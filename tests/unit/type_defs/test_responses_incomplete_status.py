"""Responses API bodies for generations that stop early.

When ``max_output_tokens`` (or a content filter) cuts a generation short, the
server answers 200 with ``status="incomplete"``, an ``incomplete_details``
object naming the reason, and the partial output blocks, each also marked
``"incomplete"``. The SDK must hand that partial output back to the caller
instead of failing the parse, and must keep individual output blocks typed
rather than letting them fall through to the unknown-block fallback.

Status fields on Responses API models are open strings, with the documented
values published as ``KNOWN_*`` tuples, so a status the server adds later is
preserved rather than rejected.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from venice_ai._client import VeniceClient
from venice_ai.exceptions import APIResponseValidationError
from venice_ai.types.api import responses as responses_module
from venice_ai.types.api.responses import (
    ResponsesFunctionCallOutput,
    ResponsesMessageOutput,
    ResponsesResponse,
)

# Values the /responses schema documents for each status field.
SPEC_RESPONSE_STATUSES = ("completed", "failed", "in_progress", "cancelled", "incomplete")
SPEC_MESSAGE_STATUSES = ("completed", "in_progress", "failed", "incomplete")
SPEC_FUNCTION_CALL_STATUSES = ("completed", "in_progress", "incomplete")
SPEC_INCOMPLETE_REASONS = ("max_output_tokens", "content_filter")


def _truncated_body() -> dict[str, Any]:
    """A /responses body for a call capped by ``max_output_tokens``."""
    return {
        "id": "resp_trunc",
        "object": "response",
        "created_at": 1759017600,
        "model": "venice-uncensored",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "incomplete",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "The palace of"}],
            }
        ],
        "usage": {"input_tokens": 18, "output_tokens": 5, "total_tokens": 23},
    }


class TestIncompleteResponseParses:
    def test_top_level_incomplete_status_is_accepted(self) -> None:
        resp = ResponsesResponse.model_validate(_truncated_body())
        assert resp.status == "incomplete"

    def test_partial_message_block_stays_typed_and_keeps_text(self) -> None:
        resp = ResponsesResponse.model_validate(_truncated_body())
        block = resp.output[0]
        assert isinstance(block, ResponsesMessageOutput), (
            f"incomplete message block degraded to {type(block).__name__}"
        )
        assert block.status == "incomplete"
        assert block.content[0].text == "The palace of"

    def test_partial_message_block_alone_stays_typed(self) -> None:
        body = _truncated_body()
        body["status"] = "completed"
        body.pop("incomplete_details")
        resp = ResponsesResponse.model_validate(body)
        assert isinstance(resp.output[0], ResponsesMessageOutput), (
            f"incomplete message block degraded to {type(resp.output[0]).__name__}"
        )

    def test_partial_function_call_block_stays_typed(self) -> None:
        body = _truncated_body()
        body["status"] = "completed"
        body.pop("incomplete_details")
        body["output"] = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Ven',
                "status": "incomplete",
            }
        ]
        resp = ResponsesResponse.model_validate(body)
        block = resp.output[0]
        assert isinstance(block, ResponsesFunctionCallOutput), (
            f"incomplete function_call block degraded to {type(block).__name__}"
        )
        assert block.arguments == '{"city": "Ven'

    def test_incomplete_details_is_a_declared_field(self) -> None:
        assert "incomplete_details" in ResponsesResponse.model_fields, (
            "incomplete_details is only reachable as an untyped extra"
        )

    @pytest.mark.parametrize("reason", SPEC_INCOMPLETE_REASONS)
    def test_incomplete_details_reason_is_typed(self, reason: str) -> None:
        body = _truncated_body()
        body["incomplete_details"] = {"reason": reason}
        resp = ResponsesResponse.model_validate(body)
        details = resp.incomplete_details
        assert not isinstance(details, dict), "incomplete_details was left as a raw dict"
        assert details is not None
        assert details.reason == reason

    def test_completed_body_with_null_incomplete_details_parses(self) -> None:
        body = _truncated_body()
        body["status"] = "completed"
        body["incomplete_details"] = None
        body["output"][0]["status"] = "completed"
        resp = ResponsesResponse.model_validate(body)
        assert resp.status == "completed"
        assert resp.incomplete_details is None

    def test_usage_survives_on_truncated_response(self) -> None:
        resp = ResponsesResponse.model_validate(_truncated_body())
        assert resp.usage is not None
        assert resp.usage.output_tokens == 5


class TestStatusFieldsAreOpen:
    @pytest.mark.parametrize("status", SPEC_RESPONSE_STATUSES)
    def test_every_documented_response_status_parses(self, status: str) -> None:
        body = _truncated_body()
        body["status"] = status
        assert ResponsesResponse.model_validate(body).status == status

    def test_undocumented_response_status_is_preserved(self) -> None:
        body = _truncated_body()
        body["status"] = "queued"
        assert ResponsesResponse.model_validate(body).status == "queued"

    def test_undocumented_message_status_keeps_block_typed(self) -> None:
        body = _truncated_body()
        body["status"] = "completed"
        body["output"][0]["status"] = "paused"
        block = ResponsesResponse.model_validate(body).output[0]
        assert isinstance(block, ResponsesMessageOutput), (
            f"message block with a new status degraded to {type(block).__name__}"
        )
        assert block.status == "paused"

    def test_undocumented_function_call_status_keeps_block_typed(self) -> None:
        body = _truncated_body()
        body["status"] = "completed"
        body["output"] = [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": "{}",
                "status": "queued",
            }
        ]
        block = ResponsesResponse.model_validate(body).output[0]
        assert isinstance(block, ResponsesFunctionCallOutput), (
            f"function_call block with a new status degraded to {type(block).__name__}"
        )
        assert block.status == "queued"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("KNOWN_RESPONSE_STATUSES", SPEC_RESPONSE_STATUSES),
            ("KNOWN_RESPONSE_MESSAGE_STATUSES", SPEC_MESSAGE_STATUSES),
            ("KNOWN_RESPONSE_FUNCTION_CALL_STATUSES", SPEC_FUNCTION_CALL_STATUSES),
            ("KNOWN_RESPONSE_INCOMPLETE_REASONS", SPEC_INCOMPLETE_REASONS),
        ],
    )
    def test_known_values_are_published(self, name: str, expected: tuple[str, ...]) -> None:
        known = getattr(responses_module, name, None)
        assert known is not None, f"venice_ai.types.api.responses.{name} is not defined"
        assert set(expected) <= set(known), f"{name} is missing {set(expected) - set(known)}"


def _json_response(body: dict[str, Any]) -> MagicMock:
    fake = MagicMock(spec=aiohttp.ClientResponse)
    fake.status = 200
    fake.headers = {"content-type": "application/json"}
    fake.content_length = 512
    fake.json = AsyncMock(return_value=body)
    return fake


@pytest.mark.asyncio
async def test_client_returns_partial_output_for_truncated_generation() -> None:
    client = VeniceClient(api_key="test")
    client._prepare_and_send_request = AsyncMock(  # type: ignore[method-assign]
        return_value=_json_response(_truncated_body())
    )
    try:
        resp = await client.responses.create(
            model="venice-uncensored", input="Describe Versailles.", max_output_tokens=5
        )
    except APIResponseValidationError as exc:
        pytest.fail(f"truncated /responses body was rejected: {exc.validation_error}")
    finally:
        await client.close()

    assert isinstance(resp, ResponsesResponse)
    assert resp.status == "incomplete"
    assert isinstance(resp.output[0], ResponsesMessageOutput)
    assert resp.output[0].content[0].text == "The palace of"
