"""``ChatStream.e2ee`` and ``ChatCompletionResponse.e2ee`` carry the TEE trust level."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from venice_ai.streaming import ChatStream
from venice_ai.tee.types import TeeTrust
from venice_ai.types.api.streaming import ChatCompletionChunk


def _trust() -> TeeTrust:
    return TeeTrust(
        model="e2ee-x",
        verification="baseline",
        verifier=None,
        server_verified=True,
        nonce="cd" * 32,
        model_public_key="04" + "ab" * 64,
        signing_address="0x" + "ab" * 20,
    )


def _chunk(content: str | None = None, finish_reason: str | None = None) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "e2ee-x",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )


async def _aiter(chunks):
    for c in chunks:
        yield c


def _chunks() -> list[ChatCompletionChunk]:
    return [_chunk("Hi"), _chunk(None, "stop")]


@pytest.mark.asyncio
async def test_chat_stream_exposes_trust_before_iteration_and_on_collected_response():
    trust = _trust()
    stream = ChatStream(_aiter(_chunks()), client=Mock(), e2ee=trust)
    assert stream.e2ee is trust

    result = await stream.collect()
    assert result.e2ee is trust
    assert "e2ee" not in result.model_dump()
    assert "e2ee" not in json.loads(result.model_dump_json())
    assert "e2ee" not in result.choices[0].message.model_dump()

    deltas_stream = ChatStream(_aiter(_chunks()), client=Mock(), e2ee=trust)
    deltas = [d async for d in deltas_stream.collect_with_deltas()]
    assert deltas == ["Hi"]
    assert deltas_stream.final_response is not None
    assert deltas_stream.final_response.e2ee is trust
    assert "e2ee" not in deltas_stream.final_response.model_dump()


@pytest.mark.asyncio
async def test_chat_stream_without_trust_is_none():
    stream = ChatStream(_aiter(_chunks()), client=Mock())
    assert stream.e2ee is None
    result = await stream.collect()
    assert result.e2ee is None
    assert "e2ee" not in result.model_dump()
