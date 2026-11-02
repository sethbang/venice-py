"""Unit tests for plain-mapping message input on chat.completions.

``messages=`` accepts either the typed message models or plain mappings in
the OpenAI wire shape. Mappings are validated into the corresponding model
before anything reads them, so the two forms are interchangeable and a
malformed mapping still raises rather than reaching the API.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from venice_ai.exceptions import InvalidRequestError
from venice_ai.resources.chat.completions import (
    ChatCompletions,
    _coerce_messages,
    _validate_e2ee_request,
)
from venice_ai.types.api import (
    AssistantMessage,
    ChatCompletionRequest,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)
from venice_ai.types.api.models import (
    LLMModelPricing,
    ModelResponse,
    ModelsListResponse,
    PricingTier,
)

_FAKE_CHAT_MODEL = "fake-chat-test-model"


# ---------------------------------------------------------------------------
# _coerce_messages
# ---------------------------------------------------------------------------


class TestCoerceMessages:
    @pytest.mark.parametrize(
        ("mapping", "expected"),
        [
            ({"role": "user", "content": "hi"}, UserMessage),
            ({"role": "system", "content": "sys"}, SystemMessage),
            ({"role": "assistant", "content": "reply"}, AssistantMessage),
            ({"role": "developer", "content": "dev"}, DeveloperMessage),
            ({"role": "tool", "content": "result", "tool_call_id": "call_1"}, ToolMessage),
        ],
    )
    def test_each_role_maps_to_its_model(self, mapping, expected):
        (coerced,) = _coerce_messages([mapping])
        assert isinstance(coerced, expected)
        assert coerced.content == mapping["content"]

    def test_model_instances_pass_through(self):
        original = UserMessage(content="hi")
        (coerced,) = _coerce_messages([original])
        assert isinstance(coerced, UserMessage)
        assert coerced.content == "hi"

    def test_mixed_mappings_and_models(self):
        coerced = _coerce_messages(
            [{"role": "system", "content": "sys"}, UserMessage(content="hi")]
        )
        assert [type(m).__name__ for m in coerced] == ["SystemMessage", "UserMessage"]

    def test_multimodal_mapping_content(self):
        (coerced,) = _coerce_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Describe this:"}],
                }
            ]
        )
        assert isinstance(coerced, UserMessage)
        assert coerced.content[0].text == "Describe this:"

    def test_empty_sequence(self):
        assert _coerce_messages([]) == []

    def test_caller_sequence_not_mutated(self):
        original = [{"role": "user", "content": "hi"}]
        _coerce_messages(original)
        assert original == [{"role": "user", "content": "hi"}]

    # -- validation is still enforced ---------------------------------------

    def test_unknown_role_rejected(self):
        with pytest.raises(ValidationError):
            _coerce_messages([{"role": "wizard", "content": "hi"}])

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            _coerce_messages([{"role": "user"}])

    def test_tool_message_missing_call_id_rejected(self):
        with pytest.raises(ValidationError):
            _coerce_messages([{"role": "tool", "content": "result"}])

    def test_unmodeled_field_handling_matches_the_request_model(self):
        # A per-message field the models don't declare is kept and sent, the
        # same whether the mapping is coerced here or handed straight to
        # ChatCompletionRequest.
        mapping = {"role": "user", "content": "hi", "some_future_field": 1}

        (coerced,) = _coerce_messages([dict(mapping)])
        via_request = ChatCompletionRequest(model=_FAKE_CHAT_MODEL, messages=[dict(mapping)])

        assert coerced.model_dump()["some_future_field"] == 1
        assert coerced.model_dump() == via_request.messages[0].model_dump()


class TestE2EEGuardSeesMappings:
    """The E2EE pre-flight guard inspects the caller's raw sequence.

    It must reject an E2EE-incompatible message whichever shape it arrives in,
    since a mapping reaches the guard before any coercion happens.
    """

    def test_multimodal_mapping_rejected_under_e2ee(self):
        with pytest.raises(InvalidRequestError, match="Multimodal"):
            _validate_e2ee_request(
                model="e2ee-fake-model",
                messages=[{"role": "user", "content": [{"type": "text", "text": "describe"}]}],
                tools=None,
                venice_parameters=None,
            )

    def test_text_mapping_allowed_under_e2ee(self):
        _validate_e2ee_request(
            model="e2ee-fake-model",
            messages=[{"role": "user", "content": "plain text"}],
            tools=None,
            venice_parameters=None,
        )


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


class _MockClient:
    def __init__(self):
        self.post = AsyncMock(
            return_value={
                "id": "resp-1",
                "object": "chat.completion",
                "created": 1000000,
                "model": _FAKE_CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


@pytest.mark.asyncio
class TestCreateAcceptsMappings:
    async def test_mapping_and_model_produce_identical_body(self):
        from_mapping = _MockClient()
        await ChatCompletions(from_mapping).create(
            model=_FAKE_CHAT_MODEL,
            messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        )

        from_model = _MockClient()
        await ChatCompletions(from_model).create(
            model=_FAKE_CHAT_MODEL,
            messages=[SystemMessage(content="sys"), UserMessage(content="hi")],
        )

        assert from_mapping.post.call_args == from_model.post.call_args

    async def test_invalid_mapping_raises_before_the_request(self):
        client = _MockClient()
        with pytest.raises(ValidationError):
            await ChatCompletions(client).create(
                model=_FAKE_CHAT_MODEL,
                messages=[{"role": "wizard", "content": "hi"}],
            )
        client.post.assert_not_called()


# ---------------------------------------------------------------------------
# estimate_cost()
# ---------------------------------------------------------------------------


def _build_chat_with_pricing() -> ChatCompletions:
    pricing = LLMModelPricing(
        input=PricingTier(usd=1.0, diem=1.0),
        output=PricingTier(usd=2.0, diem=2.0),
        cache_input=None,
    )
    entry = ModelResponse.model_validate(
        {
            "id": _FAKE_CHAT_MODEL,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": _FAKE_CHAT_MODEL,
                "availableContextTokens": 8192.0,
                "pricing": pricing.model_dump(),
            },
        }
    )
    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(
        return_value=ModelsListResponse(object="list", type="text", data=[entry])
    )
    return ChatCompletions(mock_client)


@pytest.mark.asyncio
class TestEstimateCostAcceptsMappings:
    async def test_mapping_matches_model_estimate(self):
        words = {"role": "user", "content": "one two three four five"}

        from_mapping = await _build_chat_with_pricing().estimate_cost(
            model=_FAKE_CHAT_MODEL, messages=[words]
        )
        from_model = await _build_chat_with_pricing().estimate_cost(
            model=_FAKE_CHAT_MODEL, messages=[UserMessage(content=words["content"])]
        )

        assert from_mapping.prompt_tokens == from_model.prompt_tokens
        assert from_mapping.total_cost_usd == from_model.total_cost_usd
        assert from_mapping.prompt_tokens > 0


# ---------------------------------------------------------------------------
# run_with_tools()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRunWithToolsAcceptsMappings:
    async def test_history_is_normalized_to_models(self):
        terminal = ChatCompletionResponse(
            id="resp-terminal",
            object="chat.completion",
            created=1000000,
            model=_FAKE_CHAT_MODEL,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Final answer."),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=ChatUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=None,
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )

        def a_tool() -> str:
            """A tool the model never calls."""
            return "unused"

        chat = ChatCompletions(MagicMock())
        chat.create = AsyncMock(return_value=terminal)  # type: ignore[method-assign]

        result = await chat.run_with_tools(
            model=_FAKE_CHAT_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            tools=[a_tool],
        )

        # The returned history must not leak the caller's raw mapping.
        assert all(not isinstance(m, dict) for m in result.messages)
        assert isinstance(result.messages[0], UserMessage)
        assert result.messages[0].content == "hi"

    async def test_caller_messages_not_mutated(self):
        terminal = ChatCompletionResponse(
            id="resp-terminal",
            object="chat.completion",
            created=1000000,
            model=_FAKE_CHAT_MODEL,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Final answer."),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=ChatUsage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                prompt_tokens_details=None,
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )

        def a_tool() -> str:
            """A tool the model never calls."""
            return "unused"

        chat = ChatCompletions(MagicMock())
        chat.create = AsyncMock(return_value=terminal)  # type: ignore[method-assign]

        messages = [{"role": "user", "content": "hi"}]
        await chat.run_with_tools(model=_FAKE_CHAT_MODEL, messages=messages, tools=[a_tool])

        assert messages == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_cost_text_counts_dict_text_parts(role):
    from venice_ai.resources.chat.completions import _concat_message_text

    messages = _coerce_messages(
        [{"role": role, "content": [{"type": "text", "text": "alpha", "future": 1}]}]
    )
    assert _concat_message_text(messages) == "alpha"
