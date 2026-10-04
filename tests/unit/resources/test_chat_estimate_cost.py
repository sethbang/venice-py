"""Unit tests for client.chat.completions.estimate_cost()."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.costs import ChatCostEstimate
from venice_ai.resources.chat.completions import (
    ChatCompletions,
    _concat_message_text,
)
from venice_ai.types.api import SystemMessage, UserMessage
from venice_ai.types.api.models import (
    LLMModelPricing,
    ModelResponse,
    ModelsListResponse,
    PricingTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fake test models — these never reach the API, so deterministic ids are safe.
# (Production code paths use client.models.resolve_*() per project policy.)
_FAKE_CHAT_MODEL = "fake-chat-test-model"
_FAKE_NO_PRICING_MODEL = "fake-chat-without-pricing"


def _make_pricing(input_usd: float, output_usd: float) -> LLMModelPricing:
    return LLMModelPricing(
        input=PricingTier(usd=input_usd, diem=input_usd),
        output=PricingTier(usd=output_usd, diem=output_usd),
        cache_input=None,
    )


def _make_model_entry(model_id: str, pricing: LLMModelPricing | None) -> ModelResponse:
    # Build through ``model_validate`` so the spec-hierarchy dispatcher routes
    # ``model_spec`` to ``TextModelSpec`` based on the parent ``type='text'``.
    # Type-specific fields (``availableContextTokens``, ``capabilities``, etc.)
    # live on the subclass.
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": model_id,
                "availableContextTokens": 8192.0,
                "pricing": pricing.model_dump() if pricing is not None else None,
            },
        }
    )


def _build_chat_with_models(*models: ModelResponse) -> ChatCompletions:
    """Construct a ChatCompletions whose client.models.list() returns *models*."""
    listing = ModelsListResponse(object="list", type="text", data=list(models))
    mock_client = MagicMock()
    mock_client.models.list = AsyncMock(return_value=listing)
    return ChatCompletions(mock_client)


# ---------------------------------------------------------------------------
# _concat_message_text
# ---------------------------------------------------------------------------


class TestConcatMessageText:
    def test_string_messages(self):
        msgs = [
            SystemMessage(content="You are helpful."),
            UserMessage(content="Hello there."),
        ]
        assert _concat_message_text(msgs) == "You are helpful. Hello there."

    def test_skips_none_content(self):
        # AssistantMessage allows content=None (tool-call only turns)
        from venice_ai.types.api import AssistantMessage

        msgs = [
            UserMessage(content="Question?"),
            AssistantMessage(content=None, tool_calls=None),
        ]
        assert _concat_message_text(msgs) == "Question?"

    def test_empty_list(self):
        assert _concat_message_text([]) == ""

    def test_multimodal_text_extraction(self):
        from venice_ai.core.models.common import TextContent

        msgs = [
            UserMessage(
                content=[
                    TextContent(type="text", text="Describe this:"),
                    TextContent(type="text", text="part two"),
                ]
            ),
        ]
        assert _concat_message_text(msgs) == "Describe this: part two"


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


class TestEstimateCost:
    @pytest.mark.asyncio
    async def test_returns_chat_cost_estimate(self):
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(1.0, 2.0)),
        )
        result = await chat.estimate_cost(
            model=_FAKE_CHAT_MODEL,
            messages=[UserMessage(content="hello world")],
            expected_completion_tokens=100,
        )
        assert isinstance(result, ChatCostEstimate)
        assert result.model == _FAKE_CHAT_MODEL

    @pytest.mark.asyncio
    async def test_token_math_matches_word_heuristic(self):
        # 5 words * 1.3 tokens/word = 6 prompt tokens (int truncation)
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(1.0, 2.0)),
        )
        result = await chat.estimate_cost(
            model=_FAKE_CHAT_MODEL,
            messages=[UserMessage(content="one two three four five")],
            expected_completion_tokens=200,
            venice_parameters={"include_venice_system_prompt": False},
        )
        assert result.prompt_tokens - result.template_overhead_tokens == 6
        assert result.venice_system_prompt_tokens == 0
        assert result.expected_completion_tokens == 200

    @pytest.mark.asyncio
    async def test_cost_math_per_million(self):
        # input pricing $3.00 per 1M, output $6.00 per 1M, 1000 prompt-tokens,
        # 500 completion-tokens → prompt $0.003, completion $0.003, total $0.006
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(3.0, 6.0)),
        )
        # 1000 / 1.3 ≈ 769.23 — pick a prompt that yields ~1000 tokens.
        # Use 770 words → 770 * 1.3 = 1001 tokens (int = 1001).
        prompt = " ".join(["word"] * 770)
        result = await chat.estimate_cost(
            model=_FAKE_CHAT_MODEL,
            messages=[UserMessage(content=prompt)],
            expected_completion_tokens=500,
            venice_parameters={"include_venice_system_prompt": False},
        )
        # (1001 + template overhead) / 1_000_000 * 3.0
        assert result.prompt_tokens == 1001 + result.template_overhead_tokens
        expected_prompt = Decimal(result.prompt_tokens) / Decimal(1_000_000) * Decimal("3.0")
        assert result.prompt_cost_usd == expected_prompt
        # 500 / 1_000_000 * 6.0 = 0.003
        assert result.completion_cost_usd == Decimal("0.003000")
        assert result.total_cost_usd == expected_prompt + Decimal("0.003000")

    @pytest.mark.asyncio
    async def test_custom_tokens_per_word(self):
        # CJK / code: 2.0 tokens/word
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(1.0, 1.0)),
        )
        result = await chat.estimate_cost(
            model=_FAKE_CHAT_MODEL,
            messages=[UserMessage(content="a b c d e")],  # 5 words
            tokens_per_word=2.0,
            venice_parameters={"include_venice_system_prompt": False},
        )
        assert result.prompt_tokens - result.template_overhead_tokens == 10  # 5 * 2.0

    @pytest.mark.asyncio
    async def test_default_completion_tokens(self):
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(1.0, 1.0)),
        )
        result = await chat.estimate_cost(
            model=_FAKE_CHAT_MODEL,
            messages=[UserMessage(content="hi")],
        )
        # Default is documented as 500
        assert result.expected_completion_tokens == 500

    @pytest.mark.asyncio
    async def test_unknown_model_raises(self):
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_CHAT_MODEL, _make_pricing(1.0, 1.0)),
        )
        with pytest.raises(ValueError, match="not found"):
            await chat.estimate_cost(
                model="does-not-exist",
                messages=[UserMessage(content="hi")],
            )

    @pytest.mark.asyncio
    async def test_model_without_llm_pricing_raises(self):
        chat = _build_chat_with_models(
            _make_model_entry(_FAKE_NO_PRICING_MODEL, None),
        )
        with pytest.raises(ValueError, match="no LLM token-based pricing"):
            await chat.estimate_cost(
                model=_FAKE_NO_PRICING_MODEL,
                messages=[UserMessage(content="hi")],
            )


class TestTopLevelExport:
    def test_chat_cost_estimate_re_exported(self):
        from venice_ai import ChatCostEstimate as TopLevel

        # Should be the same class
        assert TopLevel is ChatCostEstimate
