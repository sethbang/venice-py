"""estimate_cost() must account for the Venice system prompt the server injects.

Unless a request sets ``venice_parameters.include_venice_system_prompt=False``
(server default: ``True``), Venice prepends its own system prompt to the
conversation and bills those tokens as prompt tokens. Its size varies by model
and over time (observed roughly 1100-1750 tokens), and it is served from the
prompt cache, so it bills at ``cache_input`` when the model publishes one.

A pre-flight estimate that only counts the caller's words is therefore an
order of magnitude low for short prompts, and a budget gate built on it never
refuses anything.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.costs import BudgetManager, CostTracker, estimate_completion_cost
from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import UserMessage
from venice_ai.types.api.models import (
    LLMModelPricing,
    ModelResponse,
    ModelsListResponse,
    PricingTier,
)

_MODEL = "fake-chat-estimate-model"
_MILLION = Decimal("1000000")


def _pricing(*, cache_input: float | None = None) -> LLMModelPricing:
    return LLMModelPricing(
        input=PricingTier(usd=1.40, diem=1.40),
        output=PricingTier(usd=4.40, diem=4.40),
        cache_input=PricingTier(usd=cache_input, diem=cache_input) if cache_input else None,
    )


def _chat(pricing: LLMModelPricing) -> ChatCompletions:
    entry = ModelResponse.model_validate(
        {
            "id": _MODEL,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": _MODEL,
                "availableContextTokens": 131072.0,
                "pricing": pricing.model_dump(),
            },
        }
    )
    listing = ModelsListResponse(object="list", type="text", data=[entry])
    client = MagicMock()
    client.models.list = AsyncMock(return_value=listing)
    return ChatCompletions(client)


class TestDefaultIncludesVeniceSystemPrompt:
    @pytest.mark.asyncio
    async def test_short_prompt_estimate_counts_injected_system_prompt(self):
        chat = _chat(_pricing())
        estimate = await chat.estimate_cost(
            model=_MODEL,
            messages=[UserMessage(content="hi there")],
            expected_completion_tokens=20,
        )
        assert estimate.prompt_tokens >= 500, (
            "default requests carry the Venice system prompt, but the estimate "
            f"counted only {estimate.prompt_tokens} prompt tokens"
        )

    @pytest.mark.asyncio
    async def test_allowance_is_reported_separately(self):
        chat = _chat(_pricing())
        estimate = await chat.estimate_cost(
            model=_MODEL, messages=[UserMessage(content="hi there")], expected_completion_tokens=20
        )
        allowance = getattr(estimate, "venice_system_prompt_tokens", None)
        assert allowance is not None and allowance > 0, (
            "ChatCostEstimate should expose the injected-system-prompt allowance "
            f"as venice_system_prompt_tokens; got {allowance!r}"
        )
        assert estimate.prompt_tokens == allowance + 2  # 2 words * 1.3 -> 2 tokens

    @pytest.mark.asyncio
    async def test_allowance_priced_at_cache_input_when_published(self):
        chat = _chat(_pricing(cache_input=0.26))
        estimate = await chat.estimate_cost(
            model=_MODEL, messages=[UserMessage(content="hi there")], expected_completion_tokens=20
        )
        allowance = getattr(estimate, "venice_system_prompt_tokens", 0)
        assert allowance > 0, "no injected-system-prompt allowance in the estimate"
        expected_prompt_cost = Decimal(allowance) / _MILLION * Decimal("0.26") + Decimal(
            2
        ) / _MILLION * Decimal("1.40")
        assert estimate.prompt_cost_usd == expected_prompt_cost


class TestOptOut:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "venice_parameters",
        [
            pytest.param({"include_venice_system_prompt": False}, id="dict"),
            pytest.param({"enable_e2ee": True}, id="e2ee-forces-it-off"),
        ],
    )
    async def test_include_venice_system_prompt_false_drops_allowance(self, venice_parameters):
        chat = _chat(_pricing())
        estimate = await chat.estimate_cost(
            model=_MODEL,
            messages=[UserMessage(content="hi there")],
            expected_completion_tokens=20,
            venice_parameters=venice_parameters,
        )
        assert estimate.prompt_tokens == 2

    @pytest.mark.asyncio
    async def test_venice_parameters_model_is_accepted(self):
        from venice_ai.types.api.requests.common import VeniceParameters

        chat = _chat(_pricing())
        estimate = await chat.estimate_cost(
            model=_MODEL,
            messages=[UserMessage(content="hi there")],
            expected_completion_tokens=20,
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
        assert estimate.prompt_tokens == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "venice_parameters",
        [
            pytest.param({"enable_web_search": "off"}, id="dict-without-flag"),
            pytest.param({"include_venice_system_prompt": True}, id="dict-explicit-true"),
            pytest.param("model-default", id="model-default"),
        ],
    )
    async def test_venice_parameters_without_opt_out_keep_allowance(self, venice_parameters):
        from venice_ai.types.api.requests.common import VeniceParameters

        if venice_parameters == "model-default":
            venice_parameters = VeniceParameters(enable_web_search="off")
        chat = _chat(_pricing())
        estimate = await chat.estimate_cost(
            model=_MODEL,
            messages=[UserMessage(content="hi there")],
            expected_completion_tokens=20,
            venice_parameters=venice_parameters,
        )
        default = await chat.estimate_cost(
            model=_MODEL, messages=[UserMessage(content="hi there")], expected_completion_tokens=20
        )
        assert estimate.prompt_tokens == default.prompt_tokens, (
            "only include_venice_system_prompt=False removes the injected system prompt; "
            f"got {estimate.prompt_tokens} vs default {default.prompt_tokens}"
        )
        assert estimate.prompt_tokens > 2


class TestBudgetGate:
    @pytest.mark.asyncio
    async def test_can_afford_refuses_request_whose_real_cost_exceeds_cap(self):
        # Real bill for this request on a cache-priced model: ~7 uncached +
        # ~1744 cached prompt tokens and 20 completion tokens ~= $0.00055.
        pricing = _pricing(cache_input=0.26)
        real_cost = (
            Decimal(7) / _MILLION * Decimal("1.40")
            + Decimal(1744) / _MILLION * Decimal("0.26")
            + Decimal(20) / _MILLION * Decimal("4.40")
        )
        cap = Decimal("0.0003")
        assert real_cost > cap

        estimate = await _chat(pricing).estimate_cost(
            model=_MODEL, messages=[UserMessage(content="hi there")], expected_completion_tokens=20
        )
        budget = BudgetManager(tracker=CostTracker(), daily_usd=cap)
        assert not await budget.can_afford(estimate.total_cost_usd), (
            f"estimate ${estimate.total_cost_usd} lets through a request that really costs "
            f"${real_cost:.6f} against a ${cap} cap"
        )


# Observed live on a default-parameter chat model: (user prompt, billed prompt_tokens).
_OBSERVED_DEFAULT_PROMPTS = [
    ("Say hello and introduce yourself briefly.", 1751),
    ("Explain quantum computing in two or three simple sentences.", 1132),
]


class TestEstimateCalibratedAgainstObservedUsage:
    def test_observations_present(self):
        assert len(_OBSERVED_DEFAULT_PROMPTS) > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("prompt", "billed_prompt_tokens"), _OBSERVED_DEFAULT_PROMPTS)
    async def test_estimate_within_2x_of_billed_prompt_tokens(self, prompt, billed_prompt_tokens):
        estimate = await _chat(_pricing()).estimate_cost(
            model=_MODEL, messages=[UserMessage(content=prompt)], expected_completion_tokens=1
        )
        ratio = estimate.prompt_tokens / billed_prompt_tokens
        assert 0.5 <= ratio <= 2.0, (
            f"estimated {estimate.prompt_tokens} prompt tokens for a request billed "
            f"{billed_prompt_tokens} (ratio {ratio:.3f})"
        )

    @pytest.mark.parametrize(("prompt", "billed_prompt_tokens"), _OBSERVED_DEFAULT_PROMPTS)
    def test_module_level_estimator_within_2x_of_billed_prompt_cost(
        self, prompt, billed_prompt_tokens
    ):
        # No cache_input published, so the whole billed prompt is at the input rate.
        pricing = _pricing()
        billed = Decimal(billed_prompt_tokens) / _MILLION * Decimal("1.40")
        estimated = estimate_completion_cost(
            prompt=prompt, estimated_completion_tokens=0, model_pricing=pricing
        )["usd"]
        ratio = estimated / billed
        assert Decimal("0.5") <= ratio <= Decimal("2.0"), (
            f"estimate_completion_cost gave ${estimated} for a prompt billed ${billed} "
            f"(ratio {ratio:.3f})"
        )
