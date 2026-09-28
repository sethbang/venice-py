"""Cache-aware and long-context billing in the chat cost helpers.

Venice reports prompt-cache activity inside ``usage`` and publishes separate
per-million rates for each billing category on ``LLMModelPricing``:

* ``prompt_tokens`` is the whole prompt, *including* tokens read from and
  written to the cache.
* ``prompt_tokens_details.cached_tokens`` (mirrored at top level as
  ``cache_read_input_tokens`` on some models) is billed at ``cache_input``.
* ``prompt_tokens_details.cache_creation_input_tokens`` (mirrored as
  ``cache_creation_input_tokens``) is billed at ``cache_write``.
* The remainder is billed at ``input``; completion tokens at ``output``.
* When prompt tokens exceed ``extended.context_token_threshold`` the extended
  rates apply to the entire request.

Responses here are built with ``model_validate`` from wire-shaped dicts so the
usage objects are the real pydantic models the SDK parses, not mocks.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from venice_ai.costs import CostTracker, calculate_completion_cost
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.models import ExtendedPricing, LLMModelPricing

_MODEL = "fake-cache-priced-model"
_MILLION = Decimal("1000000")


def _tier(usd: float) -> dict[str, float]:
    return {"usd": usd, "diem": usd}


def _pricing(**tiers: Any) -> LLMModelPricing:
    body: dict[str, Any] = {"input": _tier(1.40), "output": _tier(4.40)}
    body.update(tiers)
    return LLMModelPricing.model_validate(body)


def _response(usage: dict[str, Any], model: str = _MODEL) -> ChatCompletionResponse:
    return ChatCompletionResponse.model_validate(
        {
            "id": "chatcmpl-cache",
            "object": "chat.completion",
            "created": 1_790_000_000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
    )


def _usage(
    prompt: int,
    completion: int,
    *,
    details: dict[str, int] | None = None,
    **top_level: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if details is not None:
        body["prompt_tokens_details"] = details
    body.update(top_level)
    return body


def _per_million(tokens: int, usd: str) -> Decimal:
    return Decimal(tokens) / _MILLION * Decimal(usd)


# A warm-cache request observed live: 1663 prompt tokens of which 1632 were
# served from cache, 100 completion tokens, on a model priced at
# $1.40 input / $0.26 cache_input / $4.40 output per million. The account
# balance moved by exactly the cache-aware amount below.
_WARM_PROMPT = 1663
_WARM_CACHED = 1632
_WARM_COMPLETION = 100
_WARM_BILLED = (
    _per_million(_WARM_PROMPT - _WARM_CACHED, "1.40")
    + _per_million(_WARM_CACHED, "0.26")
    + _per_million(_WARM_COMPLETION, "4.40")
)


class TestCacheReadBilling:
    def test_billed_amount_matches_observed_balance_delta(self):
        assert Decimal("0.00090772") == _WARM_BILLED

    @pytest.mark.parametrize(
        "usage",
        [
            pytest.param(
                _usage(_WARM_PROMPT, _WARM_COMPLETION, details={"cached_tokens": _WARM_CACHED}),
                id="nested-prompt_tokens_details",
            ),
            pytest.param(
                _usage(_WARM_PROMPT, _WARM_COMPLETION, cache_read_input_tokens=_WARM_CACHED),
                id="top-level-cache_read_input_tokens",
            ),
            pytest.param(
                _usage(
                    _WARM_PROMPT,
                    _WARM_COMPLETION,
                    details={"cached_tokens": _WARM_CACHED},
                    cache_read_input_tokens=_WARM_CACHED,
                ),
                id="both-mirrors-present",
            ),
        ],
    )
    def test_cached_tokens_billed_at_cache_input_rate(self, usage):
        pricing = _pricing(cache_input=_tier(0.26))
        cost = calculate_completion_cost(_response(usage), pricing)["usd"]
        assert cost == _WARM_BILLED, (
            f"cached prompt tokens must bill at cache_input: got {cost}, expected {_WARM_BILLED}"
        )

    def test_cached_tokens_fall_back_to_input_rate_without_cache_pricing(self):
        usage = _usage(_WARM_PROMPT, _WARM_COMPLETION, details={"cached_tokens": _WARM_CACHED})
        cost = calculate_completion_cost(_response(usage), _pricing())["usd"]
        assert cost == _per_million(_WARM_PROMPT, "1.40") + _per_million(_WARM_COMPLETION, "4.40")

    def test_zero_cached_tokens_bill_everything_at_input_rate(self):
        usage = _usage(1793, 6, details={"cached_tokens": 0})
        pricing = _pricing(cache_input=_tier(0.26))
        cost = calculate_completion_cost(_response(usage), pricing)["usd"]
        assert cost == _per_million(1793, "1.40") + _per_million(6, "4.40")


class TestCacheWriteBilling:
    # First turn of a cache-writing conversation: 10979 prompt tokens, 10938 of
    # them written to cache, nothing read.
    _PROMPT = 10979
    _WRITTEN = 10938
    _COMPLETION = 50

    @pytest.mark.parametrize(
        "usage",
        [
            pytest.param(
                _usage(
                    10979, 50, details={"cached_tokens": 0, "cache_creation_input_tokens": 10938}
                ),
                id="nested",
            ),
            pytest.param(_usage(10979, 50, cache_creation_input_tokens=10938), id="top-level"),
        ],
    )
    def test_cache_write_tokens_billed_at_cache_write_rate(self, usage):
        pricing = _pricing(
            input=_tier(6.00), output=_tier(30.00), cache_input=_tier(0.60), cache_write=_tier(7.50)
        )
        expected = (
            _per_million(self._PROMPT - self._WRITTEN, "6.00")
            + _per_million(self._WRITTEN, "7.50")
            + _per_million(self._COMPLETION, "30.00")
        )
        cost = calculate_completion_cost(_response(usage), pricing)["usd"]
        assert cost == expected, (
            f"cache writes must bill at cache_write: got {cost}, expected {expected}"
        )

    def test_read_and_write_in_same_request(self):
        # Turn two: 11031 prompt = 10938 read + 31 written + 62 uncached.
        usage = _usage(
            11031, 40, details={"cached_tokens": 10938, "cache_creation_input_tokens": 31}
        )
        pricing = _pricing(
            input=_tier(6.00), output=_tier(30.00), cache_input=_tier(0.60), cache_write=_tier(7.50)
        )
        expected = (
            _per_million(62, "6.00")
            + _per_million(10938, "0.60")
            + _per_million(31, "7.50")
            + _per_million(40, "30.00")
        )
        assert calculate_completion_cost(_response(usage), pricing)["usd"] == expected

    def test_cache_write_without_write_pricing_bills_at_input_rate(self):
        usage = _usage(1000, 10, details={"cache_creation_input_tokens": 900})
        pricing = _pricing(cache_input=_tier(0.26))
        expected = _per_million(1000, "1.40") + _per_million(10, "4.40")
        assert calculate_completion_cost(_response(usage), pricing)["usd"] == expected


class TestExtendedContextBilling:
    _EXTENDED = {
        "context_token_threshold": 200_000,
        "input": _tier(11.00),
        "output": _tier(41.25),
        "cache_input": _tier(1.10),
        "cache_write": _tier(13.75),
    }

    def test_below_threshold_uses_standard_rates(self):
        pricing = _pricing(input=_tier(5.50), output=_tier(27.50), extended=self._EXTENDED)
        usage = _usage(150_000, 1_000)
        expected = _per_million(150_000, "5.50") + _per_million(1_000, "27.50")
        assert calculate_completion_cost(_response(usage), pricing)["usd"] == expected

    def test_above_threshold_whole_request_uses_extended_rates(self):
        pricing = _pricing(input=_tier(5.50), output=_tier(27.50), extended=self._EXTENDED)
        usage = _usage(250_000, 1_000)
        expected = _per_million(250_000, "11.00") + _per_million(1_000, "41.25")
        cost = calculate_completion_cost(_response(usage), pricing)["usd"]
        assert cost == expected, (
            f"prompt above context_token_threshold must bill at extended rates: "
            f"got {cost}, expected {expected}"
        )

    def test_above_threshold_cached_tokens_use_extended_cache_rate(self):
        pricing = _pricing(
            input=_tier(5.50), output=_tier(27.50), cache_input=_tier(0.55), extended=self._EXTENDED
        )
        usage = _usage(250_000, 1_000, details={"cached_tokens": 240_000})
        expected = (
            _per_million(10_000, "11.00")
            + _per_million(240_000, "1.10")
            + _per_million(1_000, "41.25")
        )
        assert calculate_completion_cost(_response(usage), pricing)["usd"] == expected


class TestEveryCostEntrypointIsCacheAware:
    """Every public path that turns a chat response into dollars must agree."""

    _PRICING = _pricing(cache_input=_tier(0.26))
    _USAGE = _usage(_WARM_PROMPT, _WARM_COMPLETION, details={"cached_tokens": _WARM_CACHED})

    @pytest.mark.asyncio
    async def test_cost_tracker_track(self):
        tracker = CostTracker(pricing_map={_MODEL: self._PRICING})
        cost = await tracker.track(_response(self._USAGE))
        assert cost == _WARM_BILLED, f"CostTracker.track billed {cost}, expected {_WARM_BILLED}"
        summary = await tracker.summary()
        assert summary.total_cost_usd == _WARM_BILLED

    def test_response_summary_with_pricing(self):
        line = _response(self._USAGE).summary(pricing=self._PRICING)
        assert "$0.0009" in line, f"summary() cost segment is not cache-aware: {line!r}"


# ---------------------------------------------------------------------------
# Every published pricing rate must be able to move the computed cost.
# ---------------------------------------------------------------------------

_FULL_EXTENDED: dict[str, Any] = {
    "context_token_threshold": 1_000,
    "input": _tier(3.0),
    "output": _tier(6.0),
    "cache_input": _tier(0.3),
    "cache_write": _tier(3.75),
}


def _full_pricing(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "input": _tier(1.0),
        "output": _tier(2.0),
        "cache_input": _tier(0.1),
        "cache_write": _tier(1.25),
        "extended": dict(_FULL_EXTENDED),
    }
    body.update(overrides)
    return body


# Usage that exercises each pricing field of ``LLMModelPricing``. Standard-tier
# scenarios stay below the extended threshold; extended scenarios exceed it.
_STANDARD_USAGE = _usage(
    500, 100, details={"cached_tokens": 200, "cache_creation_input_tokens": 100}
)
_EXTENDED_USAGE = _usage(
    5_000, 100, details={"cached_tokens": 2_000, "cache_creation_input_tokens": 1_000}
)

_PRICING_FIELD_SCENARIOS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    # field -> (pricing body with only that rate changed, usage exercising it)
    "input": (_full_pricing(input=_tier(9.0)), _STANDARD_USAGE),
    "output": (_full_pricing(output=_tier(9.0)), _STANDARD_USAGE),
    "cache_input": (_full_pricing(cache_input=_tier(0.9)), _STANDARD_USAGE),
    "cache_write": (_full_pricing(cache_write=_tier(9.0)), _STANDARD_USAGE),
    "extended": (
        _full_pricing(extended={**_FULL_EXTENDED, "input": _tier(30.0)}),
        _EXTENDED_USAGE,
    ),
}

_EXTENDED_FIELD_SCENARIOS: dict[str, dict[str, Any]] = {
    "context_token_threshold": {**_FULL_EXTENDED, "context_token_threshold": 100_000},
    "input": {**_FULL_EXTENDED, "input": _tier(30.0)},
    "output": {**_FULL_EXTENDED, "output": _tier(60.0)},
    "cache_input": {**_FULL_EXTENDED, "cache_input": _tier(3.0)},
    "cache_write": {**_FULL_EXTENDED, "cache_write": _tier(37.5)},
}


def _cost(pricing_body: dict[str, Any], usage: dict[str, Any]) -> Decimal:
    pricing = LLMModelPricing.model_validate(pricing_body)
    return calculate_completion_cost(_response(usage), pricing)["usd"]


class TestEveryPricingRateIsConsumed:
    def test_scenarios_cover_every_llm_pricing_field(self):
        fields = set(LLMModelPricing.model_fields)
        assert len(fields) > 0
        assert fields == set(_PRICING_FIELD_SCENARIOS), (
            "every LLMModelPricing field needs a billing scenario here; "
            f"missing={fields - set(_PRICING_FIELD_SCENARIOS)}"
        )

    def test_scenarios_cover_every_extended_pricing_field(self):
        fields = set(ExtendedPricing.model_fields)
        assert len(fields) > 0
        assert fields == set(_EXTENDED_FIELD_SCENARIOS), (
            f"missing={fields - set(_EXTENDED_FIELD_SCENARIOS)}"
        )

    @pytest.mark.parametrize("field", sorted(_PRICING_FIELD_SCENARIOS))
    def test_changing_rate_changes_cost(self, field):
        changed_body, usage = _PRICING_FIELD_SCENARIOS[field]
        baseline = _cost(_full_pricing(), usage)
        changed = _cost(changed_body, usage)
        assert changed != baseline, (
            f"LLMModelPricing.{field} is published but never affects the computed cost "
            f"(both {baseline})"
        )

    @pytest.mark.parametrize("field", sorted(_EXTENDED_FIELD_SCENARIOS))
    def test_changing_extended_rate_changes_cost(self, field):
        baseline = _cost(_full_pricing(), _EXTENDED_USAGE)
        changed = _cost(_full_pricing(extended=_EXTENDED_FIELD_SCENARIOS[field]), _EXTENDED_USAGE)
        assert changed != baseline, (
            f"ExtendedPricing.{field} is published but never affects the computed cost "
            f"(both {baseline})"
        )
