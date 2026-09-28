"""
Comprehensive tests for src/venice_ai/costs.py module.

This test file focuses on achieving >80% coverage for cost calculation functions,
testing normal operations and edge cases.
"""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from venice_ai.costs import (
    calculate_completion_cost,
    calculate_embedding_cost,
    estimate_completion_cost,
)
from venice_ai.types.api.chat import (
    ChatCompletionResponse as ChatCompletion,
)
from venice_ai.types.api.models import (
    LLMModelPricing as ModelPricing,
)
from venice_ai.types.api.models import (
    PricingTier,
)


class TestCalculateCompletionCost:
    """Test calculate_completion_cost function."""

    def test_calculate_completion_cost_normal_operation(self):
        """Test normal cost calculation with valid data."""
        # Create mock completion with usage data
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        # Create mock pricing
        pricing = ModelPricing(
            input=PricingTier(usd=10.0, diem=0.0),  # $10 per million input tokens
            output=PricingTier(usd=20.0, diem=0.0),  # $20 per million output tokens
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        # Expected calculation:
        # USD: (100/1M * $10) + (50/1M * $20) = $0.001 + $0.001 = $0.002
        expected_usd = (Decimal("100") / Decimal("1000000")) * Decimal("10.0") + (
            Decimal("50") / Decimal("1000000")
        ) * Decimal("20.0")

        assert result["usd"] == expected_usd

    def test_calculate_completion_cost_no_pricing(self):
        """Test cost calculation with None pricing."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        result = calculate_completion_cost(completion, None)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_completion_cost_no_usage_data(self):
        """Test cost calculation when completion has no usage data."""
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = None

        pricing = ModelPricing(
            input=PricingTier(usd=10.0, diem=0.0),
            output=PricingTier(usd=20.0, diem=0.0),
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_completion_cost_missing_usage_attribute(self):
        """Test cost calculation when completion has no usage attribute."""
        completion = MagicMock(spec=ChatCompletion)
        # Remove usage attribute entirely
        del completion.usage

        pricing = ModelPricing(
            input=PricingTier(usd=10.0, diem=0.0),
            output=PricingTier(usd=20.0, diem=0.0),
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_completion_cost_zero_costs(self):
        """Test cost calculation with zero pricing."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        pricing = ModelPricing(
            input=PricingTier(usd=0.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_completion_cost_none_costs_in_pricing(self):
        """Test cost calculation with None values in pricing."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        # Create pricing with None USD values
        pricing = MagicMock(spec=ModelPricing)
        pricing.input = MagicMock()
        pricing.input.usd = None
        pricing.output = MagicMock()
        pricing.output.usd = None

        result = calculate_completion_cost(completion, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_completion_cost_partial_none_costs(self):
        """Test cost calculation with one None cost field."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        # Create pricing with partial None values
        pricing = MagicMock(spec=ModelPricing)
        pricing.input = MagicMock()
        pricing.input.usd = 10.0
        pricing.output = MagicMock()
        pricing.output.usd = None  # Only output cost is None

        result = calculate_completion_cost(completion, pricing)

        # Should only calculate input cost
        expected_usd = (Decimal("100") / Decimal("1000000")) * Decimal("10.0")

        assert result["usd"] == expected_usd

    def test_calculate_completion_cost_large_token_counts(self):
        """Test cost calculation with large token counts."""
        usage = MagicMock()
        usage.prompt_tokens = 500_000  # Half million tokens
        usage.completion_tokens = 1_000_000  # One million tokens
        usage.total_tokens = 1_500_000
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0), output=PricingTier(usd=15.0, diem=0.0)
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        # Expected calculations:
        # USD: (500K/1M * $5) + (1M/1M * $15) = $2.5 + $15 = $17.5
        expected_usd = (Decimal("500000") / Decimal("1000000")) * Decimal("5.0") + (
            Decimal("1000000") / Decimal("1000000")
        ) * Decimal("15.0")
        assert result["usd"] == expected_usd


class TestCalculateEmbeddingCost:
    """Test calculate_embedding_cost function."""

    def test_calculate_embedding_cost_normal_operation(self):
        """Test normal embedding cost calculation."""
        # Create mock embedding response with usage
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 1000

        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0),  # $2 per million tokens
            output=PricingTier(usd=0.0, diem=0.0),  # Not used for embeddings
        )  # type: ignore

        result = calculate_embedding_cost(embedding_response, pricing)

        # Expected calculation:
        # USD: (1000/1M * $2) = $0.002
        expected_usd = (Decimal("1000") / Decimal("1000000")) * Decimal("2.0")

        assert result["usd"] == expected_usd

    def test_calculate_embedding_cost_no_pricing(self):
        """Test embedding cost calculation with None pricing."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 1000

        result = calculate_embedding_cost(embedding_response, None)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_embedding_cost_no_usage_data(self):
        """Test embedding cost calculation when response has no usage data."""
        embedding_response = MagicMock()
        embedding_response.usage = None

        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_embedding_cost_missing_usage_attribute(self):
        """Test embedding cost calculation when response has no usage attribute."""
        embedding_response = MagicMock()
        # Remove usage attribute entirely
        del embedding_response.usage

        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_embedding_cost_none_cost_in_pricing(self):
        """Test embedding cost calculation with None cost in pricing."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 1000

        # Create pricing with None USD values
        pricing = MagicMock(spec=ModelPricing)
        pricing.input = MagicMock()
        pricing.input.usd = None
        pricing.output = MagicMock()
        pricing.output.usd = None

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_embedding_cost_zero_tokens(self):
        """Test embedding cost calculation with zero tokens."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 0

        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("0.00")

    def test_calculate_embedding_cost_large_token_count(self):
        """Test embedding cost calculation with large token count."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 2_000_000  # 2 million tokens

        pricing = ModelPricing(
            input=PricingTier(usd=3.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore

        result = calculate_embedding_cost(embedding_response, pricing)

        # Expected: (2M/1M * $3) = $6.0 USD
        expected_usd = (Decimal("2000000") / Decimal("1000000")) * Decimal("3.0")

        assert result["usd"] == expected_usd


class TestEstimateCompletionCost:
    """Test estimate_completion_cost function."""

    def test_estimate_completion_cost_normal_operation(self):
        """Test normal cost estimation."""
        prompt = "This is a test prompt with ten words total"  # 9 words
        estimated_completion_tokens = 100

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0), output=PricingTier(usd=15.0, diem=0.0)
        )  # type: ignore

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # Expected calculation:
        # Words: 9, estimated prompt tokens: 9 * 1.3 = 11.7 -> 11 (int conversion)
        # USD: (11/1M * $5) + (100/1M * $15) = $0.000055 + $0.0015 = $0.001555
        word_count = len(prompt.split())
        estimated_prompt_tokens = int(word_count * 1.3)
        expected_usd = (Decimal(str(estimated_prompt_tokens)) / Decimal("1000000")) * Decimal(
            "5.0"
        ) + (Decimal("100") / Decimal("1000000")) * Decimal("15.0")

        assert result["usd"] == expected_usd

    def test_estimate_completion_cost_no_pricing(self):
        """Test cost estimation with None pricing."""
        prompt = "Test prompt"
        estimated_completion_tokens = 50

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=None,
        )

        assert result["usd"] == Decimal("0.00")

    def test_estimate_completion_cost_empty_prompt(self):
        """Test cost estimation with empty prompt."""
        prompt = ""
        estimated_completion_tokens = 100

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0), output=PricingTier(usd=15.0, diem=0.0)
        )  # type: ignore

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # Only completion tokens should contribute to cost
        expected_usd = (Decimal("100") / Decimal("1000000")) * Decimal("15.0")

        assert result["usd"] == expected_usd

    def test_estimate_completion_cost_single_word_prompt(self):
        """Test cost estimation with single word prompt."""
        prompt = "Hello"
        estimated_completion_tokens = 50

        pricing = ModelPricing(
            input=PricingTier(usd=10.0, diem=0.0),
            output=PricingTier(usd=20.0, diem=0.0),
        )  # type: ignore

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # 1 word * 1.3 = 1.3 -> 1 token (int conversion)
        estimated_prompt_tokens = 1
        expected_usd = (Decimal(str(estimated_prompt_tokens)) / Decimal("1000000")) * Decimal(
            "10.0"
        ) + (Decimal(str(estimated_completion_tokens)) / Decimal("1000000")) * Decimal("20.0")

        assert result["usd"] == expected_usd

    def test_estimate_completion_cost_zero_completion_tokens(self):
        """Test cost estimation with zero estimated completion tokens."""
        prompt = "Test prompt"
        estimated_completion_tokens = 0

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0), output=PricingTier(usd=15.0, diem=0.0)
        )  # type: ignore

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # Only prompt tokens should contribute
        word_count = len(prompt.split())
        estimated_prompt_tokens = int(word_count * 1.3)
        expected_usd = (Decimal(str(estimated_prompt_tokens)) / Decimal("1000000")) * Decimal("5.0")

        assert result["usd"] == expected_usd

    def test_estimate_completion_cost_none_costs_in_pricing(self):
        """Test cost estimation with None values in pricing."""
        prompt = "Test prompt"
        estimated_completion_tokens = 100

        # Create pricing with None USD values
        pricing = MagicMock(spec=ModelPricing)
        pricing.input = MagicMock()
        pricing.input.usd = None
        pricing.output = MagicMock()
        pricing.output.usd = None

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        assert result["usd"] == Decimal("0.00")

    def test_estimate_completion_cost_partial_none_costs(self):
        """Test cost estimation with partial None costs in pricing."""
        prompt = "Test prompt"
        estimated_completion_tokens = 100

        # Create pricing with partial None values
        pricing = MagicMock(spec=ModelPricing)
        pricing.input = MagicMock()
        pricing.input.usd = 5.0
        pricing.output = MagicMock()
        pricing.output.usd = None  # Only output cost is None

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # Only input cost should be calculated
        word_count = len(prompt.split())
        estimated_prompt_tokens = int(word_count * 1.3)
        expected_usd = (Decimal(str(estimated_prompt_tokens)) / Decimal("1000000")) * Decimal("5.0")

        assert result["usd"] == expected_usd

    def test_estimate_includes_venice_system_prompt_allowance_by_default(self):
        """The default estimate prices the injected system prompt at cache_input."""
        from venice_ai.costs import VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0),
            output=PricingTier(usd=15.0, diem=0.0),
            cache_input=PricingTier(usd=0.5, diem=0.0),
        )  # type: ignore

        result = estimate_completion_cost(
            prompt="Test prompt", estimated_completion_tokens=0, model_pricing=pricing
        )

        million = Decimal("1000000")
        expected_usd = (Decimal(2) / million) * Decimal("5.0") + (
            Decimal(VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE) / million
        ) * Decimal("0.5")
        assert result["usd"] == expected_usd


class TestCostCalculationConsistency:
    """Test cost calculation consistency across functions (v2.0.0 - USD only)."""

    def test_usd_calculation_completion(self):
        """Test USD calculation consistency in completion cost."""
        usage = MagicMock()
        usage.prompt_tokens = 1_000_000  # 1M tokens for easy calculation
        usage.completion_tokens = 0
        usage.total_tokens = 1_000_000
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        pricing = ModelPricing(
            input=PricingTier(usd=1.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore # $1 per 1M tokens

        result = calculate_completion_cost(completion, pricing)

        assert result["usd"] == Decimal("1.0")  # (1M/1M * $1) = $1

    def test_usd_calculation_embedding(self):
        """Test USD calculation consistency in embedding cost."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 1_000_000  # 1M tokens

        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore # $2 per 1M tokens

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("2.0")  # (1M/1M * $2) = $2

    def test_usd_calculation_estimate(self):
        """Test USD calculation consistency in cost estimation."""
        prompt = "Single"  # 1 word -> 1 token (1 * 1.3 = 1.3 -> 1)
        estimated_completion_tokens = 1_000_000  # 1M completion tokens

        pricing = ModelPricing(
            input=PricingTier(usd=1.0, diem=0.0),  # $1 per 1M tokens
            output=PricingTier(usd=3.0, diem=0.0),  # $3 per 1M tokens
        )  # type: ignore

        result = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=estimated_completion_tokens,
            model_pricing=pricing,
            include_venice_system_prompt=False,
        )

        # Expected: input (1/1M * $1) + output (1M/1M * $3) = $0.000001 + $3 = ~$3
        expected_usd = (Decimal("1") / Decimal("1000000")) * Decimal("1.0") + (
            Decimal("1000000") / Decimal("1000000")
        ) * Decimal("3.0")

        assert result["usd"] == expected_usd


class TestCostCalculationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_token_counts(self):
        """Test cost calculations with very small token counts."""
        usage = MagicMock()
        usage.prompt_tokens = 1
        usage.completion_tokens = 1
        usage.total_tokens = 2
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        pricing = ModelPricing(
            input=PricingTier(usd=1000.0, diem=0.0),  # High cost per token
            output=PricingTier(usd=2000.0, diem=0.0),
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        # Should handle very small fractions correctly
        expected_usd = (Decimal("1") / Decimal("1000000")) * Decimal("1000.0") + (
            Decimal("1") / Decimal("1000000")
        ) * Decimal("2000.0")

        assert result["usd"] == expected_usd

        assert result["usd"] > 0  # Should not be zero

    def test_pricing_with_legacy_fields(self):
        """Test that only new pricing structure is used."""
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.total_tokens = 150
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        # Create pricing with new structure
        pricing = ModelPricing(
            input=PricingTier(usd=2.0, diem=0.0),
            output=PricingTier(usd=4.0, diem=0.0),
        )  # type: ignore

        result = calculate_completion_cost(completion, pricing)

        # Should use new pricing structure
        expected_usd = (Decimal("100") / Decimal("1000000")) * Decimal("2.0") + (
            Decimal("50") / Decimal("1000000")
        ) * Decimal("4.0")

        assert result["usd"] == expected_usd

    def test_embedding_cost_with_zero_cost_pricing(self):
        """Test embedding cost calculation with zero-cost pricing (free model)."""
        embedding_response = MagicMock()
        embedding_response.usage = MagicMock()
        embedding_response.usage.total_tokens = 1000

        pricing = ModelPricing(
            input=PricingTier(usd=0.0, diem=0.0), output=PricingTier(usd=0.0, diem=0.0)
        )  # type: ignore # Free model

        result = calculate_embedding_cost(embedding_response, pricing)

        assert result["usd"] == Decimal("0.00")


class TestCostCalculationIntegration:
    """Integration tests combining multiple cost scenarios."""

    def test_multi_model_cost_comparison(self):
        """Test cost calculations across different model pricing tiers."""
        usage = MagicMock()
        usage.prompt_tokens = 10_000
        usage.completion_tokens = 5_000
        usage.total_tokens = 15_000
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        # Low-cost model pricing
        low_cost_pricing = ModelPricing(
            input=PricingTier(usd=1.0, diem=0.0), output=PricingTier(usd=2.0, diem=0.0)
        )  # type: ignore

        # High-cost model pricing
        high_cost_pricing = ModelPricing(
            input=PricingTier(usd=50.0, diem=0.0),
            output=PricingTier(usd=100.0, diem=0.0),
        )  # type: ignore

        low_cost_result = calculate_completion_cost(completion, low_cost_pricing)
        high_cost_result = calculate_completion_cost(completion, high_cost_pricing)

        # High-cost model should be significantly more expensive
        assert high_cost_result["usd"] > low_cost_result["usd"]

        # Verify exact calculations
        low_expected_usd = (Decimal("10000") / Decimal("1000000")) * Decimal("1.0") + (
            Decimal("5000") / Decimal("1000000")
        ) * Decimal("2.0")
        high_expected_usd = (Decimal("10000") / Decimal("1000000")) * Decimal("50.0") + (
            Decimal("5000") / Decimal("1000000")
        ) * Decimal("100.0")

        assert low_cost_result["usd"] == low_expected_usd
        assert high_cost_result["usd"] == high_expected_usd

    def test_estimation_vs_actual_cost_relationship(self):
        """Test relationship between estimated and actual costs."""
        prompt = "This is a test prompt for cost estimation"
        actual_prompt_tokens = 100
        actual_completion_tokens = 200

        # Create actual completion
        usage = MagicMock()
        usage.prompt_tokens = actual_prompt_tokens
        usage.completion_tokens = actual_completion_tokens
        usage.total_tokens = actual_prompt_tokens + actual_completion_tokens
        completion = MagicMock(spec=ChatCompletion)
        completion.usage = usage

        pricing = ModelPricing(
            input=PricingTier(usd=5.0, diem=0.0), output=PricingTier(usd=10.0, diem=0.0)
        )  # type: ignore

        # Calculate actual cost
        actual_cost = calculate_completion_cost(completion, pricing)

        # Calculate estimated cost using same completion token count
        estimated_cost = estimate_completion_cost(
            prompt=prompt,
            estimated_completion_tokens=actual_completion_tokens,
            model_pricing=pricing,
        )

        # Estimated and actual costs should be close but may differ due to tokenization
        # Both should be positive
        assert actual_cost["usd"] > 0
        assert estimated_cost["usd"] > 0


class TestPricingTierSubscript:
    """``pricing.input["usd"]`` should mirror ``pricing.input.usd``.

    Background: callers building rows from ``model_dump()`` see the wire form
    where every nested model is rendered as a ``dict``. Code written against
    the dump shape (``value["usd"]``) silently broke when handed live
    PricingTier instances. ``__getitem__`` makes both shapes interchangeable.
    """

    def test_getitem_matches_attribute(self):
        tier = PricingTier(usd=1.5, diem=0.25)
        assert tier["usd"] == tier.usd == 1.5
        assert tier["diem"] == tier.diem == 0.25

    def test_unknown_key_raises_key_error(self):
        tier = PricingTier(usd=1.0, diem=0.0)
        with pytest.raises(KeyError, match="nonexistent"):
            _ = tier["nonexistent"]
