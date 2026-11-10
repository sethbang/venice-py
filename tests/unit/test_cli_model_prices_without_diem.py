"""CLI price rendering when a model publishes a USD price only."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from venice_ai.cli.commands.models.comparator import ModelComparator
from venice_ai.cli.commands.models.formatters import ModelFormatter
from venice_ai.types.api.models import ModelResponse, PricingTier


def _model() -> ModelResponse:
    entry: dict[str, Any] = {
        "id": "usd-only",
        "object": "model",
        "created": None,
        "owned_by": "venice.ai",
        "type": "text",
        "model_spec": {
            "name": "usd-only",
            "availableContextTokens": 8192,
            "pricing": {"input": {"usd": 1.0}, "output": {"usd": 2.0}},
        },
    }
    return ModelResponse.model_validate(entry)


def test_format_price_shows_na_for_an_absent_diem() -> None:
    assert ModelFormatter.format_price(PricingTier(usd=1.0)) == "$1.00 / N/A"
    assert ModelFormatter.format_price(PricingTier(usd=1.0, diem=1.0)) == "$1.00 / Ð1.00"


def test_format_gen_price_shows_na_for_an_absent_diem() -> None:
    assert ModelFormatter.format_gen_price(PricingTier(usd=1.0)) == "$1.000 / N/A"


def test_compare_models_renders_a_model_without_diem() -> None:
    table = ModelComparator.compare_models([_model(), _model()], currency="both")
    assert table is not None
    console = Console(record=True, width=120)
    console.print(table)
    assert "N/A" in console.export_text()
