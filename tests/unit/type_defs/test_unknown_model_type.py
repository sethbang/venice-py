"""
Forward-compatibility tests for model types Venice adds after an SDK release.

Venice ships new model types without an SDK release. ``decision`` (the Jev
"System One" model) appeared in the live catalog while ``ModelResponse.type``
was still a closed ``Literal``, and that single unrecognised entry failed the
whole 364-model ``/models`` parse — taking ``list()``, ``get()`` and
``get_capabilities()`` down with it, since they all read one cached listing.

Parsing a model catalog must therefore never be all-or-nothing: an
unrecognised ``type`` has to round-trip as a plain string, fall back to the
base ``ModelSpec``, and — the actual regression — must not stop its
known-type siblings from validating.

The ``decision`` payload below is recorded verbatim from the live API on
2026-09-18. It now has a dedicated spec class, so the synthetic ``teleport``
type stands in for the case these tests exist for: a type the SDK has never
heard of.
"""

import pytest

from venice_ai.types.api import KNOWN_MODEL_TYPES, ModelResponse, ModelsListResponse, TextModelSpec
from venice_ai.types.api.capabilities import GenericCapabilities
from venice_ai.types.api.models import DecisionModelSpec, ModelSpec

# Verbatim /models entry for the Jev decision model, live API 2026-09-18.
JEV_ENTRY = {
    "created": 1789689600,
    "id": "jev-latest",
    "model_spec": {
        "betaModel": True,
        "pricing": {
            "input": {"usd": 0.0525, "diem": 0.0525},
            "output": {"usd": 0, "diem": 0},
        },
        "maxStateTokens": 32000,
        "maxTotalTokens": 64000,
        "name": "Jev (System One)",
        "offline": False,
        "privacy": "anonymized",
        "traits": [],
    },
    "object": "model",
    "owned_by": "venice.ai",
    "type": "decision",
}

KNOWN_TEXT_ENTRY = {
    "id": "known-text",
    "object": "model",
    "created": 1771804800.0,
    "owned_by": "venice.ai",
    "type": "text",
    "model_spec": {"name": "Known", "availableContextTokens": 1000},
}


class TestUnknownModelTypeParses:
    """An unrecognised ``type`` must parse rather than raise."""

    def test_recorded_decision_entry_parses(self):
        m = ModelResponse.model_validate(JEV_ENTRY)
        assert m.type == "decision"
        assert m.id == "jev-latest"

    def test_wholly_unknown_type_parses(self):
        """A type no one has seen yet must not raise either."""
        m = ModelResponse.model_validate(dict(JEV_ENTRY, type="teleport", id="future-model"))
        assert m.type == "teleport"

    def test_unknown_type_falls_back_to_base_spec(self):
        """No dispatch entry means the base ``ModelSpec``, not a failure."""
        m = ModelResponse.model_validate(dict(JEV_ENTRY, type="teleport"))
        assert type(m.model_spec) is ModelSpec
        assert m.model_spec.name == "Jev (System One)"

    def test_unknown_type_keeps_its_unmodelled_fields(self):
        """Data the SDK predates must survive the round trip, not be dropped."""
        m = ModelResponse.model_validate(dict(JEV_ENTRY, type="teleport"))
        extra = m.model_spec.model_extra or {}
        assert extra.get("maxStateTokens") == 32000
        assert extra.get("maxTotalTokens") == 64000

    def test_unknown_type_does_not_poison_siblings(self):
        """The real regression: one bad entry killed the entire listing."""
        parsed = ModelsListResponse.model_validate(
            {"object": "list", "type": "all", "data": [KNOWN_TEXT_ENTRY, JEV_ENTRY]}
        )
        assert len(parsed.data) == 2
        assert isinstance(parsed.data[0].model_spec, TextModelSpec)
        assert parsed.data[1].type == "decision"


class TestDecisionModelSpec:
    """``decision`` dispatches to its own spec subclass."""

    def test_dispatches_to_decision_spec(self):
        m = ModelResponse.model_validate(JEV_ENTRY)
        assert isinstance(m.model_spec, DecisionModelSpec)

    def test_token_budget_fields_typed(self):
        m = ModelResponse.model_validate(JEV_ENTRY)
        spec = m.model_spec
        assert isinstance(spec, DecisionModelSpec)
        assert spec.maxStateTokens == 32000
        assert spec.maxTotalTokens == 64000

    def test_token_budgets_optional(self):
        """Both budgets are documented 'only present for decision models' —
        an entry omitting them must still parse."""
        entry = dict(JEV_ENTRY)
        entry["model_spec"] = {"name": "Budgetless"}
        m = ModelResponse.model_validate(entry)
        assert isinstance(m.model_spec, DecisionModelSpec)
        assert m.model_spec.maxStateTokens is None
        assert m.model_spec.maxTotalTokens is None


class TestKnownModelTypes:
    """The closed Literal is gone, so callers narrow via this constant."""

    def test_importable_from_the_public_type_namespaces(self):
        """The constant callers are pointed at must actually be reachable."""
        from venice_ai.types import KNOWN_MODEL_TYPES as from_types
        from venice_ai.types.api import KNOWN_MODEL_TYPES as from_api

        assert from_types is from_api

    def test_contains_decision(self):
        assert "decision" in KNOWN_MODEL_TYPES

    def test_covers_every_dispatchable_spec(self):
        from venice_ai.types.api.models import _SPEC_BY_TYPE

        assert set(_SPEC_BY_TYPE) <= set(KNOWN_MODEL_TYPES)


class TestGenericCapabilitiesUnknownType:
    """``get_capabilities`` funnels every non-text/image/video/inpaint type
    into ``GenericCapabilities``, whose ``type`` was its own closed Literal."""

    @pytest.mark.parametrize("model_type", ["decision", "teleport"])
    def test_accepts_unknown_type(self, model_type):
        caps = GenericCapabilities(type=model_type, privacy="anonymized")
        assert caps.type == model_type


class TestCatalogWithoutDecisionModels:
    """Accounts that cannot see decision models must be unaffected.

    ``decision`` models are beta/Pro-gated, so they are absent from the
    catalog for many accounts — which is why the closed-``Literal`` crash only
    ever reproduced on accounts that *could* see Jev. Nothing added for the
    decision type may make the SDK depend on one being present.
    """

    def test_catalog_parses_without_any_decision_entry(self):
        parsed = ModelsListResponse.model_validate(
            {"object": "list", "type": "all", "data": [KNOWN_TEXT_ENTRY]}
        )
        assert len(parsed.data) == 1
        assert all(m.type != "decision" for m in parsed.data)

    @pytest.mark.asyncio
    async def test_resolve_decision_raises_a_clear_error(self):
        """Not an obscure IndexError or a silent fallback to a chat model."""
        from unittest.mock import AsyncMock

        from venice_ai.resources.models import Models

        class StubClient:
            def __init__(self) -> None:
                self.get = AsyncMock(
                    return_value=ModelsListResponse.model_validate(
                        {"object": "list", "type": "all", "data": [KNOWN_TEXT_ENTRY]}
                    )
                )
                self.models = Models(self)

        client = StubClient()
        with pytest.raises(ValueError, match="No available decision models found"):
            await client.models.resolve_decision()
