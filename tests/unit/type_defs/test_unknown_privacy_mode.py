"""
Forward-compatibility tests for privacy modes Venice adds after an SDK release.

This is the same failure class 2.4.1 fixed for ``ModelResponse.type``.
``privacy`` is a *response* field, so a value Venice adds server-side must never
fail a parse — but it was declared as a closed ``Literal["private",
"anonymized"]`` in three places, and one unrecognised value took the whole
``/models`` listing down with it, exactly as ``decision`` once did.

The three sites matter together, not separately: ``get_capabilities()`` reads
``spec.privacy`` off :class:`ModelSpec` and passes it into
:class:`ChatCapabilities` / :class:`GenericCapabilities`, so widening only the
wire model would move the crash one layer down rather than remove it.

``confidential`` below is a stand-in for a mode this release predates. Venice
publishes two today; the point of these tests is the third.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.models.selection import DynamicModelSelector
from venice_ai.resources.models import Models
from venice_ai.types.api import KNOWN_PRIVACY_MODES, ModelResponse, ModelsListResponse
from venice_ai.types.api.capabilities import ChatCapabilities, GenericCapabilities

UNKNOWN_MODE = "confidential"

KNOWN_TEXT_ENTRY = {
    "id": "known-text",
    "object": "model",
    "created": 1771804800.0,
    "owned_by": "venice.ai",
    "type": "text",
    "model_spec": {"name": "Known", "availableContextTokens": 1000, "privacy": "private"},
}

UNKNOWN_PRIVACY_ENTRY = {
    "id": "jev-next",
    "object": "model",
    "created": 1789689600,
    "owned_by": "venice.ai",
    "type": "decision",
    "model_spec": {
        "name": "Jev (System One)",
        "betaModel": True,
        "privacy": UNKNOWN_MODE,
        "maxStateTokens": 32000,
        "maxTotalTokens": 64000,
        "traits": [],
    },
}


class TestUnknownPrivacyModeParses:
    """An unrecognised ``privacy`` value must parse rather than raise."""

    def test_unknown_mode_round_trips_as_a_string(self):
        parsed = ModelsListResponse.model_validate(
            {"object": "list", "type": "all", "data": [UNKNOWN_PRIVACY_ENTRY]}
        )

        assert parsed.data[0].model_spec.privacy == UNKNOWN_MODE

    def test_unknown_mode_does_not_fail_its_known_siblings(self):
        """The actual regression: one bad row must not sink the listing."""
        parsed = ModelsListResponse.model_validate(
            {
                "object": "list",
                "type": "all",
                "data": [KNOWN_TEXT_ENTRY, UNKNOWN_PRIVACY_ENTRY],
            }
        )

        assert [m.id for m in parsed.data] == ["known-text", "jev-next"]

    def test_known_modes_still_parse(self):
        parsed = ModelsListResponse.model_validate(
            {"object": "list", "type": "all", "data": [KNOWN_TEXT_ENTRY]}
        )

        assert parsed.data[0].model_spec.privacy == "private"


class TestCapabilitiesAcceptUnknownPrivacyMode:
    """``get_capabilities()`` forwards ``spec.privacy`` into these models, so
    widening the wire model alone would just relocate the crash."""

    def test_chat_capabilities_accepts_unknown_mode(self):
        caps = ChatCapabilities(
            context_window=1000,
            supports_function_calling=False,
            supports_vision=False,
            supports_reasoning=False,
            supports_response_schema=False,
            supports_web_search=False,
            supports_logprobs=False,
            supports_audio_input=False,
            supports_video_input=False,
            supports_multiple_images=False,
            supports_reasoning_effort=False,
            supports_tee_attestation=False,
            supports_e2ee=False,
            supports_x_search=False,
            optimized_for_code=False,
            quantization="fp16",
            privacy=UNKNOWN_MODE,
        )

        assert caps.privacy == UNKNOWN_MODE

    def test_generic_capabilities_accepts_unknown_mode(self):
        caps = GenericCapabilities(type="decision", privacy=UNKNOWN_MODE)

        assert caps.privacy == UNKNOWN_MODE


class TestKnownPrivacyModes:
    """Callers narrow ``privacy`` against this constant."""

    def test_importable_from_the_public_type_namespaces(self):
        from venice_ai.types import KNOWN_PRIVACY_MODES as from_types
        from venice_ai.types.api import KNOWN_PRIVACY_MODES as from_api

        assert from_types is from_api

    def test_contains_the_documented_modes(self):
        assert set(KNOWN_PRIVACY_MODES) == {"private", "anonymized"}


TEXT_ENTRY_WITH_UNKNOWN_PRIVACY = {
    "id": "text-next",
    "object": "model",
    "created": 1789689600,
    "owned_by": "venice.ai",
    "type": "text",
    "model_spec": {
        "name": "Text (unknown privacy mode)",
        "availableContextTokens": 1000,
        "privacy": UNKNOWN_MODE,
        "capabilities": {
            "optimizedForCode": False,
            "quantization": "fp16",
            "supportsFunctionCalling": False,
            "supportsReasoning": False,
            "supportsResponseSchema": False,
            "supportsVision": False,
            "supportsWebSearch": False,
            "supportsLogProbs": False,
        },
    },
}


class TestGetCapabilitiesForwardsUnknownMode:
    """End-to-end over the forwarding path the three sites share.

    The model-level tests above pin each site in isolation; this one walks the
    path ``get_capabilities()`` actually takes, so a *fourth* site that
    re-narrows ``privacy`` somewhere between the catalog and the returned
    capabilities object fails here even if the other tests still pass.
    """

    @pytest.mark.asyncio
    async def test_chat_path_carries_the_unknown_mode_through(self, monkeypatch):
        models = Models(client=None)  # type: ignore[arg-type]  # get_capabilities only calls self.get

        async def fake_get(model_id: str) -> ModelResponse:
            assert model_id == "text-next"
            return ModelResponse.model_validate(TEXT_ENTRY_WITH_UNKNOWN_PRIVACY)

        monkeypatch.setattr(models, "get", fake_get)

        caps = await models.get_capabilities("text-next")

        assert isinstance(caps, ChatCapabilities)
        assert caps.privacy == UNKNOWN_MODE


class TestRequirePrivateFailsClosed:
    """``require_private=True`` must not admit a mode it does not recognise.

    The filter compares ``privacy == "private"``. Rewriting it as "not
    anonymized" would still pass every test that only uses the two known modes,
    while quietly routing private workloads to a model whose guarantees the SDK
    cannot vouch for.
    """

    @staticmethod
    def _selector(*entries: dict) -> DynamicModelSelector:
        client = MagicMock()
        client.models.list = AsyncMock(
            return_value=ModelsListResponse.model_validate(
                {"object": "list", "type": "all", "data": list(entries)}
            )
        )
        return DynamicModelSelector(client)

    @pytest.mark.asyncio
    async def test_unknown_mode_is_passed_over_for_a_private_model(self):
        # The unknown-mode model comes first, so a filter that admitted it would
        # pick it; a private model winning here is the filter's doing.
        selector = self._selector(TEXT_ENTRY_WITH_UNKNOWN_PRIVACY, KNOWN_TEXT_ENTRY)

        assert await selector.select_chat_model(require_private=True) == "known-text"

    @pytest.mark.asyncio
    async def test_unknown_mode_alone_is_not_accepted_as_private(self):
        selector = self._selector(TEXT_ENTRY_WITH_UNKNOWN_PRIVACY)

        with pytest.raises(ValueError, match="private=True"):
            await selector.select_chat_model(require_private=True)
