"""Unit tests for Models.get() and Models.get_capabilities()."""

import time
from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.models import Models
from venice_ai.types.api.capabilities import (
    ChatCapabilities,
    GenericCapabilities,
    ImageCapabilities,
    InpaintCapabilities,
    VideoCapabilities,
)
from venice_ai.types.api.models import (
    ImageModelConstraints,
    InpaintModelConstraints,
    LLMModelPricing,
    ModelCapabilities,
    ModelResponse,
    ModelsListResponse,
    PricingTier,
    StepsConstraint,
    VideoModelConstraints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_pricing() -> LLMModelPricing:
    return LLMModelPricing(
        input=PricingTier(usd=1.0, diem=1.0),
        output=PricingTier(usd=2.0, diem=2.0),
        cache_input=None,
    )


def _chat_caps() -> ModelCapabilities:
    return ModelCapabilities(
        optimizedForCode=True,
        quantization="fp16",
        supportsFunctionCalling=True,
        supportsReasoning=False,
        supportsResponseSchema=True,
        supportsVision=True,
        supportsWebSearch=False,
        supportsLogProbs=True,
        supportsAudioInput=False,
        supportsVideoInput=False,
        supportsMultipleImages=True,
        supportsReasoningEffort=False,
        supportsTeeAttestation=False,
        supportsE2EE=False,
        supportsXSearch=False,
    )


def _model(
    *,
    id: str,
    type: str,
    capabilities: ModelCapabilities | None = None,
    constraints: object | None = None,
    privacy: str | None = None,
    supports_web_search: bool | None = None,
    context_tokens: float | None = None,
) -> ModelResponse:
    # Build the spec dict only with fields that match the ``type`` — the
    # spec-hierarchy router on ``ModelResponse`` will pick the right subclass
    # (``TextModelSpec`` / ``ImageModelSpec`` / ``VideoModelSpec`` / etc.) and
    # keep type-specific fields (``capabilities``, ``constraints``,
    # ``supportsWebSearch``, ``availableContextTokens``) where they belong.
    spec: dict[str, object] = {"name": id, "privacy": privacy}
    if capabilities is not None:
        spec["capabilities"] = capabilities.model_dump()
    if constraints is not None:
        # ``constraints`` is typed as ``object`` for caller flexibility (text /
        # image / video / inpaint constraints share no common base). At call
        # sites it's always a Pydantic model — narrow before dumping.
        from pydantic import BaseModel as _PydanticBase

        spec["constraints"] = (
            constraints.model_dump() if isinstance(constraints, _PydanticBase) else constraints
        )
    if context_tokens is not None:
        spec["availableContextTokens"] = context_tokens
    if supports_web_search is not None:
        spec["supportsWebSearch"] = supports_web_search

    return ModelResponse.model_validate(
        {
            "id": id,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": type,
            "model_spec": spec,
        }
    )


def _build_models(*entries: ModelResponse) -> tuple[Models, AsyncMock]:
    """Make a Models resource bound to a fake client whose list() returns *entries*."""
    listing = ModelsListResponse(object="list", type="all", data=list(entries))
    client = AsyncMock()
    resource = Models(client)
    list_mock = AsyncMock(return_value=listing)
    resource.list = list_mock  # type: ignore[method-assign]
    return resource, list_mock


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestModelsGet:
    @pytest.mark.asyncio
    async def test_returns_matching_entry(self):
        resource, _ = _build_models(
            _model(id="fake-test-a", type="text", capabilities=_chat_caps()),
            _model(id="fake-test-b", type="text", capabilities=_chat_caps()),
        )
        result = await resource.get("fake-test-b")
        assert result.id == "fake-test-b"

    @pytest.mark.asyncio
    async def test_unknown_id_raises(self):
        resource, _ = _build_models(
            _model(id="fake-test-a", type="text", capabilities=_chat_caps()),
        )
        with pytest.raises(ValueError, match="not found"):
            await resource.get("does-not-exist")

    @pytest.mark.asyncio
    async def test_caches_listing_within_ttl(self):
        resource, list_mock = _build_models(
            _model(id="fake-test-a", type="text", capabilities=_chat_caps()),
        )
        await resource.get("fake-test-a")
        await resource.get("fake-test-a")
        await resource.get("fake-test-a")
        list_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cache_expires_after_ttl(self, monkeypatch):
        resource, list_mock = _build_models(
            _model(id="fake-test-a", type="text", capabilities=_chat_caps()),
        )
        # First call seeds the cache.
        await resource.get("fake-test-a")
        assert list_mock.await_count == 1
        # Advance "now" past the 30-second TTL.
        original_monotonic = time.monotonic

        def fake_monotonic():
            return original_monotonic() + 31.0

        monkeypatch.setattr("venice_ai.resources.models.time.monotonic", fake_monotonic)
        await resource.get("fake-test-a")
        assert list_mock.await_count == 2


# ---------------------------------------------------------------------------
# get_capabilities() — chat / image / video / inpaint / generic
# ---------------------------------------------------------------------------


class TestGetCapabilitiesChat:
    @pytest.mark.asyncio
    async def test_returns_chat_capabilities_with_snake_case_fields(self):
        resource, _ = _build_models(
            _model(
                id="fake-test-chat",
                type="text",
                capabilities=_chat_caps(),
                privacy="anonymized",
                context_tokens=128000.0,
            ),
        )
        caps = await resource.get_capabilities("fake-test-chat")
        assert isinstance(caps, ChatCapabilities)
        assert caps.type == "chat"
        assert caps.context_window == 128000
        assert caps.supports_function_calling is True
        assert caps.supports_vision is True
        assert caps.supports_reasoning is False
        assert caps.optimized_for_code is True
        assert caps.quantization == "fp16"
        assert caps.privacy == "anonymized"

    @pytest.mark.asyncio
    async def test_chat_model_without_capabilities_raises(self):
        resource, _ = _build_models(
            _model(id="fake-test-chat-no-caps", type="text", capabilities=None),
        )
        with pytest.raises(ValueError, match="no capabilities payload"):
            await resource.get_capabilities("fake-test-chat-no-caps")


class TestGetCapabilitiesImage:
    @pytest.mark.asyncio
    async def test_returns_image_capabilities(self):
        constraints = ImageModelConstraints(
            promptCharacterLimit=2000.0,
            steps=StepsConstraint(default=20.0, max=50.0),
            widthHeightDivisor=8.0,
        )
        resource, _ = _build_models(
            _model(
                id="fake-test-image",
                type="image",
                constraints=constraints,
                supports_web_search=True,
            ),
        )
        caps = await resource.get_capabilities("fake-test-image")
        assert isinstance(caps, ImageCapabilities)
        assert caps.type == "image"
        assert caps.prompt_character_limit == 2000
        assert caps.width_height_divisor == 8
        assert caps.supports_web_search is True

    @pytest.mark.asyncio
    async def test_image_without_constraints_returns_minimal(self):
        resource, _ = _build_models(
            _model(id="fake-test-image-bare", type="image", constraints=None),
        )
        caps = await resource.get_capabilities("fake-test-image-bare")
        assert isinstance(caps, ImageCapabilities)
        assert caps.prompt_character_limit is None
        assert caps.width_height_divisor is None


class TestGetCapabilitiesVideo:
    @pytest.mark.asyncio
    async def test_returns_video_capabilities(self):
        constraints = VideoModelConstraints(
            model_type="text-to-video",
            aspect_ratios=["16:9", "9:16"],
            resolutions=["720p", "1080p"],
            durations=["5s", "10s"],
            audio=True,
            audio_configurable=True,
            video_input=False,
        )
        resource, _ = _build_models(
            _model(id="fake-test-video", type="video", constraints=constraints),
        )
        caps = await resource.get_capabilities("fake-test-video")
        assert isinstance(caps, VideoCapabilities)
        assert caps.model_type == "text-to-video"
        assert caps.supports_audio is True
        assert caps.audio_configurable is True
        assert caps.accepts_video_input is False
        assert caps.resolutions == ["720p", "1080p"]
        assert caps.durations == ["5s", "10s"]
        assert caps.aspect_ratios == ["16:9", "9:16"]

    @pytest.mark.asyncio
    async def test_hdr_constraints_parse_from_wire_keys(self):
        constraints = VideoModelConstraints.model_validate(
            {
                "model_type": "text-to-video",
                "resolutions": ["720p", "1080p", "4k"],
                "supportsHdr": True,
                "hdrResolutions": ["1080p", "4k"],
            }
        )
        assert constraints.supports_hdr is True
        assert constraints.hdr_resolutions == ["1080p", "4k"]
        assert "supportsHdr" not in (constraints.model_extra or {})
        dumped = constraints.model_dump()
        assert dumped["supportsHdr"] is True
        assert dumped["hdrResolutions"] == ["1080p", "4k"]
        assert "supports_hdr" not in dumped
        assert VideoModelConstraints(model_type="t", supports_hdr=True).supports_hdr is True
        resource, _ = _build_models(
            _model(id="fake-test-video-hdr", type="video", constraints=constraints),
        )
        caps = await resource.get_capabilities("fake-test-video-hdr")
        assert isinstance(caps, VideoCapabilities)
        assert caps.supports_hdr is True
        assert caps.hdr_resolutions == ["1080p", "4k"]

    @pytest.mark.asyncio
    async def test_hdr_absent_means_unsupported(self):
        constraints = VideoModelConstraints(model_type="text-to-video")
        assert constraints.supports_hdr is None
        assert constraints.hdr_resolutions is None
        resource, _ = _build_models(
            _model(id="fake-test-video-sdr", type="video", constraints=constraints),
        )
        caps = await resource.get_capabilities("fake-test-video-sdr")
        assert isinstance(caps, VideoCapabilities)
        assert caps.supports_hdr is False
        assert caps.hdr_resolutions == []

    @pytest.mark.asyncio
    async def test_video_without_constraints_raises(self):
        resource, _ = _build_models(
            _model(id="fake-test-video-broken", type="video", constraints=None),
        )
        with pytest.raises(ValueError, match="no video constraints"):
            await resource.get_capabilities("fake-test-video-broken")


class TestGetCapabilitiesInpaint:
    @pytest.mark.asyncio
    async def test_returns_inpaint_capabilities(self):
        constraints = InpaintModelConstraints(
            promptCharacterLimit=1500.0,
            combineImages=True,
        )
        resource, _ = _build_models(
            _model(id="fake-test-inpaint", type="inpaint", constraints=constraints),
        )
        caps = await resource.get_capabilities("fake-test-inpaint")
        assert isinstance(caps, InpaintCapabilities)
        assert caps.prompt_character_limit == 1500
        assert caps.combine_images is True


class TestGetCapabilitiesGeneric:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_type", ["embedding", "tts", "asr", "music", "upscale"])
    async def test_returns_generic_capabilities_for_typeless_resources(self, model_type):
        resource, _ = _build_models(_model(id=f"fake-test-{model_type}", type=model_type))
        caps = await resource.get_capabilities(f"fake-test-{model_type}")
        assert isinstance(caps, GenericCapabilities)
        assert caps.type == model_type


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


def test_top_level_export():
    import venice_ai

    for name in (
        "Capabilities",
        "ChatCapabilities",
        "ImageCapabilities",
        "VideoCapabilities",
        "InpaintCapabilities",
        "GenericCapabilities",
    ):
        assert hasattr(venice_ai, name), f"{name} missing"
        assert name in venice_ai.__all__, f"{name} not in __all__"


# ---------------------------------------------------------------------------
# Cluster D: typed context_length field on ModelResponse
# ---------------------------------------------------------------------------


class TestModelResponseContextLength:
    """context_length must be a typed field on ModelResponse, not just model_extra."""

    def test_context_length_is_a_declared_field(self):
        """context_length must appear in ModelResponse.model_fields (typed, not extra)."""
        assert "context_length" in ModelResponse.model_fields

    def test_context_length_parsed_from_top_level_field(self):
        """Parsing a raw API dict with top-level context_length sets the typed attribute."""
        raw = {
            "id": "fake-llm-ctx",
            "object": "model",
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {"name": "fake-llm-ctx"},
            "context_length": 131072,
        }
        m = ModelResponse.model_validate(raw)
        assert m.context_length == 131072

    def test_context_length_defaults_to_none_when_absent(self):
        """When the API omits context_length the field must default to None (not AttributeError)."""
        raw = {
            "id": "fake-llm-no-ctx",
            "object": "model",
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {"name": "fake-llm-no-ctx"},
        }
        m = ModelResponse.model_validate(raw)
        assert m.context_length is None
