"""
Unit tests for the per-type ``ModelSpec`` subclasses and the
``ModelResponse`` dispatcher.

Per-type ``ModelSpec`` subclasses declare the fields specific to each model
type (e.g. the music capability fields):
  - 9 subclasses cover the 9 model types
  - ``ModelResponse._coerce_spec_subclass`` routes the raw spec dict to the
    right subclass based on the parent ``type``
  - ``extra='allow'`` ensures future fields aren't silently dropped

Each subclass test below feeds a representative payload **as observed from
the live API on 2026-04-28** and asserts both the runtime class and the
specific fields we care about.
"""

from venice_ai.types.api import (
    AsrModelSpec,
    EmbeddingModelSpec,
    ImageModelSpec,
    InpaintModelSpec,
    ModelResponse,
    ModelSpec,
    MusicModelSpec,
    TextModelSpec,
    TtsModelSpec,
    UpscaleModelSpec,
    VideoModelSpec,
)
from venice_ai.types.api.models import (
    ASRModelPricing,
    AudioModelPricing,
    ImageModelConstraints,
    ImageModelPricing,
    InpaintModelConstraints,
    InpaintModelPricing,
    LLMModelPricing,
    ModelCapabilities,
    VideoModelConstraints,
)


def _wrap(model_type: str, spec: dict, model_id: str = "test-model") -> dict:
    """Shape a spec into the full ``ModelResponse`` envelope."""
    return {
        "id": model_id,
        "object": "model",
        "created": 1771804800.0,
        "owned_by": "venice.ai",
        "type": model_type,
        "model_spec": spec,
    }


class TestSpecDispatch:
    """``ModelResponse._coerce_spec_subclass`` routes ``model_spec`` to the
    right subclass based on the parent ``type``."""

    def test_text_dispatch(self):
        m = ModelResponse.model_validate(
            _wrap("text", {"name": "Test text", "availableContextTokens": 200000})
        )
        assert isinstance(m.model_spec, TextModelSpec)
        assert m.model_spec.availableContextTokens == 200000

    def test_image_dispatch(self):
        m = ModelResponse.model_validate(
            _wrap("image", {"name": "Test image", "supportsWebSearch": True})
        )
        assert isinstance(m.model_spec, ImageModelSpec)
        assert m.model_spec.supportsWebSearch is True

    def test_video_dispatch(self):
        m = ModelResponse.model_validate(_wrap("video", {"name": "Test video"}))
        assert isinstance(m.model_spec, VideoModelSpec)

    def test_inpaint_dispatch(self):
        m = ModelResponse.model_validate(_wrap("inpaint", {"name": "Test inpaint"}))
        assert isinstance(m.model_spec, InpaintModelSpec)

    def test_music_dispatch(self):
        m = ModelResponse.model_validate(_wrap("music", {"name": "Test music", "min_duration": 5}))
        assert isinstance(m.model_spec, MusicModelSpec)
        assert m.model_spec.min_duration == 5

    def test_tts_dispatch(self):
        m = ModelResponse.model_validate(
            _wrap("tts", {"name": "Test TTS", "voices": ["alpha", "beta"]})
        )
        assert isinstance(m.model_spec, TtsModelSpec)
        assert m.model_spec.voices == ["alpha", "beta"]

    def test_asr_dispatch(self):
        m = ModelResponse.model_validate(_wrap("asr", {"name": "Test ASR"}))
        assert isinstance(m.model_spec, AsrModelSpec)

    def test_embedding_dispatch(self):
        m = ModelResponse.model_validate(
            _wrap(
                "embedding",
                {
                    "name": "Test embedding",
                    "embeddingDimensions": 1024,
                    "maxInputTokens": 8192,
                    "supportsCustomDimensions": True,
                },
            )
        )
        assert isinstance(m.model_spec, EmbeddingModelSpec)
        assert m.model_spec.embeddingDimensions == 1024
        assert m.model_spec.maxInputTokens == 8192
        assert m.model_spec.supportsCustomDimensions is True

    def test_upscale_dispatch(self):
        m = ModelResponse.model_validate(_wrap("upscale", {"name": "Test upscale"}))
        assert isinstance(m.model_spec, UpscaleModelSpec)

    def test_dispatch_preserves_isinstance_against_base(self):
        """Every subclass IS a ModelSpec — generic code typed against the
        base continues to work."""
        for t in (
            "text",
            "image",
            "video",
            "inpaint",
            "music",
            "tts",
            "asr",
            "embedding",
            "upscale",
        ):
            m = ModelResponse.model_validate(_wrap(t, {"name": "X"}))
            assert isinstance(m.model_spec, ModelSpec)

    def test_dispatch_skips_already_validated_spec_instance(self):
        """If the caller hands us a Pydantic instance (not a dict) for
        ``model_spec``, the dispatcher leaves it alone — only raw API
        responses get coerced."""
        spec = MusicModelSpec(name="prebuilt", min_duration=5)
        m = ModelResponse.model_validate(
            {
                "id": "x",
                "object": "model",
                "created": 1.0,
                "owned_by": "venice.ai",
                "type": "music",
                "model_spec": spec,
            }
        )
        assert m.model_spec is spec


class TestMusicSpecFields:
    """Music's 19 capability fields must round-trip through MusicModelSpec.
    Live-API sample (ACE-Step 1.5, observed 2026-04-28) round-trips fully.
    """

    def test_full_music_spec_deserialization(self):
        spec_data = {
            "pricing": {
                "durations": {
                    "60": {"usd": 0.03, "diem": 0.03, "min_seconds": 60, "max_seconds": 60}
                }
            },
            "supports_lyrics": True,
            "lyrics_required": False,
            "supports_force_instrumental": False,
            "supports_lyrics_optimizer": False,
            "duration_options": [60, 90, 120, 150, 180, 210],
            "min_duration": 60,
            "max_duration": 210,
            "default_duration": 60,
            "supported_formats": ["flac"],
            "default_format": "flac",
            "prompt_character_limit": 512,
            "lyrics_character_limit": 4096,
            "min_prompt_length": 10,
            "supports_language_code": False,
            "supports_speed": False,
            "description": "Feature-rich song generation",
            "name": "ACE-Step 1.5",
            "modelSource": "",
            "offline": False,
            "privacy": "anonymized",
            "traits": [],
        }

        m = ModelResponse.model_validate(_wrap("music", spec_data, model_id="ace-step-15"))
        assert isinstance(m.model_spec, MusicModelSpec)

        # Every capability field round-trips:
        spec = m.model_spec
        assert spec.supports_lyrics is True
        assert spec.lyrics_required is False
        assert spec.supports_force_instrumental is False
        assert spec.supports_lyrics_optimizer is False
        assert spec.duration_options == [60, 90, 120, 150, 180, 210]
        assert spec.min_duration == 60
        assert spec.max_duration == 210
        assert spec.default_duration == 60
        assert spec.supported_formats == ["flac"]
        assert spec.default_format == "flac"
        assert spec.prompt_character_limit == 512
        assert spec.lyrics_character_limit == 4096
        assert spec.min_prompt_length == 10
        assert spec.supports_language_code is False
        assert spec.supports_speed is False
        assert spec.description == "Feature-rich song generation"
        assert spec.privacy == "anonymized"

    def test_music_speed_capability_fields(self):
        """Music models that DO support speed expose min/max/default."""
        spec_data = {
            "name": "Speedy Music",
            "supports_speed": True,
            "min_speed": 0.5,
            "max_speed": 2.0,
            "default_speed": 1.0,
        }
        m = ModelResponse.model_validate(_wrap("music", spec_data))
        assert isinstance(m.model_spec, MusicModelSpec)
        assert m.model_spec.supports_speed is True
        assert m.model_spec.min_speed == 0.5
        assert m.model_spec.max_speed == 2.0
        assert m.model_spec.default_speed == 1.0


class TestExtraAllowSafetyNet:
    """``extra='allow'`` ensures Venice can ship new fields without breaking the SDK.

    Unknown fields land in ``model_extra`` and survive ``model_dump()`` round-trips.
    """

    def test_unknown_field_preserved_on_subclass(self):
        spec = {"name": "Test", "min_duration": 5, "experimental_field": "xyz"}
        m = ModelResponse.model_validate(_wrap("music", spec))
        assert isinstance(m.model_spec, MusicModelSpec)
        assert m.model_spec.model_extra is not None
        assert m.model_spec.model_extra.get("experimental_field") == "xyz"

    def test_unknown_field_preserved_on_base(self):
        """Unknown ``type`` falls back to the base ``ModelSpec`` and still
        keeps unknown fields via ``extra='allow'``."""
        spec = {"name": "Test", "future_field": 42}
        m = ModelResponse.model_validate(_wrap("text", spec))
        assert m.model_spec.model_extra is not None
        assert m.model_spec.model_extra.get("future_field") == 42

    def test_dump_round_trips_unknown_fields(self):
        spec = {"name": "Test", "supports_lyrics": True, "future_capability": "yes"}
        m = ModelResponse.model_validate(_wrap("music", spec))
        dumped = m.model_spec.model_dump()
        assert dumped["future_capability"] == "yes"
        assert dumped["supports_lyrics"] is True


class TestBackwardCompat:
    """Existing access patterns continue to work — fields kept on the base
    are accessible on every subclass via inheritance, and ``hasattr`` returns
    the right answer for type-specific fields."""

    def test_base_fields_accessible_on_every_subclass(self):
        """``name``, ``offline``, ``privacy``, ``traits``, ``pricing`` (etc.)
        are inherited by every subclass."""
        for t in (
            "text",
            "image",
            "video",
            "inpaint",
            "music",
            "tts",
            "asr",
            "embedding",
            "upscale",
        ):
            m = ModelResponse.model_validate(_wrap(t, {"name": "X", "privacy": "private"}))
            assert m.model_spec.name == "X"
            assert m.model_spec.privacy == "private"
            assert m.model_spec.traits == []
            assert m.model_spec.offline is False

    def test_type_specific_field_hasattr_only_on_owning_subclass(self):
        """``hasattr`` for music-specific fields returns False on non-music
        subclasses — existing CLI/utility code uses this pattern as a guard."""
        text = ModelResponse.model_validate(_wrap("text", {"name": "X"}))
        assert not hasattr(text.model_spec, "min_duration")

        music = ModelResponse.model_validate(_wrap("music", {"name": "X"}))
        assert hasattr(music.model_spec, "min_duration")


class TestModelResponseExtraAllow:
    """``ModelResponse`` itself uses ``extra='allow'`` so new top-level Venice
    fields land on ``model_extra`` rather than being silently dropped.
    ``context_length`` is a typed field; the extra-allow policy remains for
    truly unknown future fields."""

    def test_unknown_top_level_field_preserved_on_model_extra(self):
        """A response carrying a brand-new top-level field round-trips
        through ``model_extra`` and survives a re-dump.

        ``context_length`` is now a *typed* field on ``ModelResponse`` so it
        no longer appears in ``model_extra``.  Unknown fields like
        ``another_future_field`` still land there.
        """
        envelope = _wrap("text", {"name": "Test"})
        envelope["context_length"] = 128_000
        envelope["another_future_field"] = {"nested": True}

        m = ModelResponse.model_validate(envelope)

        # context_length is now a declared field — accessed directly, not via extra.
        assert m.context_length == 128_000
        # Truly unknown fields still land on model_extra when extra='allow'.
        assert m.model_extra == {
            "another_future_field": {"nested": True},
        }
        # model_dump() round-trips both typed and extra fields.
        dumped = m.model_dump()
        assert dumped["context_length"] == 128_000
        assert dumped["another_future_field"] == {"nested": True}


class TestSiblingPricingExtraAllow:
    """Every sibling pricing class must set ``extra='allow'``, not just
    :class:`VideoResolutionPricing`. A bare ``BaseModel`` (``extra='ignore'``)
    would silently drop unknown live pricing keys. Each sibling must preserve
    unknown live pricing keys (e.g. quality / upscale matrices) on
    ``model_extra`` and through ``model_dump()`` rather than dropping them.

    Validated directly against each pricing class (not via the undiscriminated
    ``ModelPricing`` union, which could resolve a payload to a *sibling* class
    and test the wrong thing).
    """

    def test_image_model_pricing_preserves_unknown_key(self):
        m = ImageModelPricing.model_validate(
            {
                "generation": {"usd": 0.04, "diem": 0.4},
                "upscale": {
                    "2x": {"usd": 0.02, "diem": 0.2},
                    "4x": {"usd": 0.04, "diem": 0.4},
                },
                "quality": {"high": {"usd": 0.08, "diem": 0.8}},
            }
        )
        assert m.model_extra is not None
        assert m.model_extra.get("quality") == {"high": {"usd": 0.08, "diem": 0.8}}
        assert m.model_dump(by_alias=True)["quality"] == {"high": {"usd": 0.08, "diem": 0.8}}

    def test_inpaint_model_pricing_preserves_unknown_key(self):
        m = InpaintModelPricing.model_validate(
            {
                "inpaint": {"usd": 0.01, "diem": 0.1},
                "future_tier": {"usd": 0.02, "diem": 0.2},
            }
        )
        assert m.model_extra is not None
        assert m.model_extra.get("future_tier") == {"usd": 0.02, "diem": 0.2}
        assert m.model_dump()["future_tier"] == {"usd": 0.02, "diem": 0.2}

    def test_llm_model_pricing_preserves_unknown_key(self):
        m = LLMModelPricing.model_validate(
            {
                "input": {"usd": 0.001, "diem": 0.01},
                "output": {"usd": 0.002, "diem": 0.02},
                "reasoning": {"usd": 0.003, "diem": 0.03},
            }
        )
        assert m.model_extra is not None
        assert m.model_extra.get("reasoning") == {"usd": 0.003, "diem": 0.03}
        assert m.model_dump()["reasoning"] == {"usd": 0.003, "diem": 0.03}

    def test_audio_model_pricing_preserves_unknown_key(self):
        m = AudioModelPricing.model_validate(
            {
                "input": {"usd": 0.0001, "diem": 0.001},
                "output": {"usd": 0.0002, "diem": 0.002},
            }
        )
        assert m.model_extra is not None
        assert m.model_extra.get("output") == {"usd": 0.0002, "diem": 0.002}
        assert m.model_dump()["output"] == {"usd": 0.0002, "diem": 0.002}

    def test_asr_model_pricing_preserves_unknown_key(self):
        m = ASRModelPricing.model_validate(
            {
                "per_audio_second": {"usd": 0.0001, "diem": 0.001},
                "per_audio_minute": {"usd": 0.006, "diem": 0.06},
            }
        )
        assert m.model_extra is not None
        assert m.model_extra.get("per_audio_minute") == {"usd": 0.006, "diem": 0.06}
        assert m.model_dump()["per_audio_minute"] == {"usd": 0.006, "diem": 0.06}


class TestNestedConstraintCapabilityExtraAllow:
    """``extra='allow'`` must apply to the nested ``constraints`` /
    ``capabilities`` sub-objects, not just the outer ``ModelSpec`` and pricing
    models. Pydantic's ``extra='allow'`` does NOT recurse, so each nested model
    sets it explicitly; otherwise it would silently drop live fields such as:

      - image/inpaint ``constraints``: ``aspectRatios`` / ``resolutions`` /
        ``defaultResolution`` (needed for the ``aspect_ratio`` feature)
      - video ``constraints``: ``audio_input`` / ``per_reference_audio`` /
        ``prompt_character_limit`` / ``reference_image_*``
      - text-model ``capabilities``: ``maxImages``

    Each must preserve unknown keys on ``model_extra`` and through
    ``model_dump()`` rather than dropping them.
    """

    def test_image_constraints_preserve_aspect_ratio_fields(self):
        # Real wire shape from grok-imagine-image-quality (GET /models?type=image).
        c = ImageModelConstraints.model_validate(
            {
                "promptCharacterLimit": 1500,
                "steps": {"default": 20, "max": 50},
                "widthHeightDivisor": 8,
                "aspectRatios": ["1:1", "16:9", "9:16"],
                "resolutions": ["1024x1024", "1920x1080"],
                "defaultResolution": "1024x1024",
                "defaultAspectRatio": "1:1",
            }
        )
        assert c.model_extra is not None
        assert c.model_extra.get("aspectRatios") == ["1:1", "16:9", "9:16"]
        assert c.model_extra.get("resolutions") == ["1024x1024", "1920x1080"]
        assert c.model_extra.get("defaultResolution") == "1024x1024"
        assert c.model_dump()["aspectRatios"] == ["1:1", "16:9", "9:16"]

    def test_inpaint_constraints_preserve_aspect_ratio_fields(self):
        # Real wire shape from firered-image-edit (GET /models?type=inpaint).
        c = InpaintModelConstraints.model_validate(
            {
                "promptCharacterLimit": 1500,
                "combineImages": True,
                "aspectRatios": ["1:1", "4:3"],
                "singleImageAspectRatio": "1:1",
            }
        )
        assert c.model_extra is not None
        assert c.model_extra.get("aspectRatios") == ["1:1", "4:3"]
        assert c.model_dump()["singleImageAspectRatio"] == "1:1"

    def test_video_constraints_preserve_extended_fields(self):
        # Real wire shape (GET /models?type=video — all 97 models carry these).
        c = VideoModelConstraints.model_validate(
            {
                "model_type": "image-to-video",
                "aspect_ratios": ["16:9"],
                "resolutions": ["720p", "1080p"],
                "durations": ["5s", "10s"],
                "audio": True,
                "audio_configurable": True,
                "audio_input": True,
                "per_reference_audio": False,
                "video_input": False,
                "prompt_character_limit": 2000,
                "reference_image_min_short_side_pixels": 300,
                "reference_image_min_aspect_ratio": 0.4,
                "reference_image_max_aspect_ratio": 2.5,
            }
        )
        assert c.model_extra is not None
        assert c.model_extra.get("audio_input") is True
        assert c.model_extra.get("per_reference_audio") is False
        assert c.model_extra.get("prompt_character_limit") == 2000
        assert c.model_dump()["reference_image_min_aspect_ratio"] == 0.4

    def test_text_capabilities_preserve_max_images(self):
        # Real wire shape from z-ai-glm-5v-turbo (GET /models?type=text).
        caps = ModelCapabilities.model_validate(
            {
                "optimizedForCode": False,
                "quantization": "fp8",
                "supportsFunctionCalling": True,
                "supportsReasoning": True,
                "supportsResponseSchema": True,
                "supportsVision": True,
                "supportsWebSearch": True,
                "supportsLogProbs": False,
                "maxImages": 8,
            }
        )
        assert caps.model_extra is not None
        assert caps.model_extra.get("maxImages") == 8
        assert caps.model_dump()["maxImages"] == 8

    def test_capabilities_tolerate_novel_quantization(self):
        # quantization is a plain str, so a future server-side value must not
        # crash the whole catalog parse (it would with a constrained Literal).
        caps = ModelCapabilities.model_validate(
            {
                "optimizedForCode": False,
                "quantization": "fp6-experimental",
                "supportsFunctionCalling": True,
                "supportsReasoning": True,
                "supportsResponseSchema": True,
                "supportsVision": True,
                "supportsWebSearch": True,
                "supportsLogProbs": False,
            }
        )
        assert caps.quantization == "fp6-experimental"


class TestUncensoredFlag:
    """``model_spec.uncensored`` replaces inferring the label from model IDs.

    The API sends the field only when it is true ("Present and true when
    Venice classifies this model as uncensored... Absent for all other
    models"), so absence is a definite *not uncensored*, not "unknown".
    """

    def test_uncensored_true_is_parsed(self):
        entry = ModelResponse.model_validate(
            _wrap("text", {"name": "Venice Uncensored", "uncensored": True})
        )
        assert entry.model_spec.uncensored is True

    def test_absent_means_not_uncensored(self):
        entry = ModelResponse.model_validate(_wrap("text", {"name": "Some Model"}))
        assert entry.model_spec.uncensored is False

    def test_uncensored_applies_to_non_text_modalities(self):
        """The flag covers every modality, including video."""
        entry = ModelResponse.model_validate(
            _wrap("video", {"name": "A Video Model", "uncensored": True})
        )
        assert entry.model_spec.uncensored is True
