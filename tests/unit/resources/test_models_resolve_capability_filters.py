"""Capability filters on ``Models.resolve()`` and its ``resolve_*`` shortcuts.

These tests drive the full parse -> selector-cache -> filter chain: the mock
client returns a real :class:`ModelsListResponse` built from trimmed copies of
live ``GET /models`` entries, so a filter reading a field the selector never
copies out of the typed model fails here instead of silently matching nothing.

In every "sorts first" catalog the non-capable model is listed first *and*
carries the ``default`` trait, so a pass cannot come from catalog order or the
trait fallback.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.models import Models
from venice_ai.types.api.models import (
    ImageModelConstraints,
    InpaintModelConstraints,
    ModelsListResponse,
)

# ---------------------------------------------------------------------------
# Catalog fixtures
# ---------------------------------------------------------------------------


def _entry(model_id: str, model_type: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "venice.ai",
        "type": model_type,
        "model_spec": {"name": model_id, "privacy": "private", **spec},
    }


def _image(
    model_id: str,
    *,
    traits: list[str] | None = None,
    web_search: bool = False,
    extra_constraints: dict[str, Any] | None = None,
    uncensored: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "traits": traits or [],
        "supportsWebSearch": web_search,
        "constraints": {
            "promptCharacterLimit": 7500,
            "steps": {"default": 20, "max": 50},
            "widthHeightDivisor": 1,
            **(extra_constraints or {}),
        },
    }
    if uncensored:
        spec["uncensored"] = True
    return _entry(model_id, "image", spec)


def _inpaint(
    model_id: str,
    *,
    traits: list[str] | None = None,
    combine: bool = True,
    extra_constraints: dict[str, Any] | None = None,
    uncensored: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "traits": traits or [],
        "constraints": {
            "promptCharacterLimit": 1500,
            "combineImages": combine,
            **(extra_constraints or {}),
        },
    }
    if uncensored:
        spec["uncensored"] = True
    return _entry(model_id, "inpaint", spec)


def _video(
    model_id: str,
    *,
    traits: list[str] | None = None,
    durations: list[str],
    resolutions: list[str] | None = None,
    audio: bool = True,
    audio_configurable: bool = False,
) -> dict[str, Any]:
    return _entry(
        model_id,
        "video",
        {
            "traits": traits or [],
            "constraints": {
                "model_type": "text-to-video",
                "aspect_ratios": ["16:9"],
                "resolutions": resolutions or [],
                "durations": durations,
                "audio": audio,
                "audio_configurable": audio_configurable,
                "video_input": False,
            },
        },
    )


def _text(
    model_id: str,
    *,
    traits: list[str] | None = None,
    web_search: bool = False,
    reasoning: bool = False,
    effort_options: list[str] | None = None,
    function_calling: bool = True,
    vision: bool = False,
    response_schema: bool = True,
    context_tokens: int = 131072,
    privacy: str = "private",
    beta: bool = False,
    extra_caps: dict[str, Any] | None = None,
    pricing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    caps: dict[str, Any] = {
        "optimizedForCode": False,
        "quantization": "fp8",
        "supportsFunctionCalling": function_calling,
        "supportsReasoning": reasoning,
        "supportsResponseSchema": response_schema,
        "supportsVision": vision,
        "supportsWebSearch": web_search,
        "supportsLogProbs": False,
        "supportsReasoningEffort": effort_options is not None,
        **(extra_caps or {}),
    }
    if effort_options is not None:
        caps["reasoningEffortOptions"] = effort_options
    spec: dict[str, Any] = {
        "traits": traits or [],
        "availableContextTokens": context_tokens,
        "capabilities": caps,
        "privacy": privacy,
    }
    if beta:
        spec["betaModel"] = True
    if pricing is not None:
        spec["pricing"] = pricing
    return _entry(model_id, "text", spec)


def _resource(entries: list[dict[str, Any]]) -> Models:
    listing = ModelsListResponse.model_validate({"object": "list", "type": "all", "data": entries})
    client = MagicMock()
    client.models.list = AsyncMock(return_value=listing)
    return Models(client)


# Image: z-image-turbo (no web search, no qualities) first with `default`,
# grok-imagine-image-2-0 (qualities low/medium), nano-banana-2 (web search),
# gpt-image-2 (qualities low/medium/high).
IMAGE_CATALOG = [
    _image("z-image-turbo", traits=["default"]),
    _image(
        "grok-imagine-image-2-0",
        extra_constraints={"qualities": ["low", "medium"], "defaultQuality": "medium"},
    ),
    _image(
        "nano-banana-2",
        web_search=True,
        extra_constraints={"resolutions": ["1K", "2K", "4K"], "defaultResolution": "1K"},
    ),
    _image(
        "gpt-image-2",
        extra_constraints={
            "qualities": ["low", "medium", "high"],
            "defaultQuality": "high",
            "resolutions": ["1K", "2K", "4K"],
        },
    ),
]

# Inpaint: luma-uni-1-edit (combineImages False, no resolutions, censored)
# first with `default`; firered-image-edit (combines, no resolutions, censored);
# grok-imagine-image-2-0-edit (resolutions 1K/2K, qualities low/medium);
# qwen-image-3-edit (uncensored, resolutions 1K/2K).
INPAINT_CATALOG = [
    _inpaint("luma-uni-1-edit", traits=["default"], combine=False),
    _inpaint("firered-image-edit"),
    _inpaint(
        "grok-imagine-image-2-0-edit",
        extra_constraints={"resolutions": ["1K", "2K"], "qualities": ["low", "medium"]},
    ),
    _inpaint(
        "qwen-image-3-edit",
        extra_constraints={"resolutions": ["1K", "2K"]},
        uncensored=True,
    ),
]

# Video: ltx (no 5s, audio not configurable) first with `default`;
# happyhorse (5s, audio fixed, uppercase-P resolutions absent);
# kling (5s, audio configurable, no resolutions);
# minimax-style (5s, audio configurable, uppercase "768P").
VIDEO_CATALOG = [
    _video(
        "ltx-2-v2-3-full-text-to-video",
        traits=["default"],
        durations=["6s", "8s", "10s"],
        resolutions=["1080p", "1440p", "2160p"],
    ),
    _video(
        "happyhorse-1-1-text-to-video",
        durations=["3s", "4s", "5s", "6s"],
        resolutions=["1080p", "720p"],
    ),
    _video(
        "kling-2.6-pro-text-to-video",
        durations=["5s", "10s"],
        audio_configurable=True,
    ),
    _video(
        "minimax-h3-text-to-video",
        durations=["5s", "6s"],
        resolutions=["768P", "2K"],
        audio_configurable=True,
    ),
]

# Chat: a non-web-search, non-reasoning model first with `default`;
# a web-search model; a reasoning model whose effort options include "none";
# a reasoning model without "none".
CHAT_CATALOG = [
    _text("plain-chat", traits=["default"]),
    _text("gemini-3-5-flash", reasoning=True, effort_options=["low", "medium", "high"]),
    _text("web-chat", web_search=True),
    _text("glm-reasoner", reasoning=True, effort_options=["none", "low", "medium", "high"]),
]


# ---------------------------------------------------------------------------
# Typed parsing: filters must read parsed fields, not model_extra leftovers
# ---------------------------------------------------------------------------


class TestConstraintFieldsAreTyped:
    def test_image_constraints_type_resolutions(self) -> None:
        assert "resolutions" in ImageModelConstraints.model_fields

    def test_inpaint_constraints_type_resolutions_and_qualities(self) -> None:
        fields = InpaintModelConstraints.model_fields
        assert "resolutions" in fields
        assert "qualities" in fields


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


class TestImageFilters:
    async def test_default_unchanged_picks_default_trait(self) -> None:
        assert await _resource(IMAGE_CATALOG).resolve_image() == "z-image-turbo"

    async def test_require_web_search_skips_first_non_capable(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve_image(require_web_search=True)
        assert model == "nano-banana-2"

    async def test_require_web_search_via_resolve(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve(type="image", require_web_search=True)
        assert model == "nano-banana-2"

    async def test_require_quality_high_skips_models_without_that_tier(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve_image(require_quality="high")
        assert model == "gpt-image-2"

    async def test_require_quality_via_resolve(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve(type="image", require_quality="medium")
        assert model == "grok-imagine-image-2-0"

    async def test_require_quality_is_case_insensitive(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve_image(require_quality="HIGH")
        assert model == "gpt-image-2"

    async def test_filters_combine(self) -> None:
        with pytest.raises(ValueError, match="web_search"):
            await _resource(IMAGE_CATALOG).resolve_image(
                require_web_search=True, require_quality="high"
            )

    async def test_no_match_raises_helpful_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"image.*quality='ultra'"):
            await _resource(IMAGE_CATALOG).resolve_image(require_quality="ultra")

    async def test_preferred_model_cannot_bypass_filter(self) -> None:
        model = await _resource(IMAGE_CATALOG).resolve_image(
            require_web_search=True, preferred_models=["z-image-turbo"]
        )
        assert model == "nano-banana-2"


# ---------------------------------------------------------------------------
# Inpaint
# ---------------------------------------------------------------------------


class TestInpaintFilters:
    async def test_default_unchanged_is_catalog_first(self) -> None:
        assert await _resource(INPAINT_CATALOG).resolve_inpaint() == "luma-uni-1-edit"

    async def test_require_combine_images_skips_first_non_capable(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve_inpaint(require_combine_images=True)
        assert model == "firered-image-edit"

    async def test_require_combine_images_via_resolve(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve(
            type="inpaint", require_combine_images=True
        )
        assert model == "firered-image-edit"

    async def test_require_resolution_skips_models_without_that_tier(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve_inpaint(require_resolution="2K")
        assert model == "grok-imagine-image-2-0-edit"

    async def test_require_resolution_is_case_insensitive(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve_inpaint(require_resolution="2k")
        assert model == "grok-imagine-image-2-0-edit"

    async def test_require_uncensored(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve_inpaint(require_uncensored=True)
        assert model == "qwen-image-3-edit"

    async def test_require_uncensored_via_resolve(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve(type="inpaint", require_uncensored=True)
        assert model == "qwen-image-3-edit"

    async def test_require_quality(self) -> None:
        model = await _resource(INPAINT_CATALOG).resolve_inpaint(require_quality="low")
        assert model == "grok-imagine-image-2-0-edit"

    async def test_no_match_raises_helpful_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"inpaint.*resolution='4K'"):
            await _resource(INPAINT_CATALOG).resolve_inpaint(require_resolution="4K")


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------


class TestVideoFilters:
    async def test_default_unchanged_is_catalog_first(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve_video()
        assert model == "ltx-2-v2-3-full-text-to-video"

    async def test_min_duration_alone_still_admits_longer_only_model(self) -> None:
        # The existing minimum-duration filter is satisfied by 10s, so it keeps
        # the first model; only the exact filter rejects it.
        model = await _resource(VIDEO_CATALOG).resolve_video(min_duration="5s")
        assert model == "ltx-2-v2-3-full-text-to-video"

    async def test_require_duration_int_skips_model_without_exact_length(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve_video(require_duration=5)
        assert model == "happyhorse-1-1-text-to-video"

    async def test_require_duration_str_forms(self) -> None:
        resource = _resource(VIDEO_CATALOG)
        assert await resource.resolve_video(require_duration="5s") == (
            "happyhorse-1-1-text-to-video"
        )
        assert await resource.resolve_video(require_duration="5") == (
            "happyhorse-1-1-text-to-video"
        )
        assert await resource.resolve_video(require_duration="5 seconds") == (
            "happyhorse-1-1-text-to-video"
        )

    async def test_require_duration_via_resolve(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve(type="video", require_duration=5)
        assert model == "happyhorse-1-1-text-to-video"

    async def test_require_audio_configurable(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve_video(require_audio_configurable=True)
        assert model == "kling-2.6-pro-text-to-video"

    async def test_require_audio_configurable_via_resolve(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve(
            type="video", require_audio_configurable=True
        )
        assert model == "kling-2.6-pro-text-to-video"

    async def test_require_resolution_exact_and_case_insensitive(self) -> None:
        resource = _resource(VIDEO_CATALOG)
        assert await resource.resolve_video(require_resolution="720p") == (
            "happyhorse-1-1-text-to-video"
        )
        assert await resource.resolve_video(require_resolution="768p") == (
            "minimax-h3-text-to-video"
        )

    async def test_filters_combine(self) -> None:
        model = await _resource(VIDEO_CATALOG).resolve_video(
            require_duration=5, require_audio_configurable=True, require_resolution="768P"
        )
        assert model == "minimax-h3-text-to-video"

    async def test_no_match_raises_helpful_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"video.*duration='7s'"):
            await _resource(VIDEO_CATALOG).resolve_video(require_duration=7)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChatFilters:
    async def test_default_unchanged_picks_default_trait(self) -> None:
        assert await _resource(CHAT_CATALOG).resolve_chat() == "plain-chat"

    async def test_require_web_search_skips_first_non_capable(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(require_web_search=True)
        assert model == "web-chat"

    async def test_require_web_search_via_resolve(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve(type="chat", require_web_search=True)
        assert model == "web-chat"

    async def test_require_reasoning_effort_none_skips_model_without_it(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(
            require_reasoning=True, require_reasoning_effort="none"
        )
        assert model == "glm-reasoner"

    async def test_require_reasoning_effort_via_resolve(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve(type="chat", require_reasoning_effort="none")
        assert model == "glm-reasoner"

    async def test_no_match_raises_helpful_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"reasoning_effort='max'"):
            await _resource(CHAT_CATALOG).resolve_chat(require_reasoning_effort="max")

    async def test_filters_apply_alongside_function_calling(self) -> None:
        # Function-calling selection has its own path and its own
        # ``function_calling_default`` trait; the new filters must reach it.
        catalog = [
            _text("plain-chat", traits=["default", "function_calling_default"]),
            *CHAT_CATALOG[1:],
        ]
        resource = _resource(catalog)
        assert await resource.resolve_chat(require_function_calling=True) == "plain-chat"
        assert (
            await resource.resolve_chat(require_function_calling=True, require_web_search=True)
            == "web-chat"
        )
        assert (
            await resource.resolve_chat(
                require_function_calling=True, require_reasoning_effort="none"
            )
            == "glm-reasoner"
        )


class TestFunctionCallingKeepsEveryFilter:
    """``require_function_calling`` must combine with every other chat filter.

    ``fc-beta`` is listed first and holds both the ``default`` and the
    ``function_calling_default`` traits, but it is beta, has no vision, no
    reasoning, no response schema, a small context window and is not private.
    Each filter on its own must therefore move the pick to ``fc-full``.
    """

    CATALOG = [
        _text(
            "fc-beta",
            traits=["default", "function_calling_default"],
            beta=True,
            response_schema=False,
            context_tokens=8192,
            privacy="anonymized",
        ),
        _text("no-fc-vision", function_calling=False, vision=True, reasoning=True),
        _text("fc-full", vision=True, reasoning=True, context_tokens=262144),
    ]

    async def test_exclude_beta_default_applies(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(require_function_calling=True)
        assert model == "fc-full"

    async def test_beta_allowed_when_asked_keeps_trait(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(
            require_function_calling=True, exclude_beta=False
        )
        assert model == "fc-beta"

    @pytest.mark.parametrize(
        "flag",
        [
            {"require_vision": True},
            {"require_reasoning": True},
            {"require_response_schema": True},
            {"min_context_tokens": 100_000},
            {"require_private": True},
        ],
    )
    async def test_each_filter_applies(self, flag: dict[str, Any]) -> None:
        model = await _resource(self.CATALOG).resolve_chat(
            require_function_calling=True, exclude_beta=False, **flag
        )
        assert model == "fc-full"

    async def test_via_resolve(self) -> None:
        model = await _resource(self.CATALOG).resolve(
            type="chat", require_function_calling=True, require_vision=True
        )
        assert model == "fc-full"

    async def test_no_match_raises(self) -> None:
        catalog = [entry for entry in self.CATALOG if entry["id"] != "fc-full"]
        with pytest.raises(ValueError, match="function_calling=True"):
            await _resource(catalog).resolve_chat(
                require_function_calling=True, require_vision=True
            )


def _llm_pricing(
    input_usd: float, output_usd: float, *, cache_input_usd: float | None = None
) -> dict[str, Any]:
    pricing: dict[str, Any] = {
        "input": {"usd": input_usd, "diem": input_usd},
        "output": {"usd": output_usd, "diem": output_usd},
    }
    if cache_input_usd is not None:
        pricing["cache_input"] = {"usd": cache_input_usd, "diem": cache_input_usd}
    return pricing


class TestCachingImagesAndE2eeFilters:
    """``plain-chat`` is listed first with the ``default`` trait and has none
    of the three capabilities, so each pass must come from the filter."""

    CATALOG = [
        _text("plain-chat", traits=["default"], pricing=_llm_pricing(0.1, 0.4)),
        _text(
            "cached-chat",
            pricing=_llm_pricing(0.5, 2.0, cache_input_usd=0.05),
        ),
        _text(
            "multi-image-chat",
            vision=True,
            extra_caps={"supportsMultipleImages": True, "maxImages": 8},
            pricing=_llm_pricing(0.2, 0.8),
        ),
        _text(
            "e2ee-chat",
            extra_caps={"supportsE2EE": True},
            pricing=_llm_pricing(0.05, 0.1),
        ),
    ]

    async def test_defaults_unchanged(self) -> None:
        assert await _resource(self.CATALOG).resolve_chat() == "plain-chat"

    async def test_require_prompt_caching(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(require_prompt_caching=True)
        assert model == "cached-chat"

    async def test_require_prompt_caching_via_resolve(self) -> None:
        model = await _resource(self.CATALOG).resolve(type="chat", require_prompt_caching=True)
        assert model == "cached-chat"

    async def test_require_multiple_images(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(require_multiple_images=True)
        assert model == "multi-image-chat"

    async def test_require_e2ee(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(require_e2ee=True)
        assert model == "e2ee-chat"

    async def test_e2ee_id_prefix_counts_without_flag(self) -> None:
        catalog = [_text("plain-chat", traits=["default"]), _text("e2ee-unflagged")]
        assert await _resource(catalog).resolve_chat(require_e2ee=True) == "e2ee-unflagged"

    async def test_capability_flags_reach_the_selector_cache(self) -> None:
        resource = _resource(self.CATALOG)
        await resource.resolve_chat()
        cached = resource._get_selector()._cache.models["multi-image-chat"]
        caps = cached["model_spec"]["capabilities"]
        assert caps["supportsMultipleImages"] is True
        assert caps["maxImages"] == 8
        assert caps["supportsE2EE"] is False

    async def test_combined_with_function_calling(self) -> None:
        model = await _resource(self.CATALOG).resolve_chat(
            require_function_calling=True, require_prompt_caching=True
        )
        assert model == "cached-chat"

    async def test_no_match_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt_caching=True"):
            await _resource(self.CATALOG).resolve_chat(
                require_prompt_caching=True, require_multiple_images=True
            )


def _music(
    model_id: str,
    *,
    name: str | None = None,
    traits: list[str] | None = None,
    pricing: dict[str, Any] | None = None,
    **spec_fields: Any,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "traits": traits or [],
        "pricing": pricing or {"generation": {"usd": 0.1, "diem": 0.1}},
        **spec_fields,
    }
    entry = _entry(model_id, "music", spec)
    entry["model_spec"]["name"] = name or model_id
    return entry


# Shapes trimmed from live ``type=music`` entries: a TTS model with voices and
# per-character pricing, a sound-effects model, a voice changer, then two real
# music generators (only the last accepts ``force_instrumental``).
MUSIC_CATALOG = [
    _music(
        "elevenlabs-tts-v4",
        traits=["default"],
        pricing={"per_thousand_characters": {"usd": 0.092, "diem": 0.092}},
        voices=["Aria", "Roger"],
        default_voice="Aria",
    ),
    _music(
        "sonilo-v1-1-sound-effects",
        name="Sonilo V1.1 Sound Effects",
        pricing={"per_second": {"usd": 0.00207, "diem": 0.00207}},
        min_duration=1,
        max_duration=180,
    ),
    _music("plain-voice-changer", voice_changer=True),
    _music("ace-step-15", supports_force_instrumental=False, supports_lyrics=True),
    _music("elevenlabs-music-v2", supports_force_instrumental=True),
]


class TestMusicFilters:
    async def test_default_unchanged(self) -> None:
        assert await _resource(MUSIC_CATALOG).resolve_music() == "elevenlabs-tts-v4"

    async def test_exclude_non_music_skips_tts_sfx_and_voice_changer(self) -> None:
        model = await _resource(MUSIC_CATALOG).resolve_music(exclude_non_music=True)
        assert model == "ace-step-15"

    async def test_exclude_non_music_via_resolve(self) -> None:
        model = await _resource(MUSIC_CATALOG).resolve(type="music", exclude_non_music=True)
        assert model == "ace-step-15"

    async def test_require_force_instrumental(self) -> None:
        model = await _resource(MUSIC_CATALOG).resolve_music(require_force_instrumental=True)
        assert model == "elevenlabs-music-v2"

    async def test_no_match_raises(self) -> None:
        catalog = MUSIC_CATALOG[:3]
        with pytest.raises(ValueError, match="exclude_non_music=True"):
            await _resource(catalog).resolve_music(exclude_non_music=True)
