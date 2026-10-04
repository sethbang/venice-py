"""``prefer="cheapest"`` on ``Models.resolve()`` and the price ranking behind it.

Catalog entries are trimmed copies of live ``GET /models`` shapes and go through
the real :class:`ModelsListResponse` parse and the selector cache, so a price
the cache drops (or a pricing shape the parser reshapes) shows up here.

In every resolve test the most expensive model is listed first and carries the
``default`` trait, so a pass cannot come from catalog order or trait fallback.
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.models.selection import (
    CHAT_INPUT_WEIGHT,
    CHAT_OUTPUT_WEIGHT,
    cheapest_model_strategy,
    cheapest_selector,
    model_price,
)
from venice_ai.resources.models import Models
from venice_ai.types.api.models import ModelsListResponse


def _tier(usd: float) -> dict[str, float]:
    return {"usd": usd, "diem": usd}


def _entry(model_id: str, model_type: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "venice.ai",
        "type": model_type,
        "model_spec": {"name": model_id, "privacy": "private", "traits": [], **spec},
    }


def _chat(
    model_id: str,
    input_usd: float | None,
    output_usd: float | None,
    *,
    traits: list[str] | None = None,
    beta: bool = False,
    e2ee: bool = False,
    reasoning: bool = False,
    function_calling: bool = True,
    uncensored: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "uncensored": uncensored,
        "traits": traits or [],
        "availableContextTokens": 131072,
        "capabilities": {
            "optimizedForCode": False,
            "quantization": "fp8",
            "supportsFunctionCalling": function_calling,
            "supportsReasoning": reasoning,
            "supportsResponseSchema": True,
            "supportsVision": False,
            "supportsWebSearch": False,
            "supportsLogProbs": False,
            "supportsE2EE": e2ee,
        },
    }
    if input_usd is not None and output_usd is not None:
        spec["pricing"] = {"input": _tier(input_usd), "output": _tier(output_usd)}
    if beta:
        spec["betaModel"] = True
    return _entry(model_id, "text", spec)


def _simple(model_id: str, model_type: str, pricing: dict[str, Any], **spec: Any) -> dict[str, Any]:
    return _entry(model_id, model_type, {"pricing": pricing, **spec})


def _image(
    model_id: str,
    pricing: dict[str, Any],
    *,
    traits: list[str] | None = None,
    uncensored: bool = False,
    **constraints: Any,
) -> dict[str, Any]:
    return _entry(
        model_id,
        "image",
        {
            "traits": traits or [],
            "uncensored": uncensored,
            "pricing": {"upscale": {"2x": _tier(0.02), "4x": _tier(0.08)}, **pricing},
            "constraints": {
                "promptCharacterLimit": 7500,
                "steps": {"default": 20, "max": 50},
                "widthHeightDivisor": 1,
                **constraints,
            },
        },
    )


def _resource(entries: list[dict[str, Any]]) -> Models:
    listing = ModelsListResponse.model_validate({"object": "list", "type": "all", "data": entries})
    client = MagicMock()
    client.models.list = AsyncMock(return_value=listing)
    return Models(client)


async def _cached(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """The selector-cache dicts for *entries*, as a strategy receives them."""
    resource = _resource(entries)
    return await resource._get_selector()._fetch_models()


# ---------------------------------------------------------------------------
# model_price
# ---------------------------------------------------------------------------


class TestModelPrice:
    async def test_chat_uses_documented_blend(self) -> None:
        cache = await _cached([_chat("c", 1.0, 5.0)])
        expected = (CHAT_INPUT_WEIGHT * 1.0 + CHAT_OUTPUT_WEIGHT * 5.0) / (
            CHAT_INPUT_WEIGHT + CHAT_OUTPUT_WEIGHT
        )
        assert model_price(cache["c"]) == pytest.approx(expected)
        assert expected == pytest.approx(2.0)

    def test_missing_output_is_unpriced_not_zero(self) -> None:
        model = {
            "id": "m",
            "type": "text",
            "model_spec": {"pricing": {"input": _tier(0.1), "output": None}},
        }
        assert model_price(model) is None

    def test_negative_and_boolean_prices_are_unpriced(self) -> None:
        bad_negative = {"type": "tts", "model_spec": {"pricing": {"input": {"usd": -1}}}}
        bad_bool = {"type": "tts", "model_spec": {"pricing": {"input": {"usd": True}}}}
        assert model_price(bad_negative) is None
        assert model_price(bad_bool) is None

    def test_no_pricing_is_unpriced(self) -> None:
        assert model_price({"id": "m", "type": "text", "model_spec": {"pricing": None}}) is None
        assert model_price({"id": "m"}) is None

    async def test_decision_zero_output_is_a_real_price(self) -> None:
        cache = await _cached(
            [_simple("jev", "decision", {"input": _tier(0.042), "output": _tier(0)})]
        )
        assert model_price(cache["jev"]) == pytest.approx(0.75 * 0.042)

    async def test_embedding_uses_input_only(self) -> None:
        cache = await _cached(
            [_simple("e", "embedding", {"input": _tier(0.15), "output": _tier(0.6)})]
        )
        assert model_price(cache["e"]) == pytest.approx(0.15)

    async def test_tts_and_asr(self) -> None:
        cache = await _cached(
            [
                _simple("t", "tts", {"input": _tier(3.5)}),
                _simple("a", "asr", {"per_audio_second": _tier(0.0001)}),
            ]
        )
        assert model_price(cache["t"]) == pytest.approx(3.5)
        assert model_price(cache["a"]) == pytest.approx(0.0001)

    async def test_image_flat_generation(self) -> None:
        cache = await _cached([_image("flat", {"generation": _tier(0.05)})])
        assert model_price(cache["flat"]) == pytest.approx(0.05)

    async def test_image_resolution_tier_uses_default_resolution(self) -> None:
        cache = await _cached(
            [
                _image(
                    "tiered",
                    {"resolutions": {"1K": _tier(0.1), "2K": _tier(0.14), "4K": _tier(0.19)}},
                    resolutions=["1K", "2K", "4K"],
                    defaultResolution="2K",
                )
            ]
        )
        assert model_price(cache["tiered"]) == pytest.approx(0.14)

    async def test_image_resolution_tier_falls_back_to_lowest(self) -> None:
        cache = await _cached(
            [_image("tiered", {"resolutions": {"2K": _tier(0.14), "1K": _tier(0.1)}})]
        )
        assert model_price(cache["tiered"]) == pytest.approx(0.1)

    async def test_image_quality_uses_default_quality_not_cheapest_quality(self) -> None:
        # A request without ``quality`` pays the default ("high") tier.
        pricing = {
            "resolutions": {"1K": _tier(0.27), "2K": _tier(0.51)},
            "quality": {
                "1K": {"high": _tier(0.26), "low": _tier(0.02), "medium": _tier(0.07)},
                "2K": {"high": _tier(0.5), "low": _tier(0.04), "medium": _tier(0.12)},
            },
        }
        cache = await _cached(
            [
                _image(
                    "q",
                    pricing,
                    qualities=["low", "medium", "high"],
                    defaultQuality="high",
                    resolutions=["1K", "2K"],
                    defaultResolution="1K",
                )
            ]
        )
        assert model_price(cache["q"]) == pytest.approx(0.26)
        assert model_price(cache["q"], quality="low") == pytest.approx(0.02)
        assert model_price(cache["q"], quality="LOW") == pytest.approx(0.02)

    async def test_inpaint_flat_and_tiered(self) -> None:
        cache = await _cached(
            [
                _simple("flat-edit", "inpaint", {"inpaint": _tier(0.04)}),
                _simple(
                    "tiered-edit",
                    "inpaint",
                    {
                        "inpaint": _tier(0.08),
                        "resolutions": {"1K": _tier(0.08), "2K": _tier(0.15)},
                        "quality": {"1K": {"low": _tier(0.04), "high": _tier(0.08)}},
                    },
                    constraints={
                        "promptCharacterLimit": 1500,
                        "qualities": ["low", "high"],
                        "defaultQuality": "high",
                        "resolutions": ["1K", "2K"],
                        "defaultResolution": "1K",
                    },
                ),
            ]
        )
        assert model_price(cache["flat-edit"]) == pytest.approx(0.04)
        assert model_price(cache["tiered-edit"]) == pytest.approx(0.08)
        assert model_price(cache["tiered-edit"], quality="low") == pytest.approx(0.04)

    async def test_music_reference_clip_pricing(self) -> None:
        cache = await _cached(
            [
                _simple("flat", "music", {"generation": _tier(0.04)}),
                _simple(
                    "per-second",
                    "music",
                    {"per_second": _tier(0.002)},
                    min_duration=1,
                    max_duration=600,
                ),
                _simple(
                    "short-max",
                    "music",
                    {"per_second": _tier(0.002)},
                    min_duration=1,
                    max_duration=10,
                ),
                _simple(
                    "tiers",
                    "music",
                    {
                        "durations": {
                            "60": {**_tier(0.03), "min_seconds": 60, "max_seconds": 60},
                            "90": {**_tier(0.04), "min_seconds": 61, "max_seconds": 90},
                        }
                    },
                    min_duration=60,
                    max_duration=90,
                ),
                _simple(
                    "tiers-from-3s",
                    "music",
                    {
                        "durations": {
                            "60": {**_tier(0.69), "min_seconds": 3, "max_seconds": 60},
                            "120": {**_tier(1.38), "min_seconds": 61, "max_seconds": 120},
                        }
                    },
                ),
                _simple("tts-like", "music", {"per_thousand_characters": _tier(0.05)}),
            ]
        )
        assert model_price(cache["flat"]) == pytest.approx(0.04)
        assert model_price(cache["per-second"]) == pytest.approx(0.06)  # 30 s
        assert model_price(cache["short-max"]) is None  # cannot make a 30 s clip
        assert model_price(cache["tiers"]) == pytest.approx(0.03)  # raised to its 60 s minimum
        assert model_price(cache["tiers-from-3s"]) == pytest.approx(0.69)
        assert model_price(cache["tts-like"]) is None

    def test_video_is_never_catalog_priced(self) -> None:
        model = {"id": "v", "type": "video", "model_spec": {"pricing": {"input": _tier(1)}}}
        assert model_price(model) is None


# ---------------------------------------------------------------------------
# cheapest_model_strategy / cheapest_selector
# ---------------------------------------------------------------------------


def _priced(model_id: str, usd: float | None) -> dict[str, Any]:
    pricing = {"input": _tier(usd)} if usd is not None else None
    return {"id": model_id, "type": "tts", "model_spec": {"pricing": pricing}}


class TestCheapestStrategy:
    def test_unpriced_sorts_last(self) -> None:
        candidates = [_priced("unpriced", None), _priced("dear", 9.0), _priced("cheap", 1.0)]
        assert cheapest_model_strategy(candidates) == "cheap"
        assert cheapest_model_strategy([_priced("unpriced", None), _priced("dear", 9.0)]) == "dear"

    def test_all_unpriced_follow_the_tie_rule(self) -> None:
        candidates = [_priced("b", None), _priced("a", None)]
        assert cheapest_model_strategy(candidates) == "b"
        candidates[1]["traits"] = ["default"]
        assert cheapest_model_strategy(candidates) == "a"

    def test_tie_break_prefers_default_trait_regardless_of_order(self) -> None:
        candidates = [_priced(mid, 0.5) for mid in ("delta", "alpha", "charlie", "bravo")]
        candidates[2]["traits"] = ["default", "fastest"]
        candidates.append(_priced("expensive", 2.0))
        candidates[-1]["traits"] = ["default"]
        rng = random.Random(7)
        for _ in range(10):
            rng.shuffle(candidates)
            assert cheapest_model_strategy(candidates) == "charlie"

    def test_tie_break_then_candidate_order(self) -> None:
        candidates = [_priced(mid, 0.5) for mid in ("delta", "alpha", "charlie")]
        assert cheapest_model_strategy(candidates) == "delta"
        assert cheapest_model_strategy(list(reversed(candidates))) == "charlie"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="No candidates"):
            cheapest_model_strategy([])

    def test_selector_lets_preferred_win(self) -> None:
        candidates = [_priced("cheap", 1.0), _priced("dear", 9.0)]
        select = cheapest_selector(preferred_models=["missing", "dear"])
        assert select(candidates) == "dear"
        assert cheapest_selector(preferred_models=["missing"])(candidates) == "cheap"

    def test_test_support_reexports(self) -> None:
        from venice_ai.test_support import cheapest_model_strategy as ts_cheapest
        from venice_ai.test_support import get_model_price

        assert ts_cheapest is cheapest_model_strategy
        assert get_model_price is model_price


# ---------------------------------------------------------------------------
# resolve(prefer="cheapest")
# ---------------------------------------------------------------------------

EMBEDDING_CATALOG = [
    _simple(
        "text-embedding-bge-m3",
        "embedding",
        {"input": _tier(0.15), "output": _tier(0.6)},
        traits=["default"],
    ),
    _simple("text-embedding-zeta", "embedding", {"input": _tier(0.0125), "output": _tier(0.0125)}),
    _simple("text-embedding-3-small", "embedding", {"input": _tier(0.025), "output": _tier(0.025)}),
    _simple("text-embedding-alpha", "embedding", {"input": _tier(0.0125), "output": _tier(0.0125)}),
    _entry("text-embedding-unpriced", "embedding", {}),
]

CHAT_CATALOG = [
    _chat("glm-big", 1.40, 4.40, traits=["default", "function_calling_default"], reasoning=True),
    _chat("e2ee-tiny", 0.01, 0.02, e2ee=True),
    _chat("beta-tiny", 0.02, 0.03, beta=True),
    _chat("unpriced-chat", None, None),
    _chat("small-reasoner", 0.08, 0.40, reasoning=True),
    _chat("small-chat", 0.06, 0.30),
    _chat("small-no-tools", 0.05, 0.20, function_calling=False),
]


class TestResolveCheapest:
    async def test_default_ranking_unchanged(self) -> None:
        resource = _resource(EMBEDDING_CATALOG)
        assert await resource.resolve_embedding() == "text-embedding-bge-m3"
        assert await resource.resolve(type="embedding") == "text-embedding-bge-m3"

    async def test_embedding_cheapest_tie_breaks_on_catalog_order(self) -> None:
        # zeta and alpha tie on price; zeta is listed first and neither is the default.
        resource = _resource(EMBEDDING_CATALOG)
        assert await resource.resolve_embedding(prefer="cheapest") == "text-embedding-zeta"

    async def test_preferred_model_still_wins(self) -> None:
        resource = _resource(EMBEDDING_CATALOG)
        model = await resource.resolve_embedding(
            prefer="cheapest", preferred_models=["text-embedding-3-small"]
        )
        assert model == "text-embedding-3-small"

    async def test_chat_skips_e2ee_and_beta(self) -> None:
        resource = _resource(CHAT_CATALOG)
        assert await resource.resolve_chat(prefer="cheapest") == "small-no-tools"

    async def test_chat_require_e2ee_readmits_e2ee(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(prefer="cheapest", require_e2ee=True)
        assert model == "e2ee-tiny"

    async def test_chat_exclude_beta_false_readmits_beta(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(prefer="cheapest", exclude_beta=False)
        assert model == "beta-tiny"

    async def test_chat_function_calling(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(
            prefer="cheapest", require_function_calling=True
        )
        assert model == "small-chat"

    async def test_chat_reasoning(self) -> None:
        model = await _resource(CHAT_CATALOG).resolve_chat(
            prefer="cheapest", require_reasoning=True
        )
        assert model == "small-reasoner"

    async def test_chat_default_without_prefer_unchanged(self) -> None:
        # Without prefer the e2ee/beta skip does not apply and the trait logic runs.
        assert await _resource(CHAT_CATALOG).resolve_chat(require_function_calling=True) == (
            "glm-big"
        )

    async def test_unknown_prefer_raises(self) -> None:
        with pytest.raises(ValueError, match="prefer"):
            await _resource(CHAT_CATALOG).resolve(type="chat", prefer="fastest")  # type: ignore[arg-type]

    async def test_image_default_quality_vs_requested_quality(self) -> None:
        tiered = {
            "resolutions": {"1K": _tier(0.27)},
            "quality": {"1K": {"high": _tier(0.26), "low": _tier(0.02)}},
        }
        catalog = [
            _image("pricey-default", {"generation": _tier(0.09)}, traits=["default"]),
            _image(
                "quality-tiered",
                tiered,
                qualities=["low", "high"],
                defaultQuality="high",
                resolutions=["1K"],
                defaultResolution="1K",
            ),
            _image("flat-cheap", {"generation": _tier(0.05)}),
        ]
        resource = _resource(catalog)
        assert await resource.resolve_image(prefer="cheapest") == "flat-cheap"
        model = await resource.resolve_image(prefer="cheapest", require_quality="low")
        assert model == "quality-tiered"

    async def test_image_tie_prefers_default_trait_over_id(self) -> None:
        catalog = [
            _image("chroma", {"generation": _tier(0.01)}, uncensored=True),
            _image("lustify-sdxl", {"generation": _tier(0.01)}, uncensored=True),
            _image("z-image-turbo", {"generation": _tier(0.01)}, traits=["default", "fastest"]),
            _image("flux-dear", {"generation": _tier(0.09)}),
        ]
        assert await _resource(catalog).resolve_image(prefer="cheapest") == "z-image-turbo"

    async def test_decision_is_exempt_from_beta_skip(self) -> None:
        catalog = [
            _simple(
                "jev-latest",
                "decision",
                {"input": _tier(0.042), "output": _tier(0)},
                betaModel=True,
            )
        ]
        assert await _resource(catalog).resolve_decision(prefer="cheapest") == "jev-latest"

    async def test_beta_skipped_for_simple_types_unless_asked(self) -> None:
        catalog = [
            _simple("tts-dear", "tts", {"input": _tier(50)}, traits=["default"]),
            _simple("tts-beta", "tts", {"input": _tier(1)}, betaModel=True),
            _simple("tts-cheap", "tts", {"input": _tier(3.5)}),
        ]
        resource = _resource(catalog)
        assert await resource.resolve_tts(prefer="cheapest") == "tts-cheap"
        assert await resource.resolve(type="tts", prefer="cheapest", exclude_beta=False) == (
            "tts-beta"
        )

    async def test_asr_and_inpaint(self) -> None:
        catalog = [
            _simple("asr-dear", "asr", {"per_audio_second": _tier(0.000167)}, traits=["default"]),
            _simple("asr-cheap", "asr", {"per_audio_second": _tier(0.00003)}),
            _simple("edit-dear", "inpaint", {"inpaint": _tier(0.34)}, traits=["default"]),
            _simple("edit-cheap", "inpaint", {"inpaint": _tier(0.02)}),
        ]
        resource = _resource(catalog)
        assert await resource.resolve_asr(prefer="cheapest") == "asr-cheap"
        assert await resource.resolve_inpaint(prefer="cheapest") == "edit-cheap"


# A catalog where the cheapest model reasons, as on the live catalog.
REASONING_CHEAPEST_CATALOG = [
    _chat("glm-big", 1.40, 4.40, traits=["default", "function_calling_default"], reasoning=True),
    _chat("plain-mid", 0.10, 0.40),
    _chat("tiny-reasoner", 0.05, 0.15, reasoning=True),
    _chat("plain-uncensored", 0.08, 0.30, uncensored=True),
    _chat("plain-no-tools", 0.09, 0.35, function_calling=False),
]


class TestStrictCheapestChat:
    async def test_cheapest_is_strict_price_order(self) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        assert await resource.resolve_chat(prefer="cheapest") == "tiny-reasoner"

    async def test_function_calling_matches_plain_path(self) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        model = await resource.resolve_chat(prefer="cheapest", require_function_calling=True)
        assert model == "tiny-reasoner"

    async def test_exclude_reasoning_filters_reasoning_models(self) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        assert await resource.resolve_chat(prefer="cheapest", exclude_reasoning=True) == (
            "plain-uncensored"
        )
        model = await resource.resolve_chat(
            prefer="cheapest", exclude_reasoning=True, require_function_calling=True
        )
        assert model == "plain-uncensored"

    async def test_exclude_reasoning_applies_without_prefer(self) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        model = await resource.resolve_chat(exclude_reasoning=True, require_function_calling=True)
        assert model == "plain-mid"

    async def test_exclude_uncensored(self) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        model = await resource.resolve_chat(
            prefer="cheapest", exclude_reasoning=True, exclude_uncensored=True
        )
        assert model == "plain-no-tools"

    @pytest.mark.parametrize(
        "conflict", [{"require_reasoning": True}, {"require_reasoning_effort": "none"}]
    )
    async def test_exclude_reasoning_conflicts_raise(self, conflict: dict[str, Any]) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        with pytest.raises(ValueError, match="exclude_reasoning"):
            await resource.resolve_chat(exclude_reasoning=True, **conflict)

    @pytest.mark.parametrize("prefer", [None, "cheapest"])
    async def test_preferred_reasoning_model_wins(self, prefer: Any) -> None:
        resource = _resource(REASONING_CHEAPEST_CATALOG)
        model = await resource.resolve_chat(prefer=prefer, preferred_models=["glm-big"])
        assert model == "glm-big"

    async def test_default_ranking_still_prefers_non_reasoning(self) -> None:
        # prefer=None keeps its documented general-chat preference for a
        # model that answers without spending the budget on reasoning.
        assert await _resource(REASONING_CHEAPEST_CATALOG).resolve_chat() == "plain-mid"


class TestImageFilters:
    CATALOG = [
        _image(
            "ratio-cheap",
            {"generation": _tier(0.005)},
            aspectRatios=["1:1", "16:9"],
            defaultAspectRatio="1:1",
        ),
        _image("pixel-uncensored", {"generation": _tier(0.01)}, uncensored=True),
        _image("pixel-default", {"generation": _tier(0.01)}, traits=["default"]),
        _image("pixel-dear", {"generation": _tier(0.03)}),
    ]

    async def test_require_custom_size_skips_ratio_sized_models(self) -> None:
        resource = _resource(self.CATALOG)
        assert await resource.resolve_image(prefer="cheapest") == "ratio-cheap"
        model = await resource.resolve_image(prefer="cheapest", require_custom_size=True)
        assert model == "pixel-default"

    async def test_exclude_uncensored(self) -> None:
        resource = _resource(self.CATALOG)
        model = await resource.resolve_image(
            prefer="cheapest",
            require_custom_size=True,
            exclude_uncensored=True,
            exclude_models=["pixel-default"],
        )
        assert model == "pixel-dear"


# ---------------------------------------------------------------------------
# Music: duration-aware cheapest quoting
# ---------------------------------------------------------------------------

MUSIC_CATALOG = [
    _simple("lyria-3-pro", "music", {"generation": _tier(0.1)}, traits=["default"]),
    _simple(
        "ace-step-15",
        "music",
        {"durations": {"60": {**_tier(0.03), "min_seconds": 60, "max_seconds": 60}}},
        min_duration=60,
        max_duration=210,
        duration_options=[60, 90, 120],
    ),
    _simple(
        "sonilo-v1-1-music",
        "music",
        {"per_second": _tier(0.002875)},
        min_duration=1,
        max_duration=600,
        default_duration=90,
    ),
    _simple("minimax-music-v2", "music", {"generation": _tier(0.04)}, lyrics_required=True),
    _simple(
        "minimax-music-v25",
        "music",
        {"generation": _tier(0.18)},
        supports_force_instrumental=True,
    ),
    _simple(
        "mmaudio-v2-text-to-audio",
        "music",
        {"per_second": _tier(0.00092)},
        min_duration=1,
        max_duration=30,
    ),
]


def _music_quote(model: str, duration_seconds: int | None = None) -> float:
    """Venice's per-request quote, rounded up to the cent as the API does."""
    import math

    if model == "sonilo-v1-1-music":
        return math.ceil(0.002875 * (duration_seconds or 90) * 100) / 100
    if model == "ace-step-15":
        return {60: 0.03, 90: 0.04, 120: 0.05}[duration_seconds or 60]
    return {"lyria-3-pro": 0.1, "minimax-music-v2": 0.04, "minimax-music-v25": 0.18}[model]


def _music_resource() -> tuple[Models, list[dict[str, Any]]]:
    resource = _resource(MUSIC_CATALOG)
    calls: list[dict[str, Any]] = []

    async def _quote(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return MagicMock(quote=_music_quote(kwargs["model"], kwargs.get("duration_seconds")))

    resource._client.music.quote = AsyncMock(side_effect=_quote)
    return resource, calls


class TestMusicRequestParams:
    def test_duration_options_pick_the_smallest_covering_option(self) -> None:
        from venice_ai.models.selection import music_request_params

        spec = {"duration_options": [60, 90, 120], "min_duration": 60, "max_duration": 120}
        assert music_request_params(spec, 10) == {"duration_seconds": 60}
        assert music_request_params(spec, 61) == {"duration_seconds": 90}
        assert music_request_params(spec) == {"duration_seconds": 60}
        with pytest.raises(ValueError, match="longest"):
            music_request_params(spec, 121)

    def test_range_models_take_the_target_or_their_minimum(self) -> None:
        from venice_ai.models.selection import music_request_params

        spec = {"min_duration": 5, "max_duration": 190}
        assert music_request_params(spec, 10) == {"duration_seconds": 10}
        assert music_request_params(spec, 2) == {"duration_seconds": 5}
        assert music_request_params(spec) == {"duration_seconds": 5}
        with pytest.raises(ValueError, match="longest"):
            music_request_params(spec, 200)

    def test_models_without_duration_metadata_get_no_duration(self) -> None:
        from venice_ai.models.selection import music_request_params
        from venice_ai.types.api.models import MusicModelSpec

        assert music_request_params({}, 10) == {}
        assert music_request_params(MusicModelSpec(name="m", min_duration=3), 10) == {
            "duration_seconds": 10
        }


class TestResolveCheapestMusic:
    async def test_target_duration_prefers_the_closest_length_on_a_tie(self) -> None:
        resource, calls = _music_resource()
        result = await resource.resolve_cheapest_music(duration_seconds=10)
        assert result.model == "sonilo-v1-1-music"
        assert result.quote_usd == pytest.approx(0.03)
        assert result.request_params == {"duration_seconds": 10}
        assert result.effective_seconds == 10
        assert result.all_quotes["ace-step-15"] == pytest.approx(0.03)
        quoted = {call["model"]: call for call in calls}
        assert quoted["ace-step-15"] == {"model": "ace-step-15", "duration_seconds": 60}
        assert quoted["lyria-3-pro"] == {"model": "lyria-3-pro"}
        assert "mmaudio-v2-text-to-audio" not in quoted
        assert "lyrics" in result.skipped["minimax-music-v2"]

    async def test_without_a_target_each_model_is_quoted_at_its_minimum(self) -> None:
        resource, calls = _music_resource()
        result = await resource.resolve_cheapest_music()
        assert result.model == "sonilo-v1-1-music"
        assert result.request_params == {"duration_seconds": 1}
        assert result.quote_usd == pytest.approx(0.01)

    async def test_too_long_for_a_model_skips_it(self) -> None:
        resource, calls = _music_resource()
        result = await resource.resolve_cheapest_music(duration_seconds=150)
        assert "ace-step-15" in result.skipped
        assert result.model == "lyria-3-pro"  # flat $0.10 beats sonilo's $0.44
        assert result.effective_seconds is None

    async def test_lyrics_supplied_admits_lyrics_required_models(self) -> None:
        resource, calls = _music_resource()
        await resource.resolve_cheapest_music(duration_seconds=10, lyrics_supplied=True)
        assert "minimax-music-v2" in {call["model"] for call in calls}

    async def test_force_instrumental(self) -> None:
        resource, _ = _music_resource()
        result = await resource.resolve_cheapest_music(
            duration_seconds=10, require_force_instrumental=True
        )
        assert result.model == "minimax-music-v25"
        assert result.request_params == {}

    async def test_no_generator_long_enough_is_no_match(self) -> None:
        from venice_ai import NoMatchingModelError

        resource, calls = _music_resource()
        with pytest.raises(NoMatchingModelError) as info:
            await resource.resolve_cheapest_music(
                duration_seconds=10_000, exclude_models=["lyria-3-pro", "minimax-music-v25"]
            )
        assert calls == []
        assert info.value.resource_type == "music"
        assert "longest" in info.value.skipped["sonilo-v1-1-music"]

    async def test_every_quote_failing_is_quotes_unavailable(self) -> None:
        from venice_ai import ModelQuotesUnavailableError
        from venice_ai.exceptions import InternalServerError

        resource, _ = _music_resource()
        denied = InternalServerError("503 Service Unavailable", response=None, body=None)
        resource._client.music.quote = AsyncMock(side_effect=denied)
        with pytest.raises(ModelQuotesUnavailableError) as info:
            await resource.resolve_cheapest_music(duration_seconds=10)
        err = info.value
        assert err.resource_type == "music"
        assert err.failures and all(exc is denied for exc in err.failures.values())
        assert "minimax-music-v2" in err.skipped  # lyrics required: never quoted
        assert "minimax-music-v2" not in err.failures
        assert "InternalServerError" in str(err)

    async def test_resolve_music_prefer_cheapest_uses_quotes(self) -> None:
        resource, _ = _music_resource()
        assert await resource.resolve_music(prefer="cheapest", duration_seconds=10) == (
            "sonilo-v1-1-music"
        )
        assert await resource.resolve_music() == "lyria-3-pro"


# ---------------------------------------------------------------------------
# Video: per-model cheapest quoting
# ---------------------------------------------------------------------------


def _video(
    model_id: str,
    *,
    durations: list[str],
    resolutions: list[str] | None = None,
    model_type: str = "text-to-video",
    audio: bool = False,
    audio_configurable: bool = False,
    aspect_ratios: list[str] | None = None,
    traits: list[str] | None = None,
) -> dict[str, Any]:
    return _entry(
        model_id,
        "video",
        {
            "traits": traits or [],
            "constraints": {
                "model_type": model_type,
                "aspect_ratios": ["9:16", "16:9"] if aspect_ratios is None else aspect_ratios,
                "resolutions": resolutions or [],
                "durations": durations,
                "audio": audio,
                "audio_configurable": audio_configurable,
                "video_input": model_type == "video",
            },
        },
    )


# Live-shaped: veo3 lists no 5 s clip, ltx starts at 6 s, minimax uses "768P",
# one model offers only "Auto", one is an upscaler (model_type "video").
VIDEO_CATALOG = [
    _video(
        "veo3-fast-text-to-video",
        durations=["4s", "6s", "8s"],
        resolutions=["1080p", "720p"],
        audio=True,
        audio_configurable=True,
        traits=["default"],
    ),
    _video("ltx-text-to-video", durations=["10s", "6s", "8s"], resolutions=["2160p", "1080p"]),
    _video(
        "minimax-text-to-video",
        durations=["5s", "6s"],
        resolutions=["2K", "768P"],
        audio=True,
        audio_configurable=True,
    ),
    _video("auto-only-text-to-video", durations=["Auto"]),
    _video("topaz-video-upscale", durations=[], resolutions=["2x", "4x"], model_type="video"),
    _video("kling-image-to-video", durations=["5s", "10s"], model_type="image-to-video"),
]

QUOTES = {
    "veo3-fast-text-to-video": 0.40,
    "ltx-text-to-video": 0.24,
    "minimax-text-to-video": 0.24,
    "kling-image-to-video": 0.35,
}


def _video_resource(
    entries: list[dict[str, Any]], quotes: dict[str, float] | None = None
) -> tuple[Models, list[dict[str, Any]]]:
    resource = _resource(entries)
    calls: list[dict[str, Any]] = []
    table = QUOTES if quotes is None else quotes

    async def _quote(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if kwargs["model"] not in table:
            raise RuntimeError("400 unsupported")
        return MagicMock(quote=table[kwargs["model"]])

    resource._client.video.quote = AsyncMock(side_effect=_quote)
    return resource, calls


class TestCheapestVideoParams:
    def test_shortest_duration_lowest_resolution_landscape(self) -> None:
        from venice_ai.models.selection import cheapest_video_params

        constraints = VIDEO_CATALOG[0]["model_spec"]["constraints"]
        assert cheapest_video_params(constraints) == {
            "duration_seconds": "4s",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "audio": False,
        }

    def test_accepts_typed_constraints_and_keeps_catalog_spelling(self) -> None:
        from venice_ai.models.selection import cheapest_video_params
        from venice_ai.types.api.models import VideoModelConstraints

        typed = VideoModelConstraints.model_validate(VIDEO_CATALOG[2]["model_spec"]["constraints"])
        params = cheapest_video_params(typed, audio=None)
        assert params == {"duration_seconds": "5s", "resolution": "768P", "aspect_ratio": "16:9"}

    def test_audio_only_sent_when_configurable(self) -> None:
        from venice_ai.models.selection import cheapest_video_params

        params = cheapest_video_params(VIDEO_CATALOG[1]["model_spec"]["constraints"])
        assert "audio" not in params

    def test_pins_and_rejections(self) -> None:
        from venice_ai.models.selection import cheapest_video_params

        constraints = VIDEO_CATALOG[0]["model_spec"]["constraints"]
        pinned = cheapest_video_params(constraints, duration=8, resolution="1080P")
        assert pinned["duration_seconds"] == "8s"
        assert pinned["resolution"] == "1080p"
        with pytest.raises(ValueError, match="duration"):
            cheapest_video_params(constraints, duration="5s")
        with pytest.raises(ValueError, match="resolution"):
            cheapest_video_params(constraints, resolution="4k")
        with pytest.raises(ValueError, match="no fixed durations"):
            cheapest_video_params(VIDEO_CATALOG[3]["model_spec"]["constraints"])
        with pytest.raises(ValueError, match="audio"):
            cheapest_video_params(VIDEO_CATALOG[1]["model_spec"]["constraints"], audio=True)


class TestResolveCheapestVideo:
    async def test_each_model_quoted_at_its_own_minimum(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        result = await resource.resolve_cheapest_video(video_type="text-to-video")

        by_model = {call["model"]: call for call in calls}
        assert set(by_model) == {
            "veo3-fast-text-to-video",
            "ltx-text-to-video",
            "minimax-text-to-video",
        }
        assert by_model["veo3-fast-text-to-video"] == {
            "model": "veo3-fast-text-to-video",
            "duration_seconds": "4s",
            "resolution": "720p",
            "aspect_ratio": "16:9",
            "audio": False,
        }
        assert by_model["ltx-text-to-video"]["duration_seconds"] == "6s"
        assert by_model["ltx-text-to-video"]["resolution"] == "1080p"
        assert "audio" not in by_model["ltx-text-to-video"]
        assert by_model["minimax-text-to-video"]["resolution"] == "768P"

        # ltx and minimax tie at $0.24; the id breaks the tie.
        assert result.model == "ltx-text-to-video"
        assert result.quote_usd == pytest.approx(0.24)
        assert result.request_params == {
            "duration_seconds": "6s",
            "resolution": "1080p",
            "aspect_ratio": "16:9",
        }
        assert "veo3-fast-text-to-video" in result.all_quotes
        assert "no fixed durations" in result.skipped["auto-only-text-to-video"]

    async def test_no_type_skips_upscalers(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        await resource.resolve_cheapest_video()
        quoted = {call["model"] for call in calls}
        assert "topaz-video-upscale" not in quoted
        assert "kling-image-to-video" in quoted

    async def test_failed_quote_is_recorded_not_fatal(self) -> None:
        quotes = {k: v for k, v in QUOTES.items() if k != "ltx-text-to-video"}
        resource, _ = _video_resource(VIDEO_CATALOG, quotes)
        result = await resource.resolve_cheapest_video(video_type="text-to-video")
        assert result.model == "minimax-text-to-video"
        assert result.skipped["ltx-text-to-video"].startswith("quote failed")

    async def test_pinned_duration_filters_instead_of_failing(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        result = await resource.resolve_cheapest_video(video_type="text-to-video", duration="5s")
        assert {call["model"] for call in calls} == {"minimax-text-to-video"}
        assert result.request_params["duration_seconds"] == "5s"

    async def test_audio_none_omits_audio(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        await resource.resolve_cheapest_video(video_type="text-to-video", audio=None)
        assert all("audio" not in call for call in calls)

    async def test_audio_true_requires_audio(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        result = await resource.resolve_cheapest_video(video_type="text-to-video", audio=True)
        assert {call["model"] for call in calls} == {
            "veo3-fast-text-to-video",
            "minimax-text-to-video",
        }
        assert all(call["audio"] is True for call in calls)
        assert result.model == "minimax-text-to-video"

    async def test_require_audio_configurable(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        await resource.resolve_cheapest_video(
            video_type="text-to-video", require_audio_configurable=True
        )
        assert "ltx-text-to-video" not in {call["model"] for call in calls}

    async def test_all_quotes_fail_raises_quotes_unavailable(self) -> None:
        from venice_ai import ModelQuotesUnavailableError, NoMatchingModelError

        resource, calls = _video_resource(VIDEO_CATALOG, quotes={})
        with pytest.raises(ModelQuotesUnavailableError) as info:
            await resource.resolve_cheapest_video(video_type="text-to-video")
        err = info.value
        assert isinstance(err, ValueError)
        assert not isinstance(err, NoMatchingModelError)
        assert err.resource_type == "video"
        assert set(err.failures) == {call["model"] for call in calls}
        assert all(isinstance(exc, RuntimeError) for exc in err.failures.values())
        assert "no fixed durations" in err.skipped["auto-only-text-to-video"]
        assert isinstance(err.__cause__, ExceptionGroup)
        assert list(err.__cause__.exceptions) == list(err.failures.values())
        assert "400 unsupported" in str(err)

    async def test_prefer_cheapest_surfaces_quotes_unavailable(self) -> None:
        from venice_ai import ModelQuotesUnavailableError

        resource, _ = _video_resource(VIDEO_CATALOG, quotes={})
        with pytest.raises(ModelQuotesUnavailableError):
            await resource.resolve_video(video_type="text-to-video", prefer="cheapest")

    async def test_no_servable_request_is_no_match_not_an_outage(self) -> None:
        from venice_ai import NoMatchingModelError

        resource, calls = _video_resource(VIDEO_CATALOG)
        with pytest.raises(NoMatchingModelError) as info:
            await resource.resolve_cheapest_video(video_type="text-to-video", aspect_ratio="21:9")
        assert calls == []
        assert info.value.resource_type == "video"

    async def test_unsatisfiable_filters_raise_no_match(self) -> None:
        from venice_ai import NoMatchingModelError

        resource, calls = _video_resource(VIDEO_CATALOG)
        with pytest.raises(NoMatchingModelError, match="No video models found"):
            await resource.resolve_cheapest_video(video_type="text-to-video", min_resolution="8K")
        assert calls == []

    async def test_concurrency_is_bounded(self) -> None:
        import asyncio

        many = [_video(f"model-{i:02d}-text-to-video", durations=["5s"]) for i in range(12)]
        resource = _resource(many)
        in_flight = 0
        peak = 0

        async def _quote(**kwargs: Any) -> Any:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return MagicMock(quote=1.0)

        resource._client.video.quote = AsyncMock(side_effect=_quote)
        result = await resource.resolve_cheapest_video(max_concurrency=3)
        assert peak == 3
        assert len(result.all_quotes) == 12
        assert result.model == "model-00-text-to-video"

    async def test_resolve_video_prefer_cheapest_uses_quotes(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        assert await resource.resolve_video() == "veo3-fast-text-to-video"
        assert calls == []
        model = await resource.resolve_video(video_type="text-to-video", prefer="cheapest")
        assert model == "ltx-text-to-video"

    async def test_resolve_video_prefer_cheapest_preferred_wins_without_quotes(self) -> None:
        resource, calls = _video_resource(VIDEO_CATALOG)
        model = await resource.resolve_video(
            prefer="cheapest", preferred_models=["minimax-text-to-video"]
        )
        assert model == "minimax-text-to-video"
        assert calls == []


# ---------------------------------------------------------------------------
# Video input modes (image-to-video models that need something else)
# ---------------------------------------------------------------------------

# (id, name, expected mode): live catalog ids and names, 2026-10-02.
INPUT_MODE_CASES = [
    ("grok-imagine-1-5-lite-image-to-video", "Grok Imagine 1.5 Lite", "image"),
    ("kling-o3-standard-image-to-video", "Kling O3 Standard", "image"),
    ("runway-gen4-turbo", "Runway Gen-4 Turbo", "image"),
    ("grok-imagine-reference-to-video-private", "Grok Imagine R2V", "reference"),
    ("kling-o3-standard-reference-to-video", "Kling O3 Standard R2V", "reference"),
    ("wan-3-0-reference-to-video", "Wan 3.0 Reference", "reference"),
    ("pixverse-c1-transition", "PixVerse C1 Transition", "transition"),
    ("flux-3-first-last-frame-to-video", "Flux 3 First Last Frame", "first_last_frame"),
    ("minimax-h3-max-multi-angle", "MiniMax H3 Max Multi-Angle", "multi_angle"),
]


@pytest.mark.parametrize(("model_id", "name", "mode"), INPUT_MODE_CASES)
def test_video_input_mode(model_id: str, name: str, mode: str) -> None:
    from venice_ai.models.selection import video_input_mode

    assert video_input_mode({"id": model_id, "name": name}) == mode


I2V_CATALOG = [
    _video(
        "grok-imagine-reference-to-video-private", durations=["1s"], model_type="image-to-video"
    ),
    _video("pixverse-c1-transition", durations=["3s"], model_type="image-to-video"),
    _video("grok-imagine-1-5-lite-image-to-video", durations=["1s"], model_type="image-to-video"),
    _video("veo3-fast-text-to-video", durations=["4s"]),
]
I2V_QUOTES = {
    "grok-imagine-reference-to-video-private": 0.03,
    "pixverse-c1-transition": 0.02,
    "grok-imagine-1-5-lite-image-to-video": 0.04,
    "veo3-fast-text-to-video": 0.40,
}


class TestImageToVideoInputModes:
    async def test_image_to_video_considers_only_plain_image_models(self) -> None:
        resource, calls = _video_resource(I2V_CATALOG, I2V_QUOTES)
        result = await resource.resolve_cheapest_video(video_type="image-to-video")
        assert result.model == "grok-imagine-1-5-lite-image-to-video"
        assert {call["model"] for call in calls} == {"grok-imagine-1-5-lite-image-to-video"}

    async def test_input_mode_selects_reference_or_transition_models(self) -> None:
        resource, _ = _video_resource(I2V_CATALOG, I2V_QUOTES)
        reference = await resource.resolve_cheapest_video(input_mode="reference")
        assert reference.model == "grok-imagine-reference-to-video-private"
        transition = await resource.resolve_video(input_mode="transition")
        assert transition == "pixverse-c1-transition"

    async def test_resolve_video_image_to_video_skips_reference_models(self) -> None:
        resource, _ = _video_resource(I2V_CATALOG, I2V_QUOTES)
        assert await resource.resolve_video(video_type="image-to-video") == (
            "grok-imagine-1-5-lite-image-to-video"
        )
