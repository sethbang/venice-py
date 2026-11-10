"""Which pricing model each catalog shape parses to.

``ModelPricing`` is an undiscriminated union: pydantic picks the member that
sets the most declared fields. ``extra="allow"`` keeps unknown keys either way,
but ``isinstance`` checks and typed access depend on the choice, so the shapes
Venice publishes are pinned here. One entry per shape, from the 2026-09-28
catalog. The full dump lives under the untracked ``.metadata`` directory and is
parsed too when it is present.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from venice_ai.types.api.models import (
    ASRModelPricing,
    AudioModelPricing,
    ImageModelPricing,
    InpaintModelPricing,
    LLMModelPricing,
    ModelPricing,
    ModelResponse,
    MusicModelPricing,
    PricingTier,
    VideoResolutionPricing,
)

_TIER = {"usd": 0.01, "diem": 0.01}
_UPSCALE = {"2x": _TIER, "4x": _TIER}
_RESOLUTIONS = {"1K": _TIER, "2K": _TIER}
_QUALITY = {"1K": {"low": _TIER}}
_INPUT_IMAGES = {"included": 1, "additional": _TIER}

_DUMP = Path(__file__).parents[2] / ".metadata/models_dump_2026-09-28/models_all.json"

_ADAPTER: TypeAdapter[Any] = TypeAdapter(ModelPricing)

SHAPES = [
    ("llm", {"input": _TIER, "output": _TIER}, LLMModelPricing),
    (
        "llm cached",
        {"input": _TIER, "output": _TIER, "cache_input": _TIER, "cache_write": _TIER},
        LLMModelPricing,
    ),
    (
        "llm extended",
        {
            "input": _TIER,
            "output": _TIER,
            "extended": {"context_token_threshold": 1000, "input": _TIER, "output": _TIER},
        },
        LLMModelPricing,
    ),
    ("image flat", {"generation": _TIER, "upscale": _UPSCALE}, ImageModelPricing),
    ("image resolutions", {"resolutions": _RESOLUTIONS, "upscale": _UPSCALE}, ImageModelPricing),
    (
        "image quality",
        {"resolutions": _RESOLUTIONS, "quality": _QUALITY, "upscale": _UPSCALE},
        ImageModelPricing,
    ),
    ("inpaint flat", {"inpaint": _TIER}, InpaintModelPricing),
    ("inpaint resolutions", {"inpaint": _TIER, "resolutions": _RESOLUTIONS}, InpaintModelPricing),
    (
        "inpaint input images",
        {"inpaint": _TIER, "inputImages": _INPUT_IMAGES},
        InpaintModelPricing,
    ),
    (
        "inpaint full",
        {
            "inpaint": _TIER,
            "resolutions": _RESOLUTIONS,
            "quality": _QUALITY,
            "inputImages": _INPUT_IMAGES,
        },
        InpaintModelPricing,
    ),
    ("resolutions only", {"resolutions": _RESOLUTIONS}, VideoResolutionPricing),
    ("tts", {"input": _TIER}, AudioModelPricing),
    ("asr", {"per_audio_second": _TIER}, ASRModelPricing),
    ("music generation", {"generation": _TIER}, MusicModelPricing),
    ("music per second", {"per_second": _TIER}, MusicModelPricing),
    ("music characters", {"per_thousand_characters": _TIER}, MusicModelPricing),
    (
        "music durations",
        {"durations": {"60": {**_TIER, "min_seconds": 60, "max_seconds": 60}}},
        MusicModelPricing,
    ),
]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [(payload, expected) for _, payload, expected in SHAPES],
    ids=[name for name, _, _ in SHAPES],
)
def test_catalog_shape_parses_to_its_own_model(payload: dict[str, Any], expected: type) -> None:
    assert type(_ADAPTER.validate_python(payload)) is expected


def test_inpaint_resolution_prices_are_typed() -> None:
    pricing = _ADAPTER.validate_python(
        {
            "inpaint": _TIER,
            "resolutions": _RESOLUTIONS,
            "quality": _QUALITY,
            "inputImages": _INPUT_IMAGES,
        }
    )
    assert isinstance(pricing, InpaintModelPricing)
    assert pricing.resolutions is not None
    assert pricing.resolutions["2K"].usd == 0.01
    assert pricing.quality == _QUALITY
    assert pricing.inputImages == _INPUT_IMAGES
    assert pricing.model_extra == {}


@pytest.mark.parametrize(
    "field",
    [
        {"quality": ["x"]},
        {"quality": "x"},
        {"resolutions": {"1K": 5}},
        {"resolutions": ["1K"]},
        {"inputImages": 2},
    ],
    ids=[
        "quality list",
        "quality string",
        "resolution number",
        "resolutions list",
        "images number",
    ],
)
def test_an_unforeseen_inpaint_field_does_not_change_the_pricing_model(
    field: dict[str, Any],
) -> None:
    parsed = _ADAPTER.validate_python({"inpaint": _TIER, **field})

    assert isinstance(parsed, InpaintModelPricing)
    assert parsed.inpaint.usd == _TIER["usd"]
    name, value = next(iter(field.items()))
    assert parsed.model_dump()[name] == value


def test_unknown_pricing_keys_stay_on_the_inpaint_model() -> None:
    pricing = _ADAPTER.validate_python({"inpaint": _TIER, "vcu": 3})
    assert isinstance(pricing, InpaintModelPricing)
    assert pricing.model_extra == {"vcu": 3}


def test_unforeseen_shape_still_parses() -> None:
    pricing = _ADAPTER.validate_python({"per_frame": _TIER})
    assert isinstance(pricing, MusicModelPricing)
    assert pricing.model_extra == {"per_frame": _TIER}


@pytest.mark.parametrize(
    "payload",
    [
        {"input": {"usd": 1.0}, "output": {"usd": 2.0}},
        {"input": {"usd": 1.0, "diem": 1.0}, "output": {"usd": 2.0}},
    ],
    ids=["neither has diem", "output has no diem"],
)
def test_llm_pricing_without_diem_stays_llm(payload: dict[str, Any]) -> None:
    pricing = _ADAPTER.validate_python(payload)
    assert isinstance(pricing, LLMModelPricing)
    assert pricing.output.usd == 2.0
    assert pricing.output.diem is None


def test_tts_pricing_without_diem_stays_audio() -> None:
    assert isinstance(_ADAPTER.validate_python({"input": {"usd": 1.0}}), AudioModelPricing)


def test_tier_without_usd_is_not_an_llm_price() -> None:
    pricing = _ADAPTER.validate_python({"input": {"diem": 1.0}, "output": {"diem": 1.0}})
    assert not isinstance(pricing, LLMModelPricing)


def test_tier_subscript_returns_none_for_an_absent_diem() -> None:
    assert PricingTier(usd=1.0)["diem"] is None


@pytest.mark.skipif(not _DUMP.exists(), reason="the catalog dump is local-only")
def test_every_entry_of_the_catalog_dump_parses_to_the_model_of_its_type() -> None:
    entries = json.loads(_DUMP.read_text())["data"]
    assert len(entries) == 367

    expected = {
        "text": LLMModelPricing,
        "embedding": LLMModelPricing,
        "decision": LLMModelPricing,
        "image": ImageModelPricing,
        "upscale": ImageModelPricing,
        "inpaint": InpaintModelPricing,
        "tts": AudioModelPricing,
        "asr": ASRModelPricing,
        "music": MusicModelPricing,
        "video": type(None),
    }
    wrong = []
    seen: Counter[str] = Counter()
    for entry in entries:
        seen[entry["type"]] += 1
        pricing = ModelResponse.model_validate(entry).model_spec.pricing
        if type(pricing) is not expected[entry["type"]]:
            wrong.append((entry["id"], entry["type"], type(pricing).__name__))
        if pricing is not None:
            assert pricing.model_dump(by_alias=True, exclude_none=True) == _without_none(
                entry["model_spec"]["pricing"]
            ), entry["id"]
    assert wrong == []
    assert seen["inpaint"] == 24


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    return value
