"""HDR in video model selection: ``require_hdr`` and cheapest quoting with ``hdr``.

``GET /models`` publishes ``supportsHdr`` and ``hdrResolutions`` under a video
model's constraints. The API answers ``hdr=True`` with a 400 on a model without
HDR or at a resolution outside ``hdrResolutions``, so selection has to check
both, and the cheapest pick has to be quoted with HDR on to include its
surcharge.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.unit.resources.test_models_resolve_cheapest import _entry, _resource
from venice_ai.models.selection import cheapest_video_params
from venice_ai.resources.models import Models
from venice_ai.types.api.models import VideoModelConstraints

pytestmark = pytest.mark.asyncio


def _video(
    model_id: str,
    *,
    resolutions: list[str],
    hdr_resolutions: list[str] | None = None,
    traits: list[str] | None = None,
) -> dict[str, Any]:
    constraints: dict[str, Any] = {
        "model_type": "text-to-video",
        "aspect_ratios": ["16:9"],
        "resolutions": resolutions,
        "durations": ["5s", "10s"],
        "audio": False,
        "audio_configurable": False,
        "video_input": False,
    }
    if hdr_resolutions is not None:
        constraints["supportsHdr"] = True
        constraints["hdrResolutions"] = hdr_resolutions
    return _entry(model_id, "video", {"traits": traits or [], "constraints": constraints})


# sdr-default has no HDR and is the catalog default; hdr-4k supports HDR at 4k
# only; hdr-wide supports it at 1080p and 4k (spelled "1080P" in resolutions).
CATALOG = [
    _video("sdr-default", resolutions=["720p", "1080p"], traits=["default"]),
    _video("hdr-4k", resolutions=["1080p", "4k"], hdr_resolutions=["4k"]),
    _video("hdr-wide", resolutions=["720p", "1080P", "4k"], hdr_resolutions=["1080p", "4k"]),
]
QUOTES = {"sdr-default": 0.10, "hdr-4k": 0.50, "hdr-wide": 0.80}


def _quoting(quotes: dict[str, float] = QUOTES) -> tuple[Models, list[dict[str, Any]]]:
    resource = _resource(CATALOG)
    calls: list[dict[str, Any]] = []

    async def _quote(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return MagicMock(quote=quotes[kwargs["model"]])

    resource._client.video.quote = AsyncMock(side_effect=_quote)
    return resource, calls


def _constraints(index: int) -> dict[str, Any]:
    constraints: dict[str, Any] = CATALOG[index]["model_spec"]["constraints"]
    return constraints


class TestCheapestVideoParamsHdr:
    async def test_lowest_hdr_resolution_in_catalog_spelling(self) -> None:
        params = cheapest_video_params(_constraints(2), hdr=True)
        assert params["hdr"] is True
        assert params["resolution"] == "1080P"

    async def test_typed_constraints(self) -> None:
        typed = VideoModelConstraints.model_validate(_constraints(1))
        assert cheapest_video_params(typed, hdr=True)["resolution"] == "4k"

    async def test_pinned_resolution_must_accept_hdr(self) -> None:
        assert cheapest_video_params(_constraints(1), resolution="4K", hdr=True)["resolution"] == (
            "4k"
        )
        with pytest.raises(ValueError, match="does not support HDR"):
            cheapest_video_params(_constraints(1), resolution="1080p", hdr=True)

    async def test_model_without_hdr_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not support HDR"):
            cheapest_video_params(_constraints(0), hdr=True)

    async def test_hdr_flag_without_resolutions_is_refused(self) -> None:
        constraints = {"durations": ["5s"], "supportsHdr": True}
        with pytest.raises(ValueError, match="no HDR resolutions"):
            cheapest_video_params(constraints, hdr=True)

    async def test_hdr_off_sends_nothing(self) -> None:
        assert "hdr" not in cheapest_video_params(_constraints(2))


class TestRequireHdr:
    async def test_resolve_video_skips_models_without_hdr(self) -> None:
        model = await _resource(CATALOG).resolve_video(require_hdr=True)
        assert model in {"hdr-4k", "hdr-wide"}
        assert await _resource(CATALOG).resolve_video() == "sdr-default"

    async def test_required_resolution_must_accept_hdr(self) -> None:
        # hdr-4k lists 1080p, but not as an HDR resolution.
        model = await _resource(CATALOG).resolve_video(require_hdr=True, require_resolution="1080p")
        assert model == "hdr-wide"

    async def test_min_resolution_counts_hdr_resolutions_only(self) -> None:
        catalog = [
            _video("hdr-low", resolutions=["720p", "4k"], hdr_resolutions=["720p"]),
            _video("hdr-high", resolutions=["4k"], hdr_resolutions=["4k"]),
        ]
        model = await _resource(catalog).resolve_video(require_hdr=True, min_resolution="1080p")
        assert model == "hdr-high"

    async def test_resolve_passes_it_through(self) -> None:
        model = await _resource(CATALOG).resolve(
            type="video", require_hdr=True, require_resolution="1080p"
        )
        assert model == "hdr-wide"

    async def test_no_hdr_model_raises(self) -> None:
        with pytest.raises(ValueError, match="require_hdr=True"):
            await _resource(CATALOG[:1]).resolve_video(require_hdr=True)


class TestCheapestVideoHdr:
    async def test_quotes_hdr_models_with_hdr_on(self) -> None:
        resource, calls = _quoting()
        result = await resource.resolve_cheapest_video(hdr=True)
        assert {call["model"] for call in calls} == {"hdr-4k", "hdr-wide"}
        assert all(call["hdr"] is True for call in calls)
        assert {call["model"]: call["resolution"] for call in calls} == {
            "hdr-4k": "4k",
            "hdr-wide": "1080P",
        }
        assert result.model == "hdr-4k"
        assert result.request_params["hdr"] is True
        # Filtered up front, like require_audio, not reported as a skipped candidate.
        assert "sdr-default" not in result.skipped

    async def test_default_quotes_without_hdr(self) -> None:
        resource, calls = _quoting()
        result = await resource.resolve_cheapest_video()
        assert all("hdr" not in call for call in calls)
        assert result.model == "sdr-default"

    async def test_pinned_resolution_filters_on_hdr_resolutions(self) -> None:
        resource, calls = _quoting()
        result = await resource.resolve_cheapest_video(hdr=True, resolution="1080p")
        assert [call["model"] for call in calls] == ["hdr-wide"]
        assert result.model == "hdr-wide"

    async def test_resolve_prefer_cheapest_quotes_with_hdr(self) -> None:
        resource, calls = _quoting()
        model = await resource.resolve(type="video", require_hdr=True, prefer="cheapest")
        assert model == "hdr-4k"
        assert calls and all(call["hdr"] is True for call in calls)
