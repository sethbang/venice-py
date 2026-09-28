"""Video request models must not require what the API spec leaves optional.

A field the SDK marks required but the spec does not makes a valid request
impossible to build: the call fails client-side before it ever reaches the
server. The snapshot below records the ``required`` list of every video
request schema (and of every object nested inside one) as published in the
Venice OpenAPI spec. Each SDK model mapped to a schema may require only a
subset of it.

When a local copy of the spec is present at ``.metadata/swagger.yaml`` the
snapshot is also checked against it, so the table cannot silently drift.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from venice_ai.resources.video import Video
from venice_ai.types.api.requests.video import (
    CameraKeyframe,
    SeedanceConsents,
    VideoCompleteRequest,
    VideoConsents,
    VideoElement,
    VideoImageToVideoRequest,
    VideoKeyframe,
    VideoQuoteRequest,
    VideoRetrieveRequest,
    VideoTextToVideoRequest,
)

# (schema location in the spec) -> spec ``required`` list
SPEC_REQUIRED: dict[str, frozenset[str]] = {
    "QueueVideoRequest": frozenset({"model", "duration"}),
    "QueueVideoRequest.elements[]": frozenset(),
    "QueueVideoRequest.keyframes[]": frozenset({"image_url", "frame_index"}),
    "QueueVideoRequest.camera_trajectory[]": frozenset(
        {"time", "azimuth", "elevation", "distance"}
    ),
    "QueueVideoRequest.consents": frozenset(),
    "QueueVideoRequest.consents.seedance": frozenset(
        {
            "confirmed_terms_and_privacy",
            "confirmed_legal_right",
            "confirmed_screening_acknowledged",
        }
    ),
    "QuoteVideoRequest": frozenset({"model", "duration"}),
    "RetrieveVideoRequest": frozenset({"model", "queue_id"}),
    "CompleteVideoRequest": frozenset({"model", "queue_id"}),
}

# SDK model -> (spec location, fields the SDK may additionally require)
# ``image_url`` on the image-to-video model is the discriminator the SDK uses
# to pick that model; it is only ever built when ``image_url`` was supplied.
MODEL_TO_SPEC: dict[type[BaseModel], tuple[str, frozenset[str]]] = {
    VideoTextToVideoRequest: ("QueueVideoRequest", frozenset()),
    VideoImageToVideoRequest: ("QueueVideoRequest", frozenset({"image_url"})),
    VideoElement: ("QueueVideoRequest.elements[]", frozenset()),
    VideoKeyframe: ("QueueVideoRequest.keyframes[]", frozenset()),
    CameraKeyframe: ("QueueVideoRequest.camera_trajectory[]", frozenset()),
    VideoConsents: ("QueueVideoRequest.consents", frozenset()),
    SeedanceConsents: ("QueueVideoRequest.consents.seedance", frozenset()),
    VideoQuoteRequest: ("QuoteVideoRequest", frozenset()),
    VideoRetrieveRequest: ("RetrieveVideoRequest", frozenset()),
    VideoCompleteRequest: ("CompleteVideoRequest", frozenset()),
}


def _over_required(model: type[BaseModel]) -> set[str]:
    location, allowance = MODEL_TO_SPEC[model]
    sdk_required = {
        (info.alias or name) for name, info in model.model_fields.items() if info.is_required()
    }
    return sdk_required - SPEC_REQUIRED[location] - allowance


def test_mapping_is_not_empty():
    assert len(MODEL_TO_SPEC) > 0
    assert all(loc in SPEC_REQUIRED for loc, _ in MODEL_TO_SPEC.values())


@pytest.mark.parametrize("model", list(MODEL_TO_SPEC), ids=lambda m: m.__name__)
def test_sdk_requires_no_more_than_spec(model: type[BaseModel]):
    extra = _over_required(model)
    assert not extra, (
        f"{model.__name__} requires {sorted(extra)}, which the spec "
        f"({MODEL_TO_SPEC[model][0]}) leaves optional"
    )


# Parameters of the public video methods that map onto spec fields. A method
# parameter without a default is required at the call site.
_PARAM_TO_WIRE = {"duration_seconds": "duration"}


@pytest.mark.parametrize(
    ("method", "location"),
    [
        (Video.submit, "QueueVideoRequest"),
        (Video.run, "QueueVideoRequest"),
        (Video.quote, "QuoteVideoRequest"),
        (Video.retrieve, "RetrieveVideoRequest"),
        (Video.cancel, "CompleteVideoRequest"),
    ],
    ids=["submit", "run", "quote", "retrieve", "cancel"],
)
def test_public_method_requires_no_more_than_spec(method: Any, location: str):
    params = inspect.signature(method).parameters.values()
    required = {
        _PARAM_TO_WIRE.get(p.name, p.name)
        for p in params
        if p.name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    assert required, "signature introspection found no required parameters"
    extra = required - SPEC_REQUIRED[location]
    assert not extra, (
        f"Video.{method.__name__} requires {sorted(extra)}, which the spec ({location}) leaves optional"
    )


# ---------------------------------------------------------------------------
# Snapshot vs local spec copy
# ---------------------------------------------------------------------------

_SPEC_PATH = Path(__file__).resolve().parents[3] / ".metadata" / "swagger.yaml"


def _resolve(spec: dict, node: dict) -> dict:
    ref = node.get("$ref")
    if ref:
        return spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    return node


def _lookup(spec: dict, location: str) -> dict:
    head, *rest = location.split(".")
    node = _resolve(spec, spec["components"]["schemas"][head])
    for part in rest:
        is_array = part.endswith("[]")
        node = _resolve(spec, node["properties"][part.removesuffix("[]")])
        if is_array:
            node = _resolve(spec, node["items"])
    return node


@pytest.mark.skipif(not _SPEC_PATH.exists(), reason="local spec copy not present")
@pytest.mark.parametrize("location", sorted(SPEC_REQUIRED))
def test_snapshot_matches_local_spec(location: str):
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    assert frozenset(_lookup(spec, location).get("required", [])) == SPEC_REQUIRED[location]
