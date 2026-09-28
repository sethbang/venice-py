"""Text-model capability keys from the spec must reach the typed views.

``ModelCapabilities`` is ``extra="allow"``, so a capability key the SDK does not
declare parses without error and quietly lands on ``model_extra``. That keeps
``GET /models`` robust, but it also means an undeclared key never reaches
:class:`ChatCapabilities` and nothing notices. These tests pin the full set of
text-model capability properties from the published OpenAPI schema
(``ModelResponse.model_spec.capabilities``) and require each one to be a
declared wire field and to be surfaced on the typed chat view.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.models import Models
from venice_ai.types.api.capabilities import ChatCapabilities
from venice_ai.types.api.models import ModelCapabilities, ModelResponse, ModelsListResponse

#: ``ModelResponse.model_spec.capabilities.properties`` in the OpenAPI schema.
SPEC_TEXT_CAPABILITY_KEYS = frozenset(
    {
        "defaultReasoningEffort",
        "maxImages",
        "maxVideos",
        "optimizedForCode",
        "quantization",
        "reasoningEffortOptions",
        "supportsAudioInput",
        "supportsE2EE",
        "supportsFunctionCalling",
        "supportsLogProbs",
        "supportsMultipleImages",
        "supportsReasoning",
        "supportsReasoningEffort",
        "supportsResponseSchema",
        "supportsTeeAttestation",
        "supportsVideoInput",
        "supportsVision",
        "supportsWebSearch",
        "supportsXSearch",
    }
)

#: ``ModelCapabilities`` fields that intentionally have no ChatCapabilities
#: counterpart, with the reason.
NOT_CHAT_CAPABILITIES = {
    "supportsStyleReferences": "image-model style reference flag, not a chat capability",
    "supportsStyleReferenceStrength": "image-model style reference flag, not a chat capability",
}

#: Wire names whose snake_case form is not a plain camel-to-snake conversion.
CHAT_FIELD_RENAMES = {
    "supportsLogProbs": "supports_logprobs",
    "supportsE2EE": "supports_e2ee",
}

_SPEC_PATH = Path(__file__).resolve().parents[3] / ".metadata" / "swagger.yaml"


def _snake(name: str) -> str:
    if name in CHAT_FIELD_RENAMES:
        return CHAT_FIELD_RENAMES[name]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _text_model(capabilities: dict[str, Any]) -> ModelResponse:
    return ModelResponse.model_validate(
        {
            "id": "venice-uncensored-1-2",
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": "Venice Uncensored",
                "privacy": "private",
                "availableContextTokens": 32768,
                "capabilities": capabilities,
            },
        }
    )


def _models_resource(*entries: ModelResponse) -> Models:
    client = AsyncMock()
    resource = Models(client)
    listing = ModelsListResponse(object="list", type="all", data=list(entries))
    resource.list = AsyncMock(return_value=listing)  # type: ignore[method-assign]
    return resource


#: Capabilities block of a multi-image vision model as ``GET /models`` returns it.
_MULTI_IMAGE_CAPABILITIES: dict[str, Any] = {
    "optimizedForCode": False,
    "quantization": "fp8",
    "supportsFunctionCalling": True,
    "supportsReasoning": False,
    "supportsResponseSchema": True,
    "supportsVision": True,
    "supportsWebSearch": True,
    "supportsLogProbs": False,
    "supportsAudioInput": False,
    "supportsVideoInput": False,
    "supportsMultipleImages": True,
    "maxImages": 10,
    "supportsReasoningEffort": False,
    "supportsTeeAttestation": False,
    "supportsE2EE": False,
    "supportsXSearch": False,
}


@pytest.mark.asyncio
async def test_get_capabilities_reports_max_images() -> None:
    models = _models_resource(_text_model(_MULTI_IMAGE_CAPABILITIES))
    caps = await models.get_capabilities("venice-uncensored-1-2")
    assert isinstance(caps, ChatCapabilities)
    dumped = caps.model_dump()
    assert dumped.get("max_images") == 10, (
        "the model advertises maxImages=10 but ChatCapabilities does not expose it; "
        f"dumped keys: {sorted(dumped)}"
    )


def test_max_images_is_a_declared_wire_field() -> None:
    caps = ModelCapabilities.model_validate(_MULTI_IMAGE_CAPABILITIES)
    assert "maxImages" in ModelCapabilities.model_fields, (
        f"maxImages parsed only into model_extra={caps.model_extra!r}"
    )


def test_every_spec_capability_key_is_a_declared_wire_field() -> None:
    assert len(SPEC_TEXT_CAPABILITY_KEYS) > 0
    undeclared = sorted(SPEC_TEXT_CAPABILITY_KEYS - set(ModelCapabilities.model_fields))
    assert undeclared == [], f"spec capability keys that only land on model_extra: {undeclared}"


def test_every_wire_capability_is_surfaced_on_chat_capabilities() -> None:
    wire_fields = set(ModelCapabilities.model_fields) | SPEC_TEXT_CAPABILITY_KEYS
    assert len(wire_fields) > 0
    chat_fields = set(ChatCapabilities.model_fields)
    missing = sorted(
        name
        for name in wire_fields
        if name not in NOT_CHAT_CAPABILITIES and _snake(name) not in chat_fields
    )
    assert missing == [], (
        "text-model capabilities with no ChatCapabilities field (add one, or list the "
        f"key in NOT_CHAT_CAPABILITIES with a reason): {missing}"
    )


def test_not_chat_capabilities_allowlist_is_current() -> None:
    stale = sorted(set(NOT_CHAT_CAPABILITIES) - set(ModelCapabilities.model_fields))
    assert stale == [], f"allowlisted keys that no longer exist on ModelCapabilities: {stale}"


@pytest.mark.skipif(not _SPEC_PATH.exists(), reason="local OpenAPI schema not present")
def test_pinned_capability_keys_match_local_spec() -> None:
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(_SPEC_PATH.read_text())
    props = spec["components"]["schemas"]["ModelResponse"]["properties"]["model_spec"][
        "properties"
    ]["capabilities"]["properties"]
    assert set(props) == SPEC_TEXT_CAPABILITY_KEYS


#: Distinguishing values for non-boolean capability keys; booleans flip to True.
_NON_BOOL_VALUES: dict[str, Any] = {
    "quantization": "fp8",
    "maxImages": 7,
    "maxVideos": 3,
    "reasoningEffortOptions": ["low", "high"],
    "defaultReasoningEffort": "high",
}


def _baseline_capabilities() -> dict[str, Any]:
    baseline: dict[str, Any] = {
        name: False for name in SPEC_TEXT_CAPABILITY_KEYS if name not in _NON_BOOL_VALUES
    }
    baseline["quantization"] = "fp16"
    return baseline


@pytest.mark.asyncio
@pytest.mark.parametrize("wire_key", sorted(SPEC_TEXT_CAPABILITY_KEYS))
async def test_capability_value_reaches_chat_capabilities(wire_key: str) -> None:
    expected = _NON_BOOL_VALUES.get(wire_key, True)
    capabilities = {**_baseline_capabilities(), wire_key: expected}
    models = _models_resource(_text_model(capabilities))
    caps = await models.get_capabilities("venice-uncensored-1-2")
    field = _snake(wire_key)
    assert field in ChatCapabilities.model_fields, (
        f"{wire_key} has no ChatCapabilities field {field!r}"
    )
    assert getattr(caps, field) == expected, (
        f"{wire_key}={expected!r} on the wire but ChatCapabilities.{field}={getattr(caps, field)!r}"
    )
