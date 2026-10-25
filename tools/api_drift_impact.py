"""Sort Venice API drift by its impact on venice-py.

``api_drift.py`` finds what changed in the spec; this module decides what each
change means for the SDK, using the SDK's own pydantic models:

- **Needs SDK work**: a gap the SDK provably has. A new endpoint, a request
  field no SDK request model has, or a new value on a request field the SDK
  restricts to a fixed set.
- **Review**: anything that cannot be decided from the models, such as a
  removed field the SDK may still send, a content-type or status-code change,
  query parameters, or an endpoint with no mapped request model.
- **No SDK change**: only what the SDK provably already handles. New values on
  request fields the SDK types as open strings, and new response fields or
  values, which response models accept by construction (enforced by
  ``tests/unit/type_defs/test_response_models_tolerant.py``).

Request fields are resolved against ``model_fields`` (and aliases) through
nested models, lists and maps. Do not replace this with a grep of the source:
ordinary words like ``noise`` or ``factor`` match unrelated docstrings and read
as covered when they are not modelled.

Run through ``api_drift.py``; this module needs venice-py importable.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_drift as drift  # noqa: E402

from venice_ai.types.api.augment import AugmentScrapeRequest, AugmentSearchRequest  # noqa: E402
from venice_ai.types.api.requests.api_keys import (  # noqa: E402
    CreateApiKeyRequest,
    UpdateApiKeyRequest,
    Web3CreateApiKeyRequest,
)
from venice_ai.types.api.requests.audio import (  # noqa: E402
    AudioSpeechRequest,
    AudioTranscriptionRequest,
)
from venice_ai.types.api.requests.chat import ChatCompletionRequest  # noqa: E402
from venice_ai.types.api.requests.decisions import CreateDecisionRequest  # noqa: E402
from venice_ai.types.api.requests.embeddings import EmbeddingsRequest  # noqa: E402
from venice_ai.types.api.requests.images import (  # noqa: E402
    ImageBackgroundRemoveRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultiEditRequest,
    ImageUpscaleRequest,
    SimpleImageGenerationRequest,
)
from venice_ai.types.api.requests.music import (  # noqa: E402
    MusicCompleteRequest,
    MusicQueueRequest,
    MusicQuoteRequest,
    MusicRetrieveRequest,
)
from venice_ai.types.api.requests.responses import ResponsesRequest  # noqa: E402
from venice_ai.types.api.requests.video import (  # noqa: E402
    VideoCompleteRequest,
    VideoImageToVideoRequest,
    VideoQuoteRequest,
    VideoRetrieveRequest,
    VideoTextToVideoRequest,
)
from venice_ai.types.api.requests.voice_changer import (  # noqa: E402
    CompleteVoiceChangerRequest,
    QueueVoiceChangerRequest,
    QuoteVoiceChangerRequest,
    RetrieveVoiceChangerRequest,
)

# Endpoint -> the request models the SDK builds its body from.
MODEL_MAP: dict[str, list[type[BaseModel]]] = {
    "PATCH /api_keys": [UpdateApiKeyRequest],
    "POST /api_keys": [CreateApiKeyRequest],
    "POST /api_keys/generate_web3_key": [Web3CreateApiKeyRequest],
    "POST /audio/complete": [MusicCompleteRequest],
    "POST /audio/queue": [MusicQueueRequest],
    "POST /audio/quote": [MusicQuoteRequest],
    "POST /audio/retrieve": [MusicRetrieveRequest],
    "POST /audio/speech": [AudioSpeechRequest],
    "POST /audio/transcriptions": [AudioTranscriptionRequest],
    "POST /audio/voice-changer/complete": [CompleteVoiceChangerRequest],
    "POST /audio/voice-changer/queue": [QueueVoiceChangerRequest],
    "POST /audio/voice-changer/quote": [QuoteVoiceChangerRequest],
    "POST /audio/voice-changer/retrieve": [RetrieveVoiceChangerRequest],
    "POST /augment/scrape": [AugmentScrapeRequest],
    "POST /augment/search": [AugmentSearchRequest],
    "POST /chat/completions": [ChatCompletionRequest],
    "POST /decisions": [CreateDecisionRequest],
    "POST /embeddings": [EmbeddingsRequest],
    "POST /image/background-remove": [ImageBackgroundRemoveRequest],
    "POST /image/edit": [ImageEditRequest],
    "POST /image/generate": [ImageGenerationRequest],
    "POST /image/multi-edit": [ImageMultiEditRequest],
    "POST /image/upscale": [ImageUpscaleRequest],
    "POST /images/generations": [SimpleImageGenerationRequest],
    "POST /responses": [ResponsesRequest],
    "POST /systemone": [CreateDecisionRequest],
    "POST /video/complete": [VideoCompleteRequest],
    "POST /video/queue": [VideoTextToVideoRequest, VideoImageToVideoRequest],
    "POST /video/quote": [VideoQuoteRequest],
    "POST /video/retrieve": [VideoRetrieveRequest],
}

# Endpoints with a request body but no request model to check it against.
UNMODELLED: dict[str, str] = {
    "POST /audio/voices": "the resource builds this multipart form by hand",
    "POST /augment/text-parser": "the resource builds this multipart form by hand",
    "POST /crypto/rpc/{network}": "a JSON-RPC body forwarded as the caller wrote it",
}

# Request fields the SDK deliberately does not model, by endpoint.
KNOWN_OMISSIONS: dict[str, dict[str, str]] = {
    "POST /image/generate": {
        "inpaint": "deprecated upstream and disabled on 2025-05-19",
        "negative_prompt": "ignored by modern image models; removed in v2.0.0",
    },
    "POST /image/edit": {
        "modelId": "the spec accepts `model` too, which is what the SDK sends",
    },
    "POST /audio/transcriptions": {
        "file": "sent as a multipart file by the resource, not a body field",
    },
    "POST /chat/completions": {
        "max_tokens": "deprecated upstream; the SDK raises a TypeError naming max_completion_tokens",
    },
}

RESPONSE_GATE = "tests/unit/type_defs/test_response_models_tolerant.py"

# Reached through a request model with ``extra="allow"``: the field is not typed,
# but the model passes it through, so callers can already send it.
PASSTHROUGH: Any = object()
_RANK = {drift.NEEDS_WORK: 0, drift.REVIEW: 1, drift.NO_CHANGE: 2}


# --------------------------------------------------------------------------- #
# Resolving a spec field path against the SDK's models
# --------------------------------------------------------------------------- #


def _arms(annotation: Any) -> list[Any]:
    """Flatten ``Annotated``, unions, ``Optional`` and type aliases."""
    value = getattr(annotation, "__value__", None)
    if value is not None:
        return _arms(value)
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _arms(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [arm for arg in typing.get_args(annotation) for arm in _arms(arg)]
    return [annotation]


def _is_open(arm: Any) -> bool:
    return arm is Any or arm is object


def _tokens(path: str) -> list[str]:
    return re.findall(r"\[\]|\{\}|[^.\[\]{}]+", path)


def _step(arm: Any, token: str) -> list[Any] | None:
    """The annotations reached from ``arm`` through ``token``; ``None`` if it ends there."""
    if arm is PASSTHROUGH:
        return [PASSTHROUGH]
    if _is_open(arm):
        return [Any]
    origin = typing.get_origin(arm)
    if token == "[]":
        if origin in (list, tuple, set, frozenset) or (
            origin is not None and getattr(origin, "__name__", "") in ("Sequence", "Iterable")
        ):
            args = typing.get_args(arm)
            return [args[0]] if args else [Any]
        return None
    if token == "{}":
        if arm is dict or origin is dict:
            args = typing.get_args(arm)
            return [args[1]] if len(args) == 2 else [Any]
        return None
    if arm is dict or origin is dict:
        # A named key on a plain dict is whatever the caller writes: untyped.
        return [PASSTHROUGH]
    if isinstance(arm, type) and issubclass(arm, BaseModel):
        for name, field in arm.model_fields.items():
            if token in (name, field.alias, field.validation_alias):
                return [field.annotation]
        if arm.model_config.get("extra") == "allow":
            return [PASSTHROUGH]
    return None


def resolve(models: list[type[BaseModel]], path: str) -> list[Any] | None:
    """The leaf annotations a body field path reaches, or ``None`` if no model has it."""
    current: list[Any] = list(models)
    for token in _tokens(path):
        reached: list[Any] = []
        for annotation in current:
            for arm in _arms(annotation):
                step = _step(arm, token)
                if step is not None:
                    reached.extend(step)
        if not reached:
            return None
        current = reached
    return current


def _open_type(leaves: list[Any]) -> bool:
    """Whether any leaf accepts an arbitrary string (not just a fixed set)."""
    return any(
        _is_open(arm)
        or (isinstance(arm, type) and issubclass(arm, str))
        or getattr(arm, "__supertype__", None) is str
        for annotation in leaves
        for arm in _arms(annotation)
    )


def _passthrough_only(leaves: list[Any]) -> bool:
    return all(leaf is PASSTHROUGH for leaf in leaves)


def _typed(leaves: list[Any]) -> list[Any]:
    """The leaves a typed SDK model declares, without untyped passthrough."""
    return [leaf for leaf in leaves if leaf is not PASSTHROUGH]


def _accepts(leaves: list[Any], value: Any) -> bool:
    """Whether a field typed with ``leaves`` accepts the JSON value ``value``."""
    for annotation in leaves:
        for arm in _arms(annotation):
            if _is_open(arm):
                return True
            if typing.get_origin(arm) is Literal:
                if value in typing.get_args(arm):
                    return True
                continue
            if arm is type(None) and value is None:
                return True
            if isinstance(arm, type):
                if arm is bool:
                    if isinstance(value, bool):
                        return True
                elif arm in (int, float) and isinstance(value, (int, float)):
                    if not isinstance(value, bool):
                        return True
                elif isinstance(value, str) and issubclass(arm, str):
                    return True
            elif isinstance(value, str) and getattr(arm, "__supertype__", None) is str:
                return True
    return False


# --------------------------------------------------------------------------- #
# Where each enum is used
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class Usage:
    endpoint: str
    where: str  # "body", "response" or "parameter"
    path: str


def enum_usages(spec: dict[str, Any]) -> dict[str, list[Usage]]:
    """Map each enum key (as ``api_drift.summarize`` names it) to the fields using it."""
    out: dict[str, list[Usage]] = {}

    def walk(
        schema: Any, key: str, path: str, usage: Usage, seen: frozenset[str], depth: int
    ) -> None:
        if depth > drift.MAX_DEPTH:
            return
        if isinstance(schema, dict) and "$ref" in schema:
            ref = schema["$ref"]
            if ref in seen or not isinstance(ref, str):
                return
            _, node = drift._resolve(ref, spec)
            if ref.startswith("#/components/schemas/"):
                key = f"schemas/{ref.rsplit('/', 1)[-1]}"
            walk(node, key, path, usage, seen | {ref}, depth + 1)
            return
        if not isinstance(schema, dict):
            return
        if isinstance(schema.get("enum"), list):
            out.setdefault(key, []).append(dataclasses.replace(usage, path=path))
        for keyword in ("allOf", "anyOf", "oneOf"):
            for branch in schema.get(keyword) or []:
                walk(branch, key, path, usage, seen, depth + 1)
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, sub in properties.items():
                walk(
                    sub,
                    drift._join(key, str(name)),
                    drift._join(path, str(name)),
                    usage,
                    seen,
                    depth + 1,
                )
        items = schema.get("items")
        if isinstance(items, dict):
            walk(items, f"{key}[]", f"{path}[]", usage, seen, depth + 1)
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            walk(additional, f"{key}{{}}", f"{path}{{}}", usage, seen, depth + 1)

    for raw_path, path_item in spec["paths"].items():
        if not isinstance(path_item, dict):
            continue
        for method in drift.HTTP_METHODS:
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            endpoint = f"{method.upper()} {raw_path}"
            for param in drift._operation_parameters(path_item, operation, spec, endpoint):
                location = f"{param['in']}.{param['name']}"
                usage = Usage(endpoint, "parameter", location)
                walk(param.get("schema"), f"{endpoint} {location}", "", usage, frozenset(), 0)
            body, _ = drift._deref(operation.get("requestBody"), spec, frozenset())
            for media in ((body or {}).get("content") or {}).values():
                if isinstance(media, dict):
                    usage = Usage(endpoint, "body", "")
                    walk(media.get("schema"), f"{endpoint} body", "", usage, frozenset(), 0)
            for status, raw in (operation.get("responses") or {}).items():
                if not str(status).startswith("2"):
                    continue
                response, _ = drift._deref(raw, spec, frozenset())
                for media in ((response or {}).get("content") or {}).values():
                    if isinstance(media, dict):
                        usage = Usage(endpoint, "response", "")
                        key = f"{endpoint} response.{status}"
                        walk(media.get("schema"), key, "", usage, frozenset(), 0)
    return out


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _body_path(param: str) -> str | None:
    """``body.images[].base64`` -> ``images[].base64``; ``None`` for non-body params."""
    if param.startswith("body."):
        return param[len("body.") :]
    return None


def _top(path: str) -> str:
    return _tokens(path)[0] if path else ""


def _with(item: drift.Item, group: str, reason: str) -> drift.Item:
    return dataclasses.replace(item, group=group, reason=reason)


def _classify_param_added(item: drift.Item) -> drift.Item:
    path = _body_path(item.detail)
    if path is None:
        return _with(item, drift.REVIEW, "Query, path and header parameters are not checked")
    if item.subject in UNMODELLED:
        return _with(item, drift.REVIEW, f"No request model to check: {UNMODELLED[item.subject]}")
    models = MODEL_MAP.get(item.subject)
    if models is None:
        return _with(item, drift.REVIEW, "Endpoint has no mapped SDK request model")
    leaves = resolve(models, path)
    if leaves is not None and _passthrough_only(leaves):
        return _with(
            item,
            drift.REVIEW,
            "Not typed by the SDK, but the request model passes unknown fields through, "
            "so it can already be sent",
        )
    if leaves is not None:
        return _with(item, drift.NO_CHANGE, "The SDK request model already has this field")
    omission = KNOWN_OMISSIONS.get(item.subject, {}).get(_top(path))
    if omission:
        return _with(item, drift.NO_CHANGE, f"Deliberately not modelled: {omission}")
    names = ", ".join(m.__name__ for m in models)
    return _with(item, drift.NEEDS_WORK, f"No SDK request model has this field ({names})")


def _classify_param_removed(item: drift.Item) -> drift.Item:
    path = _body_path(item.detail)
    models = MODEL_MAP.get(item.subject)
    if path is None or models is None:
        return _with(item, drift.REVIEW, "Cannot check whether the SDK still sends it")
    if resolve(models, path) is None:
        return _with(item, drift.NO_CHANGE, "The SDK does not send this field")
    return _with(item, drift.REVIEW, "The SDK still sends this field when it is set")


def _decode(member: str) -> Any:
    try:
        return json.loads(member)
    except ValueError:
        return member


def _classify_enum_added(item: drift.Item, usages: dict[str, list[Usage]]) -> drift.Item:
    found = usages.get(item.subject, [])
    if not found:
        return _with(item, drift.REVIEW, "No endpoint field uses this enum")
    values = [_decode(v) for v in item.values]
    verdicts: list[tuple[str, str]] = []
    for usage in found:
        if usage.where == "response":
            verdicts.append(
                (
                    drift.NO_CHANGE,
                    f"Response models accept new values (enforced by `{RESPONSE_GATE}`)",
                )
            )
            continue
        if usage.where == "parameter":
            verdicts.append((drift.REVIEW, f"Used by query/path parameter on {usage.endpoint}"))
            continue
        if usage.endpoint in UNMODELLED:
            verdicts.append((drift.REVIEW, f"{usage.endpoint} has no request model to check"))
            continue
        models = MODEL_MAP.get(usage.endpoint)
        if models is None:
            verdicts.append((drift.REVIEW, f"{usage.endpoint} has no mapped SDK request model"))
            continue
        leaves = resolve(models, usage.path)
        if leaves is None:
            omission = KNOWN_OMISSIONS.get(usage.endpoint, {}).get(_top(usage.path))
            if omission:
                verdicts.append((drift.NO_CHANGE, f"Deliberately not modelled: {omission}"))
            else:
                verdicts.append(
                    (drift.NEEDS_WORK, f"The SDK does not model `{usage.path}` on {usage.endpoint}")
                )
            continue
        if _passthrough_only(leaves):
            verdicts.append(
                (drift.REVIEW, f"`{usage.path}` on {usage.endpoint} is not typed by the SDK")
            )
            continue
        typed = _typed(leaves)
        rejected = [v for v in values if not _accepts(typed, v)]
        if rejected:
            shown = ", ".join(json.dumps(v) for v in rejected)
            verdicts.append(
                (
                    drift.NEEDS_WORK,
                    f"The SDK restricts `{usage.path}` on {usage.endpoint} to a fixed set "
                    f"that does not include {shown}",
                )
            )
        elif _open_type(typed):
            verdicts.append((drift.NO_CHANGE, f"The SDK accepts any value for `{usage.path}`"))
        else:
            verdicts.append(
                (drift.NO_CHANGE, f"The SDK already accepts these values for `{usage.path}`")
            )
    group, _ = min(verdicts, key=lambda v: _RANK[v[0]])
    reasons = sorted({reason for g, reason in verdicts if g == group})
    return _with(item, group, "; ".join(reasons))


def _classify_enum_removed(item: drift.Item, usages: dict[str, list[Usage]]) -> drift.Item:
    found = usages.get(item.subject, [])
    if found and all(u.where == "response" for u in found):
        return _with(item, drift.NO_CHANGE, "Only response fields use it")
    return _with(item, drift.REVIEW, "The SDK may still allow or send the removed values")


def classify(items: list[drift.Item], spec: dict[str, Any]) -> list[drift.Item]:
    """Assign every item a group and a reason. Returns one item per input item."""
    usages = enum_usages(spec)
    out: list[drift.Item] = []
    for item in items:
        if item.kind == "endpoint_added":
            out.append(_with(item, drift.NEEDS_WORK, "No SDK method calls this endpoint"))
        elif item.kind == "endpoint_removed":
            out.append(_with(item, drift.NEEDS_WORK, "SDK methods may still call this endpoint"))
        elif item.kind == "param_added":
            out.append(_classify_param_added(item))
        elif item.kind == "param_removed":
            out.append(_classify_param_removed(item))
        elif item.kind == "response_fields_added":
            out.append(
                _with(
                    item,
                    drift.NO_CHANGE,
                    f"Response models keep unknown fields (enforced by `{RESPONSE_GATE}`)",
                )
            )
        elif item.kind == "response_fields_removed":
            out.append(_with(item, drift.REVIEW, "An SDK response model may require these fields"))
        elif item.kind in ("enum_members_added", "enum_added"):
            out.append(_classify_enum_added(item, usages))
        elif item.kind in ("enum_members_removed", "enum_removed"):
            out.append(_classify_enum_removed(item, usages))
        else:
            out.append(
                _with(item, drift.REVIEW, "Content-type and status-code changes are not checked")
            )
    return out
