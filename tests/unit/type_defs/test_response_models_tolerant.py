"""Response models accept fields and values the SDK has not seen.

A response model is validated against whatever the server sends. When the
server adds a field, a status, a content-part type or an out-of-range number,
any of these makes the *whole* response fail to parse, discarding everything
else the call returned:

- ``extra="forbid"`` (``VeniceBaseModel``'s default, which request models rely
  on to catch typos);
- a single-value ``Literal`` on a field that is not a union tag;
- a numeric or length constraint (``ge``, ``le``, ``min_length``, ...);
- a union of tagged models with no catch-all arm.

The models checked here are every model reachable from what a resource method
returns, plus the model-spec subclasses ``ModelResponse`` picks at validation
time. ``test_every_server_model_is_checked`` makes sure a new response model
cannot slip past by not being reachable. Closed multi-value ``Literal`` fields
are covered by ``test_response_outcome_fields_open.py``.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import pkgutil
import re
import types
import typing
from typing import Annotated, Any, Literal

import annotated_types
import pytest
from pydantic import BaseModel, ConfigDict, Field

import venice_ai
import venice_ai.resources
import venice_ai.types
from venice_ai._resource import APIResource
from venice_ai.types.api import models as api_models

# Single-value ``Literal`` fields that are not union tags, kept on purpose.
LITERAL_EXEMPTIONS: dict[str, str] = {
    "venice_ai.types.api.x402.X402BalanceResponse.success": (
        "a body with success=false is an error envelope, not a balance"
    ),
    "venice_ai.types.api.x402.X402TopUpResponse.success": (
        "a body with success=false is an error envelope, not a top-up receipt"
    ),
    "venice_ai.types.api.x402.X402TransactionsResponse.success": (
        "a body with success=false is an error envelope, not a transaction page"
    ),
}

# Non-request models that never hold server data. Discovery stops at them, and
# they satisfy ``test_every_server_model_is_checked``.
NOT_FROM_SERVER: dict[str, str] = {
    "venice_ai.types.api.api_keys.ApiKeysListResponse": (
        "exported body shape only; api_keys.list() validates each item into ApiKey"
    ),
    "venice_ai.types.api.api_keys.CreateApiKeyResponse": (
        "exported body shape only; api_keys.create() validates the key into CreatedApiKey"
    ),
    "venice_ai.types.api.audio.VoiceList": "built by audio.get_voices() from the model catalog",
    "venice_ai.types.api.audio.VoiceDetail": "built by audio.get_voices() from the model catalog",
    "venice_ai.types.api.capabilities.ChatCapabilities": "derived client-side from a model spec",
    "venice_ai.types.api.capabilities.ImageCapabilities": "derived client-side from a model spec",
    "venice_ai.types.api.capabilities.InpaintCapabilities": "derived client-side from a model spec",
    "venice_ai.types.api.capabilities.VideoCapabilities": "derived client-side from a model spec",
    "venice_ai.types.api.base.BaseListResponse": "not used by any resource",
    "venice_ai.types.api.base.BaseSuccessResponse": "not used by any resource",
    "venice_ai.types.api.base.DetailedError": "not used by any resource or the error parser",
    "venice_ai.types.api.base.StandardError": "not used by any resource or the error parser",
    "venice_ai.types.api.base.UsageData": "re-exported for callers; no resource parses into it",
    "venice_ai.types.api.common.ErrorDetails": "not used by any resource or the error parser",
}

# Modules whose models are configuration or bookkeeping the SDK fills in itself.
LOCAL_MODULES: dict[str, str] = {
    "venice_ai.core.config.": "client configuration the caller supplies",
    "venice_ai.core.models.metrics": "scheduler metrics the SDK records",
    "venice_ai.costs": "cost tracking the SDK computes",
    "venice_ai.observability.": "observability settings the caller supplies",
}

NOT_FROM_SERVER.update(
    {
        "venice_ai.core.models.base.VeniceBaseModel": "base class; subclasses are checked",
        "venice_ai.core.models.base.TimestampMixin": "mixin; models using it are checked",
        "venice_ai.core.models.common.AudioContent": "request content part",
        "venice_ai.core.models.common.ConsumptionLimit": "request-side API key limit",
        "venice_ai.core.models.common.FileContent": "request content part",
        "venice_ai.core.models.common.FileObject": "request content part payload",
        "venice_ai.core.models.common.VideoContent": "request content part",
        "venice_ai.core.models.common.JSONObjectFormat": "request response_format",
        "venice_ai.core.models.common.JSONSchemaFormat": "request response_format",
        "venice_ai.core.models.common.TextResponseFormat": "request response_format",
        "venice_ai.core.models.common.SpecificToolChoice": "request tool_choice",
        "venice_ai.core.models.common.ToolChoiceFunction": "request tool_choice",
        "venice_ai.core.models.common.Tool": "request tool definition",
        "venice_ai.core.models.common.ToolFunction": "request tool definition",
        "venice_ai.core.models.common.StreamOptions": "request stream_options",
        "venice_ai.core.models.common.VeniceParameters": "request venice_parameters",
        "venice_ai.core.models.common.HealthCheckResult": "built by the rate-limit backends",
        "venice_ai.core.models.common.ValidationResult": "not used by any resource",
        "venice_ai.core.models.common.ListResponse": "not used by any resource",
        "venice_ai.core.models.common.SuccessResponse": "not used by any resource",
        "venice_ai.core.models.common.RequestEcho": "not used by any resource",
        "venice_ai.core.rate_limit_discovery.RateLimitBucket": (
            "built by TierDiscovery from rate limits it sanitizes first"
        ),
        "venice_ai.tee.types.TeeVerificationResult": "built by the attestation verifier",
    }
)

# Built by the SDK from response header values it parses and picks itself, so
# no unknown field can reach them, but each value still comes from the server.
HEADER_MODELS: tuple[str, ...] = (
    "venice_ai.core.models.headers.BalanceInfo",
    "venice_ai.core.models.headers.ContentSafetyInfo",
    "venice_ai.core.models.headers.DeprecationInfo",
    "venice_ai.core.models.headers.ModelInfo",
    "venice_ai.core.models.headers.PaginationInfo",
    "venice_ai.core.models.headers.RateLimitInfo",
)

# Request models that appear as union arms in a response field only so that
# instances the caller built are kept. Server data never reaches them: the
# union is left_to_right and the response-side arms come first.
CALLER_BUILT_ARMS: dict[str, str] = {
    "venice_ai.core.models.common.TextContent": (
        "ChatMessageContentPart arm for caller-built parts; API text parts parse "
        "into ChatMessageTextContent first"
    ),
    "venice_ai.core.models.common.ImageContent": (
        "ChatMessageContentPart arm for caller-built parts; API image parts parse "
        "into ChatMessageImageContent first"
    ),
    "venice_ai.core.models.common.ImageUrl": "only reached through the ImageContent arm above",
}

_CONSTRAINTS = (
    annotated_types.Ge,
    annotated_types.Gt,
    annotated_types.Le,
    annotated_types.Lt,
    annotated_types.MinLen,
    annotated_types.MaxLen,
    annotated_types.MultipleOf,
)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _is_request_model(cls: type[BaseModel]) -> bool:
    return ".requests." in cls.__module__ or cls.__name__.endswith(("Request", "Params"))


def _all_models() -> dict[str, type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.walk_packages(venice_ai.__path__, "venice_ai."):
        if ".skills." in info.name:
            continue
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BaseModel) and cls.__module__ == module.__name__:
                found[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return found


ALL_MODELS = _all_models()
_BY_NAME: dict[str, list[type[BaseModel]]] = {}
for _cls in ALL_MODELS.values():
    _BY_NAME.setdefault(_cls.__name__, []).append(_cls)


def _key(cls: type[BaseModel], field: str | None = None) -> str:
    base = f"{cls.__module__}.{cls.__qualname__}"
    return f"{base}.{field}" if field else base


def _models_in(annotation: Any, out: set[type[BaseModel]]) -> None:
    value = getattr(annotation, "__value__", None)
    if value is not None:
        _models_in(value, out)
        return
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        out.add(annotation)
        return
    for arg in typing.get_args(annotation):
        _models_in(arg, out)


def _raw_return_annotation(fn: Any) -> str:
    """The return annotation as source text, without evaluating it.

    On Python 3.14 annotations are evaluated lazily, and one such as
    ``list[ApiKey]`` on a method named ``list`` cannot be evaluated at all.
    """
    try:
        import annotationlib  # type: ignore[import-not-found]

        annotations = annotationlib.get_annotations(fn, format=annotationlib.Format.STRING)
    except ImportError:
        annotations = getattr(fn, "__annotations__", {})
    return str(annotations.get("return", ""))


def _return_models(fn: Any) -> set[type[BaseModel]]:
    """Models named by ``fn``'s return annotation.

    Annotations that only resolve under ``TYPE_CHECKING`` fall back to looking
    the names up among all SDK models, so an unresolvable hint still counts.
    """
    out: set[type[BaseModel]] = set()
    try:
        hint = typing.get_type_hints(fn).get("return")
    except Exception:
        raw = _raw_return_annotation(fn)
        for name in re.findall(r"[A-Z]\w+", raw):
            out.update(_BY_NAME.get(name, []))
        return out
    if hint is not None:
        _models_in(hint, out)
    return out


def _roots() -> set[type[BaseModel]]:
    roots: set[type[BaseModel]] = set(api_models._SPEC_BY_TYPE.values())
    for info in pkgutil.walk_packages(venice_ai.resources.__path__, "venice_ai.resources."):
        module = importlib.import_module(info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not (issubclass(cls, APIResource) or cls.__name__.endswith("Job")):
                continue
            for name, fn in inspect.getmembers(cls, inspect.isfunction):
                if not name.startswith("_"):
                    roots |= _return_models(fn)
    # Models a resource validates the body into but does not return as is
    # (e.g. ApiKeys.list parses ApiKeysListResponse and returns its data).
    for path in pathlib.Path(venice_ai.resources.__file__).parent.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in re.findall(r"cast_to=([A-Z]\w+)|\b([A-Z]\w+)\.model_validate\(", text):
            for part in name:
                roots.update(_BY_NAME.get(part, []))
    # Streamed chunks are validated per SSE frame rather than returned whole.
    from venice_ai.types.api.responses import ResponsesStreamEvent
    from venice_ai.types.api.streaming import ChatCompletionChunk

    roots |= {ChatCompletionChunk, ResponsesStreamEvent}
    return {cls for cls in roots if not _is_request_model(cls)}


def _reachable(roots: set[type[BaseModel]]) -> set[type[BaseModel]]:
    seen: set[type[BaseModel]] = set()
    stack = list(roots)
    while stack:
        cls = stack.pop()
        if cls in seen or _key(cls) in NOT_FROM_SERVER or _key(cls) in CALLER_BUILT_ARMS:
            continue
        seen.add(cls)
        for field in cls.model_fields.values():
            nested: set[type[BaseModel]] = set()
            _models_in(field.annotation, nested)
            stack.extend(nested)
    return seen


RESPONSE_MODELS = sorted(_reachable(_roots()), key=lambda c: (c.__module__, c.__qualname__))


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #


def _union_arms(annotation: Any) -> list[Any]:
    value = getattr(annotation, "__value__", None)
    if value is not None:
        return _union_arms(value)
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _union_arms(typing.get_args(annotation)[0])
    if origin in (typing.Union, types.UnionType):
        return [arm for arg in typing.get_args(annotation) for arm in _union_arms(arg)]
    return [annotation]


def _single_literal(annotation: Any) -> tuple[Any, ...] | None:
    """The value of a one-value ``Literal``, optionally ``| None``; else ``None``."""
    arms = [arm for arm in _union_arms(annotation) if arm is not type(None)]
    if len(arms) == 1 and typing.get_origin(arms[0]) is Literal:
        values = typing.get_args(arms[0])
        return values if len(values) == 1 else None
    return None


def _single_literal_fields(cls: type[BaseModel]) -> list[str]:
    return [
        name
        for name, field in cls.model_fields.items()
        if _single_literal(field.annotation) is not None
    ]


def _is_catch_all(arm: Any) -> bool:
    """An arm that accepts a part of a type the SDK does not know."""
    if arm is Any or arm is dict or typing.get_origin(arm) is dict:
        return True
    return isinstance(arm, type) and issubclass(arm, BaseModel) and not _single_literal_fields(arm)


def _closed_unions(annotation: Any, depth: int = 0) -> list[list[str]]:
    """Unions, at any depth, of two or more tagged models with no catch-all arm."""
    if depth > 8:
        return []
    value = getattr(annotation, "__value__", None)
    if value is not None:
        return _closed_unions(value, depth + 1)
    found: list[list[str]] = []
    origin = typing.get_origin(annotation)
    if origin is Annotated:
        return _closed_unions(typing.get_args(annotation)[0], depth + 1)
    if origin in (typing.Union, types.UnionType):
        arms = _union_arms(annotation)
        models = [a for a in arms if isinstance(a, type) and issubclass(a, BaseModel)]
        if len(models) > 1 and not any(_is_catch_all(a) for a in arms):
            found.append(sorted(m.__name__ for m in models))
    for arg in typing.get_args(annotation):
        if not isinstance(arg, (str, int, float, bool)) and arg is not None:
            found.extend(_closed_unions(arg, depth + 1))
    return found


def _union_tag_models() -> set[type[BaseModel]]:
    """Models used as an arm of a union somewhere in the response models."""
    arms: set[type[BaseModel]] = set()

    def visit(annotation: Any, depth: int = 0) -> None:
        if depth > 8:
            return
        value = getattr(annotation, "__value__", None)
        if value is not None:
            visit(value, depth + 1)
            return
        if typing.get_origin(annotation) in (typing.Union, types.UnionType):
            members = _union_arms(annotation)
            models = [a for a in members if isinstance(a, type) and issubclass(a, BaseModel)]
            if models and len(members) > 1:
                arms.update(models)
        for arg in typing.get_args(annotation):
            if not isinstance(arg, (str, int, float, bool)) and arg is not None:
                visit(arg, depth + 1)

    for cls in RESPONSE_MODELS:
        for field in cls.model_fields.values():
            visit(field.annotation)
    for name in ("VideoRetrieveResponse", "MusicRetrieveResponse", "VoiceChangerRetrieveResponse"):
        visit(getattr(venice_ai.types, name))
    return arms


def _fixed_literal_problems(models: list[type[BaseModel]]) -> dict[str, tuple[Any, ...]]:
    tags = _union_tag_models()
    problems: dict[str, tuple[Any, ...]] = {}
    for cls in models:
        fields = _single_literal_fields(cls)
        if cls in tags and fields:
            fields = fields[1:] if fields[0] in ("type", "status", "object") else fields
        for name in fields:
            problems[_key(cls, name)] = _single_literal(cls.model_fields[name].annotation) or ()
    return problems


def _constraint_problems(models: list[type[BaseModel]]) -> dict[str, list[str]]:
    problems: dict[str, list[str]] = {}
    for cls in models:
        for name, field in cls.model_fields.items():
            metadata = list(field.metadata)
            for arm in _union_arms(field.annotation):
                if typing.get_origin(arm) is Annotated:
                    metadata.extend(typing.get_args(arm)[1:])
            hits = [repr(m) for m in metadata if isinstance(m, _CONSTRAINTS)]
            if hits:
                problems[_key(cls, name)] = hits
    return problems


FIXED_LITERALS = _fixed_literal_problems(RESPONSE_MODELS)


# --------------------------------------------------------------------------- #
# The detectors flag what they should
# --------------------------------------------------------------------------- #


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: Literal["list"]
    kind: Literal["x"] | None = None
    count: int = Field(..., ge=0)


class _TagA(BaseModel):
    type: Literal["a"]


class _TagB(BaseModel):
    type: Literal["b"]


class _Unknown(BaseModel):
    type: str


class _Holder(BaseModel):
    closed: list[_TagA | _TagB]
    open_dict: list[Annotated[_TagA | _TagB | dict[str, Any], Field(union_mode="left_to_right")]]
    open_model: list[_TagA | _TagB | _Unknown]


def test_detectors_flag_each_failure_mode() -> None:
    assert _Strict.model_config.get("extra") == "forbid"
    assert _fixed_literal_problems([_Strict]) == {
        _key(_Strict, "object"): ("list",),
        _key(_Strict, "kind"): ("x",),
    }
    assert _constraint_problems([_Strict]) == {_key(_Strict, "count"): ["Ge(ge=0)"]}
    closed = {
        name: _closed_unions(field.annotation) for name, field in _Holder.model_fields.items()
    }
    assert closed == {"closed": [["_TagA", "_TagB"]], "open_dict": [], "open_model": []}


def test_discovery_reaches_the_response_surface() -> None:
    names = {cls.__name__ for cls in RESPONSE_MODELS}
    assert len(RESPONSE_MODELS) > 120
    for expected in (
        "ModelsListResponse",
        "ImageModelSpec",
        "ChatCompletionResponse",
        "ChatCompletionChunk",
        "ApiKey",
        "CreatedApiKey",
        "VideoProcessingStatus",
        "X402BalanceResponse",
        "ResponsesResponse",
    ):
        assert expected in names, expected
    assert not [cls for cls in RESPONSE_MODELS if _is_request_model(cls)]


# --------------------------------------------------------------------------- #
# The gates
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", RESPONSE_MODELS, ids=[_key(c) for c in RESPONSE_MODELS])
def test_response_model_keeps_unknown_fields(model: type[BaseModel]) -> None:
    assert model.model_config.get("extra") != "forbid", (
        f"{_key(model)} forbids extra fields; a field the server adds would fail "
        'the whole response. Set model_config = ConfigDict(extra="allow").'
    )


def test_no_fixed_value_fields_outside_union_tags() -> None:
    unexpected = {k: v for k, v in FIXED_LITERALS.items() if k not in LITERAL_EXEMPTIONS}
    assert not unexpected, (
        "single-value Literal on a response field that is not a union tag; any "
        f"other server value would fail the whole response. Type it as str: {unexpected}"
    )


def test_literal_exemptions_are_current() -> None:
    stale = sorted(set(LITERAL_EXEMPTIONS) - set(FIXED_LITERALS))
    assert not stale, f"exempted fields are no longer single-value Literals: {stale}"


def test_no_value_constraints_on_response_fields() -> None:
    problems = _constraint_problems(RESPONSE_MODELS)
    assert not problems, (
        "a constraint on a response field rejects a server value outside the "
        f"documented range and fails the whole response: {problems}"
    )


def test_unions_of_tagged_models_have_a_catch_all() -> None:
    problems = {
        _key(cls, name): unions
        for cls in RESPONSE_MODELS
        for name, field in cls.model_fields.items()
        if (unions := _closed_unions(field.annotation))
    }
    assert not problems, (
        "a union of tagged models with no catch-all arm fails the whole response "
        f"on a type the SDK does not know; add a dict or open model arm: {problems}"
    )


def test_every_server_model_is_checked() -> None:
    checked = {_key(cls) for cls in RESPONSE_MODELS}
    accounted = checked | set(NOT_FROM_SERVER) | set(CALLER_BUILT_ARMS) | set(HEADER_MODELS)
    candidates = {
        key
        for key, cls in ALL_MODELS.items()
        if not _is_request_model(cls) and not cls.__module__.startswith(tuple(LOCAL_MODULES))
    }
    missing = sorted(candidates - accounted)
    assert not missing, (
        "non-request models that no resource returns; make them reachable from a "
        f"resource return type or list them, with a reason, in NOT_FROM_SERVER: {missing}"
    )


def test_header_models_take_any_server_value() -> None:
    models = [ALL_MODELS[key] for key in HEADER_MODELS]
    assert not _constraint_problems(models)
    assert not _fixed_literal_problems(models)


def test_exclusion_lists_are_current() -> None:
    listed = {**NOT_FROM_SERVER, **CALLER_BUILT_ARMS, **dict.fromkeys(HEADER_MODELS, "")}
    stale = sorted(k for k in listed if k not in ALL_MODELS)
    assert not stale, f"exclusion lists name models that no longer exist: {stale}"


def test_response_models_are_fully_defined() -> None:
    """An unresolved forward reference would hide a field's real type from these gates."""
    incomplete = [_key(cls) for cls in RESPONSE_MODELS if not cls.__pydantic_complete__]
    assert not incomplete, f"response models with unresolved annotations: {incomplete}"
