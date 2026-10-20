"""Outcome fields on response models accept values the SDK has not seen.

Lifecycle and outcome fields (``status``, ``*_status``, ``finish_reason``,
``state``, ``reason``) are the fields servers extend most often: a new
truncation mode, a new terminal state. On a response model a closed
``Literal`` there turns one new server value into a hard validation error
for the whole body, discarding everything else the call returned.

Response models therefore type these fields as ``str`` and publish the
documented values as ``KNOWN_*`` tuples. Request models keep ``Literal`` so
typos are caught before a request is sent. Single-value ``Literal`` fields
are exempt, since they are used as union tags (``status: Literal["COMPLETED"]``
on a job-status arm) that select an arm rather than describe an outcome.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
import typing
from typing import Any, Literal

import pytest
from pydantic import BaseModel

import venice_ai.core.models
import venice_ai.types

OUTCOME_FIELD_NAMES = frozenset({"status", "finish_reason", "state", "reason"})
OUTCOME_FIELD_SUFFIXES = ("_status", "_reason", "_state")


def _is_outcome_field(name: str) -> bool:
    return name in OUTCOME_FIELD_NAMES or name.endswith(OUTCOME_FIELD_SUFFIXES)


def _literal_args(annotation: Any) -> list[tuple[Any, ...]]:
    if type(annotation).__name__ == "TypeAliasType":
        return _literal_args(annotation.__value__)
    if typing.get_origin(annotation) is Literal:
        return [typing.get_args(annotation)]
    found: list[tuple[Any, ...]] = []
    for arg in typing.get_args(annotation):
        found.extend(_literal_args(arg))
    return found


def _closed_literal_values(annotation: Any) -> tuple[Any, ...] | None:
    """Values of a multi-value ``Literal`` in ``annotation`` that has no open
    ``str`` alternative, or ``None`` when the field accepts any string."""
    args = typing.get_args(annotation)
    origin = typing.get_origin(annotation)
    if annotation is str or (origin in (typing.Union, types.UnionType) and str in args):
        return None
    values = tuple(v for lit in _literal_args(annotation) for v in lit)
    return values if len(values) > 1 else None


def _is_request_model(cls: type[BaseModel]) -> bool:
    return ".requests." in cls.__module__ or cls.__name__.endswith(("Request", "Params"))


def _iter_models() -> list[type[BaseModel]]:
    seen: dict[str, type[BaseModel]] = {}
    for pkg in (venice_ai.types, venice_ai.core.models):
        for info in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            module = importlib.import_module(info.name)
            for _, cls in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(cls, BaseModel)
                    and cls.__module__ == module.__name__
                    and not _is_request_model(cls)
                ):
                    seen[f"{cls.__module__}.{cls.__qualname__}"] = cls
    return sorted(seen.values(), key=lambda c: (c.__module__, c.__qualname__))


def _outcome_fields(models: list[type[BaseModel]]) -> list[tuple[type[BaseModel], str]]:
    return [(cls, name) for cls in models for name in cls.model_fields if _is_outcome_field(name)]


RESPONSE_MODELS = _iter_models()
OUTCOME_FIELDS = _outcome_fields(RESPONSE_MODELS)


def test_scan_selects_response_models_and_outcome_fields() -> None:
    assert len(RESPONSE_MODELS) > 50
    assert len(OUTCOME_FIELDS) > 0
    names = {f"{cls.__name__}.{field}" for cls, field in OUTCOME_FIELDS}
    assert "ResponsesResponse.status" in names
    assert "ChatChoice.finish_reason" in names


def test_detector_flags_a_closed_outcome_literal() -> None:
    class _Closed(BaseModel):
        status: Literal["done", "running"]
        finish_reason: Literal["stop", "length"] | None = None

    class _Open(BaseModel):
        status: str
        finish_reason: str | None = None

    class _Tag(BaseModel):
        status: Literal["COMPLETED"]

    flagged = [
        f"{cls.__name__}.{name}"
        for cls, name in _outcome_fields([_Closed, _Open, _Tag])
        if _closed_literal_values(cls.model_fields[name].annotation) is not None
    ]
    assert flagged == ["_Closed.status", "_Closed.finish_reason"]


@pytest.mark.parametrize(
    ("model", "field"),
    OUTCOME_FIELDS,
    ids=[f"{cls.__name__}.{name}" for cls, name in OUTCOME_FIELDS],
)
def test_response_outcome_field_is_open(model: type[BaseModel], field: str) -> None:
    closed = _closed_literal_values(model.model_fields[field].annotation)
    assert closed is None, (
        f"{model.__module__}.{model.__name__}.{field} is a closed Literal {closed}; "
        "a new server value would fail the whole response parse"
    )


# Multi-value ``Literal`` fields on non-request models that are closed on
# purpose. Every other enum-like response field is an open ``str``.
CLOSED_ENUM_EXEMPTIONS: dict[str, str] = {
    "venice_ai.core.models.common.VeniceParameters.enable_web_search": (
        "request-side parameters model; a typo must fail before the request is sent"
    ),
    "venice_ai.core.models.common.ImageUrl.detail": (
        "request content part; a typo must fail before the request is sent. The "
        "chat response schema carries no image parts, and ChatMessage reuses "
        "ImageContent only so message history can be passed back in"
    ),
    "venice_ai.types.api.audio.VoiceDetail.gender": (
        "derived client-side from the voice ID prefix, never read from the server"
    ),
    "venice_ai.types.api.audio.VoiceList.gender_filter": (
        "echoes the caller's own gender filter argument"
    ),
    "venice_ai.types.api.embeddings.EmbeddingObject.encoding_format": (
        "absent from the spec's response schema; can only echo the request value, "
        "which EmbeddingRequest constrains to the same Literal"
    ),
}


def _closed_enum_fields(models: list[type[BaseModel]]) -> dict[str, tuple[Any, ...]]:
    found: dict[str, tuple[Any, ...]] = {}
    for cls in models:
        for name, field in cls.model_fields.items():
            closed = _closed_literal_values(field.annotation)
            if closed is not None:
                found[f"{cls.__module__}.{cls.__qualname__}.{name}"] = closed
    return found


CLOSED_ENUM_FIELDS = _closed_enum_fields(RESPONSE_MODELS)


def test_no_unexempted_closed_enum_on_response_models() -> None:
    unexpected = {k: v for k, v in CLOSED_ENUM_FIELDS.items() if k not in CLOSED_ENUM_EXEMPTIONS}
    assert not unexpected, (
        "closed multi-value Literal on a response model; type it as str and publish "
        f"the documented values as a KNOWN_* tuple: {unexpected}"
    )


def test_closed_enum_exemptions_are_current() -> None:
    stale = sorted(set(CLOSED_ENUM_EXEMPTIONS) - set(CLOSED_ENUM_FIELDS))
    assert not stale, f"exempted fields are no longer closed Literals: {stale}"
