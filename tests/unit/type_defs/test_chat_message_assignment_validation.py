"""Chat message models validate on assignment, not only at construction.

Every member of the ``ChatCompletionRequest.messages`` union pins its ``role``
to a single literal and constrains ``content``. Reassigning either field on an
existing message must be validated just like constructing it, otherwise a
``UserMessage`` can be turned into an ``assistant`` turn (or given non-text
content) and reach the wire unchecked.
"""

from __future__ import annotations

import typing
from typing import Any, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from venice_ai.types.api.requests.chat import ChatCompletionRequest, UserMessage


def _message_models() -> list[type[BaseModel]]:
    annotation = ChatCompletionRequest.model_fields["messages"].annotation
    assert get_origin(annotation) is list
    (item,) = get_args(annotation)
    return [m for m in get_args(item) if isinstance(m, type) and issubclass(m, BaseModel)]


MESSAGE_MODELS = _message_models()


def _instance(model: type[BaseModel]) -> BaseModel:
    kwargs: dict[str, Any] = {"content": "hello"}
    if "tool_call_id" in model.model_fields:
        kwargs["tool_call_id"] = "call_1"
    return model(**kwargs)


def _foreign_role(model: type[BaseModel]) -> str:
    (own,) = get_args(model.model_fields["role"].annotation)
    return "assistant" if own != "assistant" else "user"


def test_message_union_is_introspected() -> None:
    assert len(MESSAGE_MODELS) == 5
    assert UserMessage in MESSAGE_MODELS


def test_user_message_role_cannot_be_reassigned() -> None:
    msg = UserMessage(content="hi")
    with pytest.raises(ValidationError):
        msg.role = "assistant"  # type: ignore[assignment]


@pytest.mark.parametrize("model", MESSAGE_MODELS, ids=lambda m: m.__name__)
def test_role_reassignment_is_validated(model: type[BaseModel]) -> None:
    assert typing.get_origin(model.model_fields["role"].annotation) is typing.Literal
    msg = _instance(model)
    bad_role = _foreign_role(model)
    with pytest.raises(ValidationError):
        msg.role = bad_role  # type: ignore[attr-defined]


@pytest.mark.parametrize("model", MESSAGE_MODELS, ids=lambda m: m.__name__)
def test_content_reassignment_is_validated(model: type[BaseModel]) -> None:
    msg = _instance(model)
    with pytest.raises(ValidationError):
        msg.content = 12345  # type: ignore[attr-defined]


@pytest.mark.parametrize("model", MESSAGE_MODELS, ids=lambda m: m.__name__)
def test_valid_content_reassignment_still_allowed(model: type[BaseModel]) -> None:
    msg = _instance(model)
    msg.content = "updated"  # type: ignore[attr-defined]
    assert msg.content == "updated"  # type: ignore[attr-defined]
