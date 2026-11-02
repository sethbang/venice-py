"""Request models a caller can hand a dict to must not drop keys silently.

Pydantic's default ``extra="ignore"`` discards a key the model does not
declare, so a field Venice adds before the SDK models it would vanish from the
request without an error. Every request model that a caller's dict can reach
must say what happens instead: ``"allow"`` sends the key as given, ``"forbid"``
raises before the request is sent.

A dict reaches a model when the model is nested inside a request model
(pydantic validates the caller's dict into it), or when a resource validates a
caller's mapping into it directly. Top-level request models built only from a
method's keyword arguments are out of scope: an unknown keyword argument is
already a ``TypeError``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import typing
from typing import Any

import pytest
from pydantic import BaseModel

import venice_ai.types.api.requests as requests_pkg
from venice_ai.types.api.crypto import JsonRpcRequest
from venice_ai.types.api.requests.chat import (
    AssistantMessage,
    ChatCompletionRequest,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

# Top-level models a resource validates from a caller's mapping.
CALLER_DICT_ROOTS: dict[type[BaseModel], str] = {
    UserMessage: "chat messages= accepts mappings",
    AssistantMessage: "chat messages= accepts mappings",
    SystemMessage: "chat messages= accepts mappings",
    ToolMessage: "chat messages= accepts mappings",
    DeveloperMessage: "chat messages= accepts mappings",
    JsonRpcRequest: "crypto.batch_rpc() accepts mappings",
}


def _request_roots() -> list[type[BaseModel]]:
    roots: list[type[BaseModel]] = []
    modules = [
        importlib.import_module(f"{requests_pkg.__name__}.{info.name}")
        for info in pkgutil.iter_modules(requests_pkg.__path__)
    ]
    modules += [importlib.import_module("venice_ai.types.api.augment")]
    for module in modules:
        for name, obj in vars(module).items():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseModel)
                and obj.__module__ == module.__name__
                and name.endswith(("Request", "Params"))
            ):
                roots.append(obj)
    return roots


def _models_in(annotation: Any, found: set[type[BaseModel]]) -> None:
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        found.add(annotation)
        return
    for arg in typing.get_args(annotation):
        _models_in(arg, found)


def _caller_reachable() -> set[type[BaseModel]]:
    reachable: set[type[BaseModel]] = set(CALLER_DICT_ROOTS)
    stack = _request_roots() + list(CALLER_DICT_ROOTS)
    walked: set[type[BaseModel]] = set()
    while stack:
        model = stack.pop()
        if model in walked:
            continue
        walked.add(model)
        for field in model.model_fields.values():
            nested: set[type[BaseModel]] = set()
            _models_in(field.annotation, nested)
            reachable |= nested
            stack.extend(nested)
    return reachable


def test_the_walk_finds_nested_request_models() -> None:
    """Guard the guard: an empty or shallow walk would pass vacuously."""
    from venice_ai.core.models.common import CacheControl, TextContent
    from venice_ai.types.api.requests.common import ReasoningConfig
    from venice_ai.types.api.requests.video import SeedanceConsents, VideoElement

    reachable = _caller_reachable()
    for model in (CacheControl, TextContent, ReasoningConfig, SeedanceConsents, VideoElement):
        assert model in reachable, model.__name__


def test_no_caller_reachable_request_model_drops_unknown_keys() -> None:
    silent = sorted(
        f"{model.__module__}.{model.__qualname__}"
        for model in _caller_reachable()
        if model.model_config.get("extra", "ignore") == "ignore"
    )
    assert not silent, (
        'these request models drop unknown keys silently; set extra="allow" '
        f'(send them) or extra="forbid" (raise): {silent}'
    )


@pytest.mark.parametrize("message_cls", list(CALLER_DICT_ROOTS)[:5])
def test_message_models_send_unknown_keys(message_cls: type[BaseModel]) -> None:
    role = typing.get_args(message_cls.model_fields["role"].annotation)[0]
    mapping: dict[str, Any] = {"role": role, "content": "x", "future_field": {"a": 1}}
    if message_cls is ToolMessage:
        mapping["tool_call_id"] = "c1"
    request = ChatCompletionRequest(model="m", messages=[mapping])
    sent = request.model_dump(mode="json", exclude_none=True)["messages"][0]
    assert sent["future_field"] == {"a": 1}


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_a_text_part_dict_keeps_unknown_keys_for_every_role(role: str) -> None:
    part = {"type": "text", "text": "x", "future": 1}
    request = ChatCompletionRequest(model="m", messages=[{"role": role, "content": [part]}])
    sent = request.model_dump(mode="json", exclude_none=True)["messages"][0]
    assert sent["content"] == [part]


def test_a_typed_text_part_still_rejects_an_unknown_key() -> None:
    from pydantic import ValidationError

    from venice_ai.types.api.requests.common import TextContent

    with pytest.raises(ValidationError):
        TextContent(type="text", text="x", future=1)  # type: ignore[call-arg]
