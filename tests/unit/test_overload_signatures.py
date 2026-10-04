"""Every ``@overload`` of a resource method accepts what its implementation does.

A parameter the implementation takes but an overload omits type-checks as an
error at every call site that passes it (``Image.create(timeout=...)`` did),
even though the call works at runtime.
"""

from __future__ import annotations

import importlib
import inspect
import typing

import pytest

RESOURCE_MODULES = (
    "venice_ai.resources.audio",
    "venice_ai.resources.augment",
    "venice_ai.resources.chat.completions",
    "venice_ai.resources.image",
    "venice_ai.resources.responses",
    "venice_ai.resources.video",
)


def _overloaded_methods() -> list[tuple[str, typing.Any]]:
    found = []
    for module_name in RESOURCE_MODULES:
        module = importlib.import_module(module_name)
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            for name, fn in vars(cls).items():
                if callable(fn) and typing.get_overloads(fn):
                    found.append((f"{cls_name}.{name}", fn))
    return found


OVERLOADED = _overloaded_methods()


def test_overloaded_methods_are_found() -> None:
    names = {name for name, _ in OVERLOADED}
    assert {"Image.create", "Audio.create_speech", "ChatCompletions.create"} <= names


@pytest.mark.parametrize(("qualname", "fn"), OVERLOADED, ids=[n for n, _ in OVERLOADED])
def test_overloads_declare_every_implementation_parameter(qualname: str, fn: typing.Any) -> None:
    implementation = {
        name
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind is not inspect.Parameter.VAR_KEYWORD
    }
    for index, overload in enumerate(typing.get_overloads(fn)):
        declared = set(inspect.signature(overload).parameters)
        missing = implementation - declared
        assert not missing, f"{qualname} overload {index} omits {sorted(missing)}"


@pytest.mark.parametrize(
    ("qualname", "discriminator"),
    [
        ("Image.create", "return_binary"),
        ("Audio.create_speech", "stream"),
        ("ChatCompletions.create", "stream"),
        ("Responses.create", "stream"),
    ],
)
def test_a_non_literal_discriminator_has_an_overload(qualname: str, discriminator: str) -> None:
    fn = dict(OVERLOADED)[qualname]
    annotations = [
        str(inspect.signature(o).parameters[discriminator].annotation)
        for o in typing.get_overloads(fn)
    ]
    assert any("Literal" not in a for a in annotations), annotations
