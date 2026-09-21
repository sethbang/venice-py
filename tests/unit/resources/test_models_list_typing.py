"""Regression guards for ``Models.list(type=...)`` typing.

Two things this protects:

1. The kwarg's ``Literal`` membership matches the OpenAPI enum (``asr``,
   ``embedding``, ``image``, ``music``, ``text``, ``tts``, ``upscale``,
   ``inpaint``, ``video``, ``decision``, ``all``, ``code``) plus the
   SDK-level alias ``chat`` (which the validator normalises to ``text``).

   Note the deliberate asymmetry with :attr:`ModelResponse.type`, which is an
   open ``str``. A *request* filter stays a closed ``Literal`` so a typo fails
   at the call site; a *response* field stays open so a model type the SDK
   predates never fails the parse. Widening this kwarg is not the same fix as
   widening that field, and is not licensed by it.
2. Calling ``list()`` with no kwarg sends ``type=all`` on the wire, not the
   server's text-only default.

If someone widens the kwarg back to ``str`` or reverts the auto-``all``
default, these tests fail loudly. See
``src/venice_ai/skills/venice-py/references/response-shapes.md`` for why this matters
to consumers.
"""

from __future__ import annotations

import typing
from unittest.mock import AsyncMock

import pytest

from venice_ai.resources.models import Models
from venice_ai.types.api import ModelsListResponse

# Source of truth: the OpenAPI ``listModels`` enum served live at
# ``https://api.venice.ai/doc/api/swagger.yaml`` (operationId ``listModels``).
# Plus the SDK's ``chat`` alias and the documented but undocumented-in-enum
# ``all`` / ``code`` values. Re-fetch the spec before editing this set — the
# cached copy under ``_GLOBAL/venice-docs/`` lags the live one and was missing
# ``decision`` entirely.
_EXPECTED_TYPE_MEMBERS = frozenset(
    {
        # Official API enum
        "asr",
        "embedding",
        "image",
        "music",
        "text",
        "tts",
        "upscale",
        "inpaint",
        "video",
        "decision",
        # Documented in the description but not the enum block
        "all",
        "code",
        # SDK-level alias normalised to "text" by the validator
        "chat",
    }
)


def _list_type_literal_members() -> frozenset[str]:
    """Pull the ``type`` Literal members off ``Models.list``'s annotations."""
    hints = typing.get_type_hints(Models.list)
    annotation = hints["type"]
    # Annotation is ``ModelListType | None``. ``ModelListType`` is a PEP 695
    # ``type`` alias (a ``TypeAliasType``), so unwrap to its underlying
    # ``Literal[...]`` via ``__value__`` before reading the args.
    args = typing.get_args(annotation)
    inner = next(a for a in args if a is not type(None))
    if isinstance(inner, typing.TypeAliasType):
        inner = inner.__value__
    return frozenset(typing.get_args(inner))


def test_list_type_kwarg_is_literal_with_expected_members():
    """The ``type`` kwarg must be a closed ``Literal`` covering the API enum.

    Widening to ``str`` would silently re-admit the ``"stt"`` /
    ``"embeddings"`` typos that the outside-agent reflection report flagged.
    """
    members = _list_type_literal_members()
    assert members == _EXPECTED_TYPE_MEMBERS, f"Expected {_EXPECTED_TYPE_MEMBERS}, got {members}"


@pytest.mark.asyncio
async def test_list_no_arg_sends_type_all():
    """``list()`` without args must send ``type=all`` on the wire.

    Server's own default is text-only — this auto-``all`` behaviour is the
    SDK's contract per the docstring. Regression-guards against reverting to
    the empty-params behaviour.
    """
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=ModelsListResponse(object="list", type="all", data=[]))

    await Models(mock_client).list()

    mock_client.get.assert_called_once_with(
        "models", params={"type": "all"}, cast_to=ModelsListResponse, force_direct=True
    )


@pytest.mark.asyncio
async def test_list_explicit_text_overrides_auto_all():
    """When the caller passes ``type="text"``, the SDK MUST honour it.

    The auto-``all`` behaviour only kicks in when ``type`` is ``None``.
    """
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=ModelsListResponse(object="list", type="text", data=[])
    )

    await Models(mock_client).list(type="text")

    mock_client.get.assert_called_once_with(
        "models", params={"type": "text"}, cast_to=ModelsListResponse, force_direct=True
    )


@pytest.mark.asyncio
async def test_list_chat_alias_normalised_to_text_on_wire():
    """``type="chat"`` is an SDK-level alias normalised to ``text``.

    The wire value must be ``text`` so the server doesn't see an unrecognised
    enum member and 400.
    """
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=ModelsListResponse(object="list", type="text", data=[])
    )

    await Models(mock_client).list(type="chat")

    mock_client.get.assert_called_once_with(
        "models", params={"type": "text"}, cast_to=ModelsListResponse, force_direct=True
    )


class TestCliFetchesEveryConcreteType:
    """``venice-py models --type X`` can only narrow what it already fetched.

    The CLI lists models by fetching each type in turn and then filtering the
    union. It used to hand-list the types to fetch, so ``--type decision``
    reported "No models match the specified filters" while ``jev-latest`` was
    live in the catalog — a missing type is indistinguishable from an empty
    result. The list is derived from ``ModelListType`` now; these guard the
    derivation.
    """

    def test_every_concrete_type_is_fetched(self):
        from venice_ai.cli.commands.models.command import (
            ALIAS_MODEL_TYPES,
            concrete_model_types,
        )

        expected = _EXPECTED_TYPE_MEMBERS - ALIAS_MODEL_TYPES
        assert set(concrete_model_types()) == expected

    def test_aliases_are_real_members(self):
        """A typo'd alias name would silently fetch a bogus type instead."""
        from venice_ai.cli.commands.models.command import ALIAS_MODEL_TYPES

        assert ALIAS_MODEL_TYPES <= _EXPECTED_TYPE_MEMBERS
