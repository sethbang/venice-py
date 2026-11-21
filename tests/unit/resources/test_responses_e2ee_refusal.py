"""/responses does not support E2EE, so ``enable_e2ee`` other than False is refused.

Sending ``enable_e2ee: true`` there would ask for E2EE while the input goes out
in plaintext, so ``create()`` refuses it before any network call.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.exceptions import InvalidRequestError
from venice_ai.resources.responses import Responses
from venice_ai.types.api.requests.common import VeniceParameters

_REFUSAL = (
    r"E2EE is not supported on /responses; use "
    r"client\.chat\.completions\.create\(\.\.\., e2ee=True\)"
)


def _resource() -> Responses:
    client = Mock()
    client.post = AsyncMock(return_value=Mock())
    client._stream_request = Mock()
    return Responses(client)


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize(
    "venice_parameters",
    [
        {"enable_e2ee": True},
        {"enable_e2ee": 1},
        {"enable_e2ee": "true"},
        VeniceParameters(enable_e2ee=True),
        VeniceParameters.model_construct(enable_e2ee=1),
        VeniceParameters.model_construct(enable_e2ee=0),
    ],
    ids=["bool-true", "int-1", "str-true", "model-true", "construct-1", "construct-0"],
)
async def test_enable_e2ee_is_refused(venice_parameters: Any, stream: bool) -> None:
    resource = _resource()
    with pytest.raises(InvalidRequestError, match=_REFUSAL):
        await resource.create(
            model="m",
            input="SECRET-TEXT",
            venice_parameters=venice_parameters,
            stream=stream,
        )
    cast(Any, resource._client.post).assert_not_called()
    cast(Any, resource._client._stream_request).assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("venice_parameters", "wire_value"),
    [
        (None, None),
        ({"enable_e2ee": False}, False),
        ({"enable_e2ee": 0}, False),
        (VeniceParameters(enable_e2ee=False), False),
        (VeniceParameters.model_construct(enable_e2ee=None), None),
        ({"enable_web_search": "auto"}, None),
    ],
    ids=["absent", "bool-false", "int-0", "model-false", "construct-none", "unset"],
)
async def test_false_or_absent_enable_e2ee_is_unaffected(
    venice_parameters: Any, wire_value: Any
) -> None:
    resource = _resource()
    await resource.create(model="m", input="hi", venice_parameters=venice_parameters)
    body = cast(Any, resource._client.post).call_args.kwargs["json_data"]
    assert body.get("venice_parameters", {}).get("enable_e2ee") is wire_value
