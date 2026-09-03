"""Unit tests for plain-mapping ``venice_parameters`` input on chat.completions.

``venice_parameters=`` accepts either a :class:`VeniceParameters` or a plain
mapping of the same fields. The mapping is validated into the model while
``ChatCompletionRequest`` is built, so the two forms are interchangeable and an
invalid value still raises before the request is sent.

Unlike ``messages=``, nothing reads ``venice_parameters`` attributes ahead of the
request model, so this is a typing fix rather than a behavior change — these
tests exist to pin the equivalence the annotation now advertises.
"""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from venice_ai.resources.chat.completions import ChatCompletions, _e2ee_engaged
from venice_ai.types.api import ChatCompletionRequest
from venice_ai.types.api.requests.common import VeniceParameters

_FAKE_CHAT_MODEL = "fake-chat-test-model"

_MESSAGES = [{"role": "user", "content": "hi"}]


class _MockClient:
    def __init__(self):
        self.post = AsyncMock(
            return_value={
                "id": "resp-1",
                "object": "chat.completion",
                "created": 1000000,
                "model": _FAKE_CHAT_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


# ---------------------------------------------------------------------------
# ChatCompletionRequest — the validation boundary
# ---------------------------------------------------------------------------


class TestRequestModelCoercesMapping:
    def test_mapping_becomes_the_model(self):
        request = ChatCompletionRequest(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters={"enable_web_search": "on"},
        )

        assert isinstance(request.venice_parameters, VeniceParameters)
        assert request.venice_parameters.enable_web_search == "on"

    def test_mapping_and_model_are_equivalent(self):
        from_mapping = ChatCompletionRequest(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters={"enable_web_search": "on"},
        )
        from_model = ChatCompletionRequest(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters=VeniceParameters(enable_web_search="on"),
        )

        assert from_mapping.model_dump() == from_model.model_dump()

    def test_unrecognized_key_is_carried_through_for_both_forms(self):
        # VeniceParameters is extra="allow", so a key the model doesn't declare
        # survives coercion rather than being dropped or rejected. Pinned here
        # because it is the opposite of how the *message* models behave
        # (extra="ignore"), and the two are easy to conflate.
        from_mapping = ChatCompletionRequest(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters={"some_future_flag": True},
        )
        from_model = ChatCompletionRequest(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters=VeniceParameters(some_future_flag=True),
        )

        assert from_mapping.model_dump() == from_model.model_dump()
        assert from_mapping.model_dump()["venice_parameters"]["some_future_flag"] is True

    @pytest.mark.parametrize(
        "bad",
        [
            {"enable_web_search": "nope"},
            {"enable_web_search": 123},
            {"enable_web_citations": "notabool"},
        ],
    )
    def test_invalid_value_still_raises(self, bad):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model=_FAKE_CHAT_MODEL,
                messages=list(_MESSAGES),
                venice_parameters=bad,
            )


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCreateAcceptsMapping:
    async def test_mapping_and_model_produce_identical_body(self):
        from_mapping = _MockClient()
        await ChatCompletions(from_mapping).create(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters={"enable_web_search": "on"},
        )

        from_model = _MockClient()
        await ChatCompletions(from_model).create(
            model=_FAKE_CHAT_MODEL,
            messages=list(_MESSAGES),
            venice_parameters=VeniceParameters(enable_web_search="on"),
        )

        assert from_mapping.post.call_args == from_model.post.call_args

    async def test_invalid_mapping_raises_before_the_request(self):
        client = _MockClient()
        with pytest.raises(ValidationError):
            await ChatCompletions(client).create(
                model=_FAKE_CHAT_MODEL,
                messages=list(_MESSAGES),
                venice_parameters={"enable_web_search": "nope"},
            )
        client.post.assert_not_called()


# ---------------------------------------------------------------------------
# E2EE engagement
# ---------------------------------------------------------------------------


class TestE2EEDetectionSeesMapping:
    """``_e2ee_engaged`` inspects the caller's raw value, before coercion.

    A mapping must engage the encrypted flow exactly as the model does —
    otherwise ``venice_parameters={"enable_e2ee": True}`` would silently send
    plaintext.
    """

    def test_mapping_engages(self):
        assert _e2ee_engaged(False, {"enable_e2ee": True}) is True

    def test_model_engages(self):
        assert _e2ee_engaged(False, VeniceParameters(enable_e2ee=True)) is True

    def test_mapping_without_the_flag_does_not_engage(self):
        assert _e2ee_engaged(False, {"enable_web_search": "on"}) is False

    def test_none_does_not_engage(self):
        assert _e2ee_engaged(False, None) is False
