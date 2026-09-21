"""``anon_user_id`` must be modelled, validated, and actually sent.

Venice accepts an optional end-user identifier on seven endpoints. Before it
was modelled the SDK handled it two different wrong ways, neither of which
surfaced anything to the caller:

* the five image request models take pydantic's default ``extra="ignore"``, so
  a caller passing ``anon_user_id`` had it **silently dropped** — the request
  succeeded and the attribution never happened;
* ``ChatCompletionRequest`` and ``ResponsesRequest`` set ``extra="allow"``, so
  it reached the wire **unvalidated** — a value the API rejects produced a
  server-side 400 instead of a local error.

It is deliberately *not* an alias of the OpenAI-compatible ``user`` field,
which Venice discards.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.chat import ChatCompletionRequest
from venice_ai.types.api.requests.common import validate_anon_user_id
from venice_ai.types.api.requests.images import (
    ImageBackgroundRemoveRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultiEditRequest,
    SimpleImageGenerationRequest,
)
from venice_ai.types.api.requests.responses import ResponsesRequest

#: Every request model the API documents ``anon_user_id`` on, with the minimum
#: arguments needed to build one.
MODELS_AND_MINIMAL_ARGS = [
    (ChatCompletionRequest, {"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
    (ResponsesRequest, {"model": "m", "input": "hi"}),
    (ImageGenerationRequest, {"model": "m", "prompt": "p"}),
    (SimpleImageGenerationRequest, {"prompt": "p"}),
    (ImageEditRequest, {"prompt": "p", "image": "https://e.com/a.png"}),
    (ImageMultiEditRequest, {"prompt": "p", "images": ["https://e.com/a.png"]}),
    (ImageBackgroundRemoveRequest, {"image_url": "https://e.com/a.png"}),
]

MODEL_IDS = [m.__name__ for m, _ in MODELS_AND_MINIMAL_ARGS]


class TestModelled:
    @pytest.mark.parametrize(("model", "args"), MODELS_AND_MINIMAL_ARGS, ids=MODEL_IDS)
    def test_field_exists(self, model, args):
        assert "anon_user_id" in model.model_fields

    @pytest.mark.parametrize(("model", "args"), MODELS_AND_MINIMAL_ARGS, ids=MODEL_IDS)
    def test_reaches_the_serialised_body(self, model, args):
        body = model(**args, anon_user_id="end-user-123").model_dump(exclude_none=True)
        assert body["anon_user_id"] == "end-user-123"

    @pytest.mark.parametrize(("model", "args"), MODELS_AND_MINIMAL_ARGS, ids=MODEL_IDS)
    def test_omitted_when_unset(self, model, args):
        assert "anon_user_id" not in model(**args).model_dump(exclude_none=True)

    @pytest.mark.parametrize(("model", "args"), MODELS_AND_MINIMAL_ARGS, ids=MODEL_IDS)
    def test_constraints_apply_on_every_model(self, model, args):
        """The shared annotated type must not have been dropped on any one of them."""
        with pytest.raises(ValidationError):
            model(**args, anon_user_id="has||delimiter")


class TestConstraints:
    @pytest.mark.parametrize("value", ["e", "end-user-123", "x" * 128, "a b~!@#$%^&*()"])
    def test_accepts_printable_ascii_within_length(self, value):
        assert validate_anon_user_id(value) == value

    @pytest.mark.parametrize(
        ("value", "why"),
        [
            ("", "empty"),
            ("x" * 129, "too long"),
            ("a||b", "reserved delimiter"),
            ("héllo", "non-ascii"),
            ("tab\there", "control character"),
            ("new\nline", "control character"),
        ],
    )
    def test_rejects(self, value, why):
        with pytest.raises(ValidationError):
            validate_anon_user_id(value)

    def test_is_not_the_openai_user_field(self):
        """Venice discards `user`; the two are separate fields, not aliases."""
        body = SimpleImageGenerationRequest(
            prompt="p", user="openai-style", anon_user_id="venice-style"
        ).model_dump(exclude_none=True)
        assert body["user"] == "openai-style"
        assert body["anon_user_id"] == "venice-style"


class TestReachesTheWire:
    """Modelling the field is not the same as sending it.

    Each resource method assembles its body differently — through the request
    model, through a hand-built dict, or via a kwargs dict forwarded to
    another method — so "the model has the field" does not imply the call
    sends it.
    """

    @staticmethod
    def _client():
        from unittest.mock import AsyncMock, Mock

        c = Mock()
        c._request = AsyncMock(return_value=b"img-bytes")
        c.post = AsyncMock(return_value={"created": 0, "data": []})
        return c

    @staticmethod
    def _body(client):
        for call in (client._request.call_args, client.post.call_args):
            if call and isinstance(call.kwargs.get("json_data"), dict):
                return call.kwargs["json_data"]
        return {}

    @pytest.mark.asyncio
    async def test_create(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).create(model="m", prompt="p", anon_user_id="end-user-123")
        assert self._body(c)["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_submit_forwards_through_the_job(self):
        """``submit`` stashes kwargs on an ImageJob; the request is built later
        inside ``wait()``, so the field crosses an extra hop."""
        from venice_ai.resources.image import Image

        c = self._client()
        c.image = Image(c)
        job = await c.image.submit(model="m", prompt="p", anon_user_id="end-user-123")
        await job.wait()
        assert self._body(c)["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_edit(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).edit(prompt="p", image="https://e.com/a.png", anon_user_id="end-user-123")
        assert self._body(c)["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_background_remove(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).background_remove(
            image_url="https://e.com/a.png", anon_user_id="end-user-123"
        )
        assert self._body(c)["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_simple_generate(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).simple_generate(prompt="p", anon_user_id="end-user-123")
        assert self._body(c)["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_chat_completions(self):
        from venice_ai.resources.chat.completions import ChatCompletions

        c = self._client()
        c.post = self._client().post
        await ChatCompletions(c).create(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            anon_user_id="end-user-123",
        )
        assert self._body(c)["anon_user_id"] == "end-user-123"


class TestMultiEditValidatesWithoutItsModel:
    """``multi_edit`` is the one call site where the constraint lives twice.

    It assembles its body as a plain dict rather than through
    ``ImageMultiEditRequest``, so the model's constraints do not apply to it.
    It calls ``validate_anon_user_id`` explicitly instead — and that second
    site is exactly the kind of thing that silently diverges from the model.
    """

    @staticmethod
    def _client():
        from unittest.mock import AsyncMock, Mock

        c = Mock()
        c._request = AsyncMock(return_value=b"img-bytes")
        return c

    @pytest.mark.asyncio
    async def test_valid_value_reaches_the_payload(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).multi_edit(
            prompt="p", image="https://e.com/a.png", anon_user_id="end-user-123"
        )
        payload = c._request.call_args.kwargs["json_data"]
        assert payload["anon_user_id"] == "end-user-123"

    @pytest.mark.asyncio
    async def test_omitted_when_unset(self):
        from venice_ai.resources.image import Image

        c = self._client()
        await Image(c).multi_edit(prompt="p", image="https://e.com/a.png")
        assert "anon_user_id" not in c._request.call_args.kwargs["json_data"]

    @pytest.mark.parametrize("value", ["a||b", "héllo", "x" * 129, ""])
    @pytest.mark.asyncio
    async def test_invalid_values_are_rejected_here_too(self, value):
        """Without the explicit call these would sail past onto the wire."""
        from venice_ai.resources.image import Image

        c = self._client()
        with pytest.raises(ValidationError):
            await Image(c).multi_edit(prompt="p", image="https://e.com/a.png", anon_user_id=value)
        assert c._request.await_count == 0, "the request must not be sent"
