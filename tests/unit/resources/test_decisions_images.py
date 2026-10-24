"""
Tests for ``images`` on ``client.decisions.create`` and :class:`DecisionImage`.

Covers both wire forms the API accepts (a ``data:`` URI string and a
``{content_type, base64}`` object), the body staying unchanged when no images
are passed, the file and bytes constructors, and the local checks that reject
a bad image list before it costs an API call.

Checked against the live ``POST /decisions`` endpoint on 2026-10-24: a model
that accepts images read both forms identically, and a model that does not
answered 400.
"""

import base64
from unittest.mock import AsyncMock

import pytest

from venice_ai.exceptions import InvalidRequestError
from venice_ai.resources.decisions import Decisions
from venice_ai.types.api import DecisionImage, DecisionResponse, NoulQuestion

RESPONSE = {
    "model": "decision-model",
    "answers": {"red": {"type": "noul", "noul": 0.99}},
    "usage": {"input_tokens": 218, "output_tokens": 0},
}

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")
PNG_URI = f"data:image/png;base64,{PNG_B64}"


class MockVeniceClient:
    def __init__(self):
        self.post = AsyncMock(return_value=DecisionResponse.model_validate(RESPONSE))


@pytest.fixture
def mock_client():
    return MockVeniceClient()


@pytest.fixture
def decisions(mock_client):
    return Decisions(mock_client)


async def _create(decisions, mock_client, **kwargs):
    await decisions.create(
        model="decision-model",
        state="Look at the attached image.",
        questions={"red": NoulQuestion(instructions="Is the image mostly red?")},
        **kwargs,
    )
    return mock_client.post.await_args.kwargs["json_data"]


class TestWireBody:
    @pytest.mark.asyncio
    async def test_images_omitted_by_default(self, decisions, mock_client):
        body = await _create(decisions, mock_client)
        assert "images" not in body
        assert body == {
            "model": "decision-model",
            "state": "Look at the attached image.",
            "questions": {"red": {"type": "noul", "instructions": "Is the image mostly red?"}},
        }

    @pytest.mark.asyncio
    async def test_none_is_omitted(self, decisions, mock_client):
        body = await _create(decisions, mock_client, images=None)
        assert "images" not in body

    @pytest.mark.asyncio
    async def test_data_uri_sent_as_string(self, decisions, mock_client):
        body = await _create(decisions, mock_client, images=[PNG_URI])
        assert body["images"] == [PNG_URI]

    @pytest.mark.asyncio
    async def test_decision_image_sent_as_object(self, decisions, mock_client):
        image = DecisionImage(content_type="image/png", base64=PNG_B64)
        body = await _create(decisions, mock_client, images=[image])
        assert body["images"] == [{"content_type": "image/png", "base64": PNG_B64}]

    @pytest.mark.asyncio
    async def test_dict_image_accepted(self, decisions, mock_client):
        body = await _create(
            decisions, mock_client, images=[{"content_type": "image/jpeg", "base64": "abc="}]
        )
        assert body["images"] == [{"content_type": "image/jpeg", "base64": "abc="}]

    @pytest.mark.asyncio
    async def test_mixed_forms_keep_order(self, decisions, mock_client):
        image = DecisionImage(content_type="image/jpg", base64="abc=")
        body = await _create(decisions, mock_client, images=(PNG_URI, image))
        assert body["images"] == [PNG_URI, {"content_type": "image/jpg", "base64": "abc="}]

    @pytest.mark.asyncio
    async def test_four_images_allowed(self, decisions, mock_client):
        body = await _create(decisions, mock_client, images=[PNG_URI] * 4)
        assert len(body["images"]) == 4


class TestConstructors:
    def test_from_bytes(self):
        image = DecisionImage.from_bytes(PNG_BYTES, content_type="image/png")
        assert image == DecisionImage(content_type="image/png", base64=PNG_B64)

    @pytest.mark.parametrize(
        ("name", "content_type"),
        [("a.png", "image/png"), ("b.jpg", "image/jpeg"), ("c.JPEG", "image/jpeg")],
    )
    def test_from_file_infers_type(self, tmp_path, name, content_type):
        path = tmp_path / name
        path.write_bytes(PNG_BYTES)
        image = DecisionImage.from_file(path)
        assert image.content_type == content_type
        assert base64.b64decode(image.base64) == PNG_BYTES

    def test_from_file_accepts_str_path(self, tmp_path):
        path = tmp_path / "a.png"
        path.write_bytes(PNG_BYTES)
        assert DecisionImage.from_file(str(path)).base64 == PNG_B64

    @pytest.mark.parametrize("name", ["a.gif", "a.webp", "noext"])
    def test_from_file_rejects_other_types(self, tmp_path, name):
        path = tmp_path / name
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a PNG or JPEG file"):
            DecisionImage.from_file(path)

    def test_from_file_missing_raises_oserror(self, tmp_path):
        with pytest.raises(OSError):
            DecisionImage.from_file(tmp_path / "missing.png")


class TestLocalValidation:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "images",
        [
            pytest.param([], id="empty-list"),
            pytest.param([PNG_URI] * 5, id="five-images"),
            pytest.param(["https://example.com/a.png"], id="remote-url"),
            pytest.param([PNG_B64], id="bare-base64-string"),
            pytest.param(["data:image/webp;base64,abc="], id="webp-data-uri"),
            pytest.param([{"content_type": "image/gif", "base64": "abc="}], id="gif-object"),
            pytest.param([{"content_type": "image/png", "base64": ""}], id="empty-base64"),
            pytest.param(
                [{"content_type": "image/png", "base64": "abc=", "url": "x"}], id="extra-key"
            ),
            pytest.param([{"base64": "abc="}], id="missing-content-type"),
        ],
    )
    async def test_rejects_locally(self, decisions, mock_client, images):
        with pytest.raises(InvalidRequestError):
            await _create(decisions, mock_client, images=images)
        mock_client.post.assert_not_awaited()
