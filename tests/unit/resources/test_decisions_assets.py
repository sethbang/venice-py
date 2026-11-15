"""
Tests for ``assets`` on ``client.decisions.create`` and :class:`DecisionAsset`.

``assets`` is the Jev-only keyed map of images (``{ref: {data_base64, format,
type?}}``), separate from the positional ``images`` list. Covers the wire body,
the file and bytes constructors, and the local checks that reject a malformed
asset before it is sent.
"""

import base64
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from venice_ai import DecisionAsset as RootDecisionAsset
from venice_ai.exceptions import InvalidRequestError
from venice_ai.resources.decisions import Decisions
from venice_ai.types.api import DecisionAsset, DecisionResponse, NoulQuestion

RESPONSE = {
    "model": "jev",
    "answers": {"match": {"type": "noul", "noul": 0.9}},
    "usage": {"input_tokens": 300, "output_tokens": 0},
}
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake-png"
PNG_B64 = base64.b64encode(PNG_BYTES).decode("ascii")


@pytest.fixture
def client() -> AsyncMock:
    mock = AsyncMock()
    mock.post = AsyncMock(return_value=DecisionResponse.model_validate(RESPONSE))
    return mock


async def _create(client: AsyncMock, **kwargs: object) -> dict:
    await Decisions(client).create(
        model="jev",
        state="Does the photo in receipt_photo match the claimed total?",
        questions={"match": NoulQuestion(instructions="Do they match?")},
        **kwargs,  # type: ignore[arg-type]
    )
    body: dict = client.post.await_args.kwargs["json_data"]
    return body


class TestWireBody:
    @pytest.mark.asyncio
    async def test_model_and_dict_assets_serialize_identically(self, client: AsyncMock) -> None:
        body = await _create(
            client,
            assets={
                "receipt_photo": DecisionAsset(data_base64=PNG_B64, format="png"),
                "shelf": {"data_base64": PNG_B64, "format": "jpeg", "type": "image"},
            },
        )
        assert body["assets"] == {
            "receipt_photo": {"data_base64": PNG_B64, "format": "png"},
            "shelf": {"data_base64": PNG_B64, "format": "jpeg", "type": "image"},
        }

    @pytest.mark.asyncio
    async def test_assets_absent_when_unset(self, client: AsyncMock) -> None:
        body = await _create(client)
        assert "assets" not in body

    @pytest.mark.asyncio
    async def test_empty_assets_are_dropped(self, client: AsyncMock) -> None:
        """Non-Jev models reject the field's presence, so ``{}`` is not sent."""
        body = await _create(client, assets={})
        assert "assets" not in body

    @pytest.mark.asyncio
    async def test_assets_and_images_can_travel_together(self, client: AsyncMock) -> None:
        body = await _create(
            client,
            images=[f"data:image/png;base64,{PNG_B64}"],
            assets={"ref": DecisionAsset(data_base64=PNG_B64, format="jpg")},
        )
        assert body["images"] == [f"data:image/png;base64,{PNG_B64}"]
        assert body["assets"]["ref"]["format"] == "jpg"


class TestConstructors:
    def test_from_bytes(self) -> None:
        asset = DecisionAsset.from_bytes(PNG_BYTES, format="png")
        assert asset.data_base64 == PNG_B64
        assert asset.format == "png"

    @pytest.mark.parametrize(
        ("suffix", "fmt"), [(".png", "png"), (".JPG", "jpg"), (".jpeg", "jpg")]
    )
    def test_from_file_infers_format(self, tmp_path: Path, suffix: str, fmt: str) -> None:
        path = tmp_path / f"photo{suffix}"
        path.write_bytes(PNG_BYTES)
        asset = DecisionAsset.from_file(path)
        assert asset.format == fmt
        assert base64.b64decode(asset.data_base64) == PNG_BYTES

    def test_from_file_rejects_other_formats(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.gif"
        path.write_bytes(b"GIF89a")
        with pytest.raises(ValueError, match="not a PNG or JPEG"):
            DecisionAsset.from_file(path)

    def test_exported_from_package_root(self) -> None:
        assert RootDecisionAsset is DecisionAsset


class TestLocalChecks:
    def test_data_uri_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="without a data: prefix"):
            DecisionAsset(data_base64=f"data:image/png;base64,{PNG_B64}", format="png")

    def test_format_is_closed(self) -> None:
        with pytest.raises(ValidationError):
            DecisionAsset(data_base64=PNG_B64, format="gif")  # type: ignore[arg-type]

    def test_type_only_allows_image(self) -> None:
        with pytest.raises(ValidationError):
            DecisionAsset(data_base64=PNG_B64, format="png", type="video")  # type: ignore[arg-type]

    def test_unknown_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecisionAsset.model_validate({"data_base64": PNG_B64, "format": "png", "name": "x"})

    def test_empty_data_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecisionAsset(data_base64="", format="png")

    @pytest.mark.asyncio
    async def test_blank_key_never_reaches_the_api(self, client: AsyncMock) -> None:
        with pytest.raises(InvalidRequestError, match="asset keys"):
            await _create(client, assets={" ": DecisionAsset(data_base64=PNG_B64, format="png")})
        client.post.assert_not_awaited()
