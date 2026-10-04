"""
Tests targeting specific coverage gaps in venice_ai.resources.image module.

Coverage gaps addressed:
- Gap 1: background_remove() — lines 881–922 (entirely untested)
- Gap 2: multi_edit() — lines 985–1057 (entirely untested)
- Gap 3: _prepare_image_content() URL/base64 passthrough — lines 145, 148
- Gap 4-5: edit() mask + model in multipart/JSON modes — lines 728–731, 737, 752
- Gap 6: upscale() URL/base64 rejection — line 564
- Gap 7: _is_base64() all branches — lines 782, 785, 789
- Gap 8: _prepare_image_for_request() Path & base64 routing — lines 821, 823, 830
- Gap 9: _detect_image_format() non-PNG formats — lines 795, 799, 801
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from venice_ai.exceptions import VeniceError
from venice_ai.resources.image import Image

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_resource():
    """Create an Image resource instance with a mock client."""
    mock_client = Mock()
    return Image(mock_client)


# ===========================================================================
# Gap 1: background_remove() — lines 881–922
# ===========================================================================


class TestBackgroundRemove:
    """Tests for the background_remove() method (entirely uncovered)."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_background_remove_neither_arg_raises(self, image_resource):
        """Line 881-882: ValueError when neither image nor image_url provided."""
        with pytest.raises(ValueError, match="Either 'image' or 'image_url' must be provided"):
            await image_resource.background_remove()

    @pytest.mark.asyncio
    async def test_background_remove_multipart_with_bytes(self, image_resource):
        """Lines 891-901: Multipart path when image is raw bytes."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        image_resource._request_multipart = AsyncMock(return_value=b"no-bg-image")

        result = await image_resource.background_remove(image=png_bytes)

        assert result == b"no-bg-image"
        image_resource._request_multipart.assert_called_once()
        call_kwargs = image_resource._request_multipart.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "image/background-remove"
        assert "image" in call_kwargs["files"]
        assert call_kwargs["headers"] == {"Accept": "image/*"}

    @pytest.mark.asyncio
    async def test_background_remove_json_mode_with_url(self, image_resource):
        """Lines 869-880: JSON path when image_url is provided."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"no-bg-from-url")
        image_resource._client = mock_client

        result = await image_resource.background_remove(image_url="https://example.com/photo.jpg")

        assert result == b"no-bg-from-url"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["image_url"] == "https://example.com/photo.jpg"
        assert call_kwargs["raw_response"] is True

    @pytest.mark.asyncio
    async def test_background_remove_json_mode_with_base64_image(self, image_resource):
        """Lines 904-914: JSON path when image is a data URL (base64)."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"no-bg-from-b64")
        image_resource._client = mock_client

        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result = await image_resource.background_remove(image=data_url)

        assert result == b"no-bg-from-b64"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["image"] == data_url

    @pytest.mark.asyncio
    async def test_background_remove_response_bytes(self, image_resource):
        """Line 916-917: Response is already bytes."""
        image_resource._request_multipart = AsyncMock(return_value=b"raw-bytes")

        result = await image_resource.background_remove(image=b"\x89PNG\r\n\x1a\n")

        assert result == b"raw-bytes"

    @pytest.mark.asyncio
    async def test_background_remove_response_client_response(self, image_resource):
        """Lines 918-919: Response is aiohttp.ClientResponse."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"client-response-content")
        image_resource._request_multipart = AsyncMock(return_value=mock_response)

        result = await image_resource.background_remove(image=b"\x89PNG\r\n\x1a\n")

        assert result == b"client-response-content"

    @pytest.mark.asyncio
    async def test_background_remove_image_url_takes_priority_over_image(self, image_resource):
        """Line 869: When both image and image_url provided, image_url takes priority."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"from-url")
        image_resource._client = mock_client

        result = await image_resource.background_remove(
            image=b"\x89PNG\r\n\x1a\n",
            image_url="https://example.com/photo.jpg",
        )

        assert result == b"from-url"
        # Should have used JSON path (URL mode), not multipart (bytes path)
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["image_url"] == "https://example.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_background_remove_multipart_with_file_path(self, image_resource):
        """Lines 887-901: Multipart path when image is a file path string."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
            temp_path = f.name

        try:
            image_resource._request_multipart = AsyncMock(return_value=b"from-file")

            result = await image_resource.background_remove(image=temp_path)

            assert result == b"from-file"
            image_resource._request_multipart.assert_called_once()
        finally:
            os.unlink(temp_path)


# ===========================================================================
# Gap 2: multi_edit() — lines 985–1057
# ===========================================================================


class TestMultiEdit:
    """Tests for the multi_edit() method.

    multi_edit() always uses JSON mode via _client._request().  Bytes inputs
    are converted to base64 and placed in an ``images`` array.  When a
    ``model`` kwarg is supplied it is forwarded as ``modelId`` in the JSON
    payload (per the Venice API docs for POST /image/multi-edit).
    """

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_multi_edit_single_image(self, image_resource):
        """Single bytes image is base64-encoded and placed in images[0]."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"multi-edited")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Replace the sky",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        )

        assert result == b"multi-edited"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["method"] == "POST"
        assert call_kwargs["path"] == "image/multi-edit"
        assert call_kwargs["json_data"]["prompt"] == "Replace the sky"
        assert len(call_kwargs["json_data"]["images"]) == 1

    @pytest.mark.asyncio
    async def test_multi_edit_with_two_images(self, image_resource):
        """Two image inputs are placed in the images array; spec has no masks field."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"multi-layer-edit")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Blend images",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
            image_2=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        )

        assert result == b"multi-layer-edit"
        call_kwargs = mock_client._request.call_args[1]
        assert len(call_kwargs["json_data"]["images"]) == 2
        assert "masks" not in call_kwargs["json_data"]

    @pytest.mark.asyncio
    async def test_multi_edit_forwards_model_as_modelId(self, image_resource):
        """The ``model`` kwarg is forwarded as ``modelId`` in the JSON payload."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"with-model")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Edit with model",
            model="some-edit-model",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        )

        assert result == b"with-model"
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["modelId"] == "some-edit-model"
        assert "model" not in call_kwargs["json_data"]
        assert call_kwargs["json_data"]["prompt"] == "Edit with model"

    @pytest.mark.asyncio
    async def test_multi_edit_json_mode_with_url(self, image_resource):
        """URL strings are placed directly in the images array."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"json-multi-edit")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Blend these",
            image="https://example.com/base.jpg",
            image_2="https://example.com/overlay.jpg",
        )

        assert result == b"json-multi-edit"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["prompt"] == "Blend these"
        images = call_kwargs["json_data"]["images"]
        assert "https://example.com/base.jpg" in images
        assert "https://example.com/overlay.jpg" in images

    @pytest.mark.asyncio
    async def test_multi_edit_json_mode_forwards_modelId(self, image_resource):
        """URL path also forwards ``model`` kwarg as ``modelId``."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"json-with-model")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Edit",
            model="url-edit-model",
            image="https://example.com/photo.jpg",
        )

        assert result == b"json-with-model"
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["modelId"] == "url-edit-model"
        assert "model" not in call_kwargs["json_data"]

    @pytest.mark.asyncio
    async def test_multi_edit_no_images_raises(self, image_resource):
        """Per docs the ``images`` array is required (1-3 items); the SDK
        surfaces this as a ValueError before making the request rather than
        letting the API reject an empty array."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock()
        image_resource._client = mock_client

        with pytest.raises(ValueError, match="at least one image"):
            await image_resource.multi_edit(prompt="Generate from scratch")

        mock_client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_edit_response_bytes(self, image_resource):
        """Response is already bytes."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"raw-bytes")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(prompt="Edit", image=b"\x89PNG\r\n\x1a\n")

        assert result == b"raw-bytes"

    @pytest.mark.asyncio
    async def test_multi_edit_response_client_response(self, image_resource):
        """Response is aiohttp.ClientResponse."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.read = AsyncMock(return_value=b"cr-content")

        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.multi_edit(prompt="Edit", image=b"\x89PNG\r\n\x1a\n")

        assert result == b"cr-content"

    @pytest.mark.asyncio
    async def test_multi_edit_three_images_with_safe_mode(self, image_resource):
        """Three image inputs populate the images array; safe_mode is forwarded."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"all-fields")
        image_resource._client = mock_client
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50

        result = await image_resource.multi_edit(
            prompt="Complex edit",
            image=png,
            image_2=png,
            image_3=png,
            safe_mode=False,
        )

        assert result == b"all-fields"
        call_kwargs = mock_client._request.call_args[1]
        assert len(call_kwargs["json_data"]["images"]) == 3
        assert call_kwargs["json_data"]["safe_mode"] is False
        assert "masks" not in call_kwargs["json_data"]

    @pytest.mark.asyncio
    async def test_multi_edit_resolution_forwarded_in_payload(self, image_resource):
        """resolution kwarg is forwarded in the JSON payload to image/multi-edit."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"multi-edited-4k")
        image_resource._client = mock_client

        result = await image_resource.multi_edit(
            prompt="Edit at 4K",
            image="data:image/png;base64,AA==",
            resolution="4K",
        )

        assert result == b"multi-edited-4k"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["path"] == "image/multi-edit"
        assert call_kwargs["json_data"]["resolution"] == "4K"


# ===========================================================================
# Gap 3: _prepare_image_content() URL/base64 passthrough — lines 145, 148
# ===========================================================================


class TestPrepareImageContentPassthrough:
    """Tests for URL and base64 passthrough in _prepare_image_content()."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_prepare_image_content_url_passthrough(self, image_resource):
        """Line 144-145: URL string is returned as-is."""
        url = "https://example.com/image.png"
        result = await image_resource._prepare_image_content(url)
        assert result == url

    @pytest.mark.asyncio
    async def test_prepare_image_content_http_url_passthrough(self, image_resource):
        """Line 144-145: HTTP URL string is returned as-is."""
        url = "http://example.com/image.png"
        result = await image_resource._prepare_image_content(url)
        assert result == url

    @pytest.mark.asyncio
    async def test_prepare_image_content_base64_data_url_passthrough(self, image_resource):
        """Line 147-148: data: URL (base64) string is returned as-is."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        result = await image_resource._prepare_image_content(data_url)
        assert result == data_url

    @pytest.mark.asyncio
    async def test_prepare_image_content_raw_base64_passthrough(self, image_resource):
        """Line 147-148: Raw base64 string (long, no path separators) is returned as-is."""
        # A long string with no /, \, or . that looks like base64
        raw_b64 = "A" * 200
        result = await image_resource._prepare_image_content(raw_b64)
        assert result == raw_b64


# ===========================================================================
# Gap 4-5: edit() mask + model — always JSON mode
# ===========================================================================


class TestEditMaskAndModel:
    """Tests for edit() with mask, model, and safe_mode parameters.

    edit() always sends a JSON body via _client._request(). Binary inputs are
    base64-encoded. ``model`` and ``safe_mode`` are forwarded verbatim when
    set, and omitted from the payload when left as ``None`` (the default).
    """

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    def test_edit_has_no_mask_param(self):
        """``mask`` is dead code — the server rejects it with
        400 unrecognized_keys. ``Image.edit`` must not accept a ``mask`` kwarg."""
        import inspect

        assert "mask" not in inspect.signature(Image.edit).parameters

    def test_edit_request_model_has_no_mask_field(self):
        """``ImageEditRequest`` must not declare a ``mask`` field
        (EditImageRequest is additionalProperties:false server-side)."""
        from venice_ai.types.api.requests.images import ImageEditRequest

        assert "mask" not in ImageEditRequest.model_fields

    @pytest.mark.asyncio
    async def test_edit_model_forwarded_in_payload(self, image_resource):
        """Caller-supplied model is forwarded verbatim in the JSON payload."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"edited-with-model")
        image_resource._client = mock_client

        result = await image_resource.edit(
            prompt="Edit this",
            model="flux-2-max-edit",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        )

        assert result == b"edited-with-model"
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["model"] == "flux-2-max-edit"

    @pytest.mark.asyncio
    async def test_edit_without_model_omits_key(self, image_resource):
        """When model is not supplied, it is omitted from the payload so the
        API applies its own default (qwen-edit)."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"edited-default")
        image_resource._client = mock_client

        await image_resource.edit(
            prompt="Edit",
            image=b"\x89PNG\r\n\x1a\n" + b"\x00" * 50,
        )

        call_kwargs = mock_client._request.call_args[1]
        assert "model" not in call_kwargs["json_data"]

    @pytest.mark.asyncio
    async def test_edit_json_mode_model_forwarded(self, image_resource):
        """URL input path also forwards the model selection."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"json-model-edit")
        image_resource._client = mock_client

        result = await image_resource.edit(
            prompt="Edit",
            model="flux-2-max-edit",
            image="https://example.com/photo.jpg",
        )

        assert result == b"json-model-edit"
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["model"] == "flux-2-max-edit"

    @pytest.mark.asyncio
    async def test_edit_safe_mode_false_forwarded(self, image_resource):
        """safe_mode=False is forwarded to let callers disable blurring."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"no-blur")
        image_resource._client = mock_client

        await image_resource.edit(
            prompt="Edit",
            image="https://example.com/photo.jpg",
            safe_mode=False,
        )

        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["safe_mode"] is False

    @pytest.mark.asyncio
    async def test_edit_safe_mode_omitted_when_none(self, image_resource):
        """Default (None) leaves safe_mode out so the server default (True)
        applies — keeps request bodies byte-identical to pre-fix cassettes."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"default-blur")
        image_resource._client = mock_client

        await image_resource.edit(
            prompt="Edit",
            image="https://example.com/photo.jpg",
        )

        call_kwargs = mock_client._request.call_args[1]
        assert "safe_mode" not in call_kwargs["json_data"]


# ===========================================================================
# Gap 6: upscale() URL/base64 rejection — line 564
# ===========================================================================


class TestUpscaleURLRejection:
    """Test that upscale() rejects URL and base64 string inputs."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_upscale_rejects_url_string(self, image_resource):
        """Lines 563-567: upscale raises VeniceError for URL strings."""
        with pytest.raises(VeniceError, match="Upscale requires image file data"):
            await image_resource.upscale(image="https://example.com/photo.jpg", scale=2.0)

    @pytest.mark.asyncio
    async def test_upscale_rejects_data_url_string(self, image_resource):
        """Lines 563-567: upscale raises VeniceError for data URL strings."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        with pytest.raises(VeniceError, match="Upscale requires image file data"):
            await image_resource.upscale(image=data_url, scale=2.0)


# ===========================================================================
# Gap 7: _is_base64() all branches — lines 782, 785, 789
# ===========================================================================


class TestIsBase64:
    """Tests for _is_base64() helper method."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    def test_is_base64_returns_false_for_url(self, image_resource):
        """Line 781-782: URL returns False."""
        assert image_resource._is_base64("https://example.com/image.png") is False
        assert image_resource._is_base64("http://example.com/image.png") is False

    def test_is_base64_returns_true_for_data_url(self, image_resource):
        """Line 784-785: data: prefix returns True."""
        assert image_resource._is_base64("data:image/png;base64,abc123") is True
        assert image_resource._is_base64("data:application/octet-stream;base64,xyz") is True

    def test_is_base64_returns_true_for_long_string_without_path_chars(self, image_resource):
        """Line 788-789: Long string without /, \\, or . returns True."""
        long_b64 = "A" * 200
        assert image_resource._is_base64(long_b64) is True

    def test_is_base64_returns_false_for_file_path(self, image_resource):
        """Line 790: File path returns False."""
        assert image_resource._is_base64("/path/to/image.png") is False
        assert image_resource._is_base64("image.png") is False
        assert image_resource._is_base64("relative/path/file.jpg") is False

    def test_is_base64_returns_false_for_short_string(self, image_resource):
        """Line 790: Short string without path chars returns False (len <= 100)."""
        assert image_resource._is_base64("shortstring") is False


# ===========================================================================
# Gap 8: _prepare_image_for_request() Path & base64 routing — lines 821, 823, 830
# ===========================================================================


class TestPrepareImageForRequest:
    """Tests for _prepare_image_for_request() helper method."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_path_object(self, image_resource):
        """Lines 820-823: Path object input resolves to multipart mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            test_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            f.write(test_data)
            temp_path = f.name

        try:
            mode, content = await image_resource._prepare_image_for_request(Path(temp_path))
            assert mode == "multipart"
            assert content == test_data
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_bytes(self, image_resource):
        """Lines 824-825: Bytes input returns multipart mode."""
        test_bytes = b"raw image bytes"
        mode, content = await image_resource._prepare_image_for_request(test_bytes)
        assert mode == "multipart"
        assert content == test_bytes

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_url_string(self, image_resource):
        """Lines 826-828: URL string returns json_url mode."""
        mode, content = await image_resource._prepare_image_for_request(
            "https://example.com/image.png"
        )
        assert mode == "json_url"
        assert content is None

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_base64_string(self, image_resource):
        """Lines 829-830: Base64 string returns json_base64 mode."""
        data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        mode, content = await image_resource._prepare_image_for_request(data_url)
        assert mode == "json_base64"
        assert content is None

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_raw_base64_string(self, image_resource):
        """Lines 829-830: Raw base64 string (long, no path chars) returns json_base64."""
        raw_b64 = "A" * 200
        mode, content = await image_resource._prepare_image_for_request(raw_b64)
        assert mode == "json_base64"
        assert content is None

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_file_path_string(self, image_resource):
        """Lines 831-834: File path string resolves to multipart mode."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            test_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            f.write(test_data)
            temp_path = f.name

        try:
            mode, content = await image_resource._prepare_image_for_request(temp_path)
            assert mode == "multipart"
            assert content == test_data
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_file_like_object(self, image_resource):
        """Lines 835-838: File-like object resolves to multipart mode."""
        import io

        file_obj = io.BytesIO(b"file-like content")
        mode, content = await image_resource._prepare_image_for_request(file_obj)
        assert mode == "multipart"
        assert content == b"file-like content"

    @pytest.mark.asyncio
    async def test_prepare_image_for_request_unsupported_type(self, image_resource):
        """Lines 839-840: Unsupported type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported image type"):
            await image_resource._prepare_image_for_request(12345)


# ===========================================================================
# Gap 9: _detect_image_format() non-PNG formats — lines 795, 799, 801
# ===========================================================================


class TestDetectImageFormat:
    """Tests for _detect_image_format() helper method."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    def test_detect_jpeg(self, image_resource):
        """Line 794-795: JPEG magic bytes."""
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        filename, mime = image_resource._detect_image_format(jpeg_data)
        assert filename == "image.jpg"
        assert mime == "image/jpeg"

    def test_detect_png(self, image_resource):
        """Line 796-797: PNG magic bytes."""
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        filename, mime = image_resource._detect_image_format(png_data)
        assert filename == "image.png"
        assert mime == "image/png"

    def test_detect_webp(self, image_resource):
        """Line 798-799: WebP magic bytes."""
        webp_data = b"RIFF\x00\x00\x00\x00WEBPVP8 "
        filename, mime = image_resource._detect_image_format(webp_data)
        assert filename == "image.webp"
        assert mime == "image/webp"

    def test_detect_gif(self, image_resource):
        """Line 800-801: GIF magic bytes."""
        gif_data = b"GIF89a\x01\x00\x01\x00"
        filename, mime = image_resource._detect_image_format(gif_data)
        assert filename == "image.gif"
        assert mime == "image/gif"

    def test_detect_unknown_defaults_to_png(self, image_resource):
        """Lines 802-803: Unknown format defaults to PNG."""
        unknown_data = b"UNKNOWN\x00\x00\x00"
        filename, mime = image_resource._detect_image_format(unknown_data)
        assert filename == "image.png"
        assert mime == "image/png"


# ===========================================================================
# Gap 10: _is_url() helper
# ===========================================================================


class TestIsUrl:
    """Tests for _is_url() helper method."""

    @pytest.fixture
    def image_resource(self):
        mock_client = Mock()
        return Image(mock_client)

    def test_is_url_https(self, image_resource):
        assert image_resource._is_url("https://example.com/img.png") is True

    def test_is_url_http(self, image_resource):
        assert image_resource._is_url("http://example.com/img.png") is True

    def test_is_url_not_url(self, image_resource):
        assert image_resource._is_url("/path/to/file.png") is False
        assert image_resource._is_url("data:image/png;base64,abc") is False
        assert image_resource._is_url("just-a-string") is False
