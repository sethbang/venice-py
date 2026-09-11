"""
Comprehensive test coverage improvements for venice_ai.resources.image module.

This test file addresses the coverage gaps identified in the audit:
- Error handling in _prepare_image_content method (lines 141-142, 153-154, etc.)
- Binary response fallback logic in generate method (lines 319-332, 334-348)
- Error handling branches in upscale and edit methods (lines 504-515, 679-690)
- Branch coverage for conditional logic across all methods
"""

import io
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from venice_ai.exceptions import VeniceError
from venice_ai.resources.image import Image
from venice_ai.types.api import (
    ImageGenerationResponse,
    ImageStylesResponse,
)


class TestImagePrepareImageContent:
    """Test the _prepare_image_content method error handling and edge cases."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_prepare_image_content_file_not_found(self, image_resource):
        """Test _prepare_image_content with non-existent file path (lines 139-140)."""
        with pytest.raises(VeniceError) as exc_info:
            await image_resource._prepare_image_content("/nonexistent/file.png")

        assert "Image file not found at path:" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prepare_image_content_io_error(self, image_resource):
        """Test _prepare_image_content with IO error when reading file (lines 141-142)."""
        # Create a file and then make it unreadable
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test image data")
            temp_file_path = temp_file.name

        try:
            # Make file unreadable by changing permissions
            os.chmod(temp_file_path, 0o000)

            with pytest.raises(VeniceError) as exc_info:
                await image_resource._prepare_image_content(temp_file_path)

            assert "Error reading image file at path" in str(exc_info.value)
        finally:
            # Cleanup: restore permissions and delete file
            os.chmod(temp_file_path, 0o644)
            os.unlink(temp_file_path)

    @pytest.mark.asyncio
    async def test_prepare_image_content_string_io_text_mode(self, image_resource):
        """Test _prepare_image_content with StringIO object (lines 151-154)."""
        string_io = io.StringIO("text content")

        result = await image_resource._prepare_image_content(string_io)

        assert result == b"text content"

    @pytest.mark.asyncio
    async def test_prepare_image_content_bytes_io(self, image_resource):
        """Test _prepare_image_content with BytesIO object (lines 148-150)."""
        bytes_io = io.BytesIO(b"binary image data")

        result = await image_resource._prepare_image_content(bytes_io)

        assert result == b"binary image data"

    @pytest.mark.asyncio
    async def test_prepare_image_content_async_file_like_object(self, image_resource):
        """Test _prepare_image_content with async file-like object (lines 163-167)."""

        class AsyncFileObject:
            def __init__(self, content):
                self.content = content

            async def read(self):
                return self.content

        async_file = AsyncFileObject(b"async file content")

        result = await image_resource._prepare_image_content(async_file)

        assert result == b"async file content"

    @pytest.mark.asyncio
    async def test_prepare_image_content_sync_file_like_object(self, image_resource):
        """Test _prepare_image_content with sync file-like object (lines 163-167)."""

        class SyncFileObject:
            def __init__(self, content):
                self.content = content

            def read(self):
                return self.content

        sync_file = SyncFileObject(b"sync file content")

        result = await image_resource._prepare_image_content(sync_file)

        assert result == b"sync file content"

    @pytest.mark.asyncio
    async def test_prepare_image_content_file_object_returns_text(self, image_resource):
        """Test _prepare_image_content when file object returns text (lines 171-175)."""

        class TextFileObject:
            def read(self):
                return "text content not bytes"

        text_file = TextFileObject()

        with pytest.raises(VeniceError) as exc_info:
            await image_resource._prepare_image_content(text_file)

        assert "Image source is a file-like object that did not return bytes from read()" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_prepare_image_content_file_object_returns_invalid_type(self, image_resource):
        """Test _prepare_image_content when file object returns invalid type (lines 176-179)."""

        class InvalidFileObject:
            def read(self):
                return 12345  # Invalid type

        invalid_file = InvalidFileObject()

        with pytest.raises(TypeError) as exc_info:
            await image_resource._prepare_image_content(invalid_file)

        assert "Unsupported content type from file-like object" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prepare_image_content_file_object_os_error(self, image_resource):
        """Test _prepare_image_content when file object raises OSError (lines 180-185)."""

        class FailingFileObject:
            def read(self):
                raise OSError("File system error")

        failing_file = FailingFileObject()

        with pytest.raises(VeniceError) as exc_info:
            await image_resource._prepare_image_content(failing_file)

        assert "Error reading from image file-like object" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prepare_image_content_integer_input(self, image_resource):
        """Test _prepare_image_content with integer input (lines 188-189)."""
        with pytest.raises(VeniceError) as exc_info:
            await image_resource._prepare_image_content(12345)

        assert "Unsupported image type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prepare_image_content_other_invalid_type(self, image_resource):
        """Test _prepare_image_content with other invalid types (lines 190-191)."""
        with pytest.raises(VeniceError) as exc_info:
            await image_resource._prepare_image_content({"not": "valid"})

        assert "Unsupported image type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_prepare_image_content_valid_bytes(self, image_resource):
        """Test _prepare_image_content with valid bytes input."""
        test_bytes = b"valid image data"

        result = await image_resource._prepare_image_content(test_bytes)

        assert result == test_bytes

    @pytest.mark.asyncio
    async def test_prepare_image_content_valid_file_path(self, image_resource):
        """Test _prepare_image_content with valid file path."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            test_data = b"test image content"
            temp_file.write(test_data)
            temp_file_path = temp_file.name

        try:
            result = await image_resource._prepare_image_content(temp_file_path)
            assert result == test_data
        finally:
            os.unlink(temp_file_path)


class TestImageGenerateBinaryResponseHandling:
    """Test binary response handling and VCR fallback logic in generate method."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_generate_binary_response_bytes(self, image_resource):
        """Test generate method when _request returns bytes (lines 310-312)."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"binary image data")
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"binary image data"
        mock_client._request.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_binary_response_client_response_with_content(self, image_resource):
        """Test generate method with ClientResponse that has content (lines 313-317)."""
        mock_client = AsyncMock()

        # Mock ClientResponse
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.content = Mock()
        mock_response.content.read = AsyncMock(return_value=b"response content")

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"response content"

    @pytest.mark.asyncio
    async def test_generate_binary_response_empty_content_vcr_fallback(self, image_resource):
        """Test VCR fallback when content is empty (lines 319-331)."""
        mock_client = AsyncMock()

        # Mock ClientResponse with empty content
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.content = Mock()
        mock_response.content.read = AsyncMock(return_value=b"")  # Empty content
        mock_response.read = AsyncMock(return_value=b"fallback content")

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"fallback content"

    @pytest.mark.asyncio
    async def test_generate_binary_response_vcr_content_attribute_fallback(self, image_resource):
        """Test VCR _content attribute fallback (lines 326-330)."""
        mock_client = AsyncMock()

        # Mock ClientResponse with empty content and read exception
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.content = Mock()
        mock_response.content.read = AsyncMock(return_value=b"")  # Empty content
        mock_response.read = AsyncMock(side_effect=Exception("Read failed"))
        mock_response._content = b"vcr content attribute"

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"vcr content attribute"

    @pytest.mark.asyncio
    async def test_generate_binary_response_unknown_type_with_content_attr(self, image_resource):
        """Test unknown response type with content attribute (lines 336-346)."""
        mock_client = AsyncMock()

        # Mock unknown response type with content attribute
        mock_response = Mock()
        mock_response.content = b"content attribute bytes"

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"content attribute bytes"

    @pytest.mark.asyncio
    async def test_generate_binary_response_unknown_type_with_readable_content(
        self, image_resource
    ):
        """Test unknown response type with readable content attribute (lines 342-346)."""
        mock_client = AsyncMock()

        # Mock unknown response type with readable content
        mock_content = Mock()
        mock_content.read = AsyncMock(return_value=b"readable content")

        mock_response = Mock()
        mock_response.content = mock_content

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == b"readable content"

    @pytest.mark.asyncio
    async def test_generate_binary_response_final_fallback(self, image_resource):
        """Test final fallback casting for unknown response (lines 347-348)."""
        mock_client = AsyncMock()

        # Mock unknown response type without content attribute
        mock_response = "unknown response type"

        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model", prompt="test prompt", return_binary=True
        )

        assert result == "unknown response type"

    @pytest.mark.asyncio
    async def test_generate_json_response_path(self, image_resource):
        """Test generate method JSON response path (lines 349-353)."""
        mock_client = AsyncMock()
        # Mock timing object properly
        from venice_ai.types.api.base import TimingInfo

        timing = TimingInfo(
            inferenceDuration=500.0,
            inferencePreprocessingTime=100.0,
            inferenceQueueTime=50.0,
            total=1000.0,
        )

        mock_response = ImageGenerationResponse(
            id="test-id",
            images=["base64_image_data"],
            request={"model": "test-model", "prompt": "test prompt"},
            timing=timing,
        )
        mock_client.post = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.create(
            model="test-model",
            prompt="test prompt",
            return_binary=False,  # JSON response
        )

        assert result == mock_response
        mock_client.post.assert_called_once()


class TestImageUpscaleErrorHandling:
    """Test error handling in the upscale method."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_upscale_venice_error_file_not_found_propagation(self, image_resource):
        """Test that VeniceError for file not found propagates (lines 504-509)."""
        # Mock _prepare_image_content to raise VeniceError for file not found
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Image file not found at path: /nonexistent")

            with pytest.raises(VeniceError) as exc_info:
                await image_resource.upscale(image="/nonexistent/file.png", scale=2.0)

            assert "Image file not found at path:" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upscale_venice_error_text_mode_file_propagation(self, image_resource):
        """Test that VeniceError for text mode file propagates (lines 506-509)."""
        # Mock _prepare_image_content to raise VeniceError for text mode file
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError(
                "Image source is a file-like object that did not return bytes from read()"
            )

            with pytest.raises(VeniceError) as exc_info:
                await image_resource.upscale(image=io.StringIO("text"), scale=2.0)

            assert (
                "Image source is a file-like object that did not return bytes from read()"
                in str(exc_info.value)
            )

    @pytest.mark.asyncio
    async def test_upscale_unsupported_image_type_conversion(self, image_resource):
        """Test conversion of unsupported image type to TypeError (lines 511-513)."""
        # Mock _prepare_image_content to raise VeniceError for unsupported type
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Unsupported image type")

            with pytest.raises(TypeError) as exc_info:
                await image_resource.upscale(image=12345, scale=2.0)

            assert "Unsupported image type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upscale_other_venice_error_to_value_error(self, image_resource):
        """Test conversion of other VeniceError to ValueError (lines 514-515)."""
        # Mock _prepare_image_content to raise other VeniceError
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Some other error")

            with pytest.raises(ValueError) as exc_info:
                await image_resource.upscale(image="test", scale=2.0)

            assert "Invalid image source or parameters: Some other error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_upscale_successful_flow(self, image_resource):
        """Test successful upscale flow."""
        # Mock successful preparation and request
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = b"test image data"

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled image data"

                result = await image_resource.upscale(image=b"test", scale=2.0)

                assert result == b"upscaled image data"

    @pytest.mark.asyncio
    async def test_upscale_response_with_content_attribute(self, image_resource):
        """Test upscale when response has content attribute (lines 575-576)."""
        # Mock preparation and response with content attribute
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = b"test image data"

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_response = Mock()
                mock_response.content = b"response with content"
                mock_request.return_value = mock_response

                result = await image_resource.upscale(image=b"test", scale=2.0)

                assert result == b"response with content"


class TestImageEditErrorHandling:
    """Test error handling in the edit method."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_edit_venice_error_file_not_found_propagation(self, image_resource):
        """Test that VeniceError for file not found propagates (lines 679-684)."""
        # Mock _prepare_image_content to raise VeniceError for file not found
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Image file not found at path: /nonexistent")

            with pytest.raises(VeniceError) as exc_info:
                await image_resource.edit(prompt="test", image="/nonexistent/file.png")

            assert "Image file not found at path:" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_edit_venice_error_text_mode_file_propagation(self, image_resource):
        """Test that VeniceError for text mode file propagates (lines 680-684)."""
        # Mock _prepare_image_content to raise VeniceError for text mode file
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError(
                "Image source is a file-like object that did not return bytes from read()"
            )

            with pytest.raises(VeniceError) as exc_info:
                await image_resource.edit(prompt="test", image=io.StringIO("text"))

            assert (
                "Image source is a file-like object that did not return bytes from read()"
                in str(exc_info.value)
            )

    @pytest.mark.asyncio
    async def test_edit_unsupported_image_type_conversion(self, image_resource):
        """Test conversion of unsupported image type to TypeError (lines 686-688)."""
        # Mock _prepare_image_content to raise VeniceError for unsupported type
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Unsupported image type")

            with pytest.raises(TypeError) as exc_info:
                await image_resource.edit(prompt="test", image=12345)

            assert "Unsupported image type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_edit_other_venice_error_propagation(self, image_resource):
        """Test that other VeniceError propagates from _prepare_image_for_request."""
        # Mock _prepare_image_content to raise other VeniceError
        # (called internally by _prepare_image_for_request for file-path strings)
        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.side_effect = VeniceError("Some other error")

            with pytest.raises(VeniceError) as exc_info:
                await image_resource.edit(prompt="test", image="test")

            assert "Some other error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_edit_successful_bytes_response_json(self, image_resource):
        """Test successful edit with bytes response (JSON mode, bytes→base64)."""
        # edit() always uses JSON mode; bytes are base64-encoded
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"edited image data")
        image_resource._client = mock_client

        result = await image_resource.edit(prompt="test edit", image=b"test")

        assert result == b"edited image data"
        mock_client._request.assert_called_once()

    @pytest.mark.asyncio
    async def test_edit_client_response_type_json(self, image_resource):
        """Test edit with ClientResponse type (JSON mode, bytes→base64)."""
        mock_response = Mock(spec=aiohttp.ClientResponse)
        mock_response.content = Mock()
        mock_response.content.read = AsyncMock(return_value=b"response content")

        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.edit(prompt="test edit", image=b"test")

        assert result == b"response content"

    @pytest.mark.asyncio
    async def test_edit_json_mode_with_url(self, image_resource):
        """Test edit takes JSON path when image is a URL."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"edited via json")
        image_resource._client = mock_client

        result = await image_resource.edit(
            prompt="test edit", image="https://example.com/photo.jpg"
        )

        assert result == b"edited via json"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["image"] == "https://example.com/photo.jpg"

    @pytest.mark.asyncio
    async def test_edit_unknown_response_type(self, image_resource):
        """Test edit with unknown response type that has content attribute."""
        # edit() always uses JSON mode; mock _client._request
        mock_response = Mock()
        mock_response.content = b"content attribute bytes"

        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=mock_response)
        image_resource._client = mock_client

        result = await image_resource.edit(prompt="test edit", image=b"test")

        assert result == b"content attribute bytes"

    @pytest.mark.asyncio
    async def test_edit_resolution_forwarded_in_payload(self, image_resource):
        """Test that resolution param is forwarded in the JSON payload to image/edit."""
        mock_client = AsyncMock()
        mock_client._request = AsyncMock(return_value=b"edited with resolution")
        image_resource._client = mock_client

        result = await image_resource.edit(
            prompt="x",
            image="data:image/png;base64,AA==",
            resolution="2K",
        )

        assert result == b"edited with resolution"
        mock_client._request.assert_called_once()
        call_kwargs = mock_client._request.call_args[1]
        assert call_kwargs["json_data"]["resolution"] == "2K"


class TestImageStyleMethods:
    """Test style-related methods."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_list_styles(self, image_resource):
        """Test list_styles method (lines 616-618)."""
        mock_client = AsyncMock()
        mock_response_data = {
            "object": "list",
            "data": ["cinematic", "photorealistic", "cartoon"],
        }
        mock_client.get = AsyncMock(return_value=mock_response_data)
        image_resource._client = mock_client

        result = await image_resource.list_styles()

        assert isinstance(result, ImageStylesResponse)
        mock_client.get.assert_called_once_with("image/styles")


class TestImageUpscaleImageFormatDetection:
    """Test image format detection in upscale method."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_upscale_detect_jpeg_format(self, image_resource):
        """Test JPEG format detection (lines 533-534)."""
        jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # JPEG header

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = jpeg_data

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled"

                await image_resource.upscale(image=jpeg_data, scale=2.0)

                # Check that the files parameter included correct MIME type
                call_args = mock_request.call_args
                files = call_args[1]["files"]
                filename, content, mime_type = files["image"]
                assert filename == "image.jpg"
                assert mime_type == "image/jpeg"

    @pytest.mark.asyncio
    async def test_upscale_detect_png_format(self, image_resource):
        """Test PNG format detection (lines 535-536)."""
        png_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"  # PNG header

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = png_data

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled"

                await image_resource.upscale(image=png_data, scale=2.0)

                call_args = mock_request.call_args
                files = call_args[1]["files"]
                filename, content, mime_type = files["image"]
                assert filename == "image.png"
                assert mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_upscale_detect_webp_format(self, image_resource):
        """Test WebP format detection (lines 537-538)."""
        webp_data = b"RIFF\x00\x00\x00\x00WEBPVP8 "  # WebP header

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = webp_data

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled"

                await image_resource.upscale(image=webp_data, scale=2.0)

                call_args = mock_request.call_args
                files = call_args[1]["files"]
                filename, content, mime_type = files["image"]
                assert filename == "image.webp"
                assert mime_type == "image/webp"

    @pytest.mark.asyncio
    async def test_upscale_detect_gif_format(self, image_resource):
        """Test GIF format detection (lines 539-540)."""
        gif_data = b"GIF89a\x01\x00\x01\x00"  # GIF header

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = gif_data

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled"

                await image_resource.upscale(image=gif_data, scale=2.0)

                call_args = mock_request.call_args
                files = call_args[1]["files"]
                filename, content, mime_type = files["image"]
                assert filename == "image.gif"
                assert mime_type == "image/gif"

    @pytest.mark.asyncio
    async def test_upscale_detect_unknown_format_defaults_to_png(self, image_resource):
        """Test unknown format defaults to PNG (lines 541-543)."""
        unknown_data = b"UNKNOWN\x00\x00\x00\x00"  # Unknown format

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = unknown_data

            with patch.object(image_resource, "_request_multipart") as mock_request:
                mock_request.return_value = b"upscaled"

                await image_resource.upscale(image=unknown_data, scale=2.0)

                call_args = mock_request.call_args
                files = call_args[1]["files"]
                filename, content, mime_type = files["image"]
                assert filename == "image.png"
                assert mime_type == "image/png"


class TestImageRequestHelperMethods:
    """Test _request_multipart helper method behavior."""

    @pytest.fixture
    def image_resource(self):
        """Create an Image resource instance for testing."""
        mock_client = Mock()
        # Add the _request_multipart method to the mock client
        mock_client._request_multipart = AsyncMock()
        return Image(mock_client)

    @pytest.mark.asyncio
    async def test_request_multipart_call_structure(self, image_resource):
        """Test that _request_multipart is called with correct structure."""
        test_data = b"test image"

        with patch.object(image_resource, "_prepare_image_content") as mock_prepare:
            mock_prepare.return_value = test_data

            # Setup the _request_multipart method on the image_resource directly
            image_resource._request_multipart = AsyncMock(return_value=b"result")

            await image_resource.upscale(image=test_data, scale=2, creativity=0.01)

            # Verify _request_multipart was called with expected structure
            image_resource._request_multipart.assert_called_once()
            call_args = image_resource._request_multipart.call_args

            assert call_args[1]["method"] == "POST"
            assert call_args[1]["path"] == "image/upscale"
            assert "files" in call_args[1]
            assert "data" in call_args[1]
            assert call_args[1]["headers"] == {"Accept": "image/*"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
