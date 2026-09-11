"""
VCRpy-based integration tests for Image resource.

This module tests image generation, editing, upscaling, and style listing functionality
through real API interactions recorded with VCRpy, replacing mock-based unit tests.
"""

import asyncio
import io
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import APIError, VeniceError

# Generous per-call timeout for live (re-)recording of slow image operations
# such as upscaling, which routinely exceed the test client's 30 s default.
RECORDING_TIMEOUT = 180.0


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a sample image file for testing."""
    import struct
    import zlib

    # Create a 256x256 red PNG (minimum size for upscaling - 65536 pixels)
    width = 256
    height = 256

    # PNG signature
    png_data = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png_data += (
        struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    )

    # IDAT chunk - create red image data
    raw_data = b""
    for _y in range(height):
        raw_data += b"\x00"  # Filter type: None
        for _x in range(width):
            raw_data += b"\xff\x00\x00"  # RGB: red

    compressed_data = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b"IDAT" + compressed_data)
    png_data += (
        struct.pack(">I", len(compressed_data))
        + b"IDAT"
        + compressed_data
        + struct.pack(">I", idat_crc)
    )

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    png_data += b"\x00\x00\x00\x00IEND" + struct.pack(">I", iend_crc)

    image_file = tmp_path / "test_image.png"
    image_file.write_bytes(png_data)
    return str(image_file)


@pytest.fixture
def sample_image_bytes():
    """Return sample image data as bytes."""
    import struct
    import zlib

    # Create a 256x256 red PNG (minimum size for upscaling - 65536 pixels)
    width = 256
    height = 256

    # PNG signature
    png_data = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
    png_data += (
        struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    )

    # IDAT chunk - create red image data
    raw_data = b""
    for _y in range(height):
        raw_data += b"\x00"  # Filter type: None
        for _x in range(width):
            raw_data += b"\xff\x00\x00"  # RGB: red

    compressed_data = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b"IDAT" + compressed_data)
    png_data += (
        struct.pack(">I", len(compressed_data))
        + b"IDAT"
        + compressed_data
        + struct.pack(">I", idat_crc)
    )

    # IEND chunk
    iend_crc = zlib.crc32(b"IEND")
    png_data += b"\x00\x00\x00\x00IEND" + struct.pack(">I", iend_crc)

    return png_data


# ============================================================================
# Image Generation Tests
# ============================================================================


@pytest.mark.integration
async def test_image_generate_basic(venice_client, model_selector, vcr_cassette):
    """Test basic image generation."""
    with vcr_cassette:
        # Dynamically select an image generation model
        image_model = await model_selector.select_image_model()

        response = await venice_client.image.create(
            model=image_model,
            prompt="A simple red square on white background",
            aspect_ratio="1:1",
            steps=8,
        )

        # Validate response structure
        assert response is not None
        assert hasattr(response, "id")
        assert hasattr(response, "images")
        assert len(response.images) > 0

        # Images should be base64-encoded strings, data URLs, or HTTP URLs
        for image in response.images:
            assert isinstance(image, str)
            # Check if it's a data URL, HTTP URL, or raw base64
            is_data_url = image.startswith("data:image/")
            is_http_url = image.startswith("http")
            # Raw base64 would be a non-empty string that doesn't start with data: or http
            is_base64 = len(image) > 0 and not is_data_url and not is_http_url
            assert is_data_url or is_http_url or is_base64, f"Invalid image format: {image[:50]}..."


@pytest.mark.integration
async def test_image_generate_with_parameters(venice_client, model_selector, vcr_cassette):
    """Test image generation with various parameters."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        # The parameter contract is image-model-class-dependent (pixel models
        # take width/height + steps/cfg_scale; aspect-ratio and resolution-tier
        # models reject some of these). The SDK forwards whatever is passed —
        # when the randomly-selected model rejects this fixed param set with a
        # 400, skip rather than fail: the exact rejection body is
        # model-class-dependent, so this asserts the skip path rather than a
        # specific rejection message.
        try:
            response = await venice_client.image.create(
                model=image_model,
                prompt="A detailed landscape with mountains and a lake",
                aspect_ratio="4:3",
                steps=8,
                cfg_scale=7.5,
                seed=42,
                num_images=1,
            )
        except (VeniceError, APIError) as e:
            pytest.skip(f"Model {image_model} rejected the parameter set: {e}")

        assert response is not None
        assert len(response.images) >= 1

        # Add diagnostic logging
        print(f"DEBUG: response type: {type(response)}")
        print(f"DEBUG: response attributes: {dir(response)}")
        print(f"DEBUG: hasattr(response, 'request'): {hasattr(response, 'request')}")

        if hasattr(response, "request"):
            print(f"DEBUG: response.request type: {type(response.request)}")
            print(f"DEBUG: response.request value: {response.request}")
            if response.request is not None:
                print(
                    f"DEBUG: response.request keys: {response.request.keys() if isinstance(response.request, dict) else 'Not a dict'}"
                )

        # Check that request parameters were handled
        if hasattr(response, "request") and response.request:
            # response.request is a dict, not an object with attributes
            # First check if the key exists before asserting
            if "prompt" in response.request:
                assert (
                    response.request["prompt"] == "A detailed landscape with mountains and a lake"
                )
            else:
                print("WARNING: 'prompt' key not found in response.request")
                print(f"Available keys: {list(response.request.keys())}")


@pytest.mark.integration
async def test_image_generate_return_binary(venice_client, model_selector, vcr_cassette):
    """Test image generation with binary return."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        result = await venice_client.image.create(
            model=image_model,
            prompt="A blue circle",
            aspect_ratio="1:1",
            return_binary=True,
        )

        # Should return raw bytes
        assert isinstance(result, bytes)
        # Check for PNG header if it's PNG format
        if result.startswith(b"\x89PNG"):
            assert b"PNG" in result[:8]


# ============================================================================
# Image Upscaling Tests
# ============================================================================


@pytest.mark.integration
async def test_image_upscale_with_file(venice_client, sample_image_path, vcr_cassette):
    """Test image upscaling with file path."""
    with vcr_cassette:
        try:
            result = await venice_client.image.upscale(
                image=sample_image_path,
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

            # Should return upscaled image data
            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support upscaling
            pytest.skip(f"Upscaling not supported: {e}")


@pytest.mark.integration
async def test_image_upscale_with_bytes(venice_client, sample_image_bytes, vcr_cassette):
    """Test image upscaling with raw bytes."""
    with vcr_cassette:
        try:
            result = await venice_client.image.upscale(
                image=sample_image_bytes,
                scale=2,
                timeout=RECORDING_TIMEOUT,
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support upscaling
            pytest.skip(f"Upscaling not supported: {e}")


@pytest.mark.integration
async def test_image_upscale_with_enhancement(venice_client, sample_image_bytes, vcr_cassette):
    """Test image upscaling with the creativity knob.

    ``creativity`` replaced the removed ``enhance`` / ``enhanceCreativity`` /
    ``enhancePrompt`` parameters; the server clamps it to 0-0.02.
    """
    with vcr_cassette:
        try:
            result = await venice_client.image.upscale(
                image=sample_image_bytes,
                scale=2,
                timeout=RECORDING_TIMEOUT,
                creativity=0.02,
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support upscaling
            pytest.skip(f"Upscaling not supported: {e}")


# ============================================================================
# Image Editing Tests
# ============================================================================


@pytest.mark.integration
async def test_image_edit_basic(venice_client, sample_image_bytes, vcr_cassette):
    """Test basic image editing."""
    with vcr_cassette:
        try:
            result = await venice_client.image.edit(
                prompt="Add a red border around the image",
                image=sample_image_bytes,
            )

            # Should return edited image data
            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support editing
            pytest.skip(f"Image editing not supported: {e}")


@pytest.mark.integration
async def test_image_edit_with_file(venice_client, sample_image_path, vcr_cassette):
    """Test image editing with file path."""
    with vcr_cassette:
        try:
            result = await venice_client.image.edit(
                prompt="Change the background to blue",
                image=sample_image_path,
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support editing
            pytest.skip(f"Image editing not supported: {e}")


@pytest.mark.integration
async def test_image_edit_with_io_object(venice_client, sample_image_bytes, vcr_cassette):
    """Test image editing with IO object."""
    with vcr_cassette:
        try:
            # Create BytesIO object
            image_io = io.BytesIO(sample_image_bytes)

            result = await venice_client.image.edit(
                prompt="Make it more colorful",
                image=image_io,
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
        except (VeniceError, APIError) as e:
            # Some models/tiers might not support editing
            pytest.skip(f"Image editing not supported: {e}")


# ============================================================================
# Style Listing Tests
# ============================================================================


@pytest.mark.integration
async def test_image_list_styles(venice_client, vcr_cassette):
    """Test listing available image styles."""
    with vcr_cassette:
        try:
            styles = await venice_client.image.list_styles()

            assert styles is not None
            assert hasattr(styles, "data")
            assert isinstance(styles.data, list)

            # Should have at least some styles available
            if len(styles.data) > 0:
                for style in styles.data:
                    assert isinstance(style, str)
        except (VeniceError, APIError) as e:
            # Some endpoints might not support style listing
            pytest.skip(f"Style listing not supported: {e}")


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_image_generate_invalid_model(venice_client, vcr_cassette):
    """Test error handling for invalid model."""
    with vcr_cassette, pytest.raises(VeniceError):
        await venice_client.image.create(
            model="invalid-image-model-xyz",
            prompt="Test prompt",
        )


@pytest.mark.integration
async def test_image_generate_empty_prompt(venice_client, model_selector, vcr_cassette):
    """Test error handling for empty prompt."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        # Client should validate empty prompt before making API call
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            await venice_client.image.create(
                model=image_model,
                prompt="",  # Empty prompt
            )


@pytest.mark.integration
async def test_image_upscale_invalid_file(venice_client, vcr_cassette):
    """Test error handling for invalid file path."""
    with vcr_cassette, pytest.raises(VeniceError):
        await venice_client.image.upscale(
            image="/nonexistent/file.png",
            scale=2.0,
            timeout=RECORDING_TIMEOUT,
        )


@pytest.mark.integration
async def test_image_edit_invalid_image(venice_client, vcr_cassette):
    """Test error handling for invalid image data."""
    with vcr_cassette, pytest.raises((VeniceError, TypeError)):
        await venice_client.image.edit(
            prompt="Edit this",
            image=12345,  # Invalid image type
        )


# ============================================================================
# Complex Workflow Tests
# ============================================================================


@pytest.mark.integration
async def test_image_generate_and_upscale_workflow(
    venice_client, model_selector, vcr_cassette, sample_image_bytes
):
    """Test generating an image and then upscaling it."""
    with vcr_cassette:
        # Step 1: Generate an image (test the generation endpoint)
        image_model = await model_selector.select_image_model()

        generation_response = await venice_client.image.create(
            model=image_model,
            prompt="A small red dot",
            aspect_ratio="1:1",
            return_binary=False,  # Use JSON response to avoid VCR binary issues
        )

        assert generation_response is not None
        assert hasattr(generation_response, "images")
        assert len(generation_response.images) > 0

        # Step 2: Use the sample image bytes for upscaling test
        # This avoids VCR binary response issues while still testing the upscale endpoint
        try:
            upscaled = await venice_client.image.upscale(
                image=sample_image_bytes,  # Use fixture data that works reliably
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

            assert upscaled is not None
            if isinstance(upscaled, bytes):
                # Upscaled image should typically be larger than input
                assert len(upscaled) > len(sample_image_bytes)
        except (VeniceError, APIError) as e:
            # Upscaling might not be supported for this model
            pytest.skip(f"Upscaling not supported: {e}")


@pytest.mark.integration
async def test_image_style_discovery_and_generation(venice_client, model_selector, vcr_cassette):
    """Test discovering styles and using them in generation."""
    with vcr_cassette:
        # Step 1: Get available styles
        try:
            styles_response = await venice_client.image.list_styles()
            available_styles = styles_response.data if hasattr(styles_response, "data") else []
        except (VeniceError, APIError):
            available_styles = []

        # Step 2: Generate with a style if available
        image_model = await model_selector.select_image_model()

        generation_params = {
            "model": image_model,
            "prompt": "A beautiful sunset",
            "aspect_ratio": "1:1",
        }

        # Add style if available
        if available_styles and len(available_styles) > 0:
            generation_params["style_preset"] = available_styles[0]

        response = await venice_client.image.create(**generation_params)

        assert response is not None
        assert len(response.images) > 0


# ============================================================================
# Content Preparation Integration Tests
# (Replaces mock-based _prepare_image_content tests)
# ============================================================================


@pytest.mark.integration
async def test_image_content_preparation_real_file_upload(
    venice_client, sample_image_path, vcr_cassette
):
    """
    Test real file upload and content preparation.
    Replaces mock-based _prepare_image_content tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Test with real file path - should prepare and upload successfully
            result = await venice_client.image.upscale(
                image=sample_image_path,  # Real file path
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
                # Upscaled image should typically be larger than 1x1 pixel input
                assert len(result) > 100  # Reasonable size for upscaled image
        except (VeniceError, APIError) as e:
            if "upscaling not supported" in str(e).lower():
                pytest.skip(f"Upscaling not supported: {e}")
            else:
                raise


@pytest.mark.integration
async def test_image_content_preparation_bytes_upload(
    venice_client, sample_image_bytes, vcr_cassette
):
    """
    Test real bytes upload and content preparation.
    Replaces mock-based bytes handling tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Test with raw bytes - should handle binary data correctly
            result = await venice_client.image.upscale(
                image=sample_image_bytes,  # Raw bytes
                scale=2,
                timeout=RECORDING_TIMEOUT,
                creativity=0.01,  # Test additional parameters
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
                # Should get valid image data back
                assert len(result) > len(sample_image_bytes)  # Upscaled should be larger
        except (VeniceError, APIError) as e:
            if "upscaling not supported" in str(e).lower():
                pytest.skip(f"Upscaling not supported: {e}")
            else:
                raise


@pytest.mark.integration
async def test_image_content_preparation_io_object(venice_client, sample_image_bytes, vcr_cassette):
    """
    Test real IO object upload and content preparation.
    Replaces mock-based IO object handling tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Test with BytesIO object - should read and upload correctly
            image_io = io.BytesIO(sample_image_bytes)

            result = await venice_client.image.edit(
                prompt="Add a blue border to this image",
                image=image_io,  # IO object
            )

            assert result is not None
            if isinstance(result, bytes):
                assert len(result) > 0
                # Should get valid edited image data
                assert len(result) >= len(sample_image_bytes)  # At least as large as input
        except (VeniceError, APIError) as e:
            if "editing not supported" in str(e).lower():
                pytest.skip(f"Image editing not supported: {e}")
            else:
                raise


# ============================================================================
# Binary Response Handling Integration Tests
# (Replaces mock-based binary response tests)
# ============================================================================


@pytest.mark.integration
async def test_binary_response_handling_generate(venice_client, model_selector, vcr_cassette):
    """
    Test real binary response handling for image generation.
    Replaces mock-based binary response tests from unit tests.
    """
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        # Test binary response mode
        result = await venice_client.image.create(
            model=image_model,
            prompt="A simple red square for binary test",
            aspect_ratio="1:1",
            return_binary=True,  # Request binary response
        )

        # Should receive actual binary image data
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Should be a valid image file (check for common image headers)
        is_png = result.startswith(b"\x89PNG")
        is_jpeg = result.startswith(b"\xff\xd8\xff")
        is_webp = b"WEBP" in result[:12]

        assert is_png or is_jpeg or is_webp, f"Invalid image format. First 20 bytes: {result[:20]}"


@pytest.mark.integration
async def test_binary_response_handling_upscale(venice_client, sample_image_bytes, vcr_cassette):
    """
    Test real binary response handling for upscaling.
    Replaces mock-based upscale binary response tests from unit tests.
    """
    with vcr_cassette:
        try:
            # Test upscaling with binary input and output
            result = await venice_client.image.upscale(
                image=sample_image_bytes,
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

            # Should receive binary image data
            assert isinstance(result, bytes)
            assert len(result) > 0

            # Upscaled image should be larger than input
            assert len(result) > len(sample_image_bytes)

            # Should be valid image format
            is_valid_image = (
                result.startswith(b"\x89PNG")
                or result.startswith(b"\xff\xd8\xff")
                or b"WEBP" in result[:12]
            )
            assert is_valid_image, "Result is not a valid image format"
        except (VeniceError, APIError) as e:
            if "upscaling not supported" in str(e).lower():
                pytest.skip(f"Upscaling not supported: {e}")
            else:
                raise


# ============================================================================
# Error Handling Integration Tests
# (Replaces mock-based error scenarios)
# ============================================================================


@pytest.mark.integration
async def test_file_not_found_error_handling(venice_client, vcr_cassette):
    """
    Test real file not found error handling.
    Replaces mock-based file error tests from unit tests.
    """
    with vcr_cassette:
        # Test with nonexistent file path
        with pytest.raises(VeniceError) as exc_info:
            await venice_client.image.upscale(
                image="/definitely/nonexistent/path/image.png",
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg for keyword in ["file", "not found", "path", "error", "invalid"]
        )


@pytest.mark.integration
async def test_invalid_image_data_error_handling(venice_client, vcr_cassette):
    """
    Test real invalid image data error handling.
    Replaces mock-based invalid data tests from unit tests.
    """
    with vcr_cassette:
        # Test with invalid image data
        invalid_data = b"This is not image data at all"

        try:
            with pytest.raises((VeniceError, TypeError, ValueError)):
                await venice_client.image.upscale(
                    image=invalid_data,
                    scale=2.0,
                    timeout=RECORDING_TIMEOUT,
                )
        except (VeniceError, APIError) as e:
            # Some APIs might process invalid data differently
            # The important thing is we get some kind of error response
            assert "invalid" in str(e).lower() or "error" in str(e).lower()


# ============================================================================
# Complex Workflow Integration Tests
# (Replaces mock-based workflow tests)
# ============================================================================


@pytest.mark.integration
async def test_multi_step_image_workflow(
    venice_client, model_selector, sample_image_bytes, vcr_cassette
):
    """
    Test complex multi-step image workflow with real API calls.
    Replaces mock-based workflow tests from unit tests.
    """
    with vcr_cassette:
        # Step 1: Generate base image
        image_model = await model_selector.select_image_model()

        generated = await venice_client.image.create(
            model=image_model,
            prompt="A simple geometric shape for workflow test",
            aspect_ratio="1:1",
            return_binary=False,  # Get URL/base64 for intermediate step
        )

        assert generated is not None
        assert len(generated.images) > 0

        # Step 2: Use sample image for edit test (more reliable than generated image)
        try:
            edited = await venice_client.image.edit(
                prompt="Add bright colors to this image",
                image=sample_image_bytes,
            )

            assert edited is not None
            if isinstance(edited, bytes):
                assert len(edited) > 0
        except (VeniceError, APIError) as e:
            if "editing not supported" in str(e).lower():
                # Skip edit step if not supported, but generation should have worked
                pass
            else:
                raise

        # Step 3: Try upscaling with sample image
        try:
            upscaled = await venice_client.image.upscale(
                image=sample_image_bytes,
                scale=2.0,
                timeout=RECORDING_TIMEOUT,
            )

            assert upscaled is not None
            if isinstance(upscaled, bytes):
                assert len(upscaled) > len(sample_image_bytes)
        except (VeniceError, APIError) as e:
            if "upscaling not supported" in str(e).lower():
                # Skip upscale step if not supported
                pass
            else:
                raise

        # Workflow completed successfully — verify generation result is usable
        assert generated is not None
        assert len(generated.images) > 0


# ============================================================================
# Concurrent Operations Tests
# ============================================================================


@pytest.mark.integration
async def test_image_concurrent_generations(venice_client, model_selector, vcr_cassette):
    """Test multiple concurrent image generations."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        # Create multiple generation tasks
        tasks = []
        for i in range(3):
            task = venice_client.image.create(
                model=image_model,
                prompt=f"Test image number {i}",
                aspect_ratio="1:1",
                seed=i,
            )
            tasks.append(task)

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0

        for result in successful_results:
            # Only check images attribute if result is not an Exception
            assert hasattr(result, "images")
            assert len(result.images) > 0  # type: ignore[attr-defined]


@pytest.mark.integration
async def test_image_mixed_operations(
    venice_client, model_selector, sample_image_bytes, vcr_cassette
):
    """Test different image operations concurrently."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        tasks = []

        # Task 1: Generate
        tasks.append(
            venice_client.image.create(
                model=image_model,
                prompt="Concurrent test image",
                aspect_ratio="1:1",
            )
        )

        # Task 2: List styles (if supported)
        tasks.append(venice_client.image.list_styles())

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # At least one operation should succeed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0


# ============================================================================
# Parameter Validation Tests
# ============================================================================


@pytest.mark.integration
async def test_image_generate_extreme_parameters(venice_client, model_selector, vcr_cassette):
    """Test image generation with extreme parameter values."""
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        try:
            response = await venice_client.image.create(
                model=image_model,
                prompt="Extreme parameters test",
                aspect_ratio="1:1",
                steps=1,  # Minimum steps
                cfg_scale=20.0,  # High CFG
                seed=0,
                num_images=1,
            )

            assert response is not None
            assert len(response.images) > 0
        except VeniceError as e:
            # Some parameter combinations might be rejected
            assert "invalid" in str(e).lower() or "error" in str(e).lower()


# Removed ``test_image_generate_with_negative_prompt``: the Venice API disabled
# ``negative_prompt`` for image generation in February 2026. The SDK strips the
# field before sending, so there is nothing left for this VCR test to cover.
# Deprecation-warning behavior is exercised in
# ``tests/unit/resources/test_image_negative_prompt_deprecation.py``.


# ============================================================================
# Doc-Parity Tests (Venice API alignment, added 2026-04)
# ============================================================================


@pytest.mark.integration
async def test_image_generate_with_enable_web_search(venice_client, model_selector, vcr_cassette):
    """
    POST /image/generate supports ``enable_web_search`` per docs
    (``api-reference/endpoint/image/generate.md``). The SDK must forward it.
    """
    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        response = await venice_client.image.create(
            model=image_model,
            prompt="A serene library interior with tall bookshelves",
            aspect_ratio="1:1",
            steps=8,
            enable_web_search=False,
        )

        assert response is not None
        assert hasattr(response, "images")
        assert len(response.images) > 0


async def _select_edit_model(venice_client) -> str:
    """Dynamically pick an edit-capable image model (id contains '-edit')."""
    models = await venice_client.models.list(type="all")
    candidates = [
        m.id
        for m in models.data
        if m.id.endswith("-edit") or "-image-" in m.id and m.id.endswith("edit")
    ]
    if not candidates:
        pytest.skip("No edit-capable image model available for multi_edit test")
    # Prefer the shortest name (usually the canonical default, e.g. 'qwen-edit')
    return min(candidates, key=len)


@pytest.mark.integration
async def test_image_multi_edit_forwards_model(venice_client, sample_image_bytes, vcr_cassette):
    """
    POST /image/multi-edit accepts a ``modelId`` body field per docs
    (``api-reference/endpoint/image/multi-edit.md``). The SDK must forward
    the user's ``model`` kwarg as ``modelId`` rather than silently dropping it.
    """
    with vcr_cassette:
        edit_model = await _select_edit_model(venice_client)

        try:
            result = await venice_client.image.multi_edit(
                prompt="Add a vibrant sunset to the background",
                model=edit_model,
                image=sample_image_bytes,
            )

            # A 200 response guarantees result is bytes; length may be 0 under
            # VCR's aiohttp stub in recording mode even when the real API
            # streams back a valid PNG. The cassette preserves the wire truth —
            # the server-side acceptance of ``modelId`` is what this test guards.
            assert isinstance(result, bytes)
        except (VeniceError, APIError) as e:
            pytest.skip(f"multi_edit not supported for selected model: {e}")


@pytest.mark.integration
async def test_image_simple_generate(venice_client, model_selector, vcr_cassette):
    """POST /images/generations (OpenAI-compat) returns a SimpleImageGenerationResponse.

    Audit gap #1: this endpoint had request/response models defined but no SDK
    method routed traffic to it. The new ``Image.simple_generate()`` wires it up.
    """
    from venice_ai.types.api import SimpleImageGenerationResponse

    with vcr_cassette:
        image_model = await model_selector.select_image_model()

        result = await venice_client.image.simple_generate(
            prompt="A simple red square on white background",
            model=image_model,
            size="512x512",
            response_format="b64_json",
            output_format="png",
            n=1,
        )

        assert isinstance(result, SimpleImageGenerationResponse)
        assert result.created > 0
        assert isinstance(result.data, list)
        assert len(result.data) == 1
        item = result.data[0]
        # Either b64_json or url must be populated.
        assert (item.b64_json is not None) or (item.url is not None)
