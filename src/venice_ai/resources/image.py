"""
Venice AI Image Generation API Resources.

This module provides asynchronous client interfaces for Venice AI's comprehensive
Image Generation API, enabling developers to create, manipulate, and enhance images
using advanced AI models. The Image API supports a wide range of image generation
and editing operations with fine-grained control over output characteristics.

Key Features:
    - **AI Image Generation**: Create original images from text prompts
    - **Image Upscaling**: Enhance image resolution while preserving quality
    - **Image Editing**: Modify existing images based on text instructions
    - **Style Management**: Access and apply predefined artistic styles
    - **Format Control**: Support for JPEG, PNG, WebP output formats
    - **Advanced Parameters**: Fine-tune generation with CFG scale, steps, seeds
    - **Batch Generation**: Create multiple image variants efficiently
    - **Asynchronous Operations**: Full async/await support for scalable applications

Supported Operations:
    - **Text-to-Image Generation**: Transform descriptive text into visual content
    - **Image Enhancement**: Upscale images with optional creative enhancement
    - **Image Editing**: Modify existing images using natural language instructions
    - **Style Discovery**: List available artistic styles for image generation
    - **Format Conversion**: Generate images in various formats and resolutions

The image generation process leverages state-of-the-art diffusion models to produce
high-quality visual content suitable for diverse applications:
    - **Creative Content**: Artwork, illustrations, and concept art
    - **Marketing Materials**: Product images, advertisements, and promotional content
    - **Prototyping**: Visual mockups and design iterations
    - **Content Enhancement**: Image upscaling and quality improvement
    - **Artistic Exploration**: Style transfer and creative image manipulation

Example:
    .. code-block:: python

        import asyncio
        from venice_ai import VeniceClient

        async def generate_artwork():
            async with VeniceClient() as client:
                # Generate an original image from text
                response = await client.image.create(
                    model="venice-sd35",
                    prompt="A serene mountain landscape at sunset",
                    width=1024,
                    height=768,
                    style_preset="cinematic"
                )

                # Extract and save the generated image
                import base64
                image_data = base64.b64decode(response.images[0])
                with open("mountain_sunset.png", "wb") as f:
                    f.write(image_data)

        asyncio.run(generate_artwork())

Performance Considerations:
    - Image generation time scales with complexity and resolution
    - Batch generation is more efficient than individual requests
    - Higher step counts improve quality but increase generation time
    - Model selection affects both quality and processing speed

Note:
    All operations in this module are asynchronous and require proper async/await
    handling. The Image class is accessed through the :attr:`VeniceClient.image`
    property and provides comprehensive image generation and manipulation capabilities.
"""

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Literal,
    cast,
    overload,
)

import aiohttp

from .._resource import APIResource
from ..helpers import detect_image_format
from ..types.api import (
    ImageBackgroundRemoveRequest,
    ImageEditRequest,
    # Request models
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageStylesResponse,
    ImageUpscaleRequest,
    SimpleImageGenerationRequest,
    SimpleImageGenerationResponse,
    StyleReference,
)
from ..validation.validators import validate_model_id

logger = logging.getLogger(__name__)


async def _read_binary_response(response: Any, *, endpoint: str) -> bytes:
    """Read raw image bytes from a binary HTTP response, with fallbacks.

    aiohttp's ``content.read()`` occasionally returns empty bytes on the first
    call — under proxies, certain aiohttp versions, or non-standard servers —
    even though the body is present. This shared reader lets every binary image
    endpoint (generate / edit / upscale / multi-edit) recover the bytes instead
    of silently returning ``b""``.

    Args:
        response: The raw response object returned with ``raw_response=True``
            (an ``aiohttp.ClientResponse``), already-read ``bytes``, or a
            mock-like object exposing ``.content``.
        endpoint: API path used for the fallback metric label.

    Returns:
        The response body as ``bytes``.
    """
    if isinstance(response, bytes):
        return response

    if isinstance(response, aiohttp.ClientResponse):
        content = await response.content.read()
        if not content:
            logger.warning(
                "Image %s response: content.read() returned empty, trying fallback methods",
                endpoint,
            )
            # Track fallback metrics
            try:
                from ..observability.metrics import get_enhanced_metrics

                metrics = get_enhanced_metrics()
                if metrics._enabled:
                    metrics.streaming_fallback_total.labels(
                        endpoint=endpoint, reason="empty_content_read"
                    ).inc()
            except Exception:
                pass  # nosec B110

            try:
                # Try reading from the response directly.
                content = await response.read()
                logger.debug("response.read() fallback returned: %d bytes", len(content))
            except Exception as e:
                logger.debug("response.read() fallback failed: %s", e)
                # Some HTTP client implementations buffer the body elsewhere.
                content_attr = getattr(response, "_content", None)
                if content_attr:
                    content = content_attr
                    logger.debug("Internal _content attribute fallback: %d bytes", len(content))
        return content

    # Mock-like or already-materialized responses.
    if hasattr(response, "content"):
        content_attr = response.content
        if isinstance(content_attr, bytes):
            return content_attr
        if hasattr(content_attr, "read"):
            return cast(bytes, await content_attr.read())

    return cast(bytes, response)


if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


class ImageJob:
    """Manages a single image generation request as an async context manager.

    Mirrors :class:`venice_ai.resources.music.MusicJob` /
    :class:`venice_ai.resources.video.VideoJob` for API symmetry — but image
    generation is synchronous on the server (no queue / poll endpoint), so the
    work happens inside :meth:`wait` and there is no server-side state to
    clean up on exit. The shape matters for parallel rendering::

        async with VeniceClient() as client:
            jobs = [
                await client.image.submit(model=m, prompt=p)
                for p in prompts
            ]
            # Hold all jobs open while rendering in parallel
            async with contextlib.AsyncExitStack() as stack:
                for j in jobs:
                    await stack.enter_async_context(j)
                images = await client.gather(
                    [j.wait() for j in jobs],
                    max_concurrency=4,
                )

    For one-off rendering, ``client.image.create(...)`` is still the
    shorter path.
    """

    def __init__(
        self,
        client: "VeniceClient",
        kwargs: dict[str, Any],
    ) -> None:
        self.model: str = kwargs.get("model", "")
        self._client = client
        self._kwargs = kwargs
        self._result: ImageGenerationResponse | bytes | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "ImageJob":
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        # No server-side state to release; the request either completed or
        # was cancelled by the caller's task. Mirror the music/video shape
        # so callers can use the same async-with idiom across modalities.
        return None

    @property
    def is_complete(self) -> bool:
        return self._result is not None

    async def wait(self) -> ImageGenerationResponse | bytes:
        """Run the image generation and cache the result.

        Subsequent calls return the cached value without re-issuing the
        request. Raises whatever :meth:`Image.create` would raise.
        """
        async with self._lock:
            if self._result is None:
                self._result = await self._client.image.create(**self._kwargs)
            result = self._result
        assert result is not None  # set above, narrows for the type-checker
        return result

    async def download(self, path: str | Path) -> Path:
        """Save the rendered image to ``path`` and return the resolved Path.

        Calls :meth:`wait` if the result hasn't been fetched yet. The output
        format is inferred from ``self._kwargs.get("format")`` and the API
        response shape.
        """
        from ..exceptions import VeniceError

        result = await self.wait()
        out = Path(path)
        if isinstance(result, (bytes, bytearray)):
            out.write_bytes(bytes(result))
            return out
        # ImageGenerationResponse — first image, base64-decoded
        if not result.images:
            raise VeniceError("ImageJob.download() called but the response has no images")
        out.write_bytes(base64.b64decode(result.images[0]))
        return out


class Image(APIResource["VeniceClient"]):
    """
    Provides access to asynchronous image generation, upscaling, and style listing operations.

    This class manages asynchronous image operations using Venice AI's image API.
    It encapsulates functionality for image generation, upscaling, and style listing
    through a clean, typed interface that makes asynchronous HTTP requests.

    All methods in this class make asynchronous HTTP requests using async/await syntax.

    :param client: The Venice AI client instance used for making API requests.
    :type client: venice_ai._client.VeniceClient
    """

    async def _prepare_image_content(self, image: str | bytes | BinaryIO) -> str | bytes:
        """
        Convert different image input types to bytes or pass-through strings asynchronously.

        For HTTP/HTTPS URLs and base64/data-URL strings the original string is
        returned as-is so callers can include it directly in JSON payloads.
        For file paths the file content is read and returned as bytes.

        :param image: Image input as path string, URL, base64 string, bytes,
                      or file-like object

        :return: Image content as bytes, or the original URL / base64 string

        :raises ValueError: If image path is invalid or encoding fails
        :raises TypeError: If image content type is unsupported
        :raises VeniceError: If there are errors reading from file-like objects
        """
        from ..exceptions import VeniceError

        if isinstance(image, str):
            # Check if it's an HTTP/HTTPS URL — pass through directly
            if self._is_url(image):
                return image
            # Check if it's base64 / data-URL — pass through directly
            if self._is_base64(image):
                return image
            # Otherwise treat as file path
            image_path = Path(image)
            try:
                return image_path.read_bytes()
            except FileNotFoundError:
                raise VeniceError(f"Image file not found at path: {image}") from None
            except OSError as e:
                raise VeniceError(f"Error reading image file at path {image}: {e}") from e
        elif isinstance(image, bytes):
            # image is raw bytes
            return image
        elif isinstance(image, io.BytesIO):
            # More specific than BinaryIO for .read()
            return image.read()
        elif isinstance(image, io.StringIO):
            # Handle StringIO objects specifically - convert text to bytes
            content = image.read()
            return content.encode("utf-8")
        elif isinstance(image, (io.RawIOBase, io.BufferedIOBase, io.TextIOBase)) or (
            hasattr(image, "read")
            and callable(image.read)
            and not isinstance(image, (bytes, bytearray, memoryview))
        ):
            # Handle file-like objects with proper type narrowing
            try:
                # Handle file-like objects
                result = image.read()
                if asyncio.iscoroutine(result):
                    file_content = await result
                else:
                    file_content = result

                if isinstance(file_content, bytes):
                    return file_content
                elif isinstance(file_content, str):
                    # Text content from file-like object is not valid for image processing
                    raise VeniceError(
                        "Image source is a file-like object that did not return bytes from read()"
                    )
                else:
                    raise TypeError(
                        f"Unsupported content type from file-like object: {type(file_content)}"
                    )
            except (ValueError, TypeError, AttributeError, OSError) as e:
                if isinstance(e, (ValueError, TypeError)):
                    raise
                raise VeniceError(f"Error reading from image file-like object: {e}") from e
        else:
            # Reject any input that is not a path, bytes, or file-like object.
            raise VeniceError("Unsupported image type")

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        cfg_scale: float | None = None,
        embed_exif_metadata: bool | None = None,
        enable_web_search: bool | None = None,
        format: Literal["jpeg", "png", "webp"] | None = None,
        height: int | None = None,
        hide_watermark: bool | None = None,
        lora_strength: int | None = None,
        num_images: int | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
        style_references: list[StyleReference | dict] | None = None,
        resolution: str | None = None,
        return_binary: Literal[False] = ...,
        safe_mode: bool | None = None,
        seed: int | None = None,
        steps: int | None = None,
        style_preset: str | None = None,
        width: int | None = None,
    ) -> ImageGenerationResponse: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        cfg_scale: float | None = None,
        embed_exif_metadata: bool | None = None,
        enable_web_search: bool | None = None,
        format: Literal["jpeg", "png", "webp"] | None = None,
        height: int | None = None,
        hide_watermark: bool | None = None,
        lora_strength: int | None = None,
        num_images: int | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
        style_references: list[StyleReference | dict] | None = None,
        resolution: str | None = None,
        return_binary: Literal[True] = ...,
        safe_mode: bool | None = None,
        seed: int | None = None,
        steps: int | None = None,
        style_preset: str | None = None,
        width: int | None = None,
    ) -> bytes: ...

    async def create(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        cfg_scale: float | None = None,
        embed_exif_metadata: bool | None = None,
        enable_web_search: bool | None = None,
        format: Literal["jpeg", "png", "webp"] | None = None,
        height: int | None = None,
        hide_watermark: bool | None = None,
        lora_strength: int | None = None,
        num_images: int | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
        style_references: list[StyleReference | dict] | None = None,
        resolution: str | None = None,
        return_binary: bool | None = None,
        safe_mode: bool | None = None,
        seed: int | None = None,
        steps: int | None = None,
        style_preset: str | None = None,
        width: int | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> ImageGenerationResponse | bytes:
        """
        Generate an image using Venice AI's image generation API asynchronously.

        This method creates a new image based on a text prompt using the specified
        model, executing the request asynchronously for use in async/await contexts.
        It provides comprehensive control over the image generation process
        with multiple parameters to customize the output.

        :param model: Model ID for image generation (e.g., ``"venice-sd35"``).
        :type model: str
        :param prompt: Text prompt describing the image to generate.
        :type prompt: str
        :param aspect_ratio: Optional. Aspect ratio for the output (e.g. ``"1:1"``,
            ``"16:9"``). Supported values vary by model; inspect ``GET /models`` for
            per-model allowed values.
        :type aspect_ratio: Optional[str]
        :param cfg_scale: Optional. Classifier Free Guidance scale (range: (0, 20]). Higher values adhere more strictly to the prompt.
        :type cfg_scale: Optional[float]
        :param embed_exif_metadata: Optional. If ``True``, embed generation metadata in EXIF data.
        :type embed_exif_metadata: Optional[bool]
        :param enable_web_search: Optional. If set, the image model may incorporate
            recent web-search context. Supported by models with the ``supportsWebSearch``
            capability.
        :type enable_web_search: Optional[bool]
        :param format: Optional. Output image format.
        :type format: Optional[Literal["jpeg", "png", "webp"]]
        :param height: Optional. Height of the generated image in pixels.
        :type height: Optional[int]
        :param hide_watermark: Optional. If ``True``, hide Venice AI watermark from the generated image.
        :type hide_watermark: Optional[bool]
        :param lora_strength: Optional. Strength of LoRA model adaptation (0-100).
        :type lora_strength: Optional[int]
        :param num_images: Optional. Number of images to generate (1-4).
        :type num_images: Optional[int]
        :param quality: Optional. Output quality for quality-aware models (e.g. GPT Image 2).
            Higher values can increase the request charge. See the model spec's ``qualities``
            field for supported values per model.
        :type quality: Optional[Literal["low", "medium", "high"]]
        :param resolution: Optional. Output resolution: ``"1K"``, ``"2K"``, or ``"4K"`` for supported models with resolution-based pricing.
        :type resolution: Optional[str]
        :param return_binary: Optional. If ``True``, return raw image bytes instead of JSON response with base64 data.
        :type return_binary: Optional[bool]
        :param safe_mode: Optional. If ``True``, enable content filtering for safer outputs.
        :type safe_mode: Optional[bool]
        :param seed: Optional. Random seed for reproducible image generation results.
        :type seed: Optional[int]
        :param steps: Optional. Number of diffusion steps. Higher values generally improve quality but increase generation time.
        :type steps: Optional[int]
        :param style_preset: Optional. Style preset ID from :meth:`list_styles` to apply to the generated image.
        :type style_preset: Optional[str]
        :param width: Optional. Width of the generated image in pixels.
        :type width: Optional[int]
        :param timeout: Optional. Per-request timeout override (seconds or an
            ``aiohttp.ClientTimeout``). Useful for slow renders such as
            ``quality='high'`` that exceed the client default. Falls back to the
            client's configured timeout when omitted.
        :type timeout: Optional[Union[float, aiohttp.ClientTimeout]]

        :return: Response containing generated image data as base64 string, or raw image bytes if ``return_binary`` is ``True``.

        :raises venice_ai.exceptions.APIError: If an API error occurs during image generation.

        **Example:**

        .. code-block:: python

            async with VeniceClient() as client:
                response = await client.image.create(
                    model="venice-sd35",
                    prompt="A serene landscape with mountains and a lake",
                    width=1024,
                    height=768,
                    steps=30
                )
                # Process response.images[0] (base64 string)
        """
        # Validate model ID
        validate_model_id(model, "model")

        # Create Pydantic request model
        generation_request = ImageGenerationRequest(
            model=model,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            cfg_scale=cfg_scale,
            embed_exif_metadata=embed_exif_metadata,
            enable_web_search=enable_web_search,
            format=format,
            height=height,
            hide_watermark=hide_watermark,
            lora_strength=lora_strength,
            variants=num_images,
            quality=quality,
            enhance_prompt=enhance_prompt,
            disable_prompt_optimization_thinking=disable_prompt_optimization_thinking,
            # The public signature also takes plain dicts for convenience;
            # pydantic coerces them (and rejects malformed ones) on construction.
            style_references=cast(
                "list[StyleReference] | None",
                style_references,
            ),
            resolution=resolution,
            return_binary=return_binary,
            safe_mode=safe_mode,
            seed=seed,
            steps=steps,
            style_preset=style_preset,
            width=width,
        )

        # Convert to API payload
        body = generation_request.model_dump(exclude_none=True)

        # Determine headers based on return_binary
        headers = None
        if return_binary:
            headers = {"Accept": "image/*"}
            logger.debug("Calling _request with return_binary=True")
            response = await self._client._request(
                method="POST",
                path="image/generate",
                json_data=body,
                headers=headers,
                raw_response=True,
                timeout=timeout,
            )
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response: {response!r}")

            return await _read_binary_response(response, endpoint="image/generate")
        else:
            response = await self._client.post(
                "image/generate",
                json_data=body,
                cast_to=ImageGenerationResponse,
                timeout=timeout,
            )
            return response

    async def submit(
        self,
        *,
        model: str,
        prompt: str,
        aspect_ratio: str | None = None,
        cfg_scale: float | None = None,
        embed_exif_metadata: bool | None = None,
        enable_web_search: bool | None = None,
        format: Literal["jpeg", "png", "webp"] | None = None,
        height: int | None = None,
        hide_watermark: bool | None = None,
        lora_strength: int | None = None,
        num_images: int | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
        style_references: list[StyleReference | dict] | None = None,
        resolution: str | None = None,
        return_binary: bool | None = None,
        safe_mode: bool | None = None,
        seed: int | None = None,
        steps: int | None = None,
        style_preset: str | None = None,
        width: int | None = None,
    ) -> ImageJob:
        """Build an :class:`ImageJob` for parallel-friendly image generation.

        Mirrors the shape of :meth:`Music.run` / :meth:`Video.run` so callers
        can use the same ``async with await client.image.submit(...) as job:``
        idiom across modalities. Image generation is synchronous on the
        server (no queue endpoint), so the actual HTTP request fires inside
        :meth:`ImageJob.wait`. Use this when you want to render multiple
        images in parallel via :meth:`VeniceClient.gather` and need each
        one's lifecycle bound to a context manager.

        For a single one-shot render, :meth:`Image.create` is shorter.

        Accepts the same keyword arguments as :meth:`Image.create` minus the
        ``negative_prompt`` parameter (removed in v2.0.0).

        Returns:
            :class:`ImageJob` ready to use as an async context manager.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "cfg_scale": cfg_scale,
            "embed_exif_metadata": embed_exif_metadata,
            "enable_web_search": enable_web_search,
            "format": format,
            "height": height,
            "hide_watermark": hide_watermark,
            "lora_strength": lora_strength,
            "num_images": num_images,
            "quality": quality,
            "enhance_prompt": enhance_prompt,
            "disable_prompt_optimization_thinking": disable_prompt_optimization_thinking,
            "style_references": style_references,
            "resolution": resolution,
            "return_binary": return_binary,
            "safe_mode": safe_mode,
            "seed": seed,
            "steps": steps,
            "style_preset": style_preset,
            "width": width,
        }
        # Drop None values so each kwarg behaves identically to a direct
        # ``Image.create`` call where the caller omits the same argument.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return ImageJob(client=self._client, kwargs=kwargs)

    async def upscale(
        self,
        *,
        image: str | bytes | BinaryIO | Path,
        scale: float | None = None,
        creativity: float | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> bytes:
        """
        Upscale an image using Venice AI's image upscaling API asynchronously.

        This method allows for increasing the resolution of an image while
        maintaining or enhancing its quality using Venice AI's upscaling technology,
        in an asynchronous manner compatible with asyncio applications.

        :param image: Image to upscale. Can be a file path (string or :class:`pathlib.Path`),
            raw image bytes, or a file-like object.

        :param scale: Optional. Scaling factor for upscaling. Must be ``2`` or
                      ``4``; ``1`` is rejected. Defaults to ``2``.
        :type scale: Optional[float]
        :param creativity: Optional. How much detail and texture the upscaler
                           adds — higher adds more, lower stays closer to the
                           source. The server clamps this to ``0``-``0.02``
                           (default ``0.01``).
        :type creativity: Optional[float]
        :param timeout: Optional. Request timeout configuration.
        :type timeout: Optional[Union[float, aiohttp.ClientTimeout]]

        :return: Raw bytes of the upscaled image.

        :raises ValueError: If image path is invalid or image type is unsupported.
        :raises TypeError: If image content type is unsupported.
        :raises venice_ai.exceptions.APIError: If an API error occurs during upscaling.
        """
        from ..exceptions import VeniceError

        # Normalise Path → str so _prepare_image_content treats it as a file path
        if isinstance(image, Path):
            image = str(image)

        # Convert image input to bytes using async helper method
        try:
            image_content = await self._prepare_image_content(image)
        except VeniceError as e:
            # Let VeniceError propagate for file not found cases and text mode file errors
            if (
                str(e).startswith("Image file not found at path:")
                or str(e)
                == "Image source is a file-like object that did not return bytes from read()"
            ):
                raise
            # Surface unsupported image types as a TypeError.
            if str(e) == "Unsupported image type":
                # For async, all unsupported types use the same message
                raise TypeError("Unsupported image type") from e
            # Wrap other VeniceError in ValueError for consistency
            raise ValueError(f"Invalid image source or parameters: {e}") from e

        # Upscale requires actual image bytes (not URL strings)
        if isinstance(image_content, str):
            raise VeniceError(
                "Upscale requires image file data, not a URL or base64 string. "
                "Pass a file path, bytes, or file-like object."
            )

        # Base64 encode the image content for Pydantic validation
        image_b64 = base64.b64encode(image_content).decode("utf-8")

        # Create and validate the Pydantic request model
        upscale_kwargs: dict[str, Any] = {"image": image_b64}
        if scale is not None:
            upscale_kwargs["scale"] = scale
        if creativity is not None:
            upscale_kwargs["creativity"] = creativity
        upscale_request = ImageUpscaleRequest(**upscale_kwargs)

        ext, mime_type = detect_image_format(image_content)
        if ext == "bin":
            ext, mime_type = "png", "image/png"
        filename = f"image.{ext}"

        # For multipart form data, we still need to send the actual image bytes
        # But we've validated all parameters through Pydantic
        files = {"image": (filename, image_content, mime_type)}

        # Convert validated request to form data (excluding the image field)
        request_dict = upscale_request.model_dump(exclude_none=True, exclude={"image"})

        # Use serialize_form_value to properly handle booleans
        from ..utils import serialize_form_value

        data = {k: serialize_form_value(v) for k, v in request_dict.items()}

        # Send request as multipart form using the new _request_multipart method
        response_content = await self._request_multipart(
            method="POST",
            path="image/upscale",
            files=files,
            data=data,
            headers={"Accept": "image/*"},
            timeout=timeout,
        )

        # The multipart helper returns already-read bytes in the normal case; the
        # shared reader also recovers the body when the first read comes back empty.
        return await _read_binary_response(response_content, endpoint="image/upscale")

    async def list_styles(self) -> ImageStylesResponse:
        """
        List available image style presets asynchronously for use with image generation.

        This method retrieves all available style presets that can be used with
        the ``style_preset`` parameter in the :meth:`create` method to influence the
        aesthetic and artistic style of generated images. It performs this operation asynchronously.

        :return: A list of available image style presets with their identifiers.

        :raises venice_ai.exceptions.APIError: If an API error occurs while retrieving styles.
        """
        response = await self._client.get("image/styles")
        # Convert raw response to ImageStylesResponse object
        return ImageStylesResponse(**response)

    async def edit(
        self,
        *,
        prompt: str,
        model: str | None = None,
        image: str | bytes | BinaryIO | Path,
        aspect_ratio: str | None = None,
        safe_mode: bool | None = None,
        resolution: str | None = None,
        output_format: Literal["jpeg", "png", "webp"] | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
        timeout: float | aiohttp.ClientTimeout | None = None,
    ) -> bytes:
        """
        Edit an image based on a text prompt asynchronously.

        This method modifies an existing image according to text instructions,
        such as changing colors, removing objects, or altering scenes using
        Venice AI's image editing capabilities, in an asynchronous manner
        compatible with asyncio applications.

        Binary inputs (file paths, bytes, file-like objects) are base64-encoded
        and sent in the JSON body. Base64 strings and URLs are forwarded as-is.

        :param prompt: Text directions to edit or modify the image.
                       Per-model cap via ``promptCharacterLimit`` on GET /models;
                       the endpoint spec ceiling is 32768 characters.
        :type prompt: str
        :param model: Optional. Edit model to use (e.g., ``"flux-2-max-edit"``,
                      ``"gpt-image-1-5-edit"``, ``"nano-banana-pro-edit"``).
                      Defaults to the API's server-side default edit model when
                      omitted.
        :type model: Optional[str]
        :param image: The image to edit. Can be:
                      - A file path (string or ``Path``)
                      - Raw image bytes
                      - A file-like object opened in binary mode
                      - A base64-encoded string
                      - An HTTP/HTTPS URL
        :param aspect_ratio: Optional. Aspect ratio for the output (e.g. ``"1:1"``,
            ``"16:9"``). Omit to use the model's default; supported values vary by model.
        :type aspect_ratio: Optional[str]
        :param safe_mode: Optional. When ``True`` (the server-side default) the
            API blurs images classified as adult content. Pass ``False`` to
            disable blurring on adult-capable models.
        :type safe_mode: Optional[bool]
        :param resolution: Optional. Resolution tier for the output image.
            Supported values: ``"1K"``, ``"2K"``, ``"4K"``; defaults to
            ``"1K"`` server-side when omitted.
        :type resolution: Optional[str]
        :param output_format: Optional. Output format for the edited image.
            When omitted, the format is inferred from resolution (PNG for 1K
            edits, JPEG for 2K/4K edits).
        :type output_format: Optional[Literal["jpeg", "png", "webp"]]
        :param timeout: Optional. Request timeout configuration. Pass a float
            (seconds) or an :class:`aiohttp.ClientTimeout` instance. Overrides
            the client-level default for this call only.
        :type timeout: Optional[Union[float, aiohttp.ClientTimeout]]

        :return: The edited image as raw bytes.

        :raises ValueError: If the prompt exceeds maximum length or image format is invalid.
        :raises TypeError: If image content type is unsupported.
        :raises venice_ai.exceptions.VeniceError: If image file is not found or API error occurs.

        **Example:**

        .. code-block:: python

            async with VeniceClient() as client:
                # Edit an image file
                with open('sunset.jpg', 'rb') as f:
                    edited = await client.image.edit(
                        prompt="Change the sky to a sunrise",
                        image=f,
                    )

                # Save the result
                with open('sunrise.jpg', 'wb') as f:
                    f.write(edited)

                # Pick a specific edit model
                edited = await client.image.edit(
                    prompt="Remove the background",
                    image="photo.jpg",
                    model="flux-2-max-edit",
                )

                # Edit using a URL, with safe_mode disabled
                edited = await client.image.edit(
                    prompt="Add a rainbow",
                    image="https://example.com/photo.jpg",
                    safe_mode=False,
                )
        """
        # Binary inputs are base64-encoded into the JSON body. URL and base64
        # strings are forwarded verbatim. The endpoint validates ``model`` and
        # ``safe_mode`` server-side and rejects unknown keys, so unset fields
        # are dropped via ``exclude_none=True``.
        mode, image_bytes = await self._prepare_image_for_request(image)

        if mode == "multipart":
            if image_bytes is None:
                raise RuntimeError(
                    "Internal invariant: _prepare_image_for_request returned mode='multipart' "
                    "without bytes. Please report this as a bug."
                )
            image_value = base64.b64encode(image_bytes).decode("utf-8")
        else:
            image_value = str(image)

        headers = {"Accept": "image/*"}

        edit_request = ImageEditRequest(
            prompt=prompt,
            model=model,
            image=image_value,
            aspect_ratio=aspect_ratio,
            safe_mode=safe_mode,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
            enhance_prompt=enhance_prompt,
            disable_prompt_optimization_thinking=disable_prompt_optimization_thinking,
        )
        payload = edit_request.model_dump(exclude_none=True)

        response = await self._client._request(
            method="POST",
            path="image/edit",
            json_data=payload,
            headers=headers,
            raw_response=True,
            timeout=timeout,
        )

        return await _read_binary_response(response, endpoint="image/edit")

    def _is_url(self, value: str) -> bool:
        """Check if a string value is an HTTP/HTTPS URL."""
        return value.startswith("http://") or value.startswith("https://")

    def _is_base64(self, value: str) -> bool:
        """Check if a string value looks like base64 data (not a file path or URL)."""
        if self._is_url(value):
            return False
        # Check for data URL prefix
        if value.startswith("data:"):
            return True
        # If it doesn't look like a file path (no path separator, no extension pattern)
        # and is long enough, treat as base64
        return bool(
            "/" not in value and "\\" not in value and "." not in value and len(value) > 100
        )

    def _detect_image_format(self, data: bytes) -> tuple[str, str]:
        """Detect image format from magic bytes and return (filename, mime_type).

        Thin wrapper around :func:`venice_ai.helpers.detect_image_format` that
        composes the ``image.<ext>`` filename callers expect for multipart uploads.
        Unknown formats fall back to ``("image.png", "image/png")`` to match
        the historical default for upload paths.
        """
        ext, mime_type = detect_image_format(data)
        if ext == "bin":
            ext, mime_type = "png", "image/png"
        return f"image.{ext}", mime_type

    async def _prepare_image_for_request(
        self,
        image: str | bytes | BinaryIO | Path,
    ) -> tuple[str, bytes | None]:
        """Prepare an image input, returning (mode, content).

        Returns:
            A tuple of (mode, content) where:
            - mode is "multipart" and content is bytes if file path/bytes/BinaryIO
            - mode is "json_base64" and content is None (use original str) if base64
            - mode is "json_url" and content is None (use original str) if URL

        :raises VeniceError: If image path is invalid or cannot be read.
        :raises TypeError: If image type is unsupported.
        """
        if isinstance(image, Path):
            result = await self._prepare_image_content(str(image))
            # Path inputs always resolve to bytes (never URL/base64 passthrough)
            return "multipart", cast(bytes, result)
        elif isinstance(image, bytes):
            return "multipart", image
        elif isinstance(image, str):
            if self._is_url(image):
                return "json_url", None
            elif self._is_base64(image):
                return "json_base64", None
            else:
                # Treat as file path — always resolves to bytes
                result = await self._prepare_image_content(image)
                return "multipart", cast(bytes, result)
        elif hasattr(image, "read"):
            result = await self._prepare_image_content(image)
            # File-like objects always resolve to bytes
            return "multipart", cast(bytes, result)
        else:
            raise TypeError("Unsupported image type")

    async def background_remove(
        self,
        *,
        image: str | bytes | BinaryIO | Path | None = None,
        image_url: str | None = None,
    ) -> bytes:
        """Remove background from an image (POST /image/background-remove).

        Remove the background from an image using AI. The image can be provided
        as a file path, bytes, file-like object, base64 string, or URL.
        Returns a PNG image with transparent background.

        :param image: Image as file path, bytes, file-like object, or base64 string.
        :type image: Optional[Union[str, bytes, BinaryIO, Path]]
        :param image_url: HTTP/HTTPS URL of the image.
        :type image_url: Optional[str]

        :return: PNG image bytes with transparent background.

        :raises ValueError: If neither image nor image_url is provided.
        :raises venice_ai.exceptions.APIError: If an API error occurs.

        **Example:**

        .. code-block:: python

            async with VeniceClient() as client:
                # Remove background from a file
                result = await client.image.background_remove(
                    image="photo.jpg"
                )
                with open("no_bg.png", "wb") as f:
                    f.write(result)

                # Remove background from a URL
                result = await client.image.background_remove(
                    image_url="https://example.com/photo.jpg"
                )
        """
        if image is None and image_url is None:
            raise ValueError("Either 'image' or 'image_url' must be provided")

        headers = {"Accept": "image/*"}

        if image_url is not None:
            # URL mode — send as {"image_url": "..."} per the API spec
            request = ImageBackgroundRemoveRequest(image_url=image_url)
            payload = request.model_dump(exclude_none=True)

            response = await self._client._request(
                method="POST",
                path="image/background-remove",
                json_data=payload,
                headers=headers,
                raw_response=True,
            )
        elif image is not None:
            mode, image_bytes = await self._prepare_image_for_request(image)

            if mode == "multipart":
                if image_bytes is None:
                    raise RuntimeError(
                        "Internal invariant: _prepare_image_for_request returned mode='multipart' "
                        "without bytes. Please report this as a bug."
                    )
                filename, mime_type = self._detect_image_format(image_bytes)
                files = {"image": (filename, image_bytes, mime_type)}

                response = await self._request_multipart(
                    method="POST",
                    path="image/background-remove",
                    files=files,
                    headers=headers,
                )
            else:
                # JSON mode (base64 or data URI)
                image_value = str(image)
                request = ImageBackgroundRemoveRequest(image=image_value)
                payload = request.model_dump(exclude_none=True)

                response = await self._client._request(
                    method="POST",
                    path="image/background-remove",
                    json_data=payload,
                    headers=headers,
                    raw_response=True,
                )
        else:
            raise ValueError("Either 'image' or 'image_url' must be provided")

        if isinstance(response, bytes):
            return response
        elif isinstance(response, aiohttp.ClientResponse):
            return await response.content.read()
        if hasattr(response, "content"):
            return cast(bytes, response.content)  # pyright: ignore[reportAttributeAccessIssue]  # hasattr narrowing not propagated
        return cast(bytes, response)

    async def multi_edit(
        self,
        *,
        prompt: str,
        model: str | None = None,
        image: str | bytes | BinaryIO | Path | None = None,
        image_2: str | bytes | BinaryIO | Path | None = None,
        image_3: str | bytes | BinaryIO | Path | None = None,
        safe_mode: bool | None = None,
        resolution: str | None = None,
        aspect_ratio: str | None = None,
        output_format: Literal["jpeg", "png", "webp"] | None = None,
        quality: Literal["low", "medium", "high"] | None = None,
        enhance_prompt: bool | None = None,
        disable_prompt_optimization_thinking: bool | None = None,
    ) -> bytes:
        """Edit an image using up to 3 layered inputs (POST /image/multi-edit).

        Composite up to three images with a single prompt. The first image is
        the base; remaining images are layered on top. Supports both base64
        strings and URLs; bytes/file-like/Path inputs are encoded as base64.

        :param prompt: Edit instruction describing the desired changes.
        :type prompt: str
        :param model: Edit model to use (sent as ``modelId``).
        :type model: Optional[str]
        :param image: Base image (file path, bytes, file-like object, base64, or URL).
        :type image: Optional[Union[str, bytes, BinaryIO, Path]]
        :param image_2: Second layer image.
        :type image_2: Optional[Union[str, bytes, BinaryIO, Path]]
        :param image_3: Third layer image.
        :type image_3: Optional[Union[str, bytes, BinaryIO, Path]]
        :param safe_mode: When ``True`` (server default) blur adult content.
            Pass ``False`` to disable.
        :type safe_mode: Optional[bool]
        :param resolution: Optional. Resolution tier for the output image.
            Supported values: ``"1K"``, ``"2K"``, ``"4K"``; defaults to
            ``"1K"`` server-side when omitted.
        :type resolution: Optional[str]
        :param aspect_ratio: Optional. Aspect ratio for the output (e.g.
            ``"1:1"``, ``"16:9"``). Omit to infer from the first input image;
            supported values vary by model.
        :type aspect_ratio: Optional[str]
        :param output_format: Optional. Output format for the edited image.
            When omitted, the format is inferred from resolution (PNG for 1K
            edits, JPEG for 2K/4K edits).
        :type output_format: Optional[Literal["jpeg", "png", "webp"]]
        :param quality: Optional. Output quality for quality-aware models (e.g.
            GPT Image 2). Higher values can increase the request charge.
        :type quality: Optional[Literal["low", "medium", "high"]]

        :return: Edited image bytes.

        :raises venice_ai.exceptions.APIError: If an API error occurs.

        **Example:**

        .. code-block:: python

            async with VeniceClient() as client:
                # Multi-edit with file uploads
                result = await client.image.multi_edit(
                    prompt="Replace the sky with a sunset",
                    image="photo.jpg",
                    image_2="sky_overlay.png",
                )
                with open("edited.png", "wb") as f:
                    f.write(result)

                # Multi-edit with URLs
                result = await client.image.multi_edit(
                    prompt="Blend these images together",
                    image="https://example.com/base.jpg",
                    image_2="https://example.com/overlay.jpg",
                )
        """
        # /image/multi-edit accepts {prompt, images[], modelId, safe_mode,
        # resolution, aspect_ratio, output_format, quality} per the Venice spec.
        # Bytes/file inputs are encoded to base64 and the request body is always
        # JSON.

        async def _to_b64_or_str(
            val: str | bytes | BinaryIO | Path,
        ) -> str:
            m, b = await self._prepare_image_for_request(val)
            if m == "multipart":
                if b is None:
                    raise RuntimeError(
                        "Internal invariant: _prepare_image_for_request returned mode='multipart' "
                        "without bytes. Please report this as a bug."
                    )
                return base64.b64encode(b).decode("utf-8")
            return str(val)

        images_list: list[str] = []

        if image is not None:
            images_list.append(await _to_b64_or_str(image))
        if image_2 is not None:
            images_list.append(await _to_b64_or_str(image_2))
        if image_3 is not None:
            images_list.append(await _to_b64_or_str(image_3))

        # Per the docs (api-reference/endpoint/image/multi-edit), ``images`` is
        # required and must contain 1-3 items. Surface this as a ValueError at
        # the call site rather than letting the API reject an empty array.
        if not images_list:
            raise ValueError(
                "multi_edit requires at least one image; provide image, image_2, or image_3."
            )

        headers = {"Accept": "image/*"}

        payload: dict[str, object] = {
            "prompt": prompt,
            "images": images_list,
        }
        if model is not None:
            payload["modelId"] = model
        if safe_mode is not None:
            payload["safe_mode"] = safe_mode
        if resolution is not None:
            payload["resolution"] = resolution
        if aspect_ratio is not None:
            payload["aspect_ratio"] = aspect_ratio
        if output_format is not None:
            payload["output_format"] = output_format
        if quality is not None:
            payload["quality"] = quality
        if enhance_prompt is not None:
            payload["enhance_prompt"] = enhance_prompt
        if disable_prompt_optimization_thinking is not None:
            payload["disable_prompt_optimization_thinking"] = disable_prompt_optimization_thinking

        response = await self._client._request(
            method="POST",
            path="image/multi-edit",
            json_data=payload,
            headers=headers,
            raw_response=True,
        )

        return await _read_binary_response(response, endpoint="image/multi-edit")

    async def simple_generate(
        self,
        *,
        prompt: str,
        model: str | None = None,
        n: int | None = None,
        size: (
            Literal[
                "auto",
                "256x256",
                "512x512",
                "1024x1024",
                "1536x1024",
                "1024x1536",
                "1792x1024",
                "1024x1792",
            ]
            | None
        ) = None,
        response_format: Literal["b64_json", "url"] | None = None,
        output_format: Literal["jpeg", "png", "webp"] | None = None,
        quality: Literal["auto", "high", "medium", "low", "hd", "standard"] | None = None,
        style: Literal["vivid", "natural"] | None = None,
        background: Literal["transparent", "opaque", "auto"] | None = None,
        moderation: Literal["low", "auto"] | None = None,
        output_compression: int | None = None,
        user: str | None = None,
    ) -> SimpleImageGenerationResponse:
        """Generate an image via the OpenAI-compatible ``POST /images/generations`` endpoint.

        Drop-in replacement for the OpenAI Images API; accepts the
        OpenAI-compatible parameter set rather than Venice's native fields. For
        full Venice features (LoRA, CFG, multi-variant) use :meth:`create`.

        :param prompt: Image description (≤1500 chars per spec).
        :param model: Venice model ID. Defaults to ``"default"`` server-side.
        :param n: Number of images. Venice supports ``n=1`` only.
        :param size: Output dimensions, e.g. ``"1024x1024"``. ``"auto"`` lets the
            server pick.
        :param response_format: ``"b64_json"`` (default) returns base64-encoded
            data; ``"url"`` returns a data URL.
        :param output_format: Image encoding (``jpeg``/``png``/``webp``).
        :param quality: Output quality. Supported by quality-aware models (e.g. GPT Image 2);
            higher values can increase the request charge. Ignored by models that do not
            advertise quality support (see the model spec's ``qualities`` field).
        :param style: OpenAI-compatibility flag.
        :param background: OpenAI-compatibility flag.
        :param moderation: ``"auto"`` enables Venice safe-mode; ``"low"``
            relaxes it.
        :param output_compression: 0–100; OpenAI-compatibility flag.
        :param user: End-user identifier; OpenAI-compatibility flag.

        :return: :class:`SimpleImageGenerationResponse` with ``created`` and
            ``data`` (a list of :class:`SimpleImageData`, each with
            ``b64_json`` or ``url``).
        """
        request = SimpleImageGenerationRequest(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format=response_format,
            output_format=output_format,
            quality=quality,
            style=style,
            background=background,
            moderation=moderation,
            output_compression=output_compression,
            user=user,
        )
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "images/generations", json_data=body, cast_to=SimpleImageGenerationResponse
        )
