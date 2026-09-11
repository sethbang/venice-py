"""
Image generation models for Venice AI API.

This module contains comprehensive Pydantic models for image generation responses,
including native Venice image generation and OpenAI-compatible endpoints.
"""

import base64
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ...core.models.common import VeniceBaseModel
from ...helpers import detect_image_format
from .base import TimingInfo


class ImageGenerationResponse(VeniceBaseModel):
    """Native Venice image generation response"""

    # /image/generate has no additionalProperties:false (unlike /images/generations),
    # so tolerate and preserve forward-compatible server fields instead of forbidding.
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="The ID of the request")
    images: list[str] = Field(..., description="Base64 encoded image data")
    request: dict[str, Any] | None = Field(
        None, description="The original request data sent to the API"
    )
    timing: TimingInfo = Field(..., description="Performance timing information")

    def bytes(self, index: int = 0) -> bytes:
        """Return decoded image bytes for the image at *index* (default ``0``).

        Use this when you need the raw bytes in memory — for an HTTP upload,
        a buffer pipe, or further processing — rather than writing to disk.
        For batches, iterate ``response.bytes(i) for i in range(len(response.images))``.

        :param index: Index of the image in :attr:`images` (default ``0``).
        :return: Decoded image bytes.
        :raises IndexError: If *index* is out of range.
        """
        return base64.b64decode(self.images[index])

    def save(self, path: str | Path, index: int = 0, *, overwrite: bool = False) -> Path:
        """Decode base64 image and save to file.

        If *path* has no suffix, the correct extension is sniffed from the
        decoded image bytes (PNG, WebP, JPEG, GIF) and appended. Pass an
        explicit suffix (``"img.png"``) to override.

        Performs synchronous file I/O. When called from an async coroutine,
        wrap with ``await asyncio.to_thread(response.save, path)`` for large
        outputs.

        :param path: Destination file path. May omit the extension.
        :param index: Index of the image to save (default ``0``).
        :param overwrite: If ``False`` (default) and *path* exists, raise
            :class:`FileExistsError`.
        :return: The resolved :class:`Path` actually written (reflects any
            auto-appended extension).
        :raises FileExistsError: If the file exists and ``overwrite=False``.
        """
        path = Path(path)
        raw = base64.b64decode(self.images[index])
        if not path.suffix:
            ext, _ = detect_image_format(raw)
            if ext == "bin":
                ext = "png"
            path = path.with_suffix(f".{ext}")
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path

    def save_all(
        self,
        directory: str | Path,
        prefix: str = "image",
        ext: str | None = None,
        *,
        overwrite: bool = False,
    ) -> list[Path]:
        """Save all generated images to a directory.

        With ``ext=None`` (default) the extension is sniffed **per image**
        from each one's magic bytes — so a mixed batch of PNG and WebP
        images gets correctly-suffixed files. Pass ``ext="png"`` (or any
        extension) to force a uniform suffix.

        Performs synchronous file I/O. When called from an async coroutine,
        wrap with ``await asyncio.to_thread(response.save_all, directory)``.

        :param directory: Target directory (created if needed).
        :param prefix: Filename prefix (default ``"image"``).
        :param ext: File extension, with or without leading dot. ``None``
            (default) auto-detects from each image's magic bytes individually.
        :param overwrite: If ``False`` (default) and any target file exists,
            raise :class:`FileExistsError` before writing anything.
        :return: List of saved file paths.
        :raises FileExistsError: If a target file exists and ``overwrite=False``.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if not self.images:
            return []

        decoded = [base64.b64decode(img) for img in self.images]

        if ext is None:
            extensions: list[str] = []
            for raw in decoded:
                detected, _ = detect_image_format(raw)
                extensions.append("png" if detected == "bin" else detected)
        else:
            ext_clean = ext.lstrip(".")
            extensions = [ext_clean] * len(decoded)

        targets = [directory / f"{prefix}_{i}.{extensions[i]}" for i in range(len(decoded))]
        if not overwrite:
            for t in targets:
                if t.exists():
                    raise FileExistsError(t)
        for t, raw in zip(targets, decoded, strict=True):
            t.write_bytes(raw)
        return targets

    @property
    def enhanced_prompt(self) -> str | None:
        """The rewritten prompt, when ``enhance_prompt=True`` produced one.

        The API returns it URL-encoded in the ``x-venice-enhanced-prompt``
        response header; this decodes it. ``None`` when the header is absent,
        which is the normal case for a request that did not ask for a rewrite.
        """
        headers = self.headers
        if not headers:
            return None
        raw = headers.get("x-venice-enhanced-prompt")
        if raw is None:
            for key, value in headers.items():
                if key.lower() == "x-venice-enhanced-prompt":
                    raw = value
                    break
        if raw is None:
            return None
        from urllib.parse import unquote

        return unquote(raw)


class SimpleImageData(BaseModel):
    """OpenAI-compatible image data object"""

    model_config = ConfigDict(extra="allow")

    b64_json: str | None = Field(None, description="Base64-encoded image data")
    url: str | None = Field(None, description="Data URL of the generated image")


class SimpleImageGenerationResponse(VeniceBaseModel):
    """OpenAI-compatible image generation response"""

    created: int = Field(..., description="Unix timestamp for when the request was created")
    data: list[SimpleImageData] = Field(..., description="Array of generated image objects")


class ImageStylesResponse(BaseModel):
    """Image styles list response"""

    model_config = ConfigDict(extra="allow")

    object: Literal["list"] = Field(..., description="Object type")
    data: list[str] = Field(..., description="List of available image styles")


__all__ = [
    "ImageGenerationResponse",
    "SimpleImageData",
    "SimpleImageGenerationResponse",
    "ImageStylesResponse",
]
