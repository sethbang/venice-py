"""
Venice AI Video API resources.

This module provides classes for interacting with the Venice AI Video API,
supporting video generation operations including text-to-video and image-to-video.

The video API allows for:
- Queuing video generation requests (text-to-video and image-to-video)
- Getting price quotes before generation
- Retrieving video generation results (polling for completion)
- Marking videos as complete / deleting from storage
- High-level VideoJob abstraction for lifecycle management
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

from .._resource import APIResource
from ..exceptions import InvalidRequestError, VideoGenerationError
from ..helpers import normalize_duration_seconds
from ..types.api.models import VideoModelSpec
from ..types.api.requests.video import (
    VideoCompleteRequest,
    VideoConsents,
    VideoElement,
    VideoImageToVideoRequest,
    VideoQuoteRequest,
    VideoRetrieveRequest,
    VideoTextToVideoRequest,
    VideoTranscriptionRequest,
)
from ..types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
    VideoRetrieveResponse,
    VideoTranscriptionResponse,
)

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)


def _format_video_duration(duration_seconds: int | str) -> str:
    """Render the public ``duration_seconds`` value as the wire form ``"{n}s"``.

    Public API takes int (or a string the user already typed), but the
    Venice ``/video/queue`` endpoint expects ``"5s"`` / ``"10s"``. Translate
    once at the resource boundary so the request model still matches the
    wire schema exactly.

    Some upscale models accept the sentinel string ``"Auto"`` (or other
    model-specific values) — when the input doesn't parse as a number of
    seconds, pass it through unchanged and let the server validate.
    """
    try:
        return f"{normalize_duration_seconds(duration_seconds)}s"
    except ValueError:
        if isinstance(duration_seconds, str):
            return duration_seconds
        raise


async def _preflight_validate_video_duration(
    client: VeniceClient,
    model_id: str,
    duration_seconds: int | str | None,
) -> None:
    """Pre-flight check ``duration_seconds`` against the model's supported tier list.

    Best-effort: silent fall-through on catalog miss or non-video spec.
    Raises a ``ValueError`` only when the spec exposes an explicit
    ``constraints.durations`` enum and the requested value isn't in it.
    """
    if duration_seconds is None:
        return
    try:
        wire_form = _format_video_duration(duration_seconds)
    except ValueError:
        return  # field validator will raise the right error
    try:
        entry = await client.models.get(model_id)
    except Exception:  # noqa: BLE001  - network/catalog miss => let server validate
        return
    spec = entry.model_spec
    if not isinstance(spec, VideoModelSpec):
        return
    constraints = spec.constraints
    if constraints is None or not constraints.durations:
        return
    if wire_form not in constraints.durations:
        raise ValueError(
            f"duration_seconds={duration_seconds!r} is not supported by model "
            f"{model_id!r}; allowed: {constraints.durations}"
        )


class VideoJob:
    """Manages the lifecycle of an async video generation request.

    Use as an async context manager to guarantee server-side cleanup::

        async with await client.video.run(model=model, prompt="...", duration="5s") as job:
            status = await job.wait()
            await job.download("output.mp4", status)
    """

    def __init__(self, client: VeniceClient, queue_response: VideoQueueResponse):
        self.model: str = queue_response.model
        self.queue_id: str = queue_response.queue_id
        self._client = client
        self._status: VideoRetrieveResponse | None = None
        # For VPS-backed models the file URL is handed back at queue time and
        # /video/retrieve only returns JSON status (no url/data). Keep it so
        # download() can fall back to it.
        self._download_url: str | None = getattr(queue_response, "download_url", None)

    async def __aenter__(self) -> VideoJob:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Guarantee server-side cleanup on exit.

        Propagates any in-flight exception from the user's block. If both the
        user code and cleanup raise, the user's exception wins; the cleanup
        failure is logged.
        """
        try:
            await self.cancel()
        except Exception as e:
            # A 400 ("Request ID is invalid") means the queue entry is already
            # gone — the job reached a terminal state and the server released it,
            # or it was never completable. That's benign on a normal exit (e.g.
            # queue → poll-to-completion → exit), so log it at DEBUG rather than
            # spamming a WARNING. Genuine cleanup failures (5xx, network) stay
            # at WARNING. Cleanup is best-effort either way and never raised.
            level = logging.DEBUG if isinstance(e, InvalidRequestError) else logging.WARNING
            if exc_type is None:
                logger.log(level, "VideoJob cleanup failed for queue_id=%s: %s", self.queue_id, e)
            else:
                logger.log(
                    level,
                    "VideoJob cleanup failed during exception handling "
                    "(queue_id=%s, original=%s): %s",
                    self.queue_id,
                    exc_type.__name__,
                    e,
                )
        # Returning ``None`` lets the original exception (if any) propagate.

    @property
    def status(self) -> VideoRetrieveResponse | None:
        """Last known status from polling."""
        return self._status

    @property
    def is_complete(self) -> bool:
        return isinstance(self._status, VideoCompletedStatus)

    @property
    def is_failed(self) -> bool:
        return isinstance(self._status, VideoFailedStatus)

    @property
    def progress(self) -> float | None:
        """Progress as a fraction 0.0–1.0, or ``None`` if not processing."""
        if isinstance(self._status, VideoProcessingStatus):
            return self._status.progress_percent / 100.0
        return None

    async def poll(self) -> VideoRetrieveResponse:
        """Single poll — check current status."""
        self._status = await self._client.video.retrieve(model=self.model, queue_id=self.queue_id)
        return self._status

    async def wait(
        self,
        *,
        poll_interval: float = 5.0,
        max_polls: int = 120,
        on_progress: Callable[[VideoProcessingStatus], None] | None = None,
    ) -> VideoCompletedStatus:
        """Poll until complete or failed. Returns completed status or raises.

        :param poll_interval: Seconds between polls.
        :param max_polls: Maximum number of polls before raising ``TimeoutError``.
        :param on_progress: Optional callback invoked on each processing status update.
        :raises VideoGenerationError: If the server reports generation failure.
        :raises TimeoutError: If ``max_polls`` is exhausted.
        """
        for _ in range(max_polls):
            status = await self.poll()
            if isinstance(status, VideoCompletedStatus):
                return status
            if isinstance(status, VideoFailedStatus):
                raise VideoGenerationError(
                    f"Video generation failed: {status.error}",
                    error_code=status.error_code,
                )
            if on_progress and isinstance(status, VideoProcessingStatus):
                on_progress(status)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Video generation did not complete within {max_polls} polls")

    async def download(self, path: str | Path, status: VideoCompletedStatus) -> Path:
        """Download a completed video to *path*. Does NOT call ``cancel()`` — use the context manager.

        File I/O is offloaded to a worker thread so the event loop never blocks.
        URL downloads reuse the SDK client's managed HTTP session, so proxy,
        SSL, timeout, and retry configuration are honored.

        :param path: Destination file path.
        :param status: A completed status (from :meth:`wait` or :meth:`poll`).
        :return: The resolved :class:`Path` of the saved file.
        """
        path = Path(path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        if status.data:
            await asyncio.to_thread(path.write_bytes, status.data)
        elif status.url:
            data = await self._client.fetch_external(status.url)
            await asyncio.to_thread(path.write_bytes, data)
        elif self._download_url:
            # VPS-backed models: the status carries neither inline bytes nor a
            # url; the downloadable file lives at the queue-time download_url.
            data = await self._client.fetch_external(self._download_url)
            await asyncio.to_thread(path.write_bytes, data)
        return path

    async def cancel(self) -> VideoCompleteResponse:
        """Release server-side storage / cancel an in-progress job.

        Wraps the ``/video/complete`` endpoint, which deletes the queue
        entry server-side regardless of whether the job has finished. Named
        ``cancel`` (rather than the wire-format ``complete``) to distinguish
        it from the :attr:`is_complete` state check — terminal states are
        polled via :meth:`wait` / :attr:`status`.
        """
        return await self._client.video.cancel(model=self.model, queue_id=self.queue_id)


class Video(APIResource["VeniceClient"]):
    """
    Asynchronous interface for Venice AI's Video generation API.

    The Video class provides access to Venice's video generation endpoints
    including text-to-video, image-to-video, quoting, retrieval, and completion.

    **Core Capabilities:**
        - **Text-to-Video**: Generate video from text prompts
        - **Image-to-Video**: Animate a reference image into video
        - **Price Quoting**: Get cost estimates before generation
        - **Result Retrieval**: Poll for video generation status and download URL
        - **Cleanup**: Mark videos as complete and delete from storage

    **Usage Patterns:**
        The Video class is accessed through the Venice AI client's
        :attr:`~venice_ai._client.VeniceClient.video` property rather than
        instantiated directly.

    **Typical Workflow:**
        1. Call :meth:`quote` to get a price estimate
        2. Call :meth:`submit` to start generation (returns a ``queue_id``)
        3. Poll :meth:`retrieve` with the ``queue_id`` until status is COMPLETED
        4. Download the video from the returned URL
        5. Call :meth:`cancel` to clean up server-side storage

    Args:
        client: The Venice AI client instance providing authentication
            and connection management.

    Example:
        Basic text-to-video generation workflow::

            async with VeniceClient() as client:
                # Every call in the lifecycle must name the *same* model —
                # quote, submit, retrieve and cancel are all model-scoped.
                # Resolve it once from the live catalog rather than hardcoding.
                model = await client.models.resolve_video()

                # Get a price quote (pricing is driven by model + duration +
                # resolution; /video/quote does not accept a prompt).
                quote = await client.video.quote(
                    model=model,
                    duration="5s",
                )
                print(f"Estimated cost: ${quote.quote}")

                # Queue the generation
                result = await client.video.submit(
                    model=model,
                    prompt="A sunset over the ocean with gentle waves",
                    duration="5s",
                    aspect_ratio="16:9",
                )
                print(f"Queue ID: {result.queue_id}")

                # Poll for completion
                status = await client.video.retrieve(
                    model=model,
                    queue_id=result.queue_id,
                )

                # Clean up after download
                await client.video.cancel(
                    model=model,
                    queue_id=result.queue_id,
                )
    """

    async def submit(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | str,
        negative_prompt: str | None = None,
        resolution: str | None = None,
        audio: bool | None = None,
        aspect_ratio: str | None = None,
        image_url: str | None = None,
        upscale_factor: Literal[1, 2, 4] | None = None,
        end_image_url: str | None = None,
        audio_url: str | None = None,
        video_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        reference_audio_urls: list[str] | None = None,
        reference_video_urls: list[str] | None = None,
        elements: list[VideoElement | dict] | None = None,
        scene_image_urls: list[str] | None = None,
        consents: VideoConsents | dict | None = None,
    ) -> VideoQueueResponse:
        """
        Queue a new video generation request.

        Automatically selects the appropriate request type based on whether
        ``image_url`` is provided:
        - With ``image_url``: Uses image-to-video (I2V) request
        - Without ``image_url``: Uses text-to-video (T2V) request

        Call :meth:`quote` first to get a price estimate, then poll
        :meth:`retrieve` with the returned ``queue_id`` to get the result.

        :param model: Video model ID (e.g., ``"wan-2.6-text-to-video"``).
        :type model: str
        :param prompt: Text prompt for video generation (max length varies by
            model; default 2500 chars, up to 20000 for some models).
        :type prompt: str
        :param duration_seconds: Duration of generated video as an integer
            number of seconds (e.g. ``5``, ``10``). Liberal string parsing
            also accepts ``"5"`` / ``"5s"`` / ``"5 seconds"``. The wire
            format ``"5s"`` is generated internally. Valid values vary by
            model.
        :type duration_seconds: int | str
        :param negative_prompt: Negative prompt to avoid unwanted content.
        :type negative_prompt: Optional[str]
        :param resolution: Output resolution (e.g., ``"720p"``, ``"1080p"``).
            Valid values vary by model.
        :type resolution: Optional[str]
        :param audio: Generate audio if model supports it.
        :type audio: Optional[bool]
        :param aspect_ratio: Aspect ratio (e.g., ``"16:9"``, ``"9:16"``).
            Typically required for T2V, ignored for I2V.
        :type aspect_ratio: Optional[str]
        :param image_url: Reference image URL for image-to-video generation.
            Must start with ``http://``, ``https://``, or ``data:``.
            When provided, switches to I2V request type.
        :type image_url: Optional[str]
        :param upscale_factor: Upscale models only. ``1`` = quality enhancement,
            ``2`` = double resolution (default for topaz-video-upscale),
            ``4`` = quadruple.
        :type upscale_factor: Optional[Literal[1, 2, 4]]
        :param end_image_url: End-frame image for models that support transitions.
        :type end_image_url: Optional[str]
        :param audio_url: Background audio input (WAV/MP3, max 30s/15MB) for
            models that support it.
        :type audio_url: Optional[str]
        :param video_url: Source video for video-to-video / upscale models
            (MP4/MOV/WebM).
        :type video_url: Optional[str]
        :param reference_image_urls: Up to 9 reference images for character or
            style consistency.
        :type reference_image_urls: Optional[list[str]]
        :param reference_audio_urls: Up to 3 reference audio donors for R2V
            models (e.g. Seedance 2.0 R2V).
        :type reference_audio_urls: Optional[list[str]]
        :param reference_video_urls: Up to 3 reference video donors for R2V
            models (e.g. Seedance 2.0 R2V) used to inherit subject motion,
            camera movement, and overall style.
        :type reference_video_urls: Optional[list[str]]
        :param elements: Up to 4 structured character/object elements for
            advanced element-aware models (Kling O3 R2V). Each dict should
            include ``frontal_image_url`` and optional ``reference_image_urls``.
            Reference in the prompt as ``@Element1``, ``@Element2``, etc.
        :type elements: Optional[list[VideoElement | dict]]
        :param scene_image_urls: Up to 4 scene reference images for
            element-aware models. Reference as ``@Image1``, ``@Image2``, etc.
        :type scene_image_urls: Optional[list[str]]

        :return: Queue response containing ``model`` and ``queue_id``.
        :rtype: VideoQueueResponse

        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises pydantic.ValidationError: If request parameters are invalid.

        Example:
            Text-to-video::

                result = await client.video.submit(
                    model="wan-2.6-text-to-video",
                    prompt="A cat playing piano",
                    duration_seconds=5,
                    aspect_ratio="16:9",
                    resolution="1080p",
                )

            Image-to-video::

                result = await client.video.submit(
                    model="wan-2.6-image-to-video",
                    prompt="Make this photo come to life",
                    duration_seconds=5,
                    image_url="https://example.com/photo.jpg",
                )
        """
        await _preflight_validate_video_duration(self._client, model, duration_seconds)
        # Build request params, only including non-None values
        request_params: dict = {
            "model": model,
            "prompt": prompt,
            "duration": _format_video_duration(duration_seconds),
        }
        for key, val in {
            "negative_prompt": negative_prompt,
            "resolution": resolution,
            "audio": audio,
            "aspect_ratio": aspect_ratio,
            "upscale_factor": upscale_factor,
            "end_image_url": end_image_url,
            "audio_url": audio_url,
            "video_url": video_url,
            "reference_image_urls": reference_image_urls,
            "reference_audio_urls": reference_audio_urls,
            "reference_video_urls": reference_video_urls,
            "elements": elements,
            "scene_image_urls": scene_image_urls,
            "consents": consents,
        }.items():
            if val is not None:
                request_params[key] = val

        # Select request type based on whether image_url is provided
        request: VideoTextToVideoRequest | VideoImageToVideoRequest
        if image_url is not None:
            request_params["image_url"] = image_url
            request = VideoImageToVideoRequest.model_validate(request_params)
        else:
            request = VideoTextToVideoRequest.model_validate(request_params)

        body = request.model_dump(exclude_none=True)

        return await self._client.post(
            "video/queue",
            json_data=body,
            cast_to=VideoQueueResponse,
        )

    async def quote(
        self,
        *,
        model: str,
        duration_seconds: int | str,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        upscale_factor: Literal[1, 2, 4] | None = None,
        audio: bool | None = None,
        video_url: str | None = None,
        reference_video_total_duration: float | None = None,
    ) -> VideoQuoteResponse:
        """
        Get a price estimate for a video generation request.

        Returns the estimated cost in USD. The ``/video/quote`` endpoint
        prices based on model + duration + resolution + upscale; prompt
        text and reference images do not affect the quote and are not
        sent (see the Venice API spec).

        :param model: Video model ID (e.g., ``"wan-2-7-text-to-video"``).
        :param duration_seconds: Duration as an integer number of seconds
            (e.g. ``5``, ``10``). Liberal string parsing also accepts
            ``"5"`` / ``"5s"`` / ``"5 seconds"``. The wire form ``"5s"``
            is generated internally.
        :param aspect_ratio: Aspect ratio (e.g., ``"16:9"``, ``"9:16"``).
        :param resolution: Output resolution (e.g., ``"720p"``, ``"1080p"``).
        :param upscale_factor: For upscale models: ``1`` = quality enhancement,
            ``2`` = double resolution, ``4`` = quadruple.
        :param audio: Generate audio if the model supports it.
        :param video_url: Source video for video-to-video / upscale quotes
            (MP4/MOV/WebM — HTTP URL or ``data:`` URI).
        :param reference_video_total_duration: For R2V models (e.g. Seedance
            2.0 R2V), the aggregate duration in seconds of all reference videos
            to include in the quote. When provided, the quote reflects the
            'input with video' rate tier; when omitted, the no-reference
            baseline is returned.

        :return: Quote response containing estimated cost.
        :rtype: VideoQuoteResponse

        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises pydantic.ValidationError: If request parameters are invalid.

        Example::

            quote = await client.video.quote(
                model="wan-2-7-text-to-video",
                duration_seconds=5,
                aspect_ratio="16:9",
                resolution="720p",
            )
            print(f"Estimated cost: ${quote.quote}")
        """
        await _preflight_validate_video_duration(self._client, model, duration_seconds)
        request = VideoQuoteRequest(
            model=model,
            duration=_format_video_duration(duration_seconds),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            upscale_factor=upscale_factor,
            audio=audio,
            video_url=video_url,
            reference_video_total_duration=reference_video_total_duration,
        )
        body = request.model_dump(exclude_none=True)

        return await self._client.post(
            "video/quote",
            json_data=body,
            cast_to=VideoQuoteResponse,
        )

    async def retrieve(
        self,
        *,
        model: str,
        queue_id: str,
        delete_media_on_completion: bool = False,
    ) -> VideoRetrieveResponse:
        """
        Retrieve the result of a video generation request.

        Poll this endpoint with the ``queue_id`` from :meth:`submit` until the
        video is ready. Returns one of three status types:

        - :class:`~venice_ai.types.api.video.VideoProcessingStatus` —
          still processing, poll again
        - :class:`~venice_ai.types.api.video.VideoFailedStatus` —
          generation failed
        - :class:`~venice_ai.types.api.video.VideoCompletedStatus` —
          complete, download from ``url``

        :param model: Model ID used for generation.
        :type model: str
        :param queue_id: Queue ID from the queue response.
        :type queue_id: str
        :param delete_media_on_completion: Auto-delete media after successful
            retrieval. If ``True``, you don't need to call :meth:`cancel`.
        :type delete_media_on_completion: bool

        :return: Video retrieval response (union of status types).
        :rtype: VideoRetrieveResponse

        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises pydantic.ValidationError: If request parameters are invalid.

        Example::

            import asyncio

            # Poll until complete
            while True:
                status = await client.video.retrieve(
                    model="wan-2.6-text-to-video",
                    queue_id="queue_abc123",
                )
                if hasattr(status, 'url'):
                    print(f"Video ready: {status.url}")
                    break
                elif hasattr(status, 'error'):
                    print(f"Failed: {status.error}")
                    break
                else:
                    print(f"Processing: {status.progress_percent:.0f}%")
                    await asyncio.sleep(5)
        """
        request = VideoRetrieveRequest.model_validate(
            {
                "model": model,
                "queue_id": queue_id,
                "delete_media_on_completion": delete_media_on_completion,
            }
        )
        body = request.model_dump(exclude_none=True)

        # VideoRetrieveResponse is a Union type — the API returns JSON with a
        # `status` field that Pydantic's discriminated union can parse.
        # We need to handle this manually since cast_to expects a single type.
        # Use raw_response to inspect content-type before JSON parsing.
        raw_response = await self._client.post(
            "video/retrieve",
            json_data=body,
            raw_response=True,
        )

        content_type = raw_response.headers.get("content-type", "")
        logger.debug(
            "video.retrieve raw response: status=%s, content_type=%r, content_length=%s",
            raw_response.status,
            content_type,
            raw_response.content_length,
        )

        # If the response is not JSON, it may be binary video data (COMPLETED)
        if "application/json" not in content_type:
            logger.info(
                "video.retrieve returned non-JSON content-type %r — "
                "reading COMPLETED binary video (%s bytes declared).",
                content_type,
                raw_response.content_length,
            )
            # The API streams the completed video as binary data.
            # Read the body so callers can save it to disk.
            video_bytes = await raw_response.read()
            raw_response.close()

            result = VideoCompletedStatus.model_validate(
                {
                    "status": "COMPLETED",
                    "url": None,
                }
            )
            # Attach the binary payload via the private attribute so
            # callers can access it as ``result.data``.
            result._set_data(video_bytes)
            return result

        # JSON response path
        try:
            response_data = await raw_response.json()
        except Exception as e:
            try:
                body_preview = await raw_response.text()
            except Exception:
                body_preview = "<unable to read body>"
            logger.error(
                "Failed to parse JSON from video.retrieve response: %s. "
                "Content-Type: %r, Body preview: %.500s",
                e,
                content_type,
                body_preview,
            )
            raise

        # Parse the response into the appropriate status type
        if isinstance(response_data, dict):
            status = response_data.get("status")
            if status == "PROCESSING":
                return VideoProcessingStatus.model_validate(response_data)
            elif status == "FAILED":
                return VideoFailedStatus.model_validate(response_data)
            elif status == "COMPLETED":
                return VideoCompletedStatus.model_validate(response_data)

        # Fallback: try each status type
        for status_type in (
            VideoProcessingStatus,
            VideoFailedStatus,
            VideoCompletedStatus,
        ):
            try:
                return status_type.model_validate(response_data)
            except Exception as e:
                logger.debug("fallback validate against %s failed: %s", status_type.__name__, e)
                continue

        # If nothing worked, raise
        raise ValueError(f"Unable to parse video retrieve response: {response_data}")

    async def cancel(
        self,
        *,
        model: str,
        queue_id: str,
    ) -> VideoCompleteResponse:
        """
        Release server-side storage for a video job (cancel / cleanup).

        Wraps the ``/video/complete`` endpoint, which deletes the queue
        entry server-side regardless of whether generation has finished.
        Call this after successfully downloading the video, or to abort an
        in-progress job. Not needed if ``delete_media_on_completion`` was
        set to ``True`` in the :meth:`retrieve` request.

        :param model: Model ID used for generation.
        :type model: str
        :param queue_id: Queue ID to release.
        :type queue_id: str

        :return: Complete response indicating success.
        :rtype: VideoCompleteResponse

        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises pydantic.ValidationError: If request parameters are invalid.

        Example::

            result = await client.video.cancel(
                model="wan-2.6-text-to-video",
                queue_id="queue_abc123",
            )
            print(f"Cleanup successful: {result.success}")
        """
        request = VideoCompleteRequest.model_validate(
            {
                "model": model,
                "queue_id": queue_id,
            }
        )
        body = request.model_dump(exclude_none=True)

        return await self._client.post(
            "video/complete",
            json_data=body,
            cast_to=VideoCompleteResponse,
        )

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        duration_seconds: int | str,
        negative_prompt: str | None = None,
        resolution: str | None = None,
        audio: bool | None = None,
        aspect_ratio: str | None = None,
        image_url: str | None = None,
        upscale_factor: Literal[1, 2, 4] | None = None,
        end_image_url: str | None = None,
        audio_url: str | None = None,
        video_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        reference_audio_urls: list[str] | None = None,
        reference_video_urls: list[str] | None = None,
        elements: list[VideoElement | dict] | None = None,
        scene_image_urls: list[str] | None = None,
        consents: VideoConsents | dict | None = None,
    ) -> VideoJob:
        """Queue a video generation and return a :class:`VideoJob` for lifecycle management.

        Accepts the same parameters as :meth:`submit`. The returned job should be
        used as an async context manager to guarantee server-side cleanup::

            async with await client.video.run(
                model=model, prompt="...", duration_seconds=5
            ) as job:
                status = await job.wait()
                await job.download("output.mp4", status)

        :return: A :class:`VideoJob` handle.
        """
        queue_response = await self.submit(
            model=model,
            prompt=prompt,
            duration_seconds=duration_seconds,
            negative_prompt=negative_prompt,
            resolution=resolution,
            audio=audio,
            aspect_ratio=aspect_ratio,
            image_url=image_url,
            upscale_factor=upscale_factor,
            end_image_url=end_image_url,
            audio_url=audio_url,
            video_url=video_url,
            reference_image_urls=reference_image_urls,
            reference_audio_urls=reference_audio_urls,
            reference_video_urls=reference_video_urls,
            elements=elements,
            scene_image_urls=scene_image_urls,
            consents=consents,
        )
        return VideoJob(client=self._client, queue_response=queue_response)

    @overload
    async def transcribe(
        self,
        url: str,
        *,
        response_format: Literal["json"] = "json",
    ) -> VideoTranscriptionResponse: ...

    @overload
    async def transcribe(
        self,
        url: str,
        *,
        response_format: Literal["text"],
    ) -> str: ...

    async def transcribe(
        self,
        url: str,
        *,
        response_format: Literal["json", "text"] = "json",
    ) -> VideoTranscriptionResponse | str:
        """Transcribe a public video URL to text.

        Wraps ``POST /api/v1/video/transcriptions``. Priced at a flat $0.02
        per request.

        :param url: Publicly accessible video URL (e.g. a YouTube watch URL).
        :type url: str
        :param response_format: ``"json"`` (default) returns a
            :class:`VideoTranscriptionResponse` with ``transcript`` and
            ``lang``; ``"text"`` returns the raw transcript as ``str``.
        :type response_format: Literal["json", "text"]

        :return: Either the parsed response model or the plain transcript
            string, depending on ``response_format``.
        :rtype: VideoTranscriptionResponse | str

        :raises venice_ai.exceptions.APIError: If the API request fails.
        :raises pydantic.ValidationError: If ``url`` is not an http(s) URL.

        Example::

            result = await client.video.transcribe(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            )
            print(result.lang, result.transcript)
        """
        request = VideoTranscriptionRequest.model_validate(
            {"url": url, "response_format": response_format}
        )
        body = request.model_dump(exclude_none=True)

        if response_format == "text":
            raw_response = await self._client.post(
                "video/transcriptions",
                json_data=body,
                raw_response=True,
            )
            try:
                text: str = await raw_response.text()
                return text
            finally:
                raw_response.close()

        return await self._client.post(
            "video/transcriptions",
            json_data=body,
            cast_to=VideoTranscriptionResponse,
        )
