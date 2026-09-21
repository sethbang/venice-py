"""
Venice AI Voice Changer API resources.

Voice changing converts an existing recording into a target voice, preserving
the original timing. It uses its own async queue family under
``/audio/voice-changer/*`` (``queue`` / ``quote`` / ``retrieve`` /
``complete``), parallel to music on ``/audio/*`` — which is why this is a
top-level ``client.voice_changer`` rather than a corner of ``client.audio``.

Two things differ from the music and video families and are easy to get wrong
by analogy:

* ``quote(duration_seconds=...)`` takes a **bare count of seconds**, not the
  ``"60s"`` form the video endpoints accept.
* :meth:`VoiceChanger.retrieve` has **two** outcomes, not three. The endpoint
  declares a ``PROCESSING`` JSON body and an ``audio/mpeg`` body; there is no
  ``FAILED`` status and no JSON completed arm with a download URL. Failures
  arrive as HTTP errors, with any refund under
  ``err.body["credits_refunded"]``.

Voice-changer models are not a separate model type — they report
``type="music"`` with ``voice_changer: true``. Use
:meth:`client.models.resolve_voice_changer` rather than filtering by type.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

from .._resource import APIResource
from ..exceptions import InvalidRequestError
from ..types.api.requests.voice_changer import (
    CompleteVoiceChangerRequest,
    QueueVoiceChangerRequest,
    QuoteVoiceChangerRequest,
    RetrieveVoiceChangerRequest,
)
from ..types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerCompleteResponse,
    VoiceChangerProcessingStatus,
    VoiceChangerQueueResponse,
    VoiceChangerQuoteResponse,
    VoiceChangerRetrieveResponse,
)
from ..validation.validators import validate_model_id

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)


class VoiceChangerJob:
    """Manages the lifecycle of an async voice-conversion request.

    Use as an async context manager to guarantee server-side cleanup::

        async with VeniceClient() as client:
            model = await client.models.resolve_voice_changer()
            async with await client.voice_changer.run(
                model=model,
                file="source.mp3",
                voice="Aria",
            ) as job:
                status = await job.wait()
                await job.download("converted.mp3", status)

    Note that under :class:`~venice_ai.SyncVeniceClient` this object's methods
    stay coroutines — the sync proxy's ``_wrap_result`` only special-cases
    ``Stream`` — so they must be awaited. :class:`~venice_ai.MusicJob` and
    :class:`~venice_ai.VideoJob` share that limitation.
    """

    def __init__(self, client: VeniceClient, queue_response: VoiceChangerQueueResponse):
        self.model: str = queue_response.model
        self.queue_id: str = queue_response.queue_id
        #: Source length measured server-side — the exact quantity billed.
        self.duration_seconds: float = queue_response.duration_seconds
        self._client = client
        self._status: VoiceChangerRetrieveResponse | None = None

    async def __aenter__(self) -> VoiceChangerJob:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Guarantee server-side cleanup on exit. Mirrors :class:`MusicJob`."""
        try:
            await self.cancel()
        except InvalidRequestError as e:
            # The media is already gone — either the conversion completed with
            # delete_media_on_completion=True, or cleanup already ran. Not
            # worth a warning on every successful job.
            logger.debug(
                "VoiceChangerJob cleanup found nothing to release (queue_id=%s): %s",
                self.queue_id,
                e,
            )
        except Exception as e:
            if exc_type is None:
                logger.warning(
                    "VoiceChangerJob cleanup failed for queue_id=%s: %s", self.queue_id, e
                )
            else:
                logger.warning(
                    "VoiceChangerJob cleanup failed during exception handling "
                    "(queue_id=%s, original=%s): %s",
                    self.queue_id,
                    exc_type.__name__,
                    e,
                )

    @property
    def status(self) -> VoiceChangerRetrieveResponse | None:
        """Last known status from polling, or ``None`` before the first poll."""
        return self._status

    @property
    def is_complete(self) -> bool:
        """Whether the most recent poll returned the converted audio."""
        return isinstance(self._status, VoiceChangerCompletedStatus)

    @property
    def progress(self) -> float | None:
        """Progress as a 0.0-1.0 fraction while processing, else ``None``.

        Derived from elapsed time against the model's recent average, so it is
        a pacing hint rather than true progress.
        """
        if isinstance(self._status, VoiceChangerProcessingStatus):
            return self._status.progress_percent / 100.0
        return None

    async def poll(self) -> VoiceChangerRetrieveResponse:
        """Single poll — retrieve current status.

        Returns:
            Either a :class:`VoiceChangerProcessingStatus` or a
            :class:`VoiceChangerCompletedStatus` carrying the audio. Also
            caches the result on :attr:`status`.

        Raises:
            APIError: For HTTP-level failures retrieving the queue entry.
        """
        self._status = await self._client.voice_changer.retrieve(
            model=self.model, queue_id=self.queue_id
        )
        return self._status

    async def wait(
        self,
        *,
        poll_interval: float = 3.0,
        max_polls: int = 120,
        on_progress: Callable[[VoiceChangerProcessingStatus], None] | None = None,
    ) -> VoiceChangerCompletedStatus:
        """Poll until the converted audio is ready.

        There is no failure *status* to check for — a failed conversion raises
        an :class:`~venice_ai.exceptions.APIError` from :meth:`poll` instead.

        Args:
            poll_interval: Seconds between polls. Defaults to ``3.0``.
            max_polls: Maximum polls before giving up. Defaults to ``120``.
            on_progress: Optional callback invoked after each poll that returns
                a :class:`VoiceChangerProcessingStatus`.

        Returns:
            The terminal :class:`VoiceChangerCompletedStatus`.

        Raises:
            TimeoutError: If ``max_polls`` is exhausted before completion.
            APIError: For HTTP-level failures while polling, including a
                conversion that failed provider-side.
        """
        for _ in range(max_polls):
            status = await self.poll()
            if isinstance(status, VoiceChangerCompletedStatus):
                return status
            if on_progress:
                on_progress(status)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Voice conversion did not complete within {max_polls} polls")

    async def download(self, path: str | Path, status: VoiceChangerCompletedStatus) -> Path:
        """Write the converted audio to *path*.

        Does not call :meth:`cancel` — use the context manager for that.

        Args:
            path: Target file path; parent directories are created.
            status: A terminal :class:`VoiceChangerCompletedStatus` from
                :meth:`wait` or :meth:`poll`.

        Returns:
            The resolved path the audio was written to.

        Raises:
            ValueError: If the status carries no audio bytes.
            OSError: If the file cannot be written.
        """
        path = Path(path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        if not status.data:
            raise ValueError(
                "Completed status carries no audio bytes. The endpoint returns the "
                "audio inline; there is no URL to fall back to."
            )
        await asyncio.to_thread(path.write_bytes, status.data)
        return path

    async def cancel(self) -> VoiceChangerCompleteResponse:
        """Release the provider-held media for this conversion."""
        return await self._client.voice_changer.cancel(model=self.model, queue_id=self.queue_id)


class VoiceChanger(APIResource["VeniceClient"]):
    """Convert a recording into a target voice (asynchronous).

    Accessed through :attr:`VeniceClient.voice_changer`.

    :param client: The Venice AI client instance used to make API requests.
    """

    async def submit(
        self,
        *,
        model: str,
        file: str | bytes | BinaryIO | Path | None = None,
        audio_url: str | None = None,
        voice: str | None = None,
        remove_background_noise: bool | None = None,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> VoiceChangerQueueResponse:
        """Queue a voice conversion (``POST /api/v1/audio/voice-changer/queue``).

        Supply the source recording either as ``file`` (uploaded as multipart)
        or as ``audio_url`` (fetched by Venice), never both. Venice validates
        the bytes itself and forwards only those bytes to the provider, so an
        ``audio_url`` is never handed onward.

        Args:
            model: Voice-changer model ID. Resolve it with
                :meth:`client.models.resolve_voice_changer`.
            file: Source recording — a path, raw bytes, or binary file object.
            audio_url: Publicly reachable http(s) URL of the source recording.
            voice: Target voice. One of the model's ``voices``, or a provider
                Voice ID when the model reports ``supports_custom_voice_id``.
                Defaults to the model's ``default_voice``.
            remove_background_noise: Strip background noise before converting.
            seed: Seed for reproducible output.
            timeout: Per-call timeout override, in seconds.

        Returns:
            A :class:`VoiceChangerQueueResponse` whose ``duration_seconds`` is
            the server-measured source length — the exact quantity billed.

        Raises:
            ValueError: If neither or both of ``file`` / ``audio_url`` are given.
            APIError: For HTTP-level failures.
        """
        validate_model_id(model, "model")

        # Validate the combination before doing any file I/O.
        request = QueueVoiceChangerRequest(
            model=model,
            file=file,
            audio_url=audio_url,
            voice=voice,
            remove_background_noise=remove_background_noise,
            seed=seed,
        )

        if audio_url is not None:
            body = request.model_dump(exclude_none=True, exclude={"file"})
            return await self._client.post(
                "audio/voice-changer/queue",
                json_data=body,
                cast_to=VoiceChangerQueueResponse,
                timeout=timeout,
            )

        # Multipart path. Reuse the audio resource's input handling so paths,
        # bytes and file-like objects behave identically to transcription.
        from ..utils import serialize_form_value
        from .audio import Audio

        content, filename, content_type = await Audio(self._client)._prepare_audio_file(
            cast(Any, file)
        )
        files = {"file": (filename, content, content_type)}
        form = request.model_dump(exclude_none=True, exclude={"file", "audio_url"})
        data = {k: serialize_form_value(v) for k, v in form.items()}

        raw = await self._request_multipart(
            method="POST",
            path="audio/voice-changer/queue",
            files=files,
            data=data,
            timeout=timeout,
        )
        if not isinstance(raw, dict):
            raise ValueError(
                "Expected a JSON queue response from audio/voice-changer/queue, "
                f"got {type(raw).__name__}"
            )
        # _request_multipart does not run the cast_to path, so no _response is
        # attached here — unlike the audio_url branch above.
        return VoiceChangerQueueResponse.model_validate(raw)

    async def quote(
        self,
        *,
        model: str,
        duration_seconds: int | str,
    ) -> VoiceChangerQuoteResponse:
        """Estimate the price of converting a recording of a given length.

        Args:
            model: Voice-changer model ID.
            duration_seconds: Source length as a **bare count of seconds** —
                an ``int`` or a digits-only string. This is not the ``"60s"``
                form the video endpoints take.

        Returns:
            A :class:`VoiceChangerQuoteResponse`. The price is an estimate;
            the charge is computed from the length Venice measures when the
            recording is queued, reported as ``duration_seconds`` on the
            queue response.

        Raises:
            ValueError: If ``duration_seconds`` is not a positive whole number.
            APIError: For HTTP-level failures.
        """
        validate_model_id(model, "model")
        request = QuoteVoiceChangerRequest(model=model, duration_seconds=duration_seconds)
        return await self._client.post(
            "audio/voice-changer/quote",
            json_data=request.model_dump(exclude_none=True),
            cast_to=VoiceChangerQuoteResponse,
        )

    async def retrieve(
        self,
        *,
        model: str,
        queue_id: str,
        delete_media_on_completion: bool = False,
    ) -> VoiceChangerRetrieveResponse:
        """Poll a conversion (``POST /api/v1/audio/voice-changer/retrieve``).

        The endpoint answers in one of two ways, discriminated by content type
        rather than by a status field:

        * ``application/json`` with ``status: "PROCESSING"`` — still running.
        * ``audio/mpeg`` — the converted audio, returned inline.

        Args:
            model: The model running the conversion.
            queue_id: Queue ID from :meth:`submit`.
            delete_media_on_completion: Release the provider-held media as
                soon as this call returns the audio, making a separate
                :meth:`cancel` unnecessary. The audio cannot be retrieved
                again afterwards.

        Returns:
            A :class:`VoiceChangerProcessingStatus` or a
            :class:`VoiceChangerCompletedStatus` carrying the audio bytes.

        Raises:
            APIError: For HTTP-level failures. A conversion that failed
                provider-side surfaces here (404 media gone, 500 inference
                failed, 504 never reached the provider). Read
                ``err.body.get("credits_refunded")`` for the refund — the
                parsed JSON body is on the exception, and it is not modelled
                as a typed field because no voice-changer model exists to
                provoke the response and confirm its shape.
            ValueError: If the response is neither recognised shape.
        """
        validate_model_id(model, "model")
        request = RetrieveVoiceChangerRequest(
            model=model,
            queue_id=queue_id,
            delete_media_on_completion=delete_media_on_completion,
        )

        raw_response = await self._client.post(
            "audio/voice-changer/retrieve",
            json_data=request.model_dump(exclude_none=True),
            raw_response=True,
        )

        content_type = raw_response.headers.get("content-type", "")
        logger.debug(
            "voice_changer.retrieve raw response: status=%s, content_type=%r, length=%s",
            raw_response.status,
            content_type,
            raw_response.content_length,
        )

        if "application/json" not in content_type:
            logger.info(
                "voice_changer.retrieve returned non-JSON content-type %r — "
                "reading completed audio (%s bytes declared).",
                content_type,
                raw_response.content_length,
            )
            audio_bytes = await raw_response.read()
            raw_response.close()
            result = VoiceChangerCompletedStatus.model_validate({"status": "COMPLETED"})
            result._set_data(audio_bytes)
            return result

        try:
            response_data = await raw_response.json()
        except Exception as e:
            try:
                body_preview = await raw_response.text()
            except Exception:
                body_preview = "<unable to read body>"
            logger.error(
                "Failed to parse JSON from voice_changer.retrieve response: %s. "
                "Content-Type: %r, Body preview: %.500s",
                e,
                content_type,
                body_preview,
            )
            raise

        return VoiceChangerProcessingStatus.model_validate(response_data)

    async def cancel(
        self,
        *,
        model: str,
        queue_id: str,
    ) -> VoiceChangerCompleteResponse:
        """Release the provider-held media (``POST /audio/voice-changer/complete``).

        Args:
            model: The model running the conversion.
            queue_id: Queue ID from :meth:`submit`.

        Returns:
            A :class:`VoiceChangerCompleteResponse` reporting whether the
            media was released.
        """
        validate_model_id(model, "model")
        request = CompleteVoiceChangerRequest(model=model, queue_id=queue_id)
        return await self._client.post(
            "audio/voice-changer/complete",
            json_data=request.model_dump(exclude_none=True),
            cast_to=VoiceChangerCompleteResponse,
        )

    async def run(
        self,
        *,
        model: str,
        file: str | bytes | BinaryIO | Path | None = None,
        audio_url: str | None = None,
        voice: str | None = None,
        remove_background_noise: bool | None = None,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> VoiceChangerJob:
        """Queue a conversion and return a :class:`VoiceChangerJob` for it.

        The high-level entry point. Takes the same arguments as :meth:`submit`
        and wraps the queue response so you can ``await job.wait()`` and
        ``await job.download(...)``, with cleanup handled by the context
        manager.
        """
        queue_response = await self.submit(
            model=model,
            file=file,
            audio_url=audio_url,
            voice=voice,
            remove_background_noise=remove_background_noise,
            seed=seed,
            timeout=timeout,
        )
        return VoiceChangerJob(client=self._client, queue_response=queue_response)


__all__ = ["VoiceChanger", "VoiceChangerJob"]
