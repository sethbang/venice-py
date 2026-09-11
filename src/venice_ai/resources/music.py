"""
Venice AI Music API resources.

This module provides classes for interacting with the Venice AI Music
generation API. Music generation uses the same async queue family as video
(``submit`` / ``retrieve`` / ``cancel``), wired against the
``/audio/queue|quote|retrieve|complete`` endpoints. The high-level
:class:`MusicJob` context manager handles the lifecycle for you.

Pre-v2.0.0 these methods lived on ``client.audio`` alongside TTS / ASR;
they're now their own resource so the namespace mirrors the rest of the
SDK (one resource = one content domain).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .._resource import APIResource
from ..exceptions import MusicGenerationError
from ..helpers import normalize_duration_seconds
from ..types.api.models import MusicModelSpec
from ..types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
    MusicQueueResponse,
    MusicQuoteResponse,
    MusicRetrieveResponse,
)
from ..types.api.requests.music import (
    MusicCompleteRequest,
    MusicQueueRequest,
    MusicQuoteRequest,
    MusicRetrieveRequest,
)
from ..validation.validators import validate_model_id

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)


async def _preflight_validate_music_duration(
    client: VeniceClient,
    model_id: str,
    duration_seconds: int | str | None,
) -> None:
    """Pre-flight check ``duration_seconds`` against the model's spec.

    Best-effort: if the catalog can't be reached or the model isn't a
    :class:`MusicModelSpec`, we silently fall through and let the server
    enforce. We only ever raise for the case we *can* prove wrong: a
    ``duration_options`` enum where the requested value isn't a member,
    or a ``min_duration`` / ``max_duration`` range violation.
    """
    if duration_seconds is None:
        return
    try:
        numeric = normalize_duration_seconds(duration_seconds)
    except ValueError:
        return  # field validator will raise the right error
    try:
        entry = await client.models.get(model_id)
    except Exception:  # noqa: BLE001  - network/catalog miss => let server validate
        return
    spec = entry.model_spec
    if not isinstance(spec, MusicModelSpec):
        return
    if spec.duration_options:
        if numeric not in spec.duration_options:
            raise ValueError(
                f"duration_seconds={numeric} is not a supported value for model "
                f"{model_id!r}; allowed: {spec.duration_options}"
            )
        return
    if spec.min_duration is not None and numeric < spec.min_duration:
        raise ValueError(
            f"duration_seconds={numeric} is below the minimum {spec.min_duration} "
            f"for model {model_id!r}"
        )
    if spec.max_duration is not None and numeric > spec.max_duration:
        raise ValueError(
            f"duration_seconds={numeric} is above the maximum {spec.max_duration} "
            f"for model {model_id!r}"
        )


async def _preflight_validate_force_instrumental(
    client: VeniceClient,
    model_id: str,
    force_instrumental: bool | None,
) -> None:
    """Pre-flight check ``force_instrumental`` against the model's spec.

    Mirrors :func:`_preflight_validate_music_duration`: best-effort, only
    raises when we can prove the request will fail. The API rejects the
    *presence* of the field (not just ``True``) on models that don't support
    it, so any non-None value triggers the check.

    ``spec.supports_force_instrumental=None`` (capability undeclared) defers
    to the server — that's different from a declared ``False``.
    """
    if force_instrumental is None:
        return
    try:
        entry = await client.models.get(model_id)
    except Exception:  # noqa: BLE001 - catalog miss => let server validate
        return
    spec = entry.model_spec
    if not isinstance(spec, MusicModelSpec):
        return
    if spec.supports_force_instrumental is False:
        raise ValueError(
            f"force_instrumental is not supported by model {model_id!r}; "
            f"omit the parameter or pick a vocal-capable music model "
            f"(check ``client.models.get(model_id).model_spec.supports_force_instrumental``)."
        )


async def _preflight_validate_loop(
    client: VeniceClient,
    model_id: str,
    loop: bool | None,
) -> None:
    """Pre-flight check ``loop`` against the model's spec.

    Mirrors :func:`_preflight_validate_force_instrumental`: best-effort, only
    raises when we can prove the request will fail. The API rejects the
    *presence* of the field on models that don't support it, so any non-None
    value triggers the check.

    ``spec.supports_loop=None`` (capability undeclared) defers to the server —
    that's different from a declared ``False``.
    """
    if loop is None:
        return
    try:
        entry = await client.models.get(model_id)
    except Exception:  # noqa: BLE001 - catalog miss => let server validate
        return
    spec = entry.model_spec
    if not isinstance(spec, MusicModelSpec):
        return
    if spec.supports_loop is False:
        raise ValueError(
            f"loop is not supported by model {model_id!r}; "
            f"omit the parameter or pick a loop-capable music model "
            f"(check ``client.models.get(model_id).model_spec.supports_loop``)."
        )


class MusicJob:
    """Manages the lifecycle of an async music generation request.

    Use as an async context manager to guarantee server-side cleanup::

        async with VeniceClient() as client:
            model = await client.models.resolve_music()
            async with await client.music.run(
                model=model,
                prompt="Uplifting cinematic orchestral opener, 30 seconds",
                duration_seconds=30,
            ) as job:
                status = await job.wait()
                await job.download("opener.mp3", status)
    """

    def __init__(self, client: VeniceClient, queue_response: MusicQueueResponse):
        self.model: str = queue_response.model
        self.queue_id: str = queue_response.queue_id
        self._client = client
        self._status: MusicRetrieveResponse | None = None

    async def __aenter__(self) -> MusicJob:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Guarantee server-side cleanup on exit. Mirrors ``VideoJob``."""
        try:
            await self.cancel()
        except Exception as e:
            if exc_type is None:
                logger.warning("MusicJob cleanup failed for queue_id=%s: %s", self.queue_id, e)
            else:
                logger.warning(
                    "MusicJob cleanup failed during exception handling "
                    "(queue_id=%s, original=%s): %s",
                    self.queue_id,
                    exc_type.__name__,
                    e,
                )

    @property
    def status(self) -> MusicRetrieveResponse | None:
        """Last known status from polling.

        Returns:
            The most recent :class:`MusicRetrieveResponse` returned by
            :meth:`poll`, or ``None`` if the job has not yet been polled.
        """
        return self._status

    @property
    def is_complete(self) -> bool:
        """Whether the most recent poll returned a completed status.

        Returns:
            ``True`` if :attr:`status` is a :class:`MusicCompletedStatus`,
            otherwise ``False`` (also ``False`` before the first poll).
        """
        return isinstance(self._status, MusicCompletedStatus)

    @property
    def is_failed(self) -> bool:
        """Whether the most recent poll returned a failed status.

        Returns:
            ``True`` if :attr:`status` is a :class:`MusicFailedStatus`,
            otherwise ``False`` (also ``False`` before the first poll).
        """
        return isinstance(self._status, MusicFailedStatus)

    @property
    def progress(self) -> float | None:
        """Progress as a 0.0-1.0 fraction while processing, else ``None``.

        Returns:
            ``status.progress_percent / 100`` when the last poll returned a
            :class:`MusicProcessingStatus`. Returns ``None`` for terminal
            states (completed / failed) and before the first poll.
        """
        if isinstance(self._status, MusicProcessingStatus):
            return self._status.progress_percent / 100.0
        return None

    async def poll(self) -> MusicRetrieveResponse:
        """Single poll - retrieve current status.

        Wraps ``POST /api/v1/audio/retrieve`` for this job's ``queue_id``.

        Returns:
            The current :class:`MusicRetrieveResponse` (one of
            :class:`MusicProcessingStatus`, :class:`MusicFailedStatus`, or
            :class:`MusicCompletedStatus`). Also caches the result on
            :attr:`status`.

        Raises:
            APIError: For HTTP-level failures retrieving the queue entry
                (mapped subclasses include ``AuthenticationError``,
                ``RateLimitError``, ``NotFoundError``).
        """
        self._status = await self._client.music.retrieve(model=self.model, queue_id=self.queue_id)
        return self._status

    async def wait(
        self,
        *,
        poll_interval: float = 5.0,
        max_polls: int = 120,
        on_progress: Callable[[MusicProcessingStatus], None] | None = None,
    ) -> MusicCompletedStatus:
        """Poll until complete or failed.

        Drives :meth:`poll` on a fixed interval, returning the final
        :class:`MusicCompletedStatus` once the server reports completion.

        Args:
            poll_interval: Seconds to sleep between successive polls.
                Defaults to ``5.0``.
            max_polls: Maximum number of polls before giving up. Defaults
                to ``120`` (i.e. ten minutes at the default interval).
            on_progress: Optional callback invoked after every poll that
                returns a :class:`MusicProcessingStatus` - useful for
                forwarding progress to logs or a UI.

        Returns:
            The terminal :class:`MusicCompletedStatus`.

        Raises:
            MusicGenerationError: If the server reports generation failure.
            TimeoutError: If ``max_polls`` is exhausted before completion.
            APIError: For HTTP-level failures while polling.
        """
        for _ in range(max_polls):
            status = await self.poll()
            if isinstance(status, MusicCompletedStatus):
                return status
            if isinstance(status, MusicFailedStatus):
                raise MusicGenerationError(
                    f"Music generation failed: {status.error}",
                    error_code=status.error_code,
                )
            if on_progress and isinstance(status, MusicProcessingStatus):
                on_progress(status)
            await asyncio.sleep(poll_interval)
        raise TimeoutError(f"Music generation did not complete within {max_polls} polls")

    async def download(self, path: str | Path, status: MusicCompletedStatus) -> Path:
        """Download a completed music clip to *path*.

        Does NOT call :meth:`cancel` - use the context manager for that.
        File I/O is offloaded to a worker thread so the event loop never
        blocks. URL downloads reuse the SDK client's managed HTTP session
        so proxy, SSL, timeout, and retry configuration are honored.

        Args:
            path: Target file path. Parent directories are created
                automatically (``mkdir -p``).
            status: A terminal :class:`MusicCompletedStatus` returned from
                :meth:`wait` or :meth:`poll`. The bytes are sourced from
                ``status.data`` when present, or fetched from ``status.url``.

        Returns:
            The resolved :class:`pathlib.Path` the audio was written to.

        Raises:
            APIError: If the URL fetch fails (mapped subclasses include
                ``APIConnectionError``, ``APITimeoutError``).
            OSError: If the file cannot be written (permission denied,
                disk full, etc.).
        """
        path = Path(path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        if status.data:
            await asyncio.to_thread(path.write_bytes, status.data)
        elif status.url:
            data = await self._client.fetch_external(status.url)
            await asyncio.to_thread(path.write_bytes, data)
        return path

    async def cancel(self) -> MusicCompleteResponse:
        """Release server-side storage / cancel an in-progress job.

        Wraps ``POST /api/v1/audio/complete``, which deletes the queue
        entry server-side regardless of whether generation has finished.
        Named ``cancel`` (rather than the wire-format ``complete``) to
        distinguish it from the :attr:`is_complete` state check - terminal
        states are polled via :meth:`wait` / :attr:`status`.

        Returns:
            :class:`MusicCompleteResponse` confirming the queue entry was
            released.

        Raises:
            APIError: For HTTP-level failures (mapped subclasses include
                ``AuthenticationError``, ``NotFoundError`` if the queue id
                no longer exists).
        """
        return await self._client.music.cancel(model=self.model, queue_id=self.queue_id)


class Music(APIResource["VeniceClient"]):
    """Asynchronous interface for Venice AI's Music generation API.

    Mirrors :class:`venice_ai.resources.video.Video`: ``submit`` queues a
    job, ``run`` returns a managed :class:`MusicJob`, ``retrieve`` polls,
    ``cancel`` releases server-side storage, ``quote`` gives a price
    estimate.

    Accessed through :attr:`venice_ai.VeniceClient.music`.
    """

    async def submit(
        self,
        *,
        model: str,
        prompt: str,
        lyrics_prompt: str | None = None,
        duration_seconds: int | str | None = None,
        force_instrumental: bool | None = None,
        lyrics_optimizer: bool | None = None,
        voice: str | None = None,
        language_code: str | None = None,
        speed: float | None = None,
        loop: bool | None = None,
    ) -> MusicQueueResponse:
        """Queue a music generation request.

        Wraps ``POST /api/v1/audio/queue``. Call :meth:`quote` first for a
        price estimate, then poll :meth:`retrieve` with the returned
        ``queue_id`` to fetch the audio. Or use :meth:`run` to get a
        managed :class:`MusicJob` that handles the lifecycle.

        Parameter support varies by model - inspect ``/models?type=music``
        for per-model capability fields (``supports_lyrics``,
        ``supports_speed``, etc.).

        Args:
            model: Music model id (e.g. resolved via
                ``client.models.resolve_music()``).
            prompt: Natural-language description of the desired track.
            lyrics_prompt: Optional separate prompt to drive the lyrics on
                vocal-capable models.
            duration_seconds: Target clip duration. Accepts an int or a
                stringified int; server caps vary by model.
            force_instrumental: If ``True``, suppresses vocals even when
                the model would otherwise sing. Defaults to ``None``
                (model default applies).
            lyrics_optimizer: If ``True``, asks the model to refine
                lyrics for prosody/syllable count. Defaults to ``None``.
            voice: Named voice preset for vocal-capable models.
            language_code: BCP-47 language hint for the lyrics
                (e.g. ``"en"``, ``"ja"``).
            speed: Playback-speed multiplier where supported.
            loop: If ``True``, renders the clip so its end splices back into
                its start without an audible seam. Only supported when
                ``/models`` reports ``supports_loop=true``. Defaults to
                ``None`` (model default applies).

        Returns:
            :class:`MusicQueueResponse` containing the ``queue_id`` to poll
            with :meth:`retrieve` and the echoed ``model`` id.

        Raises:
            InvalidRequestError: If the model id fails validation, the
                prompt is empty, or any parameter is rejected server-side.
            AuthenticationError: If the API key is missing or invalid.
            RateLimitError: If the music queue is saturated for the
                account.
            APIError: For other HTTP-level failures.
        """
        validate_model_id(model, "model")
        await _preflight_validate_music_duration(self._client, model, duration_seconds)
        await _preflight_validate_force_instrumental(self._client, model, force_instrumental)
        await _preflight_validate_loop(self._client, model, loop)
        request = MusicQueueRequest.model_validate(
            {
                "model": model,
                "prompt": prompt,
                "lyrics_prompt": lyrics_prompt,
                "duration_seconds": duration_seconds,
                "force_instrumental": force_instrumental,
                "lyrics_optimizer": lyrics_optimizer,
                "voice": voice,
                "language_code": language_code,
                "speed": speed,
                "loop": loop,
            }
        )
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "audio/queue",
            json_data=body,
            cast_to=MusicQueueResponse,
        )

    async def quote(
        self,
        *,
        model: str,
        duration_seconds: int | str | None = None,
        character_count: int | None = None,
    ) -> MusicQuoteResponse:
        """Get a price quote for a music generation request.

        Wraps ``POST /api/v1/audio/quote``. Use this before :meth:`submit`
        to surface cost in your UI without committing to a generation.

        Args:
            model: Music model id whose pricing to look up.
            duration_seconds: Target clip duration to price. Accepts an
                int or stringified int.
            character_count: Optional character count for lyrics-priced
                models that bill per syllable / character.

        Returns:
            :class:`MusicQuoteResponse` with the estimated cost breakdown
            for the request.

        Raises:
            InvalidRequestError: If the model id fails validation or the
                request is rejected server-side.
            AuthenticationError: If the API key is missing or invalid.
            APIError: For other HTTP-level failures.
        """
        validate_model_id(model, "model")
        await _preflight_validate_music_duration(self._client, model, duration_seconds)
        request = MusicQuoteRequest.model_validate(
            {
                "model": model,
                "duration_seconds": duration_seconds,
                "character_count": character_count,
            }
        )
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "audio/quote",
            json_data=body,
            cast_to=MusicQuoteResponse,
        )

    async def retrieve(
        self,
        *,
        model: str,
        queue_id: str,
        delete_media_on_completion: bool = False,
    ) -> MusicRetrieveResponse:
        """Retrieve the result of a music generation request.

        Poll ``POST /api/v1/audio/retrieve`` until the audio is ready.
        When the API streams the audio inline as binary, the bytes are
        attached to the completed status's private ``data`` buffer.

        Args:
            model: Music model id used at submit time.
            queue_id: Queue identifier returned by :meth:`submit`.
            delete_media_on_completion: If ``True``, the server releases
                the cached audio after this retrieval (a one-shot fetch).
                Defaults to ``False`` so the URL remains downloadable on
                subsequent polls.

        Returns:
            One of :class:`MusicProcessingStatus`,
            :class:`MusicFailedStatus`, or :class:`MusicCompletedStatus`,
            keyed off the response's ``status`` field. Inline binary
            payloads are attached to ``MusicCompletedStatus._data``.

        Raises:
            InvalidRequestError: If the model id or queue id fails
                validation server-side.
            AuthenticationError: If the API key is missing or invalid.
            NotFoundError: If the queue id does not exist or has been
                released.
            ValueError: If the response cannot be parsed into any of the
                three status shapes.
            APIError: For other HTTP-level failures.
        """
        validate_model_id(model, "model")
        request = MusicRetrieveRequest.model_validate(
            {
                "model": model,
                "queue_id": queue_id,
                "delete_media_on_completion": delete_media_on_completion,
            }
        )
        body = request.model_dump(exclude_none=True)

        raw_response = await self._client.post(
            "audio/retrieve",
            json_data=body,
            raw_response=True,
        )

        content_type = raw_response.headers.get("content-type", "")
        logger.debug(
            "music.retrieve raw response: status=%s, content_type=%r, content_length=%s",
            raw_response.status,
            content_type,
            raw_response.content_length,
        )

        if "application/json" not in content_type:
            logger.info(
                "music.retrieve returned non-JSON content-type %r — "
                "reading COMPLETED binary audio (%s bytes declared).",
                content_type,
                raw_response.content_length,
            )
            audio_bytes = await raw_response.read()
            raw_response.close()
            completed = MusicCompletedStatus.model_validate({"status": "COMPLETED", "url": None})
            completed._set_data(audio_bytes)
            return completed

        try:
            response_data = await raw_response.json()
        except Exception as e:
            try:
                body_preview = await raw_response.text()
            except Exception:
                body_preview = "<unable to read body>"
            logger.error(
                "Failed to parse JSON from music.retrieve response: %s. "
                "Content-Type: %r, Body preview: %.500s",
                e,
                content_type,
                body_preview,
            )
            raise

        if isinstance(response_data, dict):
            status = response_data.get("status")
            if status == "PROCESSING":
                return MusicProcessingStatus.model_validate(response_data)
            if status == "FAILED":
                return MusicFailedStatus.model_validate(response_data)
            if status == "COMPLETED":
                return MusicCompletedStatus.model_validate(response_data)

        for status_type in (
            MusicProcessingStatus,
            MusicFailedStatus,
            MusicCompletedStatus,
        ):
            try:
                return status_type.model_validate(response_data)
            except Exception as e:
                logger.debug("fallback validate against %s failed: %s", status_type.__name__, e)
                continue

        raise ValueError(f"Unable to parse music retrieve response: {response_data}")

    async def cancel(
        self,
        *,
        model: str,
        queue_id: str,
    ) -> MusicCompleteResponse:
        """Release server-side storage / cancel an in-progress job.

        Wraps ``POST /api/v1/audio/complete``. Named ``cancel`` (rather
        than the wire-format ``complete``) to distinguish it from
        :attr:`MusicJob.is_complete` state checks.

        Args:
            model: Music model id used at submit time.
            queue_id: Queue identifier returned by :meth:`submit`.

        Returns:
            :class:`MusicCompleteResponse` confirming the queue entry was
            released.

        Raises:
            InvalidRequestError: If the model id or queue id fails
                validation server-side.
            AuthenticationError: If the API key is missing or invalid.
            NotFoundError: If the queue id no longer exists.
            APIError: For other HTTP-level failures.
        """
        validate_model_id(model, "model")
        request = MusicCompleteRequest.model_validate({"model": model, "queue_id": queue_id})
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "audio/complete",
            json_data=body,
            cast_to=MusicCompleteResponse,
        )

    async def run(
        self,
        *,
        model: str,
        prompt: str,
        lyrics_prompt: str | None = None,
        duration_seconds: int | str | None = None,
        force_instrumental: bool | None = None,
        lyrics_optimizer: bool | None = None,
        voice: str | None = None,
        language_code: str | None = None,
        speed: float | None = None,
        loop: bool | None = None,
    ) -> MusicJob:
        """Submit a music generation and return a managed :class:`MusicJob`.

        Calls :meth:`submit` then wraps the queue response in a
        :class:`MusicJob` async context manager. On exit, the SDK calls
        :meth:`MusicJob.cancel` to release server-side storage. Accepts the
        same parameters as :meth:`submit`.

        Wraps ``POST /api/v1/audio/queue`` (the lifecycle context manager
        also touches ``/audio/retrieve`` and ``/audio/complete``).

        Args:
            model: Music model id (e.g. resolved via
                ``client.models.resolve_music()``).
            prompt: Natural-language description of the desired track.
            lyrics_prompt: Optional separate prompt to drive the lyrics
                on vocal-capable models.
            duration_seconds: Target clip duration. Accepts an int or a
                stringified int.
            force_instrumental: If ``True``, suppresses vocals.
            lyrics_optimizer: If ``True``, asks the model to refine
                lyrics for prosody/syllable count.
            voice: Named voice preset for vocal-capable models.
            language_code: BCP-47 language hint for the lyrics.
            speed: Playback-speed multiplier where supported.
            loop: If ``True``, renders the clip so its end splices back into
                its start seamlessly. Requires ``supports_loop=true``.

        Returns:
            A :class:`MusicJob` ready to use as an async context manager.

        Raises:
            InvalidRequestError: If the model id fails validation, the
                prompt is empty, or any parameter is rejected server-side.
            AuthenticationError: If the API key is missing or invalid.
            RateLimitError: If the music queue is saturated for the
                account.
            APIError: For other HTTP-level failures.

        Example:

            .. code-block:: python

                from venice_ai import VeniceClient

                async with VeniceClient() as client:
                    model = await client.models.resolve_music()
                    async with await client.music.run(
                        model=model,
                        prompt="Upbeat synthwave track with driving bass",
                        duration_seconds=45,
                    ) as job:
                        status = await job.wait()
                        await job.download("track.mp3", status)
        """
        queue_response = await self.submit(
            model=model,
            prompt=prompt,
            lyrics_prompt=lyrics_prompt,
            duration_seconds=duration_seconds,
            force_instrumental=force_instrumental,
            lyrics_optimizer=lyrics_optimizer,
            voice=voice,
            language_code=language_code,
            speed=speed,
            loop=loop,
        )
        return MusicJob(client=self._client, queue_response=queue_response)


__all__ = ["Music", "MusicJob"]
