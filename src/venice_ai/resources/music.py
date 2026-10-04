"""
Venice AI Music API resources.

This module provides classes for interacting with the Venice AI Music
generation API. Music generation uses the same async queue family as video
(``submit`` / ``retrieve`` / ``release``), wired against the
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
from typing import TYPE_CHECKING, Any

import aiohttp

from .._resource import APIResource
from ..exceptions import (
    InvalidRequestError,
    MusicGenerationError,
    NotFoundError,
    UnprocessableEntityError,
)
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
from ..utils.errors import read_body
from ..validation.validators import validate_model_id
from ._inline_audio import read_inline_audio
from .video import _is_unknown_request_id

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
    enforce. We only ever raise for the cases we *can* prove wrong: a
    ``duration_options`` enum where the requested value isn't a member, a
    ``min_duration`` / ``max_duration`` range violation, or a model that
    declares no duration metadata at all (Venice rejects ``duration_seconds``
    on those with "This model does not support duration_seconds").
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
    if spec.min_duration is None and spec.max_duration is None:
        raise ValueError(
            f"model {model_id!r} does not take duration_seconds: it declares no "
            f"duration_options or min/max duration and chooses the clip length itself"
        )
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


def _log_unsuccessful_release(queue_id: str, model: str) -> None:
    logger.warning(
        "Releasing music job queue_id=%s (model %s) did not complete: Venice "
        "returned success=false. The stored media may remain; retry the release later.",
        queue_id,
        model,
    )


class MusicJob:
    """Manages the lifecycle of an async music generation request.

    Venice bills a music job when it is queued and has no way to stop one.
    Leaving the context without an exception releases the job's stored media
    once it has finished; a job still generating keeps running and is stored
    when it completes. If the block raises, nothing is released, so a failed
    save can be retried (see :meth:`__aexit__`). Use as an async context
    manager::

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
        # Set when /audio/retrieve rejects the job outright: the job is over
        # even though no terminal status was seen.
        self._rejected = False
        # Set once release() succeeds on a finished job, so exiting the context
        # does not release it a second time.
        self._released = False

    async def __aenter__(self) -> MusicJob:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        """Release the job's stored media on a clean exit, once it has finished.

        The media is released only when the block exits without an exception.
        If the block raised (a failed download, a disk error, a cancelled
        task), nothing is released and a WARNING names the ``queue_id``: the
        job was already billed, and its audio stays stored so the work can be
        resumed. Resume with ``client.music.retrieve(model=..., queue_id=...)``
        (or this job's :meth:`wait`), save the audio, then call
        ``client.music.release(model=..., queue_id=...)`` or :meth:`release`.
        Call :meth:`release` inside the block to discard the audio on purpose;
        a finished job already released that way is left alone on exit,
        whichever way the block exits. A job that failed or was rejected has no audio to
        keep, so it is released whichever way the block exits.

        On a clean exit, a job that has not reached a terminal status is
        polled once more. A job that has now finished is released with
        :meth:`release`. A job that is still generating is left alone:
        ``/audio/complete`` only deletes media that already exists, so calling
        it now would release nothing while the finished audio is stored later
        anyway. That case is logged at WARNING with the ``queue_id``; call
        :meth:`wait` again (a :class:`TimeoutError` from it leaves the job
        resumable) and then :meth:`release`. A queue id whose media is already
        gone (``NotFoundError``) or that Venice never issued (a 400 "Request
        ID is invalid") has nothing to release. A release that returns
        ``success: false`` is logged at WARNING.
        """
        if self._released:
            return
        failed = self._rejected or isinstance(self._status, MusicFailedStatus)
        if exc_type is not None and not failed:
            logger.warning(
                "MusicJob queue_id=%s (model %s) left its context with %s, so its stored "
                "audio was not released (last status: %s). The job is billed; to keep the "
                "audio, call client.music.retrieve(model=%r, queue_id=%r) and save it, "
                "then client.music.release(model=%r, queue_id=%r).",
                self.queue_id,
                self.model,
                exc_type.__name__,
                self._status.status if self._status is not None else "never polled",
                self.model,
                self.queue_id,
                self.model,
                self.queue_id,
            )
            return
        if not self._is_terminal:
            try:
                await self.poll()
            except NotFoundError as e:
                logger.debug(
                    "MusicJob queue_id=%s has no stored media left to release: %s",
                    self.queue_id,
                    e,
                )
                return
            except (InvalidRequestError, UnprocessableEntityError) as e:
                if _is_unknown_request_id(e):
                    logger.debug(
                        "MusicJob queue_id=%s is unknown to Venice; nothing to release: %s",
                        self.queue_id,
                        e,
                    )
                    return
                self._rejected = True
            except Exception as e:
                logger.debug("Final poll for music job queue_id=%s failed: %s", self.queue_id, e)
        if not self._is_terminal:
            logger.warning(
                "MusicJob queue_id=%s (model %s) left its context while still generating "
                "(last status: %s). The job is already billed and keeps running; Venice "
                "stores the audio when it finishes. Call wait() again, then release(), "
                "to collect and delete it.",
                self.queue_id,
                self.model,
                self._status.status if self._status is not None else "never polled",
            )
            return
        try:
            # Music.release() logs a ``success: false`` result itself.
            await self.release()
        except Exception as e:
            # A 400 "Request ID is invalid" means Venice does not know the id
            # (the job was rejected before it was stored): nothing to release.
            level = logging.DEBUG if isinstance(e, InvalidRequestError) else logging.WARNING
            logger.log(level, "MusicJob cleanup failed for queue_id=%s: %s", self.queue_id, e)

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
    def _is_terminal(self) -> bool:
        """Whether the job finished (completed, failed, or rejected by the server)."""
        return self._rejected or isinstance(self._status, (MusicCompletedStatus, MusicFailedStatus))

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
            MusicGenerationError: If the server reports generation failure,
                including a queued job that ``/audio/retrieve`` rejects
                outright with a 400 (failed server-side validation) or a 422
                (refused by the provider). The original
                :class:`~venice_ai.exceptions.InvalidRequestError` or
                :class:`~venice_ai.exceptions.UnprocessableEntityError` is
                chained as ``__cause__`` and the job counts as finished.
            TimeoutError: If ``max_polls`` is exhausted before completion.
            NotFoundError: If the job's stored media is gone: it was released
                (``release()``, or ``delete_media_on_completion``) or it
                expired. Venice answers ``"Media could not be found"``.
            APIError: For other HTTP-level failures while polling, including
                a 400 "Request ID is invalid" for a ``queue_id`` Venice never
                issued, 401/403 credential errors and 429.
        """
        for _ in range(max_polls):
            try:
                status = await self.poll()
            except (InvalidRequestError, UnprocessableEntityError) as e:
                if _is_unknown_request_id(e):
                    raise
                self._rejected = True
                raise MusicGenerationError(
                    f"Music generation failed: {e}",
                    error_code=e.code,
                    request=e.request,
                    response=e.response,
                ) from e
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

        Does NOT call :meth:`release` - use the context manager for that.
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
            MusicGenerationError: If the status carries neither inline data
                nor a ``url``; nothing is written in that case.
            APITimeoutError: If the URL fetch times out.
            APIConnectionError: If the connection fails during the URL fetch.
            aiohttp.ClientResponseError: If the URL answers with an error
                status.
            OSError: If the file cannot be written (permission denied,
                disk full, etc.).
        """
        if not (status.data or status.url):
            raise MusicGenerationError(
                f"Music job {self.queue_id} completed without downloadable audio: "
                "the status has no inline data or url"
            )
        path = Path(path)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        if status.data:
            await asyncio.to_thread(path.write_bytes, status.data)
        elif status.url:
            data = await self._client.fetch_external(status.url)
            await asyncio.to_thread(path.write_bytes, data)
        return path

    async def release(self) -> MusicCompleteResponse:
        """Delete this job's stored media and queue entry.

        Wraps ``POST /api/v1/audio/complete``. Call it after the audio has been
        downloaded. It does not stop or un-bill a job: billing happens when the
        job is queued, and a job still generating is unaffected and stores its
        audio when it finishes. Venice returns ``success: true`` for an id that
        was already released, so a ``True`` result does not mean anything was
        deleted by this call; ``success: false`` means the cleanup did not
        complete and can be retried later. Once a release of a finished job
        succeeds, leaving the ``async with`` block does not release it again.

        Returns:
            The :class:`MusicCompleteResponse`.

        Raises:
            APIError: For HTTP-level failures, including a 400 "Request ID is
                invalid" for an id Venice does not know.
        """
        result = await self._client.music.release(model=self.model, queue_id=self.queue_id)
        # Only a job that has finished has stored media to release; a release
        # sent while it is still generating deletes nothing, and the media it
        # stores later still needs releasing on exit.
        if result.success and self._is_terminal:
            self._released = True
        return result


class Music(APIResource["VeniceClient"]):
    """Asynchronous interface for Venice AI's Music generation API.

    Mirrors :class:`venice_ai.resources.video.Video`: ``submit`` queues a
    job, ``run`` returns a managed :class:`MusicJob`, ``retrieve`` polls,
    ``release`` frees server-side storage, ``quote`` gives a price
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
            InvalidRequestError: If the model id fails validation
                server-side, or the queue id was never issued (400 "Request
                ID is invalid").
            AuthenticationError: If the API key is missing or invalid.
            NotFoundError: If the job's stored media is gone because it was
                released or has expired (404 "Media could not be found").
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
            return await self._completed_from_binary(raw_response, content_type)

        # Read under the transport-error mapping, then parse: a stalled body
        # is an APITimeoutError, and a body that is not JSON is a parse error.
        body_bytes = await read_body(raw_response)
        try:
            response_data = await raw_response.json()
        except (aiohttp.ContentTypeError, ValueError) as e:
            logger.error(
                "Failed to parse JSON from music.retrieve response: %s. "
                "Content-Type: %r, Body preview: %.500s",
                e,
                content_type,
                body_bytes[:500].decode("utf-8", errors="replace"),
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

    async def release(
        self,
        *,
        model: str,
        queue_id: str,
    ) -> MusicCompleteResponse:
        """Delete a music job's stored media and queue entry.

        Wraps ``POST /api/v1/audio/complete``. Call it after the audio has been
        downloaded. It does not stop or un-bill a job: billing happens when the
        job is queued, and a job still generating stores its audio when it
        finishes. Venice returns ``success: true`` for an already released id;
        ``success: false`` means the cleanup did not complete and can be
        retried, and is logged at WARNING.

        Args:
            model: Music model id used at submit time.
            queue_id: Queue identifier returned by :meth:`submit`.

        Returns:
            The :class:`MusicCompleteResponse`.

        Raises:
            InvalidRequestError: If the model id fails validation, or Venice
                does not know the queue id (400 "Request ID is invalid").
            AuthenticationError: If the API key is missing or invalid.
            APIError: For other HTTP-level failures.
        """
        validate_model_id(model, "model")
        request = MusicCompleteRequest.model_validate({"model": model, "queue_id": queue_id})
        body = request.model_dump(exclude_none=True)
        response: MusicCompleteResponse = await self._client.post(
            "audio/complete",
            json_data=body,
            cast_to=MusicCompleteResponse,
        )
        if not response.success:
            _log_unsuccessful_release(queue_id, model)
        return response

    async def _completed_from_binary(
        self, raw_response: Any, content_type: str
    ) -> MusicCompletedStatus:
        """Build the completed status for a non-JSON retrieve response.

        Venice delivers finished audio inline with an ``audio/*`` Content-Type.
        ``application/octet-stream`` is accepted when the bytes are a known
        audio container. Anything else (an HTML proxy page, plain text, an
        empty body) is not audio and raises.
        """
        media_type, body = await read_inline_audio(
            raw_response, content_type, operation="music.retrieve"
        )
        logger.info("music.retrieve returned COMPLETED %s audio (%d bytes).", media_type, len(body))
        completed = MusicCompletedStatus.model_validate(
            {"status": "COMPLETED", "url": None, "content_type": media_type}
        )
        completed._set_data(body)
        return completed

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
        :meth:`MusicJob.release` to release server-side storage. Accepts the
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
