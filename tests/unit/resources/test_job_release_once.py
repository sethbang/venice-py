"""A job released inside its ``async with`` block is not released again on exit.

``MusicJob.release``, ``VideoJob.cancel`` and ``VoiceChangerJob.cancel`` free
a finished job's stored media. Once one has succeeded inside the block, leaving
the block -- cleanly or by an exception -- sends no second release and logs no
"was not released" warning. A release that reports ``success: false``, or one
sent before the job finished (it deletes nothing; the media is stored later),
leaves the exit-time release in place.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from venice_ai.resources.music import MusicJob
from venice_ai.resources.video import VideoJob
from venice_ai.resources.voice_changer import VoiceChangerJob
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicProcessingStatus,
    MusicQueueResponse,
)
from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoQueueResponse,
)
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerCompleteResponse,
    VoiceChangerProcessingStatus,
    VoiceChangerQueueResponse,
)


def _music(success: bool = True) -> tuple[MusicJob, AsyncMock]:
    client = Mock()
    client.music.retrieve = AsyncMock(return_value=MusicCompletedStatus(status="COMPLETED"))
    client.music.release = AsyncMock(return_value=MusicCompleteResponse(success=success))
    job = MusicJob(client, MusicQueueResponse(model="m", queue_id="q-music"))
    return job, client.music.release


def _video(success: bool = True) -> tuple[VideoJob, AsyncMock]:
    client = Mock()
    client.video.retrieve = AsyncMock(return_value=VideoCompletedStatus(status="COMPLETED"))
    client.video.cancel = AsyncMock(return_value=VideoCompleteResponse(success=success))
    job = VideoJob(client, VideoQueueResponse(model="m", queue_id="q-video"))
    return job, client.video.cancel


def _voice(success: bool = True) -> tuple[VoiceChangerJob, AsyncMock]:
    client = Mock()
    client.voice_changer.retrieve = AsyncMock(return_value=VoiceChangerCompletedStatus())
    client.voice_changer.cancel = AsyncMock(
        return_value=VoiceChangerCompleteResponse(success=success)
    )
    job = VoiceChangerJob(
        client,
        VoiceChangerQueueResponse(
            model="m", queue_id="q-vc", status="QUEUED", duration_seconds=1.0
        ),
    )
    return job, client.voice_changer.cancel


async def _release(job: Any) -> None:
    if isinstance(job, MusicJob):
        await job.release()
    else:
        await job.cancel()


async def _finish(job: Any) -> None:
    """Poll a job once, bringing it to a terminal status."""
    await job.poll()


FACTORIES = {"music": _music, "video": _video, "voice_changer": _voice}


@pytest.mark.parametrize("kind", FACTORIES)
async def test_clean_exit_after_explicit_release_does_not_release_again(kind: str) -> None:
    job, release = FACTORIES[kind]()
    async with job:
        await _finish(job)
        await _release(job)
    release.assert_awaited_once()


@pytest.mark.parametrize("kind", FACTORIES)
async def test_failing_exit_after_explicit_release_is_silent(
    kind: str, caplog: pytest.LogCaptureFixture
) -> None:
    job, release = FACTORIES[kind]()
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        async with job:
            await _finish(job)
            await _release(job)
            raise RuntimeError("after release")
    release.assert_awaited_once()
    assert "not released" not in caplog.text


@pytest.mark.parametrize("kind", FACTORIES)
async def test_unsuccessful_release_is_retried_on_exit(kind: str) -> None:
    job, release = FACTORIES[kind](success=False)
    async with job:
        await _finish(job)
        await _release(job)
    assert release.await_count == 2


async def test_music_release_before_completion_still_releases_on_exit() -> None:
    job, release = _music()
    job._client.music.retrieve.return_value = MusicProcessingStatus(
        status="PROCESSING", average_execution_time=1, execution_duration=1
    )
    async with job:
        await job.poll()
        await job.release()  # deletes nothing yet: the audio is stored when it finishes
        job._client.music.retrieve.return_value = MusicCompletedStatus(status="COMPLETED")
        await job.poll()
    assert release.await_count == 2


async def test_voice_changer_release_before_completion_still_releases_on_exit() -> None:
    job, release = _voice()
    job._client.voice_changer.retrieve.return_value = VoiceChangerProcessingStatus(
        status="PROCESSING", average_execution_time=1, execution_duration=1
    )
    async with job:
        await job.poll()
        await job.cancel()  # sent before the conversion finished
        job._client.voice_changer.retrieve.return_value = VoiceChangerCompletedStatus()
        await job.poll()
    assert release.await_count == 2


async def test_voice_changer_release_before_any_poll_still_releases_on_exit() -> None:
    job, release = _voice()
    async with job:
        await job.cancel()
    assert release.await_count == 2
