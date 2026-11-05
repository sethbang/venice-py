"""VideoJob.download_url: the queue-time link, readable so a caller can persist it."""

import logging
from unittest.mock import Mock

import pytest

from venice_ai.resources.video import VideoJob
from venice_ai.types.api.video import VideoQueueResponse

LINK = "https://cdn.example.com/queued.mp4?sig=SECRET"


def _job(download_url: str | None) -> VideoJob:
    response = VideoQueueResponse(model="m", queue_id="q", download_url=download_url)
    return VideoJob(Mock(), response)


def test_download_url_returns_the_queue_time_link() -> None:
    assert _job(LINK).download_url == LINK


def test_download_url_is_none_when_the_model_returns_no_link() -> None:
    assert _job(None).download_url is None


def test_download_url_cannot_be_reassigned() -> None:
    job = _job(LINK)
    with pytest.raises(AttributeError):
        job.download_url = "https://elsewhere.example.com/"  # type: ignore[misc]
    assert job.download_url == LINK


def test_the_link_stays_out_of_repr_and_str() -> None:
    job = _job(LINK)
    assert "SECRET" not in repr(job)
    assert "SECRET" not in str(job)


async def test_leaving_the_block_on_an_error_does_not_log_the_link(
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = _job(LINK)
    with caplog.at_level(logging.DEBUG), pytest.raises(RuntimeError):
        async with job:
            raise RuntimeError("download failed")
    assert "SECRET" not in caplog.text
