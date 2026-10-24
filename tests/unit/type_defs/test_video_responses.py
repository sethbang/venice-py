"""
Unit tests for video generation response models.

Tests parsing and validation for VideoQueueResponse, VideoQuoteResponse,
VideoProcessingStatus, VideoFailedStatus, VideoCompletedStatus, and
VideoCompleteResponse.
"""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.video import (
    VideoCompletedStatus,
    VideoCompleteResponse,
    VideoFailedStatus,
    VideoProcessingStatus,
    VideoQueueResponse,
    VideoQuoteResponse,
)


class TestVideoQueueResponse:
    """Test video queue response parsing."""

    def test_valid_response(self):
        response = VideoQueueResponse(
            model="wan-2.6-text-to-video",
            queue_id="550e8400-e29b-41d4-a716-446655440000",
        )
        assert response.model == "wan-2.6-text-to-video"
        assert response.queue_id == "550e8400-e29b-41d4-a716-446655440000"

    def test_requires_model(self):
        with pytest.raises(ValidationError):
            VideoQueueResponse(
                queue_id="550e8400-e29b-41d4-a716-446655440000",
            )  # type: ignore

    def test_requires_queue_id(self):
        with pytest.raises(ValidationError):
            VideoQueueResponse(
                model="wan-2.6-text-to-video",
            )  # type: ignore

    def test_from_dict(self):
        """Test parsing from dictionary (like API response)."""
        data = {
            "model": "wan-2.6-text-to-video",
            "queue_id": "abc-123-def",
        }
        response = VideoQueueResponse.model_validate(data)
        assert response.model == "wan-2.6-text-to-video"
        assert response.queue_id == "abc-123-def"


class TestVideoQuoteResponse:
    """Test video quote response parsing."""

    def test_integer_quote(self):
        """Test quote returned as integer."""
        response = VideoQuoteResponse(quote=5)
        assert response.quote == 5

    def test_float_quote(self):
        """Test quote returned as float."""
        response = VideoQuoteResponse(quote=0.25)
        assert response.quote == 0.25

    def test_zero_quote(self):
        """Test zero cost quote."""
        response = VideoQuoteResponse(quote=0)
        assert response.quote == 0

    def test_from_dict(self):
        data = {"quote": 1.50}
        response = VideoQuoteResponse.model_validate(data)
        assert response.quote == 1.50


class TestVideoProcessingStatus:
    """Test processing status response."""

    def test_valid_processing_status(self):
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=30000.0,
        )
        assert response.status == "PROCESSING"
        assert response.average_execution_time == 60000.0
        assert response.execution_duration == 30000.0

    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=30000.0,
        )
        assert response.progress_percent == 50.0

    def test_progress_percent_capped_at_100(self):
        """Test that progress doesn't exceed 100%."""
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=90000.0,  # Over the average
        )
        assert response.progress_percent == 100.0

    def test_progress_percent_with_zero_average(self):
        """Test progress when average is zero."""
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=0.0,
            execution_duration=30000.0,
        )
        assert response.progress_percent == 0.0

    def test_estimated_remaining_ms(self):
        """Test estimated remaining time calculation."""
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=40000.0,
        )
        assert response.estimated_remaining_ms == 20000.0

    def test_estimated_remaining_ms_never_negative(self):
        """Test that remaining time is never negative."""
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=80000.0,  # Over the average
        )
        assert response.estimated_remaining_ms == 0.0

    def test_unknown_status_is_kept(self):
        """A status the SDK does not know parses, with the timings defaulting to 0."""
        status = VideoProcessingStatus.model_validate({"status": "QUEUED"})
        assert status.status == "QUEUED"
        assert status.progress_percent == 0.0


class TestVideoFailedStatus:
    """Test failed status response."""

    def test_valid_failed_status(self):
        response = VideoFailedStatus(
            status="FAILED",
            error="Generation failed due to content policy violation",
            error_code="CONTENT_POLICY_VIOLATION",
        )
        assert response.status == "FAILED"
        assert response.error == "Generation failed due to content policy violation"
        assert response.error_code == "CONTENT_POLICY_VIOLATION"

    def test_minimal_failed_status(self):
        """Test failed status with only required fields."""
        response = VideoFailedStatus(status="FAILED")  # type: ignore
        assert response.status == "FAILED"
        assert response.error is None
        assert response.error_code is None

    def test_status_literal(self):
        """Test that only FAILED status is accepted."""
        with pytest.raises(ValidationError):
            VideoFailedStatus(
                status="PROCESSING",  # Wrong status
            )  # type: ignore


class TestVideoCompletedStatus:
    """Test completed status response."""

    def test_valid_completed_status(self):
        response = VideoCompletedStatus(
            status="COMPLETED",
            url="https://storage.example.com/video.mp4",
            expires_at="2024-12-25T00:00:00Z",
        )
        assert response.status == "COMPLETED"
        assert response.url == "https://storage.example.com/video.mp4"
        assert response.expires_at == "2024-12-25T00:00:00Z"

    def test_minimal_completed_status(self):
        """Test completed status with only required fields."""
        response = VideoCompletedStatus(status="COMPLETED")  # type: ignore
        assert response.status == "COMPLETED"
        assert response.url is None
        assert response.expires_at is None

    def test_status_literal(self):
        """Test that only COMPLETED status is accepted."""
        with pytest.raises(ValidationError):
            VideoCompletedStatus(
                status="FAILED",  # Wrong status
            )  # type: ignore


class TestVideoCompleteResponse:
    """Test complete/cleanup response."""

    def test_success_true(self):
        response = VideoCompleteResponse(success=True)
        assert response.success is True

    def test_success_false(self):
        response = VideoCompleteResponse(success=False)
        assert response.success is False

    def test_requires_success(self):
        with pytest.raises(ValidationError):
            VideoCompleteResponse()  # type: ignore


class TestVideoResponseSerialization:
    """Test serialization of video responses."""

    def test_queue_response_to_dict(self):
        response = VideoQueueResponse(
            model="wan-2.6-text-to-video",
            queue_id="abc-123",
        )
        data = response.model_dump()
        assert data["model"] == "wan-2.6-text-to-video"
        assert data["queue_id"] == "abc-123"

    def test_processing_status_to_dict(self):
        response = VideoProcessingStatus(
            status="PROCESSING",
            average_execution_time=60000.0,
            execution_duration=30000.0,
        )
        data = response.model_dump()
        assert data["status"] == "PROCESSING"
        assert data["average_execution_time"] == 60000.0
        assert data["execution_duration"] == 30000.0

    def test_failed_status_to_dict_excludes_none(self):
        response = VideoFailedStatus(status="FAILED")  # type: ignore
        data = response.model_dump(exclude_none=True)
        assert "error" not in data
        assert "error_code" not in data
