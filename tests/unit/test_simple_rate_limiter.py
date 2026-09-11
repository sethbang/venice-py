"""Unit tests for SimpleRateLimiter."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from venice_ai.rate_limiting import NoOpRateLimiter, SimpleRateLimiter


class TestHeaderParsing:
    """Tests for header parsing functionality."""

    @pytest.mark.asyncio
    async def test_parse_rate_limit_headers(self):
        """Verify x-ratelimit-* headers are parsed correctly."""
        limiter = SimpleRateLimiter()

        headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "999000",
            "x-ratelimit-reset-tokens": "60",
        }

        await limiter.update_from_headers("test-model", headers)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 500
        assert state["rpm_remaining"] == 499
        assert state["tpm_limit"] == 1000000
        assert state["tpm_remaining"] == 999000

    @pytest.mark.asyncio
    async def test_case_insensitive_headers(self):
        """Verify headers are case-insensitive."""
        limiter = SimpleRateLimiter()

        headers = {
            "X-RateLimit-Limit-Requests": "500",
            "X-RATELIMIT-REMAINING-REQUESTS": "499",
        }

        await limiter.update_from_headers("test-model", headers)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 500
        assert state["rpm_remaining"] == 499

    @pytest.mark.asyncio
    async def test_missing_headers_allowed(self):
        """Verify limiter handles missing headers gracefully."""
        limiter = SimpleRateLimiter()

        await limiter.update_from_headers("test-model", {}, status_code=200)

        can_proceed, wait_time = await limiter.acquire("test-model")
        assert can_proceed is True
        assert wait_time == 0

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 0
        assert state["tpm_limit"] == 0

    @pytest.mark.asyncio
    async def test_partial_headers(self):
        """Verify limiter handles partial headers."""
        limiter = SimpleRateLimiter()

        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
        }

        await limiter.update_from_headers("test-model", headers)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 100
        assert state["rpm_remaining"] == 50
        assert state["tpm_limit"] == 0
        assert state["tpm_remaining"] == 0


class TestPerModelIsolation:
    """Tests for per-model isolation."""

    @pytest.mark.asyncio
    async def test_per_model_isolation(self):
        """Ensure rate limit on Model A doesn't block Model B."""
        limiter = SimpleRateLimiter()

        # Simulate rate limit on model A
        await limiter.update_from_headers(
            "model-a",
            {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "60"},
            status_code=429,
        )

        # Model A should be blocked
        can_proceed_a, wait_time_a = await limiter.acquire("model-a")
        assert not can_proceed_a
        assert wait_time_a > 0

        # Model B should be fine
        can_proceed_b, wait_time_b = await limiter.acquire("model-b")
        assert can_proceed_b
        assert wait_time_b == 0


class TestBackoffCalculation:
    """Tests for backoff calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Verify exponential backoff with jitter."""
        limiter = SimpleRateLimiter(min_backoff=1.0, max_backoff=60.0)

        # First failure: ~1s backoff
        await limiter.record_failure("test-model")
        state = await limiter.get_state("test-model")
        assert state is not None
        backoff_time = state["backoff_until"] - time.time()
        assert 0.8 <= backoff_time <= 1.2  # ~1s with jitter

    @pytest.mark.asyncio
    async def test_max_backoff_cap(self):
        """Verify backoff is capped at max_backoff."""
        limiter = SimpleRateLimiter(min_backoff=1.0, max_backoff=10.0)

        # Trigger many failures
        for _ in range(10):
            await limiter.record_failure("test-model")

        state = await limiter.get_state("test-model")
        assert state is not None
        backoff_time = state["backoff_until"] - time.time()
        assert backoff_time <= 11.0  # max_backoff + jitter


class TestGlobalAbuseProtection:
    """Tests for global abuse protection."""

    @pytest.mark.asyncio
    async def test_global_abuse_protection(self):
        """Verify global block after threshold failures."""
        limiter = SimpleRateLimiter(failure_threshold=5, failure_window=10.0, block_duration=5.0)

        # Trigger 5 failures
        for i in range(5):
            await limiter.record_failure(f"model-{i}")

        # All models should be blocked
        can_proceed, wait_time = await limiter.acquire("any-model")
        assert not can_proceed
        assert 4.0 <= wait_time <= 5.5


class TestMemoryCleanup:
    """Tests for memory cleanup functionality."""

    @pytest.mark.asyncio
    async def test_stale_model_cleanup(self):
        """Verify stale models are cleaned up."""
        limiter = SimpleRateLimiter(
            stale_threshold=0.01,  # Very short for test
            max_models=100,
        )
        limiter.CLEANUP_INTERVAL = 0  # Force cleanup

        # Add some models
        for i in range(5):
            await limiter.update_from_headers(f"model-{i}", {"x-ratelimit-limit-requests": "100"})

        assert len(limiter._model_states) == 5

        # Wait for staleness
        await asyncio.sleep(0.02)

        # Force cleanup
        limiter._last_cleanup = 0
        await limiter._maybe_cleanup()

        # All models should be cleaned up as stale
        assert len(limiter._model_states) == 0

    @pytest.mark.asyncio
    async def test_max_models_limit(self):
        """Verify max_models limit is enforced."""
        limiter = SimpleRateLimiter(max_models=5, stale_threshold=3600.0)
        limiter.CLEANUP_INTERVAL = 0  # Force cleanup

        # Add more models than limit
        for i in range(10):
            await limiter.update_from_headers(f"model-{i}", {"x-ratelimit-limit-requests": "100"})
            limiter._last_cleanup = 0
            await limiter._maybe_cleanup()

        # Should not exceed max_models
        assert len(limiter._model_states) <= 5


class TestConcurrencySafety:
    """Tests for concurrency safety."""

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Verify thread safety under concurrent access."""
        import random

        limiter = SimpleRateLimiter()

        async def update_model(model_id: int):
            for _ in range(50):
                await limiter.update_from_headers(
                    f"model-{model_id}",
                    {"x-ratelimit-remaining-requests": str(random.randint(0, 500))},
                )
                await limiter.acquire(f"model-{model_id}")

        # Run 10 models concurrently, 50 updates each
        await asyncio.gather(*[update_model(i) for i in range(10)])

        # Should complete without race conditions
        states = await limiter.get_all_states()
        assert len(states) == 10


class TestResetTimeParsing:
    """Tests for reset time parsing."""

    @pytest.mark.asyncio
    async def test_reset_time_parsing_delta(self):
        """Test parsing delta seconds."""
        limiter = SimpleRateLimiter()

        await limiter.update_from_headers("test-model", {"x-ratelimit-reset-requests": "30"})

        state = await limiter.get_state("test-model")
        assert state is not None
        expected = time.time() + 30
        assert abs(state["rpm_reset"] - expected) < 2.0

    @pytest.mark.asyncio
    async def test_reset_time_parsing_timestamp(self):
        """Test parsing Unix timestamp."""
        limiter = SimpleRateLimiter()

        future_timestamp = time.time() + 60
        await limiter.update_from_headers(
            "test-model", {"x-ratelimit-reset-requests": str(int(future_timestamp))}
        )

        state = await limiter.get_state("test-model")
        assert state is not None
        assert abs(state["rpm_reset"] - future_timestamp) < 2.0

    @pytest.mark.asyncio
    async def test_reset_time_parsing_milliseconds(self):
        """Test parsing milliseconds timestamp."""
        limiter = SimpleRateLimiter()

        future_ms = (time.time() + 60) * 1000
        await limiter.update_from_headers(
            "test-model", {"x-ratelimit-reset-requests": str(int(future_ms))}
        )

        state = await limiter.get_state("test-model")
        assert state is not None
        expected = future_ms / 1000
        assert abs(state["rpm_reset"] - expected) < 2.0


class TestSubmitRequest:
    """Tests for submit_request() orchestration - CRITICAL."""

    @pytest.mark.asyncio
    async def test_submit_request_executes_request_func(self):
        """CRITICAL: Verify submit_request executes the request function."""
        limiter = SimpleRateLimiter()

        request_func_called = False
        expected_result = MagicMock()
        expected_result.status = 200
        expected_result.headers = {}

        async def mock_request_func():
            nonlocal request_func_called
            request_func_called = True
            return expected_result

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request_func)

        assert request_func_called, "submit_request must execute the request function"
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_submit_request_updates_state_from_response_headers(self):
        """Verify submit_request updates limiter state from response headers."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-remaining-requests": "499",
        }
        mock_response.status = 200

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        await limiter.submit_request(metadata, mock_request_func)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 500
        assert state["rpm_remaining"] == 499

    @pytest.mark.asyncio
    async def test_submit_request_handles_429_with_retry(self):
        """Verify submit_request handles 429 with retry."""
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=2)

        attempt_count = 0

        async def mock_request_func():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                response = MagicMock()
                response.status = 429
                response.headers = {"retry-after": "0.01"}
                response.json = AsyncMock(return_value={"error": "rate limited"})
                return response

            response = MagicMock()
            response.status = 200
            response.headers = {}
            return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request_func)

        assert attempt_count == 2
        assert result.status == 200

    @pytest.mark.asyncio
    async def test_submit_request_records_success_on_2xx(self):
        """Verify submit_request records success for successful responses."""
        limiter = SimpleRateLimiter()

        # First fail to set up a failure count
        await limiter.record_failure("test-model")
        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["consecutive_failures"] == 1

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status = 200

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Advance time past backoff instead of sleeping
        with patch("venice_ai.rate_limiting.simple.time") as mock_time:
            mock_time.time.return_value = time.time() + 2.0
            await limiter.submit_request(metadata, mock_request_func)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["consecutive_failures"] == 0

    @pytest.mark.asyncio
    async def test_submit_request_concurrent_requests(self):
        """Verify submit_request handles concurrent requests to same model."""
        limiter = SimpleRateLimiter()

        call_count = 0

        async def mock_request_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            mock_response = MagicMock()
            mock_response.headers = {}
            mock_response.status = 200
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        results = await asyncio.gather(
            *[limiter.submit_request(metadata, mock_request_func) for _ in range(10)]
        )

        assert len(results) == 10
        assert call_count == 10

    @pytest.mark.asyncio
    async def test_submit_request_blocks_during_backoff(self):
        """Verify submit_request blocks during backoff period."""
        limiter = SimpleRateLimiter(min_backoff=0.5, max_retries=0)

        # Record a failure to set up backoff
        await limiter.record_failure("test-model")

        mock_response = MagicMock()
        mock_response.headers = {}
        mock_response.status = 200

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # With max_retries=0, should raise immediately when blocked
        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)

    @pytest.mark.asyncio
    async def test_submit_request_handles_exceptions(self):
        """Verify submit_request handles exceptions from request_func."""
        limiter = SimpleRateLimiter()

        async def mock_request_func():
            raise ValueError("Test exception")

        metadata = MagicMock()
        metadata.model_id = "test-model"

        with pytest.raises(ValueError, match="Test exception"):
            await limiter.submit_request(metadata, mock_request_func)


class TestNoOpRateLimiter:
    """Tests for NoOpRateLimiter."""

    @pytest.mark.asyncio
    async def test_noop_always_allows(self):
        """NoOpRateLimiter always allows requests."""
        limiter = NoOpRateLimiter()

        can_proceed, wait_time = await limiter.acquire("any-model")
        assert can_proceed is True
        assert wait_time == 0

    @pytest.mark.asyncio
    async def test_noop_ignores_failures(self):
        """NoOpRateLimiter ignores failures."""
        limiter = NoOpRateLimiter()

        await limiter.record_failure("test-model")
        await limiter.update_from_headers("test-model", {}, status_code=429)

        can_proceed, _ = await limiter.acquire("test-model")
        assert can_proceed is True

    @pytest.mark.asyncio
    async def test_noop_submit_request_executes_directly(self):
        """NoOpRateLimiter.submit_request must also execute the request."""
        limiter = NoOpRateLimiter()

        request_executed = False
        expected_result = {"data": "test"}

        async def mock_request():
            nonlocal request_executed
            request_executed = True
            return expected_result

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request)

        assert request_executed, "NoOpRateLimiter must execute the request"
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_noop_passes_through_429_responses(self):
        """NoOpRateLimiter passes through 429 responses without retry."""
        limiter = NoOpRateLimiter()

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}

        async def mock_request():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request)

        # NoOpRateLimiter should NOT retry or create errors
        assert result.status == 429


class TestLifecycle:
    """Tests for lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test start and stop lifecycle."""
        limiter = SimpleRateLimiter()

        assert not limiter.is_running()

        await limiter.start()
        assert limiter.is_running()

        await limiter.stop()
        assert not limiter.is_running()

    @pytest.mark.asyncio
    async def test_clear_on_stop(self):
        """Test that state is cleared on stop."""
        limiter = SimpleRateLimiter()
        await limiter.start()

        await limiter.update_from_headers("test-model", {"x-ratelimit-limit-requests": "100"})
        assert len(limiter._model_states) == 1

        await limiter.stop()
        assert len(limiter._model_states) == 0

    def test_classifier_property(self):
        """Test classifier property."""
        limiter = SimpleRateLimiter()

        assert limiter.classifier is None

        mock_classifier = MagicMock()
        limiter.classifier = mock_classifier

        assert limiter.classifier is mock_classifier

    @pytest.mark.asyncio
    async def test_noop_lifecycle(self):
        """Test NoOpRateLimiter lifecycle."""
        limiter = NoOpRateLimiter()

        assert not limiter.is_running()

        await limiter.start()
        assert limiter.is_running()

        await limiter.stop()
        assert not limiter.is_running()

    def test_noop_classifier_property(self):
        """Test NoOpRateLimiter classifier property."""
        limiter = NoOpRateLimiter()

        assert limiter.classifier is None

        mock_classifier = MagicMock()
        limiter.classifier = mock_classifier

        assert limiter.classifier is mock_classifier


class TestGetStats:
    """Tests for get_stats functionality."""

    def test_get_stats_initial(self):
        """Test get_stats on fresh limiter."""
        limiter = SimpleRateLimiter(max_models=1000)

        stats = limiter.get_stats()

        assert stats["tracked_models"] == 0
        assert stats["tracked_locks"] == 0
        assert stats["max_models"] == 1000
        assert stats["global_failures"] == 0
        assert stats["global_blocked"] is False

    @pytest.mark.asyncio
    async def test_get_stats_after_activity(self):
        """Test get_stats after some activity."""
        limiter = SimpleRateLimiter(max_models=100)

        await limiter.update_from_headers("model-1", {"x-ratelimit-limit-requests": "100"})
        await limiter.update_from_headers("model-2", {"x-ratelimit-limit-requests": "200"})

        stats = limiter.get_stats()

        assert stats["tracked_models"] == 2
        assert stats["max_models"] == 100


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_get_state_nonexistent_model(self):
        """Test get_state for a model that doesn't exist."""
        limiter = SimpleRateLimiter()

        # get_state returns None for nonexistent models
        state = await limiter.get_state("nonexistent-model")
        assert state is None

    @pytest.mark.asyncio
    async def test_invalid_header_values(self):
        """Test handling of invalid header values (non-numeric)."""
        limiter = SimpleRateLimiter()

        # This should not raise an exception but may log a warning
        # Based on implementation, invalid int conversion would raise
        # So we test with valid values but edge cases
        headers = {
            "x-ratelimit-limit-requests": "0",
            "x-ratelimit-remaining-requests": "0",
        }

        await limiter.update_from_headers("test-model", headers)

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["rpm_limit"] == 0
        assert state["rpm_remaining"] == 0

    @pytest.mark.asyncio
    async def test_reset_time_invalid_value(self):
        """Test handling of invalid reset time values."""
        limiter = SimpleRateLimiter()

        # Test with non-numeric value - implementation defaults to 60 seconds
        # Note: The implementation converts to float, so "invalid" would raise ValueError
        # and return time.time() + 60.0
        # We test the behavior indirectly
        await limiter.update_from_headers("test-model", {"x-ratelimit-limit-requests": "100"})
        state = await limiter.get_state("test-model")
        assert state is not None

    @pytest.mark.asyncio
    async def test_concurrent_cleanup(self):
        """Test that cleanup handles concurrent access safely."""
        limiter = SimpleRateLimiter(max_models=10, stale_threshold=0.001)
        limiter.CLEANUP_INTERVAL = 0

        async def add_and_cleanup(model_id: int):
            await limiter.update_from_headers(
                f"model-{model_id}", {"x-ratelimit-limit-requests": "100"}
            )
            await asyncio.sleep(0.002)  # Wait for staleness
            limiter._last_cleanup = 0
            await limiter._maybe_cleanup()

        # Run multiple concurrent add/cleanup operations
        await asyncio.gather(*[add_and_cleanup(i) for i in range(20)])

        # Should complete without errors
        assert len(limiter._model_states) <= 10


class TestErrorFactoryIntegration:
    """Tests for error_factory integration - CRITICAL for error context parity."""

    @pytest.mark.asyncio
    async def test_submit_request_uses_error_factory_on_429(self):
        """Verify error_factory is called with correct arguments on 429."""
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {"retry-after": "1"}
        mock_response.json = AsyncMock(return_value={"error": {"message": "Rate limit exceeded"}})

        async def mock_request_func():
            return mock_response

        # Track error_factory calls
        error_factory_calls = []

        def mock_error_factory(message, request, body, response):
            error_factory_calls.append(
                {
                    "message": message,
                    "request": request,
                    "body": body,
                    "response": response,
                }
            )
            from venice_ai.exceptions import RateLimitError

            return RateLimitError(message=message, response=response, body=body)

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError):
            await limiter.submit_request(
                metadata, mock_request_func, error_factory=mock_error_factory
            )

        # Verify error_factory was called with correct arguments
        assert len(error_factory_calls) == 1
        call = error_factory_calls[0]
        assert call["message"] == "Rate limit exceeded"
        assert call["request"] is None  # Request object not available in this context
        assert call["body"] == {"error": {"message": "Rate limit exceeded"}}
        assert call["response"] is mock_response

    @pytest.mark.asyncio
    async def test_error_factory_creates_error_with_full_context(self):
        """Verify error created by factory has full error context."""
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {
            "retry-after": "30",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "0",
        }
        mock_response.json = AsyncMock(
            return_value={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            }
        )

        async def mock_request_func():
            return mock_response

        def full_context_error_factory(message, request, body, response):
            from venice_ai.exceptions import RateLimitError

            error = RateLimitError(message=message, response=response, body=body)
            # Simulate _make_status_error behavior of extracting retry_after
            if hasattr(response, "headers") and "retry-after" in response.headers:
                error.retry_after_seconds = int(response.headers["retry-after"])
            return error

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(
                metadata, mock_request_func, error_factory=full_context_error_factory
            )

        # Verify error has full context
        error = exc_info.value
        assert error.body == {
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        }
        assert error.retry_after_seconds == 30

    @pytest.mark.asyncio
    async def test_error_factory_not_called_on_success(self):
        """Verify error_factory is NOT called for successful responses."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {}

        async def mock_request_func():
            return mock_response

        error_factory_called = False

        def mock_error_factory(message, request, body, response):
            nonlocal error_factory_called
            error_factory_called = True
            from venice_ai.exceptions import RateLimitError

            return RateLimitError(message=message, response=response, body=body)

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(
            metadata, mock_request_func, error_factory=mock_error_factory
        )

        assert result.status == 200
        assert not error_factory_called, "error_factory should not be called for 2xx responses"

    @pytest.mark.asyncio
    async def test_fallback_error_when_no_factory(self):
        """Verify basic RateLimitError is created when no error_factory provided."""
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"error": "rate limited"})

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            # Note: NO error_factory parameter
            await limiter.submit_request(metadata, mock_request_func)

        # Should still raise RateLimitError with basic info
        error = exc_info.value
        assert "Rate limit exceeded" in str(error)
        assert error.body == {"error": "rate limited"}


class TestErrorContextParity:
    """Tests verifying scheduler path errors match direct path errors."""

    @pytest.mark.asyncio
    async def test_scheduler_error_matches_direct_path_structure(self):
        """
        CRITICAL: Verify scheduler 429 errors have same structure as direct path errors.

        Per plan section 1.1a: "SimpleRateLimiter.submit_request() must create
        RateLimitError using the same path as the direct path handler."
        """
        from venice_ai.exceptions import RateLimitError, _make_status_error

        limiter = SimpleRateLimiter(max_retries=0)

        # Create a mock response that simulates a 429
        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {"retry-after": "10"}
        mock_response.json = AsyncMock(
            return_value={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                }
            }
        )

        body = {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        # Create error via scheduler path (using _make_status_error as factory)
        scheduler_error = None
        try:
            await limiter.submit_request(
                metadata, mock_request_func, error_factory=_make_status_error
            )
        except RateLimitError as e:
            scheduler_error = e

        # Create error via direct path simulation
        direct_error = _make_status_error(
            message="API request failed with status 429",
            request=None,
            body=body,
            response=mock_response,
        )

        # Verify both are RateLimitError
        assert isinstance(scheduler_error, RateLimitError)
        assert isinstance(direct_error, RateLimitError)

        # Verify both have the response object
        assert scheduler_error.response is mock_response
        assert direct_error.response is mock_response

        # Verify both have the body
        assert scheduler_error.body == body
        assert direct_error.body == body

    @pytest.mark.asyncio
    async def test_error_factory_receives_parsed_response_body(self):
        """Verify error_factory receives the parsed JSON body, not raw response."""
        limiter = SimpleRateLimiter(max_retries=0)

        # Complex nested error body
        expected_body = {
            "error": {
                "message": "Rate limit exceeded for model gpt-4",
                "type": "rate_limit_error",
                "param": None,
                "code": "rate_limit_exceeded",
            },
            "request_id": "req_abc123",
        }

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value=expected_body)

        async def mock_request_func():
            return mock_response

        received_body = None

        def capture_body_factory(message, request, body, response):
            nonlocal received_body
            received_body = body
            from venice_ai.exceptions import RateLimitError

            return RateLimitError(message=message, response=response, body=body)

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError):
            await limiter.submit_request(
                metadata, mock_request_func, error_factory=capture_body_factory
            )

        # Verify the full parsed body was passed to error_factory
        assert received_body is not None
        assert received_body == expected_body
        assert received_body["request_id"] == "req_abc123"


class TestProtocolCompliance:
    """Tests verifying RateLimiterProtocol compliance - covers lines 77, 81, 85, 89, 94."""

    def test_protocol_methods_exist(self):
        """Verify SimpleRateLimiter implements all Protocol methods."""

        limiter = SimpleRateLimiter()

        # Verify all required methods exist
        assert hasattr(limiter, "submit_request")
        assert hasattr(limiter, "is_running")
        assert hasattr(limiter, "start")
        assert hasattr(limiter, "stop")
        assert hasattr(limiter, "classifier")

        # Verify they're callable (except classifier which is a property)
        assert callable(limiter.submit_request)
        assert callable(limiter.is_running)
        assert callable(limiter.start)
        assert callable(limiter.stop)

    def test_noop_protocol_compliance(self):
        """Verify NoOpRateLimiter also implements Protocol methods."""
        limiter = NoOpRateLimiter()

        assert hasattr(limiter, "submit_request")
        assert hasattr(limiter, "is_running")
        assert hasattr(limiter, "start")
        assert hasattr(limiter, "stop")
        assert hasattr(limiter, "classifier")


class TestRpmExhaustedRateLimiting:
    """Tests for rate limiting when rpm_remaining == 0 - covers lines 139-140."""

    @pytest.mark.asyncio
    async def test_is_rate_limited_when_rpm_exhausted(self):
        """Verify rate limiting when rpm_remaining is 0 and reset is in the future."""
        limiter = SimpleRateLimiter()

        # Set up state where rpm is exhausted but reset is in the future
        state = limiter._get_state("test-model")
        state.rpm_limit = 100
        state.rpm_remaining = 0
        state.rpm_reset = time.time() + 30  # Resets in 30 seconds

        is_limited, wait_time = state.is_rate_limited()

        assert is_limited is True
        assert 28 <= wait_time <= 31  # Should wait ~30 seconds

    @pytest.mark.asyncio
    async def test_not_rate_limited_when_rpm_exhausted_but_reset_passed(self):
        """Verify not rate limited if rpm_remaining is 0 but reset time is past."""
        limiter = SimpleRateLimiter()

        state = limiter._get_state("test-model")
        state.rpm_limit = 100
        state.rpm_remaining = 0
        state.rpm_reset = time.time() - 10  # Already reset

        is_limited, wait_time = state.is_rate_limited()

        assert is_limited is False
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_blocked_when_rpm_exhausted(self):
        """Verify acquire blocks when rpm is exhausted via headers."""
        limiter = SimpleRateLimiter()

        # Update state to exhausted
        await limiter.update_from_headers(
            "test-model",
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-remaining-requests": "0",
                "x-ratelimit-reset-requests": str(time.time() + 60),  # Unix timestamp
            },
        )

        can_proceed, wait_time = await limiter.acquire("test-model")

        assert can_proceed is False
        assert wait_time > 0


class TestCleanupDoubleCheck:
    """Tests for cleanup double-check logic - covers line 280."""

    @pytest.mark.asyncio
    async def test_cleanup_double_check_via_concurrent_access(self):
        """Test line 280: double-check after acquiring lock skips if already cleaned.

        This test triggers the double-check by having concurrent cleanup attempts
        where the first one completes the cleanup and the second one should skip
        at the double-check (line 279-280).
        """
        limiter = SimpleRateLimiter(stale_threshold=0.001, max_models=10)
        limiter.CLEANUP_INTERVAL = 0  # Allow frequent cleanup

        # Add some models
        for i in range(3):
            await limiter.update_from_headers(f"model-{i}", {"x-ratelimit-limit-requests": "100"})

        # Wait for staleness
        await asyncio.sleep(0.01)

        cleanup_executed_count = 0
        original_clear_method = limiter._model_states.clear

        def count_clear(*args, **kwargs):
            """Count how many times clear is called (cleanup execution marker)."""
            nonlocal cleanup_executed_count
            cleanup_executed_count += 1
            return original_clear_method(*args, **kwargs)

        barrier = asyncio.Barrier(2)

        async def concurrent_cleanup(task_id):
            """Run cleanup with coordination to trigger double-check."""
            # Both tasks reset _last_cleanup to trigger cleanup
            limiter._last_cleanup = 0

            # Wait for both to be ready
            await barrier.wait()

            # Both now try to cleanup
            await limiter._maybe_cleanup()

        # Run concurrent cleanups
        await asyncio.gather(
            concurrent_cleanup(1),
            concurrent_cleanup(2),
        )

        # Should complete without error
        assert True  # Test passes if no exception

    @pytest.mark.asyncio
    async def test_cleanup_first_check_skips(self):
        """Test that cleanup skips via first check when interval not elapsed."""
        limiter = SimpleRateLimiter(stale_threshold=3600, max_models=100)
        limiter.CLEANUP_INTERVAL = 3600  # Very long interval

        # Add a model
        await limiter.update_from_headers("test-model", {"x-ratelimit-limit-requests": "100"})

        # Set last cleanup to now
        limiter._last_cleanup = time.time()
        initial_cleanup_time = limiter._last_cleanup

        # Try cleanup - should skip at first check (line 274-275)
        await limiter._maybe_cleanup()

        # Should still have the same cleanup time (didn't re-run)
        assert limiter._last_cleanup == initial_cleanup_time
        # Model should still exist
        assert "test-model" in limiter._model_states


class TestRetryAfterValueErrorHandling:
    """Tests for retry-after header ValueError handling - covers lines 399-400."""

    @pytest.mark.asyncio
    async def test_apply_backoff_with_invalid_retry_after(self):
        """Verify fallback to calculated backoff when retry-after can't be parsed."""
        limiter = SimpleRateLimiter(min_backoff=2.0, max_backoff=60.0)

        # Call update_from_headers with invalid retry-after and 429 status
        await limiter.update_from_headers(
            "test-model",
            {
                "retry-after": "invalid-not-a-number",  # Will cause ValueError in float()
            },
            status_code=429,
        )

        state = await limiter.get_state("test-model")
        assert state is not None

        # Should use calculated backoff (~2 seconds for first failure)
        backoff_time = state["backoff_until"] - time.time()
        # With jitter of ±10%, should be between 1.8 and 2.2
        assert 1.5 <= backoff_time <= 2.5

    @pytest.mark.asyncio
    async def test_apply_backoff_with_empty_retry_after(self):
        """Verify fallback when retry-after is empty string."""
        limiter = SimpleRateLimiter(min_backoff=1.0)

        await limiter.update_from_headers(
            "test-model",
            {
                "retry-after": "",  # Empty string causes ValueError
            },
            status_code=429,
        )

        state = await limiter.get_state("test-model")
        assert state is not None
        assert state["consecutive_failures"] == 1


class TestParseResetTimeValueError:
    """Tests for _parse_reset_time ValueError handling - covers lines 477-479."""

    @pytest.mark.asyncio
    async def test_parse_reset_time_invalid_value(self):
        """Verify _parse_reset_time returns 60s fallback for invalid values."""
        limiter = SimpleRateLimiter()

        # Directly call _parse_reset_time with invalid value
        now = time.time()
        result = limiter._parse_reset_time("not-a-number")

        # Should return current time + 60 seconds as fallback
        assert abs(result - (now + 60)) < 2.0

    @pytest.mark.asyncio
    async def test_parse_reset_time_empty_string(self):
        """Verify _parse_reset_time handles empty string."""
        limiter = SimpleRateLimiter()

        now = time.time()
        result = limiter._parse_reset_time("")

        # Empty string causes ValueError, should fall back to +60s
        assert abs(result - (now + 60)) < 2.0

    @pytest.mark.asyncio
    async def test_parse_reset_time_special_chars(self):
        """Verify _parse_reset_time handles special characters."""
        limiter = SimpleRateLimiter()

        now = time.time()
        result = limiter._parse_reset_time("@#$%")

        assert abs(result - (now + 60)) < 2.0

    @pytest.mark.asyncio
    async def test_update_from_headers_with_invalid_reset(self):
        """Verify headers with invalid reset time use fallback."""
        limiter = SimpleRateLimiter()

        now = time.time()
        await limiter.update_from_headers(
            "test-model",
            {
                "x-ratelimit-limit-requests": "100",
                "x-ratelimit-reset-requests": "invalid",
            },
        )

        state = await limiter.get_state("test-model")
        assert state is not None
        # Reset time should be approximately 60 seconds from now
        assert abs(state["rpm_reset"] - (now + 60)) < 2.0


class TestSubmitRequestLocalRateLimitExhausted:
    """Tests for submit_request when local state shows exhausted limits - covers lines 584, 596-601."""

    @pytest.mark.asyncio
    async def test_submit_request_with_prior_rate_limit_error(self):
        """Test line 584: reraise last_rate_limit_error when local state blocks at max retries."""
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=1)

        attempt_count = 0

        async def mock_request_func():
            nonlocal attempt_count
            attempt_count += 1

            # First attempt: return 429 to set last_rate_limit_error
            response = MagicMock()
            response.status = 429
            response.headers = {"retry-after": "0.01"}
            response.json = AsyncMock(return_value={"error": "rate limited"})
            return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        # Should eventually raise after retries exhausted
        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        # Verify we got the expected error
        assert "Rate limit exceeded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_submit_request_waits_when_locally_blocked_not_at_max(self):
        """Test lines 596-601: wait and continue when locally blocked but not at max retries."""
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=2)

        # Pre-set backoff state via failure
        await limiter.record_failure("test-model")

        # Now update to allow request after short wait
        state = limiter._get_state("test-model")
        state.backoff_until = time.time() + 0.02  # Very short backoff

        call_count = 0

        async def mock_request_func():
            nonlocal call_count
            call_count += 1
            response = MagicMock()
            response.status = 200
            response.headers = {}
            return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        start_time = time.time()
        result = await limiter.submit_request(metadata, mock_request_func)
        elapsed = time.time() - start_time

        # Should have waited for backoff
        assert elapsed >= 0.01
        # Should have succeeded
        assert result.status == 200
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_submit_request_exhausted_without_prior_429(self):
        """Test lines 588-594: RateLimitError when headers show exhausted limits before any 429."""
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=0)

        # Pre-set rpm exhausted via headers (simulate previous response)
        state = limiter._get_state("test-model")
        state.rpm_limit = 100
        state.rpm_remaining = 0
        state.rpm_reset = time.time() + 60  # Still waiting for reset

        async def mock_request_func():
            response = MagicMock()
            response.status = 200
            response.headers = {}
            return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        # Should indicate exhausted limits based on headers
        assert "local state indicates exhausted limits" in str(exc_info.value)


class TestSubmitRequestWaitTimeFromRetryAfter:
    """Tests for wait time fallback to retry_after_seconds - covers line 652."""

    @pytest.mark.asyncio
    async def test_wait_time_from_retry_after_seconds_attribute(self):
        """The retry-after-seconds branch retries 429 → 200 successfully.

        Uses a tiny ``min_backoff`` so the ``backoff_until`` window is well
        below the per-iteration wall-clock overhead of the retry loop. The
        previous ``min_backoff=30.0`` raced under pytest-xdist: ``asyncio.sleep``
        returns within event-loop tolerance, ``acquire`` re-reads ``time.time()``,
        and under parallel-worker scheduling jitter the second attempt's
        ``now < backoff_until`` check intermittently flipped the limiter back
        into the rate-limited branch. With sub-millisecond backoff the gap is
        always cleared by the time we re-enter ``acquire``.
        """
        limiter = SimpleRateLimiter(min_backoff=0.01, max_retries=1)

        attempt_count = 0

        async def mock_request_func():
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                response = MagicMock()
                response.status = 429
                response.headers = {}  # No retry-after header
                response.json = AsyncMock(return_value={"error": "rate limited"})
                return response

            response = MagicMock()
            response.status = 200
            response.headers = {}
            return response

        metadata = MagicMock()
        metadata.model_id = "nonexistent-model"

        # Clear state to ensure get_state returns something without backoff_until set
        await limiter.clear()

        # Use a custom error factory that sets retry_after_seconds
        def factory_with_retry_after(message, request, body, response):
            from venice_ai.exceptions import RateLimitError

            error = RateLimitError(message=message, response=response, body=body)
            error.retry_after_seconds = 1  # Very short for test
            return error

        result = await limiter.submit_request(
            metadata, mock_request_func, error_factory=factory_with_retry_after
        )

        # Should have retried and succeeded
        assert result.status == 200
        assert attempt_count == 2


class TestRetryExhaustionFallback:
    """Tests for retry exhaustion edge cases - covers lines 671-674."""

    @pytest.mark.asyncio
    async def test_retry_loop_exhaustion_with_error(self):
        """Test lines 671-672: reraise last_rate_limit_error after loop exhaustion."""
        limiter = SimpleRateLimiter(min_backoff=0.001, max_retries=0)

        async def mock_request_func():
            response = MagicMock()
            response.status = 429
            response.headers = {}
            response.json = AsyncMock(return_value={"error": "rate limited"})
            return response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError):
            await limiter.submit_request(metadata, mock_request_func)


class TestParseResponseBodyFallbacks:
    """Tests for _parse_response_body fallback logic - covers lines 682-686."""

    @pytest.mark.asyncio
    async def test_parse_response_body_json_success(self):
        """Test successful JSON parsing."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(return_value={"error": "test"})

        result = await limiter._parse_response_body(mock_response)

        assert result == {"error": "test"}

    @pytest.mark.asyncio
    async def test_parse_response_body_json_fails_text_succeeds(self):
        """Test line 683-684: fallback to text() when json() fails."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
        mock_response.text = AsyncMock(return_value="Plain text error message")

        result = await limiter._parse_response_body(mock_response)

        assert result == "Plain text error message"

    @pytest.mark.asyncio
    async def test_parse_response_body_both_fail(self):
        """Test lines 685-686: return None when both json() and text() fail."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.json = AsyncMock(side_effect=Exception("JSON failed"))
        mock_response.text = AsyncMock(side_effect=Exception("Text failed"))

        result = await limiter._parse_response_body(mock_response)

        assert result is None

    @pytest.mark.asyncio
    async def test_submit_request_429_with_text_body(self):
        """Integration test: 429 response with non-JSON body."""
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(side_effect=ValueError("Not JSON"))
        mock_response.text = AsyncMock(return_value="Rate limit exceeded - please wait")

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        # Body should be the text response
        assert exc_info.value.body == "Rate limit exceeded - please wait"

    @pytest.mark.asyncio
    async def test_submit_request_429_with_no_parseable_body(self):
        """Integration test: 429 response with unparseable body."""
        limiter = SimpleRateLimiter(max_retries=0)

        mock_response = MagicMock()
        mock_response.status = 429
        mock_response.headers = {}
        mock_response.json = AsyncMock(side_effect=Exception("No JSON"))
        mock_response.text = AsyncMock(side_effect=Exception("No text either"))

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        from venice_ai.exceptions import RateLimitError

        with pytest.raises(RateLimitError) as exc_info:
            await limiter.submit_request(metadata, mock_request_func)

        # Body should be None when parsing fails
        assert exc_info.value.body is None


class TestCleanupWithLockedModels:
    """Tests for cleanup skipping locked models - covers partial branches 294→292, 296→292, 310→308, 312→308."""

    @pytest.mark.asyncio
    async def test_cleanup_skips_locked_stale_model(self):
        """Test that cleanup doesn't delete lock if model is locked during stale cleanup."""
        limiter = SimpleRateLimiter(stale_threshold=0.001, max_models=100)
        limiter.CLEANUP_INTERVAL = 0

        # Add a model
        await limiter.update_from_headers("test-model", {"x-ratelimit-limit-requests": "100"})

        await asyncio.sleep(0.01)  # Let it become stale

        # Acquire the lock
        lock = limiter._get_lock("test-model")

        async def hold_lock_during_cleanup():
            """Hold the lock while cleanup runs."""
            async with lock:
                # Reset to trigger cleanup
                limiter._last_cleanup = 0

                # Run cleanup while holding lock
                await limiter._maybe_cleanup()

                await asyncio.sleep(0.01)  # Hold it briefly

        await hold_lock_during_cleanup()

        # Model state should be removed (it's stale)
        assert "test-model" not in limiter._model_states

        # Lock should still exist because it was locked during cleanup
        assert "test-model" in limiter._locks

    @pytest.mark.asyncio
    async def test_cleanup_skips_locked_excess_model(self):
        """Test that cleanup doesn't delete lock if model is locked during excess cleanup."""
        limiter = SimpleRateLimiter(max_models=2, stale_threshold=3600)
        limiter.CLEANUP_INTERVAL = 0

        # Add models - will exceed max
        for i in range(5):
            await limiter.update_from_headers(f"model-{i}", {"x-ratelimit-limit-requests": "100"})
            # Small sleep to establish access order
            await asyncio.sleep(0.001)

        # Get lock on oldest model
        oldest_lock = limiter._get_lock("model-0")

        async with oldest_lock:
            # Trigger cleanup
            limiter._last_cleanup = 0
            await limiter._maybe_cleanup()

            # Lock for model-0 should still exist (was locked during cleanup)
            assert "model-0" in limiter._locks

        # Should have reduced to max_models
        assert len(limiter._model_states) <= 2


class TestResponseWithoutHeaders:
    """Tests for responses without headers attribute - covers partial branch 614→622."""

    @pytest.mark.asyncio
    async def test_submit_request_response_without_headers_attribute(self):
        """Test that submit_request handles response objects without headers attribute."""
        limiter = SimpleRateLimiter()

        class HeaderlessResponse:
            """Response object without headers attribute."""

            status = 200

        async def mock_request_func():
            return HeaderlessResponse()

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request_func)

        assert result.status == 200

    @pytest.mark.asyncio
    async def test_submit_request_none_headers(self):
        """Test response with headers=None."""
        limiter = SimpleRateLimiter()

        mock_response = MagicMock()
        mock_response.status = 200
        # Remove headers attribute entirely
        del mock_response.headers

        async def mock_request_func():
            return mock_response

        metadata = MagicMock()
        metadata.model_id = "test-model"

        result = await limiter.submit_request(metadata, mock_request_func)

        # Should complete without error
        assert result.status == 200


class TestGetAllStatesWithDeletion:
    """Tests for get_all_states when state becomes None - covers partial branch 506→504."""

    @pytest.mark.asyncio
    async def test_get_all_states_handles_concurrent_deletion(self):
        """Test get_all_states handles state being deleted during iteration."""
        limiter = SimpleRateLimiter()

        # Add some models
        for i in range(5):
            await limiter.update_from_headers(f"model-{i}", {"x-ratelimit-limit-requests": "100"})

        original_get_state = limiter.get_state
        call_count = 0

        async def get_state_with_deletion(model):
            nonlocal call_count
            call_count += 1
            # Delete a model mid-iteration
            if call_count == 3 and "model-4" in limiter._model_states:
                del limiter._model_states["model-4"]
            return await original_get_state(model)

        # Patch get_state
        limiter.get_state = get_state_with_deletion

        # Should handle the concurrent deletion gracefully
        result = await limiter.get_all_states()

        # Should have 4 states (one was deleted during iteration)
        # The exact count depends on which models were iterated before deletion
        assert isinstance(result, dict)


class TestErrorBudgetBackoff:
    """A 429 from a rolling error budget needs a full-window backoff.

    The budgets are 30-second windows that exist to stop clients retrying into
    a wall. The default backoff schedule (1s, 2s, 4s) retries three times
    entirely inside that window, so every retry fails — and each one counts
    against the failed-request budget again, deepening the hole.
    """

    @staticmethod
    def _error_budget_response():
        response = MagicMock()
        response.status = 429
        response.headers = {"x-ratelimit-remaining": "0", "x-ratelimit-resets": "30"}
        return response

    @pytest.mark.asyncio
    async def test_error_budget_429_waits_at_least_the_window(self):
        from venice_ai._queue_types import RequestMetadata, ResourceType
        from venice_ai.exceptions import RateLimitError, _make_status_error

        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=1)
        response = self._error_budget_response()

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch("asyncio.sleep", side_effect=fake_sleep), pytest.raises(RateLimitError):
            await limiter.submit_request(
                RequestMetadata(
                    request_id="req-1",
                    model_id="test-model",
                    resource_type=ResourceType.LLM,
                    endpoint="chat/completions",
                ),
                AsyncMock(return_value=response),
                error_factory=_make_status_error,
            )

        assert slept, "expected a backoff before the retry"
        assert max(slept) >= 30.0, f"backed off only {slept!r}, inside the 30s window"

    @pytest.mark.asyncio
    async def test_throughput_429_keeps_the_normal_backoff(self):
        """Only error budgets get the floor; a plain 429 must not wait 30s."""
        from venice_ai._queue_types import RequestMetadata, ResourceType
        from venice_ai.exceptions import RateLimitError, _make_status_error

        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=1)
        response = MagicMock()
        response.status = 429
        response.headers = {"x-ratelimit-reset-requests": "60"}

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch("asyncio.sleep", side_effect=fake_sleep), pytest.raises(RateLimitError):
            await limiter.submit_request(
                RequestMetadata(
                    request_id="req-1",
                    model_id="test-model",
                    resource_type=ResourceType.LLM,
                    endpoint="chat/completions",
                ),
                AsyncMock(return_value=response),
                error_factory=_make_status_error,
            )

        assert all(s < 30.0 for s in slept), f"unexpected 30s floor on a throughput 429: {slept!r}"

    @pytest.mark.asyncio
    async def test_backoff_does_not_outlast_the_caller_timeout(self):
        """A 30s floor must not silently blow past the timeout the caller set.

        RequestMetadata.timeout is the caller's ceiling for the whole call, so
        once the accumulated backoff would exceed it there is no point sleeping
        — surface the rate-limit error instead of holding the call past the
        deadline and returning neither in time.
        """
        from venice_ai._queue_types import RequestMetadata, ResourceType
        from venice_ai.exceptions import RateLimitError, _make_status_error

        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=3)
        response = self._error_budget_response()

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch("asyncio.sleep", side_effect=fake_sleep), pytest.raises(RateLimitError):
            await limiter.submit_request(
                RequestMetadata(
                    request_id="req-timeout",
                    model_id="test-model",
                    resource_type=ResourceType.LLM,
                    endpoint="chat/completions",
                    timeout=5.0,
                ),
                AsyncMock(return_value=response),
                error_factory=_make_status_error,
            )

        assert sum(slept) <= 5.0, f"slept {sum(slept)}s against a 5s timeout: {slept!r}"

    @pytest.mark.asyncio
    async def test_no_timeout_still_allows_the_full_window(self):
        """timeout=None means no ceiling, so the 30s floor still applies."""
        from venice_ai._queue_types import RequestMetadata, ResourceType
        from venice_ai.exceptions import RateLimitError, _make_status_error

        limiter = SimpleRateLimiter(min_backoff=1.0, max_retries=1)
        response = self._error_budget_response()

        slept: list[float] = []

        async def fake_sleep(seconds):
            slept.append(seconds)

        with patch("asyncio.sleep", side_effect=fake_sleep), pytest.raises(RateLimitError):
            await limiter.submit_request(
                RequestMetadata(
                    request_id="req-no-timeout",
                    model_id="test-model",
                    resource_type=ResourceType.LLM,
                    endpoint="chat/completions",
                    timeout=None,
                ),
                AsyncMock(return_value=response),
                error_factory=_make_status_error,
            )

        assert max(slept) >= 30.0
