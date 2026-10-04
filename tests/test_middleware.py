"""
Tests for the custom retry middleware.

This module tests the retry middleware's behavior including:
- Exponential backoff with jitter
- Status code-based retries
- Exception-based retries
- Idempotency logic
- Retry-After header handling
"""

import asyncio
import time
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientResponse, ClientSession, web

from venice_ai.middleware.retry import (
    RetryOptions,
    calculate_backoff_delay,
    create_retry_middleware,
    parse_retry_after_header,
)


class TestBackoffCalculation:
    """Test the exponential backoff calculation logic."""

    def test_basic_exponential_backoff(self):
        """Test basic exponential backoff without jitter."""
        # Test exponential growth: 1, 2, 4, 8 seconds
        assert calculate_backoff_delay(0, 1.0, 2.0, 60.0, 0.0) == 1.0
        assert calculate_backoff_delay(1, 1.0, 2.0, 60.0, 0.0) == 2.0
        assert calculate_backoff_delay(2, 1.0, 2.0, 60.0, 0.0) == 4.0
        assert calculate_backoff_delay(3, 1.0, 2.0, 60.0, 0.0) == 8.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        # With base=2, attempt 10 would be 1024 seconds, but capped at 10
        assert calculate_backoff_delay(10, 1.0, 2.0, 10.0, 0.0) == 10.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to the delay."""
        # With jitter, the delay should vary
        delays = [calculate_backoff_delay(2, 1.0, 2.0, 60.0, 0.5) for _ in range(10)]
        # All delays should be different (extremely unlikely to have duplicates with jitter)
        assert len(set(delays)) > 1
        # All delays should be within expected range: 4.0 ± 50% (2.0 to 6.0)
        for delay in delays:
            assert 2.0 <= delay <= 6.0

    def test_zero_jitter(self):
        """Test that zero jitter produces consistent delays."""
        delay1 = calculate_backoff_delay(2, 1.0, 2.0, 60.0, 0.0)
        delay2 = calculate_backoff_delay(2, 1.0, 2.0, 60.0, 0.0)
        assert delay1 == delay2 == 4.0


class TestRetryAfterParsing:
    """Test Retry-After header parsing."""

    def test_parse_integer_seconds(self):
        """Test parsing integer seconds from Retry-After header."""
        response = Mock(spec=ClientResponse)
        response.headers = {"Retry-After": "120"}
        assert parse_retry_after_header(response) == 120.0

    def test_parse_float_seconds(self):
        """Test parsing float seconds from Retry-After header."""
        response = Mock(spec=ClientResponse)
        response.headers = {"Retry-After": "30.5"}
        assert parse_retry_after_header(response) == 30.5

    def test_missing_header(self):
        """Test handling of missing Retry-After header."""
        response = Mock(spec=ClientResponse)
        response.headers = {}
        assert parse_retry_after_header(response) is None

    def test_invalid_header(self):
        """Test handling of invalid Retry-After header."""
        response = Mock(spec=ClientResponse)
        response.headers = {"Retry-After": "invalid"}
        assert parse_retry_after_header(response) is None


@pytest.mark.asyncio
class TestRetryMiddleware:
    """Test the retry middleware integration."""

    async def test_successful_request_no_retry(self):
        """Test that successful requests are not retried."""
        options = RetryOptions(max_attempts=3)
        middleware = create_retry_middleware(options)

        # Mock successful response
        handler = AsyncMock()
        response = Mock(spec=ClientResponse)
        response.status = 200
        handler.return_value = response

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        result = await middleware(request, handler)

        # Should only call handler once for successful request
        assert handler.call_count == 1
        assert result == response

    async def test_retry_on_503_status(self):
        """Test retry on 503 Service Unavailable status."""
        options = RetryOptions(
            max_attempts=2,
            base_delay=0.1,  # Short delay for testing
            retry_status_codes={503},
        )
        middleware = create_retry_middleware(options)

        # Mock handler that returns 503 twice, then 200
        handler = AsyncMock()
        response_503 = Mock(spec=ClientResponse)
        response_503.status = 503
        response_503.headers = {}
        response_200 = Mock(spec=ClientResponse)
        response_200.status = 200

        handler.side_effect = [response_503, response_503, response_200]

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        start_time = time.time()
        result = await middleware(request, handler)
        elapsed_time = time.time() - start_time

        # Should retry twice before succeeding
        assert handler.call_count == 3
        assert result == response_200
        # Should have delays (at least 0.2 seconds for two retries with 0.1 base)
        assert elapsed_time >= 0.2

    async def test_max_attempts_exceeded(self):
        """Test that max attempts are respected."""
        options = RetryOptions(
            max_attempts=2,
            base_delay=0.01,  # Very short delay for testing
            retry_status_codes={500},
        )
        middleware = create_retry_middleware(options)

        # Mock handler that always returns 500
        handler = AsyncMock()
        response_500 = Mock(spec=ClientResponse)
        response_500.status = 500
        response_500.headers = {}
        handler.return_value = response_500

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        result = await middleware(request, handler)

        # Should try max_attempts + 1 times (initial + retries)
        assert handler.call_count == 3
        # Should return the last failed response
        assert result.status == 500

    async def test_idempotency_post_not_retried_by_default(self):
        """Test that POST requests are not retried by default."""
        options = RetryOptions(
            max_attempts=2,
            retry_non_idempotent=False,  # Default
            retry_status_codes={503},
        )
        middleware = create_retry_middleware(options)

        # Mock handler that returns 503
        handler = AsyncMock()
        response_503 = Mock(spec=ClientResponse)
        response_503.status = 503
        handler.return_value = response_503

        # Create mock POST request
        request = Mock()
        request.method = "POST"
        request.url = "http://example.com/api"

        # Execute middleware
        result = await middleware(request, handler)

        # Should not retry POST request
        assert handler.call_count == 1
        assert result.status == 503

    async def test_idempotency_post_retried_when_configured(self):
        """Test that POST requests are retried when configured."""
        options = RetryOptions(
            max_attempts=1,
            retry_non_idempotent=True,  # Allow POST retries
            retry_status_codes={503},
            base_delay=0.01,
        )
        middleware = create_retry_middleware(options)

        # Mock handler that returns 503 then 200
        handler = AsyncMock()
        response_503 = Mock(spec=ClientResponse)
        response_503.status = 503
        response_503.headers = {}
        response_200 = Mock(spec=ClientResponse)
        response_200.status = 200

        handler.side_effect = [response_503, response_200]

        # A POST to an endpoint whose documented 503 means "at capacity"
        request = Mock()
        request.method = "POST"
        request.url = "http://example.com/api/v1/image/generate"

        # Execute middleware
        result = await middleware(request, handler)

        # Should retry POST request when configured
        assert handler.call_count == 2
        assert result.status == 200

    async def test_retry_on_timeout_exception(self):
        """Test retry on timeout exceptions."""
        options = RetryOptions(
            max_attempts=2, base_delay=0.01, retry_exceptions=[asyncio.TimeoutError]
        )
        middleware = create_retry_middleware(options)

        # Mock handler that raises TimeoutError twice, then succeeds
        handler = AsyncMock()
        response_200 = Mock(spec=ClientResponse)
        response_200.status = 200

        handler.side_effect = [
            TimeoutError("Connection timeout"),
            TimeoutError("Connection timeout"),
            response_200,
        ]

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        result = await middleware(request, handler)

        # Should retry on timeout exceptions
        assert handler.call_count == 3
        assert result == response_200

    async def test_non_retryable_exception_raised(self):
        """Test that non-retryable exceptions are raised immediately."""
        options = RetryOptions(
            max_attempts=2,
            retry_exceptions=[asyncio.TimeoutError],  # Only retry timeouts
        )
        middleware = create_retry_middleware(options)

        # Mock handler that raises a different exception
        handler = AsyncMock()
        handler.side_effect = ValueError("Invalid value")

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        with pytest.raises(ValueError, match="Invalid value"):
            await middleware(request, handler)

        # Should not retry non-configured exceptions
        assert handler.call_count == 1

    async def test_respect_retry_after_header(self):
        """Test that Retry-After headers are respected."""
        options = RetryOptions(
            max_attempts=1,
            base_delay=1.0,  # Would normally wait 1 second
            respect_retry_after=True,
            max_retry_after=0.1,  # Cap at 0.1 seconds for testing
            retry_status_codes={503},
        )
        middleware = create_retry_middleware(options)

        # Mock handler that returns 503 with Retry-After, then 200
        handler = AsyncMock()
        response_503 = Mock(spec=ClientResponse)
        response_503.status = 503
        response_503.headers = {"Retry-After": "0.05"}  # Request 50ms delay
        response_200 = Mock(spec=ClientResponse)
        response_200.status = 200

        handler.side_effect = [response_503, response_200]

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        start_time = time.time()
        result = await middleware(request, handler)
        elapsed_time = time.time() - start_time

        # Should use Retry-After header value
        assert handler.call_count == 2
        assert result == response_200
        # Should wait approximately 0.05 seconds (Retry-After value)
        assert 0.04 <= elapsed_time <= 0.15  # Allow some tolerance

    async def test_on_retry_callback(self):
        """Test that the on_retry callback is called."""
        callback_calls = []

        def on_retry(attempt, delay, exception):
            callback_calls.append({"attempt": attempt, "delay": delay, "exception": exception})

        options = RetryOptions(
            max_attempts=2, base_delay=0.01, retry_status_codes={503}, on_retry=on_retry
        )
        middleware = create_retry_middleware(options)

        # Mock handler that returns 503 twice, then 200
        handler = AsyncMock()
        response_503 = Mock(spec=ClientResponse)
        response_503.status = 503
        response_503.headers = {}
        response_200 = Mock(spec=ClientResponse)
        response_200.status = 200

        handler.side_effect = [response_503, response_503, response_200]

        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url = "http://example.com/api"

        # Execute middleware
        await middleware(request, handler)

        # Callback should be called for each retry
        assert len(callback_calls) == 2
        assert callback_calls[0]["attempt"] == 0
        assert callback_calls[1]["attempt"] == 1
        # Exception should be None for status code retries
        assert callback_calls[0]["exception"] is None
        assert callback_calls[1]["exception"] is None

    async def test_different_idempotent_methods(self):
        """Test that various idempotent methods are retried."""
        idempotent_methods = ["GET", "PUT", "DELETE", "HEAD", "OPTIONS", "TRACE"]

        for method in idempotent_methods:
            options = RetryOptions(max_attempts=1, base_delay=0.01, retry_status_codes={503})
            middleware = create_retry_middleware(options)

            # Mock handler that returns 503 then 200
            handler = AsyncMock()
            response_503 = Mock(spec=ClientResponse)
            response_503.status = 503
            response_503.headers = {}
            response_200 = Mock(spec=ClientResponse)
            response_200.status = 200

            handler.side_effect = [response_503, response_200]

            # Create mock request with idempotent method
            request = Mock()
            request.method = method
            request.url = "http://example.com/api"

            # Execute middleware
            result = await middleware(request, handler)

            # Should retry idempotent methods
            assert handler.call_count == 2, f"Method {method} should be retried"
            assert result.status == 200

            # Reset for next iteration
            handler.reset_mock()


@pytest.mark.asyncio
class TestIntegrationWithAioHTTP:
    """Integration tests with real aiohttp server."""

    async def test_real_server_retry(self):
        """Test retry middleware with a real aiohttp test server."""
        # Track request count
        request_count = 0

        async def handler(request):
            nonlocal request_count
            request_count += 1

            # Return 503 for first two requests, then 200
            if request_count <= 2:
                return web.Response(status=503, text="Service Unavailable")
            else:
                return web.Response(status=200, text="Success")

        # Create test app
        app = web.Application()
        app.router.add_get("/test", handler)

        # Start test server
        from aiohttp.test_utils import TestServer

        server = TestServer(app)
        await server.start_server()

        try:
            # Create client with retry middleware
            retry_options = RetryOptions(max_attempts=3, base_delay=0.01, retry_status_codes={503})
            middleware = create_retry_middleware(retry_options)

            async with ClientSession() as session:
                # Wrap the session's request method
                original_request = session._request

                async def wrapped_request(method, url, **kwargs):
                    # Create a mock request object for the middleware
                    request = Mock()
                    request.method = method
                    request.url = url

                    # Create handler that calls original request
                    async def handler(req):
                        # Cast the result to StreamResponse for type compatibility
                        response = await original_request(method, url, **kwargs)
                        return response  # type: ignore[return-value]

                    # Apply middleware
                    result = await middleware(request, handler)  # type: ignore[arg-type]
                    # Cast back to ClientResponse for the test
                    return cast(ClientResponse, result)

                # Type ignore to suppress the assignment warning for test purposes
                session._request = wrapped_request  # type: ignore

                # Make request
                url = f"http://localhost:{server.port}/test"
                async with session.get(url) as response:
                    assert response.status == 200
                    text = await response.text()
                    assert text == "Success"

            # Should have made 3 requests total
            assert request_count == 3

        finally:
            await server.close()

    async def test_real_server_with_retry_after(self):
        """Test retry middleware respects Retry-After header from real server."""
        request_times = []

        async def handler(request):
            request_times.append(time.time())

            if len(request_times) == 1:
                # First request: return 503 with Retry-After
                return web.Response(status=503, headers={"Retry-After": "0.1"}, text="Unavailable")
            else:
                # Second request: success
                return web.Response(status=200, text="Success")

        # Create test app
        app = web.Application()
        app.router.add_get("/test", handler)

        # Start test server
        from aiohttp.test_utils import TestServer

        server = TestServer(app)
        await server.start_server()

        try:
            # Create client with retry middleware
            retry_options = RetryOptions(
                max_attempts=1,
                base_delay=1.0,  # Would normally wait 1 second
                respect_retry_after=True,
                retry_status_codes={503},
            )
            middleware = create_retry_middleware(retry_options)

            async with ClientSession() as session:
                # Wrap the session's request method
                original_request = session._request

                async def wrapped_request(method, url, **kwargs):
                    request = Mock()
                    request.method = method
                    request.url = url

                    async def handler(req):
                        # Cast the result to StreamResponse for type compatibility
                        response = await original_request(method, url, **kwargs)
                        return response  # type: ignore[return-value]

                    # Apply middleware
                    result = await middleware(request, handler)  # type: ignore[arg-type]
                    # Cast back to ClientResponse for the test
                    return cast(ClientResponse, result)

                # Type ignore to suppress the assignment warning for test purposes
                session._request = wrapped_request  # type: ignore

                # Make request
                url = f"http://localhost:{server.port}/test"
                async with session.get(url) as response:
                    assert response.status == 200

            # Check that retry respected the Retry-After header
            assert len(request_times) == 2
            delay = request_times[1] - request_times[0]
            # Should be approximately 0.1 seconds (Retry-After value)
            assert 0.08 <= delay <= 0.20  # Allow more tolerance for system timing variations

        finally:
            await server.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
