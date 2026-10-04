"""
Shared fixtures for Venice AI testing.

This module provides reusable fixtures for testing Venice AI components,
including mock clients, responses, and common test utilities.
"""

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest


@pytest.fixture
def mock_rate_limiter():
    """Create a mock rate limiter."""
    limiter = AsyncMock()
    limiter.check_capacity = AsyncMock(return_value=(True, 0.0))
    limiter.consume_capacity = AsyncMock()
    limiter.start = AsyncMock()
    limiter.stop = AsyncMock()
    limiter.wait_if_needed = AsyncMock()
    return limiter


@pytest.fixture
def mock_http_response():
    """Create configurable mock HTTP response."""

    def _create_response(
        status: int = 200,
        json_data: dict[str, Any] | None = None,
        text: str = "",
        headers: dict[str, str] | None = None,
        content: bytes | None = None,
    ):
        response = AsyncMock()
        response.status = status
        response.ok = status < 400
        response.json = AsyncMock(return_value=json_data or {})
        response.text = AsyncMock(return_value=text)
        response.headers = headers or {}
        response.content = AsyncMock()
        response.raise_for_status = Mock()

        if content:
            response.read = AsyncMock(return_value=content)

        if status >= 400:
            from aiohttp import ClientResponseError

            response.raise_for_status.side_effect = ClientResponseError(
                request_info=Mock(),
                history=(),
                status=status,
                message=f"HTTP {status}",
                headers=response.headers,
            )

        return response

    return _create_response


@pytest.fixture
def mock_streaming_response():
    """Create mock streaming response."""

    def _create_stream(*chunks):
        response = AsyncMock()
        response.ok = True
        response.status = 200
        response.raise_for_status = Mock()

        async def content_generator():
            for chunk in chunks:
                yield chunk.encode() if isinstance(chunk, str) else chunk

        response.content = content_generator()
        return response

    return _create_stream


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.headers = {}
    session.timeout = aiohttp.ClientTimeout(total=30)
    session.closed = False

    # Create a custom mock that inherits from Mock to get assertion methods
    class MockRequestMethod(Mock):
        def __init__(self):
            super().__init__()
            self._default_return_value = None

        def __call__(self, *args, **kwargs):
            # Handle side_effect first (for exceptions)
            if self.side_effect is not None:
                if isinstance(self.side_effect, Exception):
                    raise self.side_effect
                elif callable(self.side_effect):
                    result = self.side_effect(*args, **kwargs)
                    if isinstance(result, Exception):
                        raise result
                    return result
                elif isinstance(self.side_effect, (list, tuple)):
                    call_count = len(self.call_args_list)
                    if call_count < len(self.side_effect):
                        effect = self.side_effect[call_count]
                        if isinstance(effect, Exception):
                            raise effect
                        return effect

            # Call parent Mock's __call__ to handle standard mock behavior and call tracking
            result = super().__call__(*args, **kwargs)

            # If return_value is set by test, use it, otherwise use default
            if self.return_value is not Mock() and self.return_value is not None:
                # If it's an AsyncMock (from mock_http_response), wrap it to be awaitable
                if hasattr(self.return_value, "_mock_name") and "AsyncMock" in str(
                    type(self.return_value)
                ):
                    return AwaitableWrapper(self.return_value)
                return self.return_value
            elif self._default_return_value is not None:
                return self._default_return_value

            # Return a dual-purpose mock that can be both awaited and used as context manager
            return MockResponse()

    class AwaitableWrapper:
        """Wrapper to make AsyncMock objects awaitable for session.request"""

        def __init__(self, wrapped_mock):
            self.wrapped_mock = wrapped_mock
            # Copy important attributes
            for attr in [
                "status",
                "ok",
                "headers",
                "json",
                "text",
                "read",
                "raise_for_status",
            ]:
                if hasattr(wrapped_mock, attr):
                    setattr(self, attr, getattr(wrapped_mock, attr))

        def __await__(self):
            """Make this awaitable - returns the wrapped mock when awaited"""

            async def _await():
                return self.wrapped_mock

            return _await().__await__()

        async def __aenter__(self):
            """For async with usage"""
            return self.wrapped_mock

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """For async with usage"""
            return False

        def __getattr__(self, name):
            """Delegate other attributes to the wrapped mock"""
            return getattr(self.wrapped_mock, name)

    class MockResponse:
        """Mock that can be both awaited and used as async context manager"""

        def __init__(self, json_data=None):
            self.status = 200
            self.ok = True
            self.headers = {"content-type": "application/json"}
            self.json = AsyncMock(return_value=json_data or {"test": "data"})
            self.text = AsyncMock(return_value="OK")
            self.read = AsyncMock(return_value=b"OK")
            self.raise_for_status = Mock()  # Non-async method

        def __await__(self):
            """Make this awaitable - returns self when awaited"""

            async def _await():
                return self

            return _await().__await__()

        async def __aenter__(self):
            """For async with usage"""
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            """For async with usage"""
            return False

    # Create instances for each HTTP method
    session.request = MockRequestMethod()
    session.post = MockRequestMethod()
    session.get = MockRequestMethod()
    session.delete = MockRequestMethod()
    session.put = MockRequestMethod()
    session.close = AsyncMock()
    return session


@pytest.fixture
async def venice_client_factory():
    """Factory for creating configured Venice clients."""
    clients = []

    def _create_client(**kwargs):
        from venice_ai._client import VeniceClient

        defaults = {
            "api_key": "test-key",
            "base_url": "https://api.test.com",
            "timeout": 30.0,
            "max_retries": 3,
        }
        defaults.update(kwargs)

        client = VeniceClient(**defaults)
        clients.append(client)
        return client

    yield _create_client

    # Cleanup
    for client in clients:
        if hasattr(client, "close") and not client.closed:
            await client.close()


@pytest.fixture
def mock_venice_client():
    """Create a mock Venice client for testing."""
    client = AsyncMock()
    client._api_key = "test-api-key"
    client._base_url = Mock()
    client._base_url.__truediv__ = Mock(return_value="http://test.com/path")
    client._base_url.path = "/"
    client._base_url.with_path = Mock(return_value=Mock())
    client._get_session = AsyncMock()
    client.closed = False
    client._timeout = 30.0
    return client


@pytest.fixture
def mock_venice_account():
    """Create a mock Venice account."""
    account = Mock()
    account.account_id = "test-account"
    account.can_make_request = AsyncMock(return_value=True)
    account.record_request_success = AsyncMock()
    account.record_request_failure = AsyncMock()
    account.health_check = AsyncMock(return_value={"status": "healthy"})
    account.get_metrics = Mock(return_value={})
    account.backend = Mock()
    account.failure_tracker = Mock()
    account._closed = False
    return account


@pytest.fixture
def error_samples():
    """Sample error responses for testing."""
    return {
        "rate_limit": {
            "status": 429,
            "body": {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
            "headers": {
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1234567890",
            },
        },
        "timeout": {"exception": TimeoutError("Request timed out")},
        "connection": {
            "exception": aiohttp.ClientConnectorError(
                connection_key=Mock(), os_error=OSError("Connection refused")
            )
        },
        "server_error": {
            "status": 500,
            "body": {"error": {"message": "Internal server error"}},
        },
        "validation": {
            "status": 400,
            "body": {
                "error": {
                    "message": "Invalid request",
                    "type": "validation_error",
                    "param": "model",
                }
            },
        },
    }


@pytest.fixture
def sample_file_data():
    """Provide sample file data for upload testing."""
    return {
        "text_file": {
            "filename": "test.txt",
            "content": b"This is a test text file content.",
            "content_type": "text/plain",
        },
        "json_file": {
            "filename": "data.json",
            "content": b'{"key": "value", "number": 42}',
            "content_type": "application/json",
        },
        "image_file": {
            "filename": "test.png",
            "content": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde",
            "content_type": "image/png",
        },
        "audio_file": {
            "filename": "test.wav",
            "content": b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00",
            "content_type": "audio/wav",
        },
    }


@pytest.fixture
def mock_form_data():
    """Create a mock aiohttp FormData object."""
    form_data = Mock(spec=aiohttp.FormData)
    form_data.add_field = Mock()
    form_data._fields = []
    return form_data


@pytest.fixture
def context_manager_mock():
    """Create a mock that can be used as a context manager."""

    @asynccontextmanager
    async def create_context_mock(enter_value: Any = None, exit_exception: Exception | None = None):
        """
        Create a context manager mock.

        Args:
            enter_value: Value to return from __aenter__
            exit_exception: Exception to raise in __aexit__

        Yields:
            Mock object
        """
        mock = Mock()
        mock.__aenter__ = AsyncMock(return_value=enter_value or mock)
        mock.__aexit__ = AsyncMock(side_effect=exit_exception)

        yield mock

        # Ensure __aexit__ is called even if test fails
        if not mock.__aexit__.called:
            await mock.__aexit__(None, None, None)

    return create_context_mock


@pytest.fixture
def mock_backend():
    """Create a mock state backend for testing."""
    backend = AsyncMock()
    backend.get = AsyncMock(return_value=None)
    backend.set = AsyncMock()
    backend.delete = AsyncMock()

    # Add failure tracker specific methods
    backend.record_failure = AsyncMock()
    backend.get_failure_count = AsyncMock(return_value=0)  # Return integer, not AsyncMock
    backend.clear_failures = AsyncMock()
    backend.force_circuit_break = AsyncMock()
    backend.is_circuit_broken = AsyncMock(return_value=False)  # Return boolean, not AsyncMock
    backend.exists = AsyncMock(return_value=False)
    backend.increment = AsyncMock(return_value=1)
    backend.decrement = AsyncMock(return_value=0)
    backend.get_all = AsyncMock(return_value={})
    backend.clear = AsyncMock()

    # Add account-specific backend methods
    backend.health_check = AsyncMock(return_value={"healthy": True, "status": "ok"})
    backend.check_capacity = AsyncMock(return_value=(True, 0.0))
    backend.update_rate_limits = AsyncMock()
    backend.record_request = AsyncMock()
    backend.cleanup = AsyncMock()
    backend.get_all_stats = AsyncMock(return_value={"requests_served": 100, "cache_hits": 85})
    backend.health_check = AsyncMock(return_value={"healthy": True})
    backend.record_failure = AsyncMock()
    backend.record_request = AsyncMock()
    backend.get_state = AsyncMock(return_value={})
    backend.update_state = AsyncMock()
    backend.is_connected = True
    backend.backend_type = "mock"
    return backend


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    state_manager = AsyncMock()
    state_manager.get_state = AsyncMock(return_value={})
    state_manager.set_state = AsyncMock()
    state_manager.update_state = AsyncMock()
    state_manager.delete_state = AsyncMock()
    state_manager.lock = AsyncMock()
    state_manager.unlock = AsyncMock()
    state_manager.transaction = Mock()
    state_manager.is_initialized = True
    state_manager.backend = "mock"
    return state_manager


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler for testing."""
    scheduler = AsyncMock()
    scheduler.schedule = AsyncMock()
    scheduler.execute = AsyncMock()
    scheduler.cancel = AsyncMock()
    scheduler.get_stats = Mock(
        return_value={"queued": 0, "running": 0, "completed": 0, "failed": 0}
    )
    scheduler.is_ready = Mock(return_value=True)
    scheduler.wait_for_slot = AsyncMock()
    scheduler.max_concurrent = 10
    scheduler.queue_size = 100
    scheduler.is_running = True
    return scheduler
