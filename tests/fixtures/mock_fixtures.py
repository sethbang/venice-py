"""
Mock object fixtures for Venice AI testing.

This module provides fixtures for creating mock objects and utilities
for testing various components in isolation.
"""

import asyncio
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


def with_request_plumbing(client: Any) -> Any:
    """Give a mocked client the request-header plumbing resources call.

    ``_request_headers`` merges the session defaults with the per-call headers
    (no credentials: a mocked client has none to add), and ``_siwe_resigning``
    is a no-op scope. Returns ``client`` for chaining.
    """
    from contextlib import nullcontext

    def request_headers(session: Any, headers: Any = None, siwe_auth: Any = None) -> dict:
        return {**dict(session.headers), **dict(headers or {})}

    client._request_headers = Mock(side_effect=request_headers)
    client._siwe_resigning = Mock(side_effect=lambda *a, **k: nullcontext())
    return client


def fake_event_loop() -> MagicMock:
    """
    A stand-in for the running event loop, for tests that patch
    ``asyncio.get_running_loop``.

    A patched ``get_running_loop`` is global, so code outside the test also
    receives this loop, such as a weakref callback fired by garbage collection
    of an earlier test's loop. A plain ``MagicMock`` would drop the coroutine
    passed to ``create_task`` and leave a "never awaited" warning on whichever
    test happens to be running. This one closes it instead, as a loop that
    never runs tasks should.
    """
    loop = MagicMock()
    loop.create_task.side_effect = lambda coro, **_: coro.close()
    return loop


@pytest.fixture
def mock_session():
    """
    Create a mock aiohttp ClientSession.

    Returns:
        Mock ClientSession object
    """
    session = AsyncMock()

    # Mock HTTP methods
    session.post = AsyncMock()
    session.get = AsyncMock()
    session.put = AsyncMock()
    session.delete = AsyncMock()
    session.patch = AsyncMock()
    session.head = AsyncMock()

    # Mock session lifecycle
    session.close = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)

    # Mock properties
    session.closed = False
    session.timeout = Mock()
    session.headers = {}

    return session


@pytest.fixture
def mock_scheduler():
    """
    Create a mock scheduler for rate limiting tests.

    Returns:
        Mock scheduler object
    """
    scheduler = Mock()

    # Mock scheduler methods
    scheduler.schedule = AsyncMock()
    scheduler.execute = AsyncMock()
    scheduler.cancel = AsyncMock()
    scheduler.get_stats = Mock(
        return_value={"queued": 0, "running": 0, "completed": 0, "failed": 0}
    )
    scheduler.is_ready = Mock(return_value=True)
    scheduler.wait_for_slot = AsyncMock()

    # Mock scheduler properties
    scheduler.max_concurrent = 10
    scheduler.queue_size = 100
    scheduler.is_running = True

    return scheduler


@pytest.fixture
def mock_backend():
    """
    Create a mock state backend for testing.

    Returns:
        Mock backend object
    """
    backend = Mock()

    # Mock backend methods
    backend.get = AsyncMock(return_value=None)
    backend.set = AsyncMock()
    backend.delete = AsyncMock()
    backend.exists = AsyncMock(return_value=False)
    backend.increment = AsyncMock(return_value=1)
    backend.decrement = AsyncMock(return_value=0)
    backend.get_all = AsyncMock(return_value={})
    backend.clear = AsyncMock()

    # Mock backend properties
    backend.is_connected = True
    backend.backend_type = "mock"

    return backend


@pytest.fixture
def mock_rate_limiter():
    """
    Create a mock rate limiter.

    Returns:
        Mock rate limiter object
    """
    rate_limiter = Mock()

    # Mock rate limiter methods
    rate_limiter.check_limit = AsyncMock(return_value=True)
    rate_limiter.consume = AsyncMock()
    rate_limiter.wait_if_needed = AsyncMock()
    rate_limiter.get_remaining = Mock(return_value=100)
    rate_limiter.reset = AsyncMock()

    # Mock rate limiter properties
    rate_limiter.rpm_limit = 60
    rate_limiter.tpm_limit = 10000
    rate_limiter.rpd_limit = 1000
    rate_limiter.is_enabled = True

    return rate_limiter


@pytest.fixture
def mock_retry_handler():
    """
    Create a mock retry handler.

    Returns:
        Mock retry handler object
    """
    retry_handler = Mock()

    # Mock methods
    retry_handler.should_retry = Mock(return_value=True)
    retry_handler.get_delay = Mock(return_value=1.0)
    retry_handler.on_retry = Mock()
    retry_handler.on_success = Mock()
    retry_handler.on_failure = Mock()

    # Mock properties
    retry_handler.max_retries = 3
    retry_handler.attempt = 0
    retry_handler.base_delay = 1.0
    retry_handler.max_delay = 60.0

    return retry_handler


@pytest.fixture
def mock_metrics_collector():
    """
    Create a mock metrics collector compatible with the slim MetricsCollector.

    Returns:
        Mock metrics collector object
    """
    collector = Mock()

    # Validation recording methods
    collector.record_validation_success = Mock()
    collector.record_validation_failure = Mock()

    # Lifecycle
    collector.reset = Mock()

    # Properties
    collector.enable_prometheus = False
    collector._registry = None

    return collector


@pytest.fixture
def response_mocker():
    """
    Create a response mocker for simulating various API responses.

    Returns:
        ResponseMocker instance
    """

    class ResponseMocker:
        def __init__(self):
            self.responses = []
            self.call_count = 0

        def add_response(
            self,
            status: int = 200,
            json_data: dict | None = None,
            text: str | None = None,
            headers: dict | None = None,
            delay: float = 0,
        ):
            """Add a response to the queue."""
            self.responses.append(
                {
                    "status": status,
                    "json": json_data,
                    "text": text,
                    "headers": headers or {},
                    "delay": delay,
                }
            )

        async def get_response(self):
            """Get the next response from the queue."""
            if self.call_count >= len(self.responses):
                raise ValueError("No more responses available")

            response_data = self.responses[self.call_count]
            self.call_count += 1

            # Simulate delay if specified
            if response_data["delay"] > 0:
                await asyncio.sleep(response_data["delay"])

            # Create mock response
            response = Mock()
            response.status = response_data["status"]
            response.headers = response_data["headers"]

            if response_data["json"] is not None:
                response.json = AsyncMock(return_value=response_data["json"])
            elif response_data["text"] is not None:
                response.text = AsyncMock(return_value=response_data["text"])
            else:
                response.json = AsyncMock(return_value={})

            return response

        def reset(self):
            """Reset the mocker."""
            self.call_count = 0
            self.responses.clear()

    return ResponseMocker()


@pytest.fixture
def patch_time():
    """
    Create a time patcher for controlling time in tests.

    Returns:
        TimePatcher instance
    """

    class TimePatcher:
        def __init__(self):
            self.current_time = 0
            self.patcher = None
            self.mock_time = None

        def start(self, initial_time: float = 0):
            """Start patching time."""
            self.current_time = initial_time
            self.patcher = patch("time.time", return_value=self.current_time)
            self.mock_time = self.patcher.start()
            return self.mock_time

        def advance(self, seconds: float):
            """Advance time by specified seconds."""
            self.current_time += seconds
            if self.mock_time:
                self.mock_time.return_value = self.current_time

        def stop(self):
            """Stop patching time."""
            if self.patcher:
                self.patcher.stop()

    patcher = TimePatcher()
    yield patcher
    patcher.stop()


@pytest.fixture
def async_mock_factory():
    """
    Factory for creating async mocks with custom behavior.

    Returns:
        Function to create async mocks
    """

    def create_async_mock(
        return_value: Any = None,
        side_effect: list[Any] | None = None,
        raises: Exception | None = None,
    ) -> AsyncMock:
        """
        Create an async mock with specified behavior.

        Args:
            return_value: Value to return
            side_effect: List of values to return in sequence
            raises: Exception to raise

        Returns:
            Configured AsyncMock
        """
        mock = AsyncMock()

        if raises:
            mock.side_effect = raises
        elif side_effect:
            mock.side_effect = side_effect
        else:
            mock.return_value = return_value

        return mock

    return create_async_mock


@pytest.fixture
def mock_state_manager():
    """
    Create a mock state manager for testing.

    Returns:
        Mock state manager object
    """
    state_manager = Mock()

    # Mock methods
    state_manager.get_state = AsyncMock(return_value={})
    state_manager.set_state = AsyncMock()
    state_manager.update_state = AsyncMock()
    state_manager.delete_state = AsyncMock()
    state_manager.lock = AsyncMock()
    state_manager.unlock = AsyncMock()
    state_manager.transaction = Mock()

    # Mock properties
    state_manager.is_initialized = True
    state_manager.backend = "mock"

    return state_manager


@pytest.fixture
def exception_raiser():
    """
    Create an exception raiser for testing error handling.

    Returns:
        Function to create exception-raising mocks
    """

    def create_raiser(
        exception_type: type = Exception,
        message: str = "Test error",
        after_n_calls: int = 0,
    ) -> Mock:
        """
        Create a mock that raises an exception.

        Args:
            exception_type: Type of exception to raise
            message: Exception message
            after_n_calls: Raise exception after N successful calls

        Returns:
            Mock that raises exception
        """
        call_count = [0]  # Use list to maintain state in closure

        def side_effect(*args, **kwargs):
            if call_count[0] < after_n_calls:
                call_count[0] += 1
                return Mock()  # Return mock for successful calls
            raise exception_type(message)

        mock = Mock(side_effect=side_effect)
        mock.reset_count = lambda: call_count.__setitem__(0, 0)

        return mock

    return create_raiser


@pytest.fixture
def mock_validator():
    """
    Create a mock validator for request validation testing.

    Returns:
        Mock validator object
    """
    validator = Mock()

    # Mock methods
    validator.validate = Mock(return_value=True)
    validator.validate_model = Mock()
    validator.validate_messages = Mock()
    validator.validate_parameters = Mock()
    validator.get_errors = Mock(return_value=[])

    # Mock properties
    validator.strict_mode = False
    validator.allow_extra = True

    return validator


@pytest.fixture
def context_manager_mock():
    """
    Create a mock that can be used as a context manager.

    Returns:
        Function to create context manager mocks
    """

    @contextmanager
    def create_context_mock(enter_value: Any = None, exit_exception: Exception | None = None):
        """
        Create a context manager mock.

        Args:
            enter_value: Value to return from __enter__
            exit_exception: Exception to raise in __exit__

        Yields:
            Mock object
        """
        mock = Mock()
        mock.__enter__ = Mock(return_value=enter_value or mock)
        mock.__exit__ = Mock(side_effect=exit_exception)

        yield mock

        # Ensure __exit__ is called even if test fails
        if not mock.__exit__.called:
            mock.__exit__(None, None, None)

    return create_context_mock
