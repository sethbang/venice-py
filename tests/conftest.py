"""
Root conftest.py for Venice AI test suite.

This module provides shared fixtures, configuration, and utilities for all test types
(unit, integration, and e2e). It establishes the foundation for the test infrastructure.
"""

import asyncio
import contextlib
import inspect
import logging
import os
import sys
import tempfile
import threading
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import vcr
from aiohttp import ClientSession
from vcr.record_mode import RecordMode

from tests.vcr_policy import cassette_dir_for, resolve_record_mode

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import fixtures from fixture modules to make them globally available
# Note: Fixtures are defined directly in this file and venice_fixtures
# Import the venice_fixtures module to make its fixtures available
from tests.fixtures import venice_fixtures  # noqa: F401
from venice_ai import VeniceClient
from venice_ai.core.config import SchedulerMode, VeniceAIConfig
from venice_ai.test_support.mock_utilities import (
    create_mock_api_error,
    create_mock_chat_response,
)

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG_TESTS") else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Redis Cluster mode detection for test fixtures
REDIS_CLUSTER_MODE = os.getenv("REDIS_CLUSTER_MODE", "").lower() == "true"

# ============================================================================
# Session-level Fixtures
# ============================================================================


# Module-scoped event loop fixture for parallel test execution
@pytest.fixture(scope="module")
def event_loop():
    """
    Create event loop for module-scoped fixtures.

    This is required for pytest-asyncio when using module-scoped async fixtures
    in parallel execution. Each test module gets its own event loop.
    """
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


def generate_valid_test_api_key(suffix: str = "default") -> str:
    """
    Generate a valid test API key meeting v2.0.0 requirements.

    Requirements:
    - Minimum 32 characters
    - Valid charset (alphanumeric + base64/hex chars: _, -, +, =, /)
    - No test patterns (test, fake, demo, example, xxx, sample, placeholder)

    Args:
        suffix: Optional suffix to differentiate keys (alphanumeric only)

    Returns:
        A valid test API key string
    """
    import secrets

    # Banned patterns that auth.py rejects (case-insensitive)
    _banned = ("test", "fake", "demo", "example", "xxx", "sample", "placeholder")

    # Regenerate until the random part is free of banned substrings
    while True:
        random_part = secrets.token_urlsafe(24)  # ~32 chars base64
        candidate = f"vn_{random_part}_{suffix}"
        if not any(p in candidate.lower() for p in _banned):
            return candidate


@pytest.fixture(scope="session")
def test_api_key() -> str:
    """Get test API key from environment or use a mock key."""
    return os.getenv("VENICE_TEST_API_KEY", generate_valid_test_api_key("session"))


@pytest.fixture(scope="session")
def test_base_url() -> str:
    """Get test base URL from environment or use default."""
    return os.getenv("VENICE_TEST_BASE_URL", "https://api.test.venice.ai/api/v1")


@pytest.fixture(scope="session")
def test_models() -> dict[str, str]:
    """Available test models mapping (uses dynamic cache when available)."""
    from tests.fixtures.test_models import TEST_MODELS

    return {
        "chat": os.getenv("VENICE_TEST_CHAT_MODEL", TEST_MODELS.SMALL_TEXT_MODEL),
        "vision": os.getenv("VENICE_TEST_VISION_MODEL", TEST_MODELS.VISION_MODEL),
        "embedding": os.getenv("VENICE_TEST_EMBEDDING_MODEL", TEST_MODELS.EMBEDDING_MODEL),
        "tts": os.getenv("VENICE_TEST_TTS_MODEL", TEST_MODELS.TTS_MODEL),
        "image": os.getenv("VENICE_TEST_IMAGE_MODEL", TEST_MODELS.DEFAULT_IMAGE_MODEL),
    }


# ============================================================================
# Function-level Fixtures
# ============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env(monkeypatch) -> dict[str, str]:
    """Mock environment variables for testing."""
    env_vars = {
        "VENICE_API_KEY": generate_valid_test_api_key("mock_env"),
        "VENICE_BASE_URL": "https://api.test.venice.ai/api/v1",
        "VENICE_TEST_MODE": "true",
        "VENICE_TEST_RATE_MULTIPLIER": "10.0",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


# ============================================================================
# Venice Client Fixtures
# ============================================================================


@pytest.fixture
def test_config() -> VeniceAIConfig:
    """Create a test configuration for Venice client using unified config system."""
    return VeniceAIConfig.create_test_config(
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
        test_rate_multiplier=10.0,
    )


@pytest.fixture
def legacy_test_config() -> dict[str, Any]:
    """Legacy test config for backwards compatibility during transition."""
    return {
        "api_key": "test-api-key-123",
        "base_url": "https://api.test.venice.ai/api/v1",
        "max_retries": 2,
        "timeout": 30.0,
        "test_mode": {
            "enabled": True,
            "rate_multiplier": 10.0,
            "use_mock_responses": False,
        },
    }


@pytest.fixture
def mock_client(test_config) -> Generator[VeniceClient]:
    """Create a mock Venice client for unit testing."""
    client = VeniceClient(api_key=test_config["api_key"], base_url=test_config["base_url"])

    # Mock the underlying HTTP session
    mock_session = MagicMock(spec=ClientSession)
    mock_session.post = AsyncMock()
    mock_session.get = AsyncMock()
    mock_session.delete = AsyncMock()
    mock_session.close = AsyncMock()

    # Patch the client's session
    with patch.object(client, "_session", mock_session):
        yield client


@pytest_asyncio.fixture
async def test_client(test_config) -> AsyncGenerator[VeniceClient]:
    """Create a test Venice client with shared backend for integration testing."""
    client = VeniceClient(api_key=test_config["api_key"], base_url=test_config["base_url"])
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def isolated_client(test_api_key) -> AsyncGenerator[VeniceClient]:
    """Create an isolated Venice client for independent testing."""
    client = VeniceClient(api_key=test_api_key)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def mock_venice_client():
    """Create a mock Venice client for APIResource testing."""
    mock_client = AsyncMock()
    mock_client._api_key = "test-api-key-123"

    # Mock the base URL with __truediv__ support for URL building
    mock_base_url = MagicMock()
    mock_base_url.__truediv__ = MagicMock(
        return_value="https://api.test.venice.ai/api/v1/test-path"
    )
    mock_client._base_url = mock_base_url

    # Mock the _get_session method to return a mock aiohttp session
    mock_session = AsyncMock()
    mock_session.request = AsyncMock()
    mock_session.headers = {}
    mock_client._get_session = AsyncMock(return_value=mock_session)

    return mock_client


@pytest.fixture
def mock_aiohttp_session():
    """Create a mock aiohttp ClientSession."""
    session = AsyncMock(spec=ClientSession)
    session.request = AsyncMock()
    session.headers = {}
    session.timeout = None
    return session


@pytest.fixture
def mock_http_response():
    """Create a mock HTTP response factory."""

    def _create_response(status=200, json_data=None, headers=None):
        response = AsyncMock()
        response.status = status
        response.headers = headers or {}
        response.raise_for_status = MagicMock()
        if json_data is not None:
            response.json = AsyncMock(return_value=json_data)
        return response

    return _create_response


@pytest.fixture
def sample_file_data():
    """Sample file data for upload testing."""
    return {
        "text_file": {
            "filename": "test.txt",
            "content": b"This is test file content",
            "content_type": "text/plain",
        },
        "image_file": {
            "filename": "test.png",
            "content": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde",
            "content_type": "image/png",
        },
        "binary_file": {
            "filename": "test.bin",
            "content": b"\x00\x01\x02\x03\x04\x05",
            "content_type": "application/octet-stream",
        },
    }


# ============================================================================
# Model Selection Fixtures
# ============================================================================


@pytest_asyncio.fixture
async def model_selector(venice_client):
    """
    Shared model selector with random cheap strategy for load distribution.

    It uses the random_cheap_strategy to optimize for cost while distributing
    load across multiple models.

    The random_cheap_strategy:
    - Prioritizes models in the cheapest pricing tier (within 5x of minimum price)
    - Randomly selects from the cheap pool to distribute load and avoid rate limits
    - Falls back to unpriced models if no pricing data is available

    Args:
        venice_client: The Venice client fixture (from venice_fixtures module)

    Returns:
        DynamicModelSelector: Configured with random_cheap_strategy as default
    """
    from venice_ai.models.selection import DynamicModelSelector
    from venice_ai.test_support.strategies import random_cheap_strategy

    return DynamicModelSelector(venice_client, default_selector=random_cheap_strategy)


# ============================================================================
# Mock Response Fixtures
# ============================================================================


@pytest.fixture
def mock_chat_response():
    """Create a mock chat completion response."""
    return create_mock_chat_response(
        response_id="test-chatcmpl-123",
        model="llama-3.2-3b",
        content="This is a test response.",
    )


@pytest.fixture
def mock_error_response():
    """Create a mock error response."""
    return create_mock_api_error(status_code=400, message="Invalid request")


@pytest.fixture
def mock_responses():
    """Collection of various mock responses for testing."""
    return {
        "chat": create_mock_chat_response(),
        "error_400": create_mock_api_error(400, "Bad Request"),
        "error_429": create_mock_api_error(429, "Rate limit exceeded"),
        "error_500": create_mock_api_error(500, "Internal server error"),
        "error_503": create_mock_api_error(503, "Service unavailable"),
    }


# ============================================================================
# Request Fixtures
# ============================================================================


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"},
    ]


@pytest.fixture
def sample_chat_request(sample_chat_messages, test_models):
    """Sample chat completion request."""
    return {
        "model": test_models["chat"],
        "messages": sample_chat_messages,
        "temperature": 0.7,
        "max_completion_tokens": 150,
        "stream": False,
    }


@pytest.fixture
def sample_image_request(test_models):
    """Sample image generation request."""
    return {
        "model": test_models["image"],
        "prompt": "A beautiful sunset over mountains",
        "width": 512,
        "height": 512,
        "steps": 20,
    }


@pytest.fixture
def sample_audio_request(test_models):
    """Sample audio/TTS request."""
    return {
        "model": test_models["tts"],
        "input": "Hello, this is a test.",
        "voice": "alloy",
        "speed": 1.0,
    }


@pytest.fixture
def sample_embedding_request(test_models):
    """Sample embedding request."""
    return {
        "model": test_models["embedding"],
        "input": ["This is a test sentence.", "Another test sentence."],
        "encoding_format": "float",
    }


# ============================================================================
# Test Utilities
# ============================================================================


@pytest.fixture
def assert_valid_response():
    """Fixture providing response validation assertions."""

    def _assert_valid_chat_response(response):
        """Assert that a chat response is valid."""
        assert response is not None
        assert hasattr(response, "id")
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], "message")
        assert hasattr(response.choices[0].message, "content")

    def _assert_valid_error_response(error):
        """Assert that an error response is valid."""
        assert error is not None
        assert hasattr(error, "message")
        assert hasattr(error, "response")

    return {"chat": _assert_valid_chat_response, "error": _assert_valid_error_response}


@pytest.fixture
def capture_metrics():
    """Fixture for capturing and asserting on metrics."""

    class MetricsCapture:
        def __init__(self):
            self.metrics = []

        def capture(self, metric_name, value, labels=None):
            self.metrics.append({"name": metric_name, "value": value, "labels": labels or {}})

        def assert_metric_exists(self, metric_name):
            assert any(m["name"] == metric_name for m in self.metrics)

        def get_metric(self, metric_name):
            return [m for m in self.metrics if m["name"] == metric_name]

    return MetricsCapture()


# ============================================================================
# Backend and State Management Fixtures
# ============================================================================


# Track Redis connections for resource monitoring
_redis_connection_count = 0
_redis_connection_lock = threading.Lock()


@pytest.fixture(scope="session")
def shared_rate_limit_namespace():
    """
    Shared namespace for rate limit coordination across all parallel workers.

    Returns:
        str: Shared namespace identifier for rate limit coordination
    """
    import hashlib

    api_key = os.getenv("VENICE_API_KEY", "test-default-key")
    key_hash = hashlib.md5(api_key.encode()).hexdigest()[:8]
    shared_namespace = f"test_shared_rate_limits_{key_hash}"

    logger.info(f"Using shared rate limit namespace: {shared_namespace}")
    return shared_namespace


@pytest.fixture(scope="session")
def shared_rate_limit_backend(shared_rate_limit_namespace):
    """
    Shared Redis backend for coordinating rate limits across all parallel workers.

    Returns:
        RedisBackend: Shared backend instance for rate limit coordination
    """
    from venice_ai.core.backends.redis import RedisBackend

    # We don't create a specific event loop here. RedisBackend handles multiple loops.
    # When tests run, they will use their own loops (managed by pytest-asyncio),
    # and RedisBackend will create connection pools for those loops on demand.

    backend = RedisBackend(
        redis_url=os.getenv("VENICE_TEST_REDIS_URL", "redis://localhost:6379"),
        namespace=shared_rate_limit_namespace,
        key_ttl=300,
        cluster_mode=REDIS_CLUSTER_MODE,
    )

    logger.info("Shared rate limit backend created for worker coordination")

    yield backend

    # No explicit cleanup needed here.
    # - Connection pools are cleaned up by auto_cleanup_connections fixture after each test.
    # - Any remaining state is just memory objects that will be GC'd.


@pytest.fixture(scope="module")
def backend_instance(request, shared_rate_limit_backend):
    """
    Module-scoped backend using shared rate limit coordination.

    Integration tests use shared backend for proper rate limit coordination
    across all parallel workers, enabling the intelligent scheduler to prevent
    429 errors.

    Returns:
        RedisBackend: Shared backend for rate limit coordination
    """
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")
    module_name = request.module.__name__.split(".")[-1]

    logger.info(f"Module {module_name} (worker {worker_id}) using shared rate limit backend")

    yield shared_rate_limit_backend


@pytest_asyncio.fixture(scope="function")
async def isolated_backend(request) -> AsyncGenerator:
    """
    Function-scoped backend for tests requiring complete isolation.

    Use this fixture only for:
    - Tests that modify global backend state
    - Tests that require specific backend configurations
    - Resource-intensive tests marked with @pytest.mark.resource_intensive

    Most tests should use the module-scoped backend_instance fixture.
    """
    # Generate unique namespace
    test_name = request.node.name
    test_namespace = f"test_isolated_{test_name}_{uuid.uuid4().hex[:8]}"

    from venice_ai.core.backends.redis import RedisBackend

    backend = RedisBackend(
        redis_url=os.getenv("VENICE_TEST_REDIS_URL", "redis://localhost:6379"),
        namespace=test_namespace,
        cluster_mode=REDIS_CLUSTER_MODE,
    )

    # Ensure connection in current event loop
    await backend._ensure_connected()

    yield backend

    # Cleanup
    if hasattr(backend, "cleanup"):
        await backend.cleanup()


@pytest.fixture(scope="session", autouse=True)
def monitor_resources():
    """
    Monitor resource usage during test session.

    Tracks:
    - File descriptor leaks
    - Memory usage growth
    - Unclosed connections
    """
    try:
        import gc

        import psutil

        # Baseline measurements
        process = psutil.Process()
        start_fds = process.num_fds() if hasattr(process, "num_fds") else 0
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_connections = len(process.net_connections(kind="inet"))

        yield

        # Force garbage collection
        gc.collect()

        # Final measurements
        end_fds = process.num_fds() if hasattr(process, "num_fds") else 0
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_connections = len(process.net_connections(kind="inet"))

        # Report issues
        fd_leak = end_fds - start_fds
        memory_growth = end_memory - start_memory
        connection_leak = end_connections - start_connections

        if fd_leak > 50:
            logger.warning(f"Possible file descriptor leak: {fd_leak} unclosed descriptors")

        if memory_growth > 1024:  # 1024 MB – video cassettes alone can use 600+ MB
            logger.warning(f"Significant memory growth: {memory_growth:.1f} MB")

        # In parallel testing, some connection accumulation is expected
        # Each worker may run multiple tests, and connections can accumulate
        # We use a higher threshold for parallel execution
        import os

        parallel_workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
        connection_threshold = 30 if parallel_workers == 1 else 40

        if connection_leak > connection_threshold:
            logger.warning(
                f"Possible connection leak: {connection_leak} unclosed connections (threshold: {connection_threshold})"
            )

        # Log summary
        logger.info(
            f"Resource usage - FDs: {start_fds}->{end_fds}, "
            f"Memory: {start_memory:.1f}->{end_memory:.1f} MB, "
            f"Connections: {start_connections}->{end_connections}"
        )
    except ImportError:
        # psutil not available, skip monitoring
        logger.debug("psutil not available, skipping resource monitoring")
        yield


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def performance_timer():
    """Fixture for timing test execution."""
    import time

    class PerformanceTimer:
        def __init__(self):
            self.timings = {}

        def start(self, name):
            self.timings[name] = {"start": time.perf_counter()}

        def stop(self, name):
            if name in self.timings:
                self.timings[name]["end"] = time.perf_counter()
                self.timings[name]["duration"] = (
                    self.timings[name]["end"] - self.timings[name]["start"]
                )

        def get_duration(self, name):
            return self.timings.get(name, {}).get("duration")

        def assert_duration_under(self, name, max_seconds):
            duration = self.get_duration(name)
            assert duration is not None, f"No timing found for {name}"
            assert duration < max_seconds, f"{name} took {duration:.2f}s, expected < {max_seconds}s"

    return PerformanceTimer()


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--refresh-models",
        action="store_true",
        default=False,
        help="Refresh test model cache from Venice API before running tests",
    )


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Refresh model cache if requested
    if config.getoption("--refresh-models", default=False):
        import asyncio
        import warnings

        from tests.fixtures.model_resolver import resolve_and_cache

        api_key = os.environ.get("VENICE_API_KEY") or os.environ.get("VENICE_TEST_API_KEY")
        if api_key:
            asyncio.run(resolve_and_cache(api_key))
        else:
            warnings.warn(
                "--refresh-models requires VENICE_API_KEY or VENICE_TEST_API_KEY; "
                "skipping cache refresh",
                stacklevel=1,
            )

    # Add custom markers if not already defined
    markers = [
        "unit: Unit tests",
        "integration: Integration tests",
        "e2e: End-to-end tests",
        "slow: Slow running tests",
        "fast: Fast running tests",
        "requires_api: Tests requiring API access",
        "smoke: Smoke tests",
        "benchmark: Benchmark tests",
        "requires_redis_pool: Tests requiring Redis with connection pool (50+ connections with pipelining)",
    ]

    for marker in markers:
        if marker.split(":")[0] not in config.option.markexpr:
            config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add async marker to async tests
        if inspect.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)


@pytest.fixture(autouse=True)
async def auto_cleanup_connections():
    """
    Auto-cleanup connections after each test to prevent leaks.

    This fixture runs after every test to ensure proper cleanup of:
    - Redis connections
    - Event loop tasks
    - Background processes
    """
    yield

    # Force cleanup of any Redis connections
    try:
        from venice_ai.core.backends.redis import RedisBackend

        if hasattr(RedisBackend, "cleanup_all_pools"):
            await RedisBackend.cleanup_all_pools()
    except ImportError:
        pass  # Redis not available
    except Exception as e:
        logger.debug(f"Error during Redis cleanup: {e}")

    # Force garbage collection to cleanup orphaned objects
    import gc

    gc.collect()

    # Give event loop time to process any pending cleanup
    import contextlib

    with contextlib.suppress(RuntimeError):
        await asyncio.sleep(0)


@pytest.fixture(autouse=True)
def sync_cleanup_connections():
    """
    Sync version of connection cleanup for synchronous tests.

    This ensures cleanup happens even for sync tests that can't
    use the async cleanup fixture. This helps prevent connection leaks
    when sync tests are run in the same worker as async tests.
    """
    yield

    # Force garbage collection to cleanup orphaned objects
    import gc

    gc.collect()

    # Aggressive cleanup for aiohttp sessions that might be lingering
    for obj in gc.get_objects():
        try:
            # Close any aiohttp ClientSession objects
            if (
                hasattr(obj, "__class__")
                and obj.__class__.__name__ == "ClientSession"
                and hasattr(obj, "close")
                and not getattr(obj, "closed", True)
            ):
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed() and not loop.is_running():
                        loop.run_until_complete(obj.close())
                except Exception:
                    pass
        except Exception:
            pass  # Ignore any errors during cleanup

    # Try to cleanup Redis pools synchronously if possible
    try:
        import asyncio

        # Skip sync cleanup entirely to avoid event loop conflicts with pytest-asyncio
        # The async cleanup fixture handles Redis cleanup properly
        return
    except Exception as e:
        logger.debug(f"Error during sync Redis cleanup: {e}")


@pytest.fixture(autouse=True)
def reset_test_state():
    """Automatically reset test state before each test."""
    # Reset any global state before each test
    yield
    # Cleanup after test if needed


# ============================================================================
# VCRpy Fixtures
# ============================================================================


def sanitize_dict_recursive(data):
    """
    Recursively sanitize sensitive fields in dictionary responses.

    This function sanitizes:
    - API key values and partial keys
    - Web3 addresses and signatures
    - Billing amounts (normalized to generic values)
    - Account balances
    """
    if not isinstance(data, dict):
        return

    # Sensitive fields that should be completely redacted
    sensitive_fields = {
        "apiKey": "REDACTED_API_KEY",
        "last6Chars": "XXXXXX",
        "address": "0xREDACTED",
        "signature": "0xREDACTED",
        "token": "REDACTED_TOKEN",
    }

    for key, value in list(data.items()):
        # Replace sensitive fields with redacted values
        if key in sensitive_fields:
            data[key] = sensitive_fields[key]

        # Normalize billing amounts to generic test values
        elif key in ["amount", "usd", "vcu", "diem"] and isinstance(value, (int, float, str)):
            try:
                numeric_value = float(value) if isinstance(value, str) else value
                # Normalize to 10% of original value, rounded to 2 decimals
                data[key] = round(numeric_value / 10, 2) if numeric_value else 0.0
            except (ValueError, TypeError):
                pass

        # Normalize balance objects
        elif key == "balances" and isinstance(value, dict):
            for currency in ["USD", "VCU", "DIEM", "usd", "vcu", "diem"]:
                if currency in value and value[currency] is not None:
                    try:
                        numeric_value = (
                            float(value[currency])
                            if isinstance(value[currency], str)
                            else value[currency]
                        )
                        value[currency] = round(numeric_value / 10, 2) if numeric_value else 0.0
                    except (ValueError, TypeError):
                        pass

        # Recurse into nested structures
        elif isinstance(value, dict):
            sanitize_dict_recursive(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    sanitize_dict_recursive(item)


def sanitize_sensitive_data(response):
    """
    Sanitize sensitive data in VCR cassette responses before recording.

    This function is called by VCR's before_record_response hook to
    automatically sanitize sensitive information including:
    - API keys and authentication tokens
    - Web3 blockchain addresses and signatures
    - Billing amounts and account balances

    Only processes JSON responses; CSV and other formats are left untouched.
    """
    import json

    # Only process JSON responses
    content_type = response.get("headers", {}).get("Content-Type", [""])[0]
    if not content_type.startswith("application/json"):
        return response

    try:
        # Parse JSON body
        body_string = response["body"]["string"]
        if isinstance(body_string, bytes):
            body_string = body_string.decode("utf-8")

        body = json.loads(body_string)

        # Sanitize the parsed body
        if isinstance(body, dict):
            sanitize_dict_recursive(body)
        elif isinstance(body, list):
            for item in body:
                if isinstance(item, dict):
                    sanitize_dict_recursive(item)

        # Update response body with sanitized data
        sanitized_body = json.dumps(body, indent=2)
        response["body"]["string"] = sanitized_body.encode("utf-8")

    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        # If parsing fails, log but don't modify response
        import logging

        logging.warning(f"Failed to sanitize VCR response: {e}")

    return response


def strip_unserializable_request_body(request):
    """Replace non-serializable request bodies before VCR records them.

    vcrpy 8.2+ loads cassettes with a hardened SafeLoader that refuses
    ``!!python/object`` YAML tags (an arbitrary-code-execution CVE class).
    aiohttp multipart uploads pass a live ``aiohttp.formdata.FormData``
    object as ``data=``; vcrpy's aiohttp stub records it verbatim and the
    YAML dumper writes it back as such a tag, producing a cassette that can
    no longer be replayed (it raises a misleading "your cassette files were
    generated in an older version of VCR" error on load).

    Request bodies are not part of ``match_on`` (see ``vcr_config``), so the
    recorded body content is irrelevant to replay matching. We replace any
    body the SafeLoader cannot reconstruct — i.e. anything that is not
    ``None``, ``str``/``bytes``, a ``BytesIO`` file body, or an iterator (all
    of which vcrpy whitelists) — with a safe placeholder. The real,
    unmodified body is still sent to the live API while recording.
    """
    from io import BytesIO

    body = request.body
    if body is None or isinstance(body, (str, bytes, BytesIO)) or hasattr(body, "__next__"):
        return request
    request.body = f"<non-serializable {type(body).__name__} request body stripped by VCR>"
    return request


@pytest.fixture(scope="session")
def vcr_config():
    """
    Shared VCRpy configuration for cassette-based testing.

    This fixture provides a pre-configured VCR instance that:
    - Records HTTP interactions to YAML cassettes
    - Scrubs Authorization headers to prevent API key leakage
    - Sanitizes sensitive data in response bodies (API keys, billing data)
    - Replays only by default; recording is opt-in via VENICE_VCR_RECORD
      (``all`` to re-record, ``new`` to fill gaps). Cassettes are gitignored,
      so a recording mode on a fresh clone spends real credit.
    - Stores cassettes in tests/cassettes/ directory

    CI Mode: When VENICE_CI_MODE=true, uses NONE mode to prevent recording.

    See also:
        venice_ai.test_support.vcr_utilities - Comprehensive VCR documentation,
        usage guide, and compatibility notes for production code patterns.
    """
    # Resolved centrally so this fixture and the benchmark suite's module-level
    # VCR instance can never disagree about whether a run may touch the network.
    # See tests/vcr_policy.py for the token table.
    record_mode = resolve_record_mode()

    return vcr.VCR(
        cassette_library_dir="tests/integration/cassettes",
        record_mode=record_mode,
        # Security: scrub sensitive headers. x-sign-in-with-x is the SIWE
        # wallet-auth token for /x402/* endpoints (time-limited but wallet-
        # bound); x-402-payment carries signed x402 payment payloads.
        filter_headers=[
            "authorization",
            "x-api-key",
            "x-sign-in-with-x",
            "x-402-payment",
        ],
        # Match requests on method, scheme, host, port, path, and query
        match_on=["method", "scheme", "host", "port", "path", "query"],
        # Decode compressed responses for readability
        decode_compressed_response=True,
        # Serializer for YAML format
        serializer="yaml",
        # Sanitize sensitive data before recording
        before_record_response=sanitize_sensitive_data,
        # Strip live aiohttp FormData/payload objects from recorded request
        # bodies so cassettes load under vcrpy 8.2+'s hardened SafeLoader.
        before_record_request=strip_unserializable_request_body,
        # Prevent recording cassettes on test exceptions (e.g., accidental 429/500)
        # This ensures only successful test runs create cassettes
        record_on_exception=False,
    )


@pytest.fixture
def vcr_cassette(vcr_config, request):
    """
    Function-scoped fixture that provides a VCR cassette context manager for the current test.

    The cassette name is automatically derived from the test function name and the
    cassette directory is determined by the test location (integration vs e2e).
    This fixture should be used in tests that need to record/replay HTTP interactions.

    Usage:
        def test_api_call(vcr_cassette):
            with vcr_cassette:
                # Make HTTP calls here
                response = requests.get("https://api.example.com/data")
    """
    test_name = request.node.name
    cassette_name = f"{test_name}.yaml"

    # Live mode: `--disable-vcr` (passed by `make test-ci`, and usable locally to
    # emulate a cassette-less CI run) means "don't play back from cassettes" — so
    # bypass VCR entirely and let requests hit the live API. This is what exercises
    # the SDK's real rate-limit / retry behaviour end-to-end; recorded cassettes
    # remain a local-dev convenience used only when the flag is absent.
    if request.config.getoption("--disable-vcr", default=False):
        yield contextlib.nullcontext()
        return

    # Determine cassette directory based on test path
    cassette_dir = cassette_dir_for(str(request.fspath))

    # Configure the cassette library directory
    vcr_config.cassette_library_dir = cassette_dir

    # vcrpy 8.2 ALL-mode APPENDS to (does not clear) an existing cassette, and on
    # replay serves the OLDEST matching interaction first — so stale interactions
    # mask fresh fixes (match_on excludes the body, so same-endpoint requests
    # share one match key). To force a clean overwrite we move the existing
    # cassette aside before an ALL-mode (explicit non-CI VENICE_VCR_RECORD=all)
    # record, then in teardown KEEP the freshly written one, or RESTORE the prior
    # one if the test wrote nothing (failed / skipped / timed out / 402'd). That
    # way a transient, feature-unsupported, out-of-balance, or wrong-model failure
    # during re-record never DESTROYS a good cassette — the offline replay-verify
    # keeps working from the prior recording. NEW_EPISODES and CI/NONE are
    # untouched. The first block self-heals a cassette left stranded as ``.bak``
    # by an interrupted prior run.
    cassette_path = Path(cassette_dir) / cassette_name
    backup_path = Path(f"{cassette_path}.bak")
    if backup_path.exists() and not cassette_path.exists():
        backup_path.rename(cassette_path)
    rerecording = vcr_config.record_mode == RecordMode.ALL
    if rerecording and cassette_path.exists():
        cassette_path.rename(backup_path)
    try:
        yield vcr_config.use_cassette(cassette_name)
    finally:
        if rerecording:
            if cassette_path.exists():
                backup_path.unlink(missing_ok=True)
            elif backup_path.exists():
                backup_path.rename(cassette_path)


@pytest.fixture
def vcr_config_with_errors(vcr_config):
    """
    VCR configuration that allows recording error responses (4xx, 5xx).

    Use this fixture for tests that intentionally verify error handling behavior.
    Unlike the default vcr_config, this will record cassettes even when the test
    raises exceptions (e.g., when testing that a 429 rate limit raises an exception).

    Usage:
        def test_rate_limit_handling(vcr_config_with_errors, request):
            cassette = vcr_config_with_errors.use_cassette(f"{request.node.name}.yaml")
            with cassette:
                with pytest.raises(RateLimitError):
                    # Make call that triggers 429
                    await client.chat.completions.create(...)
    """
    # Create a new VCR instance with record_on_exception=True
    # This allows recording error responses that intentionally raise exceptions
    return vcr.VCR(
        cassette_library_dir=vcr_config.cassette_library_dir,
        record_mode=vcr_config.record_mode,
        filter_headers=[
            "authorization",
            "x-api-key",
            "x-sign-in-with-x",
            "x-402-payment",
        ],
        match_on=["method", "scheme", "host", "port", "path", "query"],
        decode_compressed_response=True,
        serializer="yaml",
        before_record_response=sanitize_sensitive_data,
        before_record_request=strip_unserializable_request_body,
        # Allow recording on exceptions for error-testing scenarios
        record_on_exception=True,
    )


@pytest.fixture
def vcr_error_cassette(vcr_config_with_errors, request):
    """
    Function-scoped fixture that provides a VCR cassette for error-testing scenarios.

    Similar to vcr_cassette, but uses vcr_config_with_errors to allow recording
    even when tests raise exceptions (e.g., when testing 429/500 error handling).

    Usage:
        def test_rate_limit_raises_error(vcr_error_cassette):
            with vcr_error_cassette:
                with pytest.raises(RateLimitError):
                    await client.chat.completions.create(...)
    """
    test_name = request.node.name
    cassette_name = f"{test_name}.yaml"

    # Live mode: `--disable-vcr` (passed by `make test-ci`, and usable locally to
    # emulate a cassette-less CI run) means "don't play back from cassettes" — so
    # bypass VCR entirely and let requests hit the live API. This is what exercises
    # the SDK's real rate-limit / retry behaviour end-to-end; recorded cassettes
    # remain a local-dev convenience used only when the flag is absent.
    if request.config.getoption("--disable-vcr", default=False):
        yield contextlib.nullcontext()
        return

    # Determine cassette directory based on test path
    cassette_dir = cassette_dir_for(str(request.fspath))

    # Configure the cassette library directory
    vcr_config_with_errors.cassette_library_dir = cassette_dir

    # Same overwrite-on-success / restore-on-failure re-record safety as
    # vcr_cassette (see there): move an existing cassette aside before an ALL-mode
    # record, then keep the freshly written one or restore the prior one if the
    # test wrote nothing. Self-heals a cassette stranded as ``.bak``.
    cassette_path = Path(cassette_dir) / cassette_name
    backup_path = Path(f"{cassette_path}.bak")
    if backup_path.exists() and not cassette_path.exists():
        backup_path.rename(cassette_path)
    rerecording = vcr_config_with_errors.record_mode == RecordMode.ALL
    if rerecording and cassette_path.exists():
        cassette_path.rename(backup_path)
    try:
        yield vcr_config_with_errors.use_cassette(cassette_name)
    finally:
        if rerecording:
            if cassette_path.exists():
                backup_path.unlink(missing_ok=True)
            elif backup_path.exists():
                backup_path.rename(cassette_path)


# ============================================================================
# Cassette Validation Utilities
# ============================================================================


def validate_cassettes_for_errors(
    cassette_dirs: list | None = None,
    error_statuses: tuple = (429, 500, 502, 503, 504),
    raise_on_error: bool = False,
) -> dict[str, list]:
    """
    Scan VCR cassettes for unexpected HTTP error responses.

    This utility helps identify cassettes that may have accidentally recorded
    error responses (e.g., 429 rate limits, 500 server errors) during test
    recording. Such cassettes can cause false positives in future test runs.

    Args:
        cassette_dirs: List of directories to scan (defaults to standard test cassette dirs)
        error_statuses: Tuple of HTTP status codes to flag as errors
        raise_on_error: If True, raise an exception when errors are found

    Returns:
        Dictionary mapping cassette paths to lists of error status codes found

    Usage:
        # In a test or as a standalone check
        errors = validate_cassettes_for_errors()
        if errors:
            for path, statuses in errors.items():
                print(f"⚠️ {path}: Contains {statuses} responses")

        # As a CI check that fails on errors
        validate_cassettes_for_errors(raise_on_error=True)
    """
    import yaml

    if cassette_dirs is None:
        cassette_dirs = [
            Path("tests/integration/cassettes"),
            Path("tests/e2e/cassettes"),
        ]

    errors_found = {}

    for cassette_dir in cassette_dirs:
        cassette_path = Path(cassette_dir)
        if not cassette_path.exists():
            continue

        for cassette_file in cassette_path.rglob("*.yaml"):
            try:
                with open(cassette_file) as f:
                    content = yaml.safe_load(f)

                if not content or "interactions" not in content:
                    continue

                file_errors = []
                for interaction in content.get("interactions", []):
                    response = interaction.get("response", {})
                    status = response.get("status", {})
                    code = status.get("code") if isinstance(status, dict) else status

                    if code in error_statuses:
                        file_errors.append(code)

                if file_errors:
                    errors_found[str(cassette_file)] = file_errors

            except (yaml.YAMLError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse cassette {cassette_file}: {e}")

    if raise_on_error and errors_found:
        error_msg = "Found cassettes with error responses:\n"
        for path, statuses in errors_found.items():
            error_msg += f"  - {path}: {statuses}\n"
        raise ValueError(error_msg)

    return errors_found


@pytest.fixture
def cassette_validator():
    """
    Fixture providing cassette validation utilities.

    Usage:
        def test_cassettes_are_clean(cassette_validator):
            errors = cassette_validator()
            assert not errors, f"Found error responses in cassettes: {errors}"
    """
    return validate_cassettes_for_errors


# ============================================================================
# Test Execution Helpers
# ============================================================================


@pytest.fixture
def run_with_timeout():
    """Run an async function with timeout."""

    async def _run_with_timeout(coro, timeout_seconds=5.0):
        try:
            return await asyncio.wait_for(coro, timeout=timeout_seconds)
        except TimeoutError:
            pytest.fail(f"Test timed out after {timeout_seconds} seconds")

    return _run_with_timeout


@pytest.fixture
def skip_if_no_api():
    """Skip test if no API key is available."""

    def _skip_if_no_api():
        if not os.getenv("VENICE_TEST_API_KEY"):
            pytest.skip("No API key available for testing")

    return _skip_if_no_api
