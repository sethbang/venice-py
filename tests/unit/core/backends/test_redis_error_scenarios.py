"""
Additional tests for RedisBackend error scenarios and edge cases.

This test module focuses on achieving >80% test coverage by testing
error handling paths, connection pool management, and edge cases.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError, ResponseError

from venice_ai.core.backends.redis import RedisBackend


class TestRedisBackendConnectionPoolManagement:
    """Test connection pool management and event loop switching."""

    @pytest.mark.asyncio
    async def test_event_loop_switching(self):
        """Test _ensure_connected handles event loop switching correctly."""
        backend = RedisBackend(redis_url="redis://localhost:6379", namespace="test_namespace")

        # Mock the first event loop
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop1 = MagicMock()
            mock_loop1_id = id(mock_loop1)
            mock_get_loop.return_value = mock_loop1

            with (
                patch("venice_ai.core.backends.redis.Redis") as mock_redis_class,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                # Setup mocks
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_redis_instance = AsyncMock()
                mock_redis_instance.ping = AsyncMock(return_value=True)
                mock_redis_instance.script_load = AsyncMock(return_value="test_sha")
                mock_redis_class.return_value = mock_redis_instance

                # First connection
                await backend._ensure_connected()
                assert backend._event_loop_id == mock_loop1_id
                assert backend._connected is True

                # Simulate event loop change
                mock_loop2 = MagicMock()
                mock_loop2_id = id(mock_loop2)
                mock_get_loop.return_value = mock_loop2

                # This should trigger the event loop change branch (lines 148-154)
                await backend._ensure_connected()
                assert backend._event_loop_id == mock_loop2_id

    @pytest.mark.asyncio
    async def test_fallback_redis_connection_creation(self):
        """Test fallback Redis connection when ConnectionPool is not available."""
        backend = RedisBackend(redis_url="redis://localhost:6379", namespace="test_namespace")

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch("venice_ai.core.backends.redis.ConnectionPool", None),
                patch("venice_ai.core.backends.redis.Redis", None),
                patch("venice_ai.core.backends.redis.redis") as mock_redis_module,
            ):
                # Make ConnectionPool unavailable, and mock Redis class to return None
                mock_redis_instance = AsyncMock()
                mock_redis_instance.ping = AsyncMock(return_value=True)
                mock_redis_instance.script_load = AsyncMock(return_value="test_sha")
                mock_redis_module.from_url = AsyncMock(return_value=mock_redis_instance)

                # This should trigger fallback connection creation (lines 197-210)
                redis_client = await backend._ensure_connected()
                assert redis_client is not None
                mock_redis_module.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_ping_failure_reconnect(self):
        """Test _ensure_connected handles ping failure and reconnects."""
        backend = RedisBackend(redis_url="redis://localhost:6379", namespace="test_namespace")

        # Set up existing connection that will fail ping
        mock_existing_redis = AsyncMock()
        mock_existing_redis.ping = AsyncMock(side_effect=ConnectionError("Connection lost"))
        backend._redis = mock_existing_redis
        backend._connected = True

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            backend._event_loop_id = id(mock_loop)  # Same loop: no pool switch

            with (
                patch("venice_ai.core.backends.redis.Redis") as mock_redis_class,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                # Setup new connection mocks
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_new_redis = AsyncMock()
                mock_new_redis.ping = AsyncMock(return_value=True)
                mock_new_redis.script_load = AsyncMock(return_value="test_sha")
                mock_redis_class.return_value = mock_new_redis

                # This should trigger reconnection logic (lines 163-166)
                await backend._ensure_connected()
                assert backend._redis == mock_new_redis
                assert backend._connected is True


class TestRedisBackendFailureManagement:
    """Test failure tracking and circuit breaker functionality."""

    @pytest.fixture
    def redis_backend_with_mock(self):
        """Create a RedisBackend with a mock client that can simulate errors."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_namespace",
        )
        backend._redis = mock_client
        backend._connected = True
        return backend, mock_client

    @pytest.mark.asyncio
    async def test_record_failure_connection_error(self, redis_backend_with_mock):
        """Test record_failure handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lpush.side_effect = ConnectionError("Connection lost")

        # This should cover lines 561-562
        await backend.record_failure("TestError", "Test message")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_record_failure_response_error(self, redis_backend_with_mock):
        """Test record_failure handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lpush.side_effect = ResponseError("Redis response error")

        # This should cover lines 563-564
        await backend.record_failure("TestError", "Test message")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_record_failure_redis_error(self, redis_backend_with_mock):
        """Test record_failure handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lpush.side_effect = RedisError("Redis internal error")

        # This should cover lines 565-566
        await backend.record_failure("TestError", "Test message")
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_get_failure_count_connection_error(self, redis_backend_with_mock):
        """Test get_failure_count handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lrange.side_effect = ConnectionError("Connection lost")

        # This should cover lines 590-592
        count = await backend.get_failure_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_failure_count_response_error(self, redis_backend_with_mock):
        """Test get_failure_count handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lrange.side_effect = ResponseError("Redis response error")

        # This should cover lines 593-595
        count = await backend.get_failure_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_get_failure_count_redis_error(self, redis_backend_with_mock):
        """Test get_failure_count handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.lrange.side_effect = RedisError("Redis internal error")

        # This should cover lines 596-598
        count = await backend.get_failure_count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_clear_failures_success(self, redis_backend_with_mock):
        """Test clear_failures method."""
        backend, mock_client = redis_backend_with_mock
        mock_client.delete.return_value = 1

        # This should cover lines 930-935 (clear_failures method)
        await backend.clear_failures()
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_failures_connection_error(self, redis_backend_with_mock):
        """Test clear_failures handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.delete.side_effect = ConnectionError("Connection lost")

        # This should cover lines 936-937
        await backend.clear_failures()
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_clear_failures_response_error(self, redis_backend_with_mock):
        """Test clear_failures handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.delete.side_effect = ResponseError("Redis response error")

        # This should cover lines 938-939
        await backend.clear_failures()
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_clear_failures_redis_error(self, redis_backend_with_mock):
        """Test clear_failures handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.delete.side_effect = RedisError("Redis internal error")

        # This should cover lines 940-941
        await backend.clear_failures()
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_force_circuit_break_success(self, redis_backend_with_mock):
        """Test force_circuit_break sets dedicated Redis key with TTL."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex = AsyncMock(return_value=True)

        await backend.force_circuit_break(5.0)

        # Should set the circuit break key with TTL
        mock_client.setex.assert_called_once_with(backend.circuit_break_key, 5, "1")

    @pytest.mark.asyncio
    async def test_force_circuit_break_connection_error(self, redis_backend_with_mock):
        """Test force_circuit_break handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = ConnectionError("Connection lost")

        await backend.force_circuit_break(5.0)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_force_circuit_break_response_error(self, redis_backend_with_mock):
        """Test force_circuit_break handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = ResponseError("Redis response error")

        await backend.force_circuit_break(5.0)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_force_circuit_break_redis_error(self, redis_backend_with_mock):
        """Test force_circuit_break handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = RedisError("Redis internal error")

        await backend.force_circuit_break(5.0)
        # Should not raise exception


class TestRedisBackendHealthAndCleanup:
    """Test health check and cleanup operations."""

    @pytest.fixture
    def redis_backend_with_mock(self):
        """Create a RedisBackend with a mock client that can simulate errors."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_namespace",
        )
        backend._redis = mock_client
        backend._connected = True
        # Set _owned_redis to True so cleanup will actually clean up
        backend._owned_redis = True
        return backend, mock_client

    @pytest.mark.asyncio
    async def test_health_check_connection_error(self, redis_backend_with_mock):
        """Test health_check handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.set.side_effect = ConnectionError("Connection lost")

        # This should cover lines 810-815
        result = await backend.health_check()
        assert result.healthy is False
        assert "Connection error" in result.error

    @pytest.mark.asyncio
    async def test_health_check_response_error(self, redis_backend_with_mock):
        """Test health_check handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.set.side_effect = ResponseError("Redis response error")

        # This should cover lines 816-821
        result = await backend.health_check()
        assert result.healthy is False
        assert "Redis response error" in result.error

    @pytest.mark.asyncio
    async def test_health_check_redis_error(self, redis_backend_with_mock):
        """Test health_check handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.set.side_effect = RedisError("Redis internal error")

        # This should cover lines 822-827
        result = await backend.health_check()
        assert result.healthy is False
        assert "Redis error" in result.error

    @pytest.mark.asyncio
    async def test_health_check_unexpected_error(self, redis_backend_with_mock):
        """Test health_check handles unexpected errors."""
        backend, mock_client = redis_backend_with_mock
        mock_client.set.side_effect = ValueError("Unexpected error")

        # This should cover lines 828-833
        result = await backend.health_check()
        assert result.healthy is False
        assert "Unexpected error" in result.error

    @pytest.mark.asyncio
    async def test_get_all_stats_connection_error(self, redis_backend_with_mock):
        """Test get_all_stats handles ConnectionError from scan_iter."""
        backend, mock_client = redis_backend_with_mock

        async def _raise_scan_iter(**kwargs):
            raise ConnectionError("Connection lost")
            # Make it an async generator that raises
            yield  # pragma: no cover  # noqa: E501

        mock_client.scan_iter = _raise_scan_iter

        result = await backend.get_all_stats()
        assert result["rate_limits"] == 0
        assert result["failures"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_all_stats_response_error(self, redis_backend_with_mock):
        """Test get_all_stats handles ResponseError from scan_iter."""
        backend, mock_client = redis_backend_with_mock

        async def _raise_scan_iter(**kwargs):
            raise ResponseError("Redis response error")
            yield  # pragma: no cover  # noqa: E501

        mock_client.scan_iter = _raise_scan_iter

        result = await backend.get_all_stats()
        assert result["rate_limits"] == 0
        assert result["failures"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_all_stats_redis_error(self, redis_backend_with_mock):
        """Test get_all_stats handles RedisError from scan_iter."""
        backend, mock_client = redis_backend_with_mock

        async def _raise_scan_iter(**kwargs):
            raise RedisError("Redis internal error")
            yield  # pragma: no cover  # noqa: E501

        mock_client.scan_iter = _raise_scan_iter

        result = await backend.get_all_stats()
        assert result["rate_limits"] == 0
        assert result["failures"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cleanup_connection_error(self, redis_backend_with_mock):
        """Test cleanup handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        # AsyncMock has aclose() which is checked first in _cleanup_connection
        mock_client.aclose.side_effect = ConnectionError("Connection error during close")

        # This should cover lines 877-878
        await backend.cleanup()
        assert backend._redis is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_cleanup_unexpected_error(self, redis_backend_with_mock):
        """Test cleanup handles unexpected errors."""
        backend, mock_client = redis_backend_with_mock
        # AsyncMock has aclose() which is checked first in _cleanup_connection
        mock_client.aclose.side_effect = RuntimeError("Unexpected error")

        # This should cover lines 879-880
        await backend.cleanup()
        assert backend._redis is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_with_connection_errors(self):
        """Test cleanup_all_pools handles connection errors during disconnect."""
        mock_pool1 = MagicMock()
        mock_pool1.disconnect = AsyncMock(side_effect=ConnectionError("Connection error"))

        mock_pool2 = MagicMock()
        mock_pool2.disconnect = AsyncMock(side_effect=RuntimeError("Runtime error"))

        with patch.object(RedisBackend, "_connection_pools", {1: mock_pool1, 2: mock_pool2}):
            # This should cover lines 919-926
            await RedisBackend.cleanup_all_pools()
            # Should not raise exceptions

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_no_event_loop_fallback(self):
        """Test cleanup_all_pools skips cleanup when no event loop exists."""
        mock_pool = MagicMock()
        # Make disconnect a regular function that returns None
        mock_pool.disconnect = lambda: None

        with (
            patch.object(RedisBackend, "_connection_pools", {1: mock_pool}),
            patch("asyncio.get_running_loop", side_effect=RuntimeError("No event loop")),
            patch("asyncio.run") as mock_run,
        ):
            mock_run.return_value = None  # Make run return None

            await RedisBackend.cleanup_all_pools()

            # Verify asyncio.run was NOT called (behavior changed to skip cleanup)
            mock_run.assert_not_called()


class TestRedisBackendRateLimitErrorHandling:
    """Test error handling in rate limit operations."""

    @pytest.fixture
    def redis_backend_with_mock(self):
        """Create a RedisBackend with a mock client that can simulate errors."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_namespace",
        )
        backend._redis = mock_client
        backend._connected = True
        backend._owned_redis = True  # Ensure cleanup runs
        return backend, mock_client

    @pytest.mark.asyncio
    async def test_update_rate_limits_connection_error(self, redis_backend_with_mock):
        """Test update_rate_limits handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = ConnectionError("Connection lost")

        # This should cover lines 621-625
        await backend.update_rate_limits("test_model", {"x-ratelimit-remaining-requests": "100"})
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_update_rate_limits_response_error(self, redis_backend_with_mock):
        """Test update_rate_limits handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = ResponseError("Redis response error")

        # This should cover lines 625-627
        await backend.update_rate_limits("test_model", {"x-ratelimit-remaining-requests": "100"})
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_update_rate_limits_redis_error(self, redis_backend_with_mock):
        """Test update_rate_limits handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.setex.side_effect = RedisError("Redis internal error")

        # This should cover lines 627-628
        await backend.update_rate_limits("test_model", {"x-ratelimit-remaining-requests": "100"})
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_get_rate_limits_connection_error(self, redis_backend_with_mock):
        """Test _get_rate_limits handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.side_effect = ConnectionError("Connection lost")

        # Test internal _get_rate_limits method
        result = await backend._get_rate_limits("test_model")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_rate_limits_response_error(self, redis_backend_with_mock):
        """Test _get_rate_limits handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.side_effect = ResponseError("Redis response error")

        # Test internal _get_rate_limits method
        result = await backend._get_rate_limits("test_model")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_rate_limits_redis_error(self, redis_backend_with_mock):
        """Test _get_rate_limits handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.side_effect = RedisError("Redis internal error")

        # Test internal _get_rate_limits method
        result = await backend._get_rate_limits("test_model")
        assert result == {}

    @pytest.mark.asyncio
    async def test_record_request_connection_error(self, redis_backend_with_mock):
        """Test record_request handles ConnectionError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.return_value = json.dumps({"rpm_remaining": 100, "tpm_remaining": 10000})
        mock_client.setex.side_effect = ConnectionError("Connection lost")

        await backend.record_request("test_model", tokens_used=100)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_record_request_response_error(self, redis_backend_with_mock):
        """Test record_request handles ResponseError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.return_value = json.dumps({"rpm_remaining": 100, "tpm_remaining": 10000})
        mock_client.setex.side_effect = ResponseError("Redis response error")

        await backend.record_request("test_model", tokens_used=100)
        # Should not raise exception

    @pytest.mark.asyncio
    async def test_record_request_redis_error(self, redis_backend_with_mock):
        """Test record_request handles RedisError."""
        backend, mock_client = redis_backend_with_mock
        mock_client.get.return_value = json.dumps({"rpm_remaining": 100, "tpm_remaining": 10000})
        mock_client.setex.side_effect = RedisError("Redis internal error")

        await backend.record_request("test_model", tokens_used=100)
        # Should not raise exception
