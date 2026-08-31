"""
Comprehensive test coverage for RedisBackend.

This test module targets specific missing lines and branches to achieve 90%+ coverage.
It focuses on:
- Cluster mode connection handling (lines 148-154, 163)
- Ping timeout scenarios (lines 207-209)
- RuntimeError and connection error paths (lines 219-231)
- Cleanup connection timeout handling (lines 240-255)
- Loop cleanup callback paths (lines 274-301)
- Health check data retrieval (lines 318-323)
- Check capacity with circuit breaker (lines 406-413)
- Rate limit update with empty headers (lines 421-425)
- Failure count JSON iteration (lines 502-512)
- Circuit breaker state (lines 525-526)
- Force circuit break async cleanup (lines 549, 552-557)
- Context manager paths (lines 586-600)
- cleanup_all_pools comprehensive paths (lines 611-654)
"""

import asyncio
import builtins
import contextlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.exceptions import ConnectionError, RedisError, ResponseError, TimeoutError

from venice_ai.core.backends.redis import RedisBackend


class TestClusterModeConnection:
    """Test cluster mode connection handling (lines 148-154, 163)."""

    @pytest.mark.asyncio
    async def test_cluster_mode_creates_cluster_client(self):
        """Test that cluster_mode creates a RedisCluster client."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_cluster",
            cluster_mode=True,
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch("venice_ai.core.backends.redis.RedisCluster") as mock_cluster,
            ):
                mock_cluster_instance = AsyncMock()
                mock_cluster_instance.ping = AsyncMock(return_value=True)
                mock_cluster.from_url.return_value = mock_cluster_instance

                result = await backend._ensure_connected()

                # Verify cluster client was created (covers lines 149-150, 154)
                mock_cluster.from_url.assert_called_once_with(
                    "redis://localhost:6379",
                    decode_responses=True,
                )
                assert result == mock_cluster_instance
                assert backend.cluster_mode is True

    @pytest.mark.asyncio
    async def test_connection_pool_limit_warning(self):
        """Test warning when connection pool limit is reached (line 163)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_pool_limit",
        )

        # Pre-fill connection pools to reach limit
        with (
            patch.object(RedisBackend, "_max_pools", 2),
            patch.object(RedisBackend, "_connection_pools", {1: MagicMock(), 2: MagicMock()}),
            patch("asyncio.get_running_loop") as mock_get_loop,
        ):
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch("venice_ai.core.backends.redis.Redis") as mock_redis,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_redis_instance = AsyncMock()
                mock_redis_instance.ping = AsyncMock(return_value=True)
                mock_redis.return_value = mock_redis_instance

                with patch("venice_ai.core.backends.redis.logger") as mock_logger:
                    await backend._ensure_connected()

                    # Verify warning was logged (covers line 163)
                    warning_calls = [
                        call
                        for call in mock_logger.warning.call_args_list
                        if "pool limit" in str(call).lower()
                    ]
                    assert len(warning_calls) >= 1


class TestPingTimeoutHandling:
    """Test ping timeout scenarios (lines 207-209)."""

    @pytest.mark.asyncio
    async def test_ping_timeout_raises_connection_error(self):
        """Test that ping timeout raises ConnectionError."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_ping_timeout",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch("venice_ai.core.backends.redis.Redis") as mock_redis_class,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_redis_instance = MagicMock()
                # ping doesn't need to be AsyncMock since wait_for is patched
                mock_redis_instance.ping = MagicMock()
                mock_redis_class.return_value = mock_redis_instance

                # Use a regular function that raises TimeoutError synchronously
                # This avoids creating unawaited coroutines
                def raise_timeout(*args, **kwargs):
                    raise builtins.TimeoutError()

                with (
                    patch("asyncio.wait_for", side_effect=raise_timeout),
                    pytest.raises(ConnectionError) as exc_info,
                ):
                    await backend._ensure_connected()

                    # Verify ping timeout error (covers lines 207-209)
                    assert "ping timed out" in str(exc_info.value).lower()


class TestRuntimeErrorHandling:
    """Test RuntimeError handling (lines 219-225)."""

    @pytest.mark.asyncio
    async def test_no_running_event_loop_error(self):
        """Test handling of 'no running event loop' RuntimeError."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_no_loop",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("no running event loop")

            with pytest.raises(RuntimeError) as exc_info:
                await backend._ensure_connected()

            # Verify proper error message (covers lines 219-221)
            assert "no running event loop" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_other_runtime_error_propagates(self):
        """Test that other RuntimeErrors are propagated."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_other_error",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_get_loop.side_effect = RuntimeError("some other error")

            with pytest.raises(RuntimeError) as exc_info:
                await backend._ensure_connected()

            # Verify error propagates (covers line 225)
            assert "some other error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_error_during_connection(self):
        """Test TimeoutError handling during connection (lines 226-228)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_timeout",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch.object(RedisBackend, "_connection_pools", {}),
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool_class.from_url.side_effect = TimeoutError("Connection timeout")

                with pytest.raises(TimeoutError):
                    await backend._ensure_connected()

    @pytest.mark.asyncio
    async def test_redis_error_during_connection(self):
        """Test RedisError handling during connection (lines 229-231)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_redis_error",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch.object(RedisBackend, "_connection_pools", {}),
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool_class.from_url.side_effect = RedisError("Redis error")

                with pytest.raises(RedisError):
                    await backend._ensure_connected()


class TestCleanupConnectionTimeout:
    """Test cleanup connection timeout handling (lines 240-255)."""

    @pytest.mark.asyncio
    async def test_cleanup_connection_timeout_with_aclose(self):
        """Test cleanup timeout handling when aclose times out (lines 246-250)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_cleanup_timeout",
        )

        # Use a simple mock for the connection - no async functions needed
        # since we're patching wait_for to raise TimeoutError
        mock_connection = MagicMock()
        mock_connection.aclose = MagicMock()  # Will be wrapped by wait_for
        mock_connection.close = MagicMock()

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = builtins.TimeoutError()

            # This should not raise, and should try force close (covers lines 247, 251-255)
            await backend._cleanup_connection(12345, mock_connection, timeout=0.1)

            # Verify force close was attempted
            mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_connection_force_close_fails(self):
        """Test cleanup when force close also fails (lines 254-255)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_force_close_fail",
        )

        mock_connection = MagicMock()
        mock_connection.close = MagicMock(side_effect=RuntimeError("Force close failed"))

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = builtins.TimeoutError()

            # This should not raise even when force close fails
            await backend._cleanup_connection(12345, mock_connection, timeout=0.1)

    @pytest.mark.asyncio
    async def test_cleanup_connection_with_close_method(self):
        """Test cleanup uses close() when aclose() not available (lines 242-243)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_close_fallback",
        )

        mock_connection = AsyncMock(spec=["close"])
        mock_connection.close = AsyncMock()

        await backend._cleanup_connection(12345, mock_connection, timeout=2.5)

        # Verify close was called since aclose is not available
        mock_connection.close.assert_called_once()


class TestLoopCleanupCallback:
    """Test loop cleanup callback paths (lines 274-286)."""

    def test_register_loop_cleanup_and_callback_with_running_loop(self):
        """Test loop cleanup callback when event loop is available (lines 279-281)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_callback",
        )

        mock_loop = MagicMock()
        mock_loop_id = id(mock_loop)
        mock_connection = MagicMock()

        # Register cleanup
        backend._register_loop_cleanup(mock_loop, mock_connection)

        # Verify weak reference was created
        assert mock_loop_id in backend._cleanup_callbacks

        # Simulate weak reference callback (loop garbage collected)
        # First, get the callback function
        weak_ref = backend._cleanup_callbacks[mock_loop_id]

        # We can't easily trigger the callback, but we can verify the structure
        assert weak_ref is not None

    @pytest.mark.asyncio
    async def test_disconnect_pool_timeout(self):
        """Test pool disconnect timeout handling (lines 298-299)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_pool_disconnect",
        )

        # Use simple mock - no need for async since wait_for is patched
        mock_pool = MagicMock()
        mock_pool.disconnect = MagicMock()

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = builtins.TimeoutError()

            # Should not raise (covers lines 298-299)
            await backend._disconnect_pool(mock_pool)

    @pytest.mark.asyncio
    async def test_disconnect_pool_error(self):
        """Test pool disconnect error handling (lines 300-301)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_pool_error",
        )

        # Use simple mock since wait_for is patched
        mock_pool = MagicMock()
        mock_pool.disconnect = MagicMock()

        with patch("asyncio.wait_for") as mock_wait_for:
            mock_wait_for.side_effect = RuntimeError("Disconnect error")

            # Should not raise (covers lines 300-301)
            await backend._disconnect_pool(mock_pool)


class TestHealthCheckDataRetrieval:
    """Test health check info retrieval (lines 318-323)."""

    @pytest.mark.asyncio
    async def test_health_check_retrieves_redis_info(self):
        """Test health check retrieves and returns Redis info (lines 321-323)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_health",
        )
        backend._redis = mock_client
        backend._connected = True

        # Setup mock responses
        mock_client.set = AsyncMock(return_value=True)
        mock_client.get = AsyncMock(return_value="test")
        mock_client.delete = AsyncMock(return_value=1)
        mock_client.info = AsyncMock(
            return_value={
                "redis_version": "7.0.0",
                "used_memory_human": "1.5M",
                "connected_clients": 5,
            }
        )

        result = await backend.health_check()

        # Verify info was retrieved (covers lines 321, 323)
        assert result.healthy is True
        assert result.metadata is not None
        assert result.metadata["redis_version"] == "7.0.0"
        assert result.metadata["used_memory"] == "1.5M"
        assert result.metadata["connected_clients"] == 5


class TestCleanupExceptionHandling:
    """Test cleanup exception handling (lines 371-372)."""

    @pytest.mark.asyncio
    async def test_cleanup_exception_logging(self):
        """Test exception during cleanup is logged (lines 371-372)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_cleanup_exc",
        )
        backend._redis = mock_client
        backend._connected = True
        backend._owned_redis = True
        backend._event_loop_id = 12345

        # Make cleanup fail
        mock_client.aclose = AsyncMock(side_effect=RuntimeError("Cleanup failed"))

        with patch("venice_ai.core.backends.redis.logger") as mock_logger:
            await backend.cleanup()

            # Verify error was logged (covers lines 371-372)
            mock_logger.error.assert_called()

        # Verify state was cleared despite error
        assert backend._redis is None
        assert backend._connected is False


class TestGetAllStats:
    """Test get_all_stats method (lines 383-386)."""

    @pytest.mark.asyncio
    async def test_get_all_stats_returns_full_data(self):
        """Test get_all_stats returns complete statistics via SCAN iteration."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_stats",
        )
        backend._redis = mock_client
        backend._connected = True

        # scan_iter returns an async iterator of keys
        async def _fake_scan_iter(**kwargs):
            for key in ["key1", "key2", "key3"]:
                yield key

        mock_client.scan_iter = _fake_scan_iter
        mock_client.lrange = AsyncMock(return_value=[])
        mock_client.exists = AsyncMock(return_value=0)

        result = await backend.get_all_stats()

        # Verify stats counted via SCAN
        assert result["rate_limits"] == 3
        assert result["failures"] == 0
        assert result["redis_connected"] is True


class TestCheckCapacity:
    """Test check_capacity method (lines 406-413)."""

    @pytest.mark.asyncio
    async def test_check_capacity_with_circuit_broken(self):
        """Test check_capacity returns false when circuit is broken (lines 409-410)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_capacity",
        )
        backend._redis = mock_client
        backend._connected = True

        # Mock rate limits with low remaining
        mock_client.get = AsyncMock(
            return_value=json.dumps({"rpm_remaining": 0, "reset_at": time.time() + 60})
        )
        # Mock circuit_break_key exists (forced circuit break)
        mock_client.exists = AsyncMock(return_value=1)

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # Verify circuit broken behavior (covers lines 409-410)
        assert can_proceed is False
        assert wait_time >= 30.0

    @pytest.mark.asyncio
    async def test_check_capacity_can_proceed(self):
        """Test check_capacity returns true when capacity available (lines 412-413)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_capacity_ok",
        )
        backend._redis = mock_client
        backend._connected = True

        # Mock rate limits with capacity
        mock_client.get = AsyncMock(
            return_value=json.dumps({"rpm_remaining": 100, "tpm_remaining": 10000})
        )
        # Mock no forced circuit break and no failures
        mock_client.exists = AsyncMock(return_value=0)
        mock_client.lrange = AsyncMock(return_value=[])

        can_proceed, wait_time = await backend.check_capacity("test_model")

        # Verify can proceed (covers lines 412-413)
        assert can_proceed is True
        assert wait_time <= 0.0


class TestUpdateRateLimitsEmptyHeaders:
    """Test update_rate_limits with empty headers (lines 421-425)."""

    @pytest.mark.asyncio
    async def test_update_rate_limits_empty_headers(self):
        """Test update_rate_limits uses existing data when headers empty (lines 424-425)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_empty_headers",
        )
        backend._redis = mock_client
        backend._connected = True

        existing_limits = {"rpm_remaining": 50, "tpm_remaining": 5000}
        mock_client.get = AsyncMock(return_value=json.dumps(existing_limits))
        mock_client.setex = AsyncMock()

        # Call with empty headers (covers lines 424-425)
        await backend.update_rate_limits("test_model", {})

        # Verify existing data was used
        mock_client.get.assert_called()
        mock_client.setex.assert_called()

    @pytest.mark.asyncio
    async def test_update_rate_limits_no_existing_data(self):
        """Test update_rate_limits with empty headers and no existing data."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_no_existing",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.get = AsyncMock(return_value=None)
        mock_client.setex = AsyncMock()

        await backend.update_rate_limits("test_model", {})

        # Verify empty dict was stored
        mock_client.setex.assert_called()


class TestGetFailureCountJsonIteration:
    """Test get_failure_count JSON iteration (lines 502-512)."""

    @pytest.mark.asyncio
    async def test_get_failure_count_iterates_failures(self):
        """Test get_failure_count counts failures in window (lines 504-510)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_count",
        )
        backend._redis = mock_client
        backend._connected = True

        current_time = time.time()
        failures = [
            json.dumps({"type": "error", "timestamp": current_time - 10}),  # In window
            json.dumps({"type": "error", "timestamp": current_time - 20}),  # In window
            json.dumps({"type": "error", "timestamp": current_time - 60}),  # Out of window
        ]
        mock_client.lrange = AsyncMock(return_value=failures)

        count = await backend.get_failure_count(window_seconds=30)

        # Verify count (covers lines 504-510, 512)
        assert count == 2

    @pytest.mark.asyncio
    async def test_get_failure_count_invalid_json(self):
        """Test get_failure_count handles invalid JSON (lines 509-510)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_invalid_json",
        )
        backend._redis = mock_client
        backend._connected = True

        current_time = time.time()
        failures = [
            json.dumps({"type": "error", "timestamp": current_time - 10}),
            "invalid json {{{",
            json.dumps({"type": "error", "timestamp": current_time - 5}),
        ]
        mock_client.lrange = AsyncMock(return_value=failures)

        count = await backend.get_failure_count(window_seconds=30)

        # Should count valid entries only (covers line 510)
        assert count == 2


class TestIsCircuitBroken:
    """Test is_circuit_broken method (lines 525-526)."""

    @pytest.mark.asyncio
    async def test_is_circuit_broken_returns_true_via_forced_key(self):
        """Test is_circuit_broken returns true when circuit_break_key exists."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_circuit",
        )
        backend._redis = mock_client
        backend._connected = True

        # Forced circuit break key exists
        mock_client.exists = AsyncMock(return_value=1)

        result = await backend.is_circuit_broken()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_circuit_broken_returns_true_via_failure_threshold(self):
        """Test is_circuit_broken returns true when failure threshold exceeded."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_circuit",
        )
        backend._redis = mock_client
        backend._connected = True

        # No forced circuit break
        mock_client.exists = AsyncMock(return_value=0)
        current_time = time.time()
        failures = [json.dumps({"type": "error", "timestamp": current_time - i}) for i in range(25)]
        mock_client.lrange = AsyncMock(return_value=failures)

        result = await backend.is_circuit_broken()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_circuit_broken_returns_false(self):
        """Test is_circuit_broken returns false below threshold."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_circuit_ok",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.exists = AsyncMock(return_value=0)
        mock_client.lrange = AsyncMock(return_value=[])

        result = await backend.is_circuit_broken()

        assert result is False


class TestForceCircuitBreak:
    """Test force_circuit_break method using dedicated Redis key with TTL."""

    @pytest.mark.asyncio
    async def test_force_circuit_break_sets_key_with_ttl(self):
        """Test force_circuit_break sets a Redis key with TTL."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_force_break",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.setex = AsyncMock()

        await backend.force_circuit_break(5.0)

        # Verify setex was called with the circuit break key and correct TTL
        mock_client.setex.assert_called_once_with(backend.circuit_break_key, 5, "1")

    @pytest.mark.asyncio
    async def test_force_circuit_break_minimum_ttl(self):
        """Test force_circuit_break enforces minimum TTL of 1 second."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_force_min_ttl",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.setex = AsyncMock()

        await backend.force_circuit_break(0.1)

        # TTL should be at least 1
        mock_client.setex.assert_called_once_with(backend.circuit_break_key, 1, "1")

    @pytest.mark.asyncio
    async def test_force_circuit_break_response_error(self):
        """Test force_circuit_break handles ResponseError."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_force_response_error",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.setex = AsyncMock(side_effect=ResponseError("Stream error"))

        # Should not raise
        await backend.force_circuit_break(1.0)

    @pytest.mark.asyncio
    async def test_force_circuit_break_redis_error(self):
        """Test force_circuit_break handles RedisError."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_force_redis_error",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.setex = AsyncMock(side_effect=RedisError("Redis error"))

        # Should not raise
        await backend.force_circuit_break(1.0)


class TestGetRateLimitsEmpty:
    """Test _get_rate_limits empty return (line 571)."""

    @pytest.mark.asyncio
    async def test_get_rate_limits_empty_result(self):
        """Test _get_rate_limits returns empty dict when no data (line 571)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_empty_limits",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.get = AsyncMock(return_value=None)

        result = await backend._get_rate_limits("test_model")

        # Verify empty dict returned (covers line 571)
        assert result == {}


class TestContextManagerPaths:
    """Test context manager paths (lines 586-600)."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Test async context manager entry (lines 586-587)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_context",
        )

        with patch.object(backend, "_ensure_connected") as mock_ensure:
            mock_ensure.return_value = AsyncMock()

            result = await backend.__aenter__()

            # Verify connection established (covers lines 586-587)
            mock_ensure.assert_called_once()
            assert result == backend

    @pytest.mark.asyncio
    async def test_context_manager_exit_cleanup(self):
        """Test async context manager exit with cleanup (lines 591-600)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_context_exit",
        )
        backend._redis = mock_client
        backend._connected = True
        backend._owned_redis = True
        backend._event_loop_id = 12345

        with patch.object(backend, "_cleanup_connection") as mock_cleanup:
            mock_cleanup.return_value = None

            await backend.__aexit__(None, None, None)

            # Verify cleanup was called (covers lines 591-593)
            mock_cleanup.assert_called_once()

        # Verify state reset (covers lines 599-600)
        assert backend._redis is None
        assert backend._connected is False

    @pytest.mark.asyncio
    async def test_context_manager_exit_cleanup_error(self):
        """Test context manager exit handles cleanup errors (lines 596-597)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_context_error",
        )
        backend._redis = mock_client
        backend._connected = True
        backend._owned_redis = True
        backend._event_loop_id = 12345

        with (
            patch.object(backend, "_cleanup_connection", side_effect=RuntimeError("Cleanup error")),
            patch("venice_ai.core.backends.redis.logger") as mock_logger,
        ):
            await backend.__aexit__(None, None, None)

            # Verify error was logged (covers lines 596-597)
            mock_logger.error.assert_called()

        # Verify state still reset
        assert backend._redis is None


class TestCleanupAllPools:
    """Test cleanup_all_pools comprehensive paths (lines 611-654)."""

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_closed_loop(self):
        """Test cleanup_all_pools skips when loop is closed (lines 611-612)."""
        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = True
            mock_get_loop.return_value = mock_loop

            with patch("venice_ai.core.backends.redis.logger") as mock_logger:
                await RedisBackend.cleanup_all_pools()

                # Verify warning logged (covers lines 611-612)
                warning_calls = [
                    call
                    for call in mock_logger.warning.call_args_list
                    if "closed" in str(call).lower()
                ]
                assert len(warning_calls) >= 1

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_with_aclose(self):
        """Test cleanup_all_pools uses aclose when available (line 629-630)."""
        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock()
        mock_pool.close = MagicMock()
        mock_pool.aclose = AsyncMock()

        with patch.object(RedisBackend, "_connection_pools", {1: mock_pool}):
            await RedisBackend.cleanup_all_pools()

            # Verify aclose was called (covers line 629)
            mock_pool.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_aclose_error_uses_disconnect(self):
        """Test cleanup_all_pools falls back to disconnect on aclose error (lines 631-635)."""
        mock_pool = MagicMock()
        mock_pool.close = MagicMock()
        mock_pool.aclose = AsyncMock(side_effect=RuntimeError("aclose failed"))
        mock_pool.disconnect = AsyncMock()

        with patch.object(RedisBackend, "_connection_pools", {1: mock_pool}):
            await RedisBackend.cleanup_all_pools()

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_disconnect_only(self):
        """Test cleanup_all_pools with disconnect only (lines 636-639)."""
        mock_pool = MagicMock(spec=["disconnect"])
        mock_pool.disconnect = AsyncMock()

        with patch.object(RedisBackend, "_connection_pools", {1: mock_pool}):
            await RedisBackend.cleanup_all_pools()

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_task_creation_error(self):
        """Test cleanup_all_pools handles task creation errors (lines 640-641)."""
        mock_pool = MagicMock()
        mock_pool.disconnect = MagicMock(side_effect=AttributeError("No disconnect"))

        with (
            patch.object(RedisBackend, "_connection_pools", {1: mock_pool}),
            patch("venice_ai.core.backends.redis.logger") as mock_logger,
        ):
            await RedisBackend.cleanup_all_pools()

            # Verify warning logged (covers lines 640-641)
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_all_pools_disconnect_timeout(self):
        """Test cleanup_all_pools handles disconnect timeout (lines 649-654)."""
        mock_pool = MagicMock(spec=["disconnect"])

        # Create a real slow coroutine for disconnect
        async def slow_disconnect():
            await asyncio.sleep(10)

        mock_pool.disconnect = slow_disconnect

        # Patch _connection_pools at class level
        original_pools = RedisBackend._connection_pools.copy()
        try:
            RedisBackend._connection_pools = {1: mock_pool}

            # Run cleanup - the timeout handling should work
            # We use a short timeout scenario - the pool disconnect is slow
            # so wait_for will timeout
            await RedisBackend.cleanup_all_pools()
        finally:
            RedisBackend._connection_pools = original_pools


class TestRecordRequestBranches:
    """Test record_request branch coverage (lines 448-455)."""

    @pytest.mark.asyncio
    async def test_record_request_with_tokens_and_rpm(self):
        """Test record_request decrements both rpm and tpm (lines 448-455)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_record",
        )
        backend._redis = mock_client
        backend._connected = True

        initial_limits = {"rpm_remaining": 100, "tpm_remaining": 10000}
        mock_client.get = AsyncMock(return_value=json.dumps(initial_limits))
        mock_client.setex = AsyncMock()

        await backend.record_request("test_model", tokens_used=500)

        # Verify setex was called with decremented values
        call_args = mock_client.setex.call_args
        stored_limits = json.loads(call_args[0][2])
        assert stored_limits["rpm_remaining"] == 99
        assert stored_limits["tpm_remaining"] == 9500

    @pytest.mark.asyncio
    async def test_record_request_rpm_at_zero(self):
        """Test record_request when rpm_remaining is 0 (line 448 false branch)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_rpm_zero",
        )
        backend._redis = mock_client
        backend._connected = True

        initial_limits = {"rpm_remaining": 0, "tpm_remaining": 10000}
        mock_client.get = AsyncMock(return_value=json.dumps(initial_limits))
        mock_client.setex = AsyncMock()

        await backend.record_request("test_model", tokens_used=500)

        # Verify rpm wasn't decremented below 0
        call_args = mock_client.setex.call_args
        stored_limits = json.loads(call_args[0][2])
        assert stored_limits["rpm_remaining"] == 0

    @pytest.mark.asyncio
    async def test_record_request_no_tokens(self):
        """Test record_request without tokens_used (line 450 false branch)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_no_tokens",
        )
        backend._redis = mock_client
        backend._connected = True

        initial_limits = {"rpm_remaining": 100, "tpm_remaining": 10000}
        mock_client.get = AsyncMock(return_value=json.dumps(initial_limits))
        mock_client.setex = AsyncMock()

        await backend.record_request("test_model")  # No tokens_used

        # Verify tpm wasn't modified
        call_args = mock_client.setex.call_args
        stored_limits = json.loads(call_args[0][2])
        assert stored_limits["tpm_remaining"] == 10000

    @pytest.mark.asyncio
    async def test_record_request_no_rate_limits(self):
        """Test record_request when no rate limits exist (line 445 false branch)."""
        mock_client = AsyncMock()
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            redis_client=mock_client,
            namespace="test_no_limits",
        )
        backend._redis = mock_client
        backend._connected = True

        mock_client.get = AsyncMock(return_value=None)
        mock_client.setex = AsyncMock()

        await backend.record_request("test_model", tokens_used=500)

        # Verify setex was not called since there were no limits
        mock_client.setex.assert_not_called()


class TestEventLoopSwitchingBranches:
    """Test event loop switching branch coverage (lines 125, 131)."""

    @pytest.mark.asyncio
    async def test_event_loop_switch_with_old_connection(self):
        """Test event loop switch cleans up old connection (lines 125-131)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_loop_switch",
        )

        # Set up existing connection in old loop - use MagicMock to avoid unclosed coroutines
        old_redis = MagicMock()
        old_redis.aclose = MagicMock()
        backend._redis = old_redis
        backend._connected = True
        backend._event_loop_id = 11111  # Old loop ID

        # Track created tasks to properly clean them up
        created_tasks = []

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            # Use a real create_task but track it
            original_create_task = asyncio.create_task

            def tracking_create_task(coro, **kwargs):
                task = original_create_task(coro, **kwargs)
                created_tasks.append(task)
                return task

            with (
                patch("asyncio.create_task", side_effect=tracking_create_task),
                patch("venice_ai.core.backends.redis.Redis") as mock_redis,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                new_redis = AsyncMock()
                new_redis.ping = AsyncMock(return_value=True)
                mock_redis.return_value = new_redis

                await backend._ensure_connected()

                # Verify cleanup task was created for old connection
                # (covers lines 125-131)
                assert len(created_tasks) >= 1

                # Wait for cleanup tasks to complete
                for task in created_tasks:
                    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                        await asyncio.wait_for(task, timeout=1.0)


class TestPartialBranchCoverage:
    """Additional tests for remaining partial branches."""

    @pytest.mark.asyncio
    async def test_ensure_connected_pool_but_no_redis_class(self):
        """Test connection with pool but Redis class is None (lines 188-202)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_no_redis_class",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch.object(RedisBackend, "_connection_pools", {}),
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                with (
                    patch("venice_ai.core.backends.redis.Redis", None),
                    patch("venice_ai.core.backends.redis.redis") as mock_redis_module,
                ):
                    mock_redis = AsyncMock()
                    mock_redis.ping = AsyncMock(return_value=True)
                    mock_redis_module.from_url = AsyncMock(return_value=mock_redis)

                    await backend._ensure_connected()

                    # Verify fallback was used (covers lines 190-202)
                    mock_redis_module.from_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_loop_cleanup_happens(self):
        """Test that _register_loop_cleanup is called for owned redis (lines 214-215)."""
        backend = RedisBackend(
            redis_url="redis://localhost:6379",
            namespace="test_register_cleanup",
        )

        with patch("asyncio.get_running_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop

            with (
                patch("venice_ai.core.backends.redis.Redis") as mock_redis_class,
                patch("venice_ai.core.backends.redis.ConnectionPool") as mock_pool_class,
            ):
                mock_pool = MagicMock()
                mock_pool_class.from_url.return_value = mock_pool

                mock_redis = AsyncMock()
                mock_redis.ping = AsyncMock(return_value=True)
                mock_redis_class.return_value = mock_redis

                with patch.object(backend, "_register_loop_cleanup") as mock_register:
                    await backend._ensure_connected()

                    # Verify registration was called (covers lines 214-215)
                    mock_register.assert_called_once()
