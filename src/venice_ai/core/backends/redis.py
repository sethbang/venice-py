"""
Simplified RedisBackend for VeniceAccount operations.

This Redis backend provides distributed rate limit tracking and failure management.
For single-process applications, consider using MemoryBackend instead.
"""

import asyncio
import json
import logging
import threading
import time
import weakref
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool, Redis
    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import (
        ConnectionError,
        RedisError,
        ResponseError,
        TimeoutError,
    )

try:
    import redis.asyncio as redis
    from redis.asyncio import ConnectionPool, Redis
    from redis.asyncio.cluster import RedisCluster
    from redis.exceptions import (
        ConnectionError,
        RedisError,
        ResponseError,
        TimeoutError,
    )
except ImportError as _redis_import_error:  # pragma: no cover
    raise ImportError(
        "The 'redis' package is required to use RedisBackend. "
        "Install it with: pip install 'venice-py[redis]' or pip install redis"
    ) from _redis_import_error

import builtins
import contextlib

from .base import AccountBackend, HealthCheckResult

logger = logging.getLogger(__name__)


class RedisBackend(AccountBackend):
    """
    Redis backend for VeniceAccount with distributed failure tracking.

    Features:
    - Per-event-loop connection pooling
    - Thread-safe connection pool management
    - Distributed failure tracking and circuit breaker
    - Rate limit state caching from response headers
    """

    # Class-level connection pools indexed by event loop ID
    _connection_pools: ClassVar[dict[int, Any]] = {}
    _pool_lock: ClassVar[threading.Lock] = threading.Lock()
    _max_pools: ClassVar[int] = 20

    # WeakRef registry for automatic cleanup
    _loop_registry: ClassVar[weakref.WeakValueDictionary[int, asyncio.AbstractEventLoop]] = (
        weakref.WeakValueDictionary()
    )
    _cleanup_callbacks: ClassVar[dict[int, Any]] = {}

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_client: Any | None = None,
        namespace: str = "venice_ai",
        key_ttl: int = 3600,
        max_connections: int = 10,
        cluster_mode: bool = False,
    ) -> None:
        """
        Initialize Redis backend with connection pooling.

        Args:
            redis_url: Redis connection URL
            redis_client: Optional pre-configured Redis client
            namespace: Namespace for key isolation
            key_ttl: Default TTL for keys in seconds
            max_connections: Maximum connections per pool
            cluster_mode: Whether to use Redis Cluster client
        """
        super().__init__(namespace)

        self.redis_url = redis_url
        self.key_ttl = key_ttl
        self.max_connections = max_connections
        self.cluster_mode = cluster_mode

        self._redis: Any | None = redis_client
        self._owned_redis = redis_client is None
        self._event_loop_id: int | None = None

        # Key prefixes (simplified)
        self.key_prefix = f"venice:{namespace}"
        self.rate_limit_prefix = f"{self.key_prefix}:rate_limits"
        self.failure_prefix = f"{self.key_prefix}:failures"
        self.circuit_break_key = f"{self.key_prefix}:circuit_break_until"

        # Connection state
        self._connected = False
        self._connection_lock = asyncio.Lock()

    # === Connection Management ===

    async def _ensure_connected(self) -> Any:
        """
        Ensure Redis connection with proper cleanup on loop switches.

        Returns:
            Redis: Connected Redis client

        Raises:
            RuntimeError: If no event loop is running
        """
        try:
            current_loop = asyncio.get_running_loop()
            loop_id = id(current_loop)

            # Check if we need to switch pools (event loop changed)
            if self._event_loop_id and self._event_loop_id != loop_id:
                old_loop_id = self._event_loop_id

                # The old client's connections belong to the old loop and cannot
                # be awaited from this one. The old loop's pool is dropped when
                # that loop is garbage collected.
                logger.debug(
                    f"Event loop changed from {old_loop_id} to {loop_id}, "
                    "releasing the old client and switching connection pool"
                )

                self._redis = None
                self._connected = False

            self._event_loop_id = loop_id

            # Return existing connection if valid
            if self._redis and self._connected:
                try:
                    await self._redis.ping()
                    return self._redis
                except (ConnectionError, Exception):
                    logger.warning("Redis connection lost, reconnecting...")
                    self._redis = None
                    self._connected = False

            # Create or reuse connection for this event loop
            async with self._connection_lock:
                if self.cluster_mode:
                    if self._redis is None:
                        self._redis = RedisCluster.from_url(
                            self.redis_url,
                            decode_responses=True,
                        )
                        logger.info(f"Created Redis Cluster client for event loop {loop_id}")
                else:
                    with self._pool_lock:
                        if (
                            len(self._connection_pools) >= self._max_pools
                            and loop_id not in self._connection_pools
                        ):
                            logger.warning(
                                f"Connection pool limit ({self._max_pools}) reached. "
                                "Old event loops will be cleaned up automatically."
                            )

                        if loop_id not in self._connection_pools and ConnectionPool is not None:
                            self._connection_pools[loop_id] = ConnectionPool.from_url(
                                self.redis_url,
                                max_connections=self.max_connections,
                                decode_responses=True,
                                socket_connect_timeout=5,
                                socket_timeout=5,
                                retry_on_timeout=True,
                                health_check_interval=30,
                            )
                            logger.info(
                                f"Created Redis connection pool for event loop {loop_id} "
                                f"(total pools: {len(self._connection_pools)})"
                            )

                        pool = self._connection_pools.get(loop_id)

                    if pool and Redis is not None:
                        self._redis = Redis(connection_pool=pool)
                    elif redis:
                        self._redis = await redis.from_url(
                            self.redis_url,
                            encoding="utf-8",
                            decode_responses=True,
                            socket_connect_timeout=5,
                            socket_timeout=5,
                            retry_on_timeout=True,
                            retry_on_error=[ConnectionError]
                            if ConnectionError is not Exception
                            else [],
                            max_connections=self.max_connections,
                        )

                if self._redis:
                    try:
                        await asyncio.wait_for(self._redis.ping(), timeout=5.0)
                    except builtins.TimeoutError:
                        logger.error("Redis ping timed out after 5s")
                        raise ConnectionError("Redis ping timed out") from None

                    self._connected = True
                    logger.debug(f"Redis connected successfully in loop {loop_id}")

                    if self._redis and self._owned_redis:
                        self._register_loop_cleanup(current_loop, self._redis)

                return self._redis

        except RuntimeError as e:
            if "no running event loop" in str(e).lower():
                raise RuntimeError(
                    "No running event loop for Redis connection. "
                    "Ensure async context is properly initialized."
                ) from e
            raise
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except RedisError as e:
            logger.error(f"Redis error during connection: {e}")
            raise

    async def _cleanup_connection(
        self, loop_id: int, connection: Any, timeout: float = 2.5
    ) -> None:
        """Clean up Redis connection with timeout protection."""
        logger.debug(f"Cleaning up connection from event loop {loop_id}")

        try:
            if hasattr(connection, "aclose"):
                await asyncio.wait_for(connection.aclose(), timeout=timeout)
            elif hasattr(connection, "close"):
                await asyncio.wait_for(connection.close(), timeout=timeout)
            logger.debug(f"Successfully closed connection from loop {loop_id}")

        except builtins.TimeoutError:
            logger.warning(
                f"Connection cleanup timed out after {timeout}s for loop {loop_id}. "
                "Connection will be force-closed."
            )
            try:
                if hasattr(connection, "close"):
                    connection.close()
            except Exception as e:
                logger.error(f"Force close failed for loop {loop_id}: {e}")

        except Exception as e:
            logger.error(
                f"Unexpected error during connection cleanup for loop {loop_id}: {e}",
                exc_info=True,
            )

    def _register_loop_cleanup(self, loop: asyncio.AbstractEventLoop, connection: Any) -> None:
        """Register cleanup callback for when event loop is garbage collected."""
        loop_id = id(loop)

        def cleanup_callback(_weak_loop_ref: Any) -> None:
            # Runs from the garbage collector, possibly inside another loop or
            # while _pool_lock is held on this thread. The dead loop's
            # connections cannot be awaited anywhere, so only the references
            # are released: no task is scheduled and no lock is taken.
            RedisBackend._cleanup_callbacks.pop(loop_id, None)
            pool = RedisBackend._connection_pools.pop(loop_id, None)
            if self._event_loop_id == loop_id and self._redis is connection:
                self._redis = None
                self._connected = False
            logger.debug(
                f"Event loop {loop_id} was garbage collected; released its Redis client"
                f"{' and connection pool' if pool is not None else ''}"
            )

        weak_loop = weakref.ref(loop, cleanup_callback)
        self._cleanup_callbacks[loop_id] = weak_loop
        self._loop_registry[loop_id] = loop

    async def _disconnect_pool(self, pool: Any) -> None:
        """Asynchronously disconnect a connection pool."""
        try:
            if hasattr(pool, "disconnect"):
                await asyncio.wait_for(pool.disconnect(), timeout=2.5)
                logger.debug("Connection pool disconnected successfully")
        except builtins.TimeoutError:
            logger.warning("Pool disconnect timed out after 2.5s, forcing closure")
        except Exception as e:
            logger.error(f"Error disconnecting pool: {e}")

    # === Key Helpers ===

    def _get_rate_limit_key(self, model: str) -> str:
        """Get Redis key for rate limit storage."""
        return f"{self.rate_limit_prefix}:{model}"

    # === 11 Required Methods from AccountBackend ===

    async def health_check(self) -> HealthCheckResult:
        """Perform health check."""
        try:
            redis_client = await self._ensure_connected()

            test_key = f"health_check_{int(time.time())}"
            await redis_client.set(test_key, "test", ex=60)
            result = await redis_client.get(test_key)
            await redis_client.delete(test_key)

            info = await redis_client.info()

            return HealthCheckResult(
                healthy=result == "test",
                backend_type="redis",
                namespace=self.namespace,
                metadata={
                    "redis_url": self.redis_url,
                    "connected": self._connected,
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                },
            )
        except (ConnectionError, TimeoutError) as e:
            return HealthCheckResult(
                healthy=False,
                backend_type="redis",
                namespace=self.namespace,
                error=f"Connection error: {e}",
            )
        except ResponseError as e:
            return HealthCheckResult(
                healthy=False,
                backend_type="redis",
                namespace=self.namespace,
                error=f"Redis response error: {e}",
            )
        except RedisError as e:
            return HealthCheckResult(
                healthy=False,
                backend_type="redis",
                namespace=self.namespace,
                error=f"Redis error: {e}",
            )
        except (AttributeError, ValueError, TypeError, OSError) as e:
            return HealthCheckResult(
                healthy=False,
                backend_type="redis",
                namespace=self.namespace,
                error=f"Unexpected error: {e}",
            )

    async def cleanup(self) -> None:
        """Clean up Redis connection with proper async cleanup."""
        if self._redis and self._owned_redis:
            try:
                await self._cleanup_connection(self._event_loop_id or 0, self._redis, timeout=2.5)
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                self._redis = None
                self._event_loop_id = None
                self._connected = False

    async def get_all_stats(self) -> dict[str, Any]:
        """Get all statistics."""
        try:
            redis_client = await self._ensure_connected()

            rate_limit_count = 0
            async for _key in redis_client.scan_iter(
                match=f"{self.rate_limit_prefix}:*", count=100
            ):
                rate_limit_count += 1
            failure_count = await self.get_failure_count()

            return {
                "rate_limits": rate_limit_count,
                "failures": failure_count,
                "circuit_broken": await self.is_circuit_broken(),
                "redis_connected": self._connected,
            }
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error getting stats: {e}")
            return {"rate_limits": 0, "failures": 0, "error": str(e)}
        except (ResponseError, ValueError) as e:
            logger.error(f"Redis data error getting stats: {e}")
            return {"rate_limits": 0, "failures": 0, "error": str(e)}
        except RedisError as e:
            logger.error(f"Redis error getting stats: {e}")
            return {"rate_limits": 0, "failures": 0, "error": str(e)}

    async def check_capacity(self, model: str, request_type: str = "default") -> tuple[bool, float]:
        """Check capacity for a model."""
        del request_type  # interface-only; Redis backend ignores request_type
        rate_limits = await self._get_rate_limits(model)
        wait_time = self._calculate_wait_time(rate_limits)

        if await self.is_circuit_broken():
            return False, max(wait_time, 30.0)

        can_proceed = wait_time <= 0.0
        return can_proceed, wait_time

    async def update_rate_limits(self, model: str, headers: dict[str, str]) -> None:
        """Update rate limits from response headers."""
        try:
            redis_client = await self._ensure_connected()
            rate_limit_key = self._get_rate_limit_key(model)

            if headers:
                parsed_limits = self._parse_rate_limit_headers(headers)
            else:
                existing_json = await redis_client.get(rate_limit_key)
                parsed_limits = json.loads(existing_json) if existing_json else {}

            await redis_client.setex(rate_limit_key, self.key_ttl, json.dumps(parsed_limits))
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error updating rate limits for {model}: {e}")
        except (ResponseError, ValueError) as e:
            logger.error(f"Redis data error updating rate limits for {model}: {e}")
        except RedisError as e:
            logger.error(f"Redis error updating rate limits for {model}: {e}")

    async def record_request(self, model: str, tokens_used: int | None = None) -> None:
        """Record a request."""
        try:
            rate_limits = await self._get_rate_limits(model)
            if rate_limits:
                current_time = time.time()

                if "rpm_remaining" in rate_limits and rate_limits["rpm_remaining"] > 0:
                    rate_limits["rpm_remaining"] -= 1
                if tokens_used and "tpm_remaining" in rate_limits:
                    rate_limits["tpm_remaining"] = max(
                        0, rate_limits["tpm_remaining"] - tokens_used
                    )

                rate_limits["last_request"] = current_time

                redis_client = await self._ensure_connected()
                rate_limit_key = self._get_rate_limit_key(model)
                await redis_client.setex(rate_limit_key, self.key_ttl, json.dumps(rate_limits))
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error recording request for {model}: {e}")
        except (ResponseError, ValueError) as e:
            logger.error(f"Redis data error recording request for {model}: {e}")
        except RedisError as e:
            logger.error(f"Redis error recording request for {model}: {e}")

    async def release_streaming_reservation(
        self,
        bucket_id: str,
        reservation_id: str,
        reserved_tokens: int,
        actual_tokens: int,
    ) -> bool:
        """Release streaming reservation with refund-based accounting.

        For RedisBackend, this is largely a no-op since the simplified
        backend doesn't track individual reservations. Returns True to
        indicate success for protocol compatibility.
        """
        # RedisBackend doesn't track individual reservations
        # This is a no-op implementation for protocol compatibility
        logger.debug(
            f"release_streaming_reservation: bucket={bucket_id}, "
            f"reservation={reservation_id}, refund={reserved_tokens - actual_tokens}"
        )
        return True

    async def record_failure(self, error_type: str, error_message: str = "") -> None:
        """Record a failure."""
        try:
            redis_client = await self._ensure_connected()
            current_time = time.time()

            failure_record = {
                "type": error_type,
                "message": error_message,
                "timestamp": current_time,
            }

            failure_key = f"{self.failure_prefix}:log"
            await redis_client.lpush(failure_key, json.dumps(failure_record))
            await redis_client.expire(failure_key, 3600)
            await redis_client.ltrim(failure_key, 0, 999)
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error recording failure: {e}")
        except ResponseError as e:
            logger.error(f"Redis response error recording failure: {e}")
        except RedisError as e:
            logger.error(f"Redis error recording failure: {e}")

    async def get_failure_count(self, window_seconds: int = 30) -> int:
        """Get failure count in time window."""
        try:
            redis_client = await self._ensure_connected()
            failure_key = f"{self.failure_prefix}:log"

            current_time = time.time()
            cutoff_time = current_time - window_seconds

            failures = await redis_client.lrange(failure_key, 0, -1)
            count = 0

            for failure_json in failures:
                try:
                    failure = json.loads(failure_json)
                    if failure.get("timestamp", 0) > cutoff_time:
                        count += 1
                except json.JSONDecodeError:
                    continue

            return count
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error getting failure count: {e}")
            return 0
        except (ResponseError, ValueError) as e:
            logger.error(f"Redis data error getting failure count: {e}")
            return 0
        except RedisError as e:
            logger.error(f"Redis error getting failure count: {e}")
            return 0

    async def is_circuit_broken(self) -> bool:
        """Check if circuit breaker is triggered.

        Returns True if either:
        - A forced circuit break is active (circuit_break_until key exists), or
        - Failure count exceeds threshold (20 failures in 30s)
        """
        try:
            redis_client = await self._ensure_connected()
            forced = await redis_client.exists(self.circuit_break_key)
            if forced:
                return True
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error checking circuit break key: {e}")
        except (ResponseError, RedisError) as e:
            logger.error(f"Redis error checking circuit break key: {e}")

        failure_count = await self.get_failure_count(30)
        return failure_count >= 20

    async def clear_failures(self) -> None:
        """Clear all failure records and any forced circuit break."""
        try:
            redis_client = await self._ensure_connected()
            failure_key = f"{self.failure_prefix}:log"
            await redis_client.delete(failure_key, self.circuit_break_key)
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error clearing failures: {e}")
        except ResponseError as e:
            logger.error(f"Redis response error clearing failures: {e}")
        except RedisError as e:
            logger.error(f"Redis error clearing failures: {e}")

    async def force_circuit_break(self, duration: float) -> None:
        """Force circuit break for the specified duration.

        Sets a dedicated Redis key with a TTL equal to ``duration``.
        :meth:`is_circuit_broken` checks this key, so no fake failures
        are injected.

        Args:
            duration: Duration in seconds to keep circuit broken
        """
        try:
            redis_client = await self._ensure_connected()
            ttl = max(int(duration), 1)
            await redis_client.setex(self.circuit_break_key, ttl, "1")
            logger.debug(f"Forced circuit break for {duration}s (TTL={ttl})")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error forcing circuit break: {e}")
        except ResponseError as e:
            logger.error(f"Redis response error forcing circuit break: {e}")
        except RedisError as e:
            logger.error(f"Redis error forcing circuit break: {e}")

    # === Private Helpers ===

    async def _get_rate_limits(self, model: str) -> dict[str, Any]:
        """Get rate limits for a model (private helper)."""
        try:
            redis_client = await self._ensure_connected()
            redis_key = self._get_rate_limit_key(model)

            limits_json = await redis_client.get(redis_key)
            if limits_json:
                result: dict[str, Any] = json.loads(limits_json)
                return result
            return {}
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error getting rate limits for {model}: {e}")
            return {}
        except (ResponseError, ValueError) as e:
            logger.error(f"Redis data error getting rate limits for {model}: {e}")
            return {}
        except RedisError as e:
            logger.error(f"Redis error getting rate limits for {model}: {e}")
            return {}

    # === Context Managers ===

    async def __aenter__(self) -> "RedisBackend":
        """Async context manager entry."""
        await self._ensure_connected()
        return self

    async def __aexit__(self, exc_type: Any, _exc_val: Any, _exc_tb: Any) -> None:
        """Async context manager exit with guaranteed cleanup."""
        if self._redis and self._owned_redis:
            try:
                await self._cleanup_connection(self._event_loop_id or 0, self._redis, timeout=2.5)
            except Exception as e:
                logger.error(f"Error during context manager cleanup: {e}")
            finally:
                self._redis = None
                self._connected = False

    @classmethod
    async def cleanup_all_pools(cls) -> None:
        """
        Clean up all connection pools across all event loops.

        Call this at the end of test sessions to ensure clean shutdown.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                logger.warning("Event loop is closed, skipping pool cleanup")
                return
        except RuntimeError:
            logger.warning("No event loop available, skipping pool cleanup")
            return

        pools_to_disconnect = []
        with cls._pool_lock:
            for loop_id, pool in cls._connection_pools.items():
                pools_to_disconnect.append((loop_id, pool))
            cls._connection_pools.clear()

        disconnect_tasks = []
        for loop_id, pool in pools_to_disconnect:
            try:
                if hasattr(pool, "disconnect") and hasattr(pool, "close"):
                    try:
                        await pool.aclose() if hasattr(pool, "aclose") else pool.close()
                        logger.info(f"Closed pool for loop {loop_id}")
                    except Exception as e:
                        logger.debug(f"Error closing pool {loop_id}: {e}")
                        task = asyncio.create_task(pool.disconnect())
                        disconnect_tasks.append((loop_id, task))
                        logger.info(f"Scheduled disconnect for pool {loop_id}")
                elif hasattr(pool, "disconnect"):
                    task = asyncio.create_task(pool.disconnect())
                    disconnect_tasks.append((loop_id, task))
                    logger.info(f"Scheduled disconnect for pool {loop_id}")
            except (AttributeError, RuntimeError) as e:
                logger.warning(f"Error creating disconnect task for pool {loop_id}: {e}")

        for loop_id, task in disconnect_tasks:
            try:
                await asyncio.wait_for(task, timeout=2.0)
                logger.info(f"Disconnected pool for loop {loop_id}")
            except builtins.TimeoutError:
                logger.warning(f"Timeout disconnecting pool for loop {loop_id}")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except (ConnectionError, ResponseError, TimeoutError) as e:
                logger.debug(f"Redis error disconnecting pool for loop {loop_id}: {e}")
            except (AttributeError, RuntimeError, OSError) as e:
                logger.debug(f"Unexpected error disconnecting pool for loop {loop_id}: {e}")

        logger.info("Cleaned up all Redis connection pools")
