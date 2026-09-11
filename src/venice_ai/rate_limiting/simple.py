"""
SimpleRateLimiter - Lightweight, in-memory rate limiting for Venice AI SDK.

This module provides:
- Per-model rate limit state tracking (from response headers)
- Exponential backoff with jitter on 429 responses
- Global abuse protection (blocks all requests after threshold failures)
- Automatic cleanup of stale model state
- Memory bounds via max_models limit

For production deployments requiring distributed coordination,
use ADAPTIVE mode with the adaptive-rate-limiter package.
"""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

from ..utils.parsing import ms_epoch_to_seconds

if TYPE_CHECKING:
    from .._queue_types import RequestMetadata

logger = logging.getLogger(__name__)


class RateLimiterProtocol(Protocol):
    """
    Protocol defining the rate limiter interface for VeniceClient.

    Both SimpleRateLimiter and AdaptiveScheduler must implement this interface.
    The key method is submit_request() which orchestrates the complete request lifecycle.

    Note: acquire() and update_from_headers() are NOT part of this protocol.
    They are implementation details used internally by SimpleRateLimiter.
    AdaptiveScheduler handles rate limiting through its mode strategies instead.
    VeniceClient only uses submit_request(), lifecycle methods, and classifier.
    """

    async def submit_request(
        self,
        metadata: "RequestMetadata",
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Callable[..., Exception] | None = None,
    ) -> Any:
        """
        Submit a request for rate-limited execution.

        This is THE PRIMARY interface method. It must:
        1. Check rate limit state and wait if needed
        2. Execute the request via request_func
        3. Inspect response status - if 429, create RateLimitError using error_factory
        4. Update state from response/error headers
        5. Handle 429 errors with retry logic
        6. Return the response or re-raise after max retries

        Args:
            metadata: Request metadata containing model_id, endpoint, etc.
            request_func: Async callable that executes the actual HTTP request.
                          Returns raw response (including 429s).
            error_factory: Optional callable to create errors from response.
                          Signature: (message, request, body, response) -> Exception
                          If provided, used to create RateLimitError for 429 responses.
                          If not provided, a default RateLimitError is created.
        """
        ...

    def is_running(self) -> bool:
        """Check if the rate limiter is running."""
        ...

    async def start(self) -> None:
        """Start the rate limiter."""
        ...

    async def stop(self) -> None:
        """Stop the rate limiter."""
        ...

    @property
    def classifier(self) -> Any | None:
        """Optional request classifier for VeniceClient compatibility."""
        ...


@dataclass
class ModelBucketState:
    """
    Per-model rate limit state.

    Tracks rate limits from response headers and backoff state.
    """

    model: str

    # Rate limit values from headers
    rpm_limit: int = 0
    rpm_remaining: int = 0
    rpm_reset: float = 0.0
    tpm_limit: int = 0
    tpm_remaining: int = 0
    tpm_reset: float = 0.0

    # Backoff state
    consecutive_failures: int = 0
    backoff_until: float = 0.0

    # Timestamps
    last_updated: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def is_rate_limited(self) -> tuple[bool, float]:
        """
        Check if this model is currently rate limited.

        Returns:
            Tuple of (is_limited, wait_time_seconds)
        """
        now = time.time()
        self.last_accessed = now

        # Check backoff first (from 429 responses)
        if now < self.backoff_until:
            return True, self.backoff_until - now

        # Check if we've exhausted requests (from headers)
        if self.rpm_remaining == 0 and self.rpm_limit > 0 and now < self.rpm_reset:
            return True, self.rpm_reset - now

        return False, 0.0


class SimpleRateLimiter:
    """
    Lightweight, in-memory rate limiter for Venice AI SDK.

    Features:
    - Per-model rate limit state tracking (from response headers)
    - Exponential backoff with jitter on 429 responses
    - Global abuse protection (blocks all requests after threshold failures)
    - Automatic cleanup of stale model state
    - Memory bounds via max_models limit

    Limitations (by design):
    - Single-process only (no cross-worker coordination)
    - Reactive only (responds to 429s, does not prevent them)
    - No token prediction (cannot estimate before request)
    - No cold-start protection (concurrent cold starts may stampede)

    For production deployments, use ADAPTIVE mode with adaptive-rate-limiter.
    """

    # Constants
    BACKOFF_MULTIPLIER = 2.0

    #: Length of the rolling window behind the two error budgets (too many
    #: failed requests, too many unsupported-feature requests). A 429 from one
    #: of those clears only when the window rolls, so backing off for less is
    #: guaranteed to fail again — and for the failed-request budget, each such
    #: retry counts against the budget, digging the hole deeper. These budgets
    #: exist specifically to stop clients retrying into a wall.
    ERROR_BUDGET_WINDOW_SECONDS = 30.0
    BACKOFF_JITTER = 0.1  # ±10% jitter
    CLEANUP_INTERVAL = 300.0  # 5 minutes

    def __init__(
        self,
        min_backoff: float = 1.0,
        max_backoff: float = 60.0,
        failure_threshold: int = 20,
        failure_window: float = 30.0,
        block_duration: float = 30.0,
        max_models: int = 1000,
        stale_threshold: float = 3600.0,
        max_retries: int = 3,
    ):
        """
        Initialize SimpleRateLimiter.

        Args:
            min_backoff: Minimum backoff time in seconds (default: 1.0)
            max_backoff: Maximum backoff time in seconds (default: 60.0)
            failure_threshold: Number of failures within window to trigger global block (default: 20)
            failure_window: Time window for counting failures in seconds (default: 30.0)
            block_duration: Duration of global block in seconds (default: 30.0)
            max_models: Maximum number of models to track (default: 1000)
            stale_threshold: Time after which unused models are cleaned up (default: 3600.0)
            max_retries: Maximum number of retry attempts for 429 responses (default: 3)
        """
        self._model_states: dict[str, ModelBucketState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()

        # Global failure tracking
        self._global_failures: list[float] = []
        self._global_block_until: float = 0.0

        # Configuration
        self.min_backoff = min_backoff
        self.max_backoff = max_backoff
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window
        self.block_duration = block_duration
        self.max_models = max_models
        self.stale_threshold = stale_threshold
        self.max_retries = max_retries

        # Cleanup tracking
        self._last_cleanup: float = 0.0

        # Lifecycle state (for VeniceClient compatibility)
        self._running: bool = False
        self._classifier: Any | None = None

    # ==========================================================================
    # Lifecycle Methods (VeniceClient compatibility)
    # ==========================================================================

    def is_running(self) -> bool:
        """Check if the rate limiter is running."""
        return self._running

    async def start(self) -> None:
        """Start the rate limiter."""
        self._running = True
        logger.debug("SimpleRateLimiter started")

    async def stop(self) -> None:
        """Stop the rate limiter and cleanup."""
        self._running = False
        await self.clear()
        logger.debug("SimpleRateLimiter stopped")

    @property
    def classifier(self) -> Any | None:
        """Optional request classifier for VeniceClient compatibility."""
        return self._classifier

    @classifier.setter
    def classifier(self, value: Any) -> None:
        """Set the request classifier."""
        self._classifier = value

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _get_lock(self, model: str) -> asyncio.Lock:
        """
        Get or create lock for a model.

        Uses setdefault() for atomic get-or-create to avoid race conditions
        when multiple coroutines access the same model simultaneously.
        """
        return self._locks.setdefault(model, asyncio.Lock())

    def _get_state(self, model: str) -> ModelBucketState:
        """Get or create state for a model."""
        if model not in self._model_states:
            self._model_states[model] = ModelBucketState(model=model)
        return self._model_states[model]

    async def _maybe_cleanup(self) -> None:
        """
        Periodically clean up stale model state to prevent unbounded memory growth.

        This runs at most once per CLEANUP_INTERVAL (5 minutes).
        """
        now = time.time()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return

        async with self._global_lock:
            # Double-check after acquiring lock
            if now - self._last_cleanup < self.CLEANUP_INTERVAL:
                return

            self._last_cleanup = now

            # Find stale models
            stale_models = [
                model
                for model, state in self._model_states.items()
                if now - state.last_accessed > self.stale_threshold
            ]

            # Remove stale model state and attempt lock cleanup
            for model in stale_models:
                del self._model_states[model]
                if model in self._locks:
                    lock = self._locks[model]
                    if not lock.locked():
                        del self._locks[model]

            if stale_models:
                logger.debug(f"Cleaned up {len(stale_models)} stale model states")

            # If still over limit, remove least recently accessed
            if len(self._model_states) > self.max_models:
                sorted_models = sorted(self._model_states.items(), key=lambda x: x[1].last_accessed)
                excess = len(self._model_states) - self.max_models
                for model, _ in sorted_models[:excess]:
                    del self._model_states[model]
                    if model in self._locks:
                        lock = self._locks[model]
                        if not lock.locked():
                            del self._locks[model]
                logger.warning(
                    f"Rate limiter exceeded max_models ({self.max_models}), "
                    f"removed {excess} least recently used"
                )

    async def acquire(self, model: str) -> tuple[bool, float]:
        """
        Attempt to acquire permission to make a request.

        Args:
            model: Model identifier

        Returns:
            Tuple of (can_proceed, wait_time_seconds)
        """
        await self._maybe_cleanup()

        # Check global block first
        now = time.time()
        if now < self._global_block_until:
            return False, self._global_block_until - now

        async with self._get_lock(model):
            state = self._get_state(model)
            is_limited, wait_time = state.is_rate_limited()

            if is_limited:
                return False, wait_time

            return True, 0.0

    async def update_from_headers(
        self, model: str, headers: dict[str, str], status_code: int = 200
    ) -> None:
        """
        Update rate limit state from response headers.

        Args:
            model: Model identifier
            headers: Response headers (case-insensitive)
            status_code: HTTP status code
        """
        async with self._get_lock(model):
            state = self._get_state(model)

            # Normalize headers to lowercase
            normalized = {k.lower(): v for k, v in headers.items()}

            # Parse rate limit headers
            if "x-ratelimit-limit-requests" in normalized:
                state.rpm_limit = int(normalized["x-ratelimit-limit-requests"])
            if "x-ratelimit-remaining-requests" in normalized:
                state.rpm_remaining = int(normalized["x-ratelimit-remaining-requests"])
            if "x-ratelimit-reset-requests" in normalized:
                state.rpm_reset = self._parse_reset_time(normalized["x-ratelimit-reset-requests"])

            if "x-ratelimit-limit-tokens" in normalized:
                state.tpm_limit = int(normalized["x-ratelimit-limit-tokens"])
            if "x-ratelimit-remaining-tokens" in normalized:
                state.tpm_remaining = int(normalized["x-ratelimit-remaining-tokens"])
            if "x-ratelimit-reset-tokens" in normalized:
                state.tpm_reset = self._parse_reset_time(normalized["x-ratelimit-reset-tokens"])

            state.last_updated = time.time()
            state.last_accessed = time.time()

            # Handle 429 response
            if status_code == 429:
                await self._apply_backoff(state, normalized)

    async def _apply_backoff(self, state: ModelBucketState, headers: dict[str, str]) -> None:
        """Apply exponential backoff after a 429 response."""
        state.consecutive_failures += 1

        # Check for retry-after header first
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                wait_time = float(retry_after)
            except ValueError:
                wait_time = self._calculate_backoff(state.consecutive_failures)
        else:
            wait_time = self._calculate_backoff(state.consecutive_failures)

        state.backoff_until = time.time() + wait_time

        logger.info(
            f"Rate limited on model {state.model}, "
            f"backing off for {wait_time:.1f}s "
            f"(attempt {state.consecutive_failures})"
        )

        await self._record_global_failure()

    def _calculate_backoff(self, failures: int) -> float:
        """Calculate exponential backoff with jitter."""
        backoff = self.min_backoff * (self.BACKOFF_MULTIPLIER ** (failures - 1))
        backoff = min(backoff, self.max_backoff)

        # Add jitter (±10%) — non-cryptographic; spreads concurrent retries.
        jitter = backoff * self.BACKOFF_JITTER * (2 * random.random() - 1)  # nosec B311
        return backoff + jitter

    async def _record_global_failure(self) -> None:
        """Record a failure for global abuse protection."""
        async with self._global_lock:
            now = time.time()

            # Clean old failures
            self._global_failures = [
                t for t in self._global_failures if now - t < self.failure_window
            ]

            self._global_failures.append(now)

            # Check threshold
            if len(self._global_failures) >= self.failure_threshold:
                self._global_block_until = now + self.block_duration
                self._global_failures.clear()
                logger.warning(
                    f"Global failure threshold reached ({self.failure_threshold}), "
                    f"blocking all requests for {self.block_duration}s"
                )

    async def record_failure(self, model: str) -> None:
        """Record a failure for the given model."""
        async with self._get_lock(model):
            state = self._get_state(model)
            state.consecutive_failures += 1
            wait_time = self._calculate_backoff(state.consecutive_failures)
            state.backoff_until = time.time() + wait_time

        await self._record_global_failure()

    async def record_success(self, model: str) -> None:
        """Record a success, resetting the failure count."""
        async with self._get_lock(model):
            state = self._get_state(model)
            state.consecutive_failures = 0
            state.backoff_until = 0.0

    def _parse_reset_time(self, reset_str: str) -> float:
        """Parse a rate-limit reset header into an absolute Unix-seconds time.

        Absolute epochs (>= 1e9, i.e. year-2001+) are normalized via the
        canonical ms-epoch policy (:func:`ms_epoch_to_seconds`: values >= 1e12
        are milliseconds). Smaller values are relative delta-seconds from now.
        Aligns with ``VeniceBaseModel._ms_to_seconds`` so reset-header handling
        is uniform across the SDK.
        """
        try:
            val = float(reset_str)
        except ValueError:
            # Default to 60 seconds
            return time.time() + 60.0

        # Absolute epoch (seconds or milliseconds) vs. relative delta.
        if val >= 1e9:
            normalized = ms_epoch_to_seconds(val)
            return normalized if normalized is not None else time.time() + 60.0
        return time.time() + val

    async def get_state(self, model: str) -> dict[str, Any] | None:
        """Get the current state for a model (for debugging)."""
        async with self._get_lock(model):
            if model not in self._model_states:
                return None
            state = self._model_states[model]
            return {
                "model": state.model,
                "rpm_limit": state.rpm_limit,
                "rpm_remaining": state.rpm_remaining,
                "rpm_reset": state.rpm_reset,
                "tpm_limit": state.tpm_limit,
                "tpm_remaining": state.tpm_remaining,
                "tpm_reset": state.tpm_reset,
                "consecutive_failures": state.consecutive_failures,
                "backoff_until": state.backoff_until,
                "last_updated": state.last_updated,
                "last_accessed": state.last_accessed,
            }

    async def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Get all model states (for debugging)."""
        result = {}
        for model in list(self._model_states.keys()):
            state = await self.get_state(model)
            if state:
                result[model] = state
        return result

    async def clear(self) -> None:
        """Clear all state."""
        async with self._global_lock:
            self._model_states.clear()
            self._locks.clear()
            self._global_failures.clear()
            self._global_block_until = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get limiter statistics."""
        return {
            "tracked_models": len(self._model_states),
            "tracked_locks": len(self._locks),
            "max_models": self.max_models,
            "global_failures": len(self._global_failures),
            "global_blocked": time.time() < self._global_block_until,
            "last_cleanup": self._last_cleanup,
        }

    # ==========================================================================
    # Primary Interface: submit_request()
    # ==========================================================================

    async def submit_request(
        self,
        metadata: "RequestMetadata",
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Callable[..., Exception] | None = None,
    ) -> Any:
        """
        Submit a request for rate-limited execution.

        This is the PRIMARY interface method, invoked by ``VeniceClient`` for every request.
        It orchestrates the complete request lifecycle:

        1. Check if rate limited (wait if needed)
        2. Execute the request via request_func
        3. Inspect response - if 429, create RateLimitError using error_factory
        4. Update state from response headers
        5. Handle 429 errors with backoff and retry
        6. Return the response

        IMPORTANT: request_func() returns raw responses including 429s.
        This method is responsible for detecting 429s and creating errors.

        Args:
            metadata: Request metadata containing model_id, endpoint, etc.
            request_func: Async callable that executes the actual HTTP request.
                         Returns raw response (including 429s).
            error_factory: Optional callable to create errors from response.
                          Signature: (message, request, body, response) -> Exception
                          If provided, used to create RateLimitError for 429 responses.
                          If not provided, a basic RateLimitError is created.

        Returns:
            The response from executing request_func

        Raises:
            RateLimitError: If max retries exceeded while rate limited
        """
        from ..exceptions import RateLimitError

        model = metadata.model_id
        last_rate_limit_error: Exception | None = None
        # Total time spent in retry backoff, so it can be held under the
        # caller's own deadline rather than silently outlasting it.
        total_backoff = 0.0

        for attempt in range(self.max_retries + 1):
            # Check rate limit before proceeding (based on local state)
            can_proceed, wait_time = await self.acquire(model)

            if not can_proceed:
                if attempt >= self.max_retries:
                    # At max retries with local rate limit state blocking.
                    # We MUST have a prior RateLimitError to re-raise.
                    if last_rate_limit_error:
                        raise last_rate_limit_error
                    else:
                        # This should only happen if headers indicated exhausted limits
                        # before we ever saw a 429. Very rare edge case.
                        raise RateLimitError(
                            message=f"Rate limit for model {model} - local state indicates exhausted limits",
                            response=None,
                            body={"error": "Rate limit exhausted based on response headers"},
                        )
                else:
                    logger.info(
                        f"Rate limited on {model}, waiting {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(wait_time)
                    continue

            # Execute the request
            response = await request_func()

            # Get status code (aiohttp uses .status, httpx uses .status_code)
            status: int = (
                getattr(response, "status", None) or getattr(response, "status_code", None) or 200
            )

            # Update rate limit state from response headers
            if hasattr(response, "headers"):
                headers = dict(response.headers)
                await self.update_from_headers(model, headers, status)

            # ================================================================
            # 429 DETECTION: Create RateLimitError using error_factory
            # This ensures consistent error payloads across all paths
            # ================================================================
            if status == 429:
                # Parse response body for error context
                body = await self._parse_response_body(response)

                # Create error using factory if provided, else basic error
                if error_factory:
                    rate_limit_error = error_factory(
                        message="Rate limit exceeded",
                        request=None,
                        body=body,
                        response=response,
                    )
                else:
                    rate_limit_error = RateLimitError(
                        message="Rate limit exceeded",
                        response=response,
                        body=body,
                    )

                # Capture for potential re-raise
                last_rate_limit_error = rate_limit_error

                if attempt >= self.max_retries:
                    raise rate_limit_error

                # Wait for backoff and retry
                state = await self.get_state(model)
                if state:
                    wait_time = max(0, state["backoff_until"] - time.time())
                else:
                    wait_time = (
                        getattr(rate_limit_error, "retry_after_seconds", None) or self.min_backoff
                    )

                # An error-budget 429 only clears when its 30s window rolls;
                # the normal backoff schedule retries entirely inside it.
                if getattr(rate_limit_error, "is_error_budget", False):
                    wait_time = max(wait_time, self.ERROR_BUDGET_WINDOW_SECONDS)

                # Waiting past the caller's deadline serves nobody: they would
                # get neither the response nor their timeout on time. Surface
                # the rate-limit error instead.
                call_timeout = getattr(metadata, "timeout", None)
                if (
                    isinstance(call_timeout, (int, float))
                    and not isinstance(call_timeout, bool)
                    and total_backoff + wait_time > call_timeout
                ):
                    logger.info(
                        f"Not retrying {model} after 429: a {wait_time:.1f}s backoff would "
                        f"exceed the {call_timeout:.1f}s request timeout"
                    )
                    raise rate_limit_error

                total_backoff += wait_time

                logger.info(
                    f"Received 429 on {model}, retrying after {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                await asyncio.sleep(wait_time)
                continue

            # Success - reset failure count for 2xx responses
            if 200 <= status < 300:
                await self.record_success(model)

            return response

        # Should not reach here, but if it does, re-raise the last rate limit error
        if last_rate_limit_error:
            raise last_rate_limit_error
        # Final fallback - this really shouldn't happen
        raise RuntimeError(f"Rate limit retry exhausted for model {model} without error")

    async def _parse_response_body(self, response: Any) -> Any:
        """Parse response body for error context."""
        try:
            return await response.json()
        except Exception:
            try:
                return await response.text()
            except Exception:
                return None


class NoOpRateLimiter:
    """
    Rate limiter that does nothing (for testing/disabled mode).

    WARNING: Using this in production bypasses all rate limit protection.
    Use only for testing or when you have external rate limiting in place.

    Implements the full RateLimiterProtocol including submit_request().
    """

    def __init__(self) -> None:
        self._running = False
        self._classifier: Any | None = None

    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @property
    def classifier(self) -> Any | None:
        return self._classifier

    @classifier.setter
    def classifier(self, value: Any) -> None:
        self._classifier = value

    async def submit_request(
        self,
        metadata: "RequestMetadata",
        request_func: Callable[[], Awaitable[Any]],
        error_factory: Callable[..., Exception] | None = None,
    ) -> Any:
        """Execute request without any rate limiting.

        Note: error_factory is accepted but ignored - NoOpRateLimiter
        does not create errors, it passes responses through directly.
        """
        return await request_func()

    async def acquire(self, model: str) -> tuple[bool, float]:
        return True, 0.0

    async def update_from_headers(
        self, model: str, headers: dict[str, str], status_code: int = 200
    ) -> None:
        pass

    async def record_failure(self, model: str) -> None:
        pass

    async def record_success(self, model: str) -> None:
        pass
