"""
Rate Limit Discovery for Venice AI

This module handles the **discovery** layer of the two-layer rate limiting architecture:
- **Discovery** (this module): Fetches rate limit info from the API, groups models into
  buckets/tiers, and caches the results.
- **Enforcement** (``venice_ai.rate_limiting.simple``): Applies the discovered limits via
  token-bucket algorithm with exponential backoff.

See also: ``docs/RATE_LIMITING_ARCHITECTURE.md``
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from .._queue_types import ResourceType

if TYPE_CHECKING:
    from .._client import VeniceClient

logger = logging.getLogger(__name__)


class RateLimitBucket(BaseModel):
    """
    Rate limit tier model

    Represents a group of models sharing the same rate limits.
    All models in a tier share a single rate limit counter.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_id: str = Field(..., description="Unique identifier for this tier")
    rpm_limit: int = Field(..., gt=0, description="Requests per minute limit")
    models: set[str] = Field(default_factory=set, description="Models in this tier")
    rpd_limit: int | None = Field(default=None, gt=0, description="Requests per day limit")
    tpm_limit: int | None = Field(default=None, gt=0, description="Tokens per minute limit")

    @property
    def signature(self) -> tuple[int, int, int]:
        """Unique signature for this tier based on rate limits."""
        return (self.rpm_limit, self.rpd_limit or 0, self.tpm_limit or 0)


class RateLimitDiscovery:
    """
    Manages the discovery and grouping of rate limit tiers for different API
    models.

    In the Venice AI API, models with the same rate limits are grouped into
    "tiers." All models within a single tier share the same rate limit
    counter. This class is responsible for discovering these tiers by
    fetching rate limit information from the API and grouping models
    accordingly.

    The discovered tiers are cached to improve performance, and a fallback
    mechanism with default tiers is provided in case the API is unreachable.
    """

    def __init__(
        self,
        client: VeniceClient | None = None,
        account_id: str | None = None,
        cache_duration: int = 300,  # 5 minutes
    ) -> None:
        """
        Initialize simplified tier discovery.

        Args:
            client: Venice client for API calls
            account_id: Account identifier
            cache_duration: Cache duration in seconds
        """
        self.client = client
        self.account_id = account_id
        self.cache_duration = cache_duration

        self.tiers: dict[str, RateLimitBucket] = {}
        self.model_to_tier: dict[str, str] = {}

        self._last_refresh: datetime | None = None
        self._lock = asyncio.Lock()

        # Cache stampede prevention: track in-flight refresh operations
        self._refresh_futures: dict[str, asyncio.Future[dict[str, RateLimitBucket]]] = {}

        # Default tiers for fallback
        self._ensure_default_tiers()

        logger.info(f"TierDiscovery initialized for account: {account_id or 'client-level'}")

    def _ensure_default_tiers(self) -> None:
        """Ensure default tiers exist for each resource type."""
        for resource_type in ResourceType:
            bucket_id = f"tier_default_{resource_type.value}"
            if bucket_id not in self.tiers:
                bucket = RateLimitBucket(
                    bucket_id=bucket_id,
                    rpm_limit=10,  # Safe default
                    rpd_limit=1000,
                    tpm_limit=None,
                    models=set(),
                )
                self.tiers[bucket_id] = bucket

    async def discover_tiers(self, force_refresh: bool = False) -> dict[str, RateLimitBucket]:
        """
        Discover rate limit tiers from API with cache stampede prevention.

        Groups models that share the same rate limits into tiers (ESSENTIAL).
        Uses request coalescing to prevent multiple concurrent tier discovery
        operations when cache expires during high load.

        Cache Stampede Prevention:
            - Only the first concurrent request performs actual API call
            - Subsequent requests wait for the first request to complete
            - All waiting requests receive the same result
            - Prevents API rate limit exhaustion during cache expiration
        """
        # Fast path: cache is valid and no refresh needed
        if not force_refresh and self._is_cache_valid():
            return self.tiers

        # Check if a refresh is already in progress
        refresh_key = f"tier_refresh_{self.account_id or 'default'}"

        # Track total tier discovery requests
        self._record_tier_discovery_request()

        # Check if refresh is already in progress (first check outside lock)
        if refresh_key in self._refresh_futures:
            # Wait for the in-flight refresh to complete
            existing_future = self._refresh_futures[refresh_key]
            logger.debug(
                "Tier discovery already in progress, waiting for result (stampede prevention)"
            )

            # Track coalesced request (cache hit from stampede prevention)
            self._record_coalesced_request()

            # Track concurrent requests during coalescing
            concurrent_count = len([f for f in self._refresh_futures.values() if not f.done()])
            self._record_concurrent_requests(concurrent_count)

            try:
                start_time = time.time()
                result = await existing_future
                time_saved = time.time() - start_time
                # Record time saved by not making a duplicate API call
                self._record_time_saved(time_saved)
                return result
            except (asyncio.CancelledError, KeyError):
                # Future was cancelled or removed, fall through to retry
                pass

        # Create a new future for this refresh operation
        refresh_future: asyncio.Future[dict[str, RateLimitBucket]]

        async with self._lock:
            # Check again after acquiring lock (double-check pattern)
            if not force_refresh and self._is_cache_valid():
                return self.tiers

            # Check if someone else created the future while we waited for lock
            if refresh_key in self._refresh_futures:
                refresh_future = self._refresh_futures[refresh_key]
            else:
                # We create the future
                refresh_future = asyncio.Future()
                self._refresh_futures[refresh_key] = refresh_future

        # If we created the future, we're responsible for the refresh
        if not refresh_future.done():
            try:
                # Track actual API call (unique request)
                self._record_api_call()

                # Perform the actual tier discovery
                rate_limits_data = await self._fetch_rate_limits_simple()

                if rate_limits_data:
                    self._process_rate_limits_simple(rate_limits_data)
                    self._last_refresh = datetime.now(UTC)

                    logger.info(
                        f"Discovered {len(self.tiers)} tiers with {len(self.model_to_tier)} models"
                    )

                # Set the result for all waiting coroutines
                if not refresh_future.done():
                    refresh_future.set_result(self.tiers)

            except asyncio.CancelledError:
                # Cancelled - propagate to waiting coroutines
                if not refresh_future.done():
                    refresh_future.cancel()
                raise
            except (ValueError, TypeError, AttributeError, OSError) as e:
                logger.exception(f"Error discovering tiers: {e}")
                # Set result anyway with current tiers (fallback)
                if not refresh_future.done():
                    refresh_future.set_result(self.tiers)
            finally:
                # Clean up the future from tracking dict
                async with self._lock:
                    self._refresh_futures.pop(refresh_key, None)

        # Return current tiers
        return self.tiers

    async def _fetch_rate_limits_simple(self) -> list[dict[str, Any]] | None:
        """Fetch rate limits"""
        if not self.client:
            logger.warning("No client available for rate limit fetching")
            return None

        try:
            # A direct client request: it carries the client's authentication
            # and absolute API URL whichever session the client sends through,
            # and it bypasses the rate limiter this discovery feeds.
            data = await self.client.get("api_keys/rate_limits", force_direct=True)
            if not isinstance(data, dict):
                return None

            # Extract rate limits
            rate_limits_data = data.get("data", {})
            if isinstance(rate_limits_data, dict) and "rateLimits" in rate_limits_data:
                rate_limits = rate_limits_data["rateLimits"]
                if isinstance(rate_limits, list):
                    return rate_limits
            elif isinstance(rate_limits_data, list):
                return rate_limits_data

            return None

        except asyncio.CancelledError:
            raise  # Always re-raise for graceful shutdown
        except (ValueError, TypeError, AttributeError, OSError) as e:
            logger.exception(f"Failed to fetch rate limits: {e}")
            return None

    def _process_rate_limits_simple(self, rate_limits_list: list[dict[str, Any]]) -> None:
        """
        Process rate limits and create tiers.

        Groups models with identical rate limits into tiers.
        """
        # Build new dicts locally to ensure atomic swap
        new_tiers: dict[str, RateLimitBucket] = {}
        new_model_to_tier: dict[str, str] = {}

        # Add default tiers to new dict
        for resource_type in ResourceType:
            bucket_id = f"tier_default_{resource_type.value}"
            bucket = RateLimitBucket(
                bucket_id=bucket_id,
                rpm_limit=10,  # Safe default
                rpd_limit=1000,
                tpm_limit=None,
                models=set(),
            )
            new_tiers[bucket_id] = bucket

        for item in rate_limits_list:
            model_id = item.get("apiModelId")
            if not model_id:
                continue

            # Extract rate limits (convert to int to handle API float values)
            rpm, rpd, tpm = None, None, None
            for limit in item.get("rateLimits", []):
                limit_type = limit.get("type")
                amount = limit.get("amount")
                if limit_type == "RPM":
                    rpm = int(amount) if amount is not None else 60
                elif limit_type == "RPD":
                    rpd = int(amount) if amount is not None else None
                elif limit_type == "TPM":
                    tpm = int(amount) if amount is not None else None

            # Each model has its own independent rate-limit bucket
            # (tier_id == model_id); models are not grouped into shared tiers.

            bucket_id = model_id  # Use model_id directly as tier_id

            bucket = RateLimitBucket(
                bucket_id=bucket_id,
                rpm_limit=rpm or 60,
                rpd_limit=rpd if rpd and rpd > 0 else None,
                tpm_limit=tpm if tpm and tpm > 0 else None,
                models={model_id},
            )
            new_tiers[bucket_id] = bucket
            new_model_to_tier[model_id] = bucket_id

        # Atomic swap
        self.tiers = new_tiers
        self.model_to_tier = new_model_to_tier
        logger.debug(
            f"TierDiscovery updated: {len(self.tiers)} tiers, {len(self.model_to_tier)} models. Tiers: {list(self.tiers.keys())}"
        )

    async def get_tier_for_model(
        self, model_id: str, resource_type: ResourceType | str | None = None
    ) -> str | None:
        """
        Get the tier ID for a specific model

        Args:
            model_id: The model identifier
            resource_type: Type of resource (for fallback)

        Returns:
            Tier ID if found, None otherwise
        """
        # Ensure tiers are discovered
        if not self.tiers or not self._is_cache_valid():
            await self.discover_tiers()

        # Check if model is known
        if model_id in self.model_to_tier:
            return self.model_to_tier[model_id]

        # Try refresh once
        await self.discover_tiers()
        if model_id in self.model_to_tier:
            return self.model_to_tier[model_id]

        # Dynamic handling for "unknown" model: find any known model's tier
        # matching the resource type. This handles endpoints like image/edit
        # that don't specify a model. Uses classifier patterns instead of
        # hardcoded model lists so new models get proper rate limiting
        # without code changes.
        if model_id == "unknown" and resource_type:
            resource_type_str = (
                resource_type if isinstance(resource_type, str) else resource_type.value
            )
            tier_id = self._find_tier_for_resource_type(resource_type_str)
            if tier_id:
                return tier_id

        # Fallback to default tier if resource type provided
        if resource_type:
            if isinstance(resource_type, str):
                return f"tier_default_{resource_type}"
            else:
                return f"tier_default_{resource_type.value}"

        return None

    def _find_tier_for_resource_type(self, resource_type_str: str) -> str | None:
        """Dynamically find a tier for any model matching the given resource type.

        Instead of maintaining hardcoded model lists, uses the request
        classifier's model type patterns to match known models in
        ``model_to_tier`` to the given resource type.  The lazy import
        avoids a circular dependency (``_request_classifier`` imports
        from ``core.rate_limit_discovery``).

        Args:
            resource_type_str: Resource type value (e.g., ``"image"``,
                ``"audio"``, ``"embedding"``).

        Returns:
            Tier ID if a matching model is found, ``None`` otherwise.
        """
        # Lazy import — _request_classifier imports from core.rate_limit_discovery
        from .._request_classifier import _MODEL_TYPE_PATTERNS

        # Resolve the ResourceType enum member for the given string
        target_type: ResourceType | None = None
        for rt in ResourceType:
            if rt.value == resource_type_str:
                target_type = rt
                break

        if target_type is None or target_type not in _MODEL_TYPE_PATTERNS:
            return None

        patterns = _MODEL_TYPE_PATTERNS[target_type]
        for model_id in self.model_to_tier:
            for pattern in patterns:
                if pattern.search(model_id):
                    return self.model_to_tier[model_id]

        return None

    async def get_tier(self, bucket_id: str) -> RateLimitBucket | None:
        """Get a specific tier by ID."""
        if not self.tiers:
            await self.discover_tiers()

        return self.tiers.get(bucket_id)

    async def get_models_in_tier(self, bucket_id: str) -> set[str]:
        """Get all models in a specific tier."""
        bucket = await self.get_tier(bucket_id)
        return bucket.models if bucket else set()

    def _is_cache_valid(self) -> bool:
        """Check if the cached tiers are still valid"""
        if not self.tiers or not self._last_refresh:
            return False

        age = (datetime.now(UTC) - self._last_refresh).total_seconds()
        return age < self.cache_duration

    def _record_tier_discovery_request(self) -> None:
        """Record a tier discovery request (including coalesced)."""
        try:
            from ..observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                metrics.tier_discovery_requests_total.inc()
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    def _record_api_call(self) -> None:
        """Record an actual API call for tier discovery."""
        try:
            from ..observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                metrics.tier_discovery_api_calls_total.inc()
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    def _record_coalesced_request(self) -> None:
        """Record a request that was coalesced (stampede prevention)."""
        try:
            from ..observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                metrics.tier_discovery_coalesced_total.inc()
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    def _record_concurrent_requests(self, count: int) -> None:
        """Record number of concurrent requests during coalescing."""
        try:
            from ..observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                metrics.tier_discovery_concurrent_requests.set(count)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

    def _record_time_saved(self, seconds: float) -> None:
        """Record time saved by request coalescing."""
        try:
            from ..observability.metrics import get_enhanced_metrics

            metrics = get_enhanced_metrics()
            if metrics._enabled:
                metrics.tier_discovery_time_saved_seconds.inc(seconds)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
