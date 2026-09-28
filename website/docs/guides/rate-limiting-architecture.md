# Rate Limiting Architecture

Venice AI uses a **two-layer** rate limiting design. Understanding the separation
is essential before modifying either layer.

## Layer 1: Discovery (`core/rate_limit_discovery.py`)

**Purpose**: Learn *what* the limits are.

- Calls `api_keys/rate_limits` to fetch per-model rate limit data from Venice.
- Groups models into `RateLimitBucket` objects (keyed by model ID).
- Caches results with configurable TTL (default 5 minutes).
- Implements **cache stampede prevention** via future coalescing so concurrent
  callers share a single API request.

Key class: `RateLimitDiscovery`

## Layer 2: Enforcement (`rate_limiting/simple.py`)

**Purpose**: *Enforce* the discovered limits at request time.

- Maintains per-MODEL rate-limit state parsed from response headers (rpm/tpm limits + reset times).
- Reactive only: after a 429 it backs off with exponential backoff + jitter and gates on the backoff window and `rpm_remaining`; it does NOT proactively enforce RPM/RPD/TPM before sending (TPM is stored but not gated; there is no RPD field).
- Adds global abuse protection that blocks ALL requests after a failure threshold within a window.

Key class: `SimpleRateLimiter` (implements `RateLimiterProtocol`)

## How They Connect

```
VeniceClient
  --> builds RequestMetadata (no classifier in SIMPLE mode; minimal fallback metadata)
  --> rate_limiter.submit_request(metadata, request_func)
  --> SimpleRateLimiter.acquire(model) (checks per-model header-derived state + global block)
  --> HTTP request
  --> SimpleRateLimiter.update_from_headers(model, headers, status) (records limits/backoff)
```

`RequestClassifier` and `RateLimitDiscovery` are constructed and wired only in
ADAPTIVE mode (`factory.py`); the default SIMPLE limiter uses neither and
resolves nothing to buckets.

## Optional: Adaptive Scheduler (extracted package)

The reactive limiter above is the default. For multi-process / multi-worker
deployments, the SDK can swap it for the proactive `AdaptiveScheduler` that
ships in the standalone **`adaptive-rate-limiter`** PyPI package
(`pip install 'venice-py[adaptive]'`). When `RateLimiterMode.ADAPTIVE` is
configured, `factory._create_rate_limiter` constructs an `AdaptiveScheduler`
backed by Redis, wraps it in `AdaptiveSchedulerAdapter` (in `factory.py`)
to satisfy `RateLimiterProtocol`, and returns it in place of
`SimpleRateLimiter`. The adapter bridges Venice's `RateLimitDiscovery`
+ `RequestClassifier` to the extracted library's protocols. ADAPTIVE mode
**requires** `redis_url` to be set (either on `rate_limiter.redis_url` or
on `backend.redis.redis_url`).

> **Note:** The `AdaptiveScheduler` is always constructed in
> `SchedulerMode.INTELLIGENT` mode regardless of the `SchedulerConfig.mode`
> setting. `SchedulerConfig.mode` is not consulted in the ADAPTIVE path because
> INTELLIGENT mode is required — it uses the `VeniceProvider` and
> `VeniceClassifierAdapter` that are also constructed during ADAPTIVE
> initialisation. Setting `SchedulerConfig(mode=SchedulerMode.BASIC)` or
> `SchedulerMode.ACCOUNT` has no effect when `RateLimiterMode.ADAPTIVE` is active.
> Every other `SchedulerConfig` field is passed to the adaptive scheduler's
> configuration, and `backend.redis` supplies the Redis backend's `max_connections`
> and `cluster_mode` (with `redis_url` as the cluster seed node).

## Configuration

See `config/rate_limiter.yaml` for reference configuration with annotated fields.
