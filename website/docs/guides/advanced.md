# Advanced Features

This document covers Venice AI SDK's enterprise-grade features for production deployments.

## Rate Limiting

The SDK rate-limits automatically — no configuration is required. Behavior is
controlled by `RateLimiterConfig` (`venice_ai.rate_limiting.config`), selected
via `RateLimiterMode`:

- **`SIMPLE`** (default) — in-memory and **reactive**: backs off on `429`s
  (`min_backoff`/`max_backoff`, `max_retries`, failure-window circuit breaking).
  Single-process; no Redis.
- **`ADAPTIVE`** — **proactive** (prevents `429`s) with multi-worker coordination
  via Redis. Requires `pip install 'venice-py[adaptive]'` and a `redis_url`.
- **`DISABLED`** — no rate limiting (testing only; not recommended in production).

**Configuration** (set on the client config's `rate_limiter` field):

```python
from venice_ai import VeniceClient
from venice_ai.core.config import VeniceAIConfig
from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

# Default is SIMPLE — nothing to configure. To tune reactive backoff:
config = VeniceAIConfig(
    rate_limiter=RateLimiterConfig(min_backoff=1.0, max_backoff=60.0, max_retries=3)
)

# Proactive, multi-worker (requires the `adaptive` extra + Redis):
config = VeniceAIConfig(
    rate_limiter=RateLimiterConfig(
        mode=RateLimiterMode.ADAPTIVE,
        redis_url="redis://localhost:6379",
        account_id="acct_123",
    )
)

client = VeniceClient(config=config, api_key="your-key")
```

> `create_production_config()` and the other presets in `venice_ai.presets` set a
> sensible `RateLimiterConfig` for you.

**Monitor rate limits:**

```python
response = await client.chat.completions.create(...)

if response.response_rate_limits:
    remaining = response.response_rate_limits.remaining_requests
    reset_time = response.response_rate_limits.reset_requests
    print(f"Remaining requests: {remaining}")
    print(f"Reset at: {reset_time}")
```

## Error Handling & Retries

Comprehensive exception hierarchy for precise error handling:

```python
from venice_ai.exceptions import (
    VeniceError,
    APIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    APITimeoutError,
    APIConnectionError
)

try:
    response = await client.chat.completions.create(...)
except AuthenticationError:
    print("Invalid API key - check your credentials")
except RateLimitError as e:
    print(f"Rate limit exceeded - retry after {e.retry_after_seconds} seconds")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except APITimeoutError:
    print("Request timed out - consider increasing timeout")
except APIConnectionError:
    print("Network error - check your connection")
except APIError as e:
    print(f"API error: {e}")
except VeniceError as e:
    print(f"SDK error: {e}")
```

**Automatic retries with backoff:**

```python
import asyncio

async def make_request_with_retry(client, max_retries=3):
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(...)
        except RateLimitError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
                continue
            raise
        except (APITimeoutError, APIConnectionError):
            if attempt < max_retries:
                await asyncio.sleep(1 + attempt)
                continue
            raise
```

[**-> Full example: `examples/basic/error_handling.py`**](https://github.com/sethbang/venice-py/blob/main/examples/basic/error_handling.py)

## Distributed State Management

For multi-instance deployments, use Redis backend:

**Key features:**
- Per-event-loop connection pooling
- Distributed rate limit coordination
- Cross-instance state synchronization
- Header-based state sync with fallback to release-only mode on validation failures

**Setup:**

```python
from venice_ai.core.config import RedisBackendConfig
from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

redis_config = RedisBackendConfig(
    redis_url="redis://localhost:6379",
    max_connections=20,
)

config = VeniceAIConfig(
    backend=BackendConfig(
        backend_type=BackendType.REDIS,
        redis=redis_config
    ),
    # Redis is only used by the ADAPTIVE rate limiter; with any other mode
    # VeniceClientFactory.create_client() warns that the backend is unused.
    rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE),
)
```

The adaptive backend receives `redis_url`, `max_connections` and `cluster_mode`
(with `cluster_mode=True`, `redis_url` is the seed node for cluster discovery).

### Retry Strategy

```python
from venice_ai.middleware.retry import RetryOptions, create_retry_middleware

retry_options = RetryOptions(
    max_attempts=3,
    base_delay=1.0,
    retry_status_codes={500, 502, 503, 504},  # matches the RetryOptions default
)
retry_middleware = create_retry_middleware(retry_options)
```

> Note: `429` is intentionally omitted from `retry_status_codes`. Rate-limit
> (429) retries are handled separately by `SimpleRateLimiter`, which honors the
> `Retry-After` header with its own backoff; adding 429 here would double-retry.

## Monitoring & Observability

The `venice_ai.observability` package exposes Prometheus-style metrics for
production monitoring.  Health checks and OpenTelemetry tracing helpers are
not bundled — wire your own (e.g. via the OpenTelemetry SDK / a sidecar) if
you need them.

### Enhanced Metrics

Production-focused Prometheus metrics covering streaming fallbacks, custom
stream usage, and tier-discovery coalescing.  The class exposes its
counters/histograms/gauges as attributes so you can call the standard
`prometheus_client` API on them (`.labels(...).inc()`, `.observe(...)`,
etc.):

```python
from venice_ai.observability import EnhancedMetrics, EnhancedMetricsConfig

metrics_config = EnhancedMetricsConfig(
    enabled=True,
    include_detailed_metrics=True,
    prometheus_port=8000,
)
metrics = EnhancedMetrics(config=metrics_config)

# Counters / histograms are exposed as attributes:
metrics.streaming_fallback_total.labels(
    endpoint="chat.completions", reason="server_disconnect"
).inc()
metrics.custom_stream_duration_seconds.labels(stream_type="audio").observe(0.245)
metrics.tier_discovery_coalesced_total.inc()
```

### Response Header Access

```python
response = await client.chat.completions.create(...)

# Rate limits
if response.response_rate_limits:
    print(f"Remaining: {response.response_rate_limits.remaining_requests}")

# Deprecation warnings
if response.deprecation_info and response.deprecation_info.is_deprecated:
    print(f"Warning: {response.deprecation_info.warning}")

# Account balance
if response.balance_info:
    print(f"Balance: {response.balance_info.usd} USD")
```

[**-> Full example: `examples/headers/header_access_example.py`**](https://github.com/sethbang/venice-py/blob/main/examples/headers/header_access_example.py)

## Performance & Optimization

### Connection Pooling

```python
http_config = HttpClientConfig(
    max_connections=200,  # Connection pool size; also the concurrency cap
    timeout=30.0
)
```

### Redis Optimization

```python
redis_config = RedisBackendConfig(
    redis_url="redis://localhost:6379",
    max_connections=50,
    connection_timeout=5.0,
    max_retries=3,
    default_ttl=3600
)
```

### Caching Strategies

```python
from venice_ai.core.config import StateConfig, CachePolicy

state_config = StateConfig(
    cache_policy=CachePolicy.WRITE_BACK,
    cache_ttl=5.0,
    batch_size=100,
    enable_background_cleanup=True
)
```

### Rate Limit Tuning

`SchedulerConfig` is read only by the ADAPTIVE rate limiter
(`rate_limiter.mode=RateLimiterMode.ADAPTIVE`), which receives every field except
`mode`. With any other rate-limiter mode, `VeniceClientFactory.create_client()`
warns that a tuned scheduler has no effect.

```python
scheduler_config = SchedulerConfig(
    mode=SchedulerMode.INTELLIGENT,
    max_concurrent_executions=100,
    max_queue_size=5000,
    rate_limit_buffer_ratio=0.9,
    overflow_policy="reject"
)
```

### Tips

1. **Use streaming for long responses** - Reduce time to first token
2. **Batch embeddings requests** - More efficient than individual calls
3. **Enable connection pooling** - Reuse HTTP connections
4. **Configure appropriate timeouts** - Balance reliability and speed
5. **Use Redis in production** - Better performance for distributed state
6. **Monitor queue depths** - Adjust `max_queue_size` based on traffic

## x402 Wallet Authentication

Venice's `/x402/*` billing endpoints use **Sign-In-With-Ethereum**
(EIP-4361 SIWE) on Base chain (8453) instead of Bearer tokens. The SDK
ships an optional helper, `venice_ai.auth.x402.X402Auth`, that builds
the base64-encoded `X-Sign-In-With-X` header from a private key.

### Install

```bash
pip install 'venice-py[x402]'
```

This pulls `eth-account` (local Ethereum account + signing) and `siwe`
(EIP-4361 message builder).

### Usage

```python
import os
from venice_ai import VeniceClient
from venice_ai.auth.x402 import X402Auth

auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
# auth.wallet_address is derived from the key — no extra config needed.

async with VeniceClient() as client:
    bal = await client.x402.balance(auth=auth)
    print(f"${bal.data.balanceUsd} on {auth.wallet_address}")

    txns = await client.x402.transactions(auth=auth)
    for t in txns.data.transactions[:5]:
        print(t.createdAt, t.type, t.amount)
```

### Private-key hygiene

- **Never hardcode the private key.** Read it from an environment
  variable, a secret manager, or an HSM.
- **Never commit it.** Add any file that contains the key to
  `.gitignore`, including `.env`, `*.pem`, and ad-hoc scratch files.
- **Use a dedicated wallet for Venice billing.** Don't reuse your main
  wallet — minimise blast radius if a test key leaks.
- **Rotate periodically.** Transfer remaining balance to a new wallet,
  revoke exposure, and point the SDK at the new `X402Auth`.

### What gets sent on the wire

Each `balance()` / `transactions()` call builds a fresh SIWE message
with:

- `domain = outerface.venice.ai`, `uri = https://outerface.venice.ai`,
  `version = 1`, `chain_id = 8453`, `statement = "Sign in to Venice API"`
- A fresh 16-hex-char CSPRNG nonce (`secrets.token_hex(8)`)
- ISO-8601 `issued_at` and `expiration_time` 10 min apart (configurable
  via the `ttl_seconds=` kwarg)
- A hex signature from the wallet over the EIP-191-encoded SIWE text

The private key itself never appears in the header or in any SDK log —
the scrubber in `_client.py` redacts `X-Sign-In-With-X` (alongside
`Authorization`) at DEBUG-level request logging.

### Top-ups

`top_up()` uses standard Bearer auth (`VENICE_API_KEY`); the optional
`payment_header=` kwarg carries a pre-signed x402 payment payload. An
empty POST returns the documented `402 Payment Required` with structured
payment requirements — the SDK surfaces that as an `APIError` whose
response body contains the accept spec (chains, assets, amounts). Sign
the payment payload out-of-band (e.g. with `@venice-ai/x402-client`),
then pass the base64 string back to `top_up(payment_header=...)`.

### Testing

VCRpy tests for x402 endpoints use a deterministic throwaway key (never
funded, publicly known in the test suite). The root conftest scrubs
`X-Sign-In-With-X` and `X-402-Payment` from every recorded cassette via
`filter_headers`, so signed tokens never land on disk.
