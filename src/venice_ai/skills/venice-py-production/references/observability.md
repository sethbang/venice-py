# Observability for production Venice clients

What to log, what to measure, what to trace. The SDK exposes enough metadata via `_response`-derived properties that you don't need a separate observability layer for most production use cases — just a structured logger.

## The minimum viable observability

For every Venice call, log:

```python
import logging
log = logging.getLogger("venice")

async def traced_create(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    log.info(
        "venice.chat",
        model=kwargs.get("model"),
        prompt_tokens=response.usage.prompt_tokens if response.usage else None,
        completion_tokens=response.usage.completion_tokens if response.usage else None,
        total_tokens=response.usage.total_tokens if response.usage else None,
        balance_usd=response.balance_info.usd if response.balance_info else None,
        request_id=response.headers.get("x-request-id") if response.headers else None,
        deprecated=(response.deprecation_info.is_deprecated if response.deprecation_info else False),
    )
    return response
```

Six fields: model, three token counts, balance, request_id. With those, you can answer "what did Venice charge me for", "did we hit a deprecated model", and "let me look up this call in Venice's logs".

## Field-by-field, what each is for

| Field | Source | Used for |
|---|---|---|
| `model` | `kwargs["model"]` (or `response.model`) | Per-model spend / latency aggregation |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | `response.usage.*` | Token-budget tracking, cost attribution |
| `balance_usd` | `response.balance_info.usd` | What the key can still spend (before this call): the lesser of the account balance and the key's remaining consumption limit |
| `request_id` | `response.headers["x-request-id"]` | Correlate with Venice's server logs |
| `deprecated` | `response.deprecation_info.is_deprecated` | Alert on model retirement |
| `latency_ms` | wrap with timing | SLO / dashboard input |
| `retry_attempts` | from your retry wrapper | Detect chronic transient failures |

For STREAMING responses, the same fields live on `stream.final_response` (populated by `collect_with_deltas()` or `collect()`, NOT by `text_deltas()`).

## Structured logging libraries

The above examples use stdlib `logging` with kwargs — that works with `structlog`, `loguru`, and most observability platforms' Python SDKs (Datadog, Sentry, etc.) out of the box.

```python
# structlog
import structlog
log = structlog.get_logger("venice")
log.info("venice.chat", model=..., total_tokens=..., balance_usd=...)

# loguru
from loguru import logger
logger.info("venice.chat", model=..., total_tokens=..., balance_usd=...)
```

If you're stuck with positional `logging` formatting (`log.info("venice.chat model=%s", model)`), switch — structured fields are dramatically more useful at query time.

## Metrics (Prometheus / OpenTelemetry)

For high-volume / multi-instance deployments, log lines aren't enough — you want metrics.

### Prometheus

```python
from prometheus_client import Counter, Histogram

VENICE_CALLS  = Counter("venice_calls_total", "Venice API calls", ["model", "endpoint", "status"])
VENICE_TOKENS = Counter("venice_tokens_total", "Tokens consumed", ["model", "kind"])  # kind: prompt|completion
VENICE_LATENCY = Histogram("venice_latency_seconds", "Venice latency", ["model", "endpoint"])
VENICE_BALANCE = ... # gauge updated on each response

import time

async def traced_create(client, **kwargs):
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(**kwargs)
        VENICE_CALLS.labels(model=kwargs.get("model"), endpoint="chat", status="ok").inc()
        if response.usage:
            VENICE_TOKENS.labels(model=kwargs.get("model"), kind="prompt").inc(response.usage.prompt_tokens)
            VENICE_TOKENS.labels(model=kwargs.get("model"), kind="completion").inc(response.usage.completion_tokens)
        return response
    except Exception as e:
        VENICE_CALLS.labels(model=kwargs.get("model"), endpoint="chat", status=type(e).__name__).inc()
        raise
    finally:
        VENICE_LATENCY.labels(model=kwargs.get("model"), endpoint="chat").observe(time.perf_counter() - start)
```

Labels to favor:
- **`model`** — but cardinality matters. If you use 50 different model IDs, that's 50× metric series. For most apps a few are fine.
- **`endpoint`** — chat / images / audio / video / embeddings — bounded set, low cardinality.
- **`status`** — "ok" / "rate_limited" / "auth_error" / etc. Use the exception class name; bounded.

Don't label by:
- `request_id` (every call is unique → cardinality explosion)
- `prompt` (same reason)
- `user_id` (high-cardinality user-data; goes in logs, not metrics)

### OpenTelemetry traces

For request-flow tracing across services:

```python
from opentelemetry import trace

tracer = trace.get_tracer("venice")

async def traced_create(client, **kwargs):
    with tracer.start_as_current_span("venice.chat.completions.create") as span:
        span.set_attribute("venice.model", kwargs.get("model"))
        response = await client.chat.completions.create(**kwargs)
        if response.usage:
            span.set_attribute("venice.tokens.total", response.usage.total_tokens)
        if response.headers:
            span.set_attribute("venice.request_id", response.headers.get("x-request-id"))
        return response
```

Useful if your agent calls Venice from inside a larger flow (HTTP request → DB query → Venice → DB write → HTTP response) — the trace stitches it together.

## Deprecation alerting

`response.deprecation_info.is_deprecated == True` should be a high-signal alert, not a debug log. Models get retired; if your code is still hitting one, you have weeks (not months) before it stops working:

```python
if response.deprecation_info and response.deprecation_info.is_deprecated:
    log.warning(
        "venice.deprecated_model",
        model=response.model,
        warning=response.deprecation_info.warning,
        sunset=response.deprecation_info.date,
    )
    # Optional: page if a critical workload is using a deprecated model
    if kwargs.get("model") in CRITICAL_MODELS:
        alert.page("Critical workload using deprecated Venice model")
```

Pair with `client.models.resolve_*()` so the migration is automatic — when the model retires, the resolver picks its successor.

## Balance / spend monitoring

`response.balance_info.usd` is what the calling key can still spend: the lesser of the account balance and what remains under the key's consumption limit, read before the request was processed. Alert on it to catch either running out:

```python
if response.balance_info and response.balance_info.usd is not None:
    if response.balance_info.usd < CRITICAL_BALANCE_USD:
        alert.page(f"Venice key can spend only ${response.balance_info.usd}")
    elif response.balance_info.usd < WARN_BALANCE_USD:
        log.warning("venice.balance_low", balance_usd=response.balance_info.usd)
```

A low value can mean the key's limit is nearly used up while the account still has funds; check `client.billing.get_balance()` for the account itself. For per-call spend and budgets, use `CostTracker` + `BudgetManager` (see `cost-tracking.md`).

## Request-ID correlation

Every Venice response with `response.headers` exposes `x-request-id`. Save this on EVERY successful AND failed call so you can give Venice support a single ID to look up:

```python
async def traced_create(client, **kwargs):
    try:
        response = await client.chat.completions.create(**kwargs)
        log.info("venice.chat", request_id=_extract_request_id(response.headers), ...)
        return response
    except APIError as e:
        log.error("venice.chat.failed",
                  exc=type(e).__name__,
                  status=e.status_code,
                  request_id=(e.response.headers.get("x-request-id") if getattr(e, "response", None) is not None else None),
                  body=e.body)
        raise


def _extract_request_id(headers) -> str | None:
    if not headers:
        return None
    # Case-insensitive lookup; Venice uses 'x-request-id' but be defensive
    for k in ("x-request-id", "X-Request-Id", "x-venice-request-id"):
        if k in headers:
            return headers[k]
    return None
```

`APIError` (and all subclasses) expose the raw response via `e.response`; read headers as `e.response.headers` — the request_id is available on errors too. Don't forget the error path.

## Sampling

If you log every call, prepare for log-volume problems on high-traffic services. Strategies:

- **Sample successful calls**: log 1 in 100 (or in 1000) at INFO level; log all errors at ERROR.
- **Aggregate before logging**: count tokens / requests over a 60s window, emit a single summary log per window.
- **Push to metrics, not logs, for high-volume signals** — log lines for low-cardinality events, metrics for high-cardinality.

```python
import random
SAMPLE_RATE = 0.01

async def traced_create(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    if random.random() < SAMPLE_RATE:
        log.info("venice.chat", ...)
    METRICS.observe(...)                          # always emit metrics
    return response
```

## What NOT to log

- **Full prompts and completions on every call.** Sensitive data leakage; log size explosion.
- **API keys / private keys.** Ever. Even at DEBUG.
- **Full response objects.** They're large, mostly redundant with the structured fields, and may contain user-specific content.
- **`response._response`** — the raw HTTP response object. Internal SDK state.

For sensitive workloads, consider hashing prompts/completions (`hashlib.sha256(text.encode()).hexdigest()[:16]`) so you can correlate without storing content.

## Common bugs

- **Reading `response.usage.total_tokens` without null-checking `response.usage`** — usage is `None` when the model doesn't emit it (some streaming setups).
- **Logging at ERROR for transient retries** — drowns the "real" errors. INFO/WARN for retries; ERROR only for surfaced failures.
- **High-cardinality label on Prometheus metrics** (e.g., `request_id`) — explodes the time-series count.
- **Logging `response.choices[0].message.content`** — large, often sensitive. Hash or sample.
- **No `request_id` in error logs** — defeats the most useful debugging input you have.

## Related references

- `cost-tracking.md` — `CostTracker.summary()` is the natural input for periodic spend reports.
- `error-taxonomy.md` — what each exception class means in observability terms (transient vs terminal).
- `retries.md` — instrument retry attempts as a metric (helps detect "the API is degraded").
- `venice-py/references/headers-and-metadata.md` — full surface of `_response`-derived properties.
