# Venice exception taxonomy — canonical reference

This is the authoritative table for the `venice-py` SDK's exception classes (sourced from `src/venice_ai/exceptions.py`). Import everything from `venice_ai.exceptions`.

## Tree

```
VeniceError                              (base — catch this only as a last resort)
├── APIError                             (HTTP request reached the server, server responded with error)
│   ├── APIStatusError                   (generic HTTP status error with a status code attached)
│   ├── AuthenticationError              (401)
│   ├── PermissionDeniedError            (403)
│   ├── InvalidRequestError              (400)
│   ├── NotFoundError                    (404)
│   ├── ModelGoneError                   (410 — model retired/unroutable)
│   ├── ConflictError                    (409)
│   ├── UnprocessableEntityError         (422)
│   ├── RateLimitError                   (429) — has .retry_after_seconds
│   ├── PaymentRequiredError             (402) — body has structured payment requirements
│   ├── InternalServerError              (5xx — generic)
│   └── ServiceUnavailableError          (503)
│
├── APIConnectionError                   (network-level: DNS, TCP, SSL/proxy)
├── APITimeoutError                      (request exceeded configured timeout)
│   └── BillingTimeoutError              (specific to the billing API, which is known to hang)
│
├── APIResponseProcessingError           (parsing/validation failure)
│   └── APIResponseValidationError       (Pydantic validation failed on a response)
│
├── StreamConsumedError                  (tried to read a stream twice)
├── StreamClosedError                    (tried to read after the stream was closed)
│
├── VideoGenerationError                 (server-side video job failed) — has .error_code
├── MusicGenerationError                 (server-side music job failed)
└── MaxIterationsExceededError           (run_with_tools loop hit max_iterations) — has .iterations
```

Plus the `VeniceAPIErrorCode` enum (server error code strings) — useful for matching specific server failure modes.

## What to do with each class

### Retry — yes, with backoff

| Class | Retry strategy |
|---|---|
| `RateLimitError` | Read `e.retry_after_seconds`; sleep at least that long, then retry. Cap total wait. |
| `APITimeoutError` | Exponential backoff (1s, 2s, 4s, 8s, 16s, capped at 30s). Max 5 retries. |
| `APIConnectionError` | Same as `APITimeoutError`. |
| `InternalServerError` (5xx) | 1-2 retries with brief backoff. After that, surface — it's likely a server-side bug. |
| `ServiceUnavailableError` (503) | Same as `InternalServerError`, but lean on the side of more retries. |

### Retry — NO

| Class | Reason | What to do instead |
|---|---|---|
| `AuthenticationError` (401) | API key is wrong/missing. Retry won't fix. | Surface to operator. Check `VENICE_API_KEY`. |
| `PermissionDeniedError` (403) | Account lacks access to this resource/model. | Surface; check ACLs / tier. |
| `PaymentRequiredError` (402) | Out of prepaid balance OR x402 endpoint requires payment. | **Top up balance** (see `venice-py-x402`). The 402 body has structured payment instructions. |
| `InvalidRequestError` (400) | Request body / parameters wrong. | Fix the request. Don't loop. |
| `NotFoundError` (404) | Resource doesn't exist. | Fix the URL/ID; don't loop. |
| `ModelGoneError` (410) | Model was retired/unrouted. Retry won't bring it back. | Migrate to `model_spec.deprecation.replacementModelId` (or re-`resolve_*()`); distinct from 404. |
| `UnprocessableEntityError` (422) | Schema validation failed server-side. | Fix the body; don't loop. |
| `NoMatchingModelError` | A `resolve*()` call found no catalog model passing its filters (a `ValueError`). Nothing was called or billed. | Relax the filters or skip the feature; retrying returns the same answer until the catalog changes. |
| `MaxIterationsExceededError` | Agent loop didn't converge in budget. | **Don't retry** — investigate the model's behavior or add tools/prompts to break the cycle. |
| `StreamConsumedError` / `StreamClosedError` | Code bug — tried to consume a stream twice or after close. | Fix the code. |

### Conditional retry

| Class | When to retry |
|---|---|
| `ModelQuotesUnavailableError` | Raised by `resolve_cheapest_video` / `resolve_cheapest_music` when models match but every free quote failed (a `ValueError`). Inspect `e.failures` (model ID → exception): retry later for `RateLimitError` / 5xx / `APIConnectionError`; an `InvalidRequestError` means Venice rejected the quote request itself, so surface it. |
| `ConflictError` (409) | Depends on the resource. For idempotent operations, retry once. For state-mutating operations, surface. |
| `VideoGenerationError` / `MusicGenerationError` | Inspect `e.error_code` first — some codes (e.g., transient render failures) are retryable; others (e.g., content-policy failures) are terminal. |

## Common attribute access patterns

```python
from venice_ai.exceptions import RateLimitError, PaymentRequiredError, VideoGenerationError

try:
    response = await client.chat.completions.create(...)
except RateLimitError as e:
    wait = e.retry_after_seconds or 1.0
    await asyncio.sleep(min(wait, 30.0))
except PaymentRequiredError as e:
    # 402 body carries structured payment requirements (e.g., x402 accepts list)
    payment_info = e.body                     # NOT e.payment_instructions (no such attr)
    log.error("venice.payment_required", **payment_info)
    raise
except VideoGenerationError as e:
    if e.error_code in ("CONTENT_POLICY_VIOLATION", "INVALID_PROMPT"):
        raise                                 # terminal
    # else maybe re-submit
```

`APIError` (the parent of all HTTP-status errors) has these attributes set by the SDK:
- `e.status_code: int | None` — HTTP status, if available
- `e.body: dict | None` — parsed JSON body of the error response
- `e.response`: `aiohttp.ClientResponse | None` — the raw HTTP response; read headers as `e.response.headers` (there is no `e.headers` attribute)
- `e.code: str | None` — Venice's own error code string (matches `VeniceAPIErrorCode` enum values when applicable)
- `e.message: str` — human-readable message (may also be the str(e))

## Anti-patterns

- **`except Exception`** — drops the structured info on every typed subclass. Always catch specific.
- **Retrying `PaymentRequiredError` with backoff** — a 402 won't go away on its own. Top up.
- **Treating `MaxIterationsExceededError` as a transient retry** — it's a logic bug.
- **Reading `e.payment_instructions`** — that attribute doesn't exist. Use `e.body`.
- **Calling `e.retry_after_seconds` on non-`RateLimitError`** — only `RateLimitError` carries it.
- **Catching `VeniceError` in production code paths** — use it only at the very outermost boundary as a "log and surface unknown failure" net.

## Related references

- `retries.md` — full retry decision tree with jitter strategies and code patterns
- `rate-limiting.md` — proactive throttling using `response.response_rate_limits`
- `cost-tracking.md` — `BalanceInfo` and `BudgetManager`; what `PaymentRequiredError` looks like in the wild
