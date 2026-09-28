# Prompt caching on Venice

Venice supports prompt caching for chat completions — when a prefix of your prompt is reused across calls, the server can charge a reduced rate for the cached tokens. Done right, this can drop the cost of long-prefix workloads (RAG with retrieved docs, agents with multi-thousand-token system prompts) by 50-90%.

## When prompt caching pays off

- **System prompt is large and stable.** A 5,000-token system prompt with assistant guidelines, examples, and tool descriptions, reused across thousands of conversations — primary use case.
- **RAG with retrieved context.** The retrieved documents are the same across multiple turns of one conversation; the user query is what changes. Cache the retrieved-context prefix.
- **Agent tools and few-shot examples.** Agents often have a long preamble of tool definitions and one-shot examples that's identical per session.

## When it doesn't

- **Prompts are short.** The cache write overhead (slightly higher cost on the FIRST call) doesn't pay back if the prefix is small.
- **Prompts are unique per call.** No prefix reuse → no cache hit → no benefit.
- **Single-shot scripts.** One call, no reuse → caching is purely overhead.

## How Venice exposes caching

Caching is opt-in via Venice's chat-completions parameters. Check the model's `model_spec.capabilities` for cache support — not every model is cache-enabled. Capability filter:

```python
# Some models advertise cache support; if you need a guaranteed-cacheable model,
# query the catalog and inspect capabilities (the SDK's capability filters cover
# function-calling / vision / reasoning but not caching directly today).
catalog = await client.models.list(type="chat")
cache_capable = [m.id for m in catalog.data
                 if m.model_spec and "prompt_cache" in (m.model_spec.capabilities or [])]
```

(Capability key names may vary by Venice's catalog; check current docs.)

## Pattern: stable prefix → cached tokens

```python
SYSTEM_PROMPT = """You are an internal HR assistant for Acme Corp.
You answer benefits, payroll, and policy questions for Acme employees.

Style:
- Concise, professional tone.
- Cite the relevant policy section when applicable.
- For confidential info, redirect to the HR business partner.

Policies (excerpt):
... [4,000 more tokens of policy text, examples, and guidelines] ...
"""

async def answer(question: str) -> str:
    response = await client.chat.completions.create(
        model=cache_capable_model,
        messages=[
            SystemMessage(content=SYSTEM_PROMPT),    # same on every call → cached after first
            UserMessage(content=question),            # varies → not cached
        ],
        max_completion_tokens=300,
    )
    return response.text
```

The first call pays full price for `SYSTEM_PROMPT`. Subsequent calls within the cache window pay the reduced cache-hit rate. Cache hit rate is exposed in usage:

```python
if response.usage:
    print(f"Total: {response.usage.total_tokens}")
    # Cache stats — keys are model-dependent; check usage.model_extra or specific fields
    cached = response.usage.cache_read_input_tokens
    if cached is not None:
        print(f"Cached: {cached} / {response.usage.prompt_tokens} prompt tokens")
```

(The exact field name for cache hits depends on the model and Venice version. Inspect `response.usage` for cache-related fields.)

## Cache windows

Caches typically expire on a 5-minute (300s) sliding window — if no call uses the prefix in 5 minutes, the cache is dropped. Some Venice tiers may have longer windows (1 hour). Plan your call cadence around this:

- **High-traffic apps** (request every few seconds): cache stays warm, hit rate near 100%.
- **Bursty apps** (requests every few minutes): hits and misses interleave.
- **Idle apps** (requests every few hours): no cache hits; caching is overhead.

For apps with predictable bursts, **prime the cache** before the burst by issuing a no-op call with the cached prefix.

## Prefix structure for maximum reuse

The cache works on **prefix matching** — if your messages list has the same first N items as a prior call, those items are cached. Order matters:

```
[SystemMessage(content=BIG_PROMPT)]                          # cached after first call
[SystemMessage(content=BIG_PROMPT), UserMessage(content="Q1")]  # SystemMessage cached; UserMessage isn't
[SystemMessage(content=BIG_PROMPT), UserMessage(content="Q2")]  # SystemMessage cached again; UserMessage isn't
```

For multi-turn conversations:
```
[SystemMessage(content=BIG_PROMPT), UserMessage(content="Hi"), AssistantMessage(content="Hi back"), UserMessage(content="Q")]
```
The longest cacheable prefix is `[SystemMessage]` if the user message changes per call. Keep your stable-prefix content at the top of the messages list.

## Anti-patterns that defeat caching

- **Inserting a timestamp / nonce / per-call ID into the system prompt** → cache miss on every call.
- **Per-user content in the system prompt** (`f"You are talking to {user.name}"`) → cache miss per user.
- **Reordering messages between calls** → cache invalidates.
- **Slightly different whitespace / punctuation** → fully different prefix → cache miss.

If you need per-call variation, put it in the USER message, not the system prompt.

## Combining with structured output

Cache and structured output (`response_format=BaseModel`) are independent. The system prompt + tool schemas + Pydantic-derived JSON schema can all be cached together if you reuse them.

```python
result = await client.chat.completions.parse(
    model=cache_capable_model,
    messages=[
        SystemMessage(content=BIG_STABLE_PROMPT),
        UserMessage(content=user_question),
    ],
    response_format=Invoice,                  # schema is part of the cacheable prefix
)
```

## Combining with tool calling

`run_with_tools` can use cached prefixes — the tools list and system prompt are stable across loop iterations, so the cache hit rate within a single agent run is typically very high.

```python
result = await client.chat.completions.run_with_tools(
    model=cache_capable_model,
    messages=[SystemMessage(content=BIG_STABLE_AGENT_PROMPT), UserMessage(content="...")],
    tools=[lookup_order, issue_refund],       # tool schemas part of the prefix
    max_iterations=5,
)
```

The first iteration of the loop pays full price; iterations 2-N benefit from the cache.

## Measuring the savings

Wrap your calls with a counter that tracks `usage.prompt_tokens` vs `usage.cache_read_input_tokens` (also surfaced as `usage.prompt_tokens_details.cached_tokens`) and log the ratio:

```python
async def traced_with_cache_stats(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)
    if response.usage:
        prompt = response.usage.prompt_tokens
        cached = response.usage.cache_read_input_tokens or 0
        log.info("venice.chat", model=kwargs["model"], prompt_tokens=prompt, cached_tokens=cached,
                 cache_hit_pct=(cached / prompt * 100 if prompt else 0))
    return response
```

In production, surface `cache_hit_pct` as a Prometheus metric alongside cost — declines indicate prefix instability creeping in (often via accidental per-call variations).

## Cost calculation

Cached tokens are billed at a reduced rate (specifics depend on the model). The rates are in the model's pricing metadata via `client.models.list()`: `pricing.cache_input` for cache reads, `pricing.cache_write` for cache writes, and `pricing.extended` for the long-context tier above `extended.context_token_threshold` (a missing cache rate bills at `input`). With `CostTracker`:

```python
tracker = await CostTracker.from_client(client)  # fetches live pricing including cache rates
async with VeniceClient(cost_tracker=tracker) as client:
    response = await client.chat.completions.create(...)
# tracker.total_cost_usd reflects the cache-discounted cost automatically
```

## Common bugs

- **Per-call timestamps in the system prompt** — your "cache" never hits. Strip dynamic content.
- **Treating `cache_read_input_tokens` as available on every model** — it's `None` unless the server emits cache stats. Null-check.
- **Caching a 200-token prompt** — overhead exceeds savings. Caches earn back at thousands of tokens.
- **Recomputing the system prompt per call** (string interpolation, `.format()`, etc.) — even if the result is identical bytes, build it once outside the call loop to keep the code clean.
- **Cache window assumptions** — don't hardcode "5 minutes." Verify against the model's docs; some tiers have longer windows.

## Related references

- `cost-tracking.md` — `CostTracker` accounts for cache discounts via the live pricing map.
- `concurrency.md` — high-cap concurrent calls naturally keep the cache warm.
- `venice-py/references/tool-loops.md` — agent loops cache the system prompt + tools across iterations.
- `venice-py/references/structured-output.md` — `parse()` calls cache the schema as part of the prefix.
