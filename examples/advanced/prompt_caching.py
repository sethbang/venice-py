#!/usr/bin/env python3
"""
Venice AI SDK - Prompt Caching
==============================

This example demonstrates Venice AI's prompt caching: a repeated prompt prefix
is served from the provider's cache, which cuts latency and, on models with a
cached-input price, the input bill.

Key features covered:
- prompt_cache_key: routing hint that makes a cache hit more likely
- cache_control: markers on content blocks, needed by some providers
- Reading cache reads and writes from ``response.usage.cached_tokens`` and
  ``response.usage.cache_write_tokens``

Model choice
------------
``resolve_chat(require_prompt_caching=True)`` keeps only models whose catalog
entry lists a cached-input price (``pricing.cache_input``). That price is the
only caching signal the catalog has: Venice exposes no caching capability flag,
and some of the cheapest cache-priced models do not actually serve cache hits.
So this example takes the catalog's default ranking (``prefer=None``) rather
than ``prefer="cheapest"``.

How a hit is measured
---------------------
Every request passes ``include_venice_system_prompt=False``. Venice's own
system prompt is shared by all traffic and is often cached already, which
would make a "cold" request look warm. Without it, and with a per-run session
header at the top of the prompt, the first request of each section is truly
cold and reports (close to) 0 cached tokens.

Caching is best effort: some models run on several backends, and a request
that lands on a different backend than the one that cached the prefix misses
even with the same ``prompt_cache_key``. The three hit tests (basic,
multi-turn, cache_control) therefore each send one cold request, then up to
three warm requests about two seconds apart, and stop at the first hit. A hit
test passes only when a warm request reads at least
``MIN_CACHED_PREFIX_TOKENS`` more tokens from cache than its cold request did,
which can only come from this example's own prompt. The monitoring section
measures and prices whatever hit rate a short session gets.
"""

import asyncio
import sys
import time
import uuid
from collections.abc import Callable

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.types.api import (
    AssistantMessage,
    ChatCompletionResponse,
    LLMModelPricing,
    SystemMessage,
    TextModelSpec,
    UserMessage,
    VeniceParameters,
)
from venice_ai.types.api.requests.common import TextContent

# Type alias for message lists accepted by the API
Message = UserMessage | AssistantMessage | SystemMessage

# One id per run. It heads every system prompt and cache key, so nothing this
# run sends can already be in the cache when a section starts.
RUN_ID = uuid.uuid4().hex[:8]

# Bill only the prompt shown here. Venice's own system prompt is shared by all
# traffic and often already cached, which would hide whether *this* prompt hit.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# ---------------------------------------------------------------------------
# Long system prompt used across examples to make caching worthwhile.
# In production you'd use your own domain-specific system instructions.
# ---------------------------------------------------------------------------
LONG_SYSTEM_PROMPT = """\
You are a senior software architect specializing in distributed systems design.

Your expertise covers the following areas in depth:

1. **Microservices Architecture**: You understand service decomposition, bounded contexts,
   API gateway patterns, service mesh topologies, and inter-service communication strategies
   including synchronous REST/gRPC and asynchronous event-driven messaging via Kafka, RabbitMQ,
   and NATS.

2. **Database Design**: You are proficient in relational modeling (PostgreSQL, MySQL),
   document stores (MongoDB, CouchDB), key-value stores (Redis, DynamoDB), wide-column stores
   (Cassandra, ScyllaDB), and graph databases (Neo4j, ArangoDB). You can advise on schema
   design, indexing strategies, partitioning, replication, and consistency trade-offs (CAP theorem,
   PACELC).

3. **Cloud-Native Infrastructure**: You have hands-on experience with Kubernetes orchestration,
   Helm chart authoring, Terraform IaC, CI/CD pipelines (GitHub Actions, GitLab CI, ArgoCD),
   observability stacks (Prometheus, Grafana, OpenTelemetry, Jaeger), and multi-cloud deployment
   strategies across AWS, GCP, and Azure.

4. **Performance & Reliability**: You can design for low-latency, high-throughput workloads using
   caching layers (Redis, Memcached, CDN edge caching), load balancing (L4/L7, consistent hashing),
   circuit breakers, bulkheads, retries with exponential back-off, and chaos engineering practices.

5. **Security**: You follow zero-trust principles, understand OAuth 2.0 / OIDC flows, service-to-
   service mTLS, secret management (Vault, AWS Secrets Manager), and supply-chain security (SBOM,
   Sigstore, SLSA).

When answering questions, always:
- Provide concrete, production-ready recommendations rather than theoretical overviews.
- Cite trade-offs explicitly so the user can make informed decisions.
- Include code snippets or configuration examples when helpful.
- Keep answers concise but thorough. Aim for the right level of detail.
"""

# Providers cache a prefix only above a minimum size (1024 tokens on
# OpenAI-class models, growing in 128-token steps), so the reference notes pad
# the system prompt to roughly 2.5k tokens.
PADDED_SYSTEM_PROMPT = (
    LONG_SYSTEM_PROMPT
    + "\n\nADDITIONAL REFERENCE NOTES:\n"
    + (
        "- This is reference material reused verbatim across requests "
        "to demonstrate prompt-prefix caching. " * 120
    )
)

# A warm request counts as a hit only if it reads at least this many more
# cached tokens than the section's cold request: the smallest prefix providers
# cache, and far more than any shared template could account for.
MIN_CACHED_PREFIX_TOKENS = 1024

# Warm requests per section after the cold one, and the pause between them.
MAX_WARM_REQUESTS = 3
WARM_RETRY_DELAY_S = 2.0

# Exit code for "a prerequisite is missing" (here: no cache-priced chat model).
EXIT_SKIPPED = 77

# Identical requests in the monitoring session.
SESSION_REQUESTS = 4

# Words from the numbered headings above; an answer that lists the areas names them.
AREA_HEADINGS = ("Microservices", "Database", "Cloud-Native", "Performance", "Security")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def select_model(client: VeniceClient) -> tuple[str, LLMModelPricing] | None:
    """Resolve a chat model with a cached-input price and return its token pricing.

    Returns None when no catalog model lists a cached-input price.
    """
    try:
        model_id = await client.models.resolve_chat(require_prompt_caching=True)
    except NoMatchingModelError as e:
        print(f"SKIPPED: no chat model in the catalog lists a cached-input price ({e})")
        return None
    catalog = await client.models.list(type="text")
    spec = next(m.model_spec for m in catalog.data if m.id == model_id)
    if not isinstance(spec, TextModelSpec) or not isinstance(spec.pricing, LLMModelPricing):
        raise RuntimeError(f"The catalog lists no token pricing for {model_id}")
    return model_id, spec.pricing


def report(response: ChatCompletionResponse, label: str, elapsed: float) -> bool:
    """Print the answer and cache usage; fail an empty, truncated or unmeasured reply.

    ``prompt_tokens == 0`` is treated as a failed measurement, not a cache
    miss: some providers report zero usage on a truncated reasoning reply.
    """
    finish = response.choices[0].finish_reason
    text = (response.text or "").strip()
    usage = response.usage
    print(f"   ⏱️  Latency: {elapsed:.2f}s   finish_reason: {finish}")
    print(f"   📝 Response: {_indent(text, 6).strip()}")
    if usage is None or usage.prompt_tokens == 0:
        print(f"   ❌ [{label}] the response carries no usable token usage")
        return False
    print(
        f"   📊 [{label}] prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, "
        f"cache read: {usage.cached_tokens}, cache write: {usage.cache_write_tokens}"
    )
    if finish == "length" or not text:
        print(f"   ❌ [{label}] answer was truncated or empty")
        return False
    return True


def cached_of(response: ChatCompletionResponse) -> int:
    """Prompt tokens served from cache (0 when the response has no usage)."""
    return response.usage.cached_tokens if response.usage else 0


def is_hit(response: ChatCompletionResponse, label: str, baseline: int) -> bool:
    """Print whether a warm request read this prompt's prefix from cache.

    *baseline* is the section's cold ``cached_tokens``. A hit must read at
    least ``MIN_CACHED_PREFIX_TOKENS`` more than that.
    """
    cached = cached_of(response)
    if cached - baseline >= MIN_CACHED_PREFIX_TOKENS:
        print(f"   ✅ [{label}] {cached} prompt tokens read from cache (cold: {baseline})")
        return True
    print(f"   ⚪ [{label}] cache miss (cache read {cached}, cold: {baseline})")
    return False


def system_prompt(section: str) -> str:
    """The padded system prompt behind a per-run, per-section header.

    Providers cache by prompt prefix, so sections sharing one prompt would
    warm each other's cache; the header keeps each section's first request cold.
    """
    return f"Session {RUN_ID}/{section}.\n\n{PADDED_SYSTEM_PROMPT}"


def _indent(text: str, spaces: int = 6) -> str:
    """Indent every line of *text* by *spaces* spaces."""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


async def timed_create(
    client: VeniceClient, model: str, messages: list[Message], cache_key: str | None
) -> tuple[ChatCompletionResponse, float]:
    """Send one request (with a prompt_cache_key when given) and time it."""
    start = time.perf_counter()
    if cache_key is None:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=300,
            venice_parameters=OWN_PROMPT_ONLY,
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=300,
            venice_parameters=OWN_PROMPT_ONLY,
            prompt_cache_key=cache_key,
        )
    return response, time.perf_counter() - start


async def cold_then_warm(
    client: VeniceClient,
    model: str,
    messages: list[Message],
    cache_key: str | None,
    check_answer: Callable[[ChatCompletionResponse], bool] | None = None,
) -> tuple[bool, list[ChatCompletionResponse], list[float]]:
    """Send a cold request, then warm repeats until one hits the cache.

    Returns (passed, responses, latencies). *check_answer*, when given, is
    called on every response and must return True for the section to pass.
    """
    ok = True
    responses: list[ChatCompletionResponse] = []
    latencies: list[float] = []

    print("\n📤 Request 1 (cold cache)")
    cold, elapsed = await timed_create(client, model, messages, cache_key)
    ok = report(cold, "cold", elapsed) and ok
    if check_answer is not None:
        ok = check_answer(cold) and ok
    responses.append(cold)
    latencies.append(elapsed)
    baseline = cached_of(cold)

    hit = False
    for i in range(2, MAX_WARM_REQUESTS + 2):
        await asyncio.sleep(WARM_RETRY_DELAY_S)
        print(f"\n📤 Request {i} (warm cache)")
        response, elapsed = await timed_create(client, model, messages, cache_key)
        ok = report(response, f"warm {i}", elapsed) and ok
        if check_answer is not None:
            ok = check_answer(response) and ok
        responses.append(response)
        latencies.append(elapsed)
        if is_hit(response, f"warm {i}", baseline):
            hit = True
            break

    if not hit:
        print(f"\n   ❌ None of {len(responses) - 1} warm requests read the prefix from cache")
    return ok and hit, responses, latencies


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


async def basic_prompt_caching(client: VeniceClient, model: str) -> bool:
    """Send the same long prompt again under one prompt_cache_key."""
    print("💾 Basic Prompt Caching with prompt_cache_key")
    print("-" * 50)

    cache_key = f"architect-session-{RUN_ID}"
    print(f"   cache key: {cache_key}")
    messages: list[Message] = [
        SystemMessage(content=system_prompt("basic")),
        UserMessage(
            content="In three short bullet points (under 60 words in total), compare REST "
            "and gRPC for inter-service communication."
        ),
    ]

    ok, responses, latencies = await cold_then_warm(client, model, messages, cache_key)
    cold = responses[0].usage
    if cold is not None and cold.cache_write_tokens:
        print(f"\n   ✍️  The cold request wrote {cold.cache_write_tokens} tokens to the cache")
    print(f"\n⚡ Latency: {latencies[0]:.2f}s cold vs {min(latencies[1:]):.2f}s fastest warm")
    print("   💡 cached_tokens is the reliable signal; latency also depends on load.")
    return ok


async def multi_turn_caching(client: VeniceClient, model: str) -> bool:
    """Reuse one prompt_cache_key while a conversation grows.

    Each turn resends the whole history, so every turn after the first starts
    with a prefix an earlier turn already sent.
    """
    print("\n\n🔄 Multi-Turn Conversation Caching")
    print("-" * 50)

    cache_key = f"multiturn-arch-review-{RUN_ID}"
    conversation: list[Message] = [SystemMessage(content=system_prompt("multi-turn"))]
    questions = [
        "In two sentences: how should I partition a PostgreSQL database handling 50k writes/sec?",
        "In two sentences: what replication strategy would you recommend for that setup?",
        "In two sentences: how do I handle failover without data loss?",
        "In one sentence: which of these three steps should I do first?",
    ]

    ok = True
    hit = False
    baseline = 0
    for i, question in enumerate(questions, 1):
        if i > 1:
            await asyncio.sleep(WARM_RETRY_DELAY_S)
        conversation.append(UserMessage(content=question))
        print(f"\n📤 Turn {i}: {question}")
        response, elapsed = await timed_create(client, model, conversation, cache_key)
        # Keep the assistant reply in history for the next turn
        conversation.append(AssistantMessage.from_response(response))
        ok = report(response, f"turn {i}", elapsed) and ok
        if i == 1:
            baseline = cached_of(response)
        elif is_hit(response, f"turn {i}", baseline):
            hit = True
            break

    if not hit:
        print("\n   ❌ No later turn read the shared history from cache")
    print("\n💡 Keep one cache key for the conversation so the growing prefix stays cached.")
    return ok and hit


async def cache_control_markers(client: VeniceClient, model: str, unmarked_hit: bool) -> bool:
    """Send a system content block marked with ``cache_control``.

    What this shows depends on the provider. Most models cache a repeated
    prefix automatically: there the marker is accepted, costs nothing and
    changes nothing, and Venice adds markers for system prompts and history on
    Anthropic models by itself. The case where a marker decides the outcome is
    an Anthropic model reading one large single-turn document: without a
    marker on that block it is not cached. This section checks that the marked
    block reached the model intact (``prompt_tokens`` and an answer drawn from
    it) and that a repeat was served from cache. The requests carry no
    prompt_cache_key. *unmarked_hit* says whether an earlier section got a hit
    on this model with no marker at all, which decides what the hit here can
    be credited to.
    """
    print("\n\n🏷️  Cache Control Markers (cache_control on TextContent)")
    print("-" * 50)

    # The cacheable reference material goes in a single content block.
    system_message = SystemMessage(
        content=[
            TextContent(
                type="text",
                text=system_prompt("cache-control"),
                cache_control={"type": "ephemeral"},
            ),
        ],
    )
    user_message = UserMessage(
        content="List the five numbered areas of expertise from your instructions: "
        "titles only, one per line."
    )
    # English text runs about 4-6 characters per token. A prompt far below this
    # floor means the server dropped the reference block.
    min_prompt_tokens = len(PADDED_SYSTEM_PROMPT) // 8

    def drawn_from_block(response: ChatCompletionResponse) -> bool:
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        if prompt_tokens < min_prompt_tokens:
            print(
                f"   ❌ prompt_tokens={prompt_tokens} is below {min_prompt_tokens}: "
                "the reference block did not reach the model"
            )
            return False
        named = [word for word in AREA_HEADINGS if word.lower() in (response.text or "").lower()]
        if len(named) < 3:
            print(f"   ❌ The answer names {len(named)} of the 5 areas: {named}")
            return False
        return True

    ok, _, _ = await cold_then_warm(
        client, model, [system_message, user_message], None, check_answer=drawn_from_block
    )

    print('\n💡 cache_control={"type": "ephemeral"} marks a block as safe to cache.')
    print(f"   What this run shows on {model}: the marked block arrived intact and")
    print("   a repeat of it was read from cache.")
    if unmarked_hit:
        print("   The sections above hit on this model with no marker at all, so the hit")
        print("   is not evidence the marker caused it; the marker was accepted and harmless.")
    else:
        print("   No unmarked request hit on this model in this run, so whether the marker")
        print("   was needed for this hit is not established either way.")
    return ok


async def monitor_cache_performance(
    client: VeniceClient, model: str, pricing: LLMModelPricing
) -> bool:
    """Measure the hit rate of a short session and price the saving.

    This section is a measurement, not another hit test (the sections above
    each require a hit). It sends a fixed number of identical requests, sums
    reads and writes from ``usage`` and prices them from the catalog. It fails
    only when a response is unusable or the counts are inconsistent; a low
    hit rate is reported, since caching is best effort.
    """
    print("\n\n📊 Monitoring Cache Performance")
    print("-" * 50)

    cache_key = f"monitoring-session-{RUN_ID}"
    messages: list[Message] = [
        SystemMessage(content=system_prompt("monitoring")),
        UserMessage(content="Give a one-sentence summary of your expertise."),
    ]

    ok = True
    responses: list[ChatCompletionResponse] = []
    for i in range(1, SESSION_REQUESTS + 1):
        if i > 1:
            await asyncio.sleep(WARM_RETRY_DELAY_S)
        print(f"\n📤 Request {i} of {SESSION_REQUESTS}")
        response, elapsed = await timed_create(client, model, messages, cache_key)
        ok = report(response, f"request {i}", elapsed) and ok
        responses.append(response)

    prompt_total = sum(r.usage.prompt_tokens for r in responses if r.usage)
    cached_total = sum(cached_of(r) for r in responses)
    written_total = sum(r.usage.cache_write_tokens for r in responses if r.usage)
    cold = cached_of(responses[0])
    hits = sum(1 for r in responses[1:] if cached_of(r) - cold >= MIN_CACHED_PREFIX_TOKENS)
    print(f"\n   📈 Session: {cached_total}/{prompt_total} prompt tokens read from cache")
    print(f"   🎯 Repeats served from cache: {hits}/{len(responses) - 1}")
    if cached_total > prompt_total:
        print("   ❌ More tokens read from cache than were sent")
        ok = False
    elif hits == 0:
        print(
            "   ⚠️  No repeat was served from cache in this session. The requests may\n"
            "      have reached a backend that does not cache; the hit tests above\n"
            "      are what show caching works."
        )

    if pricing.cache_input is None:
        print("   💰 The catalog lists no cached-input price for this model: no saving to price")
    else:
        input_usd = pricing.input.usd
        saved = cached_total * (input_usd - pricing.cache_input.usd) / 1_000_000
        print(
            f"   💰 Saved on input: ${saved:.6f} "
            f"({cached_total} tokens at ${pricing.cache_input.usd}/1M instead of ${input_usd}/1M)"
        )
        if written_total and pricing.cache_write is not None:
            premium = written_total * (pricing.cache_write.usd - input_usd) / 1_000_000
            print(f"   ✍️  Cache writes: {written_total} tokens, ${premium:.6f} over the input rate")

    print("\n💡 Watch usage.cached_tokens and usage.cache_write_tokens in production.")
    return ok


def best_practices() -> bool:
    """Print best-practice tips for prompt caching (no API calls needed)."""
    print("\n\n📚 Prompt Caching Best Practices")
    print("-" * 50)

    tips = [
        (
            "🔑 Choose stable cache keys",
            "Use session IDs, user IDs, or conversation IDs as cache keys.\n"
            "      A shared key makes a hit more likely; it does not guarantee one.",
        ),
        (
            "📏 Cache long, static content",
            "System prompts, reference docs, and few-shot examples benefit most.\n"
            "      Prefixes under about 1024 tokens are usually not cached.",
        ),
        (
            "🧱 Put the static part first",
            "Caching matches the start of the prompt. Anything that changes per\n"
            "      request (timestamps, user input) belongs after the shared prefix.",
        ),
        (
            "🔄 Reuse keys across turns",
            "In multi-turn chats, keep the same prompt_cache_key for the entire\n"
            "      conversation so the growing prefix stays cached.",
        ),
        (
            "💰 Check the model's cache price",
            "pricing.cache_input in the model catalog is the cached-input rate;\n"
            "      a listed price does not prove the model serves hits, so measure.",
        ),
        (
            "📊 Measure with usage",
            "usage.cached_tokens counts cache reads and usage.cache_write_tokens\n"
            "      cache writes; a missing breakdown reads as 0.",
        ),
    ]

    for title, detail in tips:
        print(f"\n   {title}")
        print(f"      {detail}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all prompt caching examples and return the process exit code."""
    print("🚀 Venice AI Prompt Caching Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        selected = await select_model(client)
        if selected is None:
            return EXIT_SKIPPED
        model, pricing = selected
        print(f"🤖 Model: {model}")
        if pricing.cache_input is None:
            print(f"   Input ${pricing.input.usd}/1M tokens; no cached-input price listed\n")
        else:
            discount = (1 - pricing.cache_input.usd / pricing.input.usd) * 100
            print(
                f"   Input ${pricing.input.usd}/1M tokens, cached input "
                f"${pricing.cache_input.usd}/1M ({discount:.0f}% off)\n"
            )

        basic_ok = await basic_prompt_caching(client, model)
        multi_turn_ok = await multi_turn_caching(client, model)
        # Both sections pass only with a hit, and neither sends a marker.
        unmarked_hit = basic_ok or multi_turn_ok
        results = [
            ("Basic Prompt Caching", basic_ok),
            ("Multi-Turn Caching", multi_turn_ok),
            ("Cache Control Markers", await cache_control_markers(client, model, unmarked_hit)),
            ("Monitor Cache Performance", await monitor_cache_performance(client, model, pricing)),
            ("Best Practices", best_practices()),
        ]

    failed = [name for name, ok in results if not ok]
    print()
    for name, ok in results:
        print(f"   {'✓' if ok else '✗'} {name}")
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed")
        return 1

    print("\n✨ All prompt caching sections passed")
    print("\n💡 Key concepts demonstrated:")
    print("   - prompt_cache_key for routing affinity (top-level create() param)")
    print("   - cache_control markers on TextContent blocks")
    print("   - usage.cached_tokens / usage.cache_write_tokens")
    print("   - Multi-turn conversation caching patterns")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
