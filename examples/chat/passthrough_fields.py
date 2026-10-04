#!/usr/bin/env python3
"""
Venice AI SDK - Chat Completion Passthrough Fields
==================================================

Demonstrates the OpenAI-compatible passthrough fields plus Venice's
``prompt_cache_retention`` control on ``client.chat.completions.create()``:

- ``store`` — OpenAI-compat flag for server-side storage.
- ``text`` — OpenAI-compat text-config object (e.g. ``{"verbosity": "low"}``).
- ``include`` — OpenAI-compat include-specifier for response enrichment.
- ``metadata`` — free-form dict attached to the request (observability).
- ``prompt_cache_retention`` — ``"default"``, ``"extended"`` or ``"24h"``.
  Venice's API reference says ``"extended"`` and ``"24h"`` extend cache
  retention to 24 hours on models that support it. Pair it with
  ``prompt_cache_key`` so related requests are likely to reach the same cache.

The SDK forwards all five verbatim; whether a field changes the output is up
to the server and the upstream model. For example, most chat models accept
``text.verbosity`` but only some act on it, and a few reject values they don't
support with HTTP 400, so this example shows what was sent and what came back
rather than promising an effect.

What a run verifies:

- The request carrying all five fields is accepted and answered completely.
- Each retention tier is accepted and answered completely.
- For each tier, whether a warm repeat read the long prefix back from the
  cache (``usage.cached_tokens``). Cache reads are best-effort and can differ
  by tier, so a tier with no read is reported, not failed; the run fails only
  if no tier reads back at all.

It does not verify how long any tier keeps a prompt cached: that would take
hours of waiting, not one run. The closing summary lists, per tier, only what
the run observed.
"""

import asyncio
import secrets
import sys
from dataclasses import dataclass, field
from typing import Any

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters


def cut_off(response: ChatCompletionResponse, cap: int) -> bool:
    """True if a reply stopped at ``cap`` (its ``max_completion_tokens``) or has no usage.

    Some models report a reply cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against the cap as well.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    return finish_reason == "length" or used is None or used >= cap


def answer_ok(response: ChatCompletionResponse, cap: int) -> bool:
    """Print the answer and ``finish_reason``; ``False`` if truncated or empty."""
    text = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    print(text or "(empty response)")
    print(f"↳ finish_reason: {finish_reason}")
    if cut_off(response, cap):
        print("❌ The answer may be cut off at max_completion_tokens")
        return False
    if not text:
        print("❌ The model returned no visible answer")
        return False
    return True


async def demo_passthrough_fields(client: VeniceClient, model: str) -> bool:
    """Send a request with every passthrough field set, and show exactly what was sent."""
    print("🧾 Chat Completion — Passthrough Fields")
    print("-" * 30)
    print(f"📍 Using model: {model}")

    passthrough: dict[str, Any] = {
        "store": False,
        "text": {"verbosity": "low"},
        "include": [],
        "metadata": {"workflow": "sdk_example", "trace_id": "example-passthrough-001"},
        "prompt_cache_retention": "extended",
    }

    print("\n📤 Passthrough fields sent with this request:")
    for name, value in passthrough.items():
        print(f"   {name:<22} = {value!r}")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            UserMessage(content="Explain the solar system in exactly 3 short bullet points.")
        ],
        # Venice's own system prompt adds nothing this question needs.
        venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        max_completion_tokens=500,
        temperature=0.3,
        **passthrough,
    )

    print("\n💬 Response:")
    ok = answer_ok(response, 500)
    if response.usage:
        print(
            f"📊 Tokens: Input={response.usage.prompt_tokens}, "
            f"Output={response.usage.completion_tokens}"
        )
    print(
        "ℹ️ The request was accepted with all five fields. text.verbosity is a hint: "
        "models that don't support it answer at their normal length."
    )
    return ok


# One cold call writes the prefix to the cache; up to this many warm repeats,
# a short pause apart, look for the read. Venice serves some models from more
# than one backend, and prompt_cache_key makes reaching the same one likely,
# not certain, so a single warm miss is expected now and then.
MAX_WARM_CALLS = 3
WARM_CALL_DELAY_S = 2.0

# The cache calls ask for a one-word reply.
REPLY_CAP = 16


TIERS = ("default", "extended", "24h")


@dataclass
class CacheTierResult:
    """What the retention-tier demo observed, so the summary claims only that."""

    ok: bool
    #: Tiers whose every call returned a complete reply.
    accepted: list[str] = field(default_factory=list)
    #: Tiers where a warm repeat read the prefix back from the cache.
    read_back: list[str] = field(default_factory=list)
    #: Whether any call reported ``cache_write_tokens`` above zero.
    writes_reported: bool = False


async def demo_cache_retention_tiers(client: VeniceClient, model: str) -> CacheTierResult:
    """Send each retention tier and report whether a warm repeat read its prefix back.

    Retention (how long a cached prompt survives) can't be observed in one short
    run. What one run can show is that every tier is accepted, and whether
    repeats of the same long prefix are served from the cache:
    ``usage.cached_tokens`` counts the cached prompt tokens and
    ``usage.cache_write_tokens`` the tokens written, on models that report
    writes.

    Venice's own system prompt is a prefix shared by many requests and is often
    already cached, which would make a cold call look warm, so it is left out
    and a long prefix of our own is sent instead. A fresh run ID at the start
    of each tier's prefix keeps earlier runs from warming it.

    Pass/fail: every call must return a complete reply. Caching is best-effort,
    so a tier whose warm calls all miss is reported but is not an error; the
    run fails only if no tier at all reads its prefix back, because then it
    has not shown prompt caching.
    """
    print("\n🧊 prompt_cache_retention Tiers")
    print("-" * 30)
    print(f"📍 Using model: {model}")

    # A long, stable prefix is what prompt caching pays off for (for example an
    # agent's instructions). Caches only start above ~1024 tokens and store in
    # blocks, so the prefix is sized to about 3k tokens.
    policy = " ".join(
        f"Rule {i}: when a customer asks about order issue category {i}, apologise once, "
        f"confirm the order number, and offer the standard resolution for category {i}."
        for i in range(1, 91)
    )
    run_id = secrets.token_hex(4)
    params = VeniceParameters(include_venice_system_prompt=False)

    ok = True
    result = CacheTierResult(ok=True)
    for tier in TIERS:
        print(f"\n🕰️  Tier: {tier}")
        tier_complete = True
        system = SystemMessage(
            content=(
                f"Session {run_id}-{tier}. You are a support assistant. "
                f"Follow these rules. {policy}"
            )
        )
        cold_cached: int | None = None
        for call in range(MAX_WARM_CALLS + 1):
            if call > 0:
                await asyncio.sleep(WARM_CALL_DELAY_S)
            response = await client.chat.completions.create(
                model=model,
                messages=[system, UserMessage(content="Reply with the single word OK.")],
                max_completion_tokens=REPLY_CAP,
                temperature=0.0,
                venice_parameters=params,
                prompt_cache_retention=tier,
                prompt_cache_key=f"sdk-example-{run_id}-{tier}",
            )
            text = (response.text or "").strip()
            finish_reason = response.choices[0].finish_reason if response.choices else None
            usage = response.usage
            label = "cold" if call == 0 else f"warm {call}"
            print(
                f"   {label}: reply={text!r} finish_reason={finish_reason} "
                f"prompt_tokens={usage.prompt_tokens if usage else None} "
                f"cached_tokens={usage.cached_tokens if usage else None} "
                f"cache_write_tokens={usage.cache_write_tokens if usage else None}"
            )
            if cut_off(response, REPLY_CAP) or not text:
                print("   ❌ No complete reply")
                ok = False
                tier_complete = False
            if usage is not None and usage.cache_write_tokens:
                result.writes_reported = True
            if usage is None or usage.prompt_tokens == 0:
                # No prompt count means the usage block is not a measurement.
                print("   ⚠️ No prompt-token count was reported; this call can't show a read")
                continue
            if cold_cached is None:
                cold_cached = usage.cached_tokens
                if cold_cached:
                    print("   ℹ️ Part of the prefix was already cached before this run")
                continue
            if usage.cached_tokens > cold_cached:
                print(
                    f"   ✅ Read {usage.cached_tokens} of {usage.prompt_tokens} prompt tokens "
                    "from the cache"
                )
                result.read_back.append(tier)
                break
        else:
            print(
                f"   ⚠️ No cache read in {MAX_WARM_CALLS} warm calls for this tier; "
                "this run shows only that the tier was accepted"
            )
        if tier_complete:
            result.accepted.append(tier)

    print(f"\n📊 Tiers accepted with complete replies: {', '.join(result.accepted) or 'none'}")
    print(f"   Tiers that read their prefix back: {', '.join(result.read_back) or 'none'}")
    not_read = [tier for tier in TIERS if tier not in result.read_back]
    if not_read and result.read_back:
        print(f"   No cache read was observed for: {', '.join(not_read)}")
    if not result.read_back:
        print(
            "❌ No warm call read the cache, so this run did not show prompt caching. "
            "The catalog lists a cache price for this model, but a price does not "
            "guarantee hits are served or reported."
        )
        ok = False
    result.ok = ok
    return result


async def main() -> int:
    """Run all passthrough-field demos; return a process exit code."""
    print("🚀 Venice AI Chat — Passthrough Fields Example")
    print("=" * 50)

    async with VeniceClient() as client:
        # The cheapest model that answers directly (no hidden reasoning).
        try:
            model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return 77
        results: list[tuple[str, bool | None]] = [
            ("demo_passthrough_fields", await demo_passthrough_fields(client, model)),
        ]
        cache: CacheTierResult | None = None
        # The cache demo needs a model that caches prompts. The catalog has no
        # caching flag, only a cached-input price, so require_prompt_caching
        # filters on that price, and the catalog's default ranking picks among
        # those models.
        try:
            cache_model = await client.models.resolve_chat(require_prompt_caching=True)
        except NoMatchingModelError:
            print("\nSection skipped: no catalog model lists a cached-input price")
            results.append(("demo_cache_retention_tiers", None))
        else:
            cache = await demo_cache_retention_tiers(client, cache_model)
            results.append(("demo_cache_retention_tiers", cache.ok))

    failed = [name for name, ok in results if ok is False]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    skipped = [name for name, ok in results if ok is None]
    if skipped:
        print(f"\nℹ️ {len(skipped)} section(s) skipped: {', '.join(skipped)}")

    print("\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print("   - store / text / include / metadata (OpenAI-compat passthroughs)")
    if cache is not None:
        print(
            f"   - prompt_cache_retention with prompt_cache_key: {', '.join(cache.accepted)} "
            "accepted (retention length is not observable in one run)"
        )
        print(f"   - usage.cached_tokens showing a cache read for: {', '.join(cache.read_back)}")
        if cache.writes_reported:
            print("   - usage.cache_write_tokens reporting a cache write")
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
