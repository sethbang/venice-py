#!/usr/bin/env python3
"""
Venice AI SDK - Reasoning Effort
=================================

This example demonstrates how to use Venice AI's ``reasoning_effort`` parameter
to request a thinking depth from reasoning models, and measures what each tier
actually used: latency, reasoning tokens and whether the answer was right.

The effort enum has seven tiers:
``none | minimal | low | medium | high | xhigh | max``. Each model accepts its
own subset, listed in the catalog as ``capabilities.reasoningEffortOptions``,
so this example picks a model that supports ``reasoning_effort`` and runs the
tiers that model advertises.

Key features covered:
- reasoning_effort: top-level ``create()`` parameter
- The nested ``reasoning`` object with ``effort`` and ``summary`` controls
- Turning reasoning off with ``ReasoningConfig(enabled=False)``
- Lowest vs highest tier: both thinking tiers on a puzzle with one right
  answer, with their measured tokens and latency side by side
- Combining reasoning_effort with venice_parameters

The model is the cheapest reasoning model that accepts two or more thinking
tiers, so the whole run typically costs a few cents at most. Every request passes
``include_venice_system_prompt=False`` so only the prompt shown here is billed.

What a tier does
----------------
``supportsReasoningEffort`` means the model accepts the parameter. It does not
promise that thinking grows with the tier: some models scale their reasoning
with it and others barely change, or even think longer at a lower tier. The
catalog has no field that tells them apart, so this example checks what can be
checked on any model (``"none"`` and ``enabled=False`` turn thinking off,
every thinking tier thinks and answers correctly) and reports the reasoning
tokens each tier used as a measurement rather than asserting an order.

Answers and reasoning traces are reported separately. A truncated answer
(``finish_reason == "length"``) or an empty one fails its section, and the
script exits 1 if any section failed.
"""

import asyncio
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, get_args

from venice_ai import (
    NoMatchingModelError,
    ReasoningConfig,
    ReasoningEffortLevel,
    ReasoningSummary,
    VeniceClient,
    model_price,
)
from venice_ai.types.api import (
    ChatCompletionResponse,
    TextModelSpec,
    UserMessage,
    VeniceParameters,
)

# Room for the hidden reasoning plus the answer. Reasoning models spend most of
# their budget thinking (several thousand tokens on the puzzle below, at the
# lowest tier as well as the highest), and a smaller cap cuts the answer off.
MAX_TOKENS = 16384

# Exit code for "a prerequisite is missing" (here: no suitable reasoning model).
EXIT_SKIPPED = 77

# Bill only the prompt shown here, not Venice's own system prompt.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# A puzzle with one right answer ($0.05) and a tempting wrong one ($0.10).
PUZZLE = (
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
    "How much does the ball cost? Reply in at most two sentences and end with "
    "'Answer: $X.XX'."
)
PUZZLE_ANSWER = re.compile(r"answer:\s*\$?\s*0?\.05\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(*args: Any, **kwargs: Any) -> None:
    """``print`` that always flushes, so output survives an early process kill."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def _truncate(text: str, length: int = 200) -> str:
    """Shorten text for display with an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    return text[:length] + "..."


@dataclass
class Outcome:
    """What one request produced, with the answer and reasoning kept apart."""

    label: str
    elapsed: float
    finish_reason: str | None
    answer: str
    reasoning: str
    completion_tokens: int
    reasoning_tokens: int | None

    @property
    def ok(self) -> bool:
        return self.finish_reason != "length" and bool(self.answer.strip())

    @property
    def thought(self) -> bool:
        """Whether the response shows any reasoning (tokens or a trace)."""
        return bool(self.reasoning_tokens) or bool(self.reasoning)


def outcome(label: str, response: ChatCompletionResponse, elapsed: float) -> Outcome:
    """Collect the parts of a response this example reports on."""
    message = response.choices[0].message
    usage = response.usage
    details = usage.completion_tokens_details if usage else None
    return Outcome(
        label=label,
        elapsed=elapsed,
        finish_reason=response.choices[0].finish_reason,
        answer=str(response.text or ""),
        reasoning=str(message.reasoning_content or ""),
        completion_tokens=usage.completion_tokens if usage else 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", None),
    )


def show(result: Outcome) -> bool:
    """Print one outcome; return False (with the reason) if it failed."""
    log(f"   ⏱️  Latency: {result.elapsed:.2f}s   finish_reason: {result.finish_reason}")
    log(f"   📝 Answer ({len(result.answer)} chars): {_truncate(result.answer, 150)}")
    if result.reasoning:
        log(f"   🤔 Reasoning trace: {len(result.reasoning)} chars")
    reasoning_tokens = (
        f", reasoning: {result.reasoning_tokens}" if result.reasoning_tokens is not None else ""
    )
    log(f"   📊 [{result.label}] completion tokens: {result.completion_tokens}{reasoning_tokens}")
    if not result.ok:
        log(f"   ❌ [{result.label}] the answer was truncated or empty")
    return result.ok


async def ask(
    client: VeniceClient,
    model: str,
    prompt: str,
    label: str,
    *,
    venice_parameters: VeniceParameters = OWN_PROMPT_ONLY,
    **kwargs: Any,
) -> Outcome:
    """Send one prompt with the given reasoning settings and time it."""
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[UserMessage(content=prompt)],
        max_completion_tokens=MAX_TOKENS,
        venice_parameters=venice_parameters,
        **kwargs,
    )
    return outcome(label, response, time.perf_counter() - start)


async def select_model(client: VeniceClient) -> tuple[str, list[ReasoningEffortLevel]] | None:
    """Pick a model that accepts reasoning_effort, with its advertised tiers.

    The cheapest reasoning model is used when it supports ``reasoning_effort``
    with at least two thinking tiers; otherwise the cheapest catalog model that
    does, ranked by the same price measure ``prefer="cheapest"`` uses. Returns
    None (after printing why) when the catalog has no such model.
    """
    try:
        preferred = await client.models.resolve_chat(require_reasoning=True, prefer="cheapest")
    except NoMatchingModelError as e:
        log(f"SKIPPED: the catalog has no reasoning chat model ({e})")
        return None
    catalog = await client.models.list(type="text")

    def effort_tiers(spec: object) -> list[ReasoningEffortLevel]:
        if not isinstance(spec, TextModelSpec) or spec.beta or spec.offline:
            return []
        caps = spec.capabilities
        if caps is None or not caps.supportsReasoningEffort or caps.supportsE2EE:
            return []
        options = caps.reasoningEffortOptions or []
        return [tier for tier in get_args(ReasoningEffortLevel) if tier in options]

    def rank(model: Any) -> tuple[bool, bool, float]:
        price = model_price(model.model_dump())
        return (model.id != preferred, price is None, price or 0.0)

    candidates = sorted(catalog.data, key=rank)
    for model in candidates:
        tiers = effort_tiers(model.model_spec)
        if len([t for t in tiers if t != "none"]) >= 2:
            return model.id, tiers
    log("SKIPPED: no model in the catalog advertises two or more reasoning_effort tiers")
    return None


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


async def basic_reasoning_effort(
    client: VeniceClient, model: str, tiers: list[ReasoningEffortLevel]
) -> bool:
    """Run the same prompt at every tier the model advertises.

    ``"none"``, when offered, must answer without reasoning tokens, and every
    other tier must show some reasoning. How many reasoning tokens each tier
    used is reported, not checked: the order is up to the model.
    """
    log("🧠 Basic Reasoning Effort — Comparing Effort Tiers")
    log("-" * 60)
    log(f"   Advertised tiers: {', '.join(tiers)}; running all of them on one prompt")

    prompt = "In one short paragraph, why do a triangle's angles sum to 180°?"
    ok = True
    results: list[Outcome] = []
    for level in tiers:
        log(f'\n   ⚙️  reasoning_effort = "{level}"')
        result = await ask(client, model, prompt, level, reasoning_effort=level)
        ok = show(result) and ok
        if level == "none" and result.thought:
            log(f"   ❌ [none] still reasoned ({result.reasoning_tokens} reasoning tokens)")
            ok = False
        if level != "none" and not result.thought:
            log(f"   ❌ [{level}] showed no reasoning, so the tier did not think")
            ok = False
        results.append(result)

    log(f"\n   {'Level':<8} {'Time':>8} {'Tokens':>8} {'Reasoning':>10} {'Finish':>8}")
    for r in results:
        log(
            f"   {r.label:<8} {r.elapsed:>7.2f}s {r.completion_tokens:>8} "
            f"{r.reasoning_tokens or 0:>10} {r.finish_reason or '-':>8}"
        )
    thinking = [r for r in results if r.label != "none" and r.reasoning_tokens is not None]
    if len(thinking) >= 2:
        counts = [r.reasoning_tokens or 0 for r in thinking]
        order = "rose with every step up" if counts == sorted(counts) else "did not rise steadily"
        log(
            f"\n📏 Measured: reasoning tokens {order} across "
            f"{', '.join(f'{r.label} {r.reasoning_tokens}' for r in thinking)}."
        )
    log("💡 A tier is a request. Whether thinking grows with it depends on the model,")
    log("   so measure the tiers on your own prompts before choosing one for cost.")
    return ok


async def lowest_vs_highest_tier(
    client: VeniceClient, model: str, tiers: list[ReasoningEffortLevel]
) -> bool:
    """Run the lowest and the highest thinking tier on a puzzle with one answer.

    The bat-and-ball puzzle has a tempting wrong answer, so it shows whether a
    tier got it right, not just how long it took. The section fails if either
    tier answers wrongly, is cut off, or shows no reasoning at all (neither
    reasoning tokens nor a reasoning trace: a thinking tier that did not
    think). Which tier used more tokens or time is printed as measured, not
    assumed.
    """
    log("\n\n⚡ Lowest vs Highest Thinking Tier")
    log("-" * 60)

    thinking = [t for t in tiers if t != "none"]
    low, high = thinking[0], thinking[-1]
    log(f"   Puzzle: {PUZZLE}")

    ok = True
    results: list[Outcome] = []
    for level in (low, high):
        log(f'\n   ⚙️  reasoning_effort = "{level}"')
        result = await ask(client, model, PUZZLE, level, reasoning_effort=level)
        ok = show(result) and ok
        if not PUZZLE_ANSWER.search(result.answer):
            log(f"   ❌ [{level}] the answer does not end with the right amount ($0.05)")
            ok = False
        if not result.thought:
            log(f"   ❌ [{level}] showed no reasoning, so the tier did not think")
            ok = False
        results.append(result)

    log(f"\n   {'Level':<8} {'Time':>8} {'Tokens':>8} {'Reasoning':>10} {'Correct':>8}")
    for r in results:
        correct = "yes" if PUZZLE_ANSWER.search(r.answer) else "no"
        log(
            f"   {r.label:<8} {r.elapsed:>7.2f}s {r.completion_tokens:>8} "
            f"{r.reasoning_tokens or 0:>10} {correct:>8}"
        )
    low_r, high_r = results
    log("\n📏 Measured on this run:")
    measured = (
        (
            "completion tokens",
            f"{low_r.completion_tokens}",
            f"{high_r.completion_tokens}",
            low_r.completion_tokens - high_r.completion_tokens,
        ),
        (
            "latency",
            f"{low_r.elapsed:.2f}s",
            f"{high_r.elapsed:.2f}s",
            low_r.elapsed - high_r.elapsed,
        ),
    )
    for metric, low_text, high_text, difference in measured:
        lower = high if difference > 0 else low
        verdict = "equal" if difference == 0 else f"{lower} was lower"
        log(f"   {metric}: {low} {low_text}, {high} {high_text} ({verdict})")
    log("\n💡 One sample per tier on one prompt says little about another prompt.")
    log("   Measure tokens, latency and correctness per tier on your own workload")
    log("   before picking one; a higher tier is not guaranteed to think longer.")
    return ok


async def disabling_reasoning(
    client: VeniceClient, model: str, tiers: list[ReasoningEffortLevel]
) -> bool:
    """Turn reasoning off with ``ReasoningConfig(enabled=False)``.

    ``reasoning.enabled=False`` is the switch Venice recommends for turning
    reasoning off; ``reasoning_effort="none"`` (run in the basic section when
    the model offers it) is the other route. On a model that lists ``"none"``
    among its tiers the answer must arrive with zero reasoning tokens.
    """
    log("\n\n🔕 Disabling Reasoning — ReasoningConfig(enabled=False)")
    log("-" * 60)

    log("\n   🧾 reasoning=ReasoningConfig(enabled=False)")
    result = await ask(
        client,
        model,
        "Name the capital of France in one word.",
        "disabled",
        reasoning=ReasoningConfig(enabled=False),
    )
    ok = show(result)
    if "paris" not in result.answer.lower():
        log("   ❌ The answer does not name Paris")
        ok = False
    if "none" in tiers:
        if result.reasoning_tokens or result.reasoning:
            log(
                f"   ❌ The model still reasoned ({result.reasoning_tokens} tokens, "
                f"{len(result.reasoning)}-char trace)"
            )
            ok = False
        else:
            log("   ✅ Answered with no reasoning tokens and no trace")
    else:
        log(f"   ℹ️  {model} does not list 'none', so it may reason regardless")
    return ok


async def combining_with_venice_parameters(
    client: VeniceClient, model: str, tiers: list[ReasoningEffortLevel]
) -> bool:
    """Show reasoning_effort alongside venice_parameters.strip_thinking_response."""
    log("\n\n🔗 Combining reasoning_effort with Venice Parameters")
    log("-" * 60)

    # strip_thinking_response works the same at any tier, so use the lightest
    # one that still thinks ("none", when offered, is listed first).
    level = tiers[1] if tiers[0] == "none" else tiers[0]
    prompt = "Explain the halting problem in simple terms, in three sentences."

    log(f'\n   🧠 reasoning_effort="{level}", thinking visible (default)')
    visible = await ask(client, model, prompt, f"{level} + visible", reasoning_effort=level)
    ok = show(visible)

    log(f'\n   🧠 reasoning_effort="{level}", strip_thinking_response=True')
    stripped = await ask(
        client,
        model,
        prompt,
        f"{level} + stripped",
        reasoning_effort=level,
        venice_parameters=VeniceParameters(
            include_venice_system_prompt=False,
            strip_thinking_response=True,
        ),
    )
    ok = show(stripped) and ok

    log("\n   🔍 Observed:")
    if not visible.reasoning and not stripped.reasoning:
        log("   This model returned no reasoning trace in either request, so there was")
        log("   nothing visible for strip_thinking_response to remove.")
    else:
        log(f"   Reasoning trace without stripping: {len(visible.reasoning)} chars")
        log(f"   Reasoning trace with stripping:    {len(stripped.reasoning)} chars")
    if stripped.reasoning_tokens:
        log(
            f"   The stripped request still reported {stripped.reasoning_tokens} reasoning "
            "tokens: the model still thinks, only the returned trace is removed."
        )
    if "<think>" in visible.answer or "<think>" in stripped.answer:
        log("   A <think> block appeared inside an answer.")

    log("\n💡 Key distinctions:")
    log("   • reasoning_effort asks the model for a thinking depth")
    log("   • strip_thinking_response removes <think> blocks from the returned text")
    return ok


async def nested_reasoning_object(
    client: VeniceClient, model: str, tiers: list[ReasoningEffortLevel]
) -> bool:
    """Send the nested ``reasoning`` object: an ``effort`` and a ``summary``.

    This section sets only the nested object, not the top-level
    ``reasoning_effort``. It checks that the nested ``effort`` is applied:
    the lightest thinking tier must reason and, on a model that lists
    ``"none"``, ``effort="none"`` must answer with no reasoning at all.

    ``summary`` (``"auto"`` / ``"concise"`` / ``"detailed"``) asks for a
    style of reasoning summary. The catalog does not say which models honor
    it, so the trace lengths of two settings are reported as a measurement,
    not checked.
    """
    log("\n\n🧩 Nested reasoning Config — effort + summary")
    log("-" * 60)

    # The lightest tier that thinks ("none", when offered, is listed first).
    level = tiers[1] if tiers[0] == "none" else tiers[0]
    prompt = "Briefly explain why prime numbers matter in cryptography."
    summaries: list[ReasoningSummary] = ["concise", "detailed"]

    ok = True
    results: list[Outcome] = []
    for summary in summaries:
        log(f"\n   🧾 reasoning=ReasoningConfig(effort={level!r}, summary={summary!r})")
        result = await ask(
            client, model, prompt, summary, reasoning=ReasoningConfig(effort=level, summary=summary)
        )
        ok = show(result) and ok
        if not result.thought:
            log(f"   ❌ [{summary}] effort={level!r} showed no reasoning")
            ok = False
        results.append(result)

    if "none" in tiers:
        log("\n   🧾 reasoning=ReasoningConfig(effort='none')")
        off = await ask(
            client, model, prompt, "effort none", reasoning=ReasoningConfig(effort="none")
        )
        ok = show(off) and ok
        if off.thought:
            log(f"   ❌ effort='none' in the nested object still reasoned ({off.reasoning_tokens})")
            ok = False
        else:
            log("   ✅ The nested effort is applied: 'none' answered without reasoning")
    else:
        log(f"   ℹ️  {model} does not list 'none'; only the thinking tier was checked")

    concise, detailed = results
    log(
        f"\n   📏 Measured reasoning trace: concise {len(concise.reasoning)} chars, "
        f"detailed {len(detailed.reasoning)} chars"
    )
    if not concise.reasoning and not detailed.reasoning:
        log("   This model returned no reasoning trace, so summary had nothing to shape.")
    elif len(concise.reasoning) >= len(detailed.reasoning):
        log("   The concise trace was not shorter: this model shows no sign of honoring summary.")
    else:
        log("   The concise trace was shorter here; one pair of samples is not proof either way.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all reasoning effort examples and return the process exit code."""
    log("🚀 Venice AI Reasoning Effort Examples")
    log("=" * 60)

    async with VeniceClient() as client:
        selected = await select_model(client)
        if selected is None:
            return EXIT_SKIPPED
        model, tiers = selected
        log(f"🤖 Model: {model}")
        log(f"   reasoning_effort tiers it accepts: {', '.join(tiers)}\n")

        results = [
            ("basic_reasoning_effort", await basic_reasoning_effort(client, model, tiers)),
            ("lowest_vs_highest_tier", await lowest_vs_highest_tier(client, model, tiers)),
            ("disabling_reasoning", await disabling_reasoning(client, model, tiers)),
            (
                "combining_with_venice_parameters",
                await combining_with_venice_parameters(client, model, tiers),
            ),
            ("nested_reasoning_object", await nested_reasoning_object(client, model, tiers)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        log(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    log(f"\n✨ All {len(results)} reasoning effort sections passed")
    log("\n💡 Key concepts demonstrated:")
    log("   - reasoning_effort is a top-level create() parameter, not in venice_parameters")
    log("   - capabilities.reasoningEffortOptions lists the tiers a model accepts")
    log("   - The nested reasoning object carries effort (and an optional summary style)")
    log("   - ReasoningConfig(enabled=False) turns reasoning off")
    log("   - Combine with strip_thinking_response to hide reasoning from output")
    log("\n📚 Next Steps:")
    log("   - Read reasoningEffortOptions and defaultReasoningEffort from the catalog")
    log("   - Measure reasoning tokens per tier on your prompts; models differ in how")
    log("     much a tier changes")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
