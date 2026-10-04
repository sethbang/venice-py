#!/usr/bin/env python3
"""
Venice AI SDK - Venice Parameters Showcase
==========================================

This example demonstrates the Venice-specific parameters available through the VeniceParameters class.
These parameters provide fine-grained control over Venice AI's unique features including character personalities,
reasoning capabilities, web search integration, and system prompt behavior.

Venice Parameters Covered:
    • character_slug: Use public Venice character personalities
    • strip_thinking_response: Hide the reasoning trace from the response
    • disable_thinking: Disable reasoning on reasoning-capable models
    • enable_web_search: Enable web search for current knowledge
    • enable_web_citations: Request citation formatting in responses
    • return_search_results_as_documents: Surface search results as a tool call
    • include_venice_system_prompt: Control Venice system prompt inclusion

Each section checks that its parameter had the effect it claims (citations
returned, reasoning tokens gone, prompt tokens changed, ...) and that the
answer was not cut off at ``max_completion_tokens``. Any failed check makes the
script exit 1. A section whose model or character is missing from the catalog
prints ``Section skipped:`` and the others still run; the script exits 77 only
if every section was skipped.

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-py SDK

Performance note:
    Independent calls inside a demo are dispatched concurrently with
    ``asyncio.gather``; a failure in either call propagates and stops the run.
"""

import asyncio
import json
import math
import sys

from venice_ai import NoMatchingModelError, VeniceClient, model_price
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, TextModelSpec, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Leaving Venice's system prompt out (or adding a character prompt) must move
# the prompt size by at least this many tokens to count as an effect.
MIN_PROMPT_DELTA = 100

# Room for a reasoning model to think and still answer. Small reasoning models
# vary widely in how long they think, even on easy questions, so the cap leaves
# several thousand tokens of headroom over a typical run.
REASONING_CAP = 16384

# Enough for a short in-character introduction from a non-reasoning model.
CHARACTER_CAP = 300

# Ample for a two-or-three-sentence answer with citation markers.
WEB_ANSWER_CAP = 800

# Ample for a two-or-three-sentence explanation from a non-reasoning model.
SHORT_ANSWER_CAP = 400

# =============================================================================
# Helper Functions
# =============================================================================


def print_usage(response: ChatCompletionResponse) -> None:
    """Print token usage from a response, including reasoning tokens."""
    usage = response.usage
    if usage is None:
        print("\n📊 Token Usage: not provided by the API", flush=True)
        return
    details = usage.completion_tokens_details
    print(
        f"\n📊 Token Usage: Input={usage.prompt_tokens}, Output={usage.completion_tokens} "
        f"(reasoning={details.reasoning_tokens if details else None}), "
        f"Total={usage.total_tokens}",
        flush=True,
    )


def print_answer(response: ChatCompletionResponse, cap: int) -> bool:
    """Print the visible answer and ``finish_reason``; ``False`` if unusable.

    An empty answer is a failure: the reasoning trace is never a substitute
    for the answer. ``finish_reason`` alone cannot rule out truncation: some
    models report an answer cut off at the cap as ``"stop"`` (or as
    ``"tool_calls"`` when ``return_search_results_as_documents`` is on). So a
    completion count that reached ``cap`` (the request's
    ``max_completion_tokens``) also counts as cut off, and so does a response
    with no usage to check.
    """
    text = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(text or "(empty response)", flush=True)
    print(f"↳ finish_reason: {finish_reason} ({used} of {cap} completion tokens)", flush=True)
    if finish_reason == "length" or used is None or used >= cap:
        print("❌ The answer may be cut off at max_completion_tokens", flush=True)
        return False
    if not text:
        print("❌ The model returned no visible answer", flush=True)
        return False
    return True


def check(condition: bool, success: str, failure: str) -> bool:
    """Print a ✅/❌ line for one verified effect and return the condition."""
    print(f"{'✅' if condition else '❌'} {success if condition else failure}", flush=True)
    return condition


def reasoning_tokens(response: ChatCompletionResponse) -> int | None:
    """Reasoning tokens reported for a response, if any."""
    if response.usage is None or response.usage.completion_tokens_details is None:
        return None
    return response.usage.completion_tokens_details.reasoning_tokens


def prompt_tokens(response: ChatCompletionResponse) -> int:
    """Prompt tokens for a response (0 if usage is missing)."""
    return response.usage.prompt_tokens if response.usage else 0


def display_parameters(params: VeniceParameters, title: str = "VeniceParameters") -> None:
    """Display the fields that were set explicitly, including False/"off" values."""
    print(f"\n🔧 {title} (set explicitly):", flush=True)
    for key, value in params.model_dump(exclude_unset=True).items():
        print(f"   {key}: {value}", flush=True)


def print_citations(response: ChatCompletionResponse, title: str, limit: int = 5) -> int:
    """Print the first few web citations and return how many there were."""
    citations = response.web_search_citations
    if citations:
        print_subsection(title, "📚")
        for i, citation in enumerate(citations[:limit], 1):
            print(f"   [{i}] {citation.title} - {citation.url}", flush=True)
        if len(citations) > limit:
            print(f"   ... and {len(citations) - limit} more", flush=True)
    return len(citations)


def print_section_header(title: str, emoji: str = "📋") -> None:
    """Print a formatted section header."""
    print(f"\n{emoji} {title}", flush=True)
    print("=" * 70, flush=True)


def print_subsection(title: str, emoji: str = "📍") -> None:
    """Print a formatted subsection header."""
    print(f"\n{emoji} {title}", flush=True)
    print("-" * 50, flush=True)


# =============================================================================
# Example Functions
# =============================================================================


async def character_example(client: VeniceClient) -> bool | None:
    """Demonstrate ``character_slug`` with a real Venice character.

    A character carries its own persona prompt and names the model it was
    built for (``modelId``); chatting with it on that model gives the persona
    as designed. Venice sends the persona prompt in place of its own default
    system prompt, so the persona's size shows up against the bare question
    sent without Venice's system prompt (``include_venice_system_prompt=False``).

    Characters run on whatever model their author chose, so check the price
    first: this picks the character whose model is cheapest, preferring one
    that does not reason (a short reply then needs only a small token cap).
    Returns ``None`` (section skipped) when no listed character names a model
    from the current catalog.
    """
    print_section_header("Character-Based Chat Example", "🎭")

    characters = (await client.characters.list()).data or []
    text_models = {model.id: model for model in (await client.models.list(type="text")).data}

    def cost_rank(model_id: str) -> tuple[bool, float]:
        entry = text_models[model_id]
        spec = entry.model_spec
        caps = spec.capabilities if isinstance(spec, TextModelSpec) else None
        price = model_price(entry.model_dump())
        reasons = bool(caps and caps.supportsReasoning)
        return reasons, price if price is not None else math.inf

    # A character is only usable if the model it names is in the current catalog.
    usable = [c for c in characters if not c.adult and c.modelId in text_models]
    if not usable:
        print(
            "Section skipped: no listed character names a text model from the current catalog",
            flush=True,
        )
        return None

    character = min(usable, key=lambda c: cost_rank(c.modelId))
    model = character.modelId
    reasons, blended = cost_rank(model)
    cap = REASONING_CAP if reasons else CHARACTER_CAP
    print(f"📍 Using character: {character.slug} ({character.name})", flush=True)
    if character.description:
        print(f"   Description: {character.description[:200]}", flush=True)
    price_note = (
        f"~${blended:.3f} per 1M tokens, blended" if math.isfinite(blended) else "no catalog price"
    )
    print(f"📍 Using the character's model: {model} ({price_note})", flush=True)

    venice_params = VeniceParameters(character_slug=character.slug)
    display_parameters(venice_params)

    # An in-character question: the persona decides how it is answered.
    question = "Introduce yourself in two or three sentences: who are you and what do you do?"
    print(f"\n💬 Question: {question}", flush=True)

    async def ask(params: VeniceParameters | None) -> ChatCompletionResponse:
        return await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=question)],
            venice_parameters=params,
            max_completion_tokens=cap,
            temperature=0.7,
        )

    # The bare question, without Venice's system prompt, shows how much prompt
    # the persona adds.
    bare = VeniceParameters(include_venice_system_prompt=False)
    response, plain = await asyncio.gather(ask(venice_params), ask(bare))

    print_subsection("Character Response", "📝")
    ok = print_answer(response, cap)
    print_usage(response)

    # The bare request is not near zero: the model's chat template, and any
    # prompt its provider adds, are in both requests even with Venice's system
    # prompt off. The difference is what selecting the character changed.
    delta = prompt_tokens(response) - prompt_tokens(plain)
    print(
        f"📏 Prompt tokens: character {prompt_tokens(response)}, bare question "
        f"{prompt_tokens(plain)}",
        flush=True,
    )
    ok = (
        check(
            delta >= MIN_PROMPT_DELTA,
            f"Character persona injected: +{delta} prompt tokens vs the bare question",
            f"Prompt grew by only {delta} tokens, so no character prompt was added",
        )
        and ok
    )
    echoed = response.venice_parameters.character_slug if response.venice_parameters else None
    print(f"📡 Server echoed character_slug: {echoed}", flush=True)
    return ok


async def thinking_control_example(client: VeniceClient) -> bool | None:
    """Compare thinking on, ``disable_thinking=True`` and ``strip_thinking_response=True``.

    Disabling thinking skips reasoning; stripping still reasons but leaves the
    trace out of the response.

    The catalog has no flag for ``disable_thinking``; a model that accepts
    ``reasoning_effort="none"`` is one whose reasoning can be switched off.
    """
    print_section_header("Thinking Control Example", "🧠")

    try:
        model = await client.models.resolve_chat(
            require_reasoning=True, require_reasoning_effort="none", prefer="cheapest"
        )
    except NoMatchingModelError:
        print("Section skipped: no reasoning model can switch reasoning off", flush=True)
        return None
    print(f"📍 Using model: {model}", flush=True)

    # The clues admit exactly one solution: Alice=blue, Bob=green, Carol=red.
    puzzle = (
        "Alice, Bob, and Carol each picked a different color: red, green, or blue. "
        "Alice picked neither red nor green. Bob's color comes alphabetically before "
        "Carol's. What did each person pick? Reply with one line of the form "
        "'Alice=<color>, Bob=<color>, Carol=<color>'."
    )
    expected = ("alice=blue", "bob=green", "carol=red")
    print(f"\n💭 Question: {puzzle}", flush=True)

    # Venice's system prompt is left out of all three, so they differ only in
    # the thinking settings.
    params_on = VeniceParameters(disable_thinking=False, include_venice_system_prompt=False)
    params_off = VeniceParameters(disable_thinking=True, include_venice_system_prompt=False)
    params_stripped = VeniceParameters(
        strip_thinking_response=True, include_venice_system_prompt=False
    )
    print_subsection("Test 1: Thinking on", "🔍")
    display_parameters(params_on)
    print_subsection("Test 2: Thinking disabled", "✂️")
    display_parameters(params_off)
    print_subsection("Test 3: Thinking on, trace stripped from the response", "🙈")
    display_parameters(params_stripped)
    print("\n⏳ Dispatching all three completions concurrently...", flush=True)

    response_on, response_off, response_stripped = await asyncio.gather(
        *[
            client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=puzzle)],
                venice_parameters=params,
                # No temperature override: model publishers recommend their own
                # sampling settings for thinking mode, and they differ from
                # model to model, so the server default applies.
                max_completion_tokens=REASONING_CAP,
            )
            for params in (params_on, params_off, params_stripped)
        ]
    )

    ok = True
    for label, response in (
        ("thinking on", response_on),
        ("thinking off", response_off),
        ("trace stripped", response_stripped),
    ):
        print_subsection(f"Model Response ({label})", "📝")
        if response.thinking_blocks:
            trace = response.thinking_blocks[0].strip().splitlines()
            print(f"💭 Reasoning trace ({len(trace)} lines), first line: {trace[0]}", flush=True)
        ok = print_answer(response, REASONING_CAP) and ok
        print_usage(response)
        answer = "".join((response.text or "").lower().replace("*", "").split())
        correct = all(pair in answer for pair in expected)
        if response is response_off:
            # Without reasoning a small model may guess wrong; what this run
            # verifies for the "off" variant is that no reasoning happened.
            print(
                f"ℹ️ Without reasoning the answer is {'correct' if correct else 'wrong'}",
                flush=True,
            )
            continue
        ok = (
            check(
                correct,
                "Correct solution: Alice=blue, Bob=green, Carol=red",
                "The answer does not state the correct solution",
            )
            and ok
        )

    print_subsection("Reasoning On vs Off Comparison", "🔬")
    on_tokens, off_tokens = reasoning_tokens(response_on), reasoning_tokens(response_off)
    print(f"   reasoning_tokens: on={on_tokens}, off={off_tokens}", flush=True)
    ok = (
        check(
            bool(on_tokens) and off_tokens == 0 and not response_off.thinking_blocks,
            "disable_thinking skipped reasoning entirely (0 reasoning tokens, no trace)",
            "disable_thinking did not stop reasoning on this model",
        )
        and ok
    )
    stripped_tokens = reasoning_tokens(response_stripped)
    print(f"   reasoning_tokens with the trace stripped: {stripped_tokens}", flush=True)
    ok = (
        check(
            bool(stripped_tokens) and not response_stripped.thinking_blocks,
            "strip_thinking_response kept the reasoning but left the trace out of the response",
            "strip_thinking_response did not reason with a hidden trace on this model",
        )
        and ok
    )
    return ok


async def web_search_example(client: VeniceClient) -> bool | None:
    """Search the web, get citations, and also get the raw results as documents.

    ``enable_web_search`` runs the search and ``enable_web_citations`` returns
    the sources as ``response.web_search_citations``. With
    ``return_search_results_as_documents=True`` the response also carries a
    ``web_search`` tool call whose arguments hold the documents the answer was
    grounded on, so your own code can reuse or display them.
    """
    print_section_header("Web Search Example", "🌐")

    # A model that answers directly, so the cap goes to the answer, not to reasoning.
    try:
        web_model = await client.models.resolve_chat(
            require_web_search=True, exclude_reasoning=True, prefer="cheapest"
        )
    except NoMatchingModelError:
        print("Section skipped: no non-reasoning model supports web search", flush=True)
        return None
    print(f"📍 Using model: {web_model}", flush=True)

    question = "In two or three sentences, what is the current status of renewable energy adoption?"
    print(f"\n🔍 Question requiring current info: {question}", flush=True)

    venice_params = VeniceParameters(
        enable_web_search="on",  # Always search ("auto" lets the model decide)
        enable_web_citations=True,  # Ask for [REF] citation markers in the answer
        return_search_results_as_documents=True,  # Raw results as a tool call
        include_venice_system_prompt=False,  # Our system message is enough
    )
    display_parameters(venice_params)

    response = await client.chat.completions.create(
        model=web_model,
        messages=[
            SystemMessage(content="Provide current, accurate information and cite your sources."),
            UserMessage(content=question),
        ],
        venice_parameters=venice_params,
        max_completion_tokens=WEB_ANSWER_CAP,
        temperature=0.3,
    )

    print_subsection("Response with Web Search", "📝")
    # With return_search_results_as_documents the documents arrive as a tool
    # call, so a complete answer can end with finish_reason "tool_calls", and
    # some models report a truncated one that way too. print_answer therefore
    # also checks the completion count against the cap.
    ok = print_answer(response, WEB_ANSWER_CAP)
    count = print_citations(response, "Web Citations")
    print_usage(response)
    ok = (
        check(
            count > 0,
            f"Web search ran: {count} citations returned",
            "No citations returned, so no web search happened",
        )
        and ok
    )

    tool_calls = (response.choices[0].message.tool_calls or []) if response.choices else []
    documents: list[dict] = []
    print_subsection("Search Results as Tool Calls", "🔧")
    for tool_call in tool_calls:
        arguments = json.loads(tool_call.function.arguments or "{}")
        docs = arguments.get("documents") or []
        documents.extend(docs)
        print(f"   {tool_call.function.name}: {len(docs)} documents", flush=True)
    for doc in documents[:5]:
        print(f"   [{doc.get('id')}] {doc.get('title')} - {doc.get('url')}", flush=True)

    return (
        check(
            len(documents) > 0,
            f"{len(documents)} search documents returned as a tool call",
            "No search documents came back as a tool call",
        )
        and ok
    )


async def system_prompt_control_example(client: VeniceClient) -> bool | None:
    """Demonstrate ``include_venice_system_prompt``.

    Venice adds its own system prompt to every request unless told not to.
    The clearest evidence is the prompt size.
    """
    print_section_header("System Prompt Control Example", "🔧")

    # A model that answers directly, so the prompt-token comparison is not
    # mixed up with reasoning output.
    try:
        model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
    except NoMatchingModelError:
        print("Section skipped: the catalog lists no non-reasoning chat model", flush=True)
        return None
    print(f"📍 Using model: {model}", flush=True)

    question = "Explain quantum computing in two or three simple sentences."
    print(f"\n❓ Question: {question}", flush=True)

    params_with = VeniceParameters(include_venice_system_prompt=True)
    params_without = VeniceParameters(include_venice_system_prompt=False)

    print_subsection("Test 1: With Venice System Prompt", "📥")
    display_parameters(params_with)
    print_subsection("Test 2: Without Venice System Prompt", "🚫")
    display_parameters(params_without)
    print("\n⏳ Dispatching both completions concurrently...", flush=True)

    system_msg = SystemMessage(content="You are a helpful technical assistant.")
    response_with, response_without = await asyncio.gather(
        *[
            client.chat.completions.create(
                model=model,
                messages=[system_msg, UserMessage(content=question)],
                venice_parameters=params,
                max_completion_tokens=SHORT_ANSWER_CAP,
                temperature=0.7,
            )
            for params in (params_with, params_without)
        ]
    )

    print_subsection("Response With Venice System Prompt", "📝")
    ok = print_answer(response_with, SHORT_ANSWER_CAP)
    print_usage(response_with)

    print_subsection("Response Without Venice System Prompt", "📝")
    ok = print_answer(response_without, SHORT_ANSWER_CAP) and ok
    print_usage(response_without)

    print_subsection("Comparison", "📏")
    with_tokens, without_tokens = prompt_tokens(response_with), prompt_tokens(response_without)
    print(f"   Prompt tokens with Venice system prompt:    {with_tokens}", flush=True)
    print(f"   Prompt tokens without Venice system prompt: {without_tokens}", flush=True)
    return (
        check(
            with_tokens - without_tokens >= MIN_PROMPT_DELTA,
            f"The Venice system prompt accounts for {with_tokens - without_tokens} prompt tokens",
            "Prompt size barely changed, so the flag had no visible effect",
        )
        and ok
    )


async def comprehensive_example(client: VeniceClient) -> bool | None:
    """Combine web search, citations and a visible reasoning trace in one call."""
    print_section_header("Comprehensive Example - Combined Parameters", "🎯")

    try:
        model = await client.models.resolve_chat(
            require_web_search=True, require_reasoning=True, prefer="cheapest"
        )
    except NoMatchingModelError:
        print("Section skipped: no reasoning model supports web search", flush=True)
        return None
    print(f"📍 Using model: {model}", flush=True)

    question = (
        "I'm planning to start a renewable energy company. In three short bullet "
        "points, what are the key technologies to focus on and what makes a "
        "successful clean energy startup?"
    )
    print(f"\n💼 Business Question:\n{question}", flush=True)

    venice_params = VeniceParameters(
        strip_thinking_response=False,  # Keep the reasoning trace in the response
        disable_thinking=False,  # Let the model reason
        enable_web_search="on",  # Get current market data
        enable_web_citations=True,  # Cite sources
        include_venice_system_prompt=False,  # Our system message sets the role
    )
    display_parameters(venice_params, "Comprehensive Configuration")

    response = await client.chat.completions.create(
        model=model,
        messages=[
            SystemMessage(
                content=(
                    "You are an expert business consultant specializing in clean "
                    "energy and startups. Be concise."
                )
            ),
            UserMessage(content=question),
        ],
        venice_parameters=venice_params,
        # Server-default sampling for the reasoning model (see thinking_control_example).
        max_completion_tokens=REASONING_CAP,
    )

    ok = True
    # Reasoning models return the trace in reasoning_content (or as <think> tags);
    # thinking_blocks reads either shape.
    if response.thinking_blocks:
        print_subsection("Reasoning Trace (first lines)", "🧠")
        lines = response.thinking_blocks[0].strip().splitlines()
        for line in lines[:4]:
            print(f"     {line}", flush=True)
        if len(lines) > 4:
            print(f"     ... ({len(lines) - 4} more lines)", flush=True)
    else:
        print("❌ No reasoning trace despite strip_thinking_response=False", flush=True)
        ok = False

    print_subsection("Comprehensive Business Analysis", "📊")
    ok = print_answer(response, REASONING_CAP) and ok
    count = print_citations(response, "Market Research Sources")
    ok = (
        check(
            count > 0,
            f"Web search ran: {count} citations returned",
            "No citations returned, so no web search happened",
        )
        and ok
    )

    if response.venice_parameters:
        applied = response.venice_parameters
        print_subsection("Response Metadata", "📋")
        print(f"   Web search used: {applied.enable_web_search}", flush=True)
        print(f"   Citations enabled: {applied.enable_web_citations}", flush=True)
        print(f"   Venice prompts: {applied.include_venice_system_prompt}", flush=True)

    print_usage(response)
    return ok


# =============================================================================
# Main Function
# =============================================================================


async def main() -> int:
    """Run all Venice Parameters examples; return a process exit code."""
    print("🚀 Venice AI Parameters Showcase", flush=True)
    print("=" * 70, flush=True)
    print("Demonstrating Venice-specific parameters with real API data\n", flush=True)

    async with VeniceClient() as client:
        results: list[tuple[str, bool | None]] = [
            ("character_example", await character_example(client)),
            ("thinking_control_example", await thinking_control_example(client)),
            ("web_search_example", await web_search_example(client)),
            ("system_prompt_control_example", await system_prompt_control_example(client)),
            ("comprehensive_example", await comprehensive_example(client)),
        ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print_section_header(f"{len(failed)} of {len(results)} demos failed", "❌")
        print(f"   Failed: {', '.join(failed)}", flush=True)
        return 1
    if len(skipped) == len(results):
        print("SKIPPED: the catalog has no model or character for any section", flush=True)
        return 77
    if skipped:
        print(f"\nℹ️ {len(skipped)} section(s) skipped: {', '.join(skipped)}", flush=True)

    print_section_header("Examples Completed Successfully! ✨", "🎉")
    print("\n💡 Key Venice Parameters demonstrated:", flush=True)
    print("   • character_slug: Leverage pre-built AI personalities", flush=True)
    print("   • strip_thinking_response: Control reasoning visibility", flush=True)
    print("   • disable_thinking: Toggle reasoning capabilities", flush=True)
    print("   • enable_web_search: Access current information", flush=True)
    print("   • enable_web_citations: Get source attribution", flush=True)
    print("   • return_search_results_as_documents: OpenAI-compatible tools", flush=True)
    print("   • include_venice_system_prompt: Fine-tune system behavior", flush=True)
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
