#!/usr/bin/env python3
"""
Venice AI SDK - Web Scraping with Chat Completions
===================================================

This example demonstrates the ``enable_web_scraping`` Venice parameter, which
automatically detects URLs in user messages, fetches their content, and augments
the model's context so it can answer questions about those pages.

**Web Scraping vs Web Search**

+-------------------------+------------------------------------------------+
| Feature                 | Behaviour                                      |
+=========================+================================================+
| ``enable_web_scraping`` | Scrapes the **specific URLs** you include in   |
|                         | your message. The model sees the page content. |
+-------------------------+------------------------------------------------+
| ``enable_web_search``   | Searches the **open web** for relevant results |
|                         | related to your prompt (no URL required).      |
+-------------------------+------------------------------------------------+

A failed scrape is not an HTTP error: the model simply answers from what it
already knows. So every section checks that the scraped text really reached
the model, by comparing the prompt-token count against a request with scraping
turned off, and checks that the answer is complete (``finish_reason``, and a
completion count below the cap, since some models report a cut-off answer as
``"stop"``). Every request leaves Venice's own system prompt out, so the
prompt-token growth is the page text alone.

Sections:
    1. Basic URL scraping and targeted extraction (with a scraping-off baseline)
    2. Multiple URLs (compare / contrast)
    3. Web search, for contrast: no URL, results come back as citations
    4. Streaming with web scraping
"""

import asyncio
import sys
import textwrap
from typing import Any

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import UserMessage
from venice_ai.types.api.requests import VeniceParameters

# Generous caps so answers finish; each section also asks for a bounded answer.
MAX_COMPLETION_TOKENS = 1500

# A scraped page adds at least this many prompt tokens over the baseline.
MIN_SCRAPED_TOKENS = 500

RATE_LIMITS_DOC = "https://docs.venice.ai/api-reference/rate-limiting"
RFC_HTCPCP = "https://www.rfc-editor.org/rfc/rfc2324.txt"
RFC_HTCPCP_TEA = "https://www.rfc-editor.org/rfc/rfc7168.txt"

# The two RFCs add roughly 5k and 4k tokens; more than 7k proves both arrived.
MIN_BOTH_RFC_TOKENS = 7000


def _print_block(label: str, text: str) -> None:
    print(f"\n{label}")
    print(textwrap.indent(text.strip() or "(empty)", "   "))


def _finish_reason(response: Any) -> str | None:
    return response.choices[0].finish_reason if response.choices else None


def _prompt_tokens(response: Any) -> int:
    usage = getattr(response, "usage", None)
    return usage.prompt_tokens if usage is not None else 0


def _print_usage(response: Any, *, indent: str = "") -> None:
    """Print finish reason and token usage from a response."""
    print(f"{indent}🏁 Finish reason: {_finish_reason(response)}")
    usage = getattr(response, "usage", None)
    if usage is None:
        print(f"{indent}📊 Token usage: not provided by the API")
        return
    print(
        f"{indent}📊 Token Usage: Input={usage.prompt_tokens}, "
        f"Output={usage.completion_tokens}, Total={usage.total_tokens}"
    )


def _check(response: Any, baseline: int | None, label: str, *, scraped: bool = True) -> bool:
    """Return ``False`` (and say why) if the answer is truncated or the scrape didn't land."""
    ok = True
    usage = getattr(response, "usage", None)
    used = usage.completion_tokens if usage is not None else None
    if _finish_reason(response) == "length" or used is None or used >= MAX_COMPLETION_TOKENS:
        print(f"❌ {label}: the answer may be cut off ({used} of {MAX_COMPLETION_TOKENS} tokens)")
        ok = False
    if not (response.text or "").strip():
        print(f"❌ {label}: the model returned no text")
        ok = False
    if scraped and baseline is not None:
        added = _prompt_tokens(response) - baseline
        if added < MIN_SCRAPED_TOKENS:
            print(f"❌ {label}: only {added} tokens over the baseline, so the page was not scraped")
            ok = False
        else:
            print(f"✅ {label}: scraped content added ~{added} prompt tokens")
    return ok


# =============================================================================
# 1. Basic URL Scraping and Targeted Extraction
# =============================================================================


async def basic_url_scraping(client: VeniceClient, model: str) -> tuple[bool, int | None]:
    """Summarise a page and pull specific facts from it, against a scraping-off baseline.

    Returns ``(ok, baseline_prompt_tokens)``; the baseline is reused by the
    other sections to confirm their pages were scraped.
    """
    print("🌐 Basic URL Scraping and Extraction")
    print("-" * 40)

    prompt = (
        "From this page: (1) summarize it in 3-4 bullet points, then (2) list the exact "
        "names of the rate-limit response headers and (3) the HTTP status code returned "
        f"when a rate limit is exceeded.\n\n{RATE_LIMITS_DOC}"
    )

    try:
        # Same prompt without scraping: the model only sees the URL text. Only
        # its prompt-token count is used, so the reply is capped very low.
        without = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=prompt)],
            venice_parameters=VeniceParameters(
                enable_web_scraping=False, include_venice_system_prompt=False
            ),
            max_completion_tokens=16,
        )
        baseline = _prompt_tokens(without)
        print(f"📏 Baseline without scraping: {baseline} prompt tokens")

        response = await client.chat.completions.create(
            model=model,
            messages=[UserMessage(content=prompt)],
            # Only enable_web_scraping differs from the baseline request
            venice_parameters=VeniceParameters(
                enable_web_scraping=True, include_venice_system_prompt=False
            ),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.2,
        )
    except VeniceError as e:
        print(f"❌ Error in basic scraping example ({type(e).__name__}): {e}")
        return False, None

    content = response.text or ""
    _print_block("📄 Summary and extracted data:", content)
    print()
    _print_usage(response)
    ok = _check(response, baseline, "basic_url_scraping")

    # The prompt-token growth checked above is what proves the page was
    # scraped. These keyword checks only confirm the extraction answered the
    # question: x-ratelimit-remaining-requests is also OpenAI's header name, so
    # a model could produce it from memory.
    if "remaining-requests" not in content.lower():
        print("❌ basic_url_scraping: the page's x-ratelimit-*-requests headers are missing")
        ok = False
    if "429" not in content:
        print("❌ basic_url_scraping: the 429 status code is missing")
        ok = False
    return ok, baseline


# =============================================================================
# 2. Multiple URLs — Compare / Contrast
# =============================================================================


async def multiple_urls(client: VeniceClient, model: str, baseline: int | None) -> bool:
    """Scrape two URLs in a single message and compare them.

    The scraper caps how much page text it adds per request, so two very long
    pages (large Wikipedia articles, for example) can crowd each other out.
    This section compares two short, permanently published RFCs so both fit,
    and checks that the prompt grew by more than either page alone would add.
    """
    print("\n🔗 Multiple URL Scraping")
    print("-" * 40)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                UserMessage(
                    content=(
                        "The second document extends the first. List the 4 most important "
                        "things the second adds or changes, one short sentence each:\n"
                        f"1. {RFC_HTCPCP}\n"
                        f"2. {RFC_HTCPCP_TEA}"
                    ),
                )
            ],
            venice_parameters=VeniceParameters(
                enable_web_scraping=True, include_venice_system_prompt=False
            ),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.4,
        )
    except VeniceError as e:
        print(f"❌ Error in multiple URLs example ({type(e).__name__}): {e}")
        return False

    _print_block("🔍 Comparison:", response.text or "")
    print()
    _print_usage(response)
    ok = _check(response, baseline, "multiple_urls")
    if baseline is not None and _prompt_tokens(response) - baseline < MIN_BOTH_RFC_TOKENS:
        print("❌ multiple_urls: the prompt is too small to contain both documents")
        ok = False
    return ok


# =============================================================================
# 3. Web Search, for Contrast
# =============================================================================


async def scraping_vs_search(client: VeniceClient, model: str) -> bool:
    """Contrast enable_web_scraping (sections 1 and 2) with enable_web_search.

    Scraping reads the URLs you put in the message. Web search needs no URL:
    it finds pages on its own and returns them as citations.
    """
    print("\n⚖️  Web Scraping vs Web Search")
    print("-" * 40)
    print("   🌐 enable_web_scraping=True  → fetches the specific URL you provide (above)")
    print("   🔍 enable_web_search='on'    → searches the web and cites its sources")

    try:
        print("\n🔍 With enable_web_search='on' (searches the web, no URL given):")
        response_search = await client.chat.completions.create(
            model=model,
            messages=[
                UserMessage(
                    content="In 3 sentences: what kinds of models does the Venice AI API offer?"
                )
            ],
            venice_parameters=VeniceParameters(
                enable_web_search="on",
                enable_web_citations=True,
                include_venice_system_prompt=False,
            ),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.3,
        )
    except VeniceError as e:
        print(f"   ❌ Error with web search ({type(e).__name__}): {e}")
        return False

    print(textwrap.indent((response_search.text or "").strip(), "   "))
    _print_usage(response_search, indent="   ")

    # Web search returns the pages it found as citations; scraping does not.
    citations = response_search.web_search_citations
    print(f"   🔗 {len(citations)} citation(s) returned by web search:")
    for citation in citations[:5]:
        print(f"      • {citation.title} — {citation.url}")
    ok = _check(response_search, None, "web search", scraped=False)
    if not citations:
        print("   ❌ web search: no citations, so the web search did not run")
        ok = False

    print("\n💡 Key takeaway:")
    print("   • Web scraping reads the exact page you link to")
    print("   • Web search finds relevant pages on its own and cites them")
    return ok


# =============================================================================
# 4. Streaming with Web Scraping
# =============================================================================


async def streaming_with_scraping(client: VeniceClient, model: str, baseline: int | None) -> bool:
    """Demonstrate that web scraping works with streaming responses."""
    print("\n🌊 Streaming with Web Scraping")
    print("-" * 40)

    try:
        print("\n🤖 Assistant (streaming): ", end="", flush=True)
        stream = await client.chat.completions.stream(
            model=model,
            messages=[
                UserMessage(
                    content=(
                        "Give me a brief overview of this document in 3 bullet points: "
                        f"{RFC_HTCPCP}"
                    ),
                )
            ],
            venice_parameters=VeniceParameters(
                enable_web_scraping=True, include_venice_system_prompt=False
            ),
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.3,
        )
        # collect_with_deltas() prints tokens as they arrive and assembles the
        # final response (finish reason and usage) once the stream ends.
        chunk_count = 0
        async with stream:
            async for token in stream.collect_with_deltas():
                chunk_count += 1
                print(token, end="", flush=True)
        print()
    except VeniceError as e:
        print(f"\n❌ Error in streaming scraping example ({type(e).__name__}): {e}")
        return False

    final = stream.final_response
    if final is None:
        print("❌ The stream ended without a final response")
        return False
    print(f"\n📊 Streaming Stats: {chunk_count} text deltas, {len(final.text or '')} chars")
    _print_usage(final)
    return _check(final, baseline, "streaming_with_scraping")


# =============================================================================
# Main
# =============================================================================


async def main() -> int:
    """Run all web scraping examples: exit 1 on any failure, 77 if no web model."""
    print("🚀 Venice AI Web Scraping Examples")
    print("=" * 50)

    async with VeniceClient() as client:
        # Web scraping and web search share the model's web-search capability.
        # exclude_reasoning=True keeps the pick to models that answer directly,
        # so the whole token budget goes to the summaries.
        try:
            model = await client.models.resolve_chat(
                require_web_search=True, exclude_reasoning=True, prefer="cheapest"
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no non-reasoning model supports web search ({e})")
            return 77
        print(f"📍 Using model: {model}\n")

        ok_basic, baseline = await basic_url_scraping(client, model)
        results = [
            ("basic_url_scraping", ok_basic),
            ("multiple_urls", await multiple_urls(client, model, baseline)),
            ("scraping_vs_search", await scraping_vs_search(client, model)),
            ("streaming_with_scraping", await streaming_with_scraping(client, model, baseline)),
        ]

    failed = [name for name, ok in results if not ok]
    marker = "❌" if failed else "✨"
    print(
        f"\n{marker} Web scraping examples: {len(results) - len(failed)}/{len(results)} completed"
    )
    if failed:
        print(f"   Failed: {', '.join(failed)} — see the ❌ messages above.")
        return 1

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic URL scraping and targeted extraction with enable_web_scraping=True")
    print("   - Confirming a scrape landed via prompt-token growth")
    print("   - Multi-URL scraping for comparison tasks")
    print("   - Difference between web scraping and web search (citations)")
    print("   - Streaming responses with web scraping enabled")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
