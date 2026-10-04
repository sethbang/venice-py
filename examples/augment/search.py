#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Web Search
===================================

Demonstrates ``client.augment.search(...)`` — Venice's structured web-search
endpoint. Returns a list of results (title, URL, content snippet, date)
from either Brave (default, Zero Data Retention) or Google (anonymised
proxy). The ``date`` field is often empty, and snippets can contain HTML
highlighting such as ``<strong>``, so strip tags before showing or reusing them.

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.

Exit status: ``0`` when the searches returned results and every grounded
bullet's quote was found in its cited source, ``1`` otherwise. Search is the
core feature: if the catalog has no chat model that answers directly, the
grounded-chat section prints ``Section skipped:`` and the script still exits
``0``.
"""

import asyncio
import html
import re
import sys

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import SystemMessage, UserMessage
from venice_ai.types.api.augment import AugmentSearchResponse
from venice_ai.types.api.requests import VeniceParameters

_TAG = re.compile(r"<[^>]+>")


def clean(text: str) -> str:
    """Strip HTML tags and entities from a search snippet and collapse whitespace."""
    return " ".join(html.unescape(_TAG.sub("", text)).split())


# ---------------------------------------------------------------------------
# 1. Basic search (Brave default), then the same query through Google
# ---------------------------------------------------------------------------

COMPARE_QUERY = "EIP-4361 Sign-In-With-Ethereum"


def print_results(response: AugmentSearchResponse) -> None:
    """Print each result's title, URL, date and a short snippet."""
    for idx, result in enumerate(response.results, start=1):
        print(f"\n{idx}. {clean(result.title)}")
        print(f"   🔗 {result.url}")
        if result.date:
            print(f"   📅 {result.date}")
        print(f"   📝 {clean(result.content)[:150]}")


async def search_and_compare_providers(client: VeniceClient) -> bool:
    """Search once with the default provider, then repeat the query on Google.

    Leaving out ``search_provider`` uses Brave. The second call names
    ``search_provider="google"`` explicitly so the two result sets can be compared.
    """
    print("🔎 Basic Search (Brave, default)")
    print("-" * 30)

    try:
        brave = await client.augment.search(query=COMPARE_QUERY, limit=3)
    except VeniceError as e:
        print(f"❌ Search failed: {type(e).__name__}: {e}")
        return False

    print(f"📍 Query: {brave.query}")
    print(f"🔢 Results: {len(brave.results)}")
    print_results(brave)
    if not brave.results:
        print("❌ The search returned no results")
        return False

    print("\n🆚 Same query through Google")
    print("-" * 30)
    try:
        google = await client.augment.search(
            query=COMPARE_QUERY,
            limit=3,
            search_provider="google",
        )
    except VeniceError as e:
        print(f"❌ google failed: {type(e).__name__}: {e}")
        return False

    print(f"🔢 Results: {len(google.results)}")
    print_results(google)
    if not google.results:
        print("❌ google returned no results")
        return False

    shared = {r.url for r in brave.results} & {r.url for r in google.results}
    print(f"\n🔗 URLs returned by both providers: {len(shared)} of {len(brave.results)}")
    return True


# ---------------------------------------------------------------------------
# 2. Use search results to ground a chat completion (RAG-lite)
# ---------------------------------------------------------------------------

#: Room for a few short bullets from a model that answers directly.
GROUNDED_MAX_TOKENS = 512

_QUOTED = re.compile(r'["\u201c]([^"\u201d]+)["\u201d]')
_CITATION = re.compile(r"\[(\d+)\]")


def normalize(text: str) -> str:
    """Lower-case, keep only letters, digits and single spaces, for quote matching."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def check_bullet(bullet: str, source_texts: list[str]) -> str | None:
    """Return why a bullet's citation is not anchored, or ``None`` if it is.

    A citation is anchored when the bullet cites exactly one source that
    exists, and the phrase it quotes appears word for word (ignoring case and
    punctuation) in the text of that source. Nothing here reads the claim
    itself.
    """
    cited = [int(n) for n in _CITATION.findall(bullet)]
    quotes = [q for q in _QUOTED.findall(bullet) if len(normalize(q).split()) >= 3]
    if len(set(cited)) != 1:
        return f"cites {sorted(set(cited)) or 'no source'}, expected exactly one"
    n = cited[0]
    if not 1 <= n <= len(source_texts):
        return f"cites [{n}], which does not exist"
    if not quotes:
        return "quotes no supporting phrase of three or more words"
    source = normalize(source_texts[n - 1])
    missing = [q for q in quotes if normalize(q) not in source]
    if missing:
        return f"quotes {missing[0]!r}, which is not in source [{n}]"
    return None


async def search_grounded_chat(client: VeniceClient) -> bool | None:
    """Ground a chat completion in search results, with checkable citations.

    Each snippet is numbered and carries its URL. The model must cite one
    source per bullet and copy a short supporting phrase from it, so every
    citation can be checked against the text the model was actually given.

    The check proves one thing: each quoted phrase appears, word for word, in
    the source the bullet cites, so no citation points at invented text. It
    does not check that the quote supports the claim, or that the claim is
    correct or current; search snippets can be outdated, and a model can quote
    accurately and still overstate. Judge each claim against its quote.

    Returns ``None`` (section skipped) when the catalog has no chat model that
    answers directly.
    """
    print("\n🧠 Search-Grounded Chat Completion")
    print("-" * 30)

    question = "What are the current best practices for Python type hinting?"

    # Step 1 — search for recent context.
    search = await client.augment.search(query=question, limit=4)
    if not search.results:
        print("❌ The search returned no results to ground the answer in")
        return False

    # Step 2 — build a numbered source list for the prompt. Keep the exact
    # text sent to the model, so the citations can be checked against it.
    source_texts = [f"{clean(r.title)}: {clean(r.content)[:300]}" for r in search.results]
    sources = [
        f"[{n}] ({r.url}) {text}"
        for n, (r, text) in enumerate(zip(search.results, source_texts, strict=True), start=1)
    ]

    # Step 3 — ask a chat model that answers directly (no reasoning phase).
    try:
        chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    except NoMatchingModelError as e:
        print(f"Section skipped: no chat model in the catalog answers directly ({e})")
        return None
    print(f"📍 Chat model: {chat_model}")

    response = await client.chat.completions.create(
        model=chat_model,
        messages=[
            SystemMessage(
                content=(
                    "Answer the user's question using only the numbered sources. "
                    "Write 2 to 4 bullet points. Each bullet makes one claim, cites "
                    "exactly one source like [2], and then copies a supporting phrase "
                    "of 3 to 12 words, word for word, from that source in double "
                    "quotes. Format example (not about this topic): - Water boils "
                    'sooner at altitude. [3] "boils at a lower temperature at high '
                    'altitude"'
                ),
            ),
            UserMessage(
                content=f"Question: {question}\n\nSources:\n" + "\n".join(sources),
            ),
        ],
        temperature=0,
        # The system message above carries all the instructions the answer needs.
        venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        max_completion_tokens=GROUNDED_MAX_TOKENS,
    )

    answer = response.text or ""
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else 0
    print("\n💬 Grounded answer:")
    print(answer)
    print(f"\n🏁 Finish reason: {finish_reason} ({used} completion tokens)")

    print("\n📚 Sources:")
    for n, result in enumerate(search.results, start=1):
        print(f"   [{n}] {clean(result.title)}")
        print(f"       {result.url}")

    # Some backends report "stop" even when the reply ran into the cap.
    if finish_reason == "length" or used >= GROUNDED_MAX_TOKENS:
        print("❌ The answer was cut off by max_completion_tokens")
        return False

    bullets = [
        line.strip() for line in answer.splitlines() if re.match(r"\s*(?:[-*\u2022]|\d+\.)\s", line)
    ]
    if not bullets:
        print("❌ The answer has no bullet points to check")
        return False

    print("\n🔎 Citation anchoring (is each quoted phrase in the source it cites?)")
    print("   This does not judge the claims; read each one against its quote.")
    problems = 0
    for bullet in bullets:
        problem = check_bullet(bullet, source_texts)
        print(f"\n   {bullet}")
        if problem is None:
            cited = _CITATION.findall(bullet)[0]
            print(f"     ↳ quote found in source [{cited}]")
        else:
            print(f"     ↳ ❌ {problem}")
            problems += 1

    if problems:
        print(f"\n❌ {problems} of {len(bullets)} citations are not anchored in their source")
        return False
    print(f"\n✅ All {len(bullets)} quotes appear word for word in the source they cite")
    print("   (no citation points at invented text; the claims themselves are unchecked)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment-search examples.

    Returns ``1`` if any demo failed, otherwise ``0`` (a skipped grounded-chat
    section still leaves the search itself verified).
    """
    print("🚀 Venice AI Augment — Web Search Examples")
    print("=" * 50)

    sections = [
        ("search_and_compare_providers", search_and_compare_providers),
        ("search_grounded_chat", search_grounded_chat),
    ]

    # Each section returns True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = []
    async with VeniceClient() as client:
        for name, section in sections:
            try:
                results.append((name, await section(client)))
            except VeniceError as e:
                print(f"❌ {name} failed: {type(e).__name__}: {e}")
                results.append((name, False))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    print("\n" + "=" * 50)
    if failed:
        print(f"❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    if skipped:
        print(f"✅ Search verified; skipped: {', '.join(skipped)}")
        return 0
    print("✨ Search examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Structured search results (title, url, content, date)")
    print("   - Provider selection (brave default, google proxied)")
    print("   - Grounding a chat answer in numbered sources, with each quote checked")
    print("     against the source it cites (anchoring, not fact-checking)")
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
