#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Web Scrape
===================================

Demonstrates ``client.augment.scrape(url=...)`` — Venice's web-scraping
endpoint that returns page content as markdown. First tries Cloudflare's
native markdown extraction, then falls back to a headless browser. Sites
that block automated access, such as X/Twitter, are rejected immediately with
a ``400`` error.

Two pages show the range: a short page comes back whole, and a long
reference page comes back with its navigation and section structure as
markdown headings. Each is checked by text from its body, not by length.

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import InvalidRequestError, VeniceError


def markdown_headings(content: str) -> list[str]:
    """The markdown heading lines in scraped content."""
    return [line.strip() for line in content.splitlines() if line.lstrip().startswith("#")]


# ---------------------------------------------------------------------------
# 1. Scrape a small page
# ---------------------------------------------------------------------------

SMALL_PAGE_URL = "https://www.iana.org/help/example-domains"

#: Size of each scraped page, (characters, headings), for the closing contrast.
SCRAPED: dict[str, tuple[int, int]] = {}


async def scrape_small_page() -> bool:
    """Scrape a short page and check its title and body text."""
    print("📄 Scrape a Small Page")
    print("-" * 30)

    async with VeniceClient() as client:
        try:
            result = await client.augment.scrape(url=SMALL_PAGE_URL)
        except VeniceError as e:
            print(f"❌ Scrape failed: {type(e).__name__}: {e}")
            return False

    headings = markdown_headings(result.content)
    print(f"📍 URL: {result.url}")
    print(f"📄 Format: {result.format}")
    print(f"📏 Returned {len(result.content)} chars of markdown, {len(headings)} headings")
    SCRAPED["short page"] = (len(result.content), len(headings))
    print("\n📝 Content:")
    print("-" * 30)
    print(result.content[:600])
    print("-" * 30)

    # The page's title and a phrase from its first paragraph. Whitespace is
    # collapsed first, since a scraper may keep the page's line wrapping.
    flat = " ".join(result.content.split())
    expected = ("# Example Domains", "maintained for documentation purposes")
    missing = [text for text in expected if text not in flat]
    if missing:
        print(f"❌ Expected page text is missing: {', '.join(missing)}")
        return False
    return True


# ---------------------------------------------------------------------------
# 2. Scrape a docs page
# ---------------------------------------------------------------------------


async def scrape_documentation_page() -> bool:
    """Scrape a long documentation page and check its content came through."""
    print("\n📚 Scrape a Docs Page")
    print("-" * 30)

    docs_url = "https://docs.python.org/3/library/asyncio.html"

    async with VeniceClient() as client:
        try:
            result = await client.augment.scrape(url=docs_url)
        except VeniceError as e:
            print(f"❌ Scrape failed: {type(e).__name__}: {e}")
            return False

        print(f"📍 URL: {result.url}")
        print(f"📄 Format: {result.format}")
        print(f"📏 Returned {len(result.content)} chars of markdown")

        # Markdown headings show the page structure came through. They include
        # the site's navigation, so the body-text check below is the real test.
        headings = markdown_headings(result.content)
        print(f"📑 Markdown headings (page and navigation): {len(headings)}")
        SCRAPED["docs page"] = (len(result.content), len(headings))
        for heading in headings[:5]:
            print(f"   • {heading}")

        # Show the first paragraph under the page title, not just headings, so
        # thin or placeholder pages are visible.
        lines = [line.strip() for line in result.content.splitlines()]
        title_idx = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
        body = ""
        if title_idx is not None:
            start = next(
                (i for i in range(title_idx + 1, len(lines)) if lines[i][:1].isalpha()), None
            )
            if start is not None:
                end = next((i for i in range(start, len(lines)) if not lines[i]), len(lines))
                body = " ".join(lines[start:end])
        if body:
            print(f"\n📝 First paragraph: {body[:200]}")

        if "async/await" not in result.content or not body:
            print("❌ The page body is missing (expected the asyncio introduction)")
            return False
        return True


# ---------------------------------------------------------------------------
# 3. Error handling for a blocked site
# ---------------------------------------------------------------------------


async def error_handling() -> bool:
    """Show how the scraper rejects a site that blocks automated access.

    Venice rejects such domains (here X/Twitter) with HTTP 400, which the SDK raises as
    :class:`InvalidRequestError`. That rejection is the expected outcome here;
    a successful scrape or any other error (auth, 5xx, network) fails the
    section.
    """
    print("\n⚠️  Error Handling")
    print("-" * 30)

    url = "https://x.com/"

    async with VeniceClient() as client:
        try:
            result = await client.augment.scrape(url=url)
        except InvalidRequestError as e:
            print(f"🚫 {url} → {type(e).__name__} (expected): {e}")
            return True
    print(f"❌ {url} → scraped {len(result.content)} chars; expected a rejection")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment-scrape examples and report an honest tally."""
    print("🚀 Venice AI Augment — Web Scrape Examples")
    print("=" * 50)

    sections = {
        "Small page scrape": scrape_small_page,
        "Docs page scrape": scrape_documentation_page,
        "Error handling demo": error_handling,
    }

    results: dict[str, bool] = {}
    for name, section in sections.items():
        try:
            results[name] = await section()
        except VeniceError as e:
            print(f"❌ {name} failed: {type(e).__name__}: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    if all(results.values()):
        print("✨ Scrape examples completed!")
        print("\n📊 Same endpoint, two page sizes:")
        for label, (chars, headings) in SCRAPED.items():
            print(f"   {label:10s} {chars:>8,} chars  {headings:>4} headings")
        print("\n💡 Key concepts demonstrated:")
        print("   - URL → markdown for a short page and a long docs page,")
        print("     each checked by text from its body")
        print("   - A blocked domain raises InvalidRequestError")
        return 0

    failed = [name for name, ok in results.items() if not ok]
    print(f"❌ {len(failed)} of {len(results)} examples failed: {', '.join(failed)}")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
