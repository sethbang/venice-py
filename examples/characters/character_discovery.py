#!/usr/bin/env python3
"""
Venice AI SDK - Character Discovery and Integration
===================================================

This example demonstrates how to discover and work with AI characters in the Venice AI SDK:
- Walking the whole character catalog with ``characters.iter_all()`` (``list()``
  returns a single page)
- Analyzing tags and popularity statistics across the catalog
- Filtering server-side with ``search``, ``tags`` and ``sort_by``
- Choosing the chat model to pair with a character

Characters are user-created, and most carry no content tags at all, so the
rankings and the integration walkthrough draw from the featured, web-enabled
set rather than from the raw catalog.

All calls here are read-only catalog lookups — running this example costs nothing.

Note on content: the server-side ``adult`` flag is not reliable on its own —
characters tagged "Adult" can still report ``adult=False``. This example
therefore also leaves out characters whose tags mark them as adult or NSFW.
"""

import asyncio
import re
import sys
from collections import Counter

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types import Character

#: Tags (after normalization) that mark a character as adult content.
ADULT_TAGS = {"adult", "nsfw", "18"}

#: Minimum number of ratings before an average rating is worth ranking on.
MIN_RATINGS = 3

#: Use cases mapped to the normalized tags that identify them.
USE_CASE_TAGS = {
    "Education & Tutoring": {"education", "teacher", "tutor", "learning"},
    "Programming & Tech": {"programming", "coding", "linux", "developer"},
    "Storytelling": {"storyteller", "interactivestorytelling", "storytelling"},
    "Philosophy & Psychology": {"philosophy", "psychology"},
    "Comedy": {"comedy", "funny", "humor"},
}


def normalize_tag(tag: str) -> str:
    """Case-fold and drop punctuation so 'Role-play', 'roleplay' and 'Roleplay' match."""
    return re.sub(r"[^a-z0-9]", "", tag.casefold())


def is_adult(character: Character) -> bool:
    """Adult by the server flag or by an adult/NSFW tag."""
    return character.adult or any(normalize_tag(t) in ADULT_TAGS for t in character.tags)


def _short(text: str | None, limit: int) -> str:
    text = " ".join((text or "No description available").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def showcase_set(characters: list[Character]) -> list[Character]:
    """The featured, web-enabled characters, most imported first."""
    featured = [c for c in characters if c.featured and c.webEnabled]
    return sorted(featured, key=lambda c: c.stats.imports, reverse=True)


async def discover_characters(client: VeniceClient) -> list[Character] | None:
    """Walk the whole catalog. Returns the non-adult characters, or None on failure."""
    print("🎭 Character Discovery")
    print("-" * 40)

    try:
        everything = [c async for c in client.characters.iter_all(is_adult=False)]
    except VeniceError as e:
        print(f"❌ Error discovering characters: {e}")
        return None

    if not everything:
        print("❌ The character catalog came back empty.")
        return None

    characters = [c for c in everything if not is_adult(c)]
    print(f"🎪 {len(everything)} characters returned by iter_all(is_adult=False)")
    print(f"   {len(everything) - len(characters)} more are left out because their tags mark them")
    print(f"   as adult, so the analysis below covers {len(characters)} characters.")

    web = sum(1 for c in characters if c.webEnabled)
    featured = showcase_set(characters)
    print(f"   {web} are web-enabled; {sum(1 for c in characters if c.featured)} are featured.")
    print("   Characters are user-created; descriptions appear as the catalog returns them.")

    print("\n📋 Featured, web-enabled characters:")
    for i, character in enumerate(featured[:6], 1):
        print(f"\n   {i}. {character.name}  ({character.stats.imports:,} imports)")
        print(f"      🔗 Slug: {character.slug}")
        print(f"      📝 {_short(character.description, 100)}")
        if character.tags:
            print(f"      🏷️ Tags: {', '.join(character.tags[:5])}")
    if not featured:
        print("   ℹ️ No featured, web-enabled characters right now.")

    return characters


def analyze_tags(characters: list[Character]) -> None:
    """Count tags across the catalog, merging spelling variants."""
    print("\n📊 Tag Analysis")
    print("-" * 40)

    counts: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    for character in characters:
        for tag in {normalize_tag(t): t for t in character.tags}.items():
            key, original = tag
            if not key:
                continue
            counts[key] += 1
            spellings.setdefault(key, Counter())[original] += 1

    print(f"🏷️ {len(counts)} distinct tags after merging case and punctuation variants")
    print("   Most common:")
    for key, count in counts.most_common(12):
        variants = ", ".join(v for v, _ in spellings[key].most_common(3))
        print(f"   📌 {count:4d}  {key}  (written as: {variants})")


def analyze_statistics(showcase: list[Character]) -> bool:
    """Rank the showcase set by imports and by rating, with a minimum rating count."""
    print("\n📈 Character Statistics (featured, web-enabled characters)")
    print("-" * 40)

    if not showcase:
        print("❌ No featured, web-enabled characters to rank.")
        return False

    print("🏆 Top by imports:")
    for i, c in enumerate(showcase[:5], 1):
        print(f"   {i}. {c.name}: {c.stats.imports:,}")

    rated = [
        c
        for c in showcase
        if c.stats.averageRating is not None and (c.stats.ratingCount or 0) >= MIN_RATINGS
    ]
    print(f"\n⭐ Top by average rating (at least {MIN_RATINGS} ratings; {len(rated)} qualify):")
    rated.sort(key=lambda c: (c.stats.averageRating or 0.0, c.stats.ratingCount or 0), reverse=True)
    for i, c in enumerate(rated[:5], 1):
        print(f"   {i}. {c.name}: {c.stats.averageRating:.2f} from {c.stats.ratingCount} ratings")
    if not rated:
        print("   ℹ️ No character has enough ratings to rank yet.")
    return True


def character_selection_guide(characters: list[Character]) -> None:
    """Recommend characters per use case by exact (normalized) tag match."""
    print("\n🎯 Character Selection Guide (by tag)")
    print("-" * 40)

    for use_case, wanted in USE_CASE_TAGS.items():
        matches = [c for c in characters if wanted & {normalize_tag(t) for t in c.tags}]
        matches.sort(key=lambda c: c.stats.imports, reverse=True)
        print(f"\n🎯 {use_case} ({len(matches)} tagged):")
        for c in matches[:3]:
            print(f"   • {c.name}  [{', '.join(c.tags[:4])}]")
        if not matches:
            print("   ℹ️ No characters carry these tags today.")


async def server_side_filtering(client: VeniceClient) -> bool:
    """Let the server filter and sort instead of downloading everything."""
    print("\n🔎 Server-side Filtering")
    print("-" * 40)

    queries = [
        (
            "search='teacher', sort_by='highlyRated'",
            {"search": "teacher", "sort_by": "highlyRated"},
        ),
        ("tags=['Philosophy'], sort_by='imports'", {"tags": ["Philosophy"], "sort_by": "imports"}),
    ]
    for label, kwargs in queries:
        try:
            page = await client.characters.list(limit=10, is_adult=False, **kwargs)  # type: ignore[arg-type]
        except VeniceError as e:
            print(f"❌ {label}: {e}")
            return False
        shown = [c for c in page.data if not is_adult(c)][:3]
        print(f"\n   characters.list({label}) → {len(page.data)} results (server order):")
        for c in shown:
            print(f"   • {c.name}  [{', '.join(c.tags[:4])}]")
    return True


async def demonstrate_character_integration(
    client: VeniceClient, characters: list[Character], showcase: list[Character]
) -> bool | None:
    """Pair a character with a chat model that the API actually serves.

    Returns ``None`` (section skipped) when the catalog has no chat model to
    fall back on.
    """
    print("\n🔗 Character Integration Guide")
    print("-" * 40)

    try:
        listing = await client.models.list(type="text")
        # Listed is not enough: a deprecated model is routed to its replacement
        # and an offline one is not served, so only routable models count.
        text_models = {
            m.id
            for m in listing.data
            if not m.model_spec.offline and m.model_spec.deprecation is None
        }
        fallback = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    except NoMatchingModelError as e:
        print(f"Section skipped: no chat model in the catalog to fall back on ({e})")
        return None
    except VeniceError as e:
        print(f"❌ Error loading the text model catalog: {e}")
        return False

    off_catalog = sum(1 for c in characters if c.modelId not in text_models)
    print(f"📊 {off_catalog} of {len(characters)} characters name a modelId that is not a")
    print("   routable model in models.list(type='text') (unlisted, deprecated or offline).")
    print("   The server remaps some of those IDs and rejects others, and the remaps are")
    print("   not published, so validate against the catalog before passing a")
    print("   character's modelId as model=.")

    top = showcase[:3]
    if not top:
        print("❌ No featured, web-enabled character to walk through.")
        return False
    print("\n   Worked examples (featured, web-enabled characters):")
    for character in top:
        usable = character.modelId in text_models
        model = character.modelId if usable else fallback
        source = (
            "character's own model, if the request fits your budget"
            if usable
            else "not routable; cheapest direct-answer model instead"
        )
        print(f"\n   {character.name} ({character.slug})")
        print(f"   modelId {character.modelId} → use {model} ({source})")

    print("\n   Integration pattern (characters/character_details.py runs it live):")
    print("   ```python")
    print("   MAX_REQUEST_USD = Decimal('0.01')  # your spending policy for one request")
    print("   listing = await client.models.list(type='text')")
    print("   routable = {m.id for m in listing.data")
    print("               if not m.model_spec.offline and m.model_spec.deprecation is None}")
    print("   messages = [UserMessage(content='Hello! Can you help me?')]")
    print("   model = await client.models.resolve_chat(prefer='cheapest', exclude_reasoning=True)")
    print("   if char.modelId in routable:")
    print("       # A character can name any model, so price the request first.")
    print("       estimate = await client.chat.completions.estimate_cost(")
    print("           model=char.modelId, messages=messages, expected_completion_tokens=1500")
    print("       )")
    print("       if estimate.total_cost_usd <= MAX_REQUEST_USD:")
    print("           model = char.modelId")
    print("   response = await client.chat.completions.create(")
    print("       model=model,")
    print("       venice_parameters={'character_slug': char.slug},")
    print("       messages=messages,")
    print("       max_completion_tokens=1500,")
    print("   )")
    print("   ```")
    return True


async def main() -> int:
    """Run all character discovery and analysis examples.

    Returns ``0`` only if every section succeeded, ``1`` otherwise.
    """
    print("🚀 Venice AI Character Discovery & Integration Examples")
    print("=" * 70)

    async with VeniceClient() as client:
        characters = await discover_characters(client)
        if characters is None:
            print("\n❌ Character discovery failed; skipping the analysis sections.")
            return 1

        showcase = showcase_set(characters)
        analyze_tags(characters)
        results: list[tuple[str, bool | None]] = [
            ("analyze_statistics", analyze_statistics(showcase))
        ]
        character_selection_guide(characters)
        results += [
            ("server_side_filtering", await server_side_filtering(client)),
            (
                "demonstrate_character_integration",
                await demonstrate_character_integration(client, characters, showcase),
            ),
        ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print(f"\n❌ {len(failed)} section(s) failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\n✅ Character discovery verified; skipped: {', '.join(skipped)}")
        return 0

    print("\n✨ Character discovery examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Walking the whole catalog with characters.iter_all()")
    print("   - Merging tag spelling variants before counting")
    print("   - Ranking by rating only with enough ratings behind it")
    print("   - Server-side search, tag filters and sorting")
    print("   - Pairing a character with a model the API serves")
    print("\n⚠️ Important Notes:")
    print("   - Characters API is currently in Preview")
    print("   - Use character slugs (not names) for API integration")
    print("   - Don't rely on the adult flag alone to filter content")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have character access.", file=sys.stderr)
        sys.exit(1)
