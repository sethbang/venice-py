#!/usr/bin/env python3
"""
Venice AI SDK - Character Details and Individual Character Access
================================================================

This example demonstrates how to access individual character details with
``GET /characters/{slug}`` (``client.characters.get(slug)``):

- Fetching detailed information for specific characters
- Reading the per-character fields (photoUrl, shareUrl, modelId, stats)
- Checking a character's ``modelId`` against the live model catalog before
  using it — many characters name a model that ``models.list()`` does not
  list, or one that is deprecated or offline. The server remaps some of those
  IDs and rejects others, and the remaps are not published, so the catalog is
  the only safe check
- Pricing the chat request before sending it, since a character can name any
  model, then chatting with the character once, live
- Handling a missing slug

``photoUrl`` and ``shareUrl`` are for rendering in a browser (an ``<img>`` tag
or a link). Do not fetch them from server-side code: the image host refuses
programmatic requests.

The live chat in the integration section is the only billed call here.

Exit status: ``0`` when every demo passed, ``1`` otherwise. If the catalog has
no chat model to fall back on, the live chat prints ``Section skipped:``, the
free lookups still decide the result, and the closing summary lists only what
actually ran.
"""

import asyncio
import re
import sys
import uuid
from dataclasses import dataclass
from decimal import Decimal

from venice_ai import NoMatchingModelError, VeniceClient, model_price
from venice_ai.costs import ChatCostEstimate
from venice_ai.exceptions import NotFoundError, VeniceError
from venice_ai.types import Character
from venice_ai.types.api import ModelResponse, UserMessage

#: Tags (after normalization) that mark a character as adult content. The
#: server-side ``adult`` flag alone is not reliable.
ADULT_TAGS = {"adult", "nsfw", "18"}

#: Room for reasoning models, which spend part of the budget thinking.
MAX_COMPLETION_TOKENS = 1500

#: The most this demo will spend on its one chat request, in USD: one cent, a
#: demo-sized budget set by policy rather than by any one model's price. A
#: character can name any catalog model, so the request is priced with
#: ``estimate_cost()`` (prompt plus a full ``MAX_COMPLETION_TOKENS`` reply)
#: before it is sent; over budget, the demo chats on the cheapest model
#: instead. At this token cap one cent covers models listing up to about $6.67
#: per million output tokens, and the run prints how many routable catalog
#: models the budget admits for this exact request, so its reach is visible.
#: The character's persona prompt is added server-side; the estimate's
#: allowance for Venice's system prompt covers it, and the billed prompt
#: tokens are printed next to the estimate to show that.
MAX_REQUEST_USD = Decimal("0.01")


def is_adult(character: Character) -> bool:
    tags = {re.sub(r"[^a-z0-9]", "", t.casefold()) for t in character.tags}
    return character.adult or bool(tags & ADULT_TAGS)


def _short(text: str | None, limit: int) -> str:
    text = " ".join((text or "(no description)").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _featured(client: VeniceClient, count: int) -> list[Character]:
    """A deterministic sample: featured, web-enabled, non-adult characters."""
    page = await client.characters.list(
        sort_by="featured", is_web_enabled=True, is_adult=False, limit=50
    )
    return [c for c in page.data if not is_adult(c)][:count]


async def _text_models(client: VeniceClient) -> dict[str, ModelResponse]:
    """Every listed text model, by id (including deprecated and offline ones)."""
    listing = await client.models.list(type="text")
    return {m.id: m for m in listing.data}


def model_status(model_id: str, text_models: dict[str, ModelResponse]) -> str | None:
    """Why a character's ``modelId`` should not be used, or ``None`` if it can be.

    A listed model can still be deprecated (Venice routes it to a replacement)
    or offline, so being listed is not enough.
    """
    entry = text_models.get(model_id)
    if entry is None:
        return "not in the model catalog"
    if entry.model_spec.offline:
        return "listed, but offline"
    deprecation = entry.model_spec.deprecation
    if deprecation is not None:
        replacement = deprecation.replacementModelId or "no replacement named"
        return f"listed, but deprecated (replacement: {replacement})"
    return None


async def get_character_details(client: VeniceClient) -> bool:
    """Fetch full details for a handful of featured characters."""
    print("🔍 Character Details Access")
    print("-" * 40)

    try:
        sample = await _featured(client, 4)
        text_models = await _text_models(client)
    except VeniceError as e:
        print(f"❌ Error listing characters: {e}")
        return False

    if not sample:
        print("❌ No featured, web-enabled characters to look up.")
        return False

    print("📋 characters.list() returns one page (limit ≤ 100); iter_all() walks them all.")
    print(f"🔍 Getting details for {len(sample)} featured characters...")

    failures = 0
    for i, summary in enumerate(sample, 1):
        try:
            char = (await client.characters.get(summary.slug)).data
        except VeniceError as e:
            print(f"\n❌ Error getting details for {summary.slug}: {e}")
            failures += 1
            continue

        serves = model_status(char.modelId, text_models) or "listed text model"
        stats = char.stats
        print(f"\n═══ Character {i}: {char.name} ═══")
        print(f"   Slug:        {char.slug}")
        print(f"   Model ID:    {char.modelId} ({serves})")
        print(f"   Created:     {char.createdAt}")
        print(f"   Updated:     {char.updatedAt}")
        print(f"   Description: {_short(char.description, 160)}")
        print(f"   Web enabled: {'yes' if char.webEnabled else 'no'}")
        print(f"   Tags:        {', '.join(char.tags[:8]) or '(none)'}")
        print(f"   Imports:     {stats.imports:,}")
        if stats.averageRating is not None and stats.ratingCount:
            print(f"   Rating:      {stats.averageRating:.2f} from {stats.ratingCount} ratings")
        else:
            print("   Rating:      (no ratings yet)")
        print(f"   Share URL:   {char.shareUrl or '—'}")
        print(f"   Photo URL:   {'present (browser use only)' if char.photoUrl else '—'}")

    if failures:
        print(f"\n❌ {failures} of {len(sample)} detail lookups failed.")
        return False
    return True


async def character_metadata_showcase(client: VeniceClient) -> bool:
    """Survey metadata across the whole catalog, including modelId validity."""
    print("\n🎨 Character Metadata Showcase")
    print("-" * 40)

    try:
        characters = [c async for c in client.characters.iter_all()]
        text_models = await _text_models(client)
    except VeniceError as e:
        print(f"❌ Error walking the character catalog: {e}")
        return False

    if not characters:
        print("❌ The character catalog came back empty.")
        return False

    total = len(characters)
    with_photo = sum(1 for c in characters if c.photoUrl)
    with_share = sum(1 for c in characters if c.shareUrl)
    model_ids = {c.modelId for c in characters}
    unlisted = sorted({m for m in model_ids if m not in text_models})
    not_routable = sorted(
        {m for m in model_ids if m in text_models and model_status(m, text_models) is not None}
    )
    unlisted_chars = sum(1 for c in characters if c.modelId in unlisted)
    not_routable_chars = sum(1 for c in characters if c.modelId in not_routable)

    print(f"📊 {total} characters (whole catalog, via iter_all)")
    print(f"   with photoUrl: {with_photo}   with shareUrl: {with_share}")
    print(f"   distinct modelIds: {len(model_ids)}")
    print(
        f"   characters whose modelId is not listed in models.list(type='text'): {unlisted_chars}"
    )
    print(f"   those modelIds: {', '.join(unlisted) or '(none)'}")
    print(f"   characters whose modelId is listed but deprecated or offline: {not_routable_chars}")
    for model_id in not_routable:
        print(f"   • {model_id}: {model_status(model_id, text_models)}")
    print("\n   A character's modelId is the model the Venice app uses for it. Check it")
    print("   against models.list(type='text') before passing it as model=.")
    return True


async def safe_character_access(client: VeniceClient, char_slug: str) -> Character | None:
    """Return a character, or None if the slug does not exist.

    Only ``NotFoundError`` is handled; every other error propagates so the
    caller can tell "missing" apart from "the API is down".
    """
    try:
        return (await client.characters.get(char_slug)).data
    except NotFoundError:
        print(f"   ℹ️ Character {char_slug!r} not found")
        return None


def _list_price(entry: ModelResponse, key: str) -> Decimal | None:
    """A text model's listed USD price per million ``key`` tokens, if any."""
    price = (entry.model_dump().get("model_spec") or {}).get("pricing", {}).get(key)
    usd = price.get("usd") if isinstance(price, dict) else None
    return Decimal(str(usd)) if isinstance(usd, int | float) else None


def budget_reach(estimate: ChatCostEstimate, text_models: dict[str, ModelResponse]) -> str:
    """How many routable, priced text models fit ``MAX_REQUEST_USD`` for this request.

    Each model is priced at its listed input and output rates for the
    estimate's prompt tokens plus a full ``MAX_COMPLETION_TOKENS`` reply.
    """
    fits = priced = 0
    for model_id, entry in text_models.items():
        input_usd, output_usd = _list_price(entry, "input"), _list_price(entry, "output")
        if model_status(model_id, text_models) is not None:
            continue
        if input_usd is None or output_usd is None:
            continue
        priced += 1
        cost = (estimate.prompt_tokens * input_usd + MAX_COMPLETION_TOKENS * output_usd) / 10**6
        fits += cost <= MAX_REQUEST_USD
    return f"${MAX_REQUEST_USD} admits {fits} of {priced} routable priced text models"


async def choose_chat_model(
    client: VeniceClient,
    char: Character,
    text_models: dict[str, ModelResponse],
    messages: list[UserMessage],
) -> tuple[str, str, ChatCostEstimate | None]:
    """Pick the model for a character chat and say why.

    The character's own model is used when the catalog serves it and the
    request's worst-case estimate fits ``MAX_REQUEST_USD``. Otherwise the
    cheapest chat model is used. Returns ``(model, reason, estimate)``, where
    ``estimate`` prices the character's own model when it could be priced.

    Raises:
        NoMatchingModelError: The fallback is needed and the catalog has no
            chat model that answers directly.
    """
    estimate: ChatCostEstimate | None = None
    status = model_status(char.modelId, text_models)
    if status is not None:
        reason = f"its modelId is {status}"
    else:
        try:
            estimate = await client.chat.completions.estimate_cost(
                model=char.modelId,
                messages=messages,
                expected_completion_tokens=MAX_COMPLETION_TOKENS,
            )
        except ValueError as e:  # the catalog has no token price for it
            reason = f"its modelId cannot be priced ({e})"
        else:
            cost = estimate.total_cost_usd
            if cost <= MAX_REQUEST_USD:
                return char.modelId, f"its own modelId, estimated ≤ ${cost:.4f}", estimate
            reason = f"its modelId would cost up to ${cost:.4f}, over ${MAX_REQUEST_USD}"
    fallback = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    return fallback, f"cheapest direct-answer model instead: {reason}", estimate


@dataclass
class IntegrationResult:
    """What the integration demo did, so the summary claims only what ran."""

    ok: bool
    #: The live character chat was sent and checked.
    chat_ran: bool = False
    #: The chat request was priced with ``estimate_cost()`` before sending.
    priced: bool = False


async def advanced_character_integration(client: VeniceClient) -> IntegrationResult:
    """Run the integration patterns live."""
    print("\n🚀 Advanced Character Integration")
    print("-" * 40)

    try:
        sample = await _featured(client, 10)
        text_models = await _text_models(client)
    except VeniceError as e:
        print(f"❌ Error preparing the integration demo: {e}")
        return IntegrationResult(ok=False)
    if not sample:
        print("❌ No featured character available for the live demo.")
        return IntegrationResult(ok=False)

    ok = True
    chat_ran = priced = False

    # Pattern 1: chat on the character's own model, checked against the catalog
    # and priced before the request is sent.
    print("\n🎯 Pattern 1: Catalog-checked, priced character chat (live)")
    char = sample[0]
    messages = [UserMessage(content="In two sentences, what can you help me with?")]
    try:
        model, why, estimate = await choose_chat_model(client, char, text_models, messages)
        priced = estimate is not None
        # Only the character is set. include_venice_system_prompt=False is not
        # sent: in a character chat the billed prompt is the same with or
        # without it, and the persona is kept either way.
        response = await client.chat.completions.create(
            model=model,
            venice_parameters={"character_slug": char.slug},
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: live chat needs a fallback chat model, and none fits ({e})")
    except (VeniceError, ValueError) as e:
        print(f"   ❌ Chat with {char.slug} failed: {e}")
        ok = False
    else:
        chat_ran = True
        finish = response.choices[0].finish_reason if response.choices else None
        reply = (response.text or "").strip()
        usage = response.usage
        entry = text_models.get(model)
        price = model_price(entry.model_dump()) if entry is not None else None
        price_note = f"${price:.2f}/1M blended" if price is not None else "unpriced"
        print(f"   Character: {char.name} ({char.slug})")
        print(f"   Model:     {model} ({price_note})")
        print(f"   Why:       {why}")
        if estimate is not None:
            print(f"   Budget:    {budget_reach(estimate, text_models)}")
        print(f"   finish_reason: {finish}")
        if usage is not None:
            print(f"   Usage:     {usage}")
            if estimate is not None and model == estimate.model:
                # The persona is added server-side, so compare what was billed.
                print(
                    f"   Prompt:    estimated {estimate.prompt_tokens} tokens, "
                    f"billed {usage.prompt_tokens}"
                )
                if usage.prompt_tokens > estimate.prompt_tokens:
                    print("   ❌ The billed prompt exceeded the estimate, so the budget")
                    print("      check above understated this request.")
                    ok = False
        if response.cost is not None and response.cost.usd is not None:
            print(f"   Billed:    ${response.cost.usd:.6f}")
        print(f"   Reply: {reply}")
        # Some backends report "stop" even when the reply ran into the cap, so
        # the completion count is checked too, and a reply with no usage to
        # check cannot be shown complete.
        at_cap = usage is None or usage.completion_tokens >= MAX_COMPLETION_TOKENS
        if finish != "stop" or at_cap or not reply:
            print("   ❌ The reply was cut off or empty.")
            ok = False

    off_catalog = next((c for c in sample if model_status(c.modelId, text_models)), None)
    if off_catalog is not None:
        try:
            fallback = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        except NoMatchingModelError as e:
            print(f"\nSection skipped: fallback case, no chat model fits ({e})")
        except (VeniceError, ValueError) as e:
            print(f"   ❌ resolve_chat(prefer='cheapest', exclude_reasoning=True) failed: {e}")
            ok = False
        else:
            status = model_status(off_catalog.modelId, text_models)
            print(f"\n   Fallback case: {off_catalog.name} names {off_catalog.modelId}")
            print(f"   ({status}), so the chat would use {fallback}.")

    # Pattern 2: gallery data for a UI.
    print("\n🖼️ Pattern 2: Character Gallery Builder")
    print("   Build the gallery from get() results; render photoUrl in an <img> tag")
    print("   and shareUrl as a link. Both are for browsers, not for server fetches.")
    print("   ```python")
    print("   char = (await client.characters.get(slug)).data")
    print("   card = {'name': char.name, 'photo': char.photoUrl, 'link': char.shareUrl,")
    print("           'tags': char.tags, 'imports': char.stats.imports}")
    print("   ```")

    # Pattern 3: tell a missing slug apart from a real failure.
    print("\n✅ Pattern 3: Missing-slug handling (live)")
    missing_slug = f"no-such-character-{uuid.uuid4().hex[:8]}"
    try:
        found = await safe_character_access(client, char.slug)
        missing = await safe_character_access(client, missing_slug)
    except VeniceError as e:
        print(f"   ❌ Lookup failed for a reason other than a missing slug: {e}")
        return IntegrationResult(ok=False, chat_ran=chat_ran, priced=priced)
    print(f"   {char.slug!r} → {'found' if found else 'missing'}")
    print(f"   {missing_slug!r} → {'found' if missing else 'missing (NotFoundError handled)'}")
    if found is None or missing is not None:
        print("   ❌ Lookup results were not what the demo expected.")
        ok = False

    return IntegrationResult(ok=ok, chat_ran=chat_ran, priced=priced)


async def main() -> int:
    """Run all character details examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Character Details Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        results: list[tuple[str, bool]] = [
            ("get_character_details", await get_character_details(client)),
            ("character_metadata_showcase", await character_metadata_showcase(client)),
        ]
        integration = await advanced_character_integration(client)
        results.append(("advanced_character_integration", integration.ok))

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Character details examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Individual character access via GET /characters/{slug}")
    print("   - Character metadata fields (photoUrl, shareUrl, modelId, stats)")
    print("   - Validating a character's modelId against the live catalog")
    if integration.chat_ran and integration.priced:
        print("   - Pricing a request with estimate_cost() before sending it")
    if integration.chat_ran:
        print("   - A live character chat, checked for a complete reply")
    else:
        print("   ℹ️ The live character chat did not run (see 'Section skipped:' above)")
    print("   - Handling a missing slug without hiding other errors")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and you have character access.",
            file=sys.stderr,
        )
        sys.exit(1)
