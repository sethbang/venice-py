#!/usr/bin/env python3
"""
Venice AI SDK - Model Selection by Traits and Compatibility
===========================================================

This example demonstrates how to use Venice AI's model selection features:
- Selecting models by the semantic traits the API publishes (``default``,
  ``default_code``, ``highest_quality``, ...). The trait set differs per model
  type and changes over time, so it is always read from ``list_traits()``.
- Cross-platform compatibility mappings, and how to judge whether an alias
  still points where you want it to
- Falling back to ``resolve_*()`` when a type has no mapping
- ``client.models.resolve()`` with capability filters, and ``preferred_models`` /
  ``exclude_models`` lists built from the catalog
- ``resolve_music()`` for music models
- Your own ranking, through ``DynamicModelSelector``, ``cheapest_selector`` and
  ``cheapest_model_strategy``

All calls here are read-only catalog lookups — running this example costs nothing.
"""

import asyncio
import sys
from collections import Counter
from typing import Any

from venice_ai import (
    DynamicModelSelector,
    NoMatchingModelError,
    VeniceClient,
    cheapest_model_strategy,
    cheapest_selector,
    model_price,
)
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import KNOWN_MODEL_TYPES, ModelResponse, MusicModelSpec, TextModelSpec


def _describe(model: ModelResponse | None) -> str:
    """Short catalog description of a model, or a note that it isn't listed."""
    if model is None:
        return "not in the current catalog"
    parts = [model.model_spec.name or model.id]
    spec = model.model_spec
    if isinstance(spec, TextModelSpec) and spec.availableContextTokens:
        parts.append(f"{spec.availableContextTokens:,.0f} ctx")
    if spec.deprecation is not None:
        parts.append("DEPRECATED")
    return ", ".join(parts)


async def _load_catalog(client: VeniceClient) -> dict[str, ModelResponse]:
    """Fetch the full catalog once and index it by model id."""
    listing = await client.models.list()
    return {m.id: m for m in listing.data}


async def discover_models_by_traits() -> bool:
    """Discover every trait the API publishes, for every model type."""
    print("🏷️ Model Discovery by Traits")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            catalog = await _load_catalog(client)
            traits_by_type = {
                model_type: (await client.models.list_traits(type=model_type)).data
                for model_type in KNOWN_MODEL_TYPES
            }
    except VeniceError as e:
        print(f"❌ Error reading traits: {e}")
        return False

    without_traits = []
    for model_type, traits in traits_by_type.items():
        if not traits:
            without_traits.append(model_type)
            continue
        print(f"\n🎯 {model_type.upper()}: {len(traits)} traits")
        for trait_name, model_id in traits.items():
            print(f"   🔹 {trait_name:26s} → {model_id} ({_describe(catalog.get(model_id))})")

    if without_traits:
        print(f"\nℹ️ No traits published today for: {', '.join(without_traits)}")
        print("   Use the resolve_*() helpers or catalog constraints for those types.")

    return any(traits_by_type.values())


async def demonstrate_trait_based_selection() -> bool:
    """Pick a model from an ordered trait preference, falling back gracefully."""
    print("\n🎯 Practical Trait-Based Selection")
    print("-" * 40)

    # (title, model type, trait preference in order, rationale)
    scenarios: list[tuple[str, str, list[str], str]] = [
        (
            "💬 Chat Application",
            "text",
            ["default"],
            "The general-purpose default is the right start for chat",
        ),
        (
            "💻 Coding Assistant",
            "text",
            ["default_code", "default"],
            "Prefer the code-tuned model, fall back to the default",
        ),
        (
            "🎨 Image Generation",
            "image",
            ["highest_quality", "default"],
            "For creative work, quality is often preferred over speed",
        ),
    ]

    ok = True
    async with VeniceClient() as client:
        for title, model_type, preference, rationale in scenarios:
            print(f"\n{title}")
            print(f"   📝 {rationale}")
            try:
                traits = (await client.models.list_traits(type=model_type)).data
            except VeniceError as e:
                print(f"   ❌ Error reading {model_type} traits: {e}")
                ok = False
                continue

            selected = None
            for i, trait in enumerate(preference, 1):
                model_id = traits.get(trait)
                if model_id:
                    print(f"      {i}. {trait}: {model_id}")
                    selected = selected or (trait, model_id)
                else:
                    print(f"      {i}. {trait}: ℹ️ not offered for {model_type} today")

            if selected:
                print(f"   ✅ Would select: {selected[1]} (trait '{selected[0]}')")
            else:
                print("   ❌ None of the preferred traits exist for this type")
                ok = False

    return ok


async def _load_compat(client: VeniceClient) -> dict[str, dict[str, str]]:
    """Fetch the compatibility mapping for every known model type once."""
    return {
        model_type: (await client.models.list_compatibility(type=model_type)).data
        for model_type in KNOWN_MODEL_TYPES
    }


async def explore_compatibility_mappings() -> bool:
    """Show the live alias map, and where the aliases actually land."""
    print("\n🔄 Cross-Platform Compatibility")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            catalog = await _load_catalog(client)
            compat = await _load_compat(client)
    except VeniceError as e:
        print(f"❌ Error reading compatibility mappings: {e}")
        return False

    total = sum(len(m) for m in compat.values())
    print(f"📊 {total} compatibility aliases across {sum(1 for m in compat.values() if m)} types")

    for model_type, mapping in compat.items():
        if not mapping:
            continue
        print(f"\n🔍 {model_type.upper()} aliases ({len(mapping)}):")
        for alias, target in list(mapping.items())[:8]:
            print(f"   📄 {alias} → {target}")
        if len(mapping) > 8:
            print(f"   ... and {len(mapping) - 8} more")

        # An alias is a compatibility shim, not a recommendation: many aliases
        # can collapse onto one older target. Show where they land.
        print("   🎯 Targets:")
        for target, count in Counter(mapping.values()).most_common():
            print(f"      {count:3d} alias(es) → {target} ({_describe(catalog.get(target))})")

    empty = [t for t, m in compat.items() if not m]
    print(f"\nℹ️ No aliases published for: {', '.join(empty) or '(none)'}")
    return True


async def demonstrate_migration_workflow() -> bool | None:
    """Plan a migration: compare each alias target with the resolver's pick.

    Returns ``None`` (section skipped) when no type has a model to compare.
    """
    print("\n🚀 Migration Workflow Example")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            catalog = await _load_catalog(client)
            compat = await _load_compat(client)
        except VeniceError as e:
            print(f"❌ Error loading catalog data: {e}")
            return False

        resolvers = {
            "text": client.models.resolve_chat,
            "embedding": client.models.resolve_embedding,
            "image": client.models.resolve_image,
            "tts": client.models.resolve_tts,
        }

        print("📋 For each type: what an existing alias maps to, versus what the")
        print("   resolver would choose for new code today.")

        resolved_types: list[str] = []
        for model_type, resolve in resolvers.items():
            try:
                resolved = await resolve()
            except NoMatchingModelError:
                # A type with no model today is a catalog fact, not a failure.
                print(f"\n📦 {model_type.upper()}: ℹ️ no {model_type} model in the catalog")
                continue
            except VeniceError as e:
                print(f"\n📦 {model_type.upper()}: ❌ resolver failed: {e}")
                ok = False
                continue

            resolved_types.append(model_type)
            mapping = compat.get(model_type, {})
            print(f"\n📦 {model_type.upper()}")
            print(f"   🤖 resolver pick: {resolved} ({_describe(catalog.get(resolved))})")
            if not mapping:
                print("   ℹ️ No aliases for this type; resolve the model by capability instead.")
                continue

            targets = Counter(mapping.values())
            for target, count in targets.most_common():
                verdict = "matches the resolver" if target == resolved else "differs from resolver"
                print(f"   🔁 {count} alias(es) → {target}: {verdict}")

        print("\n📝 Migration guidance:")
        print("   • Aliases keep existing code running, but their targets are fixed")
        print("     server-side and can lag behind newer models.")
        print("   • For new or migrated code, call resolve_chat() / resolve_embedding() /")
        print("     resolve_image() / resolve_tts() so the choice tracks the catalog.")
        print("   • A trait is the one model Venice tags for a role. resolve_chat() filters")
        print("     by capability and passes over reasoning models unless you ask for")
        print("     reasoning, so its pick can differ from the 'default' trait above.")
        print("   • resolve_chat(prefer='cheapest') ranks strictly by price, reasoning")
        print("     models included; add exclude_reasoning=True for direct answers.")

    if ok and not resolved_types:
        print("Section skipped: no resolver found a model of any type in the catalog")
        return None
    return ok


def _plain_chat_models(catalog: dict[str, ModelResponse]) -> list[ModelResponse]:
    """Chat models with a listed context size, largest first, that any filter can return.

    Beta and end-to-end encrypted models are left out, because ``prefer="cheapest"``
    skips them.
    """
    models = []
    for model in catalog.values():
        spec = model.model_spec
        if not isinstance(spec, TextModelSpec) or not model.context_length or spec.beta:
            continue
        if spec.capabilities is not None and spec.capabilities.supportsE2EE:
            continue
        models.append(model)
    return sorted(models, key=lambda m: m.context_length or 0, reverse=True)


async def resolve_with_filters() -> bool | None:
    """Select with ``client.models.resolve()``, checking each pick against the catalog.

    The context size and the ``preferred_models`` list come from the live catalog, so
    no model ID or size is written down here. Returns ``None`` (section skipped) when
    the catalog has no chat model to filter.
    """
    print("\n🧰 models.resolve() with filters")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            catalog = await _load_catalog(client)
        except VeniceError as e:
            print(f"❌ Error reading the catalog: {e}")
            return False
        ranked = _plain_chat_models(catalog)
        if not ranked:
            print("Section skipped: the catalog has no chat model with a listed context size")
            return None

        # Ask for at least the median context size, which some model must meet.
        min_context = ranked[len(ranked) // 2].context_length or 0
        pick = await client.models.resolve(type="chat", min_context_tokens=min_context)
        size = catalog[pick].context_length or 0
        print(f"   min_context_tokens={min_context:,} → {pick} ({size:,} tokens)")
        if size < min_context:
            print("   ❌ The pick has less context than was asked for")
            ok = False

        # require_private: only models whose catalog privacy is "private".
        try:
            pick = await client.models.resolve(type="chat", require_private=True)
        except NoMatchingModelError:
            print("   ℹ️ require_private=True: no private chat model in the catalog today")
        else:
            privacy = catalog[pick].model_spec.privacy
            print(f"   require_private=True → {pick} (privacy: {privacy})")
            if privacy != "private":
                print("   ❌ The pick is not private")
                ok = False

        # A list built from the catalog: the largest contexts, in that order. The first
        # entry that passes the filters wins, even over prefer="cheapest".
        preferred = [m.id for m in ranked[:3]]
        pick = await client.models.resolve_chat(preferred_models=preferred, prefer="cheapest")
        print(f"   preferred_models={preferred} → {pick}")
        if pick != preferred[0]:
            print(f"   ❌ Expected the first preferred model, {preferred[0]}")
            ok = False

        # exclude_models takes a model out of the running, preferred or not.
        pick = await client.models.resolve_chat(
            preferred_models=preferred, exclude_models=preferred[:1]
        )
        print(f"   exclude_models={preferred[:1]} → {pick}")
        if pick == preferred[0]:
            print("   ❌ The excluded model was picked")
            ok = False

        # Filters no model can meet raise NoMatchingModelError, never a wrong model.
        impossible = max(m.context_length or 0 for m in ranked) + 1
        try:
            pick = await client.models.resolve(type="chat", min_context_tokens=impossible)
        except NoMatchingModelError:
            print(f"   min_context_tokens={impossible:,} → NoMatchingModelError, as expected")
        else:
            print(f"   ❌ Expected NoMatchingModelError, got {pick}")
            ok = False
    return ok


async def resolve_music_models() -> bool | None:
    """Pick music models with ``resolve_music()`` and check them against the catalog.

    Venice also types text-to-speech, sound-effect and voice-changer models as
    ``music``, and ``exclude_non_music=True`` leaves those out. These calls read only
    the catalog. ``prefer="cheapest"`` also works here, and quotes each model for free;
    ``examples/music/music_generation.py`` covers that path.

    Returns ``None`` (section skipped) when the catalog has no music model.
    """
    print("\n🎵 resolve_music()")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            catalog = await _load_catalog(client)
            pick = await client.models.resolve_music(exclude_non_music=True)
        except NoMatchingModelError as e:
            print(f"Section skipped: no music model in the catalog ({e})")
            return None
        except VeniceError as e:
            print(f"❌ Error resolving a music model: {e}")
            return False
        print(f"   exclude_non_music=True → {pick} ({_describe(catalog.get(pick))})")
        if pick not in catalog or catalog[pick].type != "music":
            print("   ❌ The pick is not a music model in the catalog")
            return False

        try:
            pick = await client.models.resolve_music(require_force_instrumental=True)
        except NoMatchingModelError:
            print("   ℹ️ require_force_instrumental=True: no such model in the catalog today")
            return True
        spec = catalog[pick].model_spec
        print(f"   require_force_instrumental=True → {pick}")
        if not (isinstance(spec, MusicModelSpec) and spec.supports_force_instrumental):
            print("   ❌ The pick does not list supports_force_instrumental")
            return False
    return True


async def custom_selection_strategies() -> bool | None:
    """Rank candidates yourself, through ``DynamicModelSelector``.

    ``client.models.resolve()`` covers the catalog's ranking and price. For any other
    ranking, a strategy is a function from the candidate model dicts to one ID, and
    ``DynamicModelSelector`` applies it after the filters. Each dict holds
    ``id``, ``availableContextTokens`` and ``privacy`` at the top level, with
    ``capabilities`` and ``pricing`` under ``model_spec``.

    Returns ``None`` (section skipped) when no chat model passes the filters.
    """
    print("\n🧮 Your own ranking: DynamicModelSelector")
    print("-" * 40)

    seen: list[dict[str, Any]] = []

    def largest_context(candidates: list[dict[str, Any]]) -> str:
        seen[:] = candidates
        best = max(candidates, key=lambda m: m.get("availableContextTokens") or 0)
        return str(best["id"])

    def cheapest_of_candidates(candidates: list[dict[str, Any]]) -> str:
        seen[:] = candidates
        return cheapest_model_strategy(candidates)

    ok = True
    async with VeniceClient() as client:
        selector = DynamicModelSelector(client, default_selector=largest_context)
        try:
            # With no selector= on the call, default_selector ranks.
            pick = await selector.select_chat_model(exclude_beta=True)
            sizes = [m.get("availableContextTokens") or 0 for m in seen]
            picked = next((m for m in seen if m["id"] == pick), None)
            size = (picked.get("availableContextTokens") or 0) if picked else 0
            print(f"   largest_context over {len(seen)} candidates → {pick} ({size:,.0f} tokens)")
            if picked is None or size != max(sizes):
                print("   ❌ The strategy's choice was not returned")
                ok = False

            # cheapest_model_strategy is the price ranking on its own, to call from a
            # strategy that adds rules of its own. selector= overrides the default.
            # exclude_e2ee: an encrypted model is often cheapest, and needs client-side
            # encryption that a plain request does not do.
            pick = await selector.select_chat_model(
                exclude_beta=True,
                exclude_e2ee=True,
                exclude_reasoning=True,
                selector=cheapest_of_candidates,
            )
            prices = [price for m in seen if (price := model_price(m)) is not None]
            picked = next((m for m in seen if m["id"] == pick), None)
            price = model_price(picked) if picked else None
            shown = f"${price:.4f}" if price is not None else "unpriced"
            print(f"   cheapest_model_strategy over {len(seen)} candidates → {pick} ({shown})")
            if price is None or price != min(prices):
                print("   ❌ The pick is not the lowest price among the candidates")
                ok = False

            # cheapest_selector builds a whole strategy: the first preferred model
            # that passed the filters, otherwise the lowest price.
            cheapest = pick
            others = [m["id"] for m in seen if m["id"] != cheapest]
            if others:
                pick = await selector.select_chat_model(
                    exclude_beta=True,
                    exclude_e2ee=True,
                    exclude_reasoning=True,
                    selector=cheapest_selector(preferred_models=others[:1]),
                )
                print(f"   cheapest_selector(preferred_models={others[:1]}) → {pick}")
                if pick != others[0]:
                    print("   ❌ The preferred model did not win over price")
                    ok = False
        except NoMatchingModelError as e:
            print(f"Section skipped: no chat model passes the filters ({e})")
            return None
        except VeniceError as e:
            print(f"❌ Error selecting a model: {e}")
            return False
    return ok


async def main() -> int:
    """Run all model selection and compatibility examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Model Selection & Compatibility Examples")
    print("=" * 70)

    # Each section returns True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = [
        ("discover_models_by_traits", await discover_models_by_traits()),
        ("demonstrate_trait_based_selection", await demonstrate_trait_based_selection()),
        ("explore_compatibility_mappings", await explore_compatibility_mappings()),
        ("demonstrate_migration_workflow", await demonstrate_migration_workflow()),
        ("resolve_with_filters", await resolve_with_filters()),
        ("resolve_music_models", await resolve_music_models()),
        ("custom_selection_strategies", await custom_selection_strategies()),
    ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\n✅ Catalog sections verified; skipped: {', '.join(skipped)}")
        return 0

    print("\n✨ Model selection examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Reading the per-type trait set from list_traits()")
    print("   - Ordered trait preferences with graceful fallback")
    print("   - Inspecting compatibility aliases and where they land")
    print("   - Choosing resolve_*() over aliases for new code")
    print("   - resolve() filters, and preferred and excluded lists built from the catalog")
    print("   - resolve_music() and its music-only filters")
    print("   - Your own ranking through DynamicModelSelector")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check your network connection and API key.", file=sys.stderr)
        sys.exit(1)
