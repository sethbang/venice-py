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

All calls here are read-only catalog lookups — running this example costs nothing.
"""

import asyncio
import sys
from collections import Counter

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import KNOWN_MODEL_TYPES, ModelResponse, TextModelSpec


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
