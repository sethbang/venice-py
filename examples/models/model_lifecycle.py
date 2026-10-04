#!/usr/bin/env python3
"""
Venice AI SDK - Model Lifecycle & Capability Metadata
=====================================================

Newer fields on the model catalog let you inspect a model's *lifecycle* and
*capability envelope* without a trial-and-error API call. This example reads
them straight off ``client.models.list(...)`` and the per-model lookups
``client.models.get(...)`` / ``client.models.get_capabilities(...)``:

- ``ModelResponse.context_length`` — max context window in tokens, surfaced as
  a typed top-level field (mirrors ``model_spec.availableContextTokens``).
- ``model_spec.deprecation`` (:class:`ModelDeprecation`) — ``startsAt`` /
  ``removesAt`` lifecycle instants, ``replacementModelId`` (where to migrate),
  and ``autoRemap`` (whether Venice silently re-routes the retired ID).
- Text capabilities — ``reasoningEffortOptions`` / ``defaultReasoningEffort``
  tell you which ``reasoning_effort`` values a model accepts *before* you send
  one, instead of guessing.
- Image constraints — ``defaultQuality`` / ``qualities`` tell you which
  ``quality`` tiers a quality-aware image model supports.
- ``client.models.get(model_id)`` — fetch one model's full
  :class:`ModelResponse` by id (the SDK abstracts the list-and-filter pattern
  Venice's catalog otherwise requires; there is no per-model GET endpoint).
- ``client.models.get_capabilities(model_id)`` — a typed, snake_case
  :class:`Capabilities` view (polymorphic by model type, e.g.
  :class:`ChatCapabilities`) so you can introspect feature flags directly
  instead of probing ``resolve_chat(require_function_calling=True)`` etc.

All of this is a read-only ``GET /models`` — running this example costs nothing.

See also ``models/model_selection.py`` for resolver-based selection, and
``image/quality_control.py`` for *using* the discovered ``quality`` tiers.
"""

import asyncio
import sys
from collections import Counter

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import ImageModelSpec, TextModelSpec


async def text_model_capabilities() -> bool:
    """Summarize context windows and reasoning-effort options across text models.

    Returns ``True`` on success, ``False`` if the catalog read failed.
    """
    print("🧠 Text model capabilities")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            models = await client.models.list(type="text")
    except VeniceError as e:
        print(f"❌ Error reading text model capabilities: {e}")
        return False

    if not models.data:
        print("❌ The catalog returned no text models.")
        return False

    # context_length is a typed top-level field (None for non-text models).
    contexts = [m.context_length for m in models.data if m.context_length]
    print(f"📊 {len(models.data)} text models")
    if contexts:
        print(f"   context_length: {min(contexts):,} to {max(contexts):,} tokens")

    # Group models by the exact reasoning_effort values they accept, so you can
    # see which values are safe to send before sending one.
    by_options: Counter[tuple[str, ...]] = Counter()
    defaults: Counter[str] = Counter()
    examples: dict[tuple[str, ...], str] = {}
    unsupported = 0
    for entry in models.data:
        spec = entry.model_spec
        caps = spec.capabilities if isinstance(spec, TextModelSpec) else None
        if caps is None or not caps.supportsReasoningEffort:
            unsupported += 1
            continue
        options = tuple(caps.reasoningEffortOptions or ())
        by_options[options] += 1
        examples.setdefault(options, entry.id)
        defaults[caps.defaultReasoningEffort or "—"] += 1

    print(f"\n   reasoning_effort not supported: {unsupported} models")
    print("   reasoning_effort option sets (count, example model):")
    for options, count in by_options.most_common():
        label = ", ".join(options) or "(no options listed)"
        print(f"      {count:3d} × [{label}]  e.g. {examples[options]}")
    if defaults:
        summary = ", ".join(f"{value} ({count})" for value, count in defaults.most_common())
        print(f"   default reasoning_effort: {summary}")

    return True


async def image_quality_tiers() -> bool:
    """Surface the quality tiers each quality-aware image model supports.

    Returns ``True`` on success, ``False`` if the catalog read failed.
    """
    print("\n\n🎨 Image model quality tiers")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            models = await client.models.list(type="image")
    except VeniceError as e:
        print(f"❌ Error reading image quality tiers: {e}")
        return False

    quality_aware = 0
    for entry in models.data:
        spec = entry.model_spec
        if not isinstance(spec, ImageModelSpec) or spec.constraints is None:
            continue
        constraints = spec.constraints
        # defaultQuality / qualities are present only on quality-aware models.
        if constraints.qualities or constraints.defaultQuality:
            quality_aware += 1
            tiers = constraints.qualities or []
            print(f"\n📦 {entry.id}")
            print(f"   qualities: {', '.join(tiers) or '—'}")
            print(f"   defaultQuality: {constraints.defaultQuality or '—'}")

    print(f"\n   {quality_aware} of {len(models.data)} image models are quality-aware.")
    return True


async def deprecation_report() -> bool:
    """Flag any model carrying deprecation metadata and where to migrate.

    One catalog call covers every model type. Returns ``True`` on success and
    ``False`` if the catalog read failed, so an outage never reads as a clean
    report.
    """
    print("\n\n⏳ Deprecation lifecycle report")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            models = await client.models.list()
    except VeniceError as e:
        print(f"❌ Error building deprecation report: {e}")
        return False

    if not models.data:
        print("❌ The catalog came back empty, so there is nothing to report on.")
        return False

    flagged = 0
    for entry in models.data:
        dep = entry.model_spec.deprecation
        if dep is None:
            continue
        flagged += 1
        print(f"\n⚠️  {entry.id} ({entry.type})")
        if dep.startsAt:
            print(f"   warnings active from: {dep.startsAt}")
        if dep.removesAt:
            print(f"   removed from catalog:  {dep.removesAt}")
        if dep.date:
            print(f"   legacy sunset date:    {dep.date}")
        if dep.replacementModelId:
            print(f"   ➡️  migrate to: {dep.replacementModelId}")
        print(f"   auto-remap retired ID: {dep.autoRemap}")

    types = len({entry.type for entry in models.data})
    print(f"\n   Checked {len(models.data)} models across {types} types; {flagged} flagged.")
    if flagged == 0:
        print("   Venice publishes deprecation metadata here (and via response headers)")
        print("   when a model is scheduled for retirement — check before pinning.")
    return True


async def model_detail_and_capabilities() -> bool | None:
    """Look up one model's detail + typed capabilities by id.

    Demonstrates the per-model lookups ``client.models.get(model_id)`` and
    ``client.models.get_capabilities(model_id)``. The id comes from a resolver
    (never hardcoded) so this keeps working as the catalog evolves.

    Returns ``True`` on success, ``False`` if either lookup failed, and
    ``None`` (section skipped) when the catalog has no chat model.
    """
    print("\n\n🔎 Per-model detail & typed capabilities")
    print("-" * 40)

    try:
        async with VeniceClient() as client:
            # Resolve a real chat model id dynamically — no hardcoded IDs.
            model_id = await client.models.resolve_chat()
            # Returns the full ModelResponse for a single id (the SDK abstracts
            # the list-and-filter pattern; Venice has no per-model GET endpoint).
            detail = await client.models.get(model_id)
            # Polymorphic, snake_case Capabilities view; for a chat model this
            # is a ChatCapabilities.
            caps = await client.models.get_capabilities(model_id)
    except NoMatchingModelError as e:
        print(f"Section skipped: no chat model in the catalog to look up ({e})")
        return None
    except (VeniceError, ValueError) as e:  # models.get: ValueError if unlisted
        print(f"❌ Error looking up model detail/capabilities: {e}")
        return False

    print(f"🤖 Resolved chat model: {model_id}")
    print("\n📦 models.get() detail")
    print(f"   id:             {detail.id}")
    print(f"   type:           {detail.type}")
    print(f"   owned_by:       {detail.owned_by}")
    ctx = detail.context_length
    print(f"   context_length: {f'{ctx:,} tokens' if ctx else '—'}")
    print(f"   name:           {detail.model_spec.name or '—'}")
    print(f"   beta:           {detail.model_spec.beta}")
    if detail.model_spec.deprecation is not None:
        dep = detail.model_spec.deprecation
        print(f"   ⚠️  deprecated — replacement: {dep.replacementModelId or '—'}")

    print("\n🧩 models.get_capabilities() flags")
    print(f"   (capabilities type: {caps.type})")
    for field, value in caps.model_dump().items():
        if field == "type":
            continue
        print(f"   {field}: {value}")

    return True


async def main() -> int:
    """Run the model-lifecycle metadata report (read-only, no cost).

    Returns ``1`` if any demo failed, otherwise ``0``; the detail lookup
    prints ``Section skipped:`` when the catalog has no chat model, and the
    other demos still decide the result.
    """
    print("🚀 Venice AI Model Lifecycle & Capability Metadata")
    print("=" * 60)

    # Each demo returns True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = [
        ("text_model_capabilities", await text_model_capabilities()),
        ("image_quality_tiers", await image_quality_tiers()),
        ("deprecation_report", await deprecation_report()),
        ("model_detail_and_capabilities", await model_detail_and_capabilities()),
    ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\n✅ Catalog metadata verified; skipped: {', '.join(skipped)}")
        return 0

    print("\n\n✨ Done.")
    print("\n💡 Key concepts demonstrated:")
    print("   - context_length as a typed top-level field")
    print("   - reasoningEffortOptions / defaultReasoningEffort discovery")
    print("   - defaultQuality / qualities for quality-aware image models")
    print("   - ModelDeprecation: startsAt / removesAt / replacementModelId")
    print("   - models.get(model_id) for a single model's full detail")
    print("   - models.get_capabilities(model_id) for typed feature flags")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that VENICE_API_KEY is set and valid.", file=sys.stderr)
        sys.exit(1)
