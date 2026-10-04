#!/usr/bin/env python3
"""
Venice AI SDK - Model Discovery and Listing
============================================

This example demonstrates how to discover and explore available models in the Venice AI SDK.
Learn how to browse models by type, understand their capabilities, and access detailed metadata.

Every model entry carries a ``type`` (``text``, ``image``, ``video``, ``inpaint``,
``music``, ``tts``, ``asr``, ``embedding``, ``upscale``, ``decision``, ...) and a
typed ``model_spec`` whose subclass matches that type. ``ModelResponse.type`` is an
open string: compare it against ``KNOWN_MODEL_TYPES`` and treat anything outside
that tuple as a type this SDK release predates.

All of this is a read-only ``GET /models`` — running this example costs nothing.
"""

import asyncio
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import (
    KNOWN_MODEL_TYPES,
    ASRModelPricing,
    AsrModelSpec,
    AudioModelPricing,
    DecisionModelSpec,
    EmbeddingModelSpec,
    ImageModelPricing,
    ImageModelSpec,
    InpaintModelPricing,
    InpaintModelSpec,
    LLMModelPricing,
    ModelResponse,
    MusicModelPricing,
    MusicModelSpec,
    TextModelSpec,
    TtsModelSpec,
    UpscaleModelSpec,
    VideoModelSpec,
    VideoResolutionPricing,
)


def _ordered_types(types: set[str]) -> list[str]:
    """Known types in SDK order, then any type this SDK release doesn't know yet."""
    known = [t for t in KNOWN_MODEL_TYPES if t in types]
    unknown = sorted(t for t in types if t not in KNOWN_MODEL_TYPES)
    return known + unknown


def _created(model: ModelResponse) -> str:
    """Format the ``created`` epoch as a calendar date."""
    if model.created is None:
        return "unknown"
    return datetime.fromtimestamp(model.created, tz=UTC).strftime("%Y-%m-%d")


def _highlights(model: ModelResponse) -> str:
    """One-line, type-specific summary of what a model supports."""
    spec = model.model_spec
    if isinstance(spec, TextModelSpec):
        caps = spec.capabilities
        parts = []
        if model.context_length:
            parts.append(f"{model.context_length:,} ctx")
        if caps is not None:
            if caps.supportsFunctionCalling:
                parts.append("tools")
            if caps.supportsVision:
                parts.append("vision")
            if caps.supportsReasoning:
                parts.append("reasoning")
            if caps.supportsWebSearch:
                parts.append("web search")
            if caps.optimizedForCode:
                parts.append("code")
        return ", ".join(parts) or "basic chat"
    if isinstance(spec, ImageModelSpec) and spec.constraints is not None:
        qualities = spec.constraints.qualities
        return f"qualities: {', '.join(qualities)}" if qualities else "single quality tier"
    if isinstance(spec, VideoModelSpec) and spec.constraints is not None:
        c = spec.constraints
        durations = ", ".join(c.durations) or "model default"
        resolutions = ", ".join(c.resolutions) or "model default"
        return f"{c.model_type}; durations {durations}; resolutions {resolutions}"
    if isinstance(spec, InpaintModelSpec) and spec.constraints is not None:
        return "multi-image input" if spec.constraints.combineImages else "single image input"
    if isinstance(spec, TtsModelSpec):
        return f"{len(spec.voices or [])} voices"
    if isinstance(spec, MusicModelSpec):
        formats = f"formats {', '.join(spec.supported_formats or []) or 'default'}"
        if spec.supports_lyrics is None:
            return formats
        lyrics = "lyrics" if spec.supports_lyrics else "instrumental only"
        return f"{lyrics}; {formats}"
    if isinstance(spec, EmbeddingModelSpec):
        dims = spec.embeddingDimensions
        return f"{dims}-dim vectors" if dims else "embedding vectors"
    if isinstance(spec, AsrModelSpec):
        return "speech-to-text"
    if isinstance(spec, UpscaleModelSpec):
        return "image upscaling"
    if isinstance(spec, DecisionModelSpec):
        return f"max state tokens: {spec.maxStateTokens or '—'}"
    return "—"


def _describe_pricing(model: ModelResponse) -> list[str]:
    """Render a model's pricing in the unit each pricing shape is billed in."""
    pricing = model.model_spec.pricing
    if pricing is None:
        return ["(no pricing published)"]
    if isinstance(pricing, LLMModelPricing):
        lines = [
            f"input:  ${pricing.input.usd} per 1M tokens",
            f"output: ${pricing.output.usd} per 1M tokens",
        ]
        if pricing.cache_input is not None:
            lines.append(f"cached input: ${pricing.cache_input.usd} per 1M tokens")
        return lines
    if isinstance(pricing, ImageModelPricing):
        lines = []
        if pricing.generation is not None:
            lines.append(f"generation: ${pricing.generation.usd} per image")
        lines.append(
            f"upscale: ${pricing.upscale.x2.usd} (2x) / ${pricing.upscale.x4.usd} (4x) per image"
        )
        return lines
    if isinstance(pricing, AudioModelPricing):
        return [f"input: ${pricing.input.usd} per 1M characters"]
    if isinstance(pricing, ASRModelPricing):
        return [f"audio: ${pricing.per_audio_second.usd} per second"]
    if isinstance(pricing, InpaintModelPricing):
        return [f"edit: ${pricing.inpaint.usd} per operation"]
    if isinstance(pricing, VideoResolutionPricing):
        tiers = ", ".join(f"{res} ${tier.usd}" for res, tier in pricing.resolutions.items())
        return [f"per resolution: {tiers}"]
    if isinstance(pricing, MusicModelPricing):
        if pricing.generation is not None:
            return [f"generation: ${pricing.generation.usd} per track"]
        if pricing.per_second is not None:
            return [f"audio: ${pricing.per_second.usd} per second"]
        if pricing.per_thousand_characters is not None:
            return [f"text: ${pricing.per_thousand_characters.usd} per 1K characters"]
        if pricing.durations:
            return [f"duration tiers: {', '.join(pricing.durations)} seconds"]
    return [f"{type(pricing).__name__}: {pricing.model_dump(by_alias=True)}"]


async def list_all_models() -> bool:
    """List the whole catalog, grouped by each model's ``type``."""
    print("📋 All Available Models")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # `models.list()` with no type returns the full catalog across every
            # model type in a single call.
            all_models = await client.models.list()
        except VeniceError as e:
            print(f"❌ Error listing models: {e}")
            return False

    models_by_type: dict[str, list[ModelResponse]] = defaultdict(list)
    for m in all_models.data:
        models_by_type[m.type].append(m)

    print(f"📊 {len(all_models.data)} models total across {len(models_by_type)} types")

    for model_type in _ordered_types(set(models_by_type)):
        models = models_by_type[model_type]
        marker = "" if model_type in KNOWN_MODEL_TYPES else "  (type not known to this SDK)"
        print(f"\n🏷️ {model_type.upper()} Models ({len(models)}){marker}:")
        for model in models[:3]:
            name = model.model_spec.name or model.id
            print(f"   📄 {name} (ID: {model.id}) - by {model.owned_by}")
        if len(models) > 3:
            print(f"   ... and {len(models) - 3} more {model_type} models")

    return bool(all_models.data)


async def list_models_by_type() -> bool:
    """Use the server-side ``type`` filter for each model type."""
    print("\n🎯 Models by Type (server-side filter)")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        for model_type in KNOWN_MODEL_TYPES:
            try:
                response = await client.models.list(type=model_type)  # type: ignore[arg-type]
            except VeniceError as e:
                print(f"\n❌ {model_type.upper()}: {e}")
                ok = False
                continue

            print(f"\n🔍 {model_type.upper()}: {len(response.data)} models")
            for model in response.data[:2]:
                name = model.model_spec.name or model.id
                print(f"   📄 {name} [{model.id}] — {_highlights(model)}")
            if len(response.data) > 2:
                print(f"   ... and {len(response.data) - 2} more")

    return ok


async def explore_model_details() -> bool | None:
    """Show the full detail and pricing of the resolver's pick for several types.

    Returns ``None`` (section skipped) when no type has a model to show.
    """
    print("\n🔬 Model Details Exploration")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        resolvers: list[tuple[str, Callable[[], Awaitable[str]]]] = [
            ("chat", client.models.resolve_chat),
            ("image", client.models.resolve_image),
            ("tts", client.models.resolve_tts),
            ("embedding", client.models.resolve_embedding),
        ]
        resolved_types: list[str] = []
        for label, resolve in resolvers:
            try:
                model_id = await resolve()
                model = await client.models.get(model_id)
            except NoMatchingModelError:
                # A type with no model today is a catalog fact, not a failure.
                print(f"\nℹ️ {label}: no {label} model in the catalog")
                continue
            except (VeniceError, ValueError) as e:  # models.get: ValueError if unlisted
                print(f"\n❌ {label}: {e}")
                ok = False
                continue

            resolved_types.append(label)
            spec = model.model_spec
            print(f"\n🤖 {label}: {spec.name or model.id}")
            print(f"   🏷️ ID: {model.id}   type: {model.type}")
            print(f"   🏢 Owner: {model.owned_by}")
            print(f"   📅 Released on Venice: {_created(model)}")
            print(f"   🔒 Privacy: {spec.privacy or '—'}   beta: {spec.beta}")
            print(f"   ✨ Highlights: {_highlights(model)}")
            if spec.traits:
                print(f"   🏷️ Traits: {', '.join(spec.traits)}")
            print("   💰 Pricing:")
            for line in _describe_pricing(model):
                print(f"      {line}")

    if ok and not resolved_types:
        print("Section skipped: no resolver found a model of any type in the catalog")
        return None
    return ok


async def compare_model_resolve_results() -> bool | None:
    """Contrast the resolver's pick with the first entry in raw catalog order.

    Returns ``None`` (section skipped) when no type has a model to compare.
    """
    print("\n⚖️ Model Resolve vs Raw Catalog Order")
    print("-" * 40)
    print("   The catalog is not sorted by suitability, so its first entry is")
    print("   rarely the model you want. The resolvers apply traits and filters.")

    ok = True
    async with VeniceClient() as client:
        pairs: list[tuple[str, Callable[[], Awaitable[str]]]] = [
            ("text", client.models.resolve_chat),
            ("embedding", client.models.resolve_embedding),
            ("image", client.models.resolve_image),
        ]
        resolved_types: list[str] = []
        for model_type, resolve in pairs:
            try:
                resolved = await resolve()
                listing = await client.models.list(type=model_type)  # type: ignore[arg-type]
            except NoMatchingModelError:
                print(f"   ℹ️ {model_type}: no {model_type} model in the catalog")
                continue
            except VeniceError as e:
                print(f"   ❌ {model_type}: {e}")
                ok = False
                continue
            resolved_types.append(model_type)
            first = listing.data[0].id if listing.data else "—"
            same = "same" if first == resolved else "differs"
            print(f"   🏷️ {model_type:9s} resolver: {resolved:32s} first listed: {first} ({same})")

    if ok and not resolved_types:
        print("Section skipped: no resolver found a model of any type in the catalog")
        return None
    return ok


async def main() -> int:
    """Run all model listing examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` otherwise, so a
    real API failure surfaces as a non-zero process exit.
    """
    print("🚀 Venice AI Model Discovery Examples")
    print("=" * 60)

    # Each section returns True (passed), False (failed) or None (skipped).
    results: list[tuple[str, bool | None]] = [
        ("list_all_models", await list_all_models()),
        ("list_models_by_type", await list_models_by_type()),
        ("explore_model_details", await explore_model_details()),
        ("compare_model_resolve_results", await compare_model_resolve_results()),
    ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    if skipped:
        print(f"\n✅ Catalog sections verified; skipped: {', '.join(skipped)}")
        return 0

    print("\n✨ Model discovery examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Listing the full catalog and grouping it by model type")
    print("   - Filtering the catalog server-side with models.list(type=...)")
    print("   - Reading typed specs: capabilities, constraints, voices, dimensions")
    print("   - Pricing in the unit each model type is billed in")
    print("   - Why resolve_*() beats taking the first catalog entry")
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
