#!/usr/bin/env python3
"""
Venice AI SDK - Image Quality Control
=====================================

Quality-aware image models accept a native ``quality`` parameter (``"low"`` /
``"medium"`` / ``"high"``) that trades render time and cost against fidelity.
This example:

1. **Selects** the cheapest model that offers the ``high`` tier with
   ``client.models.resolve_image(require_quality="high", prefer="cheapest")``,
   then reads its ``ImageModelConstraints.qualities`` and ``defaultQuality``.
   No model ID is hardcoded.
2. **Quotes** every tier from the model's catalog pricing, then **generates**
   the ``low`` and ``high`` tiers with ``client.image.create(..., quality=...)``
   using the same prompt, seed, resolution and aspect ratio. Each render must
   come back in the requested aspect ratio, and the request the server echoes
   in ``response.request["data"]`` must carry the tier that was sent. That
   echo is the only check in code that the tier reached the model. Comparing
   the two renders' pixels cannot show the tier's effect: a model that does
   not reproduce an image from its seed returns a different picture on every
   call, at the same tier or not. Nothing in the response measures fidelity
   either, so the example prints each render's size and time and leaves the
   visual comparison to you.
3. **Edits** with explicit output controls: ``edit(output_format=...)`` on the
   cheapest edit model, and ``multi_edit(quality=..., aspect_ratio=...)`` on
   the cheapest edit model selected with
   ``resolve_inpaint(require_quality=..., prefer="cheapest")``. It checks that
   the returned format and aspect ratio match the request. Edits return bare
   image bytes with no request echo, so whether ``quality`` was applied to an
   edit cannot be read from the response. ``multi_edit()`` accepts
   ``output_format`` too, but not every edit model honours it there, so check
   the returned bytes with ``detect_image_format`` rather than trusting the
   extension you asked for.

Around each tier render and the ``multi_edit()`` call, the example reads what
the API key can still spend (``GET /api_keys/rate_limits``, free) and prints
the drop next to the catalog price. Once the call's charge has posted, and
provided nothing is credited to the key in that window, the drop is at least
what the call was billed: other spending on the same key adds to it. So it is
printed as a measurement, not checked, and it can only rule a price out. A
drop below a tier's price shows that tier was not billed, but no drop shows
which cheaper tier was. If the balance cannot be read, the measurement is
reported as unavailable and the demo is still judged on its output.

Each run makes five paid generations or edits. If no catalog model offers the
needed quality tiers, the example prints a ``SKIPPED:`` line and exits with
code 77 before making any paid call. Outputs are written to ``examples/results/``.

See ``models/model_lifecycle.py`` for the read-only metadata view.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Literal

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError, detect_image_format
from venice_ai.types.api import ImageModelSpec, InpaintModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    SKIPPED,
    catalog_price_usd,
    format_price,
    generate_base_image,
    image_dimensions,
    matches_aspect_ratio,
    save_image,
    spendable_usd,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

PROMPT = "A single ripe pomegranate on a marble countertop, soft window light"
SEED = 20260928

#: The two tiers rendered: the cheapest and the most expensive one.
Tier = Literal["low", "high"]
RENDERED_TIERS: tuple[Tier, ...] = ("low", "high")

# High-quality renders are slow and can exceed the SDK's default 120s request
# timeout, so the tier renders pass a wider per-call ``timeout``.
QUALITY_RENDER_TIMEOUT = 300.0


class NoModel(Exception):
    """No catalog model offers what a demo needs."""


async def _spendable_or_none(client: VeniceClient) -> float | None:
    """What the key can still spend, or ``None`` with the reason printed.

    The balance is only a measurement here; a failed read must not fail a
    render that succeeded, so it is reported as unavailable instead.
    """
    try:
        return await spendable_usd(client)
    except VeniceError as e:
        print(f"      ⚠️ Billed amount unavailable: the key's balance could not be read ({e})")
        return None


def _drop(before: float | None, after: float | None) -> float | None:
    return before - after if before is not None and after is not None else None


async def quality_tiers(client: VeniceClient, written: list[Path], measured: list[str]) -> bool:
    """Render the ``low`` and ``high`` quality tiers on a model that offers ``high``.

    Every tier is quoted; only ``low`` and ``high`` are rendered. Returns ``True``
    only if both renders succeeded, came back in the requested aspect ratio and
    echoed the requested tier. Appends each tier whose billed amount was
    measured to ``measured``.

    :raises NoModel: If no image model offers the ``high`` tier.
    """
    print("🎨 Image quality tiers")
    print("-" * 40)

    try:
        model = await client.models.resolve_image(require_quality="high", prefer="cheapest")
    except NoMatchingModelError as e:
        raise NoModel(f"no image model offers the 'high' quality tier: {e}") from e

    spec = (await client.models.get(model)).model_spec
    if not isinstance(spec, ImageModelSpec) or not spec.constraints:
        print(f"   ❌ {model} has no image constraints in the catalog")
        return False

    constraints = spec.constraints
    tiers = list(constraints.qualities or [])
    extra = constraints.model_extra or {}
    ratio = extra.get("defaultAspectRatio") or next(iter(extra.get("aspectRatios") or []), None)
    resolution = constraints.defaultResolution
    print(f"   Quality-aware model: {model}")
    print(f"   Supported tiers: {', '.join(tiers)} (default: {constraints.defaultQuality})")
    print(f"   Requesting resolution={resolution}, aspect_ratio={ratio}")

    prices = {tier: catalog_price_usd(spec, resolution=resolution, quality=tier) for tier in tiers}
    for tier, price in prices.items():
        shown = f"${price:.2f}" if price is not None else "not listed"
        print(f"   💰 quality='{tier}': {shown}")
    rendered: list[Tier] = [tier for tier in RENDERED_TIERS if tier in tiers]
    if len(rendered) < len(RENDERED_TIERS):
        raise NoModel(f"{model} does not list both 'low' and 'high' tiers")
    known = [price for tier in rendered if (price := prices[tier]) is not None]
    if known:
        print(f"   💰 Estimated total to render {', '.join(rendered)}: ${sum(known):.2f}\n")

    ok = True
    for tier in rendered:
        print(f"   ⏳ Rendering quality='{tier}' …")
        before = await _spendable_or_none(client)
        started = time.monotonic()
        try:
            response = await client.image.create(
                model=model,
                prompt=PROMPT,
                quality=tier,
                resolution=resolution,
                aspect_ratio=ratio,
                seed=SEED,
                hide_watermark=True,
                return_binary=False,
                timeout=QUALITY_RENDER_TIMEOUT,
            )
        except VeniceError as e:
            print(f"      ❌ quality='{tier}' failed: {e}")
            ok = False
            continue
        elapsed = time.monotonic() - started
        spent = _drop(before, await _spendable_or_none(client))
        data = response.bytes(0)
        width, height = image_dimensions(data)
        path = response.save(RESULTS_DIR / f"quality_{tier}", overwrite=True)
        written.append(path)
        print(f"      ✅ {path.name}: {width}x{height}, {len(data)} bytes, {elapsed:.1f}s")
        if spent is not None:
            measured.append(tier)
            print(
                f"      💳 The key's spendable USD fell ${spent:.4f} "
                f"(catalog {format_price(prices[tier])} for quality='{tier}')"
            )

        echo = response.request.get("data") if isinstance(response.request, dict) else None
        echoed = echo.get("quality") if isinstance(echo, dict) else None
        if echoed != tier:
            print(f"      ❌ The server echoed quality={echoed!r}, not {tier!r}")
            ok = False
        else:
            print(f"      ✔ The server echoed quality={echoed!r}")
        if ratio and not matches_aspect_ratio(width, height, ratio):
            print(f"      ❌ {width}x{height} is not the requested aspect ratio {ratio}")
            ok = False
    return ok


def _pick_non_square_ratio(spec: InpaintModelSpec) -> str | None:
    """First landscape aspect ratio the edit model lists, if any."""
    ratios = (spec.constraints.model_extra or {}).get("aspectRatios") if spec.constraints else None
    for ratio in ratios or []:
        w, sep, h = str(ratio).partition(":")
        if sep and w.isdigit() and h.isdigit() and int(w) > int(h):
            return str(ratio)
    return None


def _report_multi_edit_charge(
    spent: float, tier_prices: dict[str, float | None], default_quality: str | None
) -> None:
    """Print what the balance drop around ``multi_edit()`` does and does not show.

    The edit response is bare image bytes with no request echo, so the drop
    is the only evidence of the tier billed, and it is one-sided: other
    spending on the key can only raise it (a credit to the key in the same
    window would lower it, which the example assumes does not happen). A tier priced above the drop was
    therefore not billed for this call. Every tier priced at or below the drop
    stays possible, and the drop cannot tell them apart.
    """
    listed = ", ".join(f"{tier} {format_price(price)}" for tier, price in tier_prices.items())
    print(f"      💳 The key's spendable USD fell ${spent:.4f} (catalog: {listed})")
    priced = {tier: price for tier, price in tier_prices.items() if price is not None}
    possible = [tier for tier, price in priced.items() if price <= spent]
    ruled_out = [tier for tier, price in priced.items() if price > spent]
    if not priced:
        print("      ℹ️ The catalog prices no tier, so the drop cannot be compared")
    elif not possible:
        print(
            "      ℹ️ The drop is below every tier's price: the charge had not posted by the"
            " second reading, so it shows nothing about the tier"
        )
    elif ruled_out:
        default_note = (
            f" That includes the default {default_quality!r} tier, so quality='low' was not"
            " ignored in favour of the default."
            if default_quality in ruled_out
            else ""
        )
        print(
            f"      ✔ Not billed at {', '.join(repr(t) for t in ruled_out)}: each costs more than"
            f" the whole drop.{default_note}"
        )
        if len(possible) == 1:
            print(f"      ℹ️ Of the listed tiers, only {possible[0]!r} is priced within the drop")
        else:
            print(
                f"      ℹ️ Still consistent with the drop: {', '.join(repr(t) for t in possible)};"
                " the drop cannot say which of these was billed"
            )
    else:
        print(
            "      ℹ️ The drop covers every tier's price (other spending on the key may be"
            " in it), so it rules no tier out"
        )


async def edit_output_controls(
    client: VeniceClient, written: list[Path], measured: list[str]
) -> bool:
    """Demonstrate ``output_format``, ``quality`` and ``aspect_ratio`` on edits.

    Returns ``True`` only if both edits succeeded and the returned images match
    the requested format and shape. Appends ``"multi_edit"`` to ``measured``
    when the ``multi_edit()`` call's billed amount was read.
    """
    print("\n✂️  Edit output controls (output_format / quality / aspect_ratio)")
    print("-" * 40)

    try:
        edit_model = await client.models.resolve_inpaint(prefer="cheapest")
        multi_model = await client.models.resolve_inpaint(require_quality="low", prefer="cheapest")
    except NoMatchingModelError as e:
        raise NoModel(f"no suitable edit model is available: {e}") from e

    multi_spec = (await client.models.get(multi_model)).model_spec
    if not isinstance(multi_spec, InpaintModelSpec) or multi_spec.constraints is None:
        print(f"   ❌ {multi_model} has no inpaint constraints in the catalog")
        return False
    ratio = _pick_non_square_ratio(multi_spec)
    if ratio is None:
        print(f"   ❌ {multi_model} lists no landscape aspect ratio")
        return False

    edit_price = catalog_price_usd((await client.models.get(edit_model)).model_spec)
    multi_qualities = list(multi_spec.constraints.qualities or [])
    tier_prices = {tier: catalog_price_usd(multi_spec, quality=tier) for tier in multi_qualities}
    multi_price = catalog_price_usd(multi_spec, quality="low")
    default_quality = multi_spec.constraints.defaultQuality
    default_price = catalog_price_usd(multi_spec, quality=default_quality)
    print(f"   edit() model: {edit_model} (catalog {format_price(edit_price)})")
    print(
        f"   multi_edit() model: {multi_model} (qualities {multi_qualities}, "
        f"catalog {format_price(multi_price)} at quality='low', "
        f"{format_price(default_price)} at its default {default_quality!r})"
    )

    try:
        try:
            base_bytes = await generate_base_image(client, "A plain ceramic bowl on a wooden table")
        except NoMatchingModelError as e:
            raise NoModel(f"no image model takes width/height for the source image: {e}") from e
        base_w, base_h = image_dimensions(base_bytes)
        print(f"   Source image: {base_w}x{base_h}")

        # Edits come back as PNG when no format is requested, so ask for WebP:
        # a PNG response would mean the parameter was ignored.
        print("   ⏳ edit() with output_format='webp' …")
        edited = await client.image.edit(
            prompt="Fill the bowl with fresh strawberries",
            image=base_bytes,
            model=edit_model,
            output_format="webp",
        )
        out = save_image(RESULTS_DIR / "quality_edit_output", edited)
        written.append(out)
        fmt = detect_image_format(edited)[0]
        print(f"      Returned format: {fmt} → {out.name}")
        if fmt != "webp":
            print("      ❌ output_format='webp' was not honoured")
            return False

        print(f"   ⏳ multi_edit() with quality='low', aspect_ratio='{ratio}' …")
        before = await _spendable_or_none(client)
        multi = await client.image.multi_edit(
            prompt="Add soft morning light from the left",
            image=base_bytes,
            model=multi_model,
            quality="low",
            aspect_ratio=ratio,
        )
        spent = _drop(before, await _spendable_or_none(client))
        m_out = save_image(RESULTS_DIR / "quality_multi_edit_output", multi)
        written.append(m_out)
        m_fmt = detect_image_format(multi)[0]
        m_w, m_h = image_dimensions(multi)
        print(f"      Returned format: {m_fmt}, size {m_w}x{m_h} → {m_out.name}")
        if not matches_aspect_ratio(m_w, m_h, ratio):
            print(f"      ❌ {m_w}x{m_h} is not the requested aspect ratio {ratio}")
            return False
        if spent is not None:
            measured.append("multi_edit")
            _report_multi_edit_charge(spent, tier_prices, default_quality)
    except VeniceError as e:
        print(f"   ❌ Error in edit output controls: {e}")
        return False

    print("   ✅ Output format and aspect ratio match the requests")
    return True


async def main() -> int:
    """Run the example.

    Returns ``0`` if every demo that ran succeeded, ``1`` if any failed, and
    ``77``, before any paid call, if no catalog model offers the quality tiers.
    """
    print("🚀 Venice AI Image Quality Control")
    print("=" * 50)

    written: list[Path] = []
    measured: list[str] = []
    results: list[tuple[str, bool]] = []
    skipped: dict[str, str] = {}
    async with VeniceClient() as client:
        for name, demo in [
            ("quality_tiers", quality_tiers),
            ("edit_output_controls", edit_output_controls),
        ]:
            try:
                results.append((name, await demo(client, written, measured)))
            except NoModel as e:
                # The quality tiers are this example's core feature: without
                # them it skips before the paid edit demo runs.
                if name == "quality_tiers":
                    print(f"\nSKIPPED: {e}")
                    return SKIPPED
                skipped[name] = str(e)
                print(f"Section skipped: {name}: {e}")

    failed = [name for name, ok in results if not ok]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Image quality control example completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - resolve_image(require_quality=..., prefer='cheapest')")
        print("   - ImageModelConstraints.qualities and defaultQuality")
        print("   - client.image.create(..., quality=<tier>) with a per-call timeout")
        print("   - Quoting each tier from the catalog pricing matrix")
        if measured:
            print("   - Bounding the billed amount with the key's spendable balance")
        if "edit_output_controls" not in skipped:
            print("   - client.image.edit(..., output_format=...)")
            print("   - client.image.multi_edit(..., quality=..., aspect_ratio=...)")

    print("\n📁 Files written by this run:")
    for path in written:
        print(f"   - {path}")
    if not written:
        print("   (none)")

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
