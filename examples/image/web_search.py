#!/usr/bin/env python3
"""
Venice AI SDK - Image Generation with Web Search
================================================

Demonstrates ``client.image.create(enable_web_search=True)``, an optional body
field on ``POST /image/generate`` that lets the image model pull in recent
web-search context before generating. Only image models whose spec-level
``model_spec.supportsWebSearch`` flag is true honour it; on any other model the
flag does nothing.

The example selects the cheapest capable model with
``client.models.resolve_image(require_web_search=True, prefer="cheapest")``,
prints the catalog price, then renders a prompt about a current event with
web search on. It passes ``resolution="1K"`` explicitly, so the catalog price
it quotes is the image's base price; web search adds its own charge on top
(see below).

After the render the example reads the request the server echoes back in
``response.request["data"]`` and fails unless its ``enable_web_search`` value
is ``True``. That proves the server received the flag. It does not prove a
search ran: the response carries no signal of that. Whether web search shaped
the content is judged by looking at the image, which can show details from
current coverage of the subject, though a model that already learned them may
draw them either way.

The ``/image/generate`` reference says additional credits are charged when a
web search is used, on top of the catalog image price, without giving the
amount. To show what the render cost, the example takes ``response.balance_info.usd``
(what the API key could spend before the request) and subtracts what it can
spend afterwards (``GET /api_keys/rate_limits``, free). It prints that drop and
how far it exceeds the catalog image price. Other spending on the same key in
that window adds to the drop, so it is printed as a measurement, not checked.
If the balance cannot be read after the render, the example says the cost is
unavailable and why; the render itself still counts as a success.

Each run makes one paid generation. The output is written to
``examples/results/``.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import ImageGenerationResponse, ImageModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import SKIPPED, catalog_price_usd, image_dimensions, spendable_usd  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

RESOLUTION = "1K"

PROMPT = (
    "A photorealistic image of the latest NASA Artemis lunar mission "
    "with astronauts on the lunar surface"
)


class NoWebSearchModel(Exception):
    """No catalog image model supports web search."""


async def generate_with_web_search(client: VeniceClient) -> list[Path] | None:
    """Render the prompt with web search on.

    Returns the files written, or ``None`` if the render failed or the
    server's request echo did not confirm ``enable_web_search=True``.

    :raises NoWebSearchModel: If no image model supports web search.
    """
    print("🌐 Image Generation — enable_web_search")
    print("-" * 40)

    try:
        model = await client.models.resolve_image(require_web_search=True, prefer="cheapest")
    except NoMatchingModelError as e:
        raise NoWebSearchModel(str(e)) from e

    spec = (await client.models.get(model)).model_spec
    print(f"📍 Using model: {model}")
    if isinstance(spec, ImageModelSpec):
        print(f"   supportsWebSearch: {spec.supportsWebSearch}")
    price = catalog_price_usd(spec, resolution=RESOLUTION)
    if price is not None:
        print(f"💰 Catalog price at {RESOLUTION}: ${price:.2f}, plus the web-search")
        print("   surcharge the server adds when a search runs")

    print("\n🎨 Generating with web search …")
    try:
        response: ImageGenerationResponse = await client.image.create(
            model=model,
            prompt=PROMPT,
            resolution=RESOLUTION,
            hide_watermark=True,
            enable_web_search=True,
            return_binary=False,
        )
    except VeniceError as e:
        print(f"   ❌ Generation failed: {e}")
        return None

    if not response.images:
        print("   ❌ The response contained no image")
        return None

    echo = response.request.get("data") if isinstance(response.request, dict) else None
    if not isinstance(echo, dict) or "enable_web_search" not in echo:
        print("   ❌ The response did not echo enable_web_search, so receipt is unconfirmed")
        return None
    if echo["enable_web_search"] is not True:
        print(
            f"   ❌ Sent enable_web_search=True, but the server echoed "
            f"{echo['enable_web_search']!r}"
        )
        return None
    print("   ✔ Server echoed enable_web_search=True (flag received; not proof a search ran)")

    before = response.balance_info.usd if response.balance_info else None
    if before is None:
        print("   ℹ️ The response carried no balance header, so the cost is not shown")
    else:
        # The balance is only a measurement: a failed read must not fail a
        # render that succeeded, so it is reported as unavailable instead.
        try:
            after = await spendable_usd(client)
        except VeniceError as e:
            print(
                f"   ⚠️ Cost unavailable: the key's balance could not be read after the render ({e})"
            )
        else:
            spent = before - after
            print(f"   💳 The key's spendable USD fell ${spent:.4f} across this render")
            if price is not None:
                print(f"      ${spent - price:.4f} above the catalog image price of ${price:.2f}")

    data = response.bytes(0)
    width, height = image_dimensions(data)
    path = response.save(RESULTS_DIR / "image_web_search_on", overwrite=True)
    print(f"   💾 Saved: {path} ({width}x{height}, {len(data)} bytes)")

    print("\n✅ The render was generated, and the server echoed enable_web_search=True.")
    return [path]


async def main() -> int:
    """Run the example. Returns ``0`` on success, ``1`` on failure, ``77`` if skipped."""
    print("🚀 Venice AI Image — enable_web_search Example")
    print("=" * 50)

    async with VeniceClient() as client:
        try:
            written = await generate_with_web_search(client)
        except NoWebSearchModel as e:
            print(f"SKIPPED: no image model in the catalog supports web search ({e})")
            return SKIPPED

    if written is None:
        print("\n⚠️ The web search demo failed.")
        return 1

    print("\n✨ Done.")
    print("\n📁 Files written by this run:")
    for path in written:
        print(f"   - {path}")
    print("\n💡 Look at the image. A web-search render can draw on current public")
    print("   imagery for the subject, beyond what the model learned in training.")
    print("   The request echo shows the server received the flag; the API")
    print("   reports no search result, so this run cannot confirm that a search")
    print("   took place.")
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
