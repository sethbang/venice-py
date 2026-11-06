#!/usr/bin/env python3
"""
Venice AI SDK - Batch Image Generation
=======================================

This example demonstrates efficient batch generation of multiple images.
Learn how to generate multiple images concurrently and manage batch workflows.

The API can also return several images from one request: ``num_images`` (up to 4,
sent as ``variants``) with ``response.save_all()``. That form needs a JSON
response, so ``return_binary=True`` is shown separately on a single image, and the
two are never combined. The catalog has no field for which models accept
``num_images`` above 1, so a model that refuses it skips that part.

Each returned image's pixel size is read from its header and checked against
the request. The concurrent demo counts requests in flight to show that
``client.gather(max_concurrency=N)`` holds the cap, and the seed demo renders
one seed twice and a second seed once to show that the seed reproduces an
image and that a new seed gives a new one (comparing pixels needs Pillow).

Image generation is billed per request, so the client's default retry policy
never resends one that the server may have processed. It retries a failed
connection and a 503 meaning "model at capacity"; a 429 or any other error is
recorded as that image's failure and the rest of the batch carries on.

Generation costs real money, so by default this example only resolves the model
(free), prints what it would run and exits 77 ("skipped"). Opt in to make the
16 paid images, about $0.16 at the cheapest listed $0.01 per image, and write the
outputs to ``examples/results/``::

    VENICE_RUN_PAID_IMAGES=1
"""

import asyncio
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any

from _helpers import (
    SKIPPED,
    catalog_price_usd,
    format_price,
    image_dimensions,
    mean_pixel_difference,
)

from venice_ai import (
    InvalidRequestError,
    NoMatchingModelError,
    VeniceClient,
    VeniceError,
    detect_image_format,
)
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Seed check thresholds, on the mean absolute RGB difference (0-255). Renders of
# two seeds must differ by at least DIFFERENT_SEED_MIN_DIFF. Two renders of one
# seed must differ by at most SAME_SEED_MAX_RATIO of that, which allows the small
# numeric noise some GPU pipelines add while still requiring the same picture.
DIFFERENT_SEED_MIN_DIFF = 10.0
SAME_SEED_MAX_RATIO = 0.10

RUN_PAID = os.environ.get("VENICE_RUN_PAID_IMAGES") == "1"

# Paid generations each demo makes.
PAID_GENERATIONS = {
    "simple_batch_generation": 2,
    "concurrent_batch_generation": 3,
    "batch_with_variations": 3,
    "progressive_batch_generation": 3,
    "batch_with_mixed_parameters": 2,
    "native_batching": 3,
}

#: Images asked for in the one native batch request.
NATIVE_BATCH_SIZE = 2


class InFlight:
    """Count requests in flight and remember the peak."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0

    def __enter__(self) -> "InFlight":
        self.current += 1
        self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *exc: object) -> None:
        self.current -= 1


async def generate_one(
    client: VeniceClient,
    model: str,
    prompt: str,
    stem: str,
    written: list[Path],
    *,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
    in_flight: InFlight | None = None,
) -> dict[str, Any]:
    """Generate and save one image, returning a result record.

    API errors are captured in the record (``success=False``) so one failed
    image does not abort the rest of the batch. A size mismatch between the
    request and the returned image also counts as a failure.
    """
    record: dict[str, Any] = {"prompt": prompt, "stem": stem, "success": False}
    try:
        with in_flight or InFlight():
            response: ImageGenerationResponse = await client.image.create(
                model=model,
                prompt=prompt,
                width=width,
                height=height,
                num_images=1,
                seed=seed,
                hide_watermark=True,
                return_binary=False,
            )
    except VeniceError as e:
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    data = response.bytes(0)
    size = image_dimensions(data)
    path = response.save(RESULTS_DIR / stem, overwrite=True)
    written.append(path)
    record.update(
        path=path,
        data=data,
        bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        size=size,
        timing=response.timing.inferenceDuration if response.timing else None,
    )
    if size != (width, height):
        record["error"] = f"requested {width}x{height}, received {size[0]}x{size[1]}"
        return record
    record["success"] = True
    return record


def report(results: list[dict[str, Any] | BaseException]) -> int:
    """Print one line per result and return the number of failures.

    ``client.gather`` returns exceptions in their result slot, so anything that
    is not a successful record (an exception or a failed record) is a failure.
    """
    failures = 0
    for result in results:
        if isinstance(result, BaseException):
            failures += 1
            print(f"   ❌ {type(result).__name__}: {result}")
        elif result["success"]:
            w, h = result["size"]
            timing = f", {result['timing']}ms" if result["timing"] else ""
            print(f"   ✅ {result['path'].name}: {w}x{h}, {result['bytes']} bytes{timing}")
        else:
            failures += 1
            print(f"   ❌ {result['stem']} ('{result['prompt'][:40]}'): {result['error']}")
    return failures


async def simple_batch_generation(client: VeniceClient, model: str, written: list[Path]) -> bool:
    """Generate several prompts one after another."""
    print("📦 Simple Batch Generation")
    print("-" * 40)

    prompts = [
        "A futuristic city with flying cars",
        "A cozy coffee shop interior",
    ]

    print(f"🎨 Generating {len(prompts)} images sequentially...")
    start = time.monotonic()
    results: list[dict[str, Any] | BaseException] = []
    for i, prompt in enumerate(prompts, 1):
        results.append(await generate_one(client, model, prompt, f"batch_seq_{i}", written))
    elapsed = time.monotonic() - start

    failures = report(results)
    print(f"⏱️ Total time: {elapsed:.2f}s ({elapsed / len(prompts):.2f}s per image)")
    return failures == 0


async def concurrent_batch_generation(
    client: VeniceClient, model: str, written: list[Path]
) -> bool:
    """Generate several prompts concurrently with a concurrency cap."""
    print("\n⚡ Concurrent Batch Generation")
    print("-" * 40)

    prompts = [
        "A cyberpunk street scene at night",
        "A serene Japanese zen garden",
        "A steampunk airship in the clouds",
    ]

    max_concurrency = 2
    print(f"🚀 Generating {len(prompts)} images, at most {max_concurrency} in flight...")
    in_flight = InFlight()
    start = time.monotonic()
    results = await client.gather(
        [
            generate_one(
                client, model, prompt, f"batch_concurrent_{i}", written, in_flight=in_flight
            )
            for i, prompt in enumerate(prompts, 1)
        ],
        max_concurrency=max_concurrency,
    )
    elapsed = time.monotonic() - start

    failures = report(results)
    print(f"⏱️ Total time: {elapsed:.2f}s ({elapsed / len(prompts):.2f}s per image)")
    print(f"✅ Successful: {len(prompts) - failures}/{len(prompts)}")
    print(f"🔢 Peak requests in flight: {in_flight.peak} (cap {max_concurrency})")
    if in_flight.peak != max_concurrency:
        print(f"   ❌ Expected the peak to reach the cap of {max_concurrency}")
        return False
    return failures == 0


async def batch_with_variations(
    client: VeniceClient, model: str, written: list[Path]
) -> bool | None:
    """Show that the seed controls the image.

    Renders the prompt with seed A twice and with seed B once. The seed-B
    render must clearly differ from seed A (``DIFFERENT_SEED_MIN_DIFF``), and
    the two seed-A renders must match: identical, or differing by no more than
    ``SAME_SEED_MAX_RATIO`` of the seed-A/seed-B difference.
    Distinct images alone would prove nothing: a server that ignored the seed
    and sampled randomly would return distinct images too.

    Returns ``True`` if the renders succeeded and, when Pillow is installed to
    compare pixels, both checks passed. Without Pillow the seed checks cannot
    run: the demo prints ``Section skipped:`` after the renders and returns
    ``None``.
    """
    print("\n🎲 Seed Reproducibility and Variation")
    print("-" * 40)

    base_prompt = "A majestic dragon perched on a mountain peak"
    seed_a, seed_b = 42, 123
    runs = [(seed_a, "a"), (seed_a, "b"), (seed_b, "a")]
    print(f"🎨 Rendering '{base_prompt}' with seed {seed_a} twice and seed {seed_b} once")

    results = await client.gather(
        [
            generate_one(
                client, model, base_prompt, f"variation_seed{seed}_{tag}", written, seed=seed
            )
            for seed, tag in runs
        ],
        max_concurrency=3,
    )
    if report(results) > 0:
        return False
    first, repeat, other = (r["data"] for r in results if isinstance(r, dict))

    same = mean_pixel_difference(first, repeat)
    different = mean_pixel_difference(first, other)
    if same is None or different is None:
        print(
            "Section skipped: the seed checks compare pixels and need Pillow (pip install Pillow)"
        )
        return None

    print(f"   Seed {seed_a} vs seed {seed_a}: mean pixel difference {same:.2f}")
    print(f"   Seed {seed_a} vs seed {seed_b}: mean pixel difference {different:.2f}")
    ok = True
    limit = SAME_SEED_MAX_RATIO * different
    if same > limit:
        print(f"   ❌ Seed {seed_a} did not reproduce its image (limit {limit:.2f})")
        ok = False
    else:
        exact = "pixel for pixel" if same == 0 else f"within {limit:.2f}"
        print(f"   ✅ Seed {seed_a} reproduced its image ({exact})")
    if different < DIFFERENT_SEED_MIN_DIFF:
        print(f"   ❌ Seed {seed_b} did not change the image (needs {DIFFERENT_SEED_MIN_DIFF})")
        ok = False
    else:
        print(f"   ✅ Seed {seed_b} produced a different image")
    return ok


async def progressive_batch_generation(
    client: VeniceClient, model: str, written: list[Path]
) -> bool:
    """Generate prompts in fixed-size batches, reporting after each batch."""
    print("\n📈 Progressive Batch Generation")
    print("-" * 40)

    all_prompts = [
        "A red sports car",
        "A blue ocean wave",
        "A green forest path",
    ]
    batch_size = 2
    total_batches = (len(all_prompts) + batch_size - 1) // batch_size
    print(
        f"📦 Processing {len(all_prompts)} prompts in {total_batches} batches of up to {batch_size}"
    )

    failures = 0
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        batch_prompts = all_prompts[start_idx : start_idx + batch_size]
        print(f"\n🔄 Batch {batch_num + 1}/{total_batches}")

        batch_results = await client.gather(
            [
                generate_one(client, model, prompt, f"progressive_{start_idx + i}", written)
                for i, prompt in enumerate(batch_prompts)
            ],
            max_concurrency=batch_size,
        )
        batch_failures = report(batch_results)
        failures += batch_failures
        print(f"   Completed: {len(batch_results) - batch_failures}/{len(batch_results)}")

    print(f"\n📊 Final: {len(all_prompts) - failures}/{len(all_prompts)} succeeded")
    return failures == 0


async def batch_with_mixed_parameters(
    client: VeniceClient, model: str, written: list[Path]
) -> bool:
    """Generate a batch where each image has its own size."""
    print("\n⚙️ Batch with Mixed Parameters")
    print("-" * 40)

    configs = [
        ("A portrait photograph", 512, 768, "portrait"),
        ("A panoramic view", 768, 512, "landscape"),
    ]
    print(f"🎨 Generating {len(configs)} images with different sizes (measured below)...")

    results: list[dict[str, Any] | BaseException] = []
    for prompt, width, height, name in configs:
        results.append(
            await generate_one(
                client, model, prompt, f"mixed_{name}", written, width=width, height=height
            )
        )
    return report(results) == 0


async def native_batching(client: VeniceClient, model: str, written: list[Path]) -> bool | None:
    """One request for several images, then one request for raw bytes.

    A reply can hold fewer images than asked for: moderation can drop some, and a
    model can return one whatever was asked. That is reported and accepted. A reply
    with none is a failure.
    Returns ``None`` if the model refuses ``num_images`` above 1.
    """
    print("\n🧺 Native Batching: num_images, save_all and return_binary")
    print("-" * 40)

    try:
        response: ImageGenerationResponse = await client.image.create(
            model=model,
            prompt="A lighthouse on a rocky coast at dawn",
            width=512,
            height=512,
            num_images=NATIVE_BATCH_SIZE,
            hide_watermark=True,
            return_binary=False,
        )
    except InvalidRequestError as e:
        print(f"Section skipped: native batching: {model} refused num_images: {e}")
        return None
    except VeniceError as e:
        print(f"   ❌ {type(e).__name__}: {e}")
        return False

    got = len(response.images)
    if got == 0:
        print("   ❌ The reply holds no images")
        return False
    if got < NATIVE_BATCH_SIZE:
        print(f"   ℹ️ {got} of {NATIVE_BATCH_SIZE} images came back: moderated, or not supported")
    paths = response.save_all(RESULTS_DIR, prefix="batch_native", overwrite=True)
    written.extend(paths)
    sizes = [image_dimensions(path.read_bytes()) for path in paths]
    print(f"   ✅ save_all() wrote {len(paths)} files: {sizes}")
    if len(paths) != got or any(size != (512, 512) for size in sizes):
        print("   ❌ The saved files do not match the reply or the 512x512 request")
        return False

    try:
        data = await client.image.create(
            model=model,
            prompt="A lighthouse on a rocky coast at dawn",
            width=512,
            height=512,
            format="png",
            hide_watermark=True,
            return_binary=True,
        )
    except VeniceError as e:
        print(f"   ❌ {type(e).__name__}: {e}")
        return False
    binary = RESULTS_DIR / "batch_binary.png"
    binary.write_bytes(data)
    written.append(binary)
    fmt = detect_image_format(data)[0]
    print(f"   ✅ return_binary=True: {len(data)} bytes of {fmt}")
    if fmt != "png":
        print("   ❌ Asked for format='png'")
        return False
    return True


async def main() -> int:
    """Run all batch generation examples.

    Returns ``0`` if every demo that ran succeeded (without Pillow the seed
    checks are skipped), ``1`` if any failed, and ``77`` if
    ``VENICE_RUN_PAID_IMAGES`` is unset or no catalog image model is sized by
    width/height.
    """
    print("🚀 Venice AI Batch Image Generation Examples")
    print("=" * 50)
    total_paid = sum(PAID_GENERATIONS.values())
    if RUN_PAID:
        print(f"💸 VENICE_RUN_PAID_IMAGES=1 — this run makes {total_paid} paid generations.")
    else:
        print("🧪 Preview only (default). Set VENICE_RUN_PAID_IMAGES=1 to generate images.")

    written: list[Path] = []
    async with VeniceClient() as client:
        # Resolve the model once; every demo below reuses it.
        # Pixel sizes are requested below, so pick among models sized by width/height.
        try:
            image_model = await client.models.resolve_image(
                prefer="cheapest", require_custom_size=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no image model in the catalog takes width/height ({e})")
            return SKIPPED
        print(f"📍 Using image model: {image_model}\n")

        if not RUN_PAID:
            spec = (await client.models.get(image_model)).model_spec
            print(f"🧾 This run would make {total_paid} paid generations:")
            for name, count in PAID_GENERATIONS.items():
                print(f"   - {name}: {count}")
            print(f"💰 Catalog price per image: {format_price(catalog_price_usd(spec))}")
            print("⏭️  Not generated: set VENICE_RUN_PAID_IMAGES=1 to generate (incurs this cost).")
            print("SKIPPED: VENICE_RUN_PAID_IMAGES is not set, so no image was generated")
            return SKIPPED

        results: list[tuple[str, bool | None]] = []
        for name, demo in [
            ("simple_batch_generation", simple_batch_generation),
            ("concurrent_batch_generation", concurrent_batch_generation),
            ("batch_with_variations", batch_with_variations),
            ("progressive_batch_generation", progressive_batch_generation),
            ("batch_with_mixed_parameters", batch_with_mixed_parameters),
            ("native_batching", native_batching),
        ]:
            results.append((name, await demo(client, image_model, written)))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    else:
        print("\n✨ Batch generation examples completed!")
        print("\n💡 Key concepts demonstrated:")
        print("   - Sequential batch generation")
        print("   - Concurrent generation with client.gather(max_concurrency=N)")
        if "batch_with_variations" not in skipped:
            print("   - Reproducing an image with a seed, and varying it with another")
        print("   - Checking the concurrency cap by counting requests in flight")
        print("   - Progressive batch processing")
        print("   - Mixed output sizes, checked against the returned images")
        print("   - Counting per-item failures, including exceptions returned by gather")
        if "native_batching" not in skipped:
            print("   - One request for several images, with num_images and save_all()")
            print("   - Raw image bytes with return_binary=True, on a single image")

    print(f"\n📁 Files written by this run ({len(written)}):")
    for path in written:
        print(f"   - {path.name}")
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
