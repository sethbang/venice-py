#!/usr/bin/env python3
"""
Venice AI SDK - Image Upscaling
================================

This example demonstrates how to upscale and enhance images using the Venice AI SDK.
Learn how to improve image quality and resolution with AI-powered upscaling.
"""

import asyncio
import base64
import sys
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format
from venice_ai.types.api import ImageGenerationResponse

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/image/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


async def setup_sample_image() -> str | None:
    """Generate a sample image for upscaling demonstrations.

    Returns the saved image path, or ``None`` if generation failed (so the
    caller can mark its demo as failed instead of silently skipping).
    """
    print("🎨 Generating sample image for upscaling...")

    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()

            # Generate a small test image with more detail for better enhancement demonstration
            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A detailed fantasy landscape with mountains, forest, lake, and castle at sunset, rich colors and intricate details",
                width=256,
                height=256,
                num_images=1,
                return_binary=False,
            )

            # Save the base image
            image_data = response.images[0]
            image_bytes = base64.b64decode(image_data)

            base_path = RESULTS_DIR / f"base_image_256.{detect_image_format(image_bytes)[0]}"
            with open(base_path, "wb") as f:
                f.write(image_bytes)

            print(f"✅ Generated base image: {base_path}")
            print(f"📏 Size: {len(image_bytes)} bytes")

            return str(base_path)

        except Exception as e:
            print(f"❌ Error generating sample image: {e}")
            return None


async def basic_upscaling() -> bool:
    """Demonstrate basic image upscaling.

    Returns ``True`` on success, ``False`` if the sample image or the upscale
    call failed.
    """
    print("\n🔍 Basic Image Upscaling")
    print("-" * 40)

    # First generate a sample image
    base_image_path = await setup_sample_image()
    if not base_image_path:
        return False

    async with VeniceClient() as client:
        try:
            print("\n📤 Upscaling image by 2x...")

            # Upscale the image. The `timeout` parameter (a float in seconds, or
            # an aiohttp.ClientTimeout) caps the request — large source images
            # and high scale factors take longer, so a generous per-call timeout
            # prevents a premature client-side abort on a slow upscale. The same
            # parameter is available on image.edit() (see image_editing.py).
            upscaled_bytes = await client.image.upscale(
                image=base_image_path,
                scale=2,
                timeout=120.0,  # seconds; raise for very large images
            )

            if not isinstance(upscaled_bytes, bytes):
                print("❌ Upscale did not return image bytes")
                return False

            # Save upscaled image
            output_path = (
                RESULTS_DIR / f"upscaled_2x_basic.{detect_image_format(upscaled_bytes)[0]}"
            )
            with open(output_path, "wb") as f:
                f.write(upscaled_bytes)

            # Get original size
            with open(base_image_path, "rb") as f:
                original_bytes = f.read()

            print("✅ Upscaling complete!")
            print(f"📊 Original size: {len(original_bytes)} bytes")
            print(f"📊 Upscaled size: {len(upscaled_bytes)} bytes")
            print(f"📈 Size increase: {len(upscaled_bytes) / len(original_bytes):.2f}x")
            print(f"💾 Saved to: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error during upscaling: {e}")
            if "not supported" in str(e).lower():
                print("💡 Note: Image upscaling may not be available for your API tier")
            return False


async def upscaling_with_added_detail() -> bool:
    """Demonstrate upscaling with added detail via ``creativity``.

    Returns ``True`` on success, ``False`` if the sample image or the upscale
    call failed.
    """
    print("\n✨ Upscaling with Added Detail")
    print("-" * 40)

    # Generate a sample image
    base_image_path = await setup_sample_image()
    if not base_image_path:
        return False

    async with VeniceClient() as client:
        try:
            print("\n🎨 Upscaling with extra detail...")

            # `creativity` is the only tuning knob; the server clamps it to 0-0.02.
            enhanced_bytes = await client.image.upscale(
                image=base_image_path,
                scale=2,
                creativity=0.02,
            )

            if not isinstance(enhanced_bytes, bytes):
                print("❌ Enhanced upscale did not return image bytes")
                return False

            output_path = (
                RESULTS_DIR / f"upscaled_2x_enhanced.{detect_image_format(enhanced_bytes)[0]}"
            )
            with open(output_path, "wb") as f:
                f.write(enhanced_bytes)

            print("✅ Upscaling complete!")
            print(f"💾 Saved to: {output_path}")
            print(f"📏 Size: {len(enhanced_bytes)} bytes")
            print("🎨 Maximum creativity applied for extra detail")

            return True

        except Exception as e:
            print(f"❌ Error during enhancement: {e}")
            return False


async def creativity_sweep() -> bool:
    """Compare the upscaler's ``creativity`` settings side by side.

    Returns ``True`` only if every setting succeeded; ``False`` if the sample
    image failed or any individual upscale failed.
    """
    print("\n🎭 Creativity Sweep")
    print("-" * 40)

    # Generate a sample image
    base_image_path = await setup_sample_image()
    if not base_image_path:
        return False

    ok = True
    async with VeniceClient() as client:
        try:
            # Different enhancement styles with more distinct characteristics
            # The endpoint takes no style prompt — `creativity` is the whole
            # dial, and the server clamps it to 0-0.02.
            enhancements = [
                {"creativity": 0.0, "name": "faithful"},
                {"creativity": 0.01, "name": "default"},
                {"creativity": 0.02, "name": "detailed"},
            ]

            for enhancement in enhancements:
                print(f"\n🎨 Setting: {enhancement['name']}")
                print(f"   Creativity: {enhancement['creativity']}")

                try:
                    enhanced_bytes = await client.image.upscale(
                        image=base_image_path,
                        scale=2,
                        creativity=enhancement["creativity"],
                    )

                    if not isinstance(enhanced_bytes, bytes):
                        print("   ❌ Enhancement did not return image bytes")
                        ok = False
                        continue

                    output_path = (
                        RESULTS_DIR
                        / f"upscaled_{enhancement['name']}.{detect_image_format(enhanced_bytes)[0]}"
                    )
                    with open(output_path, "wb") as f:
                        f.write(enhanced_bytes)

                    print(f"   ✅ Saved to: {output_path}")
                    print(f"   📏 Size: {len(enhanced_bytes)} bytes")

                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error during creative enhancement: {e}")
            ok = False

    return ok


async def batch_upscaling() -> bool:
    """Demonstrate batch upscaling of multiple images.

    Returns ``True`` only if every image was generated and upscaled; ``False``
    if any generation or upscale step failed.
    """
    print("\n📦 Batch Upscaling")
    print("-" * 40)

    ok = True
    async with VeniceClient() as client:
        try:
            # Generate multiple small images
            prompts = [
                "A red apple",
                "A blue car",
                "A green tree",
            ]

            print(f"🎨 Generating {len(prompts)} test images...")
            image_paths = []

            # Get image model
            image_model = await client.models.resolve_image()

            for i, prompt in enumerate(prompts):
                try:
                    response: ImageGenerationResponse = await client.image.create(
                        model=image_model,
                        prompt=prompt,
                        width=256,
                        height=256,
                        num_images=1,
                        return_binary=False,
                    )

                    image_data = response.images[0]
                    image_bytes = base64.b64decode(image_data)

                    path = RESULTS_DIR / f"batch_base_{i}.{detect_image_format(image_bytes)[0]}"
                    with open(path, "wb") as f:
                        f.write(image_bytes)

                    image_paths.append(path)
                    print(f"   ✓ Generated image {i + 1}")

                except Exception as e:
                    print(f"   ❌ Failed to generate image {i + 1}: {e}")
                    ok = False

            # Upscale all images
            print(f"\n🔍 Upscaling {len(image_paths)} images...")

            for i, path in enumerate(image_paths):
                try:
                    upscaled_bytes = await client.image.upscale(
                        image=path,
                        scale=2,
                    )

                    if not isinstance(upscaled_bytes, bytes):
                        print(f"   ❌ Image {i + 1} upscale did not return bytes")
                        ok = False
                        continue

                    output_path = (
                        RESULTS_DIR / f"batch_upscaled_{i}.{detect_image_format(upscaled_bytes)[0]}"
                    )
                    with open(output_path, "wb") as f:
                        f.write(upscaled_bytes)

                    print(f"   ✓ Upscaled image {i + 1} → {output_path}")

                except Exception as e:
                    print(f"   ❌ Failed to upscale image {i + 1}: {e}")
                    ok = False

            if ok:
                print("\n✅ Batch upscaling complete!")
            else:
                print("\n⚠️ Batch upscaling finished with failures")

        except Exception as e:
            print(f"❌ Error during batch upscaling: {e}")
            ok = False

    return ok


async def different_scale_factors() -> bool:
    """Demonstrate upscaling with different scale factors.

    Returns ``True`` only if every scale factor succeeded; ``False`` if the
    sample image failed or any scale factor failed.
    """
    print("\n📐 Different Scale Factors")
    print("-" * 40)

    # Generate a sample image
    base_image_path = await setup_sample_image()
    if not base_image_path:
        return False

    ok = True
    async with VeniceClient() as client:
        try:
            # Different scale factors
            scales = [1.5, 2.0, 3.0]

            # Get original size
            with open(base_image_path, "rb") as f:
                original_bytes = f.read()
            original_size = len(original_bytes)

            print(f"📊 Original image: {original_size} bytes")

            for scale in scales:
                print(f"\n🔍 Upscaling by {scale}x...")

                try:
                    upscaled_bytes = await client.image.upscale(
                        image=base_image_path,
                        scale=scale,
                    )

                    if not isinstance(upscaled_bytes, bytes):
                        print(f"   ❌ Scale {scale}x did not return image bytes")
                        ok = False
                        continue

                    output_path = (
                        RESULTS_DIR / f"upscaled_{scale}x.{detect_image_format(upscaled_bytes)[0]}"
                    )
                    with open(output_path, "wb") as f:
                        f.write(upscaled_bytes)

                    size_ratio = len(upscaled_bytes) / original_size
                    print(f"   ✅ Scale {scale}x complete")
                    print(f"   📊 New size: {len(upscaled_bytes)} bytes")
                    print(f"   📈 Size ratio: {size_ratio:.2f}x")
                    print(f"   💾 Saved to: {output_path}")

                except Exception as e:
                    print(f"   ❌ Failed at {scale}x: {e}")
                    ok = False

        except Exception as e:
            print(f"❌ Error testing scale factors: {e}")
            ok = False

    return ok


async def upscale_from_bytes() -> bool:
    """Demonstrate upscaling from in-memory image bytes.

    Returns ``True`` on success, ``False`` if generation or upscaling failed.
    """
    print("\n💾 Upscaling from Memory (Bytes)")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get image model
            image_model = await client.models.resolve_image()

            # Generate image and keep in memory
            print("🎨 Generating image in memory...")

            response: ImageGenerationResponse = await client.image.create(
                model=image_model,
                prompt="A simple geometric pattern",
                width=256,
                height=256,
                num_images=1,
                return_binary=False,
            )

            # Get bytes directly
            image_data = response.images[0]
            image_bytes = base64.b64decode(image_data)

            print(f"✅ Generated image: {len(image_bytes)} bytes")

            # Upscale directly from bytes
            print("\n🔍 Upscaling from memory...")

            upscaled_bytes = await client.image.upscale(
                image=image_bytes,  # Pass bytes directly
                scale=2,
            )

            if not isinstance(upscaled_bytes, bytes):
                print("❌ Upscale did not return image bytes")
                return False

            output_path = (
                RESULTS_DIR / f"upscaled_from_bytes.{detect_image_format(upscaled_bytes)[0]}"
            )
            with open(output_path, "wb") as f:
                f.write(upscaled_bytes)

            print("✅ Upscaled successfully!")
            print(f"📊 Original: {len(image_bytes)} bytes")
            print(f"📊 Upscaled: {len(upscaled_bytes)} bytes")
            print(f"💾 Saved to: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error upscaling from bytes: {e}")
            return False


async def main() -> int:
    """Run all image upscaling examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero process exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Image Upscaling Examples")
    print("=" * 50)

    results: list[tuple[str, bool]] = [
        ("basic_upscaling", await basic_upscaling()),
        ("upscaling_with_added_detail", await upscaling_with_added_detail()),
        ("creativity_sweep", await creativity_sweep()),
        ("batch_upscaling", await batch_upscaling()),
        ("different_scale_factors", await different_scale_factors()),
        ("upscale_from_bytes", await upscale_from_bytes()),
    ]

    failed = [name for name, ok in results if not ok]
    passed = len(results) - len(failed)

    if failed:
        print(f"\n⚠️ {passed}/{len(results)} demos completed; failed: {', '.join(failed)}")
    else:
        print(f"\n✨ Image upscaling examples completed! ({passed}/{len(results)})")

    print("\n💡 Key concepts demonstrated:")
    print("   - Basic image upscaling (2x, 3x)")
    print("   - AI-powered enhancement")
    print("   - Creative enhancement with prompts")
    print("   - Batch processing multiple images")
    print("   - Different scale factors")
    print("   - Upscaling from memory (bytes)")
    print("   - Quality comparison")

    # Extensions are auto-detected from the returned format (e.g. .webp), so
    # the listing below uses an <ext> placeholder rather than a hardcoded .png.
    print("\n📁 Generated files in examples/results/:")
    print("   - base_image_256.<ext> (original)")
    print("   - upscaled_*.<ext> (enhanced versions)")
    print("   - batch_*.<ext> (batch processing)")

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
