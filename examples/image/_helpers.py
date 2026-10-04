"""
Shared helpers for Venice AI image examples.

- ``generate_base_image`` produces a self-contained source image for the
  editing, upscaling and background-removal examples.
- ``save_image`` writes image bytes with the extension sniffed from their magic
  bytes (``venice_ai.detect_image_format``) and returns the path it wrote, so
  each example can list exactly the files its own run produced.
- ``image_dimensions`` reads the pixel size from a PNG, JPEG, WebP or GIF
  header, so examples can report and check what the server actually returned
  instead of echoing the requested size.
- ``alpha_coverage`` decodes a cutout and measures how much of it is see-through
  and how much is solid, not just whether the format declares an alpha channel.
- ``mean_pixel_difference`` compares two renders pixel by pixel. Both pixel
  checks need Pillow; ``pillow_available`` says whether it is installed.
- ``matches_aspect_ratio`` checks a returned size against a requested ratio.
- ``catalog_price_usd`` and ``format_price`` read and display a model's
  catalog price; ``spendable_usd`` reads what the API key can still spend, so
  an example can show how much a call actually cost.
- ``SKIPPED`` is the exit code an example returns when its core feature cannot
  run, for example because no catalog model offers it
  (``NoMatchingModelError``); it prints a line starting with ``SKIPPED:``
  first. A demo inside an example that cannot run prints
  ``Section skipped: <reason>`` instead, and the example still exits ``0`` if
  its core feature was validated.

The examples run under the client's default retry policy. Image generation,
editing, upscaling and background removal are billed per request, and that
policy never resends one after the server may have processed it: it retries a
failed connection and a 503 meaning "model at capacity", and surfaces every
other failure, including 429 (``RateLimitError``).
"""

import struct
from pathlib import Path

from venice_ai import VeniceClient, detect_image_format
from venice_ai.types.api import ImageGenerationResponse

#: Exit code for "skipped": a missing optional prerequisite, not a failure.
SKIPPED = 77


async def generate_base_image(
    client: "VeniceClient",
    prompt: str,
    *,
    model: str | None = None,
    width: int = 512,
    height: int = 512,
) -> bytes:
    """Generate a base image and return its raw bytes.

    Unless ``model`` is given, selects the cheapest image model sized by
    ``width`` / ``height`` (``require_custom_size=True``; models that list
    aspect ratios ignore pixel sizes). Generates a single image without the
    Venice watermark, so it is not carried into edits, cutouts or upscales, and
    returns the decoded bytes.

    :raises NoMatchingModelError: If no catalog model is sized by width/height.
    """
    image_model = model or await client.models.resolve_image(
        prefer="cheapest", require_custom_size=True
    )
    print(f"  📍 Using generation model: {image_model}")

    response: ImageGenerationResponse = await client.image.create(
        model=image_model,
        prompt=prompt,
        width=width,
        height=height,
        num_images=1,
        hide_watermark=True,
        return_binary=False,
    )
    return response.bytes(0)


def matches_aspect_ratio(width: int, height: int, ratio: str, tolerance: float = 0.03) -> bool:
    """Whether ``width`` x ``height`` has the aspect ratio ``ratio`` (e.g. ``"3:2"``).

    ``tolerance`` is the allowed relative error, since models snap each side to
    a size step (1216x832 is a common "3:2"). The default still separates 3:2
    from its neighbours 16:10 and 4:3.
    """
    w, sep, h = ratio.partition(":")
    if not (sep and w.isdigit() and h.isdigit() and int(h)):
        raise ValueError(f"Not an aspect ratio: {ratio!r}")
    target = int(w) / int(h)
    return abs(width / height - target) <= tolerance * target


def catalog_price_usd(
    spec: object,
    *,
    resolution: str | None = None,
    quality: str | None = None,
    input_images: int = 1,
) -> float | None:
    """Per-request USD price listed in a model's catalog ``pricing`` block.

    Handles the pricing shapes image and edit models use: a flat
    ``generation`` / ``inpaint`` price, per-``resolutions`` tiers, and a
    ``quality`` matrix keyed by resolution then tier. ``resolution`` defaults to
    the model's ``defaultResolution``. Edit models that bill extra input images
    (``inputImages``) have that surcharge added for ``input_images``. Returns
    ``None`` when the catalog lists no matching price.
    """
    pricing = getattr(spec, "pricing", None)
    if pricing is None:
        return None
    table = pricing.model_dump()
    constraints = getattr(spec, "constraints", None)
    tier = resolution or getattr(constraints, "defaultResolution", None)

    surcharge = 0.0
    extra = table.get("inputImages")
    if isinstance(extra, dict) and isinstance(extra.get("additional"), dict):
        billable = max(0, input_images - int(extra.get("included", 0)))
        surcharge = billable * float(extra["additional"]["usd"])

    if quality and tier and isinstance(table.get("quality"), dict):
        entry = table["quality"].get(tier, {}).get(quality)
        if entry:
            return float(entry["usd"]) + surcharge
    if tier and isinstance(table.get("resolutions"), dict):
        entry = table["resolutions"].get(tier)
        if entry:
            return float(entry["usd"]) + surcharge
    for key in ("generation", "inpaint"):
        entry = table.get(key)
        if isinstance(entry, dict) and "usd" in entry:
            return float(entry["usd"]) + surcharge
    return None


async def spendable_usd(client: "VeniceClient") -> float:
    """USD the calling API key can still spend.

    Reads ``balances.USD`` from ``GET /api_keys/rate_limits`` (a free call): the
    lesser of the account balance and what remains under the key's spending
    limit. Each response's ``balance_info.usd`` carries the same figure as it
    stood before that request, so the cost of one call is the drop between a
    reading taken before it and one taken after it. Any other spending on the
    key in between adds to that drop.
    """
    limits = await client.api_keys.get_rate_limits()
    return limits.data.balances.USD


def format_price(price: float | None) -> str:
    """Render a catalog price for display, or say that none is listed."""
    return f"${price:.2f}" if price is not None else "price not listed"


def save_image(stem: Path, data: bytes) -> Path:
    """Write ``data`` to ``stem`` plus the extension detected from its bytes.

    :raises ValueError: If the bytes are not a recognised image format.
    """
    ext, _mime = detect_image_format(data)
    if ext == "bin":
        raise ValueError(f"Response for {stem.name} is not a recognised image ({len(data)} bytes)")
    path = stem.with_suffix(f".{ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def image_dimensions(data: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` read from an image header.

    Supports PNG, GIF, WebP (lossy, lossless and extended) and baseline or
    progressive JPEG.

    :raises ValueError: If the format is unrecognised or the header is truncated.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data.startswith(b"GIF8") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8 ":
            w, h = struct.unpack("<HH", data[26:30])
            return w & 0x3FFF, h & 0x3FFF
        if chunk == b"VP8L":
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8X":
            w = int.from_bytes(data[24:27], "little") + 1
            h = int.from_bytes(data[27:30], "little") + 1
            return w, h
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            (length,) = struct.unpack(">H", data[i + 2 : i + 4])
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                return w, h
            i += 2 + length
    raise ValueError("Unrecognised or truncated image header")


def pillow_available() -> bool:
    """Whether Pillow, which the pixel checks need, is installed."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def alpha_coverage(data: bytes) -> tuple[float, float] | None:
    """Fractions of pixels that are see-through and solid, as ``(clear, solid)``.

    A pixel counts as see-through when its alpha is below 128 and as solid
    otherwise, so soft cutout edges fall on one side or the other. A header
    can declare an alpha channel on an image whose every pixel is opaque, so
    this decodes the pixels. An image without alpha is ``(0.0, 1.0)``.
    Returns ``None`` when Pillow (``pip install Pillow``) is not installed.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    from io import BytesIO

    with Image.open(BytesIO(data)) as img:
        if "A" not in img.getbands() and "transparency" not in img.info:
            return 0.0, 1.0
        histogram = img.convert("RGBA").getchannel("A").histogram()
    total = sum(histogram)
    clear = sum(histogram[:128]) / total
    return clear, 1.0 - clear


def mean_pixel_difference(first: bytes, second: bytes) -> float | None:
    """Mean absolute RGB difference between two same-size images, from 0 to 255.

    ``0.0`` means the pixels are identical. Returns ``None`` when Pillow
    (``pip install Pillow``) is not installed.

    :raises ValueError: If the images differ in size.
    """
    try:
        from PIL import Image, ImageChops, ImageStat
    except ImportError:
        return None
    from io import BytesIO

    with Image.open(BytesIO(first)) as a, Image.open(BytesIO(second)) as b:
        if a.size != b.size:
            raise ValueError(f"Cannot compare a {a.size} image with a {b.size} image")
        diff = ImageChops.difference(a.convert("RGB"), b.convert("RGB"))
        return sum(ImageStat.Stat(diff).mean) / 3
