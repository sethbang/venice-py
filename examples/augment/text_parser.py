#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Text Parser
====================================

Demonstrates ``client.augment.parse_text(file=..., response_format=...)`` —
Venice's multipart document-parsing endpoint. The endpoint accepts PDF, DOCX,
XLSX, and plain text files up to 25 MB; this example uploads plain text so the
extracted text can be checked against the input exactly. For other formats,
pass the file path (or bytes plus the matching ``content_type``) the same way.

Two return shapes:

- ``response_format="json"`` (default) returns an
  :class:`AugmentTextParserResponse` with ``.text`` and ``.tokens``.
- ``response_format="text"`` returns a plain ``str`` of the extracted text.

**Privacy:** Parsing runs in-memory on Venice's infrastructure with zero
data retention — documents are processed then immediately discarded.

**Pricing:** $0.01 per request.

**Note:** The Augment API is marked experimental in the Venice docs; request
and response shapes may change without notice.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError

# ---------------------------------------------------------------------------
# 1. Parse a plain text file (JSON response)
# ---------------------------------------------------------------------------


def round_trips(original: str, extracted: str) -> bool:
    """Plain text should come back unchanged (ignoring surrounding whitespace)."""
    if extracted.strip() == original.strip():
        print("✅ Extracted text matches the uploaded text exactly")
        return True
    print("❌ Extracted text differs from the uploaded text")
    return False


async def parse_plain_text() -> bool:
    """Upload a small .txt file and get back text + token count."""
    print("📄 Parse Plain Text → JSON")
    print("-" * 30)

    original = (
        "Venice AI is a privacy-first inference platform.\n"
        "It exposes a Python SDK at venice-py on PyPI.\n"
        "The Augment API lets you scrape, search, and parse documents.\n"
    )

    # Write a small sample file to a temporary directory and upload it by path.
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "augment_sample.txt"
        sample.write_text(original)
        print(f"📎 Uploading: {sample}")

        async with VeniceClient() as client:
            result = await client.augment.parse_text(file=str(sample))

    print(f"🔢 Tokens: {result.tokens}")
    print(f"📝 Extracted text ({len(result.text)} chars):")
    print("-" * 30)
    print(result.text)
    return round_trips(original, result.text)


# ---------------------------------------------------------------------------
# 2. Raw bytes upload with the plain text response format
# ---------------------------------------------------------------------------


async def parse_with_text_response() -> bool:
    """Upload raw bytes and use ``response_format='text'`` to get a plain ``str`` back.

    Passing bytes plus ``content_type`` and ``filename`` needs no file on disk.
    Unlike the default ``response_format='json'`` (which returns an
    ``AugmentTextParserResponse`` with ``.text``/``.tokens``), the ``"text"``
    format returns the extracted text directly as a ``str``.
    """
    print("\n🧾 Parse raw bytes with response_format='text'")
    print("-" * 30)

    content = (
        b"Venice AI Augment text-parser demo.\n"
        b"This document is uploaded as raw bytes and parsed back as plain text.\n"
    )

    async with VeniceClient() as client:
        text = await client.augment.parse_text(
            file=content,
            response_format="text",
            content_type="text/plain",
            filename="demo.txt",
        )

    # response_format="text" returns a plain str (no .text / .tokens).
    print(f"📝 Got {type(text).__name__} ({len(text)} chars):")
    print("-" * 30)
    print(text)
    return isinstance(text, str) and round_trips(content.decode(), text)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment text-parser examples."""
    print("🚀 Venice AI Augment — Text Parser Examples")
    print("=" * 50)

    sub_examples = [
        ("parse_plain_text", parse_plain_text),
        ("parse_with_text_response", parse_with_text_response),
    ]

    results: list[tuple[str, bool]] = []
    for name, fn in sub_examples:
        try:
            results.append((name, await fn()))
        except VeniceError as e:
            print(f"❌ {name} failed: {type(e).__name__}: {e}")
            results.append((name, False))

    print("\n" + "=" * 50)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        status = "✅" if ok else "❌"
        print(f"   {status} {name}")

    if passed != total:
        print(f"\n❌ {total - passed} of {total} text-parser sub-examples failed")
        return 1

    print(f"\n✨ {passed}/{total} text-parser sub-examples completed")
    print("\n💡 Key concepts demonstrated:")
    print("   - File-path upload with JSON response")
    print("   - Raw-bytes upload (no file on disk) with response_format='text'")
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
