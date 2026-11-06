#!/usr/bin/env python3
"""
Venice AI SDK - Augment: Text Parser
====================================

Demonstrates ``client.augment.parse_text(file=..., response_format=...)`` —
Venice's multipart document-parsing endpoint, for files up to 25 MB. It accepts
PDF, EPUB, DOCX, PPTX, XLSX and XLS (not the legacy ``.doc`` or ``.ppt``); plain
text, Markdown, CSV, JSON, YAML, TOML, XML and any ``text/*`` type; and source
code in about 140 recognized extensions.

This example uploads a plain text file by path, so the extracted text can be
checked against the input exactly, then a small PDF as raw bytes, built by this
script so nothing has to be downloaded, and checked for the sentence inside it. A
PDF that does not parse fails the run and prints the reason. For other formats,
pass the file path (or bytes plus the matching ``content_type``) the same way.

Two return shapes:

- ``response_format="json"`` (default) returns an
  :class:`AugmentTextParserResponse` with ``.text`` and ``.tokens``.
- ``response_format="text"`` returns a plain ``str`` of the extracted text.

**Privacy:** Parsing runs in-memory on Venice's infrastructure with zero
data retention — documents are processed then immediately discarded.

**Pricing:** $0.01 per request, so $0.02 per run for the two uploads here.

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
# 2. A PDF as raw bytes, with the plain text response format
# ---------------------------------------------------------------------------

PDF_SENTENCE = "Venice parsed this small PDF"


def build_minimal_pdf(sentence: str) -> bytes:
    """Build a valid one-page PDF showing ``sentence``, with a correct cross-reference table.

    ``sentence`` must be plain ASCII with no parentheses or backslashes.
    """
    stream = f"BT /F1 18 Tf 72 720 Td ({sentence}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_at = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        pdf += b"%010d 00000 n \n" % offset
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
    pdf += b"startxref\n%d\n%%%%EOF\n" % xref_at
    return bytes(pdf)


async def parse_pdf_with_text_response() -> bool:
    """Upload a PDF as raw bytes and use ``response_format='text'`` to get a ``str`` back.

    Passing bytes plus ``content_type`` and ``filename`` needs no file on disk.
    Unlike the default ``response_format='json'`` (which returns an
    ``AugmentTextParserResponse`` with ``.text``/``.tokens``), the ``"text"``
    format returns the extracted text directly as a ``str``. Whitespace is
    collapsed before comparing, since a PDF stores no line structure to return.
    """
    print("\n🧾 Parse a PDF from raw bytes with response_format='text'")
    print("-" * 30)

    content = build_minimal_pdf(PDF_SENTENCE)
    print(f"📎 Uploading a {len(content)}-byte PDF built by this script")

    async with VeniceClient() as client:
        text = await client.augment.parse_text(
            file=content,
            response_format="text",
            content_type="application/pdf",
            filename="demo.pdf",
        )

    # response_format="text" returns a plain str (no .text / .tokens).
    print(f"📝 Got {type(text).__name__} ({len(text)} chars):")
    print("-" * 30)
    print(text)
    if PDF_SENTENCE not in " ".join(text.split()):
        print(f"❌ The extracted text does not contain {PDF_SENTENCE!r}")
        return False
    print("✅ The sentence inside the PDF came back")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    """Run all augment text-parser examples."""
    print("🚀 Venice AI Augment — Text Parser Examples")
    print("=" * 50)

    sub_examples = [
        ("parse_plain_text", parse_plain_text),
        ("parse_pdf_with_text_response", parse_pdf_with_text_response),
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
    print("   - Raw-bytes PDF upload (no file on disk) with response_format='text'")
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
