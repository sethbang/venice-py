"""
Unit tests for :class:`venice_ai.resources.augment.Augment`.

Exercises the three /augment/* endpoints documented at
``api-reference/endpoint/augment/``: scrape, search, and text-parser.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.resources.augment import Augment
from venice_ai.types.api.augment import (
    AugmentScrapeResponse,
    AugmentSearchResponse,
    AugmentTextParserResponse,
)


@pytest.fixture
def augment() -> Augment:
    client = MagicMock()
    return Augment(client)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# scrape()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_posts_to_augment_scrape(augment: Augment) -> None:
    augment._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=AugmentScrapeResponse(
            url="https://example.com",
            content="# Hello",
            format="markdown",
        )
    )
    result = await augment.scrape(url="https://example.com")

    assert result.url == "https://example.com"
    assert result.content == "# Hello"
    assert result.format == "markdown"

    call_args = augment._client.post.call_args  # type: ignore[attr-defined]
    assert call_args.args[0] == "augment/scrape"
    assert call_args.kwargs["json_data"] == {"url": "https://example.com"}


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_posts_to_augment_search(augment: Augment) -> None:
    augment._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=AugmentSearchResponse(query="test", results=[])
    )
    await augment.search(query="latest news about AI", limit=5, search_provider="brave")

    call_args = augment._client.post.call_args  # type: ignore[attr-defined]
    assert call_args.args[0] == "augment/search"
    assert call_args.kwargs["json_data"] == {
        "query": "latest news about AI",
        "limit": 5,
        "search_provider": "brave",
    }


@pytest.mark.asyncio
async def test_search_omits_optional_fields_when_unset(augment: Augment) -> None:
    augment._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=AugmentSearchResponse(query="x", results=[])
    )
    await augment.search(query="x")

    body = augment._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body == {"query": "x"}


@pytest.mark.asyncio
async def test_search_rejects_overly_long_query(augment: Augment) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await augment.search(query="x" * 401)


@pytest.mark.asyncio
async def test_search_rejects_bad_provider(augment: Augment) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        await augment.search(query="x", search_provider="duckduckgo")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# parse_text()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_text_bytes_uploads_as_multipart(augment: Augment) -> None:
    augment._request_multipart = AsyncMock(return_value={"text": "hello", "tokens": 3})  # type: ignore[method-assign]

    result = await augment.parse_text(file=b"PDF-bytes")

    assert isinstance(result, AugmentTextParserResponse)
    assert result.text == "hello"
    assert result.tokens == 3

    call = augment._request_multipart.call_args  # type: ignore[attr-defined]
    assert call.kwargs["method"] == "POST"
    assert call.kwargs["path"] == "augment/text-parser"
    files = call.kwargs["files"]
    assert "file" in files
    assert files["file"][1] == b"PDF-bytes"
    assert call.kwargs["data"] == {"response_format": "json"}


@pytest.mark.asyncio
async def test_parse_text_text_mode_returns_str(augment: Augment) -> None:
    augment._request_multipart = AsyncMock(return_value=b"plain extracted text")  # type: ignore[method-assign]

    result = await augment.parse_text(file=b"PDF-bytes", response_format="text")

    assert isinstance(result, str)
    assert result == "plain extracted text"
    assert augment._request_multipart.call_args.kwargs["data"] == {  # type: ignore[attr-defined]
        "response_format": "text"
    }


@pytest.mark.asyncio
async def test_parse_text_file_path_sets_correct_content_type(augment: Augment, tmp_path) -> None:
    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF")

    augment._request_multipart = AsyncMock(return_value={"text": "t", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=str(pdf_path))

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, content, content_type = files["file"]
    assert filename == "report.pdf"
    assert content == b"%PDF-1.4\n%EOF"
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_parse_text_missing_file_raises(augment: Augment) -> None:
    with pytest.raises(ValueError):
        await augment.parse_text(file="/definitely/not/a/file.pdf")


@pytest.mark.asyncio
async def test_parse_text_explicit_content_type_overrides_default(augment: Augment) -> None:
    """Caller-provided ``content_type`` always wins over extension/sniffing."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(
        file=b"arbitrary bytes",
        content_type="application/pdf",
    )

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    _, _, content_type = files["file"]
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_parse_text_filename_hint_drives_content_type(augment: Augment) -> None:
    """A ``filename`` hint with a known extension drives content_type."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=b"arbitrary bytes", filename="quarterly.docx")

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, _, content_type = files["file"]
    assert filename == "quarterly.docx"
    assert content_type.endswith("wordprocessingml.document")


@pytest.mark.asyncio
async def test_parse_text_pptx_filename_sets_correct_content_type(augment: Augment) -> None:
    """A ``.pptx`` filename hint maps to the PowerPoint OOXML content type."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=b"PK\x03\x04pptx-bytes", filename="deck.pptx")

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, _, content_type = files["file"]
    assert filename == "deck.pptx"
    assert content_type == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


@pytest.mark.asyncio
async def test_parse_text_epub_path_sets_correct_content_type(augment: Augment, tmp_path) -> None:
    """An ``.epub`` Path maps to ``application/epub+zip`` rather than the DOCX
    sniff. EPUB is a ZIP container (``PK\\x03\\x04``); without an explicit
    extension mapping it falls through to the DOCX sniff, which the server
    rejects ("No text content could be extracted")."""
    epub_path = tmp_path / "book.epub"
    epub_path.write_bytes(b"PK\x03\x04epub-bytes")

    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=str(epub_path))

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, _, content_type = files["file"]
    assert filename == "book.epub"
    assert content_type == "application/epub+zip"


@pytest.mark.asyncio
async def test_parse_text_pptx_path_sets_correct_content_type(augment: Augment, tmp_path) -> None:
    """A ``.pptx`` Path resolves to the PowerPoint OOXML content type."""
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"PK\x03\x04pptx-bytes")

    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=str(pptx_path))

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    filename, _, content_type = files["file"]
    assert filename == "deck.pptx"
    assert content_type == (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    )


@pytest.mark.asyncio
async def test_parse_text_sniffs_pdf_magic_bytes(augment: Augment) -> None:
    """Raw bytes starting with ``%PDF-`` are sniffed even without a filename."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=b"%PDF-1.4\nfake pdf body")

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    _, _, content_type = files["file"]
    assert content_type == "application/pdf"


@pytest.mark.asyncio
async def test_parse_text_sniffs_zip_magic_as_docx(augment: Augment) -> None:
    """ZIP container bytes default to DOCX when no other hint is given."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=b"PK\x03\x04rest of zip")

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    _, _, content_type = files["file"]
    assert content_type.endswith("wordprocessingml.document")


@pytest.mark.asyncio
async def test_parse_text_unknown_bytes_falls_back_to_octet_stream(augment: Augment) -> None:
    """Unknown binary bytes still go up as octet-stream — server returns a clear error."""
    augment._request_multipart = AsyncMock(return_value={"text": "x", "tokens": 1})  # type: ignore[method-assign]

    await augment.parse_text(file=b"\x00\x01\x02\x03\x04\x05")

    files = augment._request_multipart.call_args.kwargs["files"]  # type: ignore[attr-defined]
    _, _, content_type = files["file"]
    assert content_type == "application/octet-stream"


# ---------------------------------------------------------------------------
# client wiring
# ---------------------------------------------------------------------------


def test_client_exposes_augment_namespace() -> None:
    """The async VeniceClient registers an ``augment`` attribute."""
    from venice_ai import VeniceClient

    # We don't actually create HTTP; just inspect attribute presence via
    # VeniceClient.__init__ hooks by constructing with a dummy config. A
    # cleaner check: the attribute name appears on the class.
    assert "augment" in VeniceClient.__init__.__code__.co_names or hasattr(
        VeniceClient, "__annotations__"
    )
    # Concrete check: build a minimal client via the factory and inspect.
    from venice_ai import create_test_venice_client

    client = create_test_venice_client(api_key="vn_unit_test_AugmentWiring_123456")
    try:
        assert isinstance(client.augment, Augment)
    finally:
        # Don't need to close — no HTTP was issued — but do it anyway.
        import asyncio

        asyncio.run(client.close())


@pytest.mark.asyncio
async def test_search_accepts_exa(augment: Augment) -> None:
    augment._client.post = AsyncMock(  # type: ignore[attr-defined]
        return_value=AugmentSearchResponse(query="x", results=[])
    )
    await augment.search(query="x", search_provider="exa")

    body = augment._client.post.call_args.kwargs["json_data"]  # type: ignore[attr-defined]
    assert body == {"query": "x", "search_provider": "exa"}


def test_search_rejects_an_unknown_provider() -> None:
    from pydantic import ValidationError

    from venice_ai.types.api.augment import AugmentSearchRequest

    with pytest.raises(ValidationError):
        AugmentSearchRequest(query="x", search_provider="zzz")  # type: ignore[arg-type]
