"""
Venice AI Augment API resources.

Wraps the three ``/augment/*`` endpoints described at
``api-reference/endpoint/augment/``:

- ``POST /augment/scrape`` — fetch a URL as markdown
- ``POST /augment/search`` — web search (Brave, Exa or Google)
- ``POST /augment/text-parser`` — extract text from uploaded PDF/DOCX/XLSX/TXT

The Venice docs mark this API as experimental — request and response shapes
may change without notice.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, cast, overload

from .._resource import APIResource
from ..types.api.augment import (
    AugmentScrapeRequest,
    AugmentScrapeResponse,
    AugmentSearchRequest,
    AugmentSearchResponse,
    AugmentTextParserResponse,
    SearchProvider,
)

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["Augment"]

# Filenames & content types for the text-parser multipart upload.
_TEXT_PARSER_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
    # EPUB is a ZIP container ("PK\x03\x04") and would otherwise be sniffed as
    # DOCX, which the server rejects. The extension map is consulted before the
    # byte sniffer, so this fixes .epub-by-path uploads.
    ".epub": "application/epub+zip",
}


def _sniff_content_type(content: bytes) -> str | None:
    """Best-effort sniff of an upload's content type from leading bytes.

    Used as a final fallback when the caller didn't pass ``content_type`` and
    the filename didn't carry a recognized extension. Returns ``None`` if no
    confident match — caller should then send ``application/octet-stream``,
    which the server will reject with a clear error.
    """
    if content.startswith(b"%PDF-"):
        return "application/pdf"
    # DOCX / XLSX are ZIP containers — both start with "PK\x03\x04". We can't
    # cheaply distinguish the two without unzipping, so prefer DOCX (the
    # common parser case); callers should pass ``content_type`` to override.
    if content.startswith(b"PK\x03\x04"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    # Plain ASCII / UTF-8 text usually has only printable chars + common
    # whitespace; this is rough but adequate for the common .txt-as-bytes case.
    sample = content[:1024]
    if sample:
        try:
            decoded = sample.decode("utf-8")
        except UnicodeDecodeError:
            return None
        if decoded and all(c.isprintable() or c in "\r\n\t" for c in decoded):
            return "text/plain"
    return None


class Augment(APIResource["VeniceClient"]):
    """Provides access to Venice's Augment (web-scrape / search / text-parse) API."""

    async def scrape(self, *, url: str) -> AugmentScrapeResponse:
        """Fetch a URL and return its contents as markdown.

        :param url: Publicly accessible URL to scrape. Must start with ``http://``
            or ``https://``.
        :type url: str

        :return: :class:`AugmentScrapeResponse` with ``url``, ``content``, and ``format``.
        :rtype: AugmentScrapeResponse

        :raises venice_ai.exceptions.APIError: If the target site blocks automated
            access (e.g. X/Twitter, Reddit) or another error occurs.

        Example::

            async with VeniceClient() as client:
                result = await client.augment.scrape(url="https://example.com")
                print(result.content)
        """
        request = AugmentScrapeRequest(url=url)
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "augment/scrape", json_data=body, cast_to=AugmentScrapeResponse
        )

    async def search(
        self,
        *,
        query: str,
        limit: int | None = None,
        search_provider: SearchProvider | None = None,
    ) -> AugmentSearchResponse:
        """Run a web search and return structured results.

        :param query: Search query (1–400 chars).
        :type query: str
        :param limit: Maximum number of results (1–20, default 10).
        :type limit: Optional[int]
        :param search_provider: ``"brave"`` (default; ZDR privacy), ``"exa"``
            (ZDR privacy) or ``"google"`` (proxied / anonymised).
        :type search_provider: Optional[SearchProvider]

        :return: :class:`AugmentSearchResponse` with ``query`` and ``results``.
        :rtype: AugmentSearchResponse

        Example::

            async with VeniceClient() as client:
                response = await client.augment.search(
                    query="latest news about AI",
                    limit=5,
                )
                for r in response.results:
                    print(r.title, r.url)
        """
        request = AugmentSearchRequest(
            query=query,
            limit=limit,
            search_provider=search_provider,
        )
        body = request.model_dump(exclude_none=True)
        return await self._client.post(
            "augment/search", json_data=body, cast_to=AugmentSearchResponse
        )

    @overload
    async def parse_text(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        response_format: Literal["json"] = "json",
        content_type: str | None = None,
        filename: str | None = None,
    ) -> AugmentTextParserResponse: ...

    @overload
    async def parse_text(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        response_format: Literal["text"],
        content_type: str | None = None,
        filename: str | None = None,
    ) -> str: ...

    async def parse_text(
        self,
        *,
        file: str | bytes | BinaryIO | Path,
        response_format: Literal["json", "text"] = "json",
        content_type: str | None = None,
        filename: str | None = None,
    ) -> AugmentTextParserResponse | str:
        """Extract text from a document file (PDF, DOCX, PPTX, XLSX, TXT).

        :param file: File to parse. Can be a path (``str`` / :class:`~pathlib.Path`),
            raw ``bytes``, or any binary file-like object.
        :type file: Union[str, bytes, BinaryIO, Path]
        :param response_format: ``"json"`` (default) returns an
            :class:`AugmentTextParserResponse` with ``text`` and ``tokens``.
            ``"text"`` returns the extracted text as a plain ``str``.
        :type response_format: Literal["json", "text"]
        :param content_type: Optional MIME type override for the upload. When
            ``None`` (default) the SDK derives it from the file extension, falling
            back to magic-byte sniffing for ``bytes`` / ``BinaryIO`` inputs without
            a recognised extension. Pass this explicitly when uploading raw bytes
            with no filename hint.
        :type content_type: Optional[str]
        :param filename: Optional filename hint for ``bytes`` / ``BinaryIO`` inputs.
            Used both for the multipart filename field and to derive ``content_type``
            from the extension when the latter is not given. Ignored when ``file``
            is a path.
        :type filename: Optional[str]

        :return: Either :class:`AugmentTextParserResponse` or plain ``str``
            depending on ``response_format``.

        :raises ValueError: If the file path is invalid or unreadable.
        :raises venice_ai.exceptions.APIError: If the API request fails.

        Example::

            async with VeniceClient() as client:
                # path: extension carries the type
                parsed = await client.augment.parse_text(file="report.pdf")

                # raw bytes: pass either content_type or a filename hint
                with open("report.pdf", "rb") as f:
                    parsed = await client.augment.parse_text(
                        file=f.read(),
                        content_type="application/pdf",
                    )
                print(parsed.text, parsed.tokens)
        """
        file_content, default_filename = await self._read_file(file)
        effective_filename = filename or default_filename

        if content_type is None:
            ext = Path(effective_filename).suffix.lower()
            sdk_default = _TEXT_PARSER_CONTENT_TYPES.get(ext)
            if sdk_default is not None:
                content_type = sdk_default
            else:
                sniffed = _sniff_content_type(file_content)
                content_type = sniffed or "application/octet-stream"

        files_dict: dict[str, Any] = {
            "file": (effective_filename, file_content, content_type),
        }
        form_data = {"response_format": response_format}

        # When asking for plain text, request it via Accept rather than the
        # multipart default of application/json (which mismatches the text/plain body).
        accept_headers = {"Accept": "text/plain"} if response_format == "text" else None

        response = await self._request_multipart(
            method="POST",
            path="augment/text-parser",
            files=files_dict,
            data=form_data,
            headers=accept_headers,
        )

        if response_format == "text":
            if isinstance(response, bytes):
                return response.decode("utf-8")
            if isinstance(response, str):
                return response
            if isinstance(response, dict) and "text" in response:
                return cast(str, response["text"])
            raise TypeError(f"Unexpected text-parser response type: {type(response)}")

        if isinstance(response, dict):
            return AugmentTextParserResponse.model_validate(response)
        if isinstance(response, bytes):
            import json as _json

            return AugmentTextParserResponse.model_validate(_json.loads(response))
        raise TypeError(f"Unexpected text-parser response type: {type(response)}")

    @staticmethod
    async def _read_file(
        file: str | bytes | BinaryIO | Path,
    ) -> tuple[bytes, str]:
        """Normalize the ``file`` arg into ``(bytes, filename)``."""
        if isinstance(file, (str, Path)):
            path = Path(file)
            if not path.exists():
                raise ValueError(f"File not found: {file}")
            return path.read_bytes(), path.name
        if isinstance(file, bytes):
            return file, "document"
        if isinstance(file, io.BytesIO):
            return file.read(), "document"
        file_any = cast(Any, file)
        if hasattr(file_any, "read") and callable(file_any.read):
            result = file_any.read()
            if asyncio.iscoroutine(result):
                content = await result
            else:
                content = result
            if not isinstance(content, bytes):
                raise TypeError("File-like object must return bytes from read()")
            filename = "document"
            if hasattr(file_any, "name"):
                filename = Path(file_any.name).name
            return content, filename
        raise TypeError(f"Unsupported file type: {type(file)}")
