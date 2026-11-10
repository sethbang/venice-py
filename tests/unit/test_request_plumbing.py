"""Requests to the API keep their headers however its URL is written."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from yarl import URL

from venice_ai import VeniceClient
from venice_ai._client import _redacted_headers
from venice_ai.core.config import VeniceAIConfig
from venice_ai.core.http_client import VeniceHTTPClient
from venice_ai.middleware.third_party import create_third_party_header_filter, same_origin


async def _headers_after_filter(api_base: str, request_url: str) -> dict[str, str]:
    middleware = create_third_party_header_filter(URL(api_base))
    request: Any = SimpleNamespace(
        url=URL(request_url),
        headers={"Authorization": "Bearer k", "X-Custom": "c", "Accept": "*/*"},
    )
    await middleware(request, AsyncMock())
    return request.headers


@pytest.mark.parametrize(
    ("api_base", "request_url"),
    [
        ("https://gw:443/venice", "https://gw/venice/audio/speech"),
        ("https://gw/venice", "https://gw:443/venice/audio/speech"),
        ("http://gw:80/venice", "http://gw/venice/image/edit"),
        ("https://GW/venice", "https://gw/venice/x"),
    ],
)
async def test_an_explicit_default_port_is_the_same_api(api_base: str, request_url: str) -> None:
    kept = await _headers_after_filter(api_base, request_url)
    assert kept["Authorization"] == "Bearer k"
    assert kept["X-Custom"] == "c"


@pytest.mark.parametrize(
    ("api_base", "request_url"),
    [
        ("https://gw:443/venice", "https://gw:8443/venice/x"),
        ("https://gw:443/venice", "http://gw/venice/x"),
        ("https://gw/venice", "https://gw.evil.example/venice/x"),
        ("https://gw/venice", "https://gw@evil.example/venice/x"),
    ],
)
async def test_another_origin_stays_third_party(api_base: str, request_url: str) -> None:
    kept = await _headers_after_filter(api_base, request_url)
    assert kept == {"Accept": "*/*"}


def test_same_origin_compares_scheme_host_and_effective_port() -> None:
    assert same_origin(URL("https://h:443/a"), URL("https://h/b"))
    assert not same_origin(URL("https://h:444/a"), URL("https://h/b"))
    assert not same_origin(URL("http://h/a"), URL("https://h/a"))


def test_fetch_external_matches_the_api_when_its_base_has_a_default_port() -> None:
    client = VeniceClient(api_key="k", base_url="https://gw:443/venice")
    assert same_origin(URL("https://gw/venice/file"), client._base_url)


def test_a_session_mock_with_a_spec_is_accepted() -> None:
    session = Mock(spec=aiohttp.ClientSession)
    client = VeniceClient(api_key="k", http_client=session)
    client._session = session
    assert "middlewares" in client._session_request_options()


def test_headers_given_as_pairs_are_sent() -> None:
    client = VeniceClient(api_key="k", headers=[("X-Custom", "c")])  # type: ignore[arg-type]
    assert client._request_headers(Mock(headers={}))["X-Custom"] == "c"


@pytest.mark.parametrize(
    "name",
    [
        "Cf-Access-Jwt-Assertion",
        "X-JWT",
        "X-Pass",
        "X-Passwd",
        "X-Pwd",
        "X-Passphrase",
        "X-Bearer",
        "X-Cred",
    ],
)
def test_debug_logging_hides_these_header_values(name: str) -> None:
    assert _redacted_headers({name: "SECRET"})[name] == "[REDACTED]"


def test_debug_logging_shows_ordinary_header_values() -> None:
    assert _redacted_headers({"Accept": "*/*"})["Accept"] == "*/*"


async def test_creating_a_session_logs_no_password_from_the_base_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = VeniceHTTPClient(config=VeniceAIConfig(), base_url="https://user:SECRET@gw/venice")
    with caplog.at_level(logging.DEBUG):
        session = await client.get_session()
    try:
        assert "gw" in caplog.text
        assert "SECRET" not in caplog.text
    finally:
        await session.close()
