"""Every entry point takes the API base URL in one form and resolves it the same way.

The form is the API root, version path included (``https://api.venice.ai/api/v1``);
a bare host is shorthand for it. Passing the same value to ``VeniceClient``, to
``VeniceAIConfig`` (directly or through ``VENICE_API_BASE_URL``) and to the
factory must send requests to the same URL: no ``/api/v1/api/v1``.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from venice_ai import VeniceClient
from venice_ai._base_url import normalize_base_url
from venice_ai.core.config import VeniceAIConfig
from venice_ai.factory import VeniceClientFactory

ROOT = "http://gateway.test:8080/api/v1/"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("https://api.venice.ai/api/v1", "https://api.venice.ai/api/v1"),
        ("https://api.venice.ai/api/v1/", "https://api.venice.ai/api/v1"),
        ("https://api.venice.ai", "https://api.venice.ai/api/v1"),
        ("https://api.venice.ai/", "https://api.venice.ai/api/v1"),
        ("  http://localhost:9000  ", "http://localhost:9000/api/v1"),
        ("https://proxy.example.com/venice", "https://proxy.example.com/venice"),
    ],
)
def test_normalize_base_url(given: str, expected: str) -> None:
    assert normalize_base_url(given) == expected


def test_bare_host_uses_the_requested_api_version() -> None:
    assert normalize_base_url("https://api.venice.ai", "v2") == "https://api.venice.ai/api/v2"


@pytest.mark.parametrize("version", ["/v2", "v2/", " /v2/ "])
def test_api_version_slashes_and_whitespace_are_trimmed(version: str) -> None:
    assert normalize_base_url("https://api.venice.ai", version) == "https://api.venice.ai/api/v2"


@pytest.mark.parametrize("blank", ["", "   ", "/", " // ", "\t\n"])
@pytest.mark.parametrize("url", ["https://api.venice.ai", "https://api.venice.ai/api/v1"])
def test_normalize_rejects_a_blank_api_version(url: str, blank: str) -> None:
    with pytest.raises(ValueError, match="api_version must name an API version"):
        normalize_base_url(url, blank)


@pytest.mark.parametrize("blank", ["", "  ", "/"])
def test_config_rejects_a_blank_api_version(blank: str) -> None:
    with pytest.raises(ValueError, match="api_version must name an API version"):
        VeniceAIConfig(api_version=blank, api_base_url="https://api.venice.ai")


def test_config_trims_api_version_before_building_the_root() -> None:
    config = VeniceAIConfig(api_version="/v2/", api_base_url="https://api.venice.ai")
    assert config.api_version == "v2"
    assert config.api_base_url == "https://api.venice.ai/api/v2"


@pytest.mark.parametrize("bad", ["api.venice.ai", "ftp://api.venice.ai", "https://", ""])
def test_normalize_rejects_non_http_urls(bad: str) -> None:
    with pytest.raises(ValueError, match="absolute http"):
        normalize_base_url(bad)


def _endpoint(client: VeniceClient) -> str:
    return str(
        client._build_request_kwargs("POST", "chat/completions", None, None, {}, None, None)["url"]
    )


@pytest.mark.parametrize("value", [ROOT, "http://gateway.test:8080"])
async def test_all_entry_points_agree(value: str) -> None:
    expected = "http://gateway.test:8080/api/v1/chat/completions"
    with patch.dict(os.environ, {}, clear=True):
        direct = VeniceClient(api_key="k", base_url=value)
        via_config = VeniceClient(api_key="k", config=VeniceAIConfig(api_base_url=value))
        via_factory = VeniceClientFactory.create_client(
            VeniceAIConfig(api_base_url=value), api_key="k", account_id="a"
        )
    with patch.dict(os.environ, {"VENICE_API_BASE_URL": value}, clear=True):
        via_env = VeniceClient(api_key="k")
        via_env_config = VeniceClient(api_key="k", config=VeniceAIConfig())
    clients = [direct, via_config, via_factory, via_env, via_env_config]
    try:
        assert [_endpoint(c) for c in clients] == [expected] * len(clients)
    finally:
        for c in clients:
            await c.close()


async def test_explicit_base_url_wins_over_config_and_env() -> None:
    with patch.dict(os.environ, {"VENICE_API_BASE_URL": "http://env.test"}, clear=True):
        client = VeniceClient(
            api_key="k",
            base_url="http://explicit.test/api/v1",
            config=VeniceAIConfig(api_base_url="http://config.test"),
        )
    try:
        assert _endpoint(client) == "http://explicit.test/api/v1/chat/completions"
    finally:
        await client.close()


async def test_default_is_the_official_api_root() -> None:
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(api_key="k")
    try:
        assert client.base_url == "https://api.venice.ai/api/v1/"
    finally:
        await client.close()
