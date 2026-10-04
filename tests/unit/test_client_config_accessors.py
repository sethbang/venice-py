"""Read-only views of a client's resolved retry policy and connection-pool limits."""

from __future__ import annotations

import dataclasses

import aiohttp
import pytest

from venice_ai import RetryOptions, SyncVeniceClient, VeniceClient
from venice_ai.core.config import HttpClientConfig, VeniceAIConfig
from venice_ai.core.http_client import ConnectionLimits


async def test_default_retry_options_are_the_documented_defaults() -> None:
    async with VeniceClient(api_key="sk-test") as client:
        assert client.retry_options == RetryOptions()


async def test_retry_options_reflect_constructor_inputs_and_are_frozen() -> None:
    async with VeniceClient(api_key="sk-test", max_retries=5) as client:
        options = client.retry_options
        assert options is not None
        assert options.max_attempts == 5
        with pytest.raises(dataclasses.FrozenInstanceError):
            options.max_attempts = 99  # type: ignore[misc]
        assert client.retry_options is not None
        assert client.retry_options.max_attempts == 5


async def test_retry_options_follow_config() -> None:
    config = VeniceAIConfig.create_minimal_config(
        api_key="sk-test", http_client=HttpClientConfig(max_retries=1)
    )
    async with VeniceClient(api_key="sk-test", config=config) as client:
        assert client.retry_options is not None
        assert client.retry_options.max_attempts == 1


async def test_retry_options_report_a_with_retries_override() -> None:
    override = RetryOptions(max_attempts=7)
    async with VeniceClient(api_key="sk-test") as client:
        async with client.with_retries(override):
            assert client.retry_options == override
        assert client.retry_options == RetryOptions()


async def test_caller_supplied_session_has_no_sdk_retry_policy() -> None:
    async with aiohttp.ClientSession() as session:
        client = VeniceClient(api_key="sk-test", http_client=session)
        assert client.retry_options is None
        assert client.connection_limits == ConnectionLimits(limit=100, limit_per_host=0)


async def test_connection_limits_defaults_and_overrides() -> None:
    async with VeniceClient(api_key="sk-test") as client:
        assert client.connection_limits == ConnectionLimits(limit=1000, limit_per_host=0)
    async with VeniceClient(
        api_key="sk-test", connector_limit=20, connector_limit_per_host=5
    ) as client:
        assert client.connection_limits == ConnectionLimits(limit=20, limit_per_host=5)
        session = await client._get_session()
        assert session.connector is not None
        assert session.connector.limit == 20
        assert session.connector.limit_per_host == 5


async def test_transport_option_limits_take_precedence() -> None:
    async with VeniceClient(
        api_key="sk-test",
        connector_limit=20,
        connector_limit_per_host=5,
        http_transport_options={"limit": 7, "limit_per_host": 3, "ttl_dns_cache": 60},
    ) as client:
        assert client.connection_limits == ConnectionLimits(limit=7, limit_per_host=3)
        session = await client._get_session()
        assert session.connector is not None
        assert session.connector.limit == 7
        assert session.connector.limit_per_host == 3


async def test_partial_transport_option_limits_fill_from_the_kwargs() -> None:
    async with VeniceClient(
        api_key="sk-test", connector_limit_per_host=5, http_transport_options={"limit": 7}
    ) as client:
        assert client.connection_limits == ConnectionLimits(limit=7, limit_per_host=5)
        session = await client._get_session()
        assert session.connector is not None
        assert (session.connector.limit, session.connector.limit_per_host) == (7, 5)


async def test_connection_limits_follow_config() -> None:
    config = VeniceAIConfig.create_minimal_config(
        api_key="sk-test", http_client=HttpClientConfig(max_connections=42)
    )
    async with VeniceClient(api_key="sk-test", config=config) as client:
        assert client.connection_limits.limit == 42


def test_sync_client_mirrors_the_accessors() -> None:
    client = SyncVeniceClient(api_key="sk-test", max_retries=4)
    try:
        assert client.retry_options is not None
        assert client.retry_options.max_attempts == 4
        override = RetryOptions(max_attempts=9)
        with client.with_retries(override):
            assert client.retry_options == override
        assert client.connection_limits == ConnectionLimits(limit=1000, limit_per_host=0)
    finally:
        client.close()


@pytest.mark.parametrize(("limit", "per_host"), [(None, None), (10, None), (None, 3)])
def test_resolver_defaults(limit: int | None, per_host: int | None) -> None:
    from venice_ai.core.http_client import resolve_connection_limits

    resolved = resolve_connection_limits(limit, per_host)
    assert resolved.limit == (1000 if limit is None else limit)
    assert resolved.limit_per_host == (0 if per_host is None else per_host)
