"""HTTP-layer configuration must change what actually goes out on the wire.

These tests drive real clients against a local aiohttp server and observe the
effect of each setting: how many attempts the server receives, what backoff
parameters the retry middleware computes with, which connector limits the
session was built with, and whether a proxy is honoured. Asserting only that a
value was stored on a config object (or forwarded as a keyword argument to a
patched class) cannot tell a wired setting from a dead one.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.core.config import HttpClientConfig, VeniceAIConfig
from venice_ai.factory import VeniceClientFactory, create_developer_client
from venice_ai.middleware import retry as retry_module
from venice_ai.middleware.retry import RetryOptions

_API_KEY = "test-key"


# ---------------------------------------------------------------------------
# Local server + retry spy
# ---------------------------------------------------------------------------


class _AlwaysFailingServer:
    """Answers every request with a retryable status and records what it saw."""

    def __init__(self, status: int = 503) -> None:
        self.status = status
        self.hits = 0
        self.user_agents: list[str | None] = []
        self.hosts: list[str | None] = []
        self._runner: web.AppRunner | None = None
        self.port = 0

    async def _handle(self, request: web.Request) -> web.Response:
        self.hits += 1
        self.user_agents.append(request.headers.get("User-Agent"))
        self.hosts.append(request.headers.get("Host"))
        return web.json_response({"error": "transient"}, status=self.status)

    async def start(self) -> None:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets  # type: ignore[union-attr]
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def api_url(self) -> str:
        return f"{self.root_url}/api/v1"


@asynccontextmanager
async def failing_server(status: int = 503) -> AsyncIterator[_AlwaysFailingServer]:
    """Run the server for the block (a context manager, not an async fixture,
    so it is guaranteed to share the test body's event loop)."""
    srv = _AlwaysFailingServer(status=status)
    await srv.start()
    try:
        yield srv
    finally:
        await srv.stop()


@dataclass
class _BackoffSpy:
    """Stands in for ``calculate_backoff_delay``: records its inputs, never sleeps."""

    calls: list[dict[str, float]] = field(default_factory=list)

    def __call__(
        self,
        attempt: int,
        base_delay: float,
        exponential_base: float,
        max_delay: float,
        jitter_factor: float,
    ) -> float:
        self.calls.append(
            {
                "attempt": attempt,
                "base_delay": base_delay,
                "exponential_base": exponential_base,
                "max_delay": max_delay,
                "jitter_factor": jitter_factor,
            }
        )
        return 0.0


@asynccontextmanager
async def backoff_spy() -> AsyncIterator[_BackoffSpy]:
    spy = _BackoffSpy()
    with patch.object(retry_module, "calculate_backoff_delay", spy):
        yield spy


async def _send_one(client: VeniceClient) -> None:
    """Issue one request and swallow the terminal error the 503 produces."""
    async with client:
        with pytest.raises(Exception):  # noqa: B017 - the status is under test elsewhere
            await client.get("models")


def _factory_client(config: VeniceAIConfig) -> VeniceClient:
    with patch.dict(os.environ, {}, clear=True):
        return VeniceClientFactory.create_client(config, api_key=_API_KEY, account_id="acct")


def _config_for(server: _AlwaysFailingServer, **http_fields: Any) -> VeniceAIConfig:
    return VeniceAIConfig.create_minimal_config(
        api_key=_API_KEY,
        api_base_url=server.root_url,
        http_client=HttpClientConfig(**http_fields),
    )


# ---------------------------------------------------------------------------
# max_retries / retry_backoff_factor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("max_retries", [0, 1, 5])
async def test_factory_client_honours_http_client_max_retries(max_retries: int) -> None:
    async with failing_server() as server, backoff_spy():
        client = _factory_client(_config_for(server, max_retries=max_retries))
        await _send_one(client)

    assert server.hits == max_retries + 1, (
        f"HttpClientConfig(max_retries={max_retries}) should allow exactly "
        f"{max_retries + 1} attempt(s) on a retryable 503; the server saw {server.hits}"
    )


async def test_direct_client_with_config_honours_http_client_max_retries() -> None:
    async with failing_server() as server, backoff_spy():
        config = _config_for(server, max_retries=0)
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(api_key=_API_KEY, base_url=server.api_url, config=config)
        await _send_one(client)

    assert server.hits == 1, (
        "VeniceClient(config=...) with http_client.max_retries=0 must not retry; "
        f"the server saw {server.hits} attempts"
    )


@pytest.mark.parametrize("max_retries", [0, 1])
async def test_client_max_retries_keyword_is_honoured(max_retries: int) -> None:
    async with failing_server() as server, backoff_spy():
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                api_key=_API_KEY, base_url=server.api_url, max_retries=max_retries
            )
        await _send_one(client)

    assert server.hits == max_retries + 1, (
        f"VeniceClient(max_retries={max_retries}) should allow exactly "
        f"{max_retries + 1} attempt(s); the server saw {server.hits}"
    )


async def test_developer_client_retries_once_as_documented() -> None:
    from venice_ai.presets import development

    real_preset = development.create_development_config

    async with failing_server() as server, backoff_spy():

        def preset_pointed_at_server(**kwargs: Any) -> VeniceAIConfig:
            config = real_preset(**kwargs)
            config.api_base_url = server.root_url
            return config

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(development, "create_development_config", preset_pointed_at_server),
        ):
            client = create_developer_client(api_key=_API_KEY)
        await _send_one(client)

    assert server.hits == 2, (
        "create_developer_client() documents max_retries=1 (one retry, two attempts); "
        f"the server saw {server.hits}"
    )


async def test_retry_backoff_factor_drives_the_exponential_base() -> None:
    # ``retry_backoff_factor`` is bounded ``ge=1.0`` and defaults to 2.0 — the
    # same domain and default as ``RetryOptions.exponential_base``.
    assert HttpClientConfig.model_fields["retry_backoff_factor"].default == (
        RetryOptions().exponential_base
    )
    async with failing_server() as server, backoff_spy() as spy:
        client = _factory_client(_config_for(server, max_retries=2, retry_backoff_factor=3.5))
        await _send_one(client)

    assert spy.calls, "a retryable 503 with max_retries=2 must reach the backoff calculation"
    bases = {call["exponential_base"] for call in spy.calls}
    assert bases == {3.5}, (
        "HttpClientConfig(retry_backoff_factor=3.5) should be the backoff exponential "
        f"base; the retry middleware computed delays with {sorted(bases)}"
    )


async def test_explicit_retry_options_take_precedence_over_config() -> None:
    """An explicit ``retry_options=`` is the most specific setting and wins."""
    async with failing_server() as server, backoff_spy():
        config = _config_for(server, max_retries=5)
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                api_key=_API_KEY,
                base_url=server.api_url,
                config=config,
                retry_options=RetryOptions(max_attempts=0),
            )
        await _send_one(client)

    assert server.hits == 1


# ---------------------------------------------------------------------------
# Connector limits
# ---------------------------------------------------------------------------


async def test_keepalive_setting_does_not_cap_per_host_concurrency() -> None:
    """Every SDK request goes to one host, so a per-host limit below
    ``max_connections`` is the effective concurrency cap. aiohttp has no
    keepalive-pool-size knob; the keepalive setting must not be reinterpreted
    as a concurrency limit."""
    config = VeniceAIConfig(
        http_client=HttpClientConfig(max_connections=100, max_keepalive_connections=20)
    )
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(api_key=_API_KEY, config=config)
    try:
        session = await client._get_session()
        connector = session.connector
        assert connector is not None
        per_host = connector.limit_per_host
        assert per_host == 0 or per_host >= config.http_client.max_connections, (
            f"max_keepalive_connections=20 capped per-host concurrency at {per_host} "
            f"although max_connections={config.http_client.max_connections}"
        )
    finally:
        await client.close()


async def test_explicit_connector_limits_are_honoured_alongside_config() -> None:
    config = VeniceAIConfig.create_minimal_config(api_key=_API_KEY)
    with patch.dict(os.environ, {}, clear=True):
        client = VeniceClient(
            api_key=_API_KEY, config=config, connector_limit=7, connector_limit_per_host=3
        )
    try:
        session = await client._get_session()
        connector = session.connector
        assert connector is not None
        assert (connector.limit, connector.limit_per_host) == (7, 3), (
            "explicit connector_limit=7 / connector_limit_per_host=3 were dropped "
            f"because config= was also passed: got {(connector.limit, connector.limit_per_host)}"
        )
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------


async def test_proxy_keyword_routes_requests_through_the_proxy() -> None:
    # The target is a refused local port: the request can only reach a server
    # if it is sent to the proxy.
    unreachable_target = "http://127.0.0.1:1/api/v1"
    async with failing_server(status=404) as proxy, backoff_spy():
        with patch.dict(os.environ, {}, clear=True):
            client = VeniceClient(
                api_key=_API_KEY,
                base_url=unreachable_target,
                proxy=proxy.root_url,
                retry_options=RetryOptions(max_attempts=0),
            )
        await _send_one(client)

    assert proxy.hits >= 1, (
        f"VeniceClient(proxy={proxy.root_url!r}) never sent a request to the proxy"
    )
    assert proxy.hosts[0] == "127.0.0.1:1"


# ---------------------------------------------------------------------------
# Class guard: every HttpClientConfig field has an observable effect
# ---------------------------------------------------------------------------

# A valid non-default value for every field. A new field must be added here,
# which forces a decision about how its effect is observed.
_PERTURBATIONS: dict[str, Any] = {
    "timeout": 17.0,
    "max_connections": 9,
    "max_keepalive_connections": 4,
    "max_retries": 1,
    "retry_backoff_factor": 3.5,
    "user_agent": "wiring-probe/1.0",
}


async def _fingerprint(http_fields: dict[str, Any]) -> tuple[Any, ...]:
    """Everything a caller could observe about the HTTP layer for one request."""
    async with failing_server() as server, backoff_spy() as spy:
        client = _factory_client(_config_for(server, **http_fields))
        async with client:
            session = await client._get_session()
            connector = session.connector
            assert connector is not None
            transport = (session.timeout.total, connector.limit, connector.limit_per_host)
            with pytest.raises(Exception):  # noqa: B017
                await client.get("models")
    backoff = tuple(sorted({tuple(sorted(c.items())) for c in spy.calls}))
    return (transport, tuple(server.user_agents), server.hits, backoff)


def _is_explicitly_retired(name: str) -> bool:
    return bool(HttpClientConfig.model_fields[name].deprecated)


def test_every_http_client_config_field_has_a_perturbation() -> None:
    fields = set(HttpClientConfig.model_fields)
    assert len(fields) > 0
    assert fields == set(_PERTURBATIONS), (
        f"HttpClientConfig fields without a perturbation: {sorted(fields - set(_PERTURBATIONS))}"
    )
    for name, value in _PERTURBATIONS.items():
        assert HttpClientConfig.model_fields[name].get_default(call_default_factory=True) != value


@pytest.mark.parametrize("name", sorted(_PERTURBATIONS))
async def test_every_http_client_config_field_changes_observable_behaviour(name: str) -> None:
    if _is_explicitly_retired(name):
        pytest.skip(f"{name} is marked deprecated")
    # Keep the baseline retrying so retry-shaped fields have something to act on.
    baseline_fields: dict[str, Any] = {"max_retries": 2}
    perturbed_fields = {**baseline_fields, name: _PERTURBATIONS[name]}

    baseline = await _fingerprint(baseline_fields)
    perturbed = await _fingerprint(perturbed_fields)

    assert perturbed != baseline, (
        f"HttpClientConfig.{name}={_PERTURBATIONS[name]!r} changed nothing observable "
        "(timeout, connector limits, User-Agent, attempt count, backoff inputs); "
        "the field is dead configuration"
    )


async def test_fingerprint_is_deterministic() -> None:
    """An unchanged config must fingerprint identically, or the guard above
    would read noise as an effect and pass every field."""
    baseline_fields: dict[str, Any] = {"max_retries": 2}
    assert await _fingerprint(baseline_fields) == await _fingerprint(baseline_fields)


# ---------------------------------------------------------------------------
# Deprecated keepalive setting
# ---------------------------------------------------------------------------


def test_deprecated_keepalive_warning_is_shown_by_default_and_points_at_user_code() -> None:
    """Run in a fresh interpreter with Python's default warning filters (pytest
    installs its own): the warning must be displayed and attributed to the
    caller's line, not to pydantic or SDK internals."""
    import subprocess
    import sys

    script = (
        "from venice_ai.core.config import HttpClientConfig\n"
        "HttpClientConfig(max_keepalive_connections=5)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"},
        check=True,
    )
    assert "<string>:2: FutureWarning: HttpClientConfig.max_keepalive_connections" in proc.stderr


def test_default_keepalive_value_does_not_warn() -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HttpClientConfig()
        HttpClientConfig(max_keepalive_connections=20)
