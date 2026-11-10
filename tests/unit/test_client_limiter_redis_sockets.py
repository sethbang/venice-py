"""What ``VeniceClient.close`` does to real Redis sockets, against a loopback RESP stand-in."""

import asyncio
import contextlib
from collections.abc import AsyncIterator

import pytest

from venice_ai import VeniceClient
from venice_ai.factory import VeniceClientFactory

pytest.importorskip("adaptive_rate_limiter")
pytest.importorskip("redis")

from adaptive_rate_limiter.backends import RedisBackend  # noqa: E402

from venice_ai.core.config import (  # noqa: E402
    BackendConfig,
    BackendType,
    RedisBackendConfig,
    VeniceAIConfig,
)
from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode  # noqa: E402


class _RedisStandIn:
    """Enough of a Redis server for a client to connect, ping and load scripts."""

    def __init__(self) -> None:
        self.open_connections = 0
        self.port = 0

    async def _reply(self, args: list[bytes]) -> bytes:
        command = args[0].upper()
        subcommand = args[1].upper() if len(args) > 1 else b""
        if command == b"PING":
            return b"+PONG\r\n"
        if command == b"SCRIPT":
            return b"$40\r\n" + b"a" * 40 + b"\r\n"
        if command == b"HELLO":
            return b"%2\r\n$6\r\nserver\r\n$5\r\nredis\r\n$5\r\nproto\r\n:3\r\n"
        if command == b"CLUSTER" and subcommand == b"SLOTS":
            port = str(self.port).encode()
            return (
                b"*1\r\n*3\r\n:0\r\n:16383\r\n*3\r\n$9\r\n127.0.0.1\r\n:"
                + port
                + b"\r\n$4\r\nnode\r\n"
            )
        if command in (b"COMMAND", b"CONFIG", b"CLUSTER"):
            return b"*0\r\n"
        if command == b"INFO":
            return b"$0\r\n\r\n"
        return b"+OK\r\n"

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.open_connections += 1
        try:
            while line := await reader.readline():
                if not line.startswith(b"*"):
                    continue
                args = []
                for _ in range(int(line[1:])):
                    size = int((await reader.readline())[1:])
                    args.append((await reader.readexactly(size + 2))[:-2])
                writer.write(await self._reply(args))
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            self.open_connections -= 1
            writer.close()


@contextlib.asynccontextmanager
async def _redis_stand_in() -> AsyncIterator[_RedisStandIn]:
    """Entered inside the test, so the server runs on the test's own event loop."""
    stand_in = _RedisStandIn()
    server = await asyncio.start_server(stand_in.handle, "127.0.0.1", 0)
    stand_in.port = server.sockets[0].getsockname()[1]
    try:
        yield stand_in
    finally:
        server.close()


@pytest.fixture
def isolated_pools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own class-level pool table, so no socket outlives the test's view."""
    monkeypatch.setattr(RedisBackend, "_connection_pools", {})


def _adaptive_config(port: int, *, cluster_mode: bool) -> VeniceAIConfig:
    url = f"redis://127.0.0.1:{port}"
    return VeniceAIConfig(
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(redis_url=url, cluster_mode=cluster_mode),
        ),
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url=url),
    )


async def _wait_for_connections(stand_in: _RedisStandIn, *, closed: bool) -> None:
    """Let the stand-in's handlers see the sockets open or close, then return."""
    async with asyncio.timeout(5):
        while (stand_in.open_connections == 0) != closed:
            await asyncio.sleep(0.01)


def _backend_of(client: VeniceClient) -> RedisBackend:
    """The backend the factory built: the real adapter, scheduler and state manager."""
    backend = client.rate_limiter._scheduler.state_manager.backend
    assert isinstance(backend, RedisBackend)
    return backend


# adaptive-rate-limiter passes redis-py's deprecated cluster_error_retry_attempts
# when it builds a cluster client. Drop this once its floor here carries the fix.
@pytest.mark.filterwarnings(
    "ignore:Call to '__init__' function with deprecated usage of input argument/s "
    "'cluster_error_retry_attempts':DeprecationWarning"
)
@pytest.mark.usefixtures("isolated_pools")
async def test_close_releases_cluster_connections() -> None:
    async with _redis_stand_in() as stand_in:
        client = VeniceClientFactory.create_client(
            _adaptive_config(stand_in.port, cluster_mode=True), api_key="test-key"
        )
        await _backend_of(client).start()
        await _wait_for_connections(stand_in, closed=False)

        await client.close()

        await _wait_for_connections(stand_in, closed=True)


@pytest.mark.usefixtures("isolated_pools")
async def test_closing_one_client_leaves_another_on_the_same_loop_working() -> None:
    async with _redis_stand_in() as stand_in:
        config = _adaptive_config(stand_in.port, cluster_mode=False)
        first = VeniceClientFactory.create_client(config, api_key="test-key")
        second = VeniceClientFactory.create_client(config, api_key="test-key")
        closed_backend = _backend_of(first)
        surviving = _backend_of(second)
        await closed_backend.start()
        await surviving.start()

        await first.close()

        assert closed_backend._redis is None
        assert await surviving._redis.ping() is True

        await second.close()
        await RedisBackend._connection_pools.popitem()[1].disconnect()
        await _wait_for_connections(stand_in, closed=True)
