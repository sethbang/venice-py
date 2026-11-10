#!/usr/bin/env python3
"""
Venice AI SDK - Testing Your Code Offline
=========================================

How to test code that calls Venice without a network connection, without an API
key and without spending anything. The client is a real ``VeniceClient``, so
the SDK's request building, response parsing, error mapping and retries all
run. It only talks to a small local server that stands in for the API:

    VeniceClient(config=create_testing_config(), api_key="test-key", base_url=local_url)

The script has three parts:

- ``summarize()`` and ``stream_summary()``: example application code, the thing
  under test. They take the client as an argument, which is what makes them
  testable.
- ``stand_in_api()``: a local aiohttp server. It plays back the replies a test
  scripts and records every request it receives.
- ``test_*``: checks on what the code sent and what it did with each reply. Copy
  this shape into your own test suite.

Points to carry over:

- Pass ``api_key`` and ``base_url`` explicitly. An explicit key means the tests
  need no ``VENICE_API_KEY`` and never read a real one, and an explicit URL wins
  over ``VENICE_API_BASE_URL``.
- ``create_testing_config()`` sets ``http_client.max_retries=1``, which
  ``VeniceClient`` uses. It also sets ``http_client.timeout``, which only the
  factory passes on, so give ``VeniceClient`` a ``timeout`` of its own.
- ``create_test_venice_client()`` is not for this: it takes no ``base_url``, so
  unless ``VENICE_API_BASE_URL`` is set it talks to the real API.
- A stand-in shows what your code and the SDK do with a response. It cannot show
  that Venice sends that response, so keep the replies close to real ones.

Cost: free. No API key needed, and nothing leaves this machine.

Exit status: ``0`` when every test passed, ``1`` when any test failed.
"""

import asyncio
import contextlib
import dataclasses
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

from aiohttp import web

from venice_ai import (
    AuthenticationError,
    RateLimitError,
    UserMessage,
    VeniceClient,
    VeniceError,
    create_testing_config,
)
from venice_ai.types.api.requests import VeniceParameters

# =============================================================================
# The application code under test
# =============================================================================

MAX_SUMMARY_TOKENS = 128


class SummaryError(Exception):
    """The model's reply cannot be used as a summary."""


class SummaryUnavailableError(Exception):
    """The API turned the request away; try again after ``retry_after`` seconds."""

    def __init__(self, retry_after: int | None) -> None:
        super().__init__(f"Summaries are rate limited (retry after {retry_after}s)")
        self.retry_after = retry_after


def _summary_request(text: str) -> dict[str, Any]:
    return {
        "messages": [UserMessage(content=f"Summarize in one sentence: {text}")],
        "max_completion_tokens": MAX_SUMMARY_TOKENS,
        "venice_parameters": VeniceParameters(include_venice_system_prompt=False),
    }


async def summarize(client: VeniceClient, text: str) -> str:
    """Summarize ``text`` on the cheapest model that answers directly."""
    model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    try:
        response = await client.chat.completions.create(model=model, **_summary_request(text))
    except RateLimitError as e:
        raise SummaryUnavailableError(e.retry_after_seconds) from e

    finish_reason = response.choices[0].finish_reason if response.choices else None
    if not response.text or finish_reason == "length":
        raise SummaryError(f"Unusable reply (finish_reason={finish_reason})")
    return response.text.strip()


async def stream_summary(client: VeniceClient, text: str) -> str:
    """The same summary, streamed and collected into one response."""
    model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    async with await client.chat.completions.stream(
        model=model, **_summary_request(text)
    ) as stream:
        response = await stream.collect()
    if not response.text:
        raise SummaryError("The stream carried no text")
    return response.text.strip()


# =============================================================================
# The stand-in API
# =============================================================================

TEST_KEY = "test-key"

# The catalog the stand-in serves. The cheapest entry reasons, so application
# code asking for a direct answer has to pick the next one up. These IDs exist
# only on this server.
CHEAPEST_REASONING = "stand-in-cheapest-reasoning"
CHEAPEST_DIRECT = "stand-in-cheapest-direct"
PRICIER_DIRECT = "stand-in-pricier-direct"


def _catalog_entry(model_id: str, *, reasoning: bool, input_usd: float) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "venice.ai",
        "type": "text",
        "model_spec": {
            "name": model_id,
            "capabilities": {
                "supportsReasoning": reasoning,
                "optimizedForCode": False,
                "quantization": "not-applicable",
                "supportsFunctionCalling": False,
                "supportsResponseSchema": False,
                "supportsVision": False,
                "supportsWebSearch": False,
                "supportsLogProbs": False,
            },
            "pricing": {
                "input": {"usd": input_usd, "diem": input_usd},
                "output": {"usd": input_usd * 2, "diem": input_usd * 2},
            },
        },
    }


CATALOG = {
    "object": "list",
    "type": "text",
    "data": [
        _catalog_entry(PRICIER_DIRECT, reasoning=False, input_usd=1.0),
        _catalog_entry(CHEAPEST_DIRECT, reasoning=False, input_usd=0.2),
        _catalog_entry(CHEAPEST_REASONING, reasoning=True, input_usd=0.1),
    ],
}


@dataclasses.dataclass(frozen=True)
class Reply:
    """One scripted answer: JSON by default, Server-Sent Events if ``events`` is set."""

    status: int = 200
    body: dict[str, Any] | None = None
    headers: dict[str, str] = dataclasses.field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()


def chat_reply(text: str, finish_reason: str = "stop") -> Reply:
    """A chat completion, shaped like the API's."""
    return Reply(
        body={
            "id": "chatcmpl-stand-in",
            "object": "chat.completion",
            "created": 0,
            "model": CHEAPEST_DIRECT,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
    )


def chat_stream_reply(*pieces: str) -> Reply:
    """A chat stream: one chunk per piece of text, then a chunk that finishes it."""

    def chunk(delta: dict[str, str], finish_reason: str | None) -> dict[str, Any]:
        return {
            "id": "chatcmpl-stand-in",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": CHEAPEST_DIRECT,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    events = [chunk({"role": "assistant", "content": piece}, None) for piece in pieces]
    return Reply(events=(*events, chunk({}, "stop")))


def error_reply(status: int, message: str, headers: dict[str, str] | None = None) -> Reply:
    """An error, in the ``{"error": message}`` form the API uses."""
    return Reply(status=status, body={"error": message}, headers=headers or {})


@dataclasses.dataclass(frozen=True)
class RecordedRequest:
    """What the stand-in received for one chat request."""

    authorization: str | None
    body: dict[str, Any]


class StandInAPI:
    """Handles to a running stand-in: its URL and the chat requests it received."""

    def __init__(self, base_url: str, chat_requests: list[RecordedRequest]) -> None:
        self.base_url = base_url
        self.chat_requests = chat_requests


@contextlib.asynccontextmanager
async def stand_in_api(*replies: Reply) -> AsyncIterator[StandInAPI]:
    """Serve ``CATALOG`` and play back ``replies`` to chat requests, in order.

    A chat request beyond the end of the script gets a 500, so a test that sends
    more than it expected shows up as a failure.
    """
    chat_requests: list[RecordedRequest] = []
    remaining = list(replies)

    async def models(request: web.Request) -> web.Response:
        return web.json_response(CATALOG)

    async def chat(request: web.Request) -> web.StreamResponse:
        chat_requests.append(
            RecordedRequest(request.headers.get("Authorization"), await request.json())
        )
        reply = remaining.pop(0) if remaining else error_reply(500, "Script ran out of replies")
        if not reply.events:
            return web.json_response(reply.body, status=reply.status, headers=reply.headers)

        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        for event in reply.events:
            await stream.write(f"data: {json.dumps(event)}\n\n".encode())
        await stream.write(b"data: [DONE]\n\n")
        await stream.write_eof()
        return stream

    app = web.Application()
    app.router.add_get("/api/v1/models", models)
    app.router.add_post("/api/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield StandInAPI(f"http://127.0.0.1:{port}/api/v1", chat_requests)
    finally:
        await runner.cleanup()


def offline_client(api: StandInAPI) -> VeniceClient:
    """A real client that can only reach the stand-in, with a key that is not real."""
    return VeniceClient(
        config=create_testing_config(), api_key=TEST_KEY, base_url=api.base_url, timeout=5.0
    )


# =============================================================================
# The tests
# =============================================================================


class CheckFailedError(Exception):
    """A test's expectation did not hold."""


def expect(condition: bool, description: str) -> None:
    """Print one check, and stop the test if it failed."""
    if not condition:
        raise CheckFailedError(description)
    print(f"   ✓ {description}")


TEXT = "The Rialto is the oldest of the four bridges over the Grand Canal."


async def test_sends_the_intended_request(client: VeniceClient, api: StandInAPI) -> None:
    """Check what left the application, not just what came back."""
    summary = await summarize(client, TEXT)

    expect(summary == "A very old bridge.", "returns the reply's text, stripped")
    expect(len(api.chat_requests) == 1, "makes exactly one chat request")
    sent = api.chat_requests[0]
    expect(
        sent.body.get("model") == CHEAPEST_DIRECT, "picks the cheapest model that answers directly"
    )
    expect(sent.body.get("max_completion_tokens") == MAX_SUMMARY_TOKENS, "sends its token cap")
    expect(TEXT in sent.body["messages"][0]["content"], "sends the text to summarize")
    expect(
        sent.body.get("venice_parameters", {}).get("include_venice_system_prompt") is False,
        "leaves out Venice's default system prompt",
    )
    expect(sent.authorization == f"Bearer {TEST_KEY}", "authenticates with the test key")


async def test_rejects_a_cut_off_reply(client: VeniceClient, api: StandInAPI) -> None:
    """A truncated reply must not be passed off as a summary."""
    try:
        summary = await summarize(client, TEXT)
    except SummaryError as e:
        expect("length" in str(e), f"raises SummaryError: {e}")
    else:
        expect(False, f"raises SummaryError, but returned {summary!r}")


async def test_reports_when_to_retry_after_a_429(client: VeniceClient, api: StandInAPI) -> None:
    """A 429 is not retried by a bare client, so the application handles it."""
    try:
        summary = await summarize(client, TEXT)
    except SummaryUnavailableError as e:
        expect(e.retry_after == 7, "turns Retry-After: 7 into retry_after == 7")
    else:
        expect(False, f"raises SummaryUnavailableError, but returned {summary!r}")
    expect(len(api.chat_requests) == 1, "does not resend the request")


async def test_survives_one_503(client: VeniceClient, api: StandInAPI) -> None:
    """The testing config allows one retry, which covers a single 503."""
    summary = await summarize(client, TEXT)

    expect(summary == "Recovered.", "returns the reply that followed the 503")
    expect(len(api.chat_requests) == 2, "the SDK made one retry, as max_retries=1 allows")


async def test_lets_a_bad_key_surface(client: VeniceClient, api: StandInAPI) -> None:
    """Nothing in ``summarize()`` should turn a bad key into a quiet failure."""
    try:
        summary = await summarize(client, TEXT)
    except AuthenticationError as e:
        expect(e.status_code == 401, "raises AuthenticationError with status 401")
    else:
        expect(False, f"raises AuthenticationError, but returned {summary!r}")
    expect(len(api.chat_requests) == 1, "does not resend the request")


async def test_reads_a_stream(client: VeniceClient, api: StandInAPI) -> None:
    """Scripted Server-Sent Events go through the SDK's real stream parser."""
    summary = await stream_summary(client, TEXT)

    expect(summary == "A very old bridge.", "joins the stream's pieces into the summary")
    expect(api.chat_requests[0].body.get("stream") is True, "asks for a stream")


TESTS: Sequence[tuple[Callable[[VeniceClient, StandInAPI], Awaitable[None]], tuple[Reply, ...]]] = (
    (test_sends_the_intended_request, (chat_reply("  A very old bridge.  "),)),
    (test_rejects_a_cut_off_reply, (chat_reply("The Rialto is", finish_reason="length"),)),
    (
        test_reports_when_to_retry_after_a_429,
        (error_reply(429, "Too many requests", {"Retry-After": "7"}),),
    ),
    (
        test_survives_one_503,
        (error_reply(503, "The model is at capacity"), chat_reply("Recovered.")),
    ),
    (test_lets_a_bad_key_surface, (error_reply(401, "Authentication failed"),)),
    (test_reads_a_stream, (chat_stream_reply("A very ", "old bridge."),)),
)


async def run_test(
    test: Callable[[VeniceClient, StandInAPI], Awaitable[None]], replies: tuple[Reply, ...]
) -> bool:
    """Run one test with its own server and client, so tests share no state."""
    print(f"\n▶ {test.__name__}")
    async with stand_in_api(*replies) as api, offline_client(api) as client:
        try:
            await test(client, api)
        except CheckFailedError as e:
            print(f"   ❌ {e}")
            return False
        except (SummaryError, SummaryUnavailableError, VeniceError) as e:
            print(f"   ❌ Unexpected {type(e).__name__}: {e}")
            return False
    return True


async def main() -> int:
    """Run every test and return the process exit code."""
    print("🧪 Venice AI Offline Testing Example")
    print("=" * 40)
    print("No API key needed, nothing leaves this machine, nothing is billed.")

    results = [await run_test(test, replies) for test, replies in TESTS]

    passed = sum(results)
    print("\n" + "=" * 40)
    if passed != len(results):
        print(f"❌ {passed} of {len(results)} tests passed")
        return 1
    print(f"✅ All {passed} tests passed, without a network or an API key")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
