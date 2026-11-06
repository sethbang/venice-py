"""``ChatChoice.logprobs`` accepts every shape a server might send.

The swagger describes one flat token; OpenAI-compatible servers send
``{"content": [...]}``. No recording holds a non-null value, so neither is
confirmed, and neither may fail the parse of a whole response.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from aiohttp import web

from venice_ai import VeniceClient
from venice_ai.types.api.chat import ChatChoice, ChatCompletionResponse, LogProbToken
from venice_ai.types.api.streaming import ChatCompletionChunkChoice


def _choice(logprobs: Any) -> ChatChoice:
    return ChatChoice.model_validate(
        {
            "index": 0,
            "message": {"role": "assistant", "content": "yes"},
            "finish_reason": "stop",
            "logprobs": logprobs,
        }
    )


FLAT = {
    "token": "yes",
    "logprob": -0.1,
    "bytes": [121, 101, 115],
    "top_logprobs": [{"token": "no", "logprob": -2.5}],
}


def test_flat_form_keeps_attribute_access() -> None:
    logprobs = _choice(FLAT).logprobs
    assert logprobs is not None
    assert logprobs.token == "yes"
    assert logprobs.logprob == -0.1
    assert logprobs.bytes == [121, 101, 115]
    assert logprobs.top_logprobs is not None
    assert logprobs.top_logprobs[0].token == "no"
    assert logprobs.content is None


def test_flat_form_is_one_entry_of_tokens() -> None:
    logprobs = _choice(FLAT).logprobs
    assert logprobs is not None
    assert [(t.token, t.logprob) for t in logprobs.tokens] == [("yes", -0.1)]
    assert isinstance(logprobs.tokens[0], LogProbToken)
    assert logprobs.tokens[0].top_logprobs is not None


def test_content_form_parses_and_reads_through_tokens() -> None:
    logprobs = _choice(
        {"content": [FLAT, {"token": "!", "logprob": -1.0}], "refusal": None}
    ).logprobs
    assert logprobs is not None
    assert logprobs.content is not None
    assert [t.token for t in logprobs.tokens] == ["yes", "!"]
    assert logprobs.token is None
    assert logprobs.model_extra == {"refusal": None}


def test_absent_and_null_stay_none() -> None:
    assert _choice(None).logprobs is None
    choice = ChatChoice.model_validate(
        {"index": 0, "message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}
    )
    assert choice.logprobs is None


@pytest.mark.parametrize(
    "unexpected",
    [
        [FLAT],
        "opaque",
        {"content": "not a list"},
        {"content": [{"no_token": True}]},
        {"token": "yes", "logprob": "not a number"},
    ],
    ids=["list", "string", "content not a list", "content entry", "bad logprob"],
)
def test_unexpected_shape_does_not_fail_the_response(unexpected: Any) -> None:
    response = ChatCompletionResponse.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "still readable"},
                    "finish_reason": "stop",
                    "logprobs": unexpected,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )
    logprobs = response.choices[0].logprobs
    assert response.choices[0].message.content == "still readable"
    assert logprobs is not None
    assert logprobs.tokens == []
    assert logprobs.raw == unexpected
    assert response.choices[0].model_dump()["logprobs"] == unexpected


@pytest.mark.parametrize(
    "sent",
    [FLAT, {"content": [FLAT], "refusal": "no"}],
    ids=["flat", "content"],
)
def test_dump_gives_back_what_was_sent(sent: dict[str, Any]) -> None:
    dumped = _choice(sent).model_dump(exclude_none=True)["logprobs"]
    assert dumped == sent
    assert _choice(sent).logprobs.raw is None  # type: ignore[union-attr]


def test_a_token_instance_is_accepted() -> None:
    choice = ChatChoice(
        index=0,
        message={"role": "assistant", "content": "x"},  # type: ignore[arg-type]
        finish_reason="stop",
        logprobs=LogProbToken(token="x", logprob=-0.5),  # type: ignore[arg-type]
    )
    assert choice.logprobs is not None
    assert choice.logprobs.token == "x"


@pytest.mark.parametrize(
    "sent",
    [FLAT, {"content": [FLAT]}],
    ids=["flat", "content"],
)
def test_stream_chunk_choice_parses_both_forms(sent: dict[str, Any]) -> None:
    choice = ChatCompletionChunkChoice.model_validate(
        {"index": 0, "delta": {"content": "y"}, "logprobs": sent}
    )
    assert choice.logprobs is not None


@pytest.mark.parametrize(
    "unknown",
    [
        {"foo": 1},
        {"tokens": ["a"], "token_logprobs": [-1.0]},
        {"token": "a"},
        {"logprob": -1.0},
    ],
    ids=["unknown key", "legacy completions form", "token only", "logprob only"],
)
def test_a_mapping_tokens_cannot_read_is_kept_in_raw(unknown: dict[str, Any]) -> None:
    choice = _choice(unknown)

    assert choice.logprobs is not None
    assert choice.logprobs.tokens == []
    assert choice.logprobs.raw == unknown
    assert choice.model_dump()["logprobs"] == unknown
    assert ChatChoice.model_validate(choice.model_dump()).logprobs.raw == unknown  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "readable",
    [{}, {"content": []}, FLAT, {"content": [FLAT]}],
    ids=["empty mapping", "no tokens", "flat", "content"],
)
def test_a_form_tokens_can_read_leaves_raw_empty(readable: dict[str, Any]) -> None:
    logprobs = _choice(readable).logprobs

    assert logprobs is not None
    assert logprobs.raw is None


@pytest.mark.parametrize(
    "unexpected",
    [[], "text", 0, {"content": "not a list"}],
    ids=["list", "string", "number", "content not a list"],
)
def test_stream_chunk_choice_keeps_an_unexpected_value_in_raw(unexpected: Any) -> None:
    choice = ChatCompletionChunkChoice.model_validate(
        {"index": 0, "delta": {"content": "y"}, "logprobs": unexpected}
    )

    assert choice.delta.content == "y"
    assert choice.logprobs is not None
    assert choice.logprobs.content is None
    assert choice.logprobs.raw == unexpected
    assert choice.model_dump()["logprobs"] == unexpected


@pytest.mark.parametrize(
    "logprobs_per_chunk",
    [[None, [], None], [[], [], []]],
    ids=["one bad chunk", "every chunk bad"],
)
async def test_bad_logprobs_on_the_wire_lose_no_stream_text(logprobs_per_chunk: list[Any]) -> None:
    async def chat(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for index, logprobs in enumerate(logprobs_per_chunk):
            chunk = {
                "id": "i",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {"content": f"t{index} "}, "logprobs": logprobs}],
            }
            await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post("/api/v1/chat/completions", chat)
    runner = web.AppRunner(app, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    try:
        async with VeniceClient(
            api_key="fabricated", base_url=f"http://127.0.0.1:{port}/api/v1"
        ) as client:
            stream = await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "x"}], stream=True
            )
            async with stream:
                text = [chunk.choices[0].delta.content async for chunk in stream]
    finally:
        await runner.cleanup()

    assert text == ["t0 ", "t1 ", "t2 "]
