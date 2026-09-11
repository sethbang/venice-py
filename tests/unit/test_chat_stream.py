"""Unit tests for ChatStream's text_deltas() and collect() methods."""

from unittest.mock import Mock

import pytest

from venice_ai.streaming import ChatStream


def _chunk(
    *,
    id: str = "chatcmpl-1",
    created: int = 1700000000,
    model: str = "llama-3.3-70b",
    content: str | None = None,
    reasoning_content: str | None = None,
    reasoning_encrypted: bool | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
    choice_index: int = 0,
):
    """Build a ChatCompletionChunk for tests."""
    from venice_ai.types.api.streaming import ChatCompletionChunk

    payload = {
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": choice_index,
                "delta": {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "reasoning_encrypted": reasoning_encrypted,
                    "tool_calls": tool_calls,
                },
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return ChatCompletionChunk.model_validate(payload)


async def _aiter(chunks):
    for c in chunks:
        yield c


def _stream(chunks):
    return ChatStream(_aiter(chunks), client=Mock())


# ---------------------------------------------------------------------------
# text_deltas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_text_deltas_yields_only_content():
    chunks = [
        _chunk(content="Hello"),
        _chunk(content=None),  # filtered
        _chunk(content=""),  # filtered (falsy)
        _chunk(content=" world"),
        _chunk(content=None, finish_reason="stop"),
    ]
    out = [t async for t in _stream(chunks).text_deltas()]
    assert out == ["Hello", " world"]


# ---------------------------------------------------------------------------
# collect — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_text_only():
    chunks = [
        _chunk(content="Hello"),
        _chunk(content=" world"),
        _chunk(
            content=None,
            finish_reason="stop",
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        ),
    ]
    response = await _stream(chunks).collect()
    assert response.id == "chatcmpl-1"
    assert response.created == 1700000000
    assert response.model == "llama-3.3-70b"
    assert response.choices[0].message.content == "Hello world"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.total_tokens == 7


@pytest.mark.asyncio
async def test_collect_accumulates_reasoning_content():
    chunks = [
        _chunk(reasoning_content="thinking..."),
        _chunk(reasoning_content=" more"),
        _chunk(content="Final answer", finish_reason="stop"),
    ]
    response = await _stream(chunks).collect()
    assert response.choices[0].message.reasoning_content == "thinking... more"
    assert response.choices[0].message.content == "Final answer"


@pytest.mark.asyncio
async def test_collect_merges_parallel_tool_calls_by_index():
    # Two parallel tool calls — interleaved deltas with index 0 and 1.
    chunks = [
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                },
                {
                    "index": 1,
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": ""},
                },
            ]
        ),
        _chunk(
            tool_calls=[
                {"index": 0, "function": {"arguments": '{"city":'}},
                {"index": 1, "function": {"arguments": '{"tz":'}},
            ]
        ),
        _chunk(
            tool_calls=[
                {"index": 0, "function": {"arguments": ' "NYC"}'}},
                {"index": 1, "function": {"arguments": ' "UTC"}'}},
            ],
            finish_reason="tool_calls",
        ),
    ]
    response = await _stream(chunks).collect()
    calls = response.choices[0].message.tool_calls
    assert calls is not None and len(calls) == 2
    assert calls[0].id == "call_a"
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == '{"city": "NYC"}'
    assert calls[1].id == "call_b"
    assert calls[1].function.name == "get_time"
    assert calls[1].function.arguments == '{"tz": "UTC"}'
    assert response.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_collect_usage_is_none_when_no_usage_chunk():
    chunks = [
        _chunk(content="hi"),
        _chunk(content=None, finish_reason="stop"),
    ]
    response = await _stream(chunks).collect()
    assert response.usage is None


@pytest.mark.asyncio
async def test_collect_propagates_id_and_created_from_first_chunk():
    chunks = [
        _chunk(id="chatcmpl-abc", created=1234567890, content="hi"),
        _chunk(id="chatcmpl-abc", created=1234567890, content=None, finish_reason="stop"),
    ]
    response = await _stream(chunks).collect()
    assert response.id == "chatcmpl-abc"
    assert response.created == 1234567890


# ---------------------------------------------------------------------------
# collect — error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_raises_when_no_finish_reason():
    chunks = [_chunk(content="incomplete")]  # no finish_reason ever
    with pytest.raises(ValueError, match="finish_reason"):
        await _stream(chunks).collect()


@pytest.mark.asyncio
async def test_collect_warns_on_n_greater_than_one():
    # Same chunk carries choice index 0 AND a chunk with index 1.
    chunks = [
        _chunk(content="hi", choice_index=0),
        _chunk(content="ignored", choice_index=1),
        _chunk(content=None, finish_reason="stop", choice_index=0),
    ]
    with pytest.warns(UserWarning, match="only tracks choice 0"):
        response = await _stream(chunks).collect()
    # Choice-0 content preserved, choice-1 dropped.
    assert response.choices[0].message.content == "hi"


# ---------------------------------------------------------------------------
# collect_with_deltas
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_with_deltas_yields_text_and_populates_final_response():
    chunks = [
        _chunk(content="Hello"),
        _chunk(content=" "),
        _chunk(content="world"),
        _chunk(
            content=None,
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        ),
    ]
    stream = _stream(chunks)
    deltas: list[str] = []
    async for text in stream.collect_with_deltas():
        deltas.append(text)
    assert deltas == ["Hello", " ", "world"]
    assert stream.final_response is not None
    assert stream.final_response.choices[0].message.content == "Hello world"
    assert stream.final_response.choices[0].finish_reason == "stop"
    assert stream.final_response.usage is not None
    assert stream.final_response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_collect_with_deltas_final_response_none_until_iteration_completes():
    chunks = [
        _chunk(content="a"),
        _chunk(content="b", finish_reason="stop"),
    ]
    stream = _stream(chunks)
    # Before iteration begins
    assert stream.final_response is None

    gen = stream.collect_with_deltas()
    # Pull one delta — final_response should still be None mid-stream.
    first = await gen.__anext__()
    assert first == "a"
    assert stream.final_response is None

    # Drain the rest.
    rest = [t async for t in gen]
    assert rest == ["b"]
    assert stream.final_response is not None


@pytest.mark.asyncio
async def test_collect_with_deltas_skips_reasoning_in_yielded_deltas_but_keeps_in_final():
    chunks = [
        _chunk(reasoning_content="thinking..."),
        _chunk(content="answer"),
        _chunk(content=None, finish_reason="stop"),
    ]
    stream = _stream(chunks)
    deltas = [t async for t in stream.collect_with_deltas()]
    # Only text deltas yielded.
    assert deltas == ["answer"]
    # But reasoning_content makes it into the final response.
    assert stream.final_response is not None
    assert stream.final_response.choices[0].message.reasoning_content == "thinking..."


@pytest.mark.asyncio
async def test_collect_with_deltas_accumulates_tool_calls_into_final():
    # Tool-call deltas should NOT be yielded as text but should appear
    # assembled in final_response.choices[0].message.tool_calls.
    chunks = [
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":'},
                }
            ]
        ),
        _chunk(
            tool_calls=[
                {
                    "index": 0,
                    "function": {"arguments": ' "venice"}'},
                }
            ]
        ),
        _chunk(content=None, finish_reason="tool_calls"),
    ]
    stream = _stream(chunks)
    deltas = [t async for t in stream.collect_with_deltas()]
    assert deltas == []  # No text content arrived.
    assert stream.final_response is not None
    tcs = stream.final_response.choices[0].message.tool_calls
    assert tcs is not None and len(tcs) == 1
    assert tcs[0].function.name == "lookup"
    assert tcs[0].function.arguments == '{"q": "venice"}'


@pytest.mark.asyncio
async def test_collect_with_deltas_raises_when_no_finish_reason():
    chunks = [
        _chunk(content="incomplete"),
        # No finish_reason on any chunk → interrupted stream.
    ]
    stream = _stream(chunks)
    with pytest.raises(ValueError, match="finish_reason"):
        async for _ in stream.collect_with_deltas():
            pass


@pytest.mark.asyncio
async def test_collect_also_populates_final_response():
    # collect() should populate final_response so that
    # `stream.final_response is response` after the call.
    chunks = [
        _chunk(content="hello"),
        _chunk(content=None, finish_reason="stop"),
    ]
    stream = _stream(chunks)
    response = await stream.collect()
    assert stream.final_response is response


@pytest.mark.asyncio
async def test_text_deltas_does_not_populate_final_response():
    # text_deltas() is the display-only path; final_response stays None.
    chunks = [
        _chunk(content="hi"),
        _chunk(content=None, finish_reason="stop"),
    ]
    stream = _stream(chunks)
    deltas = [t async for t in stream.text_deltas()]
    assert deltas == ["hi"]
    assert stream.final_response is None


# ---------------------------------------------------------------------------
# ChatCompletionChunk.text
# ---------------------------------------------------------------------------


class TestChunkTextProperty:
    """Convenience accessor for ``choices[0].delta.content``."""

    def test_returns_content_string(self):
        c = _chunk(content="hello")
        assert c.text == "hello"

    def test_returns_empty_string_when_content_none(self):
        # Last chunk often only carries finish_reason with delta.content=None.
        c = _chunk(content=None, finish_reason="stop")
        assert c.text == ""

    def test_returns_empty_string_when_no_choices(self):
        from venice_ai.types.api.streaming import ChatCompletionChunk

        c = ChatCompletionChunk(
            id="x",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[],
        )
        assert c.text == ""

    def test_picks_first_choice_when_multiple(self):
        # If the API returns parallel completions, .text mirrors the first
        # choice (same convention as ChatCompletionResponse.text).
        from venice_ai.types.api.streaming import (
            ChatCompletionChunk,
            ChatCompletionChunkChoice,
            ChatCompletionChunkChoiceDelta,
        )

        c = ChatCompletionChunk(
            id="x",
            object="chat.completion.chunk",
            created=0,
            model="m",
            choices=[
                ChatCompletionChunkChoice(
                    index=0, delta=ChatCompletionChunkChoiceDelta(content="first")
                ),
                ChatCompletionChunkChoice(
                    index=1, delta=ChatCompletionChunkChoiceDelta(content="second")
                ),
            ],
        )
        assert c.text == "first"


# ---------------------------------------------------------------------------
# encrypted reasoning blocks
# ---------------------------------------------------------------------------

# An encrypted reasoning block arrives whole in one delta, flagged with
# ``reasoning_encrypted``. It is unterminated, so the server reads it from its
# header to the end of the field: any summary the model streams afterwards must
# not be concatenated onto it, or the next turn is rejected with "encrypted
# content could not be decrypted or parsed".
_ENVELOPE = "\n\n__ENCRYPTED_REASONING__id=rs_abc\ngAAAAABtoken=="


def _interleaved_chunks():
    """Summary, then the encrypted block, then *more* summary — the failing shape."""
    return [
        _chunk(reasoning_content="Planning the answer. "),
        _chunk(reasoning_content=_ENVELOPE, reasoning_encrypted=True),
        _chunk(reasoning_content="Still thinking."),
        _chunk(content="Done.", finish_reason="stop"),
    ]


@pytest.mark.asyncio
async def test_collect_puts_encrypted_block_last_so_it_round_trips():
    response = await _stream(_interleaved_chunks()).collect()

    reasoning = response.choices[0].message.reasoning_content
    assert reasoning is not None
    assert reasoning.endswith(_ENVELOPE), "summary must never trail the encrypted block"
    assert reasoning == "Planning the answer. Still thinking." + _ENVELOPE


@pytest.mark.asyncio
async def test_collect_preserves_every_reasoning_delta():
    response = await _stream(_interleaved_chunks()).collect()

    reasoning = response.choices[0].message.reasoning_content or ""
    assert len(reasoning) == len("Planning the answer. ") + len(_ENVELOPE) + len("Still thinking.")


@pytest.mark.asyncio
async def test_collect_surfaces_reasoning_encrypted_flag():
    response = await _stream(_interleaved_chunks()).collect()

    # Mirrors the non-streaming endpoint, which flags the message the same way.
    assert response.choices[0].message.model_extra.get("reasoning_encrypted") is True


@pytest.mark.asyncio
async def test_reasoning_summary_excludes_the_encrypted_block():
    stream = _stream(_interleaved_chunks())
    await stream.collect()

    assert stream.reasoning_summary == "Planning the answer. Still thinking."
    assert "__ENCRYPTED_REASONING__" not in (stream.reasoning_summary or "")


@pytest.mark.asyncio
async def test_collect_with_deltas_puts_encrypted_block_last():
    stream = _stream(_interleaved_chunks())
    async for _ in stream.collect_with_deltas():
        pass

    assert stream.final_response is not None
    reasoning = stream.final_response.choices[0].message.reasoning_content
    assert reasoning is not None
    assert reasoning.endswith(_ENVELOPE)
    assert stream.reasoning_summary == "Planning the answer. Still thinking."


@pytest.mark.asyncio
async def test_unflagged_reasoning_keeps_arrival_order():
    """Models that emit no encrypted block are unaffected."""
    stream = _stream(
        [_chunk(reasoning_content="a"), _chunk(reasoning_content="b", finish_reason="stop")]
    )
    response = await stream.collect()

    assert response.choices[0].message.reasoning_content == "ab"
    assert "reasoning_encrypted" not in response.choices[0].message.model_extra
    assert stream.reasoning_summary == "ab"
