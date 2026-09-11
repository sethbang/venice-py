"""Replay real captured SSE through the client's own parsing path.

Every other streaming test builds chunks from a fixed keyword list, so it can
only exercise fields the SDK already models — a field Venice sends but the SDK
has not modelled is invisible to them, because pydantic drops unknown keys on
the delta without erroring. That blind spot is how a dropped
``reasoning_encrypted`` flag once reached users as a 400 on the following turn.

These tests read unedited wire bytes and push them through
``VeniceClient._process_stream_line`` — the same code path a live stream takes —
so the fixture, not the test author, decides which fields appear.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from venice_ai._client import VeniceClient
from venice_ai.streaming import ChatStream
from venice_ai.types.api.streaming import ChatCompletionChunk

FIXTURE = Path(__file__).parent / "fixtures" / "chat_stream_encrypted_reasoning.sse"
SENTINEL = "__ENCRYPTED_REASONING__"


async def _chunks_from_wire():
    """Parse the fixture exactly as a live response body is parsed."""
    client = Mock(spec=VeniceClient)
    for line in FIXTURE.read_text().splitlines():
        async for chunk in VeniceClient._process_stream_line(client, line, ChatCompletionChunk):
            yield chunk


async def _deltas_from_wire():
    """Yield each choice-0 delta. The final usage chunk carries no choices."""
    async for chunk in _chunks_from_wire():
        if chunk.choices:
            yield chunk.choices[0].delta


async def _reasoning_deltas() -> tuple[list[str], str]:
    """Return every reasoning delta in arrival order, plus the encrypted block."""
    parts: list[str] = []
    block = ""
    async for delta in _deltas_from_wire():
        if not delta.reasoning_content:
            continue
        parts.append(delta.reasoning_content)
        if delta.reasoning_encrypted:
            block = delta.reasoning_content
    return parts, block


async def _collected() -> tuple[ChatStream, object]:
    stream = ChatStream(_chunks_from_wire(), client=Mock())
    response = await stream.collect()
    return stream, response


@pytest.mark.asyncio
async def test_wire_carries_exactly_one_encrypted_reasoning_delta():
    """The flag must survive the wire -> pydantic hop.

    This is the assertion the suite was missing. It fails if the field is
    dropped from the delta model, or if the API renames it — neither of which a
    hand-built chunk can detect.
    """
    flagged = [d async for d in _deltas_from_wire() if d.reasoning_encrypted]

    assert len(flagged) == 1
    assert flagged[0].reasoning_content is not None
    assert SENTINEL in flagged[0].reasoning_content


@pytest.mark.asyncio
async def test_summary_deltas_arrive_after_the_encrypted_block():
    """Guards the fixture itself: drop these and the bug stops reproducing."""
    seen_block = False
    summary_after = 0
    async for delta in _deltas_from_wire():
        if delta.reasoning_encrypted:
            seen_block = True
        elif seen_block and delta.reasoning_content:
            summary_after += 1

    assert summary_after > 0, "fixture no longer reproduces the interleaved shape"


@pytest.mark.asyncio
async def test_naive_arrival_order_join_would_corrupt_the_token():
    """Pins why the ordering exists: the obvious implementation is broken."""
    parts, block = await _reasoning_deltas()

    naive = "".join(parts)
    assert SENTINEL in naive
    # Prose trails the token under arrival order. The block is unterminated, so
    # the server reads it to end-of-string and that prose corrupts it — exactly
    # what produced "encrypted content could not be decrypted or parsed".
    assert not naive.endswith(block)


@pytest.mark.asyncio
async def test_assembled_reasoning_ends_on_the_encrypted_block():
    _, response = await _collected()
    _, block = await _reasoning_deltas()

    reasoning = response.choices[0].message.reasoning_content
    assert reasoning is not None
    assert reasoning.endswith(block)
    assert reasoning.count(SENTINEL) == 1


@pytest.mark.asyncio
async def test_no_reasoning_delta_is_lost():
    stream, response = await _collected()
    parts, _ = await _reasoning_deltas()

    assert len(response.choices[0].message.reasoning_content or "") == sum(map(len, parts))
    assert stream.reasoning_summary is not None
    assert SENTINEL not in stream.reasoning_summary


@pytest.mark.asyncio
async def test_assembled_message_is_flagged_like_the_non_streaming_endpoint():
    _, response = await _collected()

    assert response.choices[0].message.model_extra.get("reasoning_encrypted") is True
