"""Integration tests for E2EE on ``chat.completions.create``.

The real flow is wired end-to-end with mocks for the two collaborators that
touch the network or generate keys:

* ``client.tee.open_session`` → returns a real :class:`TeeSession` built around a
  synthetic, verification-passing attestation whose model keypair we control, so
  encrypt/decrypt actually round-trips.
* ``client._stream_request`` → an async generator that asserts the outgoing body
  (encrypted user/system/assistant content, ``stream`` forced true)
  and emits server-encrypted SSE chunks (encrypted to the SESSION pub).

These exercise: the FAIL-LOUD guards (non-``e2ee-`` model, tools, web search /
scraping, logprobs, messages E2EE cannot carry), the wire shape, ``stream=False`` reassembly,
``stream=True`` decrypted deltas, the ``enable_e2ee`` driver, and the per-call
attestation-trust :class:`UserWarning`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

# cryptography is the optional [e2ee] extra; the session round-trip needs it.
ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee chat e2ee tests require the [e2ee] extra (cryptography)",
)

from venice_ai.exceptions import InvalidRequestError, TeeEncryptionError  # noqa: E402
from venice_ai.resources.chat.completions import (  # noqa: E402
    ChatCompletions,
    _E2EEChunkIterator,
)
from venice_ai.streaming import ChatStream  # noqa: E402
from venice_ai.tee import _crypto  # noqa: E402
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.tee.types import TeeOptions, TeeTrust  # noqa: E402
from venice_ai.types.api import ChatCompletionResponse  # noqa: E402
from venice_ai.types.api.chat import ChatMessage  # noqa: E402
from venice_ai.types.api.requests.common import VeniceParameters  # noqa: E402
from venice_ai.types.api.streaming import ChatCompletionChunk  # noqa: E402

_CURVE = ec.SECP256K1()

_E2EE_MODEL = "e2ee-gemma-3-27b-p"


def _build_session(trust: TeeTrust | None = None) -> tuple[TeeSession, ec.EllipticCurvePrivateKey]:
    """A real TeeSession around a model keypair we control (for round-trips)."""
    model_priv = ec.generate_private_key(_CURVE)
    model_pub_hex = _crypto.uncompressed_hex(model_priv.public_key())
    session_priv = _crypto.generate_session_keypair()
    session = TeeSession(
        session_private_key=session_priv,
        model_public_key_hex=model_pub_hex,
        signing_algo="ecdsa",
        trust=trust,
    )
    return session, model_priv


_REALISTIC_USAGE: dict[str, Any] = {
    "prompt_tokens": 3,
    "completion_tokens": 2,
    "total_tokens": 5,
    "prompt_tokens_details": {"cached_tokens": 1},
    "completion_tokens_details": {"reasoning_tokens": 0},
    "cache_read_input_tokens": 1,
}


#: Fields a live Venice E2EE stream sends as ``None`` on every chunk / choice.
_LIVE_NONE_CHUNK_FIELDS: dict[str, Any] = {
    "prompt_text": None,
    "prompt_token_ids": None,
    "kv_transfer_params": None,
}
_LIVE_NONE_CHOICE_FIELDS: dict[str, Any] = {
    "matched_stop": None,
    "stop_reason": None,
    "token_ids": None,
    "weight_versions": None,
}


def _make_completions(
    session: TeeSession,
    captured: dict[str, Any],
    deltas: list[list[dict[str, Any]]] | None = None,
    chunk_fields: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
) -> ChatCompletions:
    """A ChatCompletions whose client mocks tee.open_session + _stream_request.

    ``captured`` collects the outgoing body / headers and the streamed text for
    assertions. The mocked stream encrypts each delta to the SESSION pub so the
    decrypting wrapper produces real plaintext. When ``deltas`` is given, each
    entry is the full ``choices`` list of one chunk, emitted between the role
    chunk and the finish chunk in place of the default content chunks.
    ``chunk_fields`` are added at the top level of each of those content chunks.
    ``usage`` replaces the usage chunk's ``usage`` object.

    The stream has the shape of a live Venice E2EE stream: every chunk carries
    ``prompt_text``, ``prompt_token_ids`` and ``kv_transfer_params`` as
    ``None``, every choice carries ``stop_reason``, ``token_ids`` and
    ``weight_versions`` as ``None`` and ``matched_stop`` (``None`` until the
    finish chunk, which has a token id), and the usage chunk carries a numeric
    ``cost`` and every live-measured ``usage`` field.
    """
    client = MagicMock()
    client.tee.open_session = AsyncMock(return_value=session)

    session_pub = session.session_public_key_hex

    def _stream_request(
        *, method: str, path: str, json_data: dict[str, Any], cast_to: Any, **kw: Any
    ) -> AsyncIterator[ChatCompletionChunk]:
        captured["body"] = json_data
        captured["headers"] = kw.get("headers")

        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            # First chunk: role only (plaintext passthrough).
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chatcmpl-x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _E2EE_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            **_LIVE_NONE_CHOICE_FIELDS,
                        }
                    ],
                    **_LIVE_NONE_CHUNK_FIELDS,
                }
            )
            # Content chunks: each independently encrypted to the SESSION pub.
            if deltas is None:
                chunk_choices = [
                    [
                        {
                            "index": 0,
                            "delta": {"content": _crypto.encrypt_message(session_pub, piece)},
                        }
                    ]
                    for piece in ("Hello", " world")
                ]
            else:
                chunk_choices = deltas
            for choices in chunk_choices:
                choices = [{**_LIVE_NONE_CHOICE_FIELDS, **choice} for choice in choices]
                yield ChatCompletionChunk.model_validate(
                    {
                        "id": "chatcmpl-x",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": _E2EE_MODEL,
                        "choices": choices,
                        **_LIVE_NONE_CHUNK_FIELDS,
                        **(chunk_fields or {}),
                    }
                )
            # Final chunk: finish_reason + usage (plaintext passthrough).
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chatcmpl-x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _E2EE_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                            **_LIVE_NONE_CHOICE_FIELDS,
                            "matched_stop": 151645,
                        }
                    ],
                    **_LIVE_NONE_CHUNK_FIELDS,
                    "usage": _REALISTIC_USAGE if usage is None else usage,
                    "cost": {"usd": 0.0001, "diem": 0},
                }
            )

        return _gen()

    client._stream_request = MagicMock(side_effect=_stream_request)
    return ChatCompletions(client)


def _messages() -> list[Any]:
    return [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prior reply"},
    ]


# --- FAIL-LOUD guards (no network) -------------------------------------------


@pytest.mark.asyncio
async def test_e2ee_on_non_e2ee_model_raises() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match="e2ee-"):
        await comp.create(model="llama-3.3-70b", messages=_messages(), e2ee=True)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_with_tools_raises() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    tools = [
        {
            "type": "function",
            "function": {"name": "f", "description": "d", "parameters": {}},
        }
    ]
    with pytest.raises(InvalidRequestError, match="(?i)tool"):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, tools=tools)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_with_web_search_raises() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match="(?i)web"):
        await comp.create(
            model=_E2EE_MODEL,
            messages=_messages(),
            e2ee=True,
            venice_parameters={"enable_web_search": "on"},
        )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_with_web_scraping_raises() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match="(?i)scrap"):
        await comp.create(
            model=_E2EE_MODEL,
            messages=_messages(),
            e2ee=True,
            venice_parameters={"enable_web_scraping": True},
        )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_with_multimodal_content_raises() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "describe"}],
        }
    ]
    with pytest.raises(InvalidRequestError, match="(?i)image|file|content|multimodal"):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


# --- Wire shape --------------------------------------------------------------


def _plain(model_priv: Any, value: str) -> str:
    assert _crypto.looks_encrypted(value)
    return _crypto.decrypt_chunk(model_priv, value)


@pytest.mark.asyncio
async def test_e2ee_encrypts_user_system_and_assistant_history() -> None:
    session, model_priv = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "prior reply", "reasoning_content": "thought"},
        {"role": "user", "content": "again"},
    ]

    with pytest.warns(UserWarning, match="(?i)attestation"):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)

    body = captured["body"]
    assert body["stream"] is True
    msgs = body["messages"]
    assert [_plain(model_priv, m["content"]) for m in msgs] == [
        "be terse",
        "hi",
        "prior reply",
        "again",
    ]
    assert _plain(model_priv, msgs[2]["reasoning_content"]) == "thought"

    headers = captured["headers"]
    assert headers["X-Venice-TEE-Client-Pub-Key"] == session.session_public_key_hex
    assert headers["X-Venice-TEE-Model-Pub-Key"] == session.model_public_key_hex


@pytest.mark.asyncio
async def test_e2ee_leaves_empty_strings_unencrypted() -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "again"},
    ]
    with pytest.warns(UserWarning):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    assert captured["body"]["messages"][1]["content"] == ""


@pytest.mark.asyncio
async def test_e2ee_assistant_without_content_is_accepted() -> None:
    session, model_priv = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "reasoning_content": "thinking"},
        {"role": "user", "content": "again"},
    ]
    with pytest.warns(UserWarning):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    assistant = captured["body"]["messages"][1]
    assert "content" not in assistant
    assert _plain(model_priv, assistant["reasoning_content"]) == "thinking"


@pytest.mark.asyncio
async def test_e2ee_round_trips_collected_message_object() -> None:
    first, model_priv = _build_session()
    second = TeeSession(
        session_private_key=_crypto.generate_session_keypair(),
        model_public_key_hex=first.model_public_key_hex,
        signing_algo="ecdsa",
    )
    messages: list[Any] = [{"role": "user", "content": "hi"}]
    with pytest.warns(UserWarning):
        result = await _make_completions(first, {}).create(
            model=_E2EE_MODEL, messages=messages, e2ee=True
        )
    assert isinstance(result, ChatCompletionResponse)

    captured: dict[str, Any] = {}
    messages = [*messages, result.choices[0].message, {"role": "user", "content": "more"}]
    with pytest.warns(UserWarning):
        await _make_completions(second, captured).create(
            model=_E2EE_MODEL, messages=messages, e2ee=True
        )
    wire = captured["body"]["messages"]
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    assert _plain(model_priv, wire[1]["content"]) == "Hello world"


@pytest.mark.asyncio
async def test_e2ee_accepts_collected_message_with_echoed_cache_control() -> None:
    session, model_priv = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    echoed = ChatMessage.model_validate(
        {"role": "assistant", "content": "prior", "cache_control": {"type": "ephemeral"}}
    )
    messages = [{"role": "user", "content": "hi"}, echoed, {"role": "user", "content": "more"}]
    with pytest.warns(UserWarning):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    wire = captured["body"]["messages"][1]
    assert "cache_control" not in wire
    assert _plain(model_priv, wire["content"]) == "prior"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, match",
    [
        ({"role": "developer", "content": "x"}, "use role 'system'"),
        ({"role": "tool", "tool_call_id": "c1", "content": "x"}, "tool"),
        (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            "tool_calls",
        ),
        ({"role": "user", "content": "x", "name": "alice"}, "name"),
        ({"role": "user", "content": "x", "reasoning_content": "y"}, "reasoning_content"),
        ({"role": "assistant", "content": [{"type": "text", "text": "x"}]}, "content"),
    ],
)
async def test_e2ee_refuses_unencryptable_messages(message: dict[str, Any], match: str) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [{"role": "user", "content": "SECRET-TEXT"}, message]
    with pytest.raises(InvalidRequestError, match=match) as excinfo:
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    assert "messages[1]" in str(excinfo.value)
    assert "SECRET-TEXT" not in str(excinfo.value)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message, role",
    [
        ({"role": "function", "content": "SECRET"}, "function"),
        ({"role": "tool", "content": "SECRET"}, "tool"),
        (ChatMessage(role="tool", content="SECRET"), "tool"),
    ],
    ids=["function-role-dict", "tool-dict-without-call-id", "tool-chat-message"],
)
async def test_e2ee_validation_error_does_not_leak_content(message: Any, role: str) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [{"role": "user", "content": "hi"}, message]
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    error = excinfo.value
    assert "SECRET" not in str(error)
    assert "messages[1]" in str(error)
    assert repr(role) in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert error.__suppress_context__ is True
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("kwargs", [{"logprobs": True}, {"top_logprobs": 2}])
async def test_e2ee_refuses_logprobs(kwargs: dict[str, Any]) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match="logprobs"):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, **kwargs)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_forces_venice_system_prompt_off() -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    with pytest.warns(UserWarning):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    vp = captured["body"].get("venice_parameters", {})
    assert vp.get("include_venice_system_prompt") is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params, expected",
    [
        (VeniceParameters(), {"include_venice_system_prompt": False}),
        (
            VeniceParameters(enable_e2ee=True, character_slug="c"),
            {"include_venice_system_prompt": False, "enable_e2ee": True, "character_slug": "c"},
        ),
        (
            VeniceParameters(include_venice_system_prompt=True),
            {"include_venice_system_prompt": False},
        ),
    ],
)
async def test_e2ee_wire_parameters_are_only_what_was_set_plus_the_forced_flag(
    params: VeniceParameters, expected: dict[str, Any]
) -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    with pytest.warns(UserWarning):
        await comp.create(
            model=_E2EE_MODEL, messages=_messages(), e2ee=True, venice_parameters=params
        )
    assert captured["body"]["venice_parameters"] == expected


# --- Response handling -------------------------------------------------------


@pytest.mark.asyncio
async def test_e2ee_stream_false_reassembles_decrypted_response() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.content == "Hello world"
    assert result.usage is not None
    assert result.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_e2ee_stream_true_yields_decrypted_deltas() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        stream = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=True)
    deltas = [
        c.choices[0].delta.content async for c in stream if c.choices and c.choices[0].delta.content
    ]
    assert deltas == ["Hello", " world"]


@pytest.mark.asyncio
async def test_enable_e2ee_param_drives_the_flow() -> None:
    session, model_priv = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    with pytest.warns(UserWarning):
        result = await comp.create(
            model=_E2EE_MODEL,
            messages=_messages(),
            venice_parameters={"enable_e2ee": True},
        )
    # Flow engaged: encrypted wire + reassembled plaintext response.
    assert captured["body"]["stream"] is True
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.content == "Hello world"
    # enable_e2ee passed through because the caller set it.
    assert captured["body"]["venice_parameters"].get("enable_e2ee") is True


@pytest.mark.asyncio
async def test_e2ee_warning_fires_each_call() -> None:
    """The attestation-trust warning fires on engage (not gated by a module flag)."""
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning, match="(?i)attestation"):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)


@pytest.mark.asyncio
async def test_stream_shorthand_forwards_e2ee() -> None:
    """``stream(e2ee=...)`` engages the flow and yields decrypted deltas."""
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    with pytest.warns(UserWarning, match="(?i)attestation"):
        stream = await comp.stream(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(stream, ChatStream)
    assert captured["body"]["stream"] is True
    deltas = [text async for text in stream.text_deltas()]
    assert deltas == ["Hello", " world"]


def _trust() -> TeeTrust:
    return TeeTrust(
        model=_E2EE_MODEL,
        verification="baseline",
        verifier=None,
        server_verified=True,
        nonce="cd" * 32,
        model_public_key="04" + "ab" * 64,
        signing_address="0x" + "ab" * 20,
    )


@pytest.mark.asyncio
async def test_e2ee_trust_on_stream_and_collected_response() -> None:
    trust = _trust()
    session, _ = _build_session(trust)
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        stream = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=True)
    assert isinstance(stream, ChatStream)
    assert stream.e2ee is trust

    session, _ = _build_session(trust)
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(result, ChatCompletionResponse)
    assert result.e2ee is trust
    assert "e2ee" not in result.model_dump()
    assert "e2ee" not in result.choices[0].message.model_dump()


@pytest.mark.asyncio
async def test_e2ee_without_session_trust_gives_none() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(result, ChatCompletionResponse)
    assert result.e2ee is None


# --- Reasoning and fail-closed response handling -----------------------------


def _enc(session: TeeSession, text: str) -> str:
    return _crypto.encrypt_message(session.session_public_key_hex, text)


def _choice(index: int, **delta: Any) -> dict[str, Any]:
    return {"index": index, "delta": delta}


def _reasoning_then_content(session: TeeSession) -> list[list[dict[str, Any]]]:
    return [
        [_choice(0, reasoning_content=_enc(session, "Step 1."))],
        [_choice(0, reasoning_content=_enc(session, " Step 2."))],
        [_choice(0, content=_enc(session, "Done"))],
    ]


@pytest.mark.asyncio
async def test_reasoning_content_is_decrypted_streaming_and_collected() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {}, _reasoning_then_content(session))
    with pytest.warns(UserWarning):
        stream = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=True)
    reasoning = [
        c.choices[0].delta.reasoning_content
        async for c in stream
        if c.choices and c.choices[0].delta.reasoning_content
    ]
    assert reasoning == ["Step 1.", " Step 2."]

    session, _ = _build_session()
    comp = _make_completions(session, {}, _reasoning_then_content(session))
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.reasoning_content == "Step 1. Step 2."
    assert result.choices[0].message.content == "Done"


@pytest.mark.asyncio
async def test_every_choice_is_decrypted() -> None:
    session, _ = _build_session()
    deltas = [[_choice(0, content=_enc(session, "A")), _choice(1, content=_enc(session, "B"))]]
    comp = _make_completions(session, {}, deltas)
    with pytest.warns(UserWarning):
        stream = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=True)
    chunks = [c async for c in stream if len(c.choices) == 2]
    assert [ch.delta.content for ch in chunks[0].choices] == ["A", "B"]


@pytest.mark.asyncio
async def test_empty_content_delta_passes() -> None:
    session, _ = _build_session()
    deltas = [
        [_choice(0, content=_enc(session, "Hi"))],
        [_choice(0, content="")],
        [_choice(0, content=_enc(session, " there"))],
    ]
    comp = _make_completions(session, {}, deltas)
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.content == "Hi there"


def _bad_choice_cases(session: TeeSession) -> list[tuple[dict[str, Any], str]]:
    valid = _enc(session, "x")
    cases: list[tuple[dict[str, Any], str]] = [
        ({"content": "plain text"}, "content"),
        ({"reasoning_content": "plain thought"}, "reasoning_content"),
        ({"reasoning_content": valid, "reasoning_encrypted": True}, "reasoning_encrypted"),
        ({"content": valid, "thought_signature": "leaked-sig"}, "thought_signature"),
        (
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "c",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ]
            },
            "tool_calls",
        ),
        ({"content": valid, "reasoning": "leaked"}, "reasoning"),
    ]
    logprobs = {"content": [{"token": "leaked", "logprob": -0.1, "top_logprobs": []}]}
    return [({"index": 0, "delta": delta}, match) for delta, match in cases] + [
        ({"index": 0, "delta": {"content": valid}, "text": "PLAIN CHOICE EXTRA"}, "text"),
        ({"index": 0, "delta": {"content": valid}, "logprobs": logprobs}, "logprobs"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize("stream", [True, False])
async def test_contract_breaking_delta_raises(index: int, stream: bool) -> None:
    session, _ = _build_session()
    choice, match = _bad_choice_cases(session)[index]
    comp = _make_completions(session, {}, [[choice]])
    with pytest.warns(UserWarning), pytest.raises(TeeEncryptionError, match=match) as excinfo:
        result = await comp.create(
            model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=stream
        )
        if stream:
            async for _chunk in result:  # type: ignore[union-attr]
                pass
    assert "plain text" not in str(excinfo.value)
    assert "plain thought" not in str(excinfo.value)
    assert "leaked" not in str(excinfo.value)
    assert "PLAIN CHOICE EXTRA" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_plaintext_delta_on_second_choice_raises() -> None:
    session, _ = _build_session()
    deltas = [[_choice(0, content=_enc(session, "A")), _choice(1, content="plain B")]]
    comp = _make_completions(session, {}, deltas)
    with pytest.warns(UserWarning), pytest.raises(TeeEncryptionError, match="content"):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)


def _unknown_field_choices(
    session: TeeSession, choice_fields: dict[str, Any], delta_fields: dict[str, Any]
) -> list[list[dict[str, Any]]]:
    delta = {"content": _enc(session, "ok"), **delta_fields}
    return [[{"index": 0, "delta": delta, **choice_fields}]]


async def _create_and_consume(comp: ChatCompletions, stream: bool, **kwargs: Any) -> str:
    """Run an E2EE create() and return the decrypted text, consuming a stream fully."""
    result = await comp.create(
        model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=stream, **kwargs
    )
    if isinstance(result, ChatCompletionResponse):
        return result.text or ""
    return "".join([c.choices[0].delta.content or "" async for c in result if c.choices])


_TEXT_BEARING_UNKNOWN_FIELDS = [
    pytest.param({"prompt_text": "PLAIN"}, {}, {}, "chunk", "prompt_text", id="chunk"),
    pytest.param({"meta": {"notes": ["PLAIN"]}}, {}, {}, "chunk", "meta", id="chunk-nested"),
    pytest.param({"cost": {"usd": "PLAIN"}}, {}, {}, "chunk", "cost", id="chunk-cost-with-text"),
    pytest.param({}, {"meta": {"note": "PLAIN"}}, {}, "choice", "meta", id="choice-nested"),
    pytest.param({}, {}, {"extra": [1, ("PLAIN",)]}, "delta", "extra", id="delta-nested"),
    pytest.param(
        {}, {"top_logprobs": [{"PLAIN": -0.1}]}, {}, "choice", "top_logprobs", id="choice-key"
    ),
    pytest.param({"usage_detail": {"PLAIN": 1}}, {}, {}, "chunk", "usage_detail", id="chunk-key"),
    pytest.param({}, {"bar": {"PLAIN": 2}}, {}, "choice", "bar", id="choice-nested-dict"),
    pytest.param({}, {}, {"baz": [1, {"PLAIN": False}]}, "delta", "baz", id="delta-dict-in-list"),
    pytest.param({"cost": {"usd": 0.1, "PLAIN": 1}}, {}, {}, "chunk", "cost", id="cost-extra-key"),
    pytest.param({"cost": {"usd": True}}, {}, {}, "chunk", "cost", id="cost-bool"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "chunk_fields, choice_fields, delta_fields, level, key", _TEXT_BEARING_UNKNOWN_FIELDS
)
async def test_text_bearing_unknown_field_raises(
    stream: bool,
    chunk_fields: dict[str, Any],
    choice_fields: dict[str, Any],
    delta_fields: dict[str, Any],
    level: str,
    key: str,
) -> None:
    session, _ = _build_session()
    deltas = _unknown_field_choices(session, choice_fields, delta_fields)
    comp = _make_completions(session, {}, deltas, chunk_fields=chunk_fields)
    with (
        pytest.warns(UserWarning),
        pytest.raises(TeeEncryptionError, match=f"{level} carried unexpected field {key!r}") as e,
    ):
        await _create_and_consume(comp, stream)
    assert "PLAIN" not in str(e.value)


_NON_TEXT_UNKNOWN_FIELDS = [
    pytest.param({"foo": 1}, {}, {}, id="chunk-number"),
    pytest.param({"cost": {"usd": 0.0002, "diem": 0}}, {}, {}, id="chunk-numeric-cost"),
    pytest.param({}, {"bar": [2, 3.5, None]}, {}, id="choice-list-of-numbers"),
    pytest.param({"cost": {"usd": 0.0002}}, {}, {}, id="chunk-cost-usd-only"),
    pytest.param({"cost": {"usd": None, "diem": 1}}, {}, {}, id="chunk-cost-none"),
    pytest.param({}, {"matched_stop": 151645}, {}, id="choice-matched-stop-int"),
    pytest.param({}, {"matched_stop": None}, {}, id="choice-matched-stop-none"),
    pytest.param({}, {"stop_reason": 151645}, {}, id="choice-stop-reason-int"),
    pytest.param({}, {"stop_reason": None}, {}, id="choice-stop-reason-none"),
    pytest.param({}, {}, {"baz": [1, 2.5, True, None, [0]]}, id="delta-numbers"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("chunk_fields, choice_fields, delta_fields", _NON_TEXT_UNKNOWN_FIELDS)
async def test_non_text_unknown_field_passes(
    stream: bool,
    chunk_fields: dict[str, Any],
    choice_fields: dict[str, Any],
    delta_fields: dict[str, Any],
) -> None:
    session, _ = _build_session()
    deltas = _unknown_field_choices(session, choice_fields, delta_fields)
    comp = _make_completions(session, {}, deltas, chunk_fields=chunk_fields)
    with pytest.warns(UserWarning):
        assert await _create_and_consume(comp, stream) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("request_stop", ["END-HERE", ["\n\n", "END-HERE"]])
@pytest.mark.parametrize("field", ["matched_stop", "stop_reason"])
async def test_stop_field_may_be_a_requested_stop_string(
    field: str, stream: bool, request_stop: str | list[str]
) -> None:
    session, _ = _build_session()
    deltas = _unknown_field_choices(session, {field: "END-HERE"}, {})
    comp = _make_completions(session, {}, deltas)
    with pytest.warns(UserWarning):
        assert await _create_and_consume(comp, stream, stop=request_stop) == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("request_stop", [None, "END-HERE"])
@pytest.mark.parametrize("field", ["matched_stop", "stop_reason"])
async def test_stop_field_with_other_text_raises(
    field: str, stream: bool, request_stop: str | None
) -> None:
    session, _ = _build_session()
    deltas = _unknown_field_choices(session, {field: "PLAIN"}, {})
    comp = _make_completions(session, {}, deltas)
    with (
        pytest.warns(UserWarning),
        pytest.raises(TeeEncryptionError, match=f"choice carried unexpected field {field!r}") as e,
    ):
        await _create_and_consume(comp, stream, stop=request_stop)
    assert "PLAIN" not in str(e.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_plaintext_delta_raises_and_closes_session(stream: bool) -> None:
    session, _ = _build_session()
    deltas = [[_choice(0, content=_enc(session, "ok"))], [_choice(0, content="plain text")]]
    comp = _make_completions(session, {}, deltas)
    with (
        pytest.warns(UserWarning),
        pytest.raises(TeeEncryptionError, match="carried unencrypted content"),
    ):
        result = await comp.create(
            model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=stream
        )
        if stream:
            async for _chunk in result:  # type: ignore[union-attr]
                pass
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk(_enc(session, "after close"))


@pytest.mark.asyncio
async def test_e2ee_round_trips_collected_reasoning_content() -> None:
    first, model_priv = _build_session()
    second = TeeSession(
        session_private_key=_crypto.generate_session_keypair(),
        model_public_key_hex=first.model_public_key_hex,
        signing_algo="ecdsa",
    )
    messages: list[Any] = [{"role": "user", "content": "hi"}]
    with pytest.warns(UserWarning):
        result = await _make_completions(first, {}, _reasoning_then_content(first)).create(
            model=_E2EE_MODEL, messages=messages, e2ee=True
        )
    assert isinstance(result, ChatCompletionResponse)
    assert result.choices[0].message.reasoning_content == "Step 1. Step 2."

    captured: dict[str, Any] = {}
    messages = [*messages, result.choices[0].message, {"role": "user", "content": "more"}]
    with pytest.warns(UserWarning):
        await _make_completions(second, captured).create(
            model=_E2EE_MODEL, messages=messages, e2ee=True
        )
    assistant = captured["body"]["messages"][1]
    assert _plain(model_priv, assistant["reasoning_content"]) == "Step 1. Step 2."
    assert _plain(model_priv, assistant["content"]) == "Done"


_TEXT_BEARING_USAGE_EXTRAS = [
    pytest.param({"note": "PLAIN"}, "usage", "'note'", id="usage-text-value"),
    pytest.param({"detail": {"PLAIN": 1}}, "usage", "'detail'", id="usage-text-key"),
    pytest.param(
        {"prompt_tokens_details": {"cached_tokens": 1, "note": "PLAIN"}},
        "usage.prompt_tokens_details",
        "'note'",
        id="prompt-details-text",
    ),
    pytest.param(
        {"completion_tokens_details": {"reasoning_tokens": 0, "meta": {"PLAIN": 1}}},
        "usage.completion_tokens_details",
        "'meta'",
        id="completion-details-text-key",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("extra, level, key", _TEXT_BEARING_USAGE_EXTRAS)
async def test_text_bearing_usage_extra_raises(
    stream: bool, extra: dict[str, Any], level: str, key: str
) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {}, usage={**_REALISTIC_USAGE, **extra})
    with (
        pytest.warns(UserWarning),
        pytest.raises(TeeEncryptionError) as e,
    ):
        await _create_and_consume(comp, stream)
    assert f"E2EE response {level} carried unexpected field {key}." == str(e.value)
    assert "PLAIN" not in str(e.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "usage",
    [
        pytest.param(_REALISTIC_USAGE, id="realistic"),
        pytest.param({**_REALISTIC_USAGE, "foo": 1}, id="numeric-extra"),
        pytest.param(
            {**_REALISTIC_USAGE, "prompt_tokens_details": {"cached_tokens": 1, "foo": 2}},
            id="numeric-details-extra",
        ),
    ],
)
async def test_realistic_or_numeric_usage_passes(stream: bool, usage: dict[str, Any]) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {}, usage=usage)
    with pytest.warns(UserWarning):
        assert await _create_and_consume(comp, stream) == "Hello world"


_MALFORMED_MESSAGE = {"role": "tool", "content": "SECRET"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, venice_parameters",
    [(_E2EE_MODEL, None), ("plain-model", {"enable_e2ee": True})],
    ids=["e2ee-model", "enable_e2ee"],
)
async def test_estimate_cost_e2ee_validation_error_does_not_leak_content(
    model: str, venice_parameters: dict[str, Any] | None
) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    fetch_pricing = AsyncMock()
    comp._fetch_chat_pricing = fetch_pricing  # type: ignore[method-assign]
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.estimate_cost(
            model=model,
            messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE],
            venice_parameters=venice_parameters,
        )
    error = excinfo.value
    assert "SECRET" not in str(error)
    assert "messages[1]" in str(error)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    fetch_pricing.assert_not_called()


@pytest.mark.asyncio
async def test_estimate_cost_without_e2ee_raises_validation_error() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    comp._fetch_chat_pricing = AsyncMock()  # type: ignore[method-assign]
    with pytest.raises(ValidationError):
        await comp.estimate_cost(
            model="plain-model", messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE]
        )


def _add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "create_kwargs",
    [{"e2ee": True}, {"venice_parameters": {"enable_e2ee": True}}],
    ids=["e2ee", "enable_e2ee"],
)
async def test_run_with_tools_e2ee_validation_error_does_not_leak_content(
    create_kwargs: dict[str, Any],
) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    create = AsyncMock()
    comp.create = create  # type: ignore[method-assign]
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.run_with_tools(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE],
            tools=[_add],
            **create_kwargs,
        )
    error = excinfo.value
    assert "SECRET" not in str(error)
    assert "messages[1]" in str(error)
    assert error.__cause__ is None
    assert error.__suppress_context__ is True
    create.assert_not_called()


@pytest.mark.asyncio
async def test_run_with_tools_without_e2ee_raises_validation_error() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    create = AsyncMock()
    comp.create = create  # type: ignore[method-assign]
    with pytest.raises(ValidationError):
        await comp.run_with_tools(
            model="plain-model",
            messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE],
            tools=[_add],
        )
    create.assert_not_called()


# --- Absent values, protocol-vocabulary fields, closing a stream -------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [0, 0.0], ids=["int-zero", "float-zero"])
async def test_e2ee_refuses_zero_valued_extra_key(value: float) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [{"role": "user", "content": "x", "extra_k": value}]
    with pytest.raises(InvalidRequestError, match="'extra_k'"):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_e2ee_false_valued_extra_key_counts_as_absent() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [{"role": "user", "content": "x", "extra_k": False}]
    with pytest.warns(UserWarning):
        result = await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    assert isinstance(result, ChatCompletionResponse)


@pytest.mark.asyncio
async def test_e2ee_refuses_zero_reasoning_content_on_user_message() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    messages = [{"role": "user", "content": "x", "reasoning_content": 0}]
    with pytest.raises(InvalidRequestError, match="reasoning_content"):
        await comp.create(model=_E2EE_MODEL, messages=messages, e2ee=True)
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_non_assistant_delta_role_raises(stream: bool) -> None:
    session, _ = _build_session()
    deltas = [[_choice(0, role="user", content=_enc(session, "ok"))]]
    comp = _make_completions(session, {}, deltas)
    with pytest.warns(UserWarning), pytest.raises(TeeEncryptionError, match="role") as e:
        await _create_and_consume(comp, stream)
    assert "'user'" not in str(e.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "finish_reason",
    ["Here is PLAIN text", "stop\n", "PLAIN", "x" * 33],
    ids=["free-text", "trailing-newline", "uppercase", "too-long"],
)
async def test_free_text_finish_reason_raises(stream: bool, finish_reason: str) -> None:
    session, _ = _build_session()
    choice = {**_choice(0, content=_enc(session, "ok")), "finish_reason": finish_reason}
    comp = _make_completions(session, {}, [[choice]])
    with pytest.warns(UserWarning), pytest.raises(TeeEncryptionError, match="finish_reason") as e:
        await _create_and_consume(comp, stream)
    assert "PLAIN" not in str(e.value)
    assert finish_reason not in str(e.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("finish_reason", ["stop", "length", "content_filter"])
async def test_protocol_finish_reason_passes(stream: bool, finish_reason: str) -> None:
    session, _ = _build_session()
    choice = {
        **_choice(0, role="assistant", content=_enc(session, "ok")),
        "finish_reason": finish_reason,
    }
    comp = _make_completions(session, {}, [[choice]])
    with pytest.warns(UserWarning):
        assert await _create_and_consume(comp, stream) == "ok"


async def _unconsumed_e2ee_stream(session: TeeSession) -> ChatStream:
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning):
        stream = await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True, stream=True)
    assert isinstance(stream, ChatStream)
    return stream


@pytest.mark.asyncio
async def test_closing_an_unconsumed_stream_closes_the_session() -> None:
    session, _ = _build_session()
    stream = await _unconsumed_e2ee_stream(session)
    await stream.close()
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk(_enc(session, "after close"))


@pytest.mark.asyncio
async def test_async_with_on_an_unconsumed_stream_closes_the_session() -> None:
    session, _ = _build_session()
    stream = await _unconsumed_e2ee_stream(session)
    async with stream:
        pass
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk(_enc(session, "after close"))


@pytest.mark.asyncio
async def test_close_reaches_raw_iterator_and_session_when_decrypting_close_raises() -> None:
    session, _ = _build_session()
    raw = MagicMock()
    raw.aclose = AsyncMock()
    iterator = _E2EEChunkIterator(raw, session)
    await iterator._chunks.aclose()
    failing = MagicMock()
    failing.aclose = AsyncMock(side_effect=RuntimeError("decrypting close failed"))
    iterator._chunks = failing
    with pytest.raises(RuntimeError, match="decrypting close failed"):
        await iterator.aclose()
    raw.aclose.assert_awaited_once()
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk(_enc(session, "after close"))


@pytest.mark.asyncio
async def test_closing_a_partly_consumed_stream_closes_the_session() -> None:
    session, _ = _build_session()
    stream = await _unconsumed_e2ee_stream(session)
    async with stream:
        await stream.__anext__()
    with pytest.raises(TeeEncryptionError):
        session.decrypt_chunk(_enc(session, "after close"))


# --- Engagement follows the validated enable_e2ee ----------------------------


def _wire_contents(body: dict[str, Any]) -> list[str]:
    return [m["content"] for m in body["messages"] if isinstance(m.get("content"), str)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "venice_parameters",
    [
        {"enable_e2ee": 1},
        {"enable_e2ee": "true"},
        {"enable_e2ee": True},
        VeniceParameters(enable_e2ee=True),
    ],
    ids=["int-1", "str-true", "bool-true", "model"],
)
async def test_truthy_enable_e2ee_engages_the_flow(venice_parameters: Any) -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    with pytest.warns(UserWarning):
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "SECRET-TEXT"}],
            venice_parameters=venice_parameters,
            stream=True,
        )
    comp._client.tee.open_session.assert_called_once()  # type: ignore[attr-defined]
    contents = _wire_contents(captured["body"])
    assert contents
    assert all(_crypto.looks_encrypted(c) for c in contents)
    assert captured["body"]["venice_parameters"]["enable_e2ee"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "venice_parameters",
    [{"enable_e2ee": 0}, {"enable_e2ee": "false"}, {"enable_e2ee": False}],
    ids=["int-0", "str-false", "bool-false"],
)
async def test_falsy_enable_e2ee_does_not_engage(venice_parameters: Any) -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    await comp.create(
        model=_E2EE_MODEL,
        messages=[{"role": "user", "content": "hello"}],
        venice_parameters=venice_parameters,
        stream=True,
    )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]
    assert captured["body"]["venice_parameters"]["enable_e2ee"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [1, "true"], ids=["int-1", "str-true"])
async def test_estimate_cost_truthy_enable_e2ee_withholds_content(value: Any) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    comp._fetch_chat_pricing = AsyncMock()  # type: ignore[method-assign]
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.estimate_cost(
            model="plain-model",
            messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE],
            venice_parameters={"enable_e2ee": value},
        )
    assert "SECRET" not in str(excinfo.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [1, "true"], ids=["int-1", "str-true"])
async def test_run_with_tools_truthy_enable_e2ee_withholds_content(value: Any) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    create = AsyncMock()
    comp.create = create  # type: ignore[method-assign]
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.run_with_tools(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "hi"}, _MALFORMED_MESSAGE],
            tools=[_add],
            venice_parameters={"enable_e2ee": value},
        )
    assert "SECRET" not in str(excinfo.value)
    create.assert_not_called()


@pytest.mark.asyncio
async def test_invalid_venice_parameters_keep_the_validation_error() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(ValidationError):
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            venice_parameters={"enable_e2ee": "maybe"},
        )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize("e2ee", [True, TeeOptions()], ids=["bool", "options"])
@pytest.mark.parametrize(
    "venice_parameters",
    [{"enable_e2ee": False}, {"enable_e2ee": 0}, VeniceParameters(enable_e2ee=False)],
    ids=["bool-false", "int-0", "model"],
)
async def test_e2ee_true_with_enable_e2ee_false_is_refused(
    e2ee: Any, venice_parameters: Any
) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(
        InvalidRequestError,
        match=r"e2ee=True conflicts with venice_parameters\.enable_e2ee=False",
    ):
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            e2ee=e2ee,
            venice_parameters=venice_parameters,
        )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]
    comp._client._stream_request.assert_not_called()  # type: ignore[attr-defined]


class _AnyEnableE2EE(VeniceParameters):
    """A subclass that loosens ``enable_e2ee`` to accept any value."""

    enable_e2ee: Any = None


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings:UserWarning")
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "non-stream"])
@pytest.mark.parametrize(
    "venice_parameters",
    [
        VeniceParameters.model_construct(enable_e2ee=1),
        VeniceParameters.model_construct(enable_e2ee=0),
        VeniceParameters.model_construct(enable_e2ee="true"),
        _AnyEnableE2EE(enable_e2ee=1),
        _AnyEnableE2EE(enable_e2ee="yes"),
    ],
    ids=["construct-1", "construct-0", "construct-str", "subclass-1", "subclass-str"],
)
async def test_non_bool_wire_enable_e2ee_is_refused(venice_parameters: Any, stream: bool) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match=r"venice_parameters\.enable_e2ee must be a bool"):
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "SECRET-TEXT"}],
            venice_parameters=venice_parameters,
            stream=stream,
        )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]
    comp._client._stream_request.assert_not_called()  # type: ignore[attr-defined]
    comp._client._request.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("venice_parameters", "wire_value"),
    [
        (None, None),
        (VeniceParameters(enable_e2ee=False), False),
        (VeniceParameters.model_construct(enable_e2ee=False), False),
        (VeniceParameters.model_construct(enable_e2ee=None), None),
        (_AnyEnableE2EE(enable_e2ee=False), False),
        (VeniceParameters(character_slug="c"), None),
    ],
    ids=["absent", "model-false", "construct-false", "construct-none", "subclass-false", "unset"],
)
async def test_false_or_absent_wire_enable_e2ee_is_unaffected(
    venice_parameters: Any, wire_value: Any
) -> None:
    session, _ = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)
    await comp.create(
        model=_E2EE_MODEL,
        messages=[{"role": "user", "content": "hello"}],
        venice_parameters=venice_parameters,
        stream=True,
    )
    comp._client.tee.open_session.assert_not_called()  # type: ignore[attr-defined]
    assert captured["body"].get("venice_parameters", {}).get("enable_e2ee") is wire_value
    assert _wire_contents(captured["body"]) == ["hello"]


# --- Secrets on the error paths ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False])
async def test_encryption_failure_closes_the_session(stream: bool) -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.warns(UserWarning), pytest.raises(TeeEncryptionError) as excinfo:
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "SECRET-TEXT \ud800"}],
            e2ee=True,
            stream=stream,
        )
    assert session._session_private_key is None
    assert "SECRET-TEXT" not in str(excinfo.value)
    comp._client._stream_request.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_unknown_role_is_not_echoed() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError) as excinfo:
        await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "SECRETXYZ", "content": "SECRETXYZ"}],
            e2ee=True,
        )
    assert "SECRETXYZ" not in str(excinfo.value)
    assert "an unsupported role" in str(excinfo.value)


@pytest.mark.asyncio
async def test_known_role_is_named_in_validation_refusal() -> None:
    session, _ = _build_session()
    comp = _make_completions(session, {})
    with pytest.raises(InvalidRequestError, match=r"messages\[0\] \(role 'tool'\)"):
        await comp.create(
            model=_E2EE_MODEL, messages=[{"role": "tool", "content": "SECRET"}], e2ee=True
        )


# --- TeeTrust and response equality ------------------------------------------


def test_tee_trust_is_frozen() -> None:
    trust = _trust()
    with pytest.raises(dataclasses.FrozenInstanceError):
        trust.model = "other"  # type: ignore[misc]


def test_e2ee_takes_part_in_response_equality() -> None:
    payload = {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": _E2EE_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
    }
    plain = ChatCompletionResponse.model_validate(payload)
    with_trust = ChatCompletionResponse.model_validate(payload)
    with_trust._e2ee = _trust()
    assert plain.model_dump() == with_trust.model_dump()
    assert plain != with_trust


@pytest.mark.asyncio
async def test_e2ee_create_surfaces_typed_verifier_error() -> None:
    import json
    from pathlib import Path

    from venice_ai.exceptions import TeeTcbStatusError
    from venice_ai.resources.tee import Tee
    from venice_ai.tee.types import TeeAttestation

    raw = json.loads((Path(__file__).parent / "fixtures" / "attestation_gemma.json").read_text())
    raised = TeeTcbStatusError("x", status="OutOfDate", advisory_ids=[], policy="reject")

    class _Verifier:
        def verify(self, attestation: TeeAttestation) -> bool:
            raise raised

    session, _ = _build_session()
    comp = _make_completions(session, {})
    client = comp._client
    client.get = AsyncMock(return_value=TeeAttestation.model_validate(raw))
    client.tee = Tee(client)

    with pytest.raises(TeeTcbStatusError) as excinfo:
        await comp.create(
            model=_E2EE_MODEL,
            messages=_messages(),
            e2ee=TeeOptions(nonce=raw["request_nonce"], verifier=_Verifier()),
        )
    assert excinfo.value is raised
    client._stream_request.assert_not_called()
