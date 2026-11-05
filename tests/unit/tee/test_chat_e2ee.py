"""Integration tests for E2EE on ``chat.completions.create``.

The real flow is wired end-to-end with mocks for the two collaborators that
touch the network or generate keys:

* ``client.tee.open_session`` → returns a real :class:`TeeSession` built around a
  synthetic, verification-passing attestation whose model keypair we control, so
  encrypt/decrypt actually round-trips.
* ``client._stream_request`` → an async generator that asserts the outgoing body
  (encrypted user/system content, plaintext assistant, ``stream`` forced true)
  and emits server-encrypted SSE chunks (encrypted to the SESSION pub).

These exercise: the FAIL-LOUD guards (non-``e2ee-`` model, tools, web search /
scraping, multimodal content), the wire shape, ``stream=False`` reassembly,
``stream=True`` decrypted deltas, the ``enable_e2ee`` driver, and the one-time
attestation-trust :class:`UserWarning`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# cryptography is the optional [e2ee] extra; the session round-trip needs it.
ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee chat e2ee tests require the [e2ee] extra (cryptography)",
)

from venice_ai.exceptions import InvalidRequestError  # noqa: E402
from venice_ai.resources.chat.completions import ChatCompletions  # noqa: E402
from venice_ai.streaming import ChatStream  # noqa: E402
from venice_ai.tee import _crypto  # noqa: E402
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.types.api import ChatCompletionResponse  # noqa: E402
from venice_ai.types.api.requests.common import VeniceParameters  # noqa: E402
from venice_ai.types.api.streaming import ChatCompletionChunk  # noqa: E402

_CURVE = ec.SECP256K1()

_E2EE_MODEL = "e2ee-gemma-3-27b-p"


def _build_session() -> tuple[TeeSession, ec.EllipticCurvePrivateKey]:
    """A real TeeSession around a model keypair we control (for round-trips)."""
    model_priv = ec.generate_private_key(_CURVE)
    model_pub_hex = _crypto.uncompressed_hex(model_priv.public_key())
    session_priv = _crypto.generate_session_keypair()
    session = TeeSession(
        session_private_key=session_priv,
        model_public_key_hex=model_pub_hex,
        signing_algo="ecdsa",
    )
    return session, model_priv


def _make_completions(session: TeeSession, captured: dict[str, Any]) -> ChatCompletions:
    """A ChatCompletions whose client mocks tee.open_session + _stream_request.

    ``captured`` collects the outgoing body / headers and the streamed text for
    assertions. The mocked stream encrypts each delta to the SESSION pub so the
    decrypting wrapper produces real plaintext.
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
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }
            )
            # Content chunks: each independently encrypted to the SESSION pub.
            for piece in ("Hello", " world"):
                enc = _crypto.encrypt_message(session_pub, piece)
                yield ChatCompletionChunk.model_validate(
                    {
                        "id": "chatcmpl-x",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": _E2EE_MODEL,
                        "choices": [{"index": 0, "delta": {"content": enc}}],
                    }
                )
            # Final chunk: finish_reason + usage (plaintext passthrough).
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "chatcmpl-x",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": _E2EE_MODEL,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
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


@pytest.mark.asyncio
async def test_e2ee_encrypts_user_system_plaintext_assistant_and_forces_stream() -> None:
    session, model_priv = _build_session()
    captured: dict[str, Any] = {}
    comp = _make_completions(session, captured)

    with pytest.warns(UserWarning, match="(?i)attestation"):
        await comp.create(model=_E2EE_MODEL, messages=_messages(), e2ee=True)

    body = captured["body"]
    assert body["stream"] is True
    msgs = body["messages"]
    by_role = {m["role"]: m for m in msgs}

    # user + system content is encrypted hex that decrypts to the original.
    assert _crypto.looks_encrypted(by_role["user"]["content"])
    assert _crypto.decrypt_chunk(model_priv, by_role["user"]["content"]) == "hi"
    assert _crypto.looks_encrypted(by_role["system"]["content"])
    assert _crypto.decrypt_chunk(model_priv, by_role["system"]["content"]) == "be terse"

    # assistant content stays plaintext.
    assert by_role["assistant"]["content"] == "prior reply"

    # Headers carry the session pub + model pub.
    headers = captured["headers"]
    assert headers["X-Venice-TEE-Client-Pub-Key"] == session.session_public_key_hex
    assert headers["X-Venice-TEE-Model-Pub-Key"] == session.model_public_key_hex


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
