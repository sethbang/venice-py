"""The E2EE attestation-trust warning must describe the caller's actual trust posture.

The baseline attestation verifier trusts Venice's server-side ``verified``
claim, and ``chat.completions.create(e2ee=...)`` warns about that. When the
caller supplies a full client-side quote verifier through
``TeeOptions(verifier=...)`` that limitation no longer applies, so the
"does NOT perform full client-side quote verification" claim must not be
emitted. The baseline paths must still warn.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

ec = pytest.importorskip(
    "cryptography.hazmat.primitives.asymmetric.ec",
    reason="tee chat e2ee tests require the [e2ee] extra (cryptography)",
)

from venice_ai.resources.chat.completions import ChatCompletions  # noqa: E402
from venice_ai.tee import TeeOptions, _crypto  # noqa: E402
from venice_ai.tee._session import TeeSession  # noqa: E402
from venice_ai.types.api.streaming import ChatCompletionChunk  # noqa: E402

_E2EE_MODEL = "e2ee-gemma-3-27b-p"

#: Fragments of the baseline-trust limitation text. Any one of them in a
#: warning means the caller is being told no client-side quote check runs.
_BASELINE_TRUST_CLAIMS = ("TRUSTS Venice", "does NOT perform")


class _AcceptingQuoteVerifier:
    """A FullQuoteVerifier that accepts every attestation."""

    def verify(self, attestation: Any) -> bool:
        return True


def _session() -> TeeSession:
    model_priv = ec.generate_private_key(ec.SECP256K1())
    return TeeSession(
        session_private_key=_crypto.generate_session_keypair(),
        model_public_key_hex=_crypto.uncompressed_hex(model_priv.public_key()),
        signing_algo="ecdsa",
    )


def _completions(session: TeeSession) -> ChatCompletions:
    client = MagicMock()
    client.tee.open_session = AsyncMock(return_value=session)
    session_pub = session.session_public_key_hex

    def _stream_request(**_kw: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def _gen() -> AsyncIterator[ChatCompletionChunk]:
            base = {
                "id": "chatcmpl-x",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": _E2EE_MODEL,
            }
            yield ChatCompletionChunk.model_validate(
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
            )
            yield ChatCompletionChunk.model_validate(
                {
                    **base,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": _crypto.encrypt_message(session_pub, "ok")},
                        }
                    ],
                }
            )
            yield ChatCompletionChunk.model_validate(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            )

        return _gen()

    client._stream_request = MagicMock(side_effect=_stream_request)
    return ChatCompletions(client)


async def _baseline_trust_warnings(e2ee: Any, *, stream: bool = False) -> list[str]:
    """Run one E2EE call and return every baseline-trust warning it emitted."""
    comp = _completions(_session())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await comp.create(
            model=_E2EE_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            e2ee=e2ee,
            stream=stream,
        )
        if stream:
            async for _ in result:  # type: ignore[union-attr]
                pass
    return [
        str(w.message)
        for w in caught
        if any(claim in str(w.message) for claim in _BASELINE_TRUST_CLAIMS)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["create", "stream"])
async def test_full_quote_verifier_suppresses_baseline_trust_warning(stream: bool) -> None:
    emitted = await _baseline_trust_warnings(
        TeeOptions(verifier=_AcceptingQuoteVerifier()), stream=stream
    )
    assert emitted == [], (
        "a FullQuoteVerifier was supplied, yet the call still warned that no "
        f"client-side quote verification is performed: {emitted!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "e2ee",
    [True, TeeOptions(), TeeOptions(verifier=None)],
    ids=["bool", "options-default", "options-no-verifier"],
)
async def test_baseline_verification_still_warns(e2ee: Any) -> None:
    emitted = await _baseline_trust_warnings(e2ee)
    assert len(emitted) == 1, (
        f"baseline-only verification must warn exactly once per call; got {emitted!r}"
    )
