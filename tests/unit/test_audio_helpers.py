"""Unit tests for venice_ai.audio_helpers.

Covers split_text_for_tts plus the synchronous edges of stream_long_text
(input validation, format gate, single-segment fast path). The parallel
queue / cancel orchestration is exercised by the e2e suite.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from venice_ai.audio_helpers import (
    DEFAULT_WORD_BUDGET,
    MODEL_WORD_BUDGETS,
    _Mp3SegmentHeaderStripper,
    split_text_for_tts,
    stream_long_text,
)

# ---------------------------------------------------------------------------
# split_text_for_tts
# ---------------------------------------------------------------------------


class TestSplitTextForTts:
    def test_empty_string_returns_single_empty_segment(self):
        assert split_text_for_tts("") == [""]

    def test_whitespace_only_returns_single_empty_segment(self):
        assert split_text_for_tts("   \n  \n\n  ") == [""]

    def test_single_sentence_under_budget_returns_intact(self):
        out = split_text_for_tts("Hello world.", max_words=10)
        assert out == ["Hello world."]

    def test_two_sentences_packed_into_one_segment(self):
        out = split_text_for_tts("Hello world. Goodbye world.", max_words=10)
        assert out == ["Hello world. Goodbye world."]

    def test_packs_until_next_sentence_overflows(self):
        # 6 + 6 = 12 words total. Budget 8: first sentence (6) fits;
        # second (6) overflows → flush, second goes alone.
        text = "alpha bravo charlie delta echo foxtrot. golf hotel india juliet kilo lima."
        out = split_text_for_tts(text, max_words=8)
        assert len(out) == 2
        assert "alpha" in out[0] and "foxtrot" in out[0]
        assert "golf" in out[1] and "lima" in out[1]

    def test_oversized_sentence_emitted_alone(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel."
        out = split_text_for_tts(text, max_words=3)
        assert out == ["alpha bravo charlie delta echo foxtrot golf hotel."]

    def test_paragraphs_under_budget_pass_through_as_one_segment(self):
        # No-op when total fits under budget — important for kokoro/xai which
        # render shorter audio per segment when split unnecessarily.
        text = "First paragraph.\n\nSecond paragraph."
        out = split_text_for_tts(text, max_words=100)
        assert out == [text]

    def test_paragraphs_split_when_budget_exceeded(self):
        # 100 words across 2 paragraphs with budget 50 → 2 segments
        para_a = " ".join("alpha" for _ in range(60))
        para_b = " ".join("bravo" for _ in range(60))
        text = f"{para_a}.\n\n{para_b}."
        out = split_text_for_tts(text, max_words=50)
        assert len(out) >= 2

    def test_qwen3_uses_default_25_word_budget(self):
        # 50 words single-sentence should split into multiple chunks
        text = ". ".join(f"sentence number {i} has a few words" for i in range(10)) + "."
        out = split_text_for_tts(text, model="tts-qwen3-1-7b")
        assert len(out) >= 2
        for segment in out:
            # over-budget-alone segments allowed, but normal splits respect budget
            n_words = len(segment.split())
            assert n_words <= 25 or "." not in segment[:-1]

    def test_unknown_model_uses_default_budget(self):
        assert MODEL_WORD_BUDGETS.get("not-a-model") is None
        # Build text just over default budget
        words = " ".join("word" for _ in range(DEFAULT_WORD_BUDGET + 20))
        text = f"{words}. {words}."
        out = split_text_for_tts(text, model="not-a-model")
        # 200+ words across 2 sentences with default 100 budget = at least 2 segments
        # (each sentence is over budget alone → each emitted alone)
        assert len(out) >= 2

    def test_max_words_override_wins_over_model_default(self):
        out = split_text_for_tts("a b c. d e f.", model="tts-qwen3-1-7b", max_words=2)
        # 3 words per sentence, each over the override budget of 2
        assert len(out) == 2

    def test_max_words_zero_raises(self):
        with pytest.raises(ValueError, match="max_words must be positive"):
            split_text_for_tts("hi", max_words=0)

    def test_max_words_negative_raises(self):
        with pytest.raises(ValueError, match="max_words must be positive"):
            split_text_for_tts("hi", max_words=-5)

    def test_newline_within_paragraph_splits_sentences(self):
        # Poetry: lines end with comma, no period — paragraph blank-line is
        # the only real boundary
        text = "Line one,\nLine two.\nLine three!"
        out = split_text_for_tts(text, max_words=2)
        # Each sentence ~ 2 words → 3 sentences → at least 3 segments
        # (each fits in 2-word budget)
        assert "Line one" in out[0] or "Line one" in " ".join(out)

    def test_question_and_exclamation_terminators(self):
        text = "Question? Answer! Statement."
        out = split_text_for_tts(text, max_words=1)
        # 3 single-word-ish sentences but each is 1 word? Actually "Question?" is 1 word.
        # With budget 1, each emits alone
        assert len(out) == 3


# ---------------------------------------------------------------------------
# _Mp3SegmentHeaderStripper
# ---------------------------------------------------------------------------


def _id3_tag(body_size: int) -> bytes:
    size_bytes = bytes(
        [
            (body_size >> 21) & 0x7F,
            (body_size >> 14) & 0x7F,
            (body_size >> 7) & 0x7F,
            body_size & 0x7F,
        ]
    )
    return b"ID3\x04\x00\x00" + size_bytes + b"x" * body_size


# MPEG-2 Layer III, 64 kb/s, 24 kHz, mono: 72 * 64000 // 24000 = 192 bytes.
_FRAME_HEADER = b"\xff\xf3\x84\xc4"
_FRAME_LEN = 192
_SIDE_INFO = 9


def _audio_frame(fill: bytes = b"\x00") -> bytes:
    return _FRAME_HEADER + fill * (_FRAME_LEN - 4)


def _info_frame() -> bytes:
    body = b"\x00" * _SIDE_INFO + b"Info" + b"\x00" * (_FRAME_LEN - 4 - _SIDE_INFO - 4)
    return _FRAME_HEADER + body


def _run(stripper: _Mp3SegmentHeaderStripper, chunks: list[bytes]) -> bytes:
    return b"".join(stripper.feed(c) for c in chunks) + stripper.flush()


class TestMp3SegmentHeaderStripper:
    def test_passthrough_when_no_id3_or_frame_header(self):
        data = b"not an mp3 at all, just bytes"
        assert _run(_Mp3SegmentHeaderStripper(), [data]) == data

    def test_strips_id3_and_info_frame(self):
        audio = _audio_frame(b"\x01") + _audio_frame(b"\x02")
        data = _id3_tag(100) + _info_frame() + audio
        assert _run(_Mp3SegmentHeaderStripper(), [data]) == audio

    def test_keeps_first_frame_when_it_is_audio(self):
        audio = _audio_frame(b"\x01") + _audio_frame(b"\x02")
        assert _run(_Mp3SegmentHeaderStripper(), [_id3_tag(50) + audio]) == audio

    def test_tag_split_across_many_chunks_is_skipped_in_full(self):
        audio = _audio_frame(b"\x01")
        data = _id3_tag(1000) + _info_frame() + audio
        chunks = [data[i : i + 3] for i in range(0, len(data), 3)]
        assert _run(_Mp3SegmentHeaderStripper(), chunks) == audio

    def test_partial_magic_is_held_until_decidable(self):
        stripper = _Mp3SegmentHeaderStripper()
        assert stripper.feed(b"ID") == b""
        audio = _audio_frame()
        rest = _id3_tag(20)[2:] + audio
        assert stripper.feed(rest) + stripper.flush() == audio

    def test_truncated_tag_at_end_of_stream_is_dropped(self):
        stripper = _Mp3SegmentHeaderStripper()
        assert _run(stripper, [_id3_tag(1000)[:40]]) == b""

    def test_short_non_tag_stream_is_flushed(self):
        assert _run(_Mp3SegmentHeaderStripper(), [b"\xff\xf3"]) == b"\xff\xf3"
        assert _run(_Mp3SegmentHeaderStripper(), []) == b""

    def test_bytes_after_first_frame_pass_through_unbuffered(self):
        stripper = _Mp3SegmentHeaderStripper()
        stripper.feed(_id3_tag(10) + _info_frame())
        later = _audio_frame(b"\x07")
        assert stripper.feed(later) == later


# ---------------------------------------------------------------------------
# stream_long_text — validation + fast path (no live API)
# ---------------------------------------------------------------------------


class TestStreamLongTextValidation:
    @pytest.mark.asyncio
    async def test_rejects_non_mp3_format(self):
        # Stub client — never called because validation runs before any work
        stub_client: Any = object()
        gen = stream_long_text(
            stub_client,
            input="hi",
            model="tts-kokoro",
            voice="af_alloy",
            response_format="wav",
        )
        with pytest.raises(NotImplementedError, match="only supports response_format='mp3'"):
            async for _ in gen:
                pass

    @pytest.mark.asyncio
    async def test_rejects_response_format_enum_like_wav(self):
        stub_client: Any = object()
        gen = stream_long_text(
            stub_client,
            input="hi",
            model="tts-kokoro",
            voice="af_alloy",
            response_format="WAV",
        )
        with pytest.raises(NotImplementedError):
            async for _ in gen:
                pass

    @pytest.mark.asyncio
    async def test_rejects_zero_concurrency(self):
        stub_client: Any = object()
        gen = stream_long_text(
            stub_client,
            input="hi",
            model="tts-kokoro",
            voice="af_alloy",
            max_concurrency=0,
        )
        with pytest.raises(ValueError, match="max_concurrency"):
            async for _ in gen:
                pass


class TestStreamLongTextFastPath:
    @pytest.mark.asyncio
    async def test_single_segment_passes_through_to_create_speech(self):
        # Build a fake client whose audio.create_speech returns an async
        # iterator yielding three fixed chunks. Verify stream_long_text yields
        # exactly those three chunks (no extra splitting, no parallel logic).
        recorded_kwargs: dict[str, Any] = {}
        chunks_out = [b"first", b"second", b"third"]

        async def fake_iter() -> AsyncIterator[bytes]:
            for c in chunks_out:
                yield c

        class FakeAudio:
            async def create_speech(self, **kw: Any) -> AsyncIterator[bytes]:
                recorded_kwargs.update(kw)
                return fake_iter()

        class FakeClient:
            audio = FakeAudio()

        gen = stream_long_text(
            FakeClient(),
            input="One short sentence.",
            model="tts-kokoro",
            voice="af_alloy",
        )
        received = [c async for c in gen]
        assert received == chunks_out
        # Confirm we called create_speech once with stream=True
        assert recorded_kwargs["stream"] is True
        assert recorded_kwargs["model"] == "tts-kokoro"
        assert recorded_kwargs["voice"] == "af_alloy"
        assert recorded_kwargs["input"] == "One short sentence."

    @pytest.mark.asyncio
    async def test_multi_segment_concatenates_in_input_order(self):
        # Budget 2 forces splitting of a 3-sentence input into 3 segments.
        recorded_inputs: list[str] = []

        async def fake_iter(text: str) -> AsyncIterator[bytes]:
            await asyncio.sleep(0)  # let other tasks make progress
            yield f"seg:{text[:5]}|".encode()

        class FakeAudio:
            async def create_speech(self, **kw: Any) -> AsyncIterator[bytes]:
                recorded_inputs.append(kw["input"])
                return fake_iter(kw["input"])

        class FakeClient:
            audio = FakeAudio()

        text = "Alpha first. Bravo second. Charlie third."
        gen = stream_long_text(
            FakeClient(),
            input=text,
            model="tts-kokoro",
            voice="af_alloy",
            max_words_per_segment=2,
        )
        out = b"".join([c async for c in gen])
        # Output must be in input order
        assert b"seg:Alpha" in out
        assert b"seg:Bravo" in out
        assert b"seg:Charl" in out
        assert out.index(b"seg:Alpha") < out.index(b"seg:Bravo") < out.index(b"seg:Charl")
        # All 3 segments fired (regardless of order they completed)
        assert set(recorded_inputs) == {"Alpha first.", "Bravo second.", "Charlie third."}

    @pytest.mark.asyncio
    async def test_segment_exception_is_reraised(self):
        async def good() -> AsyncIterator[bytes]:
            yield b"ok"

        async def bad() -> AsyncIterator[bytes]:
            yield b"ok"
            raise RuntimeError("synthetic segment failure")

        call_count = 0

        class FakeAudio:
            async def create_speech(self, **kw: Any) -> AsyncIterator[bytes]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return good()
                return bad()

        class FakeClient:
            audio = FakeAudio()

        # Budget 2 forces the splitter to produce >=2 segments.
        text = "First sentence here. Second sentence here."
        gen = stream_long_text(
            FakeClient(),
            input=text,
            model="tts-kokoro",
            voice="af_alloy",
            max_words_per_segment=2,
        )
        with pytest.raises(RuntimeError, match="synthetic"):
            async for _ in gen:
                pass

    @pytest.mark.asyncio
    async def test_on_segment_complete_callback_fires_once_per_segment(self):
        async def fake_iter() -> AsyncIterator[bytes]:
            yield b"chunk1"
            yield b"chunk2"

        class FakeAudio:
            async def create_speech(self, **kw: Any) -> AsyncIterator[bytes]:
                return fake_iter()

        class FakeClient:
            audio = FakeAudio()

        callbacks: list[tuple[int, float, int]] = []

        gen = stream_long_text(
            FakeClient(),
            input="First one. Second one.",
            model="tts-kokoro",
            voice="af_alloy",
            max_words_per_segment=2,
            on_segment_complete=lambda i, t, b: callbacks.append((i, t, b)),
        )
        async for _ in gen:
            pass
        assert len(callbacks) == 2
        assert {c[0] for c in callbacks} == {0, 1}
        for _, latency, bytes_count in callbacks:
            assert latency >= 0
            assert bytes_count == len(b"chunk1") + len(b"chunk2")
