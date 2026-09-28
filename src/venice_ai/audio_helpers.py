"""Helpers for Venice TTS that work around model-specific output limits.

The main entry point is :func:`stream_long_text`. It splits a long input into
sentence-aligned segments, dispatches them to ``client.audio.create_speech``
in parallel, and yields the concatenated audio bytes in input order.

Two server-side issues motivate this helper:

1. **tts-qwen3-0-6b and tts-qwen3-1-7b silently truncate output at exactly
   15.896875 s (664 MP3 frames at 24 kHz, identical for both sizes).**
   Past ~25 words of input, the model either speeds up unnaturally or stops
   mid-sentence. Reproduced across all four ``Accept-Encoding`` variants and
   both qwen3 sizes — the cap is server-side, not transport.

2. **Six of ten Venice TTS models buffer the full response before sending
   any bytes**, despite ``stream=True``: qwen3 family, orpheus, chatterbox,
   inworld, gemini. Only ``tts-xai-v1`` and ``tts-kokoro`` deliver chunks
   progressively. Parallel segment fan-out converts the buffering models
   into pseudo-streaming because chunk 1 can be in flight while chunk 0
   is still generating.

See :data:`MODEL_WORD_BUDGETS` for the per-model split thresholds. Update
that constant — not the helper's logic — when Venice ships fixes or new
models. This file is intentionally the only place that knows about the
qwen3 ceiling.

Known limitation: voice drift across segments
---------------------------------------------
On qwen3-family models, successive segments rendered through this helper
may exhibit subtle voice timbre / tone differences ("voice drift") even
with identical ``voice``, ``model``, and input style. Cause: ``/audio/speech``
does not currently accept a ``seed`` parameter, so each segment call samples
fresh RNG state on the server.

Empirical test (12-line poem on tts-qwen3-1-7b/Serena):
``temperature`` / ``top_p`` adjustments (0.2/0.8 and 0.05/0.5) and a stable
style ``prompt`` were tried as anchors. None produced an audibly-clear
improvement over Venice defaults; run-to-run variance dominated any
configuration delta. ``temperature``, ``top_p``, and ``prompt`` remain
caller-controlled on :func:`stream_long_text` for use cases where they
help, but no default is baked in. The real fix is a server-side seed
parameter on ``/audio/speech``.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._client import VeniceClient
    from .types.enums import ResponseFormat, Voice

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model-specific output limits
# ---------------------------------------------------------------------------
# Per-model word budget for :func:`split_text_for_tts`. Each segment is sized
# so its expected output stays well under the server-side cap.
#
# **tts-qwen3-0-6b / tts-qwen3-1-7b**: hard cap of 664 MP3 frames =
# **15.896875 s** of audio. Identical cap on both model sizes (strong signal
# this is a hardcoded inference-path limit, not a per-model artifact).
# Natural pace for these models is ~2.2 words/sec, so 25 words ≈ 11 s of
# audio — leaves ~5 s headroom for slower deliveries.
#
# **Other Venice TTS models**: no truncation observed up to ~3500 chars of
# input (kokoro tested to 177 s output, xai-v1 to 224 s output). The default
# budget below is a parallelism-vs-overhead tradeoff, not a correctness one.
#
# If a Venice fix removes the qwen3 cap, set the entries to
# ``DEFAULT_WORD_BUDGET`` (or remove them) — do not change the
# splitter logic itself.
MODEL_WORD_BUDGETS: dict[str, int] = {
    "tts-qwen3-0-6b": 25,
    "tts-qwen3-1-7b": 25,
}

#: Words per segment for any model not in :data:`MODEL_WORD_BUDGETS`.
#: Chosen to produce ~30-45 s segments at typical TTS pace, which is large
#: enough to amortize per-request overhead and small enough that ~4 in
#: parallel saturate a typical user's perceived audio playback rate.
DEFAULT_WORD_BUDGET: int = 100


# Splits on sentence-ending punctuation that's followed by whitespace.
# Newlines inside a paragraph (single \n) are treated as sentence separators
# because TTS-friendly inputs often have line-broken poetry / formatted text.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])\n|\n(?=[A-Z])")


def split_text_for_tts(
    text: str,
    model: str | None = None,
    *,
    max_words: int | None = None,
) -> list[str]:
    """Split text into TTS-friendly segments only when the budget is exceeded.

    For text that fits under ``max_words`` the result is ``[text]`` and the
    caller can pass it to ``create_speech`` unchanged — the helper is a
    no-op for short inputs.

    When the input exceeds the budget the splitter is greedy: packs
    sentences (respecting paragraph and sentence terminators as boundaries)
    into a segment until the next sentence would push word count over
    ``max_words``, then starts a new segment.

    A single sentence longer than ``max_words`` is emitted as its own
    over-budget segment — splitting mid-sentence produces worse audio than
    one over-budget chunk.

    Empirical note: paragraph-boundary forcing was tried earlier and removed.
    Several Venice TTS models render each segment faster than the
    corresponding portion of a longer call (kokoro shrinks ~30 % when given
    one stanza at a time instead of three together), so forcing extra splits
    when not needed costs audible audio with no upside.

    :param text: Input text. Empty / whitespace-only returns ``[""]``.
    :param model: Venice TTS model id. Used to look up
        :data:`MODEL_WORD_BUDGETS` when ``max_words`` is not given.
    :param max_words: Override the per-model word budget.
    :return: Segments in input order. Always at least one element.
    :raises ValueError: If ``max_words`` is given and not positive.
    """
    if max_words is not None and max_words <= 0:
        raise ValueError(f"max_words must be positive, got {max_words!r}")

    if max_words is None:
        max_words = MODEL_WORD_BUDGETS.get(model or "", DEFAULT_WORD_BUDGET)

    text = text.strip()
    if not text:
        return [""]

    # Fast path: total input under budget → no splitting at all.
    if len(text.split()) <= max_words:
        return [text]

    # Flatten paragraphs + sentences into an ordered sentence list, then
    # greedily pack. Paragraph boundaries are treated as sentence boundaries
    # for splitting purposes but otherwise carry no weight.
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for raw in _SENTENCE_SPLIT_RE.split(paragraph):
            s = raw.strip()
            if s:
                sentences.append(s)

    segments: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            segments.append(" ".join(current))
            current = []
            current_words = 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if n_words > max_words:
            flush()
            segments.append(sentence)
            continue
        if current_words + n_words > max_words:
            flush()
        current.append(sentence)
        current_words += n_words
    flush()

    return segments or [text]


# ---------------------------------------------------------------------------
# MP3 segment stitching
# ---------------------------------------------------------------------------
# Every ``create_speech`` MP3 response is a complete file: an ID3v2 tag, then a
# LAME ``Info`` (CBR) or ``Xing`` (VBR) header frame declaring that file's
# frame count, then the audio frames. Appending responses verbatim yields a
# stream whose first header frame claims only segment 0's frame count, so
# header-trusting consumers (browsers, ``afinfo``, ASR front ends) report the
# duration of the first segment and may stop there; the later tags and header
# frames also sit mid-stream as bytes that are not audio. Multi-segment output
# therefore drops each segment's leading ID3v2 tag(s) and Xing/Info/VBRI frame,
# including segment 0's, and emits bare MPEG audio frames. Players then derive
# duration from the frame bitrate, which is exact for Venice's CBR output.

# (is_mpeg1) -> Layer III bitrate table in kb/s, indexed by the header nibble.
_LAYER3_BITRATES_KBPS: dict[bool, tuple[int, ...]] = {
    True: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320),
    False: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160),
}
# MPEG version bits -> sample rates in Hz, indexed by the header field.
_SAMPLE_RATES_HZ: dict[int, tuple[int, int, int]] = {
    3: (44100, 48000, 32000),  # MPEG-1
    2: (22050, 24000, 16000),  # MPEG-2
    0: (11025, 12000, 8000),  # MPEG-2.5
}
# (is_mpeg1, mono) -> Layer III side-information length in bytes.
_LAYER3_SIDE_INFO_BYTES: dict[tuple[bool, bool], int] = {
    (True, True): 17,
    (True, False): 32,
    (False, True): 9,
    (False, False): 17,
}
_VBRI_OFFSET = 36


def _id3v2_tag_length(data: bytes | bytearray) -> int | None:
    """Total length of the ID3v2 tag at the start of ``data`` (10+ bytes).

    Returns ``None`` if ``data`` does not start with a valid ID3v2 header.
    """
    if data[:3] != b"ID3":
        return None
    size_bytes = data[6:10]
    if any(b & 0x80 for b in size_bytes):
        return None
    size = (
        ((size_bytes[0] & 0x7F) << 21)
        | ((size_bytes[1] & 0x7F) << 14)
        | ((size_bytes[2] & 0x7F) << 7)
        | (size_bytes[3] & 0x7F)
    )
    footer = 10 if data[5] & 0x10 else 0
    return 10 + size + footer


def _layer3_frame_layout(data: bytes | bytearray) -> tuple[int, int] | None:
    """Return ``(frame_length, side_info_length)`` for a Layer III frame header.

    ``data`` must hold at least the 4 header bytes. Returns ``None`` for
    anything that is not a parseable MPEG Layer III frame header.
    """
    b0, b1, b2, b3 = data[0], data[1], data[2], data[3]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    version = (b1 >> 3) & 0x03
    layer_bits = (b1 >> 1) & 0x03
    if version == 1 or layer_bits != 1:
        return None
    bitrate_idx = b2 >> 4
    sr_idx = (b2 >> 2) & 0x03
    if bitrate_idx in (0, 15) or sr_idx == 3:
        return None
    is_mpeg1 = version == 3
    bitrate = _LAYER3_BITRATES_KBPS[is_mpeg1][bitrate_idx] * 1000
    sample_rate = _SAMPLE_RATES_HZ[version][sr_idx]
    padding = (b2 >> 1) & 0x01
    length = (144 if is_mpeg1 else 72) * bitrate // sample_rate + padding
    mono = (b3 >> 6) == 3
    return length, _LAYER3_SIDE_INFO_BYTES[(is_mpeg1, mono)]


def _is_vbr_header_frame(frame: bytes | bytearray, side_info: int) -> bool:
    """Whether a complete Layer III frame is a Xing/Info/VBRI header frame."""
    tag_at = 4 + side_info
    if frame[tag_at : tag_at + 4] in (b"Xing", b"Info"):
        return True
    return frame[_VBRI_OFFSET : _VBRI_OFFSET + 4] == b"VBRI"


class _Mp3SegmentHeaderStripper:
    """Drop one MP3 segment's leading ID3v2 tag(s) and Xing/Info/VBRI frame.

    Stateful across network chunks: a tag split over any number of chunks is
    skipped in full, and at most one audio frame (under 1.5 KB) is buffered
    while deciding whether the first frame is a header frame. Once the first
    frame has been handled, every later byte passes through unchanged.
    """

    def __init__(self) -> None:
        self._buf = bytearray()
        self._skip = 0
        self._done = False

    def feed(self, chunk: bytes) -> bytes:
        """Consume one chunk and return the bytes that are safe to emit."""
        if self._done:
            return chunk
        if self._skip:
            n = min(self._skip, len(chunk))
            self._skip -= n
            chunk = chunk[n:]
            if self._skip:
                return b""
        self._buf += chunk
        return self._advance(final=False)

    def flush(self) -> bytes:
        """Return any bytes still held once the segment's stream has ended."""
        if self._done:
            return b""
        return self._advance(final=True)

    def _finish(self) -> bytes:
        self._done = True
        out = bytes(self._buf)
        self._buf.clear()
        return out

    def _advance(self, *, final: bool) -> bytes:
        buf = self._buf
        while True:
            if len(buf) < 10 and b"ID3".startswith(bytes(buf[:3])):
                if not final:
                    return b""
                if buf[:3] == b"ID3":
                    # Truncated tag at end of stream: it carries no audio.
                    buf.clear()
                return self._finish()
            tag_len = _id3v2_tag_length(buf)
            if tag_len is None:
                break
            if tag_len <= len(buf):
                del buf[:tag_len]
                continue
            self._skip = tag_len - len(buf)
            buf.clear()
            if final:
                return self._finish()
            return b""

        if len(buf) < 4:
            return self._finish() if final else b""
        layout = _layer3_frame_layout(buf)
        if layout is None:
            return self._finish()
        frame_len, side_info = layout
        if len(buf) < frame_len:
            return self._finish() if final else b""
        if _is_vbr_header_frame(buf[:frame_len], side_info):
            del buf[:frame_len]
        return self._finish()


# Type alias for the progress callback
SegmentCallback = Callable[[int, float, int], None]


async def stream_long_text(
    client: VeniceClient,
    *,
    input: str,
    model: str,
    voice: str | Voice,
    response_format: str | ResponseFormat = "mp3",
    speed: float | None = None,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_words_per_segment: int | None = None,
    max_concurrency: int = 4,
    on_segment_complete: SegmentCallback | None = None,
    timeout: float | None = None,
) -> AsyncIterator[bytes]:
    """Stream TTS audio for long input by splitting into parallel segments.

    Splits ``input`` via :func:`split_text_for_tts`, dispatches each segment
    to ``client.audio.create_speech(stream=True, ...)`` with bounded
    concurrency, and yields the resulting mp3 bytes in input order.

    Only ``response_format="mp3"`` is supported. Other formats require
    container-level demuxing to concatenate cleanly and would produce
    malformed output if naively appended.

    If ``input`` fits in a single segment, this short-circuits to
    ``create_speech`` directly — no extra task scheduling overhead — and the
    response bytes are yielded unchanged.

    When the input spans several segments, each segment's leading ID3v2 tag
    and Xing/Info/VBRI header frame are removed as its bytes stream through,
    so the output is one continuous MPEG audio stream with no tag metadata.
    A per-segment header frame would otherwise declare only that segment's
    frame count, and players that trust it report a truncated duration.

    :param client: A connected ``VeniceClient``.
    :param input: Text to synthesize.
    :param model: Venice TTS model id (e.g. ``"tts-qwen3-1-7b"``).
    :param voice: Model-specific voice id.
    :param response_format: Only ``"mp3"`` supported in this helper.
    :param max_words_per_segment: Override the per-model budget.
    :param max_concurrency: Concurrent in-flight create_speech calls.
        Default 4 keeps headroom under Venice's 60 req/min limit for typical
        single-document use.
    :param on_segment_complete: Optional callback invoked as each segment
        finishes its stream, with ``(segment_index, latency_seconds, bytes)``.
    :param timeout: Per-segment timeout in seconds, forwarded to
        ``create_speech``.

    :raises NotImplementedError: If ``response_format`` is not ``"mp3"``.
    :raises ValueError: If ``max_concurrency < 1``.

    Exceptions raised inside any segment task are re-raised when that
    segment's bytes would have been yielded; later segments are cancelled.
    """
    if str(response_format).lower() != "mp3":
        raise NotImplementedError(
            f"stream_long_text currently only supports response_format='mp3'; "
            f"got {response_format!r}. wav/flac/opus/aac/pcm need container-"
            f"level concatenation and are not yet implemented."
        )
    if max_concurrency < 1:
        raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

    segments = split_text_for_tts(input, model, max_words=max_words_per_segment)

    # Forward only non-None per-segment kwargs so model defaults stay intact.
    common_kwargs: dict[str, Any] = {
        "model": model,
        "voice": voice,
        "response_format": response_format,
        "stream": True,
    }
    for k, v in (
        ("speed", speed),
        ("language", language),
        ("prompt", prompt),
        ("temperature", temperature),
        ("top_p", top_p),
        ("timeout", timeout),
    ):
        if v is not None:
            common_kwargs[k] = v

    # Fast path: single segment, no parallel orchestration overhead.
    if len(segments) == 1:
        stream = await client.audio.create_speech(input=segments[0], **common_kwargs)
        async for chunk in stream:
            yield chunk
        return

    logger.info(
        "stream_long_text: %d chars → %d segments (model=%s, max_concurrency=%d)",
        len(input),
        len(segments),
        model,
        max_concurrency,
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    # Each queue carries chunks for one segment. Sentinel: None for clean
    # end-of-segment, a BaseException for segment failure.
    queues: list[asyncio.Queue[bytes | None | BaseException]] = [asyncio.Queue() for _ in segments]

    async def fetch_segment(idx: int, segment_text: str) -> None:
        t0 = time.monotonic()
        total_bytes = 0
        try:
            async with semaphore:
                stream = await client.audio.create_speech(input=segment_text, **common_kwargs)
                stripper = _Mp3SegmentHeaderStripper()
                async for chunk in stream:
                    out = stripper.feed(chunk)
                    if out:
                        total_bytes += len(out)
                        await queues[idx].put(out)
                tail = stripper.flush()
                if tail:
                    total_bytes += len(tail)
                    await queues[idx].put(tail)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised via queue
            await queues[idx].put(exc)
        else:
            if on_segment_complete is not None:
                try:
                    on_segment_complete(idx, time.monotonic() - t0, total_bytes)
                except Exception:  # noqa: BLE001
                    logger.warning("on_segment_complete callback raised; ignoring")
        finally:
            await queues[idx].put(None)

    tasks = [asyncio.create_task(fetch_segment(i, seg)) for i, seg in enumerate(segments)]

    try:
        for q in queues:
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
