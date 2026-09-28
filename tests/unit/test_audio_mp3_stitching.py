"""Stream-level contract for MP3 audio stitched by ``stream_long_text``.

When a long input is split into several segments, each ``create_speech``
response is a complete MP3 file: an ID3v2 tag, a LAME ``Info``/``Xing``
header frame declaring that segment's frame count, then the audio frames.
The concatenated output must still be *one* well-formed MP3 stream:

* at most one ID3v2 tag, and only at offset 0;
* at most one Xing/Info/VBRI header frame, placed first, whose declared
  frame count matches the audio frames actually present (header-trusting
  consumers such as ``afinfo``, browsers and ASR front ends read duration
  from it);
* no bytes that are neither a tag nor a complete MPEG audio frame;
* every audio frame from every segment, in order.

The fixture segments in ``data/tts_mp3`` were produced with ffmpeg/LAME in
the same shape the Venice TTS endpoint returns (24 kHz mono, 64 kb/s CBR,
MPEG-2 Layer III, ID3v2.4 tag followed by a LAME ``Info`` frame). The tag
body is ~1.5 KB so that realistic network chunk sizes split it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from venice_ai.audio_helpers import stream_long_text

FIXTURE_DIR = Path(__file__).parent / "data" / "tts_mp3"
SEGMENT_FILES = sorted(FIXTURE_DIR.glob("segment_*.mp3"))

# Three sentences, one per segment when max_words_per_segment=2.
THREE_SENTENCES = "Alpha first. Bravo second. Charlie third."


# ---------------------------------------------------------------------------
# Minimal MPEG audio / ID3v2 stream parser
# ---------------------------------------------------------------------------

_BITRATES_KBPS = {
    # (is_mpeg1, layer) -> table ; only Layer III is produced by Venice TTS
    (True, 3): [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
    (False, 3): [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}
_SAMPLE_RATES = {
    3: [44100, 48000, 32000],  # MPEG-1
    2: [22050, 24000, 16000],  # MPEG-2
    0: [11025, 12000, 8000],  # MPEG-2.5
}


def _id3v2_total_size(data: bytes, pos: int) -> int | None:
    if data[pos : pos + 3] != b"ID3" or pos + 10 > len(data):
        return None
    size_bytes = data[pos + 6 : pos + 10]
    if any(b & 0x80 for b in size_bytes):
        return None
    size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
    footer = 10 if data[pos + 5] & 0x10 else 0
    return 10 + size + footer


def _frame_info(data: bytes, pos: int) -> tuple[int, int] | None:
    """Return ``(frame_length, side_info_length)`` for a Layer III header at ``pos``."""
    if pos + 4 > len(data):
        return None
    b0, b1, b2, b3 = data[pos : pos + 4]
    if b0 != 0xFF or (b1 & 0xE0) != 0xE0:
        return None
    version = (b1 >> 3) & 0x03
    layer_bits = (b1 >> 1) & 0x03
    if version == 1 or layer_bits != 1:  # reserved version / not Layer III
        return None
    bitrate_idx = b2 >> 4
    sr_idx = (b2 >> 2) & 0x03
    if bitrate_idx in (0, 15) or sr_idx == 3:
        return None
    is_mpeg1 = version == 3
    bitrate = _BITRATES_KBPS[(is_mpeg1, 3)][bitrate_idx] * 1000
    sample_rate = _SAMPLE_RATES[version][sr_idx]
    padding = (b2 >> 1) & 0x01
    coef = 144 if is_mpeg1 else 72
    length = coef * bitrate // sample_rate + padding
    mono = (b3 >> 6) == 3
    side_info_lengths = {(True, True): 17, (True, False): 32, (False, True): 9, (False, False): 17}
    return length, side_info_lengths[(is_mpeg1, mono)]


@dataclass
class Mp3Layout:
    id3_offsets: list[int] = field(default_factory=list)
    header_frames: list[tuple[int, bytes, int | None]] = field(default_factory=list)
    audio_frames: int = 0
    first_frame_offset: int | None = None
    junk_bytes: int = 0
    junk_offsets: list[int] = field(default_factory=list)

    def problems(self) -> list[str]:
        out: list[str] = []
        if len(self.id3_offsets) > 1 or any(o != 0 for o in self.id3_offsets):
            out.append(f"ID3v2 tags at offsets {self.id3_offsets} (only one, at 0, is allowed)")
        if len(self.header_frames) > 1:
            out.append(
                "multiple Xing/Info/VBRI header frames: "
                + ", ".join(
                    f"{tag.decode()}@{off} declares {n}" for off, tag, n in self.header_frames
                )
            )
        if self.header_frames:
            off, tag, declared = self.header_frames[0]
            if off != self.first_frame_offset:
                out.append(f"{tag.decode()} header frame at {off} is not the first frame")
            if declared is not None and declared != self.audio_frames:
                out.append(
                    f"{tag.decode()} header declares {declared} frames but the stream "
                    f"holds {self.audio_frames} audio frames"
                )
        if self.junk_bytes:
            out.append(
                f"{self.junk_bytes} bytes are neither ID3 nor MPEG frames "
                f"(first at offsets {self.junk_offsets[:5]})"
            )
        return out


def parse_mp3(data: bytes) -> Mp3Layout:
    layout = Mp3Layout()
    pos = 0
    in_junk = False
    while pos < len(data):
        id3_len = _id3v2_total_size(data, pos)
        if id3_len is not None and pos + id3_len <= len(data):
            layout.id3_offsets.append(pos)
            pos += id3_len
            in_junk = False
            continue
        info = _frame_info(data, pos)
        if info is not None and pos + info[0] <= len(data):
            length, side_info = info
            tag_at = pos + 4 + side_info
            tag = data[tag_at : tag_at + 4]
            vbri = data[pos + 36 : pos + 40]
            if layout.first_frame_offset is None:
                layout.first_frame_offset = pos
            if tag in (b"Xing", b"Info"):
                flags = int.from_bytes(data[tag_at + 4 : tag_at + 8], "big")
                declared = (
                    int.from_bytes(data[tag_at + 8 : tag_at + 12], "big") if flags & 0x1 else None
                )
                layout.header_frames.append((pos, tag, declared))
            elif vbri == b"VBRI":
                declared = int.from_bytes(data[pos + 36 + 14 : pos + 36 + 18], "big")
                layout.header_frames.append((pos, vbri, declared))
            else:
                layout.audio_frames += 1
            pos += length
            in_junk = False
            continue
        if not in_junk:
            layout.junk_offsets.append(pos)
            in_junk = True
        layout.junk_bytes += 1
        pos += 1
    return layout


# ---------------------------------------------------------------------------
# Fake client serving the fixture segments in fixed-size network chunks
# ---------------------------------------------------------------------------


def _segments() -> list[bytes]:
    return [p.read_bytes() for p in SEGMENT_FILES]


def _chunked(data: bytes, sizes: Sequence[int]) -> list[bytes]:
    out: list[bytes] = []
    pos = 0
    i = 0
    while pos < len(data):
        size = max(1, sizes[i % len(sizes)])
        out.append(data[pos : pos + size])
        pos += size
        i += 1
    return out


def _fake_client(segment_bytes: list[bytes], chunk_sizes: Sequence[int]) -> Any:
    by_input: dict[str, bytes] = {}
    order = ["Alpha first.", "Bravo second.", "Charlie third."]
    for text, body in zip(order, segment_bytes, strict=False):
        by_input[text] = body

    async def body_iter(body: bytes) -> AsyncIterator[bytes]:
        for chunk in _chunked(body, chunk_sizes):
            await asyncio.sleep(0)
            yield chunk

    class FakeAudio:
        async def create_speech(self, **kw: Any) -> AsyncIterator[bytes]:
            return body_iter(by_input[kw["input"]])

    class FakeClient:
        audio = FakeAudio()

    return FakeClient()


async def _stitch(
    segment_bytes: list[bytes], chunk_sizes: Sequence[int], text: str = THREE_SENTENCES
) -> bytes:
    gen = stream_long_text(
        _fake_client(segment_bytes, chunk_sizes),
        input=text,
        model="tts-kokoro",
        voice="af_alloy",
        max_words_per_segment=2,
    )
    return b"".join([c async for c in gen])


def _total_audio_frames(segment_bytes: list[bytes]) -> int:
    return sum(parse_mp3(b).audio_frames for b in segment_bytes)


def _without_id3(body: bytes) -> bytes:
    tag_len = _id3v2_total_size(body, 0)
    assert tag_len is not None
    return body[tag_len:]


def _without_header_frame(body: bytes) -> bytes:
    layout = parse_mp3(body)
    assert len(layout.header_frames) == 1
    off = layout.header_frames[0][0]
    info = _frame_info(body, off)
    assert info is not None
    return body[:off] + body[off + info[0] :]


# ---------------------------------------------------------------------------
# The checker itself must select real frames and flag naive concatenation
# ---------------------------------------------------------------------------


class TestMp3LayoutChecker:
    def test_fixtures_present(self):
        assert len(SEGMENT_FILES) == 3

    @pytest.mark.parametrize("path", SEGMENT_FILES, ids=lambda p: p.name)
    def test_each_single_response_is_well_formed(self, path: Path):
        layout = parse_mp3(path.read_bytes())
        assert layout.audio_frames > 0
        assert layout.id3_offsets == [0]
        assert len(layout.header_frames) == 1
        assert layout.header_frames[0][1] == b"Info"
        assert layout.problems() == []

    def test_naive_concatenation_is_flagged(self):
        segs = _segments()
        problems = parse_mp3(b"".join(segs)).problems()
        assert any("ID3v2 tags" in p for p in problems)
        assert any("multiple Xing/Info/VBRI" in p for p in problems)
        assert any("declares" in p and "audio frames" in p for p in problems)

    def test_leaked_partial_tag_is_flagged_as_junk(self):
        segs = _segments()
        leaked = segs[0] + segs[1][5:]  # tag magic dropped, tag body left behind
        assert parse_mp3(leaked).junk_bytes > 0


# ---------------------------------------------------------------------------
# stream_long_text output contract
# ---------------------------------------------------------------------------


class TestStitchedMp3IsSingleStream:
    @pytest.mark.asyncio
    async def test_single_header_frame_matching_total_frames(self):
        segs = _segments()
        out = await _stitch(segs, [len(max(segs, key=len))])
        layout = parse_mp3(out)
        assert len(layout.header_frames) <= 1, layout.problems()
        assert layout.problems() == []

    @pytest.mark.asyncio
    async def test_no_audio_frames_lost(self):
        segs = _segments()
        out = await _stitch(segs, [4096])
        assert parse_mp3(out).audio_frames == _total_audio_frames(segs)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunk_size", [5, 333, 4096])
    async def test_segments_without_tag_or_header_frame_keep_every_audio_frame(
        self, chunk_size: int
    ):
        seg0, seg1, seg2 = _segments()
        variants = [seg0, _without_id3(seg1), _without_header_frame(seg2)]
        assert parse_mp3(variants[1]).id3_offsets == []
        assert parse_mp3(variants[2]).header_frames == []
        out = await _stitch(variants, [chunk_size])
        layout = parse_mp3(out)
        assert layout.audio_frames == _total_audio_frames(variants)
        assert layout.junk_bytes == 0, layout.problems()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunk_size", [7, 64, 512, 1024])
    async def test_id3_tag_split_across_chunks_does_not_leak(self, chunk_size: int):
        segs = _segments()
        tag_len = _id3v2_total_size(segs[1], 0)
        assert tag_len is not None and tag_len > chunk_size
        out = await _stitch(segs, [chunk_size])
        layout = parse_mp3(out)
        assert layout.junk_bytes == 0, layout.problems()
        assert all(o == 0 for o in layout.id3_offsets), layout.problems()

    @pytest.mark.asyncio
    async def test_single_segment_fast_path_is_well_formed(self):
        seg = _segments()[0]
        gen = stream_long_text(
            _fake_client([seg], [333]),
            input="Alpha first.",
            model="tts-kokoro",
            voice="af_alloy",
            max_words_per_segment=2,
        )
        out = b"".join([c async for c in gen])
        assert parse_mp3(out).problems() == []


@settings(max_examples=40, deadline=None)
@given(
    n_segments=st.integers(min_value=2, max_value=3),
    chunk_sizes=st.lists(st.integers(min_value=1, max_value=6000), min_size=1, max_size=6),
)
def test_stitched_output_is_one_well_formed_mp3_for_any_chunking(
    n_segments: int, chunk_sizes: list[int]
):
    segs = _segments()[:n_segments]
    text = " ".join(["Alpha first.", "Bravo second.", "Charlie third."][:n_segments])
    out = asyncio.run(_stitch(segs, chunk_sizes, text))
    layout = parse_mp3(out)
    assert layout.problems() == []
    assert layout.audio_frames == _total_audio_frames(segs)
