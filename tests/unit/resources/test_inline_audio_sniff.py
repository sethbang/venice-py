"""Audio sniffing for inline retrieve bodies, checked against crafted headers.

Each header is built from the MPEG audio, ADTS and ID3v2 header layouts, so a
case documents the bits it relies on.
"""

from __future__ import annotations

import pytest

from venice_ai.resources._inline_audio import _skip_id3, read_inline_audio, sniff_audio_format
from venice_ai.types.api.audio_format import audio_format_for

# MPEG-1 Layer III, 128 kbps, 44.1 kHz: sync 0xFFE, version 11, layer 01.
MP3_FRAME = bytes([0xFF, 0xFB, 0x90, 0x64]) + b"\x00" * 16
# MPEG-2 Layer III, 64 kbps, 22.05 kHz.
MP3_MPEG2_FRAME = bytes([0xFF, 0xF3, 0x80, 0xC4]) + b"\x00" * 16
# MPEG-1 Layer II: valid MPEG audio, but not MP3.
MP2_FRAME = bytes([0xFF, 0xFD, 0x90, 0x04]) + b"\x00" * 16
# ADTS: sync 0xFFF, ID 0 (MPEG-4), layer 00, protection_absent 1.
ADTS_MPEG4 = bytes([0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC]) + b"\x00" * 8
# ADTS with ID 1 (MPEG-2) and a CRC (protection_absent 0).
ADTS_MPEG2_CRC = bytes([0xFF, 0xF8, 0x50, 0x80, 0x02, 0x1F, 0xFC]) + b"\x00" * 8
FLAC = b"fLaC" + b"\x00" * 16


def id3(payload_size: int, *, footer: bool = False) -> bytes:
    """An ID3v2.4 tag of ``payload_size`` bytes, with a footer when asked."""
    size = bytes(
        [
            (payload_size >> 21) & 0x7F,
            (payload_size >> 14) & 0x7F,
            (payload_size >> 7) & 0x7F,
            payload_size & 0x7F,
        ]
    )
    flags = 0x10 if footer else 0x00
    tag = b"ID3\x04\x00" + bytes([flags]) + size + b"T" * payload_size
    if footer:
        tag += b"3DI\x04\x00" + bytes([flags]) + size
    return tag


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (MP3_FRAME, "mp3"),
        (MP3_MPEG2_FRAME, "mp3"),
        (ADTS_MPEG4, "aac"),
        (ADTS_MPEG2_CRC, "aac"),
        (FLAC, "flac"),
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", "wav"),
        (b"OggS" + b"\x00" * 12, "ogg"),
        (b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 8, "m4a"),
        (MP2_FRAME, None),
        (b"<html>not audio</html>", None),
        (b"", None),
    ],
)
def test_sniffs_bare_streams(body: bytes, expected: str | None) -> None:
    assert sniff_audio_format(body) == expected


def test_adts_is_not_mistaken_for_mp3() -> None:
    # Both carry an all-ones sync; only the layer bits differ.
    assert sniff_audio_format(ADTS_MPEG4) == "aac"
    assert sniff_audio_format(id3(20) + ADTS_MPEG4) == "aac"


def test_id3_footer_is_skipped() -> None:
    body = id3(32, footer=True) + FLAC
    assert _skip_id3(body) == FLAC
    assert sniff_audio_format(body) == "flac"


def test_id3_without_footer_is_skipped() -> None:
    assert sniff_audio_format(id3(32) + FLAC) == "flac"
    assert sniff_audio_format(id3(32) + MP3_FRAME) == "mp3"


def test_id3_followed_by_unknown_bytes_is_not_assumed_mp3() -> None:
    assert sniff_audio_format(id3(8) + b"garbage-bytes!") is None


def test_non_syncsafe_size_is_not_a_tag() -> None:
    bogus = b"ID3\x04\x00\x00\x00\x00\x80\x00" + FLAC
    assert _skip_id3(bogus) == bogus
    assert sniff_audio_format(bogus) is None


def test_truncated_tag_header_is_left_alone() -> None:
    assert _skip_id3(b"ID3\x04") == b"ID3\x04"


class _Body:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.closed = False

    async def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_octet_stream_adts_is_reported_as_aac() -> None:
    media_type, body = await read_inline_audio(
        _Body(ADTS_MPEG4), "application/octet-stream", operation="test.retrieve"
    )
    assert media_type == "audio/aac"
    assert audio_format_for(media_type) == "aac"
    assert body == ADTS_MPEG4
