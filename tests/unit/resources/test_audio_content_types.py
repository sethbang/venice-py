"""TDD: OGG/OGA/WebM audio content-types (audit MED #6).

`.ogg` was detected by magic bytes but absent from content_type_map (so it fell
back to application/octet-stream), and WebM/EBML wasn't detected at all.
"""

import io
from unittest.mock import Mock

import pytest

from venice_ai.resources.audio import Audio


@pytest.fixture
def audio() -> Audio:
    return Audio(Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "magic,exp_name,exp_ct",
    [
        (b"OggS\x00\x00\x00\x00", "audio.ogg", "audio/ogg"),
        (b"\x1a\x45\xdf\xa3\x00\x00\x00\x00", "audio.webm", "audio/webm"),
    ],
)
async def test_prepare_detects_ogg_and_webm_from_magic(audio, magic, exp_name, exp_ct):
    _content, name, ct = await audio._prepare_audio_file(magic)
    assert name == exp_name
    assert ct == exp_ct


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fname,exp_ct",
    [("clip.ogg", "audio/ogg"), ("clip.oga", "audio/ogg"), ("clip.webm", "audio/webm")],
)
async def test_prepare_content_type_for_extension(audio, fname, exp_ct, tmp_path):
    p = tmp_path / fname
    p.write_bytes(b"\x00\x01\x02\x03")
    _content, _name, ct = await audio._prepare_audio_file(str(p))
    assert ct == exp_ct


@pytest.mark.parametrize(
    "magic,exp_name",
    [
        # MPEG-1 Layer III without the common 0x90 bitrate byte
        (b"\xff\xfb\x50\x00", "audio.mp3"),
        # MPEG-2 Layer III, e.g. a stream that starts on a bare frame header
        (b"\xff\xf3\x64\xc4", "audio.mp3"),
        (b"\xff\xf2\x64\xc4", "audio.mp3"),
        # AAC ADTS, MPEG-4 and MPEG-2 IDs, with and without CRC
        (b"\xff\xf1\x50\x80", "audio.aac"),
        (b"\xff\xf9\x50\x80", "audio.aac"),
        (b"\xff\xf0\x50\x80", "audio.aac"),
    ],
)
def test_detect_distinguishes_mpeg_frame_sync_from_adts(magic, exp_name):
    assert Audio._detect_audio_filename(magic) == exp_name


class _NamelessReader:
    """A binary file-like object with no ``name`` attribute."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


STITCHED_MP3_HEAD = b"\xff\xf3\x64\xc4" + b"\x00" * 60


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_file",
    [
        pytest.param(io.BytesIO, id="BytesIO"),
        pytest.param(_NamelessReader, id="nameless-file-like"),
    ],
)
@pytest.mark.parametrize(
    "magic,exp_name,exp_ct",
    [
        pytest.param(STITCHED_MP3_HEAD, "audio.mp3", "audio/mpeg", id="mp3-bare-frame"),
        pytest.param(b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio.wav", "audio/wav", id="wav"),
        pytest.param(b"OggS\x00\x00\x00\x00", "audio.ogg", "audio/ogg", id="ogg"),
    ],
)
async def test_prepare_detects_format_of_unnamed_file_objects(
    audio, make_file, magic, exp_name, exp_ct
):
    content, name, ct = await audio._prepare_audio_file(make_file(magic))
    assert content == magic
    assert name == exp_name
    assert ct == exp_ct


@pytest.mark.asyncio
async def test_prepare_keeps_the_name_of_a_named_file_object(audio, tmp_path):
    p = tmp_path / "clip.flac"
    p.write_bytes(STITCHED_MP3_HEAD)
    with p.open("rb") as fh:
        _content, name, ct = await audio._prepare_audio_file(fh)
    assert name == "clip.flac"
    assert ct == "audio/flac"
