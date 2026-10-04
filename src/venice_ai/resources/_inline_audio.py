"""Read audio that an async-job retrieve endpoint returns inline.

Music and voice-changer jobs deliver finished audio as the body of the
retrieve response rather than as a download URL. Both resources read it here,
so they accept and reject the same bodies and report the same media type.
"""

from __future__ import annotations

from typing import Any

from ..audio_helpers import _id3v2_tag_length, _layer3_frame_layout
from ..exceptions import APIResponseProcessingError
from ..utils.errors import read_body


def _skip_id3(data: bytes) -> bytes:
    """Return *data* after a leading ID3v2 tag, if any.

    The tag length, footer included, comes from the same ID3v2 header parser
    that MP3 stitching uses. A header whose size bytes are not syncsafe is not
    a tag, and *data* is returned unchanged. Venice prefixes some FLAC output
    with a tag.
    """
    if len(data) < 10:
        return data
    length = _id3v2_tag_length(data)
    return data if length is None else data[length:]


def _is_adts(head: bytes) -> bool:
    """Whether *head* starts with an AAC ADTS frame header.

    ADTS opens with the 12-bit sync word ``0xFFF``, an MPEG version bit and a
    2-bit layer field that is always ``00``. MPEG audio frames share the sync
    bits but never use layer ``00``, so the layer field tells the two apart.
    """
    return len(head) >= 2 and head[0] == 0xFF and head[1] & 0xF6 == 0xF0


def sniff_audio_format(data: bytes) -> str | None:
    """Identify common audio containers by their leading bytes.

    A leading ID3v2 tag is skipped first; the tag itself says nothing about
    the audio after it, so a tag followed by unrecognised bytes is ``None``.
    Returns ``"flac"``, ``"wav"``, ``"ogg"``, ``"m4a"``, ``"aac"`` (raw ADTS
    frames), ``"mp3"`` (a valid MPEG audio Layer III frame header) or ``None``.
    """
    head = _skip_id3(data)
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if _is_adts(head):
        return "aac"
    if len(head) >= 4 and _layer3_frame_layout(head) is not None:
        return "mp3"
    return None


# Canonical media type for each sniffed format.
_SNIFFED_MEDIA_TYPES: dict[str, str] = {
    "flac": "audio/flac",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "mp3": "audio/mpeg",
}


async def read_inline_audio(
    raw_response: Any, content_type: str, *, operation: str
) -> tuple[str, bytes]:
    """Read a non-JSON retrieve body as audio and return ``(media_type, body)``.

    An ``audio/*`` Content-Type is taken as given. ``application/octet-stream``
    is accepted when the bytes are a known audio container, and reported as
    that container's media type. Anything else (an HTML proxy page, plain
    text, an empty body) is not audio and raises.

    Args:
        raw_response: The unread aiohttp response; it is read and closed.
        content_type: The response's Content-Type header.
        operation: Name used in error messages, e.g. ``"music.retrieve"``.

    Raises:
        APIResponseProcessingError: If the body is empty or not audio.
    """
    media_type = content_type.split(";", 1)[0].strip().lower()
    try:
        body = await read_body(raw_response)
    finally:
        raw_response.close()
    if media_type.startswith("audio/"):
        if not body:
            raise APIResponseProcessingError(
                f"{operation} returned an empty {media_type} body",
                response=raw_response,
            )
    elif media_type == "application/octet-stream":
        sniffed = sniff_audio_format(body)
        if sniffed is None:
            raise APIResponseProcessingError(
                f"{operation} returned application/octet-stream that is not a "
                f"recognised audio format; preview: {body[:120]!r}",
                response=raw_response,
            )
        media_type = _SNIFFED_MEDIA_TYPES[sniffed]
    else:
        raise APIResponseProcessingError(
            f"{operation} returned a 2xx response with Content-Type "
            f"{content_type!r}, which is neither a JSON status nor audio; "
            f"preview: {body[:200]!r}",
            response=raw_response,
        )
    return media_type, body
