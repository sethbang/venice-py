"""
Shared helpers for Venice AI audio examples.

- ``detect_audio_format`` names the container of a block of audio bytes from
  its leading bytes (``"mp3"``, ``"wav"``, ``"flac"``, ``"aac"``, ``"m4a"``,
  ``"ogg"``), so examples can check what the server actually returned. To name
  a file, prefer the format the SDK reports from the response's content type
  (``audio_format`` on music and voice-changer completed statuses) and read
  the bytes only when that is unknown.
- ``matches_format`` checks bytes against a requested ``response_format``.
- ``tts_spec`` narrows a catalog entry's ``model_spec`` to ``TtsModelSpec``,
  and ``tts_formats`` / ``tts_default_format`` read the output formats a
  TTS entry lists.
"""

from venice_ai.types.api import TtsModelSpec

#: Response formats whose container this module can recognise. ``pcm`` is raw
#: headerless samples, so there is nothing to check.
CHECKABLE_FORMATS = frozenset({"mp3", "wav", "flac", "aac", "m4a", "opus", "ogg"})


def _id3_length(data: bytes) -> int:
    """Length of a leading ID3v2 tag, or ``0`` if there is none.

    The tag size is a 4-byte syncsafe integer (7 bits per byte) in bytes 6-9,
    plus a 10-byte footer when flag bit 4 is set. Generated audio can carry
    provenance metadata (a C2PA manifest) in such a tag, in front of FLAC as
    well as MP3.
    """
    if len(data) < 10 or data[:3] != b"ID3":
        return 0
    size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
    footer = 10 if data[5] & 0x10 else 0
    return 10 + size + footer


def detect_audio_format(data: bytes) -> str | None:
    """Return the audio container ``data`` starts with, or ``None`` if unknown.

    MPEG frames and AAC ADTS frames share the 0xFFF sync word. The two layer
    bits after it tell them apart: ``00`` is ADTS, anything else is an MPEG
    audio layer (MP3 is layer III).
    """
    tag = _id3_length(data)
    head = data[tag:]
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":
        return "m4a"
    if len(head) > 1 and head[0] == 0xFF and head[1] & 0xF0 == 0xF0 and head[1] & 0x06 == 0:
        return "aac"
    if len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[1] & 0x06:
        return "mp3"
    return None


def matches_format(data: bytes, fmt: str) -> bool:
    """Whether ``data`` is non-empty and is a ``fmt`` file.

    ``opus`` is carried in an Ogg container, and AAC may arrive as raw ADTS
    frames or in an MP4 (``m4a``) container. Formats this module cannot
    recognise (``pcm``) only need to be non-empty.
    """
    if not data:
        return False
    if fmt not in CHECKABLE_FORMATS:
        return True
    found = detect_audio_format(data)
    accepted = {"opus": {"ogg"}, "aac": {"aac", "m4a"}}.get(fmt, {fmt})
    return found in accepted


def tts_spec(spec: object, model_id: str) -> TtsModelSpec:
    """Return ``spec`` typed as a ``TtsModelSpec``.

    ``models.list()`` / ``models.get()`` type ``model_spec`` as the base
    ``ModelSpec``; the concrete class depends on the model's ``type``. Checking
    it here gives the TTS fields (``voices``, ``default_voice``,
    ``supported_formats``, ``default_format``, ``voice_cloning``) their types.

    :raises LookupError: If ``model_id``'s catalog entry is not a TTS spec.
    """
    if not isinstance(spec, TtsModelSpec):
        raise LookupError(f"{model_id} has no TTS spec in the catalog")
    return spec


def tts_formats(spec: TtsModelSpec) -> list[str]:
    """Return the output formats a TTS catalog entry lists (``supported_formats``).

    An entry that lists no formats gives an empty list.
    """
    return list(spec.supported_formats or [])


def tts_default_format(spec: TtsModelSpec, model_id: str) -> str:
    """Return the output format a TTS model uses when none is requested.

    Read from the entry's ``default_format``.

    :raises LookupError: If the entry lists no default format.
    """
    if not spec.default_format:
        raise LookupError(f"{model_id} lists no default_format in the catalog")
    return spec.default_format
