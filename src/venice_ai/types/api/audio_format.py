"""Map an audio media type to the file extension a caller should use."""

_AUDIO_FORMATS: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/vnd.wave": "wav",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
}


def audio_format_for(content_type: str | None) -> str | None:
    """Map an audio media type to a file extension; ``None`` if unknown."""
    if not content_type:
        return None
    return _AUDIO_FORMATS.get(content_type.split(";", 1)[0].strip().lower())


__all__ = ["audio_format_for"]
