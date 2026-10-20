"""
Audio and TTS models for Venice AI API.

This module contains Pydantic models for audio processing and text-to-speech
functionality including voice management and audio response handling.
"""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VoiceDetail(BaseModel):
    """Detailed information about a single text-to-speech voice.

    This model represents comprehensive metadata for an individual voice in the Venice AI
    text-to-speech system. It contains all available information about a voice including
    its unique identifier, associated TTS model, gender characteristics, and detailed
    regional/language information derived from voice naming conventions.

    **Voice Naming Convention:**
    Voice IDs follow a structured pattern: ``{region_code}_{voice_name}`` where:
    * Region codes indicate language/locale and gender (e.g., "af" = American Female, "zm" = Chinese Male)
    * Voice names provide unique identification within the region category

    **Metadata Derivation:**
    Language, accent, and gender information is intelligently parsed from voice IDs
    using Venice AI's standardized naming system, providing rich metadata for voice
    selection and filtering operations.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    """The unique identifier for the voice as provided by the Venice AI API.

    Examples: ``"af_alloy"``, ``"zm_yunjian"``, ``"bf_emma"``
    This ID is used in text-to-speech requests to specify the desired voice.
    """

    model_id: str
    """The ID of the TTS model this voice is associated with.

    Example: ``"tts-kokoro"``
    Different models may have varying capabilities, quality levels, or
    specialized features for specific voice types or languages.
    """

    gender: Literal["male", "female", "unknown"] | None
    """The perceived gender of the voice, parsed from the voice ID prefix.

    * ``"male"``: Masculine voice characteristics
    * ``"female"``: Feminine voice characteristics
    * ``"unknown"``: Gender not determinable from voice ID prefix

    Derived automatically from voice naming conventions where possible.
    """

    region_code: str | None
    """The raw two-letter prefix from the voice ID indicating region/language and gender.

    Examples: ``"af"`` (American Female), ``"zm"`` (Chinese Male), ``"bf"`` (British Female)
    This code encodes both linguistic and gender information in Venice AI's naming system.
    """

    language: str | None
    """A descriptive name of the primary language associated with the voice.

    Examples: ``"American English"``, ``"Mandarin Chinese"``, ``"British English"``
    Derived from the region_code using Venice AI's language mapping system.
    """

    accent: str | None
    """A descriptive name of the accent or locale associated with the voice.

    Examples: ``"US"``, ``"Standard Chinese"``, ``"UK"``
    Provides specific regional or cultural accent information derived from
    the region_code for more precise voice characterization.
    """


class ClonedVoice(BaseModel):
    """Result of cloning a voice via ``POST /v1/audio/voices``.

    ``id`` is the ``vv_<id>`` handle to pass back as the ``voice`` parameter on
    ``create_speech`` — it must be paired with the same ``model`` it was
    created for. Handles expire after the per-model retention window (currently
    7 days); for ``tts-minimax-speech-02-hd`` each successful TTS call resets it.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Cloned-voice handle (``vv_<id>``).")
    model: str = Field(..., description="The TTS model this voice handle is paired with.")


class VoiceList(BaseModel):
    """A comprehensive list of voice details with optional filtering metadata.

    This model represents the complete response structure returned by voice listing
    operations, containing an array of VoiceDetail objects along with metadata about
    any filters that were applied to generate the list.
    """

    model_config = ConfigDict(extra="allow")

    object: Literal["list"]
    """Type identifier for the response object, always ``"list"`` for collection responses."""

    data: list[VoiceDetail]
    """Array of voice detail objects containing comprehensive metadata for each voice."""

    model_id_filter: str | None
    """The model ID that was used to filter the voices, if any."""

    gender_filter: Literal["male", "female", "unknown"] | None
    """The gender characteristic that was used to filter the voices, if any."""

    region_code_filter: str | None
    """The region code that was used to filter the voices, if any."""


class AudioResponse(BaseModel):
    """Response wrapper for audio endpoints that preserves HTTP headers.

    This class ensures that rate-limiting headers and other metadata from
    audio API responses are accessible to the scheduler for proper rate
    limit tracking.
    """

    model_config = ConfigDict(extra="allow")

    content: bytes
    """Binary audio data returned from the text-to-speech endpoint.

    Contains the complete audio file in the requested format (MP3, WAV, etc.).
    Can be written directly to a file or streamed to clients.
    """

    headers: dict[str, str] | None = None
    """HTTP response headers from the audio API request.

    Includes rate limiting headers like ``x-ratelimit-remaining-requests`` and
    ``x-ratelimit-reset-requests`` for proper rate limit tracking. Essential
    for scheduler integration and usage monitoring.
    """

    def __init__(self, content: bytes, response: Any = None, **kwargs: Any):
        """Initialize AudioResponse with content and optional response object."""
        headers = None
        if response and hasattr(response, "headers"):
            headers = dict(response.headers)
        super().__init__(content=content, headers=headers, **kwargs)

    def __len__(self) -> int:
        """Return the length of the audio content."""
        return len(self.content)

    def __getitem__(self, key: Any) -> int | bytes:
        """Support slicing operations like bytes[0:20].

        Type Coercion Logic:
        - Integer index (e.g., audio[0]) returns int (single byte value)
        - Slice operation (e.g., audio[0:10]) returns bytes object
        - The conditional logic handles both cases for proper type narrowing
        """
        result = self.content[key]
        # Ensure we return the correct type based on the slice/index operation
        if isinstance(result, (int, bytes)):
            return result
        # Fallback for edge cases - convert iterable to bytes or single value to int
        return bytes(result) if hasattr(result, "__iter__") else int(result)

    def __bytes__(self) -> bytes:
        """Support ``bytes(response)``."""
        return self.content

    def __buffer__(self, flags: int) -> Any:
        """Support buffer protocol for file writing operations (Python 3.12+)."""
        return self.content.__buffer__(flags)

    def startswith(self, prefix: Any) -> bool:
        """Support startswith operations like bytes.startswith()."""
        return self.content.startswith(prefix)

    def iter_bytes(self) -> Any:
        """Support iteration over bytes content."""
        return iter(self.content)

    def __eq__(self, other: Any) -> bool:
        """Support equality comparison."""
        if isinstance(other, AudioResponse):
            return self.content == other.content
        elif isinstance(other, bytes):
            return self.content == other
        return False

    def __hash__(self) -> int:
        """Support hashing."""
        return hash(self.content)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save audio content to file.

        Performs synchronous file I/O. When called from an async coroutine,
        wrap with ``await asyncio.to_thread(response.save, path)`` for large
        outputs.

        :param path: Destination file path.
        :param overwrite: If ``False`` (default) and *path* exists, raise
            :class:`FileExistsError`.
        :return: The resolved :class:`Path` of the saved file.
        :raises FileExistsError: If the file exists and ``overwrite=False``.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.content)
        return path


class TranscriptionWord(BaseModel):
    """Individual word with timing data."""

    model_config = ConfigDict(extra="allow")

    word: str = Field(..., description="The word")
    start: float | None = Field(default=None, description="Start time in seconds")
    end: float | None = Field(default=None, description="End time in seconds")


class TranscriptionSegment(BaseModel):
    """Individual segment with timing data (sentence/phrase-level)."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="The segment text")
    start: float | None = Field(default=None, description="Start time in seconds")
    end: float | None = Field(default=None, description="End time in seconds")


class TranscriptionChar(BaseModel):
    """Individual character with timing data."""

    model_config = ConfigDict(extra="allow")

    char: str = Field(..., description="The character")
    start: float | None = Field(default=None, description="Start time in seconds")
    end: float | None = Field(default=None, description="End time in seconds")


class TranscriptionTimestamps(BaseModel):
    """Nested timestamps envelope returned by ``/audio/transcriptions``.

    The Venice API delivers word/segment/char timing data inside a
    ``timestamps`` object — not as a top-level ``words`` array.
    """

    model_config = ConfigDict(extra="allow")

    word: list[TranscriptionWord] | None = Field(
        default=None, description="Word-level timing entries"
    )
    segment: list[TranscriptionSegment] | None = Field(
        default=None, description="Segment-level timing entries"
    )
    char: list[TranscriptionChar] | None = Field(
        default=None, description="Character-level timing entries"
    )


class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription (``POST /audio/transcriptions``).

    The API returns ``{text, duration, timestamps:{word,segment,char}}`` per
    the spec. ``words`` (legacy top-level alias) is populated from
    ``timestamps.word`` for backward compatibility.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="The transcribed text")
    duration: float | None = Field(
        default=None, description="Audio duration in seconds (when reported by the model)"
    )
    timestamps: TranscriptionTimestamps | None = Field(
        default=None,
        description="Word / segment / char level timestamps when ``timestamps=true``.",
    )

    @property
    def words(self) -> list[TranscriptionWord] | None:
        """Word-level timestamps (alias for ``timestamps.word``)."""
        return self.timestamps.word if self.timestamps else None


__all__ = [
    "ClonedVoice",
    "VoiceDetail",
    "VoiceList",
    "AudioResponse",
    "TranscriptionWord",
    "TranscriptionSegment",
    "TranscriptionChar",
    "TranscriptionTimestamps",
    "AudioTranscriptionResponse",
]
