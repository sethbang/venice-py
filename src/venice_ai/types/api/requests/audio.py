"""
Audio and TTS request models for Venice.ai API.
"""

from pydantic import BaseModel, Field

from ...identifiers import ModelId

# ============================================================================
# Audio Request Models
# ============================================================================


class AudioSpeechRequest(BaseModel):
    """Text-to-speech request"""

    input: str = Field(..., min_length=1, max_length=4096, description="Text to generate audio for")

    # Model and voice
    model: ModelId | None = Field(default="tts-kokoro", description="TTS model to use")
    voice: str | None = Field(default="af_sky", description="Voice to use for generation")

    # Audio parameters
    response_format: str | None = Field(default="mp3", description="Audio format")
    speed: float | None = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")
    streaming: bool | None = Field(default=False, description="Stream audio sentence by sentence")

    # Model-specific style / sampling controls (ignored by models that don't advertise support)
    language: str | None = Field(
        default=None,
        description=(
            "Optional language hint. Accepted values are model-specific: Qwen 3 "
            "accepts full names (English, Chinese, ...); xAI/ElevenLabs accept "
            "ISO 639-1 codes (en, ja, ...); MiniMax accepts full names. Unsupported "
            "values are silently ignored. Omit to let the model auto-detect."
        ),
    )
    prompt: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Style prompt controlling emotion and delivery. Supported by models "
            "advertising ``supportsPromptParam`` (currently Qwen 3 TTS). Ignored otherwise."
        ),
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description=(
            "Sampling temperature for generation. Supported by models advertising "
            "``supportsTemperatureParam`` (Qwen 3, Orpheus, Chatterbox HD)."
        ),
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Nucleus sampling parameter. Supported by models advertising "
            "``supportsTopPParam`` (currently Qwen 3 TTS)."
        ),
    )


class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription (STT)."""

    model: str = Field(..., description="ASR model ID (e.g., 'nvidia/parakeet-tdt-0.6b-v3')")
    response_format: str | None = Field(default=None, description="Response format: 'json'")
    timestamps: bool | None = Field(default=None, description="Include word-level timestamps")
    language: str | None = Field(default=None, description="Language of the input audio")


# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    "AudioSpeechRequest",
    "AudioTranscriptionRequest",
]
