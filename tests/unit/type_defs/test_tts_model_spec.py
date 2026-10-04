"""TtsModelSpec declares every field the live TTS catalog sends.

The samples are entries of ``GET /models?type=tts`` captured from the live
catalog (voices trimmed). A field the spec does not declare would land in
``model_extra`` as an untyped dict, so each test asserts the typed attribute
and that nothing is left over in ``model_extra``.
"""

from __future__ import annotations

from typing import Any

import pytest

from venice_ai.types.api.models import (
    KNOWN_VOICE_CLONING_MODES,
    ModelResponse,
    TtsModelSpec,
    TtsVoiceCloning,
)

KOKORO: dict[str, Any] = {
    "created": 1742418046,
    "id": "tts-kokoro",
    "model_spec": {
        "pricing": {"input": {"usd": 3.5, "diem": 3.5}},
        "default_format": "mp3",
        "supported_formats": ["mp3", "opus", "aac", "flac", "wav", "pcm"],
        "supports_custom_voice_id": False,
        "voices": ["af_alloy", "af_aoede"],
        "name": "Kokoro Text to Speech",
        "modelSource": "https://huggingface.co/hexgrad/Kokoro-82M",
        "offline": False,
        "privacy": "private",
        "traits": [],
    },
    "object": "model",
    "owned_by": "venice.ai",
    "type": "tts",
}
CHATTERBOX_HD: dict[str, Any] = {
    "created": 1776384000,
    "id": "tts-chatterbox-hd",
    "model_spec": {
        "pricing": {"input": {"usd": 50, "diem": 50}},
        "default_format": "wav",
        "supported_formats": ["wav"],
        "supports_custom_voice_id": False,
        "voices": ["Aurora", "Blade"],
        "voice_cloning": {
            "mode": "zero_shot",
            "accepted_formats": ["mp3", "wav", "flac", "mp4"],
            "min_sample_seconds": 5,
            "retention_days": 7,
        },
        "name": "Chatterbox HD (Resemble AI)",
        "offline": False,
        "privacy": "private",
        "traits": [],
    },
    "object": "model",
    "owned_by": "venice.ai",
    "type": "tts",
}
ELEVENLABS: dict[str, Any] = {
    "created": 1776384000,
    "id": "tts-elevenlabs-turbo-v2-5",
    "model_spec": {
        "pricing": {"input": {"usd": 62.5, "diem": 62.5}},
        "default_format": "mp3",
        "supported_formats": ["mp3"],
        "supports_custom_voice_id": True,
        "voices": ["Alice", "Aria"],
        "name": "ElevenLabs Turbo v2.5",
        "offline": False,
        "privacy": "anonymized",
        "traits": [],
    },
    "object": "model",
    "owned_by": "venice.ai",
    "type": "tts",
}


def _spec(entry: dict[str, Any]) -> TtsModelSpec:
    spec = ModelResponse.model_validate(entry).model_spec
    assert isinstance(spec, TtsModelSpec)
    return spec


@pytest.mark.parametrize("entry", [KOKORO, CHATTERBOX_HD, ELEVENLABS], ids=lambda e: e["id"])
def test_live_tts_entries_leave_nothing_undeclared(entry: dict[str, Any]) -> None:
    spec = _spec(entry)
    assert not spec.model_extra
    assert spec.supported_formats == entry["model_spec"]["supported_formats"]
    assert spec.default_format == entry["model_spec"]["default_format"]
    assert spec.supports_custom_voice_id is entry["model_spec"]["supports_custom_voice_id"]


def test_voice_cloning_is_typed() -> None:
    cloning = _spec(CHATTERBOX_HD).voice_cloning
    assert isinstance(cloning, TtsVoiceCloning)
    assert cloning.mode in KNOWN_VOICE_CLONING_MODES
    assert cloning.accepted_formats == ["mp3", "wav", "flac", "mp4"]
    assert cloning.min_sample_seconds == 5
    assert cloning.retention_days == 7
    assert not cloning.model_extra


def test_models_without_cloning_report_none() -> None:
    assert _spec(KOKORO).voice_cloning is None
    assert _spec(ELEVENLABS).supports_custom_voice_id is True


def test_unknown_cloning_mode_and_partial_object_still_parse() -> None:
    entry = {**CHATTERBOX_HD, "model_spec": {**CHATTERBOX_HD["model_spec"]}}
    entry["model_spec"]["voice_cloning"] = {"mode": "future_mode"}
    cloning = _spec(entry).voice_cloning
    assert cloning is not None
    assert cloning.mode == "future_mode"
    assert cloning.accepted_formats is None
