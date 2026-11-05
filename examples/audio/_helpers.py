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
- ``voice_sample`` returns a speech recording for the voice cloning and voice
  changer examples: your own (``VENICE_VOICE_SAMPLE_PATH``) or a clip it
  synthesizes with Venice TTS. ``wav_seconds`` measures a WAV file.
"""

import os
import wave
from collections.abc import Collection
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.types.api import TtsModelSpec

#: Response formats whose container this module can recognise. ``pcm`` is raw
#: headerless samples, so there is nothing to check.
CHECKABLE_FORMATS = frozenset({"mp3", "wav", "flac", "aac", "m4a", "opus", "ogg"})

#: Set this to the path of your own recording to use it instead of a synthesized
#: sample.
VOICE_SAMPLE_ENV = "VENICE_VOICE_SAMPLE_PATH"

#: What the synthesized sample says: about 9 s of speech at a normal pace, longer
#: than the catalog's recommended cloning minimum and short enough to fit in the
#: single response of a model that cuts long audio off.
VOICE_SAMPLE_TEXT = (
    "Clear morning light falls across the quiet harbor while small boats drift slowly "
    "past the old stone wall and gulls circle overhead."
)


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


def wav_seconds(path: Path) -> float:
    """Length of a PCM WAV file, in seconds.

    :raises wave.Error: If the file is not a PCM WAV file.
    :raises EOFError: If the file is truncated.
    """
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


async def _cheapest_wav_tts(client: VeniceClient) -> tuple[str, str]:
    """``(model, voice)`` for the cheapest TTS model that can output WAV.

    WAV because both endpoints that take a sample accept it and its length can
    be measured here. Prefers an English voice, since the sample text is English.

    :raises venice_ai.NoMatchingModelError: If no TTS model outputs WAV.
    """
    specs = {
        m.id: tts_spec(m.model_spec, m.id) for m in (await client.models.list(type="tts")).data
    }
    tried: list[str] = []
    while True:
        model_id = await client.models.resolve_tts(prefer="cheapest", exclude_models=tried)
        tried.append(model_id)
        spec = specs.get(model_id)
        if spec is None or "wav" not in tts_formats(spec) or not spec.voices:
            continue
        details = (await client.audio.get_voices(model_id=model_id)).data
        english = [v.id for v in details if v.language == "English"]
        return model_id, (english or spec.voices)[0]


async def voice_sample(
    client: VeniceClient,
    results_dir: Path,
    *,
    accepted_formats: Collection[str] | None = None,
    min_seconds: float = 0.0,
) -> Path:
    """Return a speech recording to clone or convert.

    Only use a voice you have consent to use. Set ``VENICE_VOICE_SAMPLE_PATH``
    to your own recording, or to one whose speaker agreed to this use, and that
    file is returned as it is, with no billed call. Otherwise a short neutral
    sentence is synthesized with the cheapest TTS model that outputs WAV and
    saved as ``voice_sample.wav`` in ``results_dir``. That TTS call is billed.

    :param accepted_formats: The containers the receiving endpoint takes, from
        the catalog. Checked before the billed call.
    :param min_seconds: The shortest clip to accept, from the catalog. A
        synthesized clip under it is rejected before anything is uploaded.
    :raises FileNotFoundError: If ``VENICE_VOICE_SAMPLE_PATH`` names no file.
    :raises venice_ai.NoMatchingModelError: If no TTS model outputs WAV.
    :raises LookupError: If the endpoint does not accept WAV.
    :raises ValueError: If the clip is not WAV or is under ``min_seconds``.
    :raises wave.Error: If the clip cannot be measured.
    :raises EOFError: If the clip is truncated.
    """
    own = os.environ.get(VOICE_SAMPLE_ENV)
    if own:
        path = Path(own).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{VOICE_SAMPLE_ENV} names no file: {path}")
        print(f"   Using your recording ({VOICE_SAMPLE_ENV}): {path}")
        return path

    if accepted_formats is not None and "wav" not in {f.lower() for f in accepted_formats}:
        raise LookupError(f"The endpoint takes {sorted(accepted_formats)}, not a WAV sample")
    model_id, voice = await _cheapest_wav_tts(client)
    print(
        f"   💸 Synthesizing a sample with {model_id} (voice {voice!r}); this TTS call is billed."
    )
    print(f"      Set {VOICE_SAMPLE_ENV} to use your own recording instead.")
    response = await client.audio.create_speech(
        model=model_id,
        input=VOICE_SAMPLE_TEXT,
        voice=voice,
        response_format="wav",
    )
    if not matches_format(response.content, "wav"):
        raise ValueError(f"{model_id} did not return a WAV file for the sample")
    path = response.save(results_dir / "voice_sample.wav", overwrite=True)
    seconds = wav_seconds(path)
    print(f"      Saved {seconds:.1f}s of speech → {path}")
    if seconds < min_seconds:
        raise ValueError(f"The {seconds:.1f}s sample is under the {min_seconds:g}s minimum")
    return path
