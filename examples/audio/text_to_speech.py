#!/usr/bin/env python3
"""
Venice AI SDK - Text-to-Speech Generation
==========================================

This example demonstrates how to generate speech from text using the Venice AI SDK.
Learn how to create high-quality audio from text with various voice options and formats.

Every request parameter is read from the resolved model's catalog entry:
voices come from ``model_spec.voices`` and formats from
``model_spec.supported_formats``, so the example keeps working whichever TTS
model ``resolve_tts(prefer="cheapest")`` returns. Each file is checked for the
container signature of the format that was requested (``_helpers.matches_format``,
shared with the other audio examples).

The style controls ``prompt``, ``temperature`` and ``top_p`` are sent only to a
model whose catalog entry advertises them, since other models ignore them. The API
reference names the flags, but a catalog can list none, so set
``VENICE_TTS_STYLE_MODEL`` to name a model and send all three to it.

With no TTS model in the catalog the example skips and exits 77. With no ASR model
to transcribe it back, the multi-language and style-control demos are skipped.

Cost: every clip bills per input character at the resolved model's price (about
1,000 characters in all, about $0.004 at the cheapest listed $3.50 per million).
The style-control clip runs on the cheapest model that advertises the parameters,
at up to $125 per million: under a cent for its 58 characters.
"""

import asyncio
import io
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

from _helpers import (
    CHECKABLE_FORMATS,
    EXIT_SKIPPED,
    matches_format,
    tts_default_format,
    tts_formats,
    tts_spec,
    word_overlap,
)

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import TtsModelSpec

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/audio/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: File extension for each response format. ``pcm`` is raw headerless samples.
FORMAT_EXTENSIONS = {
    "mp3": "mp3",
    "wav": "wav",
    "flac": "flac",
    "aac": "aac",
    "opus": "opus",
    "pcm": "pcm",
}

#: Native-language sample text (and ISO 639-1 code) per voice language.
#: Non-English voices are always given text in their own language.
NATIVE_TEXT: dict[str, tuple[str, str]] = {
    "Spanish": ("es", "Hola, ¿cómo estás hoy? Espero que tengas un buen día."),
    "French": (
        "fr",
        "Bonjour, comment allez-vous aujourd'hui ? Je vous souhaite une bonne journée.",
    ),
    "Italian": ("it", "Ciao, come stai oggi? Ti auguro una buona giornata."),
    "Portuguese": ("pt", "Olá, como você está hoje? Espero que tenha um bom dia."),
}

#: Names a model to send the style controls to, when the catalog flags none.
STYLE_MODEL_ENV_VAR = "VENICE_TTS_STYLE_MODEL"

#: Paths written by this run, listed at the end on success.
WRITTEN: list[Path] = []


class TtsModel:
    """The resolved TTS model and the catalog facts the demos need."""

    def __init__(
        self,
        model_id: str,
        spec: TtsModelSpec,
        english_voices: list[str],
        voice_languages: dict[str, str],
    ):
        self.id = model_id
        self.spec = spec
        self.default_format = tts_default_format(spec, model_id)
        self.formats = tts_formats(spec) or [self.default_format]
        self.english_voices = english_voices
        self.voice_languages = voice_languages

    def voice(self, index: int) -> str:
        """Pick an English voice, cycling through the model's list."""
        return self.english_voices[index % len(self.english_voices)]


async def load_tts_model(client: VeniceClient) -> TtsModel:
    """Resolve a TTS model and read its voices and formats from the catalog."""
    model_id = await client.models.resolve_tts(prefer="cheapest")
    catalog = await client.models.list(type="tts")
    spec = tts_spec(next(m.model_spec for m in catalog.data if m.id == model_id), model_id)
    voices = spec.voices or []

    # get_voices() adds language/gender metadata for voice IDs that follow the
    # "<region><gender>_<name>" convention. Voices without it have language None.
    details = (await client.audio.get_voices(model_id=model_id)).data
    voice_languages = {v.id: v.language for v in details if v.language}
    english = [v for v in voices if voice_languages.get(v) == "English"]
    # With no language metadata at all, every voice is a candidate.
    if not english and not voice_languages:
        english = list(voices)
    if not english:
        raise RuntimeError(f"{model_id} lists no English voice")
    return TtsModel(model_id, spec, english, voice_languages)


def wav_seconds(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


async def basic_text_to_speech(client: VeniceClient, tts: TtsModel) -> bool:
    """Generate basic speech from text."""
    print("🎤 Basic Text-to-Speech Generation")
    print("-" * 40)

    text = (
        "Hello! Welcome to Venice AI's text-to-speech service. "
        "This is a demonstration of basic speech generation."
    )
    fmt = tts.default_format
    voice = tts.voice(0)
    try:
        response = await client.audio.create_speech(
            model=tts.id,
            input=text,
            voice=voice,
            response_format=fmt,
            speed=1.0,
        )
    except VeniceError as e:
        print(f"❌ Error generating speech: {e}")
        return False

    filename = response.save(
        RESULTS_DIR / f"basic_speech.{FORMAT_EXTENSIONS.get(fmt, fmt)}", overwrite=True
    )
    WRITTEN.append(filename)
    print(f"✅ Generated speech with voice {voice!r} ({fmt})")
    print(f"💾 Saved as: {filename}")
    print(f"📏 File size: {len(response.content)} bytes")
    print(f"📝 Text: {text}")
    if response.headers:
        print(f"📊 Response headers available: {len(response.headers)} headers")
    if not matches_format(response.content, fmt):
        print(f"❌ The response is not {fmt} audio")
        return False
    return True


async def different_formats(client: VeniceClient, tts: TtsModel) -> bool:
    """Generate speech in every format the model supports."""
    print("\n🎧 Different Audio Formats")
    print("-" * 40)

    text = "This is a test of different audio formats."
    print(f"📋 Formats supported by {tts.id}: {', '.join(tts.formats)}")

    ok = True
    for fmt in tts.formats:
        extension = FORMAT_EXTENSIONS.get(fmt, fmt)
        print(f"\n🎵 Generating {fmt.upper()} format...")
        try:
            response = await client.audio.create_speech(
                model=tts.id,
                input=text,
                voice=tts.voice(1),
                response_format=fmt,
                speed=1.0,
            )
        except VeniceError as e:
            print(f"❌ Failed to generate {fmt.upper()}: {e}")
            ok = False
            continue

        filename = response.save(RESULTS_DIR / f"format_test.{extension}", overwrite=True)
        WRITTEN.append(filename)
        print(f"📏 Size: {len(response.content)} bytes")
        if not matches_format(response.content, fmt):
            print(f"❌ {filename.name} does not start like a {fmt.upper()} file")
            ok = False
            continue
        header = "header verified" if fmt in CHECKABLE_FORMATS else "no header to check"
        print(f"✅ Generated {fmt.upper()} ({header}): {filename}")
    return ok


async def speed_variations(client: VeniceClient, tts: TtsModel) -> bool:
    """Generate speech at different speeds and check the durations shrink."""
    print("\n⚡ Speed Variations")
    print("-" * 40)

    text = (
        "This sentence will be spoken at different speeds to demonstrate the speed control feature."
    )
    speeds = [0.5, 1.0, 1.5, 2.0]

    # WAV makes the duration exact to measure; fall back to the default format.
    fmt = "wav" if "wav" in tts.formats else tts.default_format
    durations: list[float] = []
    for speed in speeds:
        print(f"\n🏃 Generating speech at {speed}x speed...")
        try:
            response = await client.audio.create_speech(
                model=tts.id,
                input=text,
                voice=tts.voice(2),
                response_format=fmt,
                speed=speed,
            )
        except VeniceError as e:
            print(f"❌ Failed to generate {speed}x speed: {e}")
            return False

        filename = response.save(
            RESULTS_DIR / f"speed_{speed}x.{FORMAT_EXTENSIONS.get(fmt, fmt)}", overwrite=True
        )
        WRITTEN.append(filename)
        line = f"✅ Generated {speed}x speed: {filename} ({len(response.content)} bytes)"
        if fmt == "wav":
            durations.append(wav_seconds(response.content))
            line += f", {durations[-1]:.2f}s"
        print(line)

    if fmt != "wav":
        print(f"ℹ️ {tts.id} has no WAV output, so durations are not compared")
        return True
    if all(a > b for a, b in zip(durations, durations[1:])):
        print("\n📉 Each faster speed produced a shorter clip, as expected")
        return True
    print(f"\n❌ Durations did not shrink with speed: {[round(d, 2) for d in durations]}")
    return False


async def streaming_speech(client: VeniceClient, tts: TtsModel) -> bool:
    """Generate speech using streaming for real-time processing."""
    print("\n🌊 Streaming Speech Generation")
    print("-" * 40)

    text = (
        "This is a demonstration of streaming text-to-speech generation. "
        "The audio is generated and delivered in chunks for real-time processing applications."
    )
    fmt = tts.default_format
    print(f"📝 Text: {text}")
    print("🔄 Starting streaming generation...")

    filename = RESULTS_DIR / f"streamed_speech.{FORMAT_EXTENSIONS.get(fmt, fmt)}"
    chunk_count = 0
    total_bytes = 0
    first_chunk_at: float | None = None
    t0 = time.monotonic()
    try:
        stream = await client.audio.create_speech(
            model=tts.id,
            input=text,
            voice=tts.voice(3),
            response_format=fmt,
            speed=1.0,
            stream=True,
        )
        with open(filename, "wb") as f:
            async for chunk in stream:
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic() - t0
                chunk_count += 1
                total_bytes += len(chunk)
                f.write(chunk)
                print(f"📦 Received chunk {chunk_count}: {len(chunk)} bytes")
    except VeniceError as e:
        print(f"❌ Error in streaming generation: {e}")
        return False
    WRITTEN.append(filename)

    if not total_bytes:
        print("❌ The stream delivered no audio")
        return False
    if not matches_format(filename.read_bytes(), fmt):
        print(f"❌ The streamed bytes are not {fmt} audio")
        return False
    print("✅ Streaming complete!")
    print(f"💾 Saved as: {filename}")
    print(f"⏱️ First chunk after {first_chunk_at:.2f}s, finished after {time.monotonic() - t0:.2f}s")
    print(f"📊 Total chunks: {chunk_count}")
    print(f"📏 Total size: {total_bytes} bytes")
    return True


async def batch_text_processing(client: VeniceClient, tts: TtsModel) -> bool:
    """Generate speech for multiple texts."""
    print("\n📚 Batch Text Processing")
    print("-" * 40)

    texts = [
        "Welcome to our service.",
        "Thank you for your order.",
        "Your appointment has been confirmed.",
        "Have a great day!",
    ]
    fmt = tts.default_format
    print(f"📝 Processing {len(texts)} texts...")

    ok = True
    for i, text in enumerate(texts):
        print(f"\n🎤 Generating audio {i + 1}/{len(texts)}: {text}")
        try:
            response = await client.audio.create_speech(
                model=tts.id,
                input=text,
                voice=tts.voice(4),
                response_format=fmt,
                speed=1.0,
            )
        except VeniceError as e:
            print(f"❌ Failed to generate audio {i + 1}: {e}")
            ok = False
            continue

        filename = response.save(
            RESULTS_DIR / f"batch_{i + 1:02d}.{FORMAT_EXTENSIONS.get(fmt, fmt)}", overwrite=True
        )
        WRITTEN.append(filename)
        if not matches_format(response.content, fmt):
            print(f"❌ {filename.name} is not {fmt} audio")
            ok = False
            continue
        print(f"✅ Generated: {filename} ({len(response.content)} bytes)")
    return ok


async def multilingual_speech(client: VeniceClient, tts: TtsModel) -> bool | None:
    """Speak native-language text with a matching voice and ``language`` hint.

    The language is set by the voice for models whose voice IDs carry a
    language prefix. ``language`` is an optional hint that not every model
    reads: the catalog does not say which ones do, and models that don't read
    it ignore it. A transcription round-trip checks the result really is
    speech in the target language.

    Returns ``None`` when the model has no voice in a language this demo has
    sample text for, or the catalog has no ASR model to transcribe it back.
    """
    print("\n🌍 Multi-lingual TTS with a native voice")
    print("-" * 40)

    choice = next(
        ((voice, lang) for voice, lang in tts.voice_languages.items() if lang in NATIVE_TEXT),
        None,
    )
    if choice is None:
        print(
            f"Section skipped: multilingual speech: {tts.id} lists no voice for {', '.join(NATIVE_TEXT)}"
        )
        return None

    voice, language = choice
    code, text = NATIVE_TEXT[language]
    fmt = "wav" if "wav" in tts.formats else tts.default_format
    print(f"🗣️ Voice {voice!r} ({language}), language hint {code!r}")
    print("ℹ️ The catalog does not say which models read the hint; here the voice sets the language")
    print(f"📝 Text: {text}")
    try:
        # Before the speech is billed, since the check depends on it.
        asr_model = await client.models.resolve_asr(prefer="cheapest")
        response = await client.audio.create_speech(
            model=tts.id,
            input=text,
            voice=voice,
            response_format=fmt,
            language=code,
        )
        filename = response.save(
            RESULTS_DIR / f"speech_language_{code}.{FORMAT_EXTENSIONS.get(fmt, fmt)}",
            overwrite=True,
        )
        WRITTEN.append(filename)
        print(f"✅ Saved: {filename} ({len(response.content)} bytes)")

        transcript = await client.audio.transcribe(file=filename, model=asr_model, language=code)
    except NoMatchingModelError as e:
        print(f"Section skipped: multilingual speech: no ASR model to check it with ({e})")
        return None
    except VeniceError as e:
        print(f"❌ Error generating multi-lingual speech: {e}")
        return False

    overlap = word_overlap(text, transcript.text)
    print(f"🔁 Transcribed back ({asr_model}): {transcript.text}")
    print(f"📊 Word overlap with the source text: {overlap:.0%}")
    if overlap < 0.6:
        print(
            "❌ The transcript does not match the text: the speech is not in the expected language"
        )
        return False
    return True


def _advertises(spec: TtsModelSpec, flag: str) -> bool:
    """Whether a TTS catalog entry sets a capability flag the SDK does not type."""
    return bool((spec.model_extra or {}).get(flag))


async def style_controls(client: VeniceClient) -> bool | None:
    """Send ``prompt``, ``temperature`` and ``top_p`` where the catalog advertises them.

    The API ignores these on other models, so a request there would show
    nothing. Each one goes only to a model that sets its flag
    (``supportsPromptParam``, ``supportsTemperatureParam``, ``supportsTopPParam``).
    The audio has no field reporting the delivery, so the check is that the words
    survived; listen to the clip to judge the effect.

    ``VENICE_TTS_STYLE_MODEL`` names a model to send all three to instead. Returns
    ``None`` when neither gives a model or no ASR model can check the result.
    """
    print("\n🎚️ Style controls: prompt, temperature, top_p")
    print("-" * 40)

    text = "The harbor bell rang twice before the morning boats left."
    flags: dict[str, dict[str, Any]] = {
        "supportsPromptParam": {"prompt": "Calm and unhurried."},
        "supportsTemperatureParam": {"temperature": 0.7},
        "supportsTopPParam": {"top_p": 0.9},
    }
    specs = {
        m.id: tts_spec(m.model_spec, m.id) for m in (await client.models.list(type="tts")).data
    }
    named = os.environ.get(STYLE_MODEL_ENV_VAR)
    tried: list[str] = []
    while True:
        try:
            model_id = named or await client.models.resolve_tts(
                prefer="cheapest", exclude_models=tried
            )
        except NoMatchingModelError:
            print(
                "Section skipped: style controls: no TTS model advertises these parameters, "
                f"and {STYLE_MODEL_ENV_VAR} is not set"
            )
            return None
        tried.append(model_id)
        spec = specs.get(model_id)
        if named and spec is None:
            print(f"❌ {STYLE_MODEL_ENV_VAR}={named} is not a TTS model in the catalog")
            return False
        if spec is None or not spec.voices:
            if named:
                print(f"❌ {named} lists no voices in the catalog")
                return False
            continue
        params: dict[str, Any] = {}
        for flag, value in flags.items():
            if named or _advertises(spec, flag):
                params.update(value)
        if params:
            break

    try:
        asr_model = await client.models.resolve_asr(prefer="cheapest")
    except NoMatchingModelError as e:
        print(f"Section skipped: style controls: no ASR model to check it with ({e})")
        return None

    fmt = tts_default_format(spec, model_id)
    voice = spec.default_voice or spec.voices[0]
    print(f"🎤 {model_id}, voice {voice!r}, sending {params}")
    try:
        response = await client.audio.create_speech(
            model=model_id, input=text, voice=voice, response_format=fmt, **params
        )
        filename = response.save(
            RESULTS_DIR / f"style_controls.{FORMAT_EXTENSIONS.get(fmt, fmt)}", overwrite=True
        )
        WRITTEN.append(filename)
        transcript = await client.audio.transcribe(file=filename, model=asr_model, language="en")
    except VeniceError as e:
        print(f"❌ Error generating styled speech: {e}")
        return False

    overlap = word_overlap(text, transcript.text)
    print(f"🔁 Transcribed back ({asr_model}): {transcript.text}")
    print(f"📊 Word overlap with the source text: {overlap:.0%}")
    if not matches_format(response.content, fmt):
        print(f"❌ The response is not {fmt} audio")
        return False
    if overlap < 0.6:
        print("❌ The clip does not say the requested text")
        return False
    print(f"✅ Saved: {filename}; listen to it to judge the delivery")
    return True


async def main() -> int:
    """Run all text-to-speech examples.

    Returns ``0`` only if every demo succeeded or was skipped for lack of a
    matching voice or ASR model, ``77`` when the catalog has no TTS model, and
    ``1`` otherwise.
    """
    print("🚀 Venice AI Text-to-Speech Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        try:
            tts = await load_tts_model(client)
        except NoMatchingModelError as e:
            print(f"\nSKIPPED: no TTS model to generate speech with ({e})")
            return EXIT_SKIPPED
        print(f"📍 Using TTS model: {tts.id}")
        print(f"🎭 {len(tts.spec.voices or [])} voices, default format {tts.default_format!r}\n")

        results: list[tuple[str, bool | None]] = [
            ("basic_text_to_speech", await basic_text_to_speech(client, tts)),
            ("different_formats", await different_formats(client, tts)),
            ("speed_variations", await speed_variations(client, tts)),
            ("streaming_speech", await streaming_speech(client, tts)),
            ("batch_text_processing", await batch_text_processing(client, tts)),
            ("multilingual_speech", await multilingual_speech(client, tts)),
            ("style_controls", await style_controls(client)),
        ]

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    if skipped:
        print(f"\n⏭️ Skipped: {', '.join(skipped)}")
    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Text-to-speech examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Voices and formats read from the model's catalog entry")
    print("   - Every supported output format")
    print("   - Speed control (0.5x to 2.0x), checked against clip duration")
    print("   - Streaming audio generation")
    print("   - Batch text processing")
    if "multilingual_speech" not in skipped:
        print("   - Native-language voice and text, verified by transcription")
    if "style_controls" not in skipped:
        print("   - prompt, temperature and top_p on a model that advertises them")

    print("\n📁 Files written by this run:")
    for path in WRITTEN:
        print(f"   - {path.relative_to(RESULTS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print(
            "Check that your API key is valid and you have audio generation access.",
            file=sys.stderr,
        )
        sys.exit(1)
