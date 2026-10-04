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
"""

import asyncio
import io
import re
import sys
import time
import wave
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api import TtsModelSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import (  # noqa: E402
    CHECKABLE_FORMATS,
    matches_format,
    tts_default_format,
    tts_formats,
    tts_spec,
)

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


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_overlap(expected: str, actual: str) -> float:
    """Fraction of the expected words that appear in the transcript."""
    want = _words(expected)
    got = set(_words(actual))
    return sum(w in got for w in want) / len(want) if want else 0.0


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
    sample text for.
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

        asr_model = await client.models.resolve_asr(prefer="cheapest")
        transcript = await client.audio.transcribe(file=filename, model=asr_model, language=code)
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


async def main() -> int:
    """Run all text-to-speech examples.

    Returns ``0`` only if every demo succeeded or was skipped for lack of a
    matching voice, ``1`` otherwise.
    """
    print("🚀 Venice AI Text-to-Speech Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        tts = await load_tts_model(client)
        print(f"📍 Using TTS model: {tts.id}")
        print(f"🎭 {len(tts.spec.voices or [])} voices, default format {tts.default_format!r}\n")

        results: list[tuple[str, bool | None]] = [
            ("basic_text_to_speech", await basic_text_to_speech(client, tts)),
            ("different_formats", await different_formats(client, tts)),
            ("speed_variations", await speed_variations(client, tts)),
            ("streaming_speech", await streaming_speech(client, tts)),
            ("batch_text_processing", await batch_text_processing(client, tts)),
            ("multilingual_speech", await multilingual_speech(client, tts)),
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
