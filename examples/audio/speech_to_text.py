#!/usr/bin/env python3
"""
Venice AI SDK - Speech-to-Text Transcription
=============================================

This example demonstrates how to transcribe audio to text using the Venice AI SDK.
Learn how to convert speech audio into text with word-level timestamps, different
input methods, and model discovery.

The demo is self-contained: it first generates a short TTS audio clip, then
transcribes it back to text — a complete round-trip without external audio files.
Each transcript is compared with the text that was spoken, so a wrong or empty
transcription fails the run.
"""

import asyncio
import io
import re
import sys
import wave
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import AuthenticationError, PaymentRequiredError, VeniceError
from venice_ai.types.api import ASRModelPricing

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import tts_default_format, tts_formats, tts_spec  # noqa: E402

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/audio/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Minimum share of the spoken words a transcript must contain.
MIN_OVERLAP = 0.8

#: Paths written by this run, listed at the end on success.
WRITTEN: list[Path] = []


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_overlap(expected: str, actual: str) -> float:
    """Fraction of the expected words that appear in the transcript."""
    want = _words(expected)
    got = set(_words(actual))
    return sum(w in got for w in want) / len(want) if want else 0.0


def check_transcript(expected: str, actual: str, indent: str = "") -> bool:
    overlap = word_overlap(expected, actual)
    if overlap >= MIN_OVERLAP:
        print(f"{indent}📊 Matches the spoken text ({overlap:.0%} of words)")
        return True
    print(f"{indent}❌ Transcript matches only {overlap:.0%} of the spoken text")
    return False


async def asr_models_by_price(client: VeniceClient) -> list[str]:
    """Every ASR model, cheapest first, via repeated ``resolve_asr(prefer="cheapest")``."""
    ordered: list[str] = []
    while True:
        try:
            ordered.append(
                await client.models.resolve_asr(prefer="cheapest", exclude_models=ordered)
            )
        except NoMatchingModelError:
            return ordered


class ClipMaker:
    """Generates the TTS clips the demos transcribe.

    The voice comes from the TTS model's own voice list, and the clip is WAV
    when the model offers it (so its length can be measured exactly).
    """

    def __init__(self, client: VeniceClient, model_id: str, voice: str, fmt: str):
        self.client = client
        self.model_id = model_id
        self.voice = voice
        self.fmt = fmt

    @classmethod
    async def create(cls, client: VeniceClient) -> "ClipMaker":
        model_id = await client.models.resolve_tts(prefer="cheapest")
        catalog = await client.models.list(type="tts")
        spec = tts_spec(next(m.model_spec for m in catalog.data if m.id == model_id), model_id)
        details = (await client.audio.get_voices(model_id=model_id)).data
        english = [v.id for v in details if v.language == "English"]
        voices = english or spec.voices
        if not voices:
            raise LookupError(f"{model_id} lists no voices in the catalog")
        voice = voices[0]
        fmt = "wav" if "wav" in tts_formats(spec) else tts_default_format(spec, model_id)
        return cls(client, model_id, voice, fmt)

    async def make(self, text: str, name: str) -> tuple[Path, float | None]:
        """Speak ``text`` into ``results/<name>``; return the path and length."""
        response = await self.client.audio.create_speech(
            model=self.model_id,
            input=text,
            voice=self.voice,
            response_format=self.fmt,
            speed=1.0,
        )
        path = response.save(RESULTS_DIR / f"{name}.{self.fmt}", overwrite=True)
        WRITTEN.append(path)
        seconds = None
        if self.fmt == "wav":
            with wave.open(io.BytesIO(response.content)) as w:
                seconds = w.getnframes() / w.getframerate()
        return path, seconds


async def basic_transcription(client: VeniceClient, clips: ClipMaker, asr_model: str) -> bool:
    """Transcribe an audio file from a file path and print the text."""
    print("🎤 Basic Speech-to-Text Transcription")
    print("-" * 40)

    sample_text = "Hello! This is a speech-to-text demonstration using the Venice AI SDK."
    print(f"📝 Original text: {sample_text}")
    try:
        audio_path, _ = await clips.make(sample_text, "stt_demo")
        print(f"💾 Saved TTS audio: {audio_path}")
        print("\n📝 Transcribing audio...")
        result = await client.audio.transcribe(file=str(audio_path), model=asr_model)
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    print(f"✅ Transcription: {result.text}")
    return check_transcript(sample_text, result.text)


async def transcription_with_timestamps(client: VeniceClient, clips: ClipMaker) -> bool:
    """Use timestamps=True to get word-level timing data.

    The catalog does not say which ASR models return word timings, so this
    asks each model in turn, cheapest first, and uses the first one that does.
    """
    print("\n⏱️ Transcription with Word-Level Timestamps")
    print("-" * 40)

    sample_text = "Word-level timestamps let you know exactly when each word was spoken."
    try:
        audio_path, seconds = await clips.make(sample_text, "stt_timestamps_demo")
        print(f"💾 Saved TTS audio: {audio_path}" + (f" ({seconds:.2f}s)" if seconds else ""))
        candidates = await asr_models_by_price(client)
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    for asr_model in candidates:
        try:
            result = await client.audio.transcribe(
                file=str(audio_path), model=asr_model, timestamps=True
            )
        except (AuthenticationError, PaymentRequiredError) as e:
            print(f"❌ {type(e).__name__}: {e}")
            return False
        except VeniceError as e:
            print(f"   ⚠️ {asr_model}: {e}")
            continue
        if not result.words:
            print(f"   ℹ️ {asr_model}: no word timings returned")
            continue

        print(f"🎤 Word timings from: {asr_model}")
        print(f"✅ Full text: {result.text}")
        print(f"\n⏱️ Word-level timestamps ({len(result.words)} words):")
        for word_info in result.words:
            start = f"{word_info.start:.2f}s" if word_info.start is not None else "N/A"
            end = f"{word_info.end:.2f}s" if word_info.end is not None else "N/A"
            print(f"   📌 {word_info.word:<20} {start} → {end}")

        starts = [w.start for w in result.words if w.start is not None]
        ends = [w.end for w in result.words if w.end is not None]
        limit = (seconds or result.duration or float("inf")) + 0.1
        if not starts or not ends or starts != sorted(starts) or max(ends) > limit:
            print("❌ Timings are missing, out of order, or run past the end of the clip")
            return False
        print(f"📐 Timings are in order and end within the clip ({max(ends):.2f}s)")
        return check_transcript(sample_text, result.text)

    print(f"❌ None of the {len(candidates)} ASR models returned word timings")
    return False


async def different_input_methods(client: VeniceClient, clips: ClipMaker, asr_model: str) -> bool:
    """Show file path (str), bytes, and BinaryIO (open file) inputs."""
    print("\n📂 Different Input Methods")
    print("-" * 40)

    sample_text = "Testing different input methods for the transcription API."
    ok = True
    try:
        audio_path, _ = await clips.make(sample_text, "stt_input_methods")
        print(f"🔊 Generated test audio: {audio_path}")

        print("\n1️⃣ Input method: file path (str)")
        result = await client.audio.transcribe(file=str(audio_path), model=asr_model)
        print(f"   ✅ Result: {result.text}")
        ok = check_transcript(sample_text, result.text, "   ") and ok

        print("\n2️⃣ Input method: bytes")
        result = await client.audio.transcribe(file=audio_path.read_bytes(), model=asr_model)
        print(f"   ✅ Result: {result.text}")
        ok = check_transcript(sample_text, result.text, "   ") and ok

        print("\n3️⃣ Input method: BinaryIO (open file)")
        with open(audio_path, "rb") as f:
            result = await client.audio.transcribe(file=f, model=asr_model)
        print(f"   ✅ Result: {result.text}")
        ok = check_transcript(sample_text, result.text, "   ") and ok
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False

    if ok:
        print("\n📊 All three input methods transcribed the clip correctly")
    return ok


async def model_discovery(client: VeniceClient) -> bool:
    """List the ASR models in the catalog and show which one resolve_asr() picks."""
    print("\n🔍 ASR Model Discovery")
    print("-" * 40)

    try:
        catalog = await client.models.list(type="asr")
        default = await client.models.resolve_asr()
        selected = await client.models.resolve_asr(prefer="cheapest")
    except (VeniceError, ValueError) as e:
        print(f"❌ ASR model discovery failed: {e}")
        return False

    print(f"📊 {len(catalog.data)} ASR models in the catalog:")
    for model in catalog.data:
        spec = model.model_spec
        pricing = spec.pricing
        price = pricing.per_audio_second.usd if isinstance(pricing, ASRModelPricing) else None
        price_text = f"${price:.6f}/audio second" if price is not None else "price not listed"
        marker = "👉" if model.id == selected else "  "
        print(f"   {marker} {model.id:<32} {spec.name} ({price_text})")
    print(f"\n✅ resolve_asr() selected: {default}")
    print(f"✅ resolve_asr(prefer='cheapest') selected: {selected} (marked 👉)")
    return True


async def response_format_options(client: VeniceClient, clips: ClipMaker, asr_model: str) -> bool:
    """Compare the JSON and plain-text response formats, plus a language hint."""
    print("\n📋 Response Format Options")
    print("-" * 40)

    sample_text = "Exploring different response formats for the transcription endpoint."
    ok = True
    try:
        audio_path, _ = await clips.make(sample_text, "stt_format_demo")
        print(f"🔊 Generated test audio: {audio_path}")

        # "json" (the default) returns a structured response object.
        print("\n1️⃣ response_format='json' (default) → AudioTranscriptionResponse")
        result = await client.audio.transcribe(
            file=str(audio_path), model=asr_model, response_format="json"
        )
        print(f"   ✅ type: {type(result).__name__}")
        print(f"   ✅ text: {result.text}")
        if result.duration is not None:
            print(f"   📏 duration: {result.duration:.2f}s")
        ok = check_transcript(sample_text, result.text, "   ") and ok

        # "text" returns the transcript as a plain string. Some models answer
        # it with an empty body, so try the models in turn, cheapest first,
        # and use the first one that returns text.
        print("\n2️⃣ response_format='text' → str")
        candidates = await asr_models_by_price(client)
        text = ""
        for model_id in candidates:
            try:
                text = await client.audio.transcribe(
                    file=str(audio_path), model=model_id, response_format="text"
                )
            except (AuthenticationError, PaymentRequiredError) as e:
                print(f"   ❌ {type(e).__name__}: {e}")
                return False
            except VeniceError as e:
                print(f"   ⚠️ {model_id}: {e}")
                continue
            if not isinstance(text, str):
                print(f"   ❌ {model_id}: expected a plain string, got {type(text).__name__}")
                ok = False
                break
            if text.strip():
                print(f"   🎤 model: {model_id}")
                print(f"   ✅ type: {type(text).__name__}")
                print(f"   ✅ text: {text.strip()}")
                ok = check_transcript(sample_text, text, "   ") and ok
                break
            print(f"   ℹ️ {model_id}: empty plain-text body")
        else:
            print("   ❌ No ASR model returned a plain-text transcript")
            ok = False

        print("\n3️⃣ With language hint (en):")
        result = await client.audio.transcribe(file=str(audio_path), model=asr_model, language="en")
        print(f"   ✅ text: {result.text}")
        ok = check_transcript(sample_text, result.text, "   ") and ok
    except VeniceError as e:
        print(f"❌ Error: {e}")
        return False
    return ok


async def main() -> int:
    """Run all speech-to-text examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise.
    """
    print("🚀 Venice AI Speech-to-Text Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        clips = await ClipMaker.create(client)
        asr_model = await client.models.resolve_asr(prefer="cheapest")
        print(
            f"🔊 TTS model for the test clips: {clips.model_id} (voice {clips.voice!r}, {clips.fmt})"
        )
        print(f"🎤 ASR model: {asr_model}\n")

        results: list[tuple[str, bool]] = [
            ("basic_transcription", await basic_transcription(client, clips, asr_model)),
            ("transcription_with_timestamps", await transcription_with_timestamps(client, clips)),
            ("different_input_methods", await different_input_methods(client, clips, asr_model)),
            ("model_discovery", await model_discovery(client)),
            ("response_format_options", await response_format_options(client, clips, asr_model)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Speech-to-text examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Basic audio-to-text transcription")
    print("   - Word-level timestamps with timing data")
    print("   - Multiple input methods (file path, bytes, BinaryIO)")
    print("   - ASR model discovery and selection")
    print("   - JSON vs plain-text response formats, and language hints")
    print("   - Round-trip TTS → STT demo (self-contained)")

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
            "Check that your API key is valid and you have audio transcription access.",
            file=sys.stderr,
        )
        sys.exit(1)
