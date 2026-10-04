#!/usr/bin/env python3
"""
Venice AI SDK - Voice Options Demonstration
===========================================

This example demonstrates the various voice options available in the Venice AI SDK.
Learn how to use different voices, discover available voices, and customize speech characteristics.

Voices are taken from the model's catalog entry, never from a fixed list.
``client.audio.get_voices()`` adds language, accent and gender metadata for
voice IDs that follow the ``<region><gender>_<name>`` convention, and this
example picks a TTS model whose voices carry that metadata. Every voice is
given text in its own language: a Japanese voice reading English text does
not produce accented English, it produces garbled speech.
"""

import asyncio
import re
import sys
from collections import defaultdict
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.audio import VoiceDetail

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import tts_default_format, tts_spec  # noqa: E402

# Resolve results dir relative to this file's location.
# All example scripts live one level below examples/ (e.g., examples/audio/).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Sample text in each voice language. Voices in other languages are skipped.
NATIVE_TEXT: dict[str, str] = {
    "English": "Hello! I'm demonstrating the different voices available in Venice AI.",
    "Spanish": "¡Hola! Estoy mostrando las diferentes voces disponibles en Venice AI.",
    "French": "Bonjour ! Je présente les différentes voix disponibles dans Venice AI.",
    "Italian": "Ciao! Sto mostrando le diverse voci disponibili in Venice AI.",
    "Portuguese": "Olá! Estou demonstrando as diferentes vozes disponíveis no Venice AI.",
    "German": "Hallo! Ich stelle die verschiedenen Stimmen von Venice AI vor.",
    "Japanese": "こんにちは。ベニスAIで使えるさまざまな声をご紹介します。",
    "Mandarin Chinese": "你好！我正在为你展示这里提供的各种声音。",
    "Hindi": "नमस्ते! यह Venice AI की अलग-अलग आवाज़ों का प्रदर्शन है।",
}

#: Languages whose clips are transcribed back and compared with the text.
#: The others are generated but not checked automatically: listen to them.
CHECKED_LANGUAGES: dict[str, str] = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Portuguese": "pt",
    "German": "de",
}

#: Minimum share of the spoken words a transcript must contain.
MIN_OVERLAP = 0.6

GENDER_ICONS = {"female": "♀️", "male": "♂️"}

#: Paths written by this run, listed at the end on success.
WRITTEN: list[Path] = []

#: Clips in languages that are not transcribed back, listed at the end.
UNCHECKED: list[Path] = []

#: Languages whose clips were transcribed back during this run.
TRANSCRIBED: set[str] = set()


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def word_overlap(expected: str, actual: str) -> float:
    want = _words(expected)
    got = set(_words(actual))
    return sum(w in got for w in want) / len(want) if want else 0.0


class VoiceDemo:
    """Shared state: the client, the chosen TTS model and its voices."""

    def __init__(
        self,
        client: VeniceClient,
        model_id: str,
        fmt: str,
        voices: list[VoiceDetail],
        asr_model: str,
    ):
        self.client = client
        self.model_id = model_id
        self.fmt = fmt
        self.voices = voices
        self.asr_model = asr_model

    def by_language(self) -> dict[str, list[VoiceDetail]]:
        groups: dict[str, list[VoiceDetail]] = defaultdict(list)
        for voice in self.voices:
            if voice.language:
                groups[voice.language].append(voice)
        return groups

    async def speak(self, voice: VoiceDetail, text: str, stem: str) -> bool:
        """Generate one clip and, for checked languages, transcribe it back.

        A clip in an unchecked language is not counted as verified: it fails
        only if it is empty, and is listed at the end for you to listen to.
        """
        try:
            response = await self.client.audio.create_speech(
                model=self.model_id,
                input=text,
                voice=voice.id,
                response_format=self.fmt,
                speed=1.0,
            )
        except VeniceError as e:
            print(f"   ❌ {voice.id}: {e}")
            return False
        path = response.save(RESULTS_DIR / f"{stem}.{self.fmt}", overwrite=True)
        WRITTEN.append(path)
        line = f"   {GENDER_ICONS.get(voice.gender or '', '❓')} {voice.id:<14} → {path.name} ({len(response.content)} bytes)"

        code = CHECKED_LANGUAGES.get(voice.language or "")
        if code is None:
            print(f"{line}  [not transcribed: listen to check]")
            UNCHECKED.append(path)
            return bool(response.content)
        try:
            transcript = await self.client.audio.transcribe(
                file=path, model=self.asr_model, language=code
            )
        except VeniceError as e:
            print(f"{line}\n      ❌ transcription failed: {e}")
            return False
        TRANSCRIBED.add(voice.language or "")
        overlap = word_overlap(text, transcript.text)
        mark = "✅" if overlap >= MIN_OVERLAP else "❌"
        print(f"{line}  {mark} transcript matches {overlap:.0%}")
        if overlap < MIN_OVERLAP:
            print(f"      heard: {transcript.text}")
        return overlap >= MIN_OVERLAP


async def choose_model(client: VeniceClient) -> tuple[str, str, list[VoiceDetail]]:
    """Pick a TTS model whose voices carry language and gender metadata.

    ``resolve_tts(prefer="cheapest")`` is used when its model qualifies;
    otherwise the catalog model with the most tagged voices.
    """
    catalog = await client.models.list(type="tts")
    all_voices = (await client.audio.get_voices()).data
    tagged: dict[str, list[VoiceDetail]] = defaultdict(list)
    for voice in all_voices:
        if voice.language and voice.gender in GENDER_ICONS:
            tagged[voice.model_id].append(voice)
    if not tagged:
        raise RuntimeError("No TTS model in the catalog lists voices with language metadata")

    resolved = await client.models.resolve_tts(prefer="cheapest")
    model_id = resolved if resolved in tagged else max(tagged, key=lambda m: len(tagged[m]))
    spec = tts_spec(next(m.model_spec for m in catalog.data if m.id == model_id), model_id)
    return model_id, tts_default_format(spec, model_id), tagged[model_id]


async def list_available_voices(demo: VoiceDemo) -> bool:
    """List voices per model, then the chosen model's voices by language."""
    print("📋 Available Voice Discovery")
    print("-" * 40)

    try:
        everything = (await demo.client.audio.get_voices()).data
        female = (await demo.client.audio.get_voices(model_id=demo.model_id, gender="female")).data
        male = (await demo.client.audio.get_voices(model_id=demo.model_id, gender="male")).data
    except VeniceError as e:
        print(f"❌ Error listing voices: {e}")
        return False

    per_model: dict[str, int] = defaultdict(int)
    for voice in everything:
        per_model[voice.model_id] += 1
    print(f"📊 {len(everything)} voices across {len(per_model)} TTS models:")
    for model_id, count in sorted(per_model.items(), key=lambda kv: -kv[1]):
        print(f"   🎤 {model_id:<28} {count} voices")

    print(f"\n🌍 {demo.model_id} voices by language:")
    for language, voices in sorted(demo.by_language().items()):
        accents = sorted({v.accent for v in voices if v.accent})
        print(f"   {language} ({', '.join(accents)}): {len(voices)}")
        for voice in voices[:4]:
            print(f"      {GENDER_ICONS.get(voice.gender or '', '❓')} {voice.id}")
        if len(voices) > 4:
            print(f"      ... and {len(voices) - 4} more")

    print("\n🔍 Filter example: get_voices(model_id=..., gender=...)")
    print(f"   Female voices: {len(female)}")
    print(f"   Male voices: {len(male)}")
    return bool(demo.voices)


async def voice_variety_by_language(demo: VoiceDemo) -> bool:
    """One female and one male voice per language, each speaking its own language."""
    print("\n🎭 Voice Variety by Language (female and male)")
    print("-" * 40)

    ok = True
    for language, voices in sorted(demo.by_language().items()):
        text = NATIVE_TEXT.get(language)
        if text is None:
            print(f"\n⏭️ {language}: no sample text in this example, skipped")
            continue
        print(f"\n🌍 {language}: {text}")
        for gender in ("female", "male"):
            voice = next((v for v in voices if v.gender == gender), None)
            if voice is None:
                print(f"   ℹ️ no {gender} {language} voice")
                continue
            ok = await demo.speak(voice, text, f"voice_{voice.id}") and ok
    return ok


async def english_accent_showcase(demo: VoiceDemo) -> bool:
    """The same English sentence in each English accent the model offers."""
    print("\n🌏 English Accent Showcase")
    print("-" * 40)

    text = "This sentence demonstrates regional accent variations in English."
    english = demo.by_language().get("English", [])
    accents: dict[str, VoiceDetail] = {}
    for voice in english:
        accents.setdefault(voice.accent or "Unknown", voice)
    if len(accents) < 2:
        print(
            f"Section skipped: accent showcase: {demo.model_id} offers fewer than two English accents"
        )
        return True

    print(f"📝 Text: {text}")
    ok = True
    for accent, voice in accents.items():
        print(f"\n🗣️ {accent} English")
        ok = await demo.speak(voice, text, f"accent_{accent.lower().replace(' ', '_')}") and ok
    return ok


async def voice_personality_showcase(demo: VoiceDemo) -> bool:
    """Different English voices reading lines written for different tones."""
    print("\n✨ Voice Personality Showcase")
    print("-" * 40)

    personality_tests = [
        (
            "Warm & Friendly",
            "Hello! I'm so excited to help you today. How can I make your day better?",
        ),
        (
            "Professional",
            "Good morning. I'll be assisting you with your inquiries in a professional manner.",
        ),
        ("Energetic", "Hey there! Ready for an amazing adventure? Let's dive right in!"),
        ("Calm & Soothing", "Take a deep breath and relax. Everything will be just fine."),
        ("Authoritative", "Please follow these instructions carefully and precisely."),
    ]
    english = demo.by_language().get("English", [])
    if not english:
        print(f"Section skipped: personality showcase: {demo.model_id} has no English voices")
        return True

    ok = True
    # Spread the picks across the list so each line gets a different voice.
    stride = max(1, len(english) // len(personality_tests))
    picks = [english[(i * stride) % len(english)] for i in range(len(personality_tests))]
    for (personality, text), voice in zip(personality_tests, picks):
        print(f"\n🎭 {personality}: {text}")
        stem = "personality_" + personality.lower().replace(" & ", "_").replace(" ", "_")
        ok = await demo.speak(voice, text, stem) and ok
    return ok


async def main() -> int:
    """Run all voice option examples. Returns ``0`` only if every demo succeeded."""
    print("🚀 Venice AI Voice Options Examples")
    print("=" * 60)

    async with VeniceClient() as client:
        model_id, fmt, voices = await choose_model(client)
        asr_model = await client.models.resolve_asr(prefer="cheapest")
        print(f"📍 TTS model: {model_id} ({len(voices)} voices with language metadata, {fmt})")
        print(f"🎤 ASR model for checking clips: {asr_model}\n")
        demo = VoiceDemo(client, model_id, fmt, voices, asr_model)

        results: list[tuple[str, bool]] = [
            ("list_available_voices", await list_available_voices(demo)),
            ("voice_variety_by_language", await voice_variety_by_language(demo)),
            ("english_accent_showcase", await english_accent_showcase(demo)),
            ("voice_personality_showcase", await voice_personality_showcase(demo)),
        ]

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1

    print("\n✨ Voice options examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - Voice discovery with get_voices() and its model/gender filters")
    print("   - Female and male voices per language, each given native text")
    print("   - English accent variations")
    print("   - Voice personality and tone")
    checked = ", ".join(sorted(lang for lang in TRANSCRIBED if lang))
    if checked:
        print(f"   - Transcription round-trip for {checked} clips")
    if UNCHECKED:
        print(f"\n🎧 {len(UNCHECKED)} clips were not transcribed; listen to check them:")
        for path in UNCHECKED:
            print(f"   - {path.relative_to(RESULTS_DIR.parent.parent)}")

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
