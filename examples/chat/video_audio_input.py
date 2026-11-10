#!/usr/bin/env python3
"""
Venice AI SDK - Video and Audio Chat Input
==========================================

Some chat models read video and audio, attached to a user message with
``UserMessage.builder().video(...)`` and ``.audio(...)``. This is also how to
analyze a video: the ``/video/transcriptions`` endpoint answers 410.

This example demonstrates:

- Finding a model: ``resolve_chat()`` has no filter for these capabilities, so
  every catalog entry whose ``supportsVideoInput`` or ``supportsAudioInput``
  flag is off goes into ``exclude_models``, and ``prefer="cheapest"`` ranks the
  rest
- Video by public URL. The API also takes a ``data:video/...`` URL; formats are
  mp4, mpeg, mov and webm. A remote URL must be publicly reachable, must not
  redirect, and must return a video content type
- Audio as base64 bytes plus a format, the only form the API takes: it does not
  accept audio URLs

How the checks work:

- Video: the model has to say what a short CC0 clip of a flower shows. The file
  name also says "flower", so this shows an answer that agrees with the clip,
  not proof that frames were decoded. The prompt-token count is printed too.
- Audio: Venice TTS speaks a phrase generated for this example, and the model
  has to write down its unusual word. Nothing else in the request contains it.

Cost: under $0.005 per run. The video and audio sections are each one chat
request, and the audio section adds one short TTS call. All of these are billed.
Every model in the catalog with these inputs may reason, so the token cap leaves
room for that, and reasoning is switched off where the model allows it. How
Venice counts video and audio as input tokens is model-specific, so the figure
is an estimate; the usage block printed by each section gives the real count.

A section with no suitable model prints a ``Section skipped:`` line. The script
exits 1 if any section failed, 77 (with a ``SKIPPED:`` line) if neither section
could run, and 0 otherwise.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"
"""

import asyncio
import base64
import re
import sys
from collections.abc import Callable
from typing import Literal

from venice_ai import NoMatchingModelError, UserMessage, VeniceClient, VeniceError
from venice_ai.types.api import ChatCompletionResponse, TextModelSpec, TtsModelSpec
from venice_ai.types.api.requests import VeniceParameters

# A public CC0 clip: a flower, 960x540 H.264, 5 seconds, 1 MB.
VIDEO_URL = "https://developer.mozilla.org/shared-assets/videos/flower.mp4"
VIDEO_SUBJECT = re.compile(r"\b(flowers?|blossoms?|blooms?|blooming|petals?)\b", re.IGNORECASE)

# What the TTS clip says. Nothing else in the audio request contains the word.
AUDIO_PHRASE = "The harbor password today is marigold."
AUDIO_WORD = "marigold"

# Room for hidden thinking on a model that cannot switch it off.
MAX_COMPLETION_TOKENS = 2048

# The question and the attachment are all the context these requests need, so
# Venice's own system prompt is left out.
PARAMS = VeniceParameters(include_venice_system_prompt=False)

EXIT_SKIPPED = 77

Capable = Callable[[TextModelSpec], bool]


def supports_video(spec: TextModelSpec) -> bool:
    return bool(spec.capabilities and spec.capabilities.supportsVideoInput)


def supports_audio(spec: TextModelSpec) -> bool:
    return bool(spec.capabilities and spec.capabilities.supportsAudioInput)


async def cheapest_capable_model(
    client: VeniceClient, capable: Capable
) -> tuple[str, Literal["none"] | None]:
    """Return the cheapest capable chat model, and how to switch off its reasoning.

    The second value is ``"none"`` if the catalog says the model accepts
    ``reasoning_effort="none"``, otherwise ``None``, which leaves the field out.

    :raises NoMatchingModelError: If no chat model has the capability.
    """
    catalog = await client.models.list(type="text")
    specs = {model.id: model.model_spec for model in catalog.data}
    lacking = [
        model_id
        for model_id, spec in specs.items()
        if not (isinstance(spec, TextModelSpec) and capable(spec))
    ]
    model = await client.models.resolve_chat(exclude_models=lacking, prefer="cheapest")
    spec = specs.get(model)
    options = (
        spec.capabilities.reasoningEffortOptions
        if isinstance(spec, TextModelSpec) and spec.capabilities
        else None
    )
    return model, "none" if options and "none" in options else None


async def synthesize_wav(client: VeniceClient, text: str) -> bytes:
    """Speak ``text`` with the cheapest TTS model that outputs WAV. This call is billed.

    :raises NoMatchingModelError: If no TTS model outputs WAV.
    :raises ValueError: If what comes back is not a WAV file.
    """
    catalog = await client.models.list(type="tts")
    specs = {model.id: model.model_spec for model in catalog.data}
    voices = {
        model_id: spec.voices
        for model_id, spec in specs.items()
        if isinstance(spec, TtsModelSpec)
        and "wav" in (spec.supported_formats or [])
        and spec.voices
    }
    model = await client.models.resolve_tts(
        prefer="cheapest", exclude_models=[model_id for model_id in specs if model_id not in voices]
    )
    # The phrase is English, so an English voice speaks it most clearly.
    details = (await client.audio.get_voices(model_id=model)).data
    english = [voice.id for voice in details if voice.language == "English"]
    voice = (english or voices[model])[0]

    print(f"💸 Synthesizing the clip with {model} (voice {voice!r}); this TTS call is billed")
    speech = await client.audio.create_speech(
        model=model, input=text, voice=voice, response_format="wav"
    )
    clip = speech.content
    if clip[:4] != b"RIFF" or clip[8:12] != b"WAVE":
        raise ValueError(f"{model} did not return a WAV file")
    return clip


def answer_of(response: ChatCompletionResponse) -> str | None:
    """Print the answer and its usage; return its text, or ``None`` if unusable.

    Some models report an answer cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against the cap as well.
    """
    if not response.choices:
        print("❌ The response has no choices")
        return None
    answer = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason
    usage = response.usage
    print(f"🗣️ Assistant: {answer}")
    print(f"   ↳ finish_reason: {finish_reason}")
    if usage is None:
        print("❌ The response reports no usage, so a cut-off answer can't be ruled out")
        return None
    print(f"   📊 {usage.prompt_tokens} prompt tokens, {usage.completion_tokens} completion tokens")
    if finish_reason == "length" or usage.completion_tokens >= MAX_COMPLETION_TOKENS:
        print("❌ The answer may be cut off at max_completion_tokens")
        return None
    if not answer:
        print("❌ The model returned no visible text")
        return None
    return answer


async def video_input(client: VeniceClient) -> bool | None:
    """Ask about a video by URL. Returns ``None`` when no model reads video."""
    print("🎬 Video Input")
    print("-" * 40)

    try:
        model, reasoning_effort = await cheapest_capable_model(client, supports_video)
    except NoMatchingModelError as e:
        print(f"Section skipped: no chat model reads video ({e})")
        return None
    print(f"🤖 Using model: {model}")
    print("💸 This section is billed: the clip counts toward the prompt tokens")

    message = (
        UserMessage.builder()
        .text("In one short sentence, what does this video show?")
        .video(VIDEO_URL)
        .build()
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[message],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=reasoning_effort,
        venice_parameters=PARAMS,
    )

    answer = answer_of(response)
    if answer is None:
        return False
    if VIDEO_SUBJECT.search(answer) is None:
        print("❌ Expected the answer to say the clip shows a flower")
        return False
    print("✅ The answer names what the clip shows")
    return True


async def audio_input(client: VeniceClient) -> bool | None:
    """Ask a model to write down a spoken phrase.

    Returns ``None`` when no model reads audio or no TTS model outputs WAV.
    """
    print("\n🎧 Audio Input")
    print("-" * 40)

    try:
        model, reasoning_effort = await cheapest_capable_model(client, supports_audio)
    except NoMatchingModelError as e:
        print(f"Section skipped: no chat model reads audio ({e})")
        return None
    print(f"🤖 Using model: {model}")
    print("💸 This section is billed: the clip counts toward the prompt tokens")

    try:
        clip = await synthesize_wav(client, AUDIO_PHRASE)
    except NoMatchingModelError as e:
        print(f"Section skipped: no TTS model can make the clip ({e})")
        return None
    except ValueError as e:
        print(f"❌ {e}")
        return False
    print(f"🔊 Clip: {len(clip)} bytes of WAV")

    message = (
        UserMessage.builder()
        .text("Write down exactly what is said in this audio clip.")
        .audio(base64.b64encode(clip).decode("ascii"), "wav")
        .build()
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[message],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort=reasoning_effort,
        venice_parameters=PARAMS,
    )

    answer = answer_of(response)
    if answer is None:
        return False
    if AUDIO_WORD not in answer.lower():
        print(f"❌ Expected the answer to contain the spoken word {AUDIO_WORD!r}")
        return False
    print("✅ The answer contains the word spoken in the clip")
    return True


async def main() -> int:
    """Run both sections; return a process exit code."""
    print("🚀 Venice AI Video and Audio Chat Input")
    print("=" * 50)

    results: list[tuple[str, bool]] = []
    skipped: list[str] = []
    async with VeniceClient() as client:
        for name, section in (("video_input", video_input), ("audio_input", audio_input)):
            try:
                outcome = await section(client)
            except VeniceError as e:
                print(f"❌ {name} failed: {type(e).__name__}: {e}")
                results.append((name, False))
                continue
            if outcome is None:
                skipped.append(name)
            else:
                results.append((name, outcome))

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1
    if not results:
        print("\nSKIPPED: the catalog lists no usable model for video or audio input")
        return EXIT_SKIPPED

    print(f"\n✨ Passed: {', '.join(name for name, _ in results)}")
    if skipped:
        print(f"   Skipped: {', '.join(skipped)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
