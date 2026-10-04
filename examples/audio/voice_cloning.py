#!/usr/bin/env python3
"""
Venice AI SDK - Voice Cloning
=============================

Clone a voice from a short audio sample, then synthesize new speech in that
voice. Two steps:

1. ``client.audio.create_voice(file=...)`` uploads the sample to
   ``POST /v1/audio/voices`` and returns a :class:`ClonedVoice` whose ``id`` is
   a ``vv_<id>`` handle, paired with the model on ``.model``.
2. ``client.audio.create_speech(input=..., model=voice.model, voice=voice.id)``
   generates speech using that handle.

**Pair the handle with the same model it was created for** — that's why we read
``voice.model`` back rather than resolving a TTS model independently for the
synthesis call. The output format also comes from that model's catalog entry
(``default_format``): not every TTS model can produce MP3.

**Treat the handle as a secret.** It encodes a link to the uploaded voice
sample, so this example never prints it.

The sample at ``examples/voice-to-clone.wav`` is used as input. A clean 5–10s
speech recording works best. Output is written to ``examples/results/``.

Handles expire after the model's retention window (``voice_cloning.retention_days``
in its catalog entry).
"""

import asyncio
import re
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import PermissionDeniedError, VeniceError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import tts_default_format, tts_formats, tts_spec  # noqa: E402

# Resolve sample + results dir relative to this file's location.
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "voice-to-clone.wav"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LINE = "Hello! This sentence was generated in a cloned voice using the Venice AI SDK."

#: Exit status for a run that skipped because a prerequisite is missing.
EXIT_SKIPPED = 77

#: Minimum share of the line's words the transcript of the result must contain.
MIN_OVERLAP = 0.8


def word_overlap(expected: str, actual: str) -> float:
    want = re.findall(r"\w+", expected.lower())
    got = set(re.findall(r"\w+", actual.lower()))
    return sum(w in got for w in want) / len(want) if want else 0.0


async def clone_and_speak() -> bool | None:
    """Clone the sample voice, then synthesize a new line in it.

    Returns ``True`` on success, ``False`` if the result fails its check, and
    ``None`` when voice cloning is not enabled for the account (HTTP 403).
    """
    print("🎙️  Voice cloning")
    print("-" * 40)

    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"Voice sample not found at {SAMPLE_PATH}. Drop a short WAV recording there."
        )

    async with VeniceClient() as client:
        # Step 1 — clone. Omitting model lets the API pick its default and
        # report it back on voice.model.
        print(f"   Uploading sample: {SAMPLE_PATH.name}")
        try:
            voice = await client.audio.create_voice(file=SAMPLE_PATH)
        except PermissionDeniedError as e:
            # Voice cloning is a gated capability. A 403 is an entitlement
            # condition, not a code error; any other error still propagates.
            print("   ⏭️  Voice cloning is not enabled on this account.")
            print(f"      ({e})")
            print("      Contact support@venice.ai to request access.")
            return None
        print(f"   ✅ Cloned handle created ({voice.id[:3]}… — kept private)")
        print(f"      paired model: {voice.model}")

        # Read the paired model's catalog entry for its output format and the
        # cloning limits it advertises.
        catalog = await client.models.list(type="tts")
        spec = tts_spec(
            next(m.model_spec for m in catalog.data if m.id == voice.model), voice.model
        )
        fmt = tts_default_format(spec, voice.model)
        print(f"      output format: {fmt} (supported: {', '.join(tts_formats(spec) or [fmt])})")
        cloning = spec.voice_cloning
        if cloning is not None:
            print(
                f"      sample of at least {cloning.min_sample_seconds}s, "
                f"handle kept {cloning.retention_days} days"
            )

        # Step 2 — synthesize using the handle + its paired model.
        response = await client.audio.create_speech(
            input=LINE,
            model=voice.model,  # MUST match the model the handle was made for
            voice=voice.id,  # the vv_<id> handle as a plain string
            response_format=fmt,
        )
        saved = response.save(RESULTS_DIR / f"cloned_voice.{fmt}", overwrite=True)
        print(f"   🔊 Saved synthesized speech → {saved} ({len(response.content)} bytes)")

        # Check the result says the line (whether it *sounds* like the sample
        # is for your ears to judge).
        asr_model = await client.models.resolve_asr(prefer="cheapest")
        transcript = await client.audio.transcribe(file=saved, model=asr_model, language="en")
        overlap = word_overlap(LINE, transcript.text)
        print(f"   🔁 Transcribed back ({asr_model}): {transcript.text}")
        print(f"   📊 Matches the requested line: {overlap:.0%}")
        if overlap < MIN_OVERLAP:
            print("   ❌ The synthesized audio does not say the requested line")
            return False
        return True


async def main() -> int:
    """Run the voice-cloning example.

    Returns ``0`` on success, ``77`` when cloning is not enabled for the
    account, and ``1`` on any failure. The success banner prints only after a
    real clone.
    """
    print("🚀 Venice AI Voice Cloning")
    print("=" * 50)

    try:
        ok = await clone_and_speak()
    except VeniceError as e:
        print(f"\n❌ API error: {e}", file=sys.stderr)
        ok = False

    if ok is None:
        print("\nSKIPPED: voice cloning is not enabled on this account (HTTP 403).")
        return EXIT_SKIPPED
    if not ok:
        print("\n⚠️ Voice cloning failed.")
        return 1

    print("\n✨ Voice cloning example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.audio.create_voice(file=...) → ClonedVoice (vv_<id>)")
    print("   - Pairing voice.id with voice.model on create_speech")
    print("   - Taking the output format from the paired model's catalog entry")
    print("   - Keeping the handle out of logs")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
