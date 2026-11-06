#!/usr/bin/env python3
"""
Venice AI SDK - Voice Cloning
=============================

Clone a voice from a short audio sample, then synthesize new speech in that
voice. Two steps:

1. ``client.audio.create_voice(file=..., model=...)`` uploads the sample to
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

**Only clone a voice you have consent to use.** By default the sample is a neutral
sentence synthesized with Venice TTS when the example runs. To clone your own
voice, or that of someone who agreed to it, set ``VENICE_VOICE_SAMPLE_PATH`` to a
clean recording; 5–10s of speech works best. The containers each model accepts and
its recommended minimum length are in its catalog entry (``voice_cloning``).

Everything free runs first, so a catalog with no cloning model or no ASR model to
check the result skips (exit 77) before any charge. Then each run pays for the
sample's TTS call (unless you supply a recording), the clone, the new speech and
the transcription that checks it. Output is written to ``examples/results/``.

Cost: about $0.054 per run was measured, about 14 times what the catalog's TTS
price predicts for the two clips. The API documents the clone endpoint as billable
at a dynamic price, so the catalog cannot quote it. Which step takes the rest has
not been measured.

Handles expire after the model's retention window (``voice_cloning.retention_days``
in its catalog entry).
"""

import asyncio
import sys
import wave
from pathlib import Path

from _helpers import (
    EXIT_SKIPPED,
    tts_default_format,
    tts_formats,
    tts_spec,
    voice_sample,
    word_overlap,
)

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import ApiKeyIpRestrictedError, PermissionDeniedError, VeniceError
from venice_ai.types.api import TtsModelSpec

# Resolve results dir relative to this file's location.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LINE = "Hello! This sentence was generated in a cloned voice using the Venice AI SDK."

#: Minimum share of the line's words the transcript of the result must contain.
MIN_OVERLAP = 0.8


async def _cheapest_cloning_model(
    client: VeniceClient, specs: dict[str, TtsModelSpec]
) -> str | None:
    """The cheapest TTS model that can clone, or ``None`` if none is listed."""
    if not any(s.voice_cloning is not None for s in specs.values()):
        return None
    tried: list[str] = []
    while True:
        try:
            model_id = await client.models.resolve_tts(prefer="cheapest", exclude_models=tried)
        except NoMatchingModelError:
            return None
        spec = specs.get(model_id)
        if spec is not None and spec.voice_cloning is not None:
            return model_id
        tried.append(model_id)


async def clone_and_speak() -> bool | str:
    """Clone a sample voice, then synthesize a new line in it.

    Returns ``True`` on success, ``False`` if the result fails its check, and
    the reason when skipped: no cloning or ASR model in the catalog, no TTS model
    to make the sample with, or voice cloning not enabled for the account
    (HTTP 403).
    """
    print("🎙️  Voice cloning")
    print("-" * 40)

    async with VeniceClient() as client:
        # Free lookups first, so a run that cannot finish pays for nothing.
        catalog = await client.models.list(type="tts")
        specs = {m.id: tts_spec(m.model_spec, m.id) for m in catalog.data}
        clone_model = await _cheapest_cloning_model(client, specs)
        if clone_model is None:
            return "no TTS model in the catalog offers voice cloning to this key"
        limits = specs[clone_model].voice_cloning
        try:
            asr_model = await client.models.resolve_asr(prefer="cheapest")
        except NoMatchingModelError as e:
            return f"no ASR model to check the result with ({e})"

        # The sample has to suit the model it is uploaded for.
        try:
            sample = await voice_sample(
                client,
                RESULTS_DIR,
                accepted_formats=limits.accepted_formats if limits else None,
                min_seconds=(limits.min_sample_seconds or 0.0) if limits else 0.0,
            )
        except NoMatchingModelError as e:
            return f"no TTS model can synthesize the sample ({e})"

        # Step 1 — clone. model= names the model the handle is for; leaving it out
        # lets the API pick one, and either way voice.model reports it.
        print(f"   Uploading sample for {clone_model}: {sample.name}")
        try:
            voice = await client.audio.create_voice(file=sample, model=clone_model)
        except ApiKeyIpRestrictedError as e:
            print(f"   ❌ This API key does not allow this source IP: {e}")
            return False
        except PermissionDeniedError as e:
            # A 403 can mean the capability is not enabled for this account. Any
            # other error still propagates.
            print("   ⏭️  Voice cloning is not enabled on this account.")
            print(f"      ({e})")
            print("      Contact support@venice.ai to request access.")
            return "voice cloning is not enabled on this account (HTTP 403)"
        print(f"   ✅ Cloned handle created ({voice.id[:3]}… — kept private)")
        print(f"      paired model: {voice.model}")

        # Read the paired model's catalog entry for its output format and the
        # cloning limits it advertises.
        spec = specs.get(voice.model)
        if spec is None:
            raise LookupError(f"{voice.model} is not in the TTS catalog")
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

    Returns ``0`` on success, ``77`` when a catalog model it needs is missing or
    cloning is not enabled for the account, and ``1`` on any failure. The success
    banner prints only after a real clone.
    """
    print("🚀 Venice AI Voice Cloning")
    print("=" * 50)

    try:
        ok = await clone_and_speak()
    except VeniceError as e:
        print(f"\n❌ API error: {e}", file=sys.stderr)
        ok = False
    except (OSError, LookupError, ValueError, wave.Error, EOFError) as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        ok = False

    if isinstance(ok, str):
        print(f"\nSKIPPED: {ok}.")
        return EXIT_SKIPPED
    if not ok:
        print("\n⚠️ Voice cloning failed.")
        return 1

    print("\n✨ Voice cloning example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.audio.create_voice(file=..., model=...) → ClonedVoice (vv_<id>)")
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
