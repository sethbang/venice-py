#!/usr/bin/env python3
"""
Venice AI SDK - Voice Changer
=============================

Convert an existing recording into a different voice. Unlike text-to-speech,
the *source* is audio: the model re-voices what was already said and preserves
the original timing, so the source length is what you are billed for.

The flow is the async job family, same shape as music and video::

    quote   → estimate the price for a source of N seconds
    run     → queue the conversion, returning a VoiceChangerJob
    wait    → poll until the converted audio is ready
    download→ write the bytes out

Two things differ from the other job families and are worth knowing before you
reach for muscle memory:

* ``quote(duration_seconds=...)`` takes a **bare number of seconds** — ``60``,
  not the ``"60s"`` form the video endpoints accept.
* ``retrieve`` has two outcomes, not three: still processing, or the converted
  audio itself. There is no FAILED status to branch on — a failed conversion
  raises instead, and the error carries ``credits_refunded``.

Voice-changer models are not their own model type. They report
``type="music"`` with ``voice_changer=true``, so resolve them with
``models.resolve_voice_changer()`` — ``resolve_music()`` would happily hand
back a music *generator*, which these endpoints reject.

**This capability is not enabled on every account.** When no voice-changer
model is in the catalog, this example reports that and exits cleanly rather
than failing.
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "voice-to-clone.wav"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Length to price in the quote step, in whole seconds.
QUOTE_SECONDS = 30


async def change_voice() -> bool:
    """Quote, then convert the sample recording into another voice.

    Returns ``True`` on success, and also on a clean skip when the account has
    no voice-changer model — that is an entitlement condition, not a failure.
    """
    print("🎚️  Voice changer")
    print("-" * 40)

    async with VeniceClient() as client:
        # Never hardcode a model id. This filters type="music" by the
        # voice_changer capability flag.
        try:
            model = await client.models.resolve_voice_changer()
        except ValueError as e:
            print("   ⏭️  No voice-changer model is available on this account.")
            print(f"      ({e})")
            print("      Contact support@venice.ai to request access.")
            return True

        print(f"   Model: {model}")

        # Price it first — a bare count of seconds, not "30s".
        quote = await client.voice_changer.quote(model=model, duration_seconds=QUOTE_SECONDS)
        print(f"   Quote for {QUOTE_SECONDS}s: ${quote.quote:.4f} USD")
        print("   (an estimate — you are billed on the length Venice measures)")

        if not SAMPLE_PATH.exists():
            raise FileNotFoundError(
                f"Source recording not found at {SAMPLE_PATH}. Drop a short WAV/MP3 there."
            )

        print(f"\n   Converting: {SAMPLE_PATH.name}")
        # The context manager releases the provider-held media on exit.
        async with await client.voice_changer.run(model=model, file=SAMPLE_PATH) as job:
            print(f"   Queued: {job.queue_id}")
            print(f"   Billed length: {job.duration_seconds:.0f}s")

            def show(status) -> None:
                print(f"   ⏳ {status.progress_percent:.0f}% …")

            status = await job.wait(poll_interval=3.0, on_progress=show)
            out_path = await job.download(RESULTS_DIR / "voice_changed.mp3", status)

        print(f"\n   🔊 Saved converted audio → {out_path}")
    return True


async def main() -> int:
    """Run the voice-changer example.

    Returns ``0`` only if the demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Voice Changer")
    print("=" * 50)

    try:
        ok = await change_voice()
    except (VeniceError, APIError) as e:
        print(f"\n❌ API error: {e}", file=sys.stderr)
        ok = False

    if not ok:
        print("\n⚠️ Voice conversion failed.")
        return 1

    print("\n✨ Voice changer example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.models.resolve_voice_changer() — a capability, not a type")
    print("   - voice_changer.quote(duration_seconds=30) — bare seconds, not '30s'")
    print("   - voice_changer.run(...) → VoiceChangerJob, as an async context manager")
    print("   - job.wait() → the audio itself; there is no FAILED status to check")
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
