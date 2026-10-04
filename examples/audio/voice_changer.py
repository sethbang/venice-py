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
  raises instead, and the error carries ``credits_refunded``. The completed
  status reports the audio's media type as ``content_type``, and
  ``audio_format`` turns it into a file extension.

Voice-changer models are not their own model type. They report
``type="music"`` with ``voice_changer=true``, so resolve them with
``models.resolve_voice_changer()`` — ``resolve_music()`` would happily hand
back a music *generator*, which these endpoints reject.

**Voice-changer models do not always appear in the catalog.** When none is
listed, this example prints a ``SKIPPED:`` line and exits 77 without claiming
to have run anything. Any other error, including a failed catalog fetch or a
source recording that is not a PCM WAV file, exits 1.
"""

import asyncio
import math
import sys
import wave
from pathlib import Path

from venice_ai import APITimeoutError, NoMatchingModelError, VeniceClient
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerProcessingStatus,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import detect_audio_format  # noqa: E402

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "voice-to-clone.wav"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


#: Exit status for a run that skipped because a prerequisite is missing.
EXIT_SKIPPED = 77


def sample_seconds(path: Path) -> int:
    """Length of a PCM WAV recording, rounded up to the whole seconds ``quote`` takes.

    Raises:
        wave.Error: If the file is not a PCM WAV file.
        EOFError: If the file is truncated.
    """
    with wave.open(str(path)) as w:
        return max(1, math.ceil(w.getnframes() / w.getframerate()))


def output_extension(status: VoiceChangerCompletedStatus) -> str | None:
    """File extension for the converted audio.

    ``status.audio_format`` comes from the response's Content-Type. An audio
    type the SDK has no extension for leaves it ``None``; the container is
    then read from the bytes, and the example says so.
    """
    if status.audio_format is not None:
        return status.audio_format
    sniffed = detect_audio_format(status.data or b"")
    if sniffed is not None:
        print(f"   ℹ️  No extension for {status.content_type}; the bytes are {sniffed}")
    return sniffed


async def change_voice(client: VeniceClient) -> int:
    """Quote, then convert the sample recording into another voice.

    Returns the exit status: ``0`` after a real conversion, ``77`` when the
    catalog lists no voice-changer model, ``1`` on any failure.
    """
    print("🎚️  Voice changer")
    print("-" * 40)

    # Never hardcode a model id. This filters type="music" by the
    # voice_changer capability flag.
    try:
        model = await client.models.resolve_voice_changer()
    except NoMatchingModelError as e:
        print("\nSKIPPED: no voice-changer model in the catalog, so nothing was converted.")
        print(f"   {e}")
        return EXIT_SKIPPED

    print(f"   Model: {model}")

    # Price the sample's own length first, as a bare count of seconds, not "9s".
    # Measuring it needs a PCM WAV file; anything else fails before any paid
    # call.
    try:
        seconds = sample_seconds(SAMPLE_PATH)
    except (OSError, wave.Error, EOFError) as e:
        print(f"   ❌ {SAMPLE_PATH} could not be read as a PCM WAV file: {e}")
        return 1
    quote = await client.voice_changer.quote(model=model, duration_seconds=seconds)
    print(f"   Quote for the {seconds}s sample: ${quote.quote:.4f} USD")
    print("   (an estimate — you are billed on the length Venice measures)")

    print(f"\n   Converting: {SAMPLE_PATH.name}")
    # A clean exit from the block releases the provider-held media.
    async with await client.voice_changer.run(model=model, file=SAMPLE_PATH) as job:
        print(f"   Queued: {job.queue_id}")
        print(f"   Billed length: {job.duration_seconds:.0f}s")

        def show(status: VoiceChangerProcessingStatus) -> None:
            print(f"   ⏳ {status.progress_percent:.0f}% …", end="\r")

        try:
            status = await job.wait(poll_interval=3.0, on_progress=show)
            print()
            extension = output_extension(status)
            if extension is None:
                print(
                    f"   ❌ The converted result ({status.content_type}) is not a known audio format"
                )
                return 1
            out_path = await job.download(RESULTS_DIR / f"voice_changed.{extension}", status)
        except TimeoutError as e:
            # wait() ran out of polls; the conversion is still running.
            print(f"\n   ❌ {e} (queue_id={job.queue_id})")
            return 1
        except APITimeoutError as e:
            # One status or download request timed out. The job keeps its
            # media when this block raises, so it can be retrieved again.
            print(f"\n   ❌ A status or download request timed out (queue_id={job.queue_id}): {e}")
            return 1

    size = out_path.stat().st_size
    print(f"\n   🔊 Saved {size} bytes of {status.content_type} → {out_path}")

    print("\n✨ Voice changer example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.models.resolve_voice_changer() — a capability, not a type")
    print(f"   - voice_changer.quote(duration_seconds={seconds}) — bare seconds, not '{seconds}s'")
    print("   - voice_changer.run(...) → VoiceChangerJob, as an async context manager")
    print("   - job.wait() → the audio itself; there is no FAILED status to check")
    print("   - status.audio_format names the file from the response's content type")
    return 0


async def main() -> int:
    """Run the voice-changer example and return its exit status.

    ``0`` if the conversion succeeded, ``77`` if it was skipped for lack of a
    model, and ``1`` otherwise. The success banner prints only after a real
    conversion.
    """
    print("🚀 Venice AI Voice Changer")
    print("=" * 50)

    try:
        async with VeniceClient() as client:
            status = await change_voice(client)
    except (VeniceError, OSError) as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        status = 1

    if status == 1:
        print("\n⚠️ Voice conversion failed.")
    return status


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
