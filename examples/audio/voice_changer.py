#!/usr/bin/env python3
"""
Venice AI SDK - Voice Changer
=============================

Convert an existing recording into a different voice, chosen from the model's
catalog entry. Unlike text-to-speech, the *source* is audio: the model re-voices
what was already said and preserves the original timing, so you are billed for
the source's length, rounded up to the next whole minute.

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

**Only convert a recording you have consent to use.** By default the source is a
neutral sentence synthesized with Venice TTS when the example runs, and that TTS
call is billed on top of the conversion. To convert your own recording, set
``VENICE_VOICE_SAMPLE_PATH`` to a PCM WAV file; its length is measured to price it,
and it is checked against the model's accepted formats and longest source.

**Quote only by default.** A conversion bills at least one whole minute, $0.35 on
the model listed when this was written, so by default this example prints the
quote for that minute (free) and exits 77 without synthesizing a sample or queueing
anything. To convert, set::

    VENICE_RUN_PAID_VOICE_CHANGER=1

After saving, a paid run retrieves the finished conversion once more with
``delete_media_on_completion=True``, the one-shot form that releases the media as
it returns the audio, so there is no separate release to send. It is safe here
because this endpoint returns the audio inline and the first copy is already on
disk. Do not use it first on a result you still have to fetch from a link.

**Voice-changer models do not always appear in the catalog.** When none is
listed, this example prints a ``SKIPPED:`` line and exits 77 without claiming
to have run anything or paying for a sample. Any other error, including a failed
catalog fetch or a source recording that is not a PCM WAV file, exits 1.

Cost with the opt-in: one minute of conversion plus under a cent for the sample's
TTS call, about $0.35 in all on the model listed when this was written.
"""

import asyncio
import math
import os
import sys
import wave
from pathlib import Path

from _helpers import EXIT_SKIPPED, detect_audio_format, voice_sample, wav_seconds

from venice_ai import APIConnectionError, APITimeoutError, NoMatchingModelError, VeniceClient
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.types.api import MusicModelSpec
from venice_ai.types.api.voice_changer import (
    VoiceChangerCompletedStatus,
    VoiceChangerProcessingStatus,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

#: Opt-in for the paid conversion (quoted either way).
RUN_PAID = os.environ.get("VENICE_RUN_PAID_VOICE_CHANGER") == "1"

#: The shortest length billed: a conversion is rounded up to a whole minute.
BILLED_MINUTE_SECONDS = 60


def sample_seconds(path: Path) -> int:
    """Length of a PCM WAV recording, rounded up to the whole seconds ``quote`` takes.

    Raises:
        wave.Error: If the file is not a PCM WAV file.
        EOFError: If the file is truncated.
    """
    return max(1, math.ceil(wav_seconds(path)))


def choose_target_voice(spec: MusicModelSpec) -> str | None:
    """A target voice from the model's catalog entry, other than its default.

    Picking a listed voice shows the ``voice`` parameter doing something. A model
    that lists none leaves ``voice`` unset, and the server uses its default.
    """
    voices = spec.voices or []
    others = [v for v in voices if v != spec.default_voice]
    candidates = others or voices
    return candidates[0] if candidates else None


def source_problem(path: Path, spec: MusicModelSpec) -> str | None:
    """Why a source recording cannot go to this model, or ``None`` if it can.

    Checked before the paid conversion. The container is read from the file's
    bytes, not its name. A model that declares no list or limit is not checked on
    that point: the server decides.

    Raises:
        OSError: If the file cannot be read.
        wave.Error: If the length of a file named as WAV cannot be measured.
        EOFError: If the file is truncated.
    """
    accepted = {f.lower() for f in spec.accepted_audio_formats or []}
    found = detect_audio_format(path.read_bytes())
    if accepted and found not in accepted:
        return f"{path.name} is {found or 'an unknown format'}; the model takes {sorted(accepted)}"
    limit = spec.max_source_audio_duration_seconds
    seconds = wav_seconds(path)
    if limit is not None and seconds > limit:
        return f"{path.name} runs {seconds:.1f}s; the model takes at most {limit}s"
    return None


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


def print_resume_hint(model: str, queue_id: str) -> None:
    """Say how to pick up a job that left its block with an error, so was not released."""
    print("      The job was not released. Pick it up with")
    print(f"      client.voice_changer.retrieve(model={model!r}, queue_id={queue_id!r})")


def print_refund(error: APIError) -> bool:
    """Say what a failed retrieve reports about the conversion's charge.

    A conversion the provider failed answers with ``credits_refunded`` in the
    error body. A body without it is some other failure, and says nothing about
    the charge. Returns whether the body reported a refund outcome, which means
    the conversion is over.
    """
    refunded = error.body.get("credits_refunded") if isinstance(error.body, dict) else None
    if refunded is True:
        print("      The charge was refunded.")
        return True
    if refunded is False:
        print("      Not refunded: either nothing was charged, or the error says what to do.")
        return True
    print("      The response does not report a refund.")
    return False


async def change_voice(client: VeniceClient) -> int:
    """Quote, then convert a sample recording into another voice.

    Returns the exit status: ``0`` after a real conversion, ``77`` when the
    catalog lists no voice-changer model or the paid opt-in is unset, ``1`` on
    any failure.
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

    # The model has to be known first: it may say which containers it takes. A list
    # without WAV stops this example before it pays for a sample. A model that declares
    # no list gets the billed sample anyway, and the server decides whether to take it.
    spec = (await client.models.get(model)).model_spec
    if not isinstance(spec, MusicModelSpec):
        raise LookupError(f"{model} has no music spec in the catalog")
    voice = choose_target_voice(spec)
    print(f"   Target voice: {voice or 'the model default'} ({len(spec.voices or [])} listed)")

    # Free, and before anything billed: a short clip is billed as a whole minute. A
    # model whose longest source is under a minute is quoted at that longest source.
    longest = spec.max_source_audio_duration_seconds or BILLED_MINUTE_SECONDS
    minute = await client.voice_changer.quote(
        model=model, duration_seconds=min(BILLED_MINUTE_SECONDS, longest)
    )
    print(f"   Quote for the minute a short clip is billed as: ${minute.quote:.4f} USD")
    if not RUN_PAID:
        print("\nSKIPPED: VENICE_RUN_PAID_VOICE_CHANGER is not set, so nothing was converted.")
        return EXIT_SKIPPED

    try:
        sample = await voice_sample(
            client, RESULTS_DIR, accepted_formats=spec.accepted_audio_formats
        )
    except NoMatchingModelError as e:
        print("\nSKIPPED: no TTS model can synthesize the source recording.")
        print(f"   {e}")
        return EXIT_SKIPPED

    # Price the clip's own length first, as a bare count of seconds, not "9s".
    # Measuring it needs a PCM WAV file; anything else fails before the paid
    # conversion.
    try:
        seconds = sample_seconds(sample)
        problem = source_problem(sample, spec)
    except (OSError, wave.Error, EOFError) as e:
        print(f"   ❌ {sample} could not be read as a PCM WAV file: {e}")
        return 1
    if problem is not None:
        print(f"   ❌ {problem}")
        return 1
    quote = await client.voice_changer.quote(model=model, duration_seconds=seconds)
    print(f"   Quote for the {seconds}s sample: ${quote.quote:.4f} USD")
    print("   (an estimate: you are billed on the length Venice measures, rounded up to a minute)")

    print(f"\n   Converting: {sample.name}")
    job = await client.voice_changer.run(model=model, file=sample, voice=voice)
    print(f"   Queued: {job.queue_id}")
    print(f"   Measured length: {job.duration_seconds:.1f}s, billed rounded up to a whole minute")

    def show(status: VoiceChangerProcessingStatus) -> None:
        print(f"   ⏳ {status.progress_percent:.0f}% …", end="\r")

    # The handlers sit outside the block: a clean exit releases the
    # provider-held media, so a failure has to raise through it to keep them.
    try:
        async with job:
            status = await job.wait(poll_interval=3.0, on_progress=show)
            print()
            extension = output_extension(status)
            if extension is None:
                raise ValueError(
                    f"The converted result ({status.content_type}) is not a known audio format"
                )
            out_path = await job.download(RESULTS_DIR / f"voice_changed.{extension}", status)

            # The file above is the copy kept. One-shot retrieval returns the audio
            # again and releases the media in the same call.
            one_shot_worked = False
            try:
                final = await client.voice_changer.retrieve(
                    model=job.model, queue_id=job.queue_id, delete_media_on_completion=True
                )
            except VeniceError as e:
                print(f"   ℹ️  One-shot retrieve not confirmed: {type(e).__name__}: {e}")
            else:
                if isinstance(final, VoiceChangerCompletedStatus) and final.data:
                    one_shot_worked = True
                else:
                    print("   ℹ️  One-shot retrieve returned no audio; the saved copy stands")
    except TimeoutError as e:
        # wait() ran out of polls; the conversion is still running.
        print(f"\n   ❌ {e} (queue_id={job.queue_id})")
        print_resume_hint(job.model, job.queue_id)
        return 1
    except (APITimeoutError, APIConnectionError) as e:
        # One status request failed; download() only writes bytes already held.
        print(f"\n   ❌ A status request failed (queue_id={job.queue_id}): {e}")
        print_resume_hint(job.model, job.queue_id)
        return 1
    except APIError as e:
        # There is no FAILED status: a failed conversion arrives as an HTTP error.
        print(f"\n   ❌ Retrieving the conversion failed (queue_id={job.queue_id}): {e}")
        if not print_refund(e):
            print_resume_hint(job.model, job.queue_id)
        return 1
    except ValueError as e:
        print(f"\n   ❌ {e} (queue_id={job.queue_id})")
        print_resume_hint(job.model, job.queue_id)
        return 1

    size = out_path.stat().st_size
    print(f"\n   🔊 Saved {size} bytes of {status.content_type} → {out_path}")

    print("\n✨ Voice changer example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.models.resolve_voice_changer() — a capability, not a type")
    print("   - A target voice read from the model's catalog entry")
    print(f"   - voice_changer.quote(duration_seconds={seconds}) — bare seconds, not '{seconds}s'")
    print("   - voice_changer.run(...) → VoiceChangerJob, as an async context manager")
    print("   - job.wait() → the audio itself; there is no FAILED status to check")
    print("   - status.audio_format names the file from the response's content type")
    if one_shot_worked:
        print("   - retrieve(delete_media_on_completion=True) — fetch and release in one call")
    return 0


async def main() -> int:
    """Run the voice-changer example and return its exit status.

    ``0`` if the conversion succeeded, ``77`` if it was skipped for lack of a
    model or of the paid opt-in, and ``1`` otherwise. The success banner prints
    only after a real conversion.
    """
    print("🚀 Venice AI Voice Changer")
    print("=" * 50)

    try:
        async with VeniceClient() as client:
            status = await change_voice(client)
    except (VeniceError, OSError, LookupError, ValueError, wave.Error, EOFError) as e:
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
