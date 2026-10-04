#!/usr/bin/env python3
"""
Venice AI SDK — Music Generation
================================

End-to-end example of the async music generation flow. The SDK exposes
music generation on its own resource — :attr:`VeniceClient.music` — even
though it shares the underlying ``/audio/*`` queue endpoints with TTS /
ASR. The lifecycle mirrors video: ``submit`` → ``retrieve`` → ``release``,
or use ``run()`` for the high-level :class:`MusicJob` context manager.

Key features covered:
- ``models.resolve_cheapest_music(duration_seconds=...)`` instead of a
  hardcoded ID. It quotes every music generator (free) at the request that
  model would need for the clip length you want, and returns the cheapest
  one together with those ``request_params`` and the clip length they
  produce. Passing ``request_params`` to ``quote`` / ``run`` / ``submit``
  bills you the price it was ranked by. Sound-effect, TTS and voice-changer
  models that Venice also types as ``music`` are left out.
- ``require_force_instrumental=True`` to find the cheapest model that
  accepts ``force_instrumental``, which suppresses vocals
- Price quoting before submission
- The high-level :class:`MusicJob` lifecycle: the example saves the audio,
  then releases the job's stored copy with ``job.release()`` and prints the
  result. A block that raises (a failed download, a full disk) releases
  nothing, so a billed clip is never deleted before it is saved
- A timeout is resumable: the job is billed when queued and keeps running,
  so the example waits on the *same* job again instead of queueing another,
  and prints the ``queue_id`` to resume from whenever a queued job is left
  unfinished
- Low-level ``client.music.submit`` / ``retrieve`` / ``release``, releasing
  only once the audio is on disk (or the job failed and has none)
- Naming the saved file from the format the server reported
  (``MusicCompletedStatus.audio_format``)

Cost: each run makes two paid generations on the cheapest model at the
target length (the price is printed before each one). The
``force_instrumental`` section is quoted for free on every run and generates
only when you opt in, because the models that support it cost more::

    VENICE_RUN_PAID_MUSIC=1 poetry run python examples/music/music_generation.py

Outputs are written to ``examples/results/music/``.

Exit status: ``0`` when both core demos produced audio, ``1`` on any failure,
``77`` (with a ``SKIPPED:`` line) when no music generator in the catalog can
make the target clip length (``NoMatchingModelError``). Candidates whose
quotes all failed (``ModelQuotesUnavailableError``) are a failure, not a
skip: the catalog has the models, but they could not be priced. The
``force_instrumental`` section prints ``Section skipped:`` when it only
quotes or no model accepts the flag; that does not change the exit status.
"""

import asyncio
import math
import os
import sys
from collections.abc import Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp

from venice_ai import (
    CheapestMusicResult,
    ModelQuotesUnavailableError,
    MusicGenerationError,
    NoMatchingModelError,
    VeniceClient,
    VeniceError,
)
from venice_ai.types.api.models import MusicModelSpec
from venice_ai.types.api.music import (
    MusicCompletedStatus,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicProcessingStatus,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "music"

#: Clip length to ask for. The resolver turns it into each model's nearest
#: valid request (some only offer fixed lengths, some choose their own).
TARGET_SECONDS = 10
#: How long to wait for a job. Generation time varies by model and by load
#: (each processing status reports the model's typical time as
#: ``average_execution_time``), so this is the same ten minutes
#: ``MusicJob.wait`` allows by default, not a figure tuned to one model. A
#: job still generating after that is already billed: it is waited on once
#: more, or its ``queue_id`` is printed to resume from, never queued again.
WAIT_BUDGET_SECONDS = 600
POLL_INTERVAL_SECONDS = 5.0
MAX_POLLS = math.ceil(WAIT_BUDGET_SECONDS / POLL_INTERVAL_SECONDS)

#: Opt-in for the paid force_instrumental generation (quoted either way).
RUN_PAID = os.environ.get("VENICE_RUN_PAID_MUSIC") == "1"

#: Exit status for a run that skipped because a prerequisite is missing.
EXIT_SKIPPED = 77


def _ts() -> str:
    """Compact timestamp suffix for unique output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _length(pick: CheapestMusicResult) -> str:
    if pick.effective_seconds is None:
        return "model-chosen length (it takes no duration)"
    return f"{pick.effective_seconds}s clip"


def print_pick(pick: CheapestMusicResult) -> None:
    """Show the chosen model and how the alternatives were quoted."""
    print(f"   🤖 Cheapest: {pick.model} at ${pick.quote_usd:.2f}, {_length(pick)}")
    print(f"      request_params={pick.request_params}")
    ranked = sorted(pick.all_quotes.items(), key=lambda kv: kv[1])
    others = ", ".join(f"{mid} ${usd:.2f}" for mid, usd in ranked if mid != pick.model)
    if others:
        print(f"      Other quotes: {others}")
    for mid, reason in pick.skipped.items():
        print(f"      Not considered: {mid} ({reason})")


def _print_quote_failures(error: ModelQuotesUnavailableError) -> None:
    """List each candidate's quote failure.

    A rate limit or server error is worth retrying later; a rejected quote
    request (``InvalidRequestError``) is not, so the type is shown per model.
    """
    for mid, exc in error.failures.items():
        print(f"      {mid}: {type(exc).__name__}: {exc}")
    for mid, reason in error.skipped.items():
        print(f"      Not considered: {mid} ({reason})")


async def music_spec(client: VeniceClient, model: str, prompt: str) -> MusicModelSpec:
    """Read the model's catalog entry and check the prompt against its limits.

    The resolver does not see the prompt, and prompt-length limits are per
    model, so they are checked here before anything is billed.
    """
    spec = (await client.models.get(model)).model_spec
    if not isinstance(spec, MusicModelSpec):
        raise LookupError(f"{model} has no music spec in the catalog")
    if spec.prompt_character_limit is not None and len(prompt) > spec.prompt_character_limit:
        raise ValueError(f"{model} accepts prompts of at most {spec.prompt_character_limit} chars")
    if spec.min_prompt_length is not None and len(prompt) < spec.min_prompt_length:
        raise ValueError(f"{model} needs a prompt of at least {spec.min_prompt_length} chars")
    return spec


def output_path(stem: str, status: MusicCompletedStatus, spec: MusicModelSpec) -> Path:
    """Name the file from the format the server reported.

    Inline audio carries its Content-Type, exposed as ``status.audio_format``.
    A status that links to the audio by ``url`` has no content type, so the
    model's declared ``default_format`` names the file instead, and that is
    said out loud.
    """
    fmt = status.audio_format
    if fmt is None:
        fmt = spec.default_format or "audio"
        print(f"   ℹ️  No content type on the status; naming the file from the catalog: .{fmt}")
    elif spec.supported_formats and fmt not in spec.supported_formats:
        print(
            f"   ℹ️  The server sent {fmt} ({status.content_type}); the catalog lists "
            f"{', '.join(spec.supported_formats)}"
        )
    return RESULTS_DIR / f"{stem}_{_ts()}.{fmt}"


def _typical_time(status: object) -> str:
    """The server's typical run time for the model, from a processing status."""
    if isinstance(status, MusicProcessingStatus):
        return (
            f" (the server's typical time for this model is "
            f"{status.average_execution_time / 1000:.0f}s; "
            f"this job has run {status.execution_duration / 1000:.0f}s)"
        )
    return ""


def _print_resume_hint(model: str, queue_id: str, *, finished: bool) -> None:
    """Say how to pick up a billed job that was not released."""
    if finished:
        print("   The job is billed and its audio is still stored. Save it, don't queue it again:")
    else:
        print("   The job is billed and may still be generating. Resume it, don't queue it again:")
    print(f"      client.music.retrieve(model={model!r}, queue_id={queue_id!r})")
    print("      then client.music.release(...) once the audio is saved.")


async def _release(
    release: Coroutine[Any, Any, MusicCompleteResponse], model: str, queue_id: str, *, what: str
) -> bool:
    """Await a release call (``what`` names what it deletes) and report the result.

    Call it only once nothing the job holds is still needed: its audio is
    saved, or it failed and has none. Returns ``True`` only if the server
    confirmed the release.
    """
    try:
        response = await release
    except (VeniceError, aiohttp.ClientError) as e:
        print(f"   ❌ Could not release the server-side copy: {type(e).__name__}: {e}")
    else:
        if response.success:
            print(f"   🧹 Released {what}")
            return True
        print("   ❌ Release returned success=false: the server did not confirm the cleanup")
    print(f"      Retry later: client.music.release(model={model!r}, queue_id={queue_id!r})")
    return False


async def generate_with_job(
    client: VeniceClient,
    pick: CheapestMusicResult,
    prompt: str,
    stem: str,
    *,
    force_instrumental: bool | None = None,
) -> Path | None:
    """Quote, generate and download with the managed :class:`MusicJob`.

    Returns the written file, or ``None`` if any step failed.
    """
    job = None
    try:
        spec = await music_spec(client, pick.model, prompt)
        quote = await client.music.quote(
            model=pick.model, character_count=len(prompt), **pick.request_params
        )
        print(f"   💰 Quoted cost: ${quote.quote:.4f}")

        # A block that raises leaves the finished job's audio stored (only a
        # failed or rejected job, which has none, is released on the way out),
        # so a clip is never deleted before it is saved. After the save the
        # block releases the job itself to report the result; the release the
        # context manager repeats on a clean exit then has nothing to delete.
        async with await client.music.run(
            model=pick.model,
            prompt=prompt,
            force_instrumental=force_instrumental,
            **pick.request_params,
        ) as job:
            print(f"   ⏳ Queued with queue_id={job.queue_id}")

            def _on_progress(status: MusicProcessingStatus) -> None:
                print(f"   ⏱  Progress: {status.progress_percent:5.1f}%", end="\r")

            try:
                status = await job.wait(
                    poll_interval=POLL_INTERVAL_SECONDS,
                    max_polls=MAX_POLLS,
                    on_progress=_on_progress,
                )
            except TimeoutError:
                # A TimeoutError leaves the job resumable: wait on it again.
                print(
                    f"\n   ⏳ Still generating after {WAIT_BUDGET_SECONDS}s"
                    f"{_typical_time(job.status)}; waiting on the same job"
                )
                status = await job.wait(
                    poll_interval=POLL_INTERVAL_SECONDS,
                    max_polls=MAX_POLLS,
                    on_progress=_on_progress,
                )
            print()

            saved = await job.download(output_path(stem, status, spec), status)
            kind = status.content_type or "audio"
            print(f"   💾 Saved {saved.stat().st_size:,} bytes of {kind} to: {saved}")
            released = await _release(
                job.release(),
                job.model,
                job.queue_id,
                what="the job's server-side copy of the audio",
            )
    except MusicGenerationError as e:
        # wait() raises this for a failed or rejected job, which has no audio
        # and is released on exit; download() raises it for a completed status
        # that carries no audio, and that job stays stored.
        print(f"\n   ❌ Generation failed: {e} (code={e.error_code})")
        if job is not None and job.is_complete:
            print("      The job was not released; clear it with")
            print(f"      client.music.release(model={job.model!r}, queue_id={job.queue_id!r})")
        return None
    except TimeoutError as e:
        print(f"\n   ❌ {e}{_typical_time(job.status if job is not None else None)}")
        if job is not None:
            _print_resume_hint(job.model, job.queue_id, finished=job.is_complete)
        return None
    except (LookupError, ValueError, VeniceError, aiohttp.ClientError, OSError) as e:
        print(f"\n   ❌ {type(e).__name__}: {e}")
        # A billed job that did not fail keeps its audio when the block
        # raises, so say how to pick it up again.
        if job is not None and not job.is_failed:
            _print_resume_hint(job.model, job.queue_id, finished=job.is_complete)
        return None

    return saved if released else None


async def quote_then_generate(client: VeniceClient, pick: CheapestMusicResult) -> Path | None:
    """The high-level flow: quote, then generate with :class:`MusicJob`."""
    print("\n🎵 Music Generation — Quote → Generate → Download")
    print("-" * 60)
    prompt = (
        "Upbeat cinematic orchestral opener with bright strings, "
        "light percussion, and a warm brass swell at the end."
    )
    return await generate_with_job(client, pick, prompt, "music_out")


async def low_level_flow(client: VeniceClient, pick: CheapestMusicResult) -> Path | None:
    """Same task using the raw submit / retrieve / release methods.

    Useful if you want to interleave other work between polls, or integrate
    the job state into a custom task system. Returns the written file, or
    ``None`` if the job failed, is still generating, or any call raised.
    """
    print("\n🎚️  Music Generation — Low-level submit/retrieve/release")
    print("-" * 60)

    prompt = "Gentle lo-fi hip-hop beat with warm vinyl crackle and a soft piano loop."
    model = pick.model
    queue_id: str | None = None
    status = None

    try:
        spec = await music_spec(client, model, prompt)
        queued = await client.music.submit(model=model, prompt=prompt, **pick.request_params)
        queue_id = queued.queue_id
        print(f"   ⏳ queue_id={queue_id}")

        # MusicJob.wait() hides this loop; it is spelled out here.
        for _ in range(MAX_POLLS):
            status = await client.music.retrieve(model=model, queue_id=queue_id)
            if not isinstance(status, MusicProcessingStatus):
                break
            print(f"   ⏱  Progress: {status.progress_percent:5.1f}%", end="\r")
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        if status is None or isinstance(status, MusicProcessingStatus):
            # Releasing now would delete nothing: the audio does not exist yet.
            print(
                f"\n   ❌ Job did not complete within {WAIT_BUDGET_SECONDS}s{_typical_time(status)}"
            )
            _print_resume_hint(model, queue_id, finished=False)
            return None
        print()

        if isinstance(status, MusicFailedStatus):
            # A failed job has no audio to keep, so its entry is released now.
            print(f"   ❌ Failed: {status.error} (code={status.error_code})")
            await _release(
                client.music.release(model=model, queue_id=queue_id),
                model,
                queue_id,
                what="the failed job's queue entry",
            )
            return None

        # Audio arrives either inline on ``status.data`` or as a download link
        # on ``status.url``. It is written to disk before anything is released:
        # if the download or the write fails, the stored copy is the only one.
        if status.data:
            audio = status.data
        elif status.url:
            audio = await client.fetch_external(status.url)
        else:
            audio = b""
        if not audio:
            print("   ❌ Completed status carried no audio; the job was not released")
            print(f"      client.music.release(model={model!r}, queue_id={queue_id!r}) clears it")
            return None
        written = output_path("music_out_lowlevel", status, spec)
        written.write_bytes(audio)
        kind = status.content_type or "audio"
        print(f"   💾 Saved {len(audio):,} bytes of {kind} to: {written}")
    except (LookupError, ValueError, VeniceError, aiohttp.ClientError, OSError) as e:
        print(f"\n   ❌ {type(e).__name__}: {e}")
        # Nothing was released, so a queued job keeps whatever it produced.
        if queue_id is not None:
            _print_resume_hint(model, queue_id, finished=isinstance(status, MusicCompletedStatus))
        return None

    released = await _release(
        client.music.release(model=model, queue_id=queue_id),
        model,
        queue_id,
        what="the job's server-side copy of the audio",
    )
    return written if released else None


async def instrumental_demo(client: VeniceClient) -> Path | bool | None:
    """Find the cheapest model that takes ``force_instrumental`` and quote it.

    Generates only with ``VENICE_RUN_PAID_MUSIC=1``. Returns the written file,
    ``False`` on failure, and ``None`` when it only quoted or no model in the
    catalog accepts the flag.
    """
    print("\n🎻 force_instrumental — the cheapest model that accepts it")
    print("-" * 60)
    try:
        pick = await client.models.resolve_cheapest_music(
            duration_seconds=TARGET_SECONDS, require_force_instrumental=True
        )
    except NoMatchingModelError as e:
        print(f"Section skipped: force_instrumental: no catalog music model accepts it: {e}")
        return None
    except ModelQuotesUnavailableError as e:
        print("   ❌ force_instrumental models exist, but none could be quoted:")
        _print_quote_failures(e)
        return False
    except (VeniceError, aiohttp.ClientError) as e:
        print(f"   ❌ Could not rank force_instrumental models: {type(e).__name__}: {e}")
        return False
    print_pick(pick)

    if not RUN_PAID:
        print(
            "Section skipped: force_instrumental generation: quoted only. "
            "Set VENICE_RUN_PAID_MUSIC=1 to generate it (incurs the cost)."
        )
        return None

    prompt = "Calm ambient piano and soft strings, slow tempo, no vocals."
    saved = await generate_with_job(
        client, pick, prompt, "music_instrumental", force_instrumental=True
    )
    return saved if saved is not None else False


async def main() -> int:
    """Run the demos; return ``0`` only if both core demos wrote audio."""
    print("🚀 Venice AI Music Generation Examples")
    print("=" * 60)
    if RUN_PAID:
        print("💸 VENICE_RUN_PAID_MUSIC=1: the force_instrumental section will generate (paid).")
    else:
        print("🧪 force_instrumental is quote only. Set VENICE_RUN_PAID_MUSIC=1 to generate it.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async with VeniceClient() as client:
        print(f"\n🔎 Cheapest music generator for a {TARGET_SECONDS}s clip (free quotes)")
        try:
            pick = await client.models.resolve_cheapest_music(duration_seconds=TARGET_SECONDS)
        except NoMatchingModelError as e:
            print(
                f"\nSKIPPED: no music generator in the catalog can make a {TARGET_SECONDS}s clip: {e}"
            )
            return EXIT_SKIPPED
        except ModelQuotesUnavailableError as e:
            print(
                f"\n❌ Music generators match a {TARGET_SECONDS}s clip, but none could be quoted:"
            )
            _print_quote_failures(e)
            return 1
        except (VeniceError, aiohttp.ClientError) as e:
            print(f"\n❌ Could not rank music models: {type(e).__name__}: {e}")
            return 1
        print_pick(pick)

        core: list[tuple[str, Path | None]] = [
            ("quote_then_generate", await quote_then_generate(client, pick)),
            ("low_level_flow", await low_level_flow(client, pick)),
        ]
        instrumental = await instrumental_demo(client)

    failed = [name for name, path in core if path is None]
    if instrumental is False:
        failed.append("instrumental_demo")
    written = [path for _, path in core if path is not None]
    if isinstance(instrumental, Path):
        written.append(instrumental)

    if written:
        print("\n📁 Files written by this run:")
        for path in written:
            print(f"   {path}")

    if failed:
        print(f"\n❌ Demos failed: {', '.join(failed)}")
        return 1
    if instrumental is None:
        print("\n⏭️  force_instrumental was not generated this run (see Section skipped above).")

    print("\n✨ Done!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
