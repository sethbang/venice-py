#!/usr/bin/env python3
"""
Venice AI SDK — Long-text TTS via stream_long_text
===================================================

Demonstrates ``client.audio.stream_long_text``, the helper that splits long
input into parallel segments and yields concatenated audio in order.

Use this helper when either:

1. **Output truncation:** some TTS models cap each response at a fixed audio
   length (about 15.9 s) regardless of input length, so a long poem renders
   only its first stanza and a half. The helper knows these models and splits
   the input so each segment stays under the cap and the full text is voiced.

2. **Pseudo-streaming on buffering models:** many TTS models buffer the whole
   response server-side before sending any bytes. Splitting and dispatching
   segments in parallel starts later segments while the first one is still
   generating.

For models that truly stream and don't truncate, the helper short-circuits to
a single ``create_speech`` call when the input fits its word budget, so it is
safe to call whatever model you use.

Neither demo names a model. ``venice_ai.audio_helpers.split_text_for_tts``
applies the same per-model budget as the helper, so asking it how it would
split the poem finds a model it splits for and one it passes through. The
candidates come from ``resolve_tts(prefer="cheapest")`` in price order, so
each demo runs on the cheapest model that shows its behavior. Each result is
transcribed back to confirm the whole poem, last line included, made it into
the audio.

Run::

    poetry run python examples/audio/long_text_streaming.py

(Requires ``VENICE_API_KEY`` in the environment.)
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.audio_helpers import DEFAULT_WORD_BUDGET, MODEL_WORD_BUDGETS, split_text_for_tts
from venice_ai.exceptions import VeniceError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import tts_formats, tts_spec  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LONG_POEM = (
    "Beneath the silver moon's soft glow,\n"
    "Where rivers hum and willows sway,\n"
    "The night air holds a tender flow,\n"
    "As dreamers drift in twilight's play.\n"
    "\n"
    "A whispered breeze through pines aloft,\n"
    "Carries tales of stars long-burned,\n"
    "Their light, though distant, lingers soft,\n"
    "On every wave the ocean turned."
)

#: The last line: if it is in the transcript, the audio was not cut short.
LAST_LINE = LONG_POEM.splitlines()[-1]

#: Minimum share of the poem's words the transcript must contain.
MIN_OVERLAP = 0.8

#: Exit status for a run that skipped because a prerequisite is missing.
EXIT_SKIPPED = 77


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def word_overlap(expected: str, actual: str) -> float:
    want = _words(expected)
    got = set(_words(actual))
    return sum(w in got for w in want) / len(want) if want else 0.0


async def pick_models(
    client: VeniceClient,
) -> tuple[tuple[str, str] | None, tuple[str, str] | None]:
    """Return ``(model, voice)`` for a model the helper splits and one it passes through.

    Walks the TTS models cheapest first: each ``resolve_tts(prefer="cheapest")``
    call excludes the models already tried. Only models that output MP3
    qualify, since both demos write ``.mp3``.
    """
    specs = {
        m.id: tts_spec(m.model_spec, m.id) for m in (await client.models.list(type="tts")).data
    }
    split: tuple[str, str] | None = None
    passthrough: tuple[str, str] | None = None
    tried: list[str] = []
    while split is None or passthrough is None:
        try:
            model_id = await client.models.resolve_tts(prefer="cheapest", exclude_models=tried)
        except NoMatchingModelError:
            break
        tried.append(model_id)
        spec = specs.get(model_id)
        if spec is None or "mp3" not in tts_formats(spec) or not spec.voices:
            continue
        details = (await client.audio.get_voices(model_id=model_id)).data
        english = [v.id for v in details if v.language == "English"]
        voice = (english or spec.voices)[0]
        segments = split_text_for_tts(LONG_POEM, model=model_id)
        if len(segments) > 1 and split is None:
            split = (model_id, voice)
        elif len(segments) == 1 and passthrough is None:
            passthrough = (model_id, voice)
    return split, passthrough


async def render(
    client: VeniceClient, model: str, voice: str, output: Path, timeout: float
) -> tuple[bytes, list[tuple[int, float, int]], float]:
    """Run stream_long_text and write the audio to ``output``."""
    segment_log: list[tuple[int, float, int]] = []
    t0 = time.monotonic()
    chunks: list[bytes] = []
    stream = await client.audio.stream_long_text(
        input=LONG_POEM,
        model=model,
        voice=voice,
        response_format="mp3",
        max_concurrency=4,
        on_segment_complete=lambda i, t, b: segment_log.append((i, t, b)),
        timeout=timeout,
    )
    async for chunk in stream:
        chunks.append(chunk)
    body = b"".join(chunks)
    output.write_bytes(body)
    return body, segment_log, time.monotonic() - t0


async def verify_transcript(client: VeniceClient, output: Path) -> bool:
    """Transcribe the stitched audio and check the whole poem is in it.

    Also prints the audio length the ASR model measured, so the split demo
    can be compared with the per-response cap described above.
    """
    asr_model = await client.models.resolve_asr(prefer="cheapest")
    transcript = await client.audio.transcribe(file=output, model=asr_model, language="en")
    overall = word_overlap(LONG_POEM, transcript.text)
    ending = word_overlap(LAST_LINE, transcript.text)
    print(f"🔁 Transcribed back with {asr_model}:")
    print(f"   {transcript.text}")
    if transcript.duration is not None:
        print(f"⏱  Audio length measured by {asr_model}: {transcript.duration:.1f}s")
    print(
        f"📊 Poem words heard: {overall:.0%}; last line ({LAST_LINE!r}) words heard: {ending:.0%}"
    )
    if overall >= MIN_OVERLAP and ending >= MIN_OVERLAP:
        return True
    print("❌ The audio does not contain the whole poem")
    return False


async def split_long_text(client: VeniceClient, model: str, voice: str) -> bool:
    """Render the full poem on a model the helper splits for."""
    planned = split_text_for_tts(LONG_POEM, model=model)
    budget = MODEL_WORD_BUDGETS.get(model, DEFAULT_WORD_BUDGET)
    print(f"\n🎙  {model} long text via stream_long_text (split)")
    print("-" * 50)
    print(f"🗣  Voice: {voice}")
    # MODEL_WORD_BUDGETS lists the models whose single responses are cut
    # short; the budget keeps each segment's audio under that cap.
    print(
        f"📏 The poem is {len(LONG_POEM.split())} words; {model} is budgeted "
        f"{budget} words per request (audio_helpers.MODEL_WORD_BUDGETS)"
    )
    print(f"✂️  The helper plans {len(planned)} segments for this model:")
    for idx, segment in enumerate(planned):
        print(f"   {idx}: {len(segment.split())} words, {segment[:40]!r}…")

    output = RESULTS_DIR / "long_text_split.mp3"
    try:
        body, segment_log, elapsed = await render(client, model, voice, output, timeout=180.0)
    except VeniceError as e:
        print(f"❌ stream_long_text failed: {e}")
        return False

    print(f"✅ Saved {len(body)} bytes to {output}")
    print(f"⏱  Wall time: {elapsed:.1f}s")
    print(f"🧩 Segments completed: {len(segment_log)}")
    # All segments are dispatched at once, so these are completion times
    # measured from dispatch, not per-segment durations.
    for idx, finished, n_bytes in sorted(segment_log):
        print(f"   segment {idx}: finished {finished:.1f}s after dispatch, {n_bytes} bytes")
    if len(segment_log) != len(planned) or not body:
        print(f"❌ Expected {len(planned)} completed segments")
        return False
    try:
        ok = await verify_transcript(client, output)
    except VeniceError as e:
        print(f"❌ Transcription failed: {e}")
        return False
    print(f"▶  Play with:  afplay {output}")
    return ok


async def passthrough_long_text(client: VeniceClient, model: str, voice: str) -> bool:
    """Same input on a model the poem fits: the helper makes a single call.

    The poem fits under this model's word budget, so the helper passes the
    whole text to ``create_speech`` unchanged and reports no segments.
    """
    budget = MODEL_WORD_BUDGETS.get(model, DEFAULT_WORD_BUDGET)
    print(f"\n🎙  {model} long text via stream_long_text (passthrough)")
    print("-" * 50)
    print(f"🗣  Voice: {voice}")
    print(
        f"📏 The poem is {len(LONG_POEM.split())} words; {model} is budgeted "
        f"{budget} words per request, so it goes out in one call"
    )

    output = RESULTS_DIR / "long_text_passthrough.mp3"
    try:
        body, segment_log, elapsed = await render(client, model, voice, output, timeout=60.0)
    except VeniceError as e:
        print(f"❌ stream_long_text failed: {e}")
        return False

    print(f"✅ Saved {len(body)} bytes to {output}")
    print(f"⏱  Wall time: {elapsed:.1f}s")
    print(f"🧩 Split segments: {len(segment_log)} (single passthrough call)")
    if segment_log or not body:
        print("❌ Expected a single passthrough call with no split segments")
        return False
    try:
        ok = await verify_transcript(client, output)
    except VeniceError as e:
        print(f"❌ Transcription failed: {e}")
        return False
    print(f"▶  Play with:  afplay {output}")
    return ok


async def main() -> int:
    """Run both demos.

    Returns ``0`` if every demo that ran succeeded (one may be skipped when
    the catalog has no model of its kind), ``77`` if both were skipped, and
    ``1`` if any demo failed.
    """
    print("🚀 Venice AI long-text TTS")
    print("=" * 50)

    async with VeniceClient() as client:
        split, passthrough = await pick_models(client)
        results: list[tuple[str, bool | None]] = []

        if split is None:
            print(
                "\nSection skipped: split demo: no catalog TTS model needs splitting for this poem"
            )
            results.append(("split_long_text", None))
        else:
            results.append(("split_long_text", await split_long_text(client, *split)))

        if passthrough is None:
            print(
                "\nSection skipped: passthrough demo: no catalog TTS model fits the poem in one call"
            )
            results.append(("passthrough_long_text", None))
        else:
            results.append(
                ("passthrough_long_text", await passthrough_long_text(client, *passthrough))
            )

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if skipped:
        print(f"\n⏭️  Skipped: {', '.join(skipped)}")
    if failed:
        print(f"\n⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
        return 1
    if len(skipped) == len(results):
        print("\nSKIPPED: no catalog TTS model fits either demo, so nothing ran.")
        return EXIT_SKIPPED
    print("\n✨ Long-text TTS example completed!")
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
