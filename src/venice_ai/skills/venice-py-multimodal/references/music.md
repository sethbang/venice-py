# Music generation

Sourced from `src/venice_ai/resources/music.py`. **Music is its own resource in v2** (`client.music`), not under `client.audio` — there is no `client.audio.generate_music(...)`. The lifecycle pattern is identical to video; see `job-lifecycle.md` for shared patterns.

## The high-level pattern

```python
import asyncio
from pathlib import Path
from venice_ai import VeniceClient


async def make_track(prompt: str, out_path: Path, duration: int = 30) -> Path:
    async with VeniceClient() as client:
        pick = await client.models.resolve_cheapest_music(duration_seconds=duration)
        async with await client.music.run(            # NOTE: `await` BEFORE `async with`
            model=pick.model,
            prompt=prompt,
            **pick.request_params,                    # the duration the model was quoted at
        ) as job:
            status = await job.wait(
                on_progress=lambda s: print(f"\r{s.progress_percent:.0f}%", end=""),
                max_polls=60,
            )
            return await job.download(out_path, status)
```

`client.music.run(...)` is `async def` returning `MusicJob`. The form is `async with await client.music.run(...) as job:`.

`resolve_cheapest_music()` quotes every generator at a request that fits the length you ask for and returns `request_params` to submit with. Models differ: some take any length in a range, some only fixed options (ace-step makes 60-second clips at minimum), and some (minimax, lyria) take no `duration_seconds` at all and choose the length themselves (`pick.effective_seconds is None`). Sending `duration_seconds` to one of those raises `ValueError` before the request. Name the file from `status.audio_format` (`"flac"`, `"mp3"`, `"wav"`, `"m4a"`).

## Methods

| Method | Returns | Use when |
|---|---|---|
| `client.music.run(...)` | `MusicJob` (lifecycle manager) | Default — 90% of cases |
| `client.music.submit(...)` | `MusicQueueResponse` (`.model` / `.queue_id` / `.status`) | Producer/consumer split — submit now, retrieve later |
| `client.music.retrieve(*, model=, queue_id=)` | `MusicRetrieveResponse` (status object) | Poll a queued job from its `model` + `queue_id` |
| `client.music.release(*, model=, queue_id=)` | `MusicCompleteResponse` | Delete a finished job's stored media by `model` + `queue_id` without a context manager (does not stop generation) |
| `client.music.quote(...)` | `MusicQuoteResponse` (`.quote`) | Pre-flight cost estimation |

## Parameters

```python
await client.music.run(
    model=...,                                # str — required
    prompt=...,                               # str — required (description of the track)
    duration_seconds=30,                      # int | str — seconds; varies by model
    # Other parameters depend on the model — check client.models.list(type="music")
)
```

The exact parameter set is model-specific. Common controls (model-dependent):
- **Genre / style hints in `prompt`** — "lo-fi beats, 90 BPM, mellow"
- **`duration_seconds`** — typically 10s to 120s; check the model's max
- **`negative_prompt`** — what to avoid (some models)

## Cost-quote BEFORE running

Music can be expensive depending on duration and model. Always quote first:

```python
quote = await client.music.quote(
    model=await client.models.resolve_music(),
    duration_seconds=30,                      # quote takes NO prompt — just model + duration
)
print(f"Estimated cost: ${quote.quote}")      # MusicQuoteResponse.quote (a number)

if float(quote.quote) > 0.20:
    raise SystemExit("over budget")

async with await client.music.run(...) as job:
    ...
```

## Producer/consumer pattern

If you want to submit jobs in one process and download them in another:

```python
# Producer
async with VeniceClient() as client:
    queued = await client.music.submit(        # -> MusicQueueResponse
        model=await client.models.resolve_music(),
        prompt="...",
        duration_seconds=30,
    )
    db.save_pending_music_job(queued.model, queued.queue_id)

# Consumer (later, possibly different process)
async with VeniceClient() as client:
    status = await client.music.retrieve(       # keyword-only -> MusicRetrieveResponse
        model=saved_model,
        queue_id=saved_queue_id,
    )
    # status carries the job state/result; download once complete
```

`retrieve()` is keyword-only (`model=` + `queue_id=`) and returns a `MusicRetrieveResponse` status object — it does **not** rebuild a `MusicJob`. The file/URL state is server-side; poll `retrieve()` until the job reports complete.

## Releasing storage (`release`)

`release()` wraps `POST /audio/complete`: it deletes the job's stored audio and queue entry. It does **not** stop or un-bill a job — billing happens when the job is queued, and a job still generating stores its audio when it finishes. Call it after downloading. Two paths:

```python
# A. The async with block releases a finished job on exit
async with await client.music.run(...) as job:
    status = await job.wait()
    await job.download(path, status)

# B. By model + queue_id, no context manager (keyword-only)
await client.music.release(model=stored_model, queue_id=stored_queue_id)
```

Leaving the `async with` block with an exception (a failed download, a disk error, a `wait()` timeout) releases nothing: the WARNING names the `queue_id`, and `client.music.retrieve(model=..., queue_id=...)` gets the audio back until you `release()` it. Leaving it cleanly before the job finished polls once more; if the job is still generating it is left alone and a WARNING names the `queue_id`. Call `wait()` again, download, then `release()`. Venice answers `success: true` even for an id that was already released; `success: false` means the cleanup did not complete and is logged. After a release, `retrieve()` raises `NotFoundError` ("Media could not be found").

## Common bugs

- **`client.audio.generate_music(...)`** — not a v2 method. Use `client.music.run(...)`. (Lint V102.)
- **`async with client.music.run(...)` (no `await`)** — TypeError. `run()` is async; use `async with await ...`.
- **Relying on the default `max_polls=120`** — for longer tracks raise it explicitly via `wait(max_polls=N, poll_interval=...)`.
- **Treating a `MusicJob` like a `VideoJob`** — same shape, different class. Don't try to interchange instances.
- **Skipping `client.music.quote()` for long tracks** — surprise costs.
- **Sending `duration_seconds` to a model that chooses its own length** — raises `ValueError`. Build the request with `resolve_cheapest_music()` or `venice_ai.music_request_params(spec, seconds)`.
- **Trusting `progress_percent`** — it is elapsed time over the model's typical run time, capped at 100. A slow job sits at 100% while still processing.

## Related references

- `job-lifecycle.md` — `async with await ... as job:` pattern shared with video.
- `video.md` — sister modality with the same lifecycle.
- `audio-tts-stt.md` — speech (NOT music) lives on `client.audio.*`.
- `venice-py/references/model-resolution.md` — `resolve_music()`.
- `venice-py-production/references/error-taxonomy.md` — `MusicGenerationError`.
