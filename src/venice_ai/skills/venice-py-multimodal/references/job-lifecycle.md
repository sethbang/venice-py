# Async-job lifecycle (video + music)

Sourced from `src/venice_ai/resources/video.py` and `src/venice_ai/resources/music.py`. Both resources share the same lifecycle pattern — a job submitted to the server runs for seconds-to-minutes, and the SDK gives you a typed `VideoJob` / `MusicJob` to manage it.

## The shape

```
client.video.run(...)        client.music.run(...)
    │                            │
    └─→ VideoJob                 └─→ MusicJob
         ├── async with job:          ├── async with job:
         │     await job.wait()       │     await job.wait()
         │     await job.download()   │     await job.download()
         ├── await job.cancel()       ├── await job.cancel()
         ├── await job.poll()         ├── await job.poll()
         └── job.queue_id             └── job.queue_id
```

`VideoJob` and `MusicJob` are different classes but expose the same async-context-manager + lifecycle methods. The patterns below apply to both.

## Canonical pattern: `async with` + `wait()` + `download()`

```python
import asyncio
from pathlib import Path
from venice_ai import VeniceClient


async def make_clip(prompt: str, out_path: Path) -> Path:
    async with VeniceClient() as client:
        async with await client.video.run(
            model=await client.models.resolve_video(video_type="text-to-video"),
            prompt=prompt,
            duration_seconds=5,
        ) as job:
            print(f"Job submitted: {job.queue_id}")

            status = await job.wait(
                on_progress=lambda s: print(f"\r{s.progress_percent:.0f}%", end=""),
                max_polls=300,                 # caps the wait; raises TimeoutError when exhausted
                poll_interval=2.0,             # seconds between polls
            )
            print()                            # newline after progress

            saved = await job.download(out_path, status)
            return saved
```

`status` (returned from `wait()`) is a typed `VideoCompletedStatus` / `MusicCompletedStatus` describing the finished job. `download()` takes both the path and the status — passing the status is what tells `download` which output URL to fetch.

## `async with job:` is mandatory

The `async with` block guarantees server-side cleanup if your code exits early — exception, timeout, KeyboardInterrupt. Without it, the job continues running on Venice's side until it completes naturally; you keep paying.

```python
# WRONG — the stored media is never released if wait() raises
job = await client.video.run(...)
status = await job.wait()
await job.download(path, status)

# RIGHT
async with await client.video.run(...) as job:   # __aexit__ releases stored media
    status = await job.wait()
    await job.download(path, status)
```

## Watching progress

`wait()` accepts `on_progress: Callable[[VideoProcessingStatus], None] | None`. The callback fires whenever the server reports progress (typically 0%, 10%, 25%, 50%, 75%, 100% or similar — server-determined cadence).

```python
def render_bar(s) -> None:
    p = s.progress_percent / 100.0
    bar = "█" * int(p * 40)
    print(f"\r[{bar:<40}] {p:.0%}", end="", flush=True)

status = await job.wait(on_progress=render_bar)
```

If you want async progress (e.g., updating a database), wrap a sync callback that spawns a task:

```python
import asyncio
def on_progress(s) -> None:
    asyncio.create_task(persist_progress(job.queue_id, s.progress_percent))
```

## Timeouts

`wait(max_polls=N)` raises `TimeoutError` once `N` polls (each `poll_interval` seconds apart) elapse without completion. The `async with` block still calls `cancel()` on the way out, but that only releases storage: **a job that has not finished keeps generating on the server and is still billed**. The SDK logs a WARNING when the block exits before a terminal status.

```python
try:
    async with await client.video.run(...) as job:
        status = await job.wait(max_polls=60)
        await job.download(path, status)
except asyncio.TimeoutError:
    log.warning("video timed out after 5 minutes")
    # the job is still running (and billed) server-side; persist
    # job.queue_id if you want to pick up the result later
```

By default `wait()` polls up to `max_polls=120` times (~10 min at `poll_interval=5.0`) then raises `TimeoutError`. **Tune `max_polls`/`poll_interval`** for your expected render time; timing out does not save money.

## Errors during the job

The server may report a job failure in the polled status. `wait()` raises:

- `VideoGenerationError(error_code=..., message=...)` for video failures
- `MusicGenerationError(error_code=..., message=...)` for music failures

Inspect `e.error_code` to decide whether to retry. Common codes:

| Code | Retry? | Reason |
|---|---|---|
| `INFERENCE_FAILED` | maybe | Transient render failure |
| `UPSCALE_FAILED` | maybe | Same as above |
| `CONTENT_POLICY_VIOLATION` | **no** | Prompt rejected; surface to operator |
| `INVALID_PROMPT` | **no** | Schema/content rejected |
| `TIMEOUT` (server-side) | yes | Server's own render queue timed out |

```python
from venice_ai.exceptions import VideoGenerationError

try:
    async with await client.video.run(...) as job:
        status = await job.wait()
        await job.download(path, status)
except VideoGenerationError as e:
    if e.error_code in ("CONTENT_POLICY_VIOLATION", "INVALID_PROMPT"):
        raise                        # terminal — fix the input
    # else maybe re-submit
```

## Manual polling — `poll()` instead of `wait()`

If you don't want to block on `wait()` (e.g., you're rendering UI or running multiple jobs), poll explicitly:

```python
async with await client.video.run(...) as job:
    while True:
        status = await job.poll()                  # returns VideoRetrieveResponse
        if status.status == "COMPLETED":
            break
        await asyncio.sleep(2.0)
        # do other work between polls
    await job.download(path, status)
```

`poll()` is cheap (one GET per call). `wait()` is just a `while not done: await asyncio.sleep(poll_interval); await poll()` loop with progress hooks.

## Releasing storage with `cancel()`

Despite its name, `await job.cancel()` does **not** stop a generation that is still running: it wraps the `/complete` endpoint, which deletes the job's stored media and queue entry (best effort). The job keeps running and is billed. Call it once you have downloaded the result; the `async with` block does this for you on exit:

```python
job = await client.video.run(...)
try:
    status = await job.wait()
    await job.download(path, status)
finally:
    await job.cancel()           # release stored media
```

The SDK has no call that aborts a queued job, so check the price with `client.video.quote(...)` before submitting.

## Low-level: `submit()` + `retrieve()`

`run()` is sugar for `submit()` (returns a `VideoQueueResponse` carrying the `model` + `queue_id`) + polling via `retrieve(*, model=, queue_id=)`. Use the low-level path when:

- You want to persist the `model` + `queue_id` to a database and pick the job up in a different process / worker.
- You're building a queue manager that submits many jobs and polls them later.

```python
# Producer
queued = await client.video.submit(model=..., prompt=..., duration_seconds=5)  # -> VideoQueueResponse
db.save_pending_job(queued.model, queued.queue_id)

# Consumer (later, possibly different process)
status = await client.video.retrieve(          # keyword-only -> VideoRetrieveResponse
    model=saved_model,
    queue_id=saved_queue_id,
)
# poll status until complete, then fetch the output URL it carries
```

`retrieve()` is keyword-only (`model=` + `queue_id=`) and returns a `VideoRetrieveResponse` status object — it does **not** rebuild a `VideoJob`. Poll it until the job reports complete.

## Cost-quote-before-run

Both video and music expose `client.<resource>.quote(...)` to get a USD price estimate before launching. Always quote before expensive jobs:

```python
quote = await client.video.quote(
    model=await client.models.resolve_video(video_type="text-to-video"),
    duration_seconds=10,                          # quote takes NO prompt — just model + duration (+ optional resolution)
    resolution="1080p",
)
print(f"Estimated cost: ${quote.quote}")          # VideoQuoteResponse.quote (a number)
if float(quote.quote) > 0.50:
    raise SystemExit("over budget")

async with await client.video.run(...) as job:
    ...
```

For video specifically, `client.models.resolve_cheapest_video(...)` quotes all candidates and returns the cheapest model — cheaper than running quote yourself.

## Concurrency caveats

You can run multiple jobs concurrently — each gets its own server-side worker:

```python
results = await client.gather(
    [
        run_one(client, prompt) for prompt in prompts
    ],
    max_concurrency=3,                     # cap to avoid rate limits
)
```

But because each job already runs server-side asynchronously, the bottleneck is usually the server's per-account concurrency limit, not your client. Empirically 3-5 concurrent video jobs is a safe ceiling.

**Don't use `client.gather` to parallelize the `wait()` portion of a single job** — `wait()` is just polling, no parallelism gained. Spawn separate jobs.

## Common bugs

- **Bare `client.video.run(...)` without `async with`** — leaks server-side resources on early exit.
- **Calling `download()` without passing `status`** — `download(path, status)` is the signature; the status holds the output URL.
- **Forgetting that `wait()` defaults to `max_polls=120`** — raise it for long renders, e.g. `wait(max_polls=300, poll_interval=2.0)`.
- **Retrying `VideoGenerationError` blindly** — check `e.error_code` first; content-policy violations are terminal.
- **Treating a `MusicJob` like a `VideoJob` (or vice versa)** — they have the same shape but they're different classes; don't try to `pickle` and reconstruct cross-type.
- **Passing the original prompt to a retrieved job's `download()`** — the job's status (from `wait()` or `poll()`) is what carries the output URL, not the inputs.

## Related references

- `video.md` — text-to-video / image-to-video / upscale parameter details.
- `music.md` — music-specific parameters (genre, BPM, etc.).
- `image.md` — image generation is sync (no job lifecycle); contrast for context.
- `venice-py-production/references/error-taxonomy.md` — the full `VideoGenerationError` / `MusicGenerationError` taxonomy.
