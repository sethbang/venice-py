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
         ├── await job.cancel()       ├── await job.release()
         ├── await job.poll()         ├── await job.poll()
         └── job.queue_id             └── job.queue_id
```

`VideoJob` and `MusicJob` are different classes but expose the same async-context-manager + lifecycle methods, except that the call releasing stored media is `cancel()` on a `VideoJob` and `release()` on a `MusicJob`. The patterns below apply to both.

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

The `async with` block releases the job's stored media when the block finishes cleanly after the job has finished (you have saved the result). A clean exit before the job finished releases nothing and sends no request: releasing a job that is still generating deletes nothing, and its output is stored when it finishes anyway, so a WARNING names the `queue_id` and tells you to call `wait()` again, then `cancel()` / `release()`. If the block raises (a failed download, a disk error, a timeout, a cancelled task), the media is **not** released: the job is already billed, and leaving its output on the server lets you retry the save. A WARNING names the `queue_id` and the calls that resume it. A job that failed or was rejected has nothing to keep and is released either way. Without the block, nothing is ever released.

```python
# WRONG — the stored media is never released
job = await client.video.run(...)
status = await job.wait()
await job.download(path, status)

# RIGHT
async with await client.video.run(...) as job:   # released on a clean exit after the job finished
    status = await job.wait()
    await job.download(path, status)
```

To resume after an exception, retrieve the job by `model` and `queue_id` (`client.video.retrieve(...)`, `client.music.retrieve(...)`, `client.voice_changer.retrieve(...)`), save the output, then release it (`client.video.cancel(...)`, `client.music.release(...)`, `client.voice_changer.cancel(...)`). For a video job whose queue response carried a `download_url`, keep using the `VideoJob` (`wait()`, `download()`, `cancel()`): the file lives behind that link, and `client.video.cancel(...)` cannot delete it (see below). Call `await job.cancel()` (or `job.release()` for music) inside the block when you want the output discarded even though the block will raise.

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

`wait(max_polls=N)` raises `TimeoutError` once `N` polls (each `poll_interval` seconds apart) elapse without completion. Releasing could not help anyway: **a job that has not finished keeps generating on the server and is still billed**, and releasing it deletes nothing. Because the block exits with an exception, nothing is released, and the SDK logs a WARNING with the `queue_id`.

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

The same errors are raised when the retrieve endpoint rejects the job outright: a 400 for a job that failed server-side validation, or a 422 for one the provider refused (for example on content policy, with the credits refunded). The original `InvalidRequestError` / `UnprocessableEntityError` is on `e.__cause__`, and the job counts as finished, so leaving the `async with` block logs no "still billed" WARNING. A 400 saying the request ID is invalid, and 401/403/404/429, are not verdicts on the job and propagate as their own `APIError` subclasses.

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

Despite its name, `await job.cancel()` does **not** stop a generation that is still running: it deletes a finished job's stored video (best effort). A job that has not finished keeps running, is billed, and its video is stored when it finishes, so a `cancel()` sent before then deletes nothing. Call it once you have downloaded the result; the `async with` block does this for you on a clean exit after the job finished:

```python
job = await client.video.run(...)
status = await job.wait()
await job.download(path, status)
await job.cancel()               # release stored media only after the save succeeded
```

For most video models `cancel()` calls `/video/complete`. Some models return a `download_url` with the queue response instead: `retrieve()` then only ever returns JSON status, the file stays behind that link for up to 24 hours, and `/video/complete` answers 400 "Request ID is invalid" for the job whether it is running or finished. For those jobs `job.cancel()` sends `DELETE` to the `download_url` first, without the API key (the link authorizes itself; a 404 means it is already gone), then calls `/video/complete` and accepts that 400. The resource-level `client.video.cancel(model=..., queue_id=...)` cannot see the link and only calls `/video/complete`. The SDK never logs the link: anyone holding it can fetch the video.

The SDK has no call that aborts a queued job, so check the price with `client.video.quote(...)` before submitting.

Music jobs name the call for what it does: `await job.release()` (there is no `MusicJob.cancel()`). Leaving a `MusicJob` block cleanly releases a finished job; a job still generating is left alone with a WARNING, as for video. (A `MusicJob` polls once more on that exit first; a `VideoJob` does not, because a completed retrieve for an inline-video model carries the whole MP4.) Call `wait()` again (a `TimeoutError` leaves the job resumable), download, then `release()`. Once a job's media is released or has expired, `retrieve()` and `wait()` raise `NotFoundError` ("Media could not be found"); a `queue_id` Venice never issued gets a 400 "Request ID is invalid" (`InvalidRequestError`). A completed music status carries `content_type` and `audio_format` (`"flac"`, `"mp3"`, ...) for naming the file.

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

- **Bare `client.video.run(...)` without `async with`** — the finished job's stored media is never released.
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
