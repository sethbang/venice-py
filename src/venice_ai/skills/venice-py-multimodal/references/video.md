# Video generation

Sourced from `src/venice_ai/resources/video.py`. Async-job lifecycle (see `job-lifecycle.md` for shared patterns); rich parameter surface covering text-to-video, image-to-video, upscaling, advanced fields, and even video transcription.

## The high-level pattern

```python
import asyncio
from pathlib import Path
from venice_ai import VeniceClient


async def make_clip(prompt: str, out_path: Path) -> Path:
    async with VeniceClient() as client:
        async with await client.video.run(           # NOTE: `await` BEFORE `async with`
            model=await client.models.resolve_video(
                video_type="text-to-video",
                require_duration="5s",                # model must offer exactly this length
                require_resolution="1080p",           # ...and this resolution tier
            ),
            prompt=prompt,
            duration_seconds="5s",                            # str — "5s", "10s", etc.
            resolution="1080p",                       # str | None
            aspect_ratio="16:9",                      # str | None
        ) as job:
            status = await job.wait(
                on_progress=lambda s: print(f"\r{s.progress_percent:.0f}%", end=""),
                max_polls=300,
                poll_interval=2.0,
            )
            return await job.download(out_path, status)
```

`client.video.run(...)` is `async def` and returns a `VideoJob`. The form is `async with await client.video.run(...) as job:` — drop the `await` and you'll be awaiting a coroutine in `async with`, which raises `TypeError`.

## Full parameter surface

```python
await client.video.run(
    model=...,                                  # str — required
    prompt=...,                                 # str | None — required by generation models; upscale takes none
    duration_seconds="5s",                      # int | str | None — required by generation models (5, "5", "5s", "5 seconds"); omit for upscale / video-to-video
    negative_prompt=None,                       # str | None — what to avoid
    resolution=None,                            # str | None — "720p", "1080p", "4k"
    audio=None,                                 # bool | None — include audio track
    aspect_ratio=None,                          # str | None — "16:9", "9:16", "1:1"
    image_url=None,                             # str | None — seed frame for image-to-video
    upscale_factor=None,                        # 1 | 2 | 4 | None — for upscale models
    end_image_url=None,                         # str | None — target end frame (interpolation)
    audio_url=None,                             # str | None — source audio to sync to
    video_url=None,                             # str | None — input video for video-to-video
    reference_image_urls=None,                  # list[str] | None — ≤9 style/identity references
    reference_audio_urls=None,                  # list[str] | None — ≤3 R2V audio donors (Seedance 2.0)
    elements=None,                              # list[VideoElement] | None — scene composition
    scene_image_urls=None,                      # list[str] | None — multi-shot composition
)
```

**`duration_seconds`, not `duration`.** The kwarg is `duration_seconds` (int or
str); `duration="5s"` raises `TypeError: unexpected keyword argument 'duration'`.
Leave it out for upscale and other video-to-video models (their catalog
`durations` is `["Auto"]`): the length comes from `video_url` and no duration is
sent. For a generation model that lists numeric durations, leaving it out raises
`ValueError` before the request is sent.

Most parameters are model-dependent — `seedance-1-5-pro-image-to-video` accepts `image_url`, `seedream-1-pro-text-to-video` accepts `prompt`, an upscale model accepts `video_url` + `upscale_factor`. Check the model's metadata via `client.models.list(type="video")` to see what each accepts.

## Text-to-video

```python
async with await client.video.run(
    model=await client.models.resolve_video(
        video_type="text-to-video", require_duration="5s", require_resolution="1080p"
    ),
    prompt="A jellyfish drifting through neon kelp at midnight, photorealistic",
    duration_seconds="5s",
    resolution="1080p",
) as job:
    status = await job.wait()
    await job.download(Path("clip.mp4"), status)
```

## Image-to-video

```python
# 1. Generate or have a seed image
image_resp = await client.image.create(
    model=await client.models.resolve_image(require_custom_size=True),  # takes width/height
    prompt="A jellyfish drifting through neon kelp",
    width=1024, height=1024,
)
seed_path = image_resp.save(Path("./seed"), overwrite=True)

# 2. Encode the local image as a data: URL (Venice accepts data URIs for image_url)
import base64
image_bytes = seed_path.read_bytes()
ext = seed_path.suffix.lstrip(".")
data_uri = f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode()}"

# 3. Feed it to image-to-video
async with await client.video.run(
    model=await client.models.resolve_video(video_type="image-to-video"),
    prompt="The jellyfish slowly turns and drifts to the left",
    duration_seconds="5s",
    image_url=data_uri,
) as job:
    status = await job.wait()
    await job.download(Path("clip.mp4"), status)
```

`video_type="image-to-video"` resolves a plain image-to-video model, one that animates the start image. Venice types reference-to-video (R2V), transition, first/last-frame and multi-angle models `image-to-video` too, and they need other inputs; ask for them with `input_mode="reference"`, `"transition"`, `"first_last_frame"` or `"multi_angle"` on `resolve_video()` / `resolve_cheapest_video()`.

You can also pass a public HTTPS URL for `image_url` if the image is hosted somewhere Venice can reach.

## Upscaling existing video

```python
async with await client.video.run(
    model=await client.models.resolve_video_upscale(),
    video_url="https://example.com/source.mp4",   # or a data: URL
    upscale_factor=2,                              # 2× or 4×
) as job:
    status = await job.wait()
    await job.download(Path("upscaled.mp4"), status)
```

## End-image / interpolation

For "morph from frame A to frame B" effects:

```python
async with await client.video.run(
    model="...",                                   # check the model supports end_image_url
    image_url=start_frame_data_uri,
    end_image_url=end_frame_data_uri,
    duration_seconds="5s",
    prompt="Smooth transition between frames",
) as job:
    ...
```

## Cost-quote BEFORE running

Video is the most expensive Venice modality. Always quote first:

```python
quote = await client.video.quote(
    model=await client.models.resolve_video(video_type="text-to-video"),
    duration_seconds="10s",                       # quote takes NO prompt — just model + duration (+ optional resolution)
    resolution="1080p",
)
print(f"Estimated cost: ${quote.quote}")          # VideoQuoteResponse.quote (a number)
if float(quote.quote) > MY_BUDGET:
    raise SystemExit("over budget")
```

For automatic cheapest-model selection across all candidates:

```python
result = await client.models.resolve_cheapest_video(
    video_type="text-to-video",
    exclude_beta=True,
)
print(f"Cheapest: {result.model} at ${result.quote_usd}")
print(f"Quoted with: {result.request_params}")     # duration_seconds, resolution, aspect_ratio, audio
job = await client.video.submit(model=result.model, prompt="...", **result.request_params)
```

Each candidate is quoted at its own cheapest valid request (shortest listed duration, lowest listed resolution, 16:9 when listed, `audio=False` where configurable), so models without a 5-second clip are still compared. Pass `duration=` or `resolution=` to pin one; models that don't list it are skipped. `result.all_quotes` maps every quoted model to its USD price and `result.skipped` says why any candidate was left out. `cheapest_video_params(constraints)` from `venice_ai.models.selection` builds the same request for a model you picked yourself.

`resolve_cheapest_video` issues one quote call per candidate (at most 8 at once). Quotes are free, but cache the result rather than calling it in a loop.

## Multi-shot composition (`elements` and `scene_image_urls`)

For scene-by-scene composition, advanced models accept structured `elements`:

```python
from venice_ai.types.api.requests.video import VideoElement

# Each element is a character/object you reference in the prompt as @Element1, @Element2, …
elements = [
    VideoElement(
        frontal_image_url="https://.../hero_front.png",          # URL or data: URI
        reference_image_urls=["https://.../hero_side.png"],      # optional extra references
    ),
    VideoElement(frontal_image_url="https://.../prop.png"),
]

async with await client.video.run(
    model="...",
    prompt="@Element1 picks up @Element2 in a cinematic tracking shot",
    duration_seconds="5s",
    elements=elements,
    aspect_ratio="16:9",                                          # element-aware models require it
) as job:
    ...
```

Each `VideoElement` takes exactly one kind of media source: images
(`frontal_image_url` and/or up to 3 `reference_image_urls`) or a single reference
`video_url` — never both, and at least one. The fields are not `type`/`url`/`duration`. You can also pass `elements` as plain
dicts. This is model-specific — many video models don't accept `elements` (it's
for Kling O3 R2V and similar). Plain dicts also work; see `examples/video/advanced_fields.py`.

## Reference-image style transfer

Some models accept reference images for style:

```python
async with await client.video.run(
    model="...",
    prompt="A jellyfish in this artistic style",
    duration_seconds="5s",
    reference_image_urls=["https://.../style_reference.png"],
) as job:
    ...
```

## Reference audio (R2V — Seedance 2.0)

Reference-to-video models (e.g. `seedance-2-0-reference-to-video`,
`seedance-2-0-fast-reference-to-video`) accept up to 3 `reference_audio_urls` —
donor clips for vocal timbre / narration / SFX. Each clip is 2–15s,
`.wav`/`.mp3` (aggregate ≤15s), supplied as a public URL or a `data:` URL, and
**must be paired with at least one reference image or video**:

```python
async with await client.video.run(
    model="...",                                   # a *-reference-to-video model
    prompt="A narrator's voice over a slow pan across a misty valley",
    duration_seconds="5s",
    image_url="data:image/png;base64,...",         # the required paired reference
    reference_image_urls=["data:image/png;base64,..."],
    reference_audio_urls=["data:audio/wav;base64,..."],
    aspect_ratio="16:9",
) as job:
    ...
```

`reference_audio_urls` is wired through both `run()` and `submit()`. See
`examples/video/advanced_fields.py` for a runnable queue-and-cleanup demo.

## Releasing storage (`cancel`)

`job.cancel()` deletes a finished job's stored video (best effort). It does
**not** stop a generation that is still running: that job keeps going on the
server, is billed, and its video is stored when it finishes, so a call made
before then deletes nothing.

What it deletes depends on the model:

- **Inline or `url` models**: it calls `/video/complete`, which deletes the
  stored media and queue entry.
- **Models whose queue response carries a `download_url`**: the file lives
  behind that link for up to 24 hours, and `/video/complete` answers 400
  "Request ID is invalid" for the job whether it is running or finished.
  `job.cancel()` first sends `DELETE` to the `download_url` without the API key
  (the link authorizes itself; a 404 means it is already gone), then calls
  `/video/complete` and accepts that 400. The link is never logged.

The resource-level `client.video.cancel(model=..., queue_id=...)` only calls
`/video/complete`, so for a `download_url` job either use `job.cancel()` or
send the `DELETE` to the link yourself.

The `async with` block calls `job.cancel()` on a clean exit once the job has
finished. Leaving cleanly before that sends nothing and logs a WARNING naming
the `queue_id`: call `wait()` again, save the video, then `cancel()`. If the
block raises, it releases nothing (so the video can still be saved) and the
WARNING names the `queue_id`. A failed or rejected job is released either way.

```python
async with await client.video.run(...) as job:
    status = await job.wait()
    await job.download(...)
# stored media released here
```

## Video transcription

Distinct from job-based generation — synchronous for hosted URLs:

```python
result = await client.video.transcribe(
    "https://example.com/video.mp4",
    response_format="json",
)
print(result.transcript)
```

`response_format` accepts `"json"` (the default; returns `VideoTranscriptionResponse`) or `"text"` (returns plain string). Useful for caption generation / search indexing.

## Concurrent video jobs

The server-side queue depth is limited (typically 2-3 concurrent jobs per account). Don't fan out 20 jobs and expect them all to start immediately — they queue:

```python
results = await client.gather(
    [make_clip(client, prompt, Path(f"./out/{i}.mp4")) for i, prompt in enumerate(prompts)],
    max_concurrency=3,                             # match the server's queue depth
    return_exceptions=True,
)
```

## Pre-flight validation

Many video failures are content-policy violations or invalid prompts. Pre-flight by:

1. Quoting first (`client.video.quote`) — catches "model doesn't accept these params".
2. Catching `VideoGenerationError(error_code=...)` — surface terminal failures, retry transients.
3. Logging the prompt + model at submit time so you can correlate when something fails.

```python
from venice_ai.exceptions import VideoGenerationError

try:
    async with await client.video.run(...) as job:
        status = await job.wait()
        await job.download(out_path, status)
except VideoGenerationError as e:
    if e.error_code in ("CONTENT_POLICY_VIOLATION", "INVALID_PROMPT"):
        log.error("video.terminal", error_code=e.error_code)
        raise
    log.warning("video.transient", error_code=e.error_code)
    # retry with same prompt
```

## Common bugs

- **`async with client.video.run(...)` (no `await`)** — TypeError. Use `async with await client.video.run(...)`.
- **`client.video.queue(...)` / `client.video.complete(...)`** — not v2 methods; use `client.video.submit(...)` / `run(...)` and `cancel(...)`. (Lint V104, V105.)
- **Skipping `client.video.quote(...)` and getting hit with surprise costs** — video is expensive; always quote.
- **Hardcoded `image_url="https://..."` from a private host** — Venice's render workers must be able to reach it. Use a data URI for local files.
- **Relying on the default `max_polls=120` (~10 min at `poll_interval=5.0`)** — for long renders raise it explicitly, e.g. `wait(max_polls=300, poll_interval=2.0)`.
- **Retrying `VideoGenerationError` blindly** — inspect `e.error_code` first.

## Related references

- `job-lifecycle.md` — the `async with await ... as job:` pattern shared with music.
- `music.md` — sister modality with the same lifecycle.
- `image.md` — image generation as input to image-to-video.
- `venice-py/references/model-resolution.md` — `resolve_video()`, `resolve_video_upscale()`, `resolve_cheapest_video()`.
- `venice-py-production/references/error-taxonomy.md` — `VideoGenerationError.error_code` semantics.
