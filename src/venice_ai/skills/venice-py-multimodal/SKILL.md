---
name: venice-py-multimodal
description: Generate, edit, and download images, audio (TTS + STT), music, and video with the Venice AI Python SDK (v2+). Use this skill whenever the user wants to make a picture, illustration, transcript, voiceover, song, or video clip via Venice — including phrases like "generate an image with Venice", "remove the background", "upscale this", "transcribe audio", "text-to-speech", "make a song", "image-to-video", "queue a video job and wait for it". Covers v2 verb consolidation (`create`/`submit`/`run`/`cancel`), the `async with job:` lifecycle for async jobs (video, music), `.save()` / `.save_all()` / `.download()` patterns, voice and style options, and music being its own resource (`client.music`, not under `client.audio` anymore). Use this — not generic image-gen advice — because Venice has explicit job lifecycle, format constraints, and cost-quoting that other SDKs lack. For chat/vision-as-input, use the `venice-py` skill instead.
---

# Venice AI multimodal: image, audio, video, music

> _Unofficial, community-maintained — not affiliated with or endorsed by Venice AI._

This skill covers the **generation** side: producing images, speech, transcripts, music, and video with Venice. For chat and vision-as-input (multimodal chat), see the `venice-py` skill.

The cardinal rule from `venice-py` carries over: **never hardcode model IDs**. Resolve via `client.models.resolve_image()`, `resolve_tts()`, `resolve_asr()`, `resolve_music()`, `resolve_video()`, or `resolve_cheapest_video()`.

## Shared chassis — same four steps every modality

1. **Resolve** the model dynamically.
2. **Configure** output (size, voice, format, duration, style preset, quality).
3. **Submit** the request — sync `create()` for fast modalities, async `run()` / `submit()` for jobs.
4. **Save** the asset — typed responses provide `.save()` / `.save_all()` / `.download()`; never assume an extension.

## Image — `client.image`

```python
from pathlib import Path
from venice_ai import VeniceClient

async with VeniceClient() as client:
    response = await client.image.create(
        model=await client.models.resolve_image(),
        prompt="A neon-lit izakaya at midnight, photorealistic, shallow depth of field",
        width=1024,
        height=1024,
        num_images=1,                    # NOT n= — Venice uses num_images
        style_preset="photographic",     # see client.image.list_styles()
    )
    saved = response.save_all(Path("./out"), prefix="izakaya", overwrite=True)
    # saved is List[Path]; extension auto-detected from each image's magic bytes (don't assume .png)
```

**Don't hardcode `.png`.** Some image models (e.g., turbo variants) return WebP. `save()` and `save_all()` sniff magic bytes and write the correct extension. Passing `ext=None` (default) does the right thing.

Other image methods:
- `client.image.upscale(image=..., scale=..., creativity=...)` — no `model` param
- `client.image.edit(prompt=..., model=..., image=...)` — returns raw bytes
- `client.image.multi_edit(prompt=..., model=..., image=..., image_2=..., image_3=..., ...)`
- `client.image.list_styles()` → `ImageStylesResponse` (preset catalog)
- Background removal, batch generation, style variants — see `examples/image/`.

**Parallel rendering with `client.image.submit()`** — for symmetry with music
and video, `client.image.submit(...)` returns an `ImageJob` async context
manager. Image generation is synchronous server-side (no queue / poll
endpoint), so the actual HTTP call fires lazily inside `await job.wait()`.
Use this when you want to render N images in parallel under bounded
concurrency:

```python
async with VeniceClient() as client:
    model = await client.models.resolve_image()
    jobs = [await client.image.submit(model=model, prompt=p) for p in prompts]
    async with contextlib.AsyncExitStack() as stack:
        for j in jobs:
            await stack.enter_async_context(j)
        results = await client.gather([j.wait() for j in jobs], max_concurrency=4)
```

For one-off rendering, `await client.image.create(...)` is shorter — both
paths share the same kwargs, so swapping is mechanical.

**Quality tiers** — quality-aware models (e.g. GPT Image 2) accept
`quality="low" | "medium" | "high"` on `create()`/`submit()`, trading cost and
render time against fidelity. Discover support before sending one: the image
model spec exposes `constraints.qualities` (the supported tiers) and
`constraints.defaultQuality`. Models without quality tiers leave both `None`.

```python
from venice_ai.types.api import ImageModelSpec

async with VeniceClient() as client:
    model = await client.models.resolve_image()
    entry = await client.models.get(model)
    spec = entry.model_spec
    tiers = spec.constraints.qualities if isinstance(spec, ImageModelSpec) and spec.constraints else None
    response = await client.image.create(
        model=model,
        prompt="A ripe pomegranate on marble, soft window light",
        quality=(tiers[-1] if tiers else None),   # highest available tier, or None
    )
```

**Resolution tier + timeout** — `edit()` and `multi_edit()` accept
`resolution="1K" | "2K" | "4K"` for models with resolution-based pricing (other
models reject it with a 400 — catch `InvalidRequestError` and retry without).
`edit()` and `upscale()` also take `timeout=` (a float in seconds or an
`aiohttp.ClientTimeout`) — raise it for large/high-res jobs that outlast the
default.

## Audio — TTS via `client.audio.create_speech`

```python
from venice_ai.types.enums import Voice, ResponseFormat   # NOT venice_ai.types.api

response = await client.audio.create_speech(
    model=await client.models.resolve_tts(),
    input="Welcome to Acme — how can I help today?",
    voice=Voice.AF_ALLOY,                      # see client.audio.get_voices(model_id=...) for the live catalog
    response_format=ResponseFormat.MP3,        # MP3 / AAC / OPUS / FLAC / WAV / PCM
    speed=1.0,
)
response.save(Path("greeting.mp3"), overwrite=True)
```

`get_voices(model_id=...)` returns the voice catalog for a given TTS model — voices vary by model.

## Audio — STT via `client.audio.transcribe`

```python
response = await client.audio.transcribe(
    model=await client.models.resolve_asr(),
    file=open("interview.mp3", "rb"),          # NOTE: kwarg is `file=`, NOT `audio=`
    language="en",
)
print(response.text)
```

The kwarg is `file=` (matching OpenAI's parameter name), not `audio=`. Common gotcha.

## Audio — voice cloning via `client.audio.create_voice`

Clone a voice from a short sample, then synthesize with the returned handle:

```python
async with VeniceClient() as client:
    # Omit model to let the API pick its default and report it back on .model.
    voice = await client.audio.create_voice(file="sample.wav")   # str | bytes | BinaryIO | Path
    audio = await client.audio.create_speech(
        input="Hello in my cloned voice.",
        model=voice.model,        # MUST pair the handle with its own model
        voice=voice.id,           # the vv_<id> handle, as a plain string
    )
    audio.save(Path("cloned.mp3"), overwrite=True)
```

`create_voice` returns a `ClonedVoice` with `.id` (a `vv_<id>` handle) and
`.model`. **Pair the handle with the same model it was created for** — that's
why you read `voice.model` back rather than resolving a TTS model separately.
A clean 5–10s clip works best; accepted containers depend on the model
(`tts-chatterbox-hd`: MP3/WAV/FLAC/M4A; `tts-minimax-speech-02-hd`: MP3/WAV).
Handles expire after the per-model retention window (~7 days). Voice cloning is
a gated capability — accounts without access get a `403`
(`PermissionDeniedError`); handle it gracefully.

## Music — `client.music`

Music is its own resource in v2 (`client.music`), not under `client.audio`. There is no `client.audio.generate_music(...)` — use `client.music.run(...)`.

```python
async with VeniceClient() as client:
    # Cost-quote before launching expensive jobs
    quote = await client.music.quote(
        model=await client.models.resolve_music(),
        duration_seconds=30,           # quote takes NO prompt — just model + duration
    )
    print(f"Estimated cost: ${quote.quote}")   # MusicQuoteResponse.quote (a number)

    async with await client.music.run(
        model=await client.models.resolve_music(),
        prompt="Lo-fi beats, 90 BPM, mellow vibes",
        duration_seconds=30,
    ) as job:
        status = await job.wait(on_progress=lambda s: print(f"\r{s.progress_percent:.0f}%", end=""))
        await job.download(Path("track.mp3"), status)
```

The `async with job:` block is **mandatory** — it guarantees server-side cleanup if your code exits early. `await job.wait()` polls until completion or timeout; `job.download()` writes the asset.

**Why both `async` keywords?** `client.music.run(...)` is an async function (it submits the queue request), and its return value `MusicJob` is itself an async context manager (it owns the server-side cleanup). The `await` resolves the coroutine; the `async with` then enters the context manager. So the full shape is `async with await client.music.run(...) as job:`. The same pattern applies to `client.video.run(...)` and `client.image.submit(...)`. If you only see one keyword, you're missing one — `async with client.music.run(...)` won't work because you're trying to enter a coroutine, and `await client.music.run(...) as job` is a syntax error.

For lower-level control:
- `client.music.submit(...)` returns a `MusicQueueResponse` (with `.model` / `.queue_id` / `.status`) — no lifecycle manager
- `client.music.retrieve(*, model=..., queue_id=...)` returns a `MusicRetrieveResponse` status object (keyword-only; no positional `job_id`)

## Video — `client.video`

Same lifecycle pattern as music. `client.video.run(...)` returns a `VideoJob`; use `async with`.

```python
# Find the cheapest model for the duration/resolution we need
best = await client.models.resolve_cheapest_video(
    duration="5s",
    resolution="1080p",
    video_type="text-to-video",
)
print(f"Picked {best.model} @ ${best.quote_usd}")

async with await client.video.run(
    model=best.model,
    prompt="A jellyfish drifting through neon kelp",
    duration_seconds=5,           # int seconds; "5s" / "5 seconds" also accepted
) as job:
    status = await job.wait(on_progress=lambda s: print(f"\r{s.progress_percent:.0f}%", end=""))
    await job.download(Path("clip.mp4"), status)
```

**Duration shape** — both `client.music.run()` and `client.video.run()` take
`duration_seconds: int | str` as of 2.0. Liberal parsing: `5`, `"5"`, `"5s"`,
and `"5 seconds"` all coerce to the integer 5 internally. The wire form
`"5s"` is generated for you. Per-model enums (e.g. `ace-step-15` only
accepts `[60, 90, 120, 150, 180, 210]`) are pre-validated client-side
against `spec.duration_options` when the catalog is reachable; otherwise
the server is the backstop.

Other patterns:
- **Image-to-video**: pass `image_url="https://..."` (a public or `data:` URL string) to `client.video.run()`.
- **Upscale**: `client.video.run(model=..., video_url=..., upscale_factor=2)` (model-dependent).
- **Reference fields** (`run()`/`submit()`): `reference_image_urls` (≤9, style/identity),
  `reference_audio_urls` (≤3, R2V audio donors for vocal timbre/narration/SFX on
  Seedance 2.0 R2V models — each 2–15s `.wav`/`.mp3`, **must be paired with a
  reference image/video**), `end_image_url` (transitions), and `elements` /
  `scene_image_urls` (element-aware models). Each accepts a public URL or a `data:` URL.
- **Cancel a running job**: `await client.video.cancel(model=..., queue_id=...)` (keyword-only).
- `client.video.quote(...)` returns USD cost before launching.

## Cost-quote-before-run

Music and video jobs can be expensive — quote first when budgets matter:

```python
quote = await client.video.quote(
    model=await client.models.resolve_video(video_type="text-to-video"),
    duration_seconds=10,           # quote takes NO prompt — just model + duration (+ optional resolution)
    resolution="1080p",
)
if quote.quote > BUDGET_USD:
    raise ValueError(f"Job would cost ${quote.quote}, over budget")
```

## VideoGenerationError / MusicGenerationError

Server-side job failures raise the typed exceptions with `.error_code`:

```python
from venice_ai.exceptions import VideoGenerationError

try:
    async with await client.video.run(...) as job:
        status = await job.wait()
        await job.download(path, status)
except VideoGenerationError as e:
    log.error(f"Video failed: {e.error_code} — {e}")
```

## Pitfalls AI assistants reliably get wrong

1. **Calling `client.audio.generate_music(...)`** — not a v2 method; music is its own resource — use `client.music.run(...)`.
2. **Skipping `async with job:`** — leaks server-side resources on early exit. Always wrap the job.
3. **Hardcoding `.png` extension** — `.save()` / `.save_all()` auto-detect format; let them.
4. **Treating `.save()` like `.write_bytes()`** — it returns `Path` (or `List[Path]` for `save_all`), not bytes.
5. **Mixing aspect ratios in image-to-video upscale** — input frame aspect must match the requested output; otherwise the model produces letterboxing.
6. **Forgetting that voices are per-model** — query `client.audio.get_voices(model_id=...)` rather than picking a name from memory.
7. **Hardcoding model IDs** — same rule as core; `resolve_image()` / `resolve_tts()` / etc. always.
8. **Calling `.queue()` instead of `.run()` / `.submit()`** — verb consolidated in v2.

## References

- `references/image.md` — generation, editing, multi-edit, upscaling, background removal, batch, style variants
- `references/audio-tts-stt.md` — voice catalog, formats, streaming TTS, STT options
- `references/music.md` — v2 separation rationale, job lifecycle, quote-before-run
- `references/video.md` — text-to-video, image-to-video, upscale, advanced fields
- `references/job-lifecycle.md` — the unified `async with job:` pattern shared by music + video

## Examples to read

Paths below are relative to the SDK repo's `examples/` directory.

- `image/text_to_image.py`, `image/image_upscaling.py`, `image/image_editing.py`, `image/quality_control.py`, `image/background_removal.py`, `image/style_variants.py`, `image/batch_generation.py`, `image/multi_edit.py`
- `audio/text_to_speech.py`, `audio/speech_to_text.py`, `audio/voice_cloning.py`, `audio/voice_options.py`
- `music/music_generation.py`
- `video/text_to_video.py`, `video/image_to_video.py`, `video/upscale.py`, `video/advanced_fields.py`
