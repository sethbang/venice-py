# Image generation

Sourced from `src/venice_ai/resources/image.py`. The image resource exposes 8 public methods: `create`, `submit`, `upscale`, `list_styles`, `edit`, `background_remove`, `multi_edit`, `simple_generate`. They follow the resolver-based model selection pattern, but return types differ:
- `create` → `ImageGenerationResponse | bytes` — this is the model with the `.save()` / `.save_all()` / `.bytes()` helpers.
- `edit`, `multi_edit`, `upscale`, `background_remove` → raw `bytes` — write them yourself; `.save()` does NOT apply.
- `submit` → `ImageJob` (parallel-render context manager), `list_styles` → `ImageStylesResponse`, `simple_generate` → `SimpleImageGenerationResponse` (OpenAI-compat).

## `create` — text-to-image

```python
response = await client.image.create(
    model=await client.models.resolve_image(),
    prompt="A neon-lit izakaya at midnight, photorealistic, shallow depth of field",
    width=1024,                              # int | None — pixels
    height=1024,                             # int | None
    num_images=1,                            # int | None — NOT `n=` (OpenAI muscle memory)
    style_preset=None,                       # str | None — see client.image.list_styles()
    aspect_ratio=None,                       # str | None — e.g. "16:9", "1:1"
    cfg_scale=None,                          # float | None — guidance scale
    embed_exif_metadata=None,                # bool | None
    enable_web_search=None,                  # bool | None
    format=None,                             # "jpeg" | "png" | "webp" | None — output format
    hide_watermark=None,                     # bool | None
    lora_strength=None,                      # int | None
    resolution=None,                         # str | None — "1K"/"2K"/"4K" tier on resolution-priced models
    quality=None,                            # "low"|"medium"|"high"|None — quality-aware models (GPT Image 2)
    return_binary=False,                     # bool — True returns raw bytes instead of response
    safe_mode=None,                          # bool | None
    seed=None,                               # int | None — for reproducibility
    steps=None,                              # int | None — diffusion steps
)

# Save with auto-detected extension
saved = response.save_all(Path("./out"), prefix="izakaya", overwrite=True)
```

Key parameters:
- **`num_images`** (NOT `n=`) — Venice uses `num_images`. OpenAI uses `n=`. The lint script flags `n=` if you leave it in.
- **`return_binary=True`** — returns `bytes` directly instead of `ImageGenerationResponse`. Useful for streaming or piping to another service without saving to disk.
- **`format`** — request a specific output format. Default lets the model decide; many fast models return WebP.
- **`style_preset`** — preset name from the live catalog. See `list_styles()`.

## Saving — auto-detect format

Don't assume PNG. Different models return different formats; `save_all` and `save` sniff the magic bytes:

```python
# CORRECT — let save_all() detect per-image
saved_paths = response.save_all(
    directory=Path("./out"),
    prefix="image",        # files become image_0.{ext}, image_1.{ext}, ...
    ext=None,              # None = auto-detect per image (default)
    overwrite=False,       # raises FileExistsError if any target exists
)

# Force a specific extension (only if you know the model)
saved_paths = response.save_all(directory=..., ext="png", overwrite=True)

# Single image
saved = response.save(Path("./out/image"), index=0, overwrite=True)
# Path was './out/image' (no extension); save() appends '.webp' or '.png' from magic bytes
```

`save()` and `save_all()` are sync; if the file write is slow (large batches), wrap with `asyncio.to_thread`:

```python
saved = await asyncio.to_thread(response.save, Path("./out/image"), overwrite=True)
```

## Quality tiers — discover and use

Quality-aware models (e.g. GPT Image 2) take `quality="low" | "medium" | "high"`
on `create()`/`submit()`. Discover support off the model spec before sending one:

```python
from venice_ai.types.api import ImageModelSpec

entry = await client.models.get(await client.models.resolve_image())
spec = entry.model_spec
if isinstance(spec, ImageModelSpec) and spec.constraints:
    print(spec.constraints.qualities)        # e.g. ["low", "medium", "high"] or None
    print(spec.constraints.defaultQuality)   # e.g. "high" or None
```

Both fields are `None` on models without quality tiers. Higher tiers can
increase the request charge. (This is distinct from the OpenAI-compat
`simple_generate(quality=...)` enum, which accepts `auto`/`hd`/`standard` too.)

## `edit` — image-to-image edit

```python
edited = await client.image.edit(
    prompt="Make the sky purple",            # str — required, keyword-only
    model="...",                             # str | None — a specific edit-capable model, or None for the API default
    image=open("input.png", "rb"),           # str | bytes | BinaryIO | Path
    aspect_ratio=None,                       # "1:1"/"16:9"/... — model-dependent
    safe_mode=None,                          # bool | None — None uses the server default
    resolution=None,                         # "1K"/"2K"/"4K" — resolution-priced models only
    timeout=None,                            # float seconds | aiohttp.ClientTimeout — raise for big edits
)
# edit() returns raw bytes — write them yourself.
Path("edited.png").write_bytes(edited)
```

`edit()` takes a single `prompt` and one `image`, and returns the edited image
as raw `bytes`. There is no mask, strength, seed, or steps parameter; the model
edits the whole image per the prompt.

`resolution` is honored only by models with resolution-based pricing; others
reject it with a `400`. Since the catalog exposes no per-model flag for it, the
robust pattern is to try with `resolution` and retry without on
`InvalidRequestError` whose message mentions resolution:

```python
from venice_ai.exceptions import InvalidRequestError

try:
    out = await client.image.edit(model=m, image=img, prompt=p, resolution="2K", timeout=180.0)
except InvalidRequestError as e:
    if "resolution" not in str(e).lower():
        raise
    out = await client.image.edit(model=m, image=img, prompt=p, timeout=180.0)
```

## `multi_edit` — multiple edits in one call

```python
image_bytes = await client.image.multi_edit(
    prompt="Composite these into one sunny scene with a rainbow",  # str — required; one prompt
    model="...",                             # str | None — optional
    image=open("a.png", "rb"),               # primary image
    image_2=open("b.png", "rb"),             # up to image_2 / image_3 additional inputs
    image_3=None,
    resolution=None,                         # "1K"/"2K"/"4K" — same semantics as edit()
    safe_mode=None,
)
# multi_edit takes a SINGLE prompt and up to three images (image / image_2 / image_3),
# NOT lists of images/prompts. It returns raw bytes.
Path("./out.png").write_bytes(image_bytes)
```

Useful for compositing several reference images under one editing prompt.

## `upscale` — increase resolution

```python
image_bytes = await client.image.upscale(
    image=open("input.png", "rb"),           # no `model` param — upscale has its own backend
    scale=2,                                  # 2 or 4 — 1 is rejected
    creativity=0.01,                          # detail/texture added; server clamps to 0–0.02
    timeout=None,                             # float seconds | aiohttp.ClientTimeout — raise for large images
)
Path("./upscaled.png").write_bytes(image_bytes)
```

`upscale()` takes **no `model` argument**, returns raw `bytes`, and uses `scale`
(not `upscale_factor`). The endpoint accepts exactly `image`, `scale` and
`creativity`. A higher `creativity` adds more detail and texture but strays
further from the source — good for thumbnails / hero images, bad for evidence /
forensics. Large source images and high scale factors take longer; raise
`timeout` to avoid a premature client-side abort.

## `list_styles` — discover available presets

```python
styles_resp = await client.image.list_styles()
for style in styles_resp.data:
    print(style)
```

`styles_resp.data` is a `list[str]`. Use any of these strings for `style_preset=`:

```python
response = await client.image.create(
    model=await client.models.resolve_image(),
    prompt="A cyberpunk cityscape",
    style_preset="anime",                    # from the catalog
)
```

## Background removal

Background removal IS a dedicated method — `client.image.background_remove()`. It takes either `image=` (file path / bytes / file-like / base64) or `image_url=` (an HTTP/HTTPS URL), and returns raw PNG `bytes` with a transparent background — write them yourself.

```python
result = await client.image.background_remove(image="photo.jpg")  # or image_url="https://..."
Path("no_bg.png").write_bytes(result)
```

For a runnable end-to-end demo, see `examples/image/background_removal.py` in the SDK.

## Batch generation

```python
responses = await client.gather(
    [
        client.image.create(
            model=await client.models.resolve_image(),
            prompt=prompt,
            width=1024,
            height=1024,
            num_images=1,
        )
        for prompt in prompts
    ],
    max_concurrency=2,                       # image gen is heavy; small cap
    return_exceptions=True,
)
for prompt, response in zip(prompts, responses):
    if isinstance(response, Exception):
        log.error("image_gen_failed", prompt=prompt, exc=str(response))
    else:
        response.save(Path(f"./out/{slug(prompt)}"))
```

`max_concurrency=2` is conservative; image generation has tighter per-account limits than chat. Validate against `response.response_rate_limits.remaining_requests` after the first batch.

## Cost estimation

Image cost is harder to predict than chat — model + size + steps all matter. The cheapest path is to read `response.balance_info.usd` after the first call to your model and extrapolate. For pre-call quotes, the SDK doesn't currently expose `client.image.quote(...)` (that's video / music only).

If you need a budget guard for image batches, set a `BudgetManager` on the client and pre-compute a conservative per-call estimate (e.g., $0.05).

## Common bugs

- **`n=` instead of `num_images=`** — OpenAI muscle memory. The Venice kwarg is `num_images`.
- **Hardcoded `.png` extension** when saving — different models return different formats. Let `.save()` / `.save_all()` auto-detect.
- **`await response.save(...)`** — `save()` is sync; the await raises `TypeError`. Wrap with `asyncio.to_thread` if you need async I/O.
- **`client.image.generate(...)`** — gone in v2. Use `create()`. Lint catches this.
- **`response.bytes(0)` then `open(...).write(...)`** — fine, but `response.save()` handles base64 decoding for you.
- **Calling `list_styles()` per generation** — cache the result; it's stable.

## Related references

- `job-lifecycle.md` — image generation is sync (no job); contrast with video / music.
- `venice-py/references/model-resolution.md` — `resolve_image()` and capability filters.
- `venice-py-production/references/rate-limiting.md` — image-resource rate limits are tighter; throttle accordingly.
