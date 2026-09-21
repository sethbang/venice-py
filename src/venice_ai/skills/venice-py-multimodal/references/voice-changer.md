# `client.voice_changer` — re-voice an existing recording

Voice changing takes audio in and gives audio back: the model re-voices what was already said, preserving the original timing. That is the whole reason the billing works the way it does — the **source** length is the billable quantity, not the output.

Reach for this when you have a recording and want it in a different voice. Reach for `client.audio.create_speech` when you have *text*, and `client.audio.create_voice` when you want to clone a voice for later TTS use.

## The shape

It is the async job family — `quote` / `run` / `wait` / `download` — the same as music and video.

```python
from venice_ai import VeniceClient

async with VeniceClient() as client:
    model = await client.models.resolve_voice_changer()   # never hardcode the ID

    quote = await client.voice_changer.quote(model=model, duration_seconds=30)
    print(quote.quote)          # USD estimate

    async with await client.voice_changer.run(model=model, file="source.mp3") as job:
        print(job.duration_seconds)      # server-measured — what you are billed
        status = await job.wait()
        await job.download("converted.mp3", status)
```

The context manager releases the provider-held media on exit. Pass `delete_media_on_completion=True` to `retrieve()` instead if you are driving the poll loop yourself.

## Source: file or URL, never both

```python
await client.voice_changer.run(model=model, file="source.mp3")                 # multipart upload
await client.voice_changer.run(model=model, audio_url="https://e.com/a.mp3")   # Venice fetches it
```

Passing both, or neither, raises a `ValidationError` before anything is uploaded. Venice fetches and validates an `audio_url` itself and forwards only the bytes to the provider — the URL is never handed onward.

## Gotchas that break working code

- **`duration_seconds` is a bare number.** `quote(duration_seconds=60)` or `"60"`. It is **not** video's `"60s"` form, and `"60s"` is rejected. The two families look alike and this is the easiest thing to get wrong by analogy.
- **There is no `FAILED` status.** `retrieve()` has exactly two outcomes: a `PROCESSING` JSON body, or the converted audio. A failed conversion raises an `APIError` (404 media gone, 500 inference failed, 504 never reached the provider) — so `wait()` never returns a failure for you to branch on, unlike `MusicJob`/`VideoJob`. Any refund is on the exception's parsed body, `err.body.get("credits_refunded")`, not a typed field.
- **There is no download URL.** The audio comes back inline as `audio/mpeg`, attached to `status.data`. `VoiceChangerCompletedStatus` has no `url` or `expires_at`; `download()` raises if the bytes are missing rather than trying to fetch.
- **Voice changing is not a model type.** These models report `type="music"` with `voice_changer: true` on `model_spec`. `resolve_music()` will return a music *generator*, which these endpoints reject with `The model \`X\` is not a voice-changer model`. Use `models.resolve_voice_changer()`.
- **The quote is an estimate.** Pricing is in whole-minute tiers against the length *Venice* measures server-side when the recording is queued, reported as `duration_seconds` on the queue response and on `job.duration_seconds`. Reconcile against that, not against what you asked to be quoted.
- **`job.progress` is a pacing hint, not real progress.** It is elapsed time over the model's recent average, clamped to 1.0 — a slower-than-average run sits at 100% while still converting.

## Model capabilities

Read these off `MusicModelSpec` before sending options the model does not take:

| Field | Meaning |
|---|---|
| `voice_changer` | This model re-voices rather than generates. Sent only when true. |
| `voices` / `default_voice` | Target voices. `voice` defaults to `default_voice`. |
| `supports_custom_voice_id` | `voice` may be a provider Voice ID, not just a member of `voices`. |
| `supports_background_noise_removal` | `remove_background_noise=True` is accepted. |
| `supports_seed` | `seed=` is accepted for reproducible output. |
| `accepted_audio_formats` | Source containers, e.g. `["mp3", "wav"]`. |
| `max_source_audio_duration_seconds` | Longest source the model will convert. |

The four booleans follow the `uncensored` convention rather than the `supports_*` one used elsewhere on this spec: the API sends them **only when true**, so an absent field is a definite "no", not "undeclared".

## CLI

```bash
venice-py audio voice-change recording.mp3 -o converted.mp3
venice-py audio voice-change recording.mp3 --voice Aria --remove-background-noise
venice-py audio voice-change --url https://example.com/clip.mp3 -o out.mp3
venice-py audio voice-change --quote-only --duration 90
```

## Availability

The endpoints are live, but the capability is not enabled on every account — a catalog with no `voice_changer` model means `resolve_voice_changer()` raises `ValueError`. Treat that as an entitlement condition to handle, not a bug. See `examples/audio/voice_changer.py`, which skips cleanly on it.
