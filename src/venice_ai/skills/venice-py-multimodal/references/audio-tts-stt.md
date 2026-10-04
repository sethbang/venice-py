# Audio: TTS and STT

Sourced from `src/venice_ai/resources/audio.py`. The audio resource has three primary methods: `create_speech` (TTS), `transcribe` (STT), and `get_voices` (catalog). **Music is NOT here in v2** — it's its own resource, `client.music.*` (see `music.md`).

## TTS — `create_speech`

```python
from pathlib import Path
from venice_ai import VeniceClient
from venice_ai.types.enums import Voice, ResponseFormat   # NOTE: types.enums, not types.api


async def speak(text: str, out_path: Path) -> Path:
    async with VeniceClient() as client:
        response = await client.audio.create_speech(
            model=await client.models.resolve_tts(),
            input=text,                          # the text to speak
            voice=Voice.AF_ALLOY,                # or a string from get_voices(model_id=...)
            response_format=ResponseFormat.MP3,  # or "mp3" / "wav" / "flac" / "aac" / "opus" / "pcm"
            speed=1.0,                            # 0.25 to 4.0; default 1.0
            language=None,                        # str | None — model-specific (e.g. "en", "ja", or "English")
            prompt=None,                          # str | None — emotion / style hint (Qwen 3 only)
            temperature=None,                     # float | None — sampling temp (Qwen 3 / Orpheus / Chatterbox HD)
            top_p=None,                           # float | None
            stream=False,                         # bool — see "Streaming TTS" below
        )
        return response.save(out_path, overwrite=True)
```

Returns an `AudioResponse` (binary content via `.content`, with `.save(path)`, `.iter_bytes()`, `.startswith(...)` for magic-byte checks).

### Voices

The voice catalog is per-model. Always query the live list:

```python
voices = await client.audio.get_voices(model_id=await client.models.resolve_tts())
for voice in voices.data:
    print(voice.id, voice.gender, voice.region_code, voice.language, voice.accent)
```

For convenience, the `Voice` enum (in `venice_ai.types.enums`) has named constants matching the catalog (e.g., `Voice.AF_ALLOY = "af_alloy"`). Naming convention:
- `af_*` American female, `am_*` American male
- `bf_*` British female, `bm_*` British male
- `zf_*` / `zm_*` Chinese (Mandarin)
- `ff_*` French female, `hf_*` Hindi female, `if_*` Italian female, `jf_*` Japanese female, `pf_*` Portuguese female, `ef_*` Spanish female (and `*m_*` male variants)

Pass either the enum or the raw string.

### Response format

`ResponseFormat` enum (also in `venice_ai.types.enums`):

| Value | Container | Use case |
|---|---|---|
| `MP3` | MP3 (lossy) | Web, broad compatibility (default) |
| `AAC` | AAC (lossy, better than MP3 at same bitrate) | Mobile |
| `OPUS` | OPUS (lossy, low-latency) | Real-time streaming |
| `FLAC` | FLAC (lossless) | Archival |
| `WAV` | WAV (uncompressed PCM) | Maximum quality, large |
| `PCM` | Raw PCM | Audio processing pipelines |

### Streaming TTS

For long inputs where you want to start playback before the whole audio is ready:

```python
response = await client.audio.create_speech(
    model=...,
    input=long_text,
    voice=Voice.AF_ALLOY,
    response_format=ResponseFormat.MP3,
    stream=True,                                 # returns AsyncIterator[bytes]
)
async for chunk in response:                     # yields raw audio bytes
    audio_player.write(chunk)
```

When `stream=True`, the return type changes to `AsyncIterator[bytes]` instead of `AudioResponse` — note your type hints if you switch dynamically.

### Saving

`AudioResponse.save(path, *, overwrite=False)` writes the binary content to disk. Sync method; wrap in `asyncio.to_thread` for large files in async contexts.

```python
saved_path = response.save(Path("greeting.mp3"), overwrite=True)
```

If you want bytes in memory:

```python
audio_bytes = response.content        # full payload as bytes
# or stream chunks:
for chunk in response.iter_bytes():
    ...
```

## STT — `transcribe`

```python
async def transcribe_audio(audio_path: Path) -> str:
    async with VeniceClient() as client:
        result = await client.audio.transcribe(
            file=open(audio_path, "rb"),         # NOTE: kwarg is `file=`, NOT `audio=`
            model=await client.models.resolve_asr(),
            response_format=None,                # str | None — e.g. "json" (default)
            timestamps=False,                    # bool | None — word-level timestamps
            language=None,                       # str | None — e.g. "en", "ja"
        )
        return result.text                       # str — the transcribed text
```

### `file=` not `audio=`

A common mistake is using `audio=`. The kwarg matches OpenAI's parameter name (`file`), NOT a generic `audio` name. The lint script's V107 rule catches `audio=`.

### Accepted file types

`file` accepts `str | bytes | BinaryIO | Path`:

```python
# Path or string path
result = await client.audio.transcribe(file="recording.mp3", model=...)

# File handle
with open("recording.wav", "rb") as f:
    result = await client.audio.transcribe(file=f, model=...)

# Raw bytes
audio_bytes = b"\x00..."
result = await client.audio.transcribe(file=audio_bytes, model=...)
```

Supported audio formats: WAV, FLAC, MP3, M4A, AAC, MP4 (audio track).

### Word-level timestamps

```python
result = await client.audio.transcribe(file="meeting.mp3", model=..., timestamps=True)
print(result.text)
if result.words:
    for word in result.words:
        print(f"{word.start:.2f}s - {word.end:.2f}s: {word.word}")
```

Useful for video captions, transcript indexing, speaker-diarization preprocessing.

### Language hint

```python
result = await client.audio.transcribe(file="japanese.mp3", model=..., language="ja")
```

Helps with code-switching audio or under-represented languages where auto-detect underperforms. Omit to let the model auto-detect.

## Voice cloning — `create_voice`

Clone a voice from a short sample, then synthesize with the returned handle.

```python
async with VeniceClient() as client:
    # file accepts a path (str/Path), raw bytes, or a binary file-like object.
    # Omit `model` to let the API pick its default and report it on .model.
    voice = await client.audio.create_voice(file="sample.wav")
    audio = await client.audio.create_speech(
        input="Hello in my cloned voice.",
        model=voice.model,         # pair the handle with the SAME model it was made for
        voice=voice.id,            # the vv_<id> handle, as a plain string
    )
    audio.save(Path("./out/cloned.mp3"), overwrite=True)
```

`create_voice` → `POST /v1/audio/voices`, returning a `ClonedVoice`:

| Attr | Meaning |
|---|---|
| `.id` | `vv_<id>` handle — pass as `voice=` to `create_speech` |
| `.model` | the TTS model the handle is bound to — pass as `model=` |

Notes:
- **Pair `voice.id` with `voice.model`.** A handle only works with the model it
  was created for, so read `voice.model` back rather than resolving separately.
- A clean **5–10s** speech clip works best. Accepted containers are model-specific:
  `tts-chatterbox-hd` → MP3/WAV/FLAC/M4A; `tts-minimax-speech-02-hd` → MP3/WAV.
- Handles expire after the per-model retention window (~7 days);
  `tts-minimax-speech-02-hd` resets the clock on each successful TTS call.
- Voice cloning is **gated** — accounts without access get a `403`
  (`PermissionDeniedError`). Catch it and degrade gracefully rather than crashing.

## Roundtrip — TTS → STT verification

Useful pattern for verifying TTS output (and for eval test-cases):

```python
async with VeniceClient() as client:
    # Generate
    tts = await client.audio.create_speech(
        model=await client.models.resolve_tts(),
        input="The quick brown fox jumps over the lazy dog.",
        voice=Voice.AF_ALLOY,
        response_format=ResponseFormat.MP3,
    )
    tts.save(Path("./out/spoken.mp3"), overwrite=True)

    # Transcribe back
    result = await client.audio.transcribe(
        file=open("./out/spoken.mp3", "rb"),
        model=await client.models.resolve_asr(),
    )
    print(result.text)
```

The roundtrip is rarely 100% identical — punctuation, capitalization, and homophones drift — but the content should match.

## Cost-quote

The audio resource doesn't expose a `quote()` method — TTS / STT are billed per character (TTS) or per second (STT). Costs are typically small ($0.001-$0.01 per call); if precision matters, track the drop in `response.balance_info.usd` between consecutive responses (each value is what the key could spend before that request).

## Common bugs

- **`client.audio.transcribe(audio=...)`** — wrong kwarg. Use `file=`. (Lint V107.)
- **`from venice_ai.types.api import Voice, ResponseFormat`** — wrong path. Use `from venice_ai.types.enums import Voice, ResponseFormat`.
- **`client.audio.generate_speech(...)`** — not a v2 method; use `create_speech()`.
- **`client.audio.generate_music(...)`** — not a v2 method; music is its own resource, `client.music.*`. (Lint V102.)
- **`client.audio.speech.create(...)` / `client.audio.transcriptions.create(...)`** — OpenAI-style nesting, not present in Venice. Methods live directly on `client.audio`.
- **`await response.save(...)`** for `AudioResponse` — sync method.
- **Hardcoded voice strings without checking `get_voices(model_id=...)`** — voices are per-model.

## Related references

- `music.md` — `client.music.*` (split from audio in v2).
- `image.md` — sister modality with similar `.save()` pattern.
- `venice-py/references/model-resolution.md` — `resolve_tts()` and `resolve_asr()`.
- `venice-py-production/references/rate-limiting.md` — audio-route rate limits differ from chat.
