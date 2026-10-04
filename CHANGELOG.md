# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`prefer="cheapest"` on `client.models.resolve()` and every `resolve_*` shortcut** (chat, embedding, image, video, tts, asr, inpaint, music, decision). It picks the lowest-priced model that passes every filter, in strict price order, instead of the catalog's own ranking, which is unchanged when `prefer` is left at `None`. Reasoning-capable chat models compete on price like any other; pass `exclude_reasoning=True` to leave them out. Unpriced models sort after every priced one (a missing price is never read as $0). Equal prices break by Venice's `default` trait first, then catalog order, then model ID, so a price tie lands where `prefer=None` would (image returns `z-image-turbo`, not the alphabetically first `chroma`). The first `preferred_models` entry that passes the filters still wins over price. Price order is only as accurate as the catalog: a model whose listed capability or price is wrong upstream can still win. Prices compare within a type:
  - chat and decision: a blended per-million-token price, `(3 × input + 1 × output) / 4` (`CHAT_INPUT_WEIGHT`, `CHAT_OUTPUT_WEIGHT`);
  - embedding: the input price; tts: the input price; asr: the per-audio-second price;
  - image and inpaint: one request at the model's default resolution and default quality, which is what a request that sends neither pays, or at `require_quality` when given;
  - music: free `POST /audio/quote` calls at the clip length asked for (`music_duration_seconds`, or `duration_seconds` on `resolve_music()`), as in `resolve_cheapest_music()`; models that require lyrics are skipped;
  - video: the catalog carries no price, so candidates are ranked by free `POST /video/quote` calls, as in `resolve_cheapest_video()`.

  Under `"cheapest"`, beta models are skipped unless `exclude_beta=False` is passed to `resolve()`, `resolve_chat()` or `resolve_video()` (decision models are exempt, since every one is beta), end-to-end encrypted chat models are skipped unless `require_e2ee=True` (`require_private=True` does not re-admit them, because a plain request to one fails), and music implies `exclude_non_music=True`. `SyncVeniceClient` forwards the keyword unchanged.

- **Public price ranking, importable from `venice_ai`, `venice_ai.models` and `venice_ai.models.selection`.** `model_price(model, *, quality=None)` returns the comparable USD price described above for a selector-cache model dict, or `None`. `cheapest_model_strategy(candidates, *, quality=None)` picks deterministically by it, and `cheapest_selector(*, quality=None, preferred_models=None)` builds the `ModelSelectorType` that `prefer="cheapest"` uses, for `DynamicModelSelector(default_selector=...)` or a per-call `selector=`. `cheapest_model_strategy` previously lived only in `venice_ai.test_support`, which re-exports it.

- **`resolve_cheapest_music(duration_seconds=None, *, require_force_instrumental=False, lyrics_supplied=False, exclude_models=None, exclude_beta=True, max_concurrency=8)`** quotes every music generator with the free `POST /audio/quote` at the request `music_request_params()` builds for the clip length and returns a `CheapestMusicResult` (`model`, `quote_usd`, `request_params`, `effective_seconds`, `all_quotes`, `skipped`). A model with fixed `duration_options` is quoted at the smallest option covering the target (ace-step makes 60-second clips at minimum), a ranged model at the target raised to its minimum, and a model without duration metadata at its own length (`effective_seconds=None`); a model that cannot make a clip that long is skipped rather than quoted shorter. The quote is authoritative because Venice rounds every music quote and bill up to the cent. Equal quotes go to the clip closest to the target, then the `default` trait, catalog order and model ID. `lyrics_required` models are skipped unless `lyrics_supplied=True`. `venice-py models resolve --type cheapest-music --duration 10` prints the result. `model_price()` prices music the same way at `MUSIC_REFERENCE_SECONDS`: a model whose minimum is longer is priced at its minimum, one that cannot make a 30-second clip is unpriced instead of being priced at a shorter clip, and prices are rounded up to the cent.

- **`music_request_params(spec, target_seconds=None)`** returns the `music.quote()` / `music.submit()` arguments for a clip length from a model's duration metadata, never a `duration_seconds` the model would reject. `music_request_params`, `cheapest_video_params`, `model_price` and `CheapestMusicResult` are importable from `venice_ai` and `venice_ai.models`.

- **`MusicCompletedStatus.content_type` and `.audio_format`.** Inline audio from `music.retrieve()` records the response's media type (`audio/flac`) and an extension (`flac`, `mp3`, `wav`, `m4a`), so callers no longer sniff the bytes.

- **Image-to-video input modes.** Venice types plain image-to-video, reference-to-video, transition, first/last-frame and multi-angle models all `image-to-video`, with constraints that do not tell them apart, so `resolve_cheapest_video(video_type="image-to-video")` could return a reference-to-video model that ignores `image_url` whenever it quoted a few cents cheaper. `video_input_mode(model)` classifies a model by its id and name (`"image"`, `"reference"`, `"first_last_frame"`, `"transition"`, `"multi_angle"`). `video_type="image-to-video"` now means plain image-to-video models, and a new `input_mode=` on `resolve()`, `resolve_video()`, `resolve_cheapest_video()` and the selector methods (CLI `--input-mode`) picks the others; it implies `video_type="image-to-video"`.

- **`client.retry_options` and `client.connection_limits`.** Read-only views of what a client actually uses: `retry_options` returns the resolved policy (the `with_retries()` override inside such a block, otherwise the construction-time policy from `retry_options`, `max_retries` and `config.http_client`, with defaults filled in), or `None` for a caller-supplied `http_client`, where the SDK installs no retry middleware. `connection_limits` returns a frozen `ConnectionLimits(limit, limit_per_host)` read from a caller's session connector, or resolved from `http_transport_options` (`limit`, `limit_per_host`, which take precedence), then `connector_limit` / `connector_limit_per_host`, then the SDK defaults (1000, no per-host cap), which now live in one place. The session the SDK builds uses the same resolved values. `SyncVeniceClient` mirrors both.

- **`MusicJob.release()` and `Music.release()`** name what `/audio/complete` does: delete the stored media and queue entry. A `success: false` result is logged at WARNING.

- **`cheapest_video_params(constraints, *, duration=None, resolution=None, aspect_ratio=None, audio=False)`** in `venice_ai.models.selection` returns the cheapest valid `video.quote()` / `video.submit()` keyword arguments for a model's `VideoModelConstraints` (or their dict form): its shortest fixed duration, its lowest listed resolution in the catalog's own spelling, 16:9 when listed, and `audio=False` only where audio is configurable. It raises `ValueError` when the model lists no fixed duration or does not list a pinned value.

- **`exclude_reasoning`, `exclude_uncensored` and `require_custom_size` filters.** `resolve_chat(exclude_reasoning=True)` keeps only models with `supportsReasoning` false, for callers that need visible `content` under a small token budget; it raises `ValueError` with `require_reasoning` or `require_reasoning_effort`. The other route to direct answers keeps reasoning-capable models: `require_reasoning_effort="none"`, then send `reasoning_effort="none"`. `exclude_uncensored=True` (chat and image) skips models Venice flags `uncensored`. `resolve_image(require_custom_size=True)` keeps models sized by explicit `width`/`height`: Venice lists `aspectRatios` only for models that size by `aspect_ratio` (and `resolution`), and those reject or ignore pixel dimensions. The CLI takes `--exclude-reasoning`, `--exclude-uncensored` and `--custom-size`.

- **More chat filters.** `resolve_chat()`, `resolve(type="chat")` and `DynamicModelSelector.select_chat_model()` take `require_prompt_caching=True` (the model lists a `pricing.cache_input` price; Venice has no prompt-caching capability flag, and a listed price does not guarantee a cache hit is served or reported), `require_multiple_images=True` (`capabilities.supportsMultipleImages`) and `require_e2ee=True` (`capabilities.supportsE2EE`, or an `e2ee-` ID prefix). The selector cache now keeps `supportsMultipleImages`, `supportsE2EE` and `maxImages`.

- **Music filters.** `resolve_music()`, `resolve(type="music")` and `select_music_model()` take `exclude_non_music=True`, which skips the text-to-speech, sound-effect and voice-changer models Venice also types as `music` (voice changers, models that list TTS voices or are priced per thousand characters, and IDs or names marking sound effects, text-to-audio or speech), and `require_force_instrumental=True` (`supports_force_instrumental`).

- **`resolve_cheapest_video()` options.** `require_audio`, `require_audio_configurable`, `min_resolution`, `min_duration` and `max_concurrency` (default 8 quote requests in flight; the method adds no retries beyond the client's own policy). `CheapestVideoResult` gains `request_params`, the keyword arguments the winner was quoted with (pass them to `video.submit()` to be billed the quote), and `skipped`, mapping each candidate that was not quoted or whose quote failed to the reason.

- **`venice-py models resolve` flags.** `--prefer cheapest`, `--prompt-caching`, `--multiple-images`, `--e2ee`, `--quality` (image, inpaint), `--music-only`, `--force-instrumental` and `--audio-configurable`. `--type cheapest-video` output lists the parameters the winner was quoted with.

- **`ResponsesResponse.incomplete_details`.** A new `ResponsesIncompleteDetails` model carries the `reason` an `"incomplete"` response was cut short: `"max_output_tokens"` or `"content_filter"`. It defaults to `None` and accepts an explicit `null`.

- **Known-value tuples for the response fields that are now open.** `KNOWN_RESPONSE_STATUSES`, `KNOWN_RESPONSE_MESSAGE_STATUSES`, `KNOWN_RESPONSE_FUNCTION_CALL_STATUSES`, `KNOWN_RESPONSE_WEB_SEARCH_CALL_STATUSES`, `KNOWN_RESPONSE_INCOMPLETE_REASONS` and `KNOWN_FINISH_REASONS`, importable from `venice_ai.types` and `venice_ai.types.api`. Treat a value outside them as one this release predates, not as invalid.

- **More model capabilities are typed.** `ModelCapabilities` declares `maxImages` and `maxVideos`, which previously landed only on `model_extra`. `ChatCapabilities`, returned by `client.models.get_capabilities()`, exposes `max_images`, `max_videos`, `reasoning_effort_options` and `default_reasoning_effort`. All four default to `None`.

- **Capability filters on the model resolvers.** Each one narrows the candidate pool before `preferred_models`, a custom selector or the `default` trait is consulted, so a model that can't use the feature is never returned. When nothing matches, the resolver raises its usual `ValueError`, naming the filter. All filters are keyword-only, default to off, and are accepted by `client.models.resolve()` and the matching shortcut:
  - `resolve_image(require_web_search=True)` reads the image spec's `supportsWebSearch`. `resolve_image(require_quality="high")` reads `constraints.qualities`.
  - `resolve_inpaint()` takes `require_combine_images=True` (reads `constraints.combineImages`, which `multi_edit` needs), `require_quality=...`, `require_resolution="2K"` (reads `constraints.resolutions`) and `require_uncensored=True` (reads the spec's `uncensored` flag).
  - `resolve_video()` takes `require_duration=5` (also `"5"` or `"5s"`), which must appear in `constraints.durations`. Unlike `min_duration`, a model offering only longer clips fails it. It also takes `require_resolution="720p"`, which must appear in `constraints.resolutions`, and `require_audio_configurable=True`, which reads `constraints.audio_configurable`.
  - `resolve_chat()` takes `require_web_search=True` (reads `capabilities.supportsWebSearch`) and `require_reasoning_effort="none"`, which must appear in `capabilities.reasoningEffortOptions`.

  Quality, resolution and duration values match without regard to case, since the catalog mixes `"1080p"` and `"1080P"`. `DynamicModelSelector.select_image_model`, `select_inpaint_model`, `select_video_model` and `select_chat_model` accept the same keywords. `SyncVeniceClient` forwards them unchanged.

- **Image and inpaint resolution tiers are typed.** `ImageModelConstraints` and `InpaintModelConstraints` declare `resolutions` and `defaultResolution`, and `InpaintModelConstraints` also declares `qualities` and `defaultQuality`. All are optional open `str` values that default to `None`. They used to land on `model_extra`, so read them as attributes now.

- **`chat.completions.estimate_cost(venice_parameters=...)`** accepts a dict or `VeniceParameters`, so an estimate can reflect whether Venice's own system prompt will be injected. See *Changed* for what this does to the default estimate.

- **`chat.completions.estimate_cost()` counts every billed prompt component.** It took only message words × `tokens_per_word`, which undercounted every model measured: one 9-word prompt estimated at 11 tokens was billed 14 to 27, because each model's chat template adds tokens and tools and schemas are billed too. The estimate now adds `CHAT_TEMPLATE_TOKEN_ALLOWANCE` (20) per request and `CHAT_MESSAGE_TOKEN_ALLOWANCE` (4) per message, reported as `template_overhead_tokens`, counts message `name`s, and takes new `tools=` and `response_format=` arguments, reported as `schema_tokens`. Their compact JSON is counted at `SCHEMA_JSON_CHARS_PER_TOKEN` (2) characters per token, plus `TOOLS_TEMPLATE_TOKEN_ALLOWANCE` (200) once when tools are sent, for the tool-calling instructions chat templates add. Measured on five models from five providers, one small tool cost 43 to 270 prompt tokens and a 1,410-character one 250 to 741; four characters per token estimated about 65 and 353. The allowance covers the most expensive template measured with at least 20% headroom and can overstate a compact template several times over. A Pydantic class passed as `response_format` is counted as the `json_schema` format `create()` actually sends. `estimate_completion_cost()` adds the same allowance for one message (`template_overhead_tokens=` overrides it). `ChatCostEstimate.prompt_tokens` includes both. Venice has no tokenize endpoint, so the allowances are sized to err high. The estimate does not cover a system prompt built into a model's own chat template, which the catalog does not publish (one measured model bills about 550 prompt tokens for a one-word message with the Venice prompt off); measure it once with a one-token request.

- **`ChatUsage.cached_tokens` and `ChatUsage.cache_write_tokens`.** Read-only `int` properties for prompt-cache reads and writes. Each reads the nested `prompt_tokens_details` count and falls back to the top-level `cache_read_input_tokens` / `cache_creation_input_tokens` mirror, never summing the two, and returns `0` when the model reports nothing (some models omit `prompt_tokens_details` entirely on a miss). `ResponsesUsage.cached_tokens` does the same for `input_tokens_details`. `str(usage)` now shows `(cache read: N, write: M)` whenever either is non-zero, and reads the nested counts too.

- **`classify_request` is exported from `venice_ai`**, next to `RetryClass` and `RetryOptions`. `classify_request(method, path, headers)` returns the `RetryClass` the retry policy assigns a request, which is also the signature a `RetryOptions(classifier=...)` replacement takes.

- **CSV export of a whole usage window: `client.billing.iter_usage_history_csv()`.** It walks `GET /billing/usage-history` with `format=CSV` the way `iter_usage_history()` walks the JSON form, sending the filters on the first page and only the cursor after, and yields one `BillingUsageHistoryCsvPage` per page. Each page is a complete CSV document with its own header row; pages can be saved as separate files, named by their position in the walk. `page.filename` carries the name from `Content-Disposition`, but nothing guarantees those names sort in walk order, so order pages by walk index rather than by name. Takes `page_size` (server default 1000), `max_pages`, `currency`, `startTimestamp` and `endTimestamp`.

- **`TtsModelSpec.supported_formats`, `.default_format`, `.supports_custom_voice_id` and `.voice_cloning`.** Every TTS catalog entry sends these; they used to land untyped in `model_extra`. `voice_cloning` is a `TtsVoiceCloning` (`mode`, `accepted_formats`, `min_sample_seconds`, `retention_days`), present only on models that can clone a voice for the caller and `None` elsewhere. `mode` is an open `str`; narrow it against the new `KNOWN_VOICE_CLONING_MODES` (`"zero_shot"`, `"persistent"`).

- **`venice-py decisions` CLI command** for `client.decisions.create()`. Questions come from repeatable `--noul ID QUESTION`, `--choice ID QUESTION OPTIONS` and `--score ID QUESTION LEVELS` flags (options and levels comma-separated, levels lowest first), from a JSON question map with `--questions FILE` (`-` reads stdin), or both. The state is the positional argument or piped stdin, and `--state-json` sends it as structured JSON. Without `--model`, the model comes from `defaults.decision_model` in the CLI config or `client.models.resolve(type="decision")`. Answers print as a table with each answer's confidence and most likely options, or as the full response with `--json`. A malformed question flag, question file or `--state-json` value exits with code 2 before any request; a missing state and API errors exit with code 1.

### Changed

- **One API-key precedence, and `VeniceAIConfig.api_key` is read.** `VeniceClient(config=VeniceAIConfig(api_key=...))`, `SyncVeniceClient(config=...)` and `VeniceClientFactory.create_client(config=...)` ignored the config's key and raised "No authentication provided" when `VENICE_API_KEY` was unset. Every entry point now resolves the key in one order: an explicit `api_key=`, then `config.api_key`, then `VENICE_API_KEY`; the first that is not `None` wins, so an explicit `api_key=""` still means "no key" (for wallet auth) even when the variable is set. The resolver is public as `venice_ai.core.auth.resolve_api_key`. Migration: a config built while `VENICE_API_KEY` was set captures that key, and it now outranks a different `VENICE_API_KEY` set later; pass `api_key=` to override it.

- **Authentication is attached to each API request, not to the session.** A caller-supplied `http_client` session got no `Authorization` header, because the key was set only on sessions the SDK created, so every request through it went out unauthenticated; a wallet-only (SIWE) client also sent an empty `Authorization: Bearer` and no `X-Sign-In-With-X` on multipart uploads, and no `X-Sign-In-With-X` on streamed speech. Every request to the Venice API now builds its headers in one place (session defaults, then the client's Bearer key or a freshly signed SIWE envelope, then the client's `headers=`, then per-call headers), whichever session sends it, and the caller's session is never modified. `VeniceClient(headers=...)` was likewise dropped when `http_client=` was given; it now applies to every API request. Rate-limit discovery now goes through the client too, so it reaches the API root on a caller session. `fetch_external()` (video and music downloads) attaches the client's credentials only to URLs on the API's own origin, so a CDN or pre-signed storage host no longer receives the API key.

- **`billing.get_usage_history(format=BillingFormatEnum.CSV)` returns a `BillingUsageHistoryCsvPage` instead of `bytes`.** The bare bytes dropped the `x-next-cursor` header that carries a CSV page's continuation token, so a CSV walk could not get past the first page. The page holds the document in `.content` (`.text` decodes it as UTF-8), the token in `.nextCursor` — the same name the JSON page uses -- and the server's file name in `.filename`. `get_usage_history` is now overloaded, so a JSON call is typed `BillingUsageHistoryResponse` and a CSV call `BillingUsageHistoryCsvPage`. Migration: replace `data = await client.billing.get_usage_history(format=CSV)` / `data.decode()` with `page = await ...` / `page.content` or `page.text`, and continue with `cursor=page.nextCursor`.

- **A job released inside its `async with` block is not released again on exit.** After `await job.release()` (music) or `await job.cancel()` (video, voice changer) succeeded on a finished job, leaving the block sent a second release, and an exception exit logged that the media "was not released". The job now records a successful release and its exit does nothing more. A release that reports `success: false`, or a release sent before the job finished (it deletes nothing; the media is stored when generation ends), still leaves the exit-time release in place. For a `VoiceChangerJob`, finished means a poll has returned the converted audio, since `/audio/voice-changer/complete` is documented for a finished conversion; an earlier release is followed by the usual one on exit, which is safe because the endpoint may be called more than once.

- **Model resolvers raise `NoMatchingModelError` or `ModelQuotesUnavailableError` instead of a bare `ValueError`.** Both are new, exported from `venice_ai`, and subclass `ValueError` and `VeniceError`, so `except ValueError` still catches them. Every `resolve*()` path that finds no catalog model passing its filters raises `NoMatchingModelError` (with `resource_type` and, where recorded, `skipped`), including `resolve_video_upscale()` and `resolve_voice_changer()`. `resolve_cheapest_video()`, `resolve_cheapest_music()` and `resolve(..., prefer="cheapest")` for video and music raise `ModelQuotesUnavailableError` when models match but every quote call fails, with `failures` mapping each model ID to the exception its quote raised (also grouped in `__cause__`), so a caller can tell an empty catalog from a rate limit, an outage or a rejected quote request. `resolve_cheapest_video()` reported a candidate pool where no model could serve the requested duration, resolution, aspect ratio or audio as "returned no valid quote"; it now raises `NoMatchingModelError` without quoting.
- **`VoiceChangerCompletedStatus` carries `content_type` and `audio_format`**, like `MusicCompletedStatus`, so the converted audio can be named without inspecting its bytes. `voice_changer.retrieve()` now reads the completed body the way `music.retrieve()` does: an `audio/*` body is taken as given, `application/octet-stream` is accepted when the bytes are a known audio container, and an empty or non-audio body (an HTML error page, plain text) raises `APIResponseProcessingError` instead of being returned as completed audio.
- **`RetryOptions` rejects 429 in `retry_status_codes` with a `ValueError`.** A 429 is retried by the client's rate limiter, which honors `Retry-After`, the error-budget window and the request deadline, and re-signs a SIWE envelope per attempt. Listing 429 in `RetryOptions` did nothing for chat, responses, embeddings and paid generation, and for idempotent requests it stacked a second retry loop under the limiter's own. The error names the setting to use instead: `VeniceClient(rate_limiter=SimpleRateLimiter(max_retries=...))`, or `RateLimiterConfig(max_retries=...)` with `VeniceClientFactory`. `RetryOptions` is now a frozen dataclass whose collections are immutable: `retry_status_codes` and `idempotent_methods` take a set and are stored as `frozenset`s, `retry_exceptions` takes a sequence and is stored as a `tuple`, and any other type raises `TypeError` (a list of status codes used to be accepted at runtime although the annotation said set). Assigning a field after construction raises `dataclasses.FrozenInstanceError`, so 429 cannot be slipped in later and `client.retry_options` returns the policy itself rather than a copy. Migration: build a variant with `dataclasses.replace(options, max_attempts=...)` (validated again) instead of assigning to a field, and pass `retry_status_codes={500, 503}` rather than a list.

- **Leaving a `MusicJob` context no longer "releases" a job that is still generating.** `/audio/complete` deletes media that already exists; called mid-generation it deleted nothing, and the finished audio was stored afterwards anyway. On exit the job is polled once more: a finished job is released, and a job still generating is left alone with a WARNING naming its `queue_id` (`wait()` can be called again after a `TimeoutError`, then `release()`). Docstrings no longer say a 400 means "already gone": `/audio/complete` answers a released id with `success: true` and a never-issued one with 400 "Request ID is invalid", and `retrieve()` / `wait()` answer a released or expired id with `NotFoundError` (404 "Media could not be found") and a never-issued one with 400 "Request ID is invalid".

- **Leaving a `VideoJob` block cleanly before the job finishes no longer calls `/video/complete`.** Measured live: a release sent while the job is still generating deletes nothing, and Venice stores the video when it finishes. The job is left alone and a WARNING names the `queue_id`; call `wait()`, save the video, then `cancel()`. A clean exit after the job finished still releases it.

- **`VideoJob.cancel()` deletes the job's `download_url`.** Models that return a `download_url` at queue time keep the file behind that link for up to 24 hours, and `/video/complete` answers 400 "Request ID is invalid" for them both before and after completion, so `cancel()` raised and the link stayed live. `cancel()` now sends `DELETE` to the link once, without the client's credentials, treating a 404 as already deleted, then calls `/video/complete` and returns `VideoCompleteResponse(success=True)` when only that call's "Request ID is invalid" 400 remains. The URL is never logged or put in an error message. `client.video.cancel(model=..., queue_id=...)` cannot see the link and is unchanged; its docstring says how to delete it.

- **A `MusicJob`, `VideoJob` or `VoiceChangerJob` block that raises no longer releases the job's output.** The context manager released stored media on every exit, so when saving a finished, already-billed track or video failed (a disk error, a cancelled task, a bug after `wait()`), the output was deleted server-side before the caller could retry. Media is now released only when the block exits cleanly. On an exception a WARNING names the `queue_id` and the calls that resume it: `client.music.retrieve(model=..., queue_id=...)` (or `video` / `voice_changer`), save the output, then `client.music.release(...)` / `client.video.cancel(...)` / `client.voice_changer.cancel(...)`. A music or video job that failed or was rejected has nothing to keep and is still released either way; the voice changer reports failures as HTTP errors rather than a status, so a `VoiceChangerJob` keeps its media on any exception exit. `MusicJob`'s exit also stops reporting "still generating" for a `queue_id` whose media is already gone (404) or that Venice never issued (400). Migration: if you relied on an exception exit to delete the output, call `await job.release()` (music) or `await job.cancel()` inside the block.

- **One form for the API base URL everywhere.** `VeniceAIConfig.api_base_url` was a bare host to which `VeniceClientFactory` appended `/api/v1`, while `VeniceClient(base_url=...)` took the URL with `/api/v1`, so the same `http://host/api/v1` sent the factory's requests to `/api/v1/api/v1/...`, and `VeniceClient(config=...)` ignored `config.api_base_url` altogether. Every entry point (`VeniceClient(base_url=...)`, `SyncVeniceClient`, `VeniceAIConfig.api_base_url`, the `VENICE_API_BASE_URL` environment variable, `VeniceClientFactory` and the CLI's `api.base_url`) now takes the API root, version path included (`https://api.venice.ai/api/v1`), and resolves it with one function. A bare host (`https://api.venice.ai`) is shorthand for `{host}/api/{api_version}`; any other path is used as given. `VeniceClient` reads an explicit `base_url` first, then `config.api_base_url`, then `VENICE_API_BASE_URL`, then the official URL. `VeniceAIConfig.api_base_url` now defaults to and reads back as `https://api.venice.ai/api/v1`. Migration: a config value that was a host plus a proxy prefix (`https://proxy.example.com/venice`, which the factory used to extend to `.../venice/api/v1`) must now spell out the full root, `https://proxy.example.com/venice/api/v1`. The one unsupported shape is a gateway that serves the API at its host root (`https://gateway.example.com/chat/completions`): a URL without a path, trailing slash or not, always takes the bare-host shorthand, and `api_version` changes only the version segment it appends. Serve such a gateway under a path and pass the URL with that path.

- **Resolver no-match messages name the model kind.** `resolve_tts()` raised "No available audio models found" and `resolve_asr()` "No available ASR models found"; they now say "text-to-speech (tts)" and "speech-to-text (asr)", matching `resource_type`.

- **`music.retrieve()` treats only audio as a completed result.** Any 2xx body that was not JSON used to be returned as finished audio, so an HTML proxy page or an empty body became a "track". Now an `audio/*` Content-Type is completed (an empty body raises), `application/octet-stream` is accepted when the bytes are a known audio container (ID3-prefixed FLAC included), and anything else raises `APIResponseProcessingError` with a body preview.

- **Music `duration_seconds` is checked against models that declare no duration.** `music.submit()` / `quote()` raise `ValueError` when the model declares neither `duration_options` nor a min/max duration, the models Venice rejects the parameter on (minimax, lyria, seed-audio).

- **The default retry policy no longer resends paid requests.** Until now every request, paid generation included, was retried up to 3 times on 500/502/503/504 and on any connection error or timeout, because `RetryOptions.retry_non_idempotent=True` treated Venice endpoints as "effectively idempotent". They are not: Venice bills image, video, music, speech and voice generation when the job is accepted, offers no `Idempotency-Key` on those endpoints, and a 504, read timeout or dropped connection can arrive after the work was done and billed. A deterministic failure also cost four attempts and about 7 seconds before it surfaced. The middleware now classifies each request by endpoint (`classify_request()`, `RetryClass`):
  - `IDEMPOTENT` (GET/HEAD/PUT/DELETE, free `*/quote`, `*/retrieve`, `*/complete` and `billing/*` POSTs, and any POST carrying `Idempotency-Key`): retried on connection failures, 500, 502, 503, 504, read timeouts and server disconnects.
  - `INFERENCE` (`chat/completions`, `responses`, `embeddings`): retried on connection failures, 502 and 503, and on a 500 at most once (`max_inference_500_retries=1`). Measured against the billing ledger, Venice does not bill chat 500s, but some are deterministic (a model that cannot honor `response_format`). A 504 or read timeout is not resent.
  - `PAID` (every other POST, including unknown paths): retried only when the connection was never established, or on a 503 that cannot have followed processing: on the generation endpoints Venice documents a 503 for, where it means the model is at capacity (`image/*`, `audio/speech`, `audio/queue`, `audio/voices`), or when the response carries `Retry-After`. A 503 from `video/queue`, `x402/top-up` or `api_keys` is surfaced.

  The defaults now match the OpenAI and Anthropic SDKs: `max_attempts=2` (was 3), `base_delay=0.5` (was 1.0), `max_delay=8` (was 60), and jitter that only shortens a delay by up to 25%. `HttpClientConfig.max_retries` defaults to 2. A `Retry-After` or `retry-after-ms` header sets the delay when it is at most `max_retry_after` (now 60s, was 120s); a longer one surfaces the error instead of being clamped and retried. 429 stays with the rate limiter. Failed responses are released before the backoff sleep, and resent attempts carry an `x-venice-sdk-retry-count` header. `client.crypto.rpc()` / `batch_rpc()` send a random `Idempotency-Key` when none is given, so their retries are deduplicated. `RetryOptions(classifier=...)` overrides the classification; `retry_non_idempotent=False` still disables every retry of a non-idempotent POST. A timeout error now says whether the request was sent and that a paid request is not resent.

- **`resolve_cheapest_video()` quotes each model at its own cheapest request by default.** `duration` now defaults to `None` (each model's shortest listed duration) instead of `"5s"`, and `resolution` to each model's lowest tier. A pinned `duration` or `resolution` filters out models that do not list it rather than quoting them to failure. `audio` defaults to `False` instead of `None`, so models with configurable audio are quoted silent; pass `audio=None` for each model's own default. Without a `video_type`, upscale and video-to-video models are skipped, since they price from a source video. `get_cheapest_video_model()`, `DynamicModelSelector.select_cheapest_video_model()` and `venice-py models resolve --type cheapest-video --duration` change the same way. A call that passed `duration="5s"` still returns the cheapest model offering a 5-second clip.

- **`venice_ai.test_support.get_model_price` is now an alias of `model_price`.** Chat prices are the blended `(3 × input + output) / 4` rather than `input + output`, a missing input or output makes the model unpriced instead of counting as $0, and ASR, resolution-tiered image and music prices are recognized. `cheapest_model_strategy` breaks ties by the `default` trait, then catalog order, then model ID.

- **Status and outcome fields on responses are now open `str` instead of closed `Literal`s.** A new value from the server is kept rather than failing the whole response. This covers `ResponsesResponse.status`, `ResponsesMessageOutput.status`, `ResponsesFunctionCallOutput.status`, `ResponsesWebSearchCallOutput.status`, `ChatChoice.finish_reason` and `ChatCompletionChunkChoice.finish_reason`. `ChatChoice.finish_reason` is still required but is now `str | None`, and it accepts the spec's `"content_filter"`.

  Runtime behavior is strictly more permissive, but this widens public `Literal`s. A caller relying on exhaustiveness checking over these fields will see their type checker stop proving the match is exhaustive; narrow against the matching `KNOWN_*` tuple and handle the fallback. Because `ChatChoice.finish_reason` is now nullable, an unguarded call such as `choice.finish_reason.startswith(...)` no longer type-checks.

- **Default cost estimates include the system prompt Venice injects.** Unless `venice_parameters` sets `include_venice_system_prompt=False` or `enable_e2ee=True` (end-to-end encryption turns the injected prompt off), `chat.completions.estimate_cost()` adds `venice_ai.costs.VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE` (1750 tokens), priced at the model's `cache_input` rate, or at `input` when it publishes none. The injected prompt's size varies (roughly 1100 to 1750 tokens have been observed), so this is a conservative allowance, not an exact count. `ChatCostEstimate` gains `venice_system_prompt_tokens`, and `prompt_tokens` now includes it. `costs.estimate_completion_cost()` gains a keyword-only `include_venice_system_prompt=True` that behaves the same way.

  Default estimates are therefore about 1750 prompt tokens higher than before. Pass the opt-out to get the old count of the caller's own words.

- **Validation error messages now carry the server's per-field reasons.** Venice sends the same `"Invalid request parameters"` for every request-validation failure, and that string was all `str(exc)` showed. Each reason is now appended as `field: message`, for example `Invalid request parameters (input: Input text exceeds the maximum token limit of 8192 tokens)`. Reasons come from the Zod `issues` array; for union failures they are taken from the nested `unionErrors` branches, and when `issues` is empty the `details` tree is used instead. At most ten are shown. `exc.body` is left unchanged.

  Code that compares `str(exc)` for exact equality will stop matching. Substring checks against the server's `error` text keep working, since the reasons are only appended.

- **Presets now use the retry count they document.** `HttpClientConfig.max_retries` and `retry_backoff_factor` used to be ignored (see *Fixed*), so every client made 4 attempts. Clients built from presets now honor their settings: the testing preset retries once, the circuit-breaker testing preset never, the development presets twice, and `create_developer_client()` once. Precedence is `retry_options=` > `VeniceClient(max_retries=)` > `config.http_client` > `RetryOptions` defaults.

- **The per-host connection limit is unlimited by default.** It was derived from `max_keepalive_connections`, which silently capped concurrency to one host at 20 by default, 50 in the production presets and 100 in the high-throughput preset despite its 500-connection pool. It is now unlimited unless `connector_limit_per_host` is given.

- **Presets that use the SIMPLE rate limiter keep a default `SchedulerConfig`** apart from `mode=BASIC`: `create_minimal_config()`, `create_development_config()`, `create_development_config_with_rate_limiting()`, `create_testing_config()`, `create_testing_config_with_intelligent_scheduler()` and `create_testing_config_for_circuit_breaker()`. The scheduler has no effect under the SIMPLE rate limiter, and the tuned values they used to set would trigger the inert-configuration warning described under *Fixed*. Every shipped preset now passes through `VeniceClientFactory.create_client()` without a warning. `create_testing_config_with_intelligent_scheduler()` no longer sets `SchedulerMode.INTELLIGENT`, which never ran.

- **`VeniceAIConfig.create_test_config()` defaults to the memory backend and a default `SchedulerConfig`.** It pairs its config with the SIMPLE rate limiter, which reads neither section, so its old defaults (a Redis backend on `redis://localhost:6379/15`, `test_mode=True`, `test_rate_multiplier=10.0`, `max_concurrent_executions=10` and `max_queue_size=100`) made `VeniceClientFactory.create_client(VeniceAIConfig.create_test_config())` emit two inert-configuration warnings. `enable_redis` now defaults to `False` and `test_rate_multiplier` to `None`, stored only when passed; `scheduler_mode` still sets `scheduler.mode`. Pass `enable_redis=True` to get the Redis backend as before. `create_test_client()` and `create_test_venice_client()` keep their own defaults (`enable_redis=True`, `test_rate_multiplier=10.0`) and stay silent, but the configs they build no longer carry `test_mode`, `max_concurrent_executions=10`, `max_queue_size=100` or `scheduler_interval=0.01` either; the scheduler section keeps its defaults apart from `mode` and `test_rate_multiplier`.

- **`test_rate_multiplier` on the testing presets defaults to `None`** and is stored only when passed. Only the ADAPTIVE scheduler reads it, so an explicit value is reported by the inert-configuration warning.

- **`validate_config()`** drops the keepalive-ratio and keepalive-exceeds-max rules and the Redis key-prefix recommendation, and reports the deprecated fields below as warnings instead.

- **`duration_seconds` is optional on `client.video.submit()`, `run()` and `quote()`** (`int | str | None = None`), and `duration` is optional on `VideoTextToVideoRequest`, `VideoImageToVideoRequest` and `VideoQuoteRequest`. When it is `None`, the request body carries no `duration`. Upscale and other video-to-video models (catalog `durations` of `["Auto"]`) take the length from `video_url`. `/video/queue` rejects `"Auto"` for `topaz-video-upscale` with "duration must be a positive number", so an upscale job could not be queued before. A generation model whose catalog lists numeric durations still needs one: leaving it out raises `ValueError` naming the allowed values before anything is sent. When the catalog lookup fails, the request goes out without a duration and the server decides. A duration that is given is checked against the catalog as before.

- **A blank `api_version` is rejected.** `VeniceAIConfig(api_version="")` (or whitespace, or `"/"`) resolved a bare-host `api_base_url` to `https://host/api/`. The config now raises `ValueError` naming the setting, whatever the URL. Surrounding slashes and whitespace are trimmed, so `"/v2/"` is read as `"v2"`. Migration: pass a version such as `"v1"`, or leave `api_version` at its default.

- **The production presets read `VENICE_BACKEND__REDIS__REDIS_URL`.** `create_production_config()`, `create_production_config_high_throughput()` and `create_production_config_conservative()` fell back only to `VENICE_REDIS_URL` when `redis_url` was not passed, so the SDK's own nested setting, the one the README documents, raised "Production Redis URL required". They now take an explicit `redis_url`, then `VENICE_BACKEND__REDIS__REDIS_URL`, then `VENICE_REDIS_URL`; an empty variable counts as unset. Migration: when both variables are set to different URLs, the nested one now wins; unset it or pass `redis_url=` to keep the other.

### Removed

- **`VeniceClientFactory.create_client()` no longer takes `account_key`.** It was documented as overriding `api_key` but was never read. Migration: pass the key as `api_key=` (or `config.api_key`).

- **`Music.cancel()` and `MusicJob.cancel()` are removed.** Both were aliases of `release()`, which wraps `/audio/complete`: it deletes a finished job's stored media and cancels nothing. Migration: call `client.music.release(model=..., queue_id=...)` and `await job.release()`.

- **`RateLimitDiscovery(account_key=...)` is removed.** The parameter was stored and never read. Migration: drop the argument; discovery uses the client it is given.

### Deprecated

- **Configuration fields nothing reads.** `HttpClientConfig.max_keepalive_connections` (aiohttp has no keepalive-pool-size setting), `RedisBackendConfig.key_prefix` (the adaptive Redis backend names its keys itself and applies no prefix) and `SchedulerConfig.strategy`, `enable_request_batching`, `model_fallbacks` and `enable_model_discovery`. They are still accepted and will be removed in the next major release.

  Setting one to a non-default value emits a `FutureWarning` attributed to your own line of code: at construction for the HTTP and scheduler fields, and when the ADAPTIVE rate limiter is built for `key_prefix`. Reading one emits pydantic's access-time `DeprecationWarning`. `validate_config()` reports non-default values as warnings. Presets, `create_minimal_config()` and `create_test_config()` no longer set any of them, so code that does not set them sees nothing.

- **The production presets' `redis_key_prefix` parameter** now defaults to `None` and is deprecated along with the field it sets.

- **`create_testing_config_with_intelligent_scheduler()`** will be removed in the next major release. It uses the SIMPLE rate limiter, so no scheduler runs, and it matches `create_testing_config()` apart from its timeout, pool size and retry count. To test the scheduler, set `rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE)`, which needs the `adaptive` extra.

### Fixed

- **Paginated iterators stop at `max_items` without fetching another page.** When the cap fell exactly on a page boundary, every `iter_*` helper requested the next page and discarded it.

- **Optional request and response fields no longer type-check as required.** 279 model fields passed their pydantic default positionally (`Field(None, ...)`), which type checkers without the pydantic plugin read as no default, so calls such as `CreateApiKeyRequest(apiKeyType=..., description=...)` or `ConsumptionLimit(usd=25.0)` were flagged for missing arguments. Every default is now passed as `default=`; runtime behavior is unchanged.
- **A response body that stalled could still raise a bare `TimeoutError`.** Bodies read after the status line arrived, outside the request path that already mapped transport errors, let `TimeoutError` / `aiohttp` errors escape: binary results of `image.create(return_binary=True)`, `edit()`, `multi_edit()`, `background_remove()` and `upscale()`, `audio.create_speech()` (whole and streamed), inline results and JSON statuses of `music.retrieve()`, `video.retrieve()` and `voice_changer.retrieve()`, `video.transcribe(response_format="text")`, `crypto.batch_rpc()`, CSV `billing.get_usage_history()` and `client.fetch_external()` (used by `download()`). Every body read now goes through the same mapping as the request (`venice_ai.utils.errors.read_body`, built on `map_aiohttp_error`), so a stall is `APITimeoutError` and a dropped connection `APIConnectionError`, with a message that says the server had already answered and the request may have been billed; a CSV usage-history stall is `BillingTimeoutError`, raised at the 10-second billing deadline, which now covers the body as well as the response. The body is read first and parsed afterwards, so a 2xx body that is not JSON from a multipart endpoint is `APIResponseProcessingError` instead of `APIConnectionError` (aiohttp's `ContentTypeError` is a client error), and a failed JSON status logs a preview of the bytes already read instead of reading the body again. `Stream` maps mid-stream failures with the same function. Multipart uploads, raw audio streams and `fetch_external()` also ignored the client's `timeout` (only the session default applied, and the audio stream dropped an `aiohttp.ClientTimeout`); every request path now resolves its timeout the same way. The image reader's fallback for an empty first `content.read()` is gone: it read the stream instead of `response.read()`, which already returns the body aiohttp holds. Migration: catch `APITimeoutError` (it does not subclass `TimeoutError`) wherever code caught `TimeoutError` or `asyncio.TimeoutError` from a request, including calls routed through a rate limiter; `wait(max_polls=...)` running out of polls is still a plain `TimeoutError`.

- **Inline audio sniffing labeled AAC as MP3 and missed an ID3 footer.** `sniff_audio_format()` (which names `music.retrieve()` and `voice_changer.retrieve()` results sent as `application/octet-stream`) took any `0xFFF` frame sync for MP3, so raw AAC ADTS was reported as `audio/mpeg`; it is now `"aac"` / `audio/aac`, told apart by its `00` layer bits. A leading ID3v2 tag whose footer flag (byte 5, bit `0x10`) is set is skipped with its 10-byte footer, so the FLAC after it is found. An ID3 tag followed by unrecognized bytes is no longer assumed to be MP3, a tag with non-syncsafe size bytes is not treated as a tag, and MP3 means a parseable MPEG audio Layer III frame header. Tag and frame parsing reuse the parsers MP3 stitching uses, and the filename chosen for raw `bytes` uploads comes from the same sniffer, so an ID3-prefixed FLAC upload is named `.flac`.

- **`Image.create()` overloads omitted `timeout`**, so `create(..., timeout=60)` type-checked as an error although it works. `Image.create`, `Audio.create_speech`, `ChatCompletions.create` and `Responses.create` also gain an overload for a discriminator that is not a literal (`return_binary=flag`, `stream=flag`), typed as the union of both results; before, mypy rejected any call that passed a `bool` variable.

- **Docstrings that overstated what a value means.** `ReasoningEffortLevel` said a higher tier allows more thinking tokens and costs more; `supportsReasoningEffort` and `reasoningEffortOptions` only say which values a model accepts, and measured usage is non-monotonic on several models. `UsageAnalyticsByDate.USD` and the `totalUsd` of `byModel` / `byKey` are gross USD-denominated spend (USD plus bundled-credit debits, refunds not netted), not plain USD, so they do not reconcile with `iter_usage_history(currency="USD")`. The DIEM balance fields now say DIEM is the staking allowance (each staked DIEM is $1 of credit per epoch, reset at 00:00 UTC) and that whether a key's DIEM headroom is capped by its DIEM limit is unverified.

- **`Balances`, `RateLimitsData.balances`, `BalanceInfo` and `response.balance_info` were documented as the account balance.** They report what the calling API key can still spend: the lesser of the account's balance and what remains under the key's consumption limit, so a key with a $1 limit reports at most $1 however much the account holds. `balance_info` (the `x-venice-balance-usd` / `-diem` headers) is that value before the request was processed, and equals `get_rate_limits().data.balances` read just before the call. The docstrings, the advanced guide and the bundled skills now say so, and point to `client.billing.get_balance()` for the account balance. The `api_keys.get_rate_limits()` docstring example also used attributes the response does not have.
- **A request timeout raised a bare `TimeoutError` instead of `APITimeoutError` on several paths.** With a rate limiter attached (`SimpleRateLimiter`, the factory default), a non-streaming call that timed out raised `builtins.TimeoutError` with an empty message, because only the direct path mapped transport errors. Under the adaptive scheduler it was worse: the scheduler's own `except asyncio.TimeoutError` caught the request's timeout and reported it as `RateLimiterError("Request timed out after {its own deadline}s")`. A timeout while the response body was still arriving raised a bare `TimeoutError` with or without a limiter, and a streamed call that timed out before its first byte raised `VeniceError("Stream error: ...")` wrapping the `APITimeoutError` instead of the `APITimeoutError` itself. Every request is now sent through one mapping, so an HTTP request timeout is `APITimeoutError` (a connection failure `APIConnectionError`) whether or not a limiter runs, for JSON and streamed responses alike, before the headers or while a JSON body is read, with the `asyncio` / `aiohttp` error as `original_error` and `__cause__`. On the rate-limited path the mapping runs inside the request the limiter executes, so a limiter never sees a bare `TimeoutError` from the request, and a timeout a limiter raises itself (waiting for capacity, before anything is sent) is not relabelled as a request timeout. An error status whose body stalls now raises the status error (for example `InternalServerError`), with the read failure as its `body`. `Stream` re-raises any `VeniceError` unchanged, not only `APIError`.

- **`ModelResponse.model_dump()` / `model_dump_json()` dropped every field a `ModelSpec` subclass declares.** The spec was promoted to the right subclass on parse, but pydantic serialized it as the declared `ModelSpec`, so text dumps lost `capabilities`, `availableContextTokens` and `maxCompletionTokens`, image dumps lost `constraints`, and embedding dumps lost `embeddingDimensions`. `venice-py models get --json` printed no capabilities as a result. The field is now `SerializeAsAny[ModelSpec]`; its static type is unchanged.

- **`ReasoningConfig(enabled=False)` was silently dropped.** The model had no `enabled` field, so the toggle Venice recommends for switching reasoning off serialized to `{}`. It is now sent; Venice ignores it when an effort is also given.

- **`resolve_chat(require_function_calling=True)` applies every other filter.** Function-calling selection returned early through its own path, which ignored `exclude_beta`, `require_vision`, `require_reasoning`, `require_response_schema`, `min_context_tokens` and `require_private`, and could return a beta model despite `exclude_beta=True`. All chat filters now narrow one candidate pool; the `function_calling_default` trait still ranks the result. `DynamicModelSelector.select_function_calling_model()` delegates to `select_chat_model(require_function_calling=True)`.

- **`resolve_cheapest_video()` no longer drops models that lack a 5-second clip.** Every candidate was quoted at a fixed `"5s"`, so models whose shortest duration is 4, 6 or 8 seconds failed their quote and were left out of the comparison without notice. See *Changed*.

- **`venice_ai.core.backends.RedisBackend` no longer leaves a "coroutine `_cleanup_connection` was never awaited" warning.** When an event loop the backend had connected on was garbage collected, a weakref callback scheduled the old client's cleanup as a task on whichever loop happened to be running, which could close before the task ran. A switch to a new loop did the same with an untracked task. Neither could work, since a redis-py asyncio connection is bound to the loop that created it. The callback now releases the dead loop's client and drops its connection pool from the per-loop pool table synchronously, and a loop switch simply releases the old client. A new loop that reuses the dead loop's `id()` no longer inherits its pool.

- **A retrieve-time 422 ends a video or music job.** A provider refusal (for example on content policy, with "Credits have been refunded") comes back from `/video/retrieve` as a 422. `VideoJob.wait()` let the `UnprocessableEntityError` escape and the job never became terminal, so leaving `async with` logged a WARNING that it was "still billed". `wait()` now raises `VideoGenerationError` chained from the 422 and marks the job finished, with no warning. Only 400 (other than "Request ID is invalid") and 422 count as a job outcome; 401, 403, 404 and 429 describe the credentials, the lookup or the rate, and still propagate unchanged.

- **`MusicJob.wait()` maps retrieve rejections to `MusicGenerationError`.** A 400 (other than "Request ID is invalid") or 422 from `/audio/retrieve` is raised as `MusicGenerationError` chained from the original, and the job counts as finished, matching `VideoJob`. A cleanup 400 on exit is logged at DEBUG, as it is for video.

- **Streaming no longer logs API errors at ERROR.** An `APIError` raised while iterating a stream (for example a 401, or a 404 whose body lists "Did you mean" model ids) was logged as `Unexpected streaming error: ...` before being raised to the caller. It is now logged at DEBUG with its class and status only, as the non-streaming path already did. Other exceptions are still logged at ERROR.

- **A Responses API generation cut short no longer fails the parse.** When `max_output_tokens` or a content filter stops a generation, Venice returns `status: "incomplete"`, which the closed `Literal` rejected, so the partial output was lost to an `APIResponseValidationError`. The response now parses with its partial `output` and `usage`, and message and function-call blocks marked `"incomplete"` stay typed as `ResponsesMessageOutput` / `ResponsesFunctionCallOutput` instead of falling through to `ResponsesUnknownOutput`.

- **Cost calculation bills the way Venice charges.** `calculate_completion_cost()`, and with it `CostTracker.track()` and `ChatCompletionResponse.summary(pricing=...)`, billed every prompt token at `input` and ignored the cache and extended rates. Cached prompt tokens are now billed at `cache_input`, read from `prompt_tokens_details.cached_tokens` or the top-level `cache_read_input_tokens` (one of the two mirrors, never their sum), and cache-write tokens at `cache_write`, read from `prompt_tokens_details.cache_creation_input_tokens` or the top-level `cache_creation_input_tokens`. The uncached remainder is billed at `input`, and a model with no cache rate falls back to `input`. When the total prompt is strictly greater than `extended.context_token_threshold`, the `extended` rates apply to the whole request; a missing extended cache rate falls back to `extended.input`, and a missing extended input or output rate to the standard one.

- **Retry settings in `HttpClientConfig` take effect.** `max_retries` and `retry_backoff_factor` were never read: every client made 4 attempts with backoff base 2.0 whatever the configuration said. See *Changed* for how this affects presets.

- **`VeniceClient(proxy=...)` sends requests through the proxy.** The value was stored and never used.

- **An explicit `connector_limit` / `connector_limit_per_host` wins when `config=` is also passed.**

- **ADAPTIVE mode forwards the scheduler and Redis settings it was ignoring.** Every `SchedulerConfig` field except `mode` now reaches the adaptive scheduler (`max_concurrent_executions`, `max_queue_size`, `overflow_policy`, `request_timeout`, `rate_limit_buffer_ratio` and the rest), so the production presets' queue and concurrency values take effect. `mode` is ignored there: the adaptive scheduler always runs INTELLIGENT. `backend.redis.max_connections` (default 20, previously the upstream default of 10) and `cluster_mode` reach the Redis backend; with `cluster_mode=True`, `redis_url` is the cluster seed node.

- **Configuration that can have no effect is no longer silent.** `VeniceClientFactory.create_client()` emits a `UserWarning` when a Redis backend, or a `SchedulerConfig` changed from its defaults (ignoring `mode`), is paired with a rate-limiter mode other than ADAPTIVE. Neither is used outside ADAPTIVE. `create_test_client()` stays silent.

- **Multipart endpoints raise the same error message and `exc.code` as JSON endpoints.** Image upscale and edit and audio transcription pass the error body along as raw text, so the message was a raw JSON dump and `exc.code` was `None`. The text is now decoded as JSON first; `exc.body` still holds the original text, and a body that is not JSON is shown unchanged. `APIStatusError` builds its message the same way as the status-specific exceptions and sets `exc.code` from the body.

- **`stream_long_text` produces one well-formed MP3 stream for multi-segment input.** Only later segments' leading ID3 tags were removed, and only from their first network chunk, so a tag split across chunks leaked its tail into the audio; every segment's Xing/Info header frame stayed in the stream. The first Info frame declared only segment 0's frame count, so browsers, `afinfo` and speech-recognition front ends reported a truncated duration. Each segment's leading ID3v2 tag and Xing/Info/VBRI header frame, segment 0's included, is now removed as bytes stream through, buffering at most one audio frame per segment. Multi-segment output carries no ID3 tag and no LAME gapless metadata; players derive the duration from the constant bitrate. Single-segment output is unchanged. The byte count passed to `on_segment_complete` is counted after stripping.

- **Raw MPEG audio is no longer uploaded as AAC.** The filename sniffer used for `bytes` input to transcription and voice-changer treated any `0xFF 0xF?` frame sync as AAC ADTS, so an MP3 starting on a bare MPEG-2 Layer III frame header, such as multi-segment `stream_long_text` output, was sent as `audio.aac`. ADTS is now recognized only with its `00` layer bits.

- **Audio uploaded from a `BytesIO` or a file object without a `name` gets a real content type.** Only raw `bytes` input had its format detected from magic bytes; the others were sent as `audio` with `application/octet-stream`. Transcription and voice-changer uploads without a name are now detected the same way as `bytes`. Named files and paths still go by their extension.

- **Chat message models validate on assignment.** `UserMessage`, `AssistantMessage`, `ToolMessage`, `SystemMessage` and `DeveloperMessage` now raise `ValidationError` for `msg.role = "assistant"` on a `UserMessage` or for non-text `content`, instead of sending it.

- **Video elements can be video-only.** `VideoElement.frontal_image_url` is optional. An element takes exactly one media source: images (`frontal_image_url` and/or `reference_image_urls`) or a single `video_url`. One with both or neither is rejected client-side, matching the API's `"Cannot provide both image URLs and video URL"` rejection. An element's `video_url` is scheme-validated.

- **The video prompt is optional.** `prompt` is optional in `client.video.submit()` / `run()` and on the video request models (`min_length=1` when given), so upscale and enhancement models that work from `video_url` no longer need a placeholder prompt; an omitted prompt is left out of the request body. The parameter stays keyword-only in the same place.

- **`VideoJob.wait()` reports a job rejected after queueing as a generation failure.** When `/video/retrieve` returns a 400 because the queued job failed server-side validation, `wait()` raises `VideoGenerationError` with the original `InvalidRequestError` as `__cause__`. A 400 saying the request ID is invalid, meaning an unknown or already released `queue_id`, still raises `InvalidRequestError`. `retrieve()` is unchanged.

- **`VideoJob.download()` / `MusicJob.download()` no longer return a path to a file that was never written.** When a completed status has no inline data and no URL (and, for video, no queue-time `download_url`), they raise `VideoGenerationError` / `MusicGenerationError` without writing or creating anything.

- **Leaving a job context early is no longer silent.** Leaving `async with VideoJob` / `MusicJob` before the job reached a terminal status (never polled, still processing, or after a `wait()` timeout) logs a WARNING. The release endpoints only free stored media of a finished job: they do not stop generation, and the job is still billed, so the job is left for you to collect. The `cancel()` / `release()` and `__aexit__` docstrings on `VideoJob`, `Video`, `MusicJob` and `Music` describe this.

- **`chat.completions.create(e2ee=TeeOptions(verifier=...))` no longer warns that no client-side quote verification is done.** The attestation-trust `UserWarning` is still emitted on every call that uses the baseline verifier.

- **Importing `venice_ai.auth.x402` no longer prints four `abnf` `GrammarWarning`s.** The filter is scoped to the `siwe` import and leaves process-wide warning settings alone.

- **Documentation matches the real signatures.** `ImageUpscaleRequest.scale` and `image.upscale()` say any value from 2 to 4 inclusive is valid, not only 2 or 4. Docstring examples that called non-existent APIs now call real ones: `resolve_video_upscale` used `client.video.quote_upscale`, the `Paginator` module used `characters.iter_all(category=...)` instead of `categories=[...]`, and the `Video` / `VideoJob` examples used `duration=` instead of `duration_seconds=`.

## [2.5.1] - 2026-09-27

### Fixed

- **A wallet-authenticated client could make exactly one request.** `VeniceClient(auth=X402Auth(...))` cached the signed SIWE envelope for the lifetime of its TTL and reattached it to every subsequent request. Venice treats a SIWE nonce as single-use, so the second call — and every call after it — came back `401 This nonce has already been used`. The first request succeeded, which is what made this look like an intermittent auth problem rather than a guaranteed one. The envelope is now signed per request, for both `X402Auth` (EVM) and `SolanaX402Auth` (Solana); signing is one local elliptic-curve operation, negligible beside the request it authenticates.

  Retries replayed spent nonces too, through two independent paths. The retry middleware re-sends the *same* request object on a retryable 5xx (`500`, `502`, `503`, `504`), headers included, so a retried attempt carried the envelope the first attempt had already spent. And the rate limiter retries a `429` by re-invoking the request callable, which resent the envelope `client.x402.balance(auth=...)` and `transactions(auth=...)` had signed once at the call site. Every attempt, retries included, now signs its own envelope — with the wallet that signed the first attempt, so a per-call `auth=` wallet is never swapped for the client's own.

  Only the SDK's own `X-Sign-In-With-X` envelope is re-signed. Bearer requests never gain one, and an envelope you build yourself and pass in `headers=` is left as you sent it. `X-402-Payment` is not re-signed either: it is a USDC transfer authorization you signed, and minting a second one on retry would authorize a second transfer. Its nonce is single-use as well, so a retried top-up is rejected rather than settled twice.

  The bundled `venice-py-x402` skill taught the same caching mistake and is corrected alongside, since it ships inside the wheel: its SIWE reference carried a `CachedSIWE` helper and advised callers to “cache freely”, both raw-HTTP examples reused one envelope across two requests, and a scored eval asked the model to implement exactly that. The reference now documents the single-use nonce and the `401`s a replay produces, the examples sign per request, and the eval checks that an `X402Auth` *instance* is reused per wallet while each request signs its own envelope.

- **A privacy mode the SDK does not know about no longer fails the entire `/models` parse.** `ModelSpec.privacy` was a closed `Literal["private", "anonymized"]`, the same failure mode 2.4.1 fixed for `ModelResponse.type`: a single catalog row carrying a third mode would raise `APIResponseValidationError` on the whole listing, taking `models.get()`, `get_capabilities()` and every `resolve_*()` helper down with it, since they all read that listing.

  The field was closed in three places, and widening only the wire model would have moved the crash one layer down rather than removing it — `get_capabilities()` reads `spec.privacy` and forwards it into `ChatCapabilities` and `GenericCapabilities`, both of which declared the same `Literal`. All three are now plain `str`.

  The known values ship as `KNOWN_PRIVACY_MODES`, importable from `venice_ai.types` and `venice_ai.types.api`. Narrowing must **fail closed**: test `privacy == "private"` rather than excluding the modes you know about, so a mode added later is never mistaken for zero-retention. `select_model(require_private=True)` already compared against `"private"` exactly and is unaffected.

  As in 2.4.1, this widens a public `Literal` to `str`, so a caller relying on exhaustiveness checking over `privacy` will see their type checker stop proving the match is exhaustive.

## [2.5.0] - 2026-09-21

### Added

- **`client.decisions` — decision ("System One") models.** `decisions.create()` calls `POST /decisions`, which evaluates a `state` against a map of typed questions and returns one structured answer per question instead of prose. Three question types are modelled: `NoulQuestion` (a yes/no judgment returned as a calibrated probability), `ChoiceQuestion` (pick one option, with the full probability distribution and a confidence) and `ScoreQuestion` (a probability-weighted position on an ordered rubric, which can land between levels). `state` accepts a string or structured data, and questions may be passed as the typed models or as plain dicts.

  Answers come back as a discriminated union — `NoulAnswer`, `ChoiceAnswer`, `ScoreAnswer` — with `DecisionResponse.noul()`, `.choice()` and `.score()` narrowing by question id so call sites do not each need an `isinstance` check. The union's fourth arm, `UnknownAnswer`, mirrors the catch-all arm in Venice's own schema: an answer type added later parses and keeps its payload rather than failing the response.

  `POST /systemone` is the same endpoint under the path TypeSafe's SDKs use, so it is reachable by pointing those at `TYPESAFE_BASE_URL=https://api.venice.ai/api`; this SDK targets `/decisions` and does not duplicate the method. The endpoint is documented as **beta** — Venice reserves the right to change its request and response schemas without notice.

- **`models.resolve_decision()`** resolves a decision model at runtime, so calling code never hardcodes a model ID. Decision models are beta-flagged today, so unlike `resolve_chat()` this shortcut does not filter beta models out — doing so would leave no candidates. `models.list(type="decision")`, `resolve(type="decision")` and the `venice-py models --type decision` / `venice-py models resolve --type decision` CLI filters accept the new type.

- **`DecisionModelSpec`** types the two token budgets decision models report in place of a context window: `maxStateTokens` (the state plus the single longest question) and `maxTotalTokens` (the state plus all questions combined).

- **`anon_user_id` on all seven endpoints that accept it**, reached through `chat.completions.create()`, `responses.create()`, `image.create()`, `image.submit()`, `image.simple_generate()`, `image.edit()`, `image.multi_edit()` and `image.background_remove()` — eight methods, because `create()` and `submit()` both post to `/image/generate`. Venice combines it with your Venice user id when attributing a request to an upstream provider. It is validated locally to the API's constraints: printable ASCII, 1-128 characters, and no `||` (the delimiter Venice joins the two ids with).

  This lands as two different changes depending on the endpoint. On the five **image** requests the field was **silently dropped** — those models take pydantic's default `extra="ignore"`, so a caller who passed it got a successful request and no attribution, with nothing to indicate it had been discarded. On **`chat/completions` and `responses`** (`extra="allow"`) it already reached the wire, but unvalidated; those calls now raise a local `ValidationError` for a value the API would have rejected with a 400. If you were passing an id that violates the constraints, you will see the failure earlier and more clearly than before.

  `anon_user_id` is **not** an alias of the OpenAI-compatible `user` field, which Venice discards. Both can be set, and they are sent as separate fields.

- **`camera_trajectory` on `video.submit()` / `video.run()`** — 2-12 `CameraKeyframe` poses describing the camera path, for H3 Max Multi-Angle. Requires `image_url`, and the output aspect ratio follows that image.

  Two of its constraints cannot be expressed in JSON Schema and so are not enforced anywhere upstream of the API: **`time` must strictly increase** across the trajectory, and **total absolute azimuth travel is capped at 32 turns**. Both are validated locally. The travel figure is cumulative and uses absolute values, so a back-and-forth path accumulates rather than cancelling out.

- **`client.voice_changer` — the `/audio/voice-changer/*` job family.** Converts an existing recording into a target voice, preserving the original timing. `run()` returns a `VoiceChangerJob` (an async context manager, like `MusicJob`/`VideoJob`) with the low-level `submit()` / `quote()` / `retrieve()` / `cancel()` underneath. The source can be a local file (uploaded as multipart) or an `audio_url` Venice fetches itself; passing both or neither raises before anything is uploaded. Also adds the `venice-py audio voice-change` CLI subcommand.

  Three things differ from the music and video job families, and reasoning by analogy from those gets each one wrong:

  * `quote(duration_seconds=...)` takes a **bare count of seconds** — `60`, not the `"60s"` form the video endpoints accept. `"60s"` is rejected.
  * `retrieve()` has **two** outcomes, not three: a `PROCESSING` JSON body or the converted audio as `audio/mpeg`, discriminated by content type. The spec declares no `FAILED` status, so `wait()` never returns a failure to branch on — a failed conversion raises an `APIError` carrying `credits_refunded`.
  * There is no download URL. The audio arrives inline on `status.data`; `VoiceChangerCompletedStatus` has no `url` or `expires_at`.

  The **source** length is the billable quantity, and the quote is only an estimate — the charge is computed from the length Venice measures when the recording is queued, reported as `duration_seconds` on the queue response and on `job.duration_seconds`.

  **Scope of verification.** The endpoints are live: a quote against a music model returns the server's own contract error. But the live catalog contains no model reporting `voice_changer: true`, and the spec's example id `elevenlabs-voice-changer` 404s, so **the queue→retrieve→complete happy path could not be exercised end to end or recorded to a cassette.** It is implemented against the spec and unit-tested with mocks. `resolve_voice_changer()` raising, and `quote()` surfacing the server's rejection, were verified live.

- **Voice-changer capability fields on `MusicModelSpec`** — `voice_changer`, `supports_background_noise_removal`, `supports_seed`, `supports_custom_voice_id`, `accepted_audio_formats` and `max_source_audio_duration_seconds`. The four booleans are typed `bool` defaulting to `False` rather than `bool | None`, following the `uncensored` precedent: the API sends them only when true, so an absent field is a definite "no" rather than "undeclared".

- **`models.resolve_voice_changer()`** — voice changing is a *capability*, not a model type. These models report `type="music"` with `voice_changer=true`, so `resolve_music()` can hand back a music generator that the voice-changer endpoints reject. This filters on the flag. It raises a `ValueError` naming the condition when no such model is in the catalog, which is the case on accounts without the capability.

### Fixed

- **`image.edit(quality=...)` no longer fails every call.** `POST /image/edit` declares `additionalProperties: false` and has no `quality` field, so the server rejected the whole request with `400 Unrecognized key(s) in object: 'quality'` — the parameter was not ignored, it was fatal. The SDK exposed `quality` on `edit()` and put it on the wire unconditionally, so **every call that set it had been failing**. Verified against the live endpoint: the identical request returns `200` without the field and `400` with it. It is now dropped before the request is built, so those calls succeed. Passing it raises a `DeprecationWarning`; the parameter is removed in 3.0.0.

  `quality` is genuinely supported on `POST /image/multi-edit`, where the SDK models it correctly — `image.multi_edit()` is the migration target. `ImageEditRequest` no longer declares the field at all, so it cannot reach the wire from a hand-built request either.

- **`venice-py models --type <t>` no longer reports "No models match" for a type it simply never fetched.** The CLI lists models by fetching each type in turn and filtering the union, but the list of types to fetch was hand-maintained and had drifted from `ModelListType`. Because the filter can only narrow what was already fetched, a missing type was indistinguishable from an empty result — `--type decision` printed "No models match the specified filters" while `jev-latest` was live in the catalog. The list is now derived from `ModelListType`, so a new type is covered as soon as it is declared.

- **`from venice_ai.resources import Decisions` now works.** The resource was wired onto `VeniceClient` but never added to the package's import block or `__all__`, so it was reachable only as `client.decisions`. Every resource attached to the client is now checked against the package exports by a test.

### Deprecated

- **`image.edit(quality=...)`** is accepted and ignored, and is removed in 3.0.0. See the entry under Fixed — the endpoint never accepted it. Use `image.multi_edit(quality=...)`.

## [2.4.1] - 2026-09-19

### Fixed

- **A model type the SDK did not know about no longer fails the entire `/models` parse.** `ModelResponse.type` was a closed `Literal`, so when Venice added the `decision` type to the live catalog, `models.list()` raised `APIResponseValidationError` on the whole 364-model listing rather than on the one unrecognised entry — taking `models.get()`, `models.get_capabilities()` and every `resolve_*()` helper down with it, since they all read that listing. A single new catalog row, added server-side with no SDK release, took working applications offline.

  `type` is now a plain `str`, matching how the SDK already treats `quantization`, model tiers and video quality. An unrecognised type parses, keeps its unmodelled `model_spec` fields, and falls back to the base `ModelSpec` instead of failing. `GenericCapabilities.type` was a second closed `Literal` with the same failure mode and is now open too; the `Capabilities` union switched to a callable discriminator so its catch-all arm can stay open, which a field discriminator does not permit.

  The known values ship as `KNOWN_MODEL_TYPES`, importable from `venice_ai.types` and `venice_ai.types.api`, for callers that want to narrow. Treat anything outside it as a type this release predates rather than as invalid.

  Note the deliberate asymmetry this introduces: *request* filters such as `models.list(type=...)` stay closed `Literal`s so a typo still fails at the call site. Only *response* fields are open, so data the SDK predates never fails a parse.

  This is a patch release because the runtime behaviour is strictly more permissive — nothing that worked before changes. It does, however, widen a public `Literal` to `str`, so a caller relying on exhaustiveness checking over `ModelResponse.type` will see their type checker stop proving the match is exhaustive. Narrow against `KNOWN_MODEL_TYPES` and handle the fallback.

## [2.4.0] - 2026-09-11

### Added

- **Every documented request field is now modelled.** A field-level coverage pass against the spec found ~25 params the SDK never sent. Image generation gains `enhance_prompt`, `disable_prompt_optimization_thinking` and `style_references` (with a `StyleReference` item model); image edit and multi-edit gain the first two. Video generation gains `omni_reference_task_type`, `reference_document_urls`, `keyframes` (with a `VideoKeyframe` item model) and the enhancement cluster used by Topaz-style models — `enhancement_model`, `compression`, `creativity`, `grain`, `halo`, `noise`, `realism`, `recover_detail`, `sharp`, `softness`, `h264_output`, `output_format`, `target_fps` and `slowdown_factor`. `video.quote()` gains the three of those that move the price: `enhancement_model`, `target_fps` and `slowdown_factor`.

- **`ImageGenerationResponse.enhanced_prompt`** returns the rewritten prompt produced by `enhance_prompt=True`, decoding it from the URL-encoded `x-venice-enhanced-prompt` response header. Without it the caller had to know the header name and unescape it themselves.

- **`modelPrivacy` now covers all three API-key paths.** It was added to key creation, but `PATCH /api_keys` and `POST /api_keys/generate_web3_key` accept it too, so `UpdateApiKeyRequest` and `Web3CreateApiKeyRequest` carry it as well and `api_keys.update()` takes a `model_privacy` argument — a key's privacy tier can be changed after creation, and Web3-created keys can set one.

- **`supportsStyleReferences` and `supportsStyleReferenceStrength`** on `ModelCapabilities`, so callers can tell which image models accept `style_references` before sending one. (There is no capability flag for `disable_prompt_optimization_thinking`; the API documents it as ignored rather than rejected by models that lack it, so it is safe to send unconditionally.)

- **`bitrate_mode` on video generation.** `video.submit()` and `video.run()` accept `"standard"` or `"high"`, selecting the output encode bitrate on public Seedance 2.0 (including Fast) and 2.5 models. Omitting it is equivalent to `"standard"`. It is queue-only and does not affect price, so `video.quote()` deliberately does not accept it.

- **`RateLimitError.is_error_budget`** distinguishes a 429 from one of the two rolling 30-second error budgets — too many failed requests, or too many requests for a feature the model does not support — from an ordinary throughput 429. Detected from the response headers, which set `x-ratelimit-resets` for a budget trip rather than the per-window `x-ratelimit-reset-requests`/`-tokens`. (The singular `X-RateLimit-Reset` that `/crypto/rpc` sets on its own 429s is deliberately not treated as a match.)

- **`model_spec.uncensored`** is now modelled on `ModelSpec`, so whether Venice classifies a model as uncensored can be read off the catalog instead of inferred from traits or model-ID substrings. Note the semantics differ from the `supports_*` capability flags in the same module: the API sends this field only when it is true and omits it otherwise, so it is typed `bool` defaulting to `False` — an absent field is a definite "not uncensored", not "undeclared".

- **`modelPrivacy` on API keys.** Keys carry a persistent privacy tier — `ALL`, `PRIVATE_TEXT`, or `PRIVATE_ONLY` — controlling which models the key may call. It is settable on `CreateApiKeyRequest` and reported on `ApiKey`. Omit it to let the account default apply.

- **`RateLimitError.custom_message`** surfaces the API's `customMessage`, which names the cap that tripped. Several distinct limits all surface as a 429 — the per-minute request cap, the per-day credit cap, and two rolling 30-second error budgets (failed requests, and requests asking a model for a feature it does not support) — and they need different responses. Previously a caller saw only "Rate limit exceeded" with no way to tell them apart. The field name is documented; its position in the body is not, so it is read from both the flat and `error`-nested shapes and is `None` when absent — this has not yet been confirmed against an observed 429.

- **`RateLimitType`** enumerates the values `RateLimitLogEntry.rateLimitType` carries (`RPD`, `RPM`, `TPM`, `FAILED_REQUESTS`, `UNSUPPORTED_FEATURE_REQUESTS`), so reading `/api_keys/rate_limits/log` no longer means hand-writing magic strings. The field stays typed `str` so a value Venice adds later still parses rather than failing the whole listing — the same arrangement as `VeniceAPIErrorCode` and `APIError.code`.

- **`VeniceAPIErrorCode.MODEL_PRIVACY_RESTRICTED`**, returned when an API key's privacy tier excludes the requested model.

- **`loop` on music generation.** `music.submit()` and `music.run()` now accept `loop`, which renders a clip whose end splices back into its start without an audible seam. The API gates it on the model, so `MusicModelSpec.supports_loop` is modelled alongside the existing capability flags and a pre-flight check raises before the request is sent when a model declares `supports_loop=false` — matching how `force_instrumental` already behaves. An undeclared capability defers to the server rather than being treated as unsupported.

### Deprecated

- **`image.upscale()`'s `enhance`, `enhanceCreativity`, `enhancePrompt` and `replication` are deprecated and ignored.** Venice removed all four from `POST /image/upscale`; the endpoint now takes exactly `image`, `scale` and `creativity`. The SDK was still sending the four dead fields on every call and had no way to set `creativity` — the only tuning knob left. Passing any of them now raises a `DeprecationWarning` and is ignored, which matches what the server already did with them, so no working call changes behaviour. They are removed in 3.0.0. The matching `venice image upscale` flags (`--enhance/--no-enhance`, `--enhance-creativity`, `--enhance-prompt`, `--replication`) behave the same way.

  Migrating: replace `enhanceCreativity` with `creativity`, noting the range differs — the server clamps `creativity` to 0–0.02 (default 0.01), not 0–1. The SDK forwards whatever you pass rather than rejecting out-of-range values, because the server clamps rather than erroring. `enhancePrompt` has no replacement; the endpoint no longer accepts a prompt.

### Fixed

- **`image.upscale()` now sends `creativity` and a valid `scale`.** `scale` must be `2` or `4`; the old validator steered callers toward `scale=1` + `enhance=True`, a combination the API now always rejects, and `scale=1` is rejected outright. Neither previously produced a successful call.

- **Source-matched video duration is no longer rejected before it is sent.** Seedance reference-to-video edit and extend can ask the output to follow the source clip — `duration="auto"` (or `"-1"`), alongside `aspect_ratio="adaptive"` (or `"auto"`). Those sentinels are not members of a model's `constraints.durations` enum, so the duration pre-flight compared them against it and raised `duration_seconds='auto' is not supported by model ...`, rejecting a request the API accepts. The sentinels are now exempt from the enum check; any other unrecognised duration is still rejected.

- **Retry backoff no longer outlasts the caller's own timeout.** `SimpleRateLimiter` never consulted `RequestMetadata.timeout` when sleeping between retries, so a caller who set a 5s timeout could sit in backoff for 30s or more and receive neither the response nor their timeout on schedule. Backoff is now accumulated and checked against that timeout: once the next sleep would exceed it, the `RateLimitError` is raised instead. A `timeout` of `None` keeps the old unbounded behaviour.

- **A 429 from an error budget now backs off for the full window.** These budgets exist to stop clients retrying into a wall, and they clear only when their rolling 30-second window rolls. `SimpleRateLimiter` applied its ordinary schedule instead — with the default `min_backoff=1.0` that is roughly 1s, 2s and 4s, so all three retries landed inside the window and were guaranteed to fail. Worse, for the failed-request budget each of those retries counted against the budget again. An error-budget 429 now backs off at least `SimpleRateLimiter.ERROR_BUDGET_WINDOW_SECONDS` (30s); throughput 429s are unaffected. Worst-case latency is bounded by `RequestMetadata.timeout` (see above); with `timeout=None` and the default `max_retries=3` an error-budget 429 can hold a call for up to 90s, where the old schedule gave up after ~7s having never had a chance of succeeding.

### Changed

- **`venice api-keys create` gained `--model-privacy`** (`ALL`, `PRIVATE_TEXT`, `PRIVATE_ONLY`), so the CLI can set the same privacy tier the SDK already exposed. Unset leaves the account default in place.

- **Video prompts accept up to 20,000 characters.** `prompt` and `negative_prompt` on the video request models were capped at 10,000, which is now half the API's limit — the SDK was rejecting prompts the API accepts. Purely a widening; no previously valid request changes behavior.

- **Embeddings input is text-only.** `input` was typed `str | list[str] | list[int] | list[list[int]]`, advertising token-ID arrays that the API now rejects with a validation error. The signature is narrowed to `str | list[str]`, and passing a token-ID array raises `InvalidRequestError` before the request is sent rather than as a bare pydantic "Input should be a valid string". No working call changes: token-ID arrays never produced an embedding — they previously failed upstream, and the API now rejects them outright. Callers passing token IDs must pass the source text instead.

## [2.3.0] - 2026-09-11

### Added

- **`ChatStream.reasoning_summary`** exposes a streamed model's plain-language reasoning with any encrypted reasoning block removed. The assembled `message.reasoning_content` must keep that block, because the API requires it back verbatim on the next turn, so rendering that field in a thinking panel shows users several KB of opaque token. Use `reasoning_summary` for display and `reasoning_content` for the round trip. `None` until the stream is consumed, and for models that emit no reasoning.

- **`ChatCompletionChunkChoiceDelta.reasoning_encrypted`** models the flag the API sets on the single streaming delta whose `reasoning_content` is an encrypted reasoning block. Clients can now tell that block apart from the summary deltas around it without parsing the block's sentinel header.

### Fixed

- **Streamed encrypted reasoning blocks now survive the round trip.** Reasoning models emit an encrypted reasoning item — an opaque, *unterminated* block the API requires back verbatim on the next turn — and mark the single delta carrying it with `reasoning_encrypted`. `ChatCompletionChunkChoiceDelta` did not model that field, so pydantic discarded it and both `ChatStream.collect()` and `collect_with_deltas()` joined every reasoning delta in arrival order. When a model emitted summary text *after* the block (routine on a long think), that prose was welded onto the end of the block, and echoing the assembled `reasoning_content` back — the standard multi-turn tool-calling pattern — failed at request setup with `400 The encrypted content for item rs_… could not be verified`. The delta now models `reasoning_encrypted`, the collectors accumulate flagged blocks separately, and the assembled `reasoning_content` places them last, preserving every delta in the shape the non-streaming endpoint already returns. The assembled message is flagged `reasoning_encrypted` to match, and the new `ChatStream.reasoning_summary` exposes the plain-language reasoning without the block — rendering `reasoning_content` in a thinking panel otherwise shows users several KB of opaque token.

- **`venice_parameters=` now type-checks with a plain dict.** Both `chat.completions.create()` overloads annotated the parameter `VeniceParameters | None`, which is narrower than what the code accepts: a mapping of the same fields has always been validated and coerced by `ChatCompletionRequest`, but type checkers rejected the call — and because `create()` is overloaded, the diagnostic reported the whole call as unmatched (`No overload variant of "create" ... matches argument types`) without naming the offending argument. The annotation is now `VeniceParameters | Mapping[str, Any] | None`. `ChatCompletionRequest.venice_parameters` deliberately stays model-only; it is the validation boundary where mappings are coerced. Note that `VeniceParameters` is `extra="allow"`, so an unrecognized key is carried through rather than rejected — identically for either form. Reported alongside the `messages=` variant in [#1](https://github.com/sethbang/venice-py/issues/1), tracked as [#58](https://github.com/sethbang/venice-py/issues/58).

## [2.2.1] - 2026-09-01

### Changed

- **`abnf` is now capped at `<2.9`** on the `x402` extra. `siwe` accepts `abnf >=2.2,<3`,
  but `abnf` 2.9.0 made redefining an RFC 5234 core rule a hard error, and `siwe`'s own
  `rfc5234` grammar redefines `ALPHA`. The result was that `import siwe` raised
  `GrammarError` outright, taking the whole SIWE and x402 authentication path with it.
  Constrained here rather than waiting on `siwe`, which is already at its latest release.

- **Dependencies refreshed.** 34 packages moved, notably `cryptography` 50.0.1, `click`
  8.5.0, `idna` 3.19, `pydantic` 2.13.5, `dcap-qvl` 0.6.3 and `filelock` 3.32.5.

  `hexbytes`, `eth-abi`, `rlp` and `eth-rlp` crossed major versions because `web3` 7.16.0
  and `eth-account` 0.13.7 declare them with no upper bound. That mixed cohort is safe
  here for a specific reason rather than by luck: this SDK imports nothing from `web3`,
  `hexbytes`, `eth_abi` or `rlp` directly, and the one `hexbytes` 2.0 behaviour change
  that does reach it — `.hex()` no longer returning a `0x` prefix — is already normalised
  at every call site. EIP-712 typed-data signing and SIWE verification were both checked
  end to end against the new cohort.

  `web3` 8, `eth-account` 0.14, `websockets` 17 and `chardet` 7 stay where they are:
  `siwe` 4.4.0 pins `web3 <8` and `eth-account <0.14`, `web3` 7 pins `websockets <16`, and
  `cyclonedx-bom` 7.3.1 pins `chardet <6`. All three are already at their latest releases,
  so these are ecosystem ceilings, not deferred work.

- **`adaptive-rate-limiter` now requires `>=1.3.0`** (was `>=1.1.0`). Venice meters requests
  but not tokens, and the limiter could not represent that. Two layers had to change for the
  adaptive path to work against this API at all.

  The header gate in INTELLIGENT mode scored all six `x-ratelimit-*` headers as one pool and
  demanded all six before syncing anything. Venice sends three, so every response fell through
  to release-only: the backend was never called, the bucket was never verified, and the limiter
  ran on fabricated cold-start limits for the life of the process. It failed silently — the
  release path returns success and logs at debug, so nothing ever surfaced. Measured on a live
  model: 427 cold-start probes across 66 minutes with the bucket never once written. The gate
  is now assessed per dimension, so a request-only provider syncs the dimension it reports.

  Underneath that, reset headers could not be parsed. Venice sends both reset stamps as absolute
  epoch milliseconds; the library rewrote its own clean integer through `str(float(...))`, then
  failed to parse the result, substituted `0`, and had its Lua reject the update on a post-2020
  sanity floor. Absent counts were separately defaulted — remaining to `0`, limit to a fabricated
  fallback — a pairing no server reports. Absent and genuinely-zero values are now distinguished,
  and an incomplete dimension is skipped rather than sinking the whole update.

  `1.2.x` is excluded deliberately rather than incidentally: `1.2.0` clamps an out-of-range token
  window into range, which rotates it early and refills the token count before the server does,
  and `1.2.1` still cannot get past the header gate on this API.

- The README now carries a short note explaining that the package installs as `venice-py`,
  that imports and `VENICE_API_KEY` are unchanged, and that the `venice-ai` bridge declares
  no extras — so `venice-ai[cli]` and friends need the name updated.

- **The repository moved to [`sethbang/venice-py`](https://github.com/sethbang/venice-py).**
  GitHub permanently redirects the old location, so existing links, clones and remotes keep
  working — no action needed. The project URLs shown on PyPI update with the next release.
  The import package is still `venice_ai` and `VENICE_API_KEY` is unchanged; this is the last
  step of the distribution rename, and it changes nothing about what gets built or installed.

### Fixed

- **Dict-form multimodal content is coerced into typed content objects again.** Passing
  `UserMessage(content=[{"type": "text", ...}])` returned plain `dict` parts instead of
  `TextContent`/`ImageContent`/`AudioContent`/`VideoContent`/`FileContent`, so attribute
  access such as `msg.content[0].text` raised `AttributeError`. Serialization — and
  therefore the request sent to the API — was unaffected.

  `MessageContentPartParam` unions the discriminated `MessageContentPart` with `TypedDict`
  mirrors of the same shapes, which exist purely so callers can pass plain dicts without
  type-checker complaints. Pydantic's smart mode picks a union member by score rather than
  by order, and the mirrors describe the same shapes by construction, so the two members
  were only ever separated by a scoring tiebreak; `pydantic` 2.13.5 adjusted that scoring
  and the `TypedDict` members began winning. The union is now pinned to
  `union_mode="left_to_right"`, which removes the dependency on scoring entirely and
  behaves identically on 2.13.4 and 2.13.5.

- **Local test runs no longer record cassettes — and no longer spend API credit — by
  default.** The VCR record mode defaulted to `NEW_EPISODES` outside CI, which replays what
  a cassette holds and sends anything it lacks to the live Venice API. Cassettes are
  gitignored, so a fresh clone has none and the first `make test` billed whoever's
  `VENICE_API_KEY` was configured, silently. Recording is now opt-in through the existing
  `VENICE_VCR_RECORD` variable (`all` re-records, `new` fills gaps); anything else replays
  only, and a request with no cassette raises instead of reaching the network.
  `make test-fresh` and `make test-quick` delete cassettes in order to re-record, so they
  now pass the opt-in themselves and announce that they spend credit.

- **Every VCR call site now resolves its record mode from one place**
  (`tests/vcr_policy.py`). The benchmark suite built its own module-level `vcr.VCR` with a
  hardcoded `RecordMode.ONCE`, under a module global that shadowed the `vcr_config` fixture
  name. `VENICE_CI_MODE=true` is only read inside that fixture, so the documented guarantee
  that CI never records did not hold for that module, and it recorded whenever a cassette
  was absent.

- **Benchmark tests now look for their cassettes where they actually live.** The cassette
  directory was chosen by test path, special-casing only `tests/e2e/`, so tests under
  `tests/benchmarks/` resolved to `tests/integration/cassettes` and could never match a
  cassette — every request either went live or failed outright. Both fixtures that made
  this choice now share one helper.

- **The adaptive rate-limiter saturation benchmark now enforces its own pass criterion.**
  It computed whether the contended p50 stayed within 50% of the control p50, wrote that
  verdict into a report, and returned without asserting it, so the test could not fail on
  the property it exists to measure. It now asserts the criterion — and asserts that
  latency samples were collected at all, since an empty sample set drove the computed delta
  to zero and produced a pass.

- **`__version__` no longer falls back to a plausible-looking version string.** When the
  package metadata cannot be read — a source tree with nothing installed — `__version__`
  now reports `0.0.0+unknown` instead of a hardcoded release number. A fallback spelled
  like a real version makes a broken metadata lookup indistinguishable from a working one,
  which is how the pre-rename lookup went unnoticed while `User-Agent` reported the wrong
  version on every request. The lookup also narrowed from `except Exception` to
  `except PackageNotFoundError`, so unrelated failures surface instead of being swallowed.

- **`venice-py configure` reads the default config path at call time.** It previously bound
  `DEFAULT_CONFIG_PATH` at import, so redirecting the config location reached
  `venice_ai.cli.config` but not the `configure` command, which kept using the path captured
  when the module was first imported.

- Four version headings in this file (`2.0.1`, `2.0.2`, `2.1.0`, `2.2.0`) were written as
  links but had no link definition, so they rendered as literal bracketed text.

### Changed

- **`make check-all` now runs pyright.** The `pyright (project)` job gates every PR, but no
  `make` target invoked it, so the first signal for a pyright-only finding was a red required
  check. mypy does not stand in for it — it does not report a name bound only inside a `try`
  block being referenced from that block's `except` clause, which pyright does.

- `make lint`, `make format`, and `make format-check` now cover `tools/` and `benchmarks/` in
  addition to `src/` and `tests/`. `format-check` runs in CI, so both are gated rather than
  merely formatted by hand. Clearing `benchmarks/` took 92 non-behavioural ruff fixes —
  whitespace, import ordering, and `List`/`Dict`/`Optional` rewritten to builtin generics.
  Every Python directory in the repo is now linted; `examples/` is covered by `examples.yml`.

## [2.2.0] - 2026-08-20

### Changed

- **The PyPI package is now `venice-py`, renamed from `venice-ai`.** v2.1.0 renamed the CLI binary for the same reason: Venice's official tooling already owns the `venice` name, and a community SDK sitting on `venice-ai` invites people to mistake it for an official release. Renaming the distribution finishes what the CLI rename started.

  ```bash
  pip install venice-ai    # before
  pip install venice-py    # after
  ```

  **Your code does not change.** The import package is still `venice_ai`, and `VENICE_API_KEY` is still `VENICE_API_KEY`:

  ```python
  from venice_ai import VeniceClient   # unchanged
  ```

  A distribution name that differs from its import name is ordinary in Python — `pillow` imports as `PIL`, `python-dotenv` as `dotenv`. Renaming the import package would have broken every existing `import venice_ai` for no benefit, so it was left alone. `VENICE_API_KEY` names the *service* rather than this package, so sharing it with Venice's own tooling is deliberate: one key works everywhere.

  Update the dependency wherever it is pinned — `requirements.txt`, `pyproject.toml`, lockfiles, Dockerfiles, CI installs. Extras are unaffected apart from the name: `pip install 'venice-py[cli]'`, `[x402]`, `[redis]`, `[adaptive]`, `[e2ee]`.

  `venice-ai` remains on PyPI permanently and is **not** yanked, so existing lockfiles keep resolving exactly as they do today.

  `venice-ai` 2.1.1 ships alongside this release as a bridge: metadata-only, with `venice-py` as its single dependency, so `pip install venice-ai` still lands a working install — it just arrives under the new name. It had to be published second, since it depends on a `venice-py` that must already exist on PyPI.

  Having both installed at once is safe: the bridge ships no importable module, so there is no `venice_ai/` directory for the two to fight over.

- **Pinning `>=2` is no longer necessary.** The old advice existed because a bare `pip install venice-ai` on Python ≤3.12 silently resolved to v1.3.x, which supported Python ≥3.11. No v1 line was ever published under `venice-py`, so there is no wrong version to land on — on an unsupported Python, pip now reports that no matching distribution exists, which is the failure the pin was engineered to force. `pip install venice-py` is enough.

  If you are staying on v1 for now, keep pinning `venice-ai<2`; the v1 line exists only under the old name.

- **The bundled skills are now `venice-py`, `venice-py-multimodal`, `venice-py-production` and `venice-py-x402`.** `venice-py skills install` removes the superseded `venice-ai*` directories it finds in the target `.claude/skills/`, so upgrading does not leave both generations installed and triggering against each other. A directory is only removed when its `SKILL.md` identifies it as one of ours, so a directory of your own that happens to share a name is left alone.

- **CLI data has moved from `~/.venice/` to `~/.venice-py/`.** `~/.venice/` collides with Venice's official CLI, which may legitimately own that path.

  The first `venice-py` command that reads the directory copies `config.yaml`, `conversations/` and `presets/` across and prints a one-line notice. Nothing is lost and nothing needs doing by hand.

  The old directory is **left exactly as it was** — it may hold the official CLI's data, and deleting another tool's files would be worse than leaving a stale copy behind. Remove it yourself once you are satisfied nothing else needs it. Only the three subpaths listed above are copied; anything else in `~/.venice/` stays put.

  Permissions are tightened rather than merely preserved: `conversations/` is narrowed to `0700` and `config.yaml` to `0600`, since transcripts hold prompt and response text and the config may hold a plaintext API key.

- **The `User-Agent` sent with every request is now `venice-py/<version>`**, previously `VeniceAI-Python-SDK/<version>`.

- **The CLI now identifies itself as `venice-py`.** `venice-py --version` printed `Venice AI CLI v<version>` and `venice-py --help` opened with `Venice AI CLI - Your AI assistant in the terminal.` Renaming the command in v2.1.0 stopped the `PATH` collision but left the tool still introducing itself as Venice's, which is the confusion the rename exists to remove. Both surfaces now say `venice-py`, and `--help` states plainly that this is the unofficial, community-maintained CLI rather than Venice's official `venice`.

### Fixed

- **`__version__` no longer reports a stale version.** It is resolved from the installed distribution's metadata, and that lookup sat inside a bare `except Exception` that fell back to a hardcoded literal. Any mismatch between the looked-up name and the built distribution therefore froze `__version__` silently — and `User-Agent` is derived from it, so every request would have misreported the version. The lookup now tracks the distribution name, with a test asserting the two cannot drift apart again.

## [2.1.0] - 2026-08-14

### Changed

- **The CLI command is now `venice-py`, renamed from `venice`.** Venice's official CLI ([`veniceai-cli`](https://www.npmjs.com/package/veniceai-cli), published March 2026) installs a binary named `venice`, and so did this SDK's `[cli]` extra as of v2.0.0. With both installed, which one ran depended on `PATH` order — typically this SDK's inside an activated virtualenv and Venice's outside it. Because the two share subcommand names (`chat`, `image`, `video`, `embeddings`, `models`, `characters`), the wrong tool would run and reject the flags rather than report the collision. The two are unrelated programs and no longer contend for the name.

  Update any scripts, aliases, and CI steps that invoke `venice`:

  ```bash
  venice chat start        # before
  venice-py chat start     # after
  ```

  Shell completions must be regenerated, and their environment variable is now `_VENICE_PY_COMPLETE`:

  ```bash
  venice-py completion zsh >> ~/.zshrc
  ```

  Upgrading in place does not remove the old script. If `venice --version` still reports a Venice **AI CLI** banner after upgrading, a stale entry point is left over from the v2.0.x install — delete it from the environment's `bin/` directory (`rm "$(command -v venice)"` while that environment is active), or the collision persists.

  This SDK is unofficial and community-maintained. For Venice's official CLI, see [veniceai/venice-cli](https://github.com/veniceai/venice-cli).

## [2.0.2] - 2026-08-14

### Fixed

- **Getting Started example raised `TypeError`.** The first code sample in the documentation site's Getting Started page constructed a message positionally (`UserMessage("Hello, Venice!")`), which Pydantic rejects with `BaseModel.__init__() takes 1 positional argument but 2 were given`. The message models take keyword arguments only; the sample now reads `UserMessage(content="Hello, Venice!")`.
- **Getting Started gave an installation sequence that could not work.** The Claude Code skills section instructed readers to run `venice skills install` after a plain `pip install venice-ai`. The `venice` CLI ships behind the optional `[cli]` extra, so that sequence fails on the install command. The section now installs `'venice-ai[cli]>=2'` first.

### Changed

- **Documented installation commands now carry a `>=2` version floor and are quoted.** On Python 3.12 and below, a bare `pip install venice-ai` resolves to v1.3.x silently — pip backtracks to the newest release whose `Requires-Python` matches, with no warning — so a reader following v2 documentation on an older interpreter would install v1 and hit confusing import errors. `pip install 'venice-ai>=2'` instead produces an explicit failure naming the cause (`Ignored the following versions that require a different python version: 2.0.2 Requires-Python >=3.13`). The specifiers are also quoted throughout because `[...]` is a glob in zsh, where an unquoted `pip install venice-ai[cli]` fails with `no matches found`. Applies to the README, the documentation site, and the bundled skills. To stay on v1, pin `venice-ai<2`.
- **The Python 3.13 requirement is stated at the point of installation.** It previously appeared only in the README's Requirements section, several hundred lines below the Quick Start, and below the install block on the Getting Started page.

## [2.0.1] - 2026-08-13

### Fixed

- **`messages=` now type-checks with plain dicts.** The parameter was annotated `Sequence[UserMessage | AssistantMessage | SystemMessage | ToolMessage | DeveloperMessage]`, which is narrower than what the code actually accepts: mappings in the OpenAI wire shape (`{"role": "user", "content": "hi"}`) have always been validated and coerced into the corresponding message model, but type checkers rejected them (`error: List item 0 has incompatible type "dict[str, str]"`). The annotation is now the public `ChatMessageParam` union, so both forms check cleanly on `create()`, `stream()`, `parse()`, `estimate_cost()`, and `run_with_tools()`. The typed models remain the documented idiom — they give completion and validation at construction — and malformed mappings still raise `ValidationError` before the request is sent. Reported in [#1](https://github.com/sethbang/venice-py/issues/1).
- **`estimate_cost()` and `run_with_tools()` accept mapping messages.** Both read the message list before it reaches the request model, so dict input previously raised `AttributeError` on `.content` (`estimate_cost`) or left raw dicts in the returned `ToolLoopResult.messages` history (`run_with_tools`). Messages are now normalized at the method boundary.

### Added

- **`ChatMessageParam`** (exported from `venice_ai.types`) — the union describing what `messages=` accepts: any of the five message models, or a plain `Mapping[str, Any]`. Use it to annotate your own message-building helpers. Note that `ChatCompletionRequest.messages` deliberately stays model-only; it is the validation boundary where mappings are coerced.

## [2.0.0] - 2026-08-12

### Breaking Changes

- **Python ≥3.13 now required** — drops support for Python 3.11 and 3.12 (v1.3.x's supported range). The SDK targets `python = ">=3.13,<4.0"`.
- **Client classes restructured — `VeniceClient` is now async-by-default.** In v1 `VeniceClient` was the *synchronous* client and `AsyncVeniceClient` the async one. In v2 `VeniceClient` IS the async client, the synchronous client is `SyncVeniceClient` (use `with SyncVeniceClient() as client:`), and `AsyncVeniceClient` has been removed. Migration: replace `from venice_ai import AsyncVeniceClient` (now an `ImportError`) with `VeniceClient` (now async); replace synchronous `VeniceClient` usage with `SyncVeniceClient`, or `await` the now-coroutine methods. Note: v2 `VeniceClient` no longer supports the synchronous `with` protocol, so v1 `with VeniceClient() as client:` code raises a `TypeError` at the `with` line until migrated. The `venice lint` V100 rule flags leftover `AsyncVeniceClient` references.
- **`client.image.generate(...)` → `client.image.create(...)`**, and **`client.image.get_available_styles()` → `client.image.list_styles()`**. No deprecation aliases — the old names raise `AttributeError`. (`client.image.simple_generate(...)` is unchanged.) The new async-job resources `client.video` and `client.music` (see Added) use a consistent verb scheme — `submit()` (low-level), `run()` (high-level managed Job), and `cancel()` (cleanup).
- **Responses are now typed Pydantic models.** Endpoints that returned `TypedDict`s in v1 now return Pydantic models, so subscript access (`resp["data"]`) must become attribute access (`resp.data`). Affects `client.embeddings.create`, `client.models.list` / `list_traits` / `list_compatibility`, and `client.audio.get_voices`; field names are preserved. Additionally:
  - `client.audio.create_speech(...)` now returns an `AudioResponse` (raw bytes on `.content`), not `bytes`.
  - `client.api_keys.retrieve()` and `delete()` now return typed models (`ApiKey`, `DeleteApiKeyResponse`) instead of untyped `dict`s; read fields by attribute (`api_key.description`). `create()` and the rate-limit / web3 helpers are likewise typed.
  - `chat` and `characters` responses were already typed in v1 and are unaffected.
- **`client.billing.get_usage(...)` → `client.billing.get_usage_history(...)`.** The Venice `/billing/usage` endpoint was deprecated upstream (rate-limited to 1 RPM; returns 410 for accounts created on/after 2026-07-07) in favour of `/billing/usage-history`, which uses cursor (keyset) pagination. The `get_usage()` and `iter_usage()` methods are removed; use `get_usage_history()` and `iter_usage_history()`. Parameters were renamed (`startDate`/`endDate` → `startTimestamp`/`endTimestamp`, `limit` → `pageSize`) and `page`/`sortOrder` are gone (the walk is always ascending by timestamp). The response is now `BillingUsageHistoryResponse` (`.data` + `.nextCursor`) rather than `BillingUsageResponse` (`.data` + `.pagination`); the `BillingUsageResponse`, `BillingPagination`, and `BillingUsageQueryParams` types are removed (replaced by `BillingUsageHistoryResponse` and `BillingUsageHistoryQueryParams`), and the usage entry `currency` is now typed `Literal["USD", "DIEM", "BUNDLED_CREDITS"] | str` (the spec's three current values, while legacy values such as `"VCU"` on historical rows still round-trip). A continuation request must send only the cursor — filters travel inside it. `client.billing.get_balance` and the beta `get_usage_analytics` are unchanged.
- **`client.get_model_pricing(model_id)` removed.** Pricing is no longer a dedicated client method; it is read off the model entry that already carries it — `(await client.models.get(model_id)).model_spec.pricing` returns that model's pricing object — an `LLMModelPricing` for chat and embedding models, exactly what the v1 method returned — without the extra round trip. To price a whole catalog at once, `CostTracker.from_client(client)` builds the `{model_id: pricing}` map in one `models.list()` call.
- **Type modules moved under `venice_ai.types.api`.** Per-resource type modules (`images`, `models`, `api_keys`, `billing`, `characters`, `embeddings`) moved from `venice_ai.types.*` to `venice_ai.types.api.*`; update direct imports. Several response classes were also renamed (e.g. `ChatCompletion` → `ChatCompletionResponse`, `ImageResponse` → `ImageGenerationResponse`).
- **`max_tokens` parameter removed** — use `max_completion_tokens` instead. The legacy `max_tokens` parameter has been fully removed; code still using it must migrate to `max_completion_tokens`.

### Added

- **Bundled Claude Code skills + `venice skills` CLI** — four skills (`venice-ai`, `venice-ai-multimodal`, `venice-ai-production`, `venice-ai-x402`) now ship as package data under `venice_ai/skills/` and install into `.claude/skills/` via `venice skills install` (project scope by default; `--global` targets `~/.claude/skills/`), with `venice skills list` and `venice skills uninstall`. The bundled skills steer Claude Code toward idiomatic v2 patterns (dynamic model resolution, `async with stream:`, `run_with_tools`, `client.gather(...)`) instead of OpenAI-style or v1 code.
- **`image.edit(quality=...)`** — opt-in `quality` (`"low" | "medium" | "high"`) on `client.image.edit(...)` and `ImageEditRequest`, for quality-aware edit models (e.g. gpt-image-2-edit) per the edit docs. Sent only when set (model-dependent), mirroring `multi_edit`.
- **`ResponsesUnknownOutput`** — a permissive catch-all output variant (exported from `venice_ai.types`) so an unmodeled `/responses` output-block `type` is preserved instead of failing the parse.
- **`SolanaX402Auth.build_header()`** — Ed25519 Sign-In-With-X (SIWS) header signing so Solana wallets can authenticate for the x402 read endpoints (`client.x402.balance` / `transactions`), mirroring `X402Auth.build_header` for EVM. `balance()` / `transactions()` now accept `X402Auth | SolanaX402Auth`. Live-verified against `GET /x402/balance`.
- **`client.tee.get_signature(model=, request_id=)`** — `GET /tee/signature`, the per-request integrity proof of the TEE flow: fetches the cryptographic signature attesting a specific completion was produced by the verified enclave (pairs with `get_attestation`). New `TeeSignatureResponse` / `TeeReceipt` / `TeeReceiptEvent` / `TeeReceiptSignature` / `TeeSignatureVerification` models (`venice_ai.tee.types`). Response shape live-captured (the endpoint is undocumented in the swagger).
- **Video face-media consents (Seedance)** — `client.video.run(...)` / `submit(...)` now accept `consents` (a `VideoConsents` / `SeedanceConsents` model, or an equivalent dict), serialized into the `POST /video/queue` body. The Venice API requires `consents.seedance.{confirmed_terms_and_privacy, confirmed_legal_right, confirmed_screening_acknowledged}` (each must be `True`) when submitted media contains faces, returning a 409 `needs_consent` otherwise; this was previously unrepresentable via the SDK. New `VideoConsents` / `SeedanceConsents` models exported from `venice_ai.types`.
- **x402 Solana settlement** — `SolanaX402Auth` (`venice_ai.auth.x402_solana`) + `client.x402.top_up_with_solana(...)` add USDC-on-Solana top-ups alongside the existing EVM/Base path. Builds the x402 "exact" SVM payment (a partially-signed `VersionedTransaction`; the facilitator sponsors gas via `feePayer`), fetches blockhash/mint context over raw JSON-RPC (`VENICE_X402_SOLANA_RPC_URL`, default mainnet-beta), and sends the x402 **V2** envelope (`{x402Version, payload, accepted}`). Requires the `x402-solana` extra (`pip install 'venice-ai[x402-solana]'`, pulls `solders`). Live-verified end-to-end against Venice's facilitator.
- **Native image `quality` parameter** — `client.image.create(...)` (and `submit(...)`) now accept `quality="low" | "medium" | "high"` for quality-aware models (e.g. GPT Image 2); higher values can increase the request charge. Distinct from the OpenAI-compat `simple_generate(quality=...)` enum. Image model specs now expose `qualities` and `defaultQuality` so callers can discover support.
- **Video reference audio** — `reference_audio_urls` (up to 3 URLs/data-URLs) for reference-to-video models (e.g. Seedance 2.0 R2V), accepted by `client.video.run(...)` and `client.video.submit(...)`; validated for URL shape and capped at 3, matching the API. Live-verified end-to-end against `seedance-2-0-fast-reference-to-video`.
- **Video reference video (R2V)** — `reference_video_urls` (up to 3 URLs/data-URLs) on `client.video.run(...)` / `submit(...)` and `reference_video_total_duration` on `client.video.quote(...)`, for Seedance 2.0 reference-to-video models. Mirrors the `reference_audio_urls` plumbing; the documented fields were previously absent from the SDK (and `VideoQuoteRequest` was `extra="forbid"`, blocking any workaround).
- **API-key billing period (`limitPeriod`)** — `client.api_keys.create(...)`, `update(...)`, and the Web3 create path now accept `limit_period` / `limitPeriod` (`"EPOCH" | "MONTH" | "LIFETIME"`), so MONTH/LIFETIME keys can be created via the SDK. The field is also now surfaced on the returned `ApiKey` model (see Fixed).
- **Model reasoning-effort discovery** — `ModelCapabilities` now surfaces `reasoningEffortOptions` (the accepted `reasoning_effort` tiers for a model) and `defaultReasoningEffort`.
- **Model deprecation metadata** — v2 adds a `ModelDeprecation` type (`date`, `replacementModelId`, `removesAt`, `startsAt`, `autoRemap`) exposed via `ModelSpec.deprecation`, mirroring the API's model-object deprecation fields. (v1 surfaced no model-level deprecation metadata.)
- **`client.image.simple_generate(...)`** — a thin wrapper around the OpenAI-compat `/images/generations` endpoint, returning a typed response.
- **`SyncVeniceClient`** — synchronous wrapper around `VeniceClient`, backed by a dedicated background event-loop thread. Use `with SyncVeniceClient() as client:` for codebases that don't (or can't) use `asyncio`. Stream results are auto-wrapped so they iterate synchronously.
- **Unified model resolution API** — `client.models.resolve()` plus type-specific shortcuts `resolve_chat()`, `resolve_embedding()`, `resolve_image()`, `resolve_video()`, `resolve_tts()`, `resolve_asr()`, `resolve_inpaint()`, and `resolve_cheapest_video()` — a single, capability-filtered call, replacing hardcoded model IDs.
- **`ChatStream` with convenience accessors** — subclass of `Stream` returned from chat completions that adds:
  - `text_deltas()` — yields only text content, filtering empty/None deltas
  - `collect()` — consumes the stream and assembles a complete `ChatCompletionResponse`
  - `client.chat.completions.stream(...)` shorthand
  - `Stream.__aenter__` / `__aexit__` for `async with` lifecycle management
- **`VideoJob` lifecycle manager** — `client.video.run()` returns a `VideoJob` that handles the submit → poll → wait → download → cleanup lifecycle, including `async with` semantics that guarantee server-side cleanup. Low-level `submit()` / `quote()` / `retrieve()` / `cancel()` remain available.
- **`venice_ai.helpers` module**:
  - `tool_from_function(fn)` — generate a `Tool` definition from a Python function's type hints
  - `tool_from_model(BaseModelSubclass)` — generate a `Tool` from a Pydantic model
  - `Conversation` — chainable builder for multi-turn message lists
- **`client.audio.stream_long_text(...)`** and the underlying `venice_ai.audio_helpers.stream_long_text(...)` — splits long inputs into sentence-aligned segments, dispatches them in parallel, and yields concatenated mp3 bytes in input order. Works around two server-side issues confirmed against live Venice on 2026-05-14:
  - `tts-qwen3-0-6b` / `tts-qwen3-1-7b` cap output at exactly 15.896875 s (664 MP3 frames @ 24 kHz) regardless of input length. A 12-line poem renders as the first stanza-and-a-half without the helper; with it, the full poem renders.
  - Six of ten Venice TTS models buffer the full response before sending any bytes (qwen3 family, orpheus, chatterbox, inworld, gemini). Parallel fan-out converts that buffering into perceived progressive streaming because later segments are in flight while the first is still generating.
  - Per-model word budget lives in `venice_ai.audio_helpers.MODEL_WORD_BUDGETS`. mp3 only; other formats raise `NotImplementedError`. Inputs that fit under the budget pass through to a single `create_speech` call (no extra overhead). Known limitation: per-segment voice timbre drift on qwen3 — Venice does not currently accept a `seed` parameter on `/audio/speech`. See module docstring for the empirical test results that led to no `temperature`/`top_p` defaults being baked in.
- **Pydantic models accepted as `response_format`** — pass a `BaseModel` subclass directly to `client.chat.completions.create(response_format=MyModel)` to enable structured output without hand-writing JSON Schema.
- **`ChatCompletionResponse.parsed` and `parse_as(model)`** — convenience accessors for structured output: `response.parsed` returns parsed JSON; `response.parse_as(MyModel)` returns a validated Pydantic instance.
- **`.save()` / `.save_all()` on response types** — `ImageGenerationResponse.save("out.png")`, `ImageGenerationResponse.save_all(directory)`, and `AudioResponse.save("speech.mp3")`. Replaces manual base64 decoding and file writes in user code.
- **Message role defaults** — `UserMessage`, `AssistantMessage`, `SystemMessage`, and `ToolMessage` now have `role` defaulted to the appropriate literal. Construct with `UserMessage(content="…")` instead of `UserMessage(role="user", content="…")`.
- **`AssistantMessage.from_response()`** — class method that extracts an `AssistantMessage` from a `ChatCompletionResponse` for multi-turn history.
- **`VideoGenerationError`** — new exception raised by `VideoJob.wait()` when the server reports a failed generation. Carries the server-provided `error_code`.
- **Expanded top-level re-exports** — `Tool`, `ToolFunction`, `ChatStream`, `VideoJob`, `SyncVeniceClient`, `Conversation`, `tool_from_function`, `tool_from_model`, `RetryOptions`, `RateLimitInfo`, `DeprecationInfo`, `BalanceInfo`, `TextContent`, `ImageContent`, `ImageUrl`, `StreamOptions`, `VeniceParameters`, `JSONSchemaFormat`, `UserMessage`, `SystemMessage`, `AssistantMessage`, `ToolMessage`, `ChatCompletionResponse`, `ChatCompletionChunk`, `ChatUsage`, `ImageGenerationResponse`, `AudioResponse`, `CreateApiKeyRequest`, and `VideoGenerationError` are now importable directly from `venice_ai`.
- **Better authentication error message** — `VeniceClient()` raises a more actionable error when no API key is found (mentions both the env var and the constructor argument).
- **`client.responses.create()`** — wraps the OpenAI-compatible `POST /responses` endpoint (tagged Alpha in the Venice docs). Returns a typed `ResponsesResponse` whose `output` array is a discriminated union of `ResponsesReasoningOutput`, `ResponsesMessageOutput`, `ResponsesFunctionCallOutput`, and `ResponsesWebSearchCallOutput`. The resource accepts `model`, `input` (string or list of structured items), plus `include`, `max_output_tokens`, `temperature`, `top_p`, `reasoning`, `tools`, `tool_choice`, `web_search`, and `venice_parameters`. `ResponsesRequest` and the response / output / usage types are exported from `venice_ai.types`; `ResponsesRequest` and `ResponsesResponse` are additionally re-exported at the top level. A new `_RE_RESPONSES` classifier pattern routes the endpoint into `ResourceType.LLM`. Streaming (`stream=true` SSE) is documented server-side but not yet wrapped by this resource.
- **Request-classifier patterns for the new Feb 2026 models** — `qwen-image` and `seedream*` now route to `ResourceType.IMAGE`, and `gpt-*` (e.g. `gpt-5.3-codex`) routes to `ResourceType.LLM`.
- **Typed `reasoning_effort` enum** — `client.chat.completions.create()` now accepts the full March 2026 reasoning-effort enum as a typed parameter: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. The `ReasoningEffortLevel` alias is exported from the top level.
- **Nested `reasoning` config object** — `client.chat.completions.create()` and its underlying request type now accept a `reasoning=ReasoningConfig(effort=..., summary=...)` parameter for the nested reasoning configuration (`summary` is one of `"auto"` / `"concise"` / `"detailed"`). Top-level `reasoning_effort` still takes precedence over `reasoning.effort` when both are set, per API spec. `ReasoningConfig` and `ReasoningSummary` are exported from the top level.
- **Request-classifier pattern for `/image/background-remove`** — the background-removal endpoint is now explicitly registered under `ResourceType.IMAGE` instead of falling through to default routing.
- **Music generation** — five new methods on `client.music` wrapping the March 2026 `/audio/queue|quote|retrieve|complete` family: `submit()`, `quote()`, `retrieve()`, `cancel()`, and the high-level `run()` which returns a `MusicJob`. `MusicJob` mirrors `VideoJob`'s context-managed lifecycle (queue → poll → download → cleanup). New types (`MusicQueueRequest`, `MusicQuoteRequest`, `MusicRetrieveRequest`, `MusicCompleteRequest`, `MusicQueueResponse`, `MusicQuoteResponse`, `MusicProcessingStatus`, `MusicFailedStatus`, `MusicCompletedStatus`, `MusicCompleteResponse`, `MusicRetrieveResponse`) are exported from `venice_ai.types`. New `MusicGenerationError` for failed jobs.
- **`client.models.resolve_music()`** — type-specific shortcut paralleling `resolve_tts()` / `resolve_asr()`. Also adds `"music"` to the supported `type` values on `resolve()` itself.
- **`ResourceType.MUSIC`** — new queue-classifier bucket so music generation (which shares the `/audio/*` path prefix with TTS/ASR) gets its own rate-limit queue. Classifier patterns cover `audio/queue`, `audio/quote`, `audio/retrieve`, `audio/complete` plus the launch-day model IDs (`elevenlabs-music`, `elevenlabs-sound-effects`, `ace-step`, `minimax-music`, `stable-audio`, `mmaudio`).
- **`client.video.transcribe()`** — wraps the new April 2026 `POST /video/transcriptions` endpoint. Accepts a public video URL (e.g. YouTube) and an optional `response_format` of `"json"` (default) or `"text"`. Returns a `VideoTranscriptionResponse` (`transcript`, `lang`) for JSON or a plain `str` for text. `VideoTranscriptionRequest` and `VideoTranscriptionResponse` are exported from `venice_ai.types`. A matching `_RE_VIDEO_TRANSCRIPTIONS` pattern routes the endpoint to `ResourceType.VIDEO` in the request classifier.
- **`client.characters.reviews()`** — wraps the new `GET /characters/{slug}/reviews` endpoint. Supports `page` / `page_size` pagination and returns a `CharacterReviewsResponse` with the page `data`, `pagination`, and aggregate `summary`. New types `CharacterReview`, `CharacterReviewsPagination`, `CharacterReviewsSummary`, and `CharacterReviewsResponse` are available from `venice_ai.types.api.characters`.
- **Characters public API expansion** — `Character` now exposes `id`, `author`, `featured`, and `isOwner` (the latter populated only on authenticated requests). `CharacterStats` gains `averageRating`, `ratingCount`, `ratingSum`, and `userRating`. All new fields are optional, so existing constructors stay backwards-compatible.
- **Typed filter kwargs on `client.characters.list()`** — `categories`, `is_adult`, `is_pro`, `is_web_enabled`, `limit`, `model_id`, `offset`, `search`, `sort_by`, `sort_order`, and `tags` are now first-class keyword arguments. List-valued filters are sent as comma-separated query values. `extra_query` is still accepted and merged last for anything not yet modelled.
- **`enable_web_search` on `client.image.create()`** — new optional boolean kwarg matching the `enable_web_search` body field documented for `POST /image/generate`. Forwarded verbatim when set; omitted otherwise. `ImageGenerationRequest` gains the matching optional field.
- **Chat-completion passthrough fields** — `client.chat.completions.create()` now accepts `prompt_cache_retention` (`"default"` / `"extended"` / `"24h"`) plus the OpenAI-compat passthroughs `store`, `text`, `include`, and `metadata`. All five are forwarded verbatim in the request body when set. Matching optional fields were added to `ChatCompletionRequest`. `prompt_logprobs` on `ChatCompletionResponse` is kept.
- **Advanced fields on `client.video.submit()`** — seven new body fields to match the full Venice video API surface: `upscale_factor` (`Literal[1, 2, 4]`, for the `topaz-video-upscale` model), `end_image_url`, `audio_url`, `video_url`, `reference_image_urls` (up to 9), `elements` (up to 4; Kling O3 R2V structured characters), and `scene_image_urls` (up to 4). These live on `VideoRequestBase` so the T2V and I2V request models both accept them. New `VideoElement` Pydantic model available from `venice_ai.types.api.requests.video`. Prompt/negative-prompt length ceiling raised to 10,000 chars (swagger `maxLength`; was previously capped at 3,500 client-side, blocking valid long prompts). `VideoElement` also carries a per-element `video_url` (Kling O3 R2V) and caps its inner `reference_image_urls` at 3, matching the API.
- **Dynamic-temperature sampling on chat completions** — `max_temp`, `min_temp`, and `min_p` are now first-class keyword arguments on `client.chat.completions.create()`, matching the documented body fields for `POST /chat/completions`. Previously the values could only reach the API via the untyped `**kwargs` passthrough.
- **`client.augment` resource** — new top-level namespace wrapping the three experimental `/augment/*` endpoints:
  - `client.augment.scrape(url=...)` — POST `/augment/scrape` returns a page as markdown.
  - `client.augment.search(query=..., limit=..., search_provider="brave" | "google")` — POST `/augment/search`.
  - `client.augment.parse_text(file=..., response_format="json" | "text")` — POST `/augment/text-parser` with multipart upload (PDF/DOCX/XLSX/TXT, ≤ 25 MB).
  New types `AugmentScrapeRequest`, `AugmentScrapeResponse`, `AugmentSearchRequest`, `AugmentSearchResult`, `AugmentSearchResponse`, and `AugmentTextParserResponse` are available from `venice_ai.types.api.augment`.
- **`client.x402` resource** — wraps the three `/x402/*` wallet-billing endpoints with a new optional `x402` extra (install via `pip install venice-ai[x402]`, pulling in `eth-account` and `siwe`):
  - `client.x402.balance(auth=...)` — GET `/x402/balance/{walletAddress}` using SIWE (EIP-4361) wallet auth.
  - `client.x402.transactions(auth=...)` — GET `/x402/transactions/{walletAddress}`, same SIWE auth.
  - `client.x402.top_up(payment_header=...)` — POST `/x402/top-up`. Standard Bearer auth, with an optional `X-402-Payment` header for pre-signed payment payloads. An empty call returns the documented 402 Payment Required with structured payment requirements.
  New `venice_ai.auth.x402.X402Auth` builds the base64-encoded `X-Sign-In-With-X` header from a wallet private key; the wallet address is derived automatically. Types (`X402BalanceData`, `X402BalanceResponse`, `X402TopUpData`, `X402TopUpResponse`, `X402Transaction`, `X402TransactionsData`, `X402TransactionsPagination`, `X402TransactionsResponse`) are exported from `venice_ai.types`.
- **`venice_parameters.enable_e2ee` / `enable_x_search` (request)** — these documented request fields are now accepted on the `VeniceParameters` request model, so callers can opt into TEE end-to-end encryption or X (Twitter) search from the SDK.
- **TEE client-side end-to-end encryption (E2EE)** — full client-side encryption for Venice confidential-compute (`e2ee-*`) chat models, with the wire path live-verified end-to-end.
  - **`client.tee` resource** — `client.tee.get_attestation(model=..., nonce=...)` fetches and **baseline-verifies** a TEE attestation from the free `GET /tee/attestation` endpoint (does not require the `[e2ee]` extra); `client.tee.open_session(model=...)` verifies fail-closed and returns a `TeeSession` that produces the `X-Venice-TEE-*` request headers and encrypts/decrypts messages. Available on both the async and sync clients.
  - **Functional `enable_e2ee` / `create(e2ee=True)`** — `client.chat.completions.create(..., e2ee=True)` (or setting `venice_parameters.enable_e2ee=True`) now runs the real flow: verify the model's attestation (fail-closed), encrypt each user/system message to the attested model key, force a wire stream with the three `X-Venice-TEE-*` headers, and decrypt the streamed response locally (reassembling a normal `ChatCompletionResponse` when `stream=False`). Pass a `TeeOptions` (exported from `venice_ai.tee`) instead of `True` to control the attestation freshness nonce or supply a `FullQuoteVerifier`. Tool calling, web search/scraping, and multimodal (image/file) content are rejected with `InvalidRequestError` before any network call, because they cannot stay inside the encrypted channel.
  - **New `[e2ee]` extra** — `pip install 'venice-ai[e2ee]'` pulls in `cryptography`. Baseline *attestation* works on a bare install; only the encrypting session (key generation / message encryption / response decryption) requires the extra, which is imported lazily and raises a clear `ImportError` with the install hint when missing.
  - **Protocol** — secp256k1 ECDH key agreement (raw 32-byte X shared secret) → HKDF-SHA256 (`info=b"ecdsa_encryption"`) → AES-256-GCM (12-byte nonce, 16-byte tag). Each encrypted message uses a fresh per-message ephemeral keypair; the response is decrypted with the session keypair whose public half rode in the `X-Venice-TEE-Client-Pub-Key` header.
  - **SECURITY LIMITATION (read before relying on it):** the *default* attestation path is **baseline**. It checks the server-side `verified` claim, the nonce echo, and the TDX report-data / signing-address binding, and rejects TDX debug flags — but on its own it **trusts Venice's server-side `verified` claim and does NOT perform full client-side Intel TDX quote verification.** A malicious Venice operator forging a self-consistent attestation would not be detected by the baseline alone. For full client-side Intel TDX verification, supply a `DcapTdxVerifier` (see below) via the `FullQuoteVerifier` extension point (`TeeOptions(verifier=...)`); the raw `intel_quote` / `nvidia_payload` evidence is retained on the attestation for it. A one-time `UserWarning` is emitted on every E2EE-engaged `create` call when no verifier is supplied. (NVIDIA GPU attestation via NRAS is still not shipped.)
  - `TeeOptions`, the `TeeSession` class, and the typed exceptions `TeeError` / `TeeAttestationError` / `TeeEncryptionError` (all subclass `VeniceError`) are exported from `venice_ai.tee`. `TeeAttestation` lives in `venice_ai.tee.types`; the `FullQuoteVerifier` protocol and the `DcapTdxVerifier` implementation are the documented extension point for full quote verification.
- **Full client-side Intel TDX attestation verification (`DcapTdxVerifier`)** — a concrete, fail-closed `FullQuoteVerifier` that closes the baseline's trust gap entirely **offline**, exported from `venice_ai.tee`. It verifies the raw `intel_quote`'s ECDSA signature and PCK certificate chain to a **pinned, baked-in** Intel SGX Root CA, evaluates the FMSPC TCB status against Intel-signed collateral, confirms the enclave is non-debug, that the E2EE signing key is bound into REPORTDATA, that the attestation event log replays to the quoted RTMRs, and — on the dstack attestation wire — that the `app_compose` binds to the quoted compose hash. (On Venice's current `attestation.evidence` wire the attestation carries no compose hash, so compose binding is reported as `unavailable` rather than passing, and workload identity is pinned via `expected_measurements` / `mr_config_id` — see **Fixed**.) The `dcap_qvl.parse_quote` policy bytes are read **only after** `verify_with_root_ca` passes for the same raw quote (the #1 correctness gate). Pass it as `client.tee.open_session(model=..., verifier=DcapTdxVerifier(...))` / `get_attestation(..., verifier=...)` or via `chat.completions.create(e2ee=TeeOptions(verifier=...))`.
  - **New `[e2ee-verify]` extra** — `pip install 'venice-ai[e2ee-verify]'` pulls in `dcap-qvl` (+ `cryptography`). The dependency is imported lazily via a `_require_dcap()` helper that raises a clear `TeeError` with the install hint when absent; it **never silently skips** (a silent skip would degrade back to baseline trust). Verified to import and verify on arm64.
  - **Security tier.** By default this proves the model runs on a *genuine, non-debug Intel TDX enclave* running a *self-consistent dstack workload* (**Tier B**). It does **NOT** independently prove this is the legitimate Venice image/app — there are no published reference measurements today — unless the caller supplies `expected_measurements` / `expected_compose_hash` from a source independent of the Venice endpoint, which upgrades those dimensions to **Tier A**.
  - **TCB-status policy.** Fail-closed **reject by default**: only `UpToDate` passes; `OutOfDate` / `Revoked` / `SWHardeningNeeded` / `ConfigurationAndSWHardeningNeeded` are rejected. An opt-in `tcb_policy="advisory"` mode accepts the hardening-needed statuses while surfacing their advisory IDs. Construct directly with a `dcap_qvl.QuoteCollateralV3` snapshot for airgapped/offline verification, or use `DcapTdxVerifier.with_fetched_collateral(...)` (the one network touch) to fetch collateral from the no-auth PCCS.
- **`DeveloperMessage` role** — new message class for the `role: "developer"` message role documented for chat completions (used by OpenAI-compatible reasoning models). Extends the `messages` union on `ChatCompletionRequest`. Exported from `venice_ai` and `venice_ai.types`.
- **`aspect_ratio` on image generate + edit** — new optional kwarg on `client.image.create()` and `client.image.edit()` matching the `aspect_ratio` body field documented for `POST /image/{generate,edit}`. Forwarded verbatim when set; omitted otherwise. Matching optional field added to `ImageGenerationRequest` and `ImageEditRequest`. Supported values vary by model — inspect `GET /models` for per-model allowed ratios.
- **`prompt` / `temperature` / `top_p` on `client.audio.create_speech()`** — three new optional kwargs matching the documented body fields on `POST /audio/speech`: style `prompt` (Qwen 3 TTS; max 500 chars), sampling `temperature` (0–2; Qwen 3 / Orpheus / Chatterbox HD), and `top_p` (0–1; Qwen 3 TTS). Forwarded verbatim when set; omitted otherwise. Ignored by models that don't advertise `supportsPromptParam` / `supportsTemperatureParam` / `supportsTopPParam`.
- **`language` on `client.audio.create_speech()` / `AudioSpeechRequest`** — optional language hint matching the documented body field on `POST /audio/speech`. Accepted formats are model-specific (Qwen 3 / MiniMax full names; xAI / ElevenLabs ISO 639-1 codes). Unsupported values are silently ignored by the server.
- **`safe_mode` on `client.image.edit()`** — optional kwarg matching the documented body field on `POST /image/edit`. Defaults to the server-side default (`True`, i.e. adult-content blur enabled) when left unset; pass `False` to disable blurring on adult-capable edit models.
- **Capability introspection fields on `ModelCapabilities`** — five new bools always returned by `GET /models?type=text` are now typed on `ModelCapabilities`: `supportsMultipleImages`, `supportsReasoningEffort`, `supportsTeeAttestation`, `supportsE2EE`, `supportsXSearch`. Default to `False` for backward-compatibility with older cached responses.
- **`CompletionTokensDetails` type** — new Pydantic model exported from `venice_ai.types` mirroring `PromptTokensDetails`. Carries `reasoning_tokens`, `audio_tokens`, and `image_tokens` for completion-side breakdowns. `ChatUsage` gains a typed `completion_tokens_details: CompletionTokensDetails | None` field plus a top-level `cache_read_input_tokens` integer. Populated by reasoning models (`openai-gpt-54-mini`, `grok-4-20`, etc.) so callers can read `response.usage.completion_tokens_details.reasoning_tokens` instead of dropping to raw dicts.
- **Cache-write token count** — `PromptTokensDetails` now models `cache_creation_input_tokens` (the swagger-documented premium cache-write count), with a symmetric top-level `ChatUsage.cache_creation_input_tokens` mirroring `cache_read_input_tokens`. Previously the cache-write count was silently dropped, so callers following the prompt-caching guide couldn't read it through the typed model.
- **`output_format` on `client.image.edit()`; `aspect_ratio` / `output_format` / `quality` on `client.image.multi_edit()`** — documented body fields on `POST /image/{edit,multi-edit}` that the SDK didn't expose. Forwarded when set, omitted otherwise.
- **CLI feature parity** — several `venice` subcommands gained flags/commands that the SDK already supported: `characters` server-side `--search` plus `--sort-by/--sort-order/--tags/--limit/--offset/--adult/--pro/--web-enabled/--model-id`, a new `venice characters reviews <slug>` command, and fuller `info --json`; `venice image generate` gained `--aspect-ratio/--resolution/--quality/--enable-web-search` and a new `venice image multi-edit` subcommand; `venice api-keys create` gained `--type/--description/--limit-usd/--limit-diem/--limit-vcu/--limit-period/--expiry`, `venice account keys update` gained `--limit-period`, and new `venice account keys rate-limits` / `rate-limit-logs` commands; `venice video generate` / `from-image` gained `--audio/--reference-image-urls/--reference-video-urls/--end-image-url`.
- **`limit` / `offset` on `client.x402.transactions()`** — optional pagination kwargs matching the documented query parameters on `GET /x402/transactions/{walletAddress}` (server defaults: `limit=50`, `offset=0`; valid `limit` range 1–100). Previously the method always fetched a single page with no way to paginate.
- **`X402Auth.build_payment_header(requirement, ...)`** — new instance method that constructs the EIP-712 typed data for a USDC `transferWithAuthorization`, signs with the wallet's private key, and base64-encodes the v2 `X-402-Payment` envelope. Validates the requirement's `network`, `asset`, and `amount` against caller-supplied expectations (`validate_network`, `validate_asset`, `max_amount_units`) BEFORE signing — refuses to sign payloads that deviate. Currently supports USDC on Base mainnet (`eip155:8453`); the `USDC_BASE_MAINNET` constant is exported alongside `X402Auth`. Required input is the dict from `PaymentRequiredError.body["accepts"][i]`. Returns the header string ready for `client.x402.top_up(payment_header=...)`.
- **`client.x402.top_up_with(auth=..., amount_usdc=..., max_amount_usdc=...)`** — one-call wrapper that performs the full x402 v2 probe-sign-submit flow: POST `/x402/top-up` with no header (probe), catch `PaymentRequiredError`, pick the first `"exact"` requirement on Base mainnet, validate against `amount_usdc` and `max_amount_usdc`, build the `X-402-Payment` header via `auth.build_payment_header(...)`, and re-POST with the signed header. Replaces ~50 lines of manual EIP-712 + EIP-3009 signing in user code with a single async call. `max_amount_usdc` defaults to `amount_usdc` (refuses to sign if the server requests more); pass `max_amount_usdc=None` only if you want to disable the cap.
- **`X402Auth.ttl_seconds` and `chain_id` properties** — promoted from internal `_ttl_seconds` / `_chain_id` to public read-only properties so callers (and SIWE-token caches in user code) can introspect TTL and chain without poking at private attributes.
- **`VeniceClient(auth=...)` — SIWE/SIWX-only authentication (Mode 2).** The constructor now accepts an optional `auth` parameter for wallet-based authentication via [`X402Auth`](src/venice_ai/auth/x402.py) (EVM, EIP-4361 SIWE) or [`SolanaX402Auth`](src/venice_ai/auth/x402_solana.py) (Solana, Ed25519 SIWX), broadening Mode 2 beyond the EVM-only path — `SolanaX402Auth` was previously usable only for the `/x402/*` reads. Live-verified end-to-end: a Solana wallet with no API key authenticated a real chat completion. When set with no `api_key`, the SDK skips `Authorization: Bearer` and attaches a cached `X-Sign-In-With-X` header on every request — debiting the wallet's prepaid Venice ledger instead of an account-level API key. Token cache uses `auth.ttl_seconds - 30s` (safety margin) so we don't re-sign on every call. When both `api_key` and `auth` are set, the API key wins for default request auth; the auth instance is retained for explicit per-call `auth=` kwargs (e.g., `client.x402.balance(auth=auth)`). When neither is set, the constructor now raises `ValueError` with a message that lists all three options (env var, `api_key=`, `auth=`).
- **`venice lint <path>` CLI subcommand** — AST-based linter for v1 / OpenAI-style / non-idiomatic Venice patterns in user code (`AsyncVeniceClient` imports, hardcoded model IDs, `max_tokens=` kwargs, `PaymentRequiredError.payment_instructions` accesses, etc.). Reports findings in flake8-compatible `path:line:col: CODE message` format. Supports `--code` filtering and `--strict` (promotes informational findings to errors). Exit 0 on clean, 1 on findings. The visitor implementation lives in [`venice_ai.cli.utils.lint_rules`](src/venice_ai/cli/utils/lint_rules.py) and is importable for tooling integrations (e.g., a future `ruff` plugin). See [`docs/cli.md`](docs/cli.md#lint) for the full rule-code table.
- **`venice health` CLI subcommand** — connectivity / balance diagnostic. Default checks: API key presence (via env var or saved config), `client.models.list(type="text")` reachability, and `client.billing.get_balance()`. Optional `--full` adds a tiny `client.embeddings.create(input="ping")` call to verify the embedding endpoint; `--wallet` (with `--wallet-env`) adds an x402 prepaid-ledger balance read using `X402Auth`. Exit 0 if every check passes, 1 if any fail. Output respects `--plain`. See [`docs/cli.md`](docs/cli.md#health).
- **`venice skills` CLI** — `venice skills install` copies the four bundled Claude Code skills into `./.claude/skills/` (or `~/.claude/skills/` with `--global`); `venice skills list` shows them with install state; `venice skills uninstall` removes them. The skills now ship as package data under `venice_ai/skills/`, so a plain `pip install venice-ai` is all that's needed.
- **Claude Code skills** under [`src/venice_ai/skills/`](src/venice_ai/skills) — four skills (`venice-ai`, `venice-ai-multimodal`, `venice-ai-production`, `venice-ai-x402`) that auto-load in [Claude Code](https://docs.claude.com/en/docs/claude-code) when their trigger contexts match (e.g., "venice chat", "venice image", "venice x402"). They steer Claude toward idiomatic v2 code (dynamic `resolve_*()`, `async with stream:`, `run_with_tools`, `client.gather(max_concurrency=N)`, `client.x402.top_up_with(...)`) instead of OpenAI-style or v1 patterns. 24 reference files (~4,400 lines) cover every method, error class, and migration; the SKILL.md files themselves stay under the 500-line skill-creator guidance. CI (`make skills-check`, `.github/workflows/skills.yml`) validates SKILL.md size, that every `examples/foo/bar.py` reference resolves, and that every Python code block in skill markdown lints clean against `venice lint`. See [`tools/skills/README.md`](tools/skills/README.md).
- **`ConsumptionLimits.vcu`** — the legacy `vcu` field (Diem predecessor) is now round-tripped on API key consumption-limit payloads. The docs note VCU is being phased out but the API still accepts it, and incoming responses could previously drop the value.
- **`VideoElement` accepted in `client.video.submit()` / `.quote()`** — the `elements` kwarg on the public video methods now accepts either typed `VideoElement` instances or raw dicts (previously only `list[dict]`). Serialized shape is unchanged.
- **Crypto RPC response headers now exposed.** `client.crypto.rpc()` and `client.crypto.batch_rpc()` surface the four billing/idempotency headers documented at `api-reference/endpoint/crypto/rpc.md` (`X-Venice-RPC-Credits`, `X-Venice-RPC-Cost-USD`, `X-Request-ID`, `Idempotent-Replayed`) via typed properties `rpc_credits`, `rpc_cost_usd`, `venice_request_id`, and `idempotent_replayed`. `JsonRpcResponse` now inherits `VeniceBaseModel` (so it also picks up the standard `headers`, `request_id`, `response_rate_limits` accessors), and a new `BatchJsonRpcResponse` wrapper carries the per-batch headers — exported from `venice_ai.types.api`.
- **`VideoTranscriptionRequest` / `VideoTranscriptionResponse` re-exported at top level.** Both types were already exported from `venice_ai.types.api` but missing from the top-level `venice_ai.types` namespace, so `from venice_ai.types import VideoTranscriptionResponse` failed while every other video type worked. Now consistent with queue / quote / complete / retrieve.
- **CLI parity flags** — `venice image edit` gains `--aspect-ratio` / `--resolution` / `--output-format` / `--safe-mode` (the SDK `edit()` already accepted them); `venice models --type` is now a `click.Choice` of the real model types and fetches `video` / `asr` / `music` (previously `--type video|asr|music` was silently accepted and returned nothing despite live models of those types); `venice video generate` / `from-image` gain `--reference-audio-urls`; `venice account usage` gains `--currency`; `venice characters list` gains `--categories`.
- **`tools/skills/check_skill_symbols.py`** — a CI checker (wired into `make skills-check` and `.github/workflows/skills.yml`) that statically verifies every SDK symbol, constructor/method keyword, and attribute referenced in skill markdown actually exists in the SDK. It conservatively skips anything it can't resolve (e.g. `extra="allow"` models, `**kwargs` signatures) for a zero-false-positive guarantee.

### Fixed

- **Model IDs are no longer lowercased (they're case-sensitive).** `ModelId`/`QueueId` normalization was applying `.lower()`, so a mixed-case model id from `GET /models` (e.g. `wai-Illustrious`) was silently rewritten to `wai-illustrious` — which the case-sensitive inference endpoints reject with `404 Specified model not found`. This broke the intended dynamic-resolution flow (`await client.models.resolve_image()` → `client.image.create(model=...)`) and even mangled a correct id passed explicitly (the request-body `model` field is also `ModelId`). Normalization now trims whitespace only and preserves case.
- **`x402.top_up_with_solana` now selects the Solana requirement by CAIP-2 id.** It matched the 402 `accepts` entry by the bare network string `"solana"`, but Venice's live challenge sends the CAIP-2 mainnet id `solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp`, so selection raised `RuntimeError` before signing and the Solana top-up path was unreachable. Requirement selection and pre-sign validation now accept the pinned mainnet CAIP-2 id **and** the legacy bare `"solana"`, echo the server's network value back verbatim, and reject any other `solana:*` cluster fail-closed — a payment path must never be steered to a different cluster (exact match, deliberately not a `solana:` prefix match). Verified end-to-end with a real on-chain USDC mainnet settlement.
- **`DcapTdxVerifier.verify()` now handles Venice's current attestation wire.** Full Tier-B verification read the measurement evidence from the old dstack `info.tcb_info` shape, but Venice migrated to an `attestation.evidence` envelope (event log carried as a JSON string; no `compose_hash` / `app_compose`), so `verify()` failed closed on every live attestation and the advertised `[e2ee-verify]` guarantee was unreachable in production. A wire-schema normalizer now maps both shapes onto one code path: signature / PCK-chain / TCB, the non-debug check, the REPORTDATA key binding, and the event-log→RTMR replay all run on the current wire. Compose identity cannot be established from the new wire, so `last_result["checks"]["compose_binding"]` is the string `"unavailable"` rather than a pass — the `checks` map is now `dict[str, bool | str]`, so compare a check with `is True`, never for truthiness — and workload identity is pinned there via `expected_measurements` (`mr_config_id`). Unsigned body fields (`os_image_hash`, `repo_commit`) are exposed as informational metadata only, never as verifiable measurements. Verified live against an entitled `e2ee-*` model.
- **Forward-compat hardening on response models.** Models whose swagger schema has no `additionalProperties: false` now use `extra="allow"` so a future server field is preserved rather than raising: `VeniceParametersResponse` (already grew `enable_x_search` live), the music retrieve/quote/complete models, the characters list/detail/reviews wrappers, and `Balances` (new currencies). Closed `Literal`s on `/models` response constraints were relaxed to open `str` (mirroring the `quantization` policy) so a new server value can't crash the `/models` parse: `ImageModelConstraints.defaultQuality`/`qualities`, `VideoModelConstraints.model_type` (and the derived `VideoCapabilities.model_type`), and `ModelCapabilities` reasoning-effort fields.
- **Chat response fidelity.** `ChatCompletionResponse.choices` is now optional (`default_factory=list`) — swagger marks it non-required ("certain models may not return this field") — and the `.parsed`/`.parse_as` accessors guard an empty list. `web_search_citations` was a phantom always-empty top-level field; it is now a read-only property delegating to `venice_parameters.web_search_citations` (where the API populates it).
- **Responses API robustness.** An unknown `/responses` output-block type no longer fails the whole parse (see `ResponsesUnknownOutput`), and `/responses` now strips the chat-only `venice_parameters` keys (`strip_thinking_response`, `disable_thinking`, `return_search_results_as_documents`, `enable_x_search`) the shared model carries but the endpoint does not document.
- **Audio upload content-types.** OGG was detected by magic bytes but absent from the content-type map (fell back to `application/octet-stream`); WebM wasn't detected at all. Added `.ogg`/`.oga`/`.webm` content types and a WebM/EBML magic-byte branch.
- **`augment.search`/`scrape` surface response headers.** Both responses were plain `BaseModel` with no header accessor, discarding the documented `X-Balance-Remaining` header; switched to `VeniceBaseModel` (`.headers`) with `extra="allow"`.
- **Over-required `/models` fields relaxed.** `ImageModelPricing.generation` → optional (swagger marks only `upscale` required; upscale-only models omit it) and `ModelSpec.name` → optional (swagger does not require it).
- **Rate-limit reset-header parsing consolidated.** `SimpleRateLimiter._parse_reset_time` used a `>1e11` millisecond threshold diverging from the canonical `ms_epoch_to_seconds` (`>=1e12`, mirrored by `VeniceBaseModel._ms_to_seconds`); now normalises absolute epochs via the canonical helper while preserving relative delta-seconds.
- **Docs corrected.** `VeniceAPIErrorCode` docstring (Venice errors are non-uniform: bare string, Zod `{details,issues}`, or top-level `response["code"]` — not `error.code`); `ADVANCED.md` rate-limiting (real `RateLimiterConfig`/`RateLimiterMode`, not the nonexistent `SchedulerConfig` API); skill `headers-and-metadata.md` (`PaginationInfo` real fields `page`/`limit`/`total`/`total_pages`).
- **Async video-job endpoints now classify as `VIDEO`.** The request classifier only mapped `video/transcriptions` to `ResourceType.VIDEO`; the async generation lifecycle (`video/queue`, `video/quote`, `video/retrieve`, `video/complete`) fell through to the LLM default, mis-categorising those requests for queue/rate-limit routing. Added endpoint patterns so they route correctly (mirroring the `/audio/*` music family).
- **`ImageGenerationResponse` tolerates forward-compatible fields.** It inherited `extra="forbid"`, but the `/image/generate` 200 schema has no `additionalProperties: false`, so a server-added top-level field would raise `APIResponseValidationError`. Switched to `extra="allow"` to preserve unknown fields. `SimpleImageGenerationResponse` stays strict — `/images/generations` does declare `additionalProperties: false`.
- **README video example used a nonexistent `VideoJob.save()`.** Corrected to `status = await job.wait()` then `await job.download("canals.mp4", status)`, matching the real `VideoJob` API.
- **`client.image.upscale(timeout=...)` now actually applies.** The `timeout` kwarg was accepted on the signature but never forwarded to the request, so long upscales could still hit the default timeout. It is now threaded through `_request_multipart` (a bare `float` is normalised to `aiohttp.ClientTimeout(total=...)`). `client.image.edit(...)` gained a matching `timeout` parameter (forwarded to the JSON request) for parity — aligning with the API's longer edit/upscale processing timeouts.
- **Stale `simple_generate(quality=...)` docstring corrected** — it claimed the flag was "passed through but unused"; quality is now honoured by quality-aware models (e.g. GPT Image 2) with pricing tied to resolution and quality.
- **`client.billing.get_balance()` response shape** — the live API returns a nested shape `{canConsume, consumptionCurrency, balances: {diem, usd}, diemEpochAllocation}`, but `BillingBalanceResponse` was modelling a flat `{diemBalance, usdBalance, totalDiemEpochAllocation}`. Every field resolved to `None` against the real endpoint. The model now matches the API — read balances via `response.balances.diem` / `response.balances.usd`, the allocation via `response.diem_epoch_allocation`, plus `response.can_consume` and `response.consumption_currency` (`"USD" | "VCU" | "DIEM" | "BUNDLED_CREDITS"`).
- **`client.image.multi_edit(model=...)` now reaches the API** — previously the `model` kwarg was accepted on the signature but silently dropped before sending. Per the docs for `POST /image/multi-edit`, it is now forwarded as the `modelId` body field so users can target specific edit models (e.g. `qwen-edit`, `flux-2-max-edit`) instead of always getting the server default.
- **`client.image.edit(model=...)` now reaches the API** — same class of bug as `multi_edit`: the `model` kwarg was accepted but rewritten to `None` before the request was built, so every call fell back to the server default (`qwen-edit`) regardless of what the caller asked for. The payload now forwards the caller's `model` verbatim. Stale comments claiming `/image/edit` rejects a `model` field have been removed — the live endpoint validates the field and requires `len >= 1` when supplied.
- **`ModelsQueryParams.type` / `ModelTraitsQueryParams.type` description strings** — previously listed a stale enum (`embedding, image, text, tts, upscale, inpaint, all, code`) that omitted `asr`, `music`, and `video`. The description now reflects the official docs enum (`asr, embedding, image, music, text, tts, upscale, inpaint, video`) and notes that `code` / `all` are accepted by the API but undocumented.
- **`client.augment.parse_text()` now sets the correct content type for `.pptx` uploads.** The text-parser MIME map listed `.pdf`, `.docx`, `.xlsx` but omitted `.pptx`, even though the docs list PowerPoint as a supported format. PowerPoint uploads now resolve to `application/vnd.openxmlformats-officedocument.presentationml.presentation` via filename extension.

- **CLI `--reasoning-effort` flag** — previously stuffed the value into `venice_parameters`, which has `extra="forbid"` and would reject it at request build time. The flag now passes `reasoning_effort` as a top-level parameter (matching the API spec) and accepts the full 7-value enum.
- **`client.video.quote()` signature now matches `POST /video/quote`** — the endpoint only accepts a pricing-relevant subset (`model`, `duration`, `aspect_ratio`, `resolution`, `upscale_factor`, `audio`, `video_url`), but the SDK was requiring `prompt` client-side (blocking valid calls with a pydantic `ValidationError`) and exposing `negative_prompt`, `image_url`, `end_image_url`, `audio_url`, `reference_image_urls`, `elements`, and `scene_image_urls` — fields the server silently drops. `VideoQuoteRequest` is now its own Pydantic model (not a `VideoRequestBase` subclass) with `extra="forbid"`, and `Video.quote()` no longer declares the removed parameters. Callers must use `client.video.submit()` for any of the prompt / reference-image fields. `models.selection.DynamicModelSelector.select_cheapest_video_model()` and `client.models.resolve_cheapest_video()` drop the matching `prompt=` / `image_url=` kwargs.
- **`client.image.create()` / `.edit()` prompt length no longer blocks valid long prompts** — the SDK capped `prompt` at 1,500 chars on `ImageGenerationRequest` and `ImageEditRequest`, but per the API docs the per-endpoint ceiling is 7,500 for `/image/generate` (and the effective cap is model-specific via `promptCharacterLimit` from `GET /models` — e.g. 5,000 on `gpt-image-2`, 10,000 on `imagineart-1.5-pro`). `ImageGenerationRequest.prompt` is now capped at 7,500 and `ImageEditRequest.prompt` at 32,768 to match the spec ceilings; the server enforces the model-specific limit.

- **Rate-limit reset headers parsed as Unix milliseconds.** `x-ratelimit-reset-requests` / `x-ratelimit-reset-tokens` arrive as 13-digit absolute Unix **ms**, but were parsed as seconds — `response_rate_limits.reset_requests` raised internally and always resolved to `None`, while `reset_tokens` stored the raw ms value. A magnitude-based detector now normalises ms→seconds symmetrically for both headers (live-verified against the wire).
- **Rate-limit ms normalization extended to the error and provider paths.** The same ms→seconds fix is now applied to `RateLimitError.reset_requests_timestamp` (previously stored the raw 13-digit ms — a `reset - time.time()` would be off by ~1000×) and to `VeniceProvider`'s adaptive-scheduler rate-limit parsing (both `rpm_reset` and `tpm_reset` are now treated as absolute ms-epochs; the stale "reset-tokens = relative seconds" assumption and its dead `_parse_relative_seconds` helper were removed). `RateLimitInfo.reset_tokens`'s field description was corrected from "duration in seconds" to "absolute Unix timestamp (seconds)" to match the normalized value. (Shared `ms_epoch_to_seconds` helper in `venice_ai.utils.parsing`.)
- **`CreatedApiKey` no longer drops `limitPeriod`.** The create / Web3-create response model omitted the swagger-required `limitPeriod`, so a created MONTH/LIFETIME key's period was silently dropped (the companion to the `ApiKey` fix above, which only covered the list/get model). Added.
- **Sibling pricing models no longer silently drop unknown keys.** The earlier `quality`/`upscale` fix added `extra="allow"` only to `VideoResolutionPricing`; the sibling pricing classes (`ImageModelPricing`, `InpaintModelPricing`, `LLMModelPricing`, `AudioModelPricing`, `ASRModelPricing`, `MusicModelPricing`) were still bare `BaseModel` and would drop any unmodeled live pricing key. All now preserve extras.
- **`VideoJob.download()` now works for private/VPS models.** The queue-time `download_url` was discarded at `VideoJob` construction, so for private models (where retrieve returns JSON status only) `download()` wrote nothing. It is now retained and used as the final fallback after `status.data` / `status.url`.
- **Pydantic models no longer silently drop live fields.** Several typed models dropped fields the API actually returns/accepts: `ChatCompletionRequest` now allows forward-compat passthrough kwargs (`extra="allow"`); response `ChatMessage` (and `AssistantMessage.from_response`) now carry `reasoning_details` (required to preserve thought signatures for Gemini-3-Pro-class models across `run_with_tools` turns); image-model pricing now preserves `quality`/`upscale` keys (e.g. `gpt-image-2`); `ApiKey` now surfaces `limitPeriod`, `currentPeriodUsage`, and `usage.trailingSevenDays.vcu`; and `UsageAnalyticsResponse` now surfaces the USD daily charts `byKeyDailyUsd` / `byModelDailyUsd`.
- **`client.augment.parse_text()` now sets the correct content type for `.epub` uploads.** EPUB is a ZIP container (`PK\x03\x04`) and was being sniffed as DOCX, which the server rejected ("No text content could be extracted"). `.epub` now maps to `application/epub+zip` via the filename extension (live-verified).
- **x402 EVM payments now use the V2 `accepted`-wrapper envelope.** The EVM/Base path still emitted the old flat `{x402Version, scheme, network, payload}` shape — byte-identical to the Solana shape that the facilitator rejects with HTTP 400 — and was never live-exercised. EVM now emits `{x402Version, payload, accepted}` with `maxTimeoutSeconds`, matching the Solana fix. The `eip155:8453` network value is unchanged.
- **CLI config file is no longer world-readable.** `venice configure` wrote `~/.venice/config.yaml` (which holds the plaintext API key) with `0644`; it is now `chmod 0o600` after every write (re-secures pre-existing files on the next save too). The `~/.venice` directory is now also `chmod 0o700`, and saved conversation transcripts (`~/.venice/conversations/*.json`) are `chmod 0o600`.
- **`venice --config PATH` is now honored by all subcommands.** The global `--config` file was loaded into the click context but never consulted for API-key resolution, and its `api.base_url` was never applied to the client — both were effectively decorative. `--config` now resolves the key (env var still wins) and the configured `base_url` is threaded into every command's client construction; `venice configure` reads/writes the `--config` path too.
- **`venice --plain health` no longer leaks ✓/✗ glyphs.** The health-check printer ignored plain mode despite its docstring; it now emits ASCII `[OK]`/`[FAIL]` markers when `--plain` is set.
- **`venice --version` no longer lies.** `venice_ai.__version__` and the HTTP `User-Agent` were hardcoded `2.0.0` on a `2.0.0rc1` build; both now derive from the installed package metadata.
- **`venice_parameters.enable_e2ee=True` engages real client-side E2EE.** Setting the flag (or passing `e2ee=True` to `client.chat.completions.create(...)`) runs the full client-side encryption flow described in the **Added** entry below. See **§ TEE client-side end-to-end encryption** in the Added section for the protocol and its attestation-verification limitation.

- **Model-spec capability/constraint fields no longer silently dropped.** The earlier `extra="allow"` work covered the pricing family + `ModelSpec` but did not recurse into the nested sub-objects, so `ModelCapabilities` (e.g. `maxImages`) and `ImageModelConstraints` / `InpaintModelConstraints` / `TextModelConstraints` / `VideoModelConstraints` (the documented `aspectRatios` / `resolutions` / `defaultResolution` discovery keys, video `audio_input` / `per_reference_audio` / `prompt_character_limit` / `reference_image_*`) were dropped from `GET /models` responses. All five now use `extra="allow"`. Wire-verified against the live catalog.
- **`quantization` is now a plain `str`** (on both the wire `ModelCapabilities` and the derived `ChatCapabilities`) instead of a required restrictive `Literal`, so a new server-side quantization value can no longer crash the entire `GET /models?type=text` parse.
- **`client.audio.transcribe(response_format="text")` no longer crashes.** The live endpoint returns `Content-Type: text/plain`, but the SDK ran `json.loads()` on every response; `transcribe()` now returns a `str` for the `text` format (overloads mirror `client.video.transcribe()`).
- **Request-classifier image routing.** The image endpoint-tier patterns were plural (`images/generate`) while the SDK sends singular paths (`image/generate`, `image/multi-edit`) — so endpoint routing matched nothing — and `qwen-image` / `gpt-image-2` fell through the generic `qwen` / `gpt-` rules into the LLM rate-limit queue. Fixed the paths and added IMAGE model patterns ahead of the LLM rules.
- **Streaming chunk usage no longer drops detail/cache fields.** `ChatCompletionChunk.usage` used the bare `UsageData` and dropped `completion_tokens_details` / `cache_read_input_tokens` / `cache_creation_input_tokens` that the wire sends; it now uses `ChatUsage`. `UsageData`, `ChatMessage`, `ChatChoice`, `ChatUsage`, and `LogProbToken` all gained `extra="allow"`.
- **In-band streaming errors are surfaced, not swallowed.** A mid-stream SSE `data: {"error": ...}` frame was caught and dropped at DEBUG level, silently truncating the response with no exception; it now raises `APIError`. Benign keepalives and `[DONE]` are still skipped.
- **Responses API types re-exported from `venice_ai.types`.** `ResponsesResponse` plus its 12 output / usage / stream-event siblings were missing from the `venice_ai.types` namespace (`from venice_ai.types import ResponsesResponse` raised `ImportError`); now exported.
- **Docs / examples / skills drift corrected.** README video snippet (`duration` → `duration_seconds`); `docs/MIGRATION.md` (`cancel()` signature, prompt-cap "3,500" → "10,000"); `docs/cli.md` (removed the nonexistent `--show-tier-info` and the removed `--negative-prompt`, `--max-tokens` → `--max-completion-tokens`, de-hardcoded default model IDs). Skill reference docs: removed the dead `negative_prompt` template, `RetryOptions(max_retries=)` → `max_attempts=`, real `VoiceDetail` fields, non-empty upscale prompt, accurate image-resource method list. Five examples now exit non-zero on API failure instead of swallowing to exit 0, and the embeddings examples no longer require the dev-only `numpy`.

### Deprecated

- **`create_model_selector(client)`** — emits `DeprecationWarning` directing users to `client.models.resolve()` (or the type-specific `resolve_*()` shortcuts). The factory still works for backwards compatibility within the v2.x line.

### Changed

- **Request-classifier model-pattern iteration order** — `IMAGE` / `AUDIO` / `EMBEDDING` patterns are now checked before `LLM` so specialised variants (e.g. `qwen-image`) reach their correct queue instead of being sucked into the generic LLM bucket by a substring match.
- **`redis` is an optional dependency (new in v2)** — the Redis backend is not installed by default. Install it with:
  ```bash
  pip install venice-ai[redis]
  ```
  Or as part of the `enterprise` or `adaptive` extras which bundle Redis support.
- **Rate-limiting backend defaults to in-memory** — `BackendConfig()` uses `BackendType.MEMORY`, so the SDK works out-of-the-box without external services; set `backend_type=BackendType.REDIS` (with a `RedisBackendConfig`) to use Redis.
- **`RateLimiterMode.ADAPTIVE` requires the `[adaptive]` extra** — selecting ADAPTIVE without `adaptive-rate-limiter` installed raises `ImportError` (install via `pip install "venice-ai[adaptive]"`); use `RateLimiterMode.SIMPLE` or `RateLimiterMode.DISABLED` otherwise.
- **`pydantic` bumped to `^2.13.4`** — projects pinning to earlier Pydantic v2 releases must update.
- **`aiohttp` widened to `>=3.13.4,<3.15`** (with `speedups` extras) — the earlier `<3.14` ceiling (aiohttp 3.14 removed `aiohttp.streams.AsyncStreamReaderMixin`, which broke every VCR-based test at import time) has been lifted now that vcrpy 8.2.0 shipped the 3.14 compatibility fix ([vcrpy#995](https://github.com/kevin1024/vcrpy/issues/995)). The dev/test `vcrpy` pin is now `^8.2.1`. Verified against the VCR integration suite and the aiohttp-backed HTTP-client unit tests on aiohttp 3.14.3.
- **`cryptography` ceiling widened to `>=46.0.0,<51.0.0`** (optional `e2ee` / `e2ee-verify` extras) — allows cryptography 46–50 instead of only 46.x. cryptography 49 drops prebuilt wheels for x86_64 macOS and 32-bit Windows, but the TEE primitives the SDK uses (secp256k1 ECDH, HKDF-SHA256, AES-256-GCM, key serialization) are unchanged across all five majors. The floor stays at 46 so downstream installs on those platforms are not forced off prebuilt wheels; CVE-2026-69247 (fixed in cryptography 50) is a Bleichenbacher oracle in PKCS#7 `EnvelopedData` decryption, an API the SDK never calls.
- **`solders` bumped to `^0.28.0`** (optional `x402-solana` extra) — the ed25519 signing, base58, and versioned-transaction APIs the SDK uses are unchanged.
- **`dcap-qvl` bumped to `^0.6.1`** (optional `e2ee-verify` extra) — `parse_quote`, `verify`, and `QuoteCollateralV3` are unchanged.
- **`prometheus-client` bumped to `^0.26.0`** (optional `metrics` / `enterprise` / `all` extras).
- **Dependency lockfile refresh** — all dependencies updated to their latest in-constraint versions (aiohttp 3.14.3, cryptography 50.0.0, Pillow 12.3.0, redis 8.1.0, pydantic-settings 2.15.0, OpenTelemetry 1.44.0, vcrpy 8.3.0, pytest 9.1.1, ruff 0.16.2, mypy 2.3.0, setuptools 84.0.0, idna 3.18, plus transitive bumps). `pip-audit` runs with **no suppressions** — the previous `--ignore-vuln` list (two pip advisories and two aiohttp advisories) is gone, since every entry is now fixed in a version the lockfile ships — and reports no known vulnerabilities.
- **`ModelResponse` now uses `extra="allow"`** — matches the existing `ModelSpec` policy. Brand-new top-level Venice fields (e.g. `context_length` added late 2025) land on `BaseModel.model_extra` and survive `model_dump()` round-trips instead of being silently dropped.
- **Balance header rename: `x-venice-balance-usd` → `x-venice-balance-diem`** — the live API renamed this header. The SDK already reads both names so existing access patterns continue to work; new code should prefer `.balance_diem` accessors where applicable.
- **Coverage gate raised from 80 → 90%** (`pyproject.toml` `tool.coverage.report.fail_under`). Reflects the actual ~95 % coverage achieved by the 4110-test suite. Internal change only — no impact on consumers.

### Removed

- **`negative_prompt` removed from image generation** — the Venice API disabled this parameter for image models in February 2026, so modern image models ignore it server-side. The keyword is now removed from `client.image.create()` (both overloads), from the underlying `ImageGenerationRequest` model, from the `--negative-prompt` / `-np` CLI flag on `venice image generate`, from the interactive wizard, and from the batch `_batch_generate_async` plumbing. Passing `negative_prompt=...` raises `TypeError`. Video generation is unaffected — `negative_prompt` remains a valid parameter for `client.video.run()` / `client.video.submit()`.
- **`Video.submit()` / `Video.quote()` / `Video.run()` parameter renamed** `duration` → `duration_seconds` — symmetry with `Music.run()` (which already used `duration_seconds`) and with the in-progress unification across modalities. The new parameter accepts `int | str` and parses liberally: `5`, `"5"`, `"5s"`, `"5 seconds"` all become 5 internally. The wire format `"5s"` is generated by the SDK before the request is sent, so the server contract is unchanged. Upscale-style sentinel strings like `"Auto"` pass through unchanged. The new helper `venice_ai.helpers.normalize_duration_seconds()` exposes the parser. The CLI's `--duration` flag is unchanged for end users; internally it now binds to `duration_seconds=` on the SDK call.
- **`typing_extensions` dependency removed** — no longer needed; the SDK now uses Python 3.13+ native typing syntax exclusively.

---

## [1.3.0] - 2025-06-24

### Added

#### **🚨 Enhanced Exception Handling**

- **New exception classes** for better error handling:
  - [`PaymentRequiredError`](src/venice_ai/exceptions.py) (HTTP 402) - Raised when payment is required to access the service
  - [`ServiceUnavailableError`](src/venice_ai/exceptions.py) (HTTP 503) - Raised when the service is temporarily unavailable
- **Improved error mapping** in [`_make_status_error()`](src/venice_ai/exceptions.py) function for more specific exception types

#### **🔧 Embeddings API Enhancements**

- **Input validation** for embeddings API:
  - Maximum array length validation (2048 items limit)
  - Raises `InvalidRequestError` with descriptive message when limit is exceeded
- **Base64 encoding support**:
  - Embedding responses can now return base64-encoded strings in addition to float arrays
  - Support for `encoding_format` parameter with values "float" or "base64"
- **OpenAI compatibility improvements**:
  - `user` parameter now accepted (though discarded by Venice API) for better OpenAI client compatibility
  - Enhanced documentation clarifying parameter behavior

#### **🎯 Model Capabilities Expansion**

- **New model capabilities** in `ModelCapabilities`:
  - `supportsVision` - Indicates if model supports image inputs
  - `supportsReasoning` - Indicates if model has reasoning capabilities
  - `quantization` - Specifies model quantization type (e.g., "fp16", "int8")
- **Beta field support** in `ModelSpec` for identifying beta models
- **Enhanced model filtering** in `get_filtered_models()`:
  - New capability-based filtering parameters
  - Deprecated `supports_capabilities` parameter in favor of specific capability flags

#### **📚 Documentation & Analysis**

- **New test suite** `tests/test_embeddings_api_alignment.py` with 28 new tests for embeddings API

#### **💻 Developer Resources**

- Added `recommended_model_updates.py` providing example utility classes (e.g., `ModelWrapper`, `ModelSelector`) for advanced model interaction and management.

### Changed

- **Test Suites**: Updated 13 test files to align with new exception handling and model capabilities.
- **E2E Tests**: Enhanced `e2e_tests/test_01_models.py` to verify new model fields and capabilities. *(Note: e2e test files are not tracked in the repository)*

### Fixed

- Corrected various tests to handle new exception types and model filtering logic.
- Enhanced client-side robustness in stream handling, pricing information retrieval, and cost calculations.

## [1.2.0] - 2025-06-22

### Added

#### **💰 Cost Management & Estimation**

- **New cost calculation module** ([`venice_ai.costs`](src/venice_ai/costs.py)):
  - [`calculate_completion_cost()`](src/venice_ai/costs.py) - Calculate actual costs from chat completion responses
  - [`calculate_embedding_cost()`](src/venice_ai/costs.py) - Calculate costs for embedding operations
  - [`estimate_completion_cost()`](src/venice_ai/costs.py) - Estimate costs before making API calls
- **Dual currency support**: All cost calculations now support both USD and VCU (Venice Compute Units)
- **New client method** [`get_model_pricing()`](src/venice_ai/_client.py) to fetch detailed pricing information for any model

#### **🧠 Enhanced Chat Completions**

- **Web Search Integration**:
  - `enable_web_search` - Control web search behavior ("on", "off", "auto")
  - `enable_web_citations` - Request citations in `[REF]0[/REF]` format
  - `include_search_results_in_stream` - Include search results in streaming responses
- **Reasoning/Thinking Controls**:
  - `strip_thinking_response` - Remove `<think></think>` blocks from responses
  - `disable_thinking` - Disable thinking mode entirely on supported models
- **Advanced Sampling Parameters**:
  - `logit_bias` - Modify token likelihood with bias values (-100 to 100)
  - `parallel_tool_calls` - Enable parallel function calling
  - `max_temp`, `min_temp` - Dynamic temperature scaling
  - `min_p` - Minimum probability threshold for token selection

#### **🔧 Utility Enhancements**

- New `get_models_by_capability()` function to filter models by specific capabilities
- Improved model filtering and capability detection

### Changed

#### **🏗️ Model Type Structure Refactoring**

- Model metadata (capabilities, constraints, pricing) is now consolidated under `model_spec`
- Pricing structure now uses dedicated `PricingUnit` and `PricingDetail` types
- Legacy pricing fields are maintained for backward compatibility but are now optional

#### **📦 Response Type Updates**

- Chat completion responses now use Pydantic models instead of TypedDict
- New [`VeniceParametersResponse`](src/venice_ai/types/chat.py) type for Venice-specific response metadata
- `web_search_citations` moved into `venice_parameters` response field

#### **🏃‍♂️ Dependency Optimization**

- Made `tiktoken` optional - because not everyone needs to count their tokens obsessively
- Relocated `numpy`, `Pillow`, `beautifulsoup4`, and `pypandoc` to dev dependencies where they can contemplate their existence without affecting your production builds
- **Installation options**:
  ```bash
  pip install venice-ai              # Lean and mean
  pip install venice-ai[tokenizers]  # With token counting
  ```

### Fixed

- Improved error handling in model listing operations
- Fixed edge cases in token estimation fallback logic
- Enhanced type safety throughout the codebase

### Security

- Project status upgraded from Beta to Production/Stable
- Enhanced input validation for new chat completion parameters

### Performance

- Reduced package size and installation time through dependency optimization
- Streamlined test suite for improved CI/CD performance

## [1.1.2] - 2025-06-19

### Changed

- **Documentation Updates**: Updated documentation to reflect Venice.ai API improvements
  - Added information about Venice Large model's increased context window (32k → 128k tokens)
  - Enhanced `README.md` with Venice Large examples and context window guidance
  - Updated client utilities documentation with model capability notes and token management best practices
  - Enhanced async chat streaming guide with large context window usage guidance
  - Added practical examples showing how to leverage the 128k context window with `max_completion_tokens`

### Notes

- **API Compatibility**: No SDK code changes required - existing functionality automatically benefits from API improvements
  - Venice Large's increased context size can be utilized through existing `max_completion_tokens` parameter
  - Non-streaming chat completions now receive cleaner responses due to server-side "thinking" message processing improvements
  - Streaming behavior remains unchanged and continues to pass through all API-sent events

## [1.1.1] - 2025-06-13

### Fixed

- **Documentation Build Issues**: Fixed empty sections in Sphinx API reference documentation that were appearing in Read the Docs builds
  - Updated `.readthedocs.yaml` to properly install the `venice_ai` package during documentation builds
  - Added missing imports in `src/venice_ai/resources/__init__.py` for `ApiKeys`, `Audio`, `Billing`, `Embeddings`, and `Models`
  - Added comprehensive type imports in `src/venice_ai/types/__init__.py` for image, api_keys, audio, embeddings, and billing modules
  - Added explicit `__all__` list to [`src/venice_ai/exceptions.py`](src/venice_ai/exceptions.py) for better module discovery
  - Fixed missing `ModelTraitList` and `ModelCompatibilityList` exports in types package
- **Test Runner & Coverage**: Refactored `test_runner.py` to use `pytest-cov` directly, resolving significant code coverage reporting inaccuracies when running tests in parallel with `pytest-xdist`.
- **Embedding Tests**: Updated `e2e_tests/test_05_embeddings.py` with improved and corrected end-to-end tests for embedding functionalities. *(Note: e2e test files are not tracked in the repository)*
- **CI Workflow**: Modified `.github/workflows/python-publish.yaml` to enhance test execution, enabling or optimizing parallel test runs.

## [1.1.0] - 2025-06-09

### Added

- Implemented support for `logprobs` and `top_logprobs` parameters in Chat Completions API, allowing users to retrieve token likelihoods. Includes E2E tests and documentation updates.

#### **🏗️ Core SDK Architecture & Client Enhancements**

- **BaseClient Foundation**: Introduced [`BaseClient`](src/venice_ai/_client.py) class providing shared functionality for both sync and async clients, including common initialization logic, retry configuration, and transport setup.
- **Advanced HTTP Configuration**: Added comprehensive HTTP client configuration options to `VeniceClient`:
  - Support for custom `httpx.Client`/`httpx.AsyncClient` instances
  - Direct configuration of proxy, transport, limits, cert, verify, trust_env, HTTP/1.1, HTTP/2 settings
  - Custom event hooks and default encoding support
  - Follow redirects and max redirects configuration
- **Global Timeout Management**: Implemented `default_timeout` parameter for setting global timeout defaults across all API calls, with per-request override capability.
- **Automatic Retry System**: Integrated `httpx-retries` library with configurable retry behavior:
  - Configurable `max_retries` (default: 2)
  - Adjustable `retry_backoff_factor` (default: 0.1)
  - Customizable `retry_status_forcelist` (default: [429, 500, 502, 503, 504])
  - Respect for `Retry-After` headers in rate limit responses
- **Sentinel Type System**: Added `NotGiven` sentinel type and `NOT_GIVEN` constant for distinguishing between `None` and not-provided parameters.

#### **🎵 Audio API Major Expansion**

- **Streaming Audio Support**: Implemented method overloads for [`create_speech()`](src/venice_ai/resources/audio.py) supporting both streaming and non-streaming audio generation:
  - `stream=False`: Returns `bytes` for immediate audio data
  - `stream=True`: Returns `Iterator[bytes]` for streaming audio chunks
- **Voice Management System**: Added comprehensive [`get_voices()`](src/venice_ai/resources/audio.py) method with advanced filtering:
  - Filter by model ID, gender (male/female/unknown), and region code
  - Automatic voice metadata parsing from voice IDs
  - Language and accent detection for 15+ supported regions
- **Enhanced Voice Metadata**: Implemented [`REGION_LANGUAGE_MAPPING`](src/venice_ai/resources/audio.py) supporting:
  - English variants: American, British, Canadian, Scottish, Welsh, Australian, Indian
  - International languages: German, Spanish, French, Italian, Japanese, Korean, Portuguese, Russian, Mandarin Chinese
- **Improved Parameter Handling**: Set sensible defaults for audio generation (`response_format="mp3"`, `speed=1.0`).
- **Raw Response Support**: Added [`_request_raw_response()`](src/venice_ai/_resource.py) and [`_arequest_raw_response()`](src/venice_ai/_resource.py) methods for handling binary audio content and streaming responses.

#### **👥 Characters API Implementation**

- **Character Listing**: Implemented [`Characters.list()`](src/venice_ai/resources/characters.py) method with support for extra headers, query parameters, and custom timeouts.
- **Enhanced Character Model**: Completely redesigned `Character` Pydantic model with modern fields:
  - Core identification: `slug`, `name`, `description`
  - AI capabilities: `system_prompt`, `user_prompt`, `vision_enabled`
  - Media support: `image_url`, `voice_id`
  - Organization: `category_tags`
  - Timestamps: `created_at`, `updated_at` with proper datetime handling
- **Simplified Character List**: Streamlined `CharacterList` model for cleaner API responses.

#### **🔧 Enhanced Error Handling & Resilience**

- **Retry-After Header Parsing**: Implemented [`_parse_retry_after_header()`](src/venice_ai/exceptions.py) function supporting:
  - Integer seconds format (e.g., "120")
  - HTTP-date format (e.g., "Wed, 21 Oct 2015 07:28:00 GMT")
  - Timezone-aware datetime calculations
  - Server time synchronization using response `Date` header
- **Enhanced RateLimitError**: Extended [`RateLimitError`](src/venice_ai/exceptions.py) with `retry_after_seconds` attribute for intelligent retry logic.
- **Improved Error Context**: Better error message formatting and context preservation across the exception hierarchy.

#### **🧪 Comprehensive Testing Infrastructure**

- **Massive Test Suites**: Added extensive functional test coverage:
  - `venice_sdk_async_test.py`: 126k lines of async functionality tests
  - `venice_sdk_sync_test.py`: 49k lines of sync functionality tests
- **HTTP Configuration Testing**: New `tests/test_client_http_config.py` for validating advanced HTTP client options.
- **Enhanced API Coverage**: Expanded test coverage for:
  - Audio streaming and non-streaming modes with various parameters
  - Characters API functionality and error handling
  - Chat completions with tool usage, JSON format, and streaming
  - Image generation with advanced parameters (negative_prompt, seed, format)
  - API key management including Web3 token functionality
  - Retry mechanism behavior and configuration
  - Global timeout functionality across all endpoints

#### **📚 Documentation & Project Infrastructure**

- **Comprehensive Changelog**: Created this detailed changelog following Keep a Changelog format.
- **Contributing Guidelines**: Added [`CONTRIBUTING.md`](CONTRIBUTING.md) with clear issue reporting guidelines.
- **Enhanced API Documentation**: Updated `docs/api.rst` with 168 new lines covering:
  - Advanced HTTP client configuration examples
  - Retry mechanism documentation
  - Global timeout usage patterns
- **Utility Documentation**: Added `docs/client_utilities.rst` documenting `estimate_token_count` and `validate_chat_messages` utilities.
- **README Overhaul**: Major [`README.md`](README.md) updates (144 lines changed) including:
  - Advanced HTTP Client Configuration section with three configuration approaches
  - Updated all code examples to include `default_timeout` parameter
  - Enhanced feature list highlighting automatic retry functionality
  - Improved error handling examples and best practices

### Changed

#### **🔄 Client Architecture Improvements**

- **Inheritance Hierarchy**: `VeniceClient` now inherits from `BaseClient` for shared functionality and consistent behavior.
- **Request Method Simplification**: Removed manual HTTP 503 retry loops from client request methods (`_request`, `_arequest`, and related stream/multipart methods) in favor of `httpx-retries` integration.
- **Enhanced Documentation**: Significantly expanded docstrings for both sync and async clients with detailed parameter descriptions and usage examples.

#### **📦 Project Configuration & Metadata**

- **Version Bump**: Updated from `1.0.3` to `1.1.0` reflecting significant new features and improvements.
- **Dependency Management**: Added `httpx-retries = "^0.4.0"` as a core dependency for retry functionality.
- **Enhanced Discoverability**: Expanded keywords from 5 to 9 terms: `ai`, `api-client`, `generative-ai`, `llm`, `machine-learning`, `ml`, `sdk`, `venice`, `venice-ai`.
- **Refined Classifiers**: Updated PyPI classifiers:
  - Removed Python 3.10 support (now requires Python 3.11+)
  - Added "Development Status :: 4 - Beta"
  - Added comprehensive topic classifiers for chat, image generation, speech, text processing
  - Added "Typing :: Typed" classifier for type hint support
- **Project URLs**: Added "Issue Tracker" and "Changelog" links for better project navigation.
- **Test Configuration**: Enabled parallel test execution with `pytest-xdist` (`addopts = "-n auto"`).

#### **🎯 API Method Enhancements**

- **Characters API**: Enhanced [`Characters.list()`](src/venice_ai/resources/characters.py) with additional parameters for headers, query parameters, body, and timeout customization.
- **Audio API**: Improved [`create_speech()`](src/venice_ai/resources/audio.py) with better error handling, streaming support, and parameter validation.
- **Consistent Parameter Patterns**: Standardized optional parameter handling across all API methods using the new `NotGiven` sentinel system.

### Fixed

#### **🐛 API Functionality Corrections**

- **API Key Management**: Corrected API key delete method to use query parameters instead of request body, aligning with API specification.
- **Image Upscale Response Handling**: Fixed Image Upscale functional tests to correctly handle `bytes` response type instead of expecting JSON.
- **Audio Error Processing**: Improved error handling in audio generation to properly consume response bodies before raising exceptions, preventing connection leaks.

#### **🧪 Testing Reliability**

- **Embeddings Test Stability**: Made Embeddings functional tests robustly skipped due to persistent API authentication issues in test environments, preventing false test failures.
- **Response Type Validation**: Enhanced test assertions to properly validate response types across different API endpoints.

### Removed

#### **🗑️ Cleanup & Simplification**

- **Legacy Files**: Removed development and example files:
  - `app.py`: 883-line example/demo application
  - `dummy_image.png` and `dummy_image_async.png`: Test image files
  - `tests/resources/test_billing.py`: 79-line billing test file
- **Billing API Simplification**: Removed `export()` method (71 lines) from [`billing.py`](src/venice_ai/resources/billing.py) that provided CSV billing data export functionality.
- **Obsolete Type Definitions**: Removed incorrect/placeholder `CharacterChatCompletionRequest` Pydantic model and associated `Stats` model.
- **Deprecated Features**: Removed plans for `ResponseTransformer`/`AsyncResponseTransformer` as existing streaming utilities were deemed sufficient.

#### **📋 Documentation Cleanup**

- **Streamlined Character Documentation**: Simplified character model documentation to focus on current functionality rather than legacy fields.

### Security

#### **🔒 Enhanced Error Information**

- **Rate Limit Intelligence**: `RateLimitError` now safely parses and exposes `Retry-After` header information without leaking sensitive data.
- **Timeout Configuration**: Global timeout settings provide better protection against hanging requests and resource exhaustion.

### Performance

#### **⚡ Efficiency Improvements**

- **Automatic Retries**: Intelligent retry mechanism reduces manual retry logic and improves success rates for transient failures.
- **Parallel Testing**: Enabled parallel test execution reducing CI/CD pipeline duration.
- **Streaming Optimization**: Enhanced audio streaming implementation for better memory efficiency with large audio files.
- **Connection Management**: Improved HTTP connection lifecycle management through better integration with `httpx` features.

---

## [1.0.3] - 2025-06-06

### Added

- Initial release of the Venice AI Python SDK with comprehensive API coverage
- Support for Chat Completions, Image Generation, Audio (TTS), Models, API Keys, Billing, and Characters endpoints
- Both synchronous and asynchronous client implementations
- Comprehensive error handling with custom exception hierarchy
- Type-hinted interfaces for better developer experience
- Resource-oriented client design pattern
- Streaming support for chat completions
- Comprehensive test suite with functional and unit tests
- Sphinx-based documentation system
- Poetry-based dependency management and packaging

### Changed

- Established baseline functionality and API coverage

### Fixed

- Initial bug fixes and stabilization for public release

## [1.0.2] - 2025-06-05

_No retroactive release notes. See git history for changes between v1.0.2 and v1.0.3._

---

## [1.0.1] - 2025-06-04

_No retroactive release notes. See git history for changes between v1.0.1 and v1.0.2._

---

## [1.0.0] - 2025-06-03

_Initial public release. No retroactive release notes documented._

---

**Note**: This changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format. For detailed technical information about any changes, please refer to the git commit history or the linked source files.

[Unreleased]: https://github.com/sethbang/venice-py/compare/v2.5.1...HEAD
[2.5.1]: https://github.com/sethbang/venice-py/compare/v2.5.0...v2.5.1
[2.5.0]: https://github.com/sethbang/venice-py/compare/v2.4.1...v2.5.0
[2.4.1]: https://github.com/sethbang/venice-py/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/sethbang/venice-py/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/sethbang/venice-py/compare/v2.2.1...v2.3.0
[2.2.1]: https://github.com/sethbang/venice-py/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/sethbang/venice-py/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/sethbang/venice-py/compare/v2.0.2...v2.1.0
[2.0.2]: https://github.com/sethbang/venice-py/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/sethbang/venice-py/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/sethbang/venice-py/compare/v1.3.0...v2.0.0
[1.3.0]: https://github.com/sethbang/venice-py/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/sethbang/venice-py/compare/v1.1.2...v1.2.0
[1.1.2]: https://github.com/sethbang/venice-py/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/sethbang/venice-py/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/sethbang/venice-py/compare/v1.0.3...v1.1.0
[1.0.3]: https://github.com/sethbang/venice-py/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/sethbang/venice-py/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/sethbang/venice-py/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/sethbang/venice-py/releases/tag/v1.0.0
