# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/sethbang/venice-py/compare/v2.2.0...HEAD
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
