"""One-shot parity check between SDK resources and the CLI surface.

Outputs a markdown table to stdout: ``resource.method | status | command | note``.

Status values:

- ``covered``  — There is a CLI command that directly invokes this exact SDK method.
- ``partial``  — A CLI command invokes a related/parent method, or only exposes a
  subset of this method's capability (e.g. CLI uses ``chat.completions.create``
  with ``stream=True`` but never calls the convenience helper ``.stream``).
- ``missing``  — No CLI surface for this method.

Run with::

    poetry run python tools/cli_parity_check.py

Pipe to a file (``> cli-parity.md``) to capture the table for review.

The mapping is maintained by hand here. When the CLI grows, update
``KNOWN_CLI_MAP`` accordingly.
"""

from __future__ import annotations

import inspect

from venice_ai import VeniceClient

# SDK resources to walk. Order matches ``VeniceClient`` attachment in
# ``src/venice_ai/_client.py`` so the output reads top-to-bottom in API order.
RESOURCES: list[str] = [
    "chat",
    "image",
    "video",
    "audio",
    "music",
    "embeddings",
    "models",
    "api_keys",
    "characters",
    "billing",
    "augment",
    "x402",
    "crypto",
    "responses",
    "decisions",
    "voice_changer",
]

# Mapping from "resource.method" → (status, [cli_invocations], note).
#
# Sources of truth used to populate this map:
#  - ``poetry run venice <cmd> --help`` walked recursively for every command
#  - ``grep -rn "client\." src/venice_ai/cli/commands/`` to confirm which
#    SDK method each CLI command actually calls.
#
# When in doubt, mark partial and explain in the note.
KNOWN_CLI_MAP: dict[str, tuple[str, list[str], str]] = {
    # ---- chat.completions ----
    "chat.completions.batch": (
        "missing",
        [],
        "No CLI surface for batched parallel completions.",
    ),
    "chat.completions.create": (
        "covered",
        ["venice-py chat start"],
        "Both stream=True (default) and stream=False paths are reachable.",
    ),
    "chat.completions.estimate_cost": (
        "missing",
        [],
        "No `venice-py chat estimate-cost` (or similar) command.",
    ),
    "chat.completions.parse": (
        "missing",
        [],
        "Structured output / Pydantic parsing not exposed by `chat start`.",
    ),
    "chat.completions.run_with_tools": (
        "missing",
        [],
        "Tool/function calling agent loop not exposed.",
    ),
    "chat.completions.stream": (
        "partial",
        ["venice-py chat start --stream"],
        "CLI calls `create(stream=True)` directly, not the `.stream()` helper.",
    ),
    # ---- image ----
    "image.background_remove": (
        "covered",
        ["venice-py image remove-bg"],
        "",
    ),
    "image.create": (
        "covered",
        ["venice-py image generate", "venice-py image batch"],
        "Full parameter surface incl. style presets, LoRA, watermark, EXIF.",
    ),
    "image.edit": (
        "covered",
        ["venice-py image edit"],
        "Exposes --aspect-ratio, --resolution, --output-format, --safe-mode/--no-safe-mode.",
    ),
    "image.list_styles": (
        "covered",
        ["venice-py image list-styles"],
        "",
    ),
    "image.multi_edit": (
        "covered",
        ["venice-py image multi-edit"],
        "Layers up to 3 images (--image, --image-2, --image-3) with one prompt.",
    ),
    "image.simple_generate": (
        "missing",
        [],
        "OpenAI-compatible generation shim not exposed.",
    ),
    "image.upscale": (
        "covered",
        ["venice-py image upscale"],
        "",
    ),
    # ---- video ----
    "video.cancel": (
        "partial",
        ["venice-py video status (auto-cancels on Ctrl-C)"],
        "Cancel is invoked implicitly by the polling loop, not as a standalone "
        "`venice-py video cancel <job_id>`.",
    ),
    "video.quote": (
        "missing",
        [],
        "Pricing quote endpoint not exposed in CLI.",
    ),
    "video.retrieve": (
        "covered",
        ["venice-py video status"],
        "Used to poll job status by queue_id.",
    ),
    "video.run": (
        "partial",
        ["venice-py video generate", "venice-py video from-image"],
        "CLI uses `submit` + `retrieve` polling rather than the high-level "
        "`run` helper. Equivalent end result.",
    ),
    "video.submit": (
        "covered",
        ["venice-py video generate --no-poll", "venice-py video from-image --no-poll"],
        "",
    ),
    "video.transcribe": (
        "missing",
        [],
        "Video URL → transcription not exposed.",
    ),
    # ---- audio ----
    "audio.create_speech": (
        "covered",
        ["venice-py audio speak"],
        "",
    ),
    "audio.get_voices": (
        "covered",
        ["venice-py audio voices"],
        "Filters by --model / --gender / --region; --json for scripting.",
    ),
    "audio.transcribe": (
        "covered",
        ["venice-py audio transcribe"],
        "",
    ),
    # ---- music ----
    "music.cancel": (
        "missing",
        [],
        "Whole `music` resource has no CLI surface.",
    ),
    "music.quote": (
        "missing",
        [],
        "Whole `music` resource has no CLI surface.",
    ),
    "music.retrieve": (
        "missing",
        [],
        "Whole `music` resource has no CLI surface.",
    ),
    "music.run": (
        "missing",
        [],
        "Whole `music` resource has no CLI surface.",
    ),
    "music.submit": (
        "missing",
        [],
        "Whole `music` resource has no CLI surface.",
    ),
    # ---- embeddings ----
    "embeddings.create": (
        "covered",
        ["venice-py embeddings"],
        "",
    ),
    # ---- models ----
    "models.get": (
        "covered",
        ["venice-py models get <id>"],
        "Single-model lookup via the cached list; --json for raw payload.",
    ),
    "models.get_capabilities": (
        "covered",
        ["venice-py models capabilities <id>"],
        "Renders the typed Capabilities discriminated union.",
    ),
    "models.list": (
        "covered",
        ["venice-py models", "venice-py configure", "venice-py chat start --select-model"],
        "Used by multiple commands; primary discovery path.",
    ),
    "models.list_compatibility": (
        "missing",
        [],
        "OpenAI-compatibility mapping endpoint not exposed.",
    ),
    "models.list_traits": (
        "partial",
        ["venice-py models --trait <t>"],
        "CLI filters traits client-side from `list` results; never calls "
        "`list_traits` to get the canonical trait set.",
    ),
    "models.resolve": (
        "covered",
        ["venice-py models resolve --type <type>"],
        "Single command exposes resolve() + every resolve_* shortcut via --type.",
    ),
    "models.resolve_asr": (
        "covered",
        ["venice-py models resolve --type asr"],
        "Routed through resolve() with type='asr'.",
    ),
    "models.resolve_chat": (
        "covered",
        ["venice-py models resolve --type chat"],
        "Carries chat capability flags (--function-calling, --vision, etc).",
    ),
    "models.resolve_cheapest_video": (
        "covered",
        ["venice-py models resolve --type cheapest-video"],
        "Renders the CheapestVideoResult with all-quotes table.",
    ),
    "models.resolve_decision": (
        "covered",
        ["venice-py models resolve --type decision"],
        "Beta-only model class; the decision arm never applies --include-beta.",
    ),
    "models.resolve_voice_changer": (
        "missing",
        [],
        "Voice-changer is a capability on type='music', not a resolve() type, so "
        "`models resolve --type` cannot express it. The audio voice-change command "
        "calls it internally to pick a default model.",
    ),
    "models.resolve_embedding": (
        "covered",
        ["venice-py models resolve --type embedding"],
        "",
    ),
    "models.resolve_image": (
        "covered",
        ["venice-py models resolve --type image"],
        "",
    ),
    "models.resolve_inpaint": (
        "covered",
        ["venice-py models resolve --type inpaint"],
        "",
    ),
    "models.resolve_music": (
        "covered",
        ["venice-py models resolve --type music"],
        "",
    ),
    "models.resolve_tts": (
        "covered",
        ["venice-py models resolve --type tts"],
        "",
    ),
    "models.resolve_video": (
        "covered",
        ["venice-py models resolve --type video"],
        "Carries video filters (--video-type, --audio, --min-resolution, --min-duration).",
    ),
    "models.resolve_video_upscale": (
        "covered",
        ["venice-py models resolve --type video-upscale"],
        "",
    ),
    # ---- api_keys ----
    "api_keys.create": (
        "covered",
        ["venice-py api-keys create"],
        "",
    ),
    "api_keys.create_web3_key": (
        "missing",
        [],
        "Web3 key creation flow not in CLI.",
    ),
    "api_keys.delete": (
        "covered",
        ["venice-py api-keys delete"],
        "",
    ),
    "api_keys.get_rate_limit_logs": (
        "missing",
        [],
        "Rate-limit logs not exposed.",
    ),
    "api_keys.get_rate_limits": (
        "missing",
        [],
        "Rate-limit summary not exposed.",
    ),
    "api_keys.get_web3_token": (
        "missing",
        [],
        "Web3 nonce token not exposed.",
    ),
    "api_keys.iter_all": (
        "partial",
        ["venice-py api-keys list"],
        "CLI calls `list()` (single page). Auto-paginating helper not used.",
    ),
    "api_keys.list": (
        "covered",
        ["venice-py api-keys list"],
        "",
    ),
    "api_keys.retrieve": (
        "covered",
        ["venice-py account keys get <id>"],
        "Renders the bare ApiKey via Rich panel; --json for scripting.",
    ),
    "api_keys.update": (
        "covered",
        ["venice-py account keys update <id> [--description ... | --expiry ... | --limit-* ...]"],
        "Updates description / expiry / consumption limits.",
    ),
    # ---- characters ----
    "characters.get": (
        "covered",
        ["venice-py characters info"],
        "",
    ),
    "characters.iter_all": (
        "partial",
        ["venice-py characters list"],
        "CLI uses `list()` only; no pagination over full catalogue.",
    ),
    "characters.iter_reviews": (
        "missing",
        [],
        "Reviews paginator not exposed.",
    ),
    "characters.list": (
        "covered",
        ["venice-py characters list"],
        "Exposes --search, --sort-by/--sort-order, --tags, --categories, and status filters.",
    ),
    "characters.reviews": (
        "missing",
        [],
        "Per-character reviews not exposed.",
    ),
    # ---- billing ----
    "billing.get_balance": (
        "covered",
        ["venice-py account balance"],
        "",
    ),
    "billing.get_usage_history": (
        "covered",
        ["venice-py account usage"],
        "",
    ),
    "billing.get_usage_analytics": (
        "missing",
        [],
        "Aggregated analytics endpoint not exposed.",
    ),
    "billing.iter_usage_history": (
        "partial",
        ["venice-py account usage"],
        "CLI shows the first usage-history page plus a 'more available' hint; no auto-pagination.",
    ),
    # ---- augment ----
    "augment.parse_text": (
        "missing",
        [],
        "Whole `augment` resource has no CLI surface.",
    ),
    "augment.scrape": (
        "missing",
        [],
        "Whole `augment` resource has no CLI surface.",
    ),
    "augment.search": (
        "missing",
        [],
        "Whole `augment` resource has no CLI surface.",
    ),
    # ---- x402 ----
    "x402.balance": (
        "missing",
        [],
        "Whole `x402` resource has no CLI surface.",
    ),
    "x402.iter_transactions": (
        "missing",
        [],
        "Whole `x402` resource has no CLI surface.",
    ),
    "x402.top_up": (
        "missing",
        [],
        "Whole `x402` resource has no CLI surface.",
    ),
    "x402.transactions": (
        "missing",
        [],
        "Whole `x402` resource has no CLI surface.",
    ),
    # ---- crypto ----
    "crypto.batch_rpc": (
        "missing",
        [],
        "Whole `crypto` resource has no CLI surface.",
    ),
    "crypto.networks": (
        "missing",
        [],
        "Whole `crypto` resource has no CLI surface.",
    ),
    "crypto.rpc": (
        "missing",
        [],
        "Whole `crypto` resource has no CLI surface.",
    ),
    # ---- voice_changer ----
    "voice_changer.submit": (
        "partial",
        ["venice-py audio voice-change <source>"],
        "CLI drives run() (submit + poll + download); submit() alone is not exposed.",
    ),
    "voice_changer.run": (
        "covered",
        ["venice-py audio voice-change <source>", "venice-py audio voice-change --url <url>"],
        "",
    ),
    "voice_changer.quote": (
        "covered",
        ["venice-py audio voice-change --quote-only --duration <seconds>"],
        "",
    ),
    "voice_changer.retrieve": (
        "partial",
        ["venice-py audio voice-change <source>"],
        "Polled internally by the job; no standalone retrieve command.",
    ),
    "voice_changer.cancel": (
        "partial",
        ["venice-py audio voice-change <source>"],
        "Runs via the job context manager on exit; not separately invocable.",
    ),
    # ---- decisions ----
    "decisions.create": (
        "missing",
        [],
        "Beta /decisions endpoint has no CLI surface; questions are a nested "
        "typed map that does not reduce cleanly to flags.",
    ),
    # ---- responses ----
    "responses.create": (
        "missing",
        [],
        "OpenAI-style /responses endpoint has no CLI surface.",
    ),
}


def all_resource_methods() -> list[str]:
    """Walk SDK resources and return every public callable as ``resource.method``.

    Excludes dunders/underscored names. Special-cases ``chat`` to descend into
    ``chat.completions`` (the ``Chat`` resource itself only exposes the nested
    ``completions`` accessor).
    """
    out: list[str] = []
    c = VeniceClient(api_key="x")
    for rn in RESOURCES:
        r = getattr(c, rn)
        if rn == "chat":
            for name, m in inspect.getmembers(r.completions):
                if name.startswith("_"):
                    continue
                if not callable(m):
                    continue
                out.append(f"chat.completions.{name}")
            continue
        for name, m in inspect.getmembers(r):
            if name.startswith("_"):
                continue
            if not callable(m):
                continue
            out.append(f"{rn}.{name}")
    return out


def main() -> None:
    rows: list[tuple[str, str, str, str]] = []
    sdk_methods = all_resource_methods()

    for sdk_path in sorted(sdk_methods):
        entry = KNOWN_CLI_MAP.get(sdk_path)
        if entry is None:
            # Method exists in SDK but not in our map — needs hand-classification.
            rows.append((sdk_path, "unknown", "—", "Not yet classified in KNOWN_CLI_MAP."))
            continue
        status, cli_list, note = entry
        cli = ", ".join(cli_list) if cli_list else "—"
        rows.append((sdk_path, status, cli, note))

    # Identify any KNOWN_CLI_MAP entries that no longer correspond to a real
    # SDK method (e.g. a method got removed). Surface them so the table stays
    # honest.
    sdk_set = set(sdk_methods)
    stale = sorted(set(KNOWN_CLI_MAP) - sdk_set)
    for s in stale:
        rows.append(
            (s, "stale-map-entry", "—", "Method no longer in SDK; remove from KNOWN_CLI_MAP.")
        )

    width = max(len(r[0]) for r in rows)
    print(f"| {'SDK method':<{width}} | Status   | CLI command | Note |")
    print(f"| {'-' * width} | -------- | ----------- | ---- |")
    for path, status, cli, note in rows:
        print(f"| {path:<{width}} | {status:<8} | {cli} | {note} |")

    covered = sum(1 for r in rows if r[1] == "covered")
    partial = sum(1 for r in rows if r[1] == "partial")
    missing = sum(1 for r in rows if r[1] == "missing")
    unknown = sum(1 for r in rows if r[1] == "unknown")
    stale_n = sum(1 for r in rows if r[1] == "stale-map-entry")
    total = len(rows) - stale_n  # don't count stale rows in the denominator

    print()
    print(f"**Total SDK methods:** {total}")
    print(f"**Covered:** {covered}")
    print(f"**Partial:** {partial}")
    print(f"**Missing:** {missing}")
    if unknown:
        print(f"**Unknown (please add to KNOWN_CLI_MAP):** {unknown}")
    if stale_n:
        print(f"**Stale map entries (please remove):** {stale_n}")

    pct = (covered / total * 100) if total else 0.0
    pct_with_partial = ((covered + partial) / total * 100) if total else 0.0
    print(
        f"\n**Coverage:** {covered}/{total} fully ({pct:.0f}%); "
        f"{covered + partial}/{total} including partial ({pct_with_partial:.0f}%)."
    )


if __name__ == "__main__":
    main()
