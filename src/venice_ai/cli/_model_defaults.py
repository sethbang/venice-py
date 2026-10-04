"""Runtime resolution of default models for the CLI.

The CLI hardcodes NO model IDs. When the user has not explicitly chosen a
model (via a flag or via a saved value in ``venice-py configure``), the model is
resolved from the live Venice ``/models`` API at call time, so newly added and
deprecated models are picked up automatically without code changes.
"""

from __future__ import annotations

from typing import Any

import click

# CLI "kind" → arguments for ``client.models.resolve(...)``.
# ``stt`` maps to the API's ``asr`` type; video kinds carry a sub-mode.
_RESOLVE_KWARGS: dict[str, dict[str, Any]] = {
    "chat": {"type": "chat"},
    "image": {"type": "image"},
    "tts": {"type": "tts"},
    "stt": {"type": "asr"},
    "embedding": {"type": "embedding"},
    "decision": {"type": "decision"},
    "video_t2v": {"type": "video", "video_type": "text-to-video"},
    "video_i2v": {"type": "video", "video_type": "image-to-video"},
}


async def resolve_default_model(
    client: Any,
    config: dict[str, Any],
    kind: str,
    explicit: str | None = None,
) -> str:
    """Resolve the model ID a CLI command should use.

    Precedence: ``explicit`` flag > user's saved ``config`` default >
    live ``/models`` API resolution. Raises :class:`click.ClickException`
    with an actionable message if the API cannot be reached and no default
    is available.

    :param client: An open ``VeniceClient``.
    :param config: The loaded CLI config dict.
    :param kind: One of ``chat, image, tts, stt, embedding, decision, video_t2v,
        video_i2v``.
    :param explicit: A model ID the user passed via ``--model`` (or ``None``).
    """
    if explicit:
        return explicit

    configured: str | None = config.get("defaults", {}).get(f"{kind}_model")
    if configured:
        return configured

    try:
        kwargs = _RESOLVE_KWARGS[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown model kind: {kind!r}") from exc

    try:
        resolved: str = await client.models.resolve(**kwargs)
    except Exception as exc:  # network down, no matching model, etc.
        raise click.ClickException(
            f"Could not resolve a default {kind} model from the Venice API "
            f"({exc}). Pass an explicit model with --model, or run "
            f"'venice-py configure' while online."
        ) from exc
    return resolved
