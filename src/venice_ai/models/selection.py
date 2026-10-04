"""
Dynamic Model Selection for Venice AI SDK.

This module provides intelligent model selection capabilities with caching,
preference handling, capability-based filtering, and custom selection strategies
for production use.

Classes:
    ModelCache: Cache for model information with TTL support
    DynamicModelSelector: Intelligent model selector with capability filtering

Functions:
    create_model_selector: Factory function to create a model selector
    get_chat_model: Quick helper to get a chat model
    get_embedding_model: Quick helper to get an embedding model
    get_video_model: Quick helper to get a video model
    get_cheapest_video_model: Quick helper to find the cheapest video model via quoting
    get_multiple_models: Quick helper to get multiple models for concurrency

Types:
    ModelSelectorType: Type alias for custom model selection functions
    CheapestVideoResult: Dataclass returned by select_cheapest_video_model

Example:
    >>> from venice_ai import VeniceClient, create_model_selector
    >>>
    >>> async with VeniceClient(api_key="...") as client:
    ...     selector = create_model_selector(client)
    ...     model = await selector.select_chat_model(
    ...         preferred_models=["llama-3.3-70b"],
    ...         require_function_calling=True
    ...     )

    # With custom selector for cost optimization:
    >>> def cheapest_model(candidates):
    ...     # Custom logic to pick cheapest model
    ...     return candidates[0]["id"]
    >>> selector = create_model_selector(client, default_selector=cheapest_model)
"""

import asyncio
import logging
import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from venice_ai.exceptions import ModelQuotesUnavailableError, NoMatchingModelError
from venice_ai.types.api.models import VideoModelConstraints

logger = logging.getLogger(__name__)

# Type alias for model selector functions
# Selectors receive a list of model dictionaries and return a selected model ID
ModelSelectorType = Callable[[list[dict[str, Any]]], str]


@dataclass
class CheapestVideoResult:
    """Result from cost-aware video model selection via the quote API.

    Attributes:
        model: The model ID with the lowest quoted price.
        quote_usd: The quoted cost in USD for the selected model.
        all_quotes: Mapping of every successfully quoted model ID to its
            USD price.  Useful for debugging or displaying alternatives.
        request_params: The ``video.quote`` / ``video.submit`` keyword
            arguments (``duration_seconds``, ``resolution``, ``aspect_ratio``,
            ``audio``) the selected model was quoted with. Pass them to
            ``submit`` to be billed the quoted price.
        skipped: Candidates that were not quoted or whose quote failed,
            mapped to the reason.
    """

    model: str
    quote_usd: float
    all_quotes: dict[str, float] = field(default_factory=dict)
    request_params: dict[str, Any] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass
class CheapestMusicResult:
    """Result from cost-aware music model selection via the quote API.

    Attributes:
        model: The model ID with the lowest quoted price.
        quote_usd: The quoted cost in USD for the selected model.
        request_params: The ``music.quote`` / ``music.submit`` keyword
            arguments the model was quoted with (``duration_seconds``, or
            nothing for models that take no duration). Pass them to ``submit``
            to be billed the quoted price.
        effective_seconds: The clip length that request produces, or ``None``
            when the model takes no duration and chooses the length itself.
        all_quotes: Every successfully quoted model ID mapped to its USD price.
        skipped: Candidates that were not quoted or whose quote failed,
            mapped to the reason.
    """

    model: str
    quote_usd: float
    request_params: dict[str, Any] = field(default_factory=dict)
    effective_seconds: int | None = None
    all_quotes: dict[str, float] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelCache:
    """Cache for model information with TTL support."""

    models: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    ttl_seconds: float = 300.0  # 5 minutes

    def is_expired(self) -> bool:
        """Check if cache has expired."""
        # Cache is always expired if empty, regardless of timestamp
        if not self.models:
            return True

        age = (datetime.now(UTC) - self.last_updated).total_seconds()
        return age > self.ttl_seconds

    def get_models(self, resource_type: str | None = None) -> list[str]:
        """Get list of model IDs, optionally filtered by resource type."""
        if self.is_expired():
            return []

        if resource_type:
            filtered = [
                model_id
                for model_id, model_info in self.models.items()
                if model_info.get("type") == resource_type
            ]
            return filtered

        return list(self.models.keys())

    def update(self, models: dict[str, Any]) -> None:
        """Update cache with new model data."""
        self.models = models
        self.last_updated = datetime.now(UTC)
        logger.info(f"Model cache updated with {len(models)} models")


# ---------------------------------------------------------------------------
# Price ranking (used by ``prefer="cheapest"`` and the cheapest strategies)
# ---------------------------------------------------------------------------

#: Weights for blending a token-priced model's input and output prices into one
#: comparable number: ``(3 * input + 1 * output) / 4`` USD per million tokens.
#: Three input tokens per output token is the usual blended-price convention and
#: matches prompt-heavy chat, RAG and tool-calling traffic.
CHAT_INPUT_WEIGHT = 3.0
CHAT_OUTPUT_WEIGHT = 1.0

#: Clip length, in seconds, at which :func:`model_price` prices music models so
#: that flat, per-second and duration-tier pricing compare on one footing. A model
#: whose minimum is longer is priced at its minimum; a model that cannot make a
#: clip this long is unpriced. ``prefer="cheapest"`` does not use it: music is
#: ranked by quotes at the caller's duration (see
#: :meth:`DynamicModelSelector.select_cheapest_music_model`).
MUSIC_REFERENCE_SECONDS = 30


def _usd(tier: Any) -> float | None:
    """Return the non-negative USD price in a cached pricing tier, else ``None``."""
    if not isinstance(tier, dict):
        return None
    value = tier.get("usd")
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _resolution_height(tier: str) -> float:
    """Sort key for resolution tiers: ``"720p"`` -> 720, ``"1K"`` -> 1024, ``"4k"`` -> 2160.

    Case-insensitive. Unrecognized tiers sort last.
    """
    text = tier.strip().lower()
    if text.endswith("p") and text[:-1].isdigit():
        return float(text[:-1])
    if text.endswith("k") and text[:-1].isdigit():
        # Image tiers ("1K", "2K", "4K") and video tiers ("2k", "4k") share
        # this spelling; only the relative order matters here.
        return {1: 1024.0, 2: 1440.0, 4: 2160.0, 8: 4320.0}.get(int(text[:-1]), float("inf"))
    return float("inf")


def _lookup(mapping: dict[str, Any], key: str) -> Any:
    """Case-insensitive ``mapping[key]``, ``None`` if absent."""
    target = key.strip().lower()
    for candidate, value in mapping.items():
        if isinstance(candidate, str) and candidate.strip().lower() == target:
            return value
    return None


def _tiered_image_price(
    pricing: dict[str, Any], constraints: dict[str, Any], flat_key: str, quality: str | None
) -> float | None:
    """Price one image or inpaint request at the tier the caller is billed for.

    The resolution is the model's ``defaultResolution``, else its lowest listed
    tier. With *quality* the price is ``pricing.quality[resolution][quality]``;
    without it the model's ``defaultQuality`` is used, which is what a request
    that sends no ``quality`` pays. Models without tiers fall back to the flat
    ``generation`` (image) or ``inpaint`` price.
    """
    resolution_prices = pricing.get("resolutions")
    quality_prices = pricing.get("quality")
    tiers = [
        key
        for source in (resolution_prices, quality_prices)
        if isinstance(source, dict)
        for key in source
    ]
    resolution: str | None = None
    default_resolution = constraints.get("defaultResolution")
    if isinstance(default_resolution, str) and any(
        t.lower() == default_resolution.lower() for t in tiers
    ):
        resolution = default_resolution
    elif tiers:
        resolution = min(tiers, key=_resolution_height)

    wanted_quality = quality or constraints.get("defaultQuality")
    if resolution is not None and isinstance(wanted_quality, str):
        by_quality = (
            _lookup(quality_prices, resolution) if isinstance(quality_prices, dict) else None
        )
        if isinstance(by_quality, dict):
            price = _usd(_lookup(by_quality, wanted_quality))
            if price is not None:
                return price
    if resolution is not None and isinstance(resolution_prices, dict):
        price = _usd(_lookup(resolution_prices, resolution))
        if price is not None:
            return price
    return _usd(pricing.get(flat_key))


def _spec_value(spec: Any, key: str) -> Any:
    """Read *key* from a music spec given as a model or as its dict form."""
    if isinstance(spec, Mapping):
        return spec.get(key)
    return getattr(spec, key, None)


def music_request_params(spec: Any, target_seconds: int | None = None) -> dict[str, Any]:
    """Return the ``music.quote`` / ``music.submit`` arguments for a clip length.

    Built from the duration metadata the model declares, so the request never
    carries a ``duration_seconds`` the model would reject:

    - ``duration_options`` (a fixed set, e.g. ``[60, 90, 120]``): the smallest
      option at or above *target_seconds*, or the smallest option when no
      target is given.
    - ``min_duration`` / ``max_duration``: *target_seconds* raised to the
      minimum, or the minimum when no target is given.
    - Neither: no ``duration_seconds`` at all. Venice rejects the parameter on
      these models, which choose the clip length themselves.

    Args:
        spec: A :class:`~venice_ai.types.api.models.MusicModelSpec` or its dict
            form (the selector cache's ``model_spec``).
        target_seconds: The clip length wanted. ``None`` asks for the model's
            shortest valid request.

    Raises:
        ValueError: If *target_seconds* is longer than the longest clip the
            model can make.
    """
    options = _spec_value(spec, "duration_options")
    if isinstance(options, list) and options:
        numeric = sorted(int(o) for o in options if isinstance(o, int | float))
        if not numeric:
            return {}
        if target_seconds is None:
            return {"duration_seconds": numeric[0]}
        covering = [o for o in numeric if o >= target_seconds]
        if not covering:
            raise ValueError(
                f"{target_seconds}s is above the longest duration option {numeric[-1]}s"
            )
        return {"duration_seconds": covering[0]}

    min_duration = _spec_value(spec, "min_duration")
    max_duration = _spec_value(spec, "max_duration")
    has_min = isinstance(min_duration, int | float)
    has_max = isinstance(max_duration, int | float)
    if not has_min and not has_max:
        return {}
    if target_seconds is None:
        seconds = int(min_duration) if has_min else 1
    else:
        seconds = max(target_seconds, int(min_duration)) if has_min else target_seconds
    if has_max and seconds > max_duration:
        raise ValueError(f"{seconds}s is above the longest supported duration {max_duration}s")
    return {"duration_seconds": seconds}


def _ceil_to_cent(usd: float) -> float:
    """Round up to the cent, as Venice rounds every music quote and bill."""
    return math.ceil(round(usd * 100, 6)) / 100


def _music_price(spec: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    """Price a music model for one :data:`MUSIC_REFERENCE_SECONDS` clip.

    The clip length is the request :func:`music_request_params` builds for that
    target; a model that cannot make a clip that long is unpriced. Prices are
    rounded up to the cent, as Venice quotes and bills them. Per-thousand-
    character pricing belongs to text-to-speech models typed as music; it has
    no clip-length equivalent, so those models are unpriced too.
    """
    flat = _usd(pricing.get("generation"))
    if flat is not None:
        return flat
    try:
        params = music_request_params(spec, MUSIC_REFERENCE_SECONDS)
    except ValueError:
        return None
    seconds = float(params.get("duration_seconds", MUSIC_REFERENCE_SECONDS))

    per_second = _usd(pricing.get("per_second"))
    if per_second is not None:
        return _ceil_to_cent(per_second * seconds)
    durations = pricing.get("durations")
    if isinstance(durations, dict):
        covering: list[tuple[float, float]] = []
        for tier in durations.values():
            price = _usd(tier)
            if price is None:
                continue
            low = tier.get("min_seconds", 0)
            high = tier.get("max_seconds")
            if not isinstance(high, int | float):
                continue
            if isinstance(low, int | float) and low <= seconds <= high:
                return price
            if high >= seconds:
                covering.append((float(high), price))
        if covering:
            return min(covering)[1]
    return None


def model_price(model: dict[str, Any], *, quality: str | None = None) -> float | None:
    """Return a comparable USD price for a cached model dict, or ``None``.

    The unit depends on the model type, so prices compare only within a type:

    - chat (``text``) and ``decision``: blended per-million-token price,
      ``(CHAT_INPUT_WEIGHT * input + CHAT_OUTPUT_WEIGHT * output) /
      (CHAT_INPUT_WEIGHT + CHAT_OUTPUT_WEIGHT)``.
    - ``embedding``: input price per million tokens (embeddings produce no
      output tokens).
    - ``tts``: input price per million characters.
    - ``asr``: price per audio second.
    - ``image`` / ``inpaint``: one request at the model's default resolution and
      quality; *quality* prices that quality tier instead.
    - ``music``: one :data:`MUSIC_REFERENCE_SECONDS` clip (or the model's
      longer minimum), rounded up to the cent; unpriced when the model cannot
      make a clip that long.
    - ``video``: always ``None``; the catalog does not price video, use
      ``POST /video/quote`` (see :meth:`DynamicModelSelector.select_cheapest_video_model`).

    A missing or malformed price returns ``None`` and is never read as ``$0``.

    Args:
        model: A model dict from the selector cache (as passed to a
            :data:`ModelSelectorType`).
        quality: Image/inpaint quality tier to price (e.g. ``"low"``).
    """
    spec = model.get("model_spec") or {}
    pricing = spec.get("pricing")
    if not isinstance(pricing, dict):
        return None
    model_type = model.get("type")

    if model_type == "video":
        return None
    if model_type == "embedding":
        return _usd(pricing.get("input"))
    if model_type == "music":
        return _music_price(spec, pricing)
    if model_type in ("text", "decision") or (
        pricing.get("input") is not None and pricing.get("output") is not None
    ):
        input_usd = _usd(pricing.get("input"))
        output_usd = _usd(pricing.get("output"))
        if input_usd is None or output_usd is None:
            return None
        return (CHAT_INPUT_WEIGHT * input_usd + CHAT_OUTPUT_WEIGHT * output_usd) / (
            CHAT_INPUT_WEIGHT + CHAT_OUTPUT_WEIGHT
        )
    if pricing.get("per_audio_second") is not None:
        return _usd(pricing["per_audio_second"])
    if pricing.get("input") is not None:
        return _usd(pricing["input"])
    constraints = spec.get("constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    flat_key = "inpaint" if model_type == "inpaint" or "inpaint" in pricing else "generation"
    return _tiered_image_price(pricing, constraints, flat_key, quality)


def _price_sort_key(
    index: int, model: dict[str, Any], quality: str | None
) -> tuple[bool, float, bool, int, str]:
    """Sort key for :func:`cheapest_model_strategy`; see its tie-break rule."""
    price = model_price(model, quality=quality)
    traits = model.get("traits") or []
    return (
        price is None,
        price if price is not None else 0.0,
        "default" not in traits,
        index,
        str(model.get("id", "")),
    )


def cheapest_model_strategy(candidates: list[dict[str, Any]], *, quality: str | None = None) -> str:
    """Select the single cheapest model by :func:`model_price`.

    Unpriced models sort after every priced one. Models at the same price are
    ordered by a fixed rule so a price tie lands where ``prefer=None`` would:

    1. a model carrying Venice's ``default`` trait first,
    2. then the order the candidates were passed in (catalog order, or the
       ``prefer_recommended`` order when that reordering is on),
    3. then model id, so the result is deterministic.

    Args:
        candidates: Model dicts from the selector cache.
        quality: Image/inpaint quality tier to price (see :func:`model_price`).

    Returns:
        The selected model id.

    Raises:
        NoMatchingModelError: If *candidates* is empty.
    """
    if not candidates:
        raise NoMatchingModelError("No candidates available for selection")
    ranked = min(enumerate(candidates), key=lambda pair: _price_sort_key(pair[0], pair[1], quality))
    return str(ranked[1]["id"])


def _quotes_unavailable(
    resource_type: str, failures: dict[str, Exception], skipped: dict[str, str]
) -> ModelQuotesUnavailableError:
    """Build the error for a quote-ranked pick where every quote failed.

    The per-model exceptions are attached as ``failures`` and, grouped, as
    ``__cause__``, so a caller can tell an invalid key or an outage from a
    catalog with no matching model.
    """
    reasons = {mid: f"{type(exc).__name__}: {exc}" for mid, exc in failures.items()}
    error = ModelQuotesUnavailableError(
        f"None of the {len(failures)} candidate {resource_type} models returned a "
        f"valid quote. Failures: {reasons}",
        resource_type=resource_type,
        failures=failures,
        skipped=skipped,
    )
    if failures:
        error.__cause__ = ExceptionGroup(f"{resource_type} quote failures", list(failures.values()))
    return error


class _CheapestSelector:
    """The :data:`ModelSelectorType` built by :func:`cheapest_selector`.

    ``ranks_full_pool`` tells the chat selector to hand this strategy every
    model that passes the caller's filters. Other selectors receive the
    general-chat pool, which leaves out reasoning models unless reasoning was
    requested (see :meth:`DynamicModelSelector.select_chat_model`).
    """

    ranks_full_pool = True

    def __init__(self, quality: str | None, preferred_models: list[str]) -> None:
        self._quality = quality
        self._preferred = preferred_models

    def __call__(self, candidates: list[dict[str, Any]]) -> str:
        ids = {str(m.get("id")) for m in candidates}
        for model_id in self._preferred:
            if model_id in ids:
                return model_id
        return cheapest_model_strategy(candidates, quality=self._quality)


def cheapest_selector(
    *, quality: str | None = None, preferred_models: list[str] | None = None
) -> ModelSelectorType:
    """Build a :data:`ModelSelectorType` that picks the cheapest candidate.

    The first of *preferred_models* that survives filtering wins outright;
    otherwise :func:`cheapest_model_strategy` decides. This is what
    ``client.models.resolve(..., prefer="cheapest")`` passes to the selector.
    The chat selector gives it every model that passes the explicit filters,
    reasoning models included, so the result is the strictly cheapest match.

    Args:
        quality: Image/inpaint quality tier to price (see :func:`model_price`).
        preferred_models: Model ids that take precedence over price, in order.
    """
    return _CheapestSelector(quality, list(preferred_models or []))


# ---------------------------------------------------------------------------
# Private helpers for video constraint filtering (shared by select_video_model
# and select_cheapest_video_model).
# ---------------------------------------------------------------------------

_VIDEO_RESOLUTION_ORDER = [
    "360p",
    "480p",
    "540p",
    "580p",
    "720p",
    "1080p",
    "1440p",
    "2160p",
    "4k",
]


def _parse_duration_seconds(d: str) -> int:
    """Parse a duration string like ``'5s'`` to integer seconds."""
    try:
        return int(d.rstrip("s"))
    except (ValueError, AttributeError):
        logger.warning(
            "Unparseable duration string %r — defaulting to 0 seconds. "
            "Expected a format like '5s' or '10s'.",
            d,
        )
        return 0


def _meets_resolution(model_resolutions: list[str], min_res: str) -> bool:
    """Return ``True`` if any resolution in *model_resolutions* meets *min_res*."""
    if not model_resolutions:
        return False
    min_idx = _VIDEO_RESOLUTION_ORDER.index(min_res) if min_res in _VIDEO_RESOLUTION_ORDER else -1
    if min_idx == -1:
        return False
    return any(
        _VIDEO_RESOLUTION_ORDER.index(r) >= min_idx
        for r in model_resolutions
        if r in _VIDEO_RESOLUTION_ORDER
    )


def _meets_duration(model_durations: list[str], min_dur: str) -> bool:
    """Return ``True`` if any duration in *model_durations* meets *min_dur*."""
    if not model_durations:
        return False
    min_secs = _parse_duration_seconds(min_dur)
    return any(_parse_duration_seconds(d) >= min_secs for d in model_durations)


_DURATION_PATTERN = re.compile(r"^(\d+)\s*(?:s|secs?|seconds?)?$")


def _normalize_duration(duration: int | str) -> str:
    """Normalize ``5`` / ``"5"`` / ``"5s"`` / ``"5 seconds"`` to the catalog's ``"5s"``.

    Anything else (e.g. ``"Auto"``) is lowercased and compared as-is.
    """
    if isinstance(duration, int):
        return f"{duration}s"
    text = duration.strip().lower()
    match = _DURATION_PATTERN.match(text)
    return f"{match.group(1)}s" if match else text


def _offers(values: Any, wanted: str) -> bool:
    """Return ``True`` if the catalog list *values* contains *wanted*, ignoring case.

    Catalog tiers are not consistently cased (``"1080p"`` vs ``"1080P"``), and a
    missing or non-list field means the model does not advertise the option.
    """
    if not isinstance(values, list):
        return False
    target = wanted.strip().lower()
    return any(isinstance(v, str) and v.strip().lower() == target for v in values)


# MusicModelSpec fields copied into the selector cache (see _fetch_models).
_MUSIC_SPEC_FIELDS = (
    "voice_changer",
    "supports_force_instrumental",
    "voices",
    "default_voice",
    "min_duration",
    "max_duration",
    "default_duration",
    "duration_options",
    "lyrics_required",
)

# Substrings of a ``type == "music"`` model's id or name that mark it as speech,
# sound-effect or voice-changer audio rather than music. The name check matters
# because the catalog has no positive "generates music" flag; descriptions are
# not checked because real music models mention sound effects in passing.
_NON_MUSIC_PATTERNS = (
    "sound-effect",
    "sound effect",
    "sfx",
    "text-to-audio",
    "tts",
    "speech",
    "voice-chang",
    "voice chang",
)


def _is_music_generation_model(model_data: dict[str, Any]) -> bool:
    """Return ``True`` for music generators, ``False`` for other music-typed models.

    Venice types text-to-speech, sound-effect and voice-changer models as
    ``music`` too. A model is treated as non-music when it is a voice changer,
    lists TTS voices, is priced per thousand characters, or its id or name
    matches :data:`_NON_MUSIC_PATTERNS`.
    """
    spec = model_data.get("model_spec", {})
    if spec.get("voice_changer") or spec.get("voices") or spec.get("default_voice"):
        return False
    if _has_price(spec.get("pricing"), "per_thousand_characters"):
        return False
    haystack = " ".join(str(model_data.get(key, "")) for key in ("id", "name")).lower()
    return not any(pattern in haystack for pattern in _NON_MUSIC_PATTERNS)


def _has_price(pricing: Any, key: str) -> bool:
    """Return ``True`` if the cached *pricing* dict lists a USD price under *key*."""
    if not isinstance(pricing, dict):
        return False
    tier = pricing.get(key)
    return isinstance(tier, dict) and isinstance(tier.get("usd"), int | float)


def _is_e2ee_model(model_data: dict[str, Any]) -> bool:
    """Return ``True`` for end-to-end encrypted chat models.

    These need the client-side encryption flow, so a plain request to them
    fails. The catalog flags them with ``supportsE2EE``; the ``e2ee-`` id
    prefix is checked too in case the flag is missing.
    """
    capabilities = model_data.get("model_spec", {}).get("capabilities", {})
    if capabilities.get("supportsE2EE", False):
        return True
    return str(model_data.get("id", "")).startswith("e2ee-")


def _video_constraints(model_data: dict[str, Any]) -> dict[str, Any]:
    """The cached ``model_spec.constraints`` dict of a video model (``{}`` if absent)."""
    constraints = model_data.get("model_spec", {}).get("constraints")
    return constraints if isinstance(constraints, dict) else {}


def _fixed_duration_seconds(tier: Any) -> int | None:
    """``"4s"`` / ``"4"`` -> ``4``; ``None`` for non-numeric tiers such as ``"Auto"``."""
    if not isinstance(tier, str):
        return None
    match = _DURATION_PATTERN.match(tier.strip().lower())
    return int(match.group(1)) if match else None


def cheapest_video_params(
    constraints: VideoModelConstraints | Mapping[str, Any],
    *,
    duration: int | str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    audio: bool | None = False,
) -> dict[str, Any]:
    """Return the cheapest valid ``video.quote`` / ``video.submit`` arguments for a model.

    Built only from values the model lists, so the request never carries an
    option the model would reject:

    - ``duration_seconds``: the shortest fixed duration (``"Auto"`` tiers are
      ignored), or *duration* when given.
    - ``resolution``: the lowest listed tier, spelled as the catalog spells it,
      or *resolution* when given. Omitted when the model lists none.
    - ``aspect_ratio``: *aspect_ratio* when given, else ``"16:9"`` when listed,
      else the first listed ratio. Omitted when the model lists none.
    - ``audio``: *audio*, sent only when the model's ``audio_configurable`` is
      set. ``False`` (default) asks for the cheaper silent clip; ``None``
      leaves it to the model.

    Args:
        constraints: A :class:`~venice_ai.types.api.models.VideoModelConstraints`
            or its dict form (``model_spec.constraints``).
        duration: Duration to use instead of the shortest (``5``, ``"5"``, ``"5s"``).
        resolution: Resolution tier to use instead of the lowest.
        aspect_ratio: Aspect ratio to use instead of the 16:9 preference.
        audio: Audio setting for models that make it configurable.

    Raises:
        ValueError: If the model lists no fixed duration, or does not list a
            requested *duration*, *resolution* or *aspect_ratio*, or cannot
            produce audio when ``audio=True``.
    """
    data: Mapping[str, Any] = (
        constraints.model_dump() if isinstance(constraints, VideoModelConstraints) else constraints
    )
    params: dict[str, Any] = {}

    durations = [d for d in data.get("durations") or [] if _fixed_duration_seconds(d) is not None]
    if duration is not None:
        wanted = _normalize_duration(duration)
        match = [d for d in durations if _normalize_duration(d) == wanted]
        if not match:
            raise ValueError(f"duration {wanted!r} not offered; listed: {data.get('durations')}")
        params["duration_seconds"] = match[0]
    elif durations:
        params["duration_seconds"] = min(durations, key=lambda d: _fixed_duration_seconds(d) or 0)
    else:
        raise ValueError(f"no fixed durations listed: {data.get('durations')}")

    resolutions = [r for r in data.get("resolutions") or [] if isinstance(r, str)]
    if resolution is not None:
        match = [r for r in resolutions if r.strip().lower() == resolution.strip().lower()]
        if not match:
            raise ValueError(f"resolution {resolution!r} not offered; listed: {resolutions}")
        params["resolution"] = match[0]
    elif resolutions:
        params["resolution"] = min(resolutions, key=_resolution_height)

    ratios = [r for r in data.get("aspect_ratios") or [] if isinstance(r, str)]
    if aspect_ratio is not None:
        if ratios and aspect_ratio not in ratios:
            raise ValueError(f"aspect ratio {aspect_ratio!r} not offered; listed: {ratios}")
        params["aspect_ratio"] = aspect_ratio
    elif ratios:
        params["aspect_ratio"] = "16:9" if "16:9" in ratios else ratios[0]

    if audio is True and not data.get("audio", False):
        raise ValueError("model does not generate audio")
    if audio is not None and data.get("audio_configurable", False):
        params["audio"] = audio

    return params


#: What an ``image-to-video`` model needs besides the prompt.
#:
#: - ``"image"``: one start image (plain image-to-video).
#: - ``"reference"``: reference images that guide subjects or style
#:   (reference-to-video, "R2V").
#: - ``"first_last_frame"``: a first and a last frame.
#: - ``"transition"``: two images to transition between.
#: - ``"multi_angle"``: several views of one subject.
type VideoInputMode = Literal["image", "reference", "first_last_frame", "transition", "multi_angle"]

# Venice types all of these models ``image-to-video`` and gives them
# indistinguishable constraints, so the id and display name are the only signal.
_VIDEO_INPUT_MODE_PATTERNS: tuple[tuple[VideoInputMode, re.Pattern[str]], ...] = (
    ("first_last_frame", re.compile(r"first[- ]last[- ]frame")),
    ("multi_angle", re.compile(r"multi[- ]angle")),
    ("transition", re.compile(r"transition")),
    ("reference", re.compile(r"reference|\br2v\b")),
)


def video_input_mode(model: Mapping[str, Any]) -> VideoInputMode:
    """Classify what an ``image-to-video`` model needs as input.

    The Venice catalog types plain image-to-video, reference-to-video,
    transition, first/last-frame and multi-angle models all as
    ``model_type="image-to-video"``, with constraints that do not tell them
    apart. The model id and display name do, so this matches on them:
    ``reference`` / ``R2V``, ``transition``, ``first-last-frame`` and
    ``multi-angle``. Anything else is a plain ``"image"`` model.

    Args:
        model: A selector-cache model dict, or any mapping with ``id`` and
            optionally ``name``.
    """
    haystack = " ".join(str(model.get(key) or "") for key in ("id", "name")).lower()
    for mode, pattern in _VIDEO_INPUT_MODE_PATTERNS:
        if pattern.search(haystack):
            return mode
    return "image"


def _filter_video_candidates(
    candidates: list[str],
    models_data: dict[str, Any],
    *,
    model_type: str | None = None,
    require_audio: bool = False,
    min_resolution: str | None = None,
    min_duration: str | None = None,
    require_duration: int | str | None = None,
    require_resolution: str | None = None,
    require_audio_configurable: bool = False,
    exclude_beta: bool = False,
    input_mode: VideoInputMode | None = None,
) -> list[str]:
    """Filter video model candidates by constraint criteria.

    This is the single implementation of video-constraint filtering used by
    both :meth:`DynamicModelSelector.select_video_model` and
    :meth:`DynamicModelSelector.select_cheapest_video_model`.

    ``model_type="image-to-video"`` without an *input_mode* keeps plain
    image-to-video models only (``input_mode="image"``). An *input_mode*
    without a *model_type* implies ``"image-to-video"``.
    """
    if input_mode is not None and model_type is None:
        model_type = "image-to-video"
    if model_type == "image-to-video" and input_mode is None:
        input_mode = "image"
    filtered: list[str] = []
    for model_id in candidates:
        model_data = models_data.get(model_id, {})
        model_spec = model_data.get("model_spec", {})
        constraints = model_spec.get("constraints", {})

        # Filter by model_type
        if model_type and constraints.get("model_type") != model_type:
            continue

        # Filter image-to-video models by the input they need
        if input_mode is not None and video_input_mode(model_data or {"id": model_id}) != (
            input_mode
        ):
            continue

        # Filter by audio support
        if require_audio and not constraints.get("audio", False):
            continue

        # Filter by minimum resolution
        if min_resolution and not _meets_resolution(
            constraints.get("resolutions", []), min_resolution
        ):
            continue

        # Filter by minimum duration
        if min_duration and not _meets_duration(constraints.get("durations", []), min_duration):
            continue

        # Filter by exact duration (the model must list it, not just exceed it)
        if require_duration is not None and not _offers(
            constraints.get("durations"), _normalize_duration(require_duration)
        ):
            continue

        # Filter by exact resolution tier
        if require_resolution is not None and not _offers(
            constraints.get("resolutions"), require_resolution
        ):
            continue

        # Filter by whether audio can be switched on/off per request
        if require_audio_configurable and not constraints.get("audio_configurable", False):
            continue

        # Filter by beta status
        if exclude_beta and model_data.get("beta", False):
            continue

        filtered.append(model_id)
    return filtered


# ---------------------------------------------------------------------------
# Private helper for image-generation capability filtering
# (used by select_image_model).
# ---------------------------------------------------------------------------

# Substrings identifying ``type == "image"`` models that do NOT perform
# text-to-image *generation* and would 400 on an ``image.create(prompt=...)``
# call (e.g. ``bria-bg-remover``, a background remover).
#
# Why a denylist instead of a positive capability check: the models catalog
# exposes no clean positive "supports generation" signal for image models —
# ``model_spec.capabilities`` is empty ``{}`` for every image-type model, and
# ``traits`` / ``constraints.aspectRatios`` are inconsistent across genuine
# generators (e.g. ``venice-sd35`` and ``z-image-turbo`` are real generators
# that carry neither). Upscalers and inpainters are already excluded upstream
# because Venice types them separately (``upscale`` / ``inpaint``), not
# ``image``, so the only non-generators currently mis-typed as ``image`` are
# background removers. The denylist also covers ``upscal``/``enhance`` patterns
# defensively in case such models are ever surfaced under the ``image`` type.
_IMAGE_NON_GENERATOR_PATTERNS = (
    "bg-remover",
    "background",
    "remover",
    "removal",
    "upscal",
    "enhance",
)


def _is_image_generation_model(model_data: dict[str, Any]) -> bool:
    """Return ``True`` unless *model_data* is a known non-generative image model.

    Distinguishes text-to-image generators (``venice-sd35``, ``qwen-image``,
    ``flux-*`` …) from non-generators like ``bria-bg-remover`` that share the
    ``image`` resource type but reject ``image.create(prompt=...)`` requests.
    Matches defensively on both the model id and its human-readable name so a
    renamed background remover ("Background Remover") is still excluded.
    """
    haystack = " ".join(str(model_data.get(key, "")) for key in ("id", "name")).lower()
    return not any(pattern in haystack for pattern in _IMAGE_NON_GENERATOR_PATTERNS)


class DynamicModelSelector:
    """
    Dynamic model selector that fetches available models and provides
    intelligent selection for production and testing scenarios.

    Supports custom selection strategies via the default_selector parameter
    or per-call selector argument. Strategies receive full model dictionaries
    including pricing data for cost-aware selection.
    """

    def __init__(
        self,
        client: Any,
        cache_ttl: float = 300.0,
        default_selector: ModelSelectorType | None = None,
    ):
        """
        Initialize the model selector.

        Args:
            client: Venice AI client instance for API calls
            cache_ttl: Time-to-live for model cache in seconds
            default_selector: Optional custom selection function that receives
                a list of model dicts and returns the selected model ID.
                Used as fallback when no per-call selector is provided.
        """
        self.client = client
        self._cache = ModelCache(ttl_seconds=cache_ttl)
        self._fetch_lock = asyncio.Lock()
        self.default_selector = default_selector

    async def _fetch_models(self, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch models from API with caching."""
        if not force_refresh and not self._cache.is_expired():
            return self._cache.models

        async with self._fetch_lock:
            # Double-check after acquiring lock
            if not force_refresh and not self._cache.is_expired():
                return self._cache.models

            try:
                logger.info("Fetching available models from API...")
                # Use the models endpoint to get ALL available models
                response = await self.client.models.list(type="all")

                # Convert response to dict format
                models_dict = {}
                if hasattr(response, "data") and response.data:
                    for model in response.data:
                        model_id = model.id if hasattr(model, "id") else str(model)
                        model_data = {
                            "id": model_id,
                            "object": getattr(model, "object", "model"),
                            "type": getattr(model, "type", "unknown"),  # Store the actual type
                            "created": getattr(model, "created", time.time()),
                            "owned_by": getattr(model, "owned_by", "unknown"),
                        }

                        # Store model_spec information if available
                        if hasattr(model, "model_spec"):
                            model_spec = model.model_spec

                            # Build the model_spec dictionary with capabilities and pricing
                            model_spec_dict: dict[str, Any] = {
                                "capabilities": {},
                                "pricing": None,
                            }

                            # Extract capabilities if available
                            if hasattr(model_spec, "capabilities"):
                                capabilities = model_spec.capabilities
                                model_spec_dict["capabilities"] = {
                                    "supportsFunctionCalling": getattr(
                                        capabilities, "supportsFunctionCalling", False
                                    ),
                                    "supportsVision": getattr(
                                        capabilities, "supportsVision", False
                                    ),
                                    "supportsWebSearch": getattr(
                                        capabilities, "supportsWebSearch", False
                                    ),
                                    "optimizedForCode": getattr(
                                        capabilities, "optimizedForCode", False
                                    ),
                                    "supportsReasoning": getattr(
                                        capabilities, "supportsReasoning", False
                                    ),
                                    "supportsAudioInput": getattr(
                                        capabilities, "supportsAudioInput", False
                                    ),
                                    "supportsVideoInput": getattr(
                                        capabilities, "supportsVideoInput", False
                                    ),
                                    "supportsLogProbs": getattr(
                                        capabilities, "supportsLogProbs", False
                                    ),
                                    "supportsResponseSchema": getattr(
                                        capabilities, "supportsResponseSchema", False
                                    ),
                                    "quantization": getattr(
                                        capabilities, "quantization", "not-available"
                                    ),
                                    "reasoningEffortOptions": getattr(
                                        capabilities, "reasoningEffortOptions", None
                                    ),
                                    "supportsMultipleImages": getattr(
                                        capabilities, "supportsMultipleImages", False
                                    ),
                                    "supportsE2EE": getattr(capabilities, "supportsE2EE", False),
                                    "maxImages": getattr(capabilities, "maxImages", None),
                                }

                            # Music-typed models carry their capabilities on the
                            # spec itself. Keep the fields that tell real music
                            # generators apart from TTS, sound-effect and
                            # voice-changer models sharing the type.
                            for attr in _MUSIC_SPEC_FIELDS:
                                value = getattr(model_spec, attr, None)
                                if value is not None:
                                    model_spec_dict[attr] = value

                            # Image models declare web search on the spec itself
                            # rather than in a capabilities block.
                            spec_web_search = getattr(model_spec, "supportsWebSearch", None)
                            if spec_web_search is True:
                                model_spec_dict["supportsWebSearch"] = True

                            # Extract pricing if available
                            if hasattr(model_spec, "pricing") and model_spec.pricing:
                                pricing = model_spec.pricing
                                # Convert pricing to dict, handling both Pydantic models
                                # and plain dicts
                                if hasattr(pricing, "model_dump"):
                                    model_spec_dict["pricing"] = pricing.model_dump()
                                elif isinstance(pricing, dict):
                                    model_spec_dict["pricing"] = pricing
                                else:
                                    # Manual extraction for edge cases
                                    pricing_dict = {}
                                    for attr in [
                                        "input",
                                        "output",
                                        "cache_input",
                                        "generation",
                                        "upscale",
                                    ]:
                                        if hasattr(pricing, attr):
                                            val = getattr(pricing, attr)
                                            if val is not None:
                                                if hasattr(val, "model_dump"):
                                                    pricing_dict[attr] = val.model_dump()
                                                elif isinstance(val, dict):
                                                    pricing_dict[attr] = val
                                                else:
                                                    # Extract usd/diem from tier
                                                    pricing_dict[attr] = {
                                                        "usd": getattr(val, "usd", None),
                                                        "diem": getattr(val, "diem", None),
                                                    }
                                    if pricing_dict:
                                        model_spec_dict["pricing"] = pricing_dict

                            # Extract constraints if available (for image, video, inpaint models)
                            if hasattr(model_spec, "constraints") and model_spec.constraints:
                                constraints = model_spec.constraints
                                if hasattr(constraints, "model_dump"):
                                    model_spec_dict["constraints"] = constraints.model_dump()
                                elif isinstance(constraints, dict):
                                    model_spec_dict["constraints"] = constraints
                                else:
                                    # Manual extraction for video/image constraints
                                    constraints_dict = {}
                                    for attr in [
                                        "model_type",
                                        "aspect_ratios",
                                        "resolutions",
                                        "durations",
                                        "audio",
                                        "audio_configurable",
                                        "video_input",
                                        "promptCharacterLimit",
                                        "steps",
                                        "widthHeightDivisor",
                                        "combineImages",
                                    ]:
                                        if hasattr(constraints, attr):
                                            val = getattr(constraints, attr)
                                            if val is not None:
                                                if hasattr(val, "model_dump"):
                                                    constraints_dict[attr] = val.model_dump()
                                                else:
                                                    constraints_dict[attr] = val
                                    if constraints_dict:
                                        model_spec_dict["constraints"] = constraints_dict

                            # Assign the complete model_spec dictionary
                            model_data["model_spec"] = model_spec_dict

                            # Extract additional metadata from model_spec
                            model_data["availableContextTokens"] = getattr(
                                model_spec, "availableContextTokens", None
                            )
                            model_data["beta"] = getattr(model_spec, "beta", False) or getattr(
                                model_spec, "betaModel", False
                            )
                            model_data["privacy"] = getattr(model_spec, "privacy", None)
                            model_data["uncensored"] = (
                                getattr(model_spec, "uncensored", False) is True
                            )
                            model_data["model_sets"] = getattr(model_spec, "model_sets", []) or []
                            model_data["name"] = getattr(model_spec, "name", "")
                            model_data["description"] = getattr(model_spec, "description", "")
                            model_data["offline"] = getattr(model_spec, "offline", False)
                            deprecation = getattr(model_spec, "deprecation", None)
                            model_data["deprecation_date"] = (
                                getattr(deprecation, "date", None) if deprecation else None
                            )

                            # Extract traits if available
                            if hasattr(model_spec, "traits"):
                                model_data["traits"] = (
                                    list(model_spec.traits) if model_spec.traits else []
                                )

                        models_dict[model_id] = model_data

                self._cache.update(models_dict)
                logger.info(f"Successfully fetched {len(models_dict)} models")
                return models_dict

            except asyncio.CancelledError:
                raise  # Always re-raise for graceful shutdown
            except (ValueError, TypeError, AttributeError, OSError) as e:
                logger.exception(f"Failed to fetch models: {e}")
                if self._cache.models:
                    logger.warning("Using cached models despite fetch failure")
                    return self._cache.models
                raise

    async def cheapest_exclusions(
        self, resource_type: str, *, exclude_beta: bool, exclude_e2ee: bool
    ) -> set[str]:
        """Return the ids that ``prefer="cheapest"`` skips by default.

        Beta models (when *exclude_beta*) and end-to-end encrypted chat models
        (when *exclude_e2ee*) are often the cheapest entries in the catalog, but
        a plain request to an e2ee model fails and beta models can change or
        disappear without notice.

        Args:
            resource_type: Catalog type to scan (``"text"``, ``"image"`` ...).
            exclude_beta: Include beta models in the result.
            exclude_e2ee: Include end-to-end encrypted models in the result.
        """
        models = await self._fetch_models()
        skipped: set[str] = set()
        for model_id, model_data in models.items():
            if model_data.get("type") != resource_type:
                continue
            if (exclude_beta and model_data.get("beta", False)) or (
                exclude_e2ee and _is_e2ee_model(model_data)
            ):
                skipped.add(model_id)
        return skipped

    async def select_by_trait(self, trait: str, resource_type: str | None = None) -> str | None:
        """
        Select the model assigned to a specific Venice trait.

        Venice assigns traits like "default", "fastest", "default_code",
        "default_reasoning", "default_vision", "function_calling_default",
        "most_intelligent", "most_uncensored" to indicate canonical model roles.

        Args:
            trait: The trait to search for (e.g., "default", "fastest", "default_code")
            resource_type: Optional filter by resource type (e.g., "text", "image")

        Returns:
            The model ID with the matching trait, or None if no model has that trait
        """
        models = await self._fetch_models()

        for model_id, model_data in models.items():
            # Filter by resource type if specified
            if resource_type and model_data.get("type") != resource_type:
                continue

            traits = model_data.get("traits", [])
            if trait in traits:
                logger.info(f"Found model '{model_id}' with trait '{trait}'")
                return model_id

        logger.debug(
            f"No model found with trait '{trait}'"
            + (f" and type '{resource_type}'" if resource_type else "")
        )
        return None

    def _get_trait_model(self, trait: str, resource_type: str | None = None) -> str | None:
        """
        Synchronous trait lookup against the already-populated cache.

        This is a non-async helper for use inside methods that have already
        called _fetch_models(). Returns None if cache is empty or no match.
        """
        if not self._cache.models:
            return None

        for model_id, model_data in self._cache.models.items():
            if resource_type and model_data.get("type") != resource_type:
                continue
            traits = model_data.get("traits", [])
            if trait in traits:
                return model_id

        return None

    async def _select_simple_model(
        self,
        resource_type: str,
        label: str,
        *,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
        candidate_filter: Callable[[dict[str, Any]], bool] | None = None,
        filter_description: str = "",
    ) -> str:
        """Generic model selection for simple resource types.

        Shared implementation for resource types that follow the same
        pattern: fetch available models, apply exclusions, run custom
        selector, try preferred list, then fall back to first candidate.

        ``candidate_filter`` receives each candidate's cached model dict and
        narrows the pool before any selector, preference or fallback runs;
        ``filter_description`` names the active requirements in the error
        raised when nothing passes.
        """
        available = await self.get_available_models(resource_type=resource_type)
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise NoMatchingModelError(
                f"No available {label} models found", resource_type=resource_type
            )

        if candidate_filter is not None:
            candidates = [m for m in candidates if candidate_filter(self._cache.models.get(m, {}))]
            if not candidates:
                raise NoMatchingModelError(
                    f"No {label} models found matching requirements: {filter_description}",
                    resource_type=resource_type,
                )

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected {label} model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred {label} model: {preferred}")
                    return preferred

        # Fallback to first available model
        selected = candidates[0]
        logger.info(f"Selected fallback {label} model: {selected}")
        return selected

    async def _select_trait_model(
        self,
        trait_name: str,
        label: str,
        capability_kwarg: str,
        *,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """Select model by trait, falling back to chat model with capability.

        Shared implementation for text models that first check a Venice
        trait (e.g. ``default_code``) and, if that isn't available,
        delegate to :meth:`select_chat_model` with the appropriate
        capability requirement.
        """
        # Try trait-based selection first
        await self._fetch_models()
        trait_model = self._get_trait_model(trait_name, resource_type="text")

        exclude_models = exclude_models or set()
        available = await self.get_available_models(resource_type="text")

        if trait_model and trait_model in available and trait_model not in exclude_models:
            if preferred_models:
                for preferred in preferred_models:
                    if preferred in available and preferred not in exclude_models:
                        logger.info(f"Selected preferred {label} model: {preferred}")
                        return preferred

            logger.info(f"Selected {label} model via '{trait_name}' trait: {trait_model}")
            return trait_model

        # Fall back to capability-based selection
        capability_flag: dict[str, Any] = {capability_kwarg: True}
        return await self.select_chat_model(
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
            **capability_flag,
        )

    @staticmethod
    def _is_past_deprecation(model_data: dict[str, Any]) -> bool:
        """Return True iff the model has a deprecation date that has already passed.

        Models with no ``deprecation_date``, an unparseable date, or a date in
        the future are considered active.
        """
        date_str = model_data.get("deprecation_date")
        if not date_str:
            return False
        try:
            # The API returns dates like "2026-04-15T00:00:00.000Z"; fromisoformat
            # accepts the trailing offset since Python 3.11 but not the literal "Z"
            # before 3.11, so normalize defensively.
            normalized = date_str.replace("Z", "+00:00")
            deprecation_at = datetime.fromisoformat(normalized)
            if deprecation_at.tzinfo is None:
                deprecation_at = deprecation_at.replace(tzinfo=UTC)
            return datetime.now(UTC) >= deprecation_at
        except (ValueError, TypeError):
            return False

    async def get_available_models(
        self, resource_type: str | None = None, force_refresh: bool = False
    ) -> list[str]:
        """
        Get list of available models, excluding offline and past-deprecation models.

        Args:
            resource_type: Filter by resource type ('text', 'image', 'video', etc.)
            force_refresh: Force refresh of model cache

        Returns:
            List of available model IDs (excludes offline models and models whose
            ``deprecation.date`` has passed). Callers who need a deprecated model
            should pass it directly to ``client.chat.completions.create(model=...)``
            rather than going through the resolver.
        """
        await self._fetch_models(force_refresh=force_refresh)
        all_models = self._cache.get_models(resource_type=resource_type)

        available = []
        for model_id in all_models:
            model_data = self._cache.models.get(model_id, {})
            if model_data.get("offline", False):
                continue
            if self._is_past_deprecation(model_data):
                logger.debug(
                    "Skipping deprecated model %s (deprecation_date=%s)",
                    model_id,
                    model_data.get("deprecation_date"),
                )
                continue
            available.append(model_id)
        return available

    def _is_reasoning_model(self, model_id: str) -> bool:
        """Return True if ``model_id`` advertises reasoning support.

        Reasoning models spend their token budget on internal thinking and
        frequently return an empty ``message.content`` under small
        ``max_completion_tokens`` limits. Callers that want guaranteed visible
        content (general chat, comparison/concurrency tests) use this to filter
        them out unless reasoning is explicitly required.
        """
        return bool(
            self._cache.models.get(model_id, {})
            .get("model_spec", {})
            .get("capabilities", {})
            .get("supportsReasoning", False)
        )

    async def select_chat_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        require_function_calling: bool = False,
        require_vision: bool = False,
        require_reasoning: bool = False,
        require_code_optimization: bool = False,
        require_response_schema: bool = False,
        min_context_tokens: int | None = None,
        require_private: bool = False,
        exclude_beta: bool = False,
        prefer_recommended: bool = False,
        selector: ModelSelectorType | None = None,
        require_web_search: bool = False,
        require_reasoning_effort: str | None = None,
        require_prompt_caching: bool = False,
        require_multiple_images: bool = False,
        require_e2ee: bool = False,
        exclude_reasoning: bool = False,
        exclude_uncensored: bool = False,
    ) -> str:
        """
        Select a suitable chat completion model.

        Ranking, once every filter has applied:

        1. A custom ``selector`` (or ``default_selector``). A price-ranking
           selector built by :func:`cheapest_selector` sees every filtered
           model. Any other selector sees the general-chat pool described in
           step 3.
        2. The first of ``preferred_models`` that passed the filters.
        3. For general chat (``require_reasoning`` not set), models that do not
           reason are preferred when any passed the filters: a reasoning model
           can spend a small ``max_completion_tokens`` budget on thinking and
           return empty ``content``. Within that pool, the model with Venice's
           ``default`` trait, then catalog order. Pass ``exclude_reasoning=True``
           to make this a hard filter instead of a preference.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            require_function_calling: If True, only select models that support function calling
            require_vision: If True, only select models that support vision (image input)
            require_reasoning: If True, only select models that support reasoning
                with thinking blocks
            require_code_optimization: If True, only select models optimized for
                code generation
            require_response_schema: If True, only select models that support
                structured output via response schema
            min_context_tokens: If set, only select models with at least this many
                context tokens
            require_private: If True, only select models with privacy="private"
                (no data stored by provider)
            exclude_beta: If True, exclude models marked as beta
            prefer_recommended: If True, prefer models in Venice's
                "venice_recommendations" model set
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.
            require_web_search: If True, only select models that support web search
            require_reasoning_effort: Only select models whose
                ``reasoningEffortOptions`` include this value (e.g. ``"none"`` for
                models that can run with reasoning switched off)
            require_prompt_caching: If True, only select models whose catalog
                entry lists a cached-input price (``pricing.cache_input``).
                Venice exposes no prompt-caching capability flag; a listed
                price does not guarantee that a cache hit is served or reported
            require_multiple_images: If True, only select models that accept
                several images in one request (``supportsMultipleImages``)
            require_e2ee: If True, only select end-to-end encrypted models
                (``supportsE2EE``)
            exclude_reasoning: If True, only select models that never reason
                (``supportsReasoning`` is false). Use it when the caller needs
                visible ``content`` under a small token budget. Cannot be
                combined with ``require_reasoning`` or
                ``require_reasoning_effort``. To keep reasoning-capable models
                but switch reasoning off, use
                ``require_reasoning_effort="none"`` and send
                ``reasoning_effort="none"`` with the request instead.
            exclude_uncensored: If True, skip models Venice flags as
                ``uncensored``.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable model is found.
            ValueError: If ``exclude_reasoning`` is combined with
                ``require_reasoning`` or ``require_reasoning_effort``.
        """
        if exclude_reasoning and (require_reasoning or require_reasoning_effort is not None):
            raise ValueError(
                "exclude_reasoning=True cannot be combined with require_reasoning or "
                "require_reasoning_effort: the filters admit disjoint sets of models"
            )
        available = await self.get_available_models(resource_type="text")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise NoMatchingModelError("No available chat models found", resource_type="chat")

        # Every capability filter, function calling included, narrows the same
        # pool, so the requirements always apply together.
        has_capability_filter = any(
            [
                require_function_calling,
                require_vision,
                require_reasoning,
                require_code_optimization,
                require_response_schema,
                min_context_tokens is not None,
                require_private,
                exclude_beta,
                require_web_search,
                require_reasoning_effort is not None,
                require_prompt_caching,
                require_multiple_images,
                require_e2ee,
                exclude_reasoning,
                exclude_uncensored,
            ]
        )

        if has_capability_filter:
            models_data = await self._fetch_models()
            filtered = []
            for model_id in candidates:
                model_data = models_data.get(model_id, {})
                model_spec = model_data.get("model_spec", {})
                capabilities = model_spec.get("capabilities", {})

                # Check function calling support
                if require_function_calling and not capabilities.get(
                    "supportsFunctionCalling", False
                ):
                    continue

                # Check vision support
                if require_vision and not capabilities.get("supportsVision", False):
                    continue

                # Check reasoning support
                if require_reasoning and not capabilities.get("supportsReasoning", False):
                    continue

                # Check code optimization
                if require_code_optimization and not capabilities.get("optimizedForCode", False):
                    continue

                # Check response schema support
                if require_response_schema and not capabilities.get(
                    "supportsResponseSchema", False
                ):
                    continue

                # Check minimum context tokens
                if min_context_tokens is not None:
                    available_context = model_data.get("availableContextTokens")
                    if available_context is None or available_context < min_context_tokens:
                        continue

                # Check privacy requirement
                if require_private and model_data.get("privacy") != "private":
                    continue

                # Check beta exclusion
                if exclude_beta and model_data.get("beta", False):
                    continue

                # Check web search support
                if require_web_search and not capabilities.get("supportsWebSearch", False):
                    continue

                # Check the model accepts the requested reasoning_effort value
                if require_reasoning_effort is not None and not _offers(
                    capabilities.get("reasoningEffortOptions"), require_reasoning_effort
                ):
                    continue

                # Check for a cached-input price, the only caching signal the
                # catalog carries.
                if require_prompt_caching and not _has_price(
                    model_spec.get("pricing"), "cache_input"
                ):
                    continue

                # Check several images per request
                if require_multiple_images and not capabilities.get(
                    "supportsMultipleImages", False
                ):
                    continue

                # Check end-to-end encryption
                if require_e2ee and not _is_e2ee_model(model_data):
                    continue

                # Check models that never reason
                if exclude_reasoning and capabilities.get("supportsReasoning", False):
                    continue

                # Check the uncensored flag
                if exclude_uncensored and model_data.get("uncensored", False):
                    continue

                filtered.append(model_id)

            if not filtered:
                prefix = (
                    "No function calling capable models found"
                    if require_function_calling
                    else "No chat models found"
                )
                raise NoMatchingModelError(
                    f"{prefix} matching requirements: "
                    f"function_calling={require_function_calling}, "
                    f"vision={require_vision}, reasoning={require_reasoning}, "
                    f"code={require_code_optimization}, schema={require_response_schema}, "
                    f"min_context={min_context_tokens}, private={require_private}, "
                    f"exclude_beta={exclude_beta}, web_search={require_web_search}, "
                    f"reasoning_effort={require_reasoning_effort!r}, "
                    f"prompt_caching={require_prompt_caching}, "
                    f"multiple_images={require_multiple_images}, e2ee={require_e2ee}, "
                    f"exclude_reasoning={exclude_reasoning}, "
                    f"exclude_uncensored={exclude_uncensored}",
                    resource_type="chat",
                )
            candidates = filtered

        if require_function_calling:
            return self._pick_function_calling_model(
                candidates, preferred_models=preferred_models, selector=selector
            )

        # Prefer Venice-recommended models if requested
        if prefer_recommended and len(candidates) > 1:
            recommended = []
            non_recommended = []
            models_data_for_ranking = await self._fetch_models()
            for mid in candidates:
                model_data = models_data_for_ranking.get(mid, {})
                model_sets = model_data.get("model_sets", [])
                if "venice_recommendations" in model_sets:
                    recommended.append(mid)
                else:
                    non_recommended.append(mid)
            if recommended:
                candidates = recommended + non_recommended
                logger.debug(
                    f"Reordered candidates to prefer {len(recommended)} Venice-recommended models"
                )

        # The general-chat pool: models that answer without reasoning, when any
        # passed the filters. Reasoning models consume thinking tokens from the
        # completion budget and can return empty content under small limits.
        general_pool = candidates
        if not require_reasoning and len(candidates) > 1:
            non_reasoning = [m for m in candidates if not self._is_reasoning_model(m)]
            if non_reasoning:
                general_pool = non_reasoning

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            pool = (
                candidates if getattr(active_selector, "ranks_full_pool", False) else general_pool
            )
            candidate_objects = [self._cache.models[mid] for mid in pool]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected chat model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred chat model: {preferred}")
                    return preferred

        # Try trait-based selection (Venice's canonical default)
        trait_model = self._get_trait_model("default", resource_type="text")
        if trait_model and trait_model in general_pool:
            logger.info(f"Selected chat model via 'default' trait: {trait_model}")
            return trait_model

        # Fallback to the first model of the pool
        selected = general_pool[0]
        logger.info(f"Selected fallback chat model: {selected}")
        return selected

    def _pick_function_calling_model(
        self,
        candidates: list[str],
        *,
        preferred_models: list[str] | None,
        selector: ModelSelectorType | None,
    ) -> str:
        """Rank an already-filtered function-calling pool.

        Custom selector first, then ``preferred_models``, then Venice's
        ``function_calling_default`` trait, then catalog order.
        """
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected function calling model via custom selector: {selected}")
            return selected

        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred function calling model: {preferred}")
                    return preferred

        trait_model = self._get_trait_model("function_calling_default", resource_type="text")
        if trait_model and trait_model in candidates:
            logger.info(f"Selected function calling model via trait: {trait_model}")
            return trait_model

        selected = candidates[0]
        logger.info(f"Selected fallback function calling model: {selected}")
        return selected

    async def select_function_calling_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
        require_web_search: bool = False,
        require_reasoning_effort: str | None = None,
    ) -> str:
        """
        Select a chat model that supports function calling/tools.

        Equivalent to ``select_chat_model(require_function_calling=True, ...)``;
        use that method to combine function calling with other filters.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.
            require_web_search: If True, only select models that support web search
            require_reasoning_effort: Only select models whose
                ``reasoningEffortOptions`` include this value

        Returns:
            Selected model ID that supports function calling

        Raises:
            NoMatchingModelError: If no suitable function calling model found
        """
        return await self.select_chat_model(
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            require_function_calling=True,
            selector=selector,
            require_web_search=require_web_search,
            require_reasoning_effort=require_reasoning_effort,
        )

    async def select_code_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model optimized for code generation.

        Uses the 'default_code' trait as the primary selection, falling back
        to any model with optimizedForCode=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID optimized for code

        Raises:
            NoMatchingModelError: If no suitable code model found
        """
        return await self._select_trait_model(
            "default_code",
            "code",
            "require_code_optimization",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_vision_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model that supports vision (image input).

        Uses the 'default_vision' trait as the primary selection, falling back
        to any model with supportsVision=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID with vision support

        Raises:
            NoMatchingModelError: If no suitable vision model found
        """
        return await self._select_trait_model(
            "default_vision",
            "vision",
            "require_vision",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_reasoning_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a model that supports reasoning with thinking blocks.

        Uses the 'default_reasoning' trait as the primary selection, falling back
        to any model with supportsReasoning=True capability.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID with reasoning support

        Raises:
            NoMatchingModelError: If no suitable reasoning model found
        """
        return await self._select_trait_model(
            "default_reasoning",
            "reasoning",
            "require_reasoning",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_embedding_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable embedding model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable model found
        """
        return await self._select_simple_model(
            "embedding",
            "embedding",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_image_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
        require_web_search: bool = False,
        require_quality: str | None = None,
        require_custom_size: bool = False,
        exclude_uncensored: bool = False,
    ) -> str:
        """
        Select a suitable image generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.
            require_web_search: If True, only select models whose spec sets
                ``supportsWebSearch`` (i.e. that honor ``enable_web_search``).
            require_quality: Quality tier the model must list in its
                ``qualities`` constraint (e.g. ``"high"``; case-insensitive).
            require_custom_size: If True, only select models sized by explicit
                ``width`` / ``height``. Venice lists ``aspectRatios`` only for
                models that size by ``aspect_ratio`` (and ``resolution``
                tiers), and those models reject or ignore ``width`` /
                ``height``, so a model without ``aspectRatios`` takes pixel
                dimensions. The catalog does not say whether such a model
                returns exactly the requested size.
            exclude_uncensored: If True, skip models Venice flags as
                ``uncensored``.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable model found
        """
        available = await self.get_available_models(resource_type="image")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise NoMatchingModelError("No available image models found", resource_type="image")

        # Restrict to text-to-image *generation* models. The "image" resource
        # type also covers non-generative models such as background removers
        # (e.g. bria-bg-remover), which 400 on an image.create(prompt=...) call.
        # Filtering the candidate pool itself — before any selector runs —
        # constrains every downstream path equally (custom ``selector`` such as
        # random_cheap_strategy, ``preferred_models``, and trait selection), so a
        # cost-driven selector can never hand back a background remover for
        # generation. Falls back to the unfiltered pool if filtering would empty
        # it, so this never hard-fails on an unexpected catalog shape.
        generators = [
            m for m in candidates if _is_image_generation_model(self._cache.models.get(m, {}))
        ]
        if generators:
            candidates = generators

        if (
            require_web_search
            or require_quality is not None
            or require_custom_size
            or exclude_uncensored
        ):
            filtered = []
            for model_id in candidates:
                model_data = self._cache.models.get(model_id, {})
                model_spec = model_data.get("model_spec", {})
                constraints = model_spec.get("constraints") or {}
                if require_web_search and not model_spec.get("supportsWebSearch", False):
                    continue
                if require_quality is not None and not _offers(
                    constraints.get("qualities"), require_quality
                ):
                    continue
                if require_custom_size and constraints.get("aspectRatios"):
                    continue
                if exclude_uncensored and model_data.get("uncensored", False):
                    continue
                filtered.append(model_id)
            if not filtered:
                raise NoMatchingModelError(
                    f"No image models found matching requirements: "
                    f"web_search={require_web_search}, quality={require_quality!r}, "
                    f"custom_size={require_custom_size}, "
                    f"exclude_uncensored={exclude_uncensored}",
                    resource_type="image",
                )
            candidates = filtered

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            # Pass full model objects to the selector
            candidate_objects = [self._cache.models[mid] for mid in candidates]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected image model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in candidates:
                    logger.info(f"Selected preferred image model: {preferred}")
                    return preferred

        # Try trait-based selection (Venice's canonical default image model)
        trait_model = self._get_trait_model("default", resource_type="image")
        if trait_model and trait_model in candidates:
            logger.info(f"Selected image model via 'default' trait: {trait_model}")
            return trait_model

        # Fallback to first available model
        selected = candidates[0]
        logger.info(f"Selected fallback image model: {selected}")
        return selected

    async def select_video_model(
        self,
        model_type: str | None = None,  # "text-to-video" or "image-to-video"
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        exclude_beta: bool = False,
        selector: ModelSelectorType | None = None,
        require_duration: int | str | None = None,
        require_resolution: str | None = None,
        require_audio_configurable: bool = False,
        input_mode: VideoInputMode | None = None,
    ) -> str:
        """
        Select a suitable video generation model.

        Args:
            model_type: Filter by video model type ("text-to-video" or
                "image-to-video"). "image-to-video" keeps plain image-to-video
                models unless *input_mode* says otherwise.
            require_audio: If True, only select models that support audio generation
            min_resolution: Minimum resolution (e.g., "720p", "1080p", "4k").
                Models must support this resolution or higher.
            min_duration: Minimum duration (e.g., "5s", "10s").
                Models must support at least this duration.
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            exclude_beta: If True, exclude models marked as beta
            selector: Optional custom selection function
            require_duration: Exact duration the model must list in its
                ``durations`` constraint (``5``, ``"5"`` or ``"5s"``). Unlike
                ``min_duration``, a model offering only longer clips is rejected.
            require_resolution: Exact resolution tier the model must list in its
                ``resolutions`` constraint (e.g. ``"720p"``; case-insensitive).
            require_audio_configurable: If True, only select models whose
                ``audio_configurable`` constraint is set, i.e. that accept an
                explicit ``audio=True/False``.
            input_mode: What an image-to-video model must take as input (see
                :data:`VideoInputMode` and :func:`video_input_mode`), e.g.
                ``"reference"`` for reference-to-video. Implies
                ``model_type="image-to-video"``.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable video model found
        """
        available = await self.get_available_models(resource_type="video")
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if not candidates:
            raise NoMatchingModelError("No available video models found", resource_type="video")

        # Get full model data for constraint filtering
        models_data = await self._fetch_models()

        # Apply constraint-based filtering via shared helper
        filtered = _filter_video_candidates(
            candidates,
            models_data,
            model_type=model_type,
            require_audio=require_audio,
            min_resolution=min_resolution,
            min_duration=min_duration,
            require_duration=require_duration,
            require_resolution=require_resolution,
            require_audio_configurable=require_audio_configurable,
            exclude_beta=exclude_beta,
            input_mode=input_mode,
        )

        if not filtered:
            exact_duration = (
                _normalize_duration(require_duration) if require_duration is not None else None
            )
            raise NoMatchingModelError(
                f"No video models found matching criteria: "
                f"model_type={model_type}, require_audio={require_audio}, "
                f"min_resolution={min_resolution}, min_duration={min_duration}, "
                f"require_duration={exact_duration!r}, "
                f"require_resolution={require_resolution!r}, "
                f"require_audio_configurable={require_audio_configurable}, "
                f"exclude_beta={exclude_beta}, input_mode={input_mode!r}",
                resource_type="video",
            )

        # Apply custom selector if present
        active_selector = selector or self.default_selector
        if active_selector:
            candidate_objects = [self._cache.models[mid] for mid in filtered]
            selected = active_selector(candidate_objects)
            logger.info(f"Selected video model via custom selector: {selected}")
            return selected

        # Try preferred models first
        if preferred_models:
            for preferred in preferred_models:
                if preferred in filtered:
                    logger.info(f"Selected preferred video model: {preferred}")
                    return preferred

        # Fallback to first available model
        selected = filtered[0]
        logger.info(f"Selected video model: {selected}")
        return selected

    async def select_cheapest_video_model(
        self,
        *,
        duration: int | str | None = None,
        model_type: str | None = None,
        resolution: str | None = None,
        audio: bool | None = False,
        aspect_ratio: str | None = None,
        require_audio: bool = False,
        min_resolution: str | None = None,
        min_duration: str | None = None,
        require_audio_configurable: bool = False,
        exclude_models: set[str] | None = None,
        exclude_beta: bool = True,
        max_concurrency: int = 8,
        input_mode: VideoInputMode | None = None,
    ) -> CheapestVideoResult:
        """
        Select the cheapest video model by quoting every viable candidate.

        Candidates are filtered with the same constraint logic as
        :meth:`select_video_model`. Each one is then quoted with
        ``POST /video/quote`` at its *own* cheapest valid request, built by
        :func:`cheapest_video_params`: its shortest listed duration, its lowest
        listed resolution, a 16:9 aspect ratio when listed, and ``audio=False``
        when the model makes audio configurable. A model that does not offer a
        fixed 5-second clip is therefore still compared, at the duration it
        does offer. Passing ``duration`` or ``resolution`` pins that value
        instead, and models that do not list it are filtered out up front.

        .. note::

           Quotes are free (no generation occurs), but each call issues one
           request per candidate. At most ``max_concurrency`` run at once and
           this method adds no retries of its own (the client's retry policy
           still applies to each request).

        Args:
            duration: Pin the quoted duration (``5``, ``"5"`` or ``"5s"``);
                only models listing it are considered. ``None`` (default)
                quotes each model at its shortest duration.
            model_type: Filter by ``"text-to-video"`` or ``"image-to-video"``.
                ``None`` considers both, but never ``"video"`` (upscale and
                video-to-video) models, which need a source video to quote.
                ``"image-to-video"`` keeps plain image-to-video models unless
                *input_mode* says otherwise.
            resolution: Pin the quoted resolution (case-insensitive); only
                models listing it are considered. ``None`` (default) quotes
                each model at its lowest resolution.
            audio: Audio setting sent to models whose audio is configurable.
                ``False`` (default) quotes the silent, cheaper variant;
                ``True`` also requires audio support; ``None`` omits the
                parameter so each model uses its own default.
            aspect_ratio: Aspect ratio to quote; models listing other ratios
                only are filtered out. ``None`` prefers 16:9.
            require_audio: Only consider models that support audio.
            min_resolution: Only consider models offering at least this resolution.
            min_duration: Only consider models offering at least this duration.
            require_audio_configurable: Only consider models that accept an
                explicit ``audio=True/False``.
            exclude_models: Model IDs to exclude from consideration.
            exclude_beta: If ``True``, exclude beta models.
            max_concurrency: Maximum quote requests in flight at once.
            input_mode: What an image-to-video model must take as input (see
                :data:`VideoInputMode`). Implies ``model_type="image-to-video"``.

        Returns:
            A :class:`CheapestVideoResult` with the cheapest model, its quote,
            the request parameters it was quoted with, every successful quote
            and the reason each remaining candidate was skipped.

        Raises:
            NoMatchingModelError: If no candidate passes the filters or can
                serve the requested settings.
            ModelQuotesUnavailableError: If candidates exist but every quote
                failed; ``failures`` maps each model to its exception.
            ValueError: If ``max_concurrency`` is below 1.

        Example:
            >>> selector = create_model_selector(client)
            >>> result = await selector.select_cheapest_video_model(
            ...     model_type="text-to-video",
            ... )
            >>> print(f"Cheapest: {result.model} at ${result.quote_usd:.4f}")
            >>> job = await client.video.submit(
            ...     model=result.model, prompt="...", **result.request_params
            ... )
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")

        # --- Step 1: Get constraint-filtered candidates -----------------
        available = await self.get_available_models(resource_type="video")
        exclude_models = exclude_models or set()

        candidates = [m for m in available if m not in exclude_models]
        if not candidates:
            raise NoMatchingModelError("No available video models found", resource_type="video")

        models_data = await self._fetch_models()

        filtered = _filter_video_candidates(
            candidates,
            models_data,
            model_type=model_type,
            require_audio=require_audio or audio is True,
            min_resolution=min_resolution,
            min_duration=min_duration,
            require_duration=duration,
            require_resolution=resolution,
            require_audio_configurable=require_audio_configurable,
            exclude_beta=exclude_beta,
            input_mode=input_mode,
        )
        if model_type is None:
            # Upscale and video-to-video models price from a source video.
            filtered = [
                m
                for m in filtered
                if _video_constraints(models_data.get(m, {})).get("model_type") != "video"
            ]
        if aspect_ratio is not None:
            filtered = [
                m
                for m in filtered
                if not _video_constraints(models_data.get(m, {})).get("aspect_ratios")
                or _offers(
                    _video_constraints(models_data.get(m, {})).get("aspect_ratios"), aspect_ratio
                )
            ]

        if not filtered:
            raise NoMatchingModelError(
                f"No video models found matching criteria: "
                f"model_type={model_type}, require_audio={require_audio}, audio={audio}, "
                f"min_resolution={min_resolution}, min_duration={min_duration}, "
                f"duration={duration!r}, resolution={resolution!r}, "
                f"aspect_ratio={aspect_ratio!r}, "
                f"require_audio_configurable={require_audio_configurable}, "
                f"exclude_beta={exclude_beta}",
                resource_type="video",
            )

        # --- Step 2: Build each model's cheapest valid request ----------
        skipped: dict[str, str] = {}
        plans: list[tuple[str, dict[str, Any]]] = []
        for mid in filtered:
            try:
                params = cheapest_video_params(
                    _video_constraints(models_data.get(mid, {})),
                    duration=duration,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    audio=audio,
                )
            except ValueError as exc:
                skipped[mid] = str(exc)
                continue
            plans.append((mid, params))

        if not plans:
            raise NoMatchingModelError(
                f"None of the {len(filtered)} candidate video models can serve "
                f"duration={duration!r}, resolution={resolution!r}, "
                f"aspect_ratio={aspect_ratio!r}, audio={audio}. Skipped: {skipped}",
                resource_type="video",
                skipped=skipped,
            )
        unquotable = dict(skipped)
        failures: dict[str, Exception] = {}

        # --- Step 3: Quote, at most max_concurrency at a time -----------
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _quote_model(
            mid: str, params: dict[str, Any]
        ) -> tuple[str, float, dict[str, Any]] | None:
            async with semaphore:
                try:
                    quote_resp = await self.client.video.quote(model=mid, **params)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug(f"Quote failed for {mid}: {exc}")
                    failures[mid] = exc
                    skipped[mid] = f"quote failed: {exc}"
                    return None
            cost = float(quote_resp.quote)
            logger.debug(f"Video quote for {mid} with {params}: ${cost:.4f}")
            return (mid, cost, params)

        results = await asyncio.gather(*[_quote_model(mid, params) for mid, params in plans])
        valid = [r for r in results if r is not None]

        if not valid:
            raise _quotes_unavailable("video", failures, unquotable)

        # --- Step 4: Pick the cheapest -----------------------------------
        # Ties follow cheapest_model_strategy: the ``default`` trait, then
        # catalog order, then model id.
        order = {mid: index for index, mid in enumerate(filtered)}

        def _tie_key(item: tuple[str, float, dict[str, Any]]) -> tuple[float, bool, int, str]:
            traits = models_data.get(item[0], {}).get("traits") or []
            return (item[1], "default" not in traits, order.get(item[0], len(order)), item[0])

        valid.sort(key=_tie_key)
        cheapest_model, cheapest_price, cheapest_params = valid[0]

        logger.info(
            f"Selected cheapest video model: {cheapest_model} "
            f"(${cheapest_price:.4f}) out of {len(valid)} quoted models"
        )

        return CheapestVideoResult(
            model=cheapest_model,
            quote_usd=cheapest_price,
            all_quotes={mid: price for mid, price, _ in valid},
            request_params=cheapest_params,
            skipped=skipped,
        )

    async def select_audio_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable audio/speech generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable model found
        """
        return await self._select_simple_model(
            "tts",
            "text-to-speech (tts)",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_inpaint_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
        require_combine_images: bool = False,
        require_quality: str | None = None,
        require_resolution: str | None = None,
        require_uncensored: bool = False,
    ) -> str:
        """
        Select a suitable inpaint (image editing) model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function
            require_combine_images: If True, only select models whose
                ``combineImages`` constraint is set (needed for ``multi_edit``).
            require_quality: Quality tier the model must list in its
                ``qualities`` constraint (e.g. ``"low"``; case-insensitive).
            require_resolution: Resolution tier the model must list in its
                ``resolutions`` constraint (e.g. ``"2K"``; case-insensitive).
                Models without the constraint reject a ``resolution`` argument.
            require_uncensored: If True, only select models Venice flags as
                ``uncensored``.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable inpaint model found
        """
        has_filter = (
            require_combine_images
            or require_quality is not None
            or require_resolution is not None
            or require_uncensored
        )

        def _matches(model_data: dict[str, Any]) -> bool:
            constraints = model_data.get("model_spec", {}).get("constraints", {})
            if require_combine_images and not constraints.get("combineImages", False):
                return False
            if require_quality is not None and not _offers(
                constraints.get("qualities"), require_quality
            ):
                return False
            if require_resolution is not None and not _offers(
                constraints.get("resolutions"), require_resolution
            ):
                return False
            return not (require_uncensored and not model_data.get("uncensored", False))

        return await self._select_simple_model(
            "inpaint",
            "inpaint",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
            candidate_filter=_matches if has_filter else None,
            filter_description=(
                f"combine_images={require_combine_images}, quality={require_quality!r}, "
                f"resolution={require_resolution!r}, uncensored={require_uncensored}"
            ),
        )

    async def select_asr_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable ASR (Automatic Speech Recognition) model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable ASR model found
        """
        return await self._select_simple_model(
            "asr",
            "speech-to-text (asr)",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_music_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
        exclude_non_music: bool = False,
        require_force_instrumental: bool = False,
        duration_seconds: int | None = None,
    ) -> str:
        """
        Select a suitable music generation model.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.
            exclude_non_music: If True, skip the text-to-speech, sound-effect and
                voice-changer models that Venice also types as ``music``.
            require_force_instrumental: If True, only select models that accept
                ``force_instrumental`` (``supports_force_instrumental``).
            duration_seconds: If set, skip models whose declared duration
                metadata cannot make a clip this long (see
                :func:`music_request_params`). Models that declare no duration
                metadata choose their own length and are kept.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable music model found
        """

        def _matches(model_data: dict[str, Any]) -> bool:
            spec = model_data.get("model_spec", {})
            if exclude_non_music and not _is_music_generation_model(model_data):
                return False
            if require_force_instrumental and not spec.get("supports_force_instrumental"):
                return False
            if duration_seconds is not None:
                try:
                    music_request_params(spec, duration_seconds)
                except ValueError:
                    return False
            return True

        has_filter = exclude_non_music or require_force_instrumental or duration_seconds is not None

        return await self._select_simple_model(
            "music",
            "music",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
            candidate_filter=_matches if has_filter else None,
            filter_description=(
                f"exclude_non_music={exclude_non_music}, "
                f"force_instrumental={require_force_instrumental}, "
                f"duration_seconds={duration_seconds}"
            ),
        )

    async def select_cheapest_music_model(
        self,
        *,
        duration_seconds: int | None = None,
        require_force_instrumental: bool = False,
        lyrics_supplied: bool = False,
        exclude_models: set[str] | None = None,
        exclude_beta: bool = True,
        max_concurrency: int = 8,
    ) -> CheapestMusicResult:
        """
        Select the cheapest music generator for a clip length by quoting each one.

        Only music generators are considered (the text-to-speech, sound-effect
        and voice-changer models Venice also types as ``music`` are skipped).
        Each candidate is quoted with the free ``POST /audio/quote`` at the
        request :func:`music_request_params` builds for *duration_seconds*:

        - a model with fixed ``duration_options`` at the smallest option that
          covers the target (``ace-step`` makes 60-second clips at minimum);
        - a model with a duration range at the target, raised to its minimum;
        - a model that takes no duration at its own length, reported as
          ``effective_seconds=None``.

        A model that cannot make a clip as long as the target is skipped rather
        than quoted at a shorter one. The quote is authoritative: Venice rounds
        every music quote and bill up to the cent, which catalog per-second
        prices alone understate for short clips.

        Ranking is by quote. Among equal quotes the clip closest to the target
        length wins, then the ``default`` trait, then catalog order, then model
        id.

        Args:
            duration_seconds: The clip length wanted. ``None`` quotes each model
                at its shortest valid request.
            require_force_instrumental: Only consider models that accept
                ``force_instrumental``.
            lyrics_supplied: Consider models whose catalog entry sets
                ``lyrics_required``. Leave ``False`` unless the submit call will
                pass ``lyrics_prompt``, or the cheapest pick could reject it.
            exclude_models: Model IDs to exclude.
            exclude_beta: Exclude beta models.
            max_concurrency: Maximum quote requests in flight at once.

        Returns:
            A :class:`CheapestMusicResult`.

        Raises:
            NoMatchingModelError: If no music generator passes the filters or
                can make a clip of ``duration_seconds``.
            ModelQuotesUnavailableError: If candidates exist but every quote
                failed; ``failures`` maps each model to its exception.
            ValueError: If ``max_concurrency`` is below 1.
        """
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        available = await self.get_available_models(resource_type="music")
        excluded = exclude_models or set()
        models_data = await self._fetch_models()

        skipped: dict[str, str] = {}
        plans: list[tuple[str, dict[str, Any]]] = []
        for mid in available:
            if mid in excluded:
                continue
            model_data = models_data.get(mid, {})
            spec = model_data.get("model_spec", {})
            if not _is_music_generation_model(model_data):
                continue
            if require_force_instrumental and not spec.get("supports_force_instrumental"):
                continue
            if exclude_beta and model_data.get("beta", False):
                continue
            if spec.get("lyrics_required") and not lyrics_supplied:
                skipped[mid] = "requires lyrics (pass lyrics_supplied=True to include it)"
                continue
            try:
                params = music_request_params(spec, duration_seconds)
            except ValueError as exc:
                skipped[mid] = str(exc)
                continue
            plans.append((mid, params))

        if not plans:
            raise NoMatchingModelError(
                f"No music models found matching criteria: duration_seconds="
                f"{duration_seconds}, force_instrumental={require_force_instrumental}, "
                f"lyrics_supplied={lyrics_supplied}, exclude_beta={exclude_beta}. "
                f"Skipped: {skipped}",
                resource_type="music",
                skipped=skipped,
            )

        unquotable = dict(skipped)
        failures: dict[str, Exception] = {}
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _quote(mid: str, params: dict[str, Any]) -> tuple[str, float] | None:
            async with semaphore:
                try:
                    response = await self.client.music.quote(model=mid, **params)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug(f"Music quote failed for {mid}: {exc}")
                    failures[mid] = exc
                    skipped[mid] = f"quote failed: {exc}"
                    return None
            return (mid, float(response.quote))

        results = await asyncio.gather(*[_quote(mid, params) for mid, params in plans])
        valid = [r for r in results if r is not None]
        if not valid:
            raise _quotes_unavailable("music", failures, unquotable)

        params_by_model = dict(plans)
        order = {mid: index for index, (mid, _) in enumerate(plans)}

        def _rank(item: tuple[str, float]) -> tuple[float, float, bool, int, str]:
            mid, price = item
            seconds = params_by_model[mid].get("duration_seconds")
            if duration_seconds is None:
                distance = 0.0
            elif seconds is None:
                distance = math.inf
            else:
                distance = float(abs(seconds - duration_seconds))
            traits = models_data.get(mid, {}).get("traits") or []
            return (price, distance, "default" not in traits, order[mid], mid)

        best_model, best_price = min(valid, key=_rank)
        best_params = params_by_model[best_model]
        logger.info(
            f"Selected cheapest music model: {best_model} (${best_price:.2f}, "
            f"{best_params or 'model-chosen length'}) out of {len(valid)} quoted models"
        )
        return CheapestMusicResult(
            model=best_model,
            quote_usd=best_price,
            request_params=dict(best_params),
            effective_seconds=best_params.get("duration_seconds"),
            all_quotes=dict(valid),
            skipped=skipped,
        )

    async def select_decision_model(
        self,
        preferred_models: list[str] | None = None,
        exclude_models: set[str] | None = None,
        selector: ModelSelectorType | None = None,
    ) -> str:
        """
        Select a suitable decision ("System One") model.

        Note that decision models are currently beta-flagged; unlike the chat
        and video selectors this one does not offer an ``exclude_beta`` filter,
        because applying it would leave no candidates at all.

        Args:
            preferred_models: List of preferred models in priority order
            exclude_models: Set of models to exclude from selection
            selector: Optional custom selection function. Takes precedence over
                default_selector. Receives list of model dicts, returns model ID.

        Returns:
            Selected model ID

        Raises:
            NoMatchingModelError: If no suitable decision model found
        """
        return await self._select_simple_model(
            "decision",
            "decision",
            preferred_models=preferred_models,
            exclude_models=exclude_models,
            selector=selector,
        )

    async def select_models_for_concurrency_test(
        self,
        count: int = 2,
        resource_type: str = "text",
        exclude_models: set[str] | None = None,
    ) -> list[str]:
        """
        Select multiple models for concurrency testing or production use.

        Args:
            count: Number of models to select
            resource_type: Type of models to select (default "text"). Use "text"
                for chat models, "image" for image models, etc.
            exclude_models: Set of models to exclude from selection

        Returns:
            List of selected model IDs

        Raises:
            NoMatchingModelError: If not enough models available
        """
        available = await self.get_available_models(resource_type=resource_type)
        exclude_models = exclude_models or set()

        # Filter out excluded models
        candidates = [m for m in available if m not in exclude_models]

        if len(candidates) < count:
            raise NoMatchingModelError(
                f"Need {count} models but only {len(candidates)} available",
                resource_type=resource_type,
            )

        # Try to get diverse models for better testing
        selected: list[str] = []

        # Try to get diverse models using traits first. Reasoning models
        # consume their token budget on thinking and often return empty
        # message.content under standard token limits, which is undesirable for
        # concurrency/comparison tests — so skip reasoning trait models here
        # (e.g. the 'default'/'most_intelligent' traits, which now resolve to
        # reasoning models). They are still eligible via the reasoning fallback
        # below if too few non-reasoning models exist.
        diversity_traits = ["default", "fastest", "most_intelligent"]
        for trait in diversity_traits:
            if len(selected) >= count:
                break
            trait_model = self._get_trait_model(trait, resource_type=resource_type)
            if (
                trait_model
                and trait_model in candidates
                and trait_model not in selected
                and not self._is_reasoning_model(trait_model)
            ):
                selected.append(trait_model)

        # Fill remaining slots, preferring non-reasoning models.
        remaining = [m for m in candidates if m not in selected]
        non_reasoning = [m for m in remaining if not self._is_reasoning_model(m)]
        reasoning = [m for m in remaining if m not in non_reasoning]
        ordered_remaining = non_reasoning + reasoning

        for model in ordered_remaining:
            if len(selected) >= count:
                break
            selected.append(model)

        logger.info(f"Selected {len(selected)} models for concurrency test: {selected}")
        return selected[:count]

    async def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific model.

        Args:
            model_id: ID of the model to get info for

        Returns:
            Model information dict or None if not found
        """
        await self._fetch_models()
        return self._cache.models.get(model_id)

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about the current cache state."""
        return {
            "model_count": len(self._cache.models),
            "last_updated": self._cache.last_updated,
            "is_expired": self._cache.is_expired(),
            "ttl_seconds": self._cache.ttl_seconds,
        }


def create_model_selector(
    client: Any,
    cache_ttl: float = 300.0,
    default_selector: ModelSelectorType | None = None,
) -> DynamicModelSelector:
    """
    Factory function to create a model selector instance.

    .. deprecated::
        Use ``client.models.resolve()`` instead. This function will be removed
        in a future release.

    Args:
        client: Venice AI client instance
        cache_ttl: Cache TTL in seconds
        default_selector: Optional custom selection function that receives
            a list of model dicts and returns the selected model ID.

    Returns:
        DynamicModelSelector instance
    """
    import warnings

    warnings.warn(
        "create_model_selector() is deprecated, use client.models.resolve()",
        DeprecationWarning,
        stacklevel=2,
    )
    return DynamicModelSelector(client, cache_ttl=cache_ttl, default_selector=default_selector)


# Utility functions for common use cases
async def get_chat_model(client: Any, preferred: list[str] | None = None) -> str:
    """Quick helper to get a chat model for production or testing."""
    selector = DynamicModelSelector(client)
    return await selector.select_chat_model(preferred_models=preferred)


async def get_embedding_model(client: Any, preferred: list[str] | None = None) -> str:
    """Quick helper to get an embedding model for production or testing."""
    selector = DynamicModelSelector(client)
    return await selector.select_embedding_model(preferred_models=preferred)


async def get_multiple_models(client: Any, count: int = 2) -> list[str]:
    """Quick helper to get multiple models for concurrency testing or production use."""
    selector = DynamicModelSelector(client)
    return await selector.select_models_for_concurrency_test(count=count)


async def get_video_model(
    client: Any,
    model_type: str | None = None,
    preferred: list[str] | None = None,
) -> str:
    """Quick helper to get a video model."""
    selector = DynamicModelSelector(client)
    return await selector.select_video_model(model_type=model_type, preferred_models=preferred)


async def get_cheapest_video_model(
    client: Any,
    model_type: str | None = None,
    duration: int | str | None = None,
    **kwargs: Any,
) -> CheapestVideoResult:
    """Quick helper to find the cheapest video model for given parameters.

    Queries the ``POST /video/quote`` endpoint for every eligible model and
    returns a :class:`CheapestVideoResult` with the model that has the lowest
    cost.

    Args:
        client: Venice AI client instance.
        model_type: ``"text-to-video"`` or ``"image-to-video"``.
        duration: Pin the quoted duration. ``None`` (default) quotes each
            model at its own shortest duration.
        **kwargs: Forwarded to
            :meth:`DynamicModelSelector.select_cheapest_video_model`.

    Returns:
        A :class:`CheapestVideoResult` with the cheapest model, its USD
        price, and all successful quotes.

    Example:
        >>> result = await get_cheapest_video_model(client, model_type="text-to-video")
        >>> print(f"Use model {result.model} (${result.quote_usd:.4f})")
    """
    selector = DynamicModelSelector(client)
    return await selector.select_cheapest_video_model(
        model_type=model_type, duration=duration, **kwargs
    )
