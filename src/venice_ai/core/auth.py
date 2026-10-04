"""Authentication utilities for Venice AI API."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import VeniceAIConfig

API_KEY_ENV_VAR = "VENICE_API_KEY"


def resolve_api_key(api_key: str | None, config: VeniceAIConfig | None = None) -> str:
    """Resolve the API key every client entry point authenticates with.

    The first source that is not ``None`` wins, in this order:

    1. ``api_key`` -- the key passed to the client or factory.
    2. ``config.api_key`` -- the key on a :class:`VeniceAIConfig`.
    3. The ``VENICE_API_KEY`` environment variable.

    An explicit empty string stops the lookup and means "no API key", so a
    caller can authenticate with a wallet even when ``VENICE_API_KEY`` is set.

    Args:
        api_key: The explicitly passed key, or ``None``.
        config: The client configuration, if any.

    Returns:
        The key with surrounding whitespace removed; ``""`` when no source
        provides one.
    """
    if api_key is None and config is not None:
        api_key = config.api_key
    if api_key is None:
        api_key = os.environ.get(API_KEY_ENV_VAR)
    return (api_key or "").strip()


def create_auth_headers(api_key: str) -> dict[str, str]:
    """Create authentication headers from an API key."""
    return {"Authorization": f"Bearer {api_key.strip()}"}


def validate_api_key_format(api_key: str) -> bool:
    """Basic validation of API key format.

    Venice API keys are non-empty strings. This function performs
    basic format validation including a minimum length check.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    api_key = api_key.strip()
    return len(api_key) > 10  # Basic length check
