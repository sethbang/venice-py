"""The one form the SDK takes a Venice API base URL in.

Every entry point (``VeniceClient(base_url=...)``, ``VeniceAIConfig.api_base_url``,
the ``VENICE_API_BASE_URL`` environment variable, ``VeniceClientFactory`` and
the CLI's ``api.base_url``) takes the **API root**: the URL that endpoint paths
such as ``chat/completions`` are appended to, version path included::

    https://api.venice.ai/api/v1

A bare host (``https://api.venice.ai``, no path) is accepted as shorthand and
gets ``/api/{api_version}`` appended, since Venice serves its API under that
path. Any other path is used as given, so a proxy or gateway can expose the API
at its own path. The two spellings of the official URL therefore resolve to
the same root wherever they are passed.

The one shape this cannot express is an API served at a host's root (endpoints
at ``https://gateway.example.com/chat/completions``). A URL with no path --
with or without a trailing slash -- is always read as the bare-host shorthand,
and ``api_version`` changes only the version segment of that shorthand, never
whether it is applied. Serve such a gateway under a path
(``https://gateway.example.com/venice``) and pass the URL with that path.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

DEFAULT_API_VERSION = "v1"


def normalize_api_version(api_version: str) -> str:
    """Return *api_version* as a bare path segment (``"v1"``, not ``"/v1/"``).

    Raises:
        ValueError: If *api_version* is empty or only whitespace and slashes,
            which would make a bare host resolve to ``/api/``.
    """
    version = str(api_version).strip().strip("/").strip()
    if not version:
        raise ValueError(
            f"api_version must name an API version such as {DEFAULT_API_VERSION!r}, "
            f"got {api_version!r}"
        )
    return version


def normalize_base_url(url: str, api_version: str = DEFAULT_API_VERSION) -> str:
    """Return the API root for *url*, without a trailing slash.

    Args:
        url: An API root (``https://api.venice.ai/api/v1``) or a bare host
            (``https://api.venice.ai``). A URL without a path always takes the
            bare-host shorthand, so an API mounted at a host's root cannot be
            addressed.
        api_version: The version path appended to a bare host.

    Raises:
        ValueError: If *url* is not an absolute ``http`` or ``https`` URL, or
            *api_version* is empty or blank.
    """
    version = normalize_api_version(api_version)
    text = str(url).strip()
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"API base URL must be an absolute http:// or https:// URL, got {text!r}")
    path = parts.path.rstrip("/")
    if not path:
        path = f"/api/{version}"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


__all__ = ["DEFAULT_API_VERSION", "normalize_api_version", "normalize_base_url"]
