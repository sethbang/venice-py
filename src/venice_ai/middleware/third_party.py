"""Keep what a client sends to the Venice API off every other host.

The SDK sends requests to other hosts: video download links, the Solana RPC, and
wherever a redirect leads. aiohttp merges a session's default headers into all of
them and, on a cross-origin redirect, removes only ``Authorization``, ``Cookie``
and ``Proxy-Authorization``. A custom secret header would survive.

aiohttp runs a request's middlewares once per redirect hop, so
:func:`create_third_party_header_filter` sees every host a request reaches.
"""

import contextlib
import contextvars
from collections.abc import Iterator
from typing import Any

from aiohttp import BasicAuth, ClientMiddlewareType
from yarl import URL

# Everything else is removed, whatever it is called, since the name of a secret
# header is the caller's choice. ``Cookie`` is removed with the rest: a cookie jar
# sends cookies stored without a domain to every host.
_SENT_TO_ANY_HOST = frozenset(
    {
        "accept",
        "accept-encoding",
        "content-encoding",
        "content-length",
        "content-type",
        "expect",
        "host",
        "transfer-encoding",
        "user-agent",
    }
)


# The Authorization value built from credentials in a URL the SDK was given, such as
# https://user:password@rpc.example.com. aiohttp removes them from the URL before a
# middleware runs, so the header alone cannot show where it came from.
_url_authorization: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "venice_url_authorization", default=None
)


@contextlib.contextmanager
def keeping_url_credentials(url: str) -> Iterator[None]:
    """Let credentials written into *url* through the filter for requests sent inside."""
    try:
        auth = BasicAuth.from_url(URL(url))
    except (TypeError, ValueError):
        auth = None
    token = _url_authorization.set(auth.encode() if auth is not None else None)
    try:
        yield
    finally:
        _url_authorization.reset(token)


def same_origin(first: URL, second: URL) -> bool:
    """Whether two URLs share a scheme, a host and an effective port.

    ``URL.origin()`` keeps a port written out in full, so ``https://h:443/`` and
    ``https://h/`` have different origins although they are the same place.
    """
    return (first.scheme, first.host, first.port) == (second.scheme, second.host, second.port)


def create_third_party_header_filter(api_origin: URL) -> ClientMiddlewareType:
    """Create the middleware that removes API-bound headers from requests to other hosts.

    A request to *api_origin* passes unchanged. A request to any other origin keeps
    only ``Accept``, ``User-Agent`` and the headers that describe its transfer. Inside
    :func:`keeping_url_credentials` it also keeps an ``Authorization`` built from that
    URL's own credentials, which were written for that host.

    Place it last in a middleware list, so it also covers what earlier middlewares
    add, on every retry and every redirect hop.

    Args:
        api_origin: Origin of the API.

    Returns:
        An aiohttp client middleware.
    """

    async def third_party_header_filter(request: Any, handler: Any) -> Any:
        if not same_origin(request.url, api_origin):
            url_authorization = _url_authorization.get()
            for name in {name for name in request.headers if name.lower() not in _SENT_TO_ANY_HOST}:
                if name.lower() == "authorization" and request.headers[name] == url_authorization:
                    continue
                del request.headers[name]
        return await handler(request)

    return third_party_header_filter


__all__ = ["create_third_party_header_filter", "keeping_url_credentials", "same_origin"]
