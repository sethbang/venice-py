"""
Middleware components for the Venice AI client.

This module provides middleware functionality that integrates into the Venice AI client's
HTTP request pipeline. Middleware components act as interceptors that can modify, enhance,
or handle requests and responses during the API communication lifecycle.

## Available Middleware

### Retry Middleware
The retry middleware resends a failed request only when the first attempt was not
processed or when processing it twice is free. Paid generation requests are never
resent once the server may have received them; see ``venice_ai.middleware.retry``.

## Integration with Venice AI Client

Middleware components are automatically integrated into the client's HTTP session during
initialization. They operate transparently, intercepting requests before they reach the
API endpoints and responses before they're returned to the calling code.

Key benefits:
- **Resilience**: Automatic retry logic for transient failures
- **Performance**: Intelligent backoff strategies to avoid overwhelming servers
- **Transparency**: Middleware operates behind the scenes without changing API interfaces
- **Configurability**: Extensive customization options for different use cases

## Usage

Middleware is typically configured through the client initialization process and doesn't
require direct interaction from end users. However, advanced users can customize retry
behavior through the RetryOptions configuration.
"""

from .retry import (
    RetryClass,
    RetryOptions,
    calculate_backoff_delay,
    classify_request,
    create_retry_middleware,
)

__all__ = [
    "RetryClass",
    "RetryOptions",
    "classify_request",
    "create_retry_middleware",
    "calculate_backoff_delay",
]
