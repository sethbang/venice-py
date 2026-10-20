"""
Response Validation and Metrics Collection
==========================================

This module provides comprehensive response validation utilities for the Venice AI SDK,
ensuring API responses conform to expected Pydantic model schemas. It includes robust
error handling, detailed metrics collection, and validation configuration management.

The validation system serves multiple purposes:
    * **Data Integrity**: Ensures API responses match expected schemas
    * **Error Detection**: Catches API format changes and data corruption early
    * **Type Safety**: Provides strongly-typed response objects for development
    * **Debugging Support**: Collects metrics for monitoring validation health
    * **Development Aid**: Clear error messages for schema mismatches

Key Features:
    * **Pydantic Integration**: Full integration with Pydantic v2 validation
    * **Metrics Collection**: Automatic tracking of validation success/failure rates
    * **Error Context**: Rich error messages with endpoint and method context
    * **Per-Model Tracking**: Granular metrics by response model type
    * **Production Ready**: Robust error handling for production environments
    * **Thread-Safe**: All metrics operations are thread-safe using Lock protection

Validation Workflow:
    1. **Response Parsing**: Raw API response data is received
    2. **Schema Validation**: Data is validated against Pydantic model
    3. **Success Handling**: Valid data is returned as typed model instance
    4. **Error Handling**: Validation failures are wrapped with context
    5. **Metrics Update**: Success/failure statistics are recorded (thread-safe)

Thread Safety:
    This module uses a module-level ``threading.Lock`` (``_validation_lock``) for
    thread-safe metric recording. All validation metric operations are protected by
    this lock, ensuring safe concurrent access from multiple threads.

Example:
    >>> from venice_ai._validation import validate_response_model
    >>> from venice_ai.types.chat import ChatCompletionResponse
    >>>
    >>> # Validate API response data
    >>> response_data = await api_call()
    >>> try:
    ...     completion = validate_response_model(
    ...         response_data,
    ...         ChatCompletionResponse,
    ...         endpoint="/chat/completions",
    ...         method="POST"
    ...     )
    ...     print(f"Valid response: {completion.choices[0].message.content}")
    ... except APIResponseValidationError as e:
    ...     print(f"Validation failed: {e}")
    ...     print(f"Model: {e.model_name}")
    ...     print(f"Errors: {e.validation_error.errors()}")
"""

import logging
import threading
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .core.metrics import get_metrics_collector
from .exceptions import APIResponseValidationError

logger = logging.getLogger(__name__)

# Module-level validation counters for fast, lock-protected in-process queries
# via get_validation_metrics(). These are intentionally maintained alongside the
# Prometheus counters (written through metrics.record_validation_success/failure)
# which serve external scraping. The two write paths are decoupled so that
# in-process callers never need to round-trip through the Prometheus client.
_validation_lock = threading.Lock()
_validation_counters: dict[str, int] = {"successful": 0, "failed": 0}
_validation_errors_by_model: dict[str, int] = {}

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def validate_response_model(
    response_data: Any,
    model_class: type[_ModelT],
    *,
    endpoint: str | None = None,
    method: str | None = None,
) -> _ModelT:
    """
    Validate API response data against a Pydantic model with comprehensive error handling.

    This function serves as the primary validation entry point for all API responses
    in the Venice AI SDK. It validates raw response data against expected Pydantic
    schemas, providing detailed error context and automatic metrics collection.

    The validation process includes:
    * Schema validation using Pydantic v2 model validation
    * Automatic success/failure metrics collection (thread-safe)
    * Rich error context with endpoint and method information
    * Per-model error tracking for debugging and monitoring

    Thread Safety:
        All metric recording operations are thread-safe, using the module-level
        ``_validation_lock``. Safe for concurrent use.

    Args:
        response_data: Raw response data from the Venice AI API. This can be
                      any data structure (dict, list, primitive) that should
                      be validated against the target model schema.
        model_class: The Pydantic model class to validate against. Must be a
                    subclass of BaseModel and should match the expected response
                    structure for the specific API endpoint.
        endpoint: Optional API endpoint path for error context and debugging.
                 Helps identify which API call caused validation failures.
        method: Optional HTTP method (GET, POST, etc.) for error context.
               Provides additional context for debugging validation issues.

    Returns:
        Validated and typed Pydantic model instance. The returned object
        provides full type safety and access to all response fields with
        proper typing and validation.

    Raises:
        APIResponseValidationError: Raised when response data fails Pydantic
                                  validation. The exception includes the original
                                  ValidationError, response data, model name, and
                                  endpoint context for debugging.

    Metrics:
        Automatically updates global validation metrics including:
        * Total successful validations
        * Total failed validations
        * Per-model error counts for monitoring

    Example:
        >>> from venice_ai._validation import validate_response_model
        >>> from venice_ai.types.chat import ChatCompletionResponse
        >>>
        >>> # Validate a chat completion response
        >>> raw_response = {
        ...     "id": "chatcmpl-123",
        ...     "object": "chat.completion",
        ...     "choices": [{"message": {"role": "assistant", "content": "Hello!"}}]
        ... }
        >>>
        >>> try:
        ...     completion = validate_response_model(
        ...         raw_response,
        ...         ChatCompletionResponse,
        ...         endpoint="/chat/completions",
        ...         method="POST"
        ...     )
        ...     print(f"Valid response: {completion.choices[0].message.content}")
        ... except APIResponseValidationError as e:
        ...     logger.error(f"Validation failed for {e.model_name}: {e}")
    """
    model_name = model_class.__name__

    try:
        validated_model = model_class.model_validate(response_data)

        # Record success metric — Prometheus + local counters
        metrics = get_metrics_collector()
        metrics.record_validation_success(model_name)
        with _validation_lock:
            _validation_counters["successful"] += 1

        logger.debug(
            f"Successfully validated response for {model_name}",
            extra={
                "model_name": model_name,
                "endpoint": endpoint,
                "method": method,
            },
        )

        return validated_model

    except ValidationError as e:
        # Record failure metric — Prometheus + local counters
        metrics = get_metrics_collector()
        metrics.record_validation_failure(model_name)
        with _validation_lock:
            _validation_counters["failed"] += 1
            _validation_errors_by_model[model_name] = (
                _validation_errors_by_model.get(model_name, 0) + 1
            )

        logger.error(
            f"Pydantic validation failed for {model_name}: {e}",
            extra={
                "model_name": model_name,
                "response_data": response_data,
                "validation_errors": e.errors(),
                "endpoint": endpoint,
                "method": method,
            },
        )

        raise APIResponseValidationError(
            f"API response validation failed for {model_name}",
            validation_error=e,
            response_data=response_data,
            model_name=model_name,
        ) from e


def get_validation_metrics() -> dict[str, Any]:
    """
    Get current validation metrics from the in-process local counters.

    Note:
        This reads from the module-level ``_validation_counters`` dict, **not**
        from the Prometheus metrics collector. Both are updated on every
        validation call, but this function is designed for lightweight in-process
        queries without depending on the Prometheus client library.

    Thread-safe: Uses module-level lock for counter access.

    Returns:
        Dictionary with structure:
        {
            "successful_validations": int,
            "failed_validations": int,
            "validation_errors_by_model": Dict[str, int]
        }
    """
    with _validation_lock:
        return {
            "successful_validations": _validation_counters["successful"],
            "failed_validations": _validation_counters["failed"],
            "validation_errors_by_model": dict(_validation_errors_by_model),
        }


def reset_validation_metrics() -> None:
    """
    Reset validation metrics to zero.

    Thread-safe: Uses module-level lock for counter access.
    """
    with _validation_lock:
        _validation_counters["successful"] = 0
        _validation_counters["failed"] = 0
        _validation_errors_by_model.clear()


__all__ = [
    "validate_response_model",
    "get_validation_metrics",
    "reset_validation_metrics",
]
