"""
Venice AI SDK Exception Hierarchy
=================================

Exception Hierarchy::

    VeniceError (base)
    ├── APIError (API-related errors)
    │   ├── APIStatusError (HTTP status errors)
    │   ├── AuthenticationError (401 Unauthorized)
    │   ├── PermissionDeniedError (403 Forbidden)
    │   ├── InvalidRequestError (400 Bad Request)
    │   ├── NotFoundError (404 Not Found)
    │   ├── ModelGoneError (410 Gone — retired/unroutable model)
    │   ├── ConflictError (409 Conflict)
    │   ├── UnprocessableEntityError (422 Unprocessable Entity)
    │   ├── RateLimitError (429 Too Many Requests)
    │   ├── PaymentRequiredError (402 Payment Required)
    │   ├── InternalServerError (500+ Server Errors)
    │   └── ServiceUnavailableError (503 Service Unavailable)
    ├── APIConnectionError (Network connectivity)
    ├── APITimeoutError (Request timeouts)
    │   └── BillingTimeoutError (Billing API timeouts)
    ├── APIResponseProcessingError (Response parsing)
    │   └── APIResponseValidationError (Pydantic validation)
    ├── StreamConsumedError (Stream already consumed)
    ├── StreamClosedError (Stream closed)
    └── MissingStreamClassError (Configuration errors)

Example::

    from venice_ai.exceptions import RateLimitError, AuthenticationError, APIError

    try:
        response = await client.chat.completions.create(...)
    except RateLimitError as e:
        print(f"Rate limited, retry after {e.retry_after_seconds}s")
    except AuthenticationError:
        print("Check your API key")
    except APIError as e:
        print(f"API error: {e.status_code} - {e}")
"""

import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error codes returned by the Venice API
# ---------------------------------------------------------------------------


class VeniceAPIErrorCode(StrEnum):
    """Machine-readable error codes the Venice API may return.

    The Venice error envelope is **not uniform**: many endpoints return a bare
    string message (``{"error": "Model is required"}``), validation failures
    return a Zod-style shape (``{"details": {...}, "issues": [...]}``), and the
    schema'd errors carry a machine-readable code as a **top-level** field
    (``response["code"]``) — not nested under ``error``. This enum models those
    top-level codes; ``exc.code`` is ``None`` when the error carried only a
    string message.

    The enum tracks the official Venice error-codes table but is **not
    guaranteed exhaustive** — the API may return new codes before this enum (or
    the public docs) is updated, so callers should treat an unrecognized
    ``exc.code`` string defensively rather than assuming enum membership.

    Using this enum enables type-safe matching instead of comparing raw strings::

        from venice_ai.exceptions import RateLimitError, VeniceAPIErrorCode

        try:
            response = await client.chat.completions.create(...)
        except RateLimitError as exc:
            if exc.code == VeniceAPIErrorCode.RATE_LIMIT_EXCEEDED:
                ...
    """

    # --- Officially documented codes (api-reference/error-codes) -----------
    # 401
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHENTICATION_FAILED_INACTIVE_KEY = "AUTHENTICATION_FAILED_INACTIVE_KEY"
    X402_INVALID_SIGN_IN = "X402_INVALID_SIGN_IN"
    PRO_ONLY_MODEL = "PRO_ONLY_MODEL"
    # 402
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    API_KEY_DIEM_SPEND_LIMIT_EXCEEDED = "API_KEY_DIEM_SPEND_LIMIT_EXCEEDED"
    API_KEY_USD_SPEND_LIMIT_EXCEEDED = "API_KEY_USD_SPEND_LIMIT_EXCEEDED"
    # 403
    UNAUTHORIZED = "UNAUTHORIZED"
    MODEL_PRIVACY_RESTRICTED = "MODEL_PRIVACY_RESTRICTED"
    API_ACCESS_DISABLED = "API_ACCESS_DISABLED"
    X402_WALLET_MISMATCH = "X402_WALLET_MISMATCH"
    # 400
    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_MODEL = "INVALID_MODEL"
    REQUEST_ID_NOT_FOUND = "REQUEST_ID_NOT_FOUND"
    INVALID_AUDIO_FORMAT = "INVALID_AUDIO_FORMAT"
    INVALID_VIDEO_FORMAT = "INVALID_VIDEO_FORMAT"
    CORRUPTED_IMAGE = "CORRUPTED_IMAGE"
    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    TOO_MANY_TOKENS = "TOO_MANY_TOKENS"
    # 404
    CHARACTER_NOT_FOUND = "CHARACTER_NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MEDIA_NOT_FOUND = "MEDIA_NOT_FOUND"
    # 413 / 415
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"
    # 422
    VIDEO_DURATION_TOO_LONG = "VIDEO_DURATION_TOO_LONG"
    VIDEO_DURATION_TOO_SHORT = "VIDEO_DURATION_TOO_SHORT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    CONTENT_POLICY_VIOLATION = "CONTENT_POLICY_VIOLATION"
    ASR_UPSTREAM_VALIDATION_FAILED = "ASR_UPSTREAM_VALIDATION_FAILED"
    # 429
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    MODEL_OVERLOADED = "MODEL_OVERLOADED"
    # 500
    INFERENCE_FAILED = "INFERENCE_FAILED"
    UPSCALE_FAILED = "UPSCALE_FAILED"
    IMAGE_EDIT_ERROR = "IMAGE_EDIT_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    # 502
    TEE_ATTESTATION_FAILED = "TEE_ATTESTATION_FAILED"
    TEE_SIGNATURE_FAILED = "TEE_SIGNATURE_FAILED"
    ASR_UPSTREAM_FAILED = "ASR_UPSTREAM_FAILED"
    # 503 / 504
    MODEL_OFFLINE = "MODEL_OFFLINE"
    MODEL_AT_CAPACITY = "MODEL_AT_CAPACITY"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"

    # --- Observed-in-wild codes not (yet) in the official error table ------
    INVALID_API_KEY = "INVALID_API_KEY"
    INVALID_FILE_SIZE = "INVALID_FILE_SIZE"
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"


__all__ = [
    "VeniceAPIErrorCode",
    "VeniceError",
    "APIError",
    "APIStatusError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "NotFoundError",
    "ModelGoneError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "BillingTimeoutError",
    "APIResponseProcessingError",
    "MissingStreamClassError",
    "VideoGenerationError",
    "MusicGenerationError",
    "StreamConsumedError",
    "StreamClosedError",
    "PaymentRequiredError",
    "ServiceUnavailableError",
    "APIResponseValidationError",
    "TeeError",
    "TeeAttestationError",
    "TeeEncryptionError",
]


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class VeniceError(Exception):
    """Base exception for all errors raised by the Venice AI SDK.

    Attributes:
        message: Human-readable description of what went wrong.
        request_obj: The request object that caused the error, if available.
        response_obj: The HTTP response object, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message)
        self.request_obj = request
        self.response_obj = response
        self.message = message

    @property
    def request(self) -> Any | None:
        return self.request_obj

    @property
    def response(self) -> aiohttp.ClientResponse | Any | None:
        return self.response_obj


# ---------------------------------------------------------------------------
# API error hierarchy  (HTTP errors from the Venice API)
# ---------------------------------------------------------------------------


class APIError(VeniceError):
    """Base exception for all API-related errors from the Venice AI service.

    Raised when the API returns a non-success HTTP status code.

    Attributes:
        status_code: HTTP status code from the API response.
        body: Parsed response body containing error details, if available.
        code: Venice API error code string, if present in the response body.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any,
        body: Any | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        # Handle both aiohttp.ClientResponse (with .status) and other response objects (with .status_code)
        if hasattr(response, "status"):
            self.status_code = response.status
        elif hasattr(response, "status_code"):
            self.status_code = response.status_code  # pyright: ignore[reportAttributeAccessIssue]
        else:
            self.status_code = None
        self.body = body
        self.code = code


class APIStatusError(APIError):
    """Raised when the API returns a non-success status code.

    Initialised with the response object and parsed body for detailed error
    information.
    """

    def __init__(
        self,
        *,
        response: aiohttp.ClientResponse,
        body: Any,
        request: Any | None = None,
    ) -> None:
        message = f"API request failed with status {response.status}"
        if body:
            if isinstance(body, dict):
                error_data = body.get("error")
                if isinstance(error_data, dict):
                    detail = error_data.get("message") or error_data.get("detail")
                    if detail:
                        message = f"{message}: {detail}"
                elif isinstance(error_data, str):
                    message = f"{message}: {error_data}"
            elif isinstance(body, str):
                message = f"{message}: {body}"

        super().__init__(message, request=request, response=response, body=body)


class AuthenticationError(APIError):
    """Raised for 401 Unauthorized errors — invalid or missing API key."""


class PermissionDeniedError(APIError):
    """Raised for 403 Forbidden errors — authenticated but not authorised."""


class InvalidRequestError(APIError):
    """Raised for 400 Bad Request errors, and also 413/415 status codes.

    Common causes: missing required fields, invalid parameter values, oversized
    files, or unsupported content types.
    """


class NotFoundError(APIError):
    """Raised for 404 Not Found errors — resource or endpoint not found."""


class ModelGoneError(APIError):
    """Raised for HTTP 410 Gone — the requested model has been retired and
    could not be auto-routed to a replacement.

    Venice auto-routes deprecated models to a similar model where possible;
    a 410 means routing was not possible (technical or safety reasons). Pick
    a current model (see ``client.models.list()``) or resolve one via traits.
    """


class ConflictError(APIError):
    """Raised for 409 Conflict errors — resource conflict or concurrent modification."""


class UnprocessableEntityError(APIError):
    """Raised for 422 Unprocessable Entity errors — server-side validation failures."""


class RateLimitError(APIError):
    """Raised for 429 Too Many Requests errors when rate limits are exceeded.

    Attributes:
        retry_after_seconds: Seconds to wait before retrying, parsed from the
            ``Retry-After`` header (may be ``None``).
        remaining_requests: Requests remaining in the current window, if reported.
        reset_requests_timestamp: Absolute Unix timestamp **in seconds** when
            the request limit resets, normalized from the millisecond-epoch
            ``x-ratelimit-reset-requests`` header.
        cached_rate_limit_headers: Pre-extracted rate-limit headers (lowercase keys)
            for use by distributed backend state synchronisation.
        custom_message: The API's ``customMessage``, naming which cap tripped
            (may be ``None``). Several different limits all surface as a 429 —
            the per-minute request cap, the per-day credit cap, and two rolling
            30-second error budgets: failed requests, and requests asking a
            model for a feature it does not support. They need different
            responses, and only this message distinguishes them. An
            unsupported-feature trip in particular will keep recurring until
            the request itself changes, so retrying it unchanged is futile.
    """

    retry_after_seconds: int | None
    remaining_requests: int | None
    reset_requests_timestamp: float | None
    cached_rate_limit_headers: dict[str, str]
    custom_message: str | None

    def __init__(
        self,
        message: str,
        *,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any,
        body: Any | None = None,
        retry_after_seconds: int | None = None,
        remaining_requests: int | None = None,
        reset_requests_timestamp: float | None = None,
        cached_rate_limit_headers: dict[str, str] | None = None,
        custom_message: str | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response, body=body)
        self.custom_message = custom_message
        self.retry_after_seconds = retry_after_seconds
        self.remaining_requests = remaining_requests
        self.reset_requests_timestamp = reset_requests_timestamp
        # Store pre-extracted headers passed from caller
        self.cached_rate_limit_headers = cached_rate_limit_headers or {}


class PaymentRequiredError(APIError):
    """Raised for 402 Payment Required errors — insufficient balance or credits."""


class InternalServerError(APIError):
    """Raised for all 5xx server-side errors except 503.

    Dispatch is by HTTP status range, not by API error code: any status in
    500-599 maps here except 503, which has its own subclass
    (:class:`ServiceUnavailableError`). Known HTTP 500 error codes include
    ``INFERENCE_FAILED``, ``UPSCALE_FAILED``, and ``UNKNOWN_ERROR``; other
    5xx statuses (e.g. 502, 504) fall through to this class as well.
    """


class ServiceUnavailableError(APIError):
    """Raised for 503 Service Unavailable errors — service temporarily down.

    Clients should implement retry logic with exponential backoff when
    encountering this error.
    """


# ---------------------------------------------------------------------------
# Network / connection errors
# ---------------------------------------------------------------------------


class APIConnectionError(VeniceError):
    """Raised when there is a network-level connectivity issue.

    Covers DNS failures, TCP connection errors, SSL/TLS problems, and
    proxy configuration issues.

    Attributes:
        original_error: The underlying exception that triggered this error.
    """

    def __init__(
        self,
        message: str = "Connection error",
        *,
        original_error: Exception | None = None,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        self.original_error = original_error


class APITimeoutError(VeniceError):
    """Raised when an API request exceeds the configured timeout.

    Attributes:
        original_error: The underlying timeout exception.
    """

    def __init__(
        self,
        message: str = "Request timed out",
        *,
        original_error: Exception | None = None,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        self.original_error = original_error


class BillingTimeoutError(APITimeoutError):
    """Raised when a billing API request times out.

    The SDK wraps billing requests in an aggressive timeout so a slow query
    fails fast instead of hanging.

    Common causes:
    - A full page is simply slow: the endpoint takes several seconds to return
      a default 1000-entry page, and slows further under concurrent load
    - Date range smaller than ~15 minutes, which the API can hang on
    - Query returns no data (empty result set)
    - Complex filters on sparse data

    Tips: cut ``pageSize`` to shrink a slow page, and use a date range of at
    least one full day where data is known to exist.
    """

    def __init__(
        self,
        message: str = (
            "Billing API request timed out. The usage-history endpoint is slow — a "
            "default 1000-entry page takes several seconds and degrades under "
            "concurrent load — and it can hang outright on very small date ranges "
            "(< 15 minutes) or filters that match no data. Try a smaller pageSize, "
            "or a date range of at least 1 day."
        ),
        *,
        original_error: Exception | None = None,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(
            message,
            original_error=original_error,
            request=request,
            response=response,
        )


# ---------------------------------------------------------------------------
# Response processing errors
# ---------------------------------------------------------------------------


class APIResponseProcessingError(VeniceError):
    """Raised when the client fails to process a received API response.

    Covers JSON parse errors, unexpected response structure, missing fields,
    and type conversion failures.

    Attributes:
        original_error: The underlying parse / processing exception.
    """

    def __init__(
        self,
        message: str,
        *,
        original_error: Exception | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, response=response)
        self.original_error = original_error


class APIResponseValidationError(APIResponseProcessingError):
    """Raised when Pydantic validation fails on an API response.

    Indicates a mismatch between the expected response schema and the actual
    response — typically caused by an API format change.

    Attributes:
        validation_error: The original ``pydantic.ValidationError``.
        response_data: The raw response data that failed validation.
        model_name: The Pydantic model name that failed validation.
    """

    def __init__(
        self,
        message: str,
        *,
        validation_error: Exception,
        response_data: Any | None = None,
        model_name: str | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, original_error=validation_error, response=response)
        self.validation_error = validation_error
        self.response_data = response_data
        self.model_name = model_name


# ---------------------------------------------------------------------------
# Streaming / configuration errors
# ---------------------------------------------------------------------------


class VideoGenerationError(VeniceError):
    """Raised when a video generation job fails on the server.

    Attributes:
        error_code: The error code returned by the API, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        self.error_code = error_code


class MusicGenerationError(VeniceError):
    """Raised when a music generation job fails on the server.

    Attributes:
        error_code: The error code returned by the API, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        request: Any | None = None,
        response: aiohttp.ClientResponse | Any | None = None,
    ) -> None:
        super().__init__(message, request=request, response=response)
        self.error_code = error_code


class MissingStreamClassError(VeniceError):
    """Raised when ``stream=True`` is passed but no ``stream_cls`` is provided."""

    pass


class StreamConsumedError(VeniceError):
    """Stream has already been consumed and cannot be read again."""

    pass


class StreamClosedError(VeniceError):
    """Stream has been closed and cannot be read."""

    pass


class TeeError(VeniceError):
    """Base exception for Venice TEE (Trusted Execution Environment) / E2EE errors.

    Raised for failures in the confidential-compute end-to-end-encryption path
    (``venice_ai.tee``): attestation verification, key agreement, and
    encrypt/decrypt of message content. Subclasses :class:`VeniceError`.
    """


class TeeAttestationError(TeeError):
    """Raised when TEE attestation verification fails (fail-closed).

    Examples: the server reported ``verified == false``; the response nonce did
    not match the client-sent nonce; the report-data binding did not match the
    expected ``signing_address || zero-pad || nonce`` layout; or TDX debug flags
    were set. The SDK fails closed — it raises rather than proceeding with an
    unverified enclave.
    """


class TeeEncryptionError(TeeError):
    """Raised when TEE end-to-end encryption or decryption fails.

    Examples: a message could not be encrypted to the model key, or an encrypted
    response chunk could not be decrypted / authenticated (GCM tag mismatch).
    """


class MaxIterationsExceededError(VeniceError):
    """Raised when ``chat.completions.run_with_tools`` doesn't converge.

    The tool-orchestration loop runs at most ``max_iterations`` model round
    trips before giving up. If every iteration ends with
    ``finish_reason="tool_calls"``, the loop is either looping
    pathologically (model keeps requesting tools that don't satisfy it) or
    the cap was set too low.

    Attributes:
        iterations: How many round trips were attempted (== ``max_iterations``).
        messages: The full message history at the time of failure, including
            every assistant tool-call turn and tool-result the loop produced.
            Useful for diagnosing why the loop didn't converge.
        last_response: The model's last (still-tool-calling) response, in
            case it carries diagnostic clues — e.g. usage information or
            unusual tool-call shapes.
    """

    def __init__(
        self,
        message: str,
        *,
        iterations: int,
        messages: list[Any],
        last_response: Any,
    ) -> None:
        super().__init__(message)
        self.iterations = iterations
        self.messages = messages
        self.last_response = last_response


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_retry_after_header(
    header_value: str, response_date_str: str | None = None
) -> int | None:
    """Parse a ``Retry-After`` header value and return the delay in seconds.

    The header can contain either an integer (seconds to wait) or an HTTP-date
    string (absolute time to retry).

    Returns:
        Number of seconds to wait, or ``None`` if parsing fails.
    """
    try:
        return int(header_value)
    except ValueError:
        try:
            retry_after_dt = parsedate_to_datetime(header_value)
            if retry_after_dt.tzinfo is None:
                retry_after_dt = retry_after_dt.replace(tzinfo=UTC)

            if response_date_str:
                server_now_dt = parsedate_to_datetime(response_date_str)
                if server_now_dt.tzinfo is None:
                    server_now_dt = server_now_dt.replace(tzinfo=UTC)
                now_dt: datetime = server_now_dt
            else:
                now_dt = datetime.now(UTC)

            delta = (retry_after_dt - now_dt).total_seconds()
            return max(0, int(delta))
        except (TypeError, ValueError) as e:
            logger.debug(
                f"Failed to parse Retry-After header '{header_value}': {e}",
                extra={"header_value": header_value},
            )
            return None


def _make_status_error(
    message: str | None,
    *,
    request: Any | None = None,
    body: Any | None,
    response: aiohttp.ClientResponse | Any,
    rate_limit_headers: dict[str, str] | None = None,
) -> APIError:
    """Create a specific APIError subclass based on the HTTP status code.

    Args:
        message: Optional error message override.
        request: The request object that caused the error.
        body: Parsed response body with error details.
        response: The HTTP response object.
        rate_limit_headers: Optional pre-extracted rate-limit headers (lowercase
            keys).  When provided these are used instead of ``response.headers``
            for rate-limit information and are cached on the resulting
            :class:`RateLimitError` for distributed backend synchronisation.
            Expected keys: ``x-ratelimit-*``, ``retry-after``.
    """
    status_code = getattr(response, "status", getattr(response, "status_code", 500))
    base_message = message if message else f"HTTP Status {status_code}"
    err_msg = base_message

    # Parse error details from response body
    error_code: str | None = None
    custom_message: str | None = None
    if isinstance(body, dict):
        error_data = body.get("error")
        if isinstance(error_data, dict):
            detail = error_data.get("message") or error_data.get("detail")
            if detail:
                err_msg = f"{base_message}: {detail}"
            error_code = error_data.get("code")
            custom_message = error_data.get("customMessage")
        elif isinstance(error_data, str):
            err_msg = f"{base_message}: {error_data}"
        if error_code is None:
            error_code = body.get("code")
        if custom_message is None:
            custom_message = body.get("customMessage")
    elif isinstance(body, str) and body.strip():
        err_msg = f"{base_message}: {body}"

    def _build(exc_cls: type[APIError]) -> APIError:
        exc = exc_cls(err_msg, request=request, response=response, body=body)
        exc.code = error_code
        return exc

    match status_code:
        case 413 | 415:
            # File size / content type → InvalidRequestError
            return _build(InvalidRequestError)

        case 429:
            # Rate limit → RateLimitError with header parsing
            headers = rate_limit_headers or {}
            retry_after_header = headers.get("retry-after") or response.headers.get("Retry-After")
            retry_after_seconds = None
            if retry_after_header:
                date_header = headers.get("date") or response.headers.get("Date")
                retry_after_seconds = _parse_retry_after_header(retry_after_header, date_header)

            remaining_requests_str = headers.get(
                "x-ratelimit-remaining-requests"
            ) or response.headers.get("x-ratelimit-remaining-requests")
            reset_requests_str = headers.get("x-ratelimit-reset-requests") or response.headers.get(
                "x-ratelimit-reset-requests"
            )

            exc: APIError = RateLimitError(
                err_msg,
                request=request,
                response=response,
                body=body,
                retry_after_seconds=retry_after_seconds,
                remaining_requests=_safe_int_parse(remaining_requests_str),
                reset_requests_timestamp=ms_epoch_to_seconds(_safe_float_parse(reset_requests_str)),
                cached_rate_limit_headers=headers,
                custom_message=custom_message,
            )
            exc.code = error_code
            return exc

        # Direct status-code → exception class mapping
        case 400:
            return _build(InvalidRequestError)
        case 401:
            return _build(AuthenticationError)
        case 402:
            return _build(PaymentRequiredError)
        case 403:
            return _build(PermissionDeniedError)
        case 404:
            return _build(NotFoundError)
        case 410:
            return _build(ModelGoneError)
        case 409:
            return _build(ConflictError)
        case 422:
            return _build(UnprocessableEntityError)
        case 503:
            return _build(ServiceUnavailableError)

        # 5xx → InternalServerError
        case code if 500 <= code < 600:
            return _build(InternalServerError)

        # Unhandled 4xx
        case code if 400 <= code < 500:
            exc = APIError(
                f"Unhandled 4xx error: {err_msg}",
                request=request,
                response=response,
                body=body,
            )
            exc.code = error_code
            return exc

        # Fallback for any other status code
        case _:
            exc = APIError(err_msg, request=request, response=response, body=body)
            exc.code = error_code
            return exc


# Re-export canonical implementations from utils.parsing so that any
# existing callers of ``_safe_int_parse`` / ``_safe_float_parse`` continue
# to work without changes.
from .utils.parsing import ms_epoch_to_seconds  # noqa: E402
from .utils.parsing import safe_float as _safe_float_parse  # noqa: E402
from .utils.parsing import safe_int as _safe_int_parse  # noqa: E402
