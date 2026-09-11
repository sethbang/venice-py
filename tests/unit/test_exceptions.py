"""
Consolidated tests for venice_ai/exceptions.py.

Merged from test_exceptions_coverage.py and test_exceptions_extended.py.
"""

import time
from email.utils import formatdate
from unittest.mock import MagicMock, Mock

import pytest

from venice_ai.exceptions import (
    APIError,
    APIStatusError,
    ConflictError,
    InvalidRequestError,
    ModelGoneError,
    NotFoundError,
    PaymentRequiredError,
    RateLimitError,
    ServiceUnavailableError,
    UnprocessableEntityError,
    _make_status_error,
    _parse_retry_after_header,
    _safe_float_parse,
    _safe_int_parse,
)


class TestAPIErrorEdgeCases:
    """Test edge cases in APIError initialization."""

    def test_api_error_with_response_status_code_attr(self):
        """Test APIError when response has status_code instead of status."""
        response = Mock()
        response.status_code = 404
        del response.status

        error = APIError("Test error", request=None, response=response, body=None)
        assert error.status_code == 404

    def test_api_error_with_no_status_attributes(self):
        """Test APIError when response has neither status nor status_code."""
        response = Mock()
        if hasattr(response, "status"):
            del response.status
        if hasattr(response, "status_code"):
            del response.status_code

        error = APIError("Test error", request=None, response=response, body=None)
        assert error.status_code is None


class TestAPIStatusErrorMessageConstruction:
    """Test complex error message construction in APIStatusError."""

    def test_api_status_error_with_string_body(self):
        """Test APIStatusError with string body."""
        response = MagicMock()
        response.status = 400

        error = APIStatusError(response=response, body="Simple string error message", request=None)
        assert "Simple string error message" in str(error)

    def test_api_status_error_with_dict_error_string(self):
        """Test APIStatusError when error dict contains string."""
        response = MagicMock()
        response.status = 400

        error = APIStatusError(response=response, body={"error": "Error as string"}, request=None)
        assert "Error as string" in str(error)

    def test_api_status_error_with_dict_error_detail(self):
        """Test APIStatusError when error dict uses 'detail' key."""
        response = MagicMock()
        response.status = 400

        error = APIStatusError(
            response=response,
            body={"error": {"detail": "Detailed error message", "code": "ERR001"}},
            request=None,
        )
        assert "Detailed error message" in str(error)

    def test_api_status_error_with_empty_body(self):
        """Test APIStatusError with empty/None body."""
        response = MagicMock()
        response.status = 500

        error = APIStatusError(response=response, body=None, request=None)
        assert "API request failed with status 500" in str(error)


class TestPaymentRequiredError:
    """Test PaymentRequiredError initialization."""

    def test_payment_required_error_initialization(self):
        """Test PaymentRequiredError creation with all parameters."""
        response = MagicMock()
        response.status = 402

        error = PaymentRequiredError(
            "Payment required for this operation",
            request={"endpoint": "/api/premium"},
            response=response,
            body={"error": {"message": "Insufficient credits"}},
        )

        assert isinstance(error, PaymentRequiredError)
        assert error.status_code == 402
        assert "Payment required" in str(error)


class TestRetryAfterHeaderParsing:
    """Test Retry-After header parsing edge cases."""

    def test_parse_retry_after_with_http_date_and_server_date(self):
        """Test parsing Retry-After with HTTP date and Date header."""
        future_time = formatdate(timeval=time.time() + 60, usegmt=True)
        server_time = formatdate(timeval=time.time(), usegmt=True)

        result = _parse_retry_after_header(future_time, server_time)
        assert result is not None
        assert 58 <= result <= 62

    def test_parse_retry_after_with_http_date_no_server_date(self):
        """Test parsing Retry-After with HTTP date but no server Date header."""
        future_time = formatdate(timeval=time.time() + 60, usegmt=True)

        result = _parse_retry_after_header(future_time)
        assert result is not None
        assert 58 <= result <= 62

    def test_parse_retry_after_with_invalid_date_format(self):
        """Test parsing Retry-After with invalid date string."""
        result = _parse_retry_after_header("Not a valid date", None)
        assert result is None

    def test_parse_retry_after_with_past_date(self):
        """Test parsing Retry-After with past date."""
        past_time = formatdate(timeval=time.time() - 60, usegmt=True)
        result = _parse_retry_after_header(past_time, None)
        assert result == 0


class TestMakeStatusError:
    """Test _make_status_error for various HTTP status codes."""

    def test_make_status_error_402_payment_required(self):
        """Test _make_status_error for 402 Payment Required."""
        response = MagicMock()
        response.status = 402
        response.headers = {}

        error = _make_status_error(
            "Payment required",
            body={"error": {"message": "Subscription expired"}},
            response=response,
        )

        assert isinstance(error, PaymentRequiredError)
        assert error.status_code == 402

    def test_make_status_error_403_permission_denied(self):
        """Test _make_status_error for 403 Forbidden."""
        response = MagicMock()
        response.status = 403
        response.headers = {}

        error = _make_status_error(
            "Forbidden", body={"error": {"message": "Access denied"}}, response=response
        )

        assert error.status_code == 403
        assert "Access denied" in str(error)

    def test_make_status_error_409_conflict(self):
        """Test _make_status_error for 409 Conflict."""
        response = MagicMock()
        response.status = 409
        response.headers = {}

        error = _make_status_error(
            "Conflict error",
            body={"error": {"message": "Resource exists"}},
            response=response,
        )

        assert isinstance(error, ConflictError)
        assert "Resource exists" in str(error)

    def test_make_status_error_413_file_too_large(self):
        """Test _make_status_error for 413 Payload Too Large."""
        response = MagicMock()
        response.status = 413
        response.headers = {}

        error = _make_status_error(
            "File too large",
            body={"error": {"message": "File exceeds 10MB limit"}},
            response=response,
        )

        assert isinstance(error, InvalidRequestError)
        assert error.status_code == 413

    def test_make_status_error_415_unsupported_media(self):
        """Test _make_status_error for 415 Unsupported Media Type."""
        response = MagicMock()
        response.status = 415
        response.headers = {}

        error = _make_status_error(
            "Unsupported media type",
            body={"error": {"message": "Only PNG and JPEG supported"}},
            response=response,
        )

        assert isinstance(error, InvalidRequestError)
        assert error.status_code == 415

    def test_make_status_error_422_unprocessable_entity(self):
        """Test _make_status_error for 422 Unprocessable Entity."""
        response = MagicMock()
        response.status = 422
        response.headers = {}

        error = _make_status_error(
            "Validation error", body={"error": {"code": "INVALID_DATA"}}, response=response
        )

        assert isinstance(error, UnprocessableEntityError)
        assert "Validation error" in str(error)

    def test_make_status_error_503_service_unavailable(self):
        """Test _make_status_error for 503 Service Unavailable."""
        response = MagicMock()
        response.status = 503
        response.headers = {}

        error = _make_status_error("Service unavailable", body=None, response=response)

        assert isinstance(error, ServiceUnavailableError)


class TestRateLimitHeaderParsing:
    """Test rate limit header parsing failures."""

    def test_make_status_error_429_with_retry_after_seconds(self):
        """Test 429 error with Retry-After as seconds."""
        response = MagicMock()
        response.status = 429
        response.headers = {"Retry-After": "30"}

        error = _make_status_error(
            "Rate limited",
            body={"error": {"message": "Too many requests"}},
            response=response,
        )

        assert isinstance(error, RateLimitError)
        assert error.retry_after_seconds == 30

    def test_make_status_error_429_with_retry_after_date(self):
        """Test 429 error with Retry-After as HTTP date."""
        response = MagicMock()
        response.status = 429
        future_time = formatdate(timeval=time.time() + 45, usegmt=True)
        response.headers = {"Retry-After": future_time}

        error = _make_status_error("Rate limited", body=None, response=response)

        assert isinstance(error, RateLimitError)
        assert error.retry_after_seconds is not None
        assert 43 <= error.retry_after_seconds <= 47

    def test_make_status_error_429_with_invalid_retry_after(self):
        """Test 429 error with invalid Retry-After header."""
        response = MagicMock()
        response.status = 429
        response.headers = {"Retry-After": "invalid_value"}

        error = _make_status_error("Rate limited", body=None, response=response)

        assert isinstance(error, RateLimitError)
        assert error.retry_after_seconds is None

    def test_make_status_error_429_reset_requests_ms_epoch_normalized(self):
        """x-ratelimit-reset-requests is a 13-digit ms-epoch.

        The header value (live-verified: 1780580108941) must be normalized to
        absolute Unix *seconds* on ``reset_requests_timestamp`` rather than
        stored raw (1000x off).
        """
        ms_epoch = 1780580108941
        response = MagicMock()
        response.status = 429
        response.headers = {"x-ratelimit-reset-requests": str(ms_epoch)}

        error = _make_status_error("Rate limited", body=None, response=response)

        assert isinstance(error, RateLimitError)
        assert error.reset_requests_timestamp == ms_epoch / 1000
        assert error.reset_requests_timestamp != float(ms_epoch)

    def test_make_status_error_429_reset_requests_seconds_epoch_unchanged(self):
        """A 10-digit seconds-epoch reset header passes through."""
        seconds_epoch = 1704067200
        response = MagicMock()
        response.status = 429
        response.headers = {"x-ratelimit-reset-requests": str(seconds_epoch)}

        error = _make_status_error("Rate limited", body=None, response=response)

        assert isinstance(error, RateLimitError)
        assert error.reset_requests_timestamp == float(seconds_epoch)


class TestSafeParsingFunctions:
    """Test _safe_int_parse and _safe_float_parse functions."""

    def test_safe_int_parse_valid(self):
        assert _safe_int_parse("123") == 123
        assert _safe_int_parse("0") == 0
        assert _safe_int_parse("-456") == -456

    def test_safe_int_parse_invalid(self):
        assert _safe_int_parse("not_a_number") is None
        assert _safe_int_parse("12.34") == 12  # Float string — truncates via int(float(...))
        assert _safe_int_parse("") is None
        assert _safe_int_parse("infinity") is None

    def test_safe_int_parse_none(self):
        assert _safe_int_parse(None) is None

    def test_safe_int_parse_type_error(self):
        assert _safe_int_parse([123]) is None  # type: ignore[arg-type]
        assert _safe_int_parse({"value": 123}) is None  # type: ignore[arg-type]

    def test_safe_float_parse_valid(self):
        assert _safe_float_parse("123.45") == 123.45
        assert _safe_float_parse("0.0") == 0.0
        assert _safe_float_parse("-67.89") == -67.89
        assert _safe_float_parse("123") == 123.0

    def test_safe_float_parse_invalid(self):
        assert _safe_float_parse("not_a_number") is None
        assert _safe_float_parse("") is None
        assert _safe_float_parse("12.34.56") is None

    def test_safe_float_parse_none(self):
        assert _safe_float_parse(None) is None

    def test_safe_float_parse_type_error(self):
        assert _safe_float_parse([123.45]) is None  # type: ignore[arg-type]
        assert _safe_float_parse({"value": 123.45}) is None  # type: ignore[arg-type]


class TestMakeStatusErrorEdgeCases:
    """Test additional edge cases in _make_status_error."""

    def test_make_status_error_with_string_body(self):
        """Test _make_status_error with plain string body."""
        response = MagicMock()
        response.status = 500
        response.headers = {}

        error = _make_status_error(None, body="Internal server error occurred", response=response)
        assert "Internal server error occurred" in str(error)

    def test_make_status_error_with_empty_string_body(self):
        """Test _make_status_error with empty string body."""
        response = MagicMock()
        response.status = 503
        response.headers = {}

        error = _make_status_error(None, body="   ", response=response)
        assert "HTTP Status 503" in str(error)

    def test_make_status_error_with_legacy_response(self):
        """Test _make_status_error with legacy response object."""
        response = Mock()
        response.status_code = 404
        if hasattr(response, "status"):
            del response.status
        response.headers = {}

        error = _make_status_error("Not found", body=None, response=response)
        assert error.status_code == 404


class TestModelGoneError:
    """Test HTTP 410 → ModelGoneError mapping (Cluster E)."""

    def test_make_status_error_410_yields_model_gone_error(self):
        """_make_status_error for 410 must return a ModelGoneError."""
        response = MagicMock()
        response.status = 410
        response.headers = {}

        error = _make_status_error(
            "Model retired",
            body={"error": {"message": "Model has been retired and cannot be auto-routed"}},
            response=response,
        )

        assert isinstance(error, ModelGoneError)
        assert error.status_code == 410

    def test_model_gone_error_is_api_error(self):
        """ModelGoneError must be a subclass of APIError."""
        response = MagicMock()
        response.status = 410
        response.headers = {}

        error = _make_status_error("Gone", body=None, response=response)

        assert isinstance(error, APIError)

    def test_model_gone_error_not_not_found_error(self):
        """ModelGoneError must not be confused with NotFoundError (404)."""
        response = MagicMock()
        response.status = 410
        response.headers = {}

        error = _make_status_error("Gone", body=None, response=response)

        assert not isinstance(error, NotFoundError)

    def test_model_gone_error_importable_from_venice_ai(self):
        """ModelGoneError must be re-exported from the top-level venice_ai package."""
        from venice_ai import ModelGoneError as _ModelGoneError

        assert _ModelGoneError is ModelGoneError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestModelPrivacyRestrictedCode:
    """``MODEL_PRIVACY_RESTRICTED`` is returned when an API key's privacy tier
    excludes the model — by ``GET /models`` when filtering the listing, and on
    inference when the key may not call the requested model."""

    def test_code_is_a_known_error_code(self):
        from venice_ai.exceptions import VeniceAPIErrorCode

        assert VeniceAPIErrorCode.MODEL_PRIVACY_RESTRICTED == "MODEL_PRIVACY_RESTRICTED"

    def test_factory_surfaces_the_code(self):
        from venice_ai.exceptions import VeniceAPIErrorCode

        response = MagicMock()
        response.status = 403
        response.headers = {}

        error = _make_status_error(
            "Forbidden",
            body={
                "error": {"message": "Model not permitted for this key"},
                "code": "MODEL_PRIVACY_RESTRICTED",
            },
            response=response,
        )
        assert error.code == VeniceAPIErrorCode.MODEL_PRIVACY_RESTRICTED


class TestRateLimitCustomMessage:
    """A 429 body carries ``customMessage`` naming which cap tripped.

    Without it a caller only sees "Rate limit exceeded" and cannot tell a
    per-minute throughput cap from the failed-request or unsupported-feature
    error budgets, which need different fixes.
    """

    def test_custom_message_parsed_from_top_level(self):
        response = MagicMock()
        response.status = 429
        response.headers = {}

        error = _make_status_error(
            "Rate limit exceeded",
            body={
                "error": "Rate limit exceeded",
                "customMessage": (
                    "Too many failed attempts (> 50) resulting in a non-success "
                    "status code. Please wait 30 seconds and try again."
                ),
            },
            response=response,
        )
        assert isinstance(error, RateLimitError)
        assert error.custom_message is not None
        assert "Too many failed attempts" in error.custom_message

    def test_custom_message_parsed_when_nested_under_error(self):
        """The factory already accepts both flat and error-nested bodies."""
        response = MagicMock()
        response.status = 429
        response.headers = {}

        error = _make_status_error(
            "Rate limit exceeded",
            body={"error": {"message": "Slow down", "customMessage": "Per-minute cap"}},
            response=response,
        )
        assert isinstance(error, RateLimitError)
        assert error.custom_message == "Per-minute cap"

    def test_custom_message_is_none_when_absent(self):
        response = MagicMock()
        response.status = 429
        response.headers = {}

        error = _make_status_error(
            "Rate limit exceeded", body={"error": "Rate limit exceeded"}, response=response
        )
        assert isinstance(error, RateLimitError)
        assert error.custom_message is None
