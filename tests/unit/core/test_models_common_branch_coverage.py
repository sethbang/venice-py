"""
Comprehensive branch coverage tests for venice_ai.core.models.common module.

This test file addresses the 0% branch coverage identified in the audit by testing
all conditional logic paths in VeniceBaseModel and related helper methods.

Coverage targets:
- VeniceBaseModel.headers property (lines 41-43)
- VeniceBaseModel.response_rate_limits property (lines 49-76)
- VeniceBaseModel.deprecation_info property (lines 82-97)
- VeniceBaseModel.balance_info property (lines 102-120)
- VeniceBaseModel._parse_int method (lines 130-135)
- VeniceBaseModel._parse_float method (lines 139-144)
- VeniceBaseModel._parse_timestamp method (lines 148-159)
- DateRangeParams validator (lines 192-196)
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from venice_ai.core.models.common import (
    BalanceInfo,
    DateRangeParams,
    DeprecationInfo,
    PaginationInfo,
    RateLimitInfo,
    TimingInfo,
    UsageInfo,
    VeniceBaseModel,
    VeniceParameters,
)
from venice_ai.types.api.common import PromptTokensDetails, WebSearchCitation


class TestVeniceBaseModelHeaders:
    """Test VeniceBaseModel.headers property branch coverage."""

    def test_headers_with_response_and_headers_attribute(self):
        """Test headers property when response has headers attribute (line 41: True branch)."""
        model = VeniceBaseModel()

        # Create mock response with headers
        mock_response = Mock()
        mock_response.headers = {"x-test": "value", "content-type": "application/json"}
        model._response = mock_response

        result = model.headers

        assert result is not None
        assert isinstance(result, dict)
        assert result["x-test"] == "value"
        assert result["content-type"] == "application/json"

    def test_headers_without_response(self):
        """Test headers property when _response is None (line 41: False branch, first condition)."""
        model = VeniceBaseModel()
        model._response = None

        result = model.headers

        assert result is None

    def test_headers_without_headers_attribute(self):
        """Test headers property when response lacks headers attribute (line 41: False branch, second condition)."""
        model = VeniceBaseModel()

        # Create mock response without headers attribute
        mock_response = Mock(spec=[])  # spec=[] ensures no attributes
        model._response = mock_response

        result = model.headers

        assert result is None


class TestVeniceBaseModelRateLimits:
    """Test VeniceBaseModel.response_rate_limits property branch coverage."""

    def test_response_rate_limits_without_headers(self):
        """Test response_rate_limits when headers is None (line 49: False branch)."""
        model = VeniceBaseModel()
        model._response = None

        result = model.response_rate_limits

        assert result is None

    def test_response_rate_limits_with_complete_data(self):
        """Test response_rate_limits with all rate limit headers (line 73: True branch)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-reset-requests": "1704067200",
            "x-ratelimit-limit-tokens": "50000",
            "x-ratelimit-remaining-tokens": "48000",
            "x-ratelimit-reset-tokens": "1704067200",
        }
        model._response = mock_response

        result = model.response_rate_limits

        assert result is not None
        assert isinstance(result, RateLimitInfo)
        assert result.limit_requests == 100
        assert result.remaining_requests == 95
        assert result.limit_tokens == 50000
        assert result.remaining_tokens == 48000

    def test_response_rate_limits_with_partial_data(self):
        """Test response_rate_limits with some rate limit headers (line 73: True branch, partial data)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-remaining-requests": "50",
        }
        model._response = mock_response

        result = model.response_rate_limits

        assert result is not None
        assert result.remaining_requests == 50
        assert result.limit_requests is None
        assert result.limit_tokens is None

    def test_response_rate_limits_with_no_data(self):
        """Test response_rate_limits when no rate limit headers exist (line 73: False branch)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}
        model._response = mock_response

        result = model.response_rate_limits

        assert result is None

    def test_response_rate_limits_ms_epoch_reset_headers(self):
        """Reset headers arrive as 13-digit ms epochs.

        Live wire capture shows ``x-ratelimit-reset-requests`` /
        ``x-ratelimit-reset-tokens`` as absolute Unix epoch MILLISECONDS
        (e.g. ``1780567876726``). They must parse to correct values:

        * ``reset_requests`` (datetime) must NOT be None (was always None
          because ``fromtimestamp`` treated the ms value as seconds → year
          out of range → ValueError → None).
        * ``reset_tokens`` (float seconds) must be the seconds-epoch
          ``1780567876.727``, NOT the raw ms ``1.78e12``.
        """
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-limit-requests": "150",
            "x-ratelimit-remaining-requests": "149",
            "x-ratelimit-reset-requests": "1780567876726",
            "x-ratelimit-limit-tokens": "3000000",
            "x-ratelimit-remaining-tokens": "3000000",
            "x-ratelimit-reset-tokens": "1780567876727",
        }
        model._response = mock_response

        result = model.response_rate_limits

        assert result is not None
        assert isinstance(result, RateLimitInfo)
        # reset_requests must be a real datetime, not None.
        assert result.reset_requests is not None
        assert result.reset_requests == datetime.fromtimestamp(1780567876.726, tz=UTC)
        # reset_tokens normalized to seconds-epoch float, not raw ms.
        assert result.reset_tokens == 1780567876.727
        # reset_tokens is an ABSOLUTE Unix timestamp in seconds, not a
        # duration. A 13-digit ms-epoch header must land in the absolute-seconds
        # band (>= 1e9, < 1e11), never the raw ms value (~1.78e12) and never a
        # small relative duration.
        assert result.reset_tokens is not None
        assert 1e9 <= result.reset_tokens < 1e11

    def test_parse_timestamp_seconds_epoch_unchanged(self):
        """Guard: 10-digit seconds epochs still parse correctly (no /1000)."""
        model = VeniceBaseModel()
        result = model._parse_timestamp("1704067200")
        assert result == datetime.fromtimestamp(1704067200, tz=UTC)


class TestVeniceBaseModelDeprecation:
    """Test VeniceBaseModel.deprecation_info property branch coverage."""

    def test_deprecation_info_without_headers(self):
        """Test deprecation_info when headers is None (line 82: False branch)."""
        model = VeniceBaseModel()
        model._response = None

        result = model.deprecation_info

        assert result is None

    def test_deprecation_info_with_both_warning_and_date(self):
        """Test deprecation_info with both warning and date (line 88: False branch)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-venice-model-deprecation-warning": "This model will be deprecated",
            "x-venice-model-deprecation-date": "1704067200",
        }
        model._response = mock_response

        result = model.deprecation_info

        assert result is not None
        assert isinstance(result, DeprecationInfo)
        assert result.warning == "This model will be deprecated"
        assert result.date is not None
        assert isinstance(result.date, datetime)

    def test_deprecation_info_with_only_warning(self):
        """Test deprecation_info with only warning (line 88: False branch, partial)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-venice-model-deprecation-warning": "Deprecation pending",
        }
        model._response = mock_response

        result = model.deprecation_info

        assert result is not None
        assert result.warning == "Deprecation pending"
        assert result.date is None

    def test_deprecation_info_with_only_date(self):
        """Test deprecation_info with only date (line 88: False branch, partial)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-venice-model-deprecation-date": "2024-12-31T23:59:59Z",
        }
        model._response = mock_response

        result = model.deprecation_info

        assert result is not None
        assert result.warning is None
        assert result.date is not None

    def test_deprecation_info_without_deprecation_headers(self):
        """Test deprecation_info when no deprecation headers exist (line 88: True branch)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}
        model._response = mock_response

        result = model.deprecation_info

        assert result is None


class TestVeniceBaseModelBalance:
    """Test VeniceBaseModel.balance_info property branch coverage."""

    def test_balance_info_without_headers(self):
        """Test balance_info when headers is None (line 103: False branch)."""
        model = VeniceBaseModel()
        model._response = None

        result = model.balance_info

        assert result is None

    def test_balance_info_with_all_balances(self):
        """Test balance_info with all balance types (line 110: False branch).

        Note: x-venice-balance-vcu header has been removed from the API.
        The vcu field is deprecated and always None.
        """
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-venice-balance-diem": "100.50",
            "x-venice-balance-usd": "50.25",
        }
        model._response = mock_response

        result = model.balance_info

        assert result is not None
        assert isinstance(result, BalanceInfo)
        assert result.diem == 100.50
        assert result.usd == 50.25

    def test_balance_info_with_partial_balances(self):
        """Test balance_info with some balance types (line 110: False branch, partial)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-venice-balance-usd": "25.00",
        }
        model._response = mock_response

        result = model.balance_info

        assert result is not None
        assert result.usd == 25.00
        assert result.diem is None

    def test_balance_info_without_balance_headers(self):
        """Test balance_info when no balance headers exist (line 110: True branch)."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {"content-type": "application/json"}
        model._response = mock_response

        result = model.balance_info

        assert result is None


class TestVeniceBaseModelParseInt:
    """Test VeniceBaseModel._parse_int method branch coverage."""

    def test_parse_int_with_none(self):
        """Test _parse_int with None value (line 130: True branch)."""
        model = VeniceBaseModel()

        result = model._parse_int(None)

        assert result is None

    def test_parse_int_with_valid_string(self):
        """Test _parse_int with valid integer string (line 133: success path)."""
        model = VeniceBaseModel()

        result = model._parse_int("42")

        assert result == 42

    def test_parse_int_with_invalid_string(self):
        """Test _parse_int with invalid string (line 134: ValueError branch)."""
        model = VeniceBaseModel()

        result = model._parse_int("not-a-number")

        assert result is None

    def test_parse_int_with_float_string(self):
        """Test _parse_int with float string — truncates to int via int(float(...))."""
        model = VeniceBaseModel()

        result = model._parse_int("42.5")

        assert result == 42

    def test_parse_int_with_type_error(self):
        """Test _parse_int with wrong type (line 134: TypeError branch)."""
        model = VeniceBaseModel()

        # Pass an object that can't be converted to int
        result = model._parse_int([1, 2, 3])  # type: ignore[arg-type]

        assert result is None


class TestVeniceBaseModelParseFloat:
    """Test VeniceBaseModel._parse_float method branch coverage."""

    def test_parse_float_with_none(self):
        """Test _parse_float with None value (line 139: True branch)."""
        model = VeniceBaseModel()

        result = model._parse_float(None)

        assert result is None

    def test_parse_float_with_valid_string(self):
        """Test _parse_float with valid float string (line 142: success path)."""
        model = VeniceBaseModel()

        result = model._parse_float("42.5")

        assert result == 42.5

    def test_parse_float_with_integer_string(self):
        """Test _parse_float with integer string (line 142: success path)."""
        model = VeniceBaseModel()

        result = model._parse_float("100")

        assert result == 100.0

    def test_parse_float_with_invalid_string(self):
        """Test _parse_float with invalid string (line 143: ValueError branch)."""
        model = VeniceBaseModel()

        result = model._parse_float("not-a-number")

        assert result is None

    def test_parse_float_with_type_error(self):
        """Test _parse_float with wrong type (line 143: TypeError branch)."""
        model = VeniceBaseModel()

        # Pass an object that can't be converted to float
        result = model._parse_float({"value": 42.5})  # type: ignore[arg-type]

        assert result is None


class TestVeniceBaseModelParseTimestamp:
    """Test VeniceBaseModel._parse_timestamp method branch coverage."""

    def test_parse_timestamp_with_none(self):
        """Test _parse_timestamp with None value (line 148: True branch)."""
        model = VeniceBaseModel()

        result = model._parse_timestamp(None)

        assert result is None

    def test_parse_timestamp_with_unix_timestamp(self):
        """Test _parse_timestamp with Unix timestamp string (line 152: True branch)."""
        model = VeniceBaseModel()

        result = model._parse_timestamp("1704067200")

        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None

    def test_parse_timestamp_with_iso_format(self):
        """Test _parse_timestamp with ISO format (line 157: alternate path)."""
        model = VeniceBaseModel()

        result = model._parse_timestamp("2024-01-01T00:00:00Z")

        assert result is not None
        assert isinstance(result, datetime)

    def test_parse_timestamp_with_iso_format_no_z(self):
        """Test _parse_timestamp with ISO format without Z (line 157: without Z replacement)."""
        model = VeniceBaseModel()

        result = model._parse_timestamp("2024-01-01T00:00:00+00:00")

        assert result is not None
        assert isinstance(result, datetime)

    def test_parse_timestamp_with_invalid_string(self):
        """Test _parse_timestamp with invalid string (line 158: ValueError branch)."""
        model = VeniceBaseModel()

        result = model._parse_timestamp("not-a-timestamp")

        assert result is None

    def test_parse_timestamp_with_invalid_unix_timestamp(self):
        """Test _parse_timestamp with invalid Unix timestamp (OSError/ValueError branch).

        Uses a value that overflows even after ms→seconds normalization
        (``/1000`` still yields a year far beyond 9999), so the
        out-of-range path is still exercised.
        """
        model = VeniceBaseModel()

        # Even divided by 1000 this is ~3e13 years out — still raises.
        result = model._parse_timestamp("999999999999999999999")

        assert result is None

    def test_parse_timestamp_with_type_error(self):
        """Test _parse_timestamp with wrong type causes AttributeError."""
        model = VeniceBaseModel()

        # Passing an integer causes AttributeError when calling .isdigit()
        with pytest.raises(AttributeError, match="'int' object has no attribute 'isdigit'"):
            model._parse_timestamp(12345)  # type: ignore[arg-type]


class TestDateRangeParamsValidator:
    """Test DateRangeParams validator branch coverage."""

    def test_date_range_params_valid_range(self):
        """Test DateRangeParams with valid date range (line 194: False branch)."""
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = datetime(2024, 12, 31, tzinfo=UTC)

        params = DateRangeParams(start_date=start, end_date=end)

        assert params.start_date == start
        assert params.end_date == end

    def test_date_range_params_end_equals_start(self):
        """Test DateRangeParams when end_date equals start_date (line 194: True branch, equal)."""
        date = datetime(2024, 1, 1, tzinfo=UTC)

        with pytest.raises(ValidationError) as exc_info:
            DateRangeParams(start_date=date, end_date=date)

        assert "end_date must be after start_date" in str(exc_info.value)

    def test_date_range_params_end_before_start(self):
        """Test DateRangeParams when end_date is before start_date (line 194: True branch, less than)."""
        start = datetime(2024, 12, 31, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)

        with pytest.raises(ValidationError) as exc_info:
            DateRangeParams(start_date=start, end_date=end)

        assert "end_date must be after start_date" in str(exc_info.value)

    def test_date_range_params_without_start_date(self):
        """Test DateRangeParams when start_date is None (line 193: first condition False)."""
        end = datetime(2024, 12, 31, tzinfo=UTC)

        # Should succeed - no validation error when start_date is None
        params = DateRangeParams(start_date=None, end_date=end)

        assert params.start_date is None
        assert params.end_date == end

    def test_date_range_params_without_end_date(self):
        """Test DateRangeParams when end_date is None (line 193: second condition False)."""
        start = datetime(2024, 1, 1, tzinfo=UTC)

        # Should succeed - no validation error when end_date is None
        params = DateRangeParams(start_date=start, end_date=None)

        assert params.start_date == start
        assert params.end_date is None

    def test_date_range_params_both_none(self):
        """Test DateRangeParams when both dates are None (line 193: both conditions False)."""
        params = DateRangeParams(start_date=None, end_date=None)

        assert params.start_date is None
        assert params.end_date is None


class TestModelCreationAndSerialization:
    """Test model creation and serialization for complete coverage."""

    def test_pagination_info_creation(self):
        """Test PaginationInfo model creation."""
        pagination = PaginationInfo(page=2, limit=50, total=150, total_pages=3)

        assert pagination.page == 2
        assert pagination.limit == 50
        assert pagination.total == 150
        assert pagination.total_pages == 3

    def test_usage_info_creation(self):
        """Test UsageInfo model creation."""
        usage = UsageInfo(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=20, audio_tokens=80),
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.prompt_tokens_details is not None

    def test_timing_info_creation(self):
        """Test TimingInfo model creation."""
        timing = TimingInfo(
            inference_duration=100.5,
            inference_preprocessing_time=10.2,
            inference_queue_time=5.3,
            total=116.0,
        )

        assert timing.inference_duration == 100.5
        assert timing.inference_preprocessing_time == 10.2
        assert timing.inference_queue_time == 5.3
        assert timing.total == 116.0

    def test_venice_parameters_default_values(self):
        """Test VeniceParameters with default values."""
        # All fields have defaults, so empty instantiation should work
        params = VeniceParameters()  # type: ignore[call-arg]

        assert params.character_slug is None
        assert params.strip_thinking_response is False
        assert params.disable_thinking is False
        assert params.enable_web_search == "off"
        assert params.enable_web_citations is False
        assert params.include_search_results_in_stream is False
        assert params.return_search_results_as_documents is None
        assert params.include_venice_system_prompt is True
        assert params.enable_e2ee is None
        assert params.enable_x_search is None

    def test_venice_parameters_custom_values(self):
        """Test VeniceParameters with custom values."""
        params = VeniceParameters(
            character_slug="alan-watts",
            strip_thinking_response=True,
            disable_thinking=True,
            enable_web_search="on",
            enable_web_citations=True,
            include_search_results_in_stream=True,
            return_search_results_as_documents=True,
            include_venice_system_prompt=False,
            enable_e2ee=True,
            enable_x_search=True,
        )

        assert params.character_slug == "alan-watts"
        assert params.strip_thinking_response is True
        assert params.disable_thinking is True
        assert params.enable_web_search == "on"
        assert params.enable_web_citations is True
        assert params.include_search_results_in_stream is True
        assert params.return_search_results_as_documents is True
        assert params.include_venice_system_prompt is False
        assert params.enable_e2ee is True
        assert params.enable_x_search is True

    def test_venice_parameters_e2ee_and_x_search_serialize(self):
        """Opt-in privacy/search flags serialize under their alias names.

        Constructing VeniceParameters no longer warns: the E2EE
        attestation-trust warning moved to ``chat.completions.create`` (it fires
        only when the real flow engages), so construction is silent.
        """
        params = VeniceParameters(enable_e2ee=True, enable_x_search=True)  # type: ignore[call-arg]
        dumped = params.model_dump(by_alias=True, exclude_none=True)
        assert dumped.get("enable_e2ee") is True
        assert dumped.get("enable_x_search") is True

    def test_venice_parameters_enable_e2ee_construction_does_not_warn(self):
        """The E2EE warning moved out of the model.

        Setting ``enable_e2ee=True`` on the model is no longer a "thinks-
        encrypted-isn't" trap: the real client-side E2EE flow now runs in
        ``chat.completions.create`` and emits the attestation-trust warning
        there. Plain model construction must therefore stay silent.
        """
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            VeniceParameters(enable_e2ee=True)  # type: ignore[call-arg]

    def test_venice_parameters_enable_e2ee_falsey_does_not_warn(self):
        """No warning when E2EE is not requested (False or unset)."""
        import warnings as _w

        with _w.catch_warnings():
            _w.simplefilter("error")
            VeniceParameters(enable_e2ee=False)  # type: ignore[call-arg]
            VeniceParameters()  # type: ignore[call-arg]

    def test_web_search_citation_creation(self):
        """Test WebSearchCitation model creation."""
        citation = WebSearchCitation(
            title="Test Article",
            url="https://example.com/article",
            content="This is an excerpt from the article.",
            date="2024-01-01",
        )

        assert citation.title == "Test Article"
        assert citation.url == "https://example.com/article"
        assert citation.content == "This is an excerpt from the article."
        assert citation.date == "2024-01-01"


class TestDeprecationInfoProperty:
    """Test DeprecationInfo.is_deprecated property."""

    def test_is_deprecated_with_warning(self):
        """Test is_deprecated when warning exists."""
        info = DeprecationInfo(warning="This model will be deprecated", date=None)

        assert info.is_deprecated is True

    def test_is_deprecated_with_date(self):
        """Test is_deprecated when date exists."""
        info = DeprecationInfo(warning=None, date=datetime(2024, 12, 31, tzinfo=UTC))

        assert info.is_deprecated is True

    def test_is_deprecated_with_both(self):
        """Test is_deprecated when both warning and date exist."""
        info = DeprecationInfo(warning="Deprecated", date=datetime(2024, 12, 31, tzinfo=UTC))

        assert info.is_deprecated is True

    def test_is_deprecated_with_neither(self):
        """Test is_deprecated when neither warning nor date exist."""
        info = DeprecationInfo(warning=None, date=None)

        assert info.is_deprecated is False


class TestComplexHeaderParsing:
    """Test complex header parsing scenarios."""

    def test_rate_limits_with_mixed_valid_invalid_headers(self):
        """Test rate limit parsing with mix of valid and invalid header values."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "invalid",  # Will be None
            "x-ratelimit-reset-requests": "1704067200",
            "x-ratelimit-limit-tokens": "not-a-number",  # Will be None
            "x-ratelimit-remaining-tokens": "48000",
            "x-ratelimit-reset-tokens": "bad-timestamp",  # Will be None
        }
        model._response = mock_response

        result = model.response_rate_limits

        assert result is not None
        assert result.limit_requests == 100
        assert result.remaining_requests is None  # Invalid parse
        assert result.limit_tokens is None  # Invalid parse
        assert result.remaining_tokens == 48000

    def test_venice_version_property(self):
        """Test venice_version property."""
        model = VeniceBaseModel()

        # With version header
        mock_response = Mock()
        mock_response.headers = {"x-venice-version": "1.0.0"}
        model._response = mock_response

        assert model.venice_version == "1.0.0"

        # Without version header
        mock_response.headers = {}
        assert model.venice_version is None

        # Without response
        model._response = None
        assert model.venice_version is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_pagination_info_accepts_any_server_values(self):
        """PaginationInfo is parsed from response headers, so it keeps whatever
        the server sends rather than raising on a value outside the documented range."""
        pagination = PaginationInfo(page=1, limit=50, total=100, total_pages=2)
        assert pagination.page == 1

        unusual = PaginationInfo(page=0, limit=1000, total=-1, total_pages=0)
        assert (unusual.page, unusual.limit, unusual.total) == (0, 1000, -1)

    def test_model_with_response_attached(self):
        """Test that model can store and access raw response."""
        model = VeniceBaseModel()

        mock_response = Mock()
        mock_response.headers = {
            "x-test": "value",
            "x-ratelimit-remaining-requests": "50",
            "x-venice-balance-usd": "100.00",
        }
        model._response = mock_response

        # All properties should work
        assert model.headers is not None
        assert model.response_rate_limits is not None
        assert model.balance_info is not None
        assert model.venice_version is None  # No version header


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
