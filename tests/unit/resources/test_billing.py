"""
Comprehensive tests for src/venice_ai/resources/billing.py module.

This test file focuses on achieving >80% coverage for billing operations,
testing the cursor-paginated usage-history endpoint, its continuation
semantics, and the beta usage-analytics endpoint.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    PermissionDeniedError,
    RateLimitError,
)
from venice_ai.resources.billing import Billing
from venice_ai.types.api.billing import BillingUsageHistoryResponse, UsageAnalyticsResponse
from venice_ai.types.enums import BillingFormatEnum


class MockVeniceClient:
    """Mock client for testing Billing resource."""

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self._request = AsyncMock()


@pytest.fixture
def mock_client():
    """Create a mock Venice client for testing."""
    return MockVeniceClient()


@pytest.fixture
def billing_resource(mock_client):
    """Create a Billing resource instance for testing."""
    return Billing(mock_client)


@pytest.fixture
def sample_usage_history_response():
    """Sample usage-history response (single page, last page)."""
    return {
        "data": [
            {
                "sku": "llama-3.2-3b-inference",
                "timestamp": "2025-01-15T12:00:00Z",
                "currency": "USD",
                "amount": 0.001,
                "units": 100.0,
                "pricePerUnitUsd": 0.00001,
                "notes": "LLM inference request",
                "inferenceDetails": {
                    "requestId": "req_123",
                    "promptTokens": 50.0,
                    "completionTokens": 50.0,
                    "inferenceExecutionTime": 120.5,
                },
            },
            {
                "sku": "text-embedding-bge-m3-inference",
                "timestamp": "2025-01-15T11:30:00Z",
                "currency": "DIEM",
                "amount": -0.01,
                "units": 50.0,
                "pricePerUnitUsd": 0.0002,
                "notes": "Embedding generation request",
                "inferenceDetails": None,
            },
        ],
        "nextCursor": None,
    }


class TestGetUsageHistory:
    """Test get_usage_history() request construction and response parsing."""

    @pytest.mark.asyncio
    async def test_json_format_default(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        """First-page JSON request sends no cursor and returns a parsed model."""
        mock_client._request.return_value = sample_usage_history_response

        result = await billing_resource.get_usage_history()

        assert isinstance(result, BillingUsageHistoryResponse)
        assert result.model_dump() == sample_usage_history_response
        mock_client._request.assert_called_once_with(
            "GET",
            "billing/usage-history",
            params={},
            headers={"accept": "application/json"},
            raw_response=False,
        )

    @pytest.mark.asyncio
    async def test_all_first_page_filters(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        """All first-page filters are forwarded as query parameters."""
        mock_client._request.return_value = sample_usage_history_response

        await billing_resource.get_usage_history(
            currency="USD",
            startTimestamp="2025-01-01T00:00:00Z",
            endTimestamp="2025-01-31T23:59:59Z",
            pageSize=500,
        )

        call_args = mock_client._request.call_args
        assert call_args[1]["params"] == {
            "currency": "USD",
            "startTimestamp": "2025-01-01T00:00:00Z",
            "endTimestamp": "2025-01-31T23:59:59Z",
            "pageSize": 500,
        }

    @pytest.mark.asyncio
    async def test_none_filters_excluded(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        """None-valued filters are dropped from the query string."""
        mock_client._request.return_value = sample_usage_history_response

        await billing_resource.get_usage_history(
            currency="DIEM",
            startTimestamp=None,
            endTimestamp="2025-01-31T23:59:59Z",
            pageSize=None,
        )

        params = mock_client._request.call_args[1]["params"]
        assert params == {"currency": "DIEM", "endTimestamp": "2025-01-31T23:59:59Z"}
        assert "startTimestamp" not in params
        assert "pageSize" not in params
        assert "cursor" not in params

    @pytest.mark.asyncio
    async def test_cursor_only_continuation(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        """A continuation request sends the cursor and nothing else."""
        mock_client._request.return_value = sample_usage_history_response

        await billing_resource.get_usage_history(cursor="TOKEN_ABC")

        call_args = mock_client._request.call_args
        assert call_args[0][1] == "billing/usage-history"
        assert call_args[1]["params"] == {"cursor": "TOKEN_ABC"}


class TestUsageHistoryCursorGuard:
    """Test the client-side guard against mixing a cursor with filters."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"currency": "USD"},
            {"startTimestamp": "2025-01-01T00:00:00Z"},
            {"endTimestamp": "2025-01-31T23:59:59Z"},
            {"pageSize": 100},
        ],
    )
    async def test_cursor_with_filter_raises(self, billing_resource, mock_client, kwargs):
        """Passing a cursor alongside any filter raises before any request."""
        with pytest.raises(ValueError, match="only 'cursor'"):
            await billing_resource.get_usage_history(cursor="TOKEN", **kwargs)

        mock_client._request.assert_not_called()


class TestUsageHistoryValidation:
    """Test timestamp validation on the first page."""

    @pytest.mark.asyncio
    async def test_end_before_start_raises(self, billing_resource, mock_client):
        """endTimestamp earlier than startTimestamp is rejected."""
        with pytest.raises(ValueError, match="cannot be before"):
            await billing_resource.get_usage_history(
                startTimestamp="2025-02-01T00:00:00Z",
                endTimestamp="2025-01-01T00:00:00Z",
            )

        mock_client._request.assert_not_called()

    @pytest.mark.asyncio
    async def test_page_size_below_minimum_raises(self, billing_resource, mock_client):
        """pageSize below the server minimum (10) fails Pydantic validation."""
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            await billing_resource.get_usage_history(pageSize=5)


class TestIterUsageHistory:
    """Test the cursor-paginated iter_usage_history() walk."""

    @pytest.mark.asyncio
    async def test_single_page(self, billing_resource, mock_client, sample_usage_history_response):
        """A single-page walk (nextCursor=None) yields every entry once."""
        mock_client._request.return_value = sample_usage_history_response

        items = [entry async for entry in billing_resource.iter_usage_history()]

        assert [e.sku for e in items] == [
            "llama-3.2-3b-inference",
            "text-embedding-bge-m3-inference",
        ]
        # Only one request — no nextCursor means the walk stops.
        assert mock_client._request.call_count == 1

    @pytest.mark.asyncio
    async def test_cursor_continuation_sends_cursor_only(self, billing_resource, mock_client):
        """The second page sends ONLY the cursor; filters live inside it.

        This is the regression guard: the endpoint 400s a continuation that
        carries filter parameters, and a mocked test would otherwise pass while
        every real page after the first failed.
        """
        page1 = {
            "data": [
                {
                    "sku": "sku-a",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "currency": "USD",
                    "amount": 0.1,
                    "units": 1.0,
                    "pricePerUnitUsd": 0.1,
                    "notes": "",
                    "inferenceDetails": None,
                }
            ],
            "nextCursor": "CURSOR_2",
        }
        page2 = {
            "data": [
                {
                    "sku": "sku-b",
                    "timestamp": "2025-01-02T00:00:00Z",
                    "currency": "USD",
                    "amount": 0.2,
                    "units": 2.0,
                    "pricePerUnitUsd": 0.1,
                    "notes": "",
                    "inferenceDetails": None,
                }
            ],
            "nextCursor": None,
        }
        mock_client._request.side_effect = [page1, page2]

        items = [
            entry
            async for entry in billing_resource.iter_usage_history(currency="USD", page_size=100)
        ]

        assert [e.sku for e in items] == ["sku-a", "sku-b"]
        assert mock_client._request.call_count == 2

        # First page carries the filters and the page size, no cursor.
        first_call = mock_client._request.call_args_list[0]
        assert first_call[0][1] == "billing/usage-history"
        assert first_call[1]["params"] == {"currency": "USD", "pageSize": 100}

        # Second page carries ONLY the cursor.
        second_call = mock_client._request.call_args_list[1]
        assert second_call[1]["params"] == {"cursor": "CURSOR_2"}

    @pytest.mark.asyncio
    async def test_max_items_caps_walk(self, billing_resource, mock_client):
        """max_items stops iteration even when more pages are available."""
        page1 = {
            "data": [
                {
                    "sku": f"sku-{i}",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "currency": "USD",
                    "amount": 0.1,
                    "units": 1.0,
                    "pricePerUnitUsd": 0.1,
                    "notes": "",
                    "inferenceDetails": None,
                }
                for i in range(3)
            ],
            "nextCursor": "CURSOR_2",
        }
        mock_client._request.return_value = page1

        items = [entry async for entry in billing_resource.iter_usage_history(max_items=2)]

        assert len(items) == 2
        # Capped inside the first page; no continuation request was made.
        assert mock_client._request.call_count == 1

    @pytest.mark.asyncio
    async def test_reiteration_restarts_walk(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        """Iterating the same Paginator again restarts from the first page."""
        mock_client._request.return_value = sample_usage_history_response

        paginator = billing_resource.iter_usage_history(currency="USD")

        first = [e async for e in paginator]
        mock_client._request.reset_mock()
        second = [e async for e in paginator]

        assert len(first) == len(second) == 2
        # The restart sends the filters again (not a stale cursor).
        assert mock_client._request.call_args[1]["params"] == {
            "currency": "USD",
            "pageSize": 100,
        }


class TestUsageHistoryErrorHandling:
    """Test error propagation from the underlying request."""

    @pytest.mark.asyncio
    async def test_authentication_error(self, billing_resource, mock_client):
        mock_client._request.side_effect = AuthenticationError(
            "Invalid API key", response=MagicMock()
        )
        with pytest.raises(AuthenticationError):
            await billing_resource.get_usage_history()

    @pytest.mark.asyncio
    async def test_permission_denied_error(self, billing_resource, mock_client):
        mock_client._request.side_effect = PermissionDeniedError(
            "Billing access denied", response=MagicMock()
        )
        with pytest.raises(PermissionDeniedError):
            await billing_resource.get_usage_history()

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, billing_resource, mock_client):
        mock_client._request.side_effect = RateLimitError(
            "Rate limit exceeded", response=MagicMock()
        )
        with pytest.raises(RateLimitError):
            await billing_resource.get_usage_history()

    @pytest.mark.asyncio
    async def test_invalid_request_error(self, billing_resource, mock_client):
        """A rejected cursor surfaces as InvalidRequestError from the server."""
        mock_client._request.side_effect = InvalidRequestError(
            "Invalid cursor", response=MagicMock()
        )
        with pytest.raises(InvalidRequestError):
            await billing_resource.get_usage_history(cursor="expired")

    @pytest.mark.asyncio
    async def test_generic_api_error(self, billing_resource, mock_client):
        mock_client._request.side_effect = APIError("Server error", response=MagicMock())
        with pytest.raises(APIError):
            await billing_resource.get_usage_history()


class TestUsageHistoryResponseCasting:
    """Test JSON response handling (CSV pages: tests/unit/resources/test_billing_csv.py)."""

    @pytest.mark.asyncio
    async def test_json_cast_to_response_model(
        self, billing_resource, mock_client, sample_usage_history_response
    ):
        mock_client._request.return_value = sample_usage_history_response

        result = await billing_resource.get_usage_history(format=BillingFormatEnum.JSON)

        assert isinstance(result, BillingUsageHistoryResponse)
        assert result.nextCursor is None
        assert len(result.data) == 2

    @pytest.mark.asyncio
    async def test_empty_page(self, billing_resource, mock_client):
        """An empty last page parses cleanly."""
        mock_client._request.return_value = {"data": [], "nextCursor": None}

        result = await billing_resource.get_usage_history()

        assert isinstance(result, BillingUsageHistoryResponse)
        assert result.data == []
        assert result.nextCursor is None

    @pytest.mark.asyncio
    async def test_legacy_currency_round_trips(self, billing_resource, mock_client):
        """A historical row with a retired currency (e.g. VCU) must not crash the page.

        usage-history returns historical rows; the spec enum only constrains new rows.
        A too-narrow Literal would raise ValidationError and kill the entire walk.
        """
        mock_client._request.return_value = {
            "data": [
                {
                    "sku": "legacy-sku",
                    "timestamp": "2024-06-01T00:00:00Z",
                    "currency": "VCU",
                    "amount": -0.5,
                    "units": 5.0,
                    "pricePerUnitUsd": 0.1,
                    "notes": "legacy",
                    "inferenceDetails": None,
                }
            ],
            "nextCursor": None,
        }

        result = await billing_resource.get_usage_history()

        assert isinstance(result, BillingUsageHistoryResponse)
        assert result.data[0].currency == "VCU"


# ============================================================================
# Usage Analytics (Beta) Tests
# ============================================================================


@pytest.fixture
def sample_usage_analytics_response():
    """Sample usage analytics response matching the API schema."""
    return {
        "lookback": "7d",
        "byDate": [
            {"date": "2025-01-14", "USD": 0.05, "DIEM": 1.25},
            {"date": "2025-01-15", "USD": 0.12, "DIEM": 3.00},
        ],
        "byModel": [
            {
                "modelName": "Llama 3.3 70B",
                "unitType": "tokens",
                "modelType": "LLM",
                "totalUsd": 0.10,
                "totalDiem": 2.50,
                "totalUnits": 5000.0,
                "breakdown": [
                    {"type": "Input", "usd": 0.04, "diem": 1.00, "units": 3000.0},
                    {"type": "Output", "usd": 0.06, "diem": 1.50, "units": 2000.0},
                ],
            },
            {
                "modelName": "FLUX Pro",
                "unitType": "images",
                "modelType": "IMAGE",
                "totalUsd": 0.07,
                "totalDiem": 1.75,
                "totalUnits": 2.0,
                "breakdown": None,
            },
        ],
        "byModelDaily": [
            {"date": "2025-01-14T00:00:00Z", "Llama 3.3 70B": 1.0, "FLUX Pro": 0.5},
            {"date": "2025-01-15T00:00:00Z", "Llama 3.3 70B": 1.5, "FLUX Pro": 1.25},
        ],
        "topModels": ["Llama 3.3 70B", "FLUX Pro"],
        "byKey": [
            {
                "apiKeyId": "key_abc123",
                "description": "Production Key",
                "totalUsd": 0.15,
                "totalDiem": 3.75,
                "totalUnits": 4500.0,
            },
            {
                "apiKeyId": None,
                "description": "Web App",
                "totalUsd": 0.02,
                "totalDiem": 0.50,
                "totalUnits": 500.0,
            },
        ],
        "byKeyDaily": [
            {"date": "2025-01-14T00:00:00Z", "Production Key": 1.0, "Web App": 0.25},
            {"date": "2025-01-15T00:00:00Z", "Production Key": 2.75, "Web App": 0.25},
        ],
        "topKeyNames": ["Production Key", "Web App"],
    }


class TestBillingGetUsageAnalytics:
    """Test get_usage_analytics() method (Beta endpoint)."""

    @pytest.mark.asyncio
    async def test_get_usage_analytics_default(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test getting usage analytics with default parameters (7d lookback)."""
        mock_client._request.return_value = sample_usage_analytics_response

        with pytest.warns(FutureWarning, match="beta"):
            result = await billing_resource.get_usage_analytics()

        assert isinstance(result, UsageAnalyticsResponse)
        assert result.lookback == "7d"
        assert len(result.byDate) == 2
        assert len(result.byModel) == 2
        assert len(result.byKey) == 2
        assert result.topModels == ["Llama 3.3 70B", "FLUX Pro"]

        mock_client._request.assert_called_once_with(
            "GET",
            "billing/usage-analytics",
            params={},
            headers={"accept": "application/json"},
        )

    @pytest.mark.asyncio
    async def test_get_usage_analytics_with_lookback(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test getting usage analytics with a lookback period."""
        mock_client._request.return_value = sample_usage_analytics_response

        with pytest.warns(FutureWarning, match="beta"):
            result = await billing_resource.get_usage_analytics(lookback="30d")

        assert isinstance(result, UsageAnalyticsResponse)

        call_args = mock_client._request.call_args
        assert call_args[1]["params"] == {"lookback": "30d"}

    @pytest.mark.asyncio
    async def test_get_usage_analytics_with_date_range(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test getting usage analytics with a custom date range."""
        mock_client._request.return_value = sample_usage_analytics_response

        with pytest.warns(FutureWarning, match="beta"):
            result = await billing_resource.get_usage_analytics(
                startDate="2025-01-01",
                endDate="2025-01-31",
            )

        assert isinstance(result, UsageAnalyticsResponse)

        call_args = mock_client._request.call_args
        assert call_args[1]["params"] == {
            "startDate": "2025-01-01",
            "endDate": "2025-01-31",
        }

    @pytest.mark.asyncio
    async def test_get_usage_analytics_response_structure(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test that the response is properly validated into Pydantic models."""
        mock_client._request.return_value = sample_usage_analytics_response

        with pytest.warns(FutureWarning, match="beta"):
            result = await billing_resource.get_usage_analytics()

        # Verify byDate structure
        assert result.byDate[0].date == "2025-01-14"
        assert result.byDate[0].USD == 0.05
        assert result.byDate[0].DIEM == 1.25

        # Verify byModel structure
        model = result.byModel[0]
        assert model.modelName == "Llama 3.3 70B"
        assert model.unitType == "tokens"
        assert model.modelType == "LLM"
        assert model.totalUsd == 0.10
        assert model.breakdown is not None
        assert len(model.breakdown) == 2
        assert model.breakdown[0].type == "Input"

        # Verify byKey structure
        key = result.byKey[0]
        assert key.apiKeyId == "key_abc123"
        assert key.description == "Production Key"

        # Verify null apiKeyId for web app
        web_key = result.byKey[1]
        assert web_key.apiKeyId is None
        assert web_key.description == "Web App"


class TestBillingUsageAnalyticsValidation:
    """Test parameter validation for get_usage_analytics()."""

    @pytest.mark.asyncio
    async def test_lookback_and_dates_mutually_exclusive(self, billing_resource):
        """Test that lookback and date range cannot be combined."""
        with (
            pytest.raises(ValueError, match="Cannot specify both"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(
                lookback="7d",
                startDate="2025-01-01",
                endDate="2025-01-31",
            )

    @pytest.mark.asyncio
    async def test_start_date_without_end_date(self, billing_resource):
        """Test that startDate requires endDate."""
        with (
            pytest.raises(ValueError, match="Both 'startDate' and 'endDate' are required"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(startDate="2025-01-01")

    @pytest.mark.asyncio
    async def test_end_date_without_start_date(self, billing_resource):
        """Test that endDate requires startDate."""
        with (
            pytest.raises(ValueError, match="Both 'startDate' and 'endDate' are required"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(endDate="2025-01-31")

    @pytest.mark.asyncio
    async def test_invalid_lookback_format(self, billing_resource):
        """Test that invalid lookback format is rejected."""
        with (
            pytest.raises(ValueError, match="Invalid lookback format"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(lookback="7days")

    @pytest.mark.asyncio
    async def test_lookback_exceeds_max(self, billing_resource):
        """Test that lookback > 90d is rejected."""
        with (
            pytest.raises(ValueError, match="between 1d and 90d"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(lookback="91d")

    @pytest.mark.asyncio
    async def test_lookback_zero_days(self, billing_resource):
        """Test that lookback of 0d is rejected (fails the ^[1-9]\\d*d$ format)."""
        with (
            pytest.raises(ValueError, match="Invalid lookback format"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(lookback="0d")

    @pytest.mark.asyncio
    async def test_date_range_end_before_start(self, billing_resource):
        """Test that end date before start date is rejected."""
        with (
            pytest.raises(ValueError, match="cannot be before"),
            pytest.warns(FutureWarning, match="beta"),
        ):
            await billing_resource.get_usage_analytics(
                startDate="2025-01-31",
                endDate="2025-01-01",
            )

    @pytest.mark.asyncio
    async def test_valid_lookback_values(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test various valid lookback values."""
        mock_client._request.return_value = sample_usage_analytics_response

        for lookback in ["1d", "7d", "30d", "60d", "90d"]:
            with pytest.warns(FutureWarning, match="beta"):
                result = await billing_resource.get_usage_analytics(lookback=lookback)
            assert isinstance(result, UsageAnalyticsResponse)
            mock_client._request.reset_mock()

    @pytest.mark.asyncio
    async def test_beta_warning_emitted(
        self, billing_resource, mock_client, sample_usage_analytics_response
    ):
        """Test that a FutureWarning is emitted for the beta endpoint."""
        mock_client._request.return_value = sample_usage_analytics_response

        with pytest.warns(FutureWarning, match="beta API endpoint"):
            await billing_resource.get_usage_analytics()
