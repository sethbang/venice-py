"""
VCR-based integration tests for Venice AI Billing Resource.

This test suite uses VCRpy to record and replay real API responses for the
billing usage-history endpoint. All sensitive data (billing amounts, account
balances) is automatically sanitized by the before_record_response hook in
conftest.py.

Tests cover:
- JSON and CSV response formats
- Filter parameters (currency, timestamp range, page size)
- Cursor-paginated walking via iter_usage_history
- Error handling scenarios
"""

import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.exceptions import BillingTimeoutError
from venice_ai.types.api.billing import BillingUsageHistoryCsvPage
from venice_ai.types.enums import BillingFormatEnum


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for billing integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    yield client
    await client.close()


pytestmark = [pytest.mark.integration, pytest.mark.vcr]


@pytest.mark.integration
async def test_billing_get_usage_history_json_default(vcr_cassette, venice_client):
    """Basic usage-history retrieval in JSON format (default)."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history()

    # Verify response structure: data list + cursor (never pagination metadata).
    assert hasattr(page, "data")
    assert isinstance(page.data, list)
    assert hasattr(page, "nextCursor")


@pytest.mark.integration
async def test_billing_get_usage_history_json_structure(vcr_cassette, venice_client):
    """Usage-history entries expose the documented fields."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history(format=BillingFormatEnum.JSON)

    assert isinstance(page.data, list)
    if page.data:
        record = page.data[0]
        assert hasattr(record, "sku")
        assert hasattr(record, "timestamp")
        assert hasattr(record, "currency")
        assert hasattr(record, "amount")


@pytest.mark.integration
async def test_billing_get_usage_history_csv_format(vcr_cassette, venice_client):
    """Usage-history retrieval in CSV format returns a typed CSV page."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history(format=BillingFormatEnum.CSV)

    assert isinstance(page, BillingUsageHistoryCsvPage)
    csv_text = page.text
    assert "timestamp" in csv_text.lower() or "sku" in csv_text.lower()


@pytest.mark.integration
async def test_billing_get_usage_history_with_currency_filter(vcr_cassette, venice_client):
    """Usage-history retrieval filtered by currency."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history(currency="USD")

    assert hasattr(page, "data")
    if page.data:
        for record in page.data:
            assert record.currency == "USD"


@pytest.mark.integration
async def test_billing_get_usage_history_with_date_range(vcr_cassette, venice_client):
    """Usage-history retrieval with a timestamp range filter."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history(
            startTimestamp="2025-05-05T00:00:00Z", endTimestamp="2025-06-06T23:59:59Z"
        )

    assert hasattr(page, "data")
    assert isinstance(page.data, list)


@pytest.mark.integration
async def test_billing_get_usage_history_with_page_size(vcr_cassette, venice_client):
    """Usage-history retrieval honours the pageSize cap."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history(pageSize=10)

    assert isinstance(page.data, list)
    # Entries in ascending timestamp order.
    if len(page.data) >= 2:
        for i in range(len(page.data) - 1):
            assert page.data[i].timestamp <= page.data[i + 1].timestamp


@pytest.mark.integration
async def test_billing_iter_usage_history_walks_pages(vcr_cassette, venice_client):
    """iter_usage_history yields entries across the cursor-paginated walk."""
    with vcr_cassette:
        collected = []
        async for entry in venice_client.billing.iter_usage_history(page_size=10, max_items=15):
            collected.append(entry)

    assert len(collected) <= 15
    for entry in collected:
        assert hasattr(entry, "sku")
        assert hasattr(entry, "timestamp")


@pytest.mark.integration
async def test_billing_get_usage_history_small_range(vcr_cassette, venice_client):
    """A narrow range either succeeds or fails fast with BillingTimeoutError.

    Billing endpoints can be slow; the SDK wraps requests in a 10-second
    timeout and raises BillingTimeoutError with actionable guidance rather
    than hanging.
    """
    with vcr_cassette:
        try:
            page = await venice_client.billing.get_usage_history(
                startTimestamp="2025-05-05T00:00:00Z", endTimestamp="2025-05-05T00:01:00Z"
            )
            assert hasattr(page, "data")
            assert isinstance(page.data, list)
        except BillingTimeoutError:
            # Expected: the SDK correctly fails fast on a slow/empty query.
            pass


@pytest.mark.integration
async def test_billing_response_structure_validation(vcr_cassette, venice_client):
    """The usage-history response matches the documented schema."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history()

    # Top-level structure: data + nextCursor, no pagination envelope.
    assert hasattr(page, "data")
    assert hasattr(page, "nextCursor")
    assert not hasattr(page, "pagination")

    if page.data:
        record = page.data[0]
        required_fields = [
            "sku",
            "timestamp",
            "currency",
            "amount",
            "units",
            "pricePerUnitUsd",
        ]
        for field in required_fields:
            assert hasattr(record, field), f"Missing required field: {field}"


@pytest.mark.integration
async def test_billing_sanitization_check(vcr_cassette, venice_client):
    """Billing amounts are sanitized (not real production values)."""
    with vcr_cassette:
        page = await venice_client.billing.get_usage_history()

    if page.data:
        for record in page.data:
            # Sanitized amounts should be small (< 100).
            assert abs(record.amount) < 100, "Amounts should be sanitized to small values"
            assert isinstance(record.amount, (int, float)), "Amount should be numeric"
