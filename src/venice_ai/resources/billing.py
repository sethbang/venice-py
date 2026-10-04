"""
Venice AI Billing and Usage Analytics Resource Module.

This module provides comprehensive billing and usage tracking functionality for Venice AI
services. It enables users to retrieve detailed usage analytics, cost breakdowns, and
billing information in multiple formats to support various integration and reporting needs.

The module supports both structured JSON responses for programmatic integration and
CSV exports for data analysis, reporting, and external system integration. All operations
are designed to work seamlessly with asynchronous workflows and provide flexible
filtering options for precise usage analysis.

Key Features:
    - Detailed usage analytics with flexible filtering
    - Multi-format support (JSON and CSV)
    - Comprehensive cost tracking and billing summaries
    - Aggregated usage analytics with breakdowns by date, model, and API key (Beta)
    - Pagination support for large datasets
    - Real-time usage monitoring and reporting

Classes:
    Billing: Asynchronous resource for billing and usage data operations
"""

import asyncio
import re
import warnings
from email.message import Message
from typing import TYPE_CHECKING, Literal, cast, overload

import aiohttp

from .._date_validation import validate_date_range, validate_date_string
from .._pagination import DEFAULT_PAGE_SIZE, Paginator, _PageResult
from .._resource import APIResource
from ..exceptions import APITimeoutError, BillingTimeoutError
from ..types.api import BillingUsageHistoryQueryParams
from ..types.api.billing import (
    BillingBalanceResponse,
    BillingUsageEntry,
    BillingUsageHistoryCsvPage,
    BillingUsageHistoryResponse,
    UsageAnalyticsQueryParams,
    UsageAnalyticsResponse,
)
from ..types.enums import BillingFormatEnum
from ..utils.errors import read_body

# Aggressive timeout for billing API requests.
# The Venice billing API is known to hang on small date ranges or empty results.
# Using a shorter timeout to fail fast and provide actionable guidance.
BILLING_REQUEST_TIMEOUT_SECONDS = 10

# The usage-history endpoint's documented default ``pageSize``, used when a
# CSV export leaves the page size to the server.
USAGE_HISTORY_DEFAULT_PAGE_SIZE = 1000

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


class Billing(APIResource["VeniceClient"]):
    """
    Asynchronous resource for comprehensive billing and usage analytics operations.

    This class provides a complete interface for retrieving and analyzing Venice AI usage
    data, billing information, and cost analytics. It supports multiple output formats
    and flexible filtering options to accommodate various reporting and integration needs.

    The class automatically handles format-specific request headers, response parsing,
    and data type conversions based on the requested output format. All operations are
    fully asynchronous and integrate seamlessly with async/await patterns.

    Key Capabilities:
        - Usage data retrieval with flexible date range filtering
        - Multiple output formats (JSON for APIs, CSV for analytics)
        - Pagination support for large datasets
        - Currency-specific cost reporting
        - Detailed usage breakdowns by model and service type

    Args:
        client: The Venice AI client instance for making authenticated API requests.

    Example:
        Basic usage analytics:

        .. code-block:: python

            async with VeniceClient() as client:
                # Walk recent usage entries lazily (cursor-paginated)
                async for entry in client.billing.iter_usage_history(
                    startTimestamp="2025-01-01T00:00:00Z",
                    endTimestamp="2025-02-01T00:00:00Z",
                ):
                    print(entry.timestamp, entry.amount)

                # Or export the same window as CSV, one document per page,
                # named by its position in the walk
                index = 0
                async for csv_page in client.billing.iter_usage_history_csv(
                    startTimestamp="2025-01-01T00:00:00Z",
                ):
                    index += 1
                    Path(f"usage-{index:04d}.csv").write_bytes(csv_page.content)
    """

    @overload
    async def get_usage_history(
        self,
        *,
        format: Literal[BillingFormatEnum.JSON] = ...,
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
        pageSize: int | None = None,
        cursor: str | None = None,
    ) -> BillingUsageHistoryResponse: ...

    @overload
    async def get_usage_history(
        self,
        *,
        format: Literal[BillingFormatEnum.CSV],
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
        pageSize: int | None = None,
        cursor: str | None = None,
    ) -> BillingUsageHistoryCsvPage: ...

    @overload
    async def get_usage_history(
        self,
        *,
        format: BillingFormatEnum,
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
        pageSize: int | None = None,
        cursor: str | None = None,
    ) -> BillingUsageHistoryResponse | BillingUsageHistoryCsvPage: ...

    async def get_usage_history(
        self,
        *,
        format: BillingFormatEnum = BillingFormatEnum.JSON,
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
        pageSize: int | None = None,
        cursor: str | None = None,
    ) -> BillingUsageHistoryResponse | BillingUsageHistoryCsvPage:
        """Fetch one page of billing usage history (GET /billing/usage-history).

        The endpoint is a cursor-paginated walk in ascending timestamp order. The
        first request of a walk takes the filter parameters
        (``currency``/``startTimestamp``/``endTimestamp``/``pageSize``); each
        page carries a ``nextCursor`` token. A continuation request passes that
        token as ``cursor`` **and nothing else** — the filters travel inside the
        cursor, and the server rejects a request that sends filters alongside one.
        Use :meth:`iter_usage_history` (JSON entries) or
        :meth:`iter_usage_history_csv` (CSV pages) to walk every page
        automatically.

        The whole exchange -- the response and, for CSV, its body -- must finish
        within ``BILLING_REQUEST_TIMEOUT_SECONDS`` (10 seconds).

        Args:
            format: ``JSON`` (default, structured) or ``CSV`` (a CSV document
                per page).
            currency: Filter by consumable currency (``"USD"``, ``"DIEM"``,
                ``"BUNDLED_CREDITS"``). First page only.
            startTimestamp: Inclusive lower bound on entry timestamps (ISO 8601 UTC).
                First page only.
            endTimestamp: Exclusive upper bound on entry timestamps (ISO 8601 UTC).
                First page only.
            pageSize: Entries per page (10-1000, server default 1000). First page only.
            cursor: Continuation token from a prior response's ``nextCursor``. When
                given, no filter parameter may be supplied.

        Returns:
            For JSON format: a :class:`BillingUsageHistoryResponse` (``data`` plus
            ``nextCursor``). For CSV format: a :class:`BillingUsageHistoryCsvPage`
            with the CSV document in ``content``, the continuation token the
            server sends in the ``x-next-cursor`` header as ``nextCursor``, and
            the server-assigned ``filename``.

        Raises:
            ValueError: If ``cursor`` is combined with any filter parameter.
            BillingTimeoutError: If the response, or a CSV body, has not
                arrived within the 10-second billing deadline.
            InvalidRequestError: If a parameter is invalid, or the cursor was
                rejected (expired, tampered with, or issued to another user) —
                restart the walk from the first page.
            AuthenticationError: If the API key is invalid or expired.
            APIError: For other API-related errors or service issues.

        Example:
            .. code-block:: python

               from venice_ai import VeniceClient

               async def walk_usage():
                   async with VeniceClient() as client:
                       page = await client.billing.get_usage_history(
                           startTimestamp="2025-01-01T00:00:00Z",
                           currency="USD",
                       )
                       for entry in page.data:
                           print(entry.timestamp, entry.amount)

                       # Fetch the next page — cursor only.
                       if page.nextCursor:
                           page = await client.billing.get_usage_history(
                               cursor=page.nextCursor
                           )
        """
        # A continuation request carries only the cursor; the filters live inside it.
        if cursor is not None and any(
            p is not None for p in (currency, startTimestamp, endTimestamp, pageSize)
        ):
            raise ValueError(
                "A continuation request must send only 'cursor'. Filter parameters "
                "(currency, startTimestamp, endTimestamp, pageSize) travel inside the "
                "cursor and are rejected when sent alongside it."
            )

        # Validate timestamp formats and ordering on the first page.
        if startTimestamp is not None:
            validate_date_string(startTimestamp, "startTimestamp")
        if endTimestamp is not None:
            validate_date_string(endTimestamp, "endTimestamp")
        if startTimestamp is not None and endTimestamp is not None:
            validate_date_range(
                start_date=startTimestamp,
                end_date=endTimestamp,
                start_param="startTimestamp",
                end_param="endTimestamp",
            )

        # Create Pydantic query parameters model
        query_params = BillingUsageHistoryQueryParams(
            currency=currency,
            cursor=cursor,
            startTimestamp=startTimestamp,
            endTimestamp=endTimestamp,
            pageSize=pageSize,
        )

        # Convert to dictionary, excluding None values
        params = query_params.model_dump(exclude_none=True)

        csv = format == BillingFormatEnum.CSV
        headers = {"accept": "text/csv" if csv else "application/json"}

        async def _exchange() -> BillingUsageHistoryResponse | BillingUsageHistoryCsvPage:
            if not csv:
                result = await self._client._request(
                    "GET",
                    "billing/usage-history",
                    params=params,
                    headers=headers,
                    raw_response=False,
                )
                return BillingUsageHistoryResponse.model_validate(result)
            # raw_response=True hands back the unread aiohttp response.
            response = cast(
                aiohttp.ClientResponse,
                await self._client._request(
                    "GET",
                    "billing/usage-history",
                    params=params,
                    headers=headers,
                    raw_response=True,
                ),
            )
            try:
                content = await read_body(response)
            finally:
                response.release()
            return BillingUsageHistoryCsvPage(
                content=content,
                nextCursor=response.headers.get("x-next-cursor") or None,
                filename=_attachment_filename(response.headers.get("Content-Disposition")),
            )

        # Billing endpoints can hang on small ranges or empty results. One
        # deadline covers the response and the CSV body together, so a body
        # that stalls fails as fast as a response that never arrives.
        try:
            return await asyncio.wait_for(_exchange(), timeout=BILLING_REQUEST_TIMEOUT_SECONDS)
        except TimeoutError as e:
            raise BillingTimeoutError(original_error=e) from e
        except APITimeoutError as e:
            raise BillingTimeoutError(original_error=e.original_error) from e

    def iter_usage_history(
        self,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
    ) -> Paginator[BillingUsageEntry]:
        """Lazily walk every billing usage entry matching the filters.

        Wraps :meth:`get_usage_history` (JSON form only) as an async iterator over
        the cursor-paginated walk. The first page sends the filters; every
        subsequent page sends only the ``nextCursor`` token from the page before
        it, exactly as the endpoint requires.

        Each page carries its own 10-second deadline; a slow page raises
        :class:`~venice_ai.exceptions.BillingTimeoutError` mid-walk, and because
        the cursor lives inside the iterator there is no resume handle — restart
        the walk from the first page. The deadline wraps each page request on
        its own, so a longer range means more pages rather than more entries
        per page; ``page_size`` is what sets the size of a page. Lower it to
        shrink each page, and use a range of at least a day where data is
        known to exist, since the endpoint can hang on very short ranges or
        filters that match nothing.

        :param page_size: Entries per page (default 100; server range 10-1000).
        :param max_items: Optional cap on total items yielded.
        :param currency: Filter by consumable currency.
        :param startTimestamp: Inclusive lower bound (ISO 8601 UTC).
        :param endTimestamp: Exclusive upper bound (ISO 8601 UTC).

        Example::

            async for entry in client.billing.iter_usage_history(currency="USD"):
                print(entry.timestamp, entry.amount)
        """
        cursor: str | None = None

        async def _fetch_page(page_index: int) -> _PageResult[BillingUsageEntry]:
            nonlocal cursor
            if page_index == 0:
                # First page: send the filters and start the walk fresh.
                cursor = None
                response = await self.get_usage_history(
                    format=BillingFormatEnum.JSON,
                    currency=currency,
                    startTimestamp=startTimestamp,
                    endTimestamp=endTimestamp,
                    pageSize=page_size,
                )
            else:
                # Continuation: cursor only — the filters travel inside the cursor.
                response = await self.get_usage_history(
                    format=BillingFormatEnum.JSON,
                    cursor=cursor,
                )
            cursor = response.nextCursor
            items = list(response.data)
            has_more = response.nextCursor is not None
            return _PageResult(items=items, has_more=has_more)

        return Paginator(_fetch_page, page_size=page_size, max_items=max_items)

    def iter_usage_history_csv(
        self,
        *,
        page_size: int | None = None,
        max_pages: int | None = None,
        currency: str | None = None,
        startTimestamp: str | None = None,
        endTimestamp: str | None = None,
    ) -> Paginator[BillingUsageHistoryCsvPage]:
        """Lazily export every usage entry matching the filters as CSV pages.

        The CSV counterpart of :meth:`iter_usage_history`: it walks the same
        cursor-paginated endpoint with ``format=CSV`` and yields one
        :class:`~venice_ai.types.api.billing.BillingUsageHistoryCsvPage` per
        page. Each page is a complete CSV document with its own header row, so
        save pages as separate files, named by their position in the walk, or
        drop the header row of every page after the first when joining them.
        Nothing guarantees that the server-assigned ``page.filename`` values
        sort in walk order, so do not rely on them for ordering.

        Each page carries its own 10-second deadline; a slow page raises
        :class:`~venice_ai.exceptions.BillingTimeoutError` mid-walk, and the
        cursor lives inside the iterator, so restart from the first page. The
        deadline wraps each page request on its own, so a longer range means
        more pages rather than more entries per page; ``page_size`` is what
        sets the size of a page. Lower it to shrink each page, and use a range
        of at least a day where data is known to exist, since the endpoint can
        hang on very short ranges or filters that match nothing.

        :param page_size: Entries per page (server range 10-1000; the server
            default of 1000 when ``None``).
        :param max_pages: Optional cap on the number of pages yielded.
        :param currency: Filter by consumable currency.
        :param startTimestamp: Inclusive lower bound (ISO 8601 UTC).
        :param endTimestamp: Exclusive upper bound (ISO 8601 UTC).

        Example::

            index = 0
            async for page in client.billing.iter_usage_history_csv(
                startTimestamp="2026-06-01T00:00:00Z",
            ):
                index += 1
                Path(f"usage-{index:04d}.csv").write_bytes(page.content)
        """
        cursor: str | None = None

        async def _fetch_page(page_index: int) -> _PageResult[BillingUsageHistoryCsvPage]:
            nonlocal cursor
            if page_index == 0:
                cursor = None
                page = await self.get_usage_history(
                    format=BillingFormatEnum.CSV,
                    currency=currency,
                    startTimestamp=startTimestamp,
                    endTimestamp=endTimestamp,
                    pageSize=page_size,
                )
            else:
                page = await self.get_usage_history(format=BillingFormatEnum.CSV, cursor=cursor)
            cursor = page.nextCursor
            return _PageResult(items=[page], has_more=page.nextCursor is not None)

        return Paginator(
            _fetch_page,
            page_size=page_size or USAGE_HISTORY_DEFAULT_PAGE_SIZE,
            max_items=max_pages,
        )

    async def get_balance(self) -> BillingBalanceResponse:
        """Get current balance information (GET /billing/balance).

        Returns the authenticated user's remaining DIEM and USD balances
        along with the total DIEM epoch allocation.

        Returns:
            :class:`BillingBalanceResponse` with DIEM/USD balances.

        Raises:
            AuthenticationError: If the API key is invalid or expired.
            APIError: If the request fails or returns an error response.
            APIConnectionError: If unable to connect to the API.

        Example:
            .. code-block:: python

                balance = await client.billing.get_balance()
                print(f"Can consume: {balance.can_consume}")
                if balance.balances:
                    print(f"DIEM: {balance.balances.diem}")
                    print(f"USD:  {balance.balances.usd}")
        """
        return await self._client.get("billing/balance", cast_to=BillingBalanceResponse)

    async def get_usage_analytics(
        self,
        *,
        lookback: str | None = None,
        startDate: str | None = None,
        endDate: str | None = None,
    ) -> UsageAnalyticsResponse:
        """Get aggregated usage analytics with breakdowns by date, model, and API key.

        .. beta::
            This method wraps a **beta** API endpoint (``GET /billing/usage-analytics``).
            The request/response schema and behaviour may change without notice.

        Provides summary views of your API usage data, ideal for building dashboards
        and monitoring consumption. Data is cached server-side for 10 minutes.

        You can specify the time period using either:

        * **lookback**: A relative period like ``"7d"`` (7 days), ``"30d"`` (30 days),
          up to ``"90d"`` (90 days).
        * **startDate** and **endDate**: A custom date range in ``YYYY-MM-DD`` format.
          Both are required if either is provided.

        If no parameters are specified, the default lookback period is 7 days.

        Args:
            lookback: Relative lookback period (e.g., ``"7d"``, ``"30d"``, up to ``"90d"``).
                Cannot be combined with ``startDate``/``endDate``.
            startDate: Start date in ``YYYY-MM-DD`` format. Required if ``endDate`` is provided.
            endDate: End date in ``YYYY-MM-DD`` format. Required if ``startDate`` is provided.

        Returns:
            :class:`~venice_ai.types.api.billing.UsageAnalyticsResponse` containing
            aggregated analytics with ``byDate``, ``byModel``, ``byKey``, and chart data.

        Raises:
            InvalidRequestError: If parameter values are invalid (e.g., lookback > 90d,
                mismatched startDate/endDate, or combining lookback with date range).
            AuthenticationError: If the API key is invalid or expired.
            BillingTimeoutError: If the request times out (10s limit).
            APIError: For other API-related errors or service issues.

        Warning:
            This endpoint is currently in **beta** and may be unstable. Request/response
            schemas and behaviour may change without notice in future API versions.

        Example:
            .. code-block:: python

                from venice_ai import VeniceClient

                async def check_analytics():
                    async with VeniceClient() as client:
                        # Default: last 7 days
                        analytics = await client.billing.get_usage_analytics()

                        # Last 30 days
                        analytics = await client.billing.get_usage_analytics(lookback="30d")

                        # Custom date range
                        analytics = await client.billing.get_usage_analytics(
                            startDate="2025-01-01",
                            endDate="2025-01-31",
                        )

                        # Inspect results
                        for day in analytics.byDate:
                            print(f"{day.date}: ${day.USD:.2f} / {day.DIEM:.2f} DIEM")

                        for model in analytics.byModel:
                            print(f"{model.modelName}: ${model.totalUsd:.4f}")
        """
        warnings.warn(
            "Billing.get_usage_analytics() wraps a beta API endpoint "
            "(GET /billing/usage-analytics). The request/response schema and "
            "behaviour may change without notice.",
            stacklevel=2,
            category=FutureWarning,
        )

        # Validate mutually exclusive parameters
        if lookback is not None and (startDate is not None or endDate is not None):
            raise ValueError(
                "Cannot specify both 'lookback' and 'startDate'/'endDate'. "
                "Use either a relative lookback period or a custom date range."
            )

        # Validate that both startDate and endDate are provided together
        if (startDate is None) != (endDate is None):
            raise ValueError(
                "Both 'startDate' and 'endDate' are required when specifying a custom date range."
            )

        # Validate lookback format (e.g., "7d", "30d", up to "90d")
        if lookback is not None:
            lookback_pattern = re.compile(r"^[1-9]\d*d$")
            if not lookback_pattern.match(lookback):
                raise ValueError(
                    f"Invalid lookback format '{lookback}'. "
                    "Expected format like '7d', '30d', up to '90d'."
                )
            days = int(lookback[:-1])
            if days < 1 or days > 90:
                raise ValueError(f"Lookback period must be between 1d and 90d, got '{lookback}'.")

        # Validate date formats if provided (YYYY-MM-DD)
        if startDate is not None:
            validate_date_string(startDate, "startDate")
        if endDate is not None:
            validate_date_string(endDate, "endDate")

        # Validate date range if both dates provided
        if startDate is not None and endDate is not None:
            validate_date_range(
                start_date=startDate,
                end_date=endDate,
                start_param="startDate",
                end_param="endDate",
            )

        # Build query parameters
        query_params = UsageAnalyticsQueryParams(
            lookback=lookback,
            startDate=startDate,
            endDate=endDate,
        )

        params = query_params.model_dump(exclude_none=True)

        # Make the API request with aggressive timeout
        try:
            result = await asyncio.wait_for(
                self._client._request(
                    "GET",
                    "billing/usage-analytics",
                    params=params,
                    headers={"accept": "application/json"},
                ),
                timeout=BILLING_REQUEST_TIMEOUT_SECONDS,
            )
        except TimeoutError as e:
            raise BillingTimeoutError(original_error=e) from e
        except APITimeoutError as e:
            raise BillingTimeoutError(original_error=e.original_error) from e

        return UsageAnalyticsResponse.model_validate(result)


def _attachment_filename(content_disposition: str | None) -> str | None:
    """Return the ``filename`` parameter of a ``Content-Disposition`` header, if any."""
    if not content_disposition:
        return None
    message = Message()
    message["Content-Disposition"] = content_disposition
    return message.get_filename()
