"""
Venice AI API Keys Resource Module.

This module provides comprehensive API key management functionality for the Venice AI platform.
API keys serve as the primary authentication mechanism for accessing Venice AI services and are
essential for controlling access to various endpoints, managing usage quotas, and enforcing
rate limits across different model types and services.

The module supports both traditional API key creation and Web3-based authentication workflows,
allowing users to manage their credentials through either conventional means or blockchain-based
identity verification.

Main Features:
    - Create, list, retrieve, and delete API keys
    - Manage rate limits and usage monitoring
    - Web3 authentication and key generation
    - Usage analytics and billing integration

Classes:
    ApiKeys: Asynchronous resource client for comprehensive API key management operations
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from .._pagination import DEFAULT_PAGE_SIZE, Paginator, _PageResult
from .._resource import APIResource
from ..exceptions import APIResponseProcessingError
from ..types.api import (
    ApiKey,
    ApiKeyDetailsResponse,
    # Request models
    CreateApiKeyRequest,
    CreatedApiKey,
    DeleteApiKeyResponse,
    RateLimitLogsResponse,
    RateLimitsResponse,
    Web3ApiKeyResponse,
    Web3CreateApiKeyRequest,
    Web3TokenResponse,
)
from ..types.api.requests.api_keys import UpdateApiKeyRequest

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


class ApiKeys(APIResource["VeniceClient"]):
    """
    Asynchronous resource for comprehensive API key management operations.

    This class provides a complete interface for managing Venice AI API keys, including
    creation, deletion, retrieval, and monitoring capabilities. It supports both standard
    API key workflows and Web3-based authentication mechanisms.

    All operations are asynchronous and return awaitable coroutines. The class automatically
    handles request formatting, response parsing, and error handling for all API key operations.

    Key Features:
        - Create and manage API keys with custom configurations
        - List and retrieve existing keys with pagination support
        - Delete keys with proper cleanup
        - Monitor rate limits and usage patterns
        - Web3 authentication and blockchain-based key generation
        - Real-time usage analytics and billing integration

    Args:
        _client: The VeniceClient instance for making API requests.

    Example:
        Basic API key management:

        .. code-block:: python

            import asyncio
            from venice_ai import VeniceClient
            from venice_ai.types.api import CreateApiKeyRequest

            async def manage_api_keys():
                async with VeniceClient() as client:
                    # List existing keys
                    keys = await client.api_keys.list(limit=10)

                    # Create a new key
                    request = CreateApiKeyRequest(
                        description="Production API Key",
                        apiKeyType="INFERENCE"
                    )
                    new_key = await client.api_keys.create(api_key_request=request)

                    # Monitor usage
                    rate_limits = await client.api_keys.get_rate_limits()

            asyncio.run(manage_api_keys())
    """

    def _normalize_response(
        self,
        response_data: Any,
        operation: str,
        *,
        unwrap_data_key: bool = True,
        allow_empty: bool = False,
    ) -> Any:
        """
        Normalize API response by unwrapping data structures.

        This helper provides consistent response unwrapping across all API key operations,
        handling various response formats (wrapped/unwrapped) with uniform error handling.

        Args:
            response_data: Raw response from API (dict, list, or other)
            operation: Description of operation for error messages (e.g., "list API keys")
            unwrap_data_key: If True, check for and unwrap {"data": ...} wrapper
            allow_empty: If True, return None for missing data instead of error

        Returns:
            Unwrapped response data ready for model validation

        Raises:
            APIResponseProcessingError: If response format is unexpected
        """
        # Handle None response
        if response_data is None:
            if allow_empty:
                return None
            raise APIResponseProcessingError(f"Unexpected None response when trying to {operation}")

        # Try to unwrap 'data' key if requested
        if unwrap_data_key and isinstance(response_data, dict) and "data" in response_data:
            return response_data["data"]

        return response_data

    def _validate_list_response(
        self,
        response_data: Any,
        operation: str,
    ) -> list[ApiKey]:
        """
        Validate and parse a list of API keys from response data.

        Args:
            response_data: Unwrapped response data (list or dict with 'data' key)
            operation: Description of operation for error messages

        Returns:
            List of validated ApiKey objects, or empty list for unexpected formats

        Note:
            This method returns an empty list for unexpected formats to maintain
            backward compatibility with existing tests and behavior.
        """
        # Unwrap if needed
        data = self._normalize_response(
            response_data, operation, unwrap_data_key=True, allow_empty=True
        )

        # Handle empty response
        if data is None or (isinstance(data, list) and len(data) == 0):
            return []

        # Handle unexpected format - return empty list for backward compatibility
        if not isinstance(data, list):
            return []

        # Validate each item
        try:
            return [ApiKey.model_validate(item) for item in data]
        except Exception as e:
            raise APIResponseProcessingError(
                f"Failed to validate API key list when trying to {operation}: {str(e)}"
            ) from e

    async def list(self, *, page: int | None = None, limit: int | None = None) -> list[ApiKey]:
        """
        Retrieve a paginated list of API keys for the authenticated account.

        Returns metadata for all API keys associated with the current account, including
        both active and inactive keys. The actual secret key values are excluded from
        responses for security purposes.

        Args:
            page: Page number for pagination (1-based). Defaults to first page.
            limit: Maximum number of keys per page. Uses server default if not specified.

        Returns:
            List of API key objects containing metadata including ID, description,
            creation timestamp, expiration, and usage statistics.

        Raises:
            AuthenticationError: If the API key is invalid or expired.
            APIError: If the request fails or returns an error response.
            APIConnectionError: If unable to connect to the API.

        Example:
            .. code-block:: python

                # List all keys
                keys = await client.api_keys.list()

                # Paginated listing
                page_keys = await client.api_keys.list(page=1, limit=10)
                for key in page_keys:
                    print(f"Key: {key.id} - {key.description}")
        """
        params: dict[str, Any] = {}
        if page is not None:
            params["page"] = page
        if limit is not None:
            params["limit"] = limit

        response_data = await self._client.get("api_keys", params=params if params else None)

        # Use normalized validation for consistent handling
        return self._validate_list_response(response_data, "list API keys")

    def iter_all(
        self,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
    ) -> Paginator[ApiKey]:
        """Lazily iterate every API key, paging through the server as needed.

        Wraps :meth:`list` for unbounded enumeration. Termination: there's
        no pagination envelope on this endpoint, so the iterator stops on
        the first short page (``len(items) < page_size``).

        :param page_size: Server page size (default 100).
        :param max_items: Optional cap on total items yielded.

        Example::

            async for key in client.api_keys.iter_all():
                print(key.id, key.description)
        """

        async def _fetch_page(page_index: int) -> _PageResult[ApiKey]:
            # API uses 1-based pages; convert from zero-based iteration index.
            items = await self.list(page=page_index + 1, limit=page_size)
            return _PageResult(items=items, has_more=len(items) == page_size)

        return Paginator(_fetch_page, page_size=page_size, max_items=max_items)

    async def create(self, *, api_key_request: CreateApiKeyRequest) -> CreatedApiKey:
        """
        Create a new API key with the specified configuration.

        Generates a new API key based on the provided parameters. The secret key value
        is returned only once in this response and cannot be retrieved later, so it
        must be stored securely immediately upon creation.

        Args:
            api_key_request: Configuration for the new API key including:
                - description: Human-readable description for identification
                - apiKeyType: Key type ("INFERENCE", "ADMIN", etc.)
                - expiresAt: Optional ISO 8601 expiration timestamp
                - consumptionLimit: Optional usage quotas and restrictions
                - modelPrivacy: Optional privacy tier restricting which models
                  the key may call ("ALL", "PRIVATE_TEXT", "PRIVATE_ONLY")

        Returns:
            Complete API key object including the secret key value, metadata,
            and configuration details.

        Raises:
            AuthenticationError: If the current credentials are invalid.
            APIError: If the request is invalid or account limits are exceeded.
            APIConnectionError: If unable to connect to the API.

        Warning:
            The secret key value is only returned once. Store it securely immediately.

        Example:
            .. code-block:: python

                from venice_ai.types.api import CreateApiKeyRequest

                # Create a production API key
                request = CreateApiKeyRequest(
                    description="Production Service Key",
                    apiKeyType="INFERENCE"
                )
                new_key = await client.api_keys.create(api_key_request=request)

                # Store the secret key securely
                secret_key = new_key.apiKey  # Only available now!
        """
        data = self._serialize_request(api_key_request)

        response = await self._client.post("api_keys", json_data=data)

        # Unwrap data key if present
        api_key_data = self._normalize_response(response, "create API key", unwrap_data_key=True)

        # Validate that we got a dict (not string or other unexpected type)
        if not isinstance(api_key_data, dict):
            raise APIResponseProcessingError(
                f"Unexpected response format from API key creation endpoint. Expected dict, got {type(api_key_data).__name__}"
            )

        # Make a copy to avoid modifying original data
        api_key_data = dict(api_key_data)

        # Normalizes legacy field name variant from older API versions.
        if "consumptionLimits" in api_key_data and "consumptionLimit" not in api_key_data:
            api_key_data["consumptionLimit"] = api_key_data.pop("consumptionLimits")
        # Provide default empty object if field is missing entirely
        elif "consumptionLimit" not in api_key_data:
            api_key_data["consumptionLimit"] = {}

        # Validate and return
        try:
            return CreatedApiKey.model_validate(api_key_data)
        except Exception as e:
            raise APIResponseProcessingError(
                f"Failed to validate created API key response: {str(e)}"
            ) from e

    async def delete(self, *, api_key_id: str) -> DeleteApiKeyResponse:
        """
        Permanently delete an API key.

        Removes the specified API key from the account, immediately invalidating it
        for all future requests. This operation cannot be undone.

        Args:
            api_key_id: Unique identifier of the API key to delete. This is the
                key's ID (not the secret value) as returned by create or list operations.

        Returns:
            Deletion confirmation response with operation status.

        Raises:
            AuthenticationError: If the current credentials are invalid.
            APIError: If the key ID doesn't exist or belongs to another account.
            APIConnectionError: If unable to connect to the API.

        Warning:
            This operation is irreversible. The deleted key cannot be recovered.

        Example:
            .. code-block:: python

                # Delete a specific key
                result = await client.api_keys.delete(api_key_id="key_123456789")

                # Batch delete test keys
                keys = await client.api_keys.list()
                for key in keys:
                    if "test" in key.description.lower():
                        await client.api_keys.delete(api_key_id=key.id)
        """
        # Construct the URL with the API key ID as a query parameter
        path = "api_keys"
        params = {"id": api_key_id}
        response = await self._client.delete(path, params=params)
        return DeleteApiKeyResponse.model_validate(response)

    async def update(
        self,
        *,
        id: str,
        description: str | None = None,
        expires_at: str | None = None,
        consumption_limit: dict[str, Any] | None = None,
        limit_period: Literal["EPOCH", "MONTH", "LIFETIME"] | None = None,
    ) -> ApiKey:
        """Update an existing API key (PATCH /api_keys).

        Args:
            id: ID of the API key to update.
            description: New description for the key.
            expires_at: New expiration date (ISO 8601 format).
            consumption_limit: Epoch consumption limits (e.g., ``{'diem': 1, 'usd': 10}``).
            limit_period: Period over which the consumption limit resets
                (``EPOCH``, ``MONTH``, or ``LIFETIME``).

        Returns:
            Updated :class:`ApiKey` object.

        Raises:
            AuthenticationError: If the API key is invalid or expired.
            APIError: If the request fails or returns an error response.
            APIConnectionError: If unable to connect to the API.

        Example:
            .. code-block:: python

                updated = await client.api_keys.update(
                    id="key_123",
                    description="Updated description",
                    consumption_limit={"diem": 5, "usd": 50},
                )
        """
        # Build the request body manually to support both dict and ConsumptionLimit
        request = UpdateApiKeyRequest(
            id=id,
            description=description,
            expiresAt=expires_at,
            consumptionLimit=consumption_limit,  # type: ignore[arg-type]  # Pydantic coerces dict to ConsumptionLimit at validate time
            limitPeriod=limit_period,
        )
        body = request.model_dump(exclude_none=True, by_alias=True)
        # Per the docs (api-reference/endpoint/api_keys/update), the response is
        # wrapped: ``{"data": {<ApiKey fields>}, "success": true}``. Cast to
        # the wrapper, then unwrap so callers get an ``ApiKey`` directly.
        response = await self._client.patch(
            "api_keys", json_data=body, cast_to=ApiKeyDetailsResponse
        )
        return response.data

    async def retrieve(self, *, api_key_id: str) -> ApiKey:
        """
        Retrieve detailed information about a specific API key.

        Fetches comprehensive metadata for the specified API key, including usage
        statistics, configuration details, and current status. The secret key value
        is never included in responses for security purposes.

        Args:
            api_key_id: Unique identifier of the API key to retrieve.

        Returns:
            :class:`ApiKey` with metadata, usage statistics, and configuration
            parameters. The wire response is ``{"data": {<ApiKey fields>}}``;
            the wrapper is unwrapped so callers receive the bare object,
            consistent with :meth:`update`.

        Raises:
            AuthenticationError: If the current credentials are invalid.
            NotFoundError: If the specified key ID doesn't exist.
            APIError: If the request fails or returns an error response.
            APIConnectionError: If unable to connect to the API.

        Example:
            .. code-block:: python

                # Get detailed key information
                api_key = await client.api_keys.retrieve(api_key_id="key_123456789")
                print(f"Description: {api_key.description}")
                print(f"Created: {api_key.createdAt}")
                print(f"Usage: {api_key.usage}")
        """
        # Per the docs (api-reference/endpoint/api_keys/get), the response is
        # wrapped: ``{"data": {<ApiKey fields>}}``. Cast to the wrapper, then
        # unwrap so callers get an ``ApiKey`` directly — matching ``update()``.
        response = await self._client.get(f"api_keys/{api_key_id}", cast_to=ApiKeyDetailsResponse)
        return response.data

    async def get_web3_token(self) -> Web3TokenResponse:
        """
        Retrieve a temporary token for Web3 API key generation.

        Generates a time-limited authentication token required for creating API keys
        through Web3 blockchain authentication. This token must be used in the
        subsequent Web3 key creation request.

        Returns:
            Response containing the temporary token and any associated metadata
            required for Web3 authentication.

        Raises:
            APIError: If token generation fails or service is unavailable.
            APIConnectionError: If unable to connect to the API.

        Note:
            The returned token has a limited lifespan and should be used immediately
            in the Web3 key creation process.
        """
        response = await self._client.get("api_keys/generate_web3_key")
        return Web3TokenResponse.model_validate(response)

    async def create_web3_key(
        self, *, web3_key_request: Web3CreateApiKeyRequest
    ) -> Web3ApiKeyResponse:
        """
        Create a new API key using Web3 blockchain authentication.

        Generates an API key authenticated through blockchain signature verification,
        enabling decentralized identity management for Venice AI access.

        Args:
            web3_key_request: Web3 authentication request containing:
                - apiKeyType: API key type (``"INFERENCE"`` or ``"ADMIN"``)
                - address: Wallet address for authentication
                - signature: Signed token for verification
                - token: Token from get_web3_token()
                - description (optional): API key description
                - consumptionLimit (optional): Spending limits
                - limitPeriod (optional): Consumption-limit reset period
                  (``"EPOCH"``/``"MONTH"``/``"LIFETIME"``)
                - expiresAt (optional): Expiration date

        Returns:
            Response containing the newly created API key and associated metadata.

        Raises:
            APIError: If Web3 verification fails or key creation is rejected.
            APIConnectionError: If unable to connect to the API.

        Note:
            Requires a valid Web3 token from get_web3_token() and proper signature
            verification against the specified blockchain network.
        """
        data = self._serialize_request(web3_key_request)

        response = await self._client.post("api_keys/generate_web3_key", json_data=data)
        return Web3ApiKeyResponse.model_validate(response)

    def _serialize_request(self, request_obj: Any) -> dict[str, Any]:
        """
        Serialize a request object to a dictionary for JSON serialization.

        Handles Pydantic models, Mapping objects, and objects with __dict__.

        Args:
            request_obj: The request object to serialize

        Returns:
            Dictionary with None values excluded
        """
        data: dict[str, Any]
        # Pydantic models use model_dump()
        if hasattr(request_obj, "model_dump"):
            data = request_obj.model_dump(exclude_none=True)
        # Check for Mapping (dict-like) - covers dict/TypedDict
        elif isinstance(request_obj, Mapping):
            data = {k: v for k, v in request_obj.items() if v is not None}
        # Fallback for objects with __dict__
        elif hasattr(request_obj, "__dict__"):
            data = {k: v for k, v in vars(request_obj).items() if v is not None}
        else:
            # Last resort
            data = dict(request_obj)
            data = {k: v for k, v in data.items() if v is not None}
        return data

    async def get_rate_limits(self) -> RateLimitsResponse:
        """
        Retrieve current rate limit information and usage statistics.

        Returns comprehensive rate limiting data for the authenticated API key, including
        configured limits across different time periods and current usage levels.

        Returns:
            Rate limit configuration and current usage statistics including:
            - Limits per minute, hour, day, and month
            - Current usage counts for each period
            - Remaining capacity and reset times

        Raises:
            AuthenticationError: If the API key is invalid or expired.
            APIError: If the request fails or returns an error response.
            APIConnectionError: If unable to connect to the API.

        Example:
            .. code-block:: python

                # Check current rate limits
                limits = await client.api_keys.get_rate_limits()
                print(f"Requests per minute: {limits.data.requests_per_minute}")
                print(f"Current usage: {limits.data.current_usage}")
        """
        response = await self._client.get("api_keys/rate_limits")
        return RateLimitsResponse.model_validate(response)

    async def get_rate_limit_logs(self) -> RateLimitLogsResponse:
        """
        Retrieves the last 50 rate limit violations for the account asynchronously.

        Returns the last 50 rate limits that the account exceeded. This endpoint
        helps monitor and troubleshoot rate limiting issues by providing a history
        of when limits were hit.

        :return: List of the last 50 rate limit violations with timestamps, models,
            and violation types.


        :raises venice_ai.exceptions.AuthenticationError: If authentication fails.
        :raises venice_ai.exceptions.APIError: If the API returns an error.
        :raises venice_ai.exceptions.APIConnectionError: If there's an issue connecting to the API.

        Example:
            .. code-block:: python

                # Get recent rate limit logs asynchronously
                logs_response = await client.api_keys.get_rate_limit_logs()
                for log_entry in logs_response.data:
                    print(f"Model: {log_entry.modelId}, Type: {log_entry.rateLimitType}, Time: {log_entry.timestamp}")
        """
        response = await self._client.get("api_keys/rate_limits/log")
        return RateLimitLogsResponse.model_validate(response)
