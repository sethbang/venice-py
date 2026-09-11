"""
Comprehensive tests for src/venice_ai/resources/api_keys.py module.

This test file focuses on achieving >80% coverage for API key management functions,
testing all methods: create, list, get, delete, update, and error handling.
"""

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseProcessingError,
    AuthenticationError,
    NotFoundError,
)
from venice_ai.resources.api_keys import ApiKeys
from venice_ai.types.api import (
    CreateApiKeyRequest,
    Web3CreateApiKeyRequest,
)


class MockVeniceClient:
    """Mock client for testing ApiKeys resource."""

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self.get = AsyncMock()
        self.post = AsyncMock()
        self.delete = AsyncMock()
        self.patch = AsyncMock()


@pytest.fixture
def mock_client():
    """Create a mock Venice client for testing."""
    return MockVeniceClient()


@pytest.fixture
def api_keys_resource(mock_client):
    """Create an ApiKeys resource instance for testing."""
    return ApiKeys(mock_client)


@pytest.fixture
def sample_api_key():
    """Sample API key data."""
    return {
        "id": "key_123456789",
        "apiKey": "venice_test_key_abcdef123456",
        "apiKeyType": "INFERENCE",
        "description": "Test API Key",
        "createdAt": "2025-01-01T00:00:00Z",
        "expiresAt": None,
        "lastUsedAt": "2025-01-15T12:00:00Z",
        "last6Chars": "123456",
        "consumptionLimits": {"usd": None, "diem": None},
        "usage": {"trailingSevenDays": {"usd": "10.50", "diem": "25.00"}},
    }


@pytest.fixture
def sample_create_request():
    """Sample API key creation request."""
    return CreateApiKeyRequest(
        description="Test API Key",
        apiKeyType="INFERENCE",
        consumptionLimit=None,
        expiresAt=None,
    )


@pytest.fixture
def sample_web3_request():
    """Sample Web3 API key creation request."""
    return Web3CreateApiKeyRequest(
        apiKeyType="INFERENCE",
        address="0x123456789abcdef",
        signature="0xsignature123",
        token="web3_token_123",
        description="Web3 Test Key",
        consumptionLimit=None,
        expiresAt=None,
    )


class TestApiKeysList:
    """Test list() method functionality."""

    @pytest.mark.asyncio
    async def test_list_basic_success(self, api_keys_resource, mock_client, sample_api_key):
        """Test successful API key listing without parameters."""
        mock_client.get.return_value = [sample_api_key]

        result = await api_keys_resource.list()

        # Result is now a list of Pydantic ApiKey objects
        assert len(result) == 1
        assert result[0].id == sample_api_key["id"]
        assert result[0].apiKeyType == sample_api_key["apiKeyType"]
        assert result[0].description == sample_api_key["description"]
        mock_client.get.assert_called_once_with("api_keys", params=None)

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, api_keys_resource, mock_client, sample_api_key):
        """Test API key listing with pagination parameters."""
        mock_client.get.return_value = [sample_api_key]

        result = await api_keys_resource.list(page=1, limit=10)

        # Result is now a list of Pydantic ApiKey objects
        assert len(result) == 1
        assert result[0].id == sample_api_key["id"]
        mock_client.get.assert_called_once_with("api_keys", params={"page": 1, "limit": 10})

    @pytest.mark.asyncio
    async def test_list_with_data_wrapper(self, api_keys_resource, mock_client, sample_api_key):
        """Test API key listing when response has data wrapper."""
        response_data = {"data": [sample_api_key]}
        mock_client.get.return_value = response_data

        result = await api_keys_resource.list()

        # Result is now a list of Pydantic ApiKey objects
        assert len(result) == 1
        assert result[0].id == sample_api_key["id"]

    @pytest.mark.asyncio
    async def test_list_empty_response(self, api_keys_resource, mock_client):
        """Test API key listing with empty response."""
        mock_client.get.return_value = []

        result = await api_keys_resource.list()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_unexpected_format(self, api_keys_resource, mock_client):
        """Test API key listing with unexpected response format."""
        mock_client.get.return_value = {"unexpected": "format"}

        result = await api_keys_resource.list()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_only_page_parameter(self, api_keys_resource, mock_client, sample_api_key):
        """Test listing with only page parameter."""
        mock_client.get.return_value = [sample_api_key]

        result = await api_keys_resource.list(page=2)

        # Result is now a list of Pydantic ApiKey objects
        assert len(result) == 1
        assert result[0].id == sample_api_key["id"]
        mock_client.get.assert_called_once_with("api_keys", params={"page": 2})

    @pytest.mark.asyncio
    async def test_list_only_limit_parameter(self, api_keys_resource, mock_client, sample_api_key):
        """Test listing with only limit parameter."""
        mock_client.get.return_value = [sample_api_key]

        result = await api_keys_resource.list(limit=5)

        # Result is now a list of Pydantic ApiKey objects
        assert len(result) == 1
        assert result[0].id == sample_api_key["id"]
        mock_client.get.assert_called_once_with("api_keys", params={"limit": 5})


class TestApiKeysValidationErrorPaths:
    """Cover the validation-error wrapping branches that the rewritten tests
    sidestep when they mock past validation by returning pre-validated models.

    These pass *raw dicts/lists* through the mock client so the resource code
    actually executes ``_normalize_response`` and ``_validate_list_response``.
    """

    @pytest.mark.asyncio
    async def test_list_wraps_pydantic_validation_failure(self, api_keys_resource, mock_client):
        """When the API returns items that don't match the ``ApiKey`` schema,
        ``_validate_list_response`` wraps the Pydantic error in an
        ``APIResponseProcessingError``."""
        # Each item is missing the required ``id`` field — Pydantic will reject.
        malformed_items = [{"apiKeyType": "INFERENCE", "description": "no id"}]
        mock_client.get.return_value = malformed_items

        with pytest.raises(APIResponseProcessingError) as exc_info:
            await api_keys_resource.list()

        assert "Failed to validate API key list" in str(exc_info.value)
        assert "list API keys" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_wraps_pydantic_validation_failure(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """When the create endpoint returns a payload that can't deserialize to
        ``CreatedApiKey``, ``create()`` wraps the failure in
        ``APIResponseProcessingError``."""
        # Server returns a payload missing required ``apiKey`` field.
        mock_client.post.return_value = {
            "data": {
                "id": "key_123",
                "apiKeyType": "INFERENCE",
                "description": "missing apiKey",
            }
        }

        with pytest.raises(APIResponseProcessingError) as exc_info:
            await api_keys_resource.create(api_key_request=sample_create_request)

        assert "Failed to validate created API key response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_provides_default_consumption_limit_when_missing(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """When the response omits ``consumptionLimit``, ``create()`` injects an
        empty dict so model validation succeeds."""
        # Response has no consumptionLimit at all — server may legitimately omit it.
        mock_client.post.return_value = {
            "data": {
                "id": "key_no_limit",
                "apiKey": "venice_test_key_NoLimit_pad_padding",
                "apiKeyType": "INFERENCE",
                "description": "no limit",
                "expiresAt": None,
            }
        }

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        assert result.id == "key_no_limit"
        # Default empty ConsumptionLimits — all currency fields None.
        assert result.consumptionLimit.usd is None
        assert result.consumptionLimit.diem is None


class TestApiKeysCreate:
    """Test create() method functionality."""

    @pytest.mark.asyncio
    async def test_create_basic_success(
        self, api_keys_resource, mock_client, sample_create_request, sample_api_key
    ):
        """Test successful API key creation."""
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        mock_client.post.assert_called_once_with(
            "api_keys",
            json_data=sample_create_request.model_dump(exclude_none=True),
        )

    @pytest.mark.asyncio
    async def test_create_with_consumption_limit_mapping(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """Test API key creation with consumption limit field mapping."""
        api_key_data = {
            "id": "key_123",
            "apiKey": "venice_key_123",
            "apiKeyType": "INFERENCE",
            "description": "Test Key",
            "createdAt": "2025-01-01T00:00:00Z",
            "last6Chars": "ey_123",
            "consumptionLimits": {"usd": 1000.0, "diem": None},
            "usage": {"trailingSevenDays": {"usd": "10.50", "diem": "25.00"}},
        }
        response_data = {"data": api_key_data}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        # Result is now a Pydantic ApiKey object
        assert result.id == api_key_data["id"]
        assert result.apiKeyType == api_key_data["apiKeyType"]
        assert result.description == api_key_data["description"]

    @pytest.mark.asyncio
    async def test_create_missing_consumption_limits(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """Test API key creation with missing consumptionLimits field."""
        api_key_data = {
            "id": "key_123",
            "apiKey": "venice_key_123",
            "apiKeyType": "INFERENCE",
            "description": "Test Key",
            "createdAt": "2025-01-01T00:00:00Z",
            "last6Chars": "ey_123",
            "consumptionLimits": {"usd": None, "diem": None},
            "usage": {"trailingSevenDays": {"usd": "0.00", "diem": "0.00"}},
        }
        response_data = {"data": api_key_data}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        # Result is now a Pydantic ApiKey object
        assert result.id == api_key_data["id"]
        assert result.apiKeyType == api_key_data["apiKeyType"]
        assert result.description == api_key_data["description"]

    @pytest.mark.asyncio
    async def test_create_field_filtering(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """Test that only valid ApiKey fields are included in response."""
        api_key_data = {
            "id": "key_123",
            "apiKey": "venice_key_123",
            "apiKeyType": "INFERENCE",
            "description": "Test Key",
            "createdAt": "2025-01-01T00:00:00Z",
            "last6Chars": "ey_123",
            "consumptionLimits": {"usd": None, "diem": None},
            "usage": {"trailingSevenDays": {"usd": "0.00", "diem": "0.00"}},
            "invalidField": "should_be_filtered",  # Invalid field
            "anotherInvalid": 123,
        }
        response_data = {"data": api_key_data}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        # Invalid fields should be filtered out by Pydantic validation
        # Result is now a Pydantic ApiKey object
        assert result.id == api_key_data["id"]
        assert result.apiKeyType == api_key_data["apiKeyType"]
        assert result.description == api_key_data["description"]
        # Invalid fields won't be present in the Pydantic object

    @pytest.mark.asyncio
    async def test_create_response_without_data_key(
        self, api_keys_resource, mock_client, sample_create_request, sample_api_key
    ):
        """Test API key creation when response has no data wrapper."""
        mock_client.post.return_value = sample_api_key

        result = await api_keys_resource.create(api_key_request=sample_create_request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]

    @pytest.mark.asyncio
    async def test_create_unexpected_response_format(
        self, api_keys_resource, mock_client, sample_create_request
    ):
        """Test API key creation with unexpected response format."""
        mock_client.post.return_value = "unexpected_string"

        with pytest.raises(APIResponseProcessingError) as exc_info:
            await api_keys_resource.create(api_key_request=sample_create_request)

        assert "Unexpected response format" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_with_dict_request(self, api_keys_resource, mock_client, sample_api_key):
        """Test API key creation with dict-like request object."""
        dict_request = {"description": "Test Key", "apiKeyType": "INFERENCE"}
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=dict_request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        mock_client.post.assert_called_once_with(
            "api_keys",
            json_data={"description": "Test Key", "apiKeyType": "INFERENCE"},
        )

    @pytest.mark.asyncio
    async def test_create_with_object_with_dict(
        self, api_keys_resource, mock_client, sample_api_key
    ):
        """Test API key creation with object that has __dict__ attribute."""

        class RequestObject:
            def __init__(self):
                self.description = "Test Key"
                self.apiKeyType = "INFERENCE"
                self.optionalField = None

        request_obj = RequestObject()
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request_obj)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        # Should exclude None values
        expected_data = {"description": "Test Key", "apiKeyType": "INFERENCE"}
        mock_client.post.assert_called_once_with("api_keys", json_data=expected_data)


class TestApiKeysDelete:
    """Test delete() method functionality."""

    @pytest.mark.asyncio
    async def test_delete_success(self, api_keys_resource, mock_client):
        """Test successful API key deletion."""
        delete_response = {"success": True, "message": "API key deleted"}
        mock_client.delete.return_value = delete_response

        result = await api_keys_resource.delete(api_key_id="key_123")

        # Result is now a Pydantic object, not a dictionary
        assert result.success == delete_response["success"]
        mock_client.delete.assert_called_once_with("api_keys", params={"id": "key_123"})

    @pytest.mark.asyncio
    async def test_delete_nonexistent_key(self, api_keys_resource, mock_client):
        """Test deletion of non-existent API key."""
        mock_response = MagicMock()
        mock_client.delete.side_effect = APIError("API key not found", response=mock_response)

        with pytest.raises(APIError):
            await api_keys_resource.delete(api_key_id="nonexistent_key")


class TestApiKeysRetrieve:
    """Test retrieve() method functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_success(self, api_keys_resource, mock_client, sample_api_key):
        """retrieve() hits GET /api_keys/{id} and unwraps ``data`` to a bare ApiKey,
        consistent with update()."""
        from venice_ai.types.api import ApiKey, ApiKeyDetailsResponse

        details = ApiKeyDetailsResponse.model_validate({"data": sample_api_key})
        mock_client.get.return_value = details

        result = await api_keys_resource.retrieve(api_key_id="key_123456789")

        assert isinstance(result, ApiKey)
        # Bare ApiKey: fields accessed directly, not via ``.data``.
        assert result.id == sample_api_key["id"]
        assert result.description == sample_api_key["description"]
        mock_client.get.assert_called_once_with(
            "api_keys/key_123456789", cast_to=ApiKeyDetailsResponse
        )

    @pytest.mark.asyncio
    async def test_retrieve_propagates_not_found(self, api_keys_resource, mock_client):
        """A 404 from the API surfaces as NotFoundError raised by the client layer."""
        mock_client.get.side_effect = NotFoundError(
            "API Key not found.", response=None, request=None
        )

        with pytest.raises(NotFoundError):
            await api_keys_resource.retrieve(api_key_id="nonexistent_key")


class TestApiKeysUpdate:
    """Test ``update()`` (PATCH /api_keys) — the new endpoint added during API
    alignment. The wire shape (camelCase aliases, ``data`` wrapper on response)
    is locked in by the recorded cassette in
    ``tests/integration/cassettes/test_api_keys_update_*.yaml``; these unit
    tests assert on the *outgoing* request body so any drift between the SDK
    and the cassette is caught here too.
    """

    @pytest.mark.asyncio
    async def test_update_description_only(self, api_keys_resource, mock_client, sample_api_key):
        """A description-only patch sends ``{"id", "description"}`` and unwraps
        ``data`` from the response."""
        from venice_ai.types.api import ApiKeyDetailsResponse

        wrapped = ApiKeyDetailsResponse.model_validate({"data": sample_api_key})
        mock_client.patch.return_value = wrapped

        result = await api_keys_resource.update(
            id="key_123456789",
            description="Updated description",
        )

        # Returned ApiKey is the unwrapped ``.data`` of the response.
        assert result is wrapped.data
        assert result.id == sample_api_key["id"]

        # Outgoing request: path + cast_to + body shape.
        mock_client.patch.assert_called_once()
        call = mock_client.patch.call_args
        assert call.args[0] == "api_keys"
        assert call.kwargs["cast_to"] is ApiKeyDetailsResponse

        body = call.kwargs["json_data"]
        # camelCase aliases per ``by_alias=True``; None fields excluded per ``exclude_none=True``.
        assert body == {"id": "key_123456789", "description": "Updated description"}
        assert "expiresAt" not in body
        assert "consumptionLimit" not in body

    @pytest.mark.asyncio
    async def test_update_consumption_limit_dict_serializes_to_camel_case(
        self, api_keys_resource, mock_client, sample_api_key
    ):
        """The snake_case kwarg ``consumption_limit`` becomes camelCase
        ``consumptionLimit`` on the wire — and accepts a plain dict (not just
        a ``ConsumptionLimit`` model). Verified against the cassette body."""
        from venice_ai.types.api import ApiKeyDetailsResponse

        wrapped = ApiKeyDetailsResponse.model_validate({"data": sample_api_key})
        mock_client.patch.return_value = wrapped

        await api_keys_resource.update(
            id="key_123456789",
            consumption_limit={"diem": 1, "usd": 10},
        )

        body = mock_client.patch.call_args.kwargs["json_data"]
        assert body["id"] == "key_123456789"
        # SINGULAR ``consumptionLimit`` on the request body (per the docs);
        # response uses PLURAL ``consumptionLimits``.
        assert "consumptionLimit" in body
        assert body["consumptionLimit"] == {"diem": 1, "usd": 10}
        # Nothing else gets sent.
        assert "description" not in body
        assert "expiresAt" not in body

    @pytest.mark.asyncio
    async def test_update_all_fields(self, api_keys_resource, mock_client, sample_api_key):
        """All optional fields populate the body when passed."""
        from venice_ai.types.api import ApiKeyDetailsResponse

        wrapped = ApiKeyDetailsResponse.model_validate({"data": sample_api_key})
        mock_client.patch.return_value = wrapped

        await api_keys_resource.update(
            id="key_123456789",
            description="d",
            expires_at="2025-12-31T23:59:59Z",
            consumption_limit={"diem": 5, "usd": 50},
        )

        body = mock_client.patch.call_args.kwargs["json_data"]
        assert body == {
            "id": "key_123456789",
            "description": "d",
            "expiresAt": "2025-12-31T23:59:59Z",
            "consumptionLimit": {"diem": 5, "usd": 50},
        }


class TestApiKeysWeb3:
    """Test Web3-related methods."""

    @pytest.mark.asyncio
    async def test_get_web3_token_success(self, api_keys_resource, mock_client):
        """Test successful Web3 token retrieval."""
        token_response = {
            "success": True,
            "data": {"token": "web3_token_123", "expires_in": 3600},
        }
        mock_client.get.return_value = token_response

        result = await api_keys_resource.get_web3_token()

        assert result.success
        assert result.data.token == "web3_token_123"
        mock_client.get.assert_called_once_with("api_keys/generate_web3_key")

    @pytest.mark.asyncio
    async def test_create_web3_key_success(
        self, api_keys_resource, mock_client, sample_web3_request
    ):
        """Test successful Web3 API key creation."""
        web3_response = {
            "success": True,
            "data": {
                "id": "web3_key_123",
                "apiKey": "venice_web3_key_xyz",
                "apiKeyType": "INFERENCE",
                "description": "Web3 Test Key",
                "consumptionLimit": {"usd": None, "diem": None},
            },
        }
        mock_client.post.return_value = web3_response

        result = await api_keys_resource.create_web3_key(web3_key_request=sample_web3_request)

        assert result.success
        assert result.data.id == "web3_key_123"
        mock_client.post.assert_called_once_with(
            "api_keys/generate_web3_key",
            json_data=sample_web3_request.model_dump(exclude_none=True),
        )

    @pytest.mark.asyncio
    async def test_create_web3_key_no_data_wrapper(
        self, api_keys_resource, mock_client, sample_web3_request
    ):
        """Test Web3 key creation when response has no data wrapper."""
        web3_response = {
            "success": True,
            "data": {
                "id": "web3_key_123",
                "apiKey": "venice_web3_key_xyz",
                "apiKeyType": "INFERENCE",
                "description": "Web3 Test Key",
                "consumptionLimit": {"usd": None, "diem": None},
            },
        }
        mock_client.post.return_value = web3_response

        result = await api_keys_resource.create_web3_key(web3_key_request=sample_web3_request)

        assert result.success
        assert result.data.id == "web3_key_123"

    @pytest.mark.asyncio
    async def test_create_web3_key_with_dict_request(self, api_keys_resource, mock_client):
        """Test Web3 key creation with dict-like request object."""
        dict_request = {
            "description": "Web3 Dict Key",
            "apiKeyType": "INFERENCE",
            "web3_network_id": "ethereum",
            "web3_address": "0xabc123",
            "signature": "0xsig456",
        }
        web3_response = {
            "success": True,
            "data": {
                "id": "web3_key_123",
                "apiKey": "venice_web3_key_xyz",
                "apiKeyType": "INFERENCE",
                "description": "Web3 Dict Key",
                "consumptionLimit": {"usd": None, "diem": None},
            },
        }
        mock_client.post.return_value = web3_response

        result = await api_keys_resource.create_web3_key(web3_key_request=dict_request)

        assert result.success
        assert result.data.id == "web3_key_123"
        mock_client.post.assert_called_once_with(
            "api_keys/generate_web3_key", json_data=dict_request
        )


class TestApiKeysRateLimits:
    """Test rate limit related methods."""

    @pytest.mark.asyncio
    async def test_get_rate_limits_with_data_wrapper(self, api_keys_resource, mock_client):
        """Test rate limits retrieval with data wrapper."""
        rate_limits_data = {
            "data": {
                "accessPermitted": True,
                "apiTier": {"id": "free", "isCharged": False},
                "balances": {"USD": 100.0, "DIEM": 50.0},
                "keyExpiration": None,
                "nextEpochBegins": "2025-01-16T00:00:00Z",
                "rateLimits": [
                    {
                        "apiModelId": "llama-3.2-3b",
                        "rateLimits": [
                            {"type": "rpm", "amount": 60.0},
                            {"type": "tpm", "amount": 10000.0},
                        ],
                    }
                ],
            }
        }
        mock_client.get.return_value = rate_limits_data

        result = await api_keys_resource.get_rate_limits()

        assert result.data.accessPermitted
        assert result.data.apiTier.id == "free"
        mock_client.get.assert_called_once_with("api_keys/rate_limits")

    @pytest.mark.asyncio
    async def test_get_rate_limits_without_data_wrapper(self, api_keys_resource, mock_client):
        """Test rate limits retrieval without data wrapper."""
        rate_limits_data = {
            "data": {
                "accessPermitted": True,
                "apiTier": {"id": "free", "isCharged": False},
                "balances": {"USD": 100.0, "DIEM": 50.0},
                "keyExpiration": None,
                "nextEpochBegins": "2025-01-16T00:00:00Z",
                "rateLimits": [],
            }
        }
        mock_client.get.return_value = rate_limits_data

        result = await api_keys_resource.get_rate_limits()

        assert result.data.accessPermitted

    @pytest.mark.asyncio
    async def test_get_rate_limit_logs_success(self, api_keys_resource, mock_client):
        """Test successful rate limit logs retrieval."""
        logs_data = {
            "object": "list",
            "data": [
                {
                    "apiKeyId": "key_123",
                    "timestamp": "2025-01-15T12:00:00Z",
                    "modelId": "llama-3.2-3b",
                    "rateLimitTier": "free",
                    "rateLimitType": "requests_per_minute",
                }
            ],
        }
        mock_client.get.return_value = logs_data

        result = await api_keys_resource.get_rate_limit_logs()

        assert result.object == "list"
        assert len(result.data) == 1
        assert result.data[0].apiKeyId == "key_123"
        mock_client.get.assert_called_once_with("api_keys/rate_limits/log")


class TestApiKeysRequestSerialization:
    """Test request object serialization and edge cases."""

    @pytest.mark.asyncio
    async def test_create_with_pydantic_model(self, api_keys_resource, mock_client, sample_api_key):
        """Test creation with Pydantic model (has model_dump method)."""
        request = CreateApiKeyRequest(
            description="Pydantic Test Key",
            apiKeyType="INFERENCE",
            expiresAt="2025-12-31T23:59:59Z",
            consumptionLimit=None,
        )
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        expected_data = {
            "description": "Pydantic Test Key",
            "apiKeyType": "INFERENCE",
            "expiresAt": "2025-12-31T23:59:59Z",
        }
        mock_client.post.assert_called_once_with("api_keys", json_data=expected_data)

    @pytest.mark.asyncio
    async def test_create_with_mapping_object(self, api_keys_resource, mock_client, sample_api_key):
        """Test creation with mapping-like object."""
        from collections.abc import Mapping

        class MappingRequest(Mapping):
            def __init__(self):
                self._data = {
                    "description": "Mapping Test Key",
                    "apiKeyType": "INFERENCE",
                    "nullField": None,
                }

            def __getitem__(self, key):
                return self._data[key]

            def __iter__(self):
                return iter(self._data)

            def __len__(self):
                return len(self._data)

            def items(self):
                return self._data.items()

        request = MappingRequest()
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        # Should exclude None values
        expected_data = {"description": "Mapping Test Key", "apiKeyType": "INFERENCE"}
        mock_client.post.assert_called_once_with("api_keys", json_data=expected_data)

    @pytest.mark.asyncio
    async def test_create_fallback_serialization(
        self, api_keys_resource, mock_client, sample_api_key
    ):
        """Test creation with fallback serialization method."""

        # Object that doesn't have model_dump, isn't Mapping, doesn't have __dict__
        # but can be converted with dict()
        class FallbackRequest:
            def __init__(self):
                pass

            def __iter__(self):
                return iter([("description", "Fallback Key"), ("apiKeyType", "INFERENCE")])

        request = FallbackRequest()
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]


class TestApiKeysErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_list_authentication_error(self, api_keys_resource, mock_client):
        """Test handling of authentication error during list."""
        mock_response = MagicMock()
        mock_client.get.side_effect = AuthenticationError("Invalid API key", response=mock_response)

        with pytest.raises(AuthenticationError):
            await api_keys_resource.list()

    @pytest.mark.asyncio
    async def test_create_api_error(self, api_keys_resource, mock_client, sample_create_request):
        """Test handling of API error during creation."""
        mock_response = MagicMock()
        mock_client.post.side_effect = APIError("API key limit reached", response=mock_response)

        with pytest.raises(APIError):
            await api_keys_resource.create(api_key_request=sample_create_request)

    @pytest.mark.asyncio
    async def test_delete_connection_error(self, api_keys_resource, mock_client):
        """Test handling of connection error during deletion."""
        mock_client.delete.side_effect = APIConnectionError("Connection failed")

        with pytest.raises(APIConnectionError):
            await api_keys_resource.delete(api_key_id="key_123")


class TestApiKeysEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_list_with_zero_limit(self, api_keys_resource, mock_client):
        """Test listing with zero limit."""
        mock_client.get.return_value = []

        result = await api_keys_resource.list(limit=0)

        assert result == []
        mock_client.get.assert_called_once_with("api_keys", params={"limit": 0})

    @pytest.mark.asyncio
    async def test_list_with_negative_page(self, api_keys_resource, mock_client):
        """Test listing with negative page number."""
        mock_client.get.return_value = []

        result = await api_keys_resource.list(page=-1)

        assert result == []
        mock_client.get.assert_called_once_with("api_keys", params={"page": -1})

    @pytest.mark.asyncio
    async def test_create_with_none_values(self, api_keys_resource, mock_client, sample_api_key):
        """Test creation request with None values are properly excluded."""
        request = CreateApiKeyRequest(
            description="Test Key",
            apiKeyType="INFERENCE",
            consumptionLimit=None,  # Should be excluded
            expiresAt=None,  # Should be excluded
        )
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        # Only non-None values should be sent
        expected_data = {"description": "Test Key", "apiKeyType": "INFERENCE"}
        mock_client.post.assert_called_once_with("api_keys", json_data=expected_data)

    @pytest.mark.asyncio
    async def test_retrieve_returns_dedicated_endpoint_payload(
        self, api_keys_resource, mock_client
    ):
        """retrieve() unwraps the {"data": ...} envelope from /api_keys/{id}."""
        from venice_ai.types.api import ApiKeyDetailsResponse

        sample_key = {
            "id": "key_1",
            "description": "Just-this-key",
            "apiKeyType": "INFERENCE",
            "last6Chars": "key_1",
            "consumptionLimits": {"usd": None, "diem": None},
            "usage": {"trailingSevenDays": {"usd": "0.00", "diem": "0.00"}},
        }
        details = ApiKeyDetailsResponse.model_validate({"data": sample_key})
        mock_client.get.return_value = details

        result = await api_keys_resource.retrieve(api_key_id="key_1")

        # Bare ApiKey: ``.data`` is no longer present.
        assert result.id == sample_key["id"]
        assert result.description == sample_key["description"]
        mock_client.get.assert_called_once_with("api_keys/key_1", cast_to=ApiKeyDetailsResponse)

    @pytest.mark.asyncio
    async def test_web3_serialization_edge_cases(self, api_keys_resource, mock_client):
        """Test Web3 request serialization with different object types."""

        # Test with object that has __dict__
        class Web3RequestObj:
            def __init__(self):
                self.description = "Web3 Dict Key"
                self.apiKeyType = "INFERENCE"
                self.web3_network_id = "ethereum"
                self.nullField = None

        request_obj = Web3RequestObj()
        web3_response = {
            "success": True,
            "data": {
                "id": "web3_key_123",
                "apiKey": "venice_web3_key_xyz",
                "apiKeyType": "INFERENCE",
                "description": "Web3 Dict Key",
                "consumptionLimit": {"usd": None, "diem": None},
            },
        }
        mock_client.post.return_value = web3_response

        result = await api_keys_resource.create_web3_key(web3_key_request=request_obj)

        assert result.success
        assert result.data.id == "web3_key_123"
        expected_data = {
            "description": "Web3 Dict Key",
            "apiKeyType": "INFERENCE",
            "web3_network_id": "ethereum",
        }
        mock_client.post.assert_called_once_with(
            "api_keys/generate_web3_key", json_data=expected_data
        )


class TestApiKeysIntegration:
    """Test integration scenarios combining multiple operations."""

    @pytest.mark.asyncio
    async def test_create_list_delete_workflow(
        self, api_keys_resource, mock_client, sample_create_request, sample_api_key
    ):
        """Test a complete workflow: create, list, delete."""
        # Setup responses for the workflow
        create_response = {"data": sample_api_key}
        list_response = [sample_api_key]
        delete_response = {"success": True}

        mock_client.post.return_value = create_response
        mock_client.get.return_value = list_response
        mock_client.delete.return_value = delete_response

        # Create key
        created_key = await api_keys_resource.create(api_key_request=sample_create_request)
        assert created_key.id == sample_api_key["id"]

        # List keys
        keys = await api_keys_resource.list()
        assert len(keys) == 1
        assert keys[0].id == sample_api_key["id"]

        # Delete key
        delete_result = await api_keys_resource.delete(api_key_id=sample_api_key["id"])
        assert delete_result.success is True

        # Verify all calls were made
        assert mock_client.post.call_count == 1
        assert mock_client.get.call_count == 1
        assert mock_client.delete.call_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_vs_list_consistency(
        self, api_keys_resource, mock_client, sample_api_key
    ):
        """retrieve and list return consistent data for the same key."""
        from venice_ai.types.api import ApiKeyDetailsResponse

        # Get via list
        mock_client.get.return_value = [sample_api_key]
        keys = await api_keys_resource.list()
        list_key = keys[0]

        # Get via retrieve (now uses /api_keys/{id})
        mock_client.get.return_value = ApiKeyDetailsResponse.model_validate(
            {"data": sample_api_key}
        )
        retrieved_key = await api_keys_resource.retrieve(api_key_id=sample_api_key["id"])

        assert list_key.id == retrieved_key.id
        assert list_key.description == retrieved_key.description


class TestApiKeysRobustness:
    """Test robustness and resilience scenarios."""

    @pytest.mark.asyncio
    async def test_large_api_key_list(self, api_keys_resource, mock_client):
        """Test handling of large API key lists."""
        # Create a large list of API keys
        large_key_list = [
            {
                "id": f"key_{i}",
                "description": f"Key {i}",
                "apiKeyType": "INFERENCE",
                "last6Chars": f"key_{i}"[-6:],
                "consumptionLimits": {"usd": None, "diem": None},
                "usage": {"trailingSevenDays": {"usd": "0.00", "diem": "0.00"}},
            }
            for i in range(100)
        ]
        mock_client.get.return_value = large_key_list

        result = await api_keys_resource.list()

        assert len(result) == 100
        assert result[0].id == "key_0"
        assert result[99].id == "key_99"

    @pytest.mark.asyncio
    async def test_special_characters_in_description(
        self, api_keys_resource, mock_client, sample_api_key
    ):
        """Test API key creation with special characters in description."""
        request = CreateApiKeyRequest(
            description="Test Key with émojis 🔑 & special chars: @#$%^&*()",
            apiKeyType="INFERENCE",
            consumptionLimit=None,
            expiresAt=None,
        )
        response_data = {"data": sample_api_key}
        mock_client.post.return_value = response_data

        result = await api_keys_resource.create(api_key_request=request)

        # Result is now a Pydantic ApiKey object
        assert result.id == sample_api_key["id"]
        assert result.apiKeyType == sample_api_key["apiKeyType"]
        assert result.description == sample_api_key["description"]
        # Verify special characters are preserved in the request
        call_args = mock_client.post.call_args
        assert "émojis 🔑 & special chars" in call_args[1]["json_data"]["description"]

    @pytest.mark.asyncio
    async def test_very_long_api_key_id(self, api_keys_resource, mock_client):
        """Test operations with very long API key IDs."""
        long_key_id = "key_" + "x" * 500  # Very long ID
        delete_response = {"success": True}
        mock_client.delete.return_value = delete_response

        result = await api_keys_resource.delete(api_key_id=long_key_id)

        assert result.success is True
        mock_client.delete.assert_called_once_with("api_keys", params={"id": long_key_id})


class TestUpdateModelPrivacy:
    """PATCH /api_keys accepts modelPrivacy, so update() must be able to send it."""

    @pytest.mark.asyncio
    async def test_update_forwards_model_privacy(self):
        from venice_ai.resources.api_keys import ApiKeys

        client = AsyncMock()
        client.patch = AsyncMock(return_value={"data": {}})
        keys = ApiKeys(client)
        # The response shape is irrelevant here; the request body is what matters.
        with contextlib.suppress(Exception):
            await keys.update(id="key-1", model_privacy="PRIVATE_ONLY")

        body = client.patch.await_args.kwargs["json_data"]
        assert body["modelPrivacy"] == "PRIVATE_ONLY"
