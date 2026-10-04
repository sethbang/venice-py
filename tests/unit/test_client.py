"""
Unit tests for VeniceClient core functionality.

Consolidated from test_client.py, test_client_branch_coverage.py,
test_client_coverage_improvements.py, and test_client_expanded_coverage.py.

This file covers: initialization, request/response processing, HTTP operations,
form serialization, streaming, and VCR compatibility.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from pydantic import BaseModel

from venice_ai import VeniceClient
from venice_ai.core.config import (
    HttpClientConfig,
    VeniceAIConfig,
)
from venice_ai.exceptions import (
    APIResponseProcessingError,
    APIResponseValidationError,
)
from venice_ai.utils import serialize_form_value


class SampleModel(BaseModel):
    """Sample Pydantic model for testing."""

    id: str
    status: str
    data: dict = {}


class SampleTestModel(BaseModel):
    """Sample Pydantic model for validation tests."""

    id: str
    name: str
    value: int = 0


class AsyncIteratorWrapper:
    """Helper for async iteration in tests."""

    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        self.iter = iter(self.items)
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration as e:
            raise StopAsyncIteration from e


# =============================================================================
# Form Value Serialization
# =============================================================================


class TestFormValueSerialization:
    """Test form value serialization function."""

    def test_serialize_boolean_true(self):
        assert serialize_form_value(True) == "true"

    def test_serialize_boolean_false(self):
        assert serialize_form_value(False) == "false"

    def test_serialize_string(self):
        assert serialize_form_value("test") == "test"
        assert serialize_form_value("") == ""

    def test_serialize_number(self):
        assert serialize_form_value(123) == "123"
        assert serialize_form_value(0) == "0"
        assert serialize_form_value(3.14) == "3.14"

    def test_serialize_none(self):
        assert serialize_form_value(None) == "None"


# =============================================================================
# Client Initialization
# =============================================================================


class TestClientInitialization:
    """Test client initialization paths."""

    def test_init_with_invalid_http_client_type(self):
        """Test initialization with invalid http_client type."""
        with pytest.raises(TypeError) as exc_info:
            VeniceClient(api_key="test", http_client="not_a_session")  # type: ignore[arg-type]

        assert "http_client must be an aiohttp.ClientSession" in str(exc_info.value)

    def test_init_with_valid_http_client(self):
        """Test initialization with valid aiohttp.ClientSession."""
        mock_session = Mock(spec=aiohttp.ClientSession)
        client = VeniceClient(api_key="test", http_client=mock_session)

        assert client._session == mock_session
        assert client._should_close_session is False

    def test_client_init_with_venice_ai_config(self):
        """Test VeniceClient initialization with VeniceAIConfig object."""
        config = VeniceAIConfig(
            http_client=HttpClientConfig(max_connections=100, max_keepalive_connections=20)
        )

        client = VeniceClient(api_key="test-api-key", config=config)

        assert client is not None
        assert client._venice_http_client is not None
        assert client._should_close_session is True

        asyncio.run(client.close())

    def test_init_with_int_timeout_is_honored(self):
        """Passing ``timeout=240`` (an int) must set the client's effective timeout to
        240 seconds — not silently fall through to DEFAULT_TIMEOUT (120s).

        Regression for outside-agent feedback: ``isinstance(effective_timeout, float)``
        rejected ints, so callers got a stricter timeout than they asked for and saw
        unexplained mid-run aborts.
        """
        client = VeniceClient(api_key="test-api-key", timeout=240)
        assert isinstance(client._timeout, aiohttp.ClientTimeout)
        assert client._timeout.total == 240
        assert client.timeout == 240

    def test_init_with_int_default_timeout_is_honored(self):
        """The ``default_timeout`` kwarg must accept ints too, for symmetry."""
        client = VeniceClient(api_key="test-api-key", default_timeout=300)
        assert client.timeout == 300

    def test_init_with_unsupported_timeout_type_falls_back(self):
        """Garbage timeout values still fall through to DEFAULT_TIMEOUT — the int
        coercion change must not break the ``unknown -> default`` safety net."""
        client = VeniceClient(api_key="test-api-key", timeout="not-a-number")  # type: ignore[arg-type]
        assert client._timeout.total == 120.0


# =============================================================================
# Request/Response Processing
# =============================================================================


class TestRequestResponse:
    """Test VeniceClient core request/response functionality."""

    @pytest.fixture
    def mock_client_session(self):
        session = AsyncMock(spec=aiohttp.ClientSession)
        session.headers = {}
        return session

    @pytest.fixture
    def mock_response(self):
        response = AsyncMock(spec=aiohttp.ClientResponse)
        response.ok = True
        response.status = 200
        response.headers = {"content-type": "application/json"}
        response.content_length = 100
        return response

    @pytest.fixture
    def client(self):
        return VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")

    @pytest.mark.asyncio
    async def test_process_non_stream_response(self, client, mock_client_session, mock_response):
        """Test non-streaming JSON response processing."""
        sample_json_data = {
            "id": "123",
            "status": "completed",
            "data": {"result": "success"},
        }
        mock_response.json = AsyncMock(return_value=sample_json_data)
        mock_response.headers = {"content-type": "application/json"}

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request(method="GET", path="/test", force_direct=True)

        assert result == sample_json_data
        mock_response.json.assert_called_once()

        mock_client_session.request.assert_called_once()
        call_kwargs = mock_client_session.request.call_args.kwargs
        assert call_kwargs["method"] == "GET"
        assert "/test" in str(call_kwargs["url"])

    @pytest.mark.asyncio
    async def test_process_non_stream_response_with_model_validation(
        self, client, mock_client_session, mock_response
    ):
        """Test non-streaming response with Pydantic model validation."""
        sample_json_data = {"id": "456", "status": "pending", "data": {"info": "test"}}
        mock_response.json = AsyncMock(return_value=sample_json_data)
        mock_response.headers = {"content-type": "application/json"}

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request(
                method="POST",
                path="/test-model",
                json_data={"query": "test"},
                cast_to=SampleModel,
                force_direct=True,
            )

        assert isinstance(result, SampleModel)
        assert result.id == "456"
        assert result.status == "pending"
        assert result.data == {"info": "test"}
        assert hasattr(result, "_response")
        assert result._response == mock_response  # pyright: ignore[reportAttributeAccessIssue]

    @pytest.mark.asyncio
    async def test_process_non_stream_response_empty_content(
        self, client, mock_client_session, mock_response
    ):
        """Test handling of empty responses (content_length == 0) without cast_to."""
        mock_response.content_length = 0
        mock_response.headers = {"content-type": "application/json"}

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request(method="GET", path="/empty", force_direct=True)

        assert result is None
        assert not mock_response.json.called

    @pytest.mark.asyncio
    async def test_process_non_stream_response_empty_content_with_cast_to(
        self, client, mock_client_session, mock_response
    ):
        """Test empty responses with cast_to raise validation error."""
        mock_response.content_length = 0
        mock_response.headers = {"content-type": "application/json"}

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            with pytest.raises(APIResponseValidationError) as exc_info:
                await client._request(
                    method="GET", path="/empty-with-model", cast_to=SampleModel, force_direct=True
                )

            assert "Expected SampleModel but received empty response" in str(exc_info.value)
            assert exc_info.value.model_name == "SampleModel"

    @pytest.mark.asyncio
    async def test_process_non_stream_response_json_parse_error(
        self, client, mock_client_session, mock_response
    ):
        """Test handling of JSON parsing errors."""
        mock_response.content_length = 50
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(request_info=MagicMock(), history=())
        )

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            with pytest.raises(APIResponseProcessingError) as exc_info:
                await client._request(method="GET", path="/invalid-json", force_direct=True)

            assert "Failed to parse JSON response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_non_stream_response_validation_error(
        self, client, mock_client_session, mock_response
    ):
        """Test handling of Pydantic validation errors."""
        invalid_json_data = {"invalid_field": "value"}
        mock_response.json = AsyncMock(return_value=invalid_json_data)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content_length = 50

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            with pytest.raises(APIResponseValidationError) as exc_info:
                await client._request(
                    method="POST",
                    path="/invalid-model-data",
                    json_data={"test": "data"},
                    cast_to=SampleModel,
                    force_direct=True,
                )

            assert "API response validation failed for SampleModel" in str(exc_info.value)
            assert exc_info.value.model_name == "SampleModel"
            assert exc_info.value.response_data == invalid_json_data

    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming_detection(self, client, mock_client_session):
        """Test content-type based streaming vs non-streaming detection."""
        non_stream_response = AsyncMock(spec=aiohttp.ClientResponse)
        non_stream_response.ok = True
        non_stream_response.headers = {"content-type": "application/json"}
        non_stream_response.content_length = 50
        non_stream_response.json = AsyncMock(return_value={"type": "non_stream"})

        mock_client_session.request = AsyncMock(return_value=non_stream_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request("GET", "/non-stream", force_direct=True)

        assert result == {"type": "non_stream"}
        assert not hasattr(result, "__aiter__")

        stream_response = AsyncMock(spec=aiohttp.ClientResponse)
        stream_response.ok = True
        stream_response.headers = {"content-type": "text/event-stream"}
        stream_response.content_length = 50

        mock_client_session.request = AsyncMock(return_value=stream_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request("GET", "/stream", cast_to=SampleModel, force_direct=True)

        from venice_ai.streaming import Stream

        assert isinstance(result, Stream)

    @pytest.mark.asyncio
    async def test_build_request_with_extra_params(
        self, client, mock_client_session, mock_response
    ):
        """Test merging of extra headers, query, and body parameters."""
        sample_json_data = {"id": "test-123", "status": "success", "data": {"merged": True}}
        mock_response.json = AsyncMock(return_value=sample_json_data)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content_length = 100

        mock_client_session.request = AsyncMock(return_value=mock_response)

        extra_headers = {"X-Custom-Header": "custom-value", "X-Test-ID": "test-123"}
        extra_query = {"filter": "active", "limit": "10"}
        extra_body = {"metadata": {"source": "test"}, "options": {"validate": True}}

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request(
                method="POST",
                path="/test-endpoint",
                headers=extra_headers,
                params=extra_query,
                json_data=extra_body,
                force_direct=True,
            )

        assert result == sample_json_data

        call_kwargs = mock_client_session.request.call_args.kwargs
        assert call_kwargs["method"] == "POST"
        request_headers = call_kwargs.get("headers", {})
        assert request_headers["X-Custom-Header"] == "custom-value"
        request_params = call_kwargs.get("params", {})
        assert request_params == extra_query
        request_json = call_kwargs.get("json", {})
        assert request_json == extra_body

    @pytest.mark.asyncio
    async def test_build_request_without_extra_params(
        self, client, mock_client_session, mock_response
    ):
        """Test request without extra parameters."""
        sample_json_data = {"id": "no-extras", "status": "success"}
        mock_response.json = AsyncMock(return_value=sample_json_data)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.content_length = 50

        mock_client_session.request = AsyncMock(return_value=mock_response)

        with patch.object(client, "_get_session", return_value=mock_client_session):
            result = await client._request(method="GET", path="/no-extras", force_direct=True)

        assert result == sample_json_data

        call_kwargs = mock_client_session.request.call_args.kwargs
        assert call_kwargs["method"] == "GET"
        assert call_kwargs.get("params") is None
        assert call_kwargs.get("json") is None


# =============================================================================
# HTTP Operations
# =============================================================================


class TestHTTPOperations:
    """Test HTTP convenience methods."""

    @pytest.mark.asyncio
    async def test_delete_method_basic(self):
        client = VeniceClient(api_key="test")

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"deleted": True}
            result = await client.delete("/test", force_direct=True)
            mock_request.assert_called_once_with("DELETE", "/test", cast_to=None, force_direct=True)
            assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_delete_method_with_cast_to(self):
        client = VeniceClient(api_key="test")

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = SampleTestModel(id="1", name="test")
            await client.delete("/test", cast_to=SampleTestModel, force_direct=True)
            mock_request.assert_called_once_with(
                "DELETE", "/test", cast_to=SampleTestModel, force_direct=True
            )

    @pytest.mark.asyncio
    async def test_put_method(self):
        client = VeniceClient(api_key="test")

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"success": True}
            await client.put("/test", json_data={"key": "value"})
            mock_request.assert_called_once()
            args = mock_request.call_args
            assert args[0][0] == "PUT"
            assert args[0][1] == "/test"

    @pytest.mark.asyncio
    async def test_file_upload_form_data_creation(self):
        """Test file upload FormData creation."""
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content_length = 10
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read = AsyncMock(return_value=b"{}")
        mock_response.json = AsyncMock(return_value={"uploaded": True})

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response
            files = {"file1": ("test.txt", b"file content", "text/plain")}
            data = {"name": "test", "enabled": True}
            await client._request("POST", "/upload", files=files, data=data, force_direct=True)

            call_kwargs = mock_prepare.call_args[1]
            assert isinstance(call_kwargs["data"], aiohttp.FormData)


# =============================================================================
# Client Properties and Expanded Coverage
# =============================================================================


class TestClientExpanded:
    """Tests to expand coverage for VeniceClient."""

    @pytest.fixture
    def client(self):
        return VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")

    @pytest.mark.asyncio
    async def test_client_properties_and_headers(self, client):
        """Test client properties and get_headers."""
        assert client.base_url.rstrip("/") == "https://api.test.venice.ai/api/v1"
        assert client.timeout == 120.0

        client._timeout = aiohttp.ClientTimeout(total=30.0)
        assert client.timeout == 30.0

        assert client.get_headers() == {}

        client._headers = {"X-Custom": "value"}
        assert client.get_headers() == {"X-Custom": "value"}

    @pytest.mark.asyncio
    async def test_prepare_request_model_from_params(self, client):
        """Test extracting model from params."""
        client.rate_limiter = MagicMock()
        client.rate_limiter.is_running.return_value = True
        client.rate_limiter.submit_request = AsyncMock()
        client.rate_limiter.classifier = None

        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.ok = True
        client.rate_limiter.submit_request.return_value = mock_response

        client._get_session = AsyncMock()

        await client._prepare_and_send_request(
            method="GET", path="/test", params={"model": "test-model"}, force_direct=False
        )

        call_args = client.rate_limiter.submit_request.call_args
        metadata = call_args[0][0]
        assert metadata.model_id == "test-model"

    @pytest.mark.asyncio
    async def test_prepare_request_with_classifier(self, client):
        """Test request classification."""
        client.rate_limiter = MagicMock()
        client.rate_limiter.is_running.return_value = True
        client.rate_limiter.classifier = AsyncMock()
        client.rate_limiter.classifier.classify = AsyncMock(return_value="mock_metadata")

        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.ok = True
        client.rate_limiter.submit_request = AsyncMock(return_value=mock_response)

        client._get_session = AsyncMock()

        await client._prepare_and_send_request(
            method="POST", path="/test", json_data={"model": "test-model"}, force_direct=False
        )

        client.rate_limiter.classifier.classify.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prepare_request_await_future(self, client):
        """Test awaiting queued request future."""
        client.rate_limiter = MagicMock()
        client.rate_limiter.is_running.return_value = True
        client.rate_limiter.classifier = None

        mock_future = asyncio.Future()
        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.ok = True
        mock_future.set_result(mock_response)

        mock_request = MagicMock()
        mock_request.future = mock_future

        mock_result = MagicMock()
        mock_result.request = mock_request

        client.rate_limiter.submit_request = AsyncMock(return_value=mock_result)

        response = await client._prepare_and_send_request(
            method="GET", path="/test", force_direct=False
        )

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_prepare_request_redact_api_key(self, client):
        """Test redaction of X-API-Key in debug logs."""
        with patch("venice_ai._client.logger") as mock_logger:
            client._get_session = AsyncMock()
            session = await client._get_session()
            session.headers = {"X-API-Key": "secret"}
            session.request = AsyncMock(return_value=MagicMock(ok=True))

            await client._prepare_and_send_request(method="GET", path="/test", force_direct=True)

            debug_calls = mock_logger.debug.call_args_list
            found_redacted = any(
                "Request headers" in str(call[0][0]) and "[REDACTED]" in str(call[0][0])
                for call in debug_calls
            )
            assert found_redacted

    @pytest.mark.asyncio
    async def test_prepare_request_numeric_timeout(self, client):
        """Test numeric timeout handling."""
        client._get_session = AsyncMock()
        session = await client._get_session()
        session.headers = {}
        session.request = AsyncMock(return_value=MagicMock(ok=True))

        await client._prepare_and_send_request(
            method="GET", path="/test", timeout=10.5, force_direct=True
        )

        call_kwargs = session.request.call_args[1]
        assert isinstance(call_kwargs["timeout"], aiohttp.ClientTimeout)
        assert call_kwargs["timeout"].total == 10.5

    @pytest.mark.asyncio
    async def test_request_without_cast_to(self):
        """Test request without cast_to returns raw dict."""
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.headers = {}
        mock_response.read = AsyncMock(return_value=b"{}")
        mock_response.json = AsyncMock(return_value={"result": "data"})

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response
            result = await client._request("GET", "/test", cast_to=None)
            assert result == {"result": "data"}

    @pytest.mark.asyncio
    async def test_stream_request_vcr_coroutine_content(self, client):
        """Test VCR compatibility with coroutine content."""
        client._prepare_and_send_request = AsyncMock()
        mock_response = MagicMock()

        async def async_string():
            return b"data: {}\n\n"

        mock_vcr_body = MagicMock()
        mock_vcr_body.string = async_string()

        mock_response._body = mock_vcr_body
        mock_response.content.read = AsyncMock(return_value=None)
        client._prepare_and_send_request.return_value = mock_response

        client._process_stream_line = MagicMock(
            return_value=AsyncIteratorWrapper([{"test": "data"}])
        )

        class DummyModel:
            pass

        gen = client._stream_request("GET", "/test", cast_to=DummyModel)
        async for _ in gen:
            pass

    @pytest.mark.asyncio
    async def test_stream_request_vcr_string_content(self, client):
        """Test VCR compatibility with string content."""
        client._prepare_and_send_request = AsyncMock()
        mock_response = MagicMock()
        mock_response._body = "data: {}\n\n"
        mock_response.content.read = AsyncMock(return_value=None)
        client._prepare_and_send_request.return_value = mock_response

        client._process_stream_line = MagicMock(
            return_value=AsyncIteratorWrapper([{"test": "data"}])
        )

        class DummyModel:
            pass

        gen = client._stream_request("GET", "/test", cast_to=DummyModel)
        async for _ in gen:
            pass

    @pytest.mark.asyncio
    async def test_process_stream_line_dict_cast(self, client):
        """Test stream processing with dict cast."""

        class DictModel(dict):
            pass

        gen = client._process_stream_line('data: {"key": "value"}', DictModel)
        result = [item async for item in gen]
        assert result[0] == {"key": "value"}


def test_http_client_config_user_agent_uses_package_version() -> None:
    """The default User-Agent must track the installed package version, not a
    hardcoded literal (which drifted from the real version, e.g. on rc builds)."""
    import venice_ai

    assert HttpClientConfig().user_agent == f"venice-py/{venice_ai.__version__}"
