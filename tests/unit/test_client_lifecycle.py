"""
Unit tests for VeniceClient lifecycle management.

Covers: scheduler initialization, session management, rate limiter injection,
error handling, error responses, streaming errors, and client close.
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
from pydantic import BaseModel

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseProcessingError,
    APIResponseValidationError,
    APITimeoutError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
)


class SampleTestModel(BaseModel):
    """Sample Pydantic model for validation tests."""

    id: str
    name: str
    value: int = 0


# =============================================================================
# Session Management
# =============================================================================


class TestSessionManagement:
    """Test session creation and loop validation."""

    @pytest.fixture
    def client(self):
        return VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")

    @pytest.mark.asyncio
    async def test_get_session_loop_mismatch(self, client):
        """Test _get_session raises RuntimeError on loop mismatch."""
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session._loop = MagicMock()
        client._session = mock_session

        with pytest.raises(RuntimeError, match="Session was created in a different event loop"):
            await client._get_session()

    @pytest.mark.asyncio
    async def test_get_session_no_http_client(self, client):
        """Test _get_session raises RuntimeError when http_client is missing."""
        client._session = None
        client._venice_http_client = None

        with pytest.raises(RuntimeError, match="VeniceHTTPClient not initialized"):
            await client._get_session()


# =============================================================================
# Scheduler and Rate Limiter
# =============================================================================


class TestSchedulerInitialization:
    """Test scheduler initialization logic."""

    @pytest.mark.asyncio
    async def test_ensure_rate_limiter_early_return(self):
        """Test early return when rate_limiter already exists."""
        client = VeniceClient(api_key="test")
        client.rate_limiter = Mock()
        original_rate_limiter = client.rate_limiter

        await client._ensure_rate_limiter_and_start()

        assert client.rate_limiter == original_rate_limiter

    @pytest.mark.asyncio
    async def test_ensure_rate_limiter_no_use(self):
        """Test when should not use rate limiter."""
        client = VeniceClient(api_key="test")
        client.rate_limiter = None

        with patch.object(client, "_should_use_rate_limiter", return_value=False):
            await client._ensure_rate_limiter_and_start()
            assert client.rate_limiter is None

    @pytest.mark.asyncio
    async def test_inject_rate_limiter_error(self):
        """Test _inject_rate_limiter raises error if already injected."""
        client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")
        client.rate_limiter = MagicMock()
        with pytest.raises(RuntimeError, match="Rate limiter already injected"):
            client._inject_rate_limiter(MagicMock())

    @pytest.mark.asyncio
    async def test_ensure_rate_limiter_scheduler_start(self):
        """Test _ensure_rate_limiter_and_start starts scheduler."""
        client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")
        client.rate_limiter = None
        mock_scheduler = MagicMock()
        mock_scheduler.is_running.return_value = False
        mock_scheduler.start = AsyncMock()

        client._scheduler_manager = mock_scheduler

        with patch.object(client, "_should_use_rate_limiter", return_value=True):
            await client._ensure_rate_limiter_and_start()

        mock_scheduler.start.assert_awaited_once()
        assert client.rate_limiter == mock_scheduler

    @pytest.mark.asyncio
    async def test_scheduler_intelligent_mode_initialization(self):
        """Test INTELLIGENT scheduler mode initialization."""
        from venice_ai.core.config import SchedulerConfig, SchedulerMode, VeniceAIConfig

        config = VeniceAIConfig(
            scheduler=SchedulerConfig(mode=SchedulerMode.INTELLIGENT, enable_rate_limiting=True)
        )

        client = VeniceClient(api_key="test", config=config)
        assert hasattr(client, "_scheduler_manager")


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling paths."""

    @pytest.mark.asyncio
    async def test_empty_response_with_cast_to_validation_error(self):
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content_length = 0
        mock_response.headers = {"content-type": "application/json"}

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            with pytest.raises(APIResponseValidationError) as exc_info:
                await client._request("GET", "/empty", cast_to=SampleTestModel, force_direct=True)

            assert "Expected SampleTestModel but received empty response" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_response_without_cast_to_returns_none(self):
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content_length = 0
        mock_response.headers = {"content-type": "application/json"}

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response
            result = await client._request("GET", "/empty", force_direct=True)
            assert result is None

    @pytest.mark.asyncio
    async def test_json_parsing_error_without_cast_to(self):
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content_length = 10
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read = AsyncMock(return_value=b"not json")
        mock_response.json = AsyncMock(side_effect=ValueError("Invalid JSON"))

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            with pytest.raises(APIResponseProcessingError):
                await client._request("GET", "/invalid-json", force_direct=True)

    @pytest.mark.asyncio
    async def test_json_parsing_error_with_cast_to(self):
        client = VeniceClient(api_key="test")

        mock_response = Mock()
        mock_response.ok = True
        mock_response.content_length = 10
        mock_response.headers = {"content-type": "application/json"}
        mock_response.read = AsyncMock(return_value=b"<html></html>")
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(request_info=MagicMock(), history=())
        )

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            with pytest.raises(APIResponseValidationError):
                await client._request(
                    "GET", "/invalid-json", cast_to=SampleTestModel, force_direct=True
                )

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}
        mock_session.request = AsyncMock(side_effect=TimeoutError("Request timeout"))

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(APITimeoutError),
        ):
            await client._prepare_and_send_request("GET", "/timeout", force_direct=True)

    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}
        mock_session.request = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=Mock(), os_error=OSError("Connection failed")
            )
        )

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(APIConnectionError),
        ):
            await client._prepare_and_send_request("GET", "/connect-fail", force_direct=True)

    @pytest.mark.asyncio
    async def test_general_client_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}
        mock_session.request = AsyncMock(side_effect=aiohttp.ClientError("General client error"))

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(APIConnectionError),
        ):
            await client._prepare_and_send_request("GET", "/client-error", force_direct=True)

    @pytest.mark.asyncio
    async def test_prepare_request_server_timeout(self):
        """Test handling of ServerTimeoutError."""
        client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")
        client._get_session = AsyncMock()
        session = await client._get_session()
        session.headers = {}
        session.request = AsyncMock(side_effect=aiohttp.ServerTimeoutError("Server timeout"))

        with pytest.raises(APITimeoutError, match="Server timeout during request"):
            await client._prepare_and_send_request(method="GET", path="/test", force_direct=True)

    @pytest.mark.asyncio
    async def test_response_text_parse_failure(self):
        """Test failure to parse text after JSON failure."""
        client = VeniceClient(api_key="test-api-key", base_url="https://api.test.venice.ai/api/v1")
        client._get_session = AsyncMock()
        session = await client._get_session()
        session.headers = {}

        mock_response = MagicMock(spec=aiohttp.ClientResponse)
        mock_response.ok = False
        mock_response.status = 500
        mock_response.json = AsyncMock(side_effect=aiohttp.ContentTypeError(MagicMock(), ()))
        mock_response.text = AsyncMock(side_effect=aiohttp.ClientError("Text parse failed"))

        session.request = AsyncMock(return_value=mock_response)

        with pytest.raises(APIError) as exc:
            await client._prepare_and_send_request(method="GET", path="/test", force_direct=True)

        assert "text parsing failed" in str(exc.value.body)

    @pytest.mark.asyncio
    async def test_request_error_without_scheduler(self):
        """Test error handling path without scheduler."""
        client = VeniceClient(api_key="test")

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            error = aiohttp.ClientResponseError(
                request_info=Mock(), history=(), status=400, message="Bad Request"
            )
            mock_prepare.side_effect = error

            with pytest.raises(aiohttp.ClientResponseError):
                await client._request("GET", "/test")


# =============================================================================
# Error Response Handling
# =============================================================================


class TestErrorResponseHandling:
    """Test error response handling paths."""

    @pytest.mark.asyncio
    async def test_error_response_json_parsing_success(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}

        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 400
        mock_response.content_length = 10
        mock_response.headers = {}
        mock_response.json = AsyncMock(return_value={"error": "Bad Request"})

        mock_session.request = AsyncMock(return_value=mock_response)

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(InvalidRequestError),
        ):
            await client._prepare_and_send_request("GET", "/error", force_direct=True)

    @pytest.mark.asyncio
    async def test_error_response_json_parsing_failure_fallback_to_text(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}

        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 500
        mock_response.content_length = 10
        mock_response.headers = {}
        mock_response.json = AsyncMock(
            side_effect=aiohttp.ContentTypeError(request_info=MagicMock(), history=())
        )
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session.request = AsyncMock(return_value=mock_response)

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(InternalServerError),
        ):
            await client._prepare_and_send_request("GET", "/error", force_direct=True)

        mock_response.json.assert_called_once()
        mock_response.text.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_response_empty_body_handling(self):
        client = VeniceClient(api_key="test")

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_session.headers = {}

        mock_response = AsyncMock()
        mock_response.ok = False
        mock_response.status = 404
        mock_response.content_length = 0
        mock_response.headers = {}

        mock_session.request = AsyncMock(return_value=mock_response)

        with (
            patch.object(client, "_get_session", return_value=mock_session),
            pytest.raises(NotFoundError),
        ):
            await client._prepare_and_send_request("GET", "/not-found", force_direct=True)


# =============================================================================
# Streaming Error Handling
# =============================================================================


class TestStreamingErrorHandling:
    """Test streaming response error handling."""

    @pytest.mark.asyncio
    async def test_streaming_vcr_compatibility_with_body_string(self):
        client = VeniceClient(api_key="test")

        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.content.read = AsyncMock(return_value=b"")
        mock_response.close = Mock()

        mock_vcr_body = Mock()
        mock_vcr_body.string = b'data: {"id": "1", "name": "test", "value": 1}\n'
        mock_response._body = mock_vcr_body

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            results = []
            async for item in client._stream_request("GET", "/stream", cast_to=SampleTestModel):
                results.append(item)

            assert len(results) == 1
            assert results[0].id == "1"
            assert results[0].name == "test"

    @pytest.mark.asyncio
    async def test_streaming_no_content_warning(self):
        client = VeniceClient(api_key="test")

        mock_response = AsyncMock()
        mock_response.ok = True
        mock_response.content.read = AsyncMock(return_value=b"")
        mock_response._body = None
        mock_response.close = Mock()

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            results = []
            async for item in client._stream_request(
                "GET", "/empty-stream", cast_to=SampleTestModel
            ):
                results.append(item)

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_streaming_json_decode_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_response = AsyncMock()
        mock_response.ok = True
        invalid_content = (
            b'data: {"invalid": json}\ndata: {"id": "1", "name": "test", "value": 1}\n'
        )
        mock_response.content.read = AsyncMock(return_value=invalid_content)
        mock_response.close = Mock()

        with patch.object(
            client, "_prepare_and_send_request", new_callable=AsyncMock
        ) as mock_prepare:
            mock_prepare.return_value = mock_response

            results = []
            async for item in client._stream_request("GET", "/stream", cast_to=SampleTestModel):
                results.append(item)

            assert len(results) == 1
            assert results[0].id == "1"


# =============================================================================
# Client Close
# =============================================================================


class TestClientClose:
    """Test client close method error handling."""

    @pytest.mark.asyncio
    async def test_close_venice_http_client_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_venice_client = AsyncMock()
        mock_venice_client.close = AsyncMock(side_effect=RuntimeError("Close failed"))
        client._venice_http_client = mock_venice_client
        client._should_close_session = False

        await client.close()

        mock_venice_client.close.assert_called_once()
        assert client._is_closed is True

    @pytest.mark.asyncio
    async def test_close_scheduler_error_handling(self):
        client = VeniceClient(api_key="test")

        mock_scheduler = AsyncMock()
        mock_scheduler.stop = AsyncMock(side_effect=OSError("Stop failed"))
        client._scheduler_manager = mock_scheduler
        client._venice_http_client = None
        client._should_close_session = False

        await client.close()

        mock_scheduler.stop.assert_called_once()
        assert client._is_closed is True

    @pytest.mark.asyncio
    async def test_close_already_closed_early_return(self):
        client = VeniceClient(api_key="test")
        client._is_closed = True

        mock_scheduler = AsyncMock()
        client._scheduler_manager = mock_scheduler

        await client.close()


# =============================================================================
# Client Init from coverage_improvements_final.py
# =============================================================================


class TestVeniceClientInit:
    """Additional VeniceClient init tests from coverage improvements."""

    def test_init_missing_api_key_with_none_env(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="No authentication provided"),
        ):
            VeniceClient(api_key=None)

    def test_init_whitespace_only_api_key(self):
        with pytest.raises(ValueError, match="No authentication provided"):
            VeniceClient(api_key="   ")

    def test_init_with_rate_limiter_config_conflict(self):
        config = {"requests_per_second": 10}
        with pytest.raises(ValueError, match="Cannot provide both"):
            VeniceClient(
                api_key="test",
                rate_limiter_config=config,
                rate_limiter_config_path="config.yaml",
            )

    def test_init_with_http_client_and_rate_limiter(self):
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        config = {"enabled": False}
        client = VeniceClient(api_key="test", http_client=mock_client, rate_limiter_config=config)
        assert client._rate_limiter_config == config

    def test_should_use_rate_limiter_with_env_var(self):
        with patch.dict(os.environ, {"VENICE_RATE_LIMITER_FEATURES_ENABLED": "true"}):
            client = VeniceClient(api_key="test")
            assert client._should_use_rate_limiter() is True

        with patch.dict(os.environ, {"VENICE_RATE_LIMITER_FEATURES_ENABLED": "false"}):
            client = VeniceClient(api_key="test")
            assert client._should_use_rate_limiter() is False

    def test_should_use_rate_limiter_with_external_client(self):
        mock_client = MagicMock(spec=aiohttp.ClientSession)
        client = VeniceClient(api_key="test", http_client=mock_client)
        assert client._should_use_rate_limiter() is False

    def test_base_url_trailing_slash_normalization(self):
        client1 = VeniceClient(api_key="test", base_url="https://api.example.com")
        client2 = VeniceClient(api_key="test", base_url="https://api.example.com/")

        assert str(client1._base_url).endswith("/")
        assert str(client2._base_url).endswith("/")
        assert str(client1._base_url) == str(client2._base_url)
