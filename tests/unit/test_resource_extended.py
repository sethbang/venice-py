"""
Comprehensive tests for src/venice_ai/_resource.py module.

This test file focuses on achieving >80% coverage for the APIResource class,
testing critical functionality including multipart uploads, error handling,
and header management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from venice_ai._client import VeniceClient
from venice_ai._resource import APIResource


class MockClient:
    """Mock client for testing APIResource.

    Header building and SIWE re-sign scoping are the real client's, so the
    multipart path is exercised with the headers a real request carries.
    """

    _request_headers = VeniceClient._request_headers
    _siwe_resigning = VeniceClient._siwe_resigning
    _resolve_siwe_resigner = VeniceClient._resolve_siwe_resigner
    _default_siwe_header = VeniceClient._default_siwe_header

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self._auth = None
        self._headers = None
        self._timeout = aiohttp.ClientTimeout(total=30.0)
        self._base_url = MagicMock()
        self._base_url.__truediv__ = MagicMock(return_value="https://api.test.venice.ai/test-path")
        self._session = None

    async def _get_session(self):
        if not self._session:
            self._session = AsyncMock()
            self._session.headers = {}
            self._session.timeout = aiohttp.ClientTimeout(total=30.0)
        return self._session


@pytest.fixture
def mock_client():
    """Create a mock client for testing."""
    return MockClient()


@pytest.fixture
def api_resource(mock_client):
    """Create an APIResource instance for testing."""
    return APIResource(mock_client)


@pytest.fixture
def mock_response():
    """Create a mock aiohttp response."""
    response = AsyncMock()
    response.status = 200
    response.headers = {}
    response.json = AsyncMock(return_value={"success": True, "data": "test"})
    response.text = AsyncMock(return_value='{"success": true}')
    response.raise_for_status = MagicMock()
    return response


class TestAPIResourceMultipart:
    """Test multipart request functionality."""

    @pytest.mark.asyncio
    async def test_request_multipart_basic(self, api_resource, mock_client):
        """Test basic multipart request with simple file."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"image": b"fake image data"}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/test-path", files=files)

        assert result == {"success": True}
        mock_session.request.assert_called_once()
        call_args = mock_session.request.call_args
        assert call_args[1]["method"] == "POST"
        assert "Authorization" in call_args[1]["headers"]
        assert call_args[1]["headers"]["Accept"] == "application/json"

    @pytest.mark.asyncio
    async def test_request_multipart_float_timeout_normalized(self, api_resource, mock_client):
        """A bare float timeout is wrapped in aiohttp.ClientTimeout(total=...) and passed through."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)
        mock_client._session = mock_session

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}
            await api_resource._request_multipart(
                "POST", "/upload", files={"image": b"x"}, timeout=42.0
            )

        passed = mock_session.request.call_args[1]["timeout"]
        assert isinstance(passed, aiohttp.ClientTimeout)
        assert passed.total == 42.0

    @pytest.mark.asyncio
    async def test_request_multipart_client_timeout_passthrough(self, api_resource, mock_client):
        """A ClientTimeout instance is forwarded unchanged; omitting it sends the client's."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)
        mock_client._session = mock_session

        explicit = aiohttp.ClientTimeout(total=12.5)
        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}
            await api_resource._request_multipart(
                "POST", "/upload", files={"image": b"x"}, timeout=explicit
            )
            assert mock_session.request.call_args[1]["timeout"] is explicit

            # No timeout arg -> the client's timeout, as on every other request path.
            mock_session.request.reset_mock()
            await api_resource._request_multipart("POST", "/upload", files={"image": b"x"})
            assert mock_session.request.call_args[1]["timeout"] is mock_client._timeout

    @pytest.mark.asyncio
    async def test_request_multipart_tuple_3_elements_bytes(self, api_resource, mock_client):
        """Test multipart with tuple format (filename, content, content_type) using bytes."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"image": ("test.jpg", b"fake image bytes", "image/jpeg")}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/upload", files=files)

        assert result == {"success": True}
        mock_session.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_multipart_tuple_3_elements_file_like(self, api_resource, mock_client):
        """Test multipart with tuple format using file-like object."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        # Create a file-like object
        file_like = MagicMock()
        file_like.read = MagicMock(return_value=b"file content")

        files = {"document": ("document.pdf", file_like, "application/pdf")}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/upload", files=files)

        assert result == {"success": True}
        mock_session.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_multipart_tuple_2_elements(self, api_resource, mock_client):
        """Test multipart with tuple format (filename, content)."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": ("test.txt", "text content")}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/upload", files=files)

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_request_multipart_with_additional_data(self, api_resource, mock_client):
        """Test multipart request with additional form data."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"image": b"image data"}
        data = {"param1": "value1", "param2": 42}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart(
                "POST", "/upload", files=files, data=data
            )

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_request_multipart_with_custom_headers(self, api_resource, mock_client):
        """Test multipart request with custom headers."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()

        # Properly mock headers.get() method
        mock_headers = MagicMock()
        mock_headers.get = MagicMock(
            side_effect=lambda key, default="": {"Content-Type": "application/json"}.get(
                key, default
            )
        )
        mock_response.headers = mock_headers

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": b"content"}
        headers = {
            "Custom-Header": "custom-value",
            "Authorization": "Bearer existing-auth",  # Should be preserved
        }

        result = await api_resource._request_multipart(
            "POST", "/upload", files=files, headers=headers
        )

        assert result == {"success": True}
        # Verify custom headers were used
        call_args = mock_session.request.call_args
        assert call_args[1]["headers"]["Custom-Header"] == "custom-value"
        assert call_args[1]["headers"]["Authorization"] == "Bearer existing-auth"

    @pytest.mark.asyncio
    async def test_request_multipart_missing_auth_header(self, api_resource, mock_client):
        """The client's key is attached when the call passes no Authorization header."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": b"content"}

        result = await api_resource._request_multipart("POST", "/upload", files=files)

        assert result == {"success": True}
        sent = mock_session.request.call_args.kwargs["headers"]
        assert sent["Authorization"] == "Bearer test-key"


class TestAPIResourceEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multipart_empty_files_dict(self, api_resource, mock_client):
        """Test multipart request with empty files dictionary."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {}  # Empty files dict

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/upload", files=files)

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_multipart_none_data(self, api_resource, mock_client):
        """Test multipart request with None data parameter."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": b"content"}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart(
                "POST", "/upload", files=files, data=None
            )

        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_accept_header_already_present(self, api_resource, mock_client):
        """Test that existing Accept header is preserved in multipart."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"success": True})
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "application/json"}

        # Create proper async context manager mock
        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": b"content"}
        headers = {"Accept": "text/plain"}  # Custom Accept header

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart(
                "POST", "/upload", files=files, headers=headers
            )

        assert result == {"success": True}
        call_args = mock_session.request.call_args

        # Should preserve custom Accept header
        assert call_args[1]["headers"]["Accept"] == "text/plain"

    @pytest.mark.asyncio
    async def test_request_multipart_text_response_returns_bytes(self, api_resource, mock_client):
        """Server-returned ``Content-Type: text/plain`` must not crash with
        ContentTypeError — _request_multipart should return raw bytes for text/*
        responses so callers can decode safely (regression test for S6)."""
        mock_session = AsyncMock()
        mock_session.headers = {}
        mock_response = AsyncMock()
        mock_response.read = AsyncMock(return_value=b"plain extracted text")
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"Content-Type": "text/plain; charset=utf-8"}
        mock_response.ok = True

        async def mock_aenter(self):
            return mock_response

        async def mock_aexit(self, *args):
            return None

        context_manager = MagicMock()
        context_manager.__aenter__ = mock_aenter
        context_manager.__aexit__ = mock_aexit
        mock_session.request = MagicMock(return_value=context_manager)

        mock_client._session = mock_session

        files = {"file": b"some bytes"}

        with patch("venice_ai.core.auth.create_auth_headers") as mock_auth:
            mock_auth.return_value = {"Authorization": "Bearer test-key"}

            result = await api_resource._request_multipart("POST", "/text-parser", files=files)

        assert result == b"plain extracted text"
        # response.json() must NOT have been called
        mock_response.json.assert_not_called()
