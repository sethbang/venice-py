"""
VCRpy-based integration tests for VeniceHTTPClient functionality.

This module tests HTTP client session management, request methods, and connection
handling through real API interactions recorded with VCRpy, replacing complex
mock-based unit tests with actual HTTP behavior verification.
"""

import asyncio
import os

import pytest
import pytest_asyncio

from venice_ai import VeniceClient, create_test_venice_client
from venice_ai.core.config import SchedulerMode, VeniceAIConfig
from venice_ai.core.http_client import VeniceHTTPClient
from venice_ai.exceptions import VeniceError


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing with shared rate limit coordination."""
    api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=True,
    )
    try:
        yield client
    finally:
        await client.close()


@pytest_asyncio.fixture
async def http_client():
    """Create a VeniceHTTPClient for direct testing."""
    api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")
    config = VeniceAIConfig.create_test_config()

    client = VeniceHTTPClient(config=config, api_key=api_key)
    try:
        yield client
    finally:
        await client.close()


# model_selector fixture is now provided by the root conftest.py


# ============================================================================
# Session Management and Configuration Tests
# ============================================================================


@pytest.mark.integration
async def test_http_client_session_creation(http_client, vcr_cassette):
    """
    Test HTTP client session creation and configuration.
    Replaces mock-based session building tests.
    """
    with vcr_cassette:
        # Test session creation by making an actual request
        session = await http_client.get_session()

        # Verify session properties
        assert session is not None
        assert not http_client.is_closed

        # The session resolves relative paths against the API root, so a live
        # run reaches /api/v1/models and gets 200 (a path off the root 404s).
        # vcrpy records and replays the URL as passed ("models"), so a replay
        # can only check the recorded status, not the resolved path.
        async with session.get("models") as response:
            assert response.status == 200


@pytest.mark.integration
async def test_http_client_with_custom_headers(vcr_cassette):
    """
    Test HTTP client with custom headers through real API calls.
    Replaces mock-based header configuration tests.
    """
    with vcr_cassette:
        api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")
        config = VeniceAIConfig.create_test_config()

        custom_headers = {
            "X-Custom-Header": "test-value",
            "User-Agent": "venice-ai-test/1.0",
        }

        client = VeniceHTTPClient(config=config, api_key=api_key, headers=custom_headers)

        try:
            session = await client.get_session()

            # Verify custom headers are set
            assert session.headers.get("X-Custom-Header") == "test-value"
            assert "venice-ai-test" in session.headers.get("User-Agent", "")

            # The custom headers ride along on an authenticated request.
            async with session.get("models") as response:
                assert response.status == 200
        finally:
            await client.close()


@pytest.mark.integration
async def test_http_client_timeout_configuration(vcr_cassette):
    """
    Test HTTP client timeout configuration through real API calls.
    Replaces mock-based timeout tests.
    """
    with vcr_cassette:
        api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")
        config = VeniceAIConfig.create_test_config()

        # Create client with aggressive timeout
        client = VeniceHTTPClient(config=config, api_key=api_key)

        try:
            session = await client.get_session()

            # Verify timeout is configured
            assert session.timeout is not None
            assert session.timeout.total == config.http_client.timeout

            # A quick request completes within the configured timeout.
            async with session.get("models") as response:
                assert response.status == 200
        finally:
            await client.close()


# ============================================================================
# HTTP Request Method Tests
# ============================================================================


@pytest.mark.integration
async def test_http_get_requests(venice_client, vcr_cassette):
    """
    Test HTTP GET request functionality through models API.
    Replaces mock-based GET request tests.
    """
    with vcr_cassette:
        # Test GET via models.list()
        models = await venice_client.models.list()

        # Verify GET request succeeded
        assert models is not None
        assert hasattr(models, "data")
        assert isinstance(models.data, list)

        # Verify we got actual model data
        if len(models.data) > 0:
            model = models.data[0]
            assert hasattr(model, "id")
            assert hasattr(model, "object")


@pytest.mark.integration
async def test_http_post_requests(venice_client, model_selector, vcr_cassette):
    """
    Test HTTP POST request functionality through chat completions.
    Replaces mock-based POST request tests.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Test POST via chat completions
        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "HTTP test"}],
            max_completion_tokens=10,
            temperature=0.1,
        )

        # Verify POST request succeeded
        assert response is not None
        assert hasattr(response, "id")
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.usage.total_tokens > 0


@pytest.mark.integration
async def test_http_request_with_json_data(venice_client, model_selector, vcr_cassette):
    """
    Test HTTP requests with JSON payload.
    Replaces mock-based JSON handling tests.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Test complex JSON payload
        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "JSON payload test"},
            ],
            max_completion_tokens=15,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )

        # Verify JSON was processed correctly
        assert response is not None
        assert len(response.choices) > 0


# ============================================================================
# Connection and Session Management Tests
# ============================================================================


@pytest.mark.integration
async def test_http_client_connection_reuse(http_client, vcr_cassette):
    """
    Test HTTP client connection reuse across multiple requests.
    Replaces mock-based connection pooling tests.
    """
    with vcr_cassette:
        session = await http_client.get_session()

        # Make multiple requests using the same session
        statuses = []
        for _i in range(3):
            async with session.get("models") as response:
                statuses.append(response.status)

        # Verify session was reused (same object)
        session2 = await http_client.get_session()
        assert session is session2

        assert statuses == [200, 200, 200]


@pytest.mark.integration
async def test_http_client_concurrent_requests(venice_client, model_selector, vcr_cassette):
    """
    Test concurrent HTTP requests through the same client.
    Replaces mock-based concurrent request tests.
    """
    with vcr_cassette:
        chat_model = await model_selector.select_chat_model()

        # Create multiple concurrent requests
        async def make_request(index: int):
            return await venice_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": f"Concurrent HTTP test {index}"}],
                max_completion_tokens=10,
                temperature=0.1,
            )

        # Execute concurrent requests
        tasks = [make_request(i) for i in range(4)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify concurrent handling
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= 2  # Allow for some rate limiting

        # Verify responses are unique (different request IDs)
        response_ids = [getattr(r, "id", None) for r in successful_responses if hasattr(r, "id")]
        assert len(set(response_ids)) == len(response_ids)


# ============================================================================
# Error Handling and Recovery Tests
# ============================================================================


@pytest.mark.integration
async def test_http_error_handling(venice_client, vcr_cassette):
    """
    Test HTTP error handling through invalid API requests.
    Replaces mock-based error handling tests.
    """
    with vcr_cassette:
        # Test with invalid model to trigger HTTP error
        with pytest.raises(VeniceError) as exc_info:
            await venice_client.chat.completions.create(
                model="definitely-invalid-model-name",
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=1,
            )

        # Verify error contains HTTP-related information
        error_str = str(exc_info.value).lower()
        # Rate limit error contains "failed", "attempts", or "status" keywords
        assert any(keyword in error_str for keyword in ["failed", "attempts", "status", "http"])


@pytest.mark.integration
async def test_http_authentication_error(vcr_cassette):
    """
    Test HTTP authentication error handling.
    Replaces mock-based auth error tests.
    """
    with vcr_cassette:
        # Use invalid API key to trigger auth error
        client = VeniceClient(api_key="invalid-key-for-auth-test")

        try:
            with pytest.raises(VeniceError) as exc_info:
                # Use chat completions which requires authentication (models endpoint is public)
                await client.chat.completions.create(
                    model="llama-3.3-70b",  # Use a known model
                    messages=[{"role": "user", "content": "Hello"}],  # type: ignore
                )

            # Verify it's an authentication-related HTTP error
            error_str = str(exc_info.value).lower()
            assert any(
                keyword in error_str
                for keyword in ["auth", "unauthorized", "invalid", "key", "forbidden"]
            )
        finally:
            await client.close()


# ============================================================================
# Request/Response Handling Tests
# ============================================================================


@pytest.mark.integration
async def test_http_request_response_headers(venice_client, vcr_cassette):
    """
    Test HTTP request and response header handling.
    Replaces mock-based header tests.
    """
    with vcr_cassette:
        # Make a request and verify headers are handled
        models = await venice_client.models.list()

        # Verify request succeeded (headers were accepted)
        assert models is not None
        assert hasattr(models, "data")

        # The fact that we got a successful response indicates
        # that headers (including Authorization) were handled correctly


@pytest.mark.integration
async def test_http_large_response_handling(venice_client, model_selector, vcr_cassette):
    """
    Test handling of larger HTTP responses.
    Replaces mock-based large response tests.
    """
    with vcr_cassette:
        # Request a longer response to test response handling
        chat_model = await model_selector.select_chat_model()

        response = await venice_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a detailed explanation of HTTP client functionality in exactly 50 words.",
                }
            ],
            max_completion_tokens=100,  # Allow for longer response
            temperature=0.3,
        )

        # Verify large response was handled properly
        assert response is not None
        assert len(response.choices) > 0
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 50  # Should be a substantial response


# ============================================================================
# Client Lifecycle Tests
# ============================================================================


@pytest.mark.integration
async def test_http_client_lifecycle(vcr_cassette):
    """
    Test complete HTTP client lifecycle.
    Replaces mock-based lifecycle tests.
    """
    with vcr_cassette:
        api_key = os.getenv("VENICE_API_KEY", "test-api-key-for-recording")

        # Test full lifecycle with context manager
        async with VeniceClient(api_key=api_key) as client:
            # Client should be fully functional
            models = await client.models.list()
            assert models is not None

            # Multiple requests should work
            models2 = await client.models.list()
            assert models2 is not None

        # Context manager should handle cleanup
        # (Verified by successful completion without hanging)


@pytest.mark.integration
async def test_http_client_cleanup_on_close(http_client, vcr_cassette):
    """
    Test HTTP client proper cleanup on close.
    Replaces mock-based cleanup tests.
    """
    with vcr_cassette:
        # Use the client
        session = await http_client.get_session()
        assert session is not None
        assert not http_client.is_closed

        # Close the client
        await http_client.close()
        assert http_client.is_closed

        # Verify getting session after close raises error
        with pytest.raises(RuntimeError, match="HTTP client has been closed"):
            await http_client.get_session()
