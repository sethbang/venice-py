"""
Comprehensive tests for src/venice_ai/resources/chat/completions.py module.

This test file focuses on achieving >80% coverage for chat completions operations,
testing both streaming and non-streaming modes, parameter validation, deprecation
warnings, and error handling.
"""

import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.streaming import Stream
from venice_ai.types.api import (
    AssistantMessage,
    DeveloperMessage,
    JSONSchemaFormat,
    SpecificToolChoice,
    StreamOptions,
    SystemMessage,
    Tool,
    ToolChoiceFunction,
    ToolFunction,
    ToolMessage,
    UserMessage,
    VeniceParameters,
)


class MockVeniceClient:
    """Mock client for testing ChatCompletions resource."""

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self.post = AsyncMock()

        # Create a proper async generator mock for _stream_request
        async def default_stream_request(*args, **kwargs):
            # Return an empty async generator to avoid warnings
            if False:  # Never yields, just makes it an async generator
                yield

        self._stream_request = default_stream_request


@pytest.fixture
def mock_client():
    """Create a mock Venice client for testing."""
    return MockVeniceClient()


@pytest.fixture
def chat_completions_resource(mock_client):
    """Create a ChatCompletions resource instance for testing."""
    return ChatCompletions(mock_client)


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Venice AI."},
    ]


@pytest.fixture
def sample_chat_response():
    """Sample chat completion response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama-3.2-3b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Venice AI is an advanced artificial intelligence platform.",
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
    }


@pytest.fixture
def sample_chat_chunk():
    """Sample chat completion chunk for streaming."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "llama-3.2-3b",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    }


class TestChatCompletionsNonStreaming:
    """Test non-streaming chat completion functionality."""

    @pytest.mark.asyncio
    async def test_create_with_all_parameters(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test non-streaming with all optional parameters."""
        mock_client.post.return_value = sample_chat_response

        venice_params = VeniceParameters(
            character_slug=None,
            strip_thinking_response=False,
            disable_thinking=False,
            enable_web_search="off",
            enable_web_citations=False,
            include_search_results_in_stream=False,
            return_search_results_as_documents=None,
            include_venice_system_prompt=True,
        )

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get weather information",
                    strict=False,
                ),
                id="tool_1",
            )
        ]

        tool_choice = SpecificToolChoice(
            type="function", function=ToolChoiceFunction(name="get_weather")
        )

        response_format = JSONSchemaFormat(
            type="json_schema",
            json_schema={"name": "weather_response", "schema": {"type": "object"}},
        )

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=False,
            max_completion_tokens=100,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            n=1,
            stop=["END"],
            seed=42,
            tools=tools,
            tool_choice=tool_choice,
            venice_parameters=venice_params,
            response_format=response_format,
            logprobs=True,
            top_logprobs=5,
            parallel_tool_calls=True,
            repetition_penalty=1.1,
            stop_token_ids=[50256],
            top_k=40,
            user="test_user",
        )

        assert result == sample_chat_response

        # Verify request body
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["model"] == "llama-3.2-3b"
        assert request_body["messages"] == sample_chat_messages
        assert request_body["stream"] is False
        assert request_body["max_completion_tokens"] == 100
        assert request_body["temperature"] == 0.7
        assert request_body["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_create_explicit_stream_false(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test explicitly setting stream=False."""
        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stream=False
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["stream"] is False


class TestChatCompletionsStreaming:
    """Test streaming chat completion functionality."""

    @pytest.mark.asyncio
    async def test_create_streaming_success(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test successful streaming chat completion."""

        # Create a proper async generator for streaming
        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stream=True
        )

        # Should return a Stream instance
        assert isinstance(result, Stream)

    @pytest.mark.asyncio
    async def test_create_streaming_with_custom_stream_cls(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming with custom stream class."""

        # Create a custom stream class that inherits from Stream
        class CustomStream(Stream):
            def __init__(self, iterator, client):
                super().__init__(iterator, client=client)
                self.custom_attr = "custom"

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=CustomStream,
        )

        # Should return CustomStream instance
        assert isinstance(result, CustomStream)
        assert hasattr(result, "custom_attr")
        assert result.custom_attr == "custom"

    @pytest.mark.asyncio
    async def test_create_streaming_with_invalid_stream_cls(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming with invalid stream class falls back to default."""

        # Create an invalid stream class without proper interface
        class InvalidStream:
            pass

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=InvalidStream,
        )

        # Should fall back to default Stream class
        assert isinstance(result, Stream)
        assert not isinstance(result, InvalidStream)

    @pytest.mark.asyncio
    async def test_create_streaming_with_compatible_custom_class(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming with compatible custom class that has proper interface."""

        # Create a compatible class with proper interface
        class CompatibleStream:
            def __init__(self, iterator, client=None):
                self.iterator = iterator
                self.client = client

            async def __aiter__(self):
                async for item in self.iterator:
                    yield item

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=CompatibleStream,
        )

        # Should use the compatible custom stream class
        assert isinstance(result, CompatibleStream)
        assert result.client == mock_client

    @pytest.mark.asyncio
    async def test_create_streaming_excludes_stream_cls_from_request(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test that stream_cls parameter is excluded from API request."""
        captured_call = {}

        # Create a function that captures the call and returns an async generator
        def mock_stream_request(*args, **kwargs):
            captured_call["args"] = args
            captured_call["kwargs"] = kwargs

            async def async_gen():
                yield {"choices": [{"delta": {"content": "test"}}]}

            return async_gen()

        mock_client._stream_request = mock_stream_request

        await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=Stream,
        )

        # Verify stream_cls is not in request body
        assert "kwargs" in captured_call
        request_body = captured_call["kwargs"]["json_data"]
        assert "stream_cls" not in request_body
        assert "stream" in request_body
        assert request_body["stream"] is True


class TestChatCompletionsParameterHandling:
    """Test parameter handling and deprecation warnings."""

    @pytest.mark.asyncio
    async def test_no_deprecation_warning_with_max_completion_tokens(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test no deprecation warning when using max_completion_tokens."""
        mock_client.post.return_value = sample_chat_response

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            await chat_completions_resource.create(
                model="llama-3.2-3b",
                messages=sample_chat_messages,
                max_completion_tokens=100,
            )

            # Should have no warnings
            assert len(w) == 0

    @pytest.mark.asyncio
    async def test_exclude_none_parameters(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test that None parameters are excluded from request."""
        mock_client.post.return_value = sample_chat_response

        await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            temperature=None,
            top_p=None,
            seed=42,
            user="test_user",
        )

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert "temperature" not in request_body
        assert "top_p" not in request_body
        assert request_body["seed"] == 42
        assert request_body["user"] == "test_user"


class TestChatCompletionsValidation:
    """Test input validation and error conditions."""

    @pytest.mark.asyncio
    async def test_create_with_empty_messages(self, chat_completions_resource, mock_client):
        """Test that empty messages list is handled by Pydantic validation."""
        # Pydantic will handle validation, so this should work at the resource level
        # but may fail at Pydantic validation level
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            await chat_completions_resource.create(model="llama-3.2-3b", messages=[])

    @pytest.mark.asyncio
    async def test_create_with_various_message_types(
        self, chat_completions_resource, mock_client, sample_chat_response
    ):
        """Test creation with different message types."""
        mock_client.post.return_value = sample_chat_response

        mixed_messages = [
            SystemMessage(role="system", content="You are helpful", name=None),
            DeveloperMessage(role="developer", content="Be concise.", name=None),
            UserMessage(role="user", content="Hello"),
            AssistantMessage(
                role="assistant",
                content="Hi there",
                name=None,
                reasoning_content=None,
                tool_calls=None,
            ),
            ToolMessage(
                role="tool",
                content="Tool result",
                tool_call_id="call_123",
                name=None,
                reasoning_content=None,
                tool_calls=None,
            ),
        ]

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=mixed_messages
        )

        assert result == sample_chat_response

    @pytest.mark.asyncio
    async def test_developer_message_role_literal_and_dict(
        self, chat_completions_resource, mock_client, sample_chat_response
    ):
        """Developer-role messages validate as both an instance and a raw dict."""
        mock_client.post.return_value = sample_chat_response

        dev = DeveloperMessage(content="Prefer short answers.")
        assert dev.role == "developer"

        messages = [
            DeveloperMessage(content="Prefer short answers."),
            {"role": "developer", "content": "Also be precise."},
            UserMessage(role="user", content="Hi"),
        ]
        result = await chat_completions_resource.create(model="llama-3.2-3b", messages=messages)
        assert result == sample_chat_response

    @pytest.mark.asyncio
    async def test_create_with_tools_and_tool_choice(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test creation with tools and tool choice."""
        mock_client.post.return_value = sample_chat_response

        tools = [
            Tool(
                type="function",
                function=ToolFunction(
                    name="get_weather",
                    description="Get current weather",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    strict=False,
                ),
                id="tool_1",
            )
        ]

        tool_choice = SpecificToolChoice(
            type="function", function=ToolChoiceFunction(name="get_weather")
        )

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            tools=tools,
            tool_choice=tool_choice,
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert "tools" in request_body
        assert "tool_choice" in request_body


class TestChatCompletionsStreamingValidation:
    """Test streaming-specific validation and behaviors."""

    @pytest.mark.asyncio
    async def test_streaming_with_none_stream_cls(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming with None stream_cls uses default."""

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=None,
        )

        # Should use default Stream class
        assert isinstance(result, Stream)

    @pytest.mark.asyncio
    async def test_streaming_with_non_class_stream_cls(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming with non-class stream_cls falls back to default."""

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls="not_a_class",  # String instead of class
        )

        # Should fall back to default Stream
        assert isinstance(result, Stream)

    @pytest.mark.asyncio
    async def test_streaming_signature_inspection_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming when signature inspection fails."""

        class ProblematicStream:
            """A class that will cause signature inspection to fail."""

            def __init__(self, *args, **kwargs):
                # This will cause TypeError during signature inspection
                pass

            def bad_init(self):
                # This will break signature inspection
                raise TypeError("Signature inspection error")

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=ProblematicStream,
        )

        # Should fall back to default Stream when inspection fails
        assert isinstance(result, Stream)
        assert not isinstance(result, ProblematicStream)


class TestChatCompletionsErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_create_authentication_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test handling of authentication error."""
        mock_response = MagicMock()
        mock_error = AuthenticationError("Invalid API key", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(AuthenticationError):
            await chat_completions_resource.create(
                model="llama-3.2-3b", messages=sample_chat_messages
            )

    @pytest.mark.asyncio
    async def test_create_permission_denied_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test handling of permission denied error."""
        mock_response = MagicMock()
        mock_error = PermissionDeniedError("Access denied to model", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(PermissionDeniedError):
            await chat_completions_resource.create(
                model="restricted-model", messages=sample_chat_messages
            )

    @pytest.mark.asyncio
    async def test_create_not_found_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test handling of model not found error."""
        mock_response = MagicMock()
        mock_error = NotFoundError("Model not found", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(NotFoundError):
            await chat_completions_resource.create(
                model="nonexistent-model", messages=sample_chat_messages
            )

    @pytest.mark.asyncio
    async def test_create_rate_limit_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test handling of rate limit error."""
        mock_response = MagicMock()
        mock_error = RateLimitError("Rate limit exceeded", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(RateLimitError):
            await chat_completions_resource.create(
                model="llama-3.2-3b", messages=sample_chat_messages
            )

    @pytest.mark.asyncio
    async def test_create_generic_api_error(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test handling of generic API error."""
        mock_response = MagicMock()
        mock_error = APIError("Server error", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(APIError):
            await chat_completions_resource.create(
                model="llama-3.2-3b", messages=sample_chat_messages
            )

    @pytest.mark.asyncio
    async def test_streaming_error_handling(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test error handling in streaming mode."""

        # Create an async generator that raises an exception when iterated
        async def failing_stream_request(*args, **kwargs):
            mock_response = MagicMock()
            raise RateLimitError("Rate limit exceeded", response=mock_response)
            yield  # This makes it an async generator, but exception is raised before yield

        mock_client._stream_request = failing_stream_request

        # Creating the stream should succeed
        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stream=True
        )

        # But iterating over it should raise the exception
        with pytest.raises(RateLimitError):
            async for _ in result:
                pass


class TestChatCompletionsRequestSerialization:
    """Test request serialization and Pydantic model handling."""

    @pytest.mark.asyncio
    async def test_create_pydantic_serialization(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test that request is properly serialized through Pydantic model."""
        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            temperature=0.7,
            max_completion_tokens=150,
            user="test_user",
        )

        assert result == sample_chat_response

        # Verify that the request went through Pydantic validation
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]

        # Should be a clean dictionary with expected fields
        assert isinstance(request_body, dict)
        assert request_body["model"] == "llama-3.2-3b"
        assert request_body["messages"] == sample_chat_messages
        assert request_body["temperature"] == 0.7
        assert request_body["max_completion_tokens"] == 150
        assert request_body["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_create_exclude_none_serialization(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test that None values are excluded during serialization."""
        mock_client.post.return_value = sample_chat_response

        await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            temperature=None,
            top_p=None,
            frequency_penalty=0.5,
            presence_penalty=None,
        )

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]

        # None values should be excluded
        assert "temperature" not in request_body
        assert "top_p" not in request_body
        assert "presence_penalty" not in request_body
        # Non-None values should be included
        assert request_body["frequency_penalty"] == 0.5


class TestChatCompletionsAdvancedFeatures:
    """Test advanced features and edge cases."""

    @pytest.mark.asyncio
    async def test_create_with_venice_parameters(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test creation with Venice-specific parameters."""
        mock_client.post.return_value = sample_chat_response

        venice_params = VeniceParameters(
            character_slug=None,
            strip_thinking_response=False,
            disable_thinking=False,
            enable_web_search="off",
            enable_web_citations=False,
            include_search_results_in_stream=False,
            return_search_results_as_documents=None,
            include_venice_system_prompt=True,
        )

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            venice_parameters=venice_params,
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert "venice_parameters" in request_body

    @pytest.mark.asyncio
    async def test_create_with_response_format(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test creation with JSON response format."""
        mock_client.post.return_value = sample_chat_response

        response_format = JSONSchemaFormat(
            type="json_schema",
            json_schema={
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "condition": {"type": "string"},
                    },
                    "required": ["temperature", "condition"],
                },
            },
        )

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            response_format=response_format,
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert "response_format" in request_body

    @pytest.mark.asyncio
    async def test_create_with_stream_options(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test creation with stream options in streaming mode."""
        captured_call = {}

        # Create a function that captures the call and returns an async generator
        def mock_stream_request(*args, **kwargs):
            captured_call["args"] = args
            captured_call["kwargs"] = kwargs

            async def async_gen():
                yield {"choices": [{"delta": {"content": "test"}}]}

            return async_gen()

        mock_client._stream_request = mock_stream_request

        stream_options = StreamOptions(include_usage=True)

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_options=stream_options,
        )

        assert isinstance(result, Stream)

        # Verify stream_options is in request body
        assert "kwargs" in captured_call
        request_body = captured_call["kwargs"]["json_data"]
        assert "stream_options" in request_body
        assert request_body["stream_options"]["include_usage"] is True

    @pytest.mark.asyncio
    async def test_create_with_stop_sequences(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test creation with stop sequences."""
        mock_client.post.return_value = sample_chat_response

        # Test with single stop string
        result1 = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stop="END"
        )

        # Test with multiple stop sequences
        result2 = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stop=["END", "STOP", "\n\n"],
        )

        assert result1 == sample_chat_response
        assert result2 == sample_chat_response


class TestChatCompletionsTypingAndOverloads:
    """Test type annotations and overload behavior."""

    @pytest.mark.asyncio
    async def test_non_streaming_return_type(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test that non-streaming returns ChatCompletion type."""
        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stream=False
        )

        # Should be the raw response dict (cast to ChatCompletion)
        assert result == sample_chat_response
        assert isinstance(result, dict)
        assert "choices" in result
        assert "usage" in result

    @pytest.mark.asyncio
    async def test_streaming_return_type(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test that streaming returns AsyncIterable type."""

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=sample_chat_messages, stream=True
        )

        # Should be an async iterable (Stream instance)
        assert hasattr(result, "__aiter__")
        assert isinstance(result, Stream)

    def test_overload_signatures_exist(self, chat_completions_resource):
        """Test that overload signatures are properly defined."""
        create_method = chat_completions_resource.create

        # Should have annotations indicating overloads
        assert hasattr(create_method, "__annotations__")

        # Check that the method is callable
        assert callable(create_method)


class TestChatCompletionsEdgeCases:
    """Test edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_create_with_valid_optional_parameters(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test that valid optional parameters are included in request."""
        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            top_k=50,  # Valid optional parameter
            min_p=0.1,  # Valid optional parameter
            repetition_penalty=1.1,  # Another valid parameter
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["top_k"] == 50
        assert request_body["min_p"] == 0.1
        assert request_body["repetition_penalty"] == 1.1

    @pytest.mark.asyncio
    async def test_streaming_with_signature_inspection_fallback(
        self, chat_completions_resource, mock_client, sample_chat_messages
    ):
        """Test streaming falls back gracefully when custom stream class inspection fails."""

        class IncompatibleStreamWithAiter:
            """A class with __aiter__ but incompatible signature."""

            def __init__(self, wrong_param):
                pass

            async def __aiter__(self):
                yield {}

        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            stream_cls=IncompatibleStreamWithAiter,
        )

        # Should fall back to default Stream
        assert isinstance(result, Stream)
        assert not isinstance(result, IncompatibleStreamWithAiter)

    @pytest.mark.asyncio
    async def test_create_with_numeric_model_name(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test creation with numeric model name."""
        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="gpt-4o",  # Model with numbers
            messages=sample_chat_messages,
        )

        assert result == sample_chat_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_create_with_complex_messages(
        self, chat_completions_resource, mock_client, sample_chat_response
    ):
        """Test creation with complex message content."""
        complex_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to tools.",
            },
            {"role": "user", "content": "What's the weather like in San Francisco?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "San Francisco"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Sunny, 72°F", "tool_call_id": "call_123"},
        ]

        mock_client.post.return_value = sample_chat_response

        result = await chat_completions_resource.create(
            model="llama-3.2-3b", messages=complex_messages
        )

        assert result == sample_chat_response


class TestChatCompletionsIntegration:
    """Test integration scenarios and workflows."""

    @pytest.mark.asyncio
    async def test_streaming_vs_non_streaming_consistency(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test consistency between streaming and non-streaming modes."""
        # Test non-streaming first
        mock_client.post.return_value = sample_chat_response

        non_streaming_result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=False,
            temperature=0.7,
        )

        # Test streaming
        async def mock_stream_request(*args, **kwargs):
            if False:  # Never yields, just makes it an async generator
                yield

        mock_client._stream_request = mock_stream_request

        streaming_result = await chat_completions_resource.create(
            model="llama-3.2-3b",
            messages=sample_chat_messages,
            stream=True,
            temperature=0.7,
        )

        # Non-streaming should return dict
        assert isinstance(non_streaming_result, dict)
        assert non_streaming_result == sample_chat_response

        # Streaming should return Stream
        assert isinstance(streaming_result, Stream)

    @pytest.mark.asyncio
    async def test_different_models_same_messages(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test same messages with different models."""
        models = ["llama-3.2-3b", "llama-3.3-70b", "venice-1"]

        for model in models:
            # Modify response for each model
            model_response = sample_chat_response.copy()
            model_response["model"] = model
            mock_client.post.return_value = model_response

            result = await chat_completions_resource.create(
                model=model, messages=sample_chat_messages
            )

            assert result["model"] == model

    @pytest.mark.asyncio
    async def test_parameter_combinations(
        self,
        chat_completions_resource,
        mock_client,
        sample_chat_messages,
        sample_chat_response,
    ):
        """Test various parameter combinations."""
        mock_client.post.return_value = sample_chat_response

        # Test multiple parameter combinations
        param_sets = [
            {"temperature": 0.1, "top_p": 0.9},
            {"frequency_penalty": 0.5, "presence_penalty": 0.3},
            {"seed": 42, "n": 2},
            {"top_k": 50, "repetition_penalty": 1.2},
        ]

        for params in param_sets:
            result = await chat_completions_resource.create(
                model="llama-3.2-3b", messages=sample_chat_messages, **params
            )

            assert result == sample_chat_response

            call_args = mock_client.post.call_args
            request_body = call_args[1]["json_data"]

            # All provided parameters should be in request
            for key, value in params.items():
                assert request_body[key] == value


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_create_sends_message_cache_control_in_full(
    chat_completions_resource, mock_client, sample_chat_response, stream
):
    """A message-level marker reaches the wire with its default ``type`` filled in."""
    from venice_ai.types.api.requests.chat import AssistantMessage, ToolMessage, UserMessage
    from venice_ai.types.api.requests.common import CacheControl

    captured = {}

    def mock_stream_request(*args, **kwargs):
        captured["kwargs"] = kwargs

        async def async_gen():
            yield {"choices": [{"delta": {"content": "ok"}}]}

        return async_gen()

    mock_client._stream_request = mock_stream_request
    mock_client.post.return_value = sample_chat_response

    await chat_completions_resource.create(
        model="m",
        messages=[
            UserMessage(content="look it up"),
            AssistantMessage(tool_calls=[{"id": "c1", "type": "function"}]),
            ToolMessage(content="r", tool_call_id="c1", cache_control=CacheControl(ttl="5m")),
            {"role": "assistant", "content": "a", "cache_control": {"type": "ephemeral"}},
        ],
        stream=stream,
    )

    body = captured["kwargs"]["json_data"] if stream else mock_client.post.call_args[1]["json_data"]
    assert body["messages"][2]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}
    assert body["messages"][3]["cache_control"] == {"type": "ephemeral"}
