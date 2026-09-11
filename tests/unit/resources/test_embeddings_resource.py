"""
Comprehensive tests for src/venice_ai/resources/embeddings.py module.

This test file focuses on achieving >80% coverage for embeddings operations,
testing embedding creation with various inputs, batch processing, and error handling.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from venice_ai.resources.embeddings import Embeddings
from venice_ai.types.api import EmbeddingsResponse

# Use the correct alias
EmbeddingList = EmbeddingsResponse


class MockVeniceClient:
    """Mock client for testing Embeddings resource."""

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self.post = AsyncMock()


@pytest.fixture
def mock_client():
    """Create a mock Venice client for testing."""
    return MockVeniceClient()


@pytest.fixture
def embeddings_resource(mock_client):
    """Create an Embeddings resource instance for testing."""
    return Embeddings(mock_client)


@pytest.fixture
def sample_embedding_response():
    """Sample embedding response data."""
    return {
        "object": "list",
        "model": "text-embedding-bge-m3",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]}],
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def sample_batch_embedding_response():
    """Sample batch embedding response data."""
    return {
        "object": "list",
        "model": "text-embedding-bge-m3",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]},
            {"object": "embedding", "index": 1, "embedding": [0.6, 0.7, 0.8, 0.9, 1.0]},
        ],
        "usage": {"prompt_tokens": 20, "total_tokens": 20},
    }


@pytest.fixture
def sample_base64_embedding_response():
    """Sample embedding response with base64 encoding."""
    return {
        "object": "list",
        "model": "text-embedding-bge-m3",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": "AAAA3D8AAADAPwAAABA/AAAA4D8AAAAQAAAAA==",  # base64 encoded
            }
        ],
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


class TestEmbeddingsCreate:
    """Test create() method functionality."""

    @pytest.mark.asyncio
    async def test_create_single_string_success(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test successful embedding creation with single string input."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input="This is a test sentence."
        )

        assert result == sample_embedding_response
        mock_client.post.assert_called_once_with(
            "embeddings",
            json_data={
                "model": "text-embedding-bge-m3",
                "input": "This is a test sentence.",
            },
            cast_to=EmbeddingList,
        )

    @pytest.mark.asyncio
    async def test_create_string_list_success(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test successful embedding creation with list of strings."""
        mock_client.post.return_value = sample_batch_embedding_response

        input_texts = ["First test sentence.", "Second test sentence."]

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=input_texts)

        assert result == sample_batch_embedding_response

        # Verify request body
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["model"] == "text-embedding-bge-m3"
        assert request_body["input"] == input_texts

    @pytest.mark.asyncio
    async def test_create_with_all_parameters(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with all optional parameters."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test with all parameters",
            dimensions=512,
            encoding_format="base64",
            user="test_user_123",
        )

        assert result == sample_embedding_response

        # Verify all parameters
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["model"] == "text-embedding-bge-m3"
        assert request_body["input"] == "Test with all parameters"
        assert request_body["dimensions"] == 512
        assert request_body["encoding_format"] == "base64"
        assert request_body["user"] == "test_user_123"

    @pytest.mark.asyncio
    async def test_create_with_base64_encoding(
        self, embeddings_resource, mock_client, sample_base64_embedding_response
    ):
        """Test embedding creation with base64 encoding format."""
        mock_client.post.return_value = sample_base64_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test base64 encoding",
            encoding_format="base64",
        )

        assert result == sample_base64_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["encoding_format"] == "base64"

    @pytest.mark.asyncio
    async def test_create_with_float_encoding(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with float encoding format (default)."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test float encoding",
            encoding_format="float",
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["encoding_format"] == "float"

    @pytest.mark.asyncio
    async def test_create_exclude_none_parameters(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that None parameters are excluded from request."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test exclude none",
            dimensions=None,  # Should be excluded
            encoding_format=None,  # Should be excluded
            user="test_user",
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert "dimensions" not in request_body
        assert "encoding_format" not in request_body
        assert request_body["user"] == "test_user"


class TestEmbeddingsValidation:
    """Test input validation and error conditions."""

    @pytest.mark.asyncio
    async def test_create_empty_model_error(self, embeddings_resource, mock_client):
        """Test that empty model raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="", input="Test input")

        assert "model parameter is required and cannot be empty" in str(exc_info.value)
        assert exc_info.value.request is None
        assert exc_info.value.response is None
        assert exc_info.value.body is None

    @pytest.mark.asyncio
    async def test_create_none_model_error(self, embeddings_resource, mock_client):
        """Test that None model raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model=None, input="Test input")

        assert "model parameter is required and cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_empty_string_input_error(self, embeddings_resource, mock_client):
        """Test that empty string input raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input="")

        assert "input cannot be empty" in str(exc_info.value)
        assert exc_info.value.request is None
        assert exc_info.value.response is None
        assert exc_info.value.body is None

    @pytest.mark.asyncio
    async def test_create_empty_list_input_error(self, embeddings_resource, mock_client):
        """Test that empty list input raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input=[])

        assert "input cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_none_input_error(self, embeddings_resource, mock_client):
        """Test that None input raises InvalidRequestError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input=None)

        assert "input cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_too_many_items_error(self, embeddings_resource, mock_client):
        """Test that input array with >2048 items raises InvalidRequestError."""
        large_input = ["item"] * 2049  # 2049 items, exceeds limit of 2048

        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input=large_input)

        assert "input array must have 2048 or fewer items" in str(exc_info.value)
        assert "but got 2049 items" in str(exc_info.value)
        assert exc_info.value.request is None
        assert exc_info.value.response is None
        assert exc_info.value.body is None

    @pytest.mark.asyncio
    async def test_create_exactly_2048_items_success(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test that input array with exactly 2048 items succeeds."""
        large_input = [f"item_{i}" for i in range(2048)]  # Exactly 2048 items
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=large_input)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert len(request_body["input"]) == 2048


class TestEmbeddingsInputTypes:
    """Test different input types and formats."""

    @pytest.mark.asyncio
    async def test_create_mixed_input_types_in_list(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test embedding creation with mixed string and token inputs."""
        mock_client.post.return_value = sample_batch_embedding_response

        mixed_input = ["First text input", "Second text input", "Third text input"]

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=mixed_input)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == mixed_input


class TestEmbeddingsParameters:
    """Test parameter handling and edge cases."""

    @pytest.mark.asyncio
    async def test_create_with_custom_dimensions(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with custom dimensions."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test custom dimensions",
            dimensions=768,
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["dimensions"] == 768

    @pytest.mark.asyncio
    async def test_create_with_zero_dimensions_validation_error(
        self, embeddings_resource, mock_client
    ):
        """Test that zero dimensions raises Pydantic validation error."""
        from pydantic_core import ValidationError

        # Zero dimensions should raise ValidationError during Pydantic validation
        with pytest.raises(ValidationError):
            await embeddings_resource.create(
                model="text-embedding-bge-m3",
                input="Test zero dimensions",
                dimensions=0,
            )

    @pytest.mark.asyncio
    async def test_create_with_large_dimensions(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with large dimensions."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test large dimensions",
            dimensions=4096,
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["dimensions"] == 4096

    @pytest.mark.asyncio
    async def test_create_user_parameter(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with user parameter."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test user parameter",
            user="user_id_12345",
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["user"] == "user_id_12345"

    @pytest.mark.asyncio
    async def test_create_exclude_none_values(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that None values are excluded from request body."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test exclude none",
            dimensions=None,
            encoding_format=None,
            user=None,
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]

        # None values should be excluded
        assert "dimensions" not in request_body
        assert "encoding_format" not in request_body
        assert "user" not in request_body
        assert request_body["model"] == "text-embedding-bge-m3"
        assert request_body["input"] == "Test exclude none"


class TestEmbeddingsValidationEdgeCases:
    """Test validation edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_create_whitespace_only_model_error(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that whitespace-only model is rejected by Pydantic validation."""
        from pydantic_core import ValidationError

        # Whitespace-only model should be rejected by Pydantic validation
        with pytest.raises(ValidationError):
            await embeddings_resource.create(
                model="   ",  # Whitespace only - rejected by Pydantic
                input="Test input",
            )

    @pytest.mark.asyncio
    async def test_create_single_item_list_success(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with single-item list."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=["Single item in list"]
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == ["Single item in list"]

    @pytest.mark.asyncio
    async def test_create_boundary_list_size_2047(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test embedding creation with 2047 items (just under limit)."""
        large_input = [f"item_{i}" for i in range(2047)]
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=large_input)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert len(request_body["input"]) == 2047


class TestEmbeddingsErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_create_authentication_error(self, embeddings_resource, mock_client):
        """Test handling of authentication error."""
        mock_response = MagicMock()
        mock_error = AuthenticationError("Invalid API key", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(AuthenticationError):
            await embeddings_resource.create(model="text-embedding-bge-m3", input="Test auth error")

    @pytest.mark.asyncio
    async def test_create_permission_denied_error(self, embeddings_resource, mock_client):
        """Test handling of permission denied error."""
        mock_response = MagicMock()
        mock_error = PermissionDeniedError("Access denied to model", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(PermissionDeniedError):
            await embeddings_resource.create(
                model="restricted-embedding-model", input="Test permission error"
            )

    @pytest.mark.asyncio
    async def test_create_not_found_error(self, embeddings_resource, mock_client):
        """Test handling of model not found error."""
        mock_response = MagicMock()
        mock_error = NotFoundError("Model not found", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(NotFoundError):
            await embeddings_resource.create(
                model="nonexistent-model", input="Test not found error"
            )

    @pytest.mark.asyncio
    async def test_create_rate_limit_error(self, embeddings_resource, mock_client):
        """Test handling of rate limit error."""
        mock_response = MagicMock()
        mock_error = RateLimitError("Rate limit exceeded", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(RateLimitError):
            await embeddings_resource.create(
                model="text-embedding-bge-m3", input="Test rate limit error"
            )

    @pytest.mark.asyncio
    async def test_create_generic_api_error(self, embeddings_resource, mock_client):
        """Test handling of generic API error."""
        mock_response = MagicMock()
        mock_error = APIError("Server error", response=mock_response)
        mock_client.post.side_effect = mock_error

        with pytest.raises(APIError):
            await embeddings_resource.create(
                model="text-embedding-bge-m3", input="Test generic error"
            )


class TestEmbeddingsBatchProcessing:
    """Test batch processing scenarios."""

    @pytest.mark.asyncio
    async def test_create_large_batch_success(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test successful batch processing with large number of inputs."""
        large_batch = [f"Sentence number {i} for batch processing." for i in range(100)]
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=large_batch)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert len(request_body["input"]) == 100
        assert request_body["input"][0] == "Sentence number 0 for batch processing."
        assert request_body["input"][99] == "Sentence number 99 for batch processing."

    @pytest.mark.asyncio
    async def test_create_mixed_length_strings(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test batch processing with strings of varying lengths (no empty strings)."""
        mixed_input = [
            "Short",
            "Medium length sentence for testing",
            "This is a much longer sentence that contains significantly more text content to test how the embedding model handles varying input lengths within a single batch request.",
            "Final sentence",  # Removed empty string as Pydantic validates against it
        ]
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=mixed_input)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == mixed_input


class TestEmbeddingsRequestSerialization:
    """Test request serialization and Pydantic model handling."""

    @pytest.mark.asyncio
    async def test_create_pydantic_serialization(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that request is properly serialized through Pydantic model."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Test Pydantic serialization",
            dimensions=256,
            encoding_format="float",
            user="test_user",
        )

        assert result == sample_embedding_response

        # Verify that the request went through Pydantic validation
        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]

        # Should be a clean dictionary with all expected fields
        expected_fields = {"model", "input", "dimensions", "encoding_format", "user"}
        assert set(request_body.keys()) == expected_fields
        assert isinstance(request_body, dict)

    @pytest.mark.asyncio
    async def test_create_request_model_validation(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that EmbeddingsRequest model is used for validation."""
        mock_client.post.return_value = sample_embedding_response

        # This should work without issues due to Pydantic validation
        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input="Validation test",
            encoding_format="base64",  # Valid literal
        )

        assert result == sample_embedding_response


class TestEmbeddingsResponseHandling:
    """Test response handling and type casting."""

    @pytest.mark.asyncio
    async def test_create_cast_to_embedding_list(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that response is properly cast to EmbeddingList."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input="Cast test")

        assert result == sample_embedding_response

        # Verify cast_to parameter
        call_args = mock_client.post.call_args
        assert call_args[1]["cast_to"] == EmbeddingList

    @pytest.mark.asyncio
    async def test_create_response_structure_validation(self, embeddings_resource, mock_client):
        """Test that response structure is validated correctly."""
        complex_response = {
            "object": "list",
            "model": "text-embedding-bge-m3",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1] * 1536,  # Full dimension embedding
                },
                {"object": "embedding", "index": 1, "embedding": [0.2] * 1536},
            ],
            "usage": {"prompt_tokens": 25, "total_tokens": 25},
        }
        mock_client.post.return_value = complex_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=["First text", "Second text"]
        )

        assert result == complex_response
        assert result["object"] == "list"
        assert len(result["data"]) == 2
        assert result["usage"]["prompt_tokens"] == 25


class TestEmbeddingsIntegration:
    """Test integration scenarios and workflows."""

    @pytest.mark.asyncio
    async def test_single_vs_batch_consistency(self, embeddings_resource, mock_client):
        """Test consistency between single and batch embedding requests."""
        # Single embedding response
        single_response = {
            "object": "list",
            "model": "text-embedding-bge-m3",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

        # Batch embedding response
        batch_response = {
            "object": "list",
            "model": "text-embedding-bge-m3",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        # Test single input
        mock_client.post.return_value = single_response
        single_result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input="Single input"
        )

        # Test batch input
        mock_client.post.return_value = batch_response
        batch_result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=["First input", "Second input"]
        )

        # Both should have consistent structure
        assert single_result["object"] == batch_result["object"] == "list"
        assert single_result["model"] == batch_result["model"]
        assert len(single_result["data"]) == 1
        assert len(batch_result["data"]) == 2

    @pytest.mark.asyncio
    async def test_different_models_same_input(self, embeddings_resource, mock_client):
        """Test same input with different models."""
        input_text = "Test input for different models"

        # Response for first model
        model1_response = {
            "object": "list",
            "model": "text-embedding-bge-m3",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1] * 1024}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

        # Response for second model
        model2_response = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.2] * 1536}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

        # Test first model
        mock_client.post.return_value = model1_response
        result1 = await embeddings_resource.create(model="text-embedding-bge-m3", input=input_text)

        # Test second model
        mock_client.post.return_value = model2_response
        result2 = await embeddings_resource.create(model="text-embedding-3-small", input=input_text)

        # Results should differ by model
        assert result1["model"] != result2["model"]
        assert len(result1["data"][0]["embedding"]) != len(result2["data"][0]["embedding"])


class TestEmbeddingsSpecialInputs:
    """Test special input cases and unicode handling."""

    @pytest.mark.asyncio
    async def test_create_unicode_input(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with unicode characters."""
        mock_client.post.return_value = sample_embedding_response

        unicode_input = "Test with émojis 🚀 and special chars: 中文, العربية, русский"

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=unicode_input
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == unicode_input

    @pytest.mark.asyncio
    async def test_create_very_long_input(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with very long input text."""
        mock_client.post.return_value = sample_embedding_response

        long_input = "This is a very long text. " * 100  # Very long string

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=long_input)

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == long_input

    @pytest.mark.asyncio
    async def test_create_newlines_and_whitespace(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test embedding creation with newlines and whitespace."""
        mock_client.post.return_value = sample_embedding_response

        input_with_whitespace = "Line 1\n\nLine 2\t\tTabbed\r\nWindows line ending"

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=input_with_whitespace
        )

        assert result == sample_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == input_with_whitespace


class TestEmbeddingsEdgeCases:
    """Test edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_create_exact_limit_boundary(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test creation at exact input limit boundary."""
        # Test at exact 2048 limit
        exact_limit_input = [f"item_{i}" for i in range(2048)]
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input=exact_limit_input
        )

        assert result == sample_batch_embedding_response

    @pytest.mark.asyncio
    async def test_create_single_character_inputs(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test embedding creation with single character inputs."""
        single_chars = ["a", "b", "c", "1", "!", "🔥"]
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=single_chars)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert request_body["input"] == single_chars


class TestEmbeddingsModelValidation:
    """Test model validation scenarios."""

    @pytest.mark.asyncio
    async def test_create_falsy_model_validation(self, embeddings_resource, mock_client):
        """Test various falsy model values."""
        falsy_models = ["", None, False, 0]

        for falsy_model in falsy_models:
            with pytest.raises(InvalidRequestError) as exc_info:
                await embeddings_resource.create(model=falsy_model, input="Test falsy model")

            assert "model parameter is required and cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_falsy_input_validation(self, embeddings_resource, mock_client):
        """Test various falsy input values."""
        falsy_inputs = ["", [], None, False]

        for falsy_input in falsy_inputs:
            with pytest.raises(InvalidRequestError) as exc_info:
                await embeddings_resource.create(model="text-embedding-bge-m3", input=falsy_input)

            assert "input cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_falsy_text_input_not_empty(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that a falsy-looking text input is not considered empty."""
        mock_client.post.return_value = sample_embedding_response

        # "0" is a non-empty string; the empty-input guard must not reject it.
        result = await embeddings_resource.create(
            model="text-embedding-bge-m3",
            input=["0"],
        )

        assert result == sample_embedding_response


class TestEmbeddingsTypeCasting:
    """Test type casting and response conversion."""

    @pytest.mark.asyncio
    async def test_create_return_type_consistency(
        self, embeddings_resource, mock_client, sample_embedding_response
    ):
        """Test that create method consistently returns EmbeddingList type."""
        mock_client.post.return_value = sample_embedding_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input="Type consistency test"
        )

        # Should be the exact response data
        assert result == sample_embedding_response
        assert isinstance(result, dict)
        assert "data" in result
        assert "usage" in result
        assert "model" in result
        assert "object" in result

    @pytest.mark.asyncio
    async def test_create_preserves_response_metadata(self, embeddings_resource, mock_client):
        """Test that all response metadata is preserved."""
        detailed_response = {
            "object": "list",
            "model": "text-embedding-bge-m3",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "usage": {"prompt_tokens": 15, "total_tokens": 15},
            "custom_field": "should_be_preserved",  # Custom field
            "metadata": {"request_id": "req_123"},
        }
        mock_client.post.return_value = detailed_response

        result = await embeddings_resource.create(
            model="text-embedding-bge-m3", input="Metadata preservation test"
        )

        # All fields should be preserved
        assert result == detailed_response
        assert result["custom_field"] == "should_be_preserved"
        assert result["metadata"]["request_id"] == "req_123"


class TestEmbeddingsPerformanceAndLimits:
    """Test performance-related scenarios and limits."""

    @pytest.mark.asyncio
    async def test_create_maximum_allowed_batch_size(
        self, embeddings_resource, mock_client, sample_batch_embedding_response
    ):
        """Test creating embeddings with maximum allowed batch size."""
        max_batch = [f"text_{i}" for i in range(2048)]  # Maximum allowed
        mock_client.post.return_value = sample_batch_embedding_response

        result = await embeddings_resource.create(model="text-embedding-bge-m3", input=max_batch)

        assert result == sample_batch_embedding_response

        call_args = mock_client.post.call_args
        request_body = call_args[1]["json_data"]
        assert len(request_body["input"]) == 2048

    @pytest.mark.asyncio
    async def test_create_just_over_limit_error(self, embeddings_resource, mock_client):
        """Test that input just over limit raises appropriate error."""
        over_limit_batch = [f"text_{i}" for i in range(2049)]  # Just over limit

        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input=over_limit_batch)

        assert "2048 or fewer items" in str(exc_info.value)
        assert "but got 2049 items" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_create_significantly_over_limit_error(self, embeddings_resource, mock_client):
        """Test that significantly over-limit input raises appropriate error."""
        way_over_limit = [f"text_{i}" for i in range(5000)]  # Way over limit

        with pytest.raises(InvalidRequestError) as exc_info:
            await embeddings_resource.create(model="text-embedding-bge-m3", input=way_over_limit)

        assert "2048 or fewer items" in str(exc_info.value)
        assert "but got 5000 items" in str(exc_info.value)


class TestEmbeddingsTextOnlyInput:
    """Token-ID arrays are rejected with the SDK's documented input error.

    The API validates embeddings input as text only. The resource raises
    InvalidRequestError for bad input, so a token array must surface that
    rather than a raw Pydantic ValidationError.
    """

    @pytest.mark.asyncio
    async def test_flat_token_array_raises_invalid_request(self, embeddings_resource, mock_client):
        with pytest.raises(InvalidRequestError, match="text"):
            await embeddings_resource.create(model="text-embedding-bge-m3", input=[101, 1045, 102])
        mock_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_nested_token_array_raises_invalid_request(
        self, embeddings_resource, mock_client
    ):
        with pytest.raises(InvalidRequestError, match="text"):
            await embeddings_resource.create(
                model="text-embedding-bge-m3", input=[[101, 102], [103]]
            )
        mock_client.post.assert_not_called()
