"""Tests for EmbeddingsRequest.validate_input edge cases."""

import pytest
from pydantic import ValidationError

from venice_ai.types.api.requests.embeddings import EmbeddingsRequest

# Common kwargs for optional fields to satisfy Pylance's strict checking
_OPTS = {"dimensions": None, "encoding_format": "float", "user": None}


class TestEmbeddingsValidateInput:
    """Test EmbeddingsRequest input validator edge cases."""

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError, match="Input string cannot be empty"):
            EmbeddingsRequest(input="", model="text-embedding-ada-002", **_OPTS)

    def test_empty_list_raises(self):
        with pytest.raises(ValidationError, match="Input array cannot be empty"):
            EmbeddingsRequest(input=[], model="text-embedding-ada-002", **_OPTS)

    def test_oversized_list_raises(self):
        big_list = [f"text_{i}" for i in range(2049)]
        with pytest.raises(ValidationError, match="Input array cannot exceed 2048 items"):
            EmbeddingsRequest(input=big_list, model="text-embedding-ada-002", **_OPTS)

    def test_empty_string_in_list_raises(self):
        with pytest.raises(ValidationError, match="Input strings cannot be empty"):
            EmbeddingsRequest(input=["valid", ""], model="text-embedding-ada-002", **_OPTS)

    def test_valid_string(self):
        req = EmbeddingsRequest(input="hello world", model="text-embedding-ada-002", **_OPTS)
        assert req.input == "hello world"

    def test_valid_list(self):
        req = EmbeddingsRequest(input=["a", "b"], model="text-embedding-ada-002", **_OPTS)
        assert req.input == ["a", "b"]

    def test_exactly_2048_items(self):
        items = [f"text_{i}" for i in range(2048)]
        req = EmbeddingsRequest(input=items, model="text-embedding-ada-002", **_OPTS)
        assert len(req.input) == 2048


class TestEmbeddingsTextOnly:
    """Token-ID array inputs are rejected client-side.

    The API stopped accepting token-ID arrays for embeddings; requests that
    pass them return a validation error rather than failing upstream, so the
    SDK fails fast with a clearer message than a server round-trip gives.
    """

    def test_flat_token_id_array_raises(self):
        with pytest.raises(ValidationError, match="text"):
            EmbeddingsRequest(input=[1, 2, 3], model="text-embedding-bge-m3", **_OPTS)  # type: ignore[arg-type]

    def test_nested_token_id_array_raises(self):
        with pytest.raises(ValidationError, match="text"):
            EmbeddingsRequest(input=[[1, 2], [3, 4]], model="text-embedding-bge-m3", **_OPTS)  # type: ignore[arg-type]

    def test_string_list_still_accepted(self):
        request = EmbeddingsRequest(
            input=["hello", "world"], model="text-embedding-bge-m3", **_OPTS
        )
        assert request.input == ["hello", "world"]
