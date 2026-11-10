"""Tests for build_model_id in venice_ai.utils.models."""

import pytest

from venice_ai.utils.models import build_model_id


class TestBuildModelId:
    """Test build_model_id function."""

    def test_no_params(self):
        assert build_model_id("llama-3.3-70b") == "llama-3.3-70b"

    def test_single_param(self):
        result = build_model_id("llama-3.3-70b", reasoning_effort="high")
        assert result == "llama-3.3-70b:reasoning_effort=high"

    def test_multiple_params(self):
        result = build_model_id(
            "llama-3.3-70b", reasoning_effort="high", max_completion_tokens=4096
        )
        assert "llama-3.3-70b:" in result
        assert "reasoning_effort=high" in result
        assert "max_completion_tokens=4096" in result
        assert "&" in result

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            build_model_id("")

    def test_colon_in_model_raises(self):
        with pytest.raises(ValueError, match="already contains a suffix"):
            build_model_id("model:existing_suffix")

    def test_bool_param(self):
        assert build_model_id("model", flag=True) == "model:flag=true"
        assert build_model_id("model", flag=False) == "model:flag=false"

    def test_bool_param_among_others(self):
        result = build_model_id("model", reasoning_effort="high", flag=False, tokens=1)
        assert result == "model:reasoning_effort=high&flag=false&tokens=1"

    def test_string_param_is_left_as_given(self):
        assert build_model_id("model", flag="True") == "model:flag=True"

    def test_int_param(self):
        result = build_model_id("model", tokens=1024)
        assert result == "model:tokens=1024"
