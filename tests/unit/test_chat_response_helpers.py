"""Unit tests for ChatCompletionResponse.parsed and parse_as."""

import pytest
from pydantic import BaseModel

from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)
from venice_ai.types.api.models import LLMModelPricing, PricingTier


def _make_response(
    content: str | list | None = '{"name": "Alice", "age": 30}',
) -> ChatCompletionResponse:
    """Helper to build a ChatCompletionResponse with given content."""
    return ChatCompletionResponse(
        id="resp-test",
        object="chat.completion",
        created=1000000,
        model="test-model",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None
        ),
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )


# ============================================================================
# text property tests
# ============================================================================


class TestTextProperty:
    """Tests for ChatCompletionResponse.text — first-choice content as plain str."""

    def test_text_str_content(self):
        resp = _make_response("hello world")
        assert resp.text == "hello world"

    def test_text_none_content(self):
        resp = _make_response(None)
        assert resp.text is None

    def test_text_empty_string_content(self):
        resp = _make_response("")
        assert resp.text == ""

    def test_text_multimodal_text_only(self):
        from venice_ai.core.models.common import TextContent

        resp = _make_response([TextContent(type="text", text="hello")])
        assert resp.text == "hello"

    def test_text_multimodal_joins_multiple_text_parts(self):
        from venice_ai.core.models.common import TextContent

        resp = _make_response(
            [
                TextContent(type="text", text="foo "),
                TextContent(type="text", text="bar"),
            ]
        )
        assert resp.text == "foo bar"

    def test_text_multimodal_skips_image_parts(self):
        from venice_ai.core.models.common import ImageContent, ImageUrl, TextContent

        resp = _make_response(
            [
                TextContent(type="text", text="describe this:"),
                ImageContent(type="image_url", image_url=ImageUrl(url="https://example.com/x.png")),
            ]
        )
        assert resp.text == "describe this:"

    def test_text_multimodal_image_only_returns_none(self):
        from venice_ai.core.models.common import ImageContent, ImageUrl

        resp = _make_response(
            [ImageContent(type="image_url", image_url=ImageUrl(url="https://example.com/x.png"))]
        )
        assert resp.text is None

    def test_text_no_choices_returns_none(self):
        resp = ChatCompletionResponse(
            id="resp-empty",
            object="chat.completion",
            created=1000000,
            model="test-model",
            choices=[],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        assert resp.text is None


# ============================================================================
# parsed property tests
# ============================================================================


class TestParsedProperty:
    """Tests for ChatCompletionResponse.parsed."""

    def test_parsed_dict(self):
        resp = _make_response('{"key": "value"}')
        result = resp.parsed
        assert result == {"key": "value"}

    def test_parsed_list(self):
        resp = _make_response("[1, 2, 3]")
        result = resp.parsed
        assert result == [1, 2, 3]

    def test_parsed_none_content(self):
        resp = _make_response(None)
        assert resp.parsed is None

    def test_parsed_multimodal_raises(self):
        from venice_ai.core.models.common import TextContent

        resp = _make_response([TextContent(type="text", text="hello")])
        with pytest.raises(TypeError, match="Cannot parse multimodal content"):
            _ = resp.parsed

    def test_parsed_invalid_json_raises(self):
        import json

        resp = _make_response("not valid json")
        with pytest.raises(json.JSONDecodeError):
            _ = resp.parsed


# ============================================================================
# parse_as method tests
# ============================================================================


class Person(BaseModel):
    name: str
    age: int


class TestParseAs:
    """Tests for ChatCompletionResponse.parse_as."""

    def test_parse_as_basic(self):
        resp = _make_response('{"name": "Alice", "age": 30}')
        person = resp.parse_as(Person)
        assert isinstance(person, Person)
        assert person.name == "Alice"
        assert person.age == 30

    def test_parse_as_none_content_raises(self):
        resp = _make_response(None)
        with pytest.raises(ValueError, match="has no content"):
            resp.parse_as(Person)

    def test_parse_as_multimodal_raises(self):
        from venice_ai.core.models.common import TextContent

        resp = _make_response([TextContent(type="text", text="hello")])
        with pytest.raises(ValueError, match="multimodal content"):
            resp.parse_as(Person)

    def test_parse_as_invalid_json_raises(self):
        from pydantic import ValidationError

        resp = _make_response("not json")
        with pytest.raises(ValidationError):
            resp.parse_as(Person)

    def test_parse_as_wrong_schema_raises(self):
        from pydantic import ValidationError

        resp = _make_response('{"wrong": "fields"}')
        with pytest.raises(ValidationError):
            resp.parse_as(Person)

    def test_parse_as_choice_index(self):
        """Test parse_as with a non-zero choice index."""
        resp = ChatCompletionResponse(
            id="resp-multi",
            object="chat.completion",
            created=1000000,
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "Alice", "age": 30}'),
                    finish_reason="stop",
                    stop_reason=None,
                ),
                ChatChoice(
                    index=1,
                    message=ChatMessage(role="assistant", content='{"name": "Bob", "age": 25}'),
                    finish_reason="stop",
                    stop_reason=None,
                ),
            ],
            usage=ChatUsage(
                prompt_tokens=10, completion_tokens=10, total_tokens=20, prompt_tokens_details=None
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        person = resp.parse_as(Person, choice_index=1)
        assert person.name == "Bob"
        assert person.age == 25


# ============================================================================
# ToolCallFunction.arguments_dict tests
# ============================================================================


class TestToolCallFunctionArgumentsDict:
    """Tests for the auto-parsed arguments_dict property on tool call functions."""

    def test_valid_object(self):
        from venice_ai.types.api.chat import ToolCallFunction

        fn = ToolCallFunction(name="lookup", arguments='{"city": "Reno", "n": 3}')
        assert fn.arguments_dict == {"city": "Reno", "n": 3}

    def test_empty_object(self):
        from venice_ai.types.api.chat import ToolCallFunction

        fn = ToolCallFunction(name="ping", arguments="{}")
        assert fn.arguments_dict == {}

    def test_invalid_json_raises_decode_error(self):
        import json as _json

        from venice_ai.types.api.chat import ToolCallFunction

        fn = ToolCallFunction(name="bad", arguments="{not json")
        with pytest.raises(_json.JSONDecodeError):
            _ = fn.arguments_dict

    def test_non_object_json_raises_type_error(self):
        from venice_ai.types.api.chat import ToolCallFunction

        fn = ToolCallFunction(name="weird", arguments="[1, 2, 3]")
        with pytest.raises(TypeError, match="JSON object"):
            _ = fn.arguments_dict

    def test_does_not_mutate_arguments(self):
        from venice_ai.types.api.chat import ToolCallFunction

        raw = '{"k": "v"}'
        fn = ToolCallFunction(name="f", arguments=raw)
        _ = fn.arguments_dict
        # Property should not affect the underlying string
        assert fn.arguments == raw


# ============================================================================
# ToolCallFunction.parse_as
# ============================================================================


class TestToolCallFunctionParseAs:
    """Tests for typed Pydantic-validated tool argument parsing."""

    def test_validates_into_model(self):
        from venice_ai.types.api.chat import ToolCallFunction

        class WeatherArgs(BaseModel):
            location: str
            unit: str = "fahrenheit"

        fn = ToolCallFunction(
            name="get_weather",
            arguments='{"location": "Paris", "unit": "celsius"}',
        )
        result = fn.parse_as(WeatherArgs)
        assert isinstance(result, WeatherArgs)
        assert result.location == "Paris"
        assert result.unit == "celsius"

    def test_uses_default_for_omitted_field(self):
        from venice_ai.types.api.chat import ToolCallFunction

        class WeatherArgs(BaseModel):
            location: str
            unit: str = "fahrenheit"

        fn = ToolCallFunction(name="get_weather", arguments='{"location": "Reno"}')
        result = fn.parse_as(WeatherArgs)
        assert result.unit == "fahrenheit"

    def test_validation_error_on_wrong_types(self):
        from pydantic import ValidationError

        from venice_ai.types.api.chat import ToolCallFunction

        class StrictArgs(BaseModel):
            count: int

        fn = ToolCallFunction(name="x", arguments='{"count": "not-a-number"}')
        with pytest.raises(ValidationError):
            fn.parse_as(StrictArgs)

    def test_invalid_json_raises_validation_error(self):
        # Pydantic v2 model_validate_json wraps both invalid-JSON and
        # schema-mismatch failures into a single ValidationError.
        from pydantic import ValidationError

        from venice_ai.types.api.chat import ToolCallFunction

        class Args(BaseModel):
            x: int

        fn = ToolCallFunction(name="x", arguments="{not json")
        with pytest.raises(ValidationError):
            fn.parse_as(Args)


# ============================================================================
# ChatCompletionResponse.thinking_blocks
# ============================================================================


class TestThinkingBlocksProperty:
    """Tests for the dual-shape reasoning extractor on chat responses."""

    @staticmethod
    def _resp(*, content=None, reasoning_content=None):
        return ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=content,
                        reasoning_content=reasoning_content,
                    ),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )

    def test_separate_reasoning_content_field(self):
        # zai-org-glm-4.7 shape: reasoning_content populated, content carries answer alone.
        r = self._resp(
            content="The answer is 3.",
            reasoning_content="I need to add 1+2.",
        )
        assert r.thinking_blocks == ["I need to add 1+2."]

    def test_inline_think_tags(self):
        # venice-uncensored shape: <think>...</think> inside content.
        r = self._resp(content="<think>step one</think>The answer is 3.")
        assert r.thinking_blocks == ["step one"]

    def test_inline_thinking_tags(self):
        r = self._resp(content="<thinking>step A</thinking>final")
        assert r.thinking_blocks == ["step A"]

    def test_inline_multiple_blocks(self):
        r = self._resp(content="<think>first</think>middle<think>second</think>end")
        assert r.thinking_blocks == ["first", "second"]

    def test_no_reasoning_anywhere_returns_empty(self):
        r = self._resp(content="Plain answer with no thinking.")
        assert r.thinking_blocks == []

    def test_empty_content_returns_empty(self):
        r = self._resp(content=None)
        assert r.thinking_blocks == []

    def test_no_choices_returns_empty(self):
        r = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="m",
            choices=[],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        assert r.thinking_blocks == []

    def test_reasoning_content_prefers_over_inline_tags(self):
        # Edge case: both populated — the dedicated field wins.
        r = self._resp(
            content="<think>inline thoughts</think>visible",
            reasoning_content="dedicated field thoughts",
        )
        assert r.thinking_blocks == ["dedicated field thoughts"]

    def test_whitespace_only_reasoning_content_falls_back_to_tags(self):
        # If reasoning_content is just whitespace, we still try regex on content.
        r = self._resp(
            content="<think>real reasoning</think>answer",
            reasoning_content="   ",
        )
        assert r.thinking_blocks == ["real reasoning"]


# ============================================================================
# ChatUsage cache accessors
# ============================================================================


class TestChatUsageCacheAccessors:
    """The three wire shapes Venice sends for prompt-cache counts."""

    @staticmethod
    def _usage(**fields):
        return ChatUsage.model_validate(
            {"prompt_tokens": 3000, "completion_tokens": 5, "total_tokens": 3005, **fields}
        )

    def test_absent_details_read_as_zero(self):
        usage = self._usage()
        assert usage.cached_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_nested_only(self):
        usage = self._usage(prompt_tokens_details={"cached_tokens": 1664})
        assert usage.cached_tokens == 1664
        assert usage.cache_write_tokens == 0

    def test_nested_with_top_level_mirror_is_not_summed(self):
        usage = self._usage(
            prompt_tokens_details={"cached_tokens": 2880, "cache_creation_input_tokens": 87},
            cache_read_input_tokens=2880,
            cache_creation_input_tokens=87,
        )
        assert usage.cached_tokens == 2880
        assert usage.cache_write_tokens == 87

    def test_top_level_only(self):
        usage = self._usage(cache_read_input_tokens=512, cache_creation_input_tokens=64)
        assert usage.cached_tokens == 512
        assert usage.cache_write_tokens == 64

    def test_responses_usage_cached_tokens(self):
        from venice_ai.types.api.responses import ResponsesUsage

        base = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
        assert ResponsesUsage.model_validate(base).cached_tokens == 0
        with_details = {**base, "input_tokens_details": {"cached_tokens": 8}}
        assert ResponsesUsage.model_validate(with_details).cached_tokens == 8


# ============================================================================
# ChatUsage.__str__
# ============================================================================


class TestChatUsageStr:
    """Concise human-readable formatting for token usage."""

    def test_basic_str(self):
        u = ChatUsage(
            prompt_tokens=1234,
            completion_tokens=567,
            total_tokens=1801,
            prompt_tokens_details=None,
        )
        assert str(u) == "prompt: 1234 / completion: 567 / total: 1801"

    def test_str_includes_cache_when_set(self):
        u = ChatUsage(
            prompt_tokens=1234,
            completion_tokens=567,
            total_tokens=1801,
            prompt_tokens_details=None,
            cache_read_input_tokens=1100,
        )
        assert str(u) == (
            "prompt: 1234 (cache read: 1100, write: 0) / completion: 567 / total: 1801"
        )

    def test_str_reads_nested_details_and_writes(self):
        u = ChatUsage.model_validate(
            {
                "prompt_tokens": 4130,
                "completion_tokens": 5,
                "total_tokens": 4135,
                "prompt_tokens_details": {"cached_tokens": 0, "cache_creation_input_tokens": 4122},
            }
        )
        assert str(u) == "prompt: 4130 (cache read: 0, write: 4122) / completion: 5 / total: 4135"

    def test_str_omits_cache_when_zero(self):
        u = ChatUsage(
            prompt_tokens=1234,
            completion_tokens=567,
            total_tokens=1801,
            prompt_tokens_details=None,
            cache_read_input_tokens=0,
        )
        assert "cache" not in str(u)

    def test_str_includes_reasoning_tokens_when_set(self):
        from venice_ai.types.api.chat import CompletionTokensDetails

        u = ChatUsage(
            prompt_tokens=10,
            completion_tokens=200,
            total_tokens=210,
            prompt_tokens_details=None,
            completion_tokens_details=CompletionTokensDetails(
                reasoning_tokens=150,
                audio_tokens=None,
                image_tokens=None,
            ),
        )
        assert str(u) == "prompt: 10 / completion: 200 (reasoning: 150) / total: 210"

    def test_str_with_zero_tokens(self):
        u = ChatUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=None,
        )
        assert str(u) == "prompt: 0 / completion: 0 / total: 0"


# ============================================================================
# ChatCompletionResponse.summary() tests
# ============================================================================


def _make_pricing(input_usd: float, output_usd: float) -> LLMModelPricing:
    """Build a minimal LLMModelPricing for tests."""
    return LLMModelPricing(
        input=PricingTier(usd=input_usd, diem=input_usd),
        output=PricingTier(usd=output_usd, diem=output_usd),
        cache_input=None,
    )


class TestSummary:
    """Tests for ChatCompletionResponse.summary()."""

    def test_basic_summary_no_pricing(self):
        # model · usage_str · finish=stop
        out = _make_response().summary()
        assert "test-model" in out
        assert "prompt: 10 / completion: 5 / total: 15" in out
        assert "finish=stop" in out
        assert "$" not in out  # no cost segment without pricing
        # Pipe-style join
        assert " · " in out

    def test_summary_with_pricing_includes_cost(self):
        # 10 prompt @ $1/M = $0.00001, 5 completion @ $2/M = $0.00001 → $0.00002
        # Formatted to 4 decimals = $0.0000
        pricing = _make_pricing(input_usd=1.0, output_usd=2.0)
        out = _make_response().summary(pricing=pricing)
        assert "$0.0000" in out
        assert "test-model" in out

    def test_summary_with_pricing_realistic_cost(self):
        # 1000 prompt @ $3/M = $0.003, 500 completion @ $15/M = $0.0075 → $0.0105
        resp = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="big-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="hi"),
                    finish_reason="stop",
                    stop_reason=None,
                )
            ],
            usage=ChatUsage(
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                prompt_tokens_details=None,
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        pricing = _make_pricing(input_usd=3.0, output_usd=15.0)
        out = resp.summary(pricing=pricing)
        assert "$0.0105" in out

    def test_summary_omits_usage_when_none(self):
        resp = _make_response()
        # ChatUsage is required-ish but optional on the response — clear it
        resp_no_usage = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="m",
            choices=resp.choices,
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        out = resp_no_usage.summary()
        assert "prompt:" not in out
        assert "m" in out
        assert "finish=stop" in out

    def test_summary_omits_finish_when_no_choices(self):
        resp = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="m",
            choices=[],
            usage=ChatUsage(
                prompt_tokens=1, completion_tokens=2, total_tokens=3, prompt_tokens_details=None
            ),
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        out = resp.summary()
        assert "finish=" not in out
        assert "m" in out
        assert "total: 3" in out

    def test_summary_returns_non_empty_str_with_minimum_fields(self):
        # Even with the absolute minimum (no usage, no choices), model is enough.
        resp = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="bare-model",
            choices=[],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        out = resp.summary()
        assert out == "bare-model"

    def test_summary_finish_reason_from_first_choice(self):
        # Different finish reasons surface correctly
        resp = ChatCompletionResponse(
            id="r",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="..."),
                    finish_reason="length",
                    stop_reason=None,
                )
            ],
            usage=None,
            prompt_logprobs=None,
            venice_parameters=None,
            service_tier=None,
            system_fingerprint=None,
            kv_transfer_params=None,
        )
        assert "finish=length" in resp.summary()

    def test_summary_pricing_segment_position(self):
        # Cost segment goes after usage and before finish_reason
        pricing = _make_pricing(input_usd=1.0, output_usd=1.0)
        out = _make_response().summary(pricing=pricing)
        assert out.index("$") > out.index("total: 15")
        assert out.index("$") < out.index("finish=stop")


class TestCacheCreationInputTokens:
    """NEW-1: cache-write token count reachable at both nesting levels."""

    def test_prompt_tokens_details_cache_creation_field(self):
        from venice_ai.types.api.common import PromptTokensDetails

        details = PromptTokensDetails.model_validate(
            {"cached_tokens": 5, "cache_creation_input_tokens": 64}
        )
        assert details.cached_tokens == 5
        assert details.cache_creation_input_tokens == 64

    def test_prompt_tokens_details_cache_creation_defaults_none(self):
        from venice_ai.types.api.common import PromptTokensDetails

        details = PromptTokensDetails.model_validate({"cached_tokens": 5})
        assert details.cache_creation_input_tokens is None

    def test_prompt_tokens_details_cache_creation_rejects_negative(self):
        from venice_ai.types.api.common import PromptTokensDetails

        with pytest.raises(ValueError):
            PromptTokensDetails.model_validate({"cache_creation_input_tokens": -1})

    def test_chat_usage_round_trips_both_cache_fields(self):
        usage = ChatUsage.model_validate(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cache_read_input_tokens": 80,
                "cache_creation_input_tokens": 64,
                "prompt_tokens_details": {
                    "cached_tokens": 80,
                    "cache_creation_input_tokens": 64,
                },
            }
        )
        assert usage.cache_read_input_tokens == 80
        assert usage.cache_creation_input_tokens == 64
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_creation_input_tokens == 64

        round_tripped = ChatUsage.model_validate(usage.model_dump())
        assert round_tripped.cache_creation_input_tokens == 64
        assert round_tripped.cache_read_input_tokens == 80
