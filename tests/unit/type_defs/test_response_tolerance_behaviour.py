"""What a response containing values the SDK has not seen parses into.

``test_response_models_tolerant.py`` gates the model structure; these tests pin
the behaviour callers rely on: unknown parts stay readable, known parts keep
their types, and what the caller built is left alone.
"""

from __future__ import annotations

from venice_ai.types.api import (
    ChatMessage,
    ChatMessageImageContent,
    ChatMessageTextContent,
    ResponsesMessageOutput,
    ResponsesOutputText,
    ResponsesReasoningOutput,
    ResponsesResponse,
    TextContent,
    ToolCall,
)

_RESPONSE = {"id": "r", "object": "response", "created_at": 1, "model": "m", "status": "completed"}


class TestChatContentParts:
    def test_api_parts_keep_types_and_unknown_fields(self):
        message = ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "", "annotations": [1]},
                    {"type": "image_url", "image_url": {"url": "u", "detail": "ultra"}},
                    {"type": "refusal", "refusal": "no"},
                ],
            }
        )
        text, image, unknown = message.content
        assert isinstance(text, ChatMessageTextContent) and isinstance(text, TextContent)
        assert text.text == "" and text.model_extra == {"annotations": [1]}
        assert isinstance(image, ChatMessageImageContent)
        assert image.image_url.detail == "ultra"
        assert unknown == {"type": "refusal", "refusal": "no"}

    def test_parts_the_caller_built_are_kept(self):
        part = TextContent(type="text", text="hi")
        message = ChatMessage(role="user", content=[part])
        assert type(message.content[0]) is TextContent


class TestToolCalls:
    def test_custom_tool_call_parses_and_keeps_its_payload(self):
        call = ToolCall.model_validate(
            {"id": "c", "type": "custom", "custom": {"name": "apply_patch", "input": "x"}}
        )
        assert call.function is None
        assert call.model_extra == {"custom": {"name": "apply_patch", "input": "x"}}

    def test_function_tool_call_is_unchanged(self):
        call = ToolCall.model_validate(
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        )
        assert call.function is not None and call.function.name == "f"


class TestResponsesOutput:
    def test_unknown_content_part_keeps_the_message_typed(self):
        response = ResponsesResponse.model_validate(
            {
                **_RESPONSE,
                "output": [
                    {
                        "type": "message",
                        "id": "m",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Hello"},
                            {"type": "refusal", "refusal": "no"},
                        ],
                    }
                ],
            }
        )
        message = response.output[0]
        assert isinstance(message, ResponsesMessageOutput)
        text, refusal = message.content
        assert isinstance(text, ResponsesOutputText) and text.text == "Hello"
        assert refusal == {"type": "refusal", "refusal": "no"}

    def test_reasoning_summary_keeps_strings_known_parts_and_unknown_parts(self):
        response = ResponsesResponse.model_validate(
            {
                **_RESPONSE,
                "output": [
                    {
                        "type": "reasoning",
                        "id": "r1",
                        "summary": [
                            "plain",
                            {"type": "summary_text", "text": "typed"},
                            {"type": "new_part", "x": 1},
                        ],
                    }
                ],
            }
        )
        reasoning = response.output[0]
        assert isinstance(reasoning, ResponsesReasoningOutput)
        plain, typed, unknown = reasoning.summary or []
        assert plain == "plain"
        assert getattr(typed, "text", None) == "typed"
        assert unknown == {"type": "new_part", "x": 1}


class TestAuditFollowUps:
    def test_text_part_with_odd_cache_control_or_null_text_keeps_its_type(self):
        message = ChatMessage.model_validate(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral", "ttl": 300},
                    },
                    {"type": "text", "text": None},
                ],
            }
        )
        assert [type(p) for p in message.content] == [ChatMessageTextContent] * 2
        assert [p.text for p in message.content] == ["hi", ""]

    def test_assistant_history_keeps_parts_the_sdk_does_not_type(self):
        from venice_ai.types.api import ChatCompletionResponse
        from venice_ai.types.api.requests.chat import AssistantMessage

        response = ChatCompletionResponse.model_validate(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "hi"},
                                {"type": "refusal", "refusal": "no"},
                                {"type": "image_url", "image_url": {"url": "u"}},
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )
        history = AssistantMessage.from_response(response)
        assert history.model_dump(exclude_none=True)["content"] == [
            {"type": "text", "text": "hi"},
            {"type": "refusal", "refusal": "no"},
            {"type": "image_url", "image_url": {"url": "u"}},
        ]

    def test_malformed_text_part_is_still_rejected_in_history(self):
        import pytest
        from pydantic import ValidationError

        from venice_ai.types.api.requests.chat import AssistantMessage

        with pytest.raises(ValidationError):
            AssistantMessage(content=[{"type": "text", "txt": "typo"}])

    def test_retrieve_unions_route_by_status(self):
        from pydantic import TypeAdapter

        from venice_ai.types import (
            MusicRetrieveResponse,
            VideoRetrieveResponse,
            VoiceChangerRetrieveResponse,
        )

        video = TypeAdapter(VideoRetrieveResponse)
        assert (
            type(video.validate_python({"status": "COMPLETED"})).__name__ == "VideoCompletedStatus"
        )
        assert type(video.validate_python({"status": "FAILED"})).__name__ == "VideoFailedStatus"
        assert type(video.validate_python({"status": "QUEUED"})).__name__ == "VideoProcessingStatus"
        music = TypeAdapter(MusicRetrieveResponse)
        assert (
            type(music.validate_python({"status": "COMPLETED"})).__name__ == "MusicCompletedStatus"
        )
        voice = TypeAdapter(VoiceChangerRetrieveResponse)
        assert (
            type(voice.validate_python({"status": "COMPLETED"})).__name__
            == "VoiceChangerCompletedStatus"
        )

    def test_streamed_custom_tool_call_assembles_with_its_payload(self):
        from venice_ai.streaming import _assembled_tool_call, _merge_tool_call_delta
        from venice_ai.types.api.streaming import ChatCompletionChunkToolCall

        calls: dict = {}
        for delta in (
            {"index": 0, "id": "t", "type": "custom", "custom": {"name": "patch", "input": "ab"}},
            {"index": 0, "custom": {"input": "cd"}},
            {
                "index": 1,
                "id": "f",
                "type": "function",
                "function": {"name": "g", "arguments": "{"},
            },
            {"index": 1, "function": {"arguments": "}"}},
        ):
            _merge_tool_call_delta(calls, ChatCompletionChunkToolCall.model_validate(delta))
        assembled = [_assembled_tool_call(entry) for _, entry in sorted(calls.items())]
        assert assembled == [
            {"id": "t", "type": "custom", "custom": {"name": "patch", "input": "abcd"}},
            {"id": "f", "type": "function", "function": {"name": "g", "arguments": "{}"}},
        ]
        call = ToolCall.model_validate(assembled[0])
        assert call.function is None and call.model_extra == {
            "custom": {"name": "patch", "input": "abcd"}
        }
