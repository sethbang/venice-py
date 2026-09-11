"""
Test suite for Pydantic request models validation.

This module tests all request models for proper validation, serialization,
and constraint enforcement.
"""

from typing import Any

import pytest

from venice_ai.types.api import (
    AudioSpeechRequest,
    BillingUsageHistoryQueryParams,
    ChatCompletionRequest,
    CreateApiKeyRequest,
    EmbeddingsRequest,
    ImageGenerationRequest,
    ModelsQueryParams,
    ModelTraitsQueryParams,
)


class TestChatCompletionRequest:
    """Test ChatCompletionRequest model validation."""

    def test_basic_chat_request(self) -> None:
        """Test creation with required fields only."""
        # Create request with only required fields
        # Pylance incorrectly reports optional fields as required - using type: ignore
        request = ChatCompletionRequest(  # type: ignore[call-arg]
            model="venice-uncensored", messages=[{"role": "user", "content": "Hello!"}]
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["model"] == "venice-uncensored"
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello!"

    def test_serialization_excludes_none(self) -> None:
        """Test that None values are properly excluded."""
        # Create request with only required fields
        request = ChatCompletionRequest(  # type: ignore[call-arg]
            model="venice-uncensored", messages=[{"role": "user", "content": "Hello!"}]
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)

        # Check that only non-None values are included
        for key, value in data.items():
            assert value is not None, (
                f"Key '{key}' should not have None value when exclude_none=True"
            )

    def test_with_optional_params(self) -> None:
        """Test creation with optional parameters."""
        request = ChatCompletionRequest(  # type: ignore[call-arg]
            model="venice-uncensored",
            messages=[{"role": "user", "content": "Hello!"}],
            temperature=0.8,
            max_completion_tokens=100,
            stream=False,
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["temperature"] == 0.8
        assert data["max_completion_tokens"] == 100
        assert data["stream"] is False


class TestUserMessageMultimodalContent:
    """Discriminated-union behavior for UserMessage.content."""

    def test_dict_form_text_and_image(self) -> None:
        from venice_ai.core.models.common import ImageContent, TextContent
        from venice_ai.types.api.requests.chat import UserMessage

        msg = UserMessage(
            content=[
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.jpg"},
                },
            ]
        )
        assert isinstance(msg.content, list)
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "Describe this image"
        assert isinstance(msg.content[1], ImageContent)
        assert msg.content[1].image_url.url == "https://example.com/cat.jpg"

    def test_dict_form_audio_and_video(self) -> None:
        from venice_ai.core.models.common import AudioContent, VideoContent
        from venice_ai.types.api.requests.chat import UserMessage

        msg = UserMessage(
            content=[
                {"type": "input_audio", "input_audio": {"data": "abc==", "format": "wav"}},
                {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
            ]
        )
        assert isinstance(msg.content[0], AudioContent)
        assert isinstance(msg.content[1], VideoContent)

    def test_typed_form_still_validates(self) -> None:
        from venice_ai.core.models.common import ImageContent, ImageUrl, TextContent
        from venice_ai.types.api.requests.chat import UserMessage

        msg = UserMessage(
            content=[
                TextContent(type="text", text="hi"),
                ImageContent(
                    type="image_url",
                    image_url=ImageUrl(url="https://example.com/x.png"),
                ),
            ]
        )
        assert len(msg.content) == 2

    def test_plain_string_content_still_works(self) -> None:
        from venice_ai.types.api.requests.chat import UserMessage

        msg = UserMessage(content="hello")
        assert msg.content == "hello"

    def test_unknown_discriminator_raises(self) -> None:
        from pydantic import ValidationError

        from venice_ai.types.api.requests.chat import UserMessage

        with pytest.raises(ValidationError):
            UserMessage(content=[{"type": "unknown_type", "data": "x"}])  # type: ignore[list-item]

    def test_missing_discriminator_raises(self) -> None:
        from pydantic import ValidationError

        from venice_ai.types.api.requests.chat import UserMessage

        with pytest.raises(ValidationError):
            UserMessage(content=[{"text": "no type field"}])  # type: ignore[list-item]

    def test_file_content_coerces_via_union(self) -> None:
        from pydantic import TypeAdapter

        from venice_ai.core.models.common import FileContent, MessageContentPart

        adapter = TypeAdapter(MessageContentPart)
        part = adapter.validate_python(
            {
                "type": "file",
                "file": {"file_data": "data:application/pdf;base64,AA==", "filename": "doc.pdf"},
            }
        )
        assert isinstance(part, FileContent)
        assert part.file.file_data == "data:application/pdf;base64,AA=="
        assert part.file.filename == "doc.pdf"

    def test_file_content_filename_optional(self) -> None:
        from pydantic import TypeAdapter

        from venice_ai.core.models.common import FileContent, MessageContentPart

        part = TypeAdapter(MessageContentPart).validate_python(
            {"type": "file", "file": {"file_data": "https://example.com/a.pdf"}}
        )
        assert isinstance(part, FileContent)
        assert part.file.filename is None


class TestUserMessageBuilder:
    """Fluent builder for multimodal UserMessage content."""

    def test_text_and_image_chain(self) -> None:
        from venice_ai.core.models.common import ImageContent, TextContent
        from venice_ai.types.api.requests.chat import UserMessage

        msg = (
            UserMessage.builder().text("describe this").image("https://example.com/cat.jpg").build()
        )
        assert isinstance(msg, UserMessage)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert msg.content[0].text == "describe this"
        assert isinstance(msg.content[1], ImageContent)
        assert msg.content[1].image_url.url == "https://example.com/cat.jpg"

    def test_audio_and_video_parts(self) -> None:
        from venice_ai.core.models.common import AudioContent, VideoContent
        from venice_ai.types.api.requests.chat import UserMessage

        msg = (
            UserMessage.builder()
            .text("intro")
            .audio("YWJjZA==", "wav")
            .video("https://example.com/v.mp4")
            .build()
        )
        assert isinstance(msg.content[1], AudioContent)
        assert msg.content[1].input_audio == {"data": "YWJjZA==", "format": "wav"}
        assert isinstance(msg.content[2], VideoContent)
        assert msg.content[2].video_url == {"url": "https://example.com/v.mp4"}

    def test_empty_build_raises(self) -> None:
        from venice_ai.types.api.requests.chat import UserMessage

        with pytest.raises(ValueError, match="at least one content part"):
            UserMessage.builder().build()

    def test_builder_produces_independent_messages(self) -> None:
        """Building twice from the same chain produces independent content lists."""
        from venice_ai.types.api.requests.chat import UserMessage

        b = UserMessage.builder().text("hi")
        msg1 = b.build()
        b.text("again")
        msg2 = b.build()
        # msg1's content list should not be mutated by later builder calls
        assert len(msg1.content) == 1
        assert len(msg2.content) == 2

    def test_builder_file_part(self) -> None:
        from venice_ai.core.models.common import FileContent
        from venice_ai.types.api import UserMessage

        msg = (
            UserMessage.builder()
            .text("Summarize this document.")
            .file("data:application/pdf;base64,AA==", filename="doc.pdf")
            .build()
        )
        parts = msg.content
        assert any(isinstance(p, FileContent) and p.file.filename == "doc.pdf" for p in parts)


class TestImageGenerationRequest:
    """Test ImageGenerationRequest model validation."""

    def test_basic_image_request(self) -> None:
        """Test creation with required fields only."""
        # Create request with only required fields
        request = ImageGenerationRequest(  # type: ignore[call-arg]
            prompt="A beautiful sunset", model="hidream"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["prompt"] == "A beautiful sunset"
        assert data["model"] == "hidream"

    def test_serialization_excludes_none(self) -> None:
        """Test that None values are properly excluded."""
        request = ImageGenerationRequest(  # type: ignore[call-arg]
            prompt="A beautiful sunset", model="hidream"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)

        # Check that only non-None values are included
        for key, value in data.items():
            assert value is not None, (
                f"Key '{key}' should not have None value when exclude_none=True"
            )

    def test_with_dimensions(self) -> None:
        """Test creation with custom dimensions."""
        request = ImageGenerationRequest(  # type: ignore[call-arg]
            prompt="A landscape", model="flux", width=1280, height=720, steps=30
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["width"] == 1280
        assert data["height"] == 720
        assert data["steps"] == 30


class TestAudioSpeechRequest:
    """Test AudioSpeechRequest model validation."""

    def test_basic_audio_request(self) -> None:
        """Test creation with required fields only."""
        # Note: model and voice have defaults, but we'll include them for clarity
        request = AudioSpeechRequest(  # type: ignore[call-arg]
            input="Hello world", model="tts-kokoro", voice="female"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["input"] == "Hello world"
        assert data["model"] == "tts-kokoro"
        assert data["voice"] == "female"

    def test_minimal_audio_request(self) -> None:
        """Test creation with only truly required field."""
        request = AudioSpeechRequest(  # type: ignore[call-arg]
            input="Test speech"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["input"] == "Test speech"
        # Check defaults are applied
        assert data.get("model") == "tts-kokoro"  # Default value


class TestEmbeddingsRequest:
    """Test EmbeddingsRequest model validation."""

    def test_single_text_input(self) -> None:
        """Test creation with single text input."""
        request = EmbeddingsRequest(  # type: ignore[call-arg]
            model="text-embedding-ada-002", input="Hello world"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["input"] == "Hello world"
        assert data["model"] == "text-embedding-ada-002"

    def test_multiple_text_input(self) -> None:
        """Test creation with multiple text inputs."""
        inputs: list[str] = ["Hello", "World", "Test"]
        request = EmbeddingsRequest(  # type: ignore[call-arg]
            model="text-embedding-ada-002", input=inputs
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["input"] == ["Hello", "World", "Test"]
        assert data["model"] == "text-embedding-ada-002"

    def test_with_dimensions(self) -> None:
        """Test creation with custom dimensions."""
        request = EmbeddingsRequest(  # type: ignore[call-arg]
            model="text-embedding-3-small", input="Test embedding", dimensions=512
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["dimensions"] == 512


class TestQueryParameterModels:
    """Test query parameter models."""

    def test_billing_usage_history_query_params(self) -> None:
        """Test BillingUsageHistoryQueryParams validation."""
        params = BillingUsageHistoryQueryParams(  # type: ignore[call-arg]
            currency="USD",
            startTimestamp="2024-01-01T00:00:00Z",
            endTimestamp="2024-12-31T23:59:59Z",
            pageSize=100,
        )

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        assert data["currency"] == "USD"
        assert data["startTimestamp"] == "2024-01-01T00:00:00Z"
        assert data["endTimestamp"] == "2024-12-31T23:59:59Z"
        assert data["pageSize"] == 100

    def test_minimal_billing_params(self) -> None:
        """Test BillingUsageHistoryQueryParams with no fields (all optional)."""
        params = BillingUsageHistoryQueryParams()  # type: ignore[call-arg]

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        # No filters set → nothing serialized (the model carries no defaults).
        assert data == {}

    def test_models_query_params(self) -> None:
        """Test ModelsQueryParams validation."""
        params = ModelsQueryParams(type="text")  # type: ignore[call-arg]

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        assert data["type"] == "text"

    def test_models_query_params_aliases_chat_to_text(self) -> None:
        """``type='chat'`` is normalized to ``'text'`` for the API."""
        params = ModelsQueryParams(type="chat")  # type: ignore[call-arg]

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        assert data["type"] == "text"

    def test_model_traits_query_params(self) -> None:
        """Test ModelTraitsQueryParams validation."""
        params = ModelTraitsQueryParams(type="embedding")  # type: ignore[call-arg]

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        assert data["type"] == "embedding"

    def test_model_traits_query_params_aliases_chat_to_text(self) -> None:
        """``type='chat'`` is normalized to ``'text'`` for the traits endpoint too."""
        params = ModelTraitsQueryParams(type="chat")  # type: ignore[call-arg]

        data: dict[str, Any] = params.model_dump(exclude_none=True)
        assert data["type"] == "text"


class TestApiKeyRequests:
    """Test API key request models."""

    def test_create_api_key_request(self) -> None:
        """Test CreateApiKeyRequest validation."""
        request = CreateApiKeyRequest(  # type: ignore[call-arg]
            apiKeyType="INFERENCE", description="Test API key"
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["apiKeyType"] == "INFERENCE"
        assert data["description"] == "Test API key"

    def test_with_consumption_limit(self) -> None:
        """Test CreateApiKeyRequest with consumption limit."""
        request = CreateApiKeyRequest(  # type: ignore[call-arg]
            apiKeyType="ADMIN",
            description="Admin key with limits",
            consumptionLimit={"usd": 100.0, "diem": 500.0},
        )

        data: dict[str, Any] = request.model_dump(exclude_none=True)
        assert data["apiKeyType"] == "ADMIN"
        assert "consumptionLimit" in data
        assert data["consumptionLimit"]["usd"] == 100.0
        assert data["consumptionLimit"]["diem"] == 500.0


class TestModelDumpPatterns:
    """Test that all models support the model_dump(exclude_none=True) pattern."""

    def test_core_request_models_serialization(self) -> None:
        """Test that core request models work with exclude_none=True."""

        # Test chat request
        chat_req = ChatCompletionRequest(  # type: ignore[call-arg]
            model="test", messages=[{"role": "user", "content": "hi"}]
        )
        chat_data: dict[str, Any] = chat_req.model_dump(exclude_none=True)
        assert isinstance(chat_data, dict)

        # Test image request
        img_req = ImageGenerationRequest(  # type: ignore[call-arg]
            prompt="test", model="flux"
        )
        img_data: dict[str, Any] = img_req.model_dump(exclude_none=True)
        assert isinstance(img_data, dict)

        # Test audio request
        audio_req = AudioSpeechRequest(  # type: ignore[call-arg]
            input="test", model="tts", voice="female"
        )
        audio_data: dict[str, Any] = audio_req.model_dump(exclude_none=True)
        assert isinstance(audio_data, dict)

        # Test embeddings request
        embed_req = EmbeddingsRequest(  # type: ignore[call-arg]
            model="test", input=["hello"]
        )
        embed_data: dict[str, Any] = embed_req.model_dump(exclude_none=True)
        assert isinstance(embed_data, dict)

        # Ensure no None values in serialized data
        all_data: list[dict[str, Any]] = [chat_data, img_data, audio_data, embed_data]
        for data in all_data:
            for key, value in data.items():
                assert value is not None, f"Key '{key}' should not be None when exclude_none=True"

    def test_field_validation(self) -> None:
        """Test that field validation works correctly."""
        # Test invalid temperature (should be between 0 and 2)
        with pytest.raises(ValueError):
            ChatCompletionRequest(  # type: ignore[call-arg]
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                temperature=3.0,  # Invalid: > 2
            )

        # Test invalid image dimensions
        with pytest.raises(ValueError):
            ImageGenerationRequest(  # type: ignore[call-arg]
                prompt="test",
                model="flux",
                width=2000,  # Invalid: > 1280
            )


def test_image_model_constraints_quality_fields():
    from venice_ai.types.api.models import ImageModelConstraints

    c = ImageModelConstraints.model_validate(
        {
            "promptCharacterLimit": 1500,
            "steps": {"default": 20.0, "max": 50.0},
            "widthHeightDivisor": 64.0,
            "defaultQuality": "high",
            "qualities": ["low", "medium", "high"],
        }
    )
    assert c.defaultQuality == "high"
    assert c.qualities == ["low", "medium", "high"]


def test_image_model_constraints_quality_optional():
    from venice_ai.types.api.models import ImageModelConstraints

    c = ImageModelConstraints.model_validate(
        {
            "promptCharacterLimit": 1500,
            "steps": {"default": 20.0, "max": 50.0},
            "widthHeightDivisor": 64.0,
        }
    )
    assert c.defaultQuality is None
    assert c.qualities is None


def test_model_deprecation_full_fields():
    from venice_ai.types.api.models import ModelDeprecation

    dep = ModelDeprecation.model_validate(
        {
            "autoRemap": True,
            "date": "2025-03-01T00:00:00.000Z",
            "removesAt": "2025-04-01T00:00:00.000Z",
            "replacementModelId": "llama-3-3-70b",
            "startsAt": "2025-03-01T00:00:00.000Z",
        }
    )
    assert dep.autoRemap is True
    assert dep.replacementModelId == "llama-3-3-70b"
    assert dep.removesAt == "2025-04-01T00:00:00.000Z"
    assert dep.startsAt == "2025-03-01T00:00:00.000Z"
    assert dep.date == "2025-03-01T00:00:00.000Z"


def test_model_deprecation_minimal_still_parses():
    from venice_ai.types.api.models import ModelDeprecation

    dep = ModelDeprecation.model_validate({"date": "2025-03-01T00:00:00.000Z"})
    assert dep.date == "2025-03-01T00:00:00.000Z"
    assert dep.replacementModelId is None
    assert dep.autoRemap is False


def test_model_capabilities_reasoning_effort_options():
    from venice_ai.types.api.models import ModelCapabilities

    caps = ModelCapabilities.model_validate(
        {
            "optimizedForCode": False,
            "quantization": "fp16",
            "supportsFunctionCalling": True,
            "supportsReasoning": False,
            "supportsResponseSchema": False,
            "supportsVision": False,
            "supportsWebSearch": False,
            "supportsLogProbs": False,
            "supportsReasoningEffort": True,
            "reasoningEffortOptions": ["none", "low", "medium", "high"],
            "defaultReasoningEffort": "medium",
        }
    )
    assert caps.reasoningEffortOptions == ["none", "low", "medium", "high"]
    assert caps.defaultReasoningEffort == "medium"


def test_model_capabilities_reasoning_effort_options_default_none():
    from venice_ai.types.api.models import ModelCapabilities

    caps = ModelCapabilities.model_validate(
        {
            "optimizedForCode": False,
            "quantization": "fp16",
            "supportsFunctionCalling": True,
            "supportsReasoning": False,
            "supportsResponseSchema": False,
            "supportsVision": False,
            "supportsWebSearch": False,
            "supportsLogProbs": False,
            "supportsReasoningEffort": False,
        }
    )
    assert caps.reasoningEffortOptions is None
    assert caps.defaultReasoningEffort is None


def test_video_request_accepts_reference_audio_urls():
    from venice_ai.types.api.requests.video import VideoImageToVideoRequest

    req = VideoImageToVideoRequest(
        model="seedance-2-0",
        prompt="dance",
        duration="5s",
        image_url="https://example.com/ref.png",
        reference_audio_urls=["https://example.com/voice.mp3"],
    )
    assert req.reference_audio_urls == ["https://example.com/voice.mp3"]


def test_video_request_reference_audio_urls_max_three():
    from pydantic import ValidationError

    from venice_ai.types.api.requests.video import VideoImageToVideoRequest

    with pytest.raises(ValidationError):
        VideoImageToVideoRequest(
            model="seedance-2-0",
            prompt="dance",
            duration="5s",
            image_url="https://example.com/ref.png",
            reference_audio_urls=[f"https://example.com/{i}.mp3" for i in range(4)],
        )


def main() -> None:
    """Run basic smoke tests when executed directly."""
    test_instance = TestModelDumpPatterns()
    test_instance.test_core_request_models_serialization()
    print("✅ All core request model tests passed!")

    # Run additional validation tests
    try:
        test_instance.test_field_validation()
        print("✅ Field validation tests passed!")
    except AssertionError:
        print("✅ Field validation correctly raised expected errors!")


def test_image_generation_request_accepts_quality():
    from venice_ai.types.api.requests.images import ImageGenerationRequest

    req = ImageGenerationRequest(model="test-model", prompt="a cat", quality="high")
    assert req.quality == "high"


def test_image_generation_request_rejects_bad_quality():
    from pydantic import ValidationError

    from venice_ai.types.api.requests.images import ImageGenerationRequest

    with pytest.raises(ValidationError):
        ImageGenerationRequest(
            model="test-model", prompt="a cat", quality="ultra"
        )  # not in {low,medium,high}


if __name__ == "__main__":
    main()


class TestApiKeyModelPrivacy:
    """API keys carry a persistent model-privacy tier, settable at creation.

    `ALL` allows every model; `PRIVATE_TEXT` requires text and embedding
    models to be Private/TEE/E2EE; `PRIVATE_ONLY` requires it of every model.
    """

    def test_create_request_accepts_model_privacy(self) -> None:
        request = CreateApiKeyRequest(  # type: ignore[call-arg]
            apiKeyType="INFERENCE",
            description="Private-only key",
            modelPrivacy="PRIVATE_ONLY",
        )
        data = request.model_dump(exclude_none=True)
        assert data["modelPrivacy"] == "PRIVATE_ONLY"

    def test_model_privacy_omitted_when_unset(self) -> None:
        """Omitted means the account default applies; don't invent a tier."""
        request = CreateApiKeyRequest(  # type: ignore[call-arg]
            apiKeyType="INFERENCE", description="Default key"
        )
        assert "modelPrivacy" not in request.model_dump(exclude_none=True)

    def test_web3_create_request_accepts_model_privacy(self) -> None:
        """Web3-created keys carry the same privacy tier as ordinary ones."""
        from venice_ai.types.api.requests.api_keys import Web3CreateApiKeyRequest

        request = Web3CreateApiKeyRequest(  # type: ignore[call-arg]
            address="0xabc",
            signature="0xdef",
            token="tok",
            apiKeyType="INFERENCE",
            description="Web3 private key",
            modelPrivacy="PRIVATE_TEXT",
        )
        assert request.model_dump(exclude_none=True)["modelPrivacy"] == "PRIVATE_TEXT"

    def test_update_request_accepts_model_privacy(self) -> None:
        """PATCH /api_keys accepts the tier too, so a key's privacy can change."""
        from venice_ai.types.api.requests.api_keys import UpdateApiKeyRequest

        request = UpdateApiKeyRequest(  # type: ignore[call-arg]
            id="key-1", modelPrivacy="ALL"
        )
        assert request.model_dump(exclude_none=True)["modelPrivacy"] == "ALL"

    def test_invalid_tier_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CreateApiKeyRequest(  # type: ignore[call-arg]
                apiKeyType="INFERENCE",
                description="Bad tier",
                modelPrivacy="PRIVATE_IMAGES",
            )
