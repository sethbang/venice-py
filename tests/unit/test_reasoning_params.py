"""Unit tests for the March 2026 reasoning-parameter widening.

Covers:
- `ChatCompletionRequest` accepts the full seven-value ``reasoning_effort`` enum.
- Chat request accepts a nested ``reasoning=ReasoningConfig(...)`` object.
- The `/image/background-remove` endpoint now routes to `ResourceType.IMAGE`.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from venice_ai import ReasoningConfig, ReasoningEffortLevel
from venice_ai._queue_types import ResourceType
from venice_ai._request_classifier import RequestClassifier
from venice_ai.types.api import (
    ChatCompletionRequest,
    UserMessage,
)

ALL_EFFORT_LEVELS: tuple[ReasoningEffortLevel, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


class TestReasoningEffortEnum:
    @pytest.mark.parametrize("effort", ALL_EFFORT_LEVELS)
    def test_chat_request_accepts_all_tiers(self, effort: ReasoningEffortLevel) -> None:
        req = ChatCompletionRequest(  # type: ignore[call-arg]
            model="llama-3.3-70b",
            messages=[UserMessage(content="hi")],
            reasoning_effort=effort,
        )
        assert req.reasoning_effort == effort

    def test_chat_request_rejects_invalid_effort(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(  # type: ignore[call-arg]
                model="llama-3.3-70b",
                messages=[UserMessage(content="hi")],
                reasoning_effort="extreme",  # type: ignore[arg-type]
            )


class TestReasoningConfigObject:
    def test_chat_request_accepts_nested_reasoning(self) -> None:
        cfg = ReasoningConfig(effort="max", summary="detailed")
        req = ChatCompletionRequest(  # type: ignore[call-arg]
            model="llama-3.3-70b",
            messages=[UserMessage(content="hi")],
            reasoning=cfg,
        )
        assert req.reasoning is not None
        assert req.reasoning.effort == "max"
        assert req.reasoning.summary == "detailed"

    def test_reasoning_config_rejects_invalid_summary(self) -> None:
        with pytest.raises(ValidationError):
            ReasoningConfig(effort="high", summary="verbose")  # type: ignore[arg-type]

    def test_chat_serialized_payload_includes_reasoning(self) -> None:
        req = ChatCompletionRequest(  # type: ignore[call-arg]
            model="llama-3.3-70b",
            messages=[UserMessage(content="hi")],
            reasoning_effort="max",
            reasoning=ReasoningConfig(effort="medium", summary="auto"),
        )
        payload = req.model_dump(exclude_none=True)
        assert payload["reasoning_effort"] == "max"
        assert payload["reasoning"] == {"effort": "medium", "summary": "auto"}


class TestBackgroundRemoveClassification:
    @pytest.fixture
    def classifier(self) -> RequestClassifier:
        return RequestClassifier(MagicMock())

    @pytest.mark.asyncio
    async def test_background_remove_routes_to_image(self, classifier: RequestClassifier) -> None:
        request = {"endpoint": "image/background-remove", "model": "rembg-bria-2.0"}
        metadata = await classifier.classify(request)
        assert metadata.resource_type == ResourceType.IMAGE


class TestReasoningEnabled:
    def test_enabled_false_is_serialized(self) -> None:
        req = ChatCompletionRequest(  # type: ignore[call-arg]
            model="m",
            messages=[UserMessage(content="hi")],
            reasoning=ReasoningConfig(enabled=False),
        )
        assert req.model_dump(exclude_none=True)["reasoning"] == {"enabled": False}

    def test_enabled_defaults_to_none(self) -> None:
        assert ReasoningConfig().enabled is None
        assert ReasoningConfig(effort="low").model_dump(exclude_none=True) == {"effort": "low"}
