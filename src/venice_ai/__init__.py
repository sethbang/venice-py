"""
venice-py - Venice AI Python SDK
================================

An unofficial, community-maintained Python SDK for the Venice AI API, with
intelligent rate limiting, circuit breakers, request queuing, and comprehensive
error handling.

Key Features:
    * **VeniceClient**: Main client for API interactions
    * **Intelligent Rate Limiting**: Automatic rate limit management with queuing
    * **Circuit Breaker Pattern**: Failure detection and recovery
    * **State Management**: Distributed state tracking with Redis backend
    * **Cost Calculation**: Built-in usage cost estimation
    * **Comprehensive Exception Handling**: Detailed error types for all scenarios
    * **Factory Pattern**: Dependency injection for testing and configuration
    * **Streaming Support**: Real-time response streaming

Quick Start:
    >>> import asyncio
    >>> from venice_ai import VeniceClient, UserMessage
    >>>
    >>> async def main():
    ...     async with VeniceClient() as client:  # reads VENICE_API_KEY
    ...         model = await client.models.resolve_chat()
    ...         response = await client.chat.completions.create(
    ...             model=model,
    ...             messages=[UserMessage(content="Hello!")],
    ...         )
    ...         print(response.choices[0].message.content)
    >>>
    >>> asyncio.run(main())

Architecture:
    The SDK is built around several core components:

    * **Client Layer**: VeniceClient orchestrates all API interactions
    * **Account Management**: Multi-account support with failure tracking
    * **Rate Limiting**: Intelligent queuing and scheduling system
    * **State Management**: Distributed state with Redis backend
    * **Recovery Patterns**: Circuit breakers and error recovery strategies
    * **Cost Tracking**: Real-time usage cost calculation

For advanced usage and configuration options, see the factory module and
configuration classes.
"""

from __future__ import annotations

# Imported at module scope, not inside the try below, so the except clause can
# name it. importlib.metadata is stdlib on every supported Python, so the import
# itself cannot be what fails.
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

# ── Static type-checking imports (not executed at runtime) ───────────────────
# These mirror the lazy _LAZY_IMPORTS below so Pylance can resolve the names
# in __all__ without actually importing the optional-dependency modules.
if TYPE_CHECKING:
    from venice_ai.core.config import VeniceAIConfig  # noqa: F401
    from venice_ai.core.config.backends import BackendConfig  # noqa: F401
    from venice_ai.core.config.enterprise import SchedulerConfig  # noqa: F401
    from venice_ai.core.config.enums import BackendType, SchedulerMode  # noqa: F401
    from venice_ai.core.config.http import HttpClientConfig  # noqa: F401
    from venice_ai.factory import (
        VeniceClientFactory,
        create_developer_client,
        create_test_venice_client,
        create_venice_client,
    )  # noqa: F401
    from venice_ai.models.selection import (
        get_chat_model,
        get_cheapest_video_model,
        get_embedding_model,
        get_video_model,
    )  # noqa: F401
    from venice_ai.presets import (
        create_development_config,
        create_production_config,
        create_testing_config,
    )  # noqa: F401

# ── Core (always-needed) eager imports ──────────────────────────────────────

from ._client import VeniceClient
from ._sync_client import SyncVeniceClient
from .core import (
    RateLimitBucket,
    RateLimitDiscovery,
    RedisBackendConfig,
)
from .core.models.common import Tool, ToolChoice, ToolFunction
from .core.models.headers import BalanceInfo, DeprecationInfo, RateLimitInfo
from .costs import (
    BudgetManager,
    BudgetRemaining,
    ChatCostEstimate,
    CostRecord,
    CostSummary,
    CostTracker,
    calculate_completion_cost,
    calculate_embedding_cost,
    estimate_completion_cost,
)
from .exceptions import (
    APIConnectionError,
    APIError,
    APIResponseProcessingError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BillingTimeoutError,
    ConflictError,
    InternalServerError,
    InvalidRequestError,
    MaxIterationsExceededError,
    ModelGoneError,
    MusicGenerationError,
    NotFoundError,
    PaymentRequiredError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    StreamClosedError,
    StreamConsumedError,
    UnprocessableEntityError,
    VeniceAPIErrorCode,
    VeniceError,
    VideoGenerationError,
)
from .helpers import (
    Conversation,
    cosine_similarity,
    detect_image_format,
    extract_thinking_blocks,
    fit_image_bytes,
    tool_from_function,
    tool_from_model,
)
from .middleware.retry import RetryOptions
from .models.selection import (
    CheapestVideoResult,
    DynamicModelSelector,
    create_model_selector,
)
from .rate_limiting import (
    RateLimiterConfig,
    RateLimiterMode,
    SimpleRateLimiter,
)
from .resources.decisions import Decisions
from .resources.image import ImageJob
from .resources.music import Music, MusicJob
from .resources.video import VideoJob
from .resources.voice_changer import VoiceChanger, VoiceChangerJob
from .streaming import BytesResponse, ChatStream, Stream
from .types.api.audio import AudioResponse
from .types.api.capabilities import (
    Capabilities,
    ChatCapabilities,
    GenericCapabilities,
    ImageCapabilities,
    InpaintCapabilities,
    VideoCapabilities,
)
from .types.api.chat import (
    ChatCompletionResponse,
    ChatUsage,
    ParsedChatCompletion,
    ToolCall,
    ToolCallFunction,
    ToolLoopResult,
)
from .types.api.decisions import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionResponse,
    DecisionUsage,
    NoulAnswer,
    ScoreAnswer,
    UnknownAnswer,
)
from .types.api.images import ImageGenerationResponse
from .types.api.requests.api_keys import CreateApiKeyRequest
from .types.api.requests.chat import (
    AssistantMessage,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
    UserMessageBuilder,
)
from .types.api.requests.common import (
    FileContent,
    FileObject,
    ImageContent,
    ImageUrl,
    JSONSchemaFormat,
    ReasoningConfig,
    ReasoningEffortLevel,
    ReasoningSummary,
    StreamOptions,
    TextContent,
    VeniceParameters,
)
from .types.api.requests.decisions import (
    ChoiceQuestion,
    CreateDecisionRequest,
    NoulCriteria,
    NoulQuestion,
    ScoreQuestion,
)
from .types.api.requests.responses import ResponsesRequest
from .types.api.responses import (
    ResponsesResponse,
)
from .types.api.streaming import ChatCompletionChunk
from .utils import build_model_id, get_filtered_models

# ── Lazy-loaded enterprise / optional imports ────────────────────────────────
# These modules import redis, pydantic-settings, etc.  We defer loading them
# until first access so that a bare ``import venice_ai`` never forces those
# optional dependencies to be present.

_LAZY_IMPORTS: dict[str, str] = {
    # Factory helpers
    "VeniceClientFactory": ".factory",
    "create_venice_client": ".factory",
    "create_test_venice_client": ".factory",
    "create_developer_client": ".factory",
    # Enterprise config (requires pydantic-settings)
    "VeniceAIConfig": ".core.config",
    # Configuration presets (depend on VeniceAIConfig / redis)
    "create_production_config": ".presets",
    "create_development_config": ".presets",
    "create_testing_config": ".presets",
    # Model selection convenience helpers (async, require client)
    "get_chat_model": ".models.selection",
    "get_embedding_model": ".models.selection",
    "get_video_model": ".models.selection",
    "get_cheapest_video_model": ".models.selection",
    # Enterprise config types (depend on pydantic, deferred for lighter import)
    "BackendConfig": ".core.config.backends",
    "BackendType": ".core.config.enums",
    "HttpClientConfig": ".core.config.http",
    "SchedulerConfig": ".core.config.enterprise",
    "SchedulerMode": ".core.config.enums",
}


def __getattr__(name: str):  # noqa: ANN001, ANN201
    if name in _LAZY_IMPORTS:
        import importlib

        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, name)
        # Cache in module globals so subsequent accesses skip __getattr__
        globals()[name] = value
        return value
    raise AttributeError(f"module 'venice_ai' has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)


__all__ = [
    # Metadata
    "__version__",
    # Core client
    "VeniceClient",
    "SyncVeniceClient",
    # Streaming & response wrappers
    "Stream",
    "ChatStream",
    "BytesResponse",
    # Common request types
    "TextContent",
    "ImageContent",
    "ImageUrl",
    "FileObject",
    "FileContent",
    "StreamOptions",
    "VeniceParameters",
    "JSONSchemaFormat",
    # Reasoning controls
    "ReasoningEffortLevel",
    "ReasoningSummary",
    "ReasoningConfig",
    # Message types
    "UserMessage",
    "UserMessageBuilder",
    "SystemMessage",
    "AssistantMessage",
    "ToolMessage",
    "DeveloperMessage",
    # Response types
    "ChatCompletionResponse",
    "ChatCompletionChunk",
    "ChatUsage",
    "ParsedChatCompletion",
    "ToolCall",
    "ToolCallFunction",
    "ToolLoopResult",
    # Model capability discovery
    "Capabilities",
    "ChoiceAnswer",
    "DecisionAnswer",
    "DecisionResponse",
    "DecisionUsage",
    "NoulAnswer",
    "ScoreAnswer",
    "UnknownAnswer",
    "ChoiceQuestion",
    "CreateDecisionRequest",
    "NoulCriteria",
    "NoulQuestion",
    "ScoreQuestion",
    "Decisions",
    "ChatCapabilities",
    "ImageCapabilities",
    "VideoCapabilities",
    "InpaintCapabilities",
    "GenericCapabilities",
    # Responses API (Alpha)
    "ResponsesRequest",
    "ResponsesResponse",
    "ImageGenerationResponse",
    "AudioResponse",
    "CreateApiKeyRequest",
    # Tool building
    "Tool",
    "ToolFunction",
    "ToolChoice",
    "tool_from_model",
    "tool_from_function",
    # Conversation helper
    "Conversation",
    # Vector similarity
    "cosine_similarity",
    # Image utilities
    "detect_image_format",
    "fit_image_bytes",
    # Reasoning helpers
    "extract_thinking_blocks",
    # Response metadata
    "RateLimitInfo",
    "DeprecationInfo",
    "BalanceInfo",
    # Retry options
    "RetryOptions",
    # Rate limiting (core)
    "RateLimitDiscovery",
    "RateLimitBucket",
    "RedisBackendConfig",
    "SimpleRateLimiter",
    "RateLimiterConfig",
    "RateLimiterMode",
    # Factory (lazy)
    "VeniceClientFactory",
    "create_venice_client",
    "create_test_venice_client",
    "create_developer_client",
    # Enterprise config (lazy)
    "VeniceAIConfig",
    "BackendConfig",
    "BackendType",
    "HttpClientConfig",
    "SchedulerConfig",
    "SchedulerMode",
    # Model selection
    "CheapestVideoResult",
    "DynamicModelSelector",
    "create_model_selector",
    # Model selection helpers (lazy)
    "get_chat_model",
    "get_embedding_model",
    "get_video_model",
    "get_cheapest_video_model",
    # Config presets (lazy)
    "create_production_config",
    "create_development_config",
    "create_testing_config",
    # Image job abstraction
    "ImageJob",
    # Video job abstraction
    "VideoJob",
    "VoiceChanger",
    "VoiceChangerJob",
    # Music resource + job abstraction
    "Music",
    "MusicJob",
    # Exceptions
    "VideoGenerationError",
    "MusicGenerationError",
    "VeniceAPIErrorCode",
    "VeniceError",
    "APIError",
    "APIStatusError",
    "AuthenticationError",
    "PermissionDeniedError",
    "InvalidRequestError",
    "ModelGoneError",
    "NotFoundError",
    "ConflictError",
    "UnprocessableEntityError",
    "RateLimitError",
    "PaymentRequiredError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "APITimeoutError",
    "BillingTimeoutError",
    "APIResponseProcessingError",
    "APIResponseValidationError",
    "StreamConsumedError",
    "StreamClosedError",
    "MaxIterationsExceededError",
    # Utility helpers
    "get_filtered_models",
    "build_model_id",
    "calculate_completion_cost",
    "calculate_embedding_cost",
    "estimate_completion_cost",
    "ChatCostEstimate",
    # Stateful cost tracking
    "CostTracker",
    "BudgetManager",
    "CostRecord",
    "CostSummary",
    "BudgetRemaining",
]

try:
    __version__ = _pkg_version("venice-py")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    # Deliberately not a plausible release version. A real-looking literal here
    # makes a broken metadata lookup indistinguishable from a working one, so
    # the version test silently stops testing anything the moment the literal
    # catches up to the shipped version.
    __version__ = "0.0.0+unknown"
