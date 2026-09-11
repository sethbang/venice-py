"""
Venice.ai API Request Models

Complete Pydantic models for all Venice.ai API endpoint requests.
Organized by functionality for better maintainability.
"""

# Import all request models from category modules
from .api_keys import (
    BillingUsageHistoryQueryParams,
    CreateApiKeyRequest,
    DeleteApiKeyQueryParams,
    ModelsQueryParams,
    ModelTraitsQueryParams,
    UpdateApiKeyRequest,
    Web3CreateApiKeyRequest,
)
from .audio import (
    AudioSpeechRequest,
    AudioTranscriptionRequest,
)
from .chat import (
    AssistantMessage,
    # Request model
    ChatCompletionRequest,
    ChatMessageParam,
    DeveloperMessage,
    SystemMessage,
    ToolMessage,
    # Message models
    UserMessage,
)
from .common import (
    ConsumptionLimit,
    DateRangeParams,
    FileContent,
    FileObject,
    ImageContent,
    ImageUrl,
    JSONObjectFormat,
    # Response format components
    JSONSchemaFormat,
    # Utility models
    PaginationParams,
    # Reasoning controls
    ReasoningConfig,
    ReasoningEffortLevel,
    ReasoningSummary,
    SpecificToolChoice,
    # Stream and tool components
    StreamOptions,
    # Content types
    TextContent,
    TextResponseFormat,
    Tool,
    ToolChoiceFunction,
    ToolFunction,
    # Venice-specific components
    VeniceParameters,
)
from .embeddings import (
    EmbeddingsRequest,
)
from .images import (
    ImageBackgroundRemoveRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageMultiEditRequest,
    ImageUpscaleRequest,
    SimpleImageGenerationRequest,
    StyleReference,
)
from .music import (
    MusicCompleteRequest,
    MusicQueueRequest,
    MusicQuoteRequest,
    MusicRetrieveRequest,
)
from .responses import ResponsesRequest
from .video import (
    SeedanceConsents,
    VideoCompleteRequest,
    VideoConsents,
    VideoImageToVideoRequest,
    VideoQueueRequest,
    VideoQuoteRequest,
    VideoRequestBase,
    VideoRetrieveRequest,
    VideoTextToVideoRequest,
    VideoTranscriptionRequest,
)

# ============================================================================
# Export All Models
# ============================================================================

__all__ = [
    # Content types
    "TextContent",
    "ImageUrl",
    "ImageContent",
    "FileObject",
    "FileContent",
    # Stream and tool components
    "StreamOptions",
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    # Response format components
    "JSONSchemaFormat",
    "JSONObjectFormat",
    "TextResponseFormat",
    # Venice-specific components
    "VeniceParameters",
    # Reasoning controls
    "ReasoningEffortLevel",
    "ReasoningSummary",
    "ReasoningConfig",
    # Utility models
    "PaginationParams",
    "DateRangeParams",
    "ConsumptionLimit",
    # Message models
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "SystemMessage",
    "ChatMessageParam",
    "DeveloperMessage",
    # Chat completion
    "ChatCompletionRequest",
    # Responses API (Alpha)
    "ResponsesRequest",
    # Image requests
    "ImageGenerationRequest",
    "SimpleImageGenerationRequest",
    "ImageUpscaleRequest",
    "StyleReference",
    "ImageEditRequest",
    "ImageBackgroundRemoveRequest",
    "ImageMultiEditRequest",
    # Audio requests
    "AudioSpeechRequest",
    "AudioTranscriptionRequest",
    # Embeddings requests
    "EmbeddingsRequest",
    # API key requests
    "CreateApiKeyRequest",
    "UpdateApiKeyRequest",
    "Web3CreateApiKeyRequest",
    # Query parameter models
    "ModelsQueryParams",
    "ModelTraitsQueryParams",
    "BillingUsageHistoryQueryParams",
    "DeleteApiKeyQueryParams",
    # Video requests
    "VideoRequestBase",
    "VideoConsents",
    "SeedanceConsents",
    "VideoTextToVideoRequest",
    "VideoImageToVideoRequest",
    "VideoQueueRequest",
    "VideoQuoteRequest",
    "VideoRetrieveRequest",
    "VideoCompleteRequest",
    "VideoTranscriptionRequest",
    # Music requests
    "MusicQueueRequest",
    "MusicQuoteRequest",
    "MusicRetrieveRequest",
    "MusicCompleteRequest",
]
