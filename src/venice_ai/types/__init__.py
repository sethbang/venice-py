"""Venice AI Types Package.

This package contains comprehensive Pydantic models and TypedDict definitions for all
Venice.ai API interactions. All types provide full validation, type safety, and
auto-completion for Python clients, ensuring robust integration with the Venice AI platform.

**Package Structure:**

This package is organized into two main components:

* **Manual types** (top-level modules): Hand-crafted response DTOs and utility types
* **API types** (``api/`` subdirectory): Request models and base types for API interactions

**Available Type Categories:**

* **Chat completions**: Request/response models, streaming support, tool calls, function calling,
  and Venice-specific features like web search integration and character-based interactions
* **Image generation**: Native Venice AI and OpenAI-compatible image generation, editing,
  upscaling, and style management with comprehensive timing and metadata support
* **Models**: Model specifications, capabilities, constraints, pricing information,
  and compatibility mappings across different AI model types
* **Audio**: Text-to-speech generation, voice management, audio processing, and
  multi-format output support with extensive voice catalog
* **Embeddings**: Text embedding generation, vector processing, and similarity computation
  with support for multiple encoding formats
* **API Keys**: Comprehensive API key management, rate limiting, consumption tracking,
  Web3 authentication, and billing integration
* **Billing**: Detailed usage tracking, consumption limits, billing information,
  pagination support, and multi-currency accounting
* **Characters**: Character-based AI interaction models with personality definitions,
  behavioral parameters, and specialized response styling

**Type Safety and Validation:**

All types are built with Pydantic v2, providing:

* Runtime validation of API requests and responses
* Automatic serialization/deserialization
* IDE auto-completion and static type checking
* Comprehensive error reporting for invalid data
* Seamless integration with Venice AI SDK components

**Usage Pattern:**

Types are typically imported from this package root and used in conjunction with the
corresponding resource classes from ``venice_ai.resources``. The types ensure that
all API interactions are properly validated and documented.
"""

# Import all models from the api directory
# Import specific modules for namespace organization
from . import api, enums, identifiers
from .api import (
    ApiKey,
    ApiKeyDetailsResponse,
    ApiKeysListResponse,
    ApiKeyUsage,
    ApiTier,
    ASRModelPricing,
    # From models module
    AsrModelSpec,
    AssistantMessage,
    AudioModelPricing,
    AudioResponse,
    # From requests module - Audio models
    AudioSpeechRequest,
    AudioTranscriptionRequest,
    AudioTranscriptionResponse,
    Balances,
    BaseListResponse,
    BaseSuccessResponse,
    BillingBalanceResponse,
    BillingUsageEntry,
    BillingUsageHistoryQueryParams,
    BillingUsageHistoryResponse,
    Character,
    CharacterResponse,
    CharactersListResponse,
    # From characters module
    CharacterStats,
    ChatChoice,
    ChatCompletionChoiceLogprobs,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkChoiceDelta,
    ChatCompletionChunkToolCall,
    ChatCompletionChunkToolCallFunction,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTokenLogprob,
    ChatCompletionTopLogprob,
    ChatMessage,
    ChatMessageParam,
    ChatUsage,
    # From streaming module
    ChunkModelFactory,
    # From audio module
    ClonedVoice,
    # From requests module - API key models
    ConsumptionLimit,
    # From api_keys module
    ConsumptionLimits,
    CreateApiKeyRequest,
    CreateApiKeyResponse,
    CreatedApiKey,
    DateRangeParams,
    DeleteApiKeyQueryParams,
    DeleteApiKeyResponse,
    DetailedError,
    DeveloperMessage,
    EmbeddingModelSpec,
    # From embeddings module
    EmbeddingObject,
    # From requests module - Embeddings models
    EmbeddingsRequest,
    EmbeddingsResponse,
    EmbeddingUsage,
    ErrorDetails,
    ImageBackgroundRemoveRequest,
    ImageContent,
    ImageEditRequest,
    # From requests module - Image generation models
    ImageGenerationRequest,
    # From images module
    ImageGenerationResponse,
    ImageModelConstraints,
    ImageModelPricing,
    ImageModelSpec,
    ImageMultiEditRequest,
    ImageStylesResponse,
    ImageUpscaleRequest,
    ImageUrl,
    # From billing module
    InferenceDetails,
    InpaintModelConstraints,
    InpaintModelPricing,
    InpaintModelSpec,
    JSONObjectFormat,
    JSONSchemaFormat,
    LLMModelPricing,
    LogProbToken,
    ModelCapabilities,
    ModelCompatibilityResponse,
    ModelConstraints,
    ModelDeprecation,
    ModelPricing,
    ModelRateLimit,
    ModelResponse,
    ModelsListResponse,
    ModelSpec,
    # From requests module - Query parameter models
    ModelsQueryParams,
    ModelTraitsQueryParams,
    ModelTraitsResponse,
    # From music module - Response models
    MusicCompletedStatus,
    MusicCompleteRequest,
    MusicCompleteResponse,
    MusicFailedStatus,
    MusicModelSpec,
    MusicProcessingStatus,
    MusicQueueRequest,
    MusicQueueResponse,
    MusicQuoteRequest,
    MusicQuoteResponse,
    MusicRetrieveRequest,
    MusicRetrieveResponse,
    # From requests module - Utility models
    PaginationParams,
    PricingTier,
    PromptTokensDetails,
    RateLimit,
    RateLimitLogEntry,
    RateLimitLogsResponse,
    RateLimitsData,
    RateLimitsResponse,
    RateLimitType,
    ReasoningConfig,
    ReasoningEffortLevel,
    ReasoningSummary,
    # From responses module + requests.responses — Responses API (Alpha)
    ResponsesError,
    ResponsesFunctionCallOutput,
    ResponsesMessageOutput,
    ResponsesOutputItem,
    ResponsesOutputText,
    ResponsesReasoningOutput,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesStreamEvent,
    ResponsesUnknownOutput,
    ResponsesUsage,
    ResponsesUsageInputDetails,
    ResponsesUsageOutputDetails,
    ResponsesWebSearchCallOutput,
    # From requests module - Video request models
    SeedanceConsents,
    SimpleImageData,
    SimpleImageGenerationRequest,
    SimpleImageGenerationResponse,
    SpecificToolChoice,
    # From base module
    StandardError,
    StepsConstraint,
    StreamOptions,
    SystemMessage,
    TemperatureConstraint,
    # From requests module - Chat completion models
    TextContent,
    TextModelConstraints,
    TextModelSpec,
    TextResponseFormat,
    TimingInfo,
    Tool,
    ToolCall,
    # From chat module
    ToolCallFunction,
    ToolChoiceFunction,
    ToolFunction,
    ToolMessage,
    TopPConstraint,
    TrailingSevenDaysUsage,
    TranscriptionWord,
    TtsModelSpec,
    UpdateApiKeyRequest,
    UpscaleModelSpec,
    UpscalePricing,
    # From billing module - Usage Analytics (Beta)
    UsageAnalyticsByDate,
    UsageAnalyticsByKey,
    UsageAnalyticsByModel,
    UsageAnalyticsModelBreakdown,
    UsageAnalyticsQueryParams,
    UsageAnalyticsResponse,
    UsageData,
    UserMessage,
    VeniceParameters,
    VideoCompletedStatus,
    VideoCompleteRequest,
    VideoCompleteResponse,
    VideoConsents,
    VideoFailedStatus,
    VideoImageToVideoRequest,
    VideoModelConstraints,
    VideoModelSpec,
    VideoProcessingStatus,
    VideoQueueRequest,
    # From video module - Response models
    VideoQueueResponse,
    VideoQuoteRequest,
    VideoQuoteResponse,
    VideoRequestBase,
    VideoResolutionPricing,
    VideoRetrieveRequest,
    VideoRetrieveResponse,
    VideoTextToVideoRequest,
    VideoTranscriptionRequest,
    VideoTranscriptionResponse,
    VoiceDetail,
    VoiceList,
    Web3ApiKeyResponse,
    Web3CreateApiKeyRequest,
    # From web3 module
    Web3TokenData,
    Web3TokenResponse,
    WebSearchCitation,
    # From x402 module - Wallet billing
    X402BalanceData,
    X402BalanceResponse,
    X402TopUpData,
    X402TopUpResponse,
    X402Transaction,
    X402TransactionsData,
    X402TransactionsPagination,
    X402TransactionsResponse,
)

# Import enums from consolidated enums module
from .enums import (
    BillingFormatEnum,
    ModelType,
    ResponseFormat,
    VideoAspectRatio,
    VideoDuration,
    # Video enums
    VideoModelType,
    VideoPrivacy,
    VideoResolution,
    VideoStatus,
    Voice,
)

# Import identifiers
from .identifiers import (
    ModelId,
    QueueId,
    normalize_model_id,
    normalize_queue_id,
)

# Export all imported types
__all__ = [
    # Enums from enums module
    "Voice",
    "ResponseFormat",
    "BillingFormatEnum",
    "ModelType",
    # Video enums
    "VideoModelType",
    "VideoStatus",
    "VideoDuration",
    "VideoResolution",
    "VideoPrivacy",
    "VideoAspectRatio",
    # Identifiers
    "ModelId",
    "QueueId",
    "normalize_model_id",
    "normalize_queue_id",
    # From base module
    "StandardError",
    "DetailedError",
    "WebSearchCitation",
    "ErrorDetails",
    "PromptTokensDetails",
    "TimingInfo",
    "UsageData",
    "BaseListResponse",
    "BaseSuccessResponse",
    # From chat module
    "ToolCallFunction",
    "ToolCall",
    "ChatMessage",
    "LogProbToken",
    "ChatChoice",
    "ChatUsage",
    "ChatCompletionResponse",
    # From images module
    "ImageGenerationResponse",
    "SimpleImageData",
    "SimpleImageGenerationResponse",
    "ImageStylesResponse",
    # From models module
    "ModelCapabilities",
    "TemperatureConstraint",
    "TopPConstraint",
    "StepsConstraint",
    "TextModelConstraints",
    "ImageModelConstraints",
    "InpaintModelConstraints",
    "VideoModelConstraints",
    "PricingTier",
    "LLMModelPricing",
    "UpscalePricing",
    "ImageModelPricing",
    "AudioModelPricing",
    "VideoResolutionPricing",
    "InpaintModelPricing",
    "ASRModelPricing",
    "ModelConstraints",
    "ModelPricing",
    "ModelDeprecation",
    "ModelSpec",
    "TextModelSpec",
    "ImageModelSpec",
    "VideoModelSpec",
    "InpaintModelSpec",
    "MusicModelSpec",
    "TtsModelSpec",
    "AsrModelSpec",
    "EmbeddingModelSpec",
    "UpscaleModelSpec",
    "ModelResponse",
    "ModelsListResponse",
    "ModelTraitsResponse",
    "ModelCompatibilityResponse",
    # From api_keys module
    "ConsumptionLimits",
    "TrailingSevenDaysUsage",
    "ApiKeyUsage",
    "ApiKey",
    "ApiKeysListResponse",
    "CreatedApiKey",
    "CreateApiKeyResponse",
    "DeleteApiKeyResponse",
    "ApiKeyDetailsResponse",
    "RateLimit",
    "ModelRateLimit",
    "ApiTier",
    "Balances",
    "RateLimitsData",
    "RateLimitsResponse",
    "RateLimitLogEntry",
    "RateLimitType",
    "RateLimitLogsResponse",
    # From billing module
    "InferenceDetails",
    "BillingUsageEntry",
    "BillingUsageHistoryResponse",
    "BillingBalanceResponse",
    # From billing module - Usage Analytics (Beta)
    "UsageAnalyticsByDate",
    "UsageAnalyticsModelBreakdown",
    "UsageAnalyticsByModel",
    "UsageAnalyticsByKey",
    "UsageAnalyticsResponse",
    "UsageAnalyticsQueryParams",
    # From audio module
    "ClonedVoice",
    "VoiceDetail",
    "VoiceList",
    "AudioResponse",
    "TranscriptionWord",
    "AudioTranscriptionResponse",
    # From embeddings module
    "EmbeddingObject",
    "EmbeddingUsage",
    "EmbeddingsResponse",
    # From characters module
    "CharacterStats",
    "Character",
    "CharactersListResponse",
    "CharacterResponse",
    # From web3 module
    "Web3TokenData",
    "Web3TokenResponse",
    "Web3ApiKeyResponse",
    # From streaming module
    "ChunkModelFactory",
    "ChatCompletionTopLogprob",
    "ChatCompletionTokenLogprob",
    "ChatCompletionChoiceLogprobs",
    "ChatCompletionChunkToolCallFunction",
    "ChatCompletionChunkToolCall",
    "ChatCompletionChunkChoiceDelta",
    "ChatCompletionChunkChoice",
    "ChatCompletionChunk",
    # From video module - Response models
    "VideoQueueResponse",
    "VideoQuoteResponse",
    "VideoProcessingStatus",
    "VideoFailedStatus",
    "VideoCompletedStatus",
    "VideoCompleteResponse",
    "VideoRetrieveResponse",
    "VideoTranscriptionResponse",
    # From music module - Response models
    "MusicQueueResponse",
    "MusicQuoteResponse",
    "MusicProcessingStatus",
    "MusicFailedStatus",
    "MusicCompletedStatus",
    "MusicCompleteResponse",
    "MusicRetrieveResponse",
    # From requests module - Music request models
    "MusicQueueRequest",
    "MusicQuoteRequest",
    "MusicRetrieveRequest",
    "MusicCompleteRequest",
    # From requests module - Chat completion models
    "TextContent",
    "ImageUrl",
    "ImageContent",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "SystemMessage",
    "ChatMessageParam",
    "DeveloperMessage",
    "StreamOptions",
    "ToolFunction",
    "Tool",
    "ToolChoiceFunction",
    "SpecificToolChoice",
    "JSONSchemaFormat",
    "JSONObjectFormat",
    "TextResponseFormat",
    "VeniceParameters",
    # Reasoning controls
    "ReasoningEffortLevel",
    "ReasoningSummary",
    "ReasoningConfig",
    "ChatCompletionRequest",
    # From responses module + requests.responses - Responses API (Alpha)
    "ResponsesRequest",
    "ResponsesError",
    "ResponsesFunctionCallOutput",
    "ResponsesMessageOutput",
    "ResponsesOutputItem",
    "ResponsesOutputText",
    "ResponsesReasoningOutput",
    "ResponsesResponse",
    "ResponsesStreamEvent",
    "ResponsesUnknownOutput",
    "ResponsesUsage",
    "ResponsesUsageInputDetails",
    "ResponsesUsageOutputDetails",
    "ResponsesWebSearchCallOutput",
    # From requests module - Image generation models
    "ImageGenerationRequest",
    "SimpleImageGenerationRequest",
    "ImageUpscaleRequest",
    "ImageEditRequest",
    "ImageBackgroundRemoveRequest",
    "ImageMultiEditRequest",
    # From requests module - Audio models
    "AudioSpeechRequest",
    "AudioTranscriptionRequest",
    # From requests module - Embeddings models
    "EmbeddingsRequest",
    # From requests module - API key models
    "ConsumptionLimit",
    "CreateApiKeyRequest",
    "UpdateApiKeyRequest",
    "Web3CreateApiKeyRequest",
    # From requests module - Query parameter models
    "ModelsQueryParams",
    "ModelTraitsQueryParams",
    "BillingUsageHistoryQueryParams",
    "DeleteApiKeyQueryParams",
    # From requests module - Utility models
    "PaginationParams",
    "DateRangeParams",
    # From requests module - Video request models
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
    # From x402 module - Wallet billing
    "X402BalanceData",
    "X402BalanceResponse",
    "X402TopUpData",
    "X402TopUpResponse",
    "X402Transaction",
    "X402TransactionsData",
    "X402TransactionsPagination",
    "X402TransactionsResponse",
    # Module exports
    "api",
    "enums",
    "identifiers",
]
