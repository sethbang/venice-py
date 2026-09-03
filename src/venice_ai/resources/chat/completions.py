"""
Venice AI Chat Completions API Resources.

This module provides asynchronous client interfaces for Venice AI's Chat Completions API,
enabling sophisticated conversational AI interactions with advanced language models.
The Chat Completions API supports both streaming and non-streaming responses, tool calling,
structured output generation, and fine-grained control over model behavior.

Key Features:
    - **Conversational AI**: Multi-turn chat conversations with context preservation
    - **Streaming Responses**: Real-time token-by-token response generation
    - **Tool Integration**: Function calling and external tool integration capabilities
    - **Structured Output**: JSON schema-guided response formatting
    - **Model Variety**: Access to multiple state-of-the-art language models
    - **Advanced Controls**: Temperature, top-p, frequency penalties, and more
    - **Asynchronous Operations**: Full async/await support for scalable applications

Supported Capabilities:
    - **Multi-turn Conversations**: Maintain context across multiple exchanges
    - **System Instructions**: Define AI behavior and personality through system messages
    - **Function Calling**: Enable AI to call external functions and APIs
    - **Response Formatting**: Control output structure with JSON schemas
    - **Content Filtering**: Optional safety and content moderation features
    - **Reproducible Generation**: Seed-based deterministic outputs
    - **Token Management**: Precise control over response length and token usage

The chat completions system enables sophisticated AI applications including:
    - **Virtual Assistants**: Intelligent chatbots and conversational interfaces
    - **Content Generation**: Creative writing, documentation, and content creation
    - **Code Assistance**: Programming help, code review, and technical guidance
    - **Data Analysis**: Structured data processing and analysis workflows
    - **Decision Support**: AI-powered recommendations and decision assistance
    - **Educational Tools**: Tutoring systems and interactive learning platforms

Example:
    .. code-block:: python

        import asyncio
        from venice_ai import VeniceClient

        async def chat_conversation():
            async with VeniceClient() as client:
                # Single-turn conversation
                response = await client.chat.completions.create(
                    model="llama-3.3-70b",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Explain quantum computing"}
                    ],
                    temperature=0.7,
                    max_completion_tokens=500
                )

                print(response["choices"][0]["message"]["content"])

        asyncio.run(chat_conversation())

Performance Considerations:
    - Streaming reduces latency for long responses
    - Batch conversations are more efficient than individual requests
    - Model selection affects both quality and response speed
    - Token limits impact both cost and conversation length

Note:
    All operations in this module are asynchronous and require proper async/await
    handling. The ChatCompletions class is accessed through the
    :attr:`VeniceClient.chat.completions` property for proper authentication and configuration.
"""

import asyncio
import inspect
import logging
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping, Sequence
from decimal import Decimal
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    cast,
    overload,
)

from pydantic import BaseModel, TypeAdapter

from ..._resource import APIResource
from ...costs import ChatCostEstimate
from ...exceptions import InvalidRequestError, MaxIterationsExceededError
from ...helpers import tool_from_function
from ...streaming import ChatStream, Stream
from ...tee._crypto import looks_encrypted
from ...tee.types import TeeOptions
from ...types.api import (
    AssistantMessage,
    # Request models
    ChatCompletionRequest,
    # Response models
    ChatCompletionResponse,
    ChatMessageParam,
    DeveloperMessage,
    JSONObjectFormat,
    JSONSchemaFormat,
    ReasoningConfig,
    ReasoningEffortLevel,
    SpecificToolChoice,
    StreamOptions,
    SystemMessage,
    TextResponseFormat,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    VeniceParameters,
)
from ...types.api.chat import ParsedChatCompletion, ToolLoopResult
from ...types.api.models import LLMModelPricing

# Import streaming models from generated.streaming module
from ...types.api.streaming import ChatCompletionChunk, ChunkModelFactory
from ...validation.validators import validate_model_id

if TYPE_CHECKING:
    from ..._client import VeniceClient  # noqa: F401
    from ...tee._session import TeeSession

logger = logging.getLogger(__name__)
_tools_logger = logging.getLogger("venice_ai.tools")

__all__ = ["ChatCompletions"]

#: Prefix that marks a Venice confidential-compute (TEE) chat model.
_E2EE_MODEL_PREFIX = "e2ee-"

#: Roles whose message content is encrypted to the model under E2EE. Assistant /
#: tool / developer content stays plaintext (the model never re-reads its own
#: prior turns as ciphertext, and tool output is not a user secret).
_E2EE_ENCRYPT_ROLES = frozenset({"user", "system"})

#: Emitted once per E2EE-engaged ``create`` call. The wire path is real, but the
#: baseline attestation verifier trusts Venice's server-side ``verified`` claim;
#: full client-side TDX / NVIDIA quote verification is not performed.
_E2EE_TRUST_WARNING = (
    "Venice E2EE engaged: messages are encrypted client-side to the attested "
    "model key and responses are decrypted locally. SECURITY LIMITATION: the "
    "baseline attestation verifier TRUSTS Venice's server-side 'verified' claim "
    "and does NOT perform full client-side Intel TDX / NVIDIA quote "
    "verification. A malicious Venice operator forging a self-consistent "
    "attestation would not be detected. Supply a FullQuoteVerifier "
    "(e2ee=TeeOptions(verifier=...)) if your threat model requires it."
)


# ---------------------------------------------------------------------------
# Tool-loop orchestration internals (used by ChatCompletions.run_with_tools)
# ---------------------------------------------------------------------------


class _ToolEntry(NamedTuple):
    """One registered tool: its definition paired with its Python dispatch handler."""

    tool: Tool
    handler: Callable[..., Any]


def _normalize_tool_registry(
    tools: Sequence[Callable[..., Any] | Tool],
) -> dict[str, _ToolEntry]:
    """Index tools by function name for run_with_tools dispatch.

    Bare callables are converted to ``Tool`` definitions via
    :func:`tool_from_function` and registered as their own dispatch handler.
    Pre-built ``Tool`` objects are rejected at registry build because
    ``run_with_tools`` cannot dispatch them — pass the underlying callable
    instead. (For the low-level ``create(tools=[...])`` path where the caller
    dispatches, ``Tool`` objects are still appropriate.)
    """
    registry: dict[str, _ToolEntry] = {}
    for item in tools:
        if isinstance(item, Tool):
            if item.function is None:
                raise ValueError(f"Tool with no function: {item!r}")
            raise ValueError(
                f"Tool {item.function.name!r} was passed to run_with_tools as a "
                f"Tool definition with no Python handler; run_with_tools cannot "
                f"dispatch it. Pass the underlying function in `tools` instead, "
                f"or use the low-level chat.completions.create(tools=[...]) path "
                f"if you want to dispatch tool calls yourself."
            )
        elif callable(item):
            tool = tool_from_function(item)
            assert tool.function is not None  # tool_from_function always sets function
            registry[tool.function.name] = _ToolEntry(tool, item)
        else:
            raise TypeError(f"tools entries must be Callable or Tool, got {type(item).__name__}")
    return registry


def _default_on_tool_error(call: ToolCall, exc: Exception) -> str:
    """Default tool-error handler: log to stderr AND format for the model.

    Pairs resilience (model can self-correct on bad inputs) with visibility
    (real bugs surface in the ``venice_ai.tools`` logger). Pass
    ``on_tool_error=raise_it`` to ``run_with_tools`` for strict propagation.
    """
    _tools_logger.error(
        "Tool %s raised %s: %s",
        call.function.name,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    return f"Error calling {call.function.name}: {type(exc).__name__}: {exc}"


async def _execute_tool_call(
    call: ToolCall,
    registry: dict[str, _ToolEntry],
    on_tool_call: Callable[[ToolCall, Any], None] | None,
    on_tool_error: Callable[[ToolCall, Exception], str],
) -> str:
    """Dispatch a single tool call and return a string for the ``ToolMessage``."""
    name = call.function.name
    entry = registry.get(name)
    if entry is None:
        raise ValueError(
            f"Model called tool {name!r} but no matching tool was registered. "
            f"Registered: {sorted(registry)}"
        )

    args = call.function.arguments_dict
    try:
        if inspect.iscoroutinefunction(entry.handler):
            result = await entry.handler(**args)
        else:
            result = entry.handler(**args)
    except Exception as exc:
        return on_tool_error(call, exc)

    if on_tool_call is not None:
        on_tool_call(call, result)
    return result if isinstance(result, str) else str(result)


_ChatMessageModel = UserMessage | AssistantMessage | SystemMessage | ToolMessage | DeveloperMessage

_MESSAGE_LIST_ADAPTER: TypeAdapter[list[_ChatMessageModel]] = TypeAdapter(list[_ChatMessageModel])


def _coerce_messages(messages: Sequence[ChatMessageParam]) -> list[_ChatMessageModel]:
    """Validate a caller's messages into the concrete message models.

    :meth:`ChatCompletions.create` gets this for free, since
    ``ChatCompletionRequest`` coerces mappings while building the request
    body. Methods that read or accumulate messages *before* that point need
    it explicitly, so that a caller passing plain dicts doesn't hit an
    ``AttributeError`` on ``.content``.
    """
    return _MESSAGE_LIST_ADAPTER.validate_python(list(messages))


def _concat_message_text(
    messages: Sequence[_ChatMessageModel],
) -> str:
    """Concatenate plain-text content across a message list.

    Used by :meth:`ChatCompletions.estimate_cost` to feed the word-count
    token heuristic. Multimodal parts (images, audio, video) are skipped —
    they are not text-token priced — and ``None`` content is treated as
    empty.
    """
    parts: list[str] = []
    for msg in messages:
        content = msg.content
        if content is None:
            continue
        if isinstance(content, str):
            parts.append(content)
            continue
        # Multimodal list — pull out only the text-shaped entries.
        for chunk in content:
            text = getattr(chunk, "text", None)
            if isinstance(text, str):
                parts.append(text)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# TEE / E2EE integration helpers
# ---------------------------------------------------------------------------


def _venice_params_as_dict(venice_parameters: Any) -> dict[str, Any]:
    """Normalize the ``venice_parameters`` argument to a plain dict (read-only).

    Accepts a :class:`VeniceParameters`, a mapping, or ``None``; never mutates the
    caller's object. Used for E2EE engagement detection and guard checks.
    """
    if venice_parameters is None:
        return {}
    if isinstance(venice_parameters, dict):
        return dict(venice_parameters)
    if hasattr(venice_parameters, "model_dump"):
        return cast(dict[str, Any], venice_parameters.model_dump(exclude_none=True))
    return {}


def _e2ee_engaged(e2ee: bool | TeeOptions, venice_parameters: Any) -> bool:
    """Whether the E2EE flow should run for this call.

    Engaged when the explicit ``e2ee`` argument is truthy OR when the caller set
    ``venice_parameters.enable_e2ee`` to ``True``.
    """
    if e2ee:
        return True
    return _venice_params_as_dict(venice_parameters).get("enable_e2ee") is True


def _validate_e2ee_request(
    *,
    model: str,
    messages: Sequence[Any],
    tools: Any,
    venice_parameters: Any,
) -> None:
    """Fail loud, before any network call, on E2EE-incompatible requests.

    Raises:
        InvalidRequestError: E2EE on a non-``e2ee-`` model; tools / web search /
            web scraping requested; or any user/system message carrying non-text
            (multimodal: image / file) content.
    """
    if not model.startswith(_E2EE_MODEL_PREFIX):
        raise InvalidRequestError(
            f"E2EE was requested but model {model!r} is not a Venice "
            f"confidential-compute model. E2EE chat requires an "
            f"'{_E2EE_MODEL_PREFIX}*' model (see client.models.list()).",
            request=None,
            response=None,
            body=None,
        )

    if tools:
        raise InvalidRequestError(
            "Tool calling is not supported under Venice E2EE: tool definitions "
            "and tool results would have to leave the encrypted channel. Drop "
            "`tools` to use E2EE.",
            request=None,
            response=None,
            body=None,
        )

    vp = _venice_params_as_dict(venice_parameters)
    if vp.get("enable_web_search", "off") != "off":
        raise InvalidRequestError(
            "Web search is not supported under Venice E2EE (it would require the "
            "server to read the encrypted prompt). Set "
            "venice_parameters.enable_web_search='off' to use E2EE.",
            request=None,
            response=None,
            body=None,
        )
    if vp.get("enable_web_scraping"):
        raise InvalidRequestError(
            "Web scraping is not supported under Venice E2EE (it would require "
            "the server to read URLs in the encrypted prompt). Disable "
            "venice_parameters.enable_web_scraping to use E2EE.",
            request=None,
            response=None,
            body=None,
        )

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role not in _E2EE_ENCRYPT_ROLES:
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if content is not None and not isinstance(content, str):
            raise InvalidRequestError(
                "Multimodal (image / file) message content is not supported under "
                "Venice E2EE; only plain-text user/system messages can be "
                "encrypted. Send text-only content to use E2EE.",
                request=None,
                response=None,
                body=None,
            )


def _encrypt_body_messages(body: dict[str, Any], session: "TeeSession") -> None:
    """Encrypt user/system message content in the dumped body in place.

    Operates on the already-``model_dump``ed body (not caller objects), so the
    caller's ``messages`` are never mutated. Assistant / tool / developer content
    is left plaintext.
    """
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") not in _E2EE_ENCRYPT_ROLES:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = session.encrypt_message(content)


async def _decrypting_chunks(
    raw_iterator: AsyncIterator[ChatCompletionChunk],
    session: "TeeSession",
) -> AsyncIterator[ChatCompletionChunk]:
    """Yield chunks with encrypted ``delta.content`` decrypted in place.

    Each SSE chunk is independently encrypted, so decryption happens per chunk
    *before* any reassembly. Non-encrypted deltas (role-only, finish, usage) pass
    through untouched. The SESSION private key must outlive consumption, so the
    session is closed in ``finally`` when the stream is exhausted or aborted.
    """
    try:
        async for chunk in raw_iterator:
            if chunk.choices:
                delta = chunk.choices[0].delta
                content = delta.content
                if content and looks_encrypted(content):
                    delta.content = session.decrypt_chunk(content)
            yield chunk
    finally:
        session.close()


# --- Resource Class ---


class ChatCompletions(APIResource["VeniceClient"]):
    """
    Provides access to asynchronous chat completion operations.

    This class manages asynchronous chat completion operations with Venice AI models,
    supporting both standard (non-streaming) and streaming response formats. It serves
    as the primary interface for chat-based interactions with Venice AI language models
    in asynchronous contexts.

    The class handles parameter validation, request formation, and response parsing
    for asynchronous chat completion requests.

    :param _client: The client instance used to make API requests.
    :type _client: venice_ai._client.VeniceClient

    Example:

        .. code-block:: python

           from venice_ai import VeniceClient
           import asyncio

           async def main():
               # Initialize the async client
               async with VeniceClient(api_key="your-api-key") as client:

                   # Create a chat completion asynchronously. Model IDs change;
                   # resolve one from the live catalog rather than hardcoding.
                   response = await client.chat.completions.create(
                       model=await client.models.resolve_chat(),
                       messages=[
                           {"role": "system", "content": "You are a helpful assistant."},
                           {"role": "user", "content": "Tell me about Venice AI."}
                       ]
                   )

                   # Access the response content
                   print(response["choices"][0]["message"]["content"])

           # Run the async function
           asyncio.run(main())
    """

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        stream: Literal[False] = False,  # Explicit non-streaming case
        # --- Common Optional Parameters ---
        frequency_penalty: float | None = None,
        max_completion_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: (
            JSONSchemaFormat | JSONObjectFormat | TextResponseFormat | type[BaseModel] | None
        ) = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_temp: float | None = None,
        min_temp: float | None = None,
        min_p: float | None = None,
        tools: Sequence[Tool] | None = None,
        tool_choice: Literal["none", "auto"] | SpecificToolChoice | None = None,
        user: str | None = None,  # Discarded but supported for OpenAI compat
        venice_parameters: VeniceParameters | Mapping[str, Any] | None = None,
        # --- Venice-Specific Params ---
        reasoning_effort: ReasoningEffortLevel | None = None,
        reasoning: ReasoningConfig | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: Literal["default", "extended", "24h"] | None = None,
        store: bool | None = None,
        text: dict[str, Any] | None = None,
        include: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        verbosity: Literal["low", "medium", "high", "auto"] | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        # --- Less Common / Newer Params from Docs ---
        logprobs: bool | None = None,  # If requesting logprobs (check API if bool or object)
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        repetition_penalty: float | None = None,
        stop_token_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        stream_options: StreamOptions | None = None,
        stream_cls: type[ChunkModelFactory[ChatCompletionChunk]] | None = None,
        e2ee: bool | TeeOptions = False,
        **kwargs: Any,
    ) -> ChatCompletionResponse:  # Return type for non-streaming
        ...

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        stream: Literal[True],
        stream_cls: type[ChunkModelFactory[ChatCompletionChunk]] | None = None,
        # --- Common Optional Parameters ---
        frequency_penalty: float | None = None,
        max_completion_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        response_format: (
            JSONSchemaFormat | JSONObjectFormat | TextResponseFormat | type[BaseModel] | None
        ) = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_temp: float | None = None,
        min_temp: float | None = None,
        min_p: float | None = None,
        tools: Sequence[Tool] | None = None,
        tool_choice: Literal["none", "auto"] | SpecificToolChoice | None = None,
        user: str | None = None,
        venice_parameters: VeniceParameters | Mapping[str, Any] | None = None,
        # --- Venice-Specific Params ---
        reasoning_effort: ReasoningEffortLevel | None = None,
        reasoning: ReasoningConfig | None = None,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: Literal["default", "extended", "24h"] | None = None,
        store: bool | None = None,
        text: dict[str, Any] | None = None,
        include: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        verbosity: Literal["low", "medium", "high", "auto"] | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        # --- Less Common / Newer Params ---
        logprobs: bool | None = None,
        top_logprobs: int | None = None,
        parallel_tool_calls: bool | None = None,
        repetition_penalty: float | None = None,
        stop_token_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        stream_options: StreamOptions | None = None,
        e2ee: bool | TeeOptions = False,
        **kwargs: Any,
    ) -> AsyncIterable[ChatCompletionChunk]:  # Return type for streaming (async iterator of dicts)
        ...

    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        stream: bool = False,
        stream_cls: type[ChunkModelFactory[ChatCompletionChunk]] | None = None,
        **kwargs: Any,  # Catch all other keyword args
    ) -> ChatCompletionResponse | AsyncIterable[ChatCompletionChunk]:
        """
        Create a model response for the given chat conversation asynchronously.

        This method handles the core functionality of the chat completions API, allowing
        for both synchronous and streaming responses in async contexts. It sends the provided
        messages and parameters to the Venice AI API and returns either a complete response or
        a stream of partial responses.

        The method automatically formats the request body, applies appropriate defaults,
        and routes the request to either the standard or streaming endpoint based on
        the ``stream`` parameter.

        Wraps ``POST /api/v1/chat/completions``. Streaming requests use the
        same path with Server-Sent Events.

        Args:
            model: ID of the model to use. Resolve via
                ``client.models.resolve_chat()`` rather than hardcoding.
            messages: Sequence of messages forming the conversation. Each
                entry is one of :class:`UserMessage`, :class:`AssistantMessage`,
                :class:`SystemMessage`, :class:`ToolMessage`, or
                :class:`DeveloperMessage` — or a plain mapping in the same
                shape (``{"role": "user", "content": "hi"}``), which is
                validated into the matching model before the request is
                built. The models are preferred: they give editor completion
                and raise on a bad field at construction rather than at call
                time.
            stream: If ``True``, stream back partial progress as
                ``AsyncIterator[ChatCompletionChunk]``. Defaults to ``False``,
                which returns a single :class:`ChatCompletionResponse`.
            stream_cls: Optional stream wrapper class for streaming responses.
                Must conform to the
                :class:`~venice_ai.types.api.streaming.ChunkModelFactory`
                protocol. Defaults to :class:`~venice_ai.streaming.Stream`.
            frequency_penalty: Number between -2.0 and 2.0. Positive values
                penalize new tokens based on their existing frequency in the
                text so far.
            max_completion_tokens: Maximum number of completion tokens to
                generate. On reasoning-capable models this is a strict cap on
                *total* completion tokens — visible output **plus** internal
                reasoning tokens — not just the visible output. (``max_tokens``
                was accepted as an alias in v1 but is removed in v2; passing it
                raises ``TypeError``.)
            n: Number of chat completion choices to generate for each input
                message.
            presence_penalty: Number between -2.0 and 2.0. Positive values
                penalize new tokens based on whether they appear in the text
                so far.
            response_format: Specifies the format the model must output
                (e.g. for JSON mode). Accepts :class:`JSONSchemaFormat`,
                :class:`JSONObjectFormat`, :class:`TextResponseFormat`, or a
                Pydantic ``BaseModel`` subclass (auto-converted to a strict
                JSON schema).
            seed: Random seed for reproducible outputs.
            stop: Up to 4 sequences where the API will stop generating
                further tokens.
            temperature: Sampling temperature between 0.0 and 2.0. Higher
                values make output more random, lower values more focused
                and deterministic. Defaults to ``0.7`` server-side.
            top_p: Nucleus sampling parameter between 0.0 and 1.0. Defaults
                to ``1.0`` server-side.
            max_temp: Upper bound for dynamic temperature scaling (0.0-2.0).
                Used with ``min_temp`` to let the model adjust temperature
                per-token within a range instead of using a fixed
                ``temperature``.
            min_temp: Lower bound for dynamic temperature scaling
                (0.0-2.0). See ``max_temp``.
            min_p: Minimum probability threshold (0.0-1.0) relative to the
                most likely token, used as an alternative to ``top_p``.
            tools: List of tools the model may call.
            tool_choice: Controls which (if any) tool is called by the
                model. Can be ``"none"``, ``"auto"``, or a
                :class:`SpecificToolChoice`.
            user: Unique identifier representing your end-user (discarded
                by API but supported for OpenAI compatibility).
            venice_parameters: Venice-specific parameters for fine-tuning
                model behavior. Accepts a :class:`VeniceParameters` or a
                plain mapping of the same fields; a mapping is validated
                into the model while the request body is built, so an
                invalid value still raises before the request is sent.
                Note that :class:`VeniceParameters` is ``extra="allow"``,
                so an unrecognized key is carried through rather than
                rejected — the same for either form.
            reasoning_effort: Controls thinking depth on reasoning models.
                One of ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``,
                ``"high"``, ``"xhigh"``, or ``"max"``. Takes precedence
                over ``reasoning.effort`` when both are set.
            reasoning: Nested reasoning configuration. Accepts ``effort``
                (same enum as ``reasoning_effort``) and ``summary``
                (``"auto"`` / ``"concise"`` / ``"detailed"``).
            prompt_cache_key: Routing hint to improve cache hit rates
                across multi-turn conversations. Requests sharing the same
                key are more likely to hit cached prompt prefixes.
            prompt_cache_retention: Cache retention tier. ``"default"``
                uses the standard TTL; ``"extended"`` or ``"24h"`` keep
                the prompt cached for longer, improving hit rates for
                long-running agents at a small storage premium.
            store: OpenAI-compat flag forwarded to the upstream model;
                controls whether the completion is stored server-side for
                replay.
            text: OpenAI-compat text configuration (e.g.
                ``{"verbosity": "low"}``). Forwarded verbatim.
            include: OpenAI-compat inclusion specifier; an array of
                response-enrichment opt-in strings passed through to the
                model.
            metadata: OpenAI-compat free-form metadata dict attached to
                the request. Forwarded verbatim - useful for client-side
                observability.
            logprobs: Whether to return log probabilities of the output
                tokens.
            top_logprobs: Number of most likely tokens to return at each
                token position if ``logprobs`` is ``True``.
            parallel_tool_calls: Whether to enable parallel function
                calling during tool use.
            repetition_penalty: Penalty for token repetition.
            stop_token_ids: List of token IDs at which to stop generation.
            top_k: Number of highest probability vocabulary tokens to keep
                for top-k-filtering.
            stream_options: Additional options for controlling streaming
                behavior.
            e2ee: Engage Venice confidential-compute (TEE) end-to-end
                encryption. The flow runs when ``e2ee`` is truthy OR when
                ``venice_parameters.enable_e2ee`` is ``True``; it requires an
                ``e2ee-*`` model and the ``[e2ee]`` extra (``cryptography``).
                ``True`` uses defaults; pass a
                :class:`~venice_ai.tee.types.TeeOptions` to control the
                attestation freshness nonce or supply a full quote verifier.
                When engaged the SDK verifies the model's attestation
                (fail-closed), encrypts each user/system message to the model
                key, forces a wire stream with the ``X-Venice-TEE-*`` headers,
                and decrypts the response locally (reassembling a normal
                :class:`ChatCompletionResponse` when ``stream`` is ``False``).
                Tools, web search/scraping, and multimodal content are rejected
                with :class:`~venice_ai.exceptions.InvalidRequestError` before
                any network call, and the Venice system prompt is forced off.
                SECURITY LIMITATION: the baseline attestation verifier trusts
                Venice's server-side ``verified`` claim and does not perform
                full client-side TDX / NVIDIA quote verification; a one-time
                :class:`UserWarning` is emitted on engagement.
            kwargs: Additional keyword arguments forwarded to the request
                body for forward-compatibility.

        Returns:
            :class:`~venice_ai.types.api.chat.ChatCompletionResponse` when
            ``stream`` is ``False``, otherwise an ``AsyncIterable`` of
            :class:`~venice_ai.types.api.streaming.ChatCompletionChunk`.

        Raises:
            InvalidRequestError: If parameters are invalid or malformed.
            AuthenticationError: If the API key is invalid or missing.
            PermissionDeniedError: If access is denied to the requested
                model or feature.
            NotFoundError: If the model or resource is not found.
            RateLimitError: If rate limits are exceeded for the account.
            TypeError: If the legacy ``max_tokens`` kwarg is supplied (use
                ``max_completion_tokens`` in v2).
            APIError: For other API-related errors not covered by specific
                exceptions.

        Example:

            .. code-block:: python

               import asyncio
               from venice_ai import VeniceClient

               async def main():
                   async with VeniceClient() as client:
                       model = await client.models.resolve_chat()
                       response = await client.chat.completions.create(
                           model=model,
                           messages=[
                               {"role": "system", "content": "You are a helpful assistant."},
                               {"role": "user", "content": "Explain async programming in Python."},
                           ],
                           temperature=0.3,
                       )
                       print(response.choices[0].message.content)

               asyncio.run(main())

               # Streaming variant
               async def stream_example():
                   async with VeniceClient() as client:
                       model = await client.models.resolve_chat()
                       async for chunk in await client.chat.completions.create(
                           model=model,
                           messages=[{"role": "user", "content": "Tell me a story."}],
                           stream=True,
                           max_completion_tokens=200,
                       ):
                           content = chunk.choices[0].delta.content or ""
                           if content:
                               print(content, end="", flush=True)

               asyncio.run(stream_example())
        """
        # Validate model ID
        validate_model_id(model, "model")

        # Guard: max_tokens was removed in v2
        if "max_tokens" in kwargs:
            raise TypeError("max_tokens has been removed in v2. Use max_completion_tokens instead.")

        # Pop e2ee FIRST — it must not leak into api_params / remaining_kwargs and
        # become part of the request body. Engagement (and the FAIL-LOUD guards)
        # are resolved below, before any network call.
        e2ee = kwargs.pop("e2ee", False)

        # Extract all optional parameters from kwargs
        frequency_penalty = kwargs.pop("frequency_penalty", None)
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        n = kwargs.pop("n", None)
        presence_penalty = kwargs.pop("presence_penalty", None)
        response_format = kwargs.pop("response_format", None)

        # Convert Pydantic BaseModel subclass → JSONSchemaFormat
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            response_format = JSONSchemaFormat(
                type="json_schema",
                json_schema={
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": response_format.model_json_schema(),
                },
            )

        seed = kwargs.pop("seed", None)
        stop = kwargs.pop("stop", None)
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        max_temp = kwargs.pop("max_temp", None)
        min_temp = kwargs.pop("min_temp", None)
        min_p = kwargs.pop("min_p", None)
        tools = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)
        user = kwargs.pop("user", None)
        venice_parameters = kwargs.pop("venice_parameters", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning = kwargs.pop("reasoning", None)
        prompt_cache_key = kwargs.pop("prompt_cache_key", None)
        prompt_cache_retention = kwargs.pop("prompt_cache_retention", None)
        store = kwargs.pop("store", None)
        text = kwargs.pop("text", None)
        include = kwargs.pop("include", None)
        metadata = kwargs.pop("metadata", None)
        verbosity = kwargs.pop("verbosity", None)
        fallbacks = kwargs.pop("fallbacks", None)
        logprobs = kwargs.pop("logprobs", None)
        top_logprobs = kwargs.pop("top_logprobs", None)
        parallel_tool_calls = kwargs.pop("parallel_tool_calls", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        stop_token_ids = kwargs.pop("stop_token_ids", None)
        top_k = kwargs.pop("top_k", None)
        stream_options = kwargs.pop("stream_options", None)

        # Build the parameters dictionary, excluding None values and non-API parameters
        api_params = {}
        if frequency_penalty is not None:
            api_params["frequency_penalty"] = frequency_penalty
        if max_completion_tokens is not None:
            api_params["max_completion_tokens"] = max_completion_tokens
        if n is not None:
            api_params["n"] = n
        if presence_penalty is not None:
            api_params["presence_penalty"] = presence_penalty
        if response_format is not None:
            api_params["response_format"] = response_format
        if seed is not None:
            api_params["seed"] = seed
        if stop is not None:
            api_params["stop"] = stop
        if temperature is not None:
            api_params["temperature"] = temperature
        if top_p is not None:
            api_params["top_p"] = top_p
        if max_temp is not None:
            api_params["max_temp"] = max_temp
        if min_temp is not None:
            api_params["min_temp"] = min_temp
        if min_p is not None:
            api_params["min_p"] = min_p
        if tools is not None:
            api_params["tools"] = tools
        if tool_choice is not None:
            api_params["tool_choice"] = tool_choice
        if user is not None:
            api_params["user"] = user
        if venice_parameters is not None:
            api_params["venice_parameters"] = venice_parameters
        if reasoning_effort is not None:
            api_params["reasoning_effort"] = reasoning_effort
        if reasoning is not None:
            api_params["reasoning"] = reasoning
        if prompt_cache_key is not None:
            api_params["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_retention is not None:
            api_params["prompt_cache_retention"] = prompt_cache_retention
        if store is not None:
            api_params["store"] = store
        if text is not None:
            api_params["text"] = text
        if include is not None:
            api_params["include"] = include
        if metadata is not None:
            api_params["metadata"] = metadata
        if verbosity is not None:
            api_params["verbosity"] = verbosity
        if fallbacks is not None:
            api_params["fallbacks"] = fallbacks
        if logprobs is not None:
            api_params["logprobs"] = logprobs
        if top_logprobs is not None:
            api_params["top_logprobs"] = top_logprobs
        if parallel_tool_calls is not None:
            api_params["parallel_tool_calls"] = parallel_tool_calls
        if repetition_penalty is not None:
            api_params["repetition_penalty"] = repetition_penalty
        if stop_token_ids is not None:
            api_params["stop_token_ids"] = stop_token_ids
        if top_k is not None:
            api_params["top_k"] = top_k
        if stream_options is not None:
            api_params["stream_options"] = stream_options

        # Add any remaining kwargs (for extensibility)
        remaining_kwargs = {k: v for k, v in kwargs.items() if k != "stream_cls" and v is not None}
        api_params.update(remaining_kwargs)

        # Create Pydantic request model
        chat_request = ChatCompletionRequest(
            model=model,
            messages=_coerce_messages(messages),  # Mappings validated into message models
            stream=stream,
            **api_params,
        )

        # Convert to API payload
        body = chat_request.model_dump(exclude_none=True)

        # Handle specific naming or structuring if needed
        # e.g. if venice_parameters needs special handling

        # --- E2EE (Venice confidential-compute) path -------------------------
        # Engaged when e2ee=... is truthy OR venice_parameters.enable_e2ee=True.
        # All guards FAIL LOUD before any network call (open_session does a GET).
        if _e2ee_engaged(e2ee, venice_parameters):
            return await self._create_e2ee(
                model=model,
                messages=messages,
                body=body,
                stream=stream,
                e2ee=e2ee,
                venice_parameters=venice_parameters,
                tools=tools,
            )

        if stream:
            logger.debug("Async create: Entered streaming logic block.")

            # Force stream_options.include_usage=True so the final chunk carries
            # token usage. This is required for refund-based accounting and rate
            # limit tracking on streaming responses.
            existing_stream_options = body.get("stream_options")
            if existing_stream_options is None:
                # No stream_options provided - create with include_usage=True
                body["stream_options"] = {"include_usage": True}
            elif isinstance(existing_stream_options, dict):
                # Dict provided - merge with include_usage=True (override if present)
                body["stream_options"] = {
                    **existing_stream_options,
                    "include_usage": True,
                }
            elif hasattr(existing_stream_options, "model_dump"):
                # Pydantic model - convert to dict then merge
                body["stream_options"] = {
                    **existing_stream_options.model_dump(exclude_none=True),
                    "include_usage": True,
                }
            else:
                # Unknown type - default to include_usage=True only
                logger.warning(
                    f"Unknown stream_options type {type(existing_stream_options)}, "
                    "replacing with include_usage=True"
                )
                body["stream_options"] = {"include_usage": True}

            user_provided_stream_cls_async = stream_cls
            effective_stream_cls_async: Any = Stream  # Default

            # Stream class validation logic
            # PRIMARY PATH: Use default Stream class for standard streaming
            # CUSTOM SUPPORT: Allows advanced users to provide custom stream classes that
            # implement the proper interface. This flexibility enables specialized streaming
            # behavior (e.g., custom parsing, filtering, or transformation) while maintaining
            # type safety and backward compatibility with the default Stream class.
            if user_provided_stream_cls_async is not None:
                if inspect.isclass(user_provided_stream_cls_async):
                    try:
                        # First check if it's a subclass of our known stream types
                        if issubclass(user_provided_stream_cls_async, Stream):
                            effective_stream_cls_async = cast(Any, user_provided_stream_cls_async)
                            logger.debug("Using custom Stream subclass")
                        else:
                            # For custom classes, check if they have the proper interface
                            # They should have __init__ with iterator and client params, and __aiter__ method
                            sig = inspect.signature(user_provided_stream_cls_async.__init__)
                            params = list(sig.parameters.keys())
                            has_proper_signature = len(params) >= 3 or "client" in params
                            has_aiter_method = hasattr(user_provided_stream_cls_async, "__aiter__")

                            if has_proper_signature and has_aiter_method:
                                effective_stream_cls_async = cast(
                                    Any, user_provided_stream_cls_async
                                )
                                logger.debug("Using custom stream class with valid interface")
                            else:
                                logger.warning(
                                    f"Custom stream class {user_provided_stream_cls_async.__name__} "
                                    "does not have proper interface, using default Stream"
                                )
                    except (TypeError, ValueError) as e:
                        # If we can't inspect the signature, fall back to default
                        logger.debug(f"Failed to inspect custom stream class ({e}), using default")
                        pass  # effective_stream_cls_async remains Stream
                else:
                    logger.warning("stream_cls is not a class, using default Stream")
            # else: stream_cls is None, use default

            # _stream_request is an async generator function, calling it returns the async generator object.
            raw_iterator = self._client._stream_request(
                method="POST",
                path="chat/completions",
                json_data=body,
                cast_to=ChatCompletionChunk,
            )
            logger.debug(
                f"Attempting to return stream_cls: {effective_stream_cls_async}, with iterator: {raw_iterator}"
            )
            instantiated_stream = effective_stream_cls_async(raw_iterator, client=self._client)
            logger.debug(f"Result of stream_cls instantiation: {instantiated_stream}")
            return cast(AsyncIterable[ChatCompletionChunk], instantiated_stream)
        else:
            # Use regular post method for non-streaming responses
            response = await self._client.post(
                "chat/completions", json_data=body, cast_to=ChatCompletionResponse
            )
            # The response is now properly validated by the client
            return response

    async def _create_e2ee(
        self,
        *,
        model: str,
        messages: Sequence[Any],
        body: dict[str, Any],
        stream: bool,
        e2ee: bool | TeeOptions,
        venice_parameters: Any,
        tools: Any,
    ) -> ChatCompletionResponse | AsyncIterable[ChatCompletionChunk]:
        """Run the real Venice E2EE chat flow.

        Validates fail-closed, opens a verified :class:`TeeSession`, encrypts the
        user/system message content, forces a wire stream with the three
        ``X-Venice-TEE-*`` headers, then either yields decrypted deltas
        (``stream=True``) or reassembles a surface-preserving
        :class:`ChatCompletionResponse` (``stream=False``).
        """
        # (1) FAIL LOUD before any network call.
        _validate_e2ee_request(
            model=model,
            messages=messages,
            tools=tools,
            venice_parameters=venice_parameters,
        )

        # (2) One-time-per-call attestation-trust limitation warning. Emitted
        # unconditionally (no module-global flag) so each engagement is honest;
        # Python's default filter handles per-callsite de-duplication for UX.
        warnings.warn(_E2EE_TRUST_WARNING, UserWarning, stacklevel=3)

        # (3) Open a verified session (attestation GET + SESSION keypair). Nonce /
        # verifier flow through from TeeOptions; bool uses defaults.
        opts = e2ee if isinstance(e2ee, TeeOptions) else TeeOptions()
        session = await self._client.tee.open_session(
            model=model,
            nonce=opts.nonce,
            verifier=opts.verifier,
        )

        # (4) Shape the wire body: force stream + include_usage, encrypt content,
        # force the Venice system prompt off. Only pass enable_e2ee through if the
        # caller set it (the body is NOT required for the flow to work).
        body["stream"] = True
        existing_stream_options = body.get("stream_options")
        if isinstance(existing_stream_options, dict):
            body["stream_options"] = {**existing_stream_options, "include_usage": True}
        else:
            body["stream_options"] = {"include_usage": True}

        vp = body.get("venice_parameters")
        if not isinstance(vp, dict):
            vp = {}
        vp["include_venice_system_prompt"] = False
        caller_enable_e2ee = _venice_params_as_dict(venice_parameters).get("enable_e2ee")
        if caller_enable_e2ee is not None:
            vp["enable_e2ee"] = caller_enable_e2ee
        body["venice_parameters"] = vp

        _encrypt_body_messages(body, session)

        # (5) Stream on the wire with the TEE headers; decrypt each delta.
        raw_iterator = self._client._stream_request(
            method="POST",
            path="chat/completions",
            json_data=body,
            headers=session.request_headers(),
            cast_to=ChatCompletionChunk,
        )
        decrypting = _decrypting_chunks(raw_iterator, session)

        if stream:
            return ChatStream(decrypting, client=self._client)
        # stream=False: consume the forced stream and reassemble a normal,
        # surface-preserving ChatCompletionResponse from the decrypted deltas.
        return await ChatStream(decrypting, client=self._client).collect()

    async def stream(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        e2ee: bool | TeeOptions = False,
        **kwargs: Any,
    ) -> ChatStream:
        """Shorthand for ``create(stream=True)`` returning a :class:`~venice_ai.streaming.ChatStream`.

        The returned stream supports async context manager,
        :meth:`~ChatStream.text_deltas`, and :meth:`~ChatStream.collect`::

            async with await client.chat.completions.stream(model=model, messages=messages) as s:
                async for text in s.text_deltas():
                    print(text, end="")

        Wraps ``POST /api/v1/chat/completions`` (Server-Sent Events).

        Args:
            model: Model id to use.
            messages: Conversation messages.
            e2ee: Engage the Venice confidential-compute (TEE) end-to-end
                encryption flow. ``True`` uses defaults; pass a
                :class:`~venice_ai.tee.types.TeeOptions` to control the
                attestation nonce or supply a full quote verifier. Requires an
                ``e2ee-*`` model and the ``[e2ee]`` extra; the deltas yielded by
                the returned stream are already decrypted plaintext. See
                :meth:`create` for the engagement rules and limitations.
            kwargs: All other parameters accepted by :meth:`create`.

        Returns:
            A :class:`~venice_ai.streaming.ChatStream` instance ready to be
            iterated as an async context manager.

        Raises:
            InvalidRequestError: If parameters fail server-side validation, or
                if E2EE is requested on an incompatible request (non-``e2ee-``
                model, tools, web search/scraping, or multimodal content).
            AuthenticationError: If the API key is missing or invalid.
            RateLimitError: If account-level rate limits are exceeded.
            APIError: For other HTTP-level failures.
        """
        kwargs.pop("stream", None)
        result = await self.create(model=model, messages=messages, stream=True, e2ee=e2ee, **kwargs)
        # result is a Stream[ChatCompletionChunk] — wrap in ChatStream
        if isinstance(result, ChatStream):
            return result
        # result is a generic Stream; transfer its iterator into a ChatStream
        if isinstance(result, Stream):
            return ChatStream(result.get_iterator(), client=self._client)
        # Fallback: bare AsyncIterable (rare path); construct ChatStream around its iterator
        return ChatStream(result.__aiter__(), client=self._client)

    async def estimate_cost(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        expected_completion_tokens: int = 500,
        tokens_per_word: float = 1.3,
    ) -> ChatCostEstimate:
        """Estimate the USD cost of a chat completion before sending it.

        Mirrors the pre-flight ``quote()`` method on ``client.video`` and
        ``client.audio`` for symmetry. Token counts are heuristic
        (word-count x ``tokens_per_word``) - the same approximation used
        by :func:`venice_ai.costs.estimate_completion_cost` - so the
        result is an estimate, not a guarantee.

        SDK-side helper. The pricing lookup wraps
        ``GET /api/v1/models`` via :meth:`client.models.list`; no other
        wire calls are made.

        Args:
            model: Chat model id whose pricing to look up.
            messages: The messages you intend to send.
            expected_completion_tokens: Caller's budget for the model's
                response. Defaults to ``500``.
            tokens_per_word: Word -> token conversion ratio. Default
                ``1.3`` is tuned for English; raise for code/CJK.

        Returns:
            :class:`~venice_ai.costs.ChatCostEstimate` with the prompt /
            completion / total USD breakdown.

        Raises:
            ValueError: If ``model`` is not present in
                :meth:`client.models.list` or has no LLM token-based
                pricing (the estimate would otherwise be meaningless).
            APIError: For HTTP-level failures while fetching the model
                catalog.

        Example::

            estimate = await client.chat.completions.estimate_cost(
                model=await client.models.resolve_chat(),
                messages=[UserMessage(content="Summarize this contract...")],
                expected_completion_tokens=1500,
            )
            if estimate.total_cost_usd > Decimal("0.10"):
                raise BudgetError(f"Too expensive: ${estimate.total_cost_usd}")
        """
        pricing = await self._fetch_chat_pricing(model)

        prompt_text = _concat_message_text(_coerce_messages(messages))
        prompt_tokens = int(len(prompt_text.split()) * tokens_per_word)

        input_usd = Decimal(str(pricing.input.usd or 0.0))
        output_usd = Decimal(str(pricing.output.usd or 0.0))
        million = Decimal("1000000")

        prompt_cost = (Decimal(prompt_tokens) / million) * input_usd
        completion_cost = (Decimal(expected_completion_tokens) / million) * output_usd

        return ChatCostEstimate(
            model=model,
            prompt_tokens=prompt_tokens,
            expected_completion_tokens=expected_completion_tokens,
            prompt_cost_usd=prompt_cost,
            completion_cost_usd=completion_cost,
            total_cost_usd=prompt_cost + completion_cost,
        )

    async def parse[T: BaseModel](
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        response_format: type[T],
        schema_name: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> ParsedChatCompletion[T]:
        """Auto-validating sibling of :meth:`create` for structured output.

        Pass a Pydantic ``BaseModel`` subclass as ``response_format``; the
        SDK builds the JSON Schema from it, sends the request, and
        validates the first choice's content against the schema before
        returning. Errors surface at this callsite
        (``pydantic.ValidationError``) instead of downstream when the
        caller would otherwise have run
        :meth:`ChatCompletionResponse.parse_as` themselves.

        Wraps ``POST /api/v1/chat/completions`` with
        ``response_format={"type": "json_schema", ...}``.

        Args:
            model: Chat model id.
            messages: Conversation messages.
            response_format: A Pydantic ``BaseModel`` subclass describing
                the desired response shape.
            schema_name: Optional schema name sent to the API. Defaults
                to ``response_format.__name__``. Some providers display
                this name in tool-style UIs.
            strict: Whether to set ``strict: true`` in the JSON Schema
                payload. Recommended; the API may then reject responses
                that don't match the schema rather than silently
                returning bad JSON.
            kwargs: All other keyword arguments accepted by :meth:`create`
                (``temperature``, ``max_completion_tokens``, ``tools``,
                etc.) are forwarded unchanged. ``stream`` is rejected -
                :meth:`parse` does not support streaming.

        Returns:
            :class:`~venice_ai.types.api.chat.ParsedChatCompletion` with
            the validated ``parsed`` instance and the underlying
            :class:`ChatCompletionResponse`.

        Raises:
            ValueError: If ``stream=True`` is passed (use :meth:`stream`
                instead) or the model returns a tool-call-only /
                multimodal choice with no text content.
            TypeError: If :meth:`create` returns an unexpected non-
                :class:`ChatCompletionResponse` value.
            pydantic.ValidationError: If the model's response doesn't
                match ``response_format``.
            InvalidRequestError: If parameters fail server-side
                validation.
            AuthenticationError: If the API key is missing or invalid.
            RateLimitError: If account-level rate limits are exceeded.
            APIError: For other HTTP-level failures.

        Example::

            class Person(BaseModel):
                name: str
                age: int

            result = await client.chat.completions.parse(
                model=await client.models.resolve_chat(),
                messages=[UserMessage(content="Tell me about Marie Curie.")],
                response_format=Person,
            )
            person: Person = result.parsed
            print(result.usage.total_tokens, person.name, person.age)
        """
        if kwargs.get("stream"):
            raise ValueError(
                "parse() does not support stream=True. Use stream() and call "
                "ChatCompletionResponse.parse_as() on the collected response."
            )

        schema_format = JSONSchemaFormat(
            type="json_schema",
            json_schema={
                "name": schema_name or response_format.__name__,
                "strict": strict,
                "schema": response_format.model_json_schema(),
            },
        )

        result = await self.create(
            model=model,
            messages=messages,
            response_format=schema_format,
            **kwargs,
        )
        # parse() never streams — narrow for the type checker.
        if not isinstance(result, ChatCompletionResponse):
            raise TypeError(
                f"Expected ChatCompletionResponse from create(), got {type(result).__name__}"
            )

        parsed = result.parse_as(response_format)
        return ParsedChatCompletion(response=result, parsed=parsed)

    async def batch(
        self,
        requests: Sequence[dict[str, Any]],
        *,
        max_concurrency: int = 10,
        return_exceptions: bool = True,
    ) -> list[ChatCompletionResponse | BaseException]:
        """Run many :meth:`create` calls in parallel with bounded concurrency.

        Each entry in *requests* is a kwargs dict unpacked into
        :meth:`create`. Result order matches input order. By default,
        per-request exceptions are collected into the result list
        (mirroring ``asyncio.gather(return_exceptions=True)``) so a single
        failure does not abort the whole batch - set
        ``return_exceptions=False`` for all-or-nothing semantics.

        Streaming is rejected: an entry with ``stream=True`` results in a
        ``ValueError`` for that slot (or aborts the batch when
        ``return_exceptions=False``). Use :meth:`stream` directly inside
        ``asyncio.gather`` if you need concurrent streams.

        SDK-side helper that fans out across :meth:`create`; each child
        call wraps ``POST /api/v1/chat/completions``.

        Args:
            requests: Sequence of kwargs dicts for :meth:`create`.
            max_concurrency: Maximum concurrent in-flight requests
                (default ``10``). Must be ``>= 1``.
            return_exceptions: If ``True`` (default), exceptions for
                individual requests appear in their slot in the result
                list. If ``False``, the first exception raises and cancels
                pending tasks.

        Returns:
            A list of :class:`ChatCompletionResponse` (and
            :class:`BaseException` instances when
            ``return_exceptions=True``) in input order.

        Raises:
            ValueError: If ``max_concurrency < 1``.
            TypeError: If a child :meth:`create` returns an unexpected
                non-:class:`ChatCompletionResponse` value (with
                ``return_exceptions=False``).
            APIError: First child failure when
                ``return_exceptions=False``.

        Example::

            results = await client.chat.completions.batch(
                [
                    {"model": model, "messages": [UserMessage(content=q)]}
                    for q in questions
                ],
                max_concurrency=5,
            )
            for q, r in zip(questions, results, strict=True):
                if isinstance(r, BaseException):
                    print(f"{q!r} failed: {r}")
                else:
                    print(f"{q!r} -> {r.choices[0].message.content!r}")
        """
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if not requests:
            return []

        sem = asyncio.Semaphore(max_concurrency)

        async def _one(req: dict[str, Any]) -> ChatCompletionResponse:
            if req.get("stream"):
                raise ValueError(
                    "batch() does not support stream=True; use stream() directly "
                    "inside asyncio.gather if you need concurrent streams."
                )
            async with sem:
                result = await self.create(**req)
            if not isinstance(result, ChatCompletionResponse):
                raise TypeError(
                    f"Unexpected non-ChatCompletionResponse result from create(): "
                    f"{type(result).__name__}"
                )
            return result

        results = await asyncio.gather(
            *(_one(req) for req in requests),
            return_exceptions=return_exceptions,
        )
        return list(results)

    async def run_with_tools(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        tools: Sequence[Callable[..., Any] | Tool],
        on_tool_call: Callable[[ToolCall, Any], None] | None = None,
        on_tool_error: Callable[[ToolCall, Exception], str] | None = None,
        parallel: bool = False,
        max_iterations: int = 10,
        **create_kwargs: Any,
    ) -> ToolLoopResult:
        """Run an automatic tool-call loop until the model produces a final answer.

        Drives the canonical "create -> check finish_reason -> execute
        tools -> re-call" loop so callers don't have to. ``tools`` accepts
        bare Python callables (auto-converted with
        :func:`venice_ai.tool_from_function` and registered as the
        dispatch handler) or pre-built :class:`Tool` definitions paired
        with a separate ``on_tool_call`` dispatcher; the two shapes can be
        mixed in one list.

        Both sync and async tool callables are supported - they're
        detected with :func:`inspect.iscoroutinefunction` and invoked or
        awaited as appropriate. The caller's ``messages`` list is **not**
        mutated; the returned :class:`ToolLoopResult` exposes a fresh
        history copy along with the final response and iteration count.

        SDK-side orchestrator that calls
        ``POST /api/v1/chat/completions`` once per iteration via
        :meth:`create`.

        Args:
            model: Model id to use for every iteration.
            messages: Initial chat messages. Not mutated.
            tools: Bare callables, :class:`Tool` definitions, or a mix.
                Bare callables are introspected with
                :func:`tool_from_function`.
            on_tool_call: Optional observation hook invoked after each
                tool runs successfully - receives the :class:`ToolCall`
                and the handler's return value. Read-only; does not
                affect what the model sees.
            on_tool_error: Optional override for tool-error handling.
                Defaults to :func:`_default_on_tool_error`, which logs the
                exception (with traceback) to the ``venice_ai.tools``
                logger and returns a formatted string sent back to the
                model so it can recover. Pass a function that re-raises
                for strict propagation.
            parallel: If ``True``, multiple tool calls in one assistant
                response run concurrently via :func:`asyncio.gather`.
                Default ``False`` (sequential) to avoid surprise
                concurrency on tool functions that share state. Only set
                to ``True`` when handlers are concurrency-safe.
            max_iterations: Maximum number of model round trips before
                giving up. Default ``10``.
            create_kwargs: Forwarded to :meth:`create` on every iteration
                (e.g. ``temperature``, ``max_completion_tokens``,
                ``response_format``, ``venice_parameters``). ``stream`` is
                managed by this method and rejected if passed.

        Returns:
            A :class:`ToolLoopResult` with the terminal response, full
            message history, and round-trip count.

        Raises:
            MaxIterationsExceededError: If the loop hits ``max_iterations``
                while still receiving ``finish_reason="tool_calls"``
                responses.
            ValueError: If ``stream`` is passed via ``create_kwargs``, if
                a tool dispatch handler is missing for a tool the model
                called, if the model returns no choices, or if
                ``max_iterations < 1``.
            TypeError: If :meth:`create` returns an unexpected
                non-:class:`ChatCompletionResponse` value.
            InvalidRequestError: If parameters fail server-side
                validation.
            AuthenticationError: If the API key is missing or invalid.
            RateLimitError: If account-level rate limits are exceeded.
            APIError: For other HTTP-level failures during any iteration.
        """
        if "stream" in create_kwargs:
            raise ValueError("run_with_tools does not support streaming")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")

        registry = _normalize_tool_registry(tools)
        tool_defs = [entry.tool for entry in registry.values()]
        on_error = on_tool_error or _default_on_tool_error

        history: list[_ChatMessageModel] = _coerce_messages(messages)

        last_response: ChatCompletionResponse | None = None
        for iteration in range(1, max_iterations + 1):
            response = await self.create(
                model=model,
                messages=history,
                tools=tool_defs,
                **create_kwargs,
            )
            if not isinstance(response, ChatCompletionResponse):
                raise TypeError(
                    f"Expected ChatCompletionResponse from create(), got {type(response).__name__}"
                )
            last_response = response

            if not response.choices:
                raise ValueError("Model returned a response with no choices")
            choice = response.choices[0]
            # Whether terminal or another tool-call turn, the assistant's
            # message goes into history. For the terminal turn this means
            # callers can feed `result.messages` straight into a follow-up
            # call without having to re-derive the final AssistantMessage.
            history.append(AssistantMessage.from_response(response))
            if choice.finish_reason != "tool_calls":
                return ToolLoopResult(response=response, messages=history, iterations=iteration)
            tool_calls = choice.message.tool_calls or []

            if parallel:
                results = await asyncio.gather(
                    *(
                        _execute_tool_call(call, registry, on_tool_call, on_error)
                        for call in tool_calls
                    )
                )
            else:
                results = []
                for call in tool_calls:
                    results.append(await _execute_tool_call(call, registry, on_tool_call, on_error))

            for call, content in zip(tool_calls, results, strict=True):
                history.append(ToolMessage(tool_call_id=call.id, content=content))

        # Loop exhausted without convergence.
        assert last_response is not None  # max_iterations >= 1, so we ran at least once
        raise MaxIterationsExceededError(
            f"Tool loop did not converge within {max_iterations} iterations",
            iterations=max_iterations,
            messages=history,
            last_response=last_response,
        )

    async def _fetch_chat_pricing(self, model: str) -> LLMModelPricing:
        """Look up LLM pricing for *model* via ``models.list``.

        :raises ValueError: If the model is unknown or not LLM-priced.
        """
        listing = await self._client.models.list()
        for entry in listing.data:
            if entry.id != model:
                continue
            spec = entry.model_spec
            if spec.pricing is None or not isinstance(spec.pricing, LLMModelPricing):
                raise ValueError(
                    f"Model {model!r} has no LLM token-based pricing — "
                    f"cost estimate is not meaningful."
                )
            return spec.pricing
        raise ValueError(f"Model {model!r} not found in models.list()")
