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
import json
import logging
import math
import re
import warnings
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Mapping,
    Sequence,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
    overload,
)

from pydantic import BaseModel, TypeAdapter, ValidationError

from ..._resource import APIResource
from ...costs import (
    CHAT_MESSAGE_TOKEN_ALLOWANCE,
    CHAT_TEMPLATE_TOKEN_ALLOWANCE,
    SCHEMA_JSON_CHARS_PER_TOKEN,
    TOOLS_TEMPLATE_TOKEN_ALLOWANCE,
    VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE,
    ChatCostEstimate,
    _estimate_costs,
)
from ...exceptions import InvalidRequestError, MaxIterationsExceededError, TeeEncryptionError
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
from ...types.api.chat import ChatMessage, ParsedChatCompletion, ToolLoopResult
from ...types.api.models import LLMModelPricing

# Import streaming models from generated.streaming module
from ...types.api.streaming import ChatCompletionChunk, ChunkModelFactory
from ...validation.validators import validate_model_id

if TYPE_CHECKING:
    from ..._client import VeniceClient  # noqa: F401
    from ...tee._session import TeeSession

logger = logging.getLogger(__name__)
_tools_logger = logging.getLogger("venice_ai.tools")

_ModelT = TypeVar("_ModelT", bound=BaseModel)

__all__ = ["ChatCompletions"]

#: Prefix that marks a Venice confidential-compute (TEE) chat model.
_E2EE_MODEL_PREFIX = "e2ee-"

#: Roles whose message fields are encrypted to the model under E2EE. The server
#: decrypts every non-empty message field it receives, so plaintext in any of
#: these roles is rejected; every other role is refused client-side.
_E2EE_ENCRYPTED_ROLES: frozenset[str] = frozenset({"user", "system", "assistant"})

#: Chat roles an E2EE refusal may name. A role outside this set is the caller's
#: own text, so a refusal calls it "an unsupported role" instead of quoting it.
_KNOWN_CHAT_ROLES: frozenset[str] = frozenset(
    {"system", "user", "assistant", "tool", "developer", "function"}
)

#: Validates ``venice_parameters`` exactly as :class:`ChatCompletionRequest` does,
#: so E2EE engagement reads the same ``enable_e2ee`` value the wire carries.
_VENICE_PARAMETERS_ADAPTER: TypeAdapter[Any] = TypeAdapter(
    cast(Any, ChatCompletionRequest.model_fields["venice_parameters"].annotation)
)

#: Emitted once per E2EE-engaged ``create`` call that uses the baseline
#: attestation verifier (no ``TeeOptions(verifier=...)``). The wire path is real,
#: but that verifier trusts Venice's server-side ``verified`` claim; full
#: client-side TDX / NVIDIA quote verification is not performed.
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
    name = call.function.name if call.function is not None else call.type
    _tools_logger.error(
        "Tool %s raised %s: %s",
        name,
        type(exc).__name__,
        exc,
        exc_info=True,
    )
    return f"Error calling {name}: {type(exc).__name__}: {exc}"


async def _execute_tool_call(
    call: ToolCall,
    registry: dict[str, _ToolEntry],
    on_tool_call: Callable[[ToolCall, Any], None] | None,
    on_tool_error: Callable[[ToolCall, Exception], str],
) -> str:
    """Dispatch a single tool call and return a string for the ``ToolMessage``."""
    if call.function is None:
        raise ValueError(
            f"Model made a {call.type!r} tool call (id {call.id!r}); run_with_tools "
            "only dispatches function tools."
        )
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


def _history_message(message: Any) -> Any:
    """A response :class:`ChatMessage` as history input; anything else unchanged.

    An assistant message goes through the same rule as
    :meth:`AssistantMessage.from_chat_message`, so the ``cache_control`` the API
    echoes is not sent back.
    """
    if not isinstance(message, ChatMessage):
        return message
    if message.role == "assistant":
        return AssistantMessage.from_chat_message(message)
    return message.model_dump(exclude_none=True)


def _coerce_messages(messages: Sequence[ChatMessageParam]) -> list[_ChatMessageModel]:
    """Validate a caller's messages into the concrete message models.

    :meth:`ChatCompletions.create` gets this for free, since
    ``ChatCompletionRequest`` coerces mappings while building the request
    body. Methods that read or accumulate messages *before* that point need
    it explicitly, so that a caller passing plain dicts doesn't hit an
    ``AttributeError`` on ``.content``. A response :class:`ChatMessage` (for
    example ``result.choices[0].message``) is accepted too, so it can be
    appended to the history of the next turn.
    """
    return _MESSAGE_LIST_ADAPTER.validate_python([_history_message(m) for m in messages])


def _schema_response_format(
    model_cls: type[BaseModel], *, name: str | None = None, strict: bool = True
) -> JSONSchemaFormat:
    """Build the ``json_schema`` response format sent for a Pydantic model class."""
    return JSONSchemaFormat(
        type="json_schema",
        json_schema={
            "name": name or model_cls.__name__,
            "strict": strict,
            "schema": model_cls.model_json_schema(),
        },
    )


def _compact_json_length(value: Any) -> int:
    """Length of *value* as the compact JSON the request body carries.

    Accepts dicts and lists and Pydantic model instances (dumped). A Pydantic
    model class passed as ``response_format`` is measured as the ``json_schema``
    format :meth:`ChatCompletions.create` sends for it. ``None`` is ``0``.
    """
    if value is None:
        return 0
    if isinstance(value, type) and issubclass(value, BaseModel):
        value = _schema_response_format(value)

    def _plain(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(exclude_none=True, by_alias=True)
        if isinstance(item, Mapping):
            return {k: _plain(v) for k, v in item.items()}
        if isinstance(item, list | tuple):
            return [_plain(v) for v in item]
        return item

    return len(json.dumps(_plain(value), separators=(",", ":"), default=str))


def _schema_token_allowance(tools: Sequence[Any] | None, response_format: Any | None) -> int:
    """Prompt-token allowance for the ``tools`` and ``response_format`` of a request.

    The compact JSON at :data:`~venice_ai.costs.SCHEMA_JSON_CHARS_PER_TOKEN`
    characters per token, plus
    :data:`~venice_ai.costs.TOOLS_TEMPLATE_TOKEN_ALLOWANCE` once when any tool
    is sent, for the tool-calling instructions the chat template adds.
    """
    chars = _compact_json_length(tools or None) + _compact_json_length(response_format)
    preamble = TOOLS_TEMPLATE_TOKEN_ALLOWANCE if tools else 0
    return preamble + math.ceil(chars / SCHEMA_JSON_CHARS_PER_TOKEN)


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
        # Multimodal list — pull out only the text-shaped entries, whether a
        # part is a model or its dict form.
        for chunk in content:
            if isinstance(chunk, Mapping):
                text = chunk.get("text") if chunk.get("type") == "text" else None
            else:
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
    if isinstance(venice_parameters, Mapping):
        return dict(venice_parameters)
    if hasattr(venice_parameters, "model_dump"):
        return cast(dict[str, Any], venice_parameters.model_dump(exclude_none=True))
    return {}


def _validated_enable_e2ee(venice_parameters: Any) -> bool | None:
    """``venice_parameters.enable_e2ee`` as the request model validates it.

    ``ChatCompletionRequest`` coerces values such as ``1`` or ``"true"`` to
    ``True`` before they reach the wire, so engagement must read the validated
    value, not the raw one. When ``venice_parameters`` itself does not validate,
    the request build raises later; until then only a literal ``True`` / ``False``
    counts.
    """
    try:
        validated = _VENICE_PARAMETERS_ADAPTER.validate_python(venice_parameters)
    except ValidationError:
        raw = _venice_params_as_dict(venice_parameters).get("enable_e2ee")
        return raw if isinstance(raw, bool) else None
    value = getattr(validated, "enable_e2ee", None)
    return value if isinstance(value, bool) else None


def _refuse_non_bool_wire_enable_e2ee(body: Mapping[str, Any]) -> None:
    """Refuse a plaintext request whose wire ``enable_e2ee`` is not ``False``.

    A :class:`VeniceParameters` built with ``model_construct`` or a subclass
    that loosens the field skips validation, so the wire can carry a value such
    as ``1`` that the SDK did not read as ``True`` and did not encrypt for. Only
    ``False`` or an absent value may go out with plaintext content.

    Raises:
        InvalidRequestError: ``enable_e2ee`` is on the wire with any value other
            than ``False`` or ``None``. The message does not quote the value.
    """
    wire = body.get("venice_parameters")
    if not isinstance(wire, Mapping):
        return
    value = wire.get("enable_e2ee")
    if value is None or value is False:
        return
    raise InvalidRequestError(
        "venice_parameters.enable_e2ee must be a bool. Pass True to use E2EE "
        "(or e2ee=True), or False / omit it for a plaintext request.",
        request=None,
        response=None,
        body=None,
    )


def _e2ee_engaged(e2ee: bool | TeeOptions, venice_parameters: Any) -> bool:
    """Whether the E2EE flow should run for this call.

    Engaged when the explicit ``e2ee`` argument is truthy OR when
    ``venice_parameters.enable_e2ee`` validates to ``True``.
    """
    if e2ee:
        return True
    return _validated_enable_e2ee(venice_parameters) is True


def _validate_e2ee_request(
    *,
    model: str,
    tools: Any,
    venice_parameters: Any,
    e2ee: bool | TeeOptions = False,
    logprobs: Any = None,
    top_logprobs: Any = None,
) -> None:
    """Fail loud, before any network call, on E2EE-incompatible requests.

    Raises:
        InvalidRequestError: ``e2ee`` set together with
            ``venice_parameters.enable_e2ee=False``; E2EE on a non-``e2ee-``
            model; tools / web search / web scraping / logprobs requested.
    """
    if e2ee and _validated_enable_e2ee(venice_parameters) is False:
        raise InvalidRequestError(
            "e2ee=True conflicts with venice_parameters.enable_e2ee=False. Drop "
            "enable_e2ee, or set it to True, to use E2EE.",
            request=None,
            response=None,
            body=None,
        )

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

    if logprobs or top_logprobs is not None:
        raise InvalidRequestError(
            "logprobs and top_logprobs are not supported under Venice E2EE: "
            "token log probabilities would be returned outside the encrypted "
            "channel. Drop `logprobs` / `top_logprobs` to use E2EE.",
            request=None,
            response=None,
            body=None,
        )


def _e2ee_refusal(message: str) -> InvalidRequestError:
    return InvalidRequestError(message, request=None, response=None, body=None)


def _message_role(message: Any) -> str | None:
    """The ``role`` of a caller's message, dict or model, when it is a string."""
    role = message.get("role") if isinstance(message, Mapping) else getattr(message, "role", None)
    return role if isinstance(role, str) else None


def _e2ee_validation_refusal(
    exc: ValidationError, messages: Sequence[ChatMessageParam]
) -> InvalidRequestError:
    """A content-free :class:`InvalidRequestError` for a failed E2EE request build.

    Pydantic's message quotes the offending input, which under E2EE is message
    text that must not leave the encrypted channel, not even in a traceback.
    The replacement names only the message index, its role when it is a known
    chat role, and the error types.
    """
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    loc = errors[0]["loc"] if errors else ()
    if loc[:1] == ("messages",):
        loc = loc[1:]
    error_types = ", ".join(sorted({str(error["type"]) for error in errors}))
    if loc and isinstance(loc[0], int) and 0 <= loc[0] < len(messages):
        index = loc[0]
        role = _message_role(messages[index])
        if role is None:
            where = f"messages[{index}]"
        elif role in _KNOWN_CHAT_ROLES:
            where = f"messages[{index}] (role {role!r})"
        else:
            where = f"messages[{index}] (an unsupported role)"
        return _e2ee_refusal(
            f"{where} is not a valid chat message ({error_types}). Details are "
            "withheld under Venice E2EE because they would quote message content."
        )
    field = str(loc[0]) if loc else "request"
    return _e2ee_refusal(
        f"Request field {field!r} is invalid ({error_types}). Details are withheld "
        "under Venice E2EE because they could quote message content."
    )


def _coerce_messages_for(
    messages: Sequence[ChatMessageParam], *, e2ee_engaged: bool
) -> list[_ChatMessageModel]:
    """:func:`_coerce_messages`, keeping message content out of errors under E2EE.

    Under E2EE a pydantic :class:`ValidationError` is replaced by a content-free
    :class:`InvalidRequestError` with no chained cause or context; otherwise it
    propagates unchanged.
    """
    try:
        return _coerce_messages(messages)
    except ValidationError as exc:
        if not e2ee_engaged:
            raise
        refusal = _e2ee_validation_refusal(exc, messages)
    raise refusal from None


def _build_chat_request(
    *,
    e2ee_engaged: bool,
    model: str,
    messages: Sequence[ChatMessageParam],
    stream: bool,
    api_params: dict[str, Any],
) -> ChatCompletionRequest:
    """Validate the caller's arguments into a :class:`ChatCompletionRequest`.

    Under E2EE a pydantic :class:`ValidationError` is replaced by a content-free
    :class:`InvalidRequestError` with no chained cause or context; otherwise it
    propagates unchanged.
    """
    coerced = _coerce_messages_for(messages, e2ee_engaged=e2ee_engaged)
    try:
        return ChatCompletionRequest(
            model=model,
            messages=coerced,
            stream=stream,
            **api_params,
        )
    except ValidationError as exc:
        if not e2ee_engaged:
            raise
        refusal = _e2ee_validation_refusal(exc, messages)
    raise refusal from None


def _is_absent(value: Any) -> bool:
    """Whether a message field counts as absent: ``None``, ``False``, ``[]`` or ``{}``.

    Uses identity and exact-type checks, so ``0`` and ``0.0`` (which compare
    equal to ``False``) are present values.
    """
    if value is None or value is False:
        return True
    return type(value) in (list, dict) and not value


def _validate_e2ee_messages(messages: list[dict[str, Any]]) -> None:
    """Refuse, before any network call, messages E2EE cannot carry.

    Runs on the dumped request body. A key whose value is ``None``, ``False``,
    ``[]`` or ``{}`` counts as absent. Errors name the message index and the
    offending role or key, never message content.

    Raises:
        InvalidRequestError: a role other than user / system / assistant, a key
            other than ``role`` / ``content`` / (assistant) ``reasoning_content``,
            or non-string ``content`` / ``reasoning_content``.
    """
    for index, msg in enumerate(messages):
        role = msg.get("role")
        if role == "developer":
            raise _e2ee_refusal(
                f"messages[{index}] has role 'developer', which is not supported "
                "under Venice E2EE; use role 'system' instead."
            )
        if role not in _E2EE_ENCRYPTED_ROLES:
            raise _e2ee_refusal(
                f"messages[{index}] has role {role!r}, which is not supported under "
                "Venice E2EE; only 'user', 'system' and 'assistant' messages can "
                "be encrypted."
            )
        for key, value in msg.items():
            if key in ("role", "content", "reasoning_content") or _is_absent(value):
                continue
            raise _e2ee_refusal(
                f"messages[{index}] (role {role!r}) has field {key!r}, which is not "
                "supported under Venice E2EE; only 'role', 'content' and, on "
                "assistant messages, 'reasoning_content' can be sent."
            )
        if not _is_absent(msg.get("reasoning_content")) and role != "assistant":
            raise _e2ee_refusal(
                f"messages[{index}] (role {role!r}) has field 'reasoning_content', "
                "which is only supported on assistant messages under Venice E2EE."
            )
        content = msg.get("content")
        if content is not None and not isinstance(content, str):
            raise _e2ee_refusal(
                f"messages[{index}] (role {role!r}) has non-text 'content'; "
                "multimodal (image / file) content is not supported under Venice "
                "E2EE. Send text-only content to use E2EE."
            )
        reasoning = msg.get("reasoning_content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise _e2ee_refusal(
                f"messages[{index}] (role {role!r}) has non-text 'reasoning_content', "
                "which is not supported under Venice E2EE."
            )


def _encrypt_body_messages(body: dict[str, Any], session: "TeeSession") -> None:
    """Encrypt message text in the dumped body in place.

    Operates on the already-``model_dump``ed body (not caller objects), so the
    caller's ``messages`` are never mutated. Every non-empty string ``content``
    (user / system / assistant) and assistant ``reasoning_content`` is
    encrypted; empty strings are sent unchanged.
    """
    for msg in body.get("messages", []):
        if msg.get("role") not in _E2EE_ENCRYPTED_ROLES:
            continue
        fields = ("content", "reasoning_content") if msg["role"] == "assistant" else ("content",)
        for field in fields:
            value = msg.get(field)
            if isinstance(value, str) and value:
                msg[field] = session.encrypt_message(value)


def _decrypt_delta_field(session: "TeeSession", delta: Any, field: str) -> None:
    """Decrypt ``delta.<field>`` in place, refusing a non-empty value that is not ciphertext."""
    value = getattr(delta, field)
    if not value:
        return
    if not looks_encrypted(value):
        raise TeeEncryptionError(
            f"E2EE response delta carried unencrypted {field}; refusing to pass it through."
        )
    setattr(delta, field, session.decrypt_chunk(value))


def _contains_text(value: Any) -> bool:
    """Whether *value* carries text: a ``str``, or any non-empty mapping, anywhere.

    Mapping keys are strings, so a non-empty mapping is text-bearing whatever
    its values (a ``{token: logprob}`` map would otherwise carry tokens in its
    keys). Lists and tuples are text-bearing if any element is.
    """
    if isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list | tuple):
        return any(_contains_text(item) for item in value)
    return False


def _refuse_text_bearing_extras(
    level: str,
    extras: Mapping[str, Any] | None,
    *,
    permitted: Mapping[str, tuple[str, ...]] | None = None,
) -> None:
    """Raise if an undeclared field at *level* carries text the enclave did not encrypt.

    A field carrying text in a key or a value is refused; numbers, booleans,
    ``None`` and lists or tuples of only those pass through. A key in
    *permitted* may also be exactly one of its listed strings. The error names
    the level and key, never the value.
    """
    for key, value in (extras or {}).items():
        if isinstance(value, str) and value in (permitted or {}).get(key, ()):
            continue
        if _contains_text(value):
            raise TeeEncryptionError(f"E2EE response {level} carried unexpected field {key!r}.")


def _request_stop_values(stop: Any) -> tuple[str, ...]:
    """The request's ``stop`` sequences as a tuple of strings (empty when unset)."""
    if isinstance(stop, str):
        return (stop,)
    if isinstance(stop, list | tuple):
        return tuple(item for item in stop if isinstance(item, str))
    return ()


#: Shape of a ``finish_reason`` under E2EE: protocol vocabulary such as
#: ``stop``, ``length`` or ``content_filter``, never free text.
_FINISH_REASON_PATTERN = re.compile(r"[a-z_]{1,32}")


def _refuse_unencryptable_delta_fields(choice: Any, stop: tuple[str, ...] = ()) -> None:
    """Raise if a choice or its delta carries anything the enclave does not encrypt.

    A choice's ``matched_stop`` or ``stop_reason`` may be a stop token id, or
    one of the request's own *stop* strings, which were sent unencrypted. A
    delta ``role`` must be ``None`` or ``"assistant"``, and a ``finish_reason``
    must be protocol vocabulary (lowercase letters and underscores, at most 32
    characters), since neither is encrypted.
    """
    delta = choice.delta
    if delta.role is not None and delta.role != "assistant":
        raise TeeEncryptionError(
            "E2EE response delta carried an unexpected role; refusing to pass it through."
        )
    finish_reason = choice.finish_reason
    if finish_reason is not None and not (
        isinstance(finish_reason, str) and _FINISH_REASON_PATTERN.fullmatch(finish_reason)
    ):
        raise TeeEncryptionError(
            "E2EE response choice carried a finish_reason outside the protocol "
            "vocabulary; refusing to pass it through."
        )
    if delta.tool_calls:
        raise TeeEncryptionError("E2EE response delta carried tool_calls, which are not encrypted.")
    if delta.reasoning_encrypted:
        raise TeeEncryptionError(
            "E2EE response delta carried reasoning_encrypted, which is not part of "
            "the E2EE contract."
        )
    if delta.thought_signature:
        raise TeeEncryptionError(
            "E2EE response delta carried thought_signature, which is not encrypted."
        )
    _refuse_text_bearing_extras("delta", delta.model_extra)
    _refuse_text_bearing_extras(
        "choice", choice.model_extra, permitted={"matched_stop": stop, "stop_reason": stop}
    )
    if choice.logprobs is not None:
        raise TeeEncryptionError("E2EE response choice carried logprobs, which are not encrypted.")


_COST_METADATA_KEYS: frozenset[str] = frozenset({"usd", "diem"})


def _is_cost_metadata(value: Any) -> bool:
    """Whether *value* is the usage chunk's numeric ``cost``: ``usd`` / ``diem`` numbers or ``None``."""
    return (
        isinstance(value, Mapping)
        and set(value) <= _COST_METADATA_KEYS
        and all(
            item is None or (isinstance(item, int | float) and not isinstance(item, bool))
            for item in value.values()
        )
    )


def _refuse_unencryptable_chunk_fields(chunk: ChatCompletionChunk) -> None:
    """Raise if a chunk or its ``usage`` carries a text-bearing field outside the protocol.

    The usage chunk's ``cost`` passes when it has the numeric shape
    :func:`_is_cost_metadata` checks; any other ``cost`` is held to the text rule.
    Undeclared fields on ``usage`` and its token-detail objects are held to the
    text rule too.
    """
    extras = {
        key: value
        for key, value in (chunk.model_extra or {}).items()
        if not (key == "cost" and _is_cost_metadata(value))
    }
    _refuse_text_bearing_extras("chunk", extras)
    usage = chunk.usage
    if usage is None:
        return
    _refuse_text_bearing_extras("usage", usage.model_extra)
    if usage.prompt_tokens_details is not None:
        _refuse_text_bearing_extras(
            "usage.prompt_tokens_details", usage.prompt_tokens_details.model_extra
        )
    if usage.completion_tokens_details is not None:
        _refuse_text_bearing_extras(
            "usage.completion_tokens_details", usage.completion_tokens_details.model_extra
        )


async def _decrypting_chunks(
    raw_iterator: AsyncIterator[ChatCompletionChunk],
    session: "TeeSession",
    *,
    stop: tuple[str, ...] = (),
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Yield chunks with encrypted ``content`` and ``reasoning_content`` decrypted in place.

    Each SSE chunk is independently encrypted, so decryption happens per chunk
    *before* any reassembly, on every choice. Fails closed: a non-empty
    ``content``/``reasoning_content`` that is not ciphertext, any
    ``tool_calls``, ``reasoning_encrypted`` or ``thought_signature``, choice
    ``logprobs``, or an undeclared field on a delta, choice, chunk or
    ``usage`` that carries text in a key or a value raises
    :class:`~venice_ai.exceptions.TeeEncryptionError` (the message names the
    field, never the value). The usage chunk's numeric ``cost`` passes
    through, and so does a choice's ``matched_stop`` or ``stop_reason`` when it
    is a token id or one of the request's own *stop* strings. A delta ``role``
    other than ``"assistant"`` or a ``finish_reason`` outside the protocol
    vocabulary also raises. Role-only, finish and usage chunks pass through.
    The SESSION private key must outlive consumption, so the session is closed in
    ``finally`` when the stream is exhausted or aborted. A stream closed before
    its first chunk never enters this body; :class:`_E2EEChunkIterator` closes
    the session in that case.
    """
    try:
        async for chunk in raw_iterator:
            _refuse_unencryptable_chunk_fields(chunk)
            for choice in chunk.choices or []:
                delta = choice.delta
                _decrypt_delta_field(session, delta, "content")
                _decrypt_delta_field(session, delta, "reasoning_content")
                _refuse_unencryptable_delta_fields(choice, stop)
            yield chunk
    finally:
        session.close()


class _E2EEChunkIterator:
    """The decrypted chunk iterator of an E2EE stream, closing the session on ``aclose()``.

    ``aclose()`` on an async generator that has not started does not run its
    ``finally``, so closing a :class:`~venice_ai.streaming.ChatStream` that was
    never iterated would leave the SESSION private key referenced until garbage
    collection. This wrapper closes the decrypting generator, the raw wire
    iterator and the session whether or not iteration began.
    """

    def __init__(
        self,
        raw_iterator: AsyncIterator[ChatCompletionChunk],
        session: "TeeSession",
        *,
        stop: tuple[str, ...] = (),
    ) -> None:
        self._raw = raw_iterator
        self._session = session
        self._chunks = _decrypting_chunks(raw_iterator, session, stop=stop)

    def __aiter__(self) -> "_E2EEChunkIterator":
        return self

    async def __anext__(self) -> ChatCompletionChunk:
        return await self._chunks.__anext__()

    async def aclose(self) -> None:
        """Close the decrypting generator and the raw iterator, then the session."""
        try:
            try:
                await self._chunks.aclose()
            finally:
                raw_aclose = getattr(self._raw, "aclose", None)
                if raw_aclose is not None:
                    await raw_aclose()
        finally:
            self._session.close()


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

    @overload
    async def create(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        stream: bool,
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
    ) -> ChatCompletionResponse | AsyncIterable[ChatCompletionChunk]:  # stream not known statically
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
                invalid value still raises before the request is sent. Only
                the fields set here are sent; the server's defaults apply to
                the rest.
                Note that :class:`VeniceParameters` is ``extra="allow"``,
                so an unrecognized key is carried through rather than
                rejected — the same for either form.
            reasoning_effort: Controls thinking depth on reasoning models.
                One of ``"none"``, ``"minimal"``, ``"low"``, ``"medium"``,
                ``"high"``, ``"xhigh"``, or ``"max"``. Takes precedence
                over ``reasoning.effort`` when both are set.
            reasoning: Nested reasoning configuration. Accepts ``enabled``
                (bool), ``effort`` (same enum as ``reasoning_effort``) and
                ``summary`` (``"auto"`` / ``"concise"`` / ``"detailed"``). A key
                :class:`ReasoningConfig` does not declare is sent as given.
            prompt_cache_key: Routing hint to improve cache hit rates
                across multi-turn conversations. Requests sharing the same
                key are more likely to hit cached prompt prefixes.
            prompt_cache_retention: Cache retention tier. ``"default"``
                uses the standard TTL; ``"extended"`` or ``"24h"`` keep
                the prompt cached for longer, improving hit rates for
                long-running agents at a small storage premium.
            store: Accepted for OpenAI compatibility; Venice does not use it.
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
                ``venice_parameters.enable_e2ee`` validates to ``True`` (so
                ``1`` and ``"true"`` engage it too); it requires an
                ``e2ee-*`` model and the ``[e2ee]`` extra (``cryptography``
                and ``pycryptodome``). ``e2ee`` set together with
                ``venice_parameters.enable_e2ee=False`` is refused.
                ``True`` uses defaults; pass a
                :class:`~venice_ai.tee.types.TeeOptions` to control the
                attestation freshness nonce or supply a full quote verifier.
                When engaged the SDK verifies the model's attestation
                (fail-closed, including that the model key derives to the
                attested signing address), encrypts ``user`` / ``system`` / ``assistant``
                message content and assistant ``reasoning_content`` to the
                model key, forces a wire stream with the ``X-Venice-TEE-*``
                headers, and decrypts response ``content`` and
                ``reasoning_content`` locally (reassembling a normal
                :class:`ChatCompletionResponse` when ``stream`` is ``False``).
                Only message content is encrypted: every other request
                parameter, such as ``stop``, ``response_format`` (including a
                JSON schema), ``prompt_cache_key``, ``user``, ``metadata``,
                ``seed`` and ``temperature``, is sent in plaintext.
                Raised as :class:`~venice_ai.exceptions.InvalidRequestError`
                before any network call: ``developer`` messages (use
                ``system``), ``tool`` messages and any other role, message
                keys other than ``role``, ``content`` and assistant
                ``reasoning_content``, non-text content, ``logprobs`` /
                ``top_logprobs``, tools, and web search/scraping; an invalid
                message shape is reported without quoting its content. The
                Venice system prompt is forced off. Fail closed: the response
                raises :class:`~venice_ai.exceptions.TeeEncryptionError` on
                any text outside the encrypted fields and the protocol
                metadata listed here, such as a non-ciphertext ``content`` or
                ``reasoning_content``, tool calls, ``logprobs``, a delta
                ``role`` other than ``"assistant"``, a ``finish_reason``
                outside the protocol vocabulary (``stop``, ``length``, ...),
                or an unknown field carrying text in a key or a value. The
                protocol metadata that passes through unencrypted is the
                chunk ``id``, ``object``, ``model``, ``created`` and
                ``system_fingerprint``, ``finish_reason``, the ``"assistant"``
                role, ``usage`` with its numeric ``cost``, and
                ``matched_stop`` / ``stop_reason`` when each is a token id or
                one of the request's own ``stop`` strings. Ciphertext length
                reveals plaintext length, for both the request messages and
                each response delta. The returned stream or response carries
                the attestation trust level as ``.e2ee``, a
                :class:`~venice_ai.tee.types.TeeTrust`.
                Consume or close a returned stream (prefer ``async with``):
                the session private key is held until the stream is
                exhausted, closed or garbage-collected.
                SECURITY LIMITATION: the baseline attestation verifier trusts
                Venice's server-side ``verified`` claim and does not perform
                full client-side TDX / NVIDIA quote verification; a
                :class:`UserWarning` is emitted on each engaged call unless a
                verifier is supplied via ``TeeOptions(verifier=...)``.
            kwargs: Additional keyword arguments forwarded to the request
                body for forward-compatibility.

        Returns:
            :class:`~venice_ai.types.api.chat.ChatCompletionResponse` when
            ``stream`` is ``False``, otherwise an ``AsyncIterable`` of
            :class:`~venice_ai.types.api.streaming.ChatCompletionChunk`.

        Raises:
            InvalidRequestError: If parameters are invalid or malformed,
                including a ``venice_parameters.enable_e2ee`` that skipped
                validation and is neither ``False`` nor ``None`` on a request
                without E2EE.
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
            response_format = _schema_response_format(response_format)

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

        e2ee_engaged = _e2ee_engaged(e2ee, venice_parameters)

        # Create Pydantic request model
        chat_request = _build_chat_request(
            e2ee_engaged=e2ee_engaged,
            model=model,
            messages=messages,
            stream=stream,
            api_params=api_params,
        )

        # Convert to API payload
        body = chat_request.model_dump(exclude_none=True)

        # Only the venice_parameters the caller set go on the wire, so the
        # server's defaults apply to the rest.
        body.pop("venice_parameters", None)
        if chat_request.venice_parameters is not None:
            wire_venice_parameters = chat_request.venice_parameters.to_wire()
            if wire_venice_parameters:
                body["venice_parameters"] = wire_venice_parameters

        if not e2ee_engaged:
            _refuse_non_bool_wire_enable_e2ee(body)

        # --- E2EE (Venice confidential-compute) path -------------------------
        # Engaged when e2ee=... is truthy OR venice_parameters.enable_e2ee=True.
        # All guards FAIL LOUD before any network call (open_session does a GET).
        if e2ee_engaged:
            return await self._create_e2ee(
                model=model,
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
        body: dict[str, Any],
        stream: bool,
        e2ee: bool | TeeOptions,
        venice_parameters: Any,
        tools: Any,
    ) -> ChatCompletionResponse | AsyncIterable[ChatCompletionChunk]:
        """Run the real Venice E2EE chat flow.

        Validates fail-closed, opens a verified :class:`TeeSession`, encrypts the
        message text, forces a wire stream with the three
        ``X-Venice-TEE-*`` headers, then either yields decrypted deltas
        (``stream=True``) or reassembles a surface-preserving
        :class:`ChatCompletionResponse` (``stream=False``).
        """
        # (1) FAIL LOUD before any network call.
        _validate_e2ee_request(
            model=model,
            tools=tools,
            venice_parameters=venice_parameters,
            e2ee=e2ee,
            logprobs=body.get("logprobs"),
            top_logprobs=body.get("top_logprobs"),
        )
        _validate_e2ee_messages(body.get("messages", []))

        opts = e2ee if isinstance(e2ee, TeeOptions) else TeeOptions()

        # (2) Attestation-trust limitation warning, emitted on every call that
        # relies on the baseline verifier (no module-global flag) so each such
        # engagement is honest. Python's default filter de-duplicates by
        # attributed location (the caller of create()). A caller-supplied full
        # quote verifier removes the limitation, so nothing is emitted then.
        if opts.verifier is None:
            warnings.warn(_E2EE_TRUST_WARNING, UserWarning, stacklevel=3)

        # (3) Open a verified session (attestation GET + SESSION keypair). Nonce /
        # verifier flow through from TeeOptions; bool uses defaults.
        session = await self._client.tee.open_session(
            model=model,
            nonce=opts.nonce,
            verifier=opts.verifier,
        )

        # (4) Shape the wire body: force stream + include_usage, encrypt content,
        # force the Venice system prompt off. E2EE mode has no Venice system
        # prompt (it would have to be encrypted client-side), so a caller's
        # include_venice_system_prompt=True cannot apply here. enable_e2ee is on
        # the wire only if the caller set it, with the value the request model
        # validated (the body is NOT required for the flow to work).
        # Until the decrypting iterator owns the session, any failure here
        # (an unencodable message, a malformed model key) must drop the
        # session private key before the exception propagates.
        try:
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
            decrypting = _E2EEChunkIterator(
                raw_iterator, session, stop=_request_stop_values(body.get("stop"))
            )
        except BaseException:
            session.close()
            raise

        if stream:
            return ChatStream(decrypting, client=self._client, e2ee=session.trust)
        # stream=False: consume the forced stream and reassemble a normal,
        # surface-preserving ChatCompletionResponse from the decrypted deltas.
        return await ChatStream(decrypting, client=self._client, e2ee=session.trust).collect()

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
                if E2EE is requested on an incompatible request, before any
                network call: ``e2ee`` with ``venice_parameters.enable_e2ee``
                set to ``False``; a non-``e2ee-`` model; ``developer`` (use
                ``system``), ``tool`` or other-role messages; message keys
                other than ``role``, ``content`` and assistant
                ``reasoning_content``; non-text content; ``logprobs`` /
                ``top_logprobs``; tools; or web search/scraping.
            TeeEncryptionError: Under E2EE, if the response carries text
                outside the encrypted fields and the protocol metadata listed
                in :meth:`create`; raised while the stream is consumed.
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
        venice_parameters: VeniceParameters | Mapping[str, Any] | None = None,
        tools: Sequence[Any] | None = None,
        response_format: Any | None = None,
    ) -> ChatCostEstimate:
        """Estimate the USD cost of a chat completion before sending it.

        Mirrors the pre-flight ``quote()`` method on ``client.video`` and
        ``client.audio`` for symmetry. Venice has no tokenize endpoint, so the
        prompt is counted from every billed component of the request with
        allowances sized to be at least what the measured models bill. The
        result is an estimate meant to err high, not a guarantee; a system
        prompt built into a model's own chat template is not included (see
        :class:`~venice_ai.costs.ChatCostEstimate`):

        - message text (and any ``name``): word count x ``tokens_per_word``,
          the approximation :func:`venice_ai.costs.estimate_completion_cost`
          uses;
        - the chat template the model wraps the request in:
          :data:`~venice_ai.costs.CHAT_TEMPLATE_TOKEN_ALLOWANCE` per request
          plus :data:`~venice_ai.costs.CHAT_MESSAGE_TOKEN_ALLOWANCE` per
          message (``template_overhead_tokens``);
        - ``tools`` and ``response_format`` (``schema_tokens``): their compact
          JSON at :data:`~venice_ai.costs.SCHEMA_JSON_CHARS_PER_TOKEN`
          characters per token, plus
          :data:`~venice_ai.costs.TOOLS_TEMPLATE_TOKEN_ALLOWANCE` once when
          tools are sent, for the tool-calling instructions the chat template
          adds. Chat templates differ widely here: measured on five models,
          the same tool cost between 43 and 270 prompt tokens. The allowance
          covers the most expensive template measured with at least 20%
          headroom, so it can overstate a model with a compact template
          several times over. A ``json_schema`` response format is free on
          models that enforce it while decoding and billed on models that put
          it in the prompt; the allowance assumes the latter.

        Unless ``venice_parameters`` sets
        ``include_venice_system_prompt=False`` (or ``enable_e2ee=True``,
        which forces it off), Venice prepends its own system prompt to the
        conversation and bills it as prompt tokens.
        The estimate adds
        :data:`~venice_ai.costs.VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE`
        tokens for it (reported as ``venice_system_prompt_tokens``),
        priced at the model's ``cache_input`` rate, or ``input`` when no
        cache rate is published. The injected prompt's real size varies
        by model, so the allowance deliberately errs high.

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
            venice_parameters: The ``venice_parameters`` you intend to
                send (a :class:`VeniceParameters` or a mapping). Only
                ``include_venice_system_prompt`` and ``enable_e2ee`` affect
                the estimate.
            tools: The ``tools`` you intend to send (dicts or models).
            response_format: The ``response_format`` you intend to send: a
                dict, a model instance, or a Pydantic model class (counted as
                the ``json_schema`` format :meth:`create` sends for it).

        Returns:
            :class:`~venice_ai.costs.ChatCostEstimate` with the prompt /
            completion / total USD breakdown.

        Raises:
            ValueError: If ``model`` is not present in
                :meth:`client.models.list` or has no LLM token-based
                pricing (the estimate would otherwise be meaningless).
            InvalidRequestError: Under E2EE (an ``e2ee-`` model or
                ``enable_e2ee``), for an invalid message, without quoting its
                content.
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
        e2ee_engaged = model.startswith("e2ee-") or _e2ee_engaged(False, venice_parameters)
        coerced = _coerce_messages_for(messages, e2ee_engaged=e2ee_engaged)
        pricing = await self._fetch_chat_pricing(model)

        names = " ".join(str(n) for msg in coerced if (n := getattr(msg, "name", None)))
        prompt_text = f"{_concat_message_text(coerced)} {names}".strip()
        template_tokens = CHAT_TEMPLATE_TOKEN_ALLOWANCE + CHAT_MESSAGE_TOKEN_ALLOWANCE * len(
            coerced
        )
        schema_tokens = _schema_token_allowance(tools, response_format)
        message_tokens = (
            int(len(prompt_text.split()) * tokens_per_word) + template_tokens + schema_tokens
        )
        opted_out = (
            _venice_params_as_dict(venice_parameters).get("include_venice_system_prompt") is False
        )
        # E2EE forces the Venice system prompt off (see ``create``).
        includes_system_prompt = not opted_out and not _e2ee_engaged(False, venice_parameters)
        system_prompt_tokens = VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE if includes_system_prompt else 0

        prompt_cost, completion_cost = _estimate_costs(
            pricing,
            prompt_tokens=message_tokens,
            system_prompt_tokens=system_prompt_tokens,
            completion_tokens=expected_completion_tokens,
        )

        return ChatCostEstimate(
            model=model,
            prompt_tokens=message_tokens + system_prompt_tokens,
            venice_system_prompt_tokens=system_prompt_tokens,
            template_overhead_tokens=template_tokens,
            schema_tokens=schema_tokens,
            expected_completion_tokens=expected_completion_tokens,
            prompt_cost_usd=prompt_cost,
            completion_cost_usd=completion_cost,
            total_cost_usd=prompt_cost + completion_cost,
        )

    async def parse(
        self,
        *,
        model: str,
        messages: Sequence[ChatMessageParam],
        response_format: type[_ModelT],
        schema_name: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> ParsedChatCompletion[_ModelT]:
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

        schema_format = _schema_response_format(response_format, name=schema_name, strict=strict)

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
                validation. Under E2EE (``e2ee`` or ``enable_e2ee``), also
                for an invalid message, before any request and without
                quoting its content.
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

        history: list[_ChatMessageModel] = _coerce_messages_for(
            messages,
            e2ee_engaged=_e2ee_engaged(
                create_kwargs.get("e2ee", False), create_kwargs.get("venice_parameters")
            ),
        )

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
