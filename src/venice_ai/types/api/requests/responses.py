"""Request model for ``POST /responses`` (beta).

The Responses API accepts OpenAI's Responses format and is tagged beta in the
Venice docs; the request surface may shift without notice. The model below
types the fields the Venice spec defines and keeps everything else permissive
(``extra="allow"``) so new fields on the server don't break callers.

Input items, tools, and tool choices accept either the typed models defined
here or plain dicts. Dicts are sent as written, so any item shape the server
accepts can still be passed.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import ANON_USER_ID_DESCRIPTION, AnonUserId, ReasoningConfig, Tool, VeniceParameters

PromptCacheRetention = Literal["in_memory", "24h", "default", "extended"]
"""Prompt-cache retention for supported OpenAI models.

``"24h"`` and ``"extended"`` keep cached prefixes for up to 24 hours.
"""

ResponsesItemStatus = Literal["completed", "in_progress", "incomplete"]
"""Status of a tool-call item replayed in ``input``."""


# ============================================================================
# Tool definitions
# ============================================================================


class ResponsesFunctionToolDefinition(BaseModel):
    """The ``function`` object of a :class:`ResponsesFunctionTool`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Function name.")
    description: str | None = Field(default=None, description="What the function does.")
    parameters: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for the function arguments."
    )
    strict: bool | None = Field(default=None, description="Enforce the schema strictly.")
    defer_loading: bool | None = Field(
        default=None,
        description=(
            "Leave the function out of the initial tool list; the model finds it "
            "through a ``tool_search`` tool."
        ),
    )


class ResponsesFunctionTool(BaseModel):
    """A function tool with optional deferred loading.

    The shared :class:`~venice_ai.types.api.requests.common.Tool` model still
    works for plain function tools; use this one when you need
    ``defer_loading``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: ResponsesFunctionToolDefinition
    defer_loading: bool | None = Field(
        default=None,
        description=(
            "Leave the tool out of the initial tool list; the model finds it "
            "through a ``tool_search`` tool."
        ),
    )


class ResponsesToolSearchTool(BaseModel):
    """A client-executed ``tool_search`` tool.

    The model emits a ``tool_search_call`` output item; your application runs
    the search and sends the matching tools back in a ``tool_search_output``
    input item (see :class:`ResponsesInputToolSearchOutput`).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_search"]
    execution: Literal["client"] = Field(
        ..., description="Where the search runs. Only ``'client'`` is supported."
    )
    parameters: dict[str, Any] = Field(
        ...,
        description='JSON Schema for the search arguments, e.g. ``{"type": "object"}``.',
    )
    description: str | None = Field(default=None, description="What the search tool does.")


ResponsesTool = Annotated[
    dict[str, Any] | ResponsesFunctionTool | ResponsesToolSearchTool,
    Field(union_mode="left_to_right"),
]
"""A tool definition: a typed model or a raw dict for any other tool type
(``web_search``, ``x_search``, ``code_interpreter``, ``file_search``,
``computer_use_preview``, or a generic ``{"type": ...}`` tool)."""


# ============================================================================
# Tool choice
# ============================================================================


class ResponsesNamedToolChoice(BaseModel):
    """Force a specific tool, in OpenAI's Responses shape.

    ``{"type": "function", "name": "get_weather"}`` or
    ``{"type": "custom", "name": "apply_patch", "namespace": "editor"}``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["function", "custom"]
    name: str = Field(..., description="Name of the tool to call.")
    namespace: str | None = Field(default=None, description="Namespace the tool belongs to.")


class ResponsesToolChoiceFunctionName(BaseModel):
    """The ``function`` object of a :class:`ResponsesFunctionToolChoice`."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Name of the function to call.")


class ResponsesFunctionToolChoice(BaseModel):
    """Force a specific function, in the Chat Completions shape
    ``{"type": "function", "function": {"name": "..."}}``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: ResponsesToolChoiceFunctionName


ResponsesToolChoice = Annotated[
    Literal["auto", "none", "required"]
    | dict[str, Any]
    | ResponsesNamedToolChoice
    | ResponsesFunctionToolChoice,
    Field(union_mode="left_to_right"),
]
"""Which tool the model calls: a mode string, a named tool, or a raw dict."""


# ============================================================================
# Input items
# ============================================================================


class ResponsesInputCustomToolCall(BaseModel):
    """A ``custom_tool_call`` item replayed in ``input``.

    Send back the ``custom_tool_call`` output item the model returned, followed
    by a :class:`ResponsesInputCustomToolCallOutput` with your result.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom_tool_call"]
    call_id: str = Field(..., description="ID linking the call to its output.")
    name: str = Field(..., description="Name of the custom tool.")
    input: str = Field(..., description="Raw input the model produced for the tool.")
    id: str | None = Field(default=None, description="Item ID from the original output.")
    namespace: str | None = Field(default=None, description="Namespace the tool belongs to.")
    status: ResponsesItemStatus | None = Field(default=None, description="Item status.")


class ResponsesInputCustomToolCallOutput(BaseModel):
    """The result of a custom tool call, sent in ``input``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom_tool_call_output"]
    call_id: str = Field(..., description="``call_id`` of the custom tool call.")
    output: str | bool | int | float | list[Any] | dict[str, Any] = Field(
        ..., description="Tool result: a string, list, object, number, or boolean."
    )


class ResponsesInputAdditionalTools(BaseModel):
    """Load more tools partway through a conversation.

    The tools become available to the model from this point in ``input``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["additional_tools"]
    tools: list[ResponsesTool] = Field(..., description="Tool definitions to add.")
    role: Literal["developer"] | None = Field(default=None, description="Item role.")
    id: str | None = Field(default=None, description="Item ID.")


class ResponsesInputToolSearchCall(BaseModel):
    """A ``tool_search_call`` item replayed in ``input``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_search_call"]
    call_id: str = Field(..., min_length=1, description="ID linking the call to its output.")
    execution: Literal["client"] = Field(..., description="Where the search runs.")
    arguments: dict[str, Any] = Field(..., description="Search arguments the model produced.")
    id: str | None = Field(default=None, description="Item ID from the original output.")
    status: ResponsesItemStatus | None = Field(default=None, description="Item status.")


class ResponsesInputToolSearchOutput(BaseModel):
    """The tools a client-executed ``tool_search`` found, sent in ``input``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["tool_search_output"]
    call_id: str = Field(..., min_length=1, description="``call_id`` of the tool search call.")
    execution: Literal["client"] = Field(..., description="Where the search ran.")
    tools: list[ResponsesTool] = Field(..., description="Tool definitions the search returned.")
    id: str | None = Field(default=None, description="Item ID.")
    status: Literal["completed"] | None = Field(default=None, description="Item status.")


ResponsesInputItem = Annotated[
    dict[str, Any]
    | ResponsesInputCustomToolCall
    | ResponsesInputCustomToolCallOutput
    | ResponsesInputAdditionalTools
    | ResponsesInputToolSearchCall
    | ResponsesInputToolSearchOutput,
    Field(union_mode="left_to_right"),
]
"""One entry of a list ``input``: a typed item or a raw dict (messages,
reasoning items, function calls and their outputs, item references)."""


# ============================================================================
# Request
# ============================================================================


class ResponsesRequest(BaseModel):
    """Body for ``POST /responses``.

    Mirrors ``ResponsesRequest`` in the Venice OpenAPI spec. ``input`` is
    either a plain string prompt or a list of message / reasoning / tool-call
    items matching the OpenAI Responses API shape.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ..., description="Model ID to use. E2EE-capable models are not supported here."
    )
    input: str | list[ResponsesInputItem] = Field(
        ...,
        description=(
            "Prompt for the model. Either a plain string or a list of structured "
            "input items (messages, reasoning items, tool calls and their outputs, etc.)."
        ),
    )
    instructions: str | None = Field(
        default=None, description="System instructions for the model, applied before ``input``."
    )
    include: list[str] | None = Field(
        default=None, description="Additional response fields to include (OpenAI-compatible)."
    )
    max_output_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate."
    )
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature (0–2)."
    )
    top_p: float | None = Field(default=None, ge=0, le=1, description="Nucleus sampling (0–1).")
    fallbacks: list[dict[str, str]] | None = Field(
        default=None,
        max_length=10,
        description=(
            "Anthropic beta parameter for Claude Fable 5 server-side refusal "
            "fallback. Array of fallback model objects (each with a 'model' key), "
            "max 10. Forwarded only for direct Anthropic routes; ignored for "
            "other providers."
        ),
    )
    reasoning: ReasoningConfig | None = Field(
        default=None, description="Nested reasoning configuration (effort + summary)."
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether the model may call more than one tool in a single turn.",
    )
    tools: list[Annotated[Tool | ResponsesTool, Field(union_mode="left_to_right")]] | None = Field(
        default=None,
        description=(
            "Tool definitions. Accepts the shared ``Tool`` model, "
            "``ResponsesFunctionTool``, ``ResponsesToolSearchTool``, or raw dicts "
            "for other tool types (web_search, x_search, code_interpreter, "
            "file_search, computer_use_preview)."
        ),
    )
    tool_choice: ResponsesToolChoice | None = Field(
        default=None,
        description=(
            "Controls which tool is called: ``'auto' | 'none' | 'required'``, "
            '``{"type": "function" | "custom", "name": ..., "namespace": ...}``, '
            'or ``{"type": "function", "function": {"name": ...}}``.'
        ),
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None,
        description=(
            "Prompt cache retention for supported OpenAI models. ``'24h'`` or "
            "``'extended'`` keeps cached prefixes for up to 24 hours."
        ),
    )
    web_search: bool | None = Field(default=None, description="Enable web search for this request.")
    stream: bool | None = Field(
        default=None, description="Whether to stream partial progress via SSE."
    )
    venice_parameters: VeniceParameters | None = Field(
        default=None, description="Venice-specific request parameters."
    )

    anon_user_id: AnonUserId | None = Field(
        default=None,
        description=ANON_USER_ID_DESCRIPTION,
    )


__all__ = [
    "PromptCacheRetention",
    "ResponsesFunctionTool",
    "ResponsesFunctionToolChoice",
    "ResponsesFunctionToolDefinition",
    "ResponsesInputAdditionalTools",
    "ResponsesInputCustomToolCall",
    "ResponsesInputCustomToolCallOutput",
    "ResponsesInputItem",
    "ResponsesInputToolSearchCall",
    "ResponsesInputToolSearchOutput",
    "ResponsesItemStatus",
    "ResponsesNamedToolChoice",
    "ResponsesRequest",
    "ResponsesTool",
    "ResponsesToolChoice",
    "ResponsesToolChoiceFunctionName",
    "ResponsesToolSearchTool",
]
