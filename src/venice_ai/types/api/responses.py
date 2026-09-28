"""Response models for ``POST /responses`` (Alpha).

The Venice Responses API is OpenAI-compatible and currently tagged Alpha,
so the output shape is expected to evolve. Models use ``extra="allow"`` to
survive server-side additions; callers that want exhaustive coverage should
read the OpenAPI spec.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

KNOWN_RESPONSE_STATUSES: Final[tuple[str, ...]] = (
    "completed",
    "failed",
    "in_progress",
    "cancelled",
    "incomplete",
)
""":attr:`ResponsesResponse.status` values this SDK release knows about.

The field is a plain ``str`` so a status the server adds later is preserved
rather than failing the whole parse. ``"incomplete"`` means generation stopped
early (see :attr:`ResponsesResponse.incomplete_details`); the partial output
and usage are still returned.
"""

KNOWN_RESPONSE_MESSAGE_STATUSES: Final[tuple[str, ...]] = (
    "completed",
    "in_progress",
    "failed",
    "incomplete",
)
""":attr:`ResponsesMessageOutput.status` values this SDK release knows about."""

KNOWN_RESPONSE_FUNCTION_CALL_STATUSES: Final[tuple[str, ...]] = (
    "completed",
    "in_progress",
    "incomplete",
)
""":attr:`ResponsesFunctionCallOutput.status` values this SDK release knows about."""

KNOWN_RESPONSE_WEB_SEARCH_CALL_STATUSES: Final[tuple[str, ...]] = ("completed",)
""":attr:`ResponsesWebSearchCallOutput.status` values this SDK release knows about."""

KNOWN_RESPONSE_INCOMPLETE_REASONS: Final[tuple[str, ...]] = (
    "max_output_tokens",
    "content_filter",
)
""":attr:`ResponsesIncompleteDetails.reason` values this SDK release knows about."""


class ResponsesUsageInputDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    cached_tokens: int | None = None


class ResponsesUsageOutputDetails(BaseModel):
    model_config = ConfigDict(extra="allow")

    reasoning_tokens: int | None = None


class ResponsesUsage(BaseModel):
    """Token usage block on a Responses API result."""

    model_config = ConfigDict(extra="allow")

    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: ResponsesUsageInputDetails | None = None
    output_tokens_details: ResponsesUsageOutputDetails | None = None


class ResponsesError(BaseModel):
    """Error block populated when ``status='failed'``."""

    model_config = ConfigDict(extra="allow")

    code: str
    message: str


class ResponsesIncompleteDetails(BaseModel):
    """Why a response ended with ``status='incomplete'``."""

    model_config = ConfigDict(extra="allow")

    reason: str = Field(
        ...,
        description=(
            "Why generation stopped early, e.g. ``'max_output_tokens'`` or "
            "``'content_filter'``. Known values are in KNOWN_RESPONSE_INCOMPLETE_REASONS."
        ),
    )


class ResponsesReasoningOutput(BaseModel):
    """``type='reasoning'`` block in the response ``output`` array."""

    model_config = ConfigDict(extra="allow")

    type: Literal["reasoning"]
    id: str
    summary: list[str] | None = None
    encrypted_content: str | None = None


class ResponsesOutputText(BaseModel):
    """``type='output_text'`` content block inside a ``message`` output."""

    model_config = ConfigDict(extra="allow")

    type: Literal["output_text"]
    text: str
    annotations: list[dict[str, Any]] | None = None


class ResponsesMessageOutput(BaseModel):
    """``type='message'`` block in the response ``output`` array."""

    model_config = ConfigDict(extra="allow")

    type: Literal["message"]
    id: str
    status: str = Field(
        ...,
        description=(
            "Block status. ``'incomplete'`` marks a block cut short; its partial "
            "content is kept. Known values are in KNOWN_RESPONSE_MESSAGE_STATUSES."
        ),
    )
    role: Literal["assistant"]
    content: list[ResponsesOutputText]


class ResponsesFunctionCallOutput(BaseModel):
    """``type='function_call'`` block."""

    model_config = ConfigDict(extra="allow")

    type: Literal["function_call"]
    id: str
    call_id: str
    name: str
    arguments: str
    status: str = Field(
        ...,
        description=(
            "Call status. ``'incomplete'`` means ``arguments`` may be truncated JSON. "
            "Known values are in KNOWN_RESPONSE_FUNCTION_CALL_STATUSES."
        ),
    )


class ResponsesWebSearchCallOutput(BaseModel):
    """``type='web_search_call'`` block."""

    model_config = ConfigDict(extra="allow")

    type: Literal["web_search_call"]
    id: str
    status: str = Field(
        ...,
        description="Search call status. Known values are in KNOWN_RESPONSE_WEB_SEARCH_CALL_STATUSES.",
    )


class ResponsesUnknownOutput(BaseModel):
    """Forward-compat fallback for an output block whose ``type`` the SDK does
    not yet model. Keeps unknown blocks (and their fields) instead of failing
    the whole /responses parse. Appended LAST so known ``Literal`` types still
    win the union match."""

    model_config = ConfigDict(extra="allow")

    type: str


# Union over the output block ``type`` field so each entry in ``output``
# deserialises into the right model; the open ``ResponsesUnknownOutput`` catches
# any future/unmodeled type.
ResponsesOutputItem = (
    ResponsesReasoningOutput
    | ResponsesMessageOutput
    | ResponsesFunctionCallOutput
    | ResponsesWebSearchCallOutput
    | ResponsesUnknownOutput
)


class ResponsesResponse(BaseModel):
    """Body of a 200 response from ``POST /responses``."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["response"]
    created_at: int
    model: str
    status: str = Field(
        ...,
        description=(
            "Response status. ``'incomplete'`` means generation stopped early; "
            "``output`` and ``usage`` still carry the partial result. Known values "
            "are in KNOWN_RESPONSE_STATUSES."
        ),
    )
    incomplete_details: ResponsesIncompleteDetails | None = Field(
        default=None,
        description="Why generation ended early; set when ``status='incomplete'``.",
    )
    output: list[ResponsesOutputItem] = Field(
        default_factory=list,
        description="Output items generated by the model (reasoning, messages, tool calls, etc.).",
    )
    usage: ResponsesUsage | None = None
    error: ResponsesError | None = None


class ResponsesStreamEvent(BaseModel):
    """A single Server-Sent Events chunk from streaming ``POST /responses``.

    Mirrors the OpenAI Responses streaming spec — many discrete event types
    (``response.created``, ``response.output_text.delta``,
    ``response.completed``, etc.) all share a common envelope. Because the
    spec is rich and Venice-side Alpha, this model accepts any extra fields
    rather than enumerating every event variant.

    The ``type`` field identifies the event; payload fields like ``delta``,
    ``response``, ``item``, or ``output_index`` are exposed via attribute
    access (Pydantic ``extra='allow'``).
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(
        ..., description="Event type identifier (e.g. ``response.output_text.delta``)."
    )
    sequence_number: int | None = Field(
        default=None, description="Monotonic sequence number assigned by the server."
    )


__all__ = [
    "KNOWN_RESPONSE_FUNCTION_CALL_STATUSES",
    "KNOWN_RESPONSE_INCOMPLETE_REASONS",
    "KNOWN_RESPONSE_MESSAGE_STATUSES",
    "KNOWN_RESPONSE_STATUSES",
    "KNOWN_RESPONSE_WEB_SEARCH_CALL_STATUSES",
    "ResponsesError",
    "ResponsesFunctionCallOutput",
    "ResponsesIncompleteDetails",
    "ResponsesMessageOutput",
    "ResponsesOutputItem",
    "ResponsesOutputText",
    "ResponsesReasoningOutput",
    "ResponsesResponse",
    "ResponsesStreamEvent",
    "ResponsesUsage",
    "ResponsesUsageInputDetails",
    "ResponsesUnknownOutput",
    "ResponsesUsageOutputDetails",
    "ResponsesWebSearchCallOutput",
]
