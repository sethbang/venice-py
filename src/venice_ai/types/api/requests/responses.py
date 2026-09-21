"""Request model for ``POST /responses`` (Alpha).

The Responses API is OpenAI-compatible and currently tagged Alpha in the
Venice docs; the request surface may shift without notice. The model below
keeps required fields typed and everything else permissive
(``extra="allow"``) so new fields on the server don't break callers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .common import ANON_USER_ID_DESCRIPTION, AnonUserId, ReasoningConfig, Tool, VeniceParameters


class ResponsesRequest(BaseModel):
    """Body for ``POST /responses``.

    Mirrors ``ResponsesRequest`` in the Venice OpenAPI spec. ``input`` is
    either a plain string prompt or a list of message / reasoning / tool-call
    blocks matching the OpenAI Responses API shape.
    """

    model_config = ConfigDict(extra="allow")

    model: str = Field(
        ..., description="Model ID to use. E2EE-capable models are not supported here."
    )
    input: str | list[dict[str, Any]] = Field(
        ...,
        description=(
            "Prompt for the model. Either a plain string or a list of structured "
            "input items (messages, reasoning blocks, function calls, etc.)."
        ),
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
    tools: list[Tool | dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool definitions. Accepts the shared ``Tool`` model (function tools) or "
            "raw dicts for Alpha tool types (web_search, x_search, code_interpreter, "
            "file_search, computer_use_preview)."
        ),
    )
    tool_choice: str | dict[str, Any] | None = Field(
        default=None,
        description="Controls which tool is called: ``'auto' | 'none' | 'required'`` or a ``{type, function}`` dict.",
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


__all__ = ["ResponsesRequest"]
