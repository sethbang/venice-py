"""Venice AI Responses API resource (beta).

Wraps ``POST /responses``, which accepts requests in OpenAI's Responses API
format. The endpoint is currently tagged beta by Venice; request and response
shapes may change without notice.

Unlike ``/chat/completions``, this endpoint returns a typed ``output`` array
containing ``reasoning``, ``message``, ``function_call``,
``custom_tool_call``, ``tool_search_call``, and ``web_search_call`` blocks.
It is stateless — each request is independent
and no conversation state is persisted between calls. E2EE-capable models
are not supported; use ``/chat/completions`` with E2EE headers instead.

Streaming is supported via Server-Sent Events when ``stream=True``; the
returned :class:`~venice_ai.streaming.Stream` yields
:class:`~venice_ai.types.api.responses.ResponsesStreamEvent` chunks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from .._resource import APIResource
from ..streaming import Stream
from ..types.api.requests.common import Tool
from ..types.api.requests.responses import (
    PromptCacheRetention,
    ResponsesInputItem,
    ResponsesRequest,
    ResponsesTool,
    ResponsesToolChoice,
)
from ..types.api.responses import ResponsesResponse, ResponsesStreamEvent

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["Responses"]


class Responses(APIResource["VeniceClient"]):
    """Access the Venice Responses API (beta).

    Access via :attr:`VeniceClient.responses`.
    """

    @overload
    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[ResponsesInputItem],  # noqa: A002 - matches API field name
        instructions: str | None = None,
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        parallel_tool_calls: bool | None = None,
        tools: Sequence[Tool | ResponsesTool] | None = None,
        tool_choice: ResponsesToolChoice | None = None,
        prompt_cache_retention: PromptCacheRetention | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: Literal[False] = False,
    ) -> ResponsesResponse: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[ResponsesInputItem],  # noqa: A002 - matches API field name
        instructions: str | None = None,
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        parallel_tool_calls: bool | None = None,
        tools: Sequence[Tool | ResponsesTool] | None = None,
        tool_choice: ResponsesToolChoice | None = None,
        prompt_cache_retention: PromptCacheRetention | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: Literal[True],
    ) -> AsyncIterable[ResponsesStreamEvent]: ...

    @overload
    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[ResponsesInputItem],  # noqa: A002 - matches API field name
        instructions: str | None = None,
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        parallel_tool_calls: bool | None = None,
        tools: Sequence[Tool | ResponsesTool] | None = None,
        tool_choice: ResponsesToolChoice | None = None,
        prompt_cache_retention: PromptCacheRetention | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: bool,
    ) -> ResponsesResponse | AsyncIterable[ResponsesStreamEvent]: ...

    async def create(
        self,
        *,
        model: str,
        input: str | Sequence[ResponsesInputItem],  # noqa: A002 - matches API field name
        instructions: str | None = None,
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        parallel_tool_calls: bool | None = None,
        tools: Sequence[Tool | ResponsesTool] | None = None,
        tool_choice: ResponsesToolChoice | None = None,
        prompt_cache_retention: PromptCacheRetention | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: bool = False,
    ) -> ResponsesResponse | AsyncIterable[ResponsesStreamEvent]:
        """Create a response using the Responses API (beta).

        Wraps ``POST /api/v1/responses``. Each call is stateless - no
        conversation history is persisted between requests.

        Args:
            model: Model ID. E2EE-capable models are not supported; use
                ``/chat/completions`` with E2EE headers instead.
            input: Prompt - either a plain string or a list of structured
                input items (messages, reasoning items, tool calls and their
                outputs) as documented in the OpenAI Responses API. Items can
                be dicts or the typed ``ResponsesInput*`` models
                (``custom_tool_call``, ``custom_tool_call_output``,
                ``additional_tools``, ``tool_search_call``,
                ``tool_search_output``).
            instructions: System instructions, applied before ``input``.
            include: Additional response fields to include.
            max_output_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling (0-1).
            fallbacks: Anthropic beta parameter for Claude Fable 5 server-side
                refusal fallback. Array of ``{"model": ...}`` objects (max 10).
                Forwarded only for direct Anthropic routes; ignored otherwise.
            reasoning: Nested reasoning config (``{"effort": "...",
                "summary": "..."}`` or ``ReasoningConfig``).
            parallel_tool_calls: Whether the model may call more than one
                tool in a single turn.
            tools: Tool definitions: function tools (``Tool`` or
                ``ResponsesFunctionTool``, which adds ``defer_loading``), a
                client-executed ``ResponsesToolSearchTool``, or dicts for
                other types (``web_search``, ``x_search``,
                ``code_interpreter``, ``file_search``,
                ``computer_use_preview``).
            tool_choice: ``"auto" | "none" | "required"``, a named tool
                ``{"type": "function" | "custom", "name": ..., "namespace": ...}``,
                or ``{"type": "function", "function": {"name": ...}}``.
            prompt_cache_retention: Prompt cache retention for supported
                OpenAI models: ``"in_memory"``, ``"24h"``, ``"default"``, or
                ``"extended"``. ``"24h"`` and ``"extended"`` keep cached
                prefixes for up to 24 hours.
            web_search: Enable web search for this request.
            venice_parameters: Venice-specific request parameters.
            stream: When ``True``, returns an async iterator of
                :class:`ResponsesStreamEvent` chunks parsed from
                Server-Sent Events. Default ``False`` returns a single
                :class:`ResponsesResponse`.

        Returns:
            :class:`ResponsesResponse` with typed ``output`` blocks
            (reasoning, message, function_call, custom_tool_call,
            tool_search_call, web_search_call), or an
            ``AsyncIterable[ResponsesStreamEvent]`` when ``stream=True``.

        Raises:
            InvalidRequestError: If parameters fail server-side validation
                (e.g. malformed ``input``, unsupported tool type, or an
                E2EE-capable model is supplied).
            AuthenticationError: If the API key is missing or invalid.
            PermissionDeniedError: If the account lacks access to the
                Responses API or the requested model.
            NotFoundError: If the model id is unknown.
            RateLimitError: If account-level rate limits are exceeded.
            APIError: For other HTTP-level failures.

        Example:

            .. code-block:: python

                from venice_ai import VeniceClient

                async with VeniceClient() as client:
                    model = await client.models.resolve_chat()
                    response = await client.responses.create(
                        model=model,
                        input="Summarize the Treaty of Versailles in two sentences.",
                        max_output_tokens=200,
                    )
                    for block in response.output:
                        if block.type == "message":
                            print(block.content)
        """
        request = ResponsesRequest(
            model=model,
            input=input if isinstance(input, str) else list(input),
            instructions=instructions,
            include=include,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            fallbacks=fallbacks,
            reasoning=reasoning,
            parallel_tool_calls=parallel_tool_calls,
            tools=list(tools) if tools is not None else None,
            tool_choice=tool_choice,
            prompt_cache_retention=prompt_cache_retention,
            web_search=web_search,
            venice_parameters=venice_parameters,
            anon_user_id=anon_user_id,
        )
        body = request.model_dump(exclude_none=True)
        # /responses documents only a subset of venice_parameters; the SDK reuses
        # the shared chat VeniceParameters model, so drop the chat-only keys the
        # Responses endpoint does not document (swagger ResponsesRequest).
        _vp = body.get("venice_parameters")
        if isinstance(_vp, dict):
            for _k in (
                "strip_thinking_response",
                "disable_thinking",
                "return_search_results_as_documents",
                "enable_x_search",
            ):
                _vp.pop(_k, None)

        if stream:
            body["stream"] = True
            raw_iterator = self._client._stream_request(
                method="POST",
                path="responses",
                json_data=body,
                cast_to=ResponsesStreamEvent,
            )
            return cast(
                AsyncIterable[ResponsesStreamEvent],
                Stream(raw_iterator, client=self._client),
            )

        return await self._client.post("responses", json_data=body, cast_to=ResponsesResponse)
