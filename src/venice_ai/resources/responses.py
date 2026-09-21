"""Venice AI Responses API resource (Alpha).

Wraps ``POST /responses`` — the OpenAI-compatible Responses API endpoint
described at ``swagger.yaml.md:7244``. The endpoint is currently tagged
Alpha by Venice; request and response shapes may change without notice.

Unlike ``/chat/completions``, this endpoint returns a typed ``output`` array
containing ``reasoning``, ``message``, ``function_call``, and
``web_search_call`` blocks. It is stateless — each request is independent
and no conversation state is persisted between calls. E2EE-capable models
are not supported; use ``/chat/completions`` with E2EE headers instead.

Streaming is supported via Server-Sent Events when ``stream=True``; the
returned :class:`~venice_ai.streaming.Stream` yields
:class:`~venice_ai.types.api.responses.ResponsesStreamEvent` chunks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterable
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from .._resource import APIResource
from ..streaming import Stream
from ..types.api.requests.common import Tool
from ..types.api.requests.responses import ResponsesRequest
from ..types.api.responses import ResponsesResponse, ResponsesStreamEvent

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = ["Responses"]


class Responses(APIResource["VeniceClient"]):
    """Access the Venice Responses API (Alpha).

    Access via :attr:`VeniceClient.responses`.
    """

    @overload
    async def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],  # noqa: A002 - matches API field name
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
        input: str | list[dict[str, Any]],  # noqa: A002 - matches API field name
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: Literal[True],
    ) -> AsyncIterable[ResponsesStreamEvent]: ...

    async def create(
        self,
        *,
        model: str,
        input: str | list[dict[str, Any]],  # noqa: A002 - matches API field name
        include: list[str] | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        fallbacks: list[dict[str, str]] | None = None,
        reasoning: Any | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        web_search: bool | None = None,
        venice_parameters: Any | None = None,
        anon_user_id: str | None = None,
        stream: bool = False,
    ) -> ResponsesResponse | AsyncIterable[ResponsesStreamEvent]:
        """Create a response using the Responses API (Alpha).

        Wraps ``POST /api/v1/responses``. Each call is stateless - no
        conversation history is persisted between requests.

        Args:
            model: Model ID. E2EE-capable models are not supported; use
                ``/chat/completions`` with E2EE headers instead.
            input: Prompt - either a plain string or a list of structured
                input items (messages, reasoning blocks, function calls,
                etc.) as documented in the OpenAI Responses API.
            include: Additional response fields to include.
            max_output_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0-2).
            top_p: Nucleus sampling (0-1).
            fallbacks: Anthropic beta parameter for Claude Fable 5 server-side
                refusal fallback. Array of ``{"model": ...}`` objects (max 10).
                Forwarded only for direct Anthropic routes; ignored otherwise.
            reasoning: Nested reasoning config (``{"effort": "...",
                "summary": "..."}`` or ``ReasoningConfig``).
            tools: Tool definitions. Function tools plus the Alpha tool
                types (``web_search``, ``x_search``, ``code_interpreter``,
                ``file_search``, ``computer_use_preview``) are supported.
            tool_choice: ``"auto" | "none" | "required"`` or a
                ``{"type": "function", "function": {"name": ...}}`` dict.
            web_search: Enable web search for this request.
            venice_parameters: Venice-specific request parameters.
            stream: When ``True``, returns an async iterator of
                :class:`ResponsesStreamEvent` chunks parsed from
                Server-Sent Events. Default ``False`` returns a single
                :class:`ResponsesResponse`.

        Returns:
            :class:`ResponsesResponse` with typed ``output`` blocks
            (reasoning, message, function_call, web_search_call), or an
            ``AsyncIterable[ResponsesStreamEvent]`` when ``stream=True``.

        Raises:
            InvalidRequestError: If parameters fail server-side validation
                (e.g. malformed ``input``, unsupported tool type, or an
                E2EE-capable model is supplied).
            AuthenticationError: If the API key is missing or invalid.
            PermissionDeniedError: If the account lacks access to the
                Responses API alpha or the requested model.
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
            input=input,
            include=include,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            fallbacks=fallbacks,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
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
