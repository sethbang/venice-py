"""Unit tests for chat.completions.run_with_tools — automatic tool-loop orchestration."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai import (
    Conversation,
    MaxIterationsExceededError,
    ToolLoopResult,
)
from venice_ai.resources.chat.completions import ChatCompletions
from venice_ai.types.api import UserMessage
from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
    ToolCall,
    ToolCallFunction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_terminal_response(content: str = "Final answer.") -> ChatCompletionResponse:
    """Build a non-tool-call (final) response."""
    return ChatCompletionResponse(
        id="resp-terminal",
        object="chat.completion",
        created=1000000,
        model="fake-test-model",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content),
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None
        ),
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )


def _make_tool_call_response(
    *,
    calls: list[tuple[str, str, str]],
    response_id: str = "resp-tools",
) -> ChatCompletionResponse:
    """Build a response whose finish_reason is ``"tool_calls"``.

    *calls* is a list of ``(call_id, fn_name, arguments_json)`` triples.
    """
    return ChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=1000000,
        model="fake-test-model",
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            type="function",
                            function=ToolCallFunction(name=name, arguments=args),
                        )
                        for call_id, name, args in calls
                    ],
                ),
                finish_reason="tool_calls",
                stop_reason=None,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=10, completion_tokens=5, total_tokens=15, prompt_tokens_details=None
        ),
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )


def _build_chat(*, side_effect: list[Any]) -> tuple[ChatCompletions, AsyncMock]:
    """Build a ChatCompletions whose create() returns the given sequence."""
    chat = ChatCompletions.__new__(ChatCompletions)
    chat._client = MagicMock()  # type: ignore[attr-defined]
    create_mock = AsyncMock(side_effect=side_effect)
    chat.create = create_mock  # type: ignore[method-assign]
    return chat, create_mock


def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Get the current weather for a specific location."""
    return f"sunny in {location} ({unit})"


def calculate(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestRunWithToolsHappyPath:
    @pytest.mark.asyncio
    async def test_no_tool_call_returns_immediately(self):
        """Model gives a final answer on the first turn, no tools dispatched."""
        chat, create_mock = _build_chat(side_effect=[_make_terminal_response("Hi!")])

        result = await chat.run_with_tools(
            model="fake-test-model",
            messages=[UserMessage(content="Hello")],
            tools=[get_weather],
        )
        assert isinstance(result, ToolLoopResult)
        assert result.iterations == 1
        assert result.text == "Hi!"
        # one round trip: no tools were called
        assert create_mock.await_count == 1

    @pytest.mark.asyncio
    async def test_single_tool_call_then_final(self):
        """Tool-call → tool-result → final answer in two round trips."""
        tool_response = _make_tool_call_response(
            calls=[("call_1", "get_weather", '{"location": "Paris"}')],
        )
        chat, create_mock = _build_chat(
            side_effect=[tool_response, _make_terminal_response("Paris is sunny.")]
        )

        result = await chat.run_with_tools(
            model="fake-test-model",
            messages=[UserMessage(content="Weather in Paris?")],
            tools=[get_weather],
        )
        assert result.iterations == 2
        assert result.text == "Paris is sunny."
        assert create_mock.await_count == 2

        # History should include: user, asst-with-tool-calls, tool-result, asst-final
        assert len(result.messages) == 4
        # The tool-result message should carry our handler's return value
        tool_result_msg = result.messages[2]
        assert tool_result_msg.tool_call_id == "call_1"
        assert tool_result_msg.content == "sunny in Paris (fahrenheit)"

    @pytest.mark.asyncio
    async def test_multiple_round_trips(self):
        """Two tool calls in sequence (one tool per turn), then final answer."""
        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c1", "get_weather", '{"location": "NYC"}')]),
                _make_tool_call_response(calls=[("c2", "calculate", '{"a": 7, "b": 11}')]),
                _make_terminal_response("All done."),
            ]
        )
        result = await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="multi-step task")],
            tools=[get_weather, calculate],
        )
        assert result.iterations == 3
        # Two intermediate round trips × 2 messages each + initial user + terminal asst.
        assert len(result.messages) == 1 + 4 + 1


# ---------------------------------------------------------------------------
# Sync vs async tool callables
# ---------------------------------------------------------------------------


class TestSyncAsyncCallables:
    @pytest.mark.asyncio
    async def test_sync_callable_invoked_directly(self):
        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c", "get_weather", '{"location": "NY"}')]),
                _make_terminal_response("done"),
            ]
        )
        result = await chat.run_with_tools(
            model="m", messages=[UserMessage(content="hi")], tools=[get_weather]
        )
        # Sync handler returned directly; tool-result message present.
        assert result.messages[2].content.startswith("sunny in NY")

    @pytest.mark.asyncio
    async def test_async_callable_awaited(self):
        async def fetch_data(url: str) -> str:
            """Fetch data from a URL (async)."""
            return f"fetched {url}"

        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(
                    calls=[("c", "fetch_data", '{"url": "https://example.com"}')]
                ),
                _make_terminal_response("done"),
            ]
        )
        result = await chat.run_with_tools(
            model="m", messages=[UserMessage(content="fetch")], tools=[fetch_data]
        )
        assert result.messages[2].content == "fetched https://example.com"


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallel:
    @pytest.mark.asyncio
    async def test_sequential_runs_one_at_a_time(self):
        """Default (parallel=False): tools run in order."""
        order: list[str] = []

        async def slow_a(x: int) -> int:
            """Slow tool A."""
            order.append(f"a-start-{x}")
            order.append(f"a-end-{x}")
            return x

        async def slow_b(x: int) -> int:
            """Slow tool B."""
            order.append(f"b-start-{x}")
            order.append(f"b-end-{x}")
            return x

        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(
                    calls=[
                        ("c1", "slow_a", '{"x": 1}'),
                        ("c2", "slow_b", '{"x": 2}'),
                    ]
                ),
                _make_terminal_response("done"),
            ]
        )
        await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="hi")],
            tools=[slow_a, slow_b],
            parallel=False,
        )
        # Strict serialization: a runs to completion before b starts.
        assert order == ["a-start-1", "a-end-1", "b-start-2", "b-end-2"]

    @pytest.mark.asyncio
    async def test_parallel_runs_concurrently(self):
        """parallel=True: both tools start before either finishes."""
        import asyncio

        active = 0
        max_active = 0

        async def gated(_x: int) -> int:
            """Tool that records concurrency level."""
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)  # yield control
            active -= 1
            return _x

        # Same handler used for both calls — different call IDs.
        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(
                    calls=[
                        ("c1", "gated", '{"_x": 1}'),
                        ("c2", "gated", '{"_x": 2}'),
                    ]
                ),
                _make_terminal_response("done"),
            ]
        )
        await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="hi")],
            tools=[gated],
            parallel=True,
        )
        assert max_active == 2  # both ran concurrently


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_default_on_tool_error_formats_as_message(self):
        """A tool that raises becomes an error string fed back to the model."""

        def broken(x: int) -> int:
            """A tool that raises."""
            raise ValueError("nope")

        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c", "broken", '{"x": 1}')]),
                _make_terminal_response("recovered"),
            ]
        )
        result = await chat.run_with_tools(
            model="m", messages=[UserMessage(content="hi")], tools=[broken]
        )
        # Error converted to a tool-message string; loop continued.
        assert result.iterations == 2
        tool_msg = result.messages[2]
        assert "Error calling broken" in tool_msg.content
        assert "ValueError: nope" in tool_msg.content

    @pytest.mark.asyncio
    async def test_on_tool_error_override_can_reraise(self):
        """Caller can pass a strict error handler that re-raises."""

        def broken(x: int) -> int:
            """A tool that raises."""
            raise RuntimeError("boom")

        def reraise(call: ToolCall, exc: Exception) -> str:
            raise exc

        chat, _ = _build_chat(
            side_effect=[_make_tool_call_response(calls=[("c", "broken", '{"x": 1}')])]
        )
        with pytest.raises(RuntimeError, match="boom"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[broken],
                on_tool_error=reraise,
            )


# ---------------------------------------------------------------------------
# Observation hook
# ---------------------------------------------------------------------------


class TestOnToolCallHook:
    @pytest.mark.asyncio
    async def test_on_tool_call_receives_call_and_result(self):
        observed: list[tuple[str, Any]] = []

        def observer(call: ToolCall, result: Any) -> None:
            observed.append((call.function.name, result))

        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c", "get_weather", '{"location": "X"}')]),
                _make_terminal_response("done"),
            ]
        )
        await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="hi")],
            tools=[get_weather],
            on_tool_call=observer,
        )
        assert observed == [("get_weather", "sunny in X (fahrenheit)")]

    @pytest.mark.asyncio
    async def test_on_tool_call_not_called_on_error(self):
        """When the handler raises, on_tool_call shouldn't fire."""
        observed: list[Any] = []

        def broken(x: int) -> int:
            raise ValueError("x")

        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c", "broken", '{"x": 1}')]),
                _make_terminal_response("done"),
            ]
        )
        await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="hi")],
            tools=[broken],
            on_tool_call=lambda c, r: observed.append((c, r)),
        )
        assert observed == []


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------


class TestMaxIterations:
    @pytest.mark.asyncio
    async def test_max_iterations_exceeded_raises(self):
        """If every response is a tool call, we hit max_iterations and raise."""

        # Always return a tool call (model never converges).
        def make_tool_call_resp() -> ChatCompletionResponse:
            return _make_tool_call_response(calls=[("c", "get_weather", '{"location": "X"}')])

        chat, _ = _build_chat(side_effect=[make_tool_call_resp() for _ in range(3)])
        with pytest.raises(MaxIterationsExceededError) as excinfo:
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[get_weather],
                max_iterations=3,
            )
        err = excinfo.value
        assert err.iterations == 3
        # Exception carries enough info to debug
        assert err.last_response is not None
        assert err.last_response.choices[0].finish_reason == "tool_calls"
        assert isinstance(err.messages, list)
        assert len(err.messages) > 1  # initial user plus loop output

    @pytest.mark.asyncio
    async def test_max_iterations_must_be_positive(self):
        chat, _ = _build_chat(side_effect=[])
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[get_weather],
                max_iterations=0,
            )


# ---------------------------------------------------------------------------
# Input safety
# ---------------------------------------------------------------------------


class TestInputSafety:
    @pytest.mark.asyncio
    async def test_messages_list_not_mutated(self):
        chat, _ = _build_chat(side_effect=[_make_terminal_response("done")])

        original = [UserMessage(content="hi")]
        snapshot = list(original)

        await chat.run_with_tools(model="m", messages=original, tools=[get_weather])
        assert original == snapshot  # input list untouched

    @pytest.mark.asyncio
    async def test_unknown_tool_raises_clear_error(self):
        """Model calls a tool that wasn't registered."""
        chat, _ = _build_chat(
            side_effect=[_make_tool_call_response(calls=[("c", "nonexistent_tool", '{"a": 1}')])]
        )
        with pytest.raises(ValueError, match="no matching tool was registered"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[get_weather],
            )

    @pytest.mark.asyncio
    async def test_handler_less_tool_rejected_at_registry_build(self):
        """Passing a Tool with no Python handler must fail at call site, not mid-loop.

        Pre-fix the registry stored handler=None silently and the agent loop
        raised only when the model actually invoked the tool. That meant a
        handler-less Tool produced a silent landmine: an unrunnable agent
        whose failure mode appeared to be a model-side error.
        """
        from venice_ai import tool_from_function
        from venice_ai.helpers import _python_type_to_json_schema  # noqa: F401  - sanity import

        def find_rhyme(word: str) -> list[str]:
            """Suggest rhyming words."""
            return []

        tool = tool_from_function(find_rhyme)
        chat, _ = _build_chat(side_effect=[])  # no responses needed; we error before any HTTP
        with pytest.raises(ValueError, match="no Python handler"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[tool],  # passing the Tool object instead of the callable
            )

    @pytest.mark.asyncio
    async def test_stream_kwarg_rejected(self):
        chat, _ = _build_chat(side_effect=[])
        with pytest.raises(ValueError, match="does not support streaming"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[get_weather],
                stream=True,
            )

    @pytest.mark.asyncio
    async def test_create_kwargs_forwarded(self):
        """temperature/etc. should reach create()."""
        chat, create_mock = _build_chat(side_effect=[_make_terminal_response("ok")])

        await chat.run_with_tools(
            model="m",
            messages=[UserMessage(content="hi")],
            tools=[get_weather],
            temperature=0.42,
            max_completion_tokens=50,
        )
        kwargs = create_mock.await_args_list[0].kwargs
        assert kwargs["temperature"] == 0.42
        assert kwargs["max_completion_tokens"] == 50


# ---------------------------------------------------------------------------
# Conversation.run_with_tools (D2)
# ---------------------------------------------------------------------------


class TestConversationRunWithTools:
    @pytest.mark.asyncio
    async def test_extends_conversation_with_loop_history(self):
        """Conversation.run_with_tools mutates the conversation in place
        with every turn the loop produced."""
        chat, _ = _build_chat(
            side_effect=[
                _make_tool_call_response(calls=[("c1", "get_weather", '{"location": "Paris"}')]),
                _make_terminal_response("Paris is sunny."),
            ]
        )

        # Build a conversation, wire its client.chat.completions to our mock chat.
        conv = Conversation(system="You are helpful.")
        conv.add_user("Weather in Paris?")
        starting = len(conv.messages)

        client = MagicMock()
        client.chat.completions = chat

        result = await conv.run_with_tools(client, model="m", tools=[get_weather])
        # Result is the same shape as D1
        assert isinstance(result, ToolLoopResult)
        assert result.text == "Paris is sunny."
        # Conversation has been extended with the loop's full new tail
        assert len(conv.messages) == starting + 3  # asst-tool-call, tool-result, asst-final
        # And matches the result.messages tail
        assert conv.messages[-len(result.messages) + (starting - 0) :] == result.messages[starting:]


class TestNonFunctionToolCalls:
    @pytest.mark.asyncio
    async def test_custom_tool_call_raises_a_clear_error(self):
        response = _make_tool_call_response(calls=[])
        response.choices[0].message.tool_calls = [
            ToolCall.model_validate(
                {"id": "c1", "type": "custom", "custom": {"name": "x", "input": "y"}}
            )
        ]
        chat, _ = _build_chat(side_effect=[response])
        with pytest.raises(ValueError, match="'custom' tool call .*only dispatches function tools"):
            await chat.run_with_tools(
                model="m",
                messages=[UserMessage(content="hi")],
                tools=[get_weather],
            )
