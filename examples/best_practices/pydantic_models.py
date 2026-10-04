#!/usr/bin/env python3
"""
Venice AI SDK - Pydantic Model Best Practices
==============================================

A reference for working with the Pydantic models in the Venice AI SDK.

PURPOSE
-------
Demonstrates the recommended patterns for building requests and reading responses.


HOW TO USE THIS FILE
--------------------
This file serves dual purposes:

1. **Executable Examples**: Run with `poetry run python examples/best_practices/pydantic_models.py`
   (requires VENICE_API_KEY environment variable) to see working examples.

2. **Reference Guide**: Read the code to learn proper patterns. Each section shows:
   - ✅ CORRECT patterns with detailed explanations
   - ❌ ANTI-PATTERNS (printed or safely demonstrated) showing what NOT to do
   - 🔧 Tool definitions and usage patterns
   - 📊 Response handling and data extraction

KEY SECTIONS OVERVIEW
---------------------
1. Message Models - Proper construction of UserMessage, SystemMessage, AssistantMessage
2. Response Models - Safe access to ChatCompletion response fields
3. Tool Calling - Defining tools, reading tool calls, and returning tool results
4. Streaming - Type-safe async iteration over ChatCompletionChunk objects
5. Request Configuration - Using StreamOptions, VeniceParameters, JSONSchemaFormat
6. Type Safety - Full type annotations for requests and responses
7. Common Pitfalls - Quick reference guide to all anti-patterns

IMPORTANT NOTES
---------------
- Responses are Pydantic models: use obj.field, never obj['field'] (that raises TypeError)
- Requests accept Pydantic models or plain dicts; models validate at construction
- Check optional fields for None before using them
- USE proper type hints for better IDE support and type checking
- READ the inline comments - they explain WHY things work this way

"""

import ast
import asyncio
import json
import operator
import sys
from collections.abc import Callable

from pydantic import ValidationError

from venice_ai import VeniceClient
from venice_ai.exceptions import NoMatchingModelError, VeniceError
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from venice_ai.types.api.requests.common import (
    ImageContent,
    ImageUrl,
    JSONSchemaFormat,
    StreamOptions,
    TextContent,
    Tool,
    ToolFunction,
    VeniceParameters,
)

# The live requests send only the messages shown here. Without this, Venice
# prepends its own system prompt and bills it as prompt tokens.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# Completion budget for the tool-calling requests. The cheapest tool-capable
# models are reasoning models, which can think for several thousand tokens even
# on an easy request before they answer; a tight cap cuts the answer off.
TOOL_CALL_MAX_TOKENS = 16384


def finish_reason_of(response: ChatCompletionResponse) -> str | None:
    """Return the first choice's finish_reason (``"length"`` means truncated)."""
    return response.choices[0].finish_reason if response.choices else None


# =============================================================================
# SECTION 1: MESSAGE MODELS
# =============================================================================
# Demonstrates proper construction and usage of message objects.
# Messages are the foundation of all chat interactions.
# =============================================================================


async def example_message_models_correct():
    """
    ✅ CORRECT: Demonstrates proper message model construction.

    All messages in the Venice AI SDK are Pydantic models, not dicts!
    This provides type safety, validation, and clear structure.
    """
    print("\n" + "=" * 70)
    print("Section 1: Message Models - CORRECT Patterns")
    print("=" * 70)

    # ✅ CORRECT: UserMessage with simple text content
    # This is the most common message type for user input
    user_msg = UserMessage(
        content="What is the capital of France?"  # Simple string content
    )
    print("\n✅ UserMessage created:")
    print(f"   role: {user_msg.role}")
    print(f"   content: {user_msg.content}")

    # ✅ CORRECT: SystemMessage for setting context
    # System messages guide the model's behavior and personality
    system_msg = SystemMessage(
        content="You are a helpful geography tutor. Provide concise, accurate answers.",
    )
    print("\n✅ SystemMessage created:")
    print(f"   role: {system_msg.role}")
    print(f"   content: {system_msg.content}")

    # ✅ CORRECT: AssistantMessage for model responses
    # Used when building conversation history or few-shot examples
    assistant_msg = AssistantMessage(
        content="The capital of France is Paris.",
    )
    print("\n✅ AssistantMessage created:")
    print(f"   role: {assistant_msg.role}")
    print(f"   content: {assistant_msg.content}")

    # ✅ CORRECT: UserMessage with multimodal content (text + image)
    # This demonstrates the more advanced content field structure
    multimodal_msg = UserMessage(
        content=[
            TextContent(type="text", text="What do you see in this image?"),
            ImageContent(
                type="image_url",
                image_url=ImageUrl(
                    url="https://example.com/image.jpg"
                    # Note: ImageUrl only has 'url' parameter
                ),
            ),
        ]
    )
    print("\n✅ Multimodal UserMessage created:")
    print(f"   role: {multimodal_msg.role}")
    print(f"   content type: {type(multimodal_msg.content)}")
    print(f"   content items: {len(multimodal_msg.content)} (text + image)")

    # ✅ CORRECT: Accessing message properties using Pydantic model attributes
    # This is type-safe and validated at runtime
    print("\n✅ Accessing message properties (Pydantic style):")
    print(f"   user_msg.role = '{user_msg.role}'")  # Attribute access
    print(f"   user_msg.content = '{user_msg.content}'")  # Attribute access

    # ✅ CORRECT: Building a conversation history
    # Messages are combined in a list for multi-turn conversations
    conversation = [system_msg, user_msg, assistant_msg]
    print("\n✅ Conversation history created:")
    print(f"   Total messages: {len(conversation)}")
    for i, msg in enumerate(conversation):
        print(f"   Message {i + 1}: {msg.role}")


def example_message_models_anti_patterns():
    """
    ❌ ANTI-PATTERNS: What NOT to do with message models.

    These patterns are WRONG and will cause runtime errors or bypass type safety.
    They are shown here for educational purposes only - DO NOT USE THEM!
    """
    print("\n" + "=" * 70)
    print("Section 1: Message Models - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n⚠️  WEAKER: Building messages as plain dicts")
    print("   # ACCEPTED, BUT PREFER THE MODEL:")
    print("   # user_msg = {'role': 'user', 'content': 'Hello'}")
    print("   # Dicts are validated into UserMessage when the request is built,")
    print("   # so a bad role still raises - but only once you call the API.")
    print("   # UserMessage(content='Hello') fails at construction instead,")
    print("   # and gives you editor completion on the fields.")

    print("\n❌ WRONG: Dict-style access on Pydantic models")
    print("   # DON'T DO THIS:")
    print("   # content = user_msg['content']  # ❌ Wrong!")
    print("   # INSTEAD USE:")
    print("   # content = user_msg.content  # ✅ Correct!")

    print("\n❌ WRONG: Not checking content type before accessing")
    print("   # DON'T DO THIS:")
    print("   # text = user_msg.content  # Might be a list!")
    print("   # INSTEAD USE:")
    print("   # if isinstance(user_msg.content, str):")
    print("   #     text = user_msg.content")
    print("   # elif isinstance(user_msg.content, list):")
    print("   #     # Handle multimodal content")

    print("\n❌ WRONG: Changing a message's role after construction")
    print("   # DON'T DO THIS:")
    print("   # user_msg.role = 'assistant'  # raises ValidationError (role is fixed)")
    print("   # INSTEAD: Create a message of the right type")
    print("   # reply = AssistantMessage(content=user_msg.content)")

    print("\n💡 Key Takeaway: Prefer the Pydantic models - they catch mistakes earlier!")


def _expect_rejection(label: str, attempt: Callable[[], object]) -> bool:
    """Run ``attempt`` and report whether Pydantic rejected it."""
    print(f"\n🛑 Attempting: {label}")
    try:
        attempt()
    except ValidationError as e:
        error = e.errors()[0]
        print(f"   ✅ Rejected: {error['msg']}")
        print(f"      loc={error['loc']}, input={error['input']!r}")
        return True
    print("   ❌ Expected ValidationError, but it was accepted")
    return False


def example_validation_rejections_live() -> bool:
    """
    🛑 LIVE DEMO: Actually trigger Pydantic validation errors.

    The anti-pattern functions above are commented strings. Here we run the
    bad code so you can see the rejected input and the error message Pydantic
    emits. Returns True only if every bad input was rejected.
    """
    print("\n" + "=" * 70)
    print("Section 1b: Validation Rejections (LIVE)")
    print("=" * 70)

    user_msg = UserMessage(content="Hello")

    def reassign_role() -> None:
        user_msg.role = "assistant"  # type: ignore[assignment]

    checks: list[tuple[str, Callable[[], object]]] = [
        # Wrong role on UserMessage: Pydantic enforces role="user".
        (
            "UserMessage(role='assistant', content='Hello')",
            lambda: UserMessage(role="assistant", content="Hello"),  # type: ignore[arg-type]
        ),
        # Missing required field: `name` on ToolFunction.
        (
            'ToolFunction(description="missing name")',
            lambda: ToolFunction(description="missing name"),  # type: ignore[call-arg]
        ),
        # Wrong type: `parameters` on ToolFunction must be a dict.
        (
            'ToolFunction(name="x", parameters="this should be a dict")',
            lambda: ToolFunction(name="x", parameters="this should be a dict"),  # type: ignore[arg-type]
        ),
        # Wrong type: StreamOptions.include_usage must be a bool.
        (
            'StreamOptions(include_usage="not a bool")',
            lambda: StreamOptions(include_usage="not a bool"),  # type: ignore[arg-type]
        ),
        # Assignment is validated too, so a message cannot change its role.
        ("user_msg.role = 'assistant'", reassign_role),
    ]
    rejected = [_expect_rejection(label, attempt) for label, attempt in checks]

    print("\n💡 Pydantic catches these before any API call is made.")
    return all(rejected)


# =============================================================================
# SECTION 2: RESPONSE MODELS
# =============================================================================
# Demonstrates safe access to ChatCompletion response objects.
# Responses contain choices, messages, usage stats, and more.
# =============================================================================


async def example_response_access_correct() -> bool:
    """
    ✅ CORRECT: Safe access to ChatCompletion response fields.

    Response objects are complex Pydantic models with nested structures.
    Always check for None and use proper attribute access.

    Returns True on success, False if the live request failed.
    """
    print("\n" + "=" * 70)
    print("Section 2: Response Access - CORRECT Patterns")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            # Make a simple request. exclude_reasoning picks a model that answers
            # directly, so a short completion budget is not spent thinking.
            model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Say 'Hello!' in exactly one word.")],
                max_completion_tokens=60,
                venice_parameters=OWN_PROMPT_ONLY,
            )

            # ✅ CORRECT: Safe access to response.choices
            # Always check if choices exist before accessing
            if response.choices:
                print("\n✅ Accessing response.choices safely:")
                print(f"   Number of choices: {len(response.choices)}")

                # ✅ CORRECT: Access first choice using attribute access
                first_choice = response.choices[0]
                print(f"   First choice index: {first_choice.index}")
                print(f"   Finish reason: {first_choice.finish_reason}")

                # ✅ CORRECT: Access message from choice
                message = first_choice.message
                print("\n✅ Accessing message:")
                print(f"   Role: {message.role}")

                # ✅ CORRECT: Safe access to optional content field
                # Content might be None if there were tool calls instead
                if message.content:
                    print(f"   Content: {message.content}")
                else:
                    print("   Content: (None - likely had tool calls)")

                # ✅ CORRECT: Check for tool_calls before accessing
                if message.tool_calls:
                    print(f"   Tool calls: {len(message.tool_calls)}")
                else:
                    print("   Tool calls: None")

            # ✅ CORRECT: Safe access to usage statistics
            # Usage is always present but good to be defensive
            if response.usage:
                usage = response.usage
                print("\n✅ Accessing usage statistics:")
                print(f"   Prompt tokens: {usage.prompt_tokens}")
                print(f"   Completion tokens: {usage.completion_tokens}")
                print(f"   Total tokens: {usage.total_tokens}")

            # ✅ CORRECT: Access response ID
            print("\n✅ Response metadata:")
            print(f"   ID: {response.id}")
            print(f"   Model: {response.model}")
            print(f"   Created: {response.created}")

        except NoMatchingModelError:
            raise  # a missing model skips the example; see __main__

        except VeniceError as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            return False

    if finish_reason_of(response) == "length" or not response.text:
        print("\n❌ Response was truncated or empty")
        return False
    return True


def example_response_access_anti_patterns():
    """
    ❌ ANTI-PATTERNS: Unsafe response access patterns.

    These patterns can cause runtime errors or produce incorrect results.
    """
    print("\n" + "=" * 70)
    print("Section 2: Response Access - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n❌ WRONG: Using the text without checking for None")
    print("   # DON'T DO THIS:")
    print("   # words = response.text.split()")
    print("   # response.text is None when the model answered with tool calls,")
    print("   # so .split() raises AttributeError.")
    print("   # INSTEAD USE:")
    print("   # if response.text:")
    print("   #     words = response.text.split()")

    print("\n❌ WRONG: Dict-style access on response")
    print("   # DON'T DO THIS:")
    print("   # content = response['choices'][0]['message']['content']")
    print("   # INSTEAD USE:")
    print("   # content = response.text")

    print("\n❌ WRONG: Assuming tool_calls always exists")
    print("   # DON'T DO THIS:")
    print("   # for tool_call in message.tool_calls:")
    print("   #     # This crashes if tool_calls is None!")
    print("   # INSTEAD USE:")
    print("   # if message.tool_calls:")
    print("   #     for tool_call in message.tool_calls:")

    print("\n❌ WRONG: Not checking if choices array is empty")
    print("   # DON'T DO THIS:")
    print("   # message = response.choices[0].message  # IndexError if empty!")
    print("   # INSTEAD USE:")
    print("   # if response.choices:")
    print("   #     message = response.choices[0].message")

    print("\n💡 Key Takeaway: Always check for None and empty arrays!")


# =============================================================================
# SECTION 3: TOOL CALLING
# =============================================================================
# Defining tools, reading tool calls, and sending tool results back.
# =============================================================================


async def example_tool_definition_correct():
    """
    ✅ PREFERRED: Define tools using Tool and ToolFunction Pydantic models.

    Plain dicts in the OpenAI tool format are accepted too, and are validated
    when the request is built. The models catch mistakes (a missing name, a
    non-dict schema) where you write them, and give editor completion.
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Definition - CORRECT Patterns")
    print("=" * 70)

    # ✅ PREFERRED: Tool definition using Pydantic models
    weather_tool = Tool(
        type="function",  # Must be "function" for function calling
        function=ToolFunction(
            name="get_weather",  # Clear, descriptive function name
            description="Get current weather for a location",  # Helps model decide when to use it
            parameters={
                # JSON Schema for function parameters
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g. 'San Francisco, CA'",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],  # Only location is required
            },
            strict=False,  # Set to True for strict schema validation
        ),
        id=None,  # Optional: tool ID
    )

    print("\n✅ Tool defined using Pydantic models:")
    print(f"   Type: {weather_tool.type}")
    assert weather_tool.function is not None
    print(f"   Function name: {weather_tool.function.name}")
    print(f"   Function description: {weather_tool.function.description}")
    print(f"   Strict validation: {weather_tool.function.strict}")

    # ✅ CORRECT: Accessing tool properties using Pydantic attributes
    func = weather_tool.function
    print("\n✅ Accessing tool properties (Pydantic style):")
    print(f"   func.name = '{func.name}'")  # Attribute access
    print(f"   func.description = '{func.description}'")  # Attribute access

    # ✅ CORRECT: Multiple tools in a list
    tools = [
        weather_tool,
        Tool(
            type="function",
            function=ToolFunction(
                name="search_web",
                description="Search the web for current information",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"],
                },
                strict=False,
            ),
            id=None,  # Optional: tool ID
        ),
    ]

    print("\n✅ Multiple tools defined:")
    for i, tool in enumerate(tools):
        assert tool.function is not None
        print(f"   Tool {i + 1}: {tool.function.name}")


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression without ``eval``."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError(f"Unsupported expression: {expression!r}")

    return walk(ast.parse(expression.replace("×", "*"), mode="eval").body)


async def example_tool_calling_correct() -> bool:
    """
    🔧 Tool calling round trip: read the tool call, run the tool, send the
    result back, and get the final answer.

    Returns True only if the model called the tool and the final answer
    contains the tool's result.
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Calling - CORRECT Patterns")
    print("=" * 70)

    calc_tool = Tool(
        type="function",
        function=ToolFunction(
            name="calculate",
            description="Evaluate a basic arithmetic expression such as '12 * 7'",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            strict=False,
        ),
        id=None,
    )
    expected = str(42 * 137)

    async with VeniceClient() as client:
        try:
            model = await client.models.resolve_chat(
                require_function_calling=True, prefer="cheapest"
            )
            messages: list = [
                UserMessage(content="What is 42 * 137? Use the calculate tool."),
            ]
            # Tool-capable models are often reasoning models, which spend
            # tokens thinking before they answer; leave them room.
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[calc_tool],
                tool_choice="auto",
                max_completion_tokens=TOOL_CALL_MAX_TOKENS,
                venice_parameters=OWN_PROMPT_ONLY,
            )
            print(f"\n   Model: {model}, finish_reason: {finish_reason_of(response)}")

            # ✅ CORRECT: Check for tool calls and access using Pydantic properties
            tool_calls = response.choices[0].message.tool_calls if response.choices else None
            if not tool_calls:
                print("\n❌ The model answered without calling the tool")
                return False

            print(f"\n🔧 Tool calls received: {len(tool_calls)}")
            # ✅ Echo the assistant turn (with its tool calls) back into history
            messages.append(AssistantMessage.from_response(response))

            for tool_call in tool_calls:
                func_name = tool_call.function.name  # ✅ Pydantic property
                func_args_str = tool_call.function.arguments  # ✅ JSON string
                print(f"\n   Function name: {func_name}")
                print(f"   Call ID: {tool_call.id}")
                print(f"   Arguments (JSON string): {func_args_str}")

                # ✅ CORRECT: Parse arguments from the JSON string
                try:
                    args = json.loads(func_args_str)
                    result = safe_calculate(args["expression"])
                    content = str(int(result)) if result == int(result) else str(result)
                except (json.JSONDecodeError, KeyError, ValueError, ZeroDivisionError) as e:
                    content = f"error: {e}"
                print(f"   Tool result: {content}")

                # ✅ Return the result as a ToolMessage tied to the call ID
                messages.append(ToolMessage(content=content, tool_call_id=tool_call.id))

            final = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[calc_tool],
                max_completion_tokens=TOOL_CALL_MAX_TOKENS,
                venice_parameters=OWN_PROMPT_ONLY,
            )
        except NoMatchingModelError:
            raise  # a missing model skips the example; see __main__
        except VeniceError as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            return False

    answer = final.text or ""
    print(f"\n   Final answer: {answer.strip()}")
    print(f"   finish_reason: {finish_reason_of(final)}")

    print("\n🎯 Pattern Summary:")
    print("   ✅ tool_call.function.name / .arguments / tool_call.id")
    print("   ✅ AssistantMessage.from_response(response) keeps the tool calls in history")
    print("   ✅ ToolMessage(content=..., tool_call_id=tool_call.id) returns each result")

    if finish_reason_of(final) == "length" or expected not in answer.replace(",", ""):
        print(f"\n❌ Final answer is truncated or does not contain {expected}")
        return False
    return True


def example_tool_calling_anti_patterns():
    """
    ❌ ANTI-PATTERNS: Tool-call mistakes to avoid.
    """
    print("\n" + "=" * 70)
    print("Section 3: Tool Calling - ANTI-PATTERNS")
    print("=" * 70)

    print("\n❌ WRONG: Dict-style access on tool calls")
    print("   This is a common mistake when accessing tool calls on the response.")
    print("\n   # ❌ WRONG: Dict-style access (DOESN'T WORK!)")
    print("   # for tool_call in message.tool_calls:")
    print("   #     func_name = tool_call['function']['name']  # ❌ ERROR!")
    print("   #     func_args = tool_call['function']['arguments']  # ❌ ERROR!")
    print("   #     call_id = tool_call['id']  # ❌ ERROR!")
    print("\n   # ✅ CORRECT: Pydantic property access")
    print("   # for tool_call in message.tool_calls:")
    print("   #     func_name = tool_call.function.name  # ✅ Works!")
    print("   #     func_args = tool_call.function.arguments  # ✅ Works!")
    print("   #     call_id = tool_call.id  # ✅ Works!")

    print("\n⚠️  WEAKER: Defining tools as plain dicts")
    print("   # ACCEPTED, BUT PREFER THE MODELS:")
    print("   # tool = {'type': 'function', 'function': {'name': 'my_func', ...}}")
    print("   # The dict is validated only when the request is built.")
    print("   # Tool(type='function', function=ToolFunction(...)) fails where you write it.")

    print("\n❌ WRONG: Not sending the tool result back")
    print("   # Reading tool_calls is only half the loop. Append")
    print("   # AssistantMessage.from_response(response) and one ToolMessage per call,")
    print("   # then call the model again for the final answer.")
    print("   # client.chat.completions.run_with_tools(...) runs this loop for you")
    print("   # (see examples/chat/agent_loop.py).")

    print("\n❌ WRONG: Not parsing arguments JSON string")
    print("   # DON'T DO THIS:")
    print("   # args = tool_call.function.arguments  # This is a JSON string!")
    print("   # value = args['param']  # ❌ Can't index a string!")
    print("   # INSTEAD USE:")
    print("   # args = json.loads(tool_call.function.arguments)")
    print("   # value = args['param']  # ✅ Now it's a dict!")

    print("\n❌ WRONG: Not checking if tool_calls exists")
    print("   # DON'T DO THIS:")
    print("   # for tool_call in message.tool_calls:  # Might be None!")
    print("   # INSTEAD USE:")
    print("   # if message.tool_calls:")
    print("   #     for tool_call in message.tool_calls:")

    print("\n💡 Takeaway: tool_call.function.name, NOT tool_call['function']['name']!")


# =============================================================================
# SECTION 4: STREAMING
# =============================================================================
# Demonstrates type-safe streaming with ChatCompletionChunk objects.
# Streaming requires careful None checking and delta handling.
# =============================================================================


async def example_streaming_correct() -> bool:
    """
    ✅ CORRECT: Type-safe async streaming with ChatCompletionChunk.

    Streaming returns chunks with delta updates, not complete messages.
    Always check for None and accumulate content properly.

    Returns True on success, False if the live request failed.
    """
    print("\n" + "=" * 70)
    print("Section 4: Streaming - CORRECT Patterns")
    print("=" * 70)

    async with VeniceClient() as client:
        try:
            print("\n🌊 Starting streaming request...")

            # ✅ CORRECT: Create streaming request (on a model that answers directly)
            model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
            stream = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content="Count from 1 to 5, one number per line.")],
                stream=True,  # Enable streaming
                max_completion_tokens=100,
                venice_parameters=OWN_PROMPT_ONLY,
            )

            print("\n✅ Streaming response (Pydantic chunks):")
            print("   ", end="")

            accumulated_content = ""
            finish_reason = None

            # ✅ CORRECT: Type-annotated async iteration
            # Each chunk is a ChatCompletionChunk Pydantic model
            async for chunk in stream:
                # ✅ CORRECT: Check if choices exist
                if chunk.choices:
                    choice = chunk.choices[0]

                    # ✅ CORRECT: Access delta from choice
                    delta = choice.delta

                    # ✅ CORRECT: Check if content exists before accessing
                    # Content might be None for chunks without new text
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        accumulated_content += delta.content

                    # ✅ CORRECT: Check finish_reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

            print()  # New line after streaming

            print("\n✅ Streaming complete:")
            print(f"   Finish reason: {finish_reason}")
            print(f"   Total content length: {len(accumulated_content)} chars")

        except NoMatchingModelError:
            raise  # a missing model skips the example; see __main__

        except VeniceError as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            return False

    if finish_reason == "length" or "5" not in accumulated_content:
        print("\n❌ Stream was truncated or did not reach 5")
        return False
    return True


def example_streaming_anti_patterns():
    """
    ❌ ANTI-PATTERNS: Unsafe streaming patterns.

    Streaming requires even more careful None checking than regular responses.
    """
    print("\n" + "=" * 70)
    print("Section 4: Streaming - ANTI-PATTERNS (What NOT to Do)")
    print("=" * 70)

    print("\n❌ WRONG: Not checking if delta.content is None")
    print("   # DON'T DO THIS:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk.choices[0].delta.content")
    print("   #     print(content)  # Might print None!")
    print("   # INSTEAD USE:")
    print("   # async for chunk in stream:")
    print("   #     if chunk.choices and chunk.choices[0].delta.content:")
    print("   #         print(chunk.choices[0].delta.content)")

    print("\n❌ WRONG: Dict-style access on chunks")
    print("   # DON'T DO THIS:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk['choices'][0]['delta']['content']")
    print("   # INSTEAD USE:")
    print("   # async for chunk in stream:")
    print("   #     content = chunk.choices[0].delta.content")

    print("\n❌ WRONG: Assuming choices always exists")
    print("   # DON'T DO THIS:")
    print("   # delta = chunk.choices[0].delta  # IndexError if empty!")
    print("   # INSTEAD USE:")
    print("   # if chunk.choices:")
    print("   #     delta = chunk.choices[0].delta")

    print("\n❌ WRONG: Not handling finish_reason")
    print("   # DON'T DO THIS:")
    print("   # Just accumulate content without checking if done")
    print("   # INSTEAD USE:")
    print("   # if choice.finish_reason:")
    print("   #     # Handle end of stream")

    print("\n💡 Key Takeaway: Streaming requires extra None checking!")


# =============================================================================
# SECTION 5: REQUEST CONFIGURATION
# =============================================================================
# Demonstrates advanced request configuration using Pydantic models.
# StreamOptions, VeniceParameters, and JSONSchemaFormat examples.
# =============================================================================


async def example_request_config_correct() -> bool:
    """
    ✅ CORRECT: Request configuration with Pydantic models, sent live.

    Builds StreamOptions, VeniceParameters and JSONSchemaFormat and sends them
    in one streaming request. The run checks that usage arrives on the stream
    and that the output parses against the schema.

    Returns True on success, False if the request failed or a setting had no
    visible effect.
    """
    print("\n" + "=" * 70)
    print("Section 5: Request Configuration - CORRECT Patterns")
    print("=" * 70)

    # ✅ StreamOptions: ask for token usage on the final stream chunk
    stream_opts = StreamOptions(include_usage=True)
    print("\n✅ StreamOptions created:")
    print(f"   include_usage: {stream_opts.include_usage}")

    # ✅ VeniceParameters: Venice-specific switches. Web search stays off so
    # the request is not billed for a search; include_venice_system_prompt=False
    # sends only your own messages to the model.
    venice_params = VeniceParameters(
        enable_web_search="off",
        include_venice_system_prompt=False,
    )
    print("\n✅ VeniceParameters created:")
    print(f"   enable_web_search: {venice_params.enable_web_search}")
    print(f"   include_venice_system_prompt: {venice_params.include_venice_system_prompt}")

    # ✅ JSONSchemaFormat: constrain the output to a JSON schema
    json_schema = JSONSchemaFormat(
        type="json_schema",
        json_schema={
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age", "email"],
                "additionalProperties": False,
            },
        },
    )
    print("\n✅ JSONSchemaFormat created:")
    print(f"   type: {json_schema.type}")
    print(f"   schema name: {json_schema.json_schema['name']}")

    print("\n🌊 Sending all three in one streaming request...")
    content = ""
    finish_reason = None
    usage = None
    async with VeniceClient() as client:
        try:
            # The catalog's default ranking, not prefer="cheapest": strict JSON
            # schema output needs a model that follows it reliably, and the
            # cheapest schema-capable models are reasoning models (which would
            # spend the budget thinking) or do not honor response_format.
            model = await client.models.resolve_chat(require_response_schema=True)
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    UserMessage(content="Extract the person: Ada Lovelace, 36, ada@example.com"),
                ],
                stream=True,
                stream_options=stream_opts,
                venice_parameters=venice_params,
                response_format=json_schema,
                max_completion_tokens=2048,
            )
            async for chunk in stream:
                if chunk.choices:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                if chunk.usage:
                    usage = chunk.usage
        except NoMatchingModelError:
            raise  # a missing model skips the example; see __main__
        except VeniceError as e:
            print(f"\n❌ Error: {type(e).__name__}: {e}")
            return False

    print(f"   Model: {model}, finish_reason: {finish_reason}")
    print(f"   Raw output: {content.strip()}")
    ok = finish_reason != "length"

    try:
        person = json.loads(content)
        print(f"   Parsed: {person}")
        if set(person) != {"name", "age", "email"} or not isinstance(person["age"], int):
            print("   ❌ Output does not match the schema")
            ok = False
        elif (person["name"], person["age"], person["email"]) != (
            "Ada Lovelace",
            36,
            "ada@example.com",
        ):
            print("   ❌ Output fits the schema but does not hold the values in the prompt")
            ok = False
    except json.JSONDecodeError as e:
        print(f"   ❌ Output is not JSON: {e}")
        ok = False

    if usage is None:
        print("   ❌ No usage chunk arrived despite include_usage=True")
        ok = False
    else:
        # include_usage=True puts token counts on the stream's final chunk.
        print(f"   Usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")

    print("\n💡 Requests also accept plain dicts for these fields; the models")
    print("   validate your settings before the request is sent.")
    return ok


# =============================================================================
# SECTION 6: TYPE SAFETY
# =============================================================================
# Demonstrates type annotations for requests and responses.
# Shows how to write fully typed Venice AI SDK code.
# =============================================================================


async def example_type_hints_correct():
    """
    ✅ CORRECT: Full type annotations for Venice AI SDK code.
    """
    print("\n" + "=" * 70)
    print("Section 6: Type Safety - CORRECT Patterns")
    print("=" * 70)

    print("\n✅ Import the response types:")
    print("   from venice_ai.types.chat import ChatCompletionChunk, ChatCompletionResponse")

    print("\n✅ Function with full type annotations:")
    print("   def answer_text(response: ChatCompletionResponse) -> str | None:")
    print("       return response.text  # None when the model returned tool calls")

    print("\n✅ Annotating a message history:")
    print("   messages: list[SystemMessage | UserMessage | AssistantMessage] = [")
    print("       SystemMessage(content='...'),")
    print("       UserMessage(content='...'),")
    print("   ]")

    print("\n✅ Type-safe async iteration:")
    print("   stream = await client.chat.completions.create(..., stream=True)")
    print("   async for chunk in stream:")
    print("       # chunk is a ChatCompletionChunk")

    print("\n💡 Type hints enable IDE autocomplete and catch errors early!")


# =============================================================================
# SECTION 7: COMMON PITFALLS REFERENCE
# =============================================================================
# Quick reference guide to all common pitfalls and their solutions.
# Scannable format for quick lookups.
# =============================================================================


def common_pitfalls_reference():
    """
    📚 QUICK REFERENCE: Common Pitfalls and Solutions

    A comprehensive, scannable list of all common mistakes and their fixes.
    Use this as a quick lookup when writing Venice AI SDK code.
    """
    print("\n" + "=" * 70)
    print("Section 7: Common Pitfalls Reference")
    print("=" * 70)

    pitfalls = """
1. MESSAGE CONSTRUCTION
   ⚠️ msg = {'role': 'user', 'content': 'Hello'}   # accepted, validated late
   ✅ msg = UserMessage(content='Hello')

2. ACCESSING MESSAGE CONTENT
   ❌ content = msg['content']
   ✅ content = msg.content

3. USING THE TEXT WITHOUT CHECKING
   ❌ words = response.text.split()   # text is None after a tool call
   ✅ if response.text:
       words = response.text.split()

4. TOOL CALL ACCESS
   ❌ name = tool_call['function']['name']
   ✅ name = tool_call.function.name

5. TOOL DEFINITION
   ⚠️ tool = {'type': 'function', 'function': {...}}   # accepted, validated late
   ✅ tool = Tool(type='function', function=ToolFunction(...))

6. TOOL ARGUMENTS PARSING
   ❌ args = tool_call.function.arguments['param']
   ✅ args = json.loads(tool_call.function.arguments)
       value = args['param']

7. STREAMING CONTENT ACCESS
   ❌ print(chunk.choices[0].delta.content)
   ✅ if chunk.choices and chunk.choices[0].delta.content:
       print(chunk.choices[0].delta.content)

8. CHECKING FOR TOOL CALLS
   ❌ for tc in message.tool_calls:
   ✅ if message.tool_calls:
       for tc in message.tool_calls:

9. MULTIMODAL CONTENT
   ❌ text = message.content  # Might be a list!
   ✅ if isinstance(message.content, str):
       text = message.content
      elif isinstance(message.content, list):
       # Handle multimodal

10. CONFIGURATION OBJECTS
    ⚠️ stream_options = {'include_usage': True}   # accepted, validated late
    ✅ stream_options = StreamOptions(include_usage=True)

11. TYPE HINTS
    ❌ def process(response):
    ✅ def process(response: ChatCompletionResponse) -> str | None:

12. FINISH REASON IN STREAMING
    ❌ # Just accumulate without checking done
    ✅ if choice.finish_reason:
        # Handle end of stream

13. EMPTY CHOICES ARRAY
    ❌ message = response.choices[0].message
    ✅ if response.choices:
        message = response.choices[0].message

14. OPTIONAL FIELDS
    ❌ usage = response.usage.total_tokens
    ✅ if response.usage:
        usage = response.usage.total_tokens

15. TRUNCATED ANSWERS
    ❌ print(response.text)   # may be cut off mid-sentence
    ✅ if response.choices[0].finish_reason == 'length':
           # raise max_completion_tokens or shorten the task
"""

    print(pitfalls)
    print("\n" + "=" * 70)
    print("💡 Remember: Prefer Pydantic models; they validate where you write them")
    print("💡 Remember: Attribute access on responses, not dict-style!")
    print("💡 Remember: Always check for None!")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


async def main() -> int:
    """
    Run all best practices examples.

    This demonstrates all the patterns in sequence, showing both correct
    approaches and anti-patterns (the anti-patterns are not executed, just shown).

    Returns 0 if every live section succeeded, 1 if any live section failed.
    """
    print("=" * 70)
    print("Venice AI SDK - Pydantic Model Best Practices")
    print("=" * 70)

    live_results: list[tuple[str, bool]] = []

    await example_message_models_correct()
    example_message_models_anti_patterns()
    live_results.append(("Validation Rejections", example_validation_rejections_live()))

    live_results.append(("Response Access", await example_response_access_correct()))
    example_response_access_anti_patterns()

    await example_tool_definition_correct()
    live_results.append(("Tool Calling", await example_tool_calling_correct()))
    example_tool_calling_anti_patterns()

    live_results.append(("Streaming", await example_streaming_correct()))
    example_streaming_anti_patterns()

    live_results.append(("Request Configuration", await example_request_config_correct()))

    await example_type_hints_correct()
    common_pitfalls_reference()

    print("\n📚 What You Learned:")
    print("   • Responses are Pydantic models: use obj.field, never obj['field']")
    print("   • Requests accept dicts, but models validate where you write them")
    print("   • Check optional fields for None and finish_reason for truncation")
    print("   • Complete the tool loop: ToolMessage results, then a follow-up call")
    print("   • Stream with None checks on every chunk")
    print("   • Type hints for better IDE support")

    print("\n🎯 Next Steps:")
    print("   1. Review the code comments for detailed explanations")
    print("   2. Use this file as a reference when writing Venice AI code")
    print("   3. Copy the ✅ CORRECT patterns into your own code")
    print("   4. Avoid the ❌ ANTI-PATTERNS shown here")

    print("\n🔗 Related Examples:")
    print("   - examples/chat/tool_calling.py - More tool calling examples")
    print("   - examples/chat/streaming_chat.py - Advanced streaming patterns")
    print("   - examples/basic/quick_start.py - Getting started guide")

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok in live_results if ok)
    failed = len(live_results) - passed
    if failed:
        print(f"❌ {passed}/{len(live_results)} live sections succeeded; {failed} failed")
        for name, ok in live_results:
            print(f"   {'✓' if ok else '✗'} {name}")
    else:
        print(f"✅ All {passed} live sections succeeded")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(130)
    except NoMatchingModelError as e:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {e}")
        sys.exit(77)
