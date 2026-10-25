# Tool calling and agent loops in Venice v2

The v2 SDK exposes **two paths** for tool/function calling:

1. **`client.chat.completions.run_with_tools(...)`** — the SDK drives the agent loop, calls your tools, sends results back, returns a terminal `ToolLoopResult`. Use this 90% of the time.
2. **`client.chat.completions.create(tools=[...])` + manual loop** — you handle dispatch and re-call yourself. Use only when you need custom control between rounds.

## `run_with_tools`: the canonical agent loop

```python
import asyncio
from typing import Literal
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage, SystemMessage
from venice_ai.exceptions import MaxIterationsExceededError


def lookup_order(order_id: str) -> dict:
    """Look up an order by id."""
    return {"order_id": order_id, "status": "delivered", "total_usd": 49.99}


def issue_refund(order_id: str, reason: str) -> str:
    """Issue a refund for an order."""
    if not order_id:
        raise ValueError("order_id required")
    return f"Refund issued for {order_id} (reason: {reason})"


async def main() -> None:
    async with VeniceClient() as client:
        try:
            result = await client.chat.completions.run_with_tools(
                model=await client.models.resolve_chat(require_function_calling=True),
                messages=[
                    SystemMessage(content="You are a customer-support agent."),
                    UserMessage(content="Refund order ORD-123, item arrived damaged."),
                ],
                tools=[lookup_order, issue_refund],   # bare callables — see below
                max_iterations=5,
                temperature=0.2,
                max_completion_tokens=500,
            )
        except MaxIterationsExceededError as e:
            print(f"Loop didn't converge in {e.iterations} iterations")
            raise

    # ToolLoopResult fields
    print(result.response.text)        # terminal assistant message (str shortcut)
    print(result.iterations)           # model round trips (>=1; 1 means the model answered without calling any tool)
    # result.messages — full history including ToolMessage entries with tool returns
    # result.response — the underlying ChatCompletionResponse from the terminal call
```

### What goes in `tools=[...]`

This is the single most-confused part of the API. Pass **bare Python callables**, NOT `tool_from_function(fn)` results:

| Item | What happens | Verdict |
|---|---|---|
| `tools=[fn]` (bare callable) | SDK auto-wraps via `tool_from_function` AND registers `fn` as the dispatch handler. | ✅ Use this |
| `tools=[tool_from_function(fn)]` | SDK accepts the schema; dispatch handler is `None`. When the model invokes the tool, `_execute_tool_call` raises a clear error. | ❌ Will fail at dispatch |
| `tools=[tool_from_model(MyBaseModel)]` | Same problem — schema only, no handler. | ❌ Will fail at dispatch |

`tool_from_function` and `tool_from_model` exist for the **lower-level path** — passing schemas to `client.chat.completions.create(tools=[...])` where YOU dispatch the tool calls yourself. Don't mix them with `run_with_tools`.

### Tool functions can be sync or async

The SDK detects coroutines via `inspect.iscoroutinefunction` and awaits them appropriately:

```python
async def fetch_user(user_id: str) -> dict:
    async with httpx.AsyncClient() as http:
        r = await http.get(f"https://internal/users/{user_id}")
        return r.json()

# Sync and async tools mix freely:
result = await client.chat.completions.run_with_tools(
    ...,
    tools=[lookup_order, fetch_user],
)
```

### Tool-error handling — read this carefully

The SDK's **default** `on_tool_error` handler (`_default_on_tool_error`):

1. Logs the exception (with traceback) to the `venice_ai.tools` logger at ERROR level.
2. Formats the exception into a string the model receives as the tool's "result".

This is **good for resilient agents**: the model sees `"ValueError: order_id required"` and can self-correct (e.g., re-extract the order ID from the user message and re-call the tool). For most production use cases this is what you want.

**It's bad if you need strict propagation** — e.g., a tool that updates a database and you want a real bug to crash the loop, not be silently absorbed by the model. In that case, pass a custom handler that re-raises:

```python
from venice_ai.types.api.chat import ToolCall

def raise_on_tool_error(call: ToolCall, exc: Exception) -> str:
    # Re-raising from on_tool_error makes run_with_tools propagate the
    # original exception out of the agent loop.
    raise exc

result = await client.chat.completions.run_with_tools(
    ...,
    on_tool_error=raise_on_tool_error,
)
```

You can also do hybrid handling (re-raise some classes, format others) by checking `type(exc)` before deciding.

### Observability hooks

`on_tool_call(call: ToolCall, result: Any) -> None` fires after every successful tool dispatch — useful for logging:

```python
import logging
log = logging.getLogger("agent")

def log_tool_call(call: ToolCall, result):
    if call.function is None:              # only function calls are dispatched
        return
    log.info(
        "tool.call",
        name=call.function.name,
        args=call.function.arguments,
        result_preview=str(result)[:200],
    )

result = await client.chat.completions.run_with_tools(
    ...,
    on_tool_call=log_tool_call,
)
```

### `max_iterations` and `MaxIterationsExceededError`

Default `max_iterations=10`. When the loop hits the cap, the SDK raises `MaxIterationsExceededError(message, iterations=N, messages=[...], last_response=...)`. **Don't retry it** — it's a logic problem (the model is in a tool-call cycle it can't escape), not a transient failure. Surface to the operator.

### `ToolLoopResult` shape

```
ToolLoopResult
├── response:     ChatCompletionResponse  # the terminal assistant turn (no tool calls)
├── messages:     list[Message]           # full conversation history (system + user + tool calls + tool results + assistant turns)
├── iterations:   int                     # model round trips before convergence (1 if the model answered immediately with no tool calls)
└── text:         str | None              # shortcut for response.text (terminal message content)
```

`result.text` and `result.response.text` are equivalent. Use whichever reads better.

## When to hand-roll the loop instead

`run_with_tools` is the right call when:
- The agent runs to completion without external interruption.
- You only need observation hooks (`on_tool_call`, `on_tool_error`), not control flow changes.

Hand-roll the loop with `client.chat.completions.create(tools=[...])` when you need:
- **Human-in-the-loop confirmation**: pause between tool calls for user approval.
- **Dynamic tool injection**: add or remove tools mid-loop based on prior outputs.
- **Mid-loop budget checks**: stop the loop if cost exceeds a cap (more granular than the iteration cap).
- **Custom message rewriting**: redact, summarize, or filter messages between rounds.

The hand-rolled pattern (sketch):

```python
from venice_ai.types.api import ToolMessage

messages: list = [SystemMessage(content="..."), UserMessage(content="...")]
tools = [tool_from_function(lookup_order), tool_from_function(issue_refund)]   # schema-only OK here
dispatch = {"lookup_order": lookup_order, "issue_refund": issue_refund}

for _ in range(5):
    response = await client.chat.completions.create(
        model=model, messages=messages, tools=tools, tool_choice="auto",
    )
    msg = response.choices[0].message
    if msg.tool_calls is None:
        break                              # terminal turn

    messages.append(msg)
    for call in msg.tool_calls:
        if call.function is None:          # not a function call (e.g. "custom")
            continue
        result = dispatch[call.function.name](**call.function.arguments_dict)
        messages.append(ToolMessage(
            tool_call_id=call.id,
            content=str(result),
        ))
    # ↑ between rounds, do whatever custom logic you need
else:
    raise RuntimeError("max iterations")
```

Note `call.function.arguments_dict` — the SDK pre-parses the JSON args; you don't need `json.loads` yourself.

## Structured tool args via `tool_from_model`

If you want a tool whose arguments are validated as a Pydantic model:

```python
from pydantic import BaseModel
from venice_ai import tool_from_model

class RefundRequest(BaseModel):
    order_id: str
    reason: str
    amount_usd: float | None = None

# Use this in the LOW-LEVEL path; not with run_with_tools
tool_def = tool_from_model(RefundRequest)
```

The model will be presented with `RefundRequest`'s JSON Schema and have to produce conformant args. You then validate `RefundRequest.model_validate(call.function.arguments_dict)` yourself.

## Common bugs

- **Mixing `tool_from_function` results with `run_with_tools`**: the most-frequent silent failure. The agent loop exits with no tool dispatched, and you don't notice until production.
- **Forgetting to set `max_iterations`** for long workflows: defaults to 10. Knowledge-base searches often need more.
- **Catching the default `on_tool_error` LOG output** as a sign of failure: it's at ERROR level by design but the agent recovered — don't alert on this without checking whether the loop ultimately succeeded.
- **Treating `MaxIterationsExceededError` as transient**: re-raise; investigate.

## Related references

- `structured-output.md` — when the model's response itself should be a Pydantic model (no tool calls).
- `migration-v1-to-v2.md` — `run_with_tools` is new in v2.
- `headers-and-metadata.md` — surfacing usage and deprecation info from `result.response`.
