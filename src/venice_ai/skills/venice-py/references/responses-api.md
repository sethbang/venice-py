# The Responses API (beta)

Sourced from `src/venice_ai/resources/responses.py`. The Responses API is a **separate, beta** endpoint at `client.responses`, distinct from `client.chat.completions`. It accepts requests in OpenAI's Responses API format and returns typed output items.

Venice serves each request in one of two modes. Most OpenAI models are forwarded to OpenAI's Responses API natively. Other models, and OpenAI requests that use Venice features such as web search, go through a translation layer onto Chat Completions, which ignores or rejects what Chat Completions cannot express. The `x-venice-responses-lane` response header says which mode served a call.

## When to use Responses vs chat.completions

| Use chat.completions when | Use responses when |
|---|---|
| You want a managed tool loop (`run_with_tools`) | Your client already speaks the Responses format (Codex-style agents) |
| You need `parse()` / `response_format` structured output | You want OpenAI models served natively, with encrypted reasoning replay |
| You need streaming with `ChatStream` helpers | You need custom (freeform-input) tools or client-executed `tool_search` |
| You're targeting an E2EE-capable model | (Responses doesn't support E2EE models — use chat.completions with E2EE headers) |

For most use cases, `client.chat.completions` is the right call.

## The basic call

```python
async with VeniceClient() as client:
    response = await client.responses.create(
        model=await client.models.resolve_chat(),
        input="What is 2+2?",                  # str OR list of input items
        instructions="Answer in one word.",     # system instructions, applied before input
        max_output_tokens=100,                  # NOT max_completion_tokens
        temperature=0.3,
    )
    for block in response.output:
        print(block)
```

Returns `ResponsesResponse` (typed; not the same shape as `ChatCompletionResponse`).

## Differences from chat.completions

| Aspect | chat.completions | responses |
|---|---|---|
| Method | `client.chat.completions.create` | `client.responses.create` |
| Conversation | `messages: list[Message]` | `input: str | list[item]` (stateless) |
| System prompt | a `system` message | `instructions="..."` or a `system`/`developer` message |
| Token limit | `max_completion_tokens` | `max_output_tokens` |
| Output | `response.choices[0].message.content` | `response.output` (list of typed blocks) |
| Reasoning | `reasoning_effort` kwarg | `reasoning={"effort": "...", "summary": "..."}` |
| Streaming | `client.chat.completions.stream(...)` → `ChatStream` | `create(stream=True)` → `AsyncIterable[ResponsesStreamEvent]` |
| Structured output | `parse()` / `response_format=` | Not modeled; use chat.completions for typed parsing |

## Conversations are stateless

Venice stores nothing. Send the whole conversation in `input` on every call: the user turns, the previous `output` items, and your tool results. `input` items are dicts or the typed `ResponsesInput*` models; dicts are sent as written.

```python
from venice_ai.types.api import ResponsesInputItem

history: list[ResponsesInputItem] = [{"type": "message", "role": "user", "content": "What is 2+2?"}]
first = await client.responses.create(model=model, input=history, max_output_tokens=100)
history += [block.model_dump(exclude_none=True) for block in first.output]
history.append({"type": "message", "role": "user", "content": "And times 3?"})
second = await client.responses.create(model=model, input=history, max_output_tokens=100)
```

Output models are not accepted in `input` directly; dump them to dicts as above. Annotate the history as `list[ResponsesInputItem]` so it can hold both dicts and the typed input models used below.

## Output blocks

`response.output` is a list of typed blocks. Match on the class:

| `type` | Model | Key fields |
|---|---|---|
| `reasoning` | `ResponsesReasoningOutput` | `summary` (strings or `ResponsesReasoningSummaryText` parts), `encrypted_content` |
| `message` | `ResponsesMessageOutput` | `content[].text`, `status` |
| `function_call` | `ResponsesFunctionCallOutput` | `call_id`, `name`, `namespace`, `arguments` (JSON string) |
| `custom_tool_call` | `ResponsesCustomToolCallOutput` | `call_id`, `name`, `namespace`, `input` (raw text) |
| `tool_search_call` | `ResponsesToolSearchCallOutput` | `call_id`, `execution`, `arguments` (dict) |
| `web_search_call` | `ResponsesWebSearchCallOutput` | `id`, `status` |
| anything else | `ResponsesUnknownOutput` | `type` plus the raw fields |

```python
from venice_ai.types.api import (
    ResponsesCustomToolCallOutput,
    ResponsesFunctionCallOutput,
    ResponsesMessageOutput,
    ResponsesReasoningOutput,
)

for block in response.output:
    if isinstance(block, ResponsesMessageOutput):
        print("REPLY:", "".join(part.text for part in block.content))
    elif isinstance(block, ResponsesReasoningOutput):
        print("REASONING:", block.summary)
    elif isinstance(block, ResponsesFunctionCallOutput):
        print(f"TOOL CALL: {block.name}({block.arguments})")
    elif isinstance(block, ResponsesCustomToolCallOutput):
        print(f"CUSTOM TOOL: {block.name} <- {block.input!r}")
```

`status` fields are plain strings; the documented values are in the `KNOWN_RESPONSE_*` tuples. A block with an unmodeled `type`, or a known `type` missing required fields, parses as `ResponsesUnknownOutput` instead of failing the response.

## Reasoning

```python
response = await client.responses.create(
    model=await client.models.resolve_chat(require_reasoning=True),
    input="Plan how to set up a Postgres backup pipeline.",
    reasoning={"effort": "medium", "summary": "auto"},
    include=["reasoning.encrypted_content"],    # return encrypted reasoning for replay
    max_output_tokens=2000,
)
```

`reasoning.effort` requests a thinking depth: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. What a tier does depends on the model. `supportsReasoningEffort` and `reasoningEffortOptions` say which values a model accepts, not that a higher tier thinks longer: some models scale with the tier, some use about the same reasoning tokens at every tier, and some have used fewer at `"high"` than at `"low"`. Measure `usage` at the tiers you plan to send before relying on an order.

In native mode, send reasoning items back in `input` to keep the model's reasoning context. In translated mode, reasoning items in `input` are not passed to the model and `reasoning.summary` is ignored.

## Function tools

```python
from venice_ai.types.api import ResponsesNamedToolChoice

response = await client.responses.create(
    model=model,
    input="What's the weather in Paris?",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }],
    tool_choice=ResponsesNamedToolChoice(type="function", name="get_weather"),
    parallel_tool_calls=False,                 # at most one tool call per turn
)
```

`tool_choice` accepts `"auto"`, `"none"`, `"required"`, a named tool `{"type": "function" | "custom", "name": ..., "namespace": ...}` (`ResponsesNamedToolChoice`), or the Chat shape `{"type": "function", "function": {"name": ...}}` (`ResponsesFunctionToolChoice`). Any other string raises a `ValidationError` before the request is sent.

To answer a `function_call`, append the call and a result item to `input`:

```python
call = next(b for b in response.output if isinstance(b, ResponsesFunctionCallOutput))
history += [b.model_dump(exclude_none=True) for b in response.output]
history.append({"type": "function_call_output", "call_id": call.call_id, "output": '{"temp_c": 18}'})
```

You dispatch function tools yourself; there is no `run_with_tools` loop on this endpoint.

## Custom tools

A custom tool takes freeform text instead of JSON arguments. The spec has no typed model for the definition, so pass a dict. The model replies with a `custom_tool_call` block; answer it with a `custom_tool_call_output` item.

```python
from venice_ai.types.api import ResponsesInputCustomToolCallOutput

response = await client.responses.create(
    model=model,
    input=history,
    tools=[{"type": "custom", "name": "apply_patch", "description": "Apply a unified diff"}],
    tool_choice={"type": "custom", "name": "apply_patch"},
)
call = next(b for b in response.output if isinstance(b, ResponsesCustomToolCallOutput))
history += [b.model_dump(exclude_none=True) for b in response.output]
history.append(
    ResponsesInputCustomToolCallOutput(type="custom_tool_call_output", call_id=call.call_id, output="applied")
)
```

In translated mode, a malformed custom-tool call, or a call to a tool not declared in the request, comes back as an ordinary `function_call` with the original arguments.

## Tool search and deferred tools

Mark large tool sets `defer_loading` and add a client-executed `tool_search` tool. The model emits a `tool_search_call`; you run the search and send the matching tools back in a `tool_search_output` item.

```python
from venice_ai.types.api import (
    ResponsesFunctionTool,
    ResponsesFunctionToolDefinition,
    ResponsesInputToolSearchOutput,
    ResponsesToolSearchCallOutput,
    ResponsesToolSearchTool,
)

deferred = ResponsesFunctionTool(
    type="function",
    defer_loading=True,
    function=ResponsesFunctionToolDefinition(name="get_weather", parameters={"type": "object"}),
)
search = ResponsesToolSearchTool(type="tool_search", execution="client", parameters={"type": "object"})

response = await client.responses.create(model=model, input=history, tools=[search, deferred])
call = next(b for b in response.output if isinstance(b, ResponsesToolSearchCallOutput))
history += [b.model_dump(exclude_none=True) for b in response.output]
history.append(
    ResponsesInputToolSearchOutput(
        type="tool_search_output", call_id=call.call_id, execution="client", tools=[deferred]
    )
)
```

`ResponsesInputAdditionalTools(type="additional_tools", tools=[...])` loads more tools partway through `input` without a search.

## Web search (Venice extension)

```python
response = await client.responses.create(
    model=...,
    input="What were the major EU AI Act provisions effective last month?",
    web_search=True,                            # same as a {"type": "web_search"} tool
    max_output_tokens=500,
)
```

The output may include a `web_search_call` block, then a `message` block with the answer. Web search, `x_search`, and Venice-only `venice_parameters` route OpenAI models to translated mode. Web search is billed as search augmentation.

For more control, use `client.augment.search(...)` and pass results into chat.completions — covered in `characters-and-augment.md`.

## Prompt caching

`prompt_cache_retention` sets how long supported OpenAI models keep cached prefixes: `"in_memory"`, `"default"`, `"24h"`, or `"extended"` (`"24h"` and `"extended"` keep them up to 24 hours). Cached input shows up in `response.usage.cached_tokens`.

## Streaming

```python
stream = await client.responses.create(
    model=...,
    input="Tell me a story.",
    stream=True,                                # returns AsyncIterable[ResponsesStreamEvent]
)
async for event in stream:
    print(event.type, getattr(event, "delta", ""))
```

Events are `ResponsesStreamEvent` envelopes: `type` plus the event's fields as attributes. In translated mode reasoning streams as `response.reasoning.delta`, and web search sends a `response.web_search.done` event before the answer. There's no `text_deltas()` / `collect_with_deltas()` helper here.

## Hosted tools

The spec accepts `code_interpreter`, `file_search`, and `computer_use_preview` tool definitions as dicts, but Venice does not run them: translated mode ignores them, and native mode cannot serve hosted tools. Run the tool in your application and expose it as a function or custom tool.

## Common bugs

- **`max_completion_tokens=` on `client.responses.create`** — wrong kwarg here. Use `max_output_tokens`.
- **`messages=...` on `client.responses.create`** — wrong; use `input=...`.
- **Treating `response.choices[0].message.content`** — that's chat.completions shape; Responses uses `response.output[]` blocks.
- **Expecting `previous_response_id` to work** — nothing is stored; send the full conversation in `input`.
- **Passing output models straight into `input`** — dump them with `model_dump(exclude_none=True)` first.
- **`tool_choice="any"`** — not a valid mode; use `"required"`.
- **Using `client.chat.completions.run_with_tools` patterns on Responses** — `run_with_tools` is chat.completions-only.

## Related references

- `tool-loops.md` — the chat.completions agent-loop path, which is what you usually want.
- `streaming.md` — chat.completions streaming with `ChatStream`.
- `structured-output.md` — `parse()` is chat.completions only.
- `characters-and-augment.md` — `client.augment.search/scrape` as an alternative to Responses-API web_search.
