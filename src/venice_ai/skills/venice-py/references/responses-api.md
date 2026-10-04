# The Responses API (Alpha)

Sourced from `src/venice_ai/resources/responses.py`. The Responses API is a **separate, alpha-status** endpoint at `client.responses` — distinct from the main `client.chat.completions`. It's an OpenAI-compatible Responses-API surface with Venice extensions (web_search, x_search, code_interpreter tools, etc.).

## When to use Responses vs chat.completions

| Use chat.completions when | Use responses when |
|---|---|
| Building a multi-turn conversation | Doing one-shot calls (Responses is stateless) |
| Need streaming with usage stats via `final_response.usage` | OK with the Responses-API streaming events |
| Want tool-calling with `run_with_tools` (managed loop) | Want OpenAI Responses-API alpha tool types (web_search, code_interpreter, etc.) |
| You're using `parse()` for structured output | You need the Responses-API output-block format (reasoning, message, function_call, web_search_call) |
| You're targeting an E2EE-capable model | (Responses doesn't support E2EE models — use chat.completions with E2EE headers) |

For most use cases, `client.chat.completions` is the right call. Use `client.responses` when you specifically need the OpenAI Responses-API surface or its alpha tool types.

## The basic call

```python
async with VeniceClient() as client:
    response = await client.responses.create(
        model=await client.models.resolve_chat(),
        input="What is 2+2?",                  # str OR list[dict] (structured)
        max_output_tokens=100,                  # NOT max_completion_tokens; this API uses max_output_tokens
        temperature=0.3,
    )
    # response.output is a list of typed blocks: reasoning / message / function_call / web_search_call
    for block in response.output:
        print(block)
```

Returns `ResponsesResponse` (typed; not the same shape as `ChatCompletionResponse`).

## Differences from chat.completions

| Aspect | chat.completions | responses |
|---|---|---|
| Method | `client.chat.completions.create` | `client.responses.create` |
| Conversation | `messages: list[Message]` | `input: str | list[dict]` (stateless) |
| Token limit | `max_completion_tokens` | `max_output_tokens` |
| Output | `response.choices[0].message.content` | `response.output` (list of typed blocks) |
| Reasoning | `reasoning_effort` kwarg | `reasoning={"effort": "...", "summary": "..."}` |
| Tools | `tools=[...]` (Tool/callable) | `tools=[...]` (Tool dicts; alpha types supported) |
| Streaming | `client.chat.completions.stream(...)` → `ChatStream` | `create(stream=True)` → `AsyncIterable[ResponsesStreamEvent]` |
| Structured output | `parse()` / `response_format=` | Not directly supported; use chat.completions for typed parsing |

## Structured input

`input` accepts either a string or a list of structured input items (matching the OpenAI Responses API):

```python
response = await client.responses.create(
    model=...,
    input=[
        {"type": "message", "role": "system", "content": "..."},
        {"type": "message", "role": "user", "content": "What is 2+2?"},
    ],
    max_output_tokens=100,
)
```

For most cases, the plain-string form is enough. Use the structured form when you need to mix message types with reasoning blocks or function calls.

## Output blocks

`response.output` is a list of typed blocks. Common types:

- **`reasoning`** — the model's chain of thought (when reasoning is requested)
- **`message`** — the assistant's reply
- **`function_call`** — tool invocations
- **`web_search_call`** — when `web_search=True` triggered a search

```python
for block in response.output:
    btype = block.type if hasattr(block, "type") else block.get("type")
    if btype == "message":
        print("REPLY:", block.content if hasattr(block, "content") else block.get("content"))
    elif btype == "reasoning":
        print("REASONING:", block.summary)
    elif btype == "function_call":
        print(f"TOOL CALL: {block.name}({block.arguments})")
    elif btype == "web_search_call":
        print(f"SEARCHED: {block.query}; results: {len(block.results)}")
```

The exact attribute names depend on the SDK's typed wrappers — inspect `response.output[0]` interactively if uncertain.

## Reasoning

```python
response = await client.responses.create(
    model=await client.models.resolve_chat(require_reasoning=True),
    input="Plan how to set up a Postgres backup pipeline.",
    reasoning={"effort": "medium", "summary": "auto"},
    max_output_tokens=2000,
)
```

`reasoning.effort` requests a thinking depth: `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"`, `"max"`. What a tier does depends on the model. `supportsReasoningEffort` and `reasoningEffortOptions` say which values a model accepts, not that a higher tier thinks longer: some models scale with the tier, some use about the same reasoning tokens at every tier, and some have used fewer at `"high"` than at `"low"`. Measure `usage` at the tiers you plan to send before relying on an order.

## Web search (Venice extension)

Venice's Responses-API supports built-in web search:

```python
response = await client.responses.create(
    model=...,
    input="What were the major EU AI Act provisions effective last month?",
    web_search=True,                            # opt-in; returns web_search_call blocks
    max_output_tokens=500,
)
```

The output will include a `web_search_call` block describing what was searched, then a `message` block with the synthesized answer.

For more control, use `client.augment.search(...)` to run the search yourself and pass results into chat.completions — covered in `characters-and-augment.md`.

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

The streaming events have a different shape from `ChatStream` — Server-Sent-Events parsed into typed event objects. There's no `text_deltas()` / `collect_with_deltas()` helper here.

## Tools (alpha types)

```python
response = await client.responses.create(
    model=...,
    input="...",
    tools=[
        {"type": "web_search"},
        {"type": "x_search"},                   # Venice-specific
        {"type": "code_interpreter"},
        {"type": "file_search"},
        {"type": "function", "function": {...}},
    ],
    tool_choice="auto",
)
```

The alpha tool types are server-managed — you don't dispatch them yourself. For function tools, you handle dispatch (no `run_with_tools`-style loop here; that's a chat.completions feature).

## When to stay on chat.completions

- **Multi-turn conversations** — Responses is stateless; you'd have to thread the full input list every call. chat.completions handles this cleanly with `messages: list[Message]`.
- **Managed agent loops** — `run_with_tools` is on chat.completions.
- **Pydantic-typed structured output** — `parse()` is on chat.completions.
- **Streaming with usage** — `ChatStream.collect_with_deltas()` populates `final_response.usage`; the Responses streaming events don't expose this directly.
- **E2EE-capable models** — explicitly NOT supported on Responses; use chat.completions.

In short: chat.completions has more features and is more stable. Responses is for OpenAI-Responses-API parity and the alpha tool types.

## Common bugs

- **`max_completion_tokens=` on `client.responses.create`** — wrong kwarg here. Use `max_output_tokens`.
- **`messages=...` on `client.responses.create`** — wrong; use `input=...`.
- **Treating `response.choices[0].message.content`** — that's chat.completions shape; Responses uses `response.output[]` blocks.
- **Using Responses for multi-turn** — restate the full conversation as `input` every call. chat.completions is much simpler.
- **Using `client.chat.completions.run_with_tools` patterns on Responses** — `run_with_tools` is chat.completions-only.

## Related references

- `tool-loops.md` — the chat.completions agent-loop path, which is what you usually want.
- `streaming.md` — chat.completions streaming with `ChatStream`.
- `structured-output.md` — `parse()` is chat.completions only.
- `characters-and-augment.md` — `client.augment.search/scrape` as an alternative to Responses-API web_search.
