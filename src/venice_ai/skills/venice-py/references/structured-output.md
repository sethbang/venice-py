# Structured output (Pydantic-typed responses)

Sourced from `ChatCompletions.parse` in `src/venice_ai/resources/chat/completions.py` and the `parsed` / `parse_as` members of `src/venice_ai/types/api/chat.py`.

`client.responses` has no `parse()`. Its `text=` argument sends a JSON Schema you write by hand and returns a JSON string; see `responses-api.md`.

## Two paths — pick `parse()` for new code

| Path | Method | Returns | Validates server-side response? | Streaming? |
|---|---|---|---|---|
| **`parse()`** (recommended) | `client.chat.completions.parse(model, messages, response_format=Cls)` | `ParsedChatCompletion[T]` with `.parsed` as the typed instance | yes (via Pydantic, raises `ValidationError` on the call) | no — explicitly unsupported |
| **`create()` + `parse_as()`** (lower-level) | `client.chat.completions.create(model, messages, response_format=Cls)` then `response.parse_as(Cls)` | `Cls` (typed instance) | yes (parse_as validates on access) | yes — works with streaming |

In both cases:
- Pass the **Pydantic class** (NOT a hand-written JSON Schema dict) as `response_format=`.
- The SDK builds the JSON Schema, sends it as `response_format={"type": "json_schema", ...}`, validates the model's reply.
- Errors surface as `pydantic.ValidationError` at the call site rather than as bad JSON downstream.

## `parse()` — the canonical pattern

```python
import asyncio
from typing import List
from pydantic import BaseModel
from venice_ai import VeniceClient
from venice_ai.types.api import UserMessage, SystemMessage


class LineItem(BaseModel):
    description: str
    amount_usd: float


class Invoice(BaseModel):
    vendor: str
    line_items: List[LineItem]
    total_usd: float


async def main() -> None:
    async with VeniceClient() as client:
        result = await client.chat.completions.parse(
            model=await client.models.resolve_chat(require_response_schema=True),
            messages=[
                SystemMessage(content="Extract structured invoice data from raw email text."),
                UserMessage(content=f"Extract from:\n\n{RAW_EMAIL}"),
            ],
            response_format=Invoice,
        )
        invoice: Invoice = result.parsed                        # typed instance — already validated
        print(invoice.model_dump())
        # result.response — the underlying ChatCompletionResponse if you need usage/headers
```

`result.parsed` IS the typed `Invoice` instance. No manual `model_validate`, no `json.loads`, no re-validation needed.

`parse()` accepts a `schema_name` kwarg (defaults to `Invoice.__name__`) and a `strict=True` kwarg (default; sets `strict: true` in the JSON Schema payload so the API rejects non-conformant responses rather than silently returning bad JSON). Other kwargs (`temperature`, `max_completion_tokens`, `tools`, etc.) forward unchanged to `create()`. **`stream=True` raises `ValueError`** — `parse()` doesn't support streaming.

## `create()` + `parse_as()` — the lower-level path

```python
response = await client.chat.completions.create(
    model=await client.models.resolve_chat(require_response_schema=True),
    messages=[UserMessage(content=f"Extract from:\n{RAW_EMAIL}")],
    response_format=Invoice,
)
invoice: Invoice = response.parse_as(Invoice)
```

Use this when:
- You need streaming: use `client.chat.completions.stream(...)` (returns a `ChatStream`), then `collected = await stream.collect()` and `collected.parse_as(Cls)`. (`create(stream=True)` returns a bare `Stream` without `.collect()`.)
- You want full control over the response object (e.g., to read `response.headers` or `response.response_rate_limits` mid-flow).
- You're mid-refactor and the existing code uses `create()`.

## `response.parsed` vs `response.parse_as(Cls)` — the distinction that bites

On a bare `ChatCompletionResponse` (returned from `create()`):

| Attribute / Method | Returns | When to use |
|---|---|---|
| `response.parsed` | **raw `dict \| list \| None`** — JSON-parses the first choice's content | When you don't have a Pydantic class (or don't want one). |
| `response.parse_as(Cls, choice_index=0)` | **typed `Cls` instance** — validates against your model | When you want the typed result. |

Treating `response.parsed` as a typed instance and calling `.model_dump()` on it raises `AttributeError: 'dict' object has no attribute 'model_dump'`.

The distinction does NOT exist on `ParsedChatCompletion[T]` (returned from `parse()`) — there, `result.parsed` IS the typed instance. The shape is asymmetric on purpose: `parse()` is the auto-validating sibling and `result.parsed` is its main API.

## Nested models work as expected

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Customer(BaseModel):
    name: str
    email: str
    address: Address                    # nested — Pydantic builds the schema recursively

result = await client.chat.completions.parse(
    model=...,
    messages=...,
    response_format=Customer,
)
print(result.parsed.address.country)
```

Lists of models (`list[LineItem]`), `Optional[T]`, `Literal[...]`, enums, `Union[A, B]` (rendered as `oneOf`) all work — anything Pydantic can produce a JSON Schema for.

## Validation failure handling

`pydantic.ValidationError` raises at the call site. The model's raw text is in `e.input_value` (sometimes useful for logging) and the validation issues in `e.errors()`. Common causes:

- **Field type mismatch** — model returned a string where you typed an int.
- **Missing required field** — the prompt didn't lead the model to populate it.
- **Strict-mode rejection** — `strict=True` is the default; with very small or under-trained models, you may see more frequent failures here. Try `strict=False` if the model can't reliably hit the schema.

```python
from pydantic import ValidationError

try:
    result = await client.chat.completions.parse(...)
except ValidationError as e:
    log.error("structured-output validation failed", errors=e.errors(), raw=e.input_value)
    # Fall back: ask again with stricter system prompt, or downgrade to dict via create() + .parsed
```

## When to use `JSONSchemaFormat` / `JSONObjectFormat`

You may not have a Pydantic class — e.g., schema comes from an external source, or you need `additionalProperties: false` on a deep tree.

```python
from venice_ai.types.api.requests.common import JSONSchemaFormat, JSONObjectFormat

response = await client.chat.completions.create(
    model=...,
    messages=...,
    response_format=JSONSchemaFormat(
        json_schema={
            "name": "Invoice",
            "schema": {
                "type": "object",
                "properties": {
                    "vendor": {"type": "string"},
                    "total_usd": {"type": "number"},
                },
                "required": ["vendor", "total_usd"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ),
)
data = response.parsed                  # raw dict — no Pydantic class to validate against
```

Or for "just give me valid JSON" with no schema, use `JSONObjectFormat()`. The model is constrained to emit JSON but not to a particular shape.

## `require_response_schema` capability filter

When resolving the model, prefer `client.models.resolve_chat(require_response_schema=True)`. Some Venice models support tool calls but not strict response_format mode; this filter excludes them.

```python
model = await client.models.resolve_chat(require_response_schema=True)
result = await client.chat.completions.parse(model=model, ...)
```

## Common bugs

- **Hand-writing the JSON Schema dict** when you have a Pydantic class — pass the class directly to `response_format=`.
- **Calling `.model_dump()` on `response.parsed`** — `.parsed` is a dict on `ChatCompletionResponse`. Use `parse_as(Cls)`.
- **Passing `stream=True` to `parse()`** — raises `ValueError`. Use `create(stream=True, response_format=Cls)` if you genuinely need streaming.
- **Forgetting `require_response_schema=True`** — picking a model without strict response_format support → degraded reliability.
- **Catching bare `Exception` around `parse()`** — drops `ValidationError`'s structured info. Catch `pydantic.ValidationError` specifically.

## Related references

- `tool-loops.md` — `tool_from_model` for typed tool arguments (similar Pydantic-to-schema flow, but for tool calls).
- `model-resolution.md` — capability filters including `require_response_schema`.
- `streaming.md` — `parse()` doesn't stream; `create(stream=True, response_format=Cls)` does.
