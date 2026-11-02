# Characters and the Augment API

Sourced from `src/venice_ai/resources/characters.py` and `src/venice_ai/resources/augment.py`. Two distinctly Venice-flavored capabilities — pre-built AI personas (Characters) and web/document tooling (Augment).

## Characters — pre-configured AI personas

`client.characters` exposes Venice's catalog of pre-built personas (think: "Socrates", "uncensored-anime-girl", "polite-customer-support-bot"). Each character has a `slug`, a `modelId`, a system prompt, optional artwork, and metadata.

### Listing and getting

```python
async with VeniceClient() as client:
    catalog = await client.characters.list()
    for character in catalog.data:
        print(f"{character.slug}: {character.name} — {character.description[:80]}")

    resp = await client.characters.get(slug="socrates")
    socrates = resp.data                            # CharacterResponse wraps the Character under .data
    print(socrates.modelId, socrates.shareUrl)
```

`list()` returns `CharactersListResponse{data: list[Character]}`. `get(slug=...)` returns a `CharacterResponse{object, data}` — the actual `Character` lives under `.data` (so read `resp.data.modelId`, not `resp.modelId`).

A `Character` has these key fields (verify against current SDK; field names occasionally evolve):

| Field | Description |
|---|---|
| `slug` | URL-safe identifier; pass to `venice_parameters` |
| `name` | Display name |
| `description` | Short blurb |
| `modelId` | The model the character is configured for |
| `photoUrl` | Avatar URL |
| `shareUrl` | Public Venice page for the character |
| `tags` | Discovery metadata (list of strings) |
| `stats` | Usage/ratings nested object — `stats.averageRating`, `stats.ratingCount`, `stats.userRating`, `stats.imports` |

### Using a character in chat

Characters aren't called via `client.characters.create_chat(...)` — they're invoked through `client.chat.completions.create(...)` by passing the character slug in `venice_parameters`:

```python
from venice_ai.types.api.requests.common import VeniceParameters

response = await client.chat.completions.create(
    model="...",                                # or use character.modelId
    messages=[UserMessage(content="Teach me about justice.")],
    venice_parameters=VeniceParameters(character_slug="socrates"),
)
```

Venice prepends the character's system prompt to the conversation. You can still add your own `SystemMessage` for additional context — it stacks with the character's.

### When to use a character vs a custom system prompt

- **Use a character** when there's a pre-built one matching your need (saves prompt-engineering work).
- **Custom system prompt** when you need a specific business voice / domain that no character matches.
- **Character + extra SystemMessage** when you want the character's personality with your domain context layered on top.

### Reviews

```python
reviews = await client.characters.reviews("socrates", page=1, page_size=10)
for r in reviews.data:
    print(r.rating, r.message[:100])               # review text is `.message`, not `.comment`
```

Useful for evaluating character quality before using one in a production app.

## Augment — web search, scrape, and document parsing

`client.augment` provides three retrieval primitives that pair well with chat completions for RAG-style workflows.

### `augment.search` — web search

```python
response = await client.augment.search(
    query="EU AI Act 2026 enforcement timeline",
    limit=10,                                   # int, 1-20, default 10
    search_provider="brave",                    # "brave" (default, ZDR), "exa" (ZDR) or "google" (proxied)
)
for r in response.results:
    print(r.title, r.url, r.content[:100])         # result text is `.content`, not `.snippet`
```

`search_provider` choices:
- **`"brave"`** (default) — Zero-Data-Retention privacy. Recommended.
- **`"exa"`** — Exa search, also Zero-Data-Retention.
- **`"google"`** — proxied / anonymized through Venice. Useful when Brave's results are sparse for your query.

Combine with chat for RAG:

```python
search_resp = await client.augment.search(query=user_question, limit=5)
context = "\n\n".join(f"[{r.title}]({r.url})\n{r.content}" for r in search_resp.results)

chat_resp = await client.chat.completions.create(
    model=...,
    messages=[
        SystemMessage(content=f"Answer using only the following sources:\n\n{context}"),
        UserMessage(content=user_question),
    ],
)
```

### `augment.scrape` — single-page scrape

```python
response = await client.augment.scrape(url="https://example.com/article")
print(response.text)                            # markdown / plain text
```

Returns the scraped content as text. Useful for "the user gave me a URL; let me read it" workflows.

For multi-page crawls, this isn't the right tool — use a dedicated scraper (firecrawl, playwright, etc.) and feed text into chat.

### `augment.parse_text` — local document → text

```python
from pathlib import Path

# JSON response (default) — gives you text + token count
result = await client.augment.parse_text(
    file=Path("./contract.pdf"),                # str | bytes | BinaryIO | Path
    response_format="json",                     # default
)
print(result.text)
print(f"Tokens: {result.tokens}")

# Plain-text response
text = await client.augment.parse_text(
    file=Path("./contract.pdf"),
    response_format="text",                     # returns str directly
)
```

Supports PDF, DOCX, XLSX, TXT, RTF, ODT, HTML — basically anything LibreOffice can read. ≤25 MB per file. Server-side parsing; no local heavy-lifting needed.

For raw bytes (e.g., from an upload):

```python
result = await client.augment.parse_text(
    file=uploaded_bytes,
    content_type="application/pdf",             # MIME hint
    filename="contract.pdf",                    # filename hint for server logs
)
```

`content_type` and `filename` are useful when uploading bytes without a path — the SDK's auto-detection can fall back to magic-byte sniffing, but explicit hints are more reliable.

### Augment + chat: the canonical RAG loop

```python
async def answer_with_search(question: str) -> str:
    async with VeniceClient() as client:
        # 1. Retrieve
        search = await client.augment.search(query=question, limit=5)
        sources = "\n\n".join(
            f"<<{r.title}>>\n{r.content}\nURL: {r.url}"
            for r in search.results
        )

        # 2. Generate with retrieved context
        response = await client.chat.completions.create(
            model=await client.models.resolve_chat(),
            messages=[
                SystemMessage(content=(
                    f"Answer the question using ONLY the sources below. "
                    f"Cite sources by URL. If the sources don't have the answer, say so.\n\n"
                    f"SOURCES:\n{sources}"
                )),
                UserMessage(content=question),
            ],
            max_completion_tokens=500,
        )
        return response.text
```

This is a 2-call RAG pattern. For multi-turn agent-style RAG (where the model decides when to search, what to search for, etc.), wrap `augment.search` as a tool and use `run_with_tools`:

```python
from venice_ai import tool_from_function   # Note: search wrapper passed as bare callable below

async def web_search(query: str) -> str:
    """Search the web for current information.

    Returns top results with title, URL, and content extract.
    """
    response = await client.augment.search(query=query, limit=5)
    return "\n\n".join(f"{r.title} ({r.url})\n{r.content}" for r in response.results)

result = await client.chat.completions.run_with_tools(
    model=await client.models.resolve_chat(require_function_calling=True),
    messages=[UserMessage(content=user_question)],
    tools=[web_search],                         # bare callable
    max_iterations=5,
)
```

The agent decides when to invoke `web_search`. The closure on `client` makes it work; for production, pass `client` as a kwarg or use a class-based agent.

## Common bugs

- **Calling `client.characters.create_chat(slug="...")`** — that's not how characters work. Pass `venice_parameters=VeniceParameters(character_slug=...)` to `client.chat.completions.create`.
- **Hardcoding character slugs** — verify with `client.characters.list()`; characters can be added/removed.
- **`augment.parse_text(file=open(...))` without explicit `response_format`** — defaults to `"json"`; you get `AugmentTextParserResponse`, not a string. Use `response_format="text"` if you want raw text.
- **Using `augment.scrape` for 100-page crawls** — single-page only; use a real crawler for bulk.
- **Looking for a `.snippet` field on search results** — there isn't one; the field is `.content` (a short extract, not the full page). Use `augment.scrape(url=r.url)` to get the full page.

## Related references

- `tool-loops.md` — wrapping `augment.search` as a tool for agent RAG.
- `responses-api.md` — Venice's Responses API also supports built-in `web_search=True` (server-driven; less control).
- `model-resolution.md` — `resolve_chat()` resolves a default model; pair with character slugs for "best chat model + this persona".
- `venice-py-x402/SKILL.md` — using characters from autonomous agents (the character slug works the same in mode-2 SIWE-auth chat calls).
