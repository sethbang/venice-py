#!/usr/bin/env python3
"""
Venice AI SDK - Chat File Inputs
================================

Venice supports an OpenAI-compatible ``type: file`` content part: you attach a
document to a user message and the server extracts its text for the model to
read. Plain-text formats (txt, md, csv, json, source code) are the most
reliable; the server also accepts office and PDF documents.

Two ways to supply the file via ``UserMessage.builder().file(...)``:

1. **Inline (``data:`` URL)** — base64-encode the bytes into a
   ``data:<mime>;base64,<...>`` URL. Best for local files.
2. **Public URL** — pass an ``https://...`` URL the API fetches for you. This
   example attaches RFC 2324 from the RFC Editor, a small, permanently
   published plain-text document. Pass a different URL as the first
   command-line argument to try your own document (at least a few pages long,
   so the extraction check below can tell its text arrived).

Both sections prove the model actually read the attachment: the inline note
holds a fact that exists nowhere else, and the public document must raise the
prompt-token count by thousands of tokens (the extracted text) and yield
details from the document itself.
"""

import asyncio
import base64
import sys

from venice_ai import NoMatchingModelError, VeniceClient, VeniceError
from venice_ai.types.api import ChatCompletionResponse, SystemMessage, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# A small Markdown document with a fact the model can't know otherwise.
_DOC_MARKDOWN = """\
# Internal Field Note — Project Halcyon

- Project codename: **Halcyon**
- Lead engineer: Marisol Okonkwo
- Launch window: the third Tuesday of November 2027
- Secret build token: HX-4417-ZULU

This note is confidential and exists only inside this document.
"""

# Default public document for the URL section, and details that only a reader
# of that document would produce together.
DEFAULT_PUBLIC_URL = "https://www.rfc-editor.org/rfc/rfc2324.txt"
_DEFAULT_URL_FACTS = ("2324", "Masinter")

# The extracted text of the default document is several thousand tokens; a
# fetch that silently failed adds almost nothing to the prompt.
_MIN_EXTRACTED_TOKENS = 2000

MAX_COMPLETION_TOKENS = 1024

# Leaving Venice's own system prompt out keeps the prompt down to the attachment
# and the question, so the token growth below is the extracted text alone.
PARAMS = VeniceParameters(include_venice_system_prompt=False)


def _markdown_data_url(markdown: str) -> str:
    """Encode markdown text as a ``data:`` URL for the file content part."""
    b64 = base64.b64encode(markdown.encode("utf-8")).decode("ascii")
    return f"data:text/markdown;base64,{b64}"


def _prompt_tokens(response: object) -> int:
    usage = getattr(response, "usage", None)
    return usage.prompt_tokens if usage is not None else 0


def _cut_off(response: ChatCompletionResponse) -> bool:
    """True if the answer stopped at ``MAX_COMPLETION_TOKENS`` (or usage is missing).

    Some models report an answer cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against the cap as well.
    """
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    return finish_reason == "length" or used is None or used >= MAX_COMPLETION_TOKENS


async def file_from_data_url(client: VeniceClient, model: str) -> int | None:
    """Attach a local document as a base64 ``data:`` URL and query it.

    Returns the request's prompt-token count on success, ``None`` on failure.
    """
    print("📎 File input via data: URL")
    print("-" * 40)

    data_url = _markdown_data_url(_DOC_MARKDOWN)

    # builder() lets you mix a file part and a text question in one message.
    user_message = (
        UserMessage.builder()
        .file(data_url, filename="field_note.md")
        .text("From the attached note, what is the secret build token?")
        .build()
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                SystemMessage(content="Answer using only the attached document."),
                user_message,
            ],
            venice_parameters=PARAMS,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.0,
        )
    except VeniceError as e:
        print(f"   ❌ Request failed ({type(e).__name__}): {e}")
        return None

    answer = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    print("   Q: secret build token?")
    print(f"   A: {answer}")
    print(f"   🏁 Finish reason: {finish_reason}")
    print(f"   📊 Prompt tokens: {_prompt_tokens(response)}")

    if _cut_off(response):
        print("   ❌ The answer may be cut off by max_completion_tokens")
        return None
    # Verify the model genuinely read the file rather than hallucinating.
    if "HX-4417-ZULU" not in answer:
        print("   ❌ The model did not return the token from the attached file")
        return None
    print("   ✅ Model extracted the token from the attached file.")
    return _prompt_tokens(response)


async def file_from_public_url(
    client: VeniceClient, model: str, url: str, baseline_prompt_tokens: int
) -> bool:
    """Attach a document by public URL and check that its text reached the model."""
    print("\n🌐 File input via public URL")
    print("-" * 40)
    print(f"   URL: {url}")

    user_message = (
        UserMessage.builder()
        .file(url)
        .text(
            "Using only the attached document: give its title, its RFC number (if any) "
            "and its author(s), then summarize it in two sentences."
        )
        .build()
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[user_message],
            venice_parameters=PARAMS,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.0,
        )
    except VeniceError as e:
        print(f"   ❌ Request failed ({type(e).__name__}): {e}")
        return False

    answer = (response.text or "").strip()
    finish_reason = response.choices[0].finish_reason if response.choices else None
    prompt_tokens = _prompt_tokens(response)
    extracted = prompt_tokens - baseline_prompt_tokens
    print("   Answer:")
    for line in answer.splitlines():
        print(f"      {line}")
    print(f"   🏁 Finish reason: {finish_reason}")
    print(f"   📊 Prompt tokens: {prompt_tokens} (~{extracted} more than the small inline note)")

    if _cut_off(response):
        print("   ❌ The answer may be cut off by max_completion_tokens")
        return False
    # A fetch or extraction failure is not an HTTP error: the model just says it
    # can't see the file. The prompt-token jump shows the text really arrived.
    if extracted < _MIN_EXTRACTED_TOKENS:
        print("   ❌ The document text did not reach the model (prompt barely grew)")
        return False
    if url == DEFAULT_PUBLIC_URL:
        missing = [fact for fact in _DEFAULT_URL_FACTS if fact not in answer]
        if missing:
            print(f"   ❌ The answer is missing details from the document: {missing}")
            return False
    print("   ✅ The server fetched the document and the model answered from it.")
    return True


async def main() -> int:
    """Run the chat file-input examples; return 1 if any section failed, 77 if no model."""
    print("🚀 Venice AI Chat File Inputs")
    print("=" * 50)

    public_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PUBLIC_URL

    async with VeniceClient() as client:
        # The cheapest model that answers directly; reasoning would only add
        # billed tokens to a lookup in the attached text.
        try:
            model = await client.models.resolve_chat(exclude_reasoning=True, prefer="cheapest")
        except NoMatchingModelError as e:
            print(f"SKIPPED: the catalog lists no non-reasoning chat model ({e})")
            return 77
        print(f"🤖 Using model: {model}\n")

        baseline = await file_from_data_url(client, model)
        results = [("data: URL", baseline is not None)]
        if baseline is None:
            print("\n⏭️  Skipping the public-URL section: it needs the inline baseline.")
            results.append(("public URL", False))
        else:
            ok = await file_from_public_url(client, model, public_url, baseline)
            results.append(("public URL", ok))

    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} sections failed: {', '.join(failed)}")
        return 1

    print("\n✨ File input examples completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - UserMessage.builder().file(data_url, filename=...)")
    print("   - Inline base64 data: URLs vs. public URLs")
    print("   - Mixing a file part with a text question in one message")
    print("   - Verifying extraction via answer content and prompt-token growth")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
