#!/usr/bin/env python3
"""
Venice AI SDK - Synchronous Client
==================================

``SyncVeniceClient`` calls the API from ordinary blocking code, with no
``async``/``await`` in sight. It runs the async client on a background thread
and has the same resource methods. This example shows the three shapes of call
that work with it:

- a chat completion;
- a chat stream, read with a plain ``for`` loop;
- a paginated ``.list()`` walked page by page.

What is still async
-------------------
The sync client turns coroutine methods into blocking ones and wraps streams.
Everything else comes back unchanged, so for these, use ``VeniceClient``:

- ``iter_*`` paginators such as ``characters.iter_all()``: they return an async
  ``Paginator``, and ``for item in paginator`` raises ``TypeError``. Walk the
  matching ``.list()`` by hand, as this example does.
- Job objects, such as the ``VideoJob`` from ``video.run()``: its methods stay
  coroutines. ``video.run()`` submits the job before it returns, so calling it
  here can start a paid job that blocking code cannot then drive.
- A sync stream has no ``collect()``, ``text_deltas()`` or ``final_response``.
  Put the text, the finish reason and the usage together from the chunks yourself.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"

Cost: two chat requests on the cheapest model that answers directly, capped at
``MAX_COMPLETION_TOKENS`` each, well under $0.001 in total. Resolving the model
and listing characters are free.

Exit status: ``0`` when the chat and the stream returned complete answers
(listing characters may print ``Section skipped:`` if the catalog has none),
``1`` for an empty or cut-off reply, one without usage to check, a page loop
that did not advance, or an API error, and ``77`` (with a ``SKIPPED:`` line)
when the catalog has no chat model that answers directly.
"""

import sys

from venice_ai import NoMatchingModelError, SyncVeniceClient, UserMessage, VeniceError
from venice_ai.types.api.requests import VeniceParameters

#: Plenty for a one-line answer from a model that answers directly.
MAX_COMPLETION_TOKENS = 64

#: Characters per page, and how many pages to read before stopping.
PAGE_SIZE = 5
MAX_PAGES = 2

#: Exit code for "skipped": a prerequisite is missing, not a failure.
EXIT_SKIPPED = 77

# Send only our own message, without Venice's default system prompt.
NO_SYSTEM_PROMPT = VeniceParameters(include_venice_system_prompt=False)


def cut_off(finish_reason: str | None, completion_tokens: int | None) -> bool:
    """True if a reply stopped at the cap, or if usage is missing so that can't be ruled out.

    Some backends report ``finish_reason="stop"`` for a reply that ran into the
    cap, so a reply that used the whole budget counts as cut off too.
    """
    return (
        finish_reason == "length"
        or completion_tokens is None
        or completion_tokens >= MAX_COMPLETION_TOKENS
    )


def chat_once(client: SyncVeniceClient, model: str) -> bool:
    """Make one blocking chat completion and check the reply is complete."""
    print("\n💬 Chat completion")
    print("-" * 40)

    response = client.chat.completions.create(
        model=model,
        messages=[UserMessage(content="Reply with one short greeting.")],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        venice_parameters=NO_SYSTEM_PROMPT,
    )

    finish_reason = response.choices[0].finish_reason if response.choices else None
    print(f"Reply: {response.text}")
    used = response.usage.completion_tokens if response.usage else None
    print(f"Finish reason: {finish_reason} ({used} of {MAX_COMPLETION_TOKENS} completion tokens)")
    if not response.text or cut_off(finish_reason, used):
        print("❌ The reply was empty or may be cut off by the token cap")
        return False
    print("✓ Complete reply")
    return True


def stream_once(client: SyncVeniceClient, model: str) -> bool:
    """Read a chat stream with a ``for`` loop and put the reply together.

    A sync stream has no ``collect()``, so this keeps the text, the finish
    reason and the usage itself. The usage arrives on the final chunk, which has
    no choices at all, so read it before checking for them.
    """
    print("\n🌊 Streaming")
    print("-" * 40)

    parts: list[str] = []
    finish_reason: str | None = None
    used: int | None = None
    chunks = 0
    with client.chat.completions.stream(
        model=model,
        messages=[UserMessage(content="Count from one to three in words.")],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        venice_parameters=NO_SYSTEM_PROMPT,
    ) as stream:
        for chunk in stream:
            chunks += 1
            if chunk.usage is not None:
                used = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                parts.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
            if choice.finish_reason:
                finish_reason = choice.finish_reason
    print()

    text = "".join(parts)
    print(
        f"Chunks: {chunks}, finish reason: {finish_reason} "
        f"({used} of {MAX_COMPLETION_TOKENS} completion tokens)"
    )
    if not text or finish_reason is None or cut_off(finish_reason, used):
        print("❌ The stream was empty, unfinished or may be cut off by the token cap")
        return False
    print("✓ Complete stream")
    return True


def list_character_pages(client: SyncVeniceClient) -> bool:
    """Walk ``characters.list()`` page by page, up to ``MAX_PAGES``. Free.

    The loop stops on a short page, which is how this endpoint signals the last
    one. It fails if a page is longer than asked for, or if a full page is
    followed by a page with nothing new, which would mean ``offset`` is having
    no effect.
    """
    print("\n📄 Paging through a list")
    print("-" * 40)

    seen: set[str] = set()
    for page in range(MAX_PAGES):
        # Both requests name the same order, so the offset walks one list.
        response = client.characters.list(
            limit=PAGE_SIZE, offset=page * PAGE_SIZE, sort_by="mostRecent"
        )
        slugs = [character.slug for character in response.data]
        print(f"Page {page + 1} (offset {page * PAGE_SIZE}): {slugs}")
        if len(slugs) > PAGE_SIZE:
            print(f"❌ Asked for {PAGE_SIZE} characters per page, got {len(slugs)}")
            return False
        if page > 0 and slugs and seen.issuperset(slugs):
            print("❌ This page repeats the previous one, so the offset had no effect")
            return False
        seen.update(slugs)
        if len(slugs) < PAGE_SIZE:
            break

    if not seen:
        print("Section skipped: the catalog returned no characters to page through")
        return True
    print(f"✓ Read {len(seen)} characters")
    return True


def main() -> int:
    """Run each kind of call on one sync client and return the exit code."""
    print("🔁 Venice AI Synchronous Client Example")
    print("=" * 40)

    # A `with` block closes the client and stops its background thread.
    with SyncVeniceClient() as client:
        # Coroutine methods on any resource are blocking calls here, so this
        # needs no await. Without exclude_reasoning=True the cheapest pick can
        # think through its whole token cap before writing the visible reply.
        try:
            model = client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
        except NoMatchingModelError as e:
            print(f"SKIPPED: no chat model in the catalog answers directly ({e})")
            return EXIT_SKIPPED
        print(f"Model: {model}")

        results = [
            chat_once(client, model),
            stream_once(client, model),
            list_character_pages(client),
        ]

    if not all(results):
        print("\n❌ One or more checks failed")
        return 1
    print("\n✅ All checks passed; see the docstring for what still needs VeniceClient")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except (VeniceError, ValueError) as e:
        # ValueError: no API key was found.
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
