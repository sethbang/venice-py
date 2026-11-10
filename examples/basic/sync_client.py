#!/usr/bin/env python3
"""
Venice AI SDK - Synchronous Client
==================================

``SyncVeniceClient`` calls the API from ordinary blocking code, with no
``async``/``await`` in sight. It runs the async client on a background thread
and has the same resource methods. This example shows three shapes of call:

- a chat completion;
- a chat stream, read with a plain ``for`` loop into one assembled response;
- an ``iter_*`` paginator, walked with a ``for`` loop and left early.

What a method returns works without ``await`` too: streams with their
``collect()``, ``text_deltas()``, ``collect_with_deltas()`` and
``final_response``, ``iter_*`` paginators, streamed audio from
``audio.create_speech(stream=True)`` and ``audio.stream_long_text()``, and job
objects such as the one ``video.run()`` returns, whose methods block and which a
plain ``with`` releases.

An ``on_progress`` callback passed to a job's ``wait()`` runs on the client's
loop thread. Keep it short and do not call the client from it, which raises
``RuntimeError``.

What stays async
----------------
- ``client.gather()`` takes awaitables, which blocking calls never produce. Use
  ``VeniceClient`` to run requests concurrently.
- A ``VideoJob`` built by hand, say from a saved queue ID, needs the async client.
  In blocking code, use ``client.video.retrieve()`` and ``client.video.cancel()``.
  A job on a ``download_url`` model is not known to ``/video/complete``: fetch its
  saved link and send ``DELETE`` to it yourself.

Prerequisites:
- Install: pip install venice-py
- Set API key: export VENICE_API_KEY="your-api-key"

Cost: two chat requests on the cheapest model that answers directly, capped at
``MAX_COMPLETION_TOKENS`` each, well under $0.001 in total. Resolving the model
and listing characters are free.

Exit status: ``0`` when the chat and the stream returned complete answers
(listing characters may print ``Section skipped:`` if the catalog has none),
``1`` for an empty or cut-off reply, one without usage to check, a stream with
no finish reason, a walk that repeats a character, or an API error, and ``77``
(with a ``SKIPPED:`` line) when the catalog has no chat model that answers
directly.
"""

import sys

from venice_ai import NoMatchingModelError, SyncVeniceClient, UserMessage, VeniceError
from venice_ai.types.api import ChatCompletionResponse
from venice_ai.types.api.requests import VeniceParameters

#: Plenty for a one-line answer from a model that answers directly.
MAX_COMPLETION_TOKENS = 64

#: Characters per page, and how many to read before leaving the walk: two pages.
PAGE_SIZE = 5
MAX_CHARACTERS = 2 * PAGE_SIZE

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


def reply_complete(response: ChatCompletionResponse) -> bool:
    """Print the finish reason and usage, and check the reply is complete."""
    finish_reason = response.choices[0].finish_reason if response.choices else None
    used = response.usage.completion_tokens if response.usage else None
    print(f"Finish reason: {finish_reason} ({used} of {MAX_COMPLETION_TOKENS} completion tokens)")
    if not response.text or cut_off(finish_reason, used):
        print("❌ The reply was empty or may be cut off by the token cap")
        return False
    print("✓ Complete reply")
    return True


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
    print(f"Reply: {response.text}")
    return reply_complete(response)


def stream_once(client: SyncVeniceClient, model: str) -> bool:
    """Print a chat stream as it arrives, then check the assembled response.

    ``collect_with_deltas()`` yields the text and leaves the whole response, with
    its finish reason and usage, on ``final_response``. To skip the live text,
    ``stream.collect()`` returns that response directly. Both raise ``ValueError``
    for a stream that ends without a finish reason.
    """
    print("\n🌊 Streaming")
    print("-" * 40)

    try:
        with client.chat.completions.stream(
            model=model,
            messages=[UserMessage(content="Count from one to three in words.")],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            venice_parameters=NO_SYSTEM_PROMPT,
        ) as stream:
            for text in stream.collect_with_deltas():
                print(text, end="", flush=True)
            print()
    except ValueError as e:
        print(f"❌ The stream did not finish: {e}")
        return False
    return reply_complete(stream.final_response)


def walk_characters(client: SyncVeniceClient) -> bool:
    """Walk ``characters.iter_all()`` with a ``for`` loop and leave it early. Free.

    The paginator fetches each page only when the walk reaches it, so stopping at
    ``MAX_CHARACTERS`` never asks for a third page. A character seen twice fails
    the check: it would mean the pages are not advancing.
    """
    print("\n📄 Walking a paginator")
    print("-" * 40)

    seen: set[str] = set()
    # Naming the order keeps every page on one list.
    for character in client.characters.iter_all(page_size=PAGE_SIZE, sort_by="mostRecent"):
        if character.slug in seen:
            print(f"❌ {character.slug} came back again, so the pages are not advancing")
            return False
        seen.add(character.slug)
        print(f"{len(seen)}. {character.slug}")
        if len(seen) == MAX_CHARACTERS:
            break

    if not seen:
        print("Section skipped: the catalog returned no characters to walk")
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
            walk_characters(client),
        ]

    if not all(results):
        print("\n❌ One or more checks failed")
        return 1
    print("\n✅ All checks passed; see the docstring for what stays async")
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
