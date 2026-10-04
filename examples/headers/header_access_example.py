#!/usr/bin/env python3
"""
Venice AI SDK - Header Access Example
====================================

This example demonstrates how to access HTTP response headers and extract
useful information like rate limits, deprecation warnings, and the balance the
calling API key can still spend from Venice AI API responses.

Features demonstrated:
- Direct header access
- Rate limit information
- Model deprecation warnings
- Spendable balance for this API key (x-venice-balance-usd): the account
  balance capped by the key's own spend limit, read before the request is
  charged. The account balance itself is client.billing.get_balance().
- DIEM credit left this epoch (x-venice-balance-diem): the staking allowance,
  where each staked DIEM grants $1 of credit per epoch, reset at 00:00 UTC.
- Server build identification

Usage:
    python examples/headers/header_access_example.py
"""

import asyncio
import sys
from pathlib import Path

from venice_ai import NoMatchingModelError, VeniceClient, detect_image_format
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.requests import UserMessage, VeniceParameters


async def discover_models(client: VeniceClient):
    """Discover available models and categorize by type."""
    print("📋 Discovering available models...")

    try:
        models_response = await client.models.list(type="all")
        models = models_response.data

        # Categorize models by type
        models_by_type = {}
        for model in models:
            model_type = model.type or "unknown"
            if model_type not in models_by_type:
                models_by_type[model_type] = []
            models_by_type[model_type].append(model.id)

        print(f"✅ Found {len(models)} models across {len(models_by_type)} types")
        for model_type, model_list in models_by_type.items():
            print(f"   • {model_type.upper()}: {len(model_list)} models")

        return models_by_type

    except VeniceError as e:
        print(f"❌ Error discovering models: {type(e).__name__}: {e}")
        return {}


# Exit codes: 0 = every section that ran verified its headers, 1 = a failure,
# 77 = skipped (no model of any of the three types was available).
EXIT_SKIPPED = 77

# Where the generated test image is written (examples/results is gitignored).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


async def demonstrate_header_access() -> dict[str, bool | None]:
    """Demonstrate accessing headers from API responses.

    Returns one outcome per section: ``True`` when the call succeeded and its
    typed header accessors agreed with the raw headers, ``False`` on any
    failure, and ``None`` when the section was skipped because no model of
    that type is available.
    """
    print("🚀 Venice AI Header Access Example")
    print("=" * 50)

    outcomes: dict[str, bool | None] = {"chat": None, "image": None, "embeddings": None}

    async with VeniceClient() as client:
        # First discover available models
        models_by_type = await discover_models(client)

        if not models_by_type:
            print("❌ Could not discover models")
            return dict.fromkeys(outcomes, False)

        # Test with a text model for chat completions
        if models_by_type.get("text"):
            outcomes["chat"] = await _chat_section(client)
        else:
            print("\nSection skipped: chat (no text models in the catalog)")

        # Test with an image model
        if models_by_type.get("image"):
            outcomes["image"] = await _image_section(client)
        else:
            print("\nSection skipped: image (no image models in the catalog)")

        # Test with embeddings model
        if models_by_type.get("embedding"):
            outcomes["embeddings"] = await _embeddings_section(client)
        else:
            print("\nSection skipped: embeddings (no embedding models in the catalog)")

    return outcomes


async def _chat_section(client: VeniceClient) -> bool | None:
    # The headers are the point here, not the reply, so take the cheapest model
    # that answers directly. exclude_reasoning keeps a reasoning model from
    # spending the small token budget on thinking and returning no text.
    try:
        text_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    except NoMatchingModelError as e:
        print(f"\nSection skipped: chat (no suitable chat model: {e})")
        return None
    print(f"\n💬 Testing Chat Completion with {text_model}...")
    try:
        chat_response = await client.chat.completions.create(
            model=text_model,
            messages=[UserMessage(content="Say hello in five words or fewer.")],
            max_completion_tokens=64,
            venice_parameters=VeniceParameters(include_venice_system_prompt=False),
        )
    except VeniceError as e:
        print(f"❌ Chat completion error: {type(e).__name__}: {e}")
        return False

    ok = True
    finish_reason = chat_response.choices[0].finish_reason if chat_response.choices else None
    print(f"✅ Response received: {chat_response.text}")
    print(f"   finish_reason: {finish_reason}")
    if finish_reason == "length" or not (chat_response.text or "").strip():
        print("❌ Chat response was truncated or empty")
        ok = False
    # Chat responses carry request and token limits.
    if not demonstrate_response_headers(chat_response, "Chat Response", expect_tokens=True):
        ok = False
    return ok


async def _image_section(client: VeniceClient) -> bool | None:
    try:
        # The request below sets pixel sizes, so only consider models sized by
        # width/height (models sized by aspect ratio would ignore them).
        image_model = await client.models.resolve_image(prefer="cheapest", require_custom_size=True)
    except NoMatchingModelError as e:
        print(f"\nSection skipped: image (no suitable image model: {e})")
        return None
    print(f"\n🖼️ Testing Image Generation with {image_model}...")
    try:
        image_response = await client.image.create(
            model=image_model,
            prompt="A simple test image of a red apple on a white table",
            width=256,
            height=256,
            num_images=1,
        )
    except VeniceError as e:
        print(f"❌ Image generation error: {type(e).__name__}: {e}")
        return False

    ok = True
    if not image_response.images:
        print("❌ Image response contained no images")
        ok = False
    else:
        raw = image_response.bytes(0)
        ext, mime = detect_image_format(raw)
        if ext == "bin":
            print(f"❌ The image bytes are not a recognized image format ({len(raw):,} bytes)")
            ok = False
        else:
            path = image_response.save(RESULTS_DIR / f"header_access_image.{ext}", overwrite=True)
            print(f"✅ Image generated: {mime}, {len(raw):,} bytes, saved to {path}")

    # The image endpoint reports request limits only, not token limits.
    if not demonstrate_response_headers(image_response, "Image Response", expect_tokens=False):
        ok = False
    return ok


async def _embeddings_section(client: VeniceClient) -> bool | None:
    try:
        embed_model = await client.models.resolve_embedding(prefer="cheapest")
    except NoMatchingModelError as e:
        print(f"\nSection skipped: embeddings (no suitable embedding model: {e})")
        return None
    print(f"\n🔢 Testing Embeddings with {embed_model}...")
    try:
        embed_response = await client.embeddings.create(model=embed_model, input="Hello world")
    except VeniceError as e:
        print(f"❌ Embeddings error: {type(e).__name__}: {e}")
        return False

    ok = True
    if embed_response.data and embed_response.data[0].embedding:
        print(f"✅ Embedding created: {len(embed_response.data[0].embedding)} dimensions")
    else:
        print("❌ Embeddings response contained no vectors")
        ok = False
    if not demonstrate_response_headers(embed_response, "Embeddings Response", expect_tokens=True):
        ok = False
    return ok


def _raw(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive lookup of one raw header value."""
    return next((v for k, v in headers.items() if k.lower() == name), None)


def _check_accessors(response, headers: dict[str, str], expect_tokens: bool) -> list[str]:
    """Compare each typed accessor with the raw header it is parsed from.

    Returns a list of problems; an empty list means every accessor that has a
    raw header behind it parsed to the same value.
    """
    problems: list[str] = []
    rate_limits = response.response_rate_limits

    def compare_int(header: str, parsed: int | None) -> None:
        raw = _raw(headers, header)
        if raw is None:
            return
        if parsed is None or parsed != int(raw):
            problems.append(f"{header}={raw!r} but the typed accessor gave {parsed!r}")

    # Request limits are sent on every authenticated response.
    for header in ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"):
        if _raw(headers, header) is None:
            problems.append(f"{header} header missing")
    compare_int("x-ratelimit-limit-requests", rate_limits.limit_requests if rate_limits else None)
    compare_int(
        "x-ratelimit-remaining-requests", rate_limits.remaining_requests if rate_limits else None
    )
    if expect_tokens:
        for header in ("x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"):
            if _raw(headers, header) is None:
                problems.append(f"{header} header missing")
    compare_int("x-ratelimit-limit-tokens", rate_limits.limit_tokens if rate_limits else None)
    compare_int(
        "x-ratelimit-remaining-tokens", rate_limits.remaining_tokens if rate_limits else None
    )

    raw_usd = _raw(headers, "x-venice-balance-usd")
    if raw_usd is not None:
        balance = response.balance_info
        parsed_usd = balance.usd if balance else None
        if parsed_usd is None or abs(parsed_usd - float(raw_usd)) > 1e-9:
            problems.append(f"x-venice-balance-usd={raw_usd!r} but balance_info.usd={parsed_usd!r}")

    raw_warning = _raw(headers, "x-venice-model-deprecation-warning")
    if raw_warning is not None:
        deprecation = response.deprecation_info
        if deprecation is None or deprecation.warning != raw_warning:
            problems.append(
                "x-venice-model-deprecation-warning is set but deprecation_info missed it"
            )

    raw_version = _raw(headers, "x-venice-version")
    if raw_version is None:
        problems.append("x-venice-version header missing")
    elif response.venice_version != raw_version:
        problems.append(
            f"x-venice-version={raw_version!r} but venice_version={response.venice_version!r}"
        )
    return problems


def demonstrate_response_headers(response, response_type: str, *, expect_tokens: bool) -> bool:
    """Demonstrate header access for any response object.

    Returns False when the response carries no HTTP headers, or when a typed
    accessor disagrees with the raw header it is parsed from.
    """
    print(f"\n📊 {response_type} Headers:")
    print("-" * 30)

    # 1. Raw headers access
    headers = response.headers
    if headers:
        print(f"📋 Raw Headers ({len(headers)} total):")
        venice_headers = {k: v for k, v in headers.items() if k.lower().startswith("x-venice")}
        rate_headers = {k: v for k, v in headers.items() if k.lower().startswith("x-ratelimit")}

        if venice_headers:
            print("   Venice Headers:")
            for key, value in venice_headers.items():
                print(f"     • {key}: {value}")

        if rate_headers:
            print("   Rate Limit Headers:")
            for key, value in rate_headers.items():
                print(f"     • {key}: {value}")
            if "x-ratelimit-resets" in {k.lower() for k in rate_headers}:
                print(
                    "     (the unsuffixed x-ratelimit-remaining / -resets pair is the rolling\n"
                    "      30-second error budget, counted per model per key; a 429 from it\n"
                    "      sets RateLimitError.is_error_budget)"
                )
    else:
        print("❌ No headers available")
        return False

    # 2. Rate limit information
    rate_limits = response.response_rate_limits
    if rate_limits:
        print("\n🚦 Rate Limits:")
        if rate_limits.limit_requests:
            print(f"   • Requests: {rate_limits.remaining_requests}/{rate_limits.limit_requests}")
            if rate_limits.reset_requests:
                print(f"     Reset: {rate_limits.reset_requests}")

        if rate_limits.limit_tokens:
            print(f"   • Tokens: {rate_limits.remaining_tokens}/{rate_limits.limit_tokens}")
            if rate_limits.reset_tokens:
                # reset_tokens is an absolute Unix timestamp in seconds, not a
                # duration. The wait is max(0, reset_tokens - time.time()).
                print(f"     Reset (Unix epoch seconds): {rate_limits.reset_tokens}")
    else:
        print("ℹ️ No rate limit information available")

    # 3. Deprecation warnings
    deprecation = response.deprecation_info
    if deprecation and deprecation.is_deprecated:
        print("\n⚠️ DEPRECATION WARNING:")
        if deprecation.warning:
            print(f"   Message: {deprecation.warning}")
        if deprecation.date:
            print(f"   Date: {deprecation.date}")
    else:
        print("✅ No deprecation warnings")

    # 4. Balance information
    balance = response.balance_info
    if balance:
        # Both headers are read before this request is charged, so they
        # reflect every request up to, but not including, this one.
        print("\n💰 Balance headers (before this request was charged):")
        if balance.usd is not None:
            # USD is what this API key can still spend: the account balance,
            # capped by the key's consumption limit when it has one.
            print(f"   • USD this key can spend: ${balance.usd:.4f}")
        if balance.diem is not None:
            # DIEM is the staking allowance left this epoch (1 DIEM = $1 of
            # credit, reset at 00:00 UTC). The header can be absent, and
            # whether a key's DIEM limit caps it, as with USD, is unverified.
            print(f"   • DIEM credit left this epoch: {balance.diem:.4f} DIEM")
    else:
        print("ℹ️ No balance information available")

    # 5. Server build (x-venice-version identifies the server build that
    # handled the request; include it when reporting issues to Venice)
    version = response.venice_version
    if version:
        print(f"\n🏷️ Server build: {version}")
    else:
        print("ℹ️ No server build information")

    # 6. Check that every typed accessor matches its raw header.
    problems = _check_accessors(response, headers, expect_tokens)
    if problems:
        for problem in problems:
            print(f"❌ {problem}")
        return False
    print("✅ Typed accessors match the raw headers")
    return True


def show_header_properties_usage():
    """Show example code for using header properties."""
    print("\n📖 Header Properties Usage Examples:")
    print("-" * 40)

    code_examples = [
        (
            "Check Rate Limits",
            """
# Check if you're approaching rate limits
response = await client.chat.completions.create(...)
if response.response_rate_limits:
    remaining = response.response_rate_limits.remaining_requests
    if remaining and remaining < 10:
        print(f"Warning: Only {remaining} requests remaining!")
        """,
        ),
        (
            "Handle Deprecation",
            """
# Check for model deprecation
image_model = await client.models.resolve_image()
response = await client.image.create(model=image_model, ...)
if response.deprecation_info and response.deprecation_info.is_deprecated:
    print(f"Model deprecated: {response.deprecation_info.warning}")
    # Switch to alternative model
        """,
        ),
        (
            "Monitor Balance",
            """
# Monitor the USD this key can spend. The header shows it before the
# current request was charged, capped by the key's limit. DIEM is a daily
# staking allowance that resets at 00:00 UTC (and its header can be absent),
# so it is not a low-balance signal on its own.
response = await client.chat.completions.create(...)
balance = response.balance_info
if balance and balance.usd is not None and balance.usd < 1.0:
    print("Low balance warning!")
        """,
        ),
        (
            "Direct Header Access",
            """
# Access any header directly
response = await client.embeddings.create(...)
headers = response.headers
custom_header = headers.get('x-custom-header') if headers else None
        """,
        ),
    ]

    for title, code in code_examples:
        print(f"\n{title}:")
        print(code.strip())


async def main() -> int:
    """Main example function.

    Returns 0 if every section that ran verified its headers, 1 if any section
    failed, and 77 if every section was skipped for lack of a model.
    """
    outcomes = await demonstrate_header_access()
    show_header_properties_usage()

    print("\nKey Benefits:")
    print("• Easy access to rate limit information for throttling")
    print("• Automatic detection of deprecated models")
    print("• Spend monitoring: the balance header is read before each request is charged")
    print("• Access to all HTTP headers for debugging")

    failed = [name for name, result in outcomes.items() if result is False]
    skipped = [name for name, result in outcomes.items() if result is None]
    if failed:
        print(f"\n❌ Section(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    if len(skipped) == len(outcomes):
        print("\nSKIPPED: no chat, image or embedding model was available to call")
        return EXIT_SKIPPED
    if skipped:
        print(f"\nSection(s) skipped for lack of a model: {', '.join(skipped)}")
    passed = [name for name, result in outcomes.items() if result]
    print(f"\n✨ Header access verified for: {', '.join(passed)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Example cancelled!")
        sys.exit(130)
    except VeniceError as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have model access.", file=sys.stderr)
        sys.exit(1)
