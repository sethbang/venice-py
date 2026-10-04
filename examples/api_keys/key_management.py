#!/usr/bin/env python3
"""
Venice AI SDK - API Key Management and Monitoring
=================================================

This example demonstrates how to manage API keys in the Venice AI SDK:
- Listing and retrieving existing API keys
- Understanding API key metadata and usage statistics
- Monitoring rate limits and consumption patterns
- Managing key lifecycle (creation should be done carefully in production)
"""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import AuthenticationError, PermissionDeniedError, VeniceError
from venice_ai.types.api import ApiKey, ConsumptionLimit, CreateApiKeyRequest, UserMessage
from venice_ai.types.api.requests import VeniceParameters

# A clearly-named, throwaway label so the ephemeral key is unmistakable in the
# dashboard and trivially distinguishable from any real, pre-existing key.
EPHEMERAL_KEY_DESCRIPTION = "venice-sdk-example-ephemeral"

# Lifetime spend cap for the ephemeral key: a second safety net (alongside the
# expiry) should cleanup ever fail. Leave generous headroom above the demo
# call's cost; a cap of a few cents can be refused with a 402 up front.
EPHEMERAL_KEY_USD_LIMIT = 1.0

# Exit codes: 0 = every section verified, 1 = a failure, 77 = skipped because
# the key in VENICE_API_KEY is not an ADMIN key (key management needs one).
EXIT_SKIPPED = 77


class AdminKeyRequired(Exception):
    """The running key is valid but is not an ADMIN key."""


def _needs_admin_key(error: VeniceError) -> bool:
    """True when Venice refused an admin-only endpoint to a valid, non-admin key.

    Venice answers an INFERENCE key with 401 "Admin API key required" (an
    unknown key gets 401 "Authentication failed"), so the message tells the
    two apart. A 403 is treated the same way.
    """
    if isinstance(error, PermissionDeniedError):
        return True
    return isinstance(error, AuthenticationError) and "admin api key required" in str(error).lower()


def _describe_limits(key: ApiKey) -> str | None:
    """Return a readable summary of a key's consumption limits, or None.

    ``consumptionLimits`` can be missing, and when present it is an object
    whose fields may all be unset, so test the individual fields rather than
    the object alone.
    """
    limits = key.consumptionLimits
    if limits is None:
        return None
    parts = []
    if limits.usd is not None:
        parts.append(f"${limits.usd:,.2f} USD")
    if limits.diem is not None:
        parts.append(f"{limits.diem:,.2f} DIEM")
    if limits.vcu is not None:
        parts.append(f"{limits.vcu:,.2f} VCU")
    return ", ".join(parts) or None


def _format_age(delta: timedelta) -> str:
    """Render a timedelta as e.g. ``2d 3h 14m``."""
    total_minutes = int(delta.total_seconds() // 60)
    days, rem = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


async def list_existing_keys() -> bool:
    """List and analyze existing API keys."""
    print("📋 API Key Inventory")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # List all API keys
            keys = await client.api_keys.list()

            if keys:
                print(f"📊 Found {len(keys)} API key(s) in your account")

                for i, key in enumerate(keys, 1):
                    print(f"\n🔑 Key #{i}:")

                    key_id = key.id
                    description = key.description or "No description"
                    api_key_type = key.apiKeyType
                    created_at = key.createdAt
                    expires_at = key.expiresAt
                    last_used = key.lastUsedAt
                    usage = key.usage

                    print(f"   📛 ID: {key_id}")
                    print(f"   📝 Description: {description}")
                    print(f"   🏷️ Type: {api_key_type}")
                    print(f"   📅 Created: {created_at}")

                    # Show expiration if set
                    if expires_at:
                        print(f"   ⏰ Expires: {expires_at}")
                    else:
                        print("   ⏰ Expires: Never")

                    # Show last usage
                    if last_used:
                        print(f"   🕐 Last used: {last_used}")
                    else:
                        print("   🕐 Last used: Never")

                    # Show usage statistics if available
                    if usage:
                        trailing_days = usage.trailingSevenDays
                        if trailing_days:
                            usd_cost = trailing_days.usd or "0.00"
                            diem_usage = trailing_days.diem or "0"
                            print(f"   💰 Last 7 days: ${usd_cost} USD, {diem_usage} DIEM")
                        else:
                            print("   💰 Usage data available")
                    else:
                        print("   💰 No usage data available")

                    limits = _describe_limits(key)
                    print(f"   🚦 Consumption limit: {limits or 'none'}")

                    # Show partial key for identification
                    if key.last6Chars:
                        print(f"   🔒 Key ends with: ***{key.last6Chars}")

                limited = sum(1 for key in keys if _describe_limits(key))
                print(f"\n📊 {limited} of {len(keys)} key(s) have a consumption limit set")

            else:
                print("ℹ️ No API keys found in your account")

            return True

        except VeniceError as e:
            if _needs_admin_key(e):
                raise AdminKeyRequired(str(e)) from e
            print(f"❌ Error listing API keys: {e}")
            print("💡 Note: This requires a valid API key with appropriate permissions")
            return False


async def monitor_rate_limits() -> bool:
    """Monitor current rate limits and usage patterns."""
    print("\n📊 Rate Limit Monitoring")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get current rate limits
            rate_limits_response = await client.api_keys.get_rate_limits()

            print("🚦 Current Rate Limit Status:")

            rate_data = rate_limits_response.data

            # Show access status
            access_status = "✅ Permitted" if rate_data.accessPermitted else "❌ Denied"
            print(f"   🔓 API Access: {access_status}")

            # Show API tier info
            tier = rate_data.apiTier
            tier_status = "💳 Paid" if tier.isCharged else "🆓 Free"
            print(f"   🏷️ API Tier: {tier.id} ({tier_status})")

            # balances.USD is what THIS key can still spend: the account
            # balance, capped by what is left under the key's consumption
            # limit. It is not the account balance (billing.get_balance()).
            balances = rate_data.balances
            print(f"   💰 USD this key can still spend: ${balances.USD:.2f}")
            print(f"   🪙 DIEM reported alongside it: {balances.DIEM:.2f}")

            # Show next epoch time
            print(f"   ⏰ Next Rate Limit Reset: {rate_data.nextEpochBegins}")

            # Show rate limits by model
            print(f"   📊 Rate Limits by Model ({len(rate_data.rateLimits)} models):")

            for i, model_limit in enumerate(rate_data.rateLimits[:5]):  # Show first 5 models
                model_name = model_limit.apiModelId or f"Model {i + 1}"
                print(f"      🤖 {model_name}:")

                for limit in model_limit.rateLimits:
                    limit_type = limit.type
                    limit_amount = limit.amount
                    print(f"         📈 {limit_type}: {int(limit_amount):,}")

            if len(rate_data.rateLimits) > 5:
                remaining = len(rate_data.rateLimits) - 5
                print(f"      ... and {remaining} more models")

            return True

        except VeniceError as e:
            print(f"❌ Error getting rate limits: {e}")
            print("💡 Note: Rate limit monitoring requires API access")
            return False


async def check_rate_limit_history() -> bool:
    """Check recent rate limit violations."""
    print("\n📈 Rate Limit Violation History")
    print("-" * 40)

    async with VeniceClient() as client:
        try:
            # Get rate limit logs
            logs_response = await client.api_keys.get_rate_limit_logs()

            if logs_response.data:
                violations = logs_response.data

                print(f"⚠️ Found {len(violations)} recent rate limit violations")

                # Show recent violations
                for i, violation in enumerate(violations[:10], 1):  # Show first 10
                    print(f"\n🚨 Violation #{i}:")

                    print(f"   🤖 Model: {violation.modelId}")
                    print(f"   📊 Type: {violation.rateLimitType}")
                    print(f"   🕐 Time: {violation.timestamp}")

                if len(violations) > 10:
                    print(f"\n... and {len(violations) - 10} more violations")

                # Analyze patterns
                print("\n📊 Violation Analysis:")

                # Count by model
                model_counts = {}
                type_counts = {}

                for violation in violations:
                    model = violation.modelId
                    v_type = violation.rateLimitType

                    model_counts[model] = model_counts.get(model, 0) + 1
                    type_counts[v_type] = type_counts.get(v_type, 0) + 1

                print("   🎯 Most affected models:")
                for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]:
                    print(f"      {model}: {count} violations")

                print("   📈 Violation types:")
                for v_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"      {v_type}: {count} violations")

            else:
                print("✅ No recent rate limit violations found")
                print("👍 Nothing in the log suggests your usage is hitting limits")

            return True

        except VeniceError as e:
            print(f"❌ Error getting rate limit logs: {e}")
            print("💡 Note: Rate limit history requires API access")
            return False


async def demonstrate_key_creation_workflow() -> bool:
    """Show how to build key-creation requests (validated locally, not sent)."""
    print("\n🔧 API Key Creation Requests")
    print("-" * 40)
    print("📝 Building request objects only — the lifecycle section below sends a real one.")

    expires = (datetime.now(UTC) + timedelta(days=30)).date().isoformat()

    minimal = CreateApiKeyRequest(apiKeyType="INFERENCE", description="Example Production Key")
    limited = CreateApiKeyRequest(
        apiKeyType="INFERENCE",
        description="Temporary Capped Key",
        expiresAt=expires,
        consumptionLimit=ConsumptionLimit(usd=25.0),
        limitPeriod="MONTH",
    )

    print("\nOption 1 - Minimal request:")
    print(f"   {minimal.model_dump(exclude_none=True)}")
    print("\nOption 2 - With expiration and a monthly $25 spend cap:")
    print(f"   {limited.model_dump(exclude_none=True)}")

    print("\n```python")
    print("new_key = await client.api_keys.create(api_key_request=request)")
    print("secret_key = new_key.apiKey  # returned ONCE — store it securely")
    print("```")

    print("\n💡 Key Management Best Practices:")
    print("   🔐 Store API keys in environment variables or secure vaults")
    print("   📅 Set expiration dates for temporary keys")
    print("   🚦 Cap spend with consumptionLimit + limitPeriod")
    print("   📝 Use descriptive names to identify key purposes")
    print("   🗑️ Delete unused keys immediately")
    print("   🔄 Rotate keys periodically for security")
    print("   🚫 Never commit keys to version control")
    return True


async def analyze_key_details() -> bool:
    """Analyze the key this script is running with (matched by its last 6 chars)."""
    print("\n🔍 Detailed Key Analysis (the key in VENICE_API_KEY)")
    print("-" * 40)

    own_suffix = os.environ.get("VENICE_API_KEY", "")[-6:]

    async with VeniceClient() as client:
        try:
            keys = await client.api_keys.list()
            match = next((k for k in keys if own_suffix and k.last6Chars == own_suffix), None)
            if match is None:
                print("❌ The key in VENICE_API_KEY is not in this account's key list")
                return False

            print(f"🔍 Analyzing key: {match.id} (***{match.last6Chars})")
            key_data = await client.api_keys.retrieve(api_key_id=match.id)
        except VeniceError as e:
            if _needs_admin_key(e):
                raise AdminKeyRequired(str(e)) from e
            print(f"❌ Error analyzing key details: {e}")
            return False

    print("\n📊 Detailed Analysis:")
    print(f"   🆔 ID: {key_data.id}")
    print(f"   📝 Description: {key_data.description}")
    print(f"   🏷️ Type: {key_data.apiKeyType}")
    print(f"   📅 Created: {key_data.createdAt}")
    print(f"   🚦 Consumption limit: {_describe_limits(key_data) or 'none'}")
    limits = key_data.consumptionLimits
    period_usage = key_data.currentPeriodUsage
    if limits is not None and limits.usd is not None and period_usage is not None:
        used = float(period_usage.usd or 0)
        print(f"   📆 Limit period: {key_data.limitPeriod}")
        print(f"   📉 USD used this period: ${used:,.4f}")
        # The rate-limits endpoint and the x-venice-balance-usd header report
        # what THIS key can still spend: the account balance, capped by this.
        print(f"   💵 USD left under this key's limit: ${limits.usd - used:,.4f}")

    trailing = key_data.usage.trailingSevenDays if key_data.usage else None
    if trailing:
        print("\n📈 Usage Analytics:")
        usd = float(trailing.usd or 0)
        diem = float(trailing.diem or 0)
        print(f"   💰 7-day cost: ${usd:,.4f} (avg ${usd / 7:,.4f}/day)")
        print(f"   ⚡ 7-day DIEM: {diem:,.4f} (avg {diem / 7:,.4f}/day)")

    print("\n🔒 Security Information:")
    if key_data.lastUsedAt:
        last_time = datetime.fromisoformat(key_data.lastUsedAt.replace("Z", "+00:00"))
        age = _format_age(datetime.now(UTC) - last_time)
        print(f"   🕐 Last activity: {key_data.lastUsedAt} ({age} ago)")
    else:
        print("   ⚠️ Never used - consider removing if not needed")
    print(f"   ⏰ Expires: {key_data.expiresAt or 'never'}")

    return True


async def _use_new_key(secret: str) -> bool | None:
    """Authenticate a brand-new client with ``secret`` and make one small chat call.

    A new key can take a moment to propagate, so an authentication failure is
    retried briefly (it is not billed); any other error fails immediately.
    Returns ``None`` when the catalog has no chat model to call with it.
    """
    print("\n🔑 Using the new key for a small chat completion...")
    async with VeniceClient(api_key=secret) as new_client:
        chat_model = None
        resp = None
        for attempt in range(1, 4):
            try:
                # The call only proves the key works, so the cheapest model that
                # answers directly (no reasoning tokens) and no Venice system
                # prompt are enough.
                chat_model = chat_model or await new_client.models.resolve_chat(
                    prefer="cheapest", exclude_reasoning=True
                )
                resp = await new_client.chat.completions.create(
                    model=chat_model,
                    messages=[UserMessage(content="Reply with the single word: ok")],
                    max_completion_tokens=64,
                    venice_parameters=VeniceParameters(include_venice_system_prompt=False),
                )
                break
            except NoMatchingModelError as e:
                print(f"Section skipped: no chat model to exercise the new key with ({e})")
                return None
            except AuthenticationError:
                print(f"   ⏳ Attempt {attempt}/3: key not accepted yet, retrying...")
                await asyncio.sleep(3)
            except VeniceError as e:
                print(f"❌ Chat request with the new key failed: {e}")
                return False

    if resp is None:
        print("❌ The new key was never accepted (AuthenticationError on every attempt)")
        return False

    finish = resp.choices[0].finish_reason if resp.choices else None
    print(f"   🤖 Model: {chat_model}")
    print(f"   💬 Reply: {resp.text!r}")
    print(f"   🏁 finish_reason: {finish}")
    if finish == "length" or not (resp.text or "").strip():
        print("❌ The reply was truncated or empty")
        return False

    # An INFERENCE key may call inference endpoints only, so key management
    # must be refused for it. This is why the other sections need an ADMIN key.
    print("\n🚫 Checking that the INFERENCE key cannot manage keys...")
    async with VeniceClient(api_key=secret) as new_client:
        try:
            await new_client.api_keys.list()
        except (PermissionDeniedError, AuthenticationError) as e:
            if not _needs_admin_key(e):
                print(f"❌ Refused, but not for lacking admin rights: {e}")
                return False
            print(f"   ✅ Refused as expected: HTTP {e.status_code} {e}")
        except VeniceError as e:
            print(f"❌ Unexpected error instead of a refusal: {type(e).__name__}: {e}")
            return False
        else:
            print("❌ An INFERENCE key was allowed to list the account's API keys")
            return False
    return True


async def _delete_ephemeral_key(admin_client: VeniceClient, key_id: str) -> bool:
    """Delete ``key_id`` and confirm it no longer appears in the key list.

    Never raises for an API error, so a cleanup failure is reported next to the
    outcome of the steps before it instead of replacing it.
    """
    print(f"\n🗑️ Deleting ephemeral key id: {key_id}")
    try:
        result = await admin_client.api_keys.delete(api_key_id=key_id)
    except VeniceError as e:
        print(f"   ❌ Deletion failed: {e} — key {key_id} may persist!")
        return False
    if not result.success:
        print(f"   ❌ Deletion reported success=False — key {key_id} may persist!")
        return False
    print("   ✅ Deletion confirmed (success=True)")

    try:
        remaining = await admin_client.api_keys.list()
    except VeniceError as e:
        print(f"   ❌ Could not re-list keys to confirm the deletion: {e}")
        return False
    if any(key.id == key_id for key in remaining):
        print(f"   ❌ Key {key_id} is still listed after deletion!")
        return False
    print(f"   ✅ Key no longer listed ({len(remaining)} key(s) remain)")
    return True


async def demonstrate_key_lifecycle() -> bool | None:
    """Run a SAFE create -> use -> delete lifecycle on an ephemeral key.

    This exercises the two mutating endpoints (``create`` and ``delete``) end to
    end against a single, clearly-named throwaway key so it never collides with
    real credentials. The key is always deleted in a ``finally`` block — even if
    the "use" step fails — so it can never leak. Defense in depth: the key is
    also created with a near-term expiry and a small lifetime spend cap, so even
    a catastrophic cleanup failure leaves only a short-lived, capped key.

    The section fails unless the new key both authenticates and completes a
    small chat request, and the key is confirmed gone afterwards. It returns
    ``None`` (skipped) when there is no chat model to use the key with; the
    key is still created and deleted.
    """
    print("\n🔁 API Key Lifecycle: create → use → delete (live)")
    print("-" * 40)
    print("⚠️ This creates a REAL, ephemeral key and deletes it before exiting.")

    async with VeniceClient() as admin_client:
        # --- CREATE ---------------------------------------------------------
        # INFERENCE so the returned secret can actually be used below. The API
        # accepts a YYYY-MM-DD date and normalizes it to a timestamp; tomorrow
        # is the soonest whole-day expiry it allows.
        expires_at = (datetime.now(UTC) + timedelta(days=1)).date().isoformat()
        request = CreateApiKeyRequest(
            apiKeyType="INFERENCE",
            description=EPHEMERAL_KEY_DESCRIPTION,
            expiresAt=expires_at,
            consumptionLimit=ConsumptionLimit(usd=EPHEMERAL_KEY_USD_LIMIT),
            limitPeriod="LIFETIME",
        )
        print(f"\n🆕 Creating ephemeral key: '{EPHEMERAL_KEY_DESCRIPTION}'")
        try:
            created = await admin_client.api_keys.create(api_key_request=request)
        except VeniceError as e:
            print(f"❌ Key creation failed: {e}")
            return False

        used: bool | None = False
        try:
            # The secret (``created.apiKey``) is returned ONCE and never again.
            # Never print a full secret — only enough to identify it.
            print(f"   ✅ Created key id: {created.id}")
            print(f"   🏷️ Type: {created.apiKeyType}")
            print(f"   📝 Description: {created.description}")
            print(f"   ⏰ Expires: {created.expiresAt}")
            secret = created.apiKey
            print(
                f"   🔒 Secret (last 6): …{secret[-6:]} (the full secret is shown only at creation)"
            )

            try:
                stored = await admin_client.api_keys.retrieve(api_key_id=created.id)
            except VeniceError as e:
                print(f"❌ Could not read the new key back: {e}")
            else:
                print(
                    f"   🚦 Consumption limit on the server: {_describe_limits(stored) or 'none'}"
                )
                # --- USE ----------------------------------------------------
                used = await _use_new_key(secret)
        finally:
            # --- DELETE (always) ----------------------------------------------
            # Guaranteed cleanup: once creation succeeded the key MUST be
            # removed, whatever happened above.
            deleted = await _delete_ephemeral_key(admin_client, created.id)

    if not deleted:
        return False
    return used


async def main() -> int:
    """Run all API key management examples.

    Returns ``0`` only if every sub-section succeeded, ``1`` on any failure,
    and ``77`` when the running key is refused the admin-only key endpoints.
    """
    print("🚀 Venice AI API Key Management Examples")
    print("=" * 60)

    try:
        results: list[tuple[str, bool | None]] = [
            ("list_existing_keys", await list_existing_keys()),
            ("monitor_rate_limits", await monitor_rate_limits()),
            ("check_rate_limit_history", await check_rate_limit_history()),
            ("demonstrate_key_creation_workflow", await demonstrate_key_creation_workflow()),
            ("analyze_key_details", await analyze_key_details()),
            ("demonstrate_key_lifecycle", await demonstrate_key_lifecycle()),
        ]
    except AdminKeyRequired as e:
        print(f"\nSKIPPED: key management needs an ADMIN key in VENICE_API_KEY ({e})")
        return EXIT_SKIPPED

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]
    if failed:
        print(f"\n❌ {len(failed)} of {len(results)} section(s) failed: {', '.join(failed)}")
        return 1

    if skipped:
        print(f"\n{len(skipped)} of {len(results)} section(s) skipped: {', '.join(skipped)}")
        print(f"✨ The other {len(results) - len(skipped)} API key management sections completed.")
    else:
        print(f"\n✨ All {len(results)} API key management sections completed.")
    print("\n💡 Key concepts demonstrated:")
    print("   - Listing and inventorying API keys, including their consumption limits")
    print("   - Monitoring rate limits and rate-limit violations")
    print("   - Building CreateApiKeyRequest objects with expiry and spend caps")
    print("   - Inspecting the key this process is running with")
    print("   - A create → use → delete lifecycle with guaranteed cleanup")
    print("\n🔒 Security reminders:")
    print("   - Never share or commit API keys to version control")
    print("   - Rotate keys regularly and delete unused ones")
    print("   - Monitor usage for unexpected patterns")
    print("   - Use environment variables for key storage")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        print("Check that your API key is valid and you have appropriate access.", file=sys.stderr)
        sys.exit(1)
