#!/usr/bin/env python3
"""
Venice AI SDK - Production API Key Management
==============================================

This example demonstrates patterns for loading, validating and rotating API keys,
and lists what a production deployment has to add around them. The code here keeps
one key string in memory: its format check only looks at the length, it writes no
audit log, and rotation does not reach a client that is already built (a
``VeniceClient`` keeps the key it was created with, so build a new one).

Least privilege: give each service its own key with the narrowest settings it can
run on. The key guide recommends an INFERENCE key (``apiKeyType``) for anything that
only calls model endpoints, plus an expiry (``expiresAt``) and a spend cap
(``consumptionLimit`` with ``limitPeriod``) for temporary, shared or higher-risk use.
``ipAllowlist`` and ``modelPrivacy`` are also available to narrow a key further.
Creating a key needs an ADMIN key; see examples/api_keys/key_management.py.

Cost: one short chat request on the cheapest direct-answer model, well under $0.001.
Everything else here reads a free endpoint or runs locally.

Exit status: ``0`` when every demo passed, ``1`` when one failed or VENICE_API_KEY
is not set (the demos that need no key still print), ``77`` when the catalog has
no model of the kind the live client demo needs.
"""

import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from venice_ai import VeniceClient
from venice_ai.exceptions import (
    AuthenticationError,
    NoMatchingModelError,
    PermissionDeniedError,
    VeniceError,
)
from venice_ai.types.api import UserMessage, VeniceParameters

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def key_fingerprint(key: str) -> str:
    """Identify a key in logs without revealing it.

    Every Venice key shares the same leading characters, so a prefix such as
    ``key[:8]`` identifies nothing. A short hash of the whole key is unique per
    key and safe to log.
    """
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()[:12]


async def test_key(api_key: str) -> bool:
    """Return True if Venice accepts ``api_key`` and permits it inference access.

    The check must call an endpoint that requires authentication: ``/models``
    is public, so listing models succeeds with any key at all. Reading the
    key's rate limits is authenticated, free, and works for both admin and
    inference keys. A 200 alone is not enough: ``accessPermitted`` says whether
    the key may use the inference APIs. A 403 (``PermissionDeniedError``) also
    counts as a rejection: a key that cannot pass the check is not rotated in.
    """
    async with VeniceClient(api_key=api_key) as client:
        try:
            limits = await client.api_keys.get_rate_limits()
        except (AuthenticationError, PermissionDeniedError):
            return False
    return limits.data.accessPermitted


class SecureKeyManager:
    """
    Holds one API key loaded from the environment or a 0600 file, and swaps it
    for a new one once Venice accepts that key.

    Practices it shows:
    - Never hardcode API keys
    - Load keys from the environment or a restricted file
    - Test a new key against the API before switching to it
    """

    def __init__(self, env_var: str = "VENICE_API_KEY"):
        self.env_var = env_var
        self._key: str | None = None
        self._key_metadata: dict = {}

    def load_key(self) -> str | None:
        """Load API key from environment variable."""
        key = os.getenv(self.env_var)
        if key:
            self._key = key
            self._key_metadata = {
                "loaded_at": datetime.now().isoformat(),
                "source": "environment",
                "env_var": self.env_var,
            }
            print(f"✅ API key loaded from {self.env_var}")
            return key
        return None

    def load_from_file(self, key_file: Path) -> str | None:
        """
        Load API key from secure file.

        Security Notes:
        - File should have restrictive permissions (0600)
        - Should not be in version control
        - Consider using encrypted storage
        """
        try:
            if not key_file.exists():
                print(f"❌ Key file not found: {key_file}")
                return None

            # Check file permissions (Unix-like systems)
            if os.name == "posix":
                stat_info = key_file.stat()
                mode = stat_info.st_mode & 0o777
                if mode != 0o600:
                    print(f"⚠️  Warning: Key file permissions are {oct(mode)}, should be 0600")

            # Load key
            key = key_file.read_text().strip()
            self._key = key
            self._key_metadata = {
                "loaded_at": datetime.now().isoformat(),
                "source": "file",
                "file_path": str(key_file),
            }
            print(f"✅ API key loaded from file: {key_file}")
            return key

        except OSError as e:
            print(f"❌ Failed to load key from file: {e}")
            return None

    def validate_key(self, key: str) -> bool:
        """
        Validate API key format.

        Add your own validation logic based on Venice AI key format.
        """
        if not key or len(key) < 10:  # noqa: SIM103 — keep extension point below
            return False

        # Add specific validation for Venice AI key format
        # Example: check prefix, length, character set, etc.

        return True

    def get_key(self) -> str | None:
        """Get the current API key."""
        return self._key

    async def rotate_key(self, new_key: str) -> bool:
        """
        Rotate to a new API key, keeping the current key unless the new one works.

        This swaps the string this manager holds. A client built from the old
        key keeps using it, so build new clients from :meth:`get_key` afterwards.

        In production:
        1. Validate the new key's format
        2. Test the new key against the API
        3. Swap it in and record the rotation for audit
        4. Revoke the old key once every service uses the new one
        """
        if not self.validate_key(new_key):
            print("🛑 New key failed format validation; keeping the current key")
            return False

        if not await test_key(new_key):
            print("🛑 Venice rejected the new key or denied it access; keeping the current key")
            return False

        old_key = self._key
        self._key = new_key
        self._key_metadata.update(
            {
                "rotated_at": datetime.now().isoformat(),
                "previous_key": key_fingerprint(old_key) if old_key else None,
                "current_key": key_fingerprint(new_key),
            }
        )

        print("✅ API key rotated successfully")
        return True


async def environment_variable_pattern() -> bool | None:
    """Demonstrate loading API key from environment variables.

    Returns ``True`` when the key loaded, ``None`` (skipped) when it is not set.
    """
    print("🔐 Environment Variable Pattern")
    print("-" * 40)

    print("✅ Best Practice: Use environment variables")
    print()
    print("1. Set environment variable:")
    print("   export VENICE_API_KEY='your-api-key-here'")
    print()
    print("2. Load in application:")
    print("   ```python")
    print("   api_key = os.getenv('VENICE_API_KEY')")
    print("   if not api_key:")
    print("       raise ValueError('VENICE_API_KEY not set')")
    print("   client = VeniceClient(api_key=api_key)")
    print("   ```")
    print()

    # Demonstration
    manager = SecureKeyManager()
    key = manager.load_key()

    if key:
        print("✅ Key loaded successfully")
        print(f"   Key fingerprint: {key_fingerprint(key)}")
        print(f"   Metadata: {json.dumps(manager._key_metadata, indent=2)}")
    else:
        print("ℹ️  No key in environment (set VENICE_API_KEY to demonstrate)")
        return None

    return True


async def key_rotation_pattern() -> bool | None:
    """Demonstrate key rotation, both accepted and refused.

    First rotates to a key Venice accepts (the current key stands in for a
    freshly created one), which must pass the live check and be swapped in.
    Then loads a stand-in key that does not exist from a 0600 file; the live
    check must reject it and the manager must keep the current key. Returns
    ``False`` if either step behaves differently, and ``None`` (skipped) when
    VENICE_API_KEY is not set.
    """
    print("\n🔄 Key Rotation Pattern")
    print("-" * 40)

    print("✅ Production Key Rotation Strategy:")
    print()
    print("1. Create the new key with client.api_keys.create(...) or in the dashboard")
    print("   (see examples/api_keys/key_management.py)")
    print()
    print("2. Test the new key against an authenticated endpoint:")
    print("   ```python")
    print("   from venice_ai.exceptions import AuthenticationError, PermissionDeniedError")
    print()
    print("   async def test_key(api_key: str) -> bool:")
    print("       async with VeniceClient(api_key=api_key) as client:")
    print("           try:")
    print("               limits = await client.api_keys.get_rate_limits()")
    print("           except (AuthenticationError, PermissionDeniedError):")
    print("               return False")
    print("       return limits.data.accessPermitted")
    print("   ```")
    print("   Do not test with client.models.list(): /models is public and accepts any key.")
    print()
    print("3. Deploy new key with rollback capability:")
    print("   - Update environment variable")
    print("   - Restart services gradually")
    print("   - Monitor error rates")
    print("   - Keep old key active during transition")
    print()
    print("4. Revoke the old key after confirmation:")
    print("   - Verify all services use the new key")
    print("   - Delete the old key (client.api_keys.delete)")
    print("   - Record the rotation in your audit log")

    manager = SecureKeyManager()
    current = manager.load_key()
    if not current:
        print("\nℹ️  No key in environment (set VENICE_API_KEY to run the live rotation demo)")
        return None

    print("\n🧪 Live demo 1: rotating to a key Venice accepts")
    print("   (the current key stands in for a newly created one)")
    accepting = SecureKeyManager()
    accepting.load_key()
    accepted = await accepting.rotate_key(current)
    rotated_at = accepting._key_metadata.get("rotated_at")
    print(f"   Active key: {key_fingerprint(accepting.get_key() or '')}")
    if not accepted or rotated_at is None or accepting.get_key() != current:
        print("❌ A key Venice accepts was not rotated in")
        return False
    print(f"✅ Rotation accepted and recorded at {rotated_at}")

    print("\n🧪 Live demo 2: rotating to a key Venice will reject")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    key_file = RESULTS_DIR / "rotation_demo_key.txt"
    key_file.write_text("venice-demo-key-that-does-not-exist-0000\n")
    key_file.chmod(0o600)
    try:
        # A separate manager reads the candidate so the active key is untouched.
        candidate = SecureKeyManager().load_from_file(key_file)
    finally:
        key_file.unlink()

    if candidate is None:
        return False

    rotated = await manager.rotate_key(candidate)
    still_current = manager.get_key() == current
    print(f"   Active key: {key_fingerprint(manager.get_key() or '')}")
    if rotated or not still_current:
        print("❌ The rejected key was swapped in")
        return False
    print("✅ Rotation refused; the current key stayed active")
    return True


def multi_environment_pattern() -> None:
    """Print guidance for managing keys across environments. Checks nothing."""
    print("\n🌍 Multi-Environment Pattern")
    print("-" * 40)

    print("✅ Managing Keys Across Environments:")
    print()
    print("Development:")
    print("   - Use .env files (not in git)")
    print("   - Separate dev API key")
    print("   - Low consumptionLimit with a limitPeriod, and an expiresAt")
    print()
    print("Staging:")
    print("   - Use secret manager (AWS Secrets, Vault)")
    print("   - Same key settings as production, with a smaller consumptionLimit")
    print("   - Test rotation procedures")
    print()
    print("Production:")
    print("   - Use secret manager (required)")
    print("   - Dedicated INFERENCE key per service, capped with consumptionLimit")
    print("   - ipAllowlist and modelPrivacy where they fit the service")
    print("   - Automated rotation")
    print("   - Comprehensive monitoring")
    print()

    print("💡 Example Multi-Environment Setup:")
    print("   ```python")
    print("   def get_api_key(environment: str) -> str:")
    print("       if environment == 'development':")
    print("           return os.getenv('VENICE_API_KEY_DEV')")
    print("       elif environment == 'staging':")
    print("           return get_from_secret_manager('staging/venice-api-key')")
    print("       elif environment == 'production':")
    print("           return get_from_secret_manager('prod/venice-api-key')")
    print("       else:")
    print("           raise ValueError(f'Unknown environment: {environment}')")
    print("   ```")


def security_best_practices() -> None:
    """Print security guidance for API key management. Checks nothing."""
    print("\n🛡️ Security Best Practices")
    print("-" * 40)

    print("✅ Critical Security Rules:")
    print()
    print("1. 🚫 NEVER commit keys to version control")
    print("   - Add .env to .gitignore")
    print("   - Scan for accidentally committed keys")
    print("   - Use git-secrets or similar tools")
    print()
    print("2. ✅ Use environment-specific keys")
    print("   - Separate keys per environment")
    print("   - INFERENCE type unless the service manages keys or reads billing")
    print("   - Set expiresAt and a consumptionLimit with a limitPeriod")
    print("   - Easy to rotate if compromised")
    print()
    print("3. ✅ Implement key rotation")
    print("   - Regular rotation schedule (90 days)")
    print("   - Automated rotation process")
    print("   - Zero-downtime rotation")
    print()
    print("4. ✅ Monitor and audit key usage")
    print("   - Log all API key operations")
    print("   - Alert on unusual usage patterns")
    print("   - Track key lifecycle events")
    print()
    print("5. ✅ Secure key storage")
    print("   - Use secret managers (AWS Secrets, Vault)")
    print("   - Encrypt at rest")
    print("   - Restrict access (IAM policies)")
    print()
    print("6. ✅ Handle key compromise")
    print("   - Immediate revocation procedure")
    print("   - Generate new key")
    print("   - Audit for damage")
    print("   - Update affected systems")


def testing_with_keys() -> None:
    """Print testing patterns that don't expose real keys. Checks nothing."""
    print("\n🧪 Testing Patterns")
    print("-" * 40)

    print("✅ Safe Testing Practices:")
    print()
    print("1. Use Test Keys:")
    print("   - Dedicated test API keys")
    print("   - Low consumptionLimit with a limitPeriod, so a runaway test stops early")
    print("   - A short expiresAt")
    print()
    print("2. Mock in Unit Tests:")
    print("   ```python")
    print("   @patch('os.getenv')")
    print("   def test_client_creation(mock_getenv):")
    print("       mock_getenv.return_value = 'test-key'")
    print("       client = VeniceClient(api_key=os.getenv('VENICE_API_KEY'))")
    print("       assert client is not None")
    print("   ```")
    print()
    print("3. Use VCR for Integration Tests:")
    print("   - Record real API responses")
    print("   - Replay without real key")
    print("   - Sanitize recordings")
    print()
    print("4. CI/CD Secrets:")
    print("   - Store in CI secret manager")
    print("   - Inject at runtime")
    print("   - Never log key values")


async def production_client_pattern() -> bool | None:
    """Check a loaded key without spending, then build a client and smoke-test it.

    ``test_key`` is the health check: it reads the key's rate limits, which is
    free. The chat afterwards is a billed smoke test of the whole path, which a
    health check that runs often should not be.

    Returns ``True`` on success, ``None`` (skipped) when VENICE_API_KEY is not
    set, and ``False`` if the live API call failed — so a real failure surfaces
    instead of being swallowed.
    """
    print("\n🏭 Production Client Pattern")
    print("-" * 40)

    # Load API key securely
    manager = SecureKeyManager()
    api_key = manager.load_key()

    if not api_key:
        print("ℹ️  No API key available — skipping live client demo")
        return None

    try:
        if not await test_key(api_key):
            print("❌ Health check failed: Venice rejected the key or denied it access")
            return False
        print("✅ Health check passed (free: rate limits read with this key)")

        async with VeniceClient(api_key=api_key) as client:
            chat_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
            response = await client.chat.completions.create(
                model=chat_model,
                messages=[UserMessage(content="Reply with the single word: ok")],
                max_completion_tokens=60,
                # Only this prompt is billed, not Venice's own system prompt.
                venice_parameters=VeniceParameters(include_venice_system_prompt=False),
            )
    except NoMatchingModelError:
        raise  # a missing model skips the example; see __main__
    except VeniceError as e:
        print(f"❌ Client check failed: {type(e).__name__}: {e}")
        print("   Check API key validity")
        print("   Verify network connectivity")
        print("   Review error logs")
        return False

    finish_reason = response.choices[0].finish_reason if response.choices else None
    print(f"   Key: {key_fingerprint(api_key)}")
    print(f"   Model: {chat_model}")
    print(f"   Response: {response.text}")
    print(f"   finish_reason: {finish_reason}")
    if finish_reason == "length" or not (response.text or "").strip():
        print("❌ Smoke-test reply was truncated or empty")
        return False
    print("✅ Smoke test passed (billed: one short chat request)")
    return True


async def main() -> int:
    """Run all API key management examples.

    Returns ``0`` only if every demo succeeded, ``1`` otherwise, so a real API
    failure or a missing key surfaces as a non-zero process exit instead of
    being masked by the success banner.
    """
    print("=" * 60)
    print("Venice AI SDK - Production API Key Management")
    print("=" * 60)

    # None marks a demo skipped because VENICE_API_KEY is not set.
    results: list[tuple[str, bool | None]] = [
        ("environment_variable_pattern", await environment_variable_pattern()),
        ("key_rotation_pattern", await key_rotation_pattern()),
    ]
    # Printed guidance only, so these are not counted as passing anything.
    multi_environment_pattern()
    security_best_practices()
    testing_with_keys()
    results.append(("production_client_pattern", await production_client_pattern()))

    failed = [name for name, ok in results if ok is False]
    skipped = [name for name, ok in results if ok is None]

    print("\n" + "=" * 60)
    if failed:
        print(f"⚠️ {len(failed)} of {len(results)} demos failed: {', '.join(failed)}")
    if skipped:
        print(f"⚠️ {len(skipped)} of {len(results)} demos skipped: {', '.join(skipped)}")
    if not failed and not skipped:
        print("✅ All examples completed!")
    print("=" * 60)
    print()
    print("🔑 Key Takeaways:")
    print("   1. Never hardcode API keys")
    print("   2. Use environment variables or secret managers")
    print("   3. Implement regular key rotation")
    print("   4. Monitor and audit key usage")
    print("   5. Have a key compromise response plan")

    if skipped:
        print(
            "\nERROR: VENICE_API_KEY is not set. Export it (e.g. via .env) and rerun.",
            file=sys.stderr,
        )
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except NoMatchingModelError as e:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {e}")
        sys.exit(77)
    except (VeniceError, OSError) as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
