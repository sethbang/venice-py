"""Unit tests for factory rate limiter creation.

Tests the VeniceClientFactory._create_rate_limiter method which determines
which rate limiter implementation to use based on configuration.
"""

import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest


class TestDefaultCreatesSimpleRateLimiter:
    """Tests verifying default config creates SimpleRateLimiter."""

    def test_default_creates_simple_rate_limiter(self):
        """Default configuration creates SimpleRateLimiter."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import SimpleRateLimiter

        config = VeniceAIConfig.create_minimal_config()
        mock_client = MagicMock()

        rate_limiter = VeniceClientFactory._create_rate_limiter(
            config, mock_client, account_id="test-account"
        )

        assert isinstance(rate_limiter, SimpleRateLimiter)

    def test_simple_mode_explicitly(self):
        """Explicit SIMPLE mode creates SimpleRateLimiter."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import RateLimiterMode, SimpleRateLimiter
        from venice_ai.rate_limiting.config import RateLimiterConfig

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(mode=RateLimiterMode.SIMPLE)
        mock_client = MagicMock()

        rate_limiter = VeniceClientFactory._create_rate_limiter(
            config, mock_client, account_id="test-account"
        )

        assert isinstance(rate_limiter, SimpleRateLimiter)

    def test_simple_rate_limiter_uses_config_values(self):
        """SimpleRateLimiter uses config values."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import SimpleRateLimiter
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.SIMPLE,
            min_backoff=2.0,
            max_backoff=120.0,
            failure_threshold=10,
            max_models=500,
        )
        mock_client = MagicMock()

        rate_limiter = VeniceClientFactory._create_rate_limiter(
            config, mock_client, account_id="test-account"
        )

        assert isinstance(rate_limiter, SimpleRateLimiter)
        assert rate_limiter.min_backoff == 2.0
        assert rate_limiter.max_backoff == 120.0
        assert rate_limiter.failure_threshold == 10
        assert rate_limiter.max_models == 500


class TestAdaptiveFallbackWithoutPackage:
    """Tests for adaptive mode strict errors when package is missing."""

    def test_adaptive_fallback_without_package(self):
        """Adaptive mode raises ImportError when package is missing (strict mode)."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url="redis://localhost:6379",
        )
        mock_client = MagicMock()

        # Mock import failure for adaptive_rate_limiter. Submodules imported
        # earlier in the session stay cached in sys.modules and would satisfy
        # ``from adaptive_rate_limiter.backends import ...`` without touching
        # the parent, so block every cached submodule as well.
        blocked = {
            name: None
            for name in list(sys.modules)
            if name == "adaptive_rate_limiter" or name.startswith("adaptive_rate_limiter.")
        }
        blocked["adaptive_rate_limiter"] = None
        with patch.dict(sys.modules, blocked):
            with pytest.raises(ImportError) as exc_info:
                VeniceClientFactory._create_rate_limiter(
                    config, mock_client, account_id="test-account"
                )

            # Should raise with informative message
            assert "adaptive-rate-limiter" in str(exc_info.value)
            assert "pip install 'venice-py[adaptive]'" in str(exc_info.value)

    def test_adaptive_fallback_warning_message(self):
        """Adaptive mode raises ImportError with install instructions (strict mode)."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url="redis://localhost:6379",
        )
        mock_client = MagicMock()

        # Mock import failure for adaptive_rate_limiter
        original_import = __builtins__.get("__import__", __import__)

        def mock_import(name, *args, **kwargs):
            if name == "adaptive_rate_limiter" or name.startswith("adaptive_rate_limiter."):
                raise ImportError("No module named 'adaptive_rate_limiter'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError) as exc_info:
                VeniceClientFactory._create_rate_limiter(
                    config, mock_client, account_id="test-account"
                )

            # Error message should contain install instructions
            error_msg = str(exc_info.value)
            assert "adaptive-rate-limiter" in error_msg
            assert "pip install 'venice-py[adaptive]'" in error_msg


class TestAdaptiveRequiresRedisUrl:
    """Tests verifying adaptive mode requires redis_url."""

    def test_adaptive_requires_redis_url(self):
        """Adaptive mode raises RuntimeError when redis_url is missing (strict mode).

        When adaptive mode is selected without a redis_url and the adaptive package
        is installed, a ValueError is raised internally and wrapped in RuntimeError.
        When the adaptive package is missing, an ImportError is raised directly.
        In both cases, no silent fallback occurs.
        """
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=None,  # No redis URL
        )
        mock_client = MagicMock()

        # Strict mode: always raises, never falls back silently
        with pytest.raises((ImportError, RuntimeError)) as exc_info:
            VeniceClientFactory._create_rate_limiter(config, mock_client, account_id="test-account")

        # If RuntimeError: should mention redis_url or configuration
        # If ImportError: should mention adaptive-rate-limiter package
        error_msg = str(exc_info.value)
        assert any(
            keyword in error_msg
            for keyword in ["redis_url", "adaptive-rate-limiter", "AdaptiveScheduler"]
        )


class TestAdaptiveIntelligentSuccessPath:
    """Verify the ADAPTIVE/INTELLIGENT scheduler is wired up correctly when all
    components are available and config is valid.

    Existing tests only cover the *failure* paths (missing redis_url, missing
    package). This class proves the success path actually composes the
    backend → state → discovery → provider → classifier → scheduler graph and
    returns it wrapped in ``AdaptiveSchedulerAdapter``.
    """

    def test_adaptive_intelligent_returns_adapted_scheduler(self):
        """ADAPTIVE mode with valid redis_url returns AdaptiveSchedulerAdapter
        wrapping a Scheduler with all six dependencies wired up."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import AdaptiveSchedulerAdapter, VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url="redis://localhost:6379",
        )
        mock_client = MagicMock()

        with (
            patch("adaptive_rate_limiter.backends.RedisBackend") as mock_backend_cls,
            patch("adaptive_rate_limiter.scheduler.StateManager") as mock_state_cls,
            patch("adaptive_rate_limiter.scheduler.Scheduler") as mock_sched_cls,
            patch("adaptive_rate_limiter.scheduler.RateLimiterConfig"),
            patch("adaptive_rate_limiter.scheduler.SchedulerMode"),
            patch("venice_ai.core.rate_limit_discovery.RateLimitDiscovery") as mock_discovery_cls,
            patch("venice_ai.provider.venice_provider.VeniceProvider") as mock_provider_cls,
            patch(
                "venice_ai.provider.classifier_adapter.VeniceClassifierAdapter"
            ) as mock_classifier_adapter_cls,
            patch("venice_ai._request_classifier.RequestClassifier") as mock_request_classifier_cls,
        ):
            mock_backend = mock_backend_cls.return_value
            mock_state = mock_state_cls.return_value
            mock_sched = mock_sched_cls.return_value
            mock_discovery = mock_discovery_cls.return_value
            mock_provider = mock_provider_cls.return_value
            mock_classifier_adapter = mock_classifier_adapter_cls.return_value
            mock_request_classifier = mock_request_classifier_cls.return_value

            rate_limiter = VeniceClientFactory._create_rate_limiter(
                config, mock_client, account_id="test-account"
            )

        # Wrapping
        assert isinstance(rate_limiter, AdaptiveSchedulerAdapter)
        assert rate_limiter._scheduler is mock_sched

        # Backend wired with redis_url + account_id
        mock_backend_cls.assert_called_once_with(
            redis_url="redis://localhost:6379",
            account_id="test-account",
        )
        # State manager wraps the backend AND receives the provider so that
        # header-based rate-limit state updates work.
        mock_state_cls.assert_called_once_with(backend=mock_backend, provider=mock_provider)
        # Discovery shared by provider + classifier
        mock_discovery_cls.assert_called_once_with(client=mock_client, account_id="test-account")
        mock_provider_cls.assert_called_once_with(
            client=mock_client, rate_limit_discovery=mock_discovery
        )
        mock_request_classifier_cls.assert_called_once_with(rate_limit_discovery=mock_discovery)
        mock_classifier_adapter_cls.assert_called_once_with(classifier=mock_request_classifier)
        # Scheduler invoked with full dependency set
        sched_kwargs = mock_sched_cls.call_args.kwargs
        assert sched_kwargs["client"] is mock_client
        assert sched_kwargs["state_manager"] is mock_state
        assert sched_kwargs["provider"] is mock_provider
        assert sched_kwargs["classifier"] is mock_classifier_adapter

    def test_adaptive_state_manager_has_provider_attached(self):
        """Regression test: AdaptiveStateManager must receive the same
        VeniceProvider instance attached to the Scheduler so that
        ``update_state_from_headers`` can resolve buckets for header-based
        state updates. Without this wiring the scheduler re-issues a
        cold-start probe on every request and never makes forward progress.

        Unlike ``test_adaptive_intelligent_returns_adapted_scheduler`` —
        which patches ``StateManager`` itself — this test lets the real
        ``StateManager`` class run so we observe its actual ``provider``
        attribute, guaranteeing the regression is caught even if the
        constructor-call assertion is later removed or weakened.
        """
        from adaptive_rate_limiter.scheduler import StateManager as RealStateManager

        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import AdaptiveSchedulerAdapter, VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url="redis://localhost:6379",
        )
        mock_client = MagicMock()

        with (
            # Mock the Redis backend so we don't need a live Redis instance,
            # but let the *real* StateManager class run so we observe its
            # provider attribute.
            patch("adaptive_rate_limiter.backends.RedisBackend"),
            patch("adaptive_rate_limiter.scheduler.Scheduler") as mock_sched_cls,
            patch("adaptive_rate_limiter.scheduler.RateLimiterConfig"),
            patch("adaptive_rate_limiter.scheduler.SchedulerMode"),
            patch("venice_ai.core.rate_limit_discovery.RateLimitDiscovery"),
            patch("venice_ai.provider.venice_provider.VeniceProvider") as mock_provider_cls,
            patch("venice_ai.provider.classifier_adapter.VeniceClassifierAdapter"),
            patch("venice_ai._request_classifier.RequestClassifier"),
        ):
            mock_provider = mock_provider_cls.return_value

            rate_limiter = VeniceClientFactory._create_rate_limiter(
                config, mock_client, account_id="test-account"
            )

        assert isinstance(rate_limiter, AdaptiveSchedulerAdapter)

        # Pull the StateManager that was passed to the Scheduler.
        sched_kwargs = mock_sched_cls.call_args.kwargs
        state_manager = sched_kwargs["state_manager"]

        # The real StateManager exposes ``provider`` as an instance attribute,
        # set by its __init__. If the factory forgot to pass it through, the
        # attribute will be None and update_state_from_headers will return 0
        # for every request (the cold-start spin we are guarding against).
        assert isinstance(state_manager, RealStateManager)
        assert state_manager.provider is not None, (
            "AdaptiveStateManager must receive a provider — without it, "
            "update_state_from_headers short-circuits and the scheduler "
            "spins on cold-start probes."
        )
        # And it must be the *same* provider attached to the scheduler.
        assert state_manager.provider is mock_provider
        assert state_manager.provider is sched_kwargs["provider"]

    def test_adaptive_warns_and_uses_default_account_id_when_omitted(self):
        """Omitting account_id in both config and call site triggers a
        UserWarning and falls back to ``"default"`` (lines 428-437)."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url="redis://localhost:6379",
            account_id=None,
        )
        mock_client = MagicMock()

        with (
            patch("adaptive_rate_limiter.backends.RedisBackend") as mock_backend_cls,
            patch("adaptive_rate_limiter.scheduler.StateManager"),
            patch("adaptive_rate_limiter.scheduler.Scheduler"),
            patch("adaptive_rate_limiter.scheduler.RateLimiterConfig"),
            patch("adaptive_rate_limiter.scheduler.SchedulerMode"),
            patch("venice_ai.core.rate_limit_discovery.RateLimitDiscovery") as mock_discovery_cls,
            patch("venice_ai.provider.venice_provider.VeniceProvider"),
            patch("venice_ai.provider.classifier_adapter.VeniceClassifierAdapter"),
            patch("venice_ai._request_classifier.RequestClassifier"),
            warnings.catch_warnings(record=True) as captured,
        ):
            warnings.simplefilter("always")
            # Empty string (the function's typed default) triggers the same
            # falsy-fallback path as ``None`` would have at runtime; we want
            # to test the default-account-id warning, so omit a real value.
            VeniceClientFactory._create_rate_limiter(config, mock_client, account_id="")

        # Warning was emitted
        assert any(
            "account_id not provided" in str(w.message) and issubclass(w.category, UserWarning)
            for w in captured
        )
        # Default account_id propagated to backend + discovery
        mock_backend_cls.assert_called_once_with(
            redis_url="redis://localhost:6379",
            account_id="default",
        )
        mock_discovery_cls.assert_called_once_with(client=mock_client, account_id="default")

    def test_adaptive_falls_back_to_backend_redis_url_when_rate_limiter_url_missing(self):
        """When ``rate_limiter.redis_url`` is None but ``backend.redis.redis_url`` is
        set, the factory uses the backend URL (lines 414-419)."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.core.config.backends import BackendConfig, RedisBackendConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=None,
        )
        # Ensure backend.redis is populated so the fallback branch is reachable.
        config.backend = BackendConfig(
            redis=RedisBackendConfig(redis_url="redis://from-backend:6379"),
        )

        mock_client = MagicMock()

        with (
            patch("adaptive_rate_limiter.backends.RedisBackend") as mock_backend_cls,
            patch("adaptive_rate_limiter.scheduler.StateManager"),
            patch("adaptive_rate_limiter.scheduler.Scheduler"),
            patch("adaptive_rate_limiter.scheduler.RateLimiterConfig"),
            patch("adaptive_rate_limiter.scheduler.SchedulerMode"),
            patch("venice_ai.core.rate_limit_discovery.RateLimitDiscovery"),
            patch("venice_ai.provider.venice_provider.VeniceProvider"),
            patch("venice_ai.provider.classifier_adapter.VeniceClassifierAdapter"),
            patch("venice_ai._request_classifier.RequestClassifier"),
        ):
            VeniceClientFactory._create_rate_limiter(config, mock_client, account_id="test-account")

        # Backend constructed with the backend.redis.redis_url, not rate_limiter.redis_url,
        # plus the pool size and cluster flag from the same RedisBackendConfig.
        mock_backend_cls.assert_called_once_with(
            redis_url="redis://from-backend:6379",
            account_id="test-account",
            max_connections=20,
            cluster_mode=False,
        )


class TestDisabledMode:
    """Tests for disabled mode creating NoOpRateLimiter."""

    def test_disabled_mode_creates_noop(self):
        """Disabled mode creates NoOpRateLimiter with warning."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import NoOpRateLimiter
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(mode=RateLimiterMode.DISABLED)
        mock_client = MagicMock()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            rate_limiter = VeniceClientFactory._create_rate_limiter(
                config, mock_client, account_id="test-account"
            )

            # Should create NoOpRateLimiter
            assert isinstance(rate_limiter, NoOpRateLimiter)

            # Should have emitted a warning
            assert any("Rate limiting is DISABLED" in str(warning.message) for warning in w)

    def test_disabled_mode_logs_critical(self):
        """Disabled mode logs a critical message."""
        import logging

        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(mode=RateLimiterMode.DISABLED)
        mock_client = MagicMock()

        with (
            patch.object(logging.getLogger("venice_ai.rate_limiting"), "critical") as mock_critical,
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")

            VeniceClientFactory._create_rate_limiter(config, mock_client, account_id="test-account")

            # Should have logged critical message
            mock_critical.assert_called_once()
            call_arg = mock_critical.call_args[0][0]
            assert "RATE LIMITING DISABLED" in call_arg

    @pytest.mark.asyncio
    async def test_disabled_rate_limiter_always_allows(self):
        """NoOpRateLimiter from disabled mode always allows requests."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(mode=RateLimiterMode.DISABLED)
        mock_client = MagicMock()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            rate_limiter = VeniceClientFactory._create_rate_limiter(
                config, mock_client, account_id="test-account"
            )

        # NoOpRateLimiter always allows requests
        from venice_ai.rate_limiting import NoOpRateLimiter

        assert isinstance(rate_limiter, NoOpRateLimiter)
        can_proceed, wait_time = await rate_limiter.acquire("any-model")
        assert can_proceed
        assert wait_time == 0


class TestFactoryIntegration:
    """Integration tests for factory rate limiter creation."""

    def test_full_client_creation_with_simple_rate_limiter(self):
        """Full client creation injects SimpleRateLimiter."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import SimpleRateLimiter

        config = VeniceAIConfig.create_minimal_config()

        client = VeniceClientFactory.create_client(
            config=config,
            api_key="test-api-key",
            account_id="test-account",
        )

        # Rate limiter should be injected
        assert hasattr(client, "rate_limiter")
        assert isinstance(client.rate_limiter, SimpleRateLimiter)

    def test_full_client_creation_with_disabled_rate_limiter(self):
        """Full client creation with disabled mode injects NoOpRateLimiter."""
        from venice_ai.core.config import VeniceAIConfig
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import NoOpRateLimiter
        from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode

        config = VeniceAIConfig.create_minimal_config()
        config.rate_limiter = RateLimiterConfig(mode=RateLimiterMode.DISABLED)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            client = VeniceClientFactory.create_client(
                config=config,
                api_key="test-api-key",
                account_id="test-account",
            )

        # Rate limiter should be NoOpRateLimiter
        assert hasattr(client, "rate_limiter")
        assert isinstance(client.rate_limiter, NoOpRateLimiter)

    def test_test_client_creation(self):
        """Test client factory creates proper rate limiter."""
        from venice_ai.factory import VeniceClientFactory
        from venice_ai.rate_limiting import SimpleRateLimiter

        client = VeniceClientFactory.create_test_client(
            enable_redis=False,
        )

        # Test client should have SimpleRateLimiter by default
        assert hasattr(client, "rate_limiter")
        assert isinstance(client.rate_limiter, SimpleRateLimiter)
