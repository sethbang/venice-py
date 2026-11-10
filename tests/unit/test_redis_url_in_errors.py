"""Errors about a Redis URL name its scheme, host and port, and nothing else from it."""

import pytest

from venice_ai.core.config import (
    BackendConfig,
    BackendType,
    RedisBackendConfig,
    VeniceAIConfig,
)
from venice_ai.presets.production import create_production_config
from venice_ai.rate_limiting.config import RateLimiterConfig, RateLimiterMode
from venice_ai.validation import config_validator
from venice_ai.validation.config_validator import validate_config

SECRET = "hunter2"
NOT_SHOWN = "<not shown: not in scheme://host[:port] form>"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("redis://prod-redis:6379/0", "redis://prod-redis:6379"),
        (f"redis://user:{SECRET}@prod-redis:6379/0", "redis://prod-redis:6379"),
        (f"rediss://:{SECRET}@prod-redis:6380", "rediss://prod-redis:6380"),
        (f"redis://prod-redis:6379/0?password={SECRET}", "redis://prod-redis:6379"),
        (f"redis://{SECRET}@prod-redis", "redis://prod-redis"),
        (f"http://user:{SECRET}@prod-redis:6379", "http://prod-redis:6379"),
        (f"redis://user:{SECRET}@[::1]:6379", "redis://[::1]:6379"),
    ],
)
def test_url_shows_scheme_host_port_only(url: str, expected: str) -> None:
    assert config_validator._redis_url_for_display(url) == expected


@pytest.mark.parametrize(
    "value",
    [
        SECRET,
        f"user:{SECRET}@prod-redis:6379",
        f"prod-redis:{SECRET}",
        f"//user:{SECRET}@prod-redis:6379",
        f"redis://user:{SECRET}/x@prod-redis:6379",
        f"redis://prod-redis:{SECRET}",
        f"redis://{SECRET}@",
        f"redis://:{SECRET}@",
        f"redis://[{SECRET}",
        f"{SECRET}://",
        "",
    ],
)
def test_a_string_that_is_not_a_clean_url_shows_nothing_of_itself(value: str) -> None:
    assert config_validator._redis_url_for_display(value) == NOT_SHOWN


@pytest.mark.parametrize(
    "url",
    [
        "redis://user:123/x@prod-redis:6379",
        "redis://user:123?x@prod-redis:6379",
        "redis://user:123#x@prod-redis:6379",
    ],
)
def test_a_numeric_password_split_by_a_delimiter_is_not_read_as_a_port(url: str) -> None:
    assert config_validator._redis_url_for_display(url) == NOT_SHOWN


def _validator_errors(url: str) -> str:
    config = VeniceAIConfig(
        backend=BackendConfig(
            backend_type=BackendType.REDIS, redis=RedisBackendConfig(redis_url=url)
        ),
        rate_limiter=RateLimiterConfig(mode=RateLimiterMode.ADAPTIVE, redis_url="redis://r:6379"),
    )
    return " ".join(validate_config(config).errors)


@pytest.mark.parametrize(
    "url",
    [SECRET, f"user:{SECRET}@prod-redis:6379", f"http://user:{SECRET}@prod-redis:6379"],
)
def test_validator_error_does_not_echo_a_password(url: str) -> None:
    message = _validator_errors(url)

    assert "Invalid Redis URL format" in message
    assert SECRET not in message


def test_validator_error_names_the_scheme_and_host() -> None:
    assert "http://prod-redis:6379" in _validator_errors(f"http://user:{SECRET}@prod-redis:6379")


@pytest.mark.parametrize(
    "url",
    [
        f"redis://user:{SECRET}@localhost:6379/0",
        f"rediss://:{SECRET}@127.0.0.1:6380",
        f"redis://localhost:6379/0?password={SECRET}",
    ],
)
def test_production_preset_error_does_not_echo_a_password(url: str) -> None:
    with pytest.raises(ValueError) as exc_info:
        create_production_config(redis_url=url)

    message = str(exc_info.value)
    assert "Invalid Redis URL for production" in message
    assert SECRET not in message


def test_production_preset_error_names_the_host() -> None:
    with pytest.raises(ValueError, match=r"redis://localhost:6379"):
        create_production_config(redis_url=f"redis://user:{SECRET}@localhost:6379/0")
