"""Guards on the VCR record-mode policy.

Recording is not a neutral default: a cassette-less request under any recording
mode reaches the live Venice API and bills the key holder. Cassettes are
gitignored, so a fresh clone has none. These tests pin the policy that keeps a
plain ``make test`` from spending money, and pin the two VCR call sites to a
single source of truth so one can never replay while the other dials out.
"""

import pytest
from vcr.record_mode import RecordMode

from tests.vcr_policy import (
    CI_ENV_VAR,
    DEFAULT_CASSETTE_DIR,
    RECORD_ENV_VAR,
    cassette_dir_for,
    resolve_record_mode,
)


@pytest.fixture(autouse=True)
def _clear_vcr_env(monkeypatch):
    """Each case states its own environment; inherit nothing from the shell."""
    monkeypatch.delenv(RECORD_ENV_VAR, raising=False)
    monkeypatch.delenv(CI_ENV_VAR, raising=False)


def test_default_never_records() -> None:
    """The unconfigured local default must not be able to reach the network."""
    assert resolve_record_mode() is RecordMode.NONE


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("all", RecordMode.ALL),
        ("ALL", RecordMode.ALL),
        ("new", RecordMode.NEW_EPISODES),
        ("", RecordMode.NONE),
        ("yes", RecordMode.NONE),
        ("true", RecordMode.NONE),
    ],
)
def test_recording_is_opt_in_by_exact_token(monkeypatch, setting, expected) -> None:
    """Only the documented tokens record; anything else stays offline.

    A truthy-looking value like ``true`` must NOT enable recording — guessing
    wrong here costs real money.
    """
    monkeypatch.setenv(RECORD_ENV_VAR, setting)
    assert resolve_record_mode() is expected


def test_ci_mode_overrides_an_explicit_record_request(monkeypatch) -> None:
    monkeypatch.setenv(CI_ENV_VAR, "true")
    monkeypatch.setenv(RECORD_ENV_VAR, "all")
    assert resolve_record_mode() is RecordMode.NONE


def test_root_fixture_uses_the_shared_policy(monkeypatch) -> None:
    """The root ``vcr_config`` fixture must not hardcode its own mode."""
    from tests.conftest import vcr_config

    built = vcr_config.__wrapped__()
    assert built.record_mode is RecordMode.NONE


def test_benchmark_module_uses_the_shared_policy() -> None:
    """Regression guard: the benchmark module used to hardcode ``RecordMode.ONCE``.

    That module-level instance shadows the fixture name, so ``VENICE_CI_MODE``
    did not reach it and it recorded — and therefore billed — whenever a
    cassette was absent, including under a run explicitly asking for CI mode.
    """
    from tests.benchmarks import test_vcr_scheduler_benchmarks as bench

    assert bench.vcr_config.record_mode is resolve_record_mode()
    assert bench.vcr_config.record_mode is not RecordMode.ONCE


def test_default_mode_blocks_an_unrecorded_request(tmp_path) -> None:
    """The default must REFUSE an unrecorded request, not forward it.

    This is the property that actually protects the credit card: under the old
    NEW_EPISODES default, a request with no cassette entry was sent to the live
    API and appended. Here the request is aimed at a closed local port, so a
    pass-through would surface as a connection error. Seeing VCR's own
    "can't overwrite" exception instead proves the request never left the
    process.
    """
    import vcr as vcr_module
    from vcr.errors import CannotOverwriteExistingCassetteException

    recorder = vcr_module.VCR(
        cassette_library_dir=str(tmp_path),
        record_mode=resolve_record_mode(),
    )

    import urllib.error
    import urllib.request

    with (
        recorder.use_cassette("absent.yaml"),
        pytest.raises(CannotOverwriteExistingCassetteException),
    ):
        urllib.request.urlopen("http://127.0.0.1:1/never", timeout=1)  # noqa: S310

    assert not list(tmp_path.iterdir()), "a blocked run must not write a cassette"


@pytest.mark.parametrize(
    ("test_path", "expected"),
    [
        ("/repo/tests/e2e/test_live.py", "tests/e2e/cassettes"),
        (
            "/repo/tests/benchmarks/test_vcr_scheduler_benchmarks.py",
            "tests/benchmarks/vcr_cassettes",
        ),
        ("/repo/tests/integration/test_chat.py", DEFAULT_CASSETTE_DIR),
        ("/repo/tests/unit/test_thing.py", DEFAULT_CASSETTE_DIR),
    ],
)
def test_cassette_dir_routing(test_path, expected) -> None:
    assert cassette_dir_for(test_path) == expected


def test_benchmarks_do_not_fall_through_to_integration() -> None:
    """Regression guard for the miss that sent every benchmark request live.

    Benchmark cassettes live in tests/benchmarks/vcr_cassettes, but the fixture
    only special-cased e2e, so benchmarks resolved to the integration directory
    and no cassette could ever match.
    """
    resolved = cassette_dir_for("/repo/tests/benchmarks/test_vcr_scheduler_benchmarks.py")
    assert resolved != DEFAULT_CASSETTE_DIR
    assert resolved == "tests/benchmarks/vcr_cassettes"
