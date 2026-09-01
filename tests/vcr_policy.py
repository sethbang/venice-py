"""Single source of truth for VCR record-mode policy.

Every VCR call site in the suite resolves its record mode here. Two call sites
previously decided independently — the root ``vcr_config`` fixture and a
module-level ``vcr.VCR`` in the benchmark suite — which meant a single run could
have one path replaying from cassettes while the other silently dialled out to
the live API.

Recording is not a neutral default. Cassettes are gitignored, so a fresh clone
has none, and under any recording mode a cassette-less request goes to the real
Venice API and bills whoever's ``VENICE_API_KEY`` is set. Recording is therefore
opt-in, by exact token.
"""

from __future__ import annotations

import os

from vcr.record_mode import RecordMode

#: Opts a local run in to recording. Unset means "replay only, never record".
RECORD_ENV_VAR = "VENICE_VCR_RECORD"

#: Set by CI. Always wins, and always means "never record".
CI_ENV_VAR = "VENICE_CI_MODE"

#: ``VENICE_VCR_RECORD`` value -> mode. Membership is exact: a truthy-looking
#: value that isn't listed here leaves the run offline rather than guessing that
#: the caller meant to spend money.
_RECORD_MODES = {
    "all": RecordMode.ALL,
    "new": RecordMode.NEW_EPISODES,
}


def resolve_record_mode() -> RecordMode:
    """Return the VCR record mode for this run.

    ==========================  ===================  ==========================
    ``VENICE_VCR_RECORD``       Mode                 Behaviour
    ==========================  ===================  ==========================
    unset (default)             ``NONE``             replay only; a request with
                                                     no cassette raises
    ``all``                     ``ALL``              re-record every interaction
                                                     the selected tests touch
    ``new``                     ``NEW_EPISODES``     replay, record only gaps
    ==========================  ===================  ==========================

    ``VENICE_CI_MODE=true`` forces ``NONE`` regardless of the above.
    """
    if os.getenv(CI_ENV_VAR, "false").lower() == "true":
        return RecordMode.NONE
    return _RECORD_MODES.get(os.getenv(RECORD_ENV_VAR, "").lower(), RecordMode.NONE)


#: Test-path fragment -> cassette directory. Ordered most-specific first.
_CASSETTE_DIRS = (
    ("tests/e2e/", "tests/e2e/cassettes"),
    # Benchmarks keep cassettes beside themselves; without an entry here they
    # fall through to the integration directory, where no cassette can match and
    # every request either goes live or fails outright.
    ("tests/benchmarks/", "tests/benchmarks/vcr_cassettes"),
)

#: Where a test lands when no fragment matches.
DEFAULT_CASSETTE_DIR = "tests/integration/cassettes"


def cassette_dir_for(test_path: str) -> str:
    """Return the cassette directory serving the test at ``test_path``."""
    for fragment, directory in _CASSETTE_DIRS:
        if fragment in test_path:
            return directory
    return DEFAULT_CASSETTE_DIR
