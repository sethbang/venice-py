"""Importing SDK modules must not print third-party warnings to the user.

Import-time warnings fire once per interpreter, so an in-process test sees
nothing if an earlier test already imported the module. Each check runs a fresh
interpreter with the stock warning filters a user's script would have.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import pytest

_PROBE = r"""
import importlib, json, sys, warnings

shown = []
def _record(message, category, filename, lineno, file=None, line=None):
    shown.append({"category": category.__name__, "message": str(message), "filename": filename})
warnings.showwarning = _record

before = list(warnings.filters)
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except ImportError:
        pass
added = [
    {
        "action": f[0],
        "message": getattr(f[1], "pattern", f[1]) or "",
        "category": f"{f[2].__module__}.{f[2].__qualname__}",
        "module": getattr(f[3], "pattern", f[3]) or "",
    }
    for f in warnings.filters
    if f not in before
]
print(json.dumps({"shown": shown, "filters_added": added}))
"""


def _import_in_fresh_interpreter(*modules: str) -> dict[str, Any]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONWARNINGS"}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, *modules],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=True,
    )
    result: dict[str, Any] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def _warnings_shown_on_import(*modules: str) -> list[dict[str, str]]:
    shown: list[dict[str, str]] = _import_in_fresh_interpreter(*modules)["shown"]
    return shown


def test_importing_x402_auth_emits_no_grammar_warnings() -> None:
    pytest.importorskip("siwe")
    shown = _warnings_shown_on_import("venice_ai.auth.x402")
    grammar = [w for w in shown if w["category"] == "GrammarWarning"]
    assert grammar == [], f"import printed {len(grammar)} abnf GrammarWarning(s): {grammar}"


def test_importing_public_auth_modules_shows_no_warnings() -> None:
    modules = (
        "venice_ai",
        "venice_ai.auth",
        "venice_ai.auth.x402",
        "venice_ai.auth.x402_solana",
        "venice_ai.tee",
    )
    assert len(modules) > 0
    shown = _warnings_shown_on_import(*modules)
    assert shown == [], (
        "importing the SDK showed warnings under default filters: "
        f"{[(w['category'], w['message'][:80]) for w in shown]}"
    )


def _silences_grammar_warnings_globally(entry: dict[str, str]) -> bool:
    if entry["action"] != "ignore":
        return False
    if entry["category"] in {"builtins.Warning", "builtins.UserWarning"}:
        return True
    if entry["category"].endswith(".GrammarWarning"):
        return True
    pattern = f"{entry['message']} {entry['module']}".lower()
    return any(token in pattern for token in ("abnf", "siwe", "redefines"))


def test_importing_sdk_modules_does_not_install_global_ignore_filters() -> None:
    added = _import_in_fresh_interpreter(
        "venice_ai",
        "venice_ai.auth",
        "venice_ai.auth.x402",
        "venice_ai.auth.x402_solana",
        "venice_ai.tee",
    )["filters_added"]
    leaked = [entry for entry in added if _silences_grammar_warnings_globally(entry)]
    assert leaked == [], (
        "importing the SDK left a process-wide ignore filter behind; silence "
        f"third-party import noise with a scoped warnings.catch_warnings(): {leaked}"
    )


@pytest.mark.parametrize(
    ("entry", "flagged"),
    [
        ({"action": "ignore", "message": "", "category": "builtins.Warning", "module": ""}, True),
        (
            {"action": "ignore", "message": "", "category": "abnf.GrammarWarning", "module": ""},
            True,
        ),
        (
            {
                "action": "ignore",
                "message": "rule '.*' redefines",
                "category": "builtins.UserWarning",
                "module": "",
            },
            True,
        ),
        (
            {
                "action": "ignore",
                "message": "",
                "category": "urllib3.exceptions.DependencyWarning",
                "module": "",
            },
            False,
        ),
        (
            {"action": "default", "message": "", "category": "builtins.UserWarning", "module": ""},
            False,
        ),
    ],
)
def test_global_ignore_filter_detection(entry: dict[str, str], flagged: bool) -> None:
    assert _silences_grammar_warnings_globally(entry) is flagged
