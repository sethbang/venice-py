"""Docstring examples must call methods that exist, with arguments they accept.

Every ``client.<resource>.<method>(...)`` call written in a module, class or
function docstring under ``src/venice_ai`` is resolved against a real
``VeniceClient``. The call target has to exist, and every keyword argument in
the example has to bind to the target's signature. Examples are what users
copy, so a renamed method or parameter that leaves a stale example behind
turns into an ``AttributeError`` or ``TypeError`` in their code.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from venice_ai import VeniceClient

SRC = Path(__file__).resolve().parents[2] / "src" / "venice_ai"

# Linters that match *removed* v1 call shapes quote those calls on purpose.
_V1_LINT_MODULES = frozenset(
    {
        "cli/utils/lint_rules.py",
        "skills/venice-py/scripts/lint_v1_usage.py",
    }
)

_CALL = re.compile(r"\bclient\.([a-z_]+(?:\.[a-z_]+){1,2})\(")
_KWARG = re.compile(r"(?<![\w.=!<>])([a-z_][a-z0-9_]*)\s*=(?!=)")


def _docstrings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                yield doc


def _call_args(text: str, open_paren: int) -> str:
    depth = 0
    for i in range(open_paren, len(text)):
        ch = text[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return text[open_paren + 1 : i]
    return text[open_paren + 1 :]


def _top_level_kwargs(args: str) -> set[str]:
    names: set[str] = set()
    depth = 0
    start = 0
    for i, ch in enumerate(args + ","):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            m = _KWARG.match(args[start:i].strip())
            if m:
                names.add(m.group(1))
            start = i + 1
    return names


def _collect() -> list[tuple[str, str, frozenset[str]]]:
    found = []
    for path in sorted(SRC.rglob("*.py")):
        if path.relative_to(SRC).as_posix() in _V1_LINT_MODULES:
            continue
        rel = str(path.relative_to(SRC.parent.parent))
        for doc in _docstrings(path):
            for m in _CALL.finditer(doc):
                kwargs = _top_level_kwargs(_call_args(doc, m.end() - 1))
                found.append((rel, m.group(1), frozenset(kwargs)))
    return found


CALLS = _collect()
CALL_FILES = sorted({rel for rel, _, _ in CALLS})


@pytest.fixture(scope="module")
def client():
    return VeniceClient(api_key="test-key")


def test_docstring_calls_were_found():
    assert len(CALLS) > 50
    assert len(CALL_FILES) > 5
    assert any(target == "video.quote" for _, target, _ in CALLS)
    assert any(
        target == "video.submit" and "duration_seconds" in kwargs for _, target, kwargs in CALLS
    )
    assert all((SRC / m).exists() for m in _V1_LINT_MODULES)


def _resolve(client, dotted: str):
    obj = client
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("source", CALL_FILES)
def test_docstring_call_targets_exist(client, source):
    missing = sorted(
        {
            f"{rel}: client.{target}(...)"
            for rel, target, _ in CALLS
            if rel == source and _resolve(client, target) is None
        }
    )
    assert not missing, "docstrings call methods that do not exist:\n" + "\n".join(missing)


@pytest.mark.parametrize("source", CALL_FILES)
def test_docstring_call_keywords_bind(client, source):
    bad = set()
    for rel, target, kwargs in CALLS:
        if rel != source:
            continue
        fn = _resolve(client, target)
        if fn is None or not callable(fn) or not kwargs:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        unknown = sorted(kwargs - set(params))
        if unknown:
            bad.add(f"{rel}: client.{target}(...) passes unknown {unknown}")
    assert not bad, "docstring examples pass arguments the target rejects:\n" + "\n".join(
        sorted(bad)
    )
