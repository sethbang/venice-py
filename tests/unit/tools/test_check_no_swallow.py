"""Unit tests for the examples swallow-to-exit-0 checker.

The checker (``tools/examples/check_no_swallow.py``) must catch the
swallow-to-exit-0 anti-pattern the live audit found, while staying zero
false-positive on the remediated tree. Three high-precision rules:

- A: a *broad* except (``Exception`` / ``BaseException`` / bare) returning a
  truthy literal — returning success from an error handler.
- B: a broad except whose body is pure print/log + fall-through (no ``raise``,
  ``sys.exit``, assignment, or value-returning ``return``).
- C: if ``main`` is annotated ``-> int``, its result must reach ``sys.exit``.

Narrow typed excepts (``except ValueError`` / ``except (VeniceError, APIError)``)
are intentional error-demos and are never flagged.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "examples" / "check_no_swallow.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_no_swallow", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution can find the module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


find_violations = _load().find_violations


def _rules(src: str) -> set[str]:
    return {v.rule for v in find_violations("snippet.py", textwrap.dedent(src))}


# ---- clean cases (must NOT flag) ----


def test_ok_tracking_return_false_is_clean():
    assert (
        _rules(
            """
        async def demo() -> bool:
            try:
                await thing()
                return True
            except Exception as e:
                print(f"err: {e}")
                return False
        """
        )
        == set()
    )


def test_assignment_ok_flag_is_clean():
    assert (
        _rules(
            """
        async def demo() -> bool:
            ok = True
            for x in items:
                try:
                    await thing(x)
                except Exception as e:
                    print(e)
                    ok = False
                    continue
            return ok
        """
        )
        == set()
    )


def test_reraise_is_clean():
    assert (
        _rules(
            """
        try:
            work()
        except Exception as e:
            print(e)
            raise
        """
        )
        == set()
    )


def test_narrow_typed_except_is_carved_out():
    # Intentional error-demo: returning True from a *typed* catch is fine.
    assert (
        _rules(
            """
        async def error_handling() -> bool:
            try:
                await scrape(bad_url)
            except (VeniceError, APIError) as e:
                print(f"handled as designed: {e}")
                return True
            return True
        """
        )
        == set()
    )


# ---- Rule A: broad except returns success ----


def test_rule_a_broad_except_returns_true():
    assert "A" in _rules(
        """
        async def demo() -> bool:
            try:
                await thing()
            except Exception as e:
                print(e)
                return True
        """
    )


# ---- Rule B: broad except swallows (print + fall-through) ----


def test_rule_b_print_then_fallthrough():
    assert "B" in _rules(
        """
        async def demo():
            try:
                await thing()
            except Exception as e:
                print(f"error: {e}")
        """
    )


def test_rule_b_print_then_continue():
    assert "B" in _rules(
        """
        async def demo():
            for x in items:
                try:
                    await thing(x)
                except Exception as e:
                    print(e)
                    continue
        """
    )


# ---- Rule C: main -> int result must reach sys.exit ----


def test_rule_c_discarded_int_main_flagged():
    assert "C" in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            asyncio.run(main())
        """
    )


def test_rule_c_inline_sys_exit_is_clean():
    assert "C" not in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            sys.exit(asyncio.run(main()))
        """
    )


def test_rule_c_assigned_then_exit_is_clean():
    assert "C" not in _rules(
        """
        async def main() -> int:
            return 0

        if __name__ == "__main__":
            rc = asyncio.run(main())
            sys.exit(rc)
        """
    )


def test_rule_c_none_main_raise_based_is_clean():
    # main() -> None that raises on failure; __main__ exits via except. Not flagged.
    assert "C" not in _rules(
        """
        async def main() -> None:
            await work()

        if __name__ == "__main__":
            try:
                asyncio.run(main())
            except Exception as e:
                print(e, file=sys.stderr)
                sys.exit(1)
        """
    )


# ---- the real audit fixture: the pre-remediation swallow must be caught ----


def test_pre_remediation_swallow_fixture_is_flagged():
    """The pre-Wave1 tool_calling.py (a real swallow) must be flagged; if it
    isn't, the checker doesn't actually work."""
    import subprocess

    old = subprocess.run(
        ["git", "show", "ec1be8a:examples/chat/tool_calling.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if old.returncode != 0:
        pytest.skip("pre-Wave1 fixture commit not available in this checkout")
    violations = find_violations("examples/chat/tool_calling.py", old.stdout)
    assert violations, "pre-remediation tool_calling.py must trip the swallow checker"


# ---- hard gate vs advisory + zero-FP on the real tree ----

_mod = _load()


def test_examples_tree_has_no_hard_violations():
    """The remediated examples/ tree must be clean of hard (A/C) findings —
    the zero-false-positive bar for the CI gate."""
    hard: list[str] = []
    files = _mod.discover_example_files(REPO_ROOT / "examples")
    assert files, "discovery found no examples; the zero-FP check would be vacuous"
    for path in files:
        for v in find_violations(str(path), path.read_text(encoding="utf-8")):
            if v.rule in _mod.HARD_RULES:
                hard.append(f"{path}:{v.line} [{v.rule}] {v.message}")
    assert not hard, "hard (A/C) findings on the current tree:\n" + "\n".join(hard)


def test_main_default_passes_on_current_tree():
    """Default gate (A/C hard, B advisory) passes on the remediated tree."""
    assert _mod.main([str(REPO_ROOT / "examples")]) == 0


def test_main_strict_fails_when_advisory_b_present(tmp_path):
    """Rule B is advisory by default and fatal under --strict."""
    (tmp_path / "demo.py").write_text(
        textwrap.dedent(
            """
            async def demo():
                try:
                    await thing()
                except Exception as e:
                    print(f"error: {e}")
            """
        )
    )
    assert _mod.main([str(tmp_path)]) == 0
    assert _mod.main([str(tmp_path), "--strict"]) == 1


# ---- file discovery: only real examples, never scratch/generated trees ----

#: A Rule A violation — a hard finding, so the gate must fail wherever it is scanned.
_VIOLATION = textwrap.dedent(
    """
    async def demo() -> bool:
        try:
            await thing()
        except Exception as e:
            print(e)
            return True
    """
)

#: Directories under an examples tree that hold generated output, virtualenvs,
#: caches, or tooling state rather than examples. ``envdir`` is an innocuous
#: name — it is excluded only because it contains ``pyvenv.cfg``.
_EXCLUDED_DIRS = [
    "results/_runlogs",
    "music/results",
    ".venv/lib/python3.13",
    "venv/lib",
    "envdir/lib/python3.13",
    "tools/site-packages/pip",
    "basic/__pycache__",
    ".claude/hooks",
    "node_modules/pkg",
]


def _polluted_tree(root: Path) -> Path:
    """An examples tree with one real violating example and the same violation
    copied into every location that is not an example."""
    real = root / "basic" / "real_example.py"
    real.parent.mkdir(parents=True)
    real.write_text(_VIOLATION, encoding="utf-8")
    (root / "basic" / "clean.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "envdir").mkdir()
    (root / "envdir" / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    for rel in _EXCLUDED_DIRS:
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "scratch.py").write_text(_VIOLATION, encoding="utf-8")
    return real


def test_main_reports_only_real_example_in_polluted_tree(tmp_path, capsys):
    """The gate still fails on a genuine violation, and reports ONLY the real
    example — never scratch/venv/cache copies of it."""
    examples = tmp_path / "examples"
    _polluted_tree(examples)
    assert _mod.main([str(examples)]) == 1
    out = capsys.readouterr().out
    flagged = [line for line in out.splitlines() if "[A]" in line]
    assert len(flagged) == 1, out
    assert "real_example.py" in flagged[0], out


def test_discover_skips_non_example_dirs(tmp_path):
    examples = tmp_path / "examples"
    real = _polluted_tree(examples)
    found = _mod.discover_example_files(examples)
    assert found == sorted([real, examples / "basic" / "clean.py"])


def test_discover_walks_root_under_dotted_parent(tmp_path):
    """Exclusions apply below the walk root only; a checkout living under a
    hidden or ``results`` parent must not make discovery vacuous."""
    examples = tmp_path / ".cache" / "results" / "examples"
    real = _polluted_tree(examples)
    assert real in _mod.discover_example_files(examples)


def test_explicit_file_in_excluded_dir_is_still_checked(tmp_path):
    """A file the user names explicitly is checked even inside an excluded dir."""
    scratch = tmp_path / "examples" / "results" / "_runlogs" / "probe.py"
    scratch.parent.mkdir(parents=True)
    scratch.write_text(_VIOLATION, encoding="utf-8")
    assert _mod.main([str(scratch)]) == 1


def test_main_fails_when_no_files_found(tmp_path, capsys):
    """Zero discovered files is a broken gate, not a clean tree."""
    empty = tmp_path / "examples"
    (empty / "results").mkdir(parents=True)
    (empty / "results" / "probe.py").write_text(_VIOLATION, encoding="utf-8")
    assert _mod.main([str(empty)]) == 1
    assert "no example files" in capsys.readouterr().out.lower()


def test_real_tree_discovery_matches_tracked_examples():
    """Discovery on the real repo finds the tracked examples and nothing under
    the gitignored ``examples/results/`` output dir."""
    import subprocess

    found = _mod.discover_example_files(REPO_ROOT / "examples")
    assert REPO_ROOT / "examples" / "basic" / "quick_start.py" in found
    results = REPO_ROOT / "examples" / "results"
    assert not [p for p in found if p.is_relative_to(results)]

    tracked = subprocess.run(
        ["git", "ls-files", "examples/*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if tracked.returncode != 0 or not tracked.stdout.strip():
        pytest.skip("git ls-files unavailable in this checkout")
    tracked_paths = {REPO_ROOT / line for line in tracked.stdout.splitlines()}
    missing = tracked_paths - set(found)
    assert not missing, f"tracked examples not discovered: {sorted(missing)}"
