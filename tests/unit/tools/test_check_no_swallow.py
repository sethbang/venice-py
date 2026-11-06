"""Unit tests for the examples swallow-to-exit-0 checker.

The checker (``tools/examples/check_no_swallow.py``) must catch the
swallow-to-exit-0 anti-pattern the live audit found, with no hard findings on the
remediated tree. Four rules:

- A: a *broad* except (``Exception`` / ``BaseException`` / bare) returning a
  truthy literal — returning success from an error handler.
- B: a broad except whose body is pure print/log + fall-through (no ``raise``,
  ``sys.exit``, assignment, or value-returning ``return``).
- C: if ``main`` is annotated ``-> int``, its result must reach ``sys.exit``.
- D: a literal swallow: exiting 0 or ``return 0`` from ``main`` in a broad except,
  an empty broad except, a report-only one in the main guard, or
  ``suppress(Exception)``.

Narrow typed excepts (``except ValueError`` / ``except (VeniceError, APIError)``)
are intentional error-demos and are never flagged.
"""

from __future__ import annotations

import importlib.util
import subprocess
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


# ---- literal swallow forms ----


@pytest.mark.parametrize(
    "exit_form",
    [
        "sys.exit(0)",
        "sys.exit()",
        "sys.exit(None)",
        "sys.exit(False)",
        "exit(0)",
        "exit()",
        "quit(0)",
        "os._exit(0)",
        "raise SystemExit",
        "raise SystemExit(0)",
        "raise SystemExit()",
    ],
)
def test_broad_handler_exiting_0_is_hard(exit_form: str) -> None:
    source = f"""
        def work():
            try:
                thing()
            except Exception as e:
                print(e)
                {exit_form}
        """
    assert _rules(source) == {"D"}


@pytest.mark.parametrize(
    "exit_form",
    [
        "sys.exit(1)",
        "sys.exit(77)",
        "exit(1)",
        "os._exit(2)",
        "raise SystemExit(1)",
        "sys.exit(rc)",
    ],
)
def test_broad_handler_exiting_nonzero_is_clean(exit_form: str) -> None:
    source = f"""
        def work():
            try:
                thing()
            except Exception as e:
                print(e)
                {exit_form}
        """
    assert _rules(source) == set()


@pytest.mark.parametrize("handler", ["except Exception:", "except BaseException:", "except:"])
def test_return_0_in_a_broad_handler_of_main_is_hard(handler: str) -> None:
    source = f"""
        async def main() -> int:
            try:
                await thing()
            {handler}
                print("failed")
                return 0
            return 0
        """
    assert _rules(source) == {"D"}


def test_return_0_in_a_broad_handler_of_a_helper_is_not_rule_d() -> None:
    source = """
        def count() -> int:
            try:
                return len(thing())
            except Exception:
                return 0
        """
    assert _rules(source) == set()


def test_return_0_from_a_function_nested_in_main_belongs_to_that_function() -> None:
    source = """
        def main() -> int:
            def count() -> int:
                try:
                    return len(thing())
                except Exception:
                    return 0
            return count()
        """
    assert _rules(source) == set()


def test_return_0_from_a_typed_handler_of_main_is_clean() -> None:
    source = """
        def main() -> int:
            try:
                thing()
            except ValueError:
                return 0
            return 1
        """
    assert _rules(source) == set()


@pytest.mark.parametrize("value", ["1", "2", "77", "-1"])
def test_nonzero_return_from_main_is_a_failure_not_a_mask(value: str) -> None:
    source = f"""
        def main() -> int:
            try:
                thing()
            except Exception as e:
                print(e)
                return {value}
            return 0
        """
    assert _rules(source) == set()


def test_nonzero_return_from_a_helper_is_still_rule_a() -> None:
    source = """
        def helper() -> int:
            try:
                thing()
            except Exception:
                return 1
        """
    assert _rules(source) == {"A"}


@pytest.mark.parametrize("body", ["pass", "...", "continue", "pass\n                    pass"])
def test_broad_handler_with_an_empty_body_is_hard(body: str) -> None:
    source = f"""
        def work():
            for item in items:
                try:
                    thing(item)
                except Exception:
                    {body}
        """
    assert _rules(source) == {"D"}


def test_typed_handler_with_an_empty_body_is_clean() -> None:
    source = """
        try:
            thing()
        except KeyError:
            pass
        """
    assert _rules(source) == set()


@pytest.mark.parametrize(
    "opener",
    [
        "contextlib.suppress(Exception)",
        "suppress(Exception)",
        "contextlib.suppress(BaseException)",
        "suppress(KeyError, Exception)",
    ],
)
def test_suppressing_broad_exceptions_is_hard(opener: str) -> None:
    source = f"""
        with {opener}:
            thing()
        """
    assert _rules(source) == {"D"}


def test_suppressing_broad_exceptions_in_async_with_is_hard() -> None:
    source = """
        async def work():
            async with limiter, suppress(Exception):
                await thing()
        """
    assert _rules(source) == {"D"}


def test_suppressing_a_typed_exception_is_clean() -> None:
    source = """
        with suppress(asyncio.CancelledError, KeyError):
            thing()
        """
    assert _rules(source) == set()


def test_main_guard_swallowing_a_run_is_hard() -> None:
    source = """
        if __name__ == "__main__":
            try:
                asyncio.run(main())
            except Exception as e:
                print(f"failed: {e}")
        """
    assert _rules(source) == {"D"}


def test_main_guard_exiting_nonzero_is_clean() -> None:
    source = """
        if __name__ == "__main__":
            try:
                asyncio.run(main())
            except Exception as e:
                print(f"failed: {e}")
                sys.exit(1)
        """
    assert _rules(source) == set()


@pytest.mark.parametrize(
    "literal",
    ["True, str(e)", "[True]", '{"ok": True}', '{"success": True, "error": str(e)}', 'f"ok {e}"'],
)
def test_broad_handler_returning_a_success_literal_is_rule_a(literal: str) -> None:
    source = f"""
        def work():
            try:
                thing()
            except Exception as e:
                return {literal}
        """
    assert _rules(source) == {"A"}


@pytest.mark.parametrize(
    "literal",
    [
        "False, str(e)",
        "[False, e]",
        '{"ok": False, "error": str(e)}',
        '{"error": str(e)}',
        "(None, e)",
    ],
)
def test_broad_handler_returning_a_failure_literal_is_clean(literal: str) -> None:
    source = f"""
        def work():
            try:
                thing()
            except Exception as e:
                return {literal}
        """
    assert _rules(source) == set()


# ---- the command itself ----


PLANTED = {
    "return 0 in main": """
import asyncio
import sys


async def main() -> int:
    try:
        await asyncio.sleep(0)
    except Exception as e:
        print(e)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
""",
    "sys.exit(0)": """
import sys

try:
    print("work")
except Exception as e:
    print(e)
    sys.exit(0)
""",
    "pass": """
try:
    print("work")
except Exception:
    pass
""",
    "suppress": """
import contextlib

with contextlib.suppress(Exception):
    print("work")
""",
}


@pytest.mark.parametrize("name", PLANTED)
@pytest.mark.parametrize("strict", [False, True], ids=["default", "strict"])
def test_command_fails_on_a_planted_file_outside_examples(
    tmp_path: Path, name: str, strict: bool
) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(PLANTED[name], encoding="utf-8")

    command = [sys.executable, str(TOOL_PATH), str(planted), *(["--strict"] if strict else [])]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 1, result.stdout
    assert "[D]" in result.stdout


def test_command_passes_a_clean_file_outside_examples(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text(
        'import sys\n\ntry:\n    print("work")\nexcept Exception as e:\n'
        "    print(e)\n    sys.exit(1)\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--strict", str(clean)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout


# ---------------------------------------------------------------------------
# What Rule D means by status 0, and where the main guard ends
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["False", "0.0"], ids=["false", "float"])
def test_falsy_non_int_from_main_is_not_return_0(value: str) -> None:
    src = f"""
    import sys

    def main():
        try:
            run()
        except Exception as e:
            print(e)
            return {value}
        return True

    sys.exit(0 if main() else 1)
    """
    assert "D" not in _rules(src)


@pytest.mark.parametrize("value", ["0, str(e)", "(0, 'failed')", "[0, str(e)]"])
def test_a_result_led_by_0_from_main_is_return_0(value: str) -> None:
    src = f"""
    def main():
        try:
            run()
        except Exception as e:
            return {value}
    """
    assert "D" in _rules(src)


def test_a_result_led_by_false_from_main_reports_the_failure() -> None:
    src = """
    def main():
        try:
            run()
        except Exception as e:
            return False, str(e)
    """
    assert _rules(src) == set()


@pytest.mark.parametrize("call", ["self.exit()", "parser.exit(2)", "app.exit()"])
def test_exit_method_of_another_object_is_not_a_process_exit_0(call: str) -> None:
    src = f"""
    def handle(self):
        try:
            run()
        except Exception as e:
            {call}
    """
    assert "D" not in _rules(src)


def test_sys_exit_ex_ok_is_status_0() -> None:
    src = """
    import os
    import sys

    try:
        run()
    except Exception:
        sys.exit(os.EX_OK)
    """
    assert "D" in _rules(src)


def test_sys_exit_float_is_not_status_0() -> None:
    src = """
    import sys

    try:
        run()
    except Exception:
        sys.exit(0.0)
    """
    assert "D" not in _rules(src)


def test_builtins_exception_is_broad() -> None:
    src = """
    import builtins

    try:
        run()
    except builtins.Exception:
        pass
    """
    assert "D" in _rules(src)


def test_else_branch_of_the_main_guard_is_not_the_guard() -> None:
    src = """
    if __name__ == "__main__":
        pass
    else:
        try:
            run()
        except Exception as e:
            print(e)
    """
    assert "D" not in _rules(src)
    assert "B" in _rules(src)


def test_body_of_the_main_guard_still_is_the_guard() -> None:
    src = """
    if __name__ == "__main__":
        try:
            run()
        except Exception as e:
            print(e)
    else:
        pass
    """
    assert "D" in _rules(src)
