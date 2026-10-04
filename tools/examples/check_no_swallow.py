#!/usr/bin/env python3
"""Flag the swallow-to-exit-0 anti-pattern in ``examples/`` scripts.

This flags example scripts that catch an error broadly, print it, and
continue — so ``main()`` returns normally, the success banner prints, and the
process exits 0 despite a real failure.

Three high-precision rules. "Broad" means ``except Exception`` / ``except
BaseException`` / bare ``except:``. Narrow *typed* excepts (``except
ValueError``, ``except (VeniceError, APIError)``, …) are intentional
error-demonstrations and are never flagged.

- **A** — a broad except that ``return``s a truthy literal (``True`` / a
  non-zero number / a non-empty string). Returning *success* from an error
  handler is masking.
- **B** — a broad except whose body is pure print/log + fall-through: no
  ``raise``, no ``sys.exit``/``exit`` call, no assignment, and no
  value-returning ``return``. That's the ``print → continue`` / bare-``return``
  swallow. Handlers that ``return False`` / set ``ok = False`` / ``raise`` pass.
- **C** — if a module defines ``main`` annotated ``-> int``, its result must
  reach ``sys.exit`` in the ``if __name__ == "__main__"`` block (recognises
  both ``sys.exit(asyncio.run(main()))`` and ``rc = asyncio.run(main());
  sys.exit(rc)``). A bare ``asyncio.run(main())`` statement discards the tally
  — the exact regression. Gated on the ``-> int`` annotation, so raise-based
  ``-> None`` mains are not flagged.

**Hard vs advisory.** Rules **A** and **C** are precise (zero false-positive in
practice) and are the CI gate. Rule **B** is irreducibly
imprecise — a ``print`` + fall-through handler is syntactically identical to a
legitimate best-effort sub-step (optional enrichment, availability probe,
failure tracked via ``results.append((name, False, e))`` or a post-handler
``return None``), so it would false-positive as a hard gate. It runs only under
``--strict`` as an advisory scan for manual swallow-hunting.

Exit 0 when clean, 1 on hard findings (``--strict`` also fails on B).
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"

#: Exception names treated as "broad". Typed/specific names are carved out as
#: intentional error-demonstrations.
BROAD_NAMES = {"Exception", "BaseException"}

#: Rules that fail CI by default. Rule B is advisory-only (``--strict``).
HARD_RULES = {"A", "C"}

#: Directory names under an examples tree that never hold examples: the
#: examples' generated output dir, virtualenvs, installed packages, and caches.
#: Hidden (dot) directories are skipped too, and so is any directory holding a
#: ``pyvenv.cfg`` whatever it is called.
EXCLUDED_DIR_NAMES = frozenset(
    {"results", "venv", "env", "site-packages", "__pycache__", "node_modules"}
)


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    rule: str
    message: str


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """A bare ``except:`` or one catching ``Exception``/``BaseException``."""
    t = handler.type
    if t is None:
        return True
    candidates = t.elts if isinstance(t, ast.Tuple) else [t]
    return any(isinstance(c, ast.Name) and c.id in BROAD_NAMES for c in candidates)


def _is_truthy_constant(node: ast.expr) -> bool:
    if not isinstance(node, ast.Constant):
        return False
    return bool(node.value)


def _is_exit_call(node: ast.AST) -> bool:
    """``sys.exit(...)`` or bare ``exit(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "exit":
        return True
    return isinstance(func, ast.Name) and func.id == "exit"


def _iter_handler_nodes(body: list[ast.stmt]):
    """Walk handler statements, descending into control flow but NOT into
    nested function/class/lambda scopes (those are separate concerns)."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        stack.extend(ast.iter_child_nodes(node))


def _handler_signals(body: list[ast.stmt]) -> tuple[bool, bool]:
    """Return ``(has_failure_signal, returns_truthy)``.

    ``has_failure_signal`` is True if the handler re-raises, exits, assigns, or
    returns a value (i.e. it does *something* with the error beyond printing).
    """
    has_signal = False
    returns_truthy = False
    for node in _iter_handler_nodes(body):
        if isinstance(node, (ast.Raise, ast.Assign, ast.AugAssign, ast.AnnAssign)):
            has_signal = True
        elif isinstance(node, ast.Return):
            value = node.value
            is_none = value is None or (isinstance(value, ast.Constant) and value.value is None)
            if not is_none:
                has_signal = True
                if _is_truthy_constant(value):
                    returns_truthy = True
        elif _is_exit_call(node):
            has_signal = True
    return has_signal, returns_truthy


def _is_main_guard(test: ast.expr) -> bool:
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_asyncio_run_main(node: ast.AST) -> bool:
    """Match ``asyncio.run(main())`` / ``run(main())``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    is_run = (isinstance(func, ast.Attribute) and func.attr == "run") or (
        isinstance(func, ast.Name) and func.id == "run"
    )
    if not is_run or not node.args:
        return False
    arg = node.args[0]
    return isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name) and arg.func.id == "main"


def _main_returns_int(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "main"
            and isinstance(node.returns, ast.Name)
            and node.returns.id == "int"
        ):
            return True
    return False


def _check_main_exit(tree: ast.Module, filename: str) -> list[Violation]:
    if not _main_returns_int(tree):
        return []
    guard = next((n for n in tree.body if isinstance(n, ast.If) and _is_main_guard(n.test)), None)
    if guard is None:
        return []
    # A bare ``asyncio.run(main())`` expression statement discards the int result.
    out: list[Violation] = []
    for node in ast.walk(guard):
        if isinstance(node, ast.Expr) and _is_asyncio_run_main(node.value):
            out.append(
                Violation(
                    filename,
                    node.lineno,
                    "C",
                    "main() -> int result is discarded; use sys.exit(asyncio.run(main())) "
                    "so the failure tally reaches the process exit code",
                )
            )
    return out


def find_violations(filename: str, source: str) -> list[Violation]:
    """Apply rules A/B/C to one example's source. Pure + unit-testable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - py_compile gate catches these too
        return [Violation(filename, exc.lineno or 0, "syntax", f"syntax error: {exc}")]

    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _is_broad(node):
            has_signal, returns_truthy = _handler_signals(node.body)
            if returns_truthy:
                violations.append(
                    Violation(
                        filename,
                        node.lineno,
                        "A",
                        "broad except returns a truthy value (masks a failure as success)",
                    )
                )
            elif not has_signal:
                violations.append(
                    Violation(
                        filename,
                        node.lineno,
                        "B",
                        "broad except swallows (print/continue without re-raise, sys.exit, "
                        "a failure flag, or a value-returning return)",
                    )
                )
    violations.extend(_check_main_exit(tree, filename))
    return violations


def _is_excluded_dir(path: Path) -> bool:
    name = path.name
    return name in EXCLUDED_DIR_NAMES or name.startswith(".") or (path / "pyvenv.cfg").is_file()


def discover_example_files(root: Path) -> list[Path]:
    """Return the example ``.py`` files under ``root``, sorted.

    Walks the filesystem and prunes non-example directories (see
    ``EXCLUDED_DIR_NAMES``, hidden dirs, and any virtualenv detected by its
    ``pyvenv.cfg``) before descending into them. Exclusions apply only to
    directories *below* ``root``, so a checkout that itself lives under a hidden
    or ``results`` parent is still scanned.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_excluded_dir(here / d)]
        found.extend(here / f for f in filenames if f.endswith(".py"))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=None,
        help="Files/dirs to check (default: examples/).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also report (and fail on) the advisory Rule B heuristic.",
    )
    args = parser.parse_args(argv)

    targets: list[Path] = []
    roots = args.paths or [EXAMPLES_ROOT]
    for root in roots:
        if root.is_dir():
            targets.extend(discover_example_files(root))
        elif root.suffix == ".py":
            # An explicitly named file is checked even inside an excluded dir.
            targets.append(root)

    if not targets:
        print(f"❌ no example files found under {', '.join(map(str, roots))}; nothing was checked.")
        return 1

    all_violations: list[Violation] = []
    for path in targets:
        rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
        all_violations.extend(find_violations(str(rel), path.read_text(encoding="utf-8")))

    hard = [v for v in all_violations if v.rule in HARD_RULES]
    advisory = [v for v in all_violations if v.rule not in HARD_RULES]

    fatal = hard + (advisory if args.strict else [])
    if fatal:
        print(f"❌ swallow-to-exit-0 check failed ({len(fatal)} finding(s)):\n")
        for v in fatal:
            print(f"  {v.file}:{v.line}  [{v.rule}]  {v.message}")
        print(
            "\nExamples must fail loudly: don't return success from a broad except "
            "(Rule A), and have main() -> int reach sys.exit(asyncio.run(main())) "
            "(Rule C). See examples/basic/quick_start.py."
        )
        return 1

    if advisory and not args.strict:
        print(
            f"ℹ️  {len(advisory)} advisory (Rule B) finding(s) — broad excepts that "
            "print + fall through. These are imprecise (best-effort sub-steps look "
            "identical to swallows); review manually via --strict. Not failing CI."
        )
    print(f"✅ {len(targets)} example file(s): no Rule A/C swallow-to-exit-0 patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
