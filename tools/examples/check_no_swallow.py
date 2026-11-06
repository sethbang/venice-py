#!/usr/bin/env python3
"""Flag the swallow-to-exit-0 anti-pattern in ``examples/`` scripts.

This flags example scripts that catch an error broadly, print it, and
continue — so ``main()`` returns normally, the success banner prints, and the
process exits 0 despite a real failure.

Four high-precision rules. "Broad" means ``except Exception`` / ``except
BaseException`` / bare ``except:``. Narrow *typed* excepts (``except
ValueError``, ``except (VeniceError, APIError)``, …) are intentional
error-demonstrations and are never flagged.

- **A** — a broad except that ``return``s a truthy literal (``True`` / a
  non-zero number / a non-empty string / a sequence led by one / a mapping with
  a truthy ``ok``, ``success`` or ``passed``). Returning *success* from an error
  handler is masking. A non-zero number returned from ``main`` is a failure exit
  code, not a mask. A literal led by a falsy value (``False, str(e)``) reports
  the failure and passes.
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
- **D** — a literal swallow, whatever else the handler does: a broad except that
  exits 0 (``sys.exit(0)``, ``sys.exit()``, ``exit()``, ``quit()``,
  ``os._exit(0)``, ``sys.exit(os.EX_OK)``, ``raise SystemExit``), that returns the
  integer ``0`` or a result led by it (``return 0, str(e)``) from ``main``, or whose
  whole body is ``pass`` / ``...`` / ``continue`` / ``break``; a broad except around
  the ``if __name__ == "__main__"`` block's work that only reports the error; and
  ``contextlib.suppress(Exception)`` or ``BaseException``.

**What this cannot see.** It reads one file's syntax, so these forms pass:
``suppress`` under an alias, held in a variable or given a tuple; ``sys`` imported
under another name; a status in a variable (``sys.exit(rc)``); a helper that returns
``0`` to ``main``; ``return False`` reaching ``sys.exit(main())`` directly, which
exits 0; ``parser.exit()`` and other ``exit`` methods, which it cannot tell from a
method of the same name; an exit from a function defined inside the handler; and a
guard handler that assigns the error and does nothing with it.

**Hard vs advisory.** Rules **A**, **C** and **D** are the CI gate: each finding is
meant to be a real swallow, and the rules are tuned against this tree, not proven
free of false positives. Rule **B** is irreducibly
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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples"

#: Exception names treated as "broad". Typed/specific names are carved out as
#: intentional error-demonstrations.
BROAD_NAMES = {"Exception", "BaseException"}

#: Rules that fail CI by default. Rule B is advisory-only (``--strict``).
HARD_RULES = {"A", "C", "D"}

#: Mapping keys whose truthy literal value claims success (Rule A).
SUCCESS_KEYS = frozenset({"ok", "success", "passed"})

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
    return any(_names_broad_exception(c) for c in candidates)


def _names_broad_exception(node: ast.expr) -> bool:
    """``Exception``, ``BaseException``, or either reached through ``builtins``."""
    if isinstance(node, ast.Attribute):
        return (
            isinstance(node.value, ast.Name)
            and node.value.id == "builtins"
            and node.attr in BROAD_NAMES
        )
    return isinstance(node, ast.Name) and node.id in BROAD_NAMES


def _is_truthy_literal(node: ast.expr | None) -> bool:
    """A literal that reads as success. A literal led by a falsy value reports failure."""
    if isinstance(node, ast.Constant):
        return bool(node.value)
    if isinstance(node, ast.JoinedStr):
        return any(_is_truthy_literal(part) for part in node.values)
    if isinstance(node, (ast.Tuple, ast.List)):
        return bool(node.elts) and _is_truthy_literal(node.elts[0])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_truthy_literal(node.operand)
    if isinstance(node, ast.Dict):
        return any(
            isinstance(key, ast.Constant)
            and key.value in SUCCESS_KEYS
            and _is_truthy_literal(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


def _is_int_literal(node: ast.expr | None) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _is_int_zero(node: ast.expr | None) -> bool:
    """The integer ``0`` itself, which is not ``False`` or ``0.0``."""
    return isinstance(node, ast.Constant) and type(node.value) is int and node.value == 0


def _is_exit_call(node: ast.AST) -> bool:
    """Anything named ``exit`` or ``_exit``, or bare ``exit(...)`` / ``quit(...)``.

    Doing this in a handler means the error was acted on. Whether it ends the
    process with status 0 is for :func:`_is_process_exit` to say.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr in {"exit", "_exit"}:
        return True
    return isinstance(func, ast.Name) and func.id in {"exit", "quit"}


def _is_process_exit(node: ast.AST) -> bool:
    """``sys.exit(...)``, ``os._exit(...)``, or bare ``exit(...)`` / ``quit(...)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return isinstance(func.value, ast.Name) and (func.value.id, func.attr) in {
            ("sys", "exit"),
            ("os", "_exit"),
        }
    return isinstance(func, ast.Name) and func.id in {"exit", "quit"}


def _is_success_status(args: list[ast.expr]) -> bool:
    """Exit-call or ``SystemExit`` arguments that mean status 0.

    Other objects, ``0.0`` among them, print and exit 1.
    """
    if not args:
        return True
    first = args[0]
    if isinstance(first, ast.Attribute):
        return (
            isinstance(first.value, ast.Name) and first.value.id == "os" and first.attr == "EX_OK"
        )
    return isinstance(first, ast.Constant) and (
        first.value is None or first.value is False or _is_int_zero(first)
    )


def _exits_zero(node: ast.AST) -> bool:
    if isinstance(node, ast.Raise):
        raised = node.exc
        if isinstance(raised, ast.Name):
            return raised.id == "SystemExit"
        return (
            isinstance(raised, ast.Call)
            and isinstance(raised.func, ast.Name)
            and raised.func.id == "SystemExit"
            and _is_success_status(raised.args)
        )
    return _is_process_exit(node) and _is_success_status(node.args)  # type: ignore[attr-defined]


def _is_empty_body(body: list[ast.stmt]) -> bool:
    """A handler that does nothing at all with the error."""
    return all(
        isinstance(stmt, (ast.Pass, ast.Continue, ast.Break))
        or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        )
        for stmt in body
    )


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


def _handler_signals(body: list[ast.stmt], *, in_main: bool = False) -> tuple[bool, bool]:
    """Return ``(has_failure_signal, returns_truthy)``.

    ``has_failure_signal`` is True if the handler re-raises, exits, assigns, or
    returns a value (i.e. it does *something* with the error beyond printing).
    ``in_main`` marks a handler of ``main``, whose returned number is an exit code.
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
                if _is_truthy_literal(value) and not (in_main and _is_int_literal(value)):
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


def _scoped_handlers(
    node: ast.AST, function: ast.AST | None = None, in_guard: bool = False
) -> Iterator[tuple[ast.ExceptHandler, ast.AST | None, bool]]:
    """Yield each ``except`` with its nearest enclosing function and main-guard status."""
    yield from _scan_scope(ast.iter_child_nodes(node), function, in_guard)


def _scan_scope(
    nodes: Iterable[ast.AST], function: ast.AST | None, in_guard: bool
) -> Iterator[tuple[ast.ExceptHandler, ast.AST | None, bool]]:
    for child in nodes:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _scoped_handlers(child, child, False)
        elif isinstance(child, ast.If) and _is_main_guard(child.test):
            yield from _scan_scope(child.body, function, function is None)
            yield from _scan_scope(child.orelse, function, in_guard)
        else:
            if isinstance(child, ast.ExceptHandler):
                yield child, function, in_guard
            yield from _scoped_handlers(child, function, in_guard)


def _is_main(function: ast.AST | None) -> bool:
    return isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.name == "main"


def _literal_swallow(
    handler: ast.ExceptHandler, function: ast.AST | None, in_guard: bool, has_signal: bool
) -> str | None:
    """Describe the Rule D form a broad handler takes, or ``None``."""
    if _is_empty_body(handler.body):
        return "broad except does nothing with the error"
    for node in _iter_handler_nodes(handler.body):
        if _exits_zero(node):
            return "broad except exits with status 0"
        if _is_main(function) and isinstance(node, ast.Return):
            value = node.value
            led_by = (
                value.elts[0] if isinstance(value, (ast.Tuple, ast.List)) and value.elts else value
            )
            if _is_int_zero(led_by):
                return "broad except in main returns 0, the success exit code"
    if in_guard and not has_signal:
        return "broad except in the main guard only reports the error, so the process exits 0"
    return None


def _check_suppress(tree: ast.Module, filename: str) -> list[Violation]:
    """Flag ``contextlib.suppress`` of a broad exception (Rule D)."""
    out: list[Violation] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name == "suppress" and any(
                isinstance(arg, ast.Name) and arg.id in BROAD_NAMES for arg in call.args
            ):
                out.append(
                    Violation(
                        filename, node.lineno, "D", "suppress() of a broad exception swallows it"
                    )
                )
    return out


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
    """Apply rules A/B/C/D to one example's source. Pure + unit-testable."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # pragma: no cover - py_compile gate catches these too
        return [Violation(filename, exc.lineno or 0, "syntax", f"syntax error: {exc}")]

    violations: list[Violation] = []
    for node, function, in_guard in _scoped_handlers(tree):
        if _is_broad(node):
            has_signal, returns_truthy = _handler_signals(node.body, in_main=_is_main(function))
            swallow = _literal_swallow(node, function, in_guard, has_signal)
            if swallow:
                violations.append(Violation(filename, node.lineno, "D", swallow))
            elif returns_truthy:
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
    violations.extend(_check_suppress(tree, filename))
    violations.extend(_check_main_exit(tree, filename))
    return sorted(violations, key=lambda v: v.line)


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
            "(Rule A), have main() -> int reach sys.exit(asyncio.run(main())) "
            "(Rule C), and don't exit 0 from or empty out a broad except (Rule D). "
            "See examples/basic/quick_start.py."
        )
        return 1

    if advisory and not args.strict:
        print(
            f"ℹ️  {len(advisory)} advisory (Rule B) finding(s) — broad excepts that "
            "print + fall through. These are imprecise (best-effort sub-steps look "
            "identical to swallows); review manually via --strict. Not failing CI."
        )
    print(f"✅ {len(targets)} example file(s): no Rule A/C/D swallow-to-exit-0 patterns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
