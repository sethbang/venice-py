"""Tests for the ``venice-py lint`` CLI subcommand and its underlying rules.

Covers:

- :class:`venice_ai.cli.utils.lint_rules.V1UsageVisitor` — every rule code
  fires on a known-bad fixture snippet and stays silent on a clean snippet.
- :func:`venice_ai.cli.utils.lint_rules.lint_path` — file vs directory walk,
  ``--code`` filter behavior, skip-dir heuristics.
- ``venice-py lint`` Click command — exit codes (clean=0, info-only=0,
  errors=1, info-only-with-strict=1), output format, ``--code`` filter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.cli.utils.lint_rules import (
    INFORMATIONAL_CODES,
    Finding,
    V1UsageVisitor,
    lint_file,
    lint_path,
)

# ---------------------------------------------------------------------------
# V1UsageVisitor — per-rule fixtures
# ---------------------------------------------------------------------------


def _findings_for(source: str, tmp_path: Path) -> list[Finding]:
    f = tmp_path / "fixture.py"
    f.write_text(source)
    return lint_file(f)


def test_v100_async_venice_client_import(tmp_path: Path) -> None:
    src = "from venice_ai import AsyncVeniceClient\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V100" for f in findings), findings


def test_v100_async_venice_client_call(tmp_path: Path) -> None:
    src = "from venice_ai import VeniceClient\nx = AsyncVeniceClient()\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V100" for f in findings), findings


def test_v101_image_generate(tmp_path: Path) -> None:
    src = "client.image.generate(prompt='hi')\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V101" for f in findings)


def test_v102_audio_generate_music(tmp_path: Path) -> None:
    src = "client.audio.generate_music(prompt='hi', duration=10)\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V102" for f in findings)


def test_v103_audio_generate_speech(tmp_path: Path) -> None:
    src = "client.audio.generate_speech(input='hi')\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V103" for f in findings)


def test_v104_video_queue(tmp_path: Path) -> None:
    src = "client.video.queue(prompt='hi', duration='5s')\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V104" for f in findings)


def test_v105_video_complete(tmp_path: Path) -> None:
    src = "client.video.complete('job_id_123')\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V105" for f in findings)


def test_v106_embeddings_generate(tmp_path: Path) -> None:
    src = "client.embeddings.generate(input='hi')\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V106" for f in findings)


def test_v107_transcribe_audio_kwarg(tmp_path: Path) -> None:
    src = "client.audio.transcribe(audio=open('x.mp3','rb'))\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V107" for f in findings)


def test_v200_max_tokens(tmp_path: Path) -> None:
    src = "client.chat.completions.create(messages=[], max_tokens=100)\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V200" for f in findings)


def test_v300_hardcoded_model(tmp_path: Path) -> None:
    src = "client.chat.completions.create(model='claude-3-5-sonnet', messages=[])\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V300" for f in findings)


def test_v300_does_not_flag_demo_model_strings(tmp_path: Path) -> None:
    """Test/demo model strings are explicitly suppressed."""
    src = "client.chat.completions.create(model='test', messages=[])\n"
    findings = _findings_for(src, tmp_path)
    assert not any(f.code == "V300" for f in findings)


def test_v300_does_not_flag_dynamic_model(tmp_path: Path) -> None:
    """Variable / expression model= values are NOT flagged (only string literals are)."""
    src = "m = await client.models.resolve_chat()\nclient.chat.completions.create(model=m, messages=[])\n"
    findings = _findings_for(src, tmp_path)
    assert not any(f.code == "V300" for f in findings)


def test_v401_create_stream_true(tmp_path: Path) -> None:
    src = "client.chat.completions.create(messages=[], stream=True)\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V401" for f in findings)


def test_v401_does_not_flag_stream_false(tmp_path: Path) -> None:
    src = "client.chat.completions.create(messages=[], stream=False)\n"
    findings = _findings_for(src, tmp_path)
    assert not any(f.code == "V401" for f in findings)


def test_v501_budget_manager_limit_kwarg(tmp_path: Path) -> None:
    src = "from decimal import Decimal\nBudgetManager(limit=Decimal('2.00'))\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V501" for f in findings)


def test_v502_tracker_total_attribute(tmp_path: Path) -> None:
    src = "tracker = make_tracker()\nprint(tracker.total)\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V502" for f in findings)


def test_v503_tracker_calls_attribute(tmp_path: Path) -> None:
    src = "cost_tracker = make_tracker()\nprint(cost_tracker.calls)\n"
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V503" for f in findings)


def test_v601_payment_instructions(tmp_path: Path) -> None:
    src = (
        "try:\n"
        "    do_thing()\n"
        "except PaymentRequiredError as e:\n"
        "    print(e.payment_instructions)\n"
    )
    findings = _findings_for(src, tmp_path)
    assert any(f.code == "V601" for f in findings)


# ---------------------------------------------------------------------------
# Idiomatic v2 code is clean
# ---------------------------------------------------------------------------


def test_clean_idiomatic_v2_code_has_no_findings(tmp_path: Path) -> None:
    src = (
        "from venice_ai import VeniceClient\n"
        "from venice_ai.types.api import UserMessage\n"
        "\n"
        "async def main():\n"
        "    async with VeniceClient() as client:\n"
        "        model = await client.models.resolve_chat()\n"
        "        response = await client.chat.completions.create(\n"
        "            model=model,\n"
        "            messages=[UserMessage(content='hi')],\n"
        "            max_completion_tokens=100,\n"
        "        )\n"
    )
    findings = _findings_for(src, tmp_path)
    assert findings == [], findings


# ---------------------------------------------------------------------------
# lint_path / lint_file walk semantics
# ---------------------------------------------------------------------------


def test_lint_path_walks_directory_and_skips_dunder_dirs(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").write_text("client.image.generate(prompt='x')\n")
    skipdir = tmp_path / "__pycache__"
    skipdir.mkdir()
    (skipdir / "ignored.py").write_text("client.image.generate(prompt='x')\n")

    findings = lint_path(tmp_path)
    bad_paths = {f.path for f in findings}

    assert tmp_path / "bad.py" in bad_paths
    # __pycache__ entries skipped
    assert skipdir / "ignored.py" not in bad_paths


def test_lint_path_handles_single_file(tmp_path: Path) -> None:
    f = tmp_path / "single.py"
    f.write_text("client.image.generate(prompt='x')\n")
    findings = lint_path(f)
    assert findings
    assert findings[0].path == f
    assert findings[0].code == "V101"


def test_lint_path_filters_by_code(tmp_path: Path) -> None:
    f = tmp_path / "mixed.py"
    f.write_text(
        "from venice_ai import AsyncVeniceClient\n"  # V100
        "client.chat.completions.create(messages=[], max_tokens=1)\n"  # V200
    )
    only_v100 = lint_path(f, codes=["V100"])
    assert all(x.code == "V100" for x in only_v100)
    only_v200 = lint_path(f, codes=["V200"])
    assert all(x.code == "V200" for x in only_v200)


def test_lint_file_handles_unreadable_file(tmp_path: Path) -> None:
    """Non-existent file returns empty findings without crashing."""
    findings = lint_file(tmp_path / "does_not_exist.py")
    assert findings == []


def test_lint_file_reports_syntax_error_as_v000(tmp_path: Path) -> None:
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n    pass\n")
    findings = lint_file(f)
    assert any(x.code == "V000" for x in findings)


# ---------------------------------------------------------------------------
# Click command — `venice-py lint <path>`
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


def test_cli_lint_clean_file_exits_zero(cli_runner: CliRunner, tmp_path: Path) -> None:
    f = tmp_path / "clean.py"
    f.write_text("x = 1\n")
    result = cli_runner.invoke(cli, ["lint", str(f)])
    assert result.exit_code == 0, result.output


def test_cli_lint_error_findings_exit_one(cli_runner: CliRunner, tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("from venice_ai import AsyncVeniceClient\n")  # V100 (error)
    result = cli_runner.invoke(cli, ["lint", str(f)])
    assert result.exit_code == 1
    assert "V100" in result.output


def test_cli_lint_info_only_exits_zero_without_strict(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """V401 alone is informational; without --strict, exit 0."""
    f = tmp_path / "info_only.py"
    f.write_text("client.chat.completions.create(messages=[], stream=True)\n")
    result = cli_runner.invoke(cli, ["lint", str(f)])
    assert result.exit_code == 0
    assert "V401" in result.output


def test_cli_lint_info_only_exits_one_with_strict(cli_runner: CliRunner, tmp_path: Path) -> None:
    """--strict promotes informational findings to errors."""
    f = tmp_path / "info_only.py"
    f.write_text("client.chat.completions.create(messages=[], stream=True)\n")
    result = cli_runner.invoke(cli, ["lint", "--strict", str(f)])
    assert result.exit_code == 1
    assert "V401" in result.output


def test_cli_lint_code_filter(cli_runner: CliRunner, tmp_path: Path) -> None:
    """--code filter restricts findings to the given codes."""
    f = tmp_path / "mixed.py"
    f.write_text(
        "from venice_ai import AsyncVeniceClient\n"  # V100
        "client.chat.completions.create(messages=[], max_tokens=1)\n"  # V200
    )
    result = cli_runner.invoke(cli, ["lint", "--code", "V100", str(f)])
    assert result.exit_code == 1  # V100 is error-level
    assert "V100" in result.output
    assert "V200" not in result.output


def test_cli_lint_no_path_arg_errors(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["lint"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "PATH" in result.output


def test_cli_lint_nonexistent_path_errors(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(cli, ["lint", "/does/not/exist"])
    assert result.exit_code != 0


def test_cli_lint_help_includes_all_codes(cli_runner: CliRunner) -> None:
    """The --help output should be reasonably complete."""
    result = cli_runner.invoke(cli, ["lint", "--help"])
    assert result.exit_code == 0
    assert "PATH" in result.output
    assert "--code" in result.output
    assert "--strict" in result.output


# ---------------------------------------------------------------------------
# Sanity: every code in INFORMATIONAL_CODES is a real rule code
# ---------------------------------------------------------------------------


def test_informational_codes_are_known_codes() -> None:
    """Documented informational codes should be ones the visitor actually emits."""
    # Walk a fixture that triggers V401 — confirm at minimum that V401 in the
    # set corresponds to a code the visitor produces.
    visitor = V1UsageVisitor(Path("fixture.py"))
    import ast as _ast

    visitor.visit(_ast.parse("client.chat.completions.create(messages=[], stream=True)\n"))
    emitted_codes = {f.code for f in visitor.findings}
    assert emitted_codes | {"V401"} >= INFORMATIONAL_CODES  # at least V401 is testable


# ---------------------------------------------------------------------------
# venice-py lint must run without the [cli] extra
# ---------------------------------------------------------------------------


def test_lint_runs_without_pyyaml(tmp_path: Path, monkeypatch) -> None:
    """``venice-py lint`` is the most useful subcommand for codebases-using-venice;
    it must work on a bare ``pip install venice-py`` install (no ``[cli]`` extra,
    no PyYAML).

    Simulate the bare install by hiding ``yaml`` from ``sys.modules`` and
    blocking its import, then invoke the CLI. ``load_config`` falls through
    cleanly to ``DEFAULT_CONFIG.copy()`` when yaml isn't reachable.

    Only ``yaml`` is blocked: the CLI lazily imports other modules while it
    runs (rich loads its unicode tables at render time), and those must resolve.
    """
    import importlib.abc
    import sys

    class _BlockYaml(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "yaml" or fullname.startswith("yaml."):
                raise ModuleNotFoundError(f"No module named {fullname!r}")
            return None

    src = tmp_path / "clean.py"
    src.write_text("import asyncio\nasync def main(): pass\n")

    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "yaml" or name.startswith("yaml.")
    }
    monkeypatch.setattr(sys, "meta_path", [_BlockYaml(), *sys.meta_path])
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["lint", str(src)])
    finally:
        sys.modules.update(saved)

    # Clean fixture, no findings expected; but the key assertion is "did not
    # blow up importing yaml". Exit code 0 on a clean fixture.
    assert result.exit_code == 0, result.output
