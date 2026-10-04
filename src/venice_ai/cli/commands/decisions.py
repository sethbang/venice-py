"""Decisions command for Venice AI CLI — Ask a decision model typed questions."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import click
from pydantic import TypeAdapter, ValidationError

from venice_ai.cli.utils.console import console, print_error
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.decisions import (
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
)
from venice_ai.types.api.requests.decisions import (
    ChoiceQuestion,
    DecisionQuestion,
    DecisionState,
    NoulQuestion,
    ScoreQuestion,
)

_QUESTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(DecisionQuestion)

#: How many entries of a probability distribution the table shows.
_TOP_PROBABILITIES = 3


@click.command("decisions")
@click.argument("state", required=False)
@click.option(
    "--noul",
    "noul_questions",
    nargs=2,
    multiple=True,
    metavar="ID QUESTION",
    help="Add a yes/no question, answered as a probability (repeatable)",
)
@click.option(
    "--choice",
    "choice_questions",
    nargs=3,
    multiple=True,
    metavar="ID QUESTION OPTIONS",
    help="Add a pick-one question; OPTIONS is a comma-separated list (repeatable)",
)
@click.option(
    "--score",
    "score_questions",
    nargs=3,
    multiple=True,
    metavar="ID QUESTION LEVELS",
    help="Add a rubric question; LEVELS is comma-separated, lowest first (repeatable)",
)
@click.option(
    "--questions",
    "-q",
    "questions_path",
    type=click.Path(dir_okay=False, allow_dash=True),
    default=None,
    help="JSON file mapping question ids to questions in the API shape ('-' reads stdin)",
)
@click.option(
    "--state-json",
    is_flag=True,
    help="Parse STATE as JSON and send it as structured state (object or array)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Decision model to use (defaults to the API-recommended model)",
)
@click.option("--json", "output_json", is_flag=True, help="Output the full response as JSON")
@click.pass_context
def decisions(
    ctx,
    state,
    noul_questions,
    choice_questions,
    score_questions,
    questions_path,
    state_json,
    model,
    output_json,
):
    """Ask a decision model typed questions about a piece of state.

    Decision ("System One") models return a calibrated answer per question
    instead of text: a probability for --noul, an option with its full
    distribution for --choice, and a position on an ordered rubric for
    --score. Every question is evaluated independently against the same
    STATE in one request. The endpoint is beta.

    STATE is text to evaluate; pipe it via stdin to omit the argument. With
    --state-json it is parsed as JSON, for records or chat logs.

    Examples:

    \b
      # Route a support ticket
      venice-py decisions "My payouts have failed for three days" \\
        --noul urgent "Does this message convey urgency?" \\
        --choice team "Which team should handle this?" "billing,technical,account" \\
        --score mood "How frustrated is the customer?" "Calm,Frustrated,Very angry"

    \b
      # Evaluate structured state
      venice-py decisions --state-json '{"plan": "free", "failed_payments": 3}' \\
        --noul at_risk "Is this account at risk of churning?"

    \b
      # Full question definitions (choice descriptions, noul labels) from a file
      cat ticket.txt | venice-py decisions --questions questions.json --json
    """
    resolved_state = read_state(state, state_json=state_json, stdin_taken=questions_path == "-")
    questions = build_questions(noul_questions, choice_questions, score_questions, questions_path)
    asyncio.run(_decisions_async(ctx, resolved_state, questions, model, output_json))


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def _split_list(raw: str, *, param_hint: str, question_id: str) -> list[str]:
    """Split a comma-separated option or level list, rejecting empty and repeated entries."""
    items = [item.strip() for item in raw.split(",")]
    items = [item for item in items if item]
    duplicates = sorted({item for item in items if items.count(item) > 1})
    if duplicates:
        raise click.BadParameter(
            f"question {question_id!r} repeats {', '.join(map(repr, duplicates))}.",
            param_hint=param_hint,
        )
    return items


def _load_questions_file(path: str) -> dict[str, Any]:
    """Read and validate a JSON question map from ``path`` (``-`` for stdin)."""
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise click.BadParameter(f"cannot read {path!r}: {exc}", param_hint="--questions") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"invalid JSON: {exc}", param_hint="--questions") from exc

    if not isinstance(data, dict):
        raise click.BadParameter(
            "expected a JSON object mapping question ids to questions.",
            param_hint="--questions",
        )

    questions: dict[str, Any] = {}
    for question_id, entry in data.items():
        try:
            questions[question_id] = _QUESTION_ADAPTER.validate_python(entry)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in err['loc']) or 'question'}: {err['msg']}"
                for err in exc.errors()
            )
            raise click.BadParameter(
                f"question {question_id!r} is invalid ({problems}).",
                param_hint="--questions",
            ) from exc
    return questions


def build_questions(
    noul_questions: tuple[tuple[str, str], ...],
    choice_questions: tuple[tuple[str, str, str], ...],
    score_questions: tuple[tuple[str, str, str], ...],
    questions_path: str | None,
) -> dict[str, Any]:
    """Merge the inline question flags and the ``--questions`` file into one map.

    :raises click.UsageError: If no question is given, or an id is used twice.
    :raises click.BadParameter: If a flag or the file describes an invalid question.
    """
    entries: list[tuple[str, Any]] = []

    for question_id, instructions in noul_questions:
        entries.append((question_id, NoulQuestion(instructions=instructions)))

    for question_id, instructions, raw_options in choice_questions:
        options = _split_list(raw_options, param_hint="--choice", question_id=question_id)
        if not options:
            raise click.BadParameter(
                f"question {question_id!r} needs at least one option.", param_hint="--choice"
            )
        criteria: dict[str, str | None] = dict.fromkeys(options)
        entries.append((question_id, ChoiceQuestion(instructions=instructions, criteria=criteria)))

    for question_id, instructions, raw_levels in score_questions:
        levels = _split_list(raw_levels, param_hint="--score", question_id=question_id)
        if len(levels) < 2:
            raise click.BadParameter(
                f"question {question_id!r} needs at least two levels.", param_hint="--score"
            )
        entries.append((question_id, ScoreQuestion(instructions=instructions, criteria=levels)))

    if questions_path is not None:
        entries.extend(_load_questions_file(questions_path).items())

    questions: dict[str, Any] = {}
    for question_id, question in entries:
        if not question_id.strip():
            raise click.UsageError("Question ids cannot be empty.")
        if question_id in questions:
            raise click.UsageError(f"Question id {question_id!r} is defined more than once.")
        questions[question_id] = question

    if not questions:
        raise click.UsageError(
            "At least one question is required: use --noul, --choice, --score or --questions."
        )
    return questions


def read_state(state: str | None, *, state_json: bool, stdin_taken: bool) -> DecisionState:
    """Return the state to evaluate, from the argument or piped stdin.

    :raises click.UsageError: If stdin is needed for both the state and ``--questions -``.
    :raises click.ClickException: If no state is available.
    :raises click.BadParameter: If ``--state-json`` is set and the state is not
        a non-empty JSON object or array.
    """
    if state is None:
        if stdin_taken:
            raise click.UsageError(
                "STATE must be given as an argument when --questions reads from stdin."
            )
        if not sys.stdin.isatty():
            state = sys.stdin.read()

    if state is None or not state.strip():
        raise click.ClickException(
            "State is required. Provide it as an argument or pipe via stdin."
        )

    if not state_json:
        return state.strip()

    try:
        parsed = json.loads(state)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"invalid JSON: {exc}", param_hint="--state-json") from exc
    if not isinstance(parsed, (dict, list)) or not parsed:
        raise click.BadParameter(
            "structured state must be a non-empty JSON object or array.",
            param_hint="--state-json",
        )
    return parsed


# ---------------------------------------------------------------------------
# Request and output
# ---------------------------------------------------------------------------


async def _decisions_async(
    ctx: click.Context,
    state: DecisionState,
    questions: dict[str, Any],
    model: str | None,
    output_json: bool,
) -> None:
    from venice_ai import VeniceClient
    from venice_ai.cli._model_defaults import resolve_default_model
    from venice_ai.cli.config import get_client_kwargs, load_config
    from venice_ai.cli.utils.console import enable_plain_mode, is_plain_mode

    plain = ctx.obj.get("plain", False) if ctx.obj else False
    if plain and not is_plain_mode():
        enable_plain_mode()
    config = ctx.obj.get("config", load_config()) if ctx.obj else load_config()

    try:
        async with VeniceClient(**get_client_kwargs()) as client:
            model = await resolve_default_model(client, config, "decision", explicit=model)
            response = await client.decisions.create(model=model, state=state, questions=questions)
    except VeniceError as e:
        print_error(f"Venice API error: {e}")
        raise SystemExit(1) from e

    if output_json:
        click.echo(json.dumps(response.model_dump(mode="json"), indent=2))
        return

    render_response(response, list(questions))


def _top_probabilities(
    probabilities: dict[str, float], labels: dict[str, str] | None = None
) -> str:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    shown = ranked[:_TOP_PROBABILITIES]
    text = ", ".join(f"{(labels or {}).get(key, key)} {p:.2f}" for key, p in shown)
    if len(ranked) > len(shown):
        text += ", ..."
    return text


def _nearest_level(answer: ScoreAnswer) -> str:
    """Label of the rubric level closest to a (possibly fractional) score."""
    if not answer.legend:
        return ""
    highest = max(int(level) for level in answer.legend)
    nearest = min(max(int(answer.score + 0.5), 0), highest)
    return answer.legend.get(str(nearest), "")


def _answer_row(question_id: str, answer: Any) -> list[str]:
    """One table row: id, type, answer, confidence, distribution."""
    if isinstance(answer, NoulAnswer):
        verdict = "yes" if answer.noul >= 0.5 else "no"
        return [question_id, "noul", f"{verdict} ({answer.noul:.2f})", "-", "-"]
    if isinstance(answer, ChoiceAnswer):
        return [
            question_id,
            "choice",
            answer.choice,
            f"{answer.confidence:.2f}",
            _top_probabilities(answer.probabilities),
        ]
    if isinstance(answer, ScoreAnswer):
        label = _nearest_level(answer)
        return [
            question_id,
            "score",
            f"{answer.score:.2f} ({label})" if label else f"{answer.score:.2f}",
            f"{answer.confidence:.2f}",
            _top_probabilities(answer.probabilities, answer.legend),
        ]
    answer_type = str(getattr(answer, "type", "unknown"))
    return [
        question_id,
        answer_type,
        "(answer type not supported by this CLI; use --json)",
        "-",
        "-",
    ]


def render_response(response: DecisionResponse, question_order: list[str]) -> None:
    """Print the answers as a table, in the order the questions were asked."""
    from rich.markup import escape

    from venice_ai.cli.utils.console import is_plain_mode
    from venice_ai.cli.utils.output import OutputManager

    ordered = [qid for qid in question_order if qid in response.answers]
    ordered += [qid for qid in response.answers if qid not in ordered]

    rows = [_answer_row(qid, response.answers[qid]) for qid in ordered]
    if not is_plain_mode():
        rows = [[escape(cell) for cell in row] for row in rows]

    OutputManager.table(
        headers=["Question", "Type", "Answer", "Confidence", "Top probabilities"],
        rows=rows,
        title="Decisions",
        col_styles=["cyan", "dim", "bold white", "green", "dim"],
    )

    missing = [qid for qid in question_order if qid not in response.answers]
    if missing:
        OutputManager.warning(f"No answer returned for: {', '.join(missing)}")

    usage = response.usage
    summary = (
        f"Model: {response.model}  Tokens: {usage.input_tokens} in / {usage.output_tokens} out"
    )
    if is_plain_mode():
        click.echo(summary)
    else:
        console.print(f"[dim]{escape(summary)}[/dim]")
