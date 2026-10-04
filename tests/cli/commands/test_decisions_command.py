"""
Tests for cli/commands/decisions.py

Covers:
- Argument parsing: --noul / --choice / --score flags, --questions file and
  stdin, positional / piped / JSON state
- Request building: the exact state, questions and model handed to
  ``client.decisions.create``, and model resolution through the resolver
- Output: the table (rich and plain) and --json
- Error paths: no questions, duplicate ids, malformed lists and files, missing
  or invalid state, and API errors exiting 1
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from venice_ai.cli.cli import cli
from venice_ai.exceptions import AuthenticationError, NotFoundError
from venice_ai.types.api import ChoiceQuestion, NoulQuestion, ScoreQuestion
from venice_ai.types.api.decisions import DecisionResponse

RESOLVED_MODEL = "resolved-decision-model"

RESPONSE_PAYLOAD = {
    "model": RESOLVED_MODEL,
    "answers": {
        "urgent": {"type": "noul", "noul": 0.93},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.86, "technical": 0.1, "account": 0.04},
            "confidence": 0.72,
        },
        "mood": {
            "type": "score",
            "score": 1.62,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.05, "1": 0.28, "2": 0.67},
            "confidence": 0.41,
        },
    },
    "usage": {"input_tokens": 57, "output_tokens": 3},
}

ALL_FLAGS = [
    "--noul",
    "urgent",
    "Is this urgent?",
    "--choice",
    "team",
    "Which team?",
    "billing, technical,account",
    "--score",
    "mood",
    "How frustrated?",
    "Calm,Frustrated,Very angry",
]


@pytest.fixture
def mock_client():
    """Patch ``VeniceClient`` and return the client the command will receive."""
    client = MagicMock()
    client.models.resolve = AsyncMock(return_value=RESOLVED_MODEL)
    client.decisions.create = AsyncMock(
        return_value=DecisionResponse.model_validate(RESPONSE_PAYLOAD)
    )
    with (
        patch("venice_ai.VeniceClient") as MockClient,
        patch("venice_ai.cli.config.ensure_api_key", return_value="test-key"),
    ):
        MockClient.return_value.__aenter__ = AsyncMock(return_value=client)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=None)
        yield client


def _invoke(args, *, input=None, plain=False):
    root = ["--plain"] if plain else []
    return CliRunner().invoke(cli, [*root, "decisions", *args], input=input)


def _sent(mock_client):
    """The keyword arguments the command passed to ``decisions.create``."""
    mock_client.decisions.create.assert_awaited_once()
    return mock_client.decisions.create.await_args.kwargs


# ---------------------------------------------------------------------------
# Parsing and request building
# ---------------------------------------------------------------------------


class TestRequestBuilding:
    def test_help(self):
        result = CliRunner().invoke(cli, ["decisions", "--help"])
        assert result.exit_code == 0
        for flag in ("--noul", "--choice", "--score", "--questions", "--state-json", "--json"):
            assert flag in result.output

    def test_inline_flags_build_typed_questions(self, mock_client):
        result = _invoke(["Payouts failing for days", *ALL_FLAGS])

        assert result.exit_code == 0, result.output
        sent = _sent(mock_client)
        assert sent["model"] == RESOLVED_MODEL
        assert sent["state"] == "Payouts failing for days"
        assert list(sent["questions"]) == ["urgent", "team", "mood"]
        assert sent["questions"]["urgent"] == NoulQuestion(instructions="Is this urgent?")
        assert sent["questions"]["team"] == ChoiceQuestion(
            instructions="Which team?",
            criteria={"billing": None, "technical": None, "account": None},
        )
        assert sent["questions"]["mood"] == ScoreQuestion(
            instructions="How frustrated?", criteria=["Calm", "Frustrated", "Very angry"]
        )

    def test_model_defaults_to_decision_resolver(self, mock_client):
        _invoke(["state", "--noul", "q", "Yes?"])
        mock_client.models.resolve.assert_awaited_once_with(type="decision")

    def test_explicit_model_skips_resolver(self, mock_client):
        result = _invoke(["state", "--noul", "q", "Yes?", "--model", "picked-model"])

        assert result.exit_code == 0, result.output
        mock_client.models.resolve.assert_not_called()
        assert _sent(mock_client)["model"] == "picked-model"

    def test_questions_file_merges_with_flags(self, mock_client, tmp_path):
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps(
                {
                    "team": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"billing": "Payments and refunds", "technical": None},
                    },
                    "safe": {
                        "type": "noul",
                        "instructions": "Is this safe?",
                        "criteria": {"true": "Safe", "false": "Unsafe"},
                    },
                }
            )
        )

        result = _invoke(["state", "--noul", "urgent", "Urgent?", "--questions", str(path)])

        assert result.exit_code == 0, result.output
        questions = _sent(mock_client)["questions"]
        assert list(questions) == ["urgent", "team", "safe"]
        assert questions["team"].criteria == {"billing": "Payments and refunds", "technical": None}
        assert questions["safe"].criteria.true == "Safe"

    def test_questions_from_stdin(self, mock_client):
        payload = json.dumps({"q": {"type": "noul", "instructions": "Yes?"}})

        result = _invoke(["the state", "--questions", "-"], input=payload)

        assert result.exit_code == 0, result.output
        assert list(_sent(mock_client)["questions"]) == ["q"]

    def test_state_from_stdin(self, mock_client):
        result = _invoke(["--noul", "q", "Yes?"], input="  piped ticket text \n")

        assert result.exit_code == 0, result.output
        assert _sent(mock_client)["state"] == "piped ticket text"

    def test_state_json_sends_structured_state(self, mock_client):
        result = _invoke(["--state-json", '{"plan": "free", "failed": 3}', "--noul", "q", "Risk?"])

        assert result.exit_code == 0, result.output
        assert _sent(mock_client)["state"] == {"plan": "free", "failed": 3}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


class TestOutput:
    def test_table_output(self, mock_client):
        result = _invoke(["state", *ALL_FLAGS])

        assert result.exit_code == 0, result.output
        out = result.output
        assert "Decisions" in out
        assert "yes (0.93)" in out
        assert "0.72" in out
        assert "Total: 3" in out
        assert f"Model: {RESOLVED_MODEL}" in out
        assert "57 in / 3 out" in out

    def test_plain_output(self, mock_client):
        result = _invoke(["state", *ALL_FLAGS], plain=True)

        assert result.exit_code == 0, result.output
        assert "\x1b[" not in result.output
        lines = result.output.splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("Question"))
        urgent, team, mood = lines[header + 2 : header + 5]
        assert urgent.startswith("urgent") and "yes (0.93)" in urgent
        assert team.startswith("team") and "0.72" in team
        assert "billing 0.86, technical 0.10, account 0.04" in team
        # 1.62 is nearest level 2; score probabilities are labelled from the legend
        assert mood.startswith("mood") and "1.62 (Very angry)" in mood
        assert "Very angry 0.67, Frustrated 0.28, Calm 0.05" in mood
        assert f"Model: {RESOLVED_MODEL}  Tokens: 57 in / 3 out" in result.output

    def test_json_output_is_pure_json(self, mock_client):
        result = _invoke(["state", *ALL_FLAGS, "--json"])

        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == RESPONSE_PAYLOAD

    def test_unknown_answer_type_renders(self, mock_client):
        mock_client.decisions.create.return_value = DecisionResponse.model_validate(
            {
                "model": RESOLVED_MODEL,
                "answers": {"q": {"type": "ranking", "order": ["a", "b"]}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )

        result = _invoke(["state", "--noul", "q", "Yes?"], plain=True)

        assert result.exit_code == 0, result.output
        assert "ranking" in result.output
        assert "use --json" in result.output

    def test_missing_answer_is_reported(self, mock_client):
        mock_client.decisions.create.return_value = DecisionResponse.model_validate(
            {
                "model": RESOLVED_MODEL,
                "answers": {"urgent": {"type": "noul", "noul": 0.1}},
                "usage": {"input_tokens": 1, "output_tokens": 0},
            }
        )

        result = _invoke(["state", "--noul", "urgent", "U?", "--noul", "other", "O?"], plain=True)

        assert result.exit_code == 0, result.output
        assert "no (0.10)" in result.output
        assert "No answer returned for: other" in result.output


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["state"], "At least one question is required"),
            (["state", "--noul", "q", "A?", "--noul", "q", "B?"], "defined more than once"),
            (["state", "--choice", "q", "Pick?", " , "], "at least one option"),
            (["state", "--choice", "q", "Pick?", "a,b,a"], "repeats 'a'"),
            (["state", "--score", "q", "Rate?", "only"], "at least two levels"),
            (["state", "--noul", " ", "Yes?"], "Question ids cannot be empty"),
            (["--state-json", "not json", "--noul", "q", "Yes?"], "invalid JSON"),
            (["--state-json", '"text"', "--noul", "q", "Yes?"], "non-empty JSON object or array"),
            (["--state-json", "{}", "--noul", "q", "Yes?"], "non-empty JSON object or array"),
            (["--questions", "-"], "STATE must be given as an argument"),
        ],
    )
    def test_usage_errors_exit_2_without_calling_api(self, mock_client, args, message):
        result = _invoke(args, input="{}")

        assert result.exit_code == 2
        assert message in result.output
        mock_client.decisions.create.assert_not_called()

    @pytest.mark.parametrize(
        ("contents", "message"),
        [
            ("{not json", "invalid JSON"),
            ("[1, 2]", "expected a JSON object"),
            ('{"q": {"type": "maybe", "instructions": "x"}}', "question 'q' is invalid"),
            ('{"q": {"type": "score", "instructions": "x", "criteria": ["one"]}}', "criteria"),
            ('{"q": {"type": "noul", "instructions": "x", "extra": 1}}', "extra"),
        ],
    )
    def test_invalid_questions_file(self, mock_client, tmp_path, contents, message):
        path = tmp_path / "q.json"
        path.write_text(contents)

        result = _invoke(["state", "--questions", str(path)])

        assert result.exit_code == 2
        assert message in result.output
        mock_client.decisions.create.assert_not_called()

    def test_unreadable_questions_file(self, mock_client, tmp_path):
        result = _invoke(["state", "--questions", str(tmp_path / "missing.json")])

        assert result.exit_code == 2
        assert "cannot read" in result.output

    def test_missing_state_exits_1(self, mock_client):
        result = _invoke(["--noul", "q", "Yes?"], input="   ")

        assert result.exit_code == 1
        assert "State is required" in result.output
        mock_client.decisions.create.assert_not_called()

    @pytest.mark.parametrize(
        "error",
        [
            AuthenticationError("Invalid API key", request=None, response=None, body=None),
            NotFoundError("Model not found", request=None, response=None, body=None),
        ],
    )
    def test_api_error_exits_1(self, mock_client, error):
        mock_client.decisions.create.side_effect = error

        result = _invoke(["state", "--noul", "q", "Yes?"], plain=True)

        assert result.exit_code == 1
        assert "Venice API error" in result.output
        assert str(error) in result.output

    def test_resolver_failure_keeps_actionable_message(self, mock_client):
        mock_client.models.resolve.side_effect = RuntimeError("network down")

        result = _invoke(["state", "--noul", "q", "Yes?"])

        assert result.exit_code == 1
        assert "Could not resolve a default decision model" in result.output
        assert "--model" in result.output
        mock_client.decisions.create.assert_not_called()
