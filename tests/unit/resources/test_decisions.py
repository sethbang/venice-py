"""
Tests for src/venice_ai/resources/decisions.py — the Decisions ("System One")
resource.

Covers the wire body the resource builds, the typed answer union (including
its open catch-all arm), the typed accessors on the response, and the local
validation that rejects a malformed request before it costs an API call.

The payloads below are recorded from the live ``POST /decisions`` endpoint on
2026-09-18 against ``jev-latest``.
"""

from unittest.mock import AsyncMock

import pytest

from venice_ai.exceptions import InvalidRequestError
from venice_ai.resources.decisions import Decisions
from venice_ai.types.api import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    UnknownAnswer,
)

# Verbatim POST /decisions response, live API 2026-09-18.
LIVE_RESPONSE = {
    "model": "jev-latest",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.96},
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"technical": 0.06, "billing": 0.94},
            "confidence": 0.89,
        },
        "frustration": {
            "type": "score",
            "score": 1.22,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.78, "2": 0.22},
            "confidence": 0.67,
        },
    },
    "usage": {"input_tokens": 410, "output_tokens": 66},
}


class MockVeniceClient:
    """Mock client for testing the Decisions resource."""

    def __init__(self, api_key: str = "test-key"):
        self._api_key = api_key
        self.post = AsyncMock(return_value=DecisionResponse.model_validate(LIVE_RESPONSE))


@pytest.fixture
def mock_client():
    return MockVeniceClient()


@pytest.fixture
def decisions(mock_client):
    return Decisions(mock_client)


def _questions():
    return {
        "is_urgent": NoulQuestion(instructions="Does this message convey urgency?"),
        "team": ChoiceQuestion(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
            },
        ),
        "frustration": ScoreQuestion(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    }


class TestRequestBody:
    """What the resource actually puts on the wire."""

    @pytest.mark.asyncio
    async def test_posts_to_decisions_path(self, decisions, mock_client):
        await decisions.create(model="m", state="hello", questions=_questions())
        assert mock_client.post.await_args.args[0] == "decisions"

    @pytest.mark.asyncio
    async def test_body_shape(self, decisions, mock_client):
        await decisions.create(model="m", state="hello", questions=_questions())
        body = mock_client.post.await_args.kwargs["json_data"]
        assert body["model"] == "m"
        assert body["state"] == "hello"
        assert set(body["questions"]) == {"is_urgent", "team", "frustration"}
        assert body["questions"]["is_urgent"]["type"] == "noul"
        assert body["questions"]["frustration"]["criteria"] == [
            "Calm",
            "Frustrated",
            "Very angry",
        ]

    @pytest.mark.asyncio
    async def test_accepts_plain_dict_questions(self, decisions, mock_client):
        """Callers shouldn't have to import the question classes."""
        await decisions.create(
            model="m",
            state="hello",
            questions={"q": {"type": "noul", "instructions": "Urgent?"}},
        )
        body = mock_client.post.await_args.kwargs["json_data"]
        assert body["questions"]["q"]["type"] == "noul"

    @pytest.mark.asyncio
    async def test_structured_state_survives(self, decisions, mock_client):
        state = {"plan": "free", "failed_payments": 3}
        await decisions.create(
            model="m",
            state=state,
            questions={"q": {"type": "noul", "instructions": "At risk?"}},
        )
        assert mock_client.post.await_args.kwargs["json_data"]["state"] == state

    @pytest.mark.asyncio
    async def test_optional_criteria_omitted_not_nulled(self, decisions, mock_client):
        """``exclude_none`` must not leave ``criteria: null`` on the wire —
        the endpoint declares ``additionalProperties: false``."""
        await decisions.create(
            model="m",
            state="hello",
            questions={"q": NoulQuestion(instructions="Urgent?")},
        )
        assert "criteria" not in mock_client.post.await_args.kwargs["json_data"]["questions"]["q"]


class TestAnswerParsing:
    """The response union narrows to the right concrete class."""

    def test_each_answer_type(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        assert isinstance(r.answers["is_urgent"], NoulAnswer)
        assert isinstance(r.answers["team"], ChoiceAnswer)
        assert isinstance(r.answers["frustration"], ScoreAnswer)

    def test_score_can_land_between_levels(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        answer = r.score("frustration")
        assert answer.score == 1.22
        assert answer.legend["1"] == "Frustrated"

    def test_unknown_answer_type_does_not_raise(self):
        """The spec's union ends in an open catch-all arm; mirror it."""
        payload = dict(LIVE_RESPONSE)
        payload["answers"] = {"q": {"type": "rank", "ranking": ["a", "b"]}}
        r = DecisionResponse.model_validate(payload)
        answer = r.answers["q"]
        assert isinstance(answer, UnknownAnswer)
        assert answer.type == "rank"
        assert answer.model_extra == {"ranking": ["a", "b"]}

    def test_usage_parsed(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        assert r.usage.input_tokens == 410
        assert r.usage.output_tokens == 66


class TestTypedAccessors:
    """``answers`` is a union; the accessors narrow it once."""

    def test_noul_returns_float(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        assert r.noul("is_urgent") == pytest.approx(0.96)

    def test_choice_returns_full_answer(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        answer = r.choice("team")
        assert answer.choice == "billing"
        assert answer.confidence == pytest.approx(0.89)

    def test_wrong_type_raises_typeerror(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        with pytest.raises(TypeError, match="answered as 'choice', not 'noul'"):
            r.noul("team")

    def test_missing_id_raises_keyerror_listing_ids(self):
        r = DecisionResponse.model_validate(LIVE_RESPONSE)
        with pytest.raises(KeyError, match="frustration, is_urgent, team"):
            r.noul("nope")


class TestLocalValidation:
    """A malformed request must fail before it costs an API call."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {
                    "model": "",
                    "state": "x",
                    "questions": {"q": {"type": "noul", "instructions": "i"}},
                },
                id="empty-model",
            ),
            pytest.param(
                {
                    "model": "m",
                    "state": "   ",
                    "questions": {"q": {"type": "noul", "instructions": "i"}},
                },
                id="blank-state",
            ),
            pytest.param(
                {
                    "model": "m",
                    "state": {},
                    "questions": {"q": {"type": "noul", "instructions": "i"}},
                },
                id="empty-mapping-state",
            ),
            pytest.param({"model": "m", "state": "x", "questions": {}}, id="no-questions"),
            pytest.param(
                {
                    "model": "m",
                    "state": "x",
                    "questions": {"q": {"type": "bogus", "instructions": "i"}},
                },
                id="unknown-question-type",
            ),
            pytest.param(
                {
                    "model": "m",
                    "state": "x",
                    "questions": {"q": {"type": "noul", "instruction": "i"}},
                },
                id="misspelled-field",
            ),
            pytest.param(
                {
                    "model": "m",
                    "state": "x",
                    "questions": {
                        "q": {"type": "score", "instructions": "i", "criteria": ["only"]}
                    },
                },
                id="score-needs-two-levels",
            ),
            pytest.param(
                {
                    "model": "m",
                    "state": "x",
                    "questions": {"q": {"type": "choice", "instructions": "i", "criteria": {}}},
                },
                id="choice-needs-an-option",
            ),
        ],
    )
    async def test_rejects_locally(self, decisions, mock_client, kwargs):
        with pytest.raises(InvalidRequestError):
            await decisions.create(**kwargs)
        mock_client.post.assert_not_awaited()
