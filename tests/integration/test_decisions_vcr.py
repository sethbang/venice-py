"""
VCRpy-based integration tests for the Decisions resource.

Records one real ``POST /decisions`` round trip carrying a question of every
answer type (``noul``, ``choice``, ``score``) and checks that the live response
parses into the typed models with every answer in range.

The assertions are structural rather than semantic: a decision model's
judgment can drift between recordings, but the shape of each answer cannot.
"""

import math
import os

import pytest
import pytest_asyncio

from venice_ai import create_test_venice_client
from venice_ai.core.config import SchedulerMode
from venice_ai.types.api import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)
from venice_ai.types.api.decisions import DecisionResponse, DecisionUsage


@pytest_asyncio.fixture
async def venice_client(backend_instance):
    """Create a Venice client for VCR testing."""
    api_key = os.getenv("VENICE_API_KEY")
    if not api_key:
        pytest.skip("VENICE_API_KEY environment variable required for integration tests")

    client = create_test_venice_client(
        api_key=api_key,
        scheduler_mode=SchedulerMode.INTELLIGENT,
        enable_redis=False,
    )
    try:
        yield client
    finally:
        await client.close()


TEAMS = {
    "billing": "Payments, payouts, invoices and refunds",
    "technical": "Bugs, outages and API integrations",
    "account": "Login, profile and account settings",
}
MOOD_LEVELS = ["Calm", "Frustrated", "Very angry"]


@pytest.mark.integration
async def test_decisions_create_all_question_types(venice_client, vcr_cassette):
    """One request answers a noul, a choice and a score question, all typed."""
    with vcr_cassette:
        model = await venice_client.models.resolve_decision()
        response = await venice_client.decisions.create(
            model=model,
            state=(
                "This is the third time I'm writing. My payouts have been failing "
                "for three days and nobody has replied. Fix it today."
            ),
            questions={
                "is_urgent": NoulQuestion(instructions="Does this message convey urgency?"),
                "team": ChoiceQuestion(
                    instructions="Which team should handle this ticket?",
                    criteria=TEAMS,
                ),
                "mood": ScoreQuestion(
                    instructions="How frustrated is the customer?",
                    criteria=MOOD_LEVELS,
                ),
            },
        )

    assert isinstance(response, DecisionResponse)
    assert response.model
    assert set(response.answers) == {"is_urgent", "team", "mood"}

    assert isinstance(response.usage, DecisionUsage)
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens >= 0

    # noul: a probability in [0, 1]
    assert isinstance(response.answers["is_urgent"], NoulAnswer)
    urgency = response.noul("is_urgent")
    assert isinstance(urgency, float)
    assert 0.0 <= urgency <= 1.0

    # choice: one of the offered options, with a full distribution over them
    team = response.choice("team")
    assert isinstance(team, ChoiceAnswer)
    assert team.choice in TEAMS
    assert set(team.probabilities) == set(TEAMS)
    assert math.isclose(sum(team.probabilities.values()), 1.0, abs_tol=1e-3)
    assert all(0.0 <= p <= 1.0 for p in team.probabilities.values())
    assert 0.0 <= team.confidence <= 1.0

    # score: a position on the rubric, levels indexed from "0"
    mood = response.score("mood")
    assert isinstance(mood, ScoreAnswer)
    assert 0.0 <= mood.score <= len(MOOD_LEVELS) - 1
    assert mood.legend == {str(i): level for i, level in enumerate(MOOD_LEVELS)}
    assert set(mood.probabilities) == set(mood.legend)
    assert math.isclose(sum(mood.probabilities.values()), 1.0, abs_tol=1e-3)
    assert 0.0 <= mood.confidence <= 1.0

    # Accessors fail loudly on a missing id or a mismatched type.
    with pytest.raises(KeyError, match="not_asked"):
        response.noul("not_asked")
    with pytest.raises(TypeError, match="'choice'"):
        response.noul("team")
    with pytest.raises(TypeError, match="'noul'"):
        response.score("is_urgent")
