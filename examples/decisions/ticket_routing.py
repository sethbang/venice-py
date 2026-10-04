#!/usr/bin/env python3
"""
Venice AI SDK - Ticket Routing with Decision Models
===================================================

Decision ("System One") models answer *typed questions* about a piece of state.
They cannot generate text at all — there is no ``messages``, no streaming, no
JSON blob to parse out of a ``content`` string. You hand them a ``state`` and a
map of questions; you get back one structured, calibrated answer per question
in a single round trip.

This routes inbound support tickets using all three question types:

* :class:`NoulQuestion`   → a probability in ``[0, 1]`` ("is this urgent?")
* :class:`ChoiceQuestion` → one of a fixed option set ("which team?")
* :class:`ScoreQuestion`  → a position on an ordered rubric ("how frustrated?")

The point is the **calibration**. Every answer carries a confidence, so the
interesting line in this file is not the routing decision — it's the threshold
that sends a low-confidence ticket to a human instead.

Three tickets, each with what the example checks:

1. An angry, time-critical payout complaint: it must go to ``billing`` with
   confidence above the floor, be judged urgent (so it routes as P1), and sound
   more frustrated than the calm ticket.
2. A vague ticket that mixes billing, API and login problems: no team is
   correct, so only the answer types are checked. It shows how the confidence
   moves; whether it falls below the floor and escalates is the model's call.
3. A calm, no-rush login-email change: it must go to ``account`` with
   confidence above the floor and must not be judged urgent, so ticket 1's
   urgency is measured against a real contrast.

**Beta.** Venice documents ``POST /decisions`` as unstable: request and
response schemas may change without notice.
"""

import asyncio
import math
import sys
from dataclasses import dataclass

from venice_ai import NoMatchingModelError, VeniceClient
from venice_ai.exceptions import PermissionDeniedError, VeniceError
from venice_ai.types.api import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)

#: Exit code for "skipped": no decision model is available to this account.
EXIT_SKIPPED = 77


@dataclass(frozen=True)
class Ticket:
    """A ticket and what a correct judgment of it must look like.

    ``None`` means "not checked": the ticket is ambiguous on that question.
    """

    text: str
    team: str | None = None
    urgent: bool | None = None


@dataclass(frozen=True)
class Judgment:
    urgency: float
    team: ChoiceAnswer
    mood: ScoreAnswer


TICKETS = [
    Ticket(
        "This is the THIRD time I'm writing. My payouts have been failing for "
        "three days straight and I've had no reply. I have contractors waiting to "
        "be paid. If this isn't fixed today I'm closing my account.",
        team="billing",
        urgent=True,
    ),
    Ticket(
        "I got charged twice this month, and since then the API returns errors "
        "and I keep getting logged out. Not sure which part is the real problem.",
    ),
    Ticket(
        "Hi! No rush at all, but whenever someone has a moment I'd like to change "
        "the email address I use to log in to my account. Thanks!",
        team="account",
        urgent=False,
    ),
]

QUESTIONS = {
    "is_urgent": NoulQuestion(
        instructions="Does this message require an urgent response?",
    ),
    "team": ChoiceQuestion(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, payouts, invoicing, refunds",
            "technical": "Bugs, outages, API and integration problems",
            "account": "Login, plan changes, cancellations",
        },
    ),
    "frustration": ScoreQuestion(
        instructions="How frustrated does the customer sound?",
        criteria=["Calm", "Annoyed", "Frustrated", "Furious"],
    ),
}

#: Route only when the model is this sure; otherwise a human triages it.
CONFIDENCE_FLOOR = 0.9

#: Urgency at or above this routes as P1, below it as P3.
URGENT_THRESHOLD = 0.7


def describe_score(answer: ScoreAnswer) -> str:
    """Describe a score that may land between two rubric levels."""

    def label(level: int) -> str:
        return answer.legend.get(str(level), str(level))

    nearest = round(answer.score)
    if abs(answer.score - nearest) <= 0.25:
        return label(nearest)
    return f"between {label(math.floor(answer.score))} and {label(math.ceil(answer.score))}"


async def route_ticket(client: VeniceClient, model: str, ticket: Ticket) -> Judgment:
    """Ask the decision model three questions about one ticket and route it."""
    print(f"\n🎫 {ticket.text[:72]}…")

    response = await client.decisions.create(model=model, state=ticket.text, questions=QUESTIONS)

    # Accessors narrow the answer union by id and raise TypeError if a question
    # came back as a different type than you asked for.
    urgency = response.noul("is_urgent")
    team = response.choice("team")
    mood = response.score("frustration")

    # `noul` is a probability, not a bool — `if urgency:` is true for 0.02.
    print(f"   Urgent:      {urgency:.0%}")
    print(f"   Team:        {team.choice}  (confidence {team.confidence:.0%})")
    # A four-level rubric can return 2.4: a probability-weighted position, so
    # describe it by its neighbouring levels rather than indexing with int(score).
    print(f"   Frustration: {mood.score:.2f}  ({describe_score(mood)})")

    print("   → ", end="")
    if team.confidence >= CONFIDENCE_FLOOR:
        priority = "P1" if urgency >= URGENT_THRESHOLD else "P3"
        print(f"Routing to {team.choice} as {priority}.")
    else:
        print(
            f"Confidence {team.confidence:.0%} is below the {CONFIDENCE_FLOOR:.0%} "
            "floor — escalating to a human triager."
        )

    usage = response.usage
    # Decision models price input only: the response still reports output
    # tokens, but the catalog bills them at zero.
    print(f"   Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
    return Judgment(urgency=urgency, team=team, mood=mood)


def check_judgment(ticket: Ticket, judgment: Judgment) -> list[str]:
    """Return what is wrong with one judgment (an empty list means it passed)."""
    team_options = set(QUESTIONS["team"].criteria)
    # The legend maps each rubric level ("0", "1", ...) to its label.
    levels = sorted(int(level) for level in judgment.mood.legend) or [0]
    low, high = levels[0], levels[-1]
    problems: list[str] = []

    # Every answer must be well-formed, whatever the ticket.
    if not 0.0 <= judgment.urgency <= 1.0:
        problems.append(f"urgency {judgment.urgency} is not a probability")
    if judgment.team.choice not in team_options:
        problems.append(f"team {judgment.team.choice!r} is not one of {sorted(team_options)}")
    if not 0.0 <= judgment.team.confidence <= 1.0:
        problems.append(f"confidence {judgment.team.confidence} is not a probability")
    if not low <= judgment.mood.score <= high:
        problems.append(f"frustration {judgment.mood.score} is outside the {low}..{high} rubric")

    if ticket.team is not None:
        if judgment.team.choice != ticket.team:
            problems.append(f"expected team {ticket.team!r}, got {judgment.team.choice!r}")
        elif judgment.team.confidence < CONFIDENCE_FLOOR:
            problems.append(
                f"a clear-cut ticket should clear the {CONFIDENCE_FLOOR:.0%} floor, "
                f"got {judgment.team.confidence:.0%}"
            )
    if ticket.urgent is not None and (judgment.urgency >= URGENT_THRESHOLD) != ticket.urgent:
        expected = "urgent" if ticket.urgent else "not urgent"
        problems.append(f"expected {expected}, urgency was {judgment.urgency:.0%}")
    return problems


async def main() -> int:
    """Run the ticket-routing example.

    Returns ``0`` when every ticket was answered and each check passed, ``1``
    when a check failed, and ``77`` (with a ``SKIPPED:`` line) when this
    account has no decision model.
    """
    print("🚀 Venice AI Decision Models")
    print("=" * 50)

    async with VeniceClient() as client:
        # Never hardcode a model id — resolve from the live catalog.
        # Decision models are all beta-flagged today, so unlike resolve_chat()
        # this helper does not filter beta models out.
        try:
            model = await client.models.resolve_decision(prefer="cheapest")
        except NoMatchingModelError as e:
            # Decision models are beta and plan-gated, so many accounts see none.
            print(f"SKIPPED: no decision model is available on this account ({e})")
            return EXIT_SKIPPED

        print(f"Model: {model}")

        judgments: list[Judgment | None] = []
        failures = 0
        for ticket in TICKETS:
            try:
                judgment = await route_ticket(client, model, ticket)
            except PermissionDeniedError as e:
                print(f"SKIPPED: this account may not call decision models ({e})")
                return EXIT_SKIPPED
            except VeniceError as e:
                print(f"   ❌ API error: {type(e).__name__}: {e}")
                judgments.append(None)
                failures += 1
                continue
            judgments.append(judgment)
            for problem in check_judgment(ticket, judgment):
                print(f"   ❌ {problem}")
                failures += 1

    # Frustration is relative: the furious ticket must sound angrier than the calm one.
    furious, calm = judgments[0], judgments[2]
    if furious is not None and calm is not None:
        if furious.mood.score <= calm.mood.score:
            print(
                f"\n❌ The furious ticket scored {furious.mood.score:.2f} frustration, "
                f"not above the calm ticket's {calm.mood.score:.2f}"
            )
            failures += 1
        else:
            print(
                f"\n✓ Frustration separates the tickets: {furious.mood.score:.2f} "
                f"(furious) vs {calm.mood.score:.2f} (calm)"
            )

    if failures:
        print(f"\n❌ {failures} check(s) failed.")
        return 1

    print("\n✨ Ticket routing example completed!")
    print("\n💡 Key concepts demonstrated:")
    print('   - client.models.resolve_decision(prefer="cheapest") instead of a hardcoded id')
    print("   - NoulQuestion / ChoiceQuestion / ScoreQuestion in one request")
    print("   - .noul() / .choice() / .score() accessors to narrow the union")
    print("   - Gating on .confidence rather than trusting the top answer")
    print("   - Urgency and frustration checked against a calm control ticket")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
