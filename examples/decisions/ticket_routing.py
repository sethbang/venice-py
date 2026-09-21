#!/usr/bin/env python3
"""
Venice AI SDK - Ticket Routing with Decision Models
===================================================

Decision ("System One") models answer *typed questions* about a piece of state.
They cannot generate text at all — there is no ``messages``, no streaming, no
JSON blob to parse out of a ``content`` string. You hand them a ``state`` and a
map of questions; you get back one structured, calibrated answer per question
in a single round trip.

This routes an inbound support ticket using all three question types:

* :class:`NoulQuestion`   → a probability in ``[0, 1]`` ("is this urgent?")
* :class:`ChoiceQuestion` → one of a fixed option set ("which team?")
* :class:`ScoreQuestion`  → a position on an ordered rubric ("how frustrated?")

The point is the **calibration**. Every answer carries a confidence, so the
interesting line in this file is not the routing decision — it's the threshold
that sends a low-confidence ticket to a human instead.

**Beta.** Venice documents ``POST /decisions`` as unstable: request and
response schemas may change without notice.
"""

import asyncio
import sys

from venice_ai import VeniceClient
from venice_ai.exceptions import APIError, VeniceError
from venice_ai.types.api import ChoiceQuestion, NoulQuestion, ScoreQuestion

# A ticket that should route to billing, with audible frustration.
TICKET = (
    "This is the THIRD time I'm writing. My payouts have been failing for "
    "three days straight and I've had no reply. I have contractors waiting to "
    "be paid. If this isn't fixed today I'm closing my account."
)

#: Route only when the model is this sure; otherwise a human triages it.
CONFIDENCE_FLOOR = 0.8


async def route_ticket() -> bool:
    """Ask one decision model three questions about a support ticket.

    Returns ``True`` on success, and also on a clean skip when no decision
    model is available — that is an entitlement condition, not a failure.
    """
    print("🎫 Ticket routing")
    print("-" * 40)
    print(f"   {TICKET[:72]}…\n")

    async with VeniceClient() as client:
        # Never hardcode a model id — resolve from the live catalog.
        # Note: decision models are all beta-flagged today, so unlike
        # resolve_chat() this helper does not filter beta models out.
        try:
            model = await client.models.resolve_decision()
        except ValueError as e:
            # Decision models are beta and Pro-gated, so plenty of accounts
            # cannot see one. That's an entitlement condition, not a code
            # error — report it and skip.
            print("   ⏭️  No decision model is available on this account.")
            print(f"      ({e})")
            return True

        print(f"   Model: {model}")

        response = await client.decisions.create(
            model=model,
            state=TICKET,
            questions={
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
            },
        )

    # Accessors narrow the answer union by id and raise TypeError if a question
    # came back as a different type than you asked for.
    urgency = response.noul("is_urgent")
    team = response.choice("team")
    mood = response.score("frustration")

    # `noul` is a probability, not a bool — `if urgency:` is true for 0.02.
    print(f"\n   Urgent:      {urgency:.0%}")
    print(f"   Team:        {team.choice}  (confidence {team.confidence:.0%})")
    # A four-level rubric can return 2.4; it is a probability-weighted
    # position, so display via .legend rather than indexing with int(score).
    print(f"   Frustration: {mood.score:.2f}  {mood.legend}")

    print("\n   → ", end="")
    if team.confidence >= CONFIDENCE_FLOOR:
        priority = "P1" if urgency >= 0.7 else "P3"
        print(f"Routing to {team.choice} as {priority}.")
    else:
        print(
            f"Confidence {team.confidence:.0%} is below the {CONFIDENCE_FLOOR:.0%} "
            "floor — escalating to a human triager."
        )

    usage = response.usage
    # Decision models price input only; output tokens bill at zero on the
    # current catalog, so a non-zero output count is expected.
    print(f"\n   Tokens: {usage.input_tokens} in / {usage.output_tokens} out")
    return True


async def main() -> int:
    """Run the ticket-routing example.

    Returns ``0`` only if the demo succeeded, ``1`` otherwise, so a real API
    failure surfaces as a non-zero exit instead of being masked by the
    success banner.
    """
    print("🚀 Venice AI Decision Models")
    print("=" * 50)

    try:
        ok = await route_ticket()
    except (VeniceError, APIError) as e:
        print(f"\n❌ API error: {e}", file=sys.stderr)
        ok = False

    if not ok:
        print("\n⚠️ Ticket routing failed.")
        return 1

    print("\n✨ Ticket routing example completed!")
    print("\n💡 Key concepts demonstrated:")
    print("   - client.models.resolve_decision() instead of a hardcoded id")
    print("   - NoulQuestion / ChoiceQuestion / ScoreQuestion in one request")
    print("   - .noul() / .choice() / .score() accessors to narrow the union")
    print("   - Gating on .confidence rather than trusting the top answer")
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
