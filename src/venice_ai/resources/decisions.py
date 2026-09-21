"""
Venice AI Decisions API resource ("System One" models).

Decision models answer *typed questions* about a piece of state instead of
generating prose. Where a chat model returns free text you then have to parse,
a decision model returns a structured, calibrated answer per question — a
probability, a category with its full distribution, or a position on a rubric —
in a single round trip.

That makes this the right resource for the high-volume, bounded-answer-set work
that chat models do awkwardly: ticket routing, content moderation, batch
scoring, triage. It is *not* a replacement for chat — a decision model cannot
generate text at all, and it can only answer questions you have already framed.

Key Features:
    - **Typed answers**: every answer is a ``noul``, ``choice`` or ``score``
      shape, validated on arrival — never a string you have to parse
    - **Calibrated confidence**: answers carry probabilities, so downstream
      code can auto-act on confident ones and escalate the rest
    - **Parallel evaluation**: every question is evaluated independently
      against the same state in one request
    - **Structured state**: evaluate a string, or a mapping/sequence such as a
      chat log, a record, or application state

Common Use Cases:
    - **Ticket routing**: one ``choice`` question over your team names
    - **Content moderation**: ``noul`` questions per policy, auto-acting above
      a confidence threshold
    - **Batch scoring**: a ``score`` question over an ordered rubric
    - **Triage and escalation**: combine an urgency ``noul`` with a severity
      ``score`` in a single call

Example:
    .. code-block:: python

        import asyncio
        from venice_ai import VeniceClient
        from venice_ai.types.api import ChoiceQuestion, NoulQuestion

        async def route_ticket():
            async with VeniceClient() as client:
                # Resolve a decision model ID at runtime (do not hardcode)
                model = await client.models.resolve_decision()
                response = await client.decisions.create(
                    model=model,
                    state="Help! My payouts have been failing for 3 days.",
                    questions={
                        "is_urgent": NoulQuestion(
                            instructions="Does this message convey urgency?"
                        ),
                        "team": ChoiceQuestion(
                            instructions="Which team should handle this?",
                            criteria={
                                "billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                            },
                        ),
                    },
                )

                team = response.choice("team")
                if team.confidence > 0.8:
                    print(f"Auto-routing to {team.choice}")
                else:
                    print("Low confidence — sending to a human")

        asyncio.run(route_ticket())

Note:
    This endpoint is **beta**; Venice documents its request and response
    schemas as subject to change without notice. It is also served at
    ``/systemone`` as a drop-in for TypeSafe's own SDKs (point them at
    ``TYPESAFE_BASE_URL=https://api.venice.ai/api``) — the two paths are the
    same endpoint, so this resource targets ``/decisions`` only.
"""

from typing import TYPE_CHECKING, Any

from .._resource import APIResource
from ..exceptions import InvalidRequestError
from ..types.api.decisions import DecisionResponse
from ..types.api.requests.decisions import CreateDecisionRequest, DecisionState

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401


class Decisions(APIResource["VeniceClient"]):
    """
    Provides access to decision ("System One") model operations (asynchronous).

    Accessed through :attr:`VeniceClient.decisions`.

    :param client: The Venice AI client instance used to make API requests.
    :type client: venice_ai._client.VeniceClient
    """

    async def create(
        self,
        *,
        model: str,
        state: DecisionState,
        questions: dict[str, Any],
    ) -> DecisionResponse:
        """
        Evaluate ``state`` against a map of typed questions.

        Every question is evaluated in parallel and in isolation against the
        same state, and the answers come back keyed by the same ids you used
        in the request. Question ids are not sent to the model, so they can be
        whatever is convenient for your code.

        :param model: The ID of the decision model to use. Resolve a valid ID
            at runtime via ``client.models.resolve_decision()`` rather than
            hardcoding one, since available models change over time.
        :type model: str
        :param state: The content to evaluate — a plain string for text, or
            structured data (mapping/sequence) for things like chat logs,
            records, or application state.
        :type state: Union[str, Dict[str, Any], List[Any]]
        :param questions: Map of question id to question. Values may be
            :class:`~venice_ai.types.api.requests.decisions.NoulQuestion`,
            :class:`~venice_ai.types.api.requests.decisions.ChoiceQuestion` or
            :class:`~venice_ai.types.api.requests.decisions.ScoreQuestion`
            instances, or equivalent dicts carrying a ``type`` key. At least
            one question is required.
        :type questions: Dict[str, Any]

        :return: The model's answers, one per question, plus token usage.
        :rtype: ~venice_ai.types.api.decisions.DecisionResponse

        :raises venice_ai.exceptions.InvalidRequestError: If parameter values
            are invalid (e.g. empty model, empty state, no questions, or a
            question whose shape does not match its declared type).
        :raises venice_ai.exceptions.AuthenticationError: If the API key is
            invalid or missing.
        :raises venice_ai.exceptions.PermissionDeniedError: If access to the
            specified model is denied.
        :raises venice_ai.exceptions.NotFoundError: If the specified model is
            not found.
        :raises venice_ai.exceptions.RateLimitError: If rate limits are
            exceeded.
        :raises venice_ai.exceptions.APIError: For other API-related errors.

        **Examples:**

        Score a review against an ordered rubric:

        .. code-block:: python

            from venice_ai.types.api import ScoreQuestion

            model = await client.models.resolve_decision()
            response = await client.decisions.create(
                model=model,
                state="Shipping took three weeks and nobody replied to my emails.",
                questions={
                    "frustration": ScoreQuestion(
                        instructions="How frustrated is the customer?",
                        criteria=["Calm", "Frustrated", "Very angry"],
                    ),
                },
            )
            answer = response.score("frustration")
            print(answer.score, answer.legend)

        Evaluate structured state rather than a string:

        .. code-block:: python

            from venice_ai.types.api import NoulQuestion

            response = await client.decisions.create(
                model=model,
                state={
                    "plan": "free",
                    "failed_payments": 3,
                    "open_tickets": 2,
                },
                questions={
                    "at_risk": NoulQuestion(
                        instructions="Is this account at risk of churning?"
                    ),
                },
            )
            if response.noul("at_risk") > 0.9:
                print("Flagging for retention outreach")
        """
        if not model:
            raise InvalidRequestError(
                "model parameter is required and cannot be empty.",
                request=None,
                response=None,
                body=None,
            )

        try:
            decision_request = CreateDecisionRequest(
                model=model,
                state=state,
                questions=questions,
            )
        except ValueError as exc:
            # Surface local schema violations as the SDK's own request error
            # rather than a bare pydantic ValidationError, so callers catch
            # them the same way they catch every other bad-request case.
            raise InvalidRequestError(
                f"Invalid decision request: {exc}",
                request=None,
                response=None,
                body=None,
            ) from exc

        body = decision_request.model_dump(exclude_none=True)

        return await self._client.post("decisions", json_data=body, cast_to=DecisionResponse)
