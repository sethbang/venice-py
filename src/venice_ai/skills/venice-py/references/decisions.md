# `client.decisions` — decision ("System One") models

**Beta.** Venice documents `POST /decisions` as unstable: request and response schemas may change without notice.

Decision models answer *typed questions* about a piece of state. They cannot generate text at all — there is no `messages`, no streaming, no `content` string to parse. You hand them a `state` and a map of questions; you get back one structured, calibrated answer per question in a single round trip.

Reach for this instead of `client.chat.completions` when the task is a bounded-answer-set judgment — ticket routing, moderation, triage, batch scoring — and you were going to ask a chat model for JSON and parse it. Reach for chat when you need prose, open-ended reasoning, or an answer set you can't enumerate up front.

## The shape

```python
from venice_ai import VeniceClient
from venice_ai.types.api import ChoiceQuestion, NoulQuestion, ScoreQuestion

async with VeniceClient() as client:
    model = await client.models.resolve_decision()   # never hardcode the ID
    response = await client.decisions.create(
        model=model,
        state="Help! My payouts have been failing for 3 days.",
        questions={
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
        },
    )
```

Question ids (`is_urgent`, `team`, …) are yours to choose; answers come back under the same ids, and the ids are not sent to the model. Every question is evaluated **in parallel and in isolation** against the same state — questions cannot see each other's answers, so don't write one that depends on another.

## The three question types

| Type | `criteria` | Answer |
|---|---|---|
| `NoulQuestion` | Optional `{"true": ..., "false": ...}` labels | `NoulAnswer.noul` — a float in `[0, 1]` |
| `ChoiceQuestion` | **Required** `{option: description \| None}` | `ChoiceAnswer` — `.choice`, `.probabilities`, `.confidence` |
| `ScoreQuestion` | **Required** ordered `[str, ...]`, ≥ 2 levels | `ScoreAnswer` — `.score`, `.legend`, `.probabilities`, `.confidence` |

`criteria` on a `ChoiceQuestion` doubles as the option set: its keys are the only values `.choice` can return.

## Reading answers

`response.answers` is a union, so `response.answers["team"].choice` does not type-check. Use the accessors, which narrow by id and raise `TypeError` if a question came back as a different type than you expected:

```python
urgency = response.noul("is_urgent")        # float
team    = response.choice("team")           # ChoiceAnswer
mood    = response.score("frustration")     # ScoreAnswer
```

The whole point is the calibration — gate on it rather than taking the top answer blindly:

```python
if team.confidence > 0.8:
    route_to(team.choice)
else:
    escalate_to_human()
```

## Gotchas that break working code

- **`noul` is a probability, not a bool.** `if response.noul("is_urgent"):` is true for `0.02`. Always compare against a threshold.
- **`score` can land between levels.** A three-level rubric can return `1.22`. It is a probability-weighted position, so `legend[str(int(score))]` silently mislabels — compare against thresholds instead, and use `.legend` for display.
- **`legend` keys are zero-indexed strings** (`{"0": "Calm", "1": "Frustrated"}`), matching the order of the `criteria` list you sent.
- **`resolve_decision()` does not filter beta models**, unlike `resolve_chat()`. Decision models are all beta-flagged today, so `exclude_beta=True` would return nothing.
- **Decision models have no `availableContextTokens`.** They report `maxStateTokens` (state + the single longest question) and `maxTotalTokens` (state + all questions) on `DecisionModelSpec`. A request can fit the per-question budget and still blow the total.
- **An unknown answer type does not raise.** It lands on `UnknownAnswer` with its payload on `.model_extra`, mirroring the catch-all arm in Venice's own schema. Match on the concrete class, not on `type`, and handle the fallback if you are branching exhaustively.
- **`POST /systemone` is the same endpoint**, exposed under TypeSafe's own path so their SDKs work against `TYPESAFE_BASE_URL=https://api.venice.ai/api`. There is no separate SDK method — `decisions.create()` is it.

## Structured state

`state` takes a string *or* structured data, which is usually the better input — you skip serialising a record into prose the model then has to re-extract:

```python
response = await client.decisions.create(
    model=model,
    state={"plan": "free", "failed_payments": 3, "open_tickets": 2},
    questions={"at_risk": NoulQuestion(instructions="Is this account at risk of churning?")},
)
```

A chat transcript (a list of message dicts) is a valid `state` too.

## Cost

Decision models price input only — output tokens bill at zero on the current catalog. `response.usage` reports both, so `output_tokens` being non-zero is expected and not a billing signal.
