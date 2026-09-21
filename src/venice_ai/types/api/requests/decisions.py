"""
Decision ("System One") request models for the Venice.ai API.

Decision models answer *typed questions* about a ``state`` instead of
generating text. A request carries one ``state`` and a map of questions keyed
by ids you choose; every question is evaluated in parallel and in isolation
against that same state, and the answers come back under the same ids.

Three question types exist, each with its own ``criteria`` shape:

``noul``
    A yes/no judgment. ``criteria`` is optional and, when given, labels what
    the two poles mean.
``choice``
    Pick one option from a fixed set. ``criteria`` maps each option name to a
    rubric description (``None`` for an option that needs no elaboration).
``score``
    A position on an ordered rubric. ``criteria`` is the ordered list of level
    descriptions, lowest first, and needs at least two entries.

Every question type rejects unknown keys (``additionalProperties: false`` in
the API spec), so a misspelled field fails locally rather than being silently
ignored by the server.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...identifiers import ModelId

# ============================================================================
# Shared aliases
# ============================================================================

DecisionState = str | dict[str, Any] | list[Any]
"""The content a decision model evaluates.

A plain string for text, or structured data (mapping/sequence) for things like
chat logs, records, or application state.
"""

DecisionInstructions = str | dict[str, Any] | list[Any]
"""A question's instructions — free text, or structured data."""


# ============================================================================
# Question models
# ============================================================================


class NoulCriteria(BaseModel):
    """What each pole of a :class:`NoulQuestion` means.

    ``true`` and ``false`` mirror the wire field names; both are ordinary
    identifiers in Python (only ``True``/``False`` are keywords).
    """

    model_config = ConfigDict(extra="forbid")

    true: str | None = Field(default=None, description="What a yes (value near 1) means.")
    false: str | None = Field(default=None, description="What a no (value near 0) means.")


class NoulQuestion(BaseModel):
    """A yes/no judgment, answered as a calibrated probability.

    The answer is a float in ``[0, 1]`` rather than a bare boolean — see
    :class:`venice_ai.types.api.decisions.NoulAnswer`. Threshold it yourself,
    or route low-confidence values to a human.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["noul"] = "noul"
    instructions: DecisionInstructions = Field(
        ..., description="The yes/no question to evaluate against the state."
    )
    criteria: NoulCriteria | None = Field(
        default=None, description="Optional labels for what a yes and a no each mean."
    )


class ChoiceQuestion(BaseModel):
    """Pick one option from a fixed set.

    ``criteria`` doubles as the option set *and* the rubric: its keys are the
    options the model may return, its values describe them. Use ``None`` as a
    value for an option that needs no extra detail.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice"] = "choice"
    instructions: DecisionInstructions = Field(..., description="What the model should decide.")
    criteria: dict[str, str | None] = Field(
        ...,
        description=(
            "Map of option name to rubric description; use ``None`` when an "
            "option needs no extra detail."
        ),
    )

    @field_validator("criteria")
    @classmethod
    def _require_an_option(cls, v: dict[str, str | None]) -> dict[str, str | None]:
        """A choice with no options has no answer the model could return."""
        if not v:
            raise ValueError("criteria must define at least one option for a 'choice' question.")
        return v


class ScoreQuestion(BaseModel):
    """Rate the state against an ordered rubric.

    ``criteria`` is ordered lowest to highest and needs at least two levels.
    The answer is a probability-weighted position that can land *between*
    levels (e.g. ``1.22`` on a three-level rubric), so treat it as continuous
    rather than as an index.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["score"] = "score"
    instructions: DecisionInstructions = Field(..., description="What the model should rate.")
    criteria: list[str] = Field(
        ...,
        min_length=2,
        description=(
            "Ordered array of level descriptions, lowest to highest. At least "
            "two levels are required."
        ),
    )


DecisionQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion,
    Field(discriminator="type"),
]
"""A single typed question in a :class:`CreateDecisionRequest`."""


# ============================================================================
# Request model
# ============================================================================


class CreateDecisionRequest(BaseModel):
    """Evaluate a ``state`` against a map of typed questions."""

    model_config = ConfigDict(extra="forbid")

    state: DecisionState = Field(
        ...,
        description=(
            "The content to evaluate: a plain string for text, or structured "
            "data (mapping/sequence) for things like chat logs, records, or "
            "application state."
        ),
    )
    model: ModelId = Field(..., description="ID of the decision model to use.")
    questions: dict[str, DecisionQuestion] = Field(
        ...,
        description=(
            "Map of typed questions, keyed by an id you choose. At least one "
            "question is required. Answers come back under the same ids. Every "
            "question is evaluated in parallel and in isolation against the "
            "same state. Question ids are not sent to the model."
        ),
    )

    @field_validator("state")
    @classmethod
    def _reject_empty_state(cls, v: DecisionState) -> DecisionState:
        """The API requires a non-empty state (``minLength``/``minItems`` 1)."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("state cannot be empty.")
        if isinstance(v, (dict, list)) and not v:
            raise ValueError("state cannot be empty.")
        return v

    @field_validator("questions")
    @classmethod
    def _require_a_question(cls, v: dict[str, Any]) -> dict[str, Any]:
        """``minProperties: 1`` — a decision with no questions has no answer."""
        if not v:
            raise ValueError("questions must contain at least one question.")
        return v


__all__ = [
    "ChoiceQuestion",
    "CreateDecisionRequest",
    "DecisionInstructions",
    "DecisionQuestion",
    "DecisionState",
    "NoulCriteria",
    "NoulQuestion",
    "ScoreQuestion",
]
