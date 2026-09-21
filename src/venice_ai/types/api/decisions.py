"""
Decision ("System One") response models for the Venice.ai API.

One answer comes back per question, keyed by the same ids used in the request.
Each answer is typed to match its question — :class:`NoulAnswer`,
:class:`ChoiceAnswer` or :class:`ScoreAnswer` — and every one carries a
calibrated probability rather than a bare verdict, so callers can auto-act on
confident answers and route the rest to a human.

The answer union is deliberately **open**. The API spec's own union ends in a
catch-all arm requiring nothing but ``type``, which is the API team saying more
answer types are coming; an unrecognised one lands on :class:`UnknownAnswer`
instead of failing the parse.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from ...core.models.base import VeniceBaseModel

# ============================================================================
# Answer models
# ============================================================================


class NoulAnswer(BaseModel):
    """A yes/no judgment as a calibrated probability."""

    model_config = ConfigDict(extra="allow")

    type: Literal["noul"] = "noul"
    noul: float = Field(
        ..., ge=0.0, le=1.0, description="The yes/no answer on a scale from 0 (no) to 1 (yes)."
    )


class ChoiceAnswer(BaseModel):
    """One option picked from the question's ``criteria`` keys."""

    model_config = ConfigDict(extra="allow")

    type: Literal["choice"] = "choice"
    choice: str = Field(..., description="The highest-probability option.")
    probabilities: dict[str, float] = Field(
        ..., description="Every option mapped to its probability (floats that sum to 1)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How certain the model is, derived from the probability distribution.",
    )


class ScoreAnswer(BaseModel):
    """A probability-weighted position on the question's ordered rubric.

    :attr:`score` can land *between* levels, so compare it against thresholds
    rather than indexing :attr:`legend` with it.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["score"] = "score"
    score: float = Field(
        ...,
        ge=0.0,
        description=("The probability-weighted answer across the levels; can land between levels."),
    )
    legend: dict[str, str] = Field(
        ..., description="Each level number mapped back to its description."
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Each level (string key) mapped to its probability (floats that sum to 1).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How certain the model is, derived from the probability distribution.",
    )


class UnknownAnswer(BaseModel):
    """An answer type this SDK release predates.

    Mirrors the catch-all arm of the API spec's answer union, which requires
    only ``type``. The payload survives in full on
    :attr:`~pydantic.BaseModel.model_extra`, so callers can read new fields
    before the SDK models them.
    """

    model_config = ConfigDict(extra="allow")

    type: str = Field(..., description="The answer type reported by the API.")


_KNOWN_ANSWER_TYPES = frozenset({"noul", "choice", "score"})


def _answer_tag(value: Any) -> str:
    """Route an answer payload to its union arm.

    A callable discriminator rather than ``Field(discriminator="type")``
    because :class:`UnknownAnswer` is the open arm: its ``type`` is a plain
    ``str``, and Pydantic requires every arm of a *field* discriminator to be
    a ``Literal``.
    """
    type_value = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return type_value if type_value in _KNOWN_ANSWER_TYPES else "unknown"


DecisionAnswer = Annotated[
    Annotated[NoulAnswer, Tag("noul")]
    | Annotated[ChoiceAnswer, Tag("choice")]
    | Annotated[ScoreAnswer, Tag("score")]
    | Annotated[UnknownAnswer, Tag("unknown")],
    Discriminator(_answer_tag),
]
"""One answer, narrowed by its ``type``.

Match on the concrete class rather than on ``type``::

    match answer:
        case NoulAnswer(noul=p) if p > 0.9: ...
        case ChoiceAnswer(choice=team, confidence=c) if c > 0.8: ...
"""


# ============================================================================
# Response model
# ============================================================================


class DecisionUsage(BaseModel):
    """Token accounting for a decision request.

    Decision models price input only; output tokens are billed at zero on the
    current catalog, but are reported so callers can see the real work done.
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int = Field(..., ge=0, description="Tokens consumed by the state and questions.")
    output_tokens: int = Field(..., ge=0, description="Tokens produced evaluating the questions.")


class DecisionResponse(VeniceBaseModel):
    """Response from ``POST /decisions``."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="The model that performed the evaluation.")
    answers: dict[str, DecisionAnswer] = Field(
        ..., description="One answer per question, keyed by the same ids used in the request."
    )
    usage: DecisionUsage = Field(..., description="Token accounting for the request.")

    # ------------------------------------------------------------------
    # Typed accessors
    #
    # ``answers`` is a union, so ``response.answers["team"].choice`` does not
    # type-check without narrowing at every call site. These accessors do the
    # narrowing once and fail loudly when a question came back as a different
    # type than the caller expected.
    # ------------------------------------------------------------------

    def _answer(self, question_id: str) -> Any:
        try:
            return self.answers[question_id]
        except KeyError:
            known = ", ".join(sorted(self.answers)) or "<none>"
            raise KeyError(
                f"No answer for question {question_id!r}. Answered ids: {known}."
            ) from None

    def _expect(self, question_id: str, expected: type, label: str) -> Any:
        answer = self._answer(question_id)
        if not isinstance(answer, expected):
            raise TypeError(
                f"Question {question_id!r} was answered as "
                f"{getattr(answer, 'type', type(answer).__name__)!r}, not {label!r}."
            )
        return answer

    def noul(self, question_id: str) -> float:
        """Return the yes/no probability for a ``noul`` question.

        :param question_id: The id used for this question in the request.
        :return: The answer on a scale from 0 (no) to 1 (yes).
        :raises KeyError: If no answer carries that id.
        :raises TypeError: If the question was not answered as a ``noul``.
        """
        return float(self._expect(question_id, NoulAnswer, "noul").noul)

    def choice(self, question_id: str) -> ChoiceAnswer:
        """Return the full :class:`ChoiceAnswer` for a ``choice`` question.

        Returns the whole answer rather than the bare option, because the
        probability distribution and confidence are usually what decide
        whether the option is safe to act on.

        :param question_id: The id used for this question in the request.
        :raises KeyError: If no answer carries that id.
        :raises TypeError: If the question was not answered as a ``choice``.
        """
        answer: ChoiceAnswer = self._expect(question_id, ChoiceAnswer, "choice")
        return answer

    def score(self, question_id: str) -> ScoreAnswer:
        """Return the full :class:`ScoreAnswer` for a ``score`` question.

        Returns the whole answer rather than the bare number, because
        :attr:`ScoreAnswer.legend` is what makes the number interpretable.

        :param question_id: The id used for this question in the request.
        :raises KeyError: If no answer carries that id.
        :raises TypeError: If the question was not answered as a ``score``.
        """
        answer: ScoreAnswer = self._expect(question_id, ScoreAnswer, "score")
        return answer


__all__ = [
    "ChoiceAnswer",
    "DecisionAnswer",
    "DecisionResponse",
    "DecisionUsage",
    "NoulAnswer",
    "ScoreAnswer",
    "UnknownAnswer",
]
