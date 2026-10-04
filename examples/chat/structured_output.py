#!/usr/bin/env python3
"""
Venice AI SDK - Structured Output Example

This example demonstrates how to get structured JSON responses from Venice AI
models using ``chat.completions.parse(response_format=PydanticModel)``. The
``parse()`` method derives the JSON schema from the model class, validates
the response against it, and returns a typed instance — no manual
``json.loads`` + field-presence checks required.

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-py SDK (with pydantic available)
"""

import asyncio
import operator
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from venice_ai import NoMatchingModelError, SystemMessage, UserMessage, VeniceClient
from venice_ai.exceptions import APIError, NotFoundError, VeniceError
from venice_ai.types.api.chat import ParsedChatCompletion
from venice_ai.types.api.requests import VeniceParameters

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _helpers import states_number  # noqa: E402

# Bounds a runaway generation; every schema here fits comfortably.
MAX_COMPLETION_TOKENS = 1024

# Re-sample at most once when a model's JSON fails validation.
MAX_PARSE_ATTEMPTS = 2

# Each request carries its own system message and the schema, so Venice's own
# system prompt is left out.
NO_VENICE_PROMPT = VeniceParameters(include_venice_system_prompt=False)


def _finished(parsed: ParsedChatCompletion) -> bool:
    """Print the finish reason and return ``False`` if the output was truncated.

    Some models report output cut off at the cap as ``finish_reason="stop"``,
    so the completion-token count is checked against the cap as well.
    """
    used = parsed.usage.completion_tokens if parsed.usage else None
    print(f"🏁 Finish reason: {parsed.finish_reason} ({used} of {MAX_COMPLETION_TOKENS} tokens)")
    if parsed.finish_reason == "length" or used is None or used >= MAX_COMPLETION_TOKENS:
        print("❌ The response may be cut off by the token limit")
        return False
    return True


# -----------------------------------------------------------------------------
# Example 1: Math Problem Solver
# -----------------------------------------------------------------------------


class MathStep(BaseModel):
    explanation: str = Field(description="What this step does, in plain words")
    output: str = Field(description="The equation or value that results from this step")


class MathResponse(BaseModel):
    steps: list[MathStep]
    final_answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


async def math_problem_example(client: VeniceClient, model: str) -> bool:
    """Demonstrate structured output for solving math problems step-by-step."""
    print("\n📊 Example 1: Math Problem Solver")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "You are a helpful math tutor. Write every step in plain text, "
                        "without LaTeX or Markdown math. Always include your confidence "
                        "level (0-1) in your response."
                    )
                ),
                UserMessage(content="Solve the equation: 3x + 15 = 42"),
            ],
            response_format=MathResponse,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.2,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
        # APITimeoutError, and APIConnectionError. A timeout on a single slow
        # request is tallied as one failed example, not a whole-run crash.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    if not _finished(parsed):
        return False
    result = parsed.parsed
    print("📋 Parsed the solution")
    print("\n🔧 Solution Steps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i}: {step.explanation}")
        print(f"          → {step.output}")

    print(f"\n🏷️ Final Answer: {result.final_answer}")
    if result.confidence is not None:
        print(f"📊 Confidence Level: {result.confidence:.2%}")

    # 3x + 15 = 42  →  x = 9
    if not states_number(result.final_answer, 9):
        print("❌ Expected the final answer to be x = 9")
        return False
    print("✅ Final answer verified: x = 9")
    return True


# -----------------------------------------------------------------------------
# Example 2: Data Extraction from Text
# -----------------------------------------------------------------------------


class Headquarters(BaseModel):
    city: str
    country: str


class CompanyInfo(BaseModel):
    company_name: str
    founded_year: int | None = None
    founders: list[str] = Field(
        default_factory=list,
        description="Full names of individual people only; empty if none are named",
    )
    industry: str
    headquarters: Headquarters
    key_products: list[str] = Field(default_factory=list)
    employee_count: int | None = Field(
        default=None, description="Only if the text states a number of employees"
    )


async def data_extraction_example(client: VeniceClient, model: str) -> bool:
    """Extract structured information from unstructured text."""
    print("\n📊 Example 2: Data Extraction")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    # A fictional company, so every expected value comes from the text alone.
    text = """
    Brightwater Robotics was founded in 2019 by Amara Lindqvist and Tomás Ferreira.
    The company is headquartered in Porto, Portugal, and builds autonomous
    hull-cleaning drones for cargo ships. Its two products are the HullSweep S2
    drone and the TideLog fleet-monitoring service. Brightwater now employs
    140 people and operates in the maritime robotics industry.
    """

    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "Extract structured information from the provided text. "
                        "Only include information explicitly mentioned."
                    )
                ),
                UserMessage(content=f"Extract company information from this text:\n\n{text}"),
            ],
            response_format=CompanyInfo,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
        # APITimeoutError, and APIConnectionError. A timeout on a single slow
        # request is tallied as one failed example, not a whole-run crash.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    if not _finished(parsed):
        return False
    result = parsed.parsed
    print("📋 Parsed the company information")
    print(f"\n🏢 Company: {result.company_name}")
    print(f"🏭 Industry: {result.industry}")
    print(f"📍 Headquarters: {result.headquarters.city}, {result.headquarters.country}")

    if result.founded_year is not None:
        print(f"📅 Founded: {result.founded_year}")
    if result.founders:
        print(f"👥 Founders: {', '.join(result.founders)}")
    if result.key_products:
        print("📦 Key Products:")
        for product in result.key_products:
            print(f"   - {product}")
    if result.employee_count is not None:
        print(f"👷 Employees: {result.employee_count}")

    # Check the extraction against facts stated in the text.
    expected = {
        "founded_year": (result.founded_year, 2019),
        "founders": (sorted(result.founders), ["Amara Lindqvist", "Tomás Ferreira"]),
        "headquarters city": (result.headquarters.city, "Porto"),
        "employee_count": (result.employee_count, 140),
    }
    mismatches = [
        f"{field}: got {got!r}, expected {want!r}"
        for field, (got, want) in expected.items()
        if got != want
    ]
    if mismatches:
        for mismatch in mismatches:
            print(f"❌ {mismatch}")
        return False
    print("✅ Every checked field matches the source text")
    return True


# -----------------------------------------------------------------------------
# Example 3: Multiple Choice Quiz Generation
# -----------------------------------------------------------------------------


# Each question is a single Python integer expression ``left <operator> right``.
# Keeping the expression in typed fields (instead of free text) lets the code
# compute the true answer and check the model's answer key against it.
_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
}


class QuizQuestion(BaseModel):
    # Field order matters: models write JSON top to bottom, so asking for the
    # working before the answer key gives the model room to compute it first.
    # The operator set is named in the prompt too: a schema enum alone does
    # not stop a model from writing a question about an operator outside it
    # (such as **) and filing it under the nearest allowed one.
    left: int = Field(ge=-100, le=100)
    operator: Literal["+", "-", "*", "//", "%"]
    right: int = Field(ge=1, le=20)
    explanation: str = Field(description="Step-by-step evaluation of the expression")
    answer: int = Field(description="The value Python computes")
    options: list[int] = Field(min_length=4, max_length=4)
    correct_answer: int = Field(ge=0, le=3, description="Index of `answer` in options")

    @field_validator("options")
    @classmethod
    def _options_distinct(cls, options: list[int]) -> list[int]:
        # JSON Schema's uniqueItems is not accepted by every structured-output
        # backend, so distinctness is enforced here instead.
        if len(set(options)) != len(options):
            raise ValueError(f"options must be distinct, got {options}")
        return options


class Quiz(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    questions: list[QuizQuestion] = Field(min_length=3, max_length=3)


async def quiz_generation_example(client: VeniceClient, model: str) -> bool:
    """Generate a structured quiz, then check its answer key in code.

    Schema validation proves the JSON has the right shape, not that its
    content is right. Because each question is a typed expression, we can
    compute the true answer and reject a quiz whose key is wrong.
    """
    print("\n📊 Example 3: Quiz Generation")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    messages: list[SystemMessage | UserMessage] = [
        SystemMessage(
            content=(
                "You are a quiz generator. Each question asks for the value of "
                "one Python integer expression 'left operator right', where "
                "operator is exactly one of +, -, *, // or %. Work out the "
                "result in the explanation first, then give the answer, four "
                "distinct integer options containing it exactly once, and the "
                "index of the answer in the options."
            )
        ),
        UserMessage(
            content=(
                "Create a quiz about Python's integer operators with 3 questions "
                "of medium difficulty. Include at least one question using // "
                "and one using % with a negative left operand."
            )
        ),
    ]

    # Bounded retry on ValidationError only (e.g. duplicate options): the
    # schema can't stop every model from repeating a value, and re-sampling
    # usually fixes it. API errors are returned immediately.
    max_attempts = MAX_PARSE_ATTEMPTS
    parsed = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = await client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=Quiz,
                venice_parameters=NO_VENICE_PROMPT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.3,
            )
            break
        except ValidationError as e:
            print(f"⚠️ Attempt {attempt}/{max_attempts}: model output didn't satisfy schema.")
            if attempt == max_attempts:
                print(f"❌ Pydantic ValidationError after {max_attempts} attempts:\n{e}")
                return False
            print("   Retrying...")
        except VeniceError as e:
            # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
            # APITimeoutError, and APIConnectionError. A timeout on a single slow
            # request is tallied as one failed example, not a whole-run crash.
            print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
            return False

    assert parsed is not None  # loop exits only via break (success) or return
    if not _finished(parsed):
        return False
    quiz = parsed.parsed
    print("📋 Parsed the quiz")
    print(f"\n📚 Topic: {quiz.topic}")
    print(f"⚡ Difficulty: {quiz.difficulty}")
    print(f"❓ Questions: {len(quiz.questions)}")
    print("\n" + "-" * 40)

    wrong_keys = 0
    for i, q in enumerate(quiz.questions, 1):
        expected = _OPERATORS[q.operator](q.left, q.right)
        print(f"\nQuestion {i}: In Python, what is {q.left} {q.operator} {q.right}?")
        for j, option in enumerate(q.options):
            marker = "✓" if j == q.correct_answer else " "
            print(f"  {marker} {chr(65 + j)}. {option}")
        print(f"💡 Explanation: {q.explanation}")
        keyed = q.options[q.correct_answer]
        if keyed != expected or q.answer != expected:
            if q.answer != expected:
                print(f"❌ Answer says {q.answer}, but Python evaluates it to {expected}")
            if keyed != expected:
                print(f"❌ Keyed option says {keyed}, but Python evaluates it to {expected}")
            wrong_keys += 1
        else:
            print(f"✅ Answer key verified: {expected}")

    if wrong_keys:
        print(f"\n❌ {wrong_keys} of {len(quiz.questions)} answer keys are wrong")
        return False
    print(f"\n✅ All {len(quiz.questions)} answer keys verified")
    return True


# -----------------------------------------------------------------------------
# Example 4: Task Planning and Organization
# -----------------------------------------------------------------------------


class ProjectTask(BaseModel):
    id: str
    name: str
    estimated_hours: float
    dependencies: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high", "critical"]


class ProjectPhase(BaseModel):
    name: str
    description: str
    tasks: list[ProjectTask]


class ProjectRisk(BaseModel):
    description: str
    mitigation: str


class ProjectPlan(BaseModel):
    # The plan's total hours are summed from the tasks in code; models are
    # unreliable at arithmetic.
    project_name: str
    phases: list[ProjectPhase]
    risks: list[ProjectRisk] = Field(default_factory=list)


async def task_planning_example(client: VeniceClient, model: str) -> bool:
    """Generate a structured task plan with dependencies and time estimates."""
    print("\n📊 Example 4: Task Planning")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    # The ProjectPlan schema is the deepest in this file (plan → phases[] →
    # tasks[] → risks[]). Soft-structured-output models occasionally emit JSON
    # that drifts from a nested schema — e.g. a risk shaped {"risk": "..."}
    # instead of {"description": ..., "mitigation": ...}. Two reinforcements
    # keep it reliable:
    #   1. Spell the exact field names/types into the prompt (response_format
    #      already sends the schema, but naming the fields in prose measurably
    #      improves compliance) and cap the plan size so the response stays
    #      small and fast — a smaller response is both quicker to generate
    #      (avoiding request timeouts on a big model) and less prone to drift.
    #   2. A bounded retry on ValidationError (below): re-sampling at a low
    #      temperature usually fixes the occasional non-conforming response.
    system_prompt = (
        "You are a project manager. Create concise, realistic project plans. "
        "Return JSON that matches this exact structure:\n"
        "- project_name: string\n"
        "- phases: array of objects, each with:\n"
        '    - "name": string\n'
        '    - "description": string\n'
        '    - "tasks": array of objects, each with:\n'
        '        - "id": string (e.g. "T1")\n'
        '        - "name": string\n'
        '        - "estimated_hours": number\n'
        '        - "dependencies": array of task-id strings (use [] if none)\n'
        '        - "priority": one of "low", "medium", "high", "critical"\n'
        "- risks: array of objects, each with EXACTLY two string fields:\n"
        '    - "description": string (what could go wrong)\n'
        '    - "mitigation": string (how to handle it)\n'
        'Do NOT use any other field names for a risk (no bare "risk" field).'
    )
    user_prompt = (
        "Create a project plan for building a simple REST API with user "
        "authentication. Keep it small: exactly 2 phases, 2-3 tasks per phase, "
        "and exactly 2 risks. Each risk must be an object with a 'description' "
        "and a 'mitigation' field."
    )

    # Bounded retry: structured-output models occasionally return JSON that
    # violates the schema. A short retry (re-sampling) usually resolves it.
    # We only retry ValidationError; VeniceError (HTTP/transport) is returned
    # immediately so a genuine API problem isn't masked by pointless retries.
    max_attempts = MAX_PARSE_ATTEMPTS
    parsed = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = await client.chat.completions.parse(
                model=model,
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ],
                response_format=ProjectPlan,
                venice_parameters=NO_VENICE_PROMPT,
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                temperature=0.2,
            )
            break
        except ValidationError as e:
            print(
                f"⚠️ Attempt {attempt}/{max_attempts}: model output didn't satisfy "
                f"schema ({type(e).__name__})."
            )
            if attempt == max_attempts:
                print(f"❌ Pydantic ValidationError after {max_attempts} attempts:\n{e}")
                return False
            print("   Retrying...")
        except VeniceError as e:
            # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
            # APITimeoutError, and APIConnectionError. A timeout on a single slow
            # request is tallied as one failed example, not a whole-run crash.
            print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
            return False

    assert parsed is not None  # loop exits only via break (success) or return
    if not _finished(parsed):
        return False
    plan = parsed.parsed
    tasks = [task for phase in plan.phases for task in phase.tasks]
    total_hours = sum(task.estimated_hours for task in tasks)
    print("📋 Parsed the project plan")
    print(f"\n🎯 Project: {plan.project_name}")
    print(f"⏱️ Total Estimated Hours: {total_hours:g} (summed from the tasks in code)")
    print("\n" + "=" * 40)

    for phase in plan.phases:
        print(f"\n📌 Phase: {phase.name}")
        print(f"   {phase.description}")
        print("   Tasks:")
        for task in phase.tasks:
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            print(
                f"   • {task.id} [{task.priority.upper()}] {task.name} - "
                f"{task.estimated_hours}h{deps}"
            )

    if plan.risks:
        print("\n⚠️ Identified Risks:")
        for risk in plan.risks:
            print(f"   • Risk: {risk.description}")
            print(f"     Mitigation: {risk.mitigation}")

    # Schema validation can't check cross-references: every dependency must
    # name a task that exists in the plan.
    task_ids = {task.id for task in tasks}
    dangling = sorted({dep for task in tasks for dep in task.dependencies} - task_ids)
    if not tasks or dangling:
        print(f"\n❌ The plan is inconsistent: no tasks or unknown dependencies {dangling}")
        return False
    print(f"\n✅ {len(tasks)} tasks; every dependency refers to a task in the plan")
    return True


# -----------------------------------------------------------------------------
# Example 5: Error Handling and Validation
# -----------------------------------------------------------------------------


class StrictResponse(BaseModel):
    status: Literal["success", "failure"]
    code: int = Field(ge=100, le=999)
    timestamp: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


async def error_handling_example(client: VeniceClient, model: str) -> bool:
    """Demonstrate the two failure classes a structured-output call can raise.

    ``parse()`` raises ``pydantic.ValidationError`` when the model's JSON
    doesn't match the schema (wrong types, missing fields, pattern mismatch),
    and a ``VeniceError`` subclass (``NotFoundError``, ``InvalidRequestError``,
    ``APITimeoutError``, ...) when the request itself fails. This demo triggers
    each one on purpose, then shows a strict schema succeeding.
    """
    print("\n🛡️ Example 5: Error Handling Demo")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")
    ok = True

    # (a) Schema violation. This is the payload shape parse() rejects when a
    # model ignores the constraints: an unknown status, an out-of-range code,
    # and a timestamp that doesn't match the pattern.
    print("\n(a) Validating a non-conforming payload against StrictResponse...")
    bad_payload = '{"status": "ok", "code": 42, "timestamp": "yesterday"}'
    try:
        StrictResponse.model_validate_json(bad_payload)
    except ValidationError as e:
        print(f"   ✅ Caught pydantic.ValidationError ({e.error_count()} errors):")
        for err in e.errors():
            field = ".".join(str(part) for part in err["loc"])
            print(f"      • {field}: {err['msg']}")
    else:
        print("   ❌ The invalid payload was accepted; the schema is not enforcing constraints")
        ok = False

    # (b) API failure. A model id that doesn't exist is rejected by the server
    # and surfaces as a classified SDK exception.
    print("\n(b) Requesting structured output from a model that doesn't exist...")
    try:
        await client.chat.completions.parse(
            model="nonexistent-model",
            messages=[UserMessage(content="Generate a success response")],
            response_format=StrictResponse,
        )
    except NotFoundError as e:
        print(f"   ✅ Caught {type(e).__name__} (HTTP {e.status_code})")
        print("   💡 Catch APIError subclasses for request failures, separately from")
        print("      ValidationError for schema failures.")
    except APIError as e:
        print(f"   ❌ Expected NotFoundError, got {type(e).__name__}: {e}")
        ok = False
    else:
        print("   ❌ The request unexpectedly succeeded")
        ok = False

    # (c) Success with a strict schema. A model can't know the current time,
    # so we supply the timestamp and check that it comes back unchanged.
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"\n(c) Strict structured output echoing timestamp {now}...")
    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "Generate a response with status='success', code=200, and "
                        f"timestamp exactly '{now}'."
                    )
                ),
                UserMessage(content="Generate a success response"),
            ],
            response_format=StrictResponse,
            venice_parameters=NO_VENICE_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.1,
        )
    except ValidationError as e:
        print(f"   ❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        print(f"   ❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    if not _finished(parsed):
        return False
    result = parsed.parsed
    print("   ✅ Valid response received:")
    print(f"      Status: {result.status}")
    print(f"      Code: {result.code}")
    print(f"      Timestamp: {result.timestamp}")
    if (result.status, result.code, result.timestamp) != ("success", 200, now):
        print("   ❌ The response does not match the requested values")
        return False
    return ok


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------


async def main() -> int:
    """Run all structured output examples: 0 if all pass, 1 if any fail, 77 if no model."""
    print("🚀 Venice AI Structured Output Examples")
    print("=" * 60)
    print("Demonstrates chat.completions.parse(response_format=PydanticModel)")
    print("for typed, schema-validated JSON responses.")

    async with VeniceClient() as client:
        print("\n✅ Client initialized successfully")

        print("\n🔍 Searching for models with structured output support...")
        # The catalog's default ranking among models that accept a JSON schema.
        # exclude_uncensored skips models Venice flags as tuned for open-ended
        # creative dialogue; this example wants careful, factual answers.
        try:
            chat_model = await client.models.resolve_chat(
                require_response_schema=True, exclude_uncensored=True
            )
        except NoMatchingModelError as e:
            print(f"SKIPPED: no catalog model supports a JSON response schema ({e})")
            return 77
        print(f"📍 Selected model: {chat_model}")

        examples = [
            ("Math Problem Solver", math_problem_example),
            ("Data Extraction", data_extraction_example),
            ("Quiz Generation", quiz_generation_example),
            ("Task Planning", task_planning_example),
            ("Error Handling Demo", error_handling_example),
        ]

        results: list[tuple[str, bool]] = []
        for name, fn in examples:
            ok = await fn(client, chat_model)
            results.append((name, ok))

        passed = sum(1 for _, ok in results if ok)
        failed = len(results) - passed

        print("\n" + "=" * 60)
        if failed:
            print(
                f"❌ {passed}/{len(results)} structured output examples completed; {failed} failed"
            )
            for name, ok in results:
                status = "✓" if ok else "✗"
                print(f"   {status} {name}")
            return 1

        print(f"✨ All {passed}/{len(results)} structured output examples completed!")
        print("\n💡 Key Takeaways:")
        print("   • Define schemas as Pydantic BaseModel classes")
        print(
            "   • chat.completions.parse(response_format=ModelClass) returns ParsedChatCompletion[T]"
        )
        print("   • parsed.parsed is a typed instance — no manual json.loads needed")
        print("   • Check finish_reason AND completion tokens against the cap: some models")
        print("     report a truncated response as 'stop', and it is not a valid result")
        print("   • Schema validation checks shape, not truth: verify content in code")
        print("   • Put reasoning fields before answer fields in the schema")
        print("   • Pydantic raises ValidationError on schema violations; request")
        print("     failures raise VeniceError subclasses such as NotFoundError")
        print("   • Lower temperature values improve consistency")
        print("\n📚 Learn more at: https://docs.venice.ai/guides/features/structured-responses")
        return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
