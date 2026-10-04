"""
Venice AI SDK - Production Cost Management

This example demonstrates how to track, monitor, and optimize costs when using
the Venice AI SDK in production:

1. Token usage tracking via the built-in :class:`venice_ai.CostTracker`
2. Cost estimation per request, checked against the actual bill
3. Model selection by price, using catalog pricing
4. Usage analytics and reporting
5. Budget management via :class:`venice_ai.BudgetManager`

Requirements:
    pip install venice-py
    export VENICE_API_KEY="your-api-key"
"""

import asyncio
import sys
import uuid
from collections.abc import Awaitable
from decimal import Decimal

from pydantic import BaseModel, ValidationError

from venice_ai import (
    BudgetManager,
    CostTracker,
    NoMatchingModelError,
    VeniceClient,
    model_price,
)
from venice_ai.exceptions import VeniceError
from venice_ai.types.api.chat import ChatCompletionResponse
from venice_ai.types.api.requests import (
    SystemMessage,
    Tool,
    ToolFunction,
    UserMessage,
    VeniceParameters,
)

MILLION = Decimal(1_000_000)

# Every request bills only the prompts shown here. Without this, Venice
# prepends its own system prompt and bills it as prompt tokens on every
# request; that prompt is shared by all traffic and often already cached, so
# it would also blur which cached tokens came from this example's own prefix.
OWN_PROMPT_ONLY = VeniceParameters(include_venice_system_prompt=False)

# Pattern 1's shared prefix: a support handbook reused verbatim by every
# request, about 2.4k tokens, well past the ~1024-token minimum providers cache.
# The run id at the top keeps it out of the cache when the run starts.
HANDBOOK_PREFIX = (
    f"Support handbook {uuid.uuid4().hex[:8]}. Answer customer questions in one "
    "sentence, using only this handbook.\n\n"
    + (
        "Policy: refunds are issued within 14 days of purchase to the original "
        "payment method, and support replies within one business day. " * 100
    )
)

# Requests Pattern 1 may send: one cold, then warm ones until a cache hit.
# Caching is best effort (a request can reach a backend without the cache),
# so a few warm attempts are allowed, about two seconds apart.
CACHE_DEMO_MAX_REQUESTS = 4
CACHE_DEMO_DELAY_S = 2.0


class UnitTestBenefits(BaseModel):
    """Pattern 2's structured answer: the response_format sent as a schema."""

    benefits: list[str]


# The price comparison in Pattern 3 stops at this blended list price (USD per
# 1M tokens, the measure prefer="cheapest" ranks by), so a demo never runs a
# premium model just because it tops the catalog.
COMPARISON_PRICE_CEILING = 1.00


def check_answer(response: ChatCompletionResponse) -> bool:
    """Print finish_reason and report whether the answer is complete."""
    finish_reason = response.choices[0].finish_reason if response.choices else None
    print(f"   finish_reason: {finish_reason}")
    if finish_reason == "length" or not (response.text or "").strip():
        print("   ❌ Answer was truncated or empty")
        return False
    return True


def usage_breakdown(response: ChatCompletionResponse) -> tuple[int, int, int, int]:
    """Return (prompt, cached, cache_written, completion) token counts."""
    usage = response.usage
    if usage is None:
        return 0, 0, 0, 0
    return (
        usage.prompt_tokens,
        usage.cached_tokens,
        usage.cache_write_tokens,
        usage.completion_tokens,
    )


# =============================================================================
# Example Patterns
# =============================================================================


async def example_basic_cost_tracking(client: VeniceClient, tracker: CostTracker) -> bool | None:
    """Track requests that share a cached prefix.

    Returns None (skipped) when no catalog model lists a cached-input price.
    """
    print("=" * 60)
    print("Pattern 1: Basic Cost Tracking — tracker.track()")
    print("=" * 60)

    # A model with a cached-input price, so the bill below can show cached
    # prompt tokens charged at the lower rate. The catalog's default ranking is
    # used rather than prefer="cheapest": a cache price is the only caching
    # signal Venice publishes, and some of the cheapest cache-priced models do
    # not serve cache hits.
    try:
        model = await client.models.resolve_chat(require_prompt_caching=True)
    except NoMatchingModelError as e:
        print(f"\nSection skipped: no chat model in the catalog lists a cached-input price ({e})")
        return None
    pricing = tracker.pricing_map.get(model)
    if pricing is None or pricing.cache_input is None:
        print(f"\n❌ The catalog lists no cached-input price for {model}")
        return False
    input_rate = Decimal(str(pricing.input.usd))
    cache_rate = Decimal(str(pricing.cache_input.usd))
    output_rate = Decimal(str(pricing.output.usd))
    # Only rates the catalog publishes are shown as rates. A model without a
    # cache-write price bills written tokens at its input rate, which is how
    # CostTracker prices them too.
    if pricing.cache_write is not None:
        write_rate = Decimal(str(pricing.cache_write.usd))
        write_note = f"cache write ${pricing.cache_write.usd}"
    else:
        write_rate = input_rate
        write_note = "no cache-write price published (written tokens bill at the input rate)"
    print(f"\nModel: {model}")
    print(
        f"Catalog rates per 1M tokens: input ${pricing.input.usd}, cached input "
        f"${pricing.cache_input.usd}, output ${pricing.output.usd}; {write_note}"
    )
    print(
        f"Every request starts with the same {len(HANDBOOK_PREFIX.split())}-word handbook, "
        "so after the first one the prefix can come from the cache.\n"
    )

    questions = [
        "How long do I have to ask for a refund?",
        "Where does my refund go?",
        "How quickly does support reply?",
        "Can I get a refund after three weeks?",
    ][:CACHE_DEMO_MAX_REQUESTS]

    ok = True
    verified_cached = 0
    for i, question in enumerate(questions):
        if i:
            await asyncio.sleep(CACHE_DEMO_DELAY_S)
        response = await client.chat.completions.create(
            model=model,
            messages=[SystemMessage(content=HANDBOOK_PREFIX), UserMessage(content=question)],
            max_completion_tokens=150,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        cost = await tracker.track(response, metadata={"prompt": question})
        prompt_tokens, cached, written, completion = usage_breakdown(response)

        print(f"Request {i + 1}: {question}")
        ok = check_answer(response) and ok
        print(f"   Answer: {(response.text or '').strip()}")
        print(
            f"   Tokens: {prompt_tokens} prompt ({cached} cache read, {written} cache write) "
            f"+ {completion} completion"
        )
        print(f"   Tracked cost: ${cost:.6f}")

        # Rebuild the bill from the published rates to show how it is made up.
        by_hand = (
            Decimal(prompt_tokens - cached - written) * input_rate
            + Decimal(cached) * cache_rate
            + Decimal(written) * write_rate
            + Decimal(completion) * output_rate
        ) / MILLION
        match = abs(by_hand - cost) < Decimal("0.000000001")
        print(f"   By hand from the rates: ${by_hand:.6f} {'✓' if match else '✗ mismatch'}")
        ok = ok and match
        if i == 0 and cached:
            # Nothing this run sent can be cached yet, so a cold read means the
            # numbers below would not show this example's own prefix.
            print("   ❌ The first request already read from cache")
            ok = False
        if i > 0 and cached:
            at_input_rate = (
                Decimal(prompt_tokens - written) * input_rate
                + Decimal(written) * write_rate
                + Decimal(completion) * output_rate
            ) / MILLION
            saved = at_input_rate - cost
            discounted = saved > 0
            print(
                f"   Without the cache discount: ${at_input_rate:.6f} "
                f"(saved ${saved:.6f}) {'✓' if discounted else '✗ no discount applied'}"
            )
            ok = ok and discounted
            if discounted and match:
                verified_cached += 1
        print()
        if verified_cached:
            break

    summary = await tracker.summary()
    print("📊 Summary:")
    print(f"   Total requests: {summary.total_requests}")
    print(f"   Total cost: ${summary.total_cost_usd:.6f}")
    print(f"   Total tokens: {summary.total_tokens}")
    print(f"   Avg cost/request: ${summary.average_cost_usd:.6f}")
    if verified_cached == 0:
        print("   ❌ No request read the handbook from cache at the cached rate")
        ok = False
    else:
        print(
            f"\n   ℹ️  {verified_cached} request(s) read the shared handbook prefix from "
            "cache, and those tokens were billed at the cached rate."
        )
    return ok and summary.total_requests >= 2


async def _estimate_and_send(
    client: VeniceClient,
    tracker: CostTracker,
    model: str,
    messages: list,
    max_completion_tokens: int,
    tools: list[Tool] | None = None,
    response_format: type[UnitTestBenefits] | None = None,
    template_floor: int = 0,
) -> tuple[bool, int]:
    """Estimate one request, send it, and check the estimate against the bill.

    Returns whether the checks passed and the prompt tokens the request billed.

    ``estimate_cost`` promises an upper bound on what the request itself
    bills: its prompt estimate is meant to be at least the prompt tokens of
    the messages, tools, schema and chat template, and its completion side is
    the cap. Some models' templates also carry a built-in system prompt that
    the catalog does not publish, so *template_floor* (measured once per model
    by :func:`_measure_template_floor`) is added before comparing. The check is
    that neither the prompt tokens nor the total cost came out above that.
    How far above the bill the estimate lands is printed, not checked: the
    tools and schema allowances are sized for the most expensive chat
    templates measured, so on a model that renders them compactly the
    estimate can be several times the actual prompt.
    """
    # Pass the estimate everything the request bills as prompt tokens: the
    # messages, the tools or schema, and the venice_parameters that decide
    # whether the Venice system prompt is added.
    estimate = await client.chat.completions.estimate_cost(
        model=model,
        messages=messages,
        expected_completion_tokens=max_completion_tokens,
        venice_parameters=OWN_PROMPT_ONLY,
        tools=tools,
        response_format=response_format,
    )
    print(
        f"   🔮 Estimate: {estimate.prompt_tokens} prompt tokens "
        f"(incl. {estimate.template_overhead_tokens} chat-template allowance, "
        f"{estimate.schema_tokens} for tools/schema, "
        f"{estimate.venice_system_prompt_tokens} for the Venice system prompt) "
        f"+ {estimate.expected_completion_tokens} completion (the cap) "
        f"= ${estimate.total_cost_usd:.6f}"
    )

    if response_format is not None:
        try:
            parsed = await client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                max_completion_tokens=max_completion_tokens,
                venice_parameters=OWN_PROMPT_ONLY,
            )
        except ValidationError as e:
            print(f"   ❌ The answer did not match the schema: {e.error_count()} error(s)")
            return False, 0
        response = parsed.response
        print(f"   🧾 Parsed: {parsed.parsed!r}")
        if not parsed.parsed.benefits:
            print("   ❌ The parsed answer lists no benefits")
            return False, 0
    elif tools:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            venice_parameters=OWN_PROMPT_ONLY,
            tools=tools,
        )
    else:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            venice_parameters=OWN_PROMPT_ONLY,
        )
    ok = check_answer(response)
    cost = await tracker.track(response)
    prompt_tokens, cached, _, completion = usage_breakdown(response)
    print(
        f"   💵 Actual:   {prompt_tokens} prompt tokens ({cached} cached) "
        f"+ {completion} completion = ${cost:.6f}"
    )

    if prompt_tokens == 0:
        print("   ❌ The response reported no prompt tokens")
        return False, 0
    covered = estimate.prompt_tokens + template_floor
    floor_note = f" + {template_floor} measured template floor" if template_floor else ""
    if covered < prompt_tokens:
        print(f"   ❌ The prompt estimate{floor_note} is {prompt_tokens - covered} tokens low")
        ok = False
    else:
        print(
            f"   ✅ Prompt estimate{floor_note} covers the bill: "
            f"{covered / prompt_tokens:.2f}x the actual prompt"
        )
    floor_cost = Decimal(0)
    if template_floor:
        input_rate = estimate.prompt_cost_usd / max(estimate.prompt_tokens, 1)
        floor_cost = input_rate * template_floor
    if cost > estimate.total_cost_usd + floor_cost:
        print("   ❌ Actual cost exceeded the estimate; budget checks would under-reserve")
        ok = False
    return ok, prompt_tokens


async def _measure_template_floor(client: VeniceClient, tracker: CostTracker, model: str) -> int:
    """Return the prompt tokens *model* bills beyond ``estimate_cost`` for a one-word request.

    ``estimate_cost`` cannot see a system prompt built into a model's own chat
    template: the catalog does not publish one. A one-token request with the
    Venice system prompt off shows it once per model, and the difference is
    added to every later comparison on that model.
    """
    probe = [UserMessage(content="Hi")]
    estimate = await client.chat.completions.estimate_cost(
        model=model,
        messages=probe,
        expected_completion_tokens=1,
        venice_parameters=OWN_PROMPT_ONLY,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=probe,
        max_completion_tokens=1,
        venice_parameters=OWN_PROMPT_ONLY,
    )
    await tracker.track(response)
    billed = response.usage.prompt_tokens if response.usage else 0
    floor = max(0, billed - estimate.prompt_tokens)
    print(
        f"   📐 Template floor on {model}: a one-word request billed {billed} prompt tokens "
        f"against an estimate of {estimate.prompt_tokens}"
        + (f", so {floor} tokens come from the model's own template" if floor else "")
    )
    return floor


async def example_estimate_vs_actual(client: VeniceClient, tracker: CostTracker) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 2: Pre-flight Estimate vs Actual Cost")
    print("=" * 60)

    # A model that answers directly: a reasoning model would spend part of the
    # small completion budget thinking.
    model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    messages = [UserMessage(content="List three benefits of unit tests, one line each.")]
    print(f"\nModel: {model}")

    print("\n📝 Plain request:")
    floor = await _measure_template_floor(client, tracker, model)
    ok, _ = await _estimate_and_send(
        client, tracker, model, messages, max_completion_tokens=200, template_floor=floor
    )

    # Tool definitions are sent as prompt tokens too, so the estimate needs them.
    tool_model = await client.models.resolve_chat(
        prefer="cheapest", exclude_reasoning=True, require_function_calling=True
    )
    lookup_tool = Tool(
        type="function",
        function=ToolFunction(
            name="lookup_order",
            description="Look up a customer order by its id and return its status.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "The order id"}},
                "required": ["order_id"],
            },
        ),
    )
    print(f"\n🔧 Same prompt with a tool definition attached ({tool_model}):")
    tool_floor = await _measure_template_floor(client, tracker, tool_model)
    tools_ok, _ = await _estimate_and_send(
        client,
        tracker,
        tool_model,
        messages,
        max_completion_tokens=200,
        tools=[lookup_tool],
        template_floor=tool_floor,
    )
    ok = tools_ok and ok

    # A json_schema response_format: some models bill the schema as prompt
    # tokens and others enforce it while decoding at no prompt cost, and the
    # estimate assumes the former. The catalog's default ranking picks the
    # model, as in examples/best_practices/pydantic_models.py, rather than
    # prefer="cheapest": some models that list schema support still reject
    # response_format requests, and the default ranking favors models that
    # serve them reliably. The same prompt is sent to that model without the
    # schema first, so the difference shows what the schema itself added to
    # the bill.
    schema_model = await client.models.resolve_chat(
        require_response_schema=True, exclude_reasoning=True
    )
    print(f"\n📝 Plain request on the schema model ({schema_model}):")
    schema_floor = await _measure_template_floor(client, tracker, schema_model)
    baseline_ok, baseline_prompt = await _estimate_and_send(
        client,
        tracker,
        schema_model,
        messages,
        max_completion_tokens=200,
        template_floor=schema_floor,
    )
    print(f"\n🧾 Same prompt with a json_schema response_format ({schema_model}):")
    schema_ok, schema_prompt = await _estimate_and_send(
        client,
        tracker,
        schema_model,
        messages,
        max_completion_tokens=200,
        response_format=UnitTestBenefits,
        template_floor=schema_floor,
    )
    ok = baseline_ok and schema_ok and ok
    if baseline_prompt and schema_prompt:
        added = schema_prompt - baseline_prompt
        print(f"   📏 The schema added {added} prompt tokens on {schema_model}")
        if added <= 0:
            print(
                "   This model enforces the schema while decoding, so this run did not\n"
                "   test the estimate's schema allowance against a schema billed as prompt."
            )
    return ok


def _pick_price_extremes(
    entries: list, exclude: set[str], ceiling: float
) -> tuple[str, str] | None:
    """Return the cheapest and the priciest plain chat model up to *ceiling*.

    Prices are :func:`venice_ai.models.selection.model_price`, the blended
    per-1M-token price that ``prefer="cheapest"`` ranks by. Reasoning, beta,
    offline, deprecated, extended-context-priced and end-to-end-encrypted
    models are skipped so the comparison is like with like.
    """
    candidates: list[tuple[float, str]] = []
    for entry in entries:
        spec = entry.model_spec
        caps = getattr(spec, "capabilities", None)
        pricing = getattr(spec, "pricing", None)
        if not spec or not caps or not pricing or entry.id in exclude:
            continue
        if caps.supportsReasoning or caps.supportsE2EE or spec.beta or spec.offline:
            continue
        if spec.deprecation is not None or pricing.extended is not None:
            continue
        price = model_price(entry.model_dump())
        if price is None or price > ceiling:
            continue
        candidates.append((price, entry.id))

    if len(candidates) < 2:
        return None
    candidates.sort()
    return candidates[0][1], candidates[-1][1]


async def example_cost_optimization(
    client: VeniceClient, tracker: CostTracker
) -> tuple[bool, tuple[str, str] | None]:
    print("\n" + "=" * 60)
    print("Pattern 3: Cost Optimization — pick the model by price")
    print("=" * 60)

    catalog = await client.models.list(type="text")
    pair = _pick_price_extremes(catalog.data, exclude=set(), ceiling=COMPARISON_PRICE_CEILING)
    if pair is None:
        print("\n❌ Fewer than two comparable chat models in the catalog")
        return False, None

    messages = [UserMessage(content="In two sentences, explain what a hash map is.")]
    max_tokens = 200
    print(
        "\n🎯 Same simple task on the cheapest plain chat model and the priciest one "
        f"listed at or under ${COMPARISON_PRICE_CEILING:.2f} per 1M tokens (blended):"
    )

    ok = True
    per_model_cost: dict[str, Decimal] = {}
    for model in pair:
        estimate = await client.chat.completions.estimate_cost(
            model=model,
            messages=messages,
            expected_completion_tokens=max_tokens,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        print(f"\n   {model} (estimate ${estimate.total_cost_usd:.6f})")
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                venice_parameters=OWN_PROMPT_ONLY,
            )
        except VeniceError as e:
            print(f"   ❌ {type(e).__name__}: {e}")
            ok = False
            continue
        ok = check_answer(response) and ok
        cost = await tracker.track(response)
        per_model_cost[model] = cost
        print(f"   Answer: {(response.text or '').strip()}")
        print(f"   Actual cost: ${cost:.6f}")

    if len(per_model_cost) == 2:
        cheap, pricey = (per_model_cost[m] for m in pair)
        ratio = f"{pricey / cheap:.1f}x" if cheap else "n/a"
        print(f"\n   📐 The pricier model cost {ratio} as much for the same task")
    else:
        ok = False

    print("\n   Other levers:")
    print("   ✅ Set max_completion_tokens to what the task needs")
    print("   ✅ Keep prompts short, and reuse prefixes so they hit the prompt cache")
    print("   ✅ Batch similar requests")
    return ok, pair


async def example_cost_analytics(
    client: VeniceClient, tracker: CostTracker, price_pair: tuple[str, str] | None
) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 4: Cost Analytics")
    print("=" * 60)

    # The cheapest direct-answer chat model by the resolver, plus the pricier model from
    # Pattern 3, so the breakdown compares two price points.
    resolved_model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    pricier = price_pair[1] if price_pair else None
    models = [resolved_model] + ([pricier] if pricier and pricier != resolved_model else [])

    scenarios = [
        ("summary", "Summarize the benefits of code review in three sentences.", 250),
        ("classification", "Is 'I love this product' positive or negative? One word.", 30),
    ]

    print("\n📈 Running each scenario on each model:\n")
    ok = True
    for model in models:
        for scenario, prompt, max_tokens in scenarios:
            response = await client.chat.completions.create(
                model=model,
                messages=[UserMessage(content=prompt)],
                max_completion_tokens=max_tokens,
                venice_parameters=OWN_PROMPT_ONLY,
            )
            print(f"{model} / {scenario}")
            ok = check_answer(response) and ok
            await tracker.track(response, metadata={"scenario": scenario})

    print("\n📊 Cost Breakdown by Model:")
    for model_id, cost in (await tracker.by_model()).items():
        print(f"   {model_id}: ${cost:.6f}")

    summary = await tracker.summary()
    print("\n📈 Overall Statistics:")
    print(f"   Requests: {summary.total_requests}")
    print(f"   Total cost: ${summary.total_cost_usd:.6f}")
    print(f"   Average per request: ${summary.average_cost_usd:.6f}")
    print(f"   Total tokens: {summary.total_tokens}")
    return ok and summary.total_requests == len(models) * len(scenarios)


async def example_budget_management(client: VeniceClient, tracker: CostTracker) -> bool:
    print("\n" + "=" * 60)
    print("Pattern 5: Budget Management — BudgetManager + can_afford()")
    print("=" * 60)

    model = await client.models.resolve_chat(prefer="cheapest", exclude_reasoning=True)
    messages = [UserMessage(content="Name one prime number. Reply with the number only.")]
    estimate = await client.chat.completions.estimate_cost(
        model=model,
        messages=messages,
        expected_completion_tokens=30,
        venice_parameters=OWN_PROMPT_ONLY,
    )

    # BudgetManager wraps an existing tracker; both daily and monthly caps are
    # optional but at least one is required. The daily cap here is 3.5 times
    # the per-request estimate so the demo reaches it. The estimate assumes the
    # whole completion cap is used, so a short reply costs less and more than
    # three requests usually fit.
    budget = BudgetManager(
        tracker=tracker,
        daily_usd=estimate.total_cost_usd * Decimal("3.5"),
        monthly_usd=Decimal("30.00"),
    )
    print(f"\n💵 Daily budget: ${budget.daily_usd:.6f} (3.5x the per-request estimate)")
    print(f"📅 Monthly budget: ${budget.monthly_usd}")
    print(f"Estimated cost per request: ${estimate.total_cost_usd:.6f}\n")

    completed = 0
    stopped_by_budget = False
    for i in range(20):
        if not await budget.can_afford(estimate.total_cost_usd):
            print(f"⚠️  Budget would be exceeded by request {i + 1}; stopping")
            stopped_by_budget = True
            break

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=30,
            venice_parameters=OWN_PROMPT_ONLY,
        )
        if not check_answer(response):
            return False
        cost = await tracker.track(response)
        remaining = await budget.remaining()
        completed += 1
        print(
            f"✅ Request {i + 1}: ${cost:.6f}, "
            f"${remaining.daily_remaining_usd:.6f} of the daily budget left"
        )

    print(f"\n   {completed} requests ran before the budget check stopped the loop")
    return stopped_by_budget and completed > 0


async def show_best_practices(client: VeniceClient) -> None:
    print("\n" + "=" * 60)
    print("Cost Management Best Practices")
    print("=" * 60)

    async def resolved(pick: Awaitable[str]) -> str:
        try:
            return await pick
        except NoMatchingModelError:
            return "(none available)"

    resolve = client.models.resolve_chat
    general = await resolved(resolve(prefer="cheapest"))
    direct = await resolved(resolve(prefer="cheapest", exclude_reasoning=True))
    tools = await resolved(resolve(prefer="cheapest", require_function_calling=True))
    reasoning = await resolved(resolve(prefer="cheapest", require_reasoning=True))
    code = await resolved(resolve(prefer="cheapest", require_code_optimization=True))
    vision = await resolved(resolve(prefer="cheapest", require_vision=True))

    practices = [
        (
            "Model Selection (cheapest match, resolved at runtime)",
            [
                f'✅ General chat: {general} (client.models.resolve_chat(prefer="cheapest"))',
                f"✅ Short direct answers: {direct} "
                "(exclude_reasoning=True skips models that spend tokens thinking)",
                f"✅ Function calling / tools: {tools}",
                f"✅ Reasoning-heavy tasks: {reasoning}",
                f"✅ Code generation: {code}",
                f"✅ Multimodal / vision: {vision}",
                "✅ Always resolve dynamically — never hardcode IDs",
            ],
        ),
        (
            "Token Management",
            [
                "✅ Set max_completion_tokens based on actual needs",
                "✅ Leave headroom for reasoning models, which spend tokens thinking",
                "✅ Optimize prompts to be concise",
                "✅ Reuse prompt prefixes; models with a cached rate bill cache hits for less",
            ],
        ),
        (
            "Monitoring",
            [
                "✅ Track every response with CostTracker.track()",
                "✅ For one-off calculations use venice_ai.costs.calculate_completion_cost",
                "✅ Reserve budget with estimate_cost(): an upper bound, not a forecast",
                "✅ Review costs regularly",
            ],
        ),
        (
            "Budget Control",
            [
                "✅ Set daily and monthly limits with BudgetManager",
                "✅ Check can_afford() before each request",
                "✅ Alert on unusual usage",
                "✅ Review and adjust budgets periodically",
            ],
        ),
    ]

    for category, items in practices:
        print(f"\n📋 {category}:")
        for item in items:
            print(f"   {item}")


async def main() -> int:
    print("=" * 60)
    print("Venice AI SDK - Production Cost Management")
    print("=" * 60)

    async with VeniceClient() as client:
        # CostTracker.from_client pre-populates the pricing map from
        # client.models.list(type="chat") so callers don't have to.
        tracker = await CostTracker.from_client(client)
        if not tracker.pricing_map:
            print("\n❌ Could not load chat-model pricing from the catalog.", file=sys.stderr)
            return 1

        # None marks a pattern skipped for a missing prerequisite.
        results: list[tuple[str, bool | None]] = []
        results.append(("Basic Cost Tracking", await example_basic_cost_tracking(client, tracker)))
        await tracker.reset()
        results.append(("Estimate vs Actual", await example_estimate_vs_actual(client, tracker)))
        await tracker.reset()
        optimization_ok, price_pair = await example_cost_optimization(client, tracker)
        results.append(("Cost Optimization", optimization_ok))
        await tracker.reset()
        results.append(
            ("Cost Analytics", await example_cost_analytics(client, tracker, price_pair))
        )
        await tracker.reset()
        results.append(("Budget Management", await example_budget_management(client, tracker)))
        await show_best_practices(client)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok is True)
    skipped = sum(1 for _, ok in results if ok is None)
    failed = len(results) - passed - skipped
    if failed == 0 and skipped == 0:
        print(f"✅ All {passed}/{len(results)} cost management patterns succeeded!")
    else:
        print(f"{'❌' if failed else '✅'} {passed} succeeded, {skipped} skipped, {failed} failed")
        for name, ok in results:
            print(f"   {'✓' if ok else '-' if ok is None else '✗'} {name}")
    print("=" * 60)

    print("\n🔑 Key Takeaways:")
    print("   1. Track token usage and costs for every request")
    print("   2. Choose models by price for the task at hand")
    print("   3. Set and monitor budgets (daily/monthly)")
    print("   4. Optimize prompts and max_completion_tokens settings")
    print("   5. Use cost analytics to identify optimization opportunities")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except NoMatchingModelError as e:
        # The catalog has no model of the kind this example needs.
        print(f"SKIPPED: {e}")
        sys.exit(77)
    except VeniceError as e:
        print(f"\n❌ {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
