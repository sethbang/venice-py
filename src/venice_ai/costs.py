"""
Cost Calculation and Estimation Utilities
=========================================

This module provides comprehensive cost calculation and estimation utilities for
Venice AI API usage. It supports real-time cost tracking, pre-request estimation,
and detailed breakdown of usage costs across different pricing models.

All calculations are based on actual token usage and current model pricing information.

Key Features:
    * **Real-time Cost Calculation**: Calculate actual costs from API responses
    * **Pre-request Estimation**: Estimate costs before making API calls
    * **Token-based Pricing**: Accurate cost calculation based on token consumption
    * **Model-specific Pricing**: Different pricing tiers for different models

Pricing Models:
    * **Input Tokens**: Cost for processing input text (prompts, messages)
    * **Output Tokens**: Cost for generating output text (completions, responses)
    * **Cache Reads / Writes**: Prompt tokens served from or written to the
      prompt cache, billed at ``cache_input`` / ``cache_write`` where the
      model publishes those rates
    * **Extended Context**: Long-context rates that apply to the whole request
      once the prompt exceeds the model's ``context_token_threshold``
    * **Flat Rate Models**: Simple per-request pricing for some operations

Cost Types:
    * **USD**: Traditional US Dollar pricing for enterprise billing
    * **DIEM**: The staking allowance. Each staked DIEM grants $1 of credit
      per epoch (resetting at 00:00 UTC), and requests are priced from the
      same USD price sheet, so a DIEM price equals the USD price

Example:
    >>> from venice_ai.costs import calculate_completion_cost, estimate_completion_cost
    >>>
    >>> # Calculate actual cost from a completion
    >>> completion = await client.chat.completions.create(...)
    >>> entry = await client.models.get(completion.model)
    >>> model_pricing = entry.model_spec.pricing
    >>> cost = calculate_completion_cost(completion, model_pricing)
    >>> print(f"Cost: ${cost['usd']:.6f} USD")
    >>>
    >>> # Estimate cost before making request
    >>> estimated_cost = estimate_completion_cost(
    ...     prompt="Your prompt here",
    ...     estimated_completion_tokens=500,
    ...     model_pricing=model_pricing
    ... )
    >>> print(f"Estimated: ${estimated_cost['usd']:.6f} USD")
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, Field

from .types.api.chat import ChatCompletionResponse as ChatCompletion
from .types.api.embeddings import EmbeddingsResponse
from .types.api.models import LLMModelPricing as ModelPricing
from .types.api.models import PricingTier

if TYPE_CHECKING:
    from ._client import VeniceClient


class ChatCostEstimate(BaseModel):
    """Pre-flight cost estimate for a chat completion request.

    Returned by :meth:`client.chat.completions.estimate_cost`. Message text
    is counted with the word heuristic (the same approach the
    :func:`estimate_completion_cost` helper uses). On top of it come an
    allowance for the chat template the model wraps the request in, an
    allowance for the ``tools`` and ``response_format`` (``schema_tokens``)
    and, unless the request opts out, an allowance for the system prompt
    Venice injects.

    Venice has no tokenize endpoint, so ``total_cost_usd`` is an estimate,
    not a guarantee. Every allowance is sized to be at least what the
    measured models bill, so for English prose the estimate is meant to err
    high; see :data:`TOOLS_TEMPLATE_TOKEN_ALLOWANCE` and
    :data:`SCHEMA_JSON_CHARS_PER_TOKEN` for how the schema allowance was
    derived and how far it can overstate. Text that tokenizes worse than
    ``tokens_per_word`` assumes (code, CJK) can still exceed the estimate.

    The estimate covers the request you send, not a system prompt built into
    a model's own chat template. The catalog does not publish one, and some
    models carry one: a measured model billed about 550 prompt tokens for a
    one-word message with the Venice system prompt off. To budget for such a
    model, send a one-word request once with ``max_completion_tokens=1`` and
    add the difference between its billed prompt tokens and its estimate.
    """

    model: str = Field(..., description="Model id used for the estimate")
    prompt_tokens: int = Field(
        ...,
        description=(
            "Estimated input token count, including template_overhead_tokens, "
            "schema_tokens and venice_system_prompt_tokens"
        ),
    )
    template_overhead_tokens: int = Field(
        default=0,
        description=(
            "Allowance for the tokens the model's chat template adds: "
            "CHAT_TEMPLATE_TOKEN_ALLOWANCE per request plus "
            "CHAT_MESSAGE_TOKEN_ALLOWANCE per message. An upper-bound constant, "
            "not an exact count."
        ),
    )
    schema_tokens: int = Field(
        default=0,
        description=(
            "Allowance for the tools and response_format: the compact JSON at "
            "SCHEMA_JSON_CHARS_PER_TOKEN characters per token, plus "
            "TOOLS_TEMPLATE_TOKEN_ALLOWANCE once when tools are sent. Sized to "
            "cover every measured chat template, so it can overstate a model "
            "whose template renders tools compactly."
        ),
    )
    venice_system_prompt_tokens: int = Field(
        default=0,
        description=(
            "Allowance for the Venice system prompt the server prepends when "
            "include_venice_system_prompt is not False (0 when opted out). A "
            "conservative constant, not an exact count."
        ),
    )
    expected_completion_tokens: int = Field(
        ..., description="Caller-provided completion token budget"
    )
    prompt_cost_usd: Decimal = Field(..., description="Estimated USD cost for prompt tokens")
    completion_cost_usd: Decimal = Field(
        ..., description="Estimated USD cost for completion tokens"
    )
    total_cost_usd: Decimal = Field(..., description="Sum of prompt + completion USD cost")


#: Prompt-token allowance added to pre-flight estimates for the system prompt
#: Venice prepends when ``venice_parameters.include_venice_system_prompt`` is
#: not ``False`` (the server default). The injected prompt's size varies by
#: model and over time (roughly 1100-1750 tokens has been observed), so this
#: sits at the top of that range: an estimate should err towards overstating
#: cost. It is an allowance, not an exact count.
VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE = 1750

#: Prompt-token allowance per request for the chat template a model wraps the
#: conversation in (start markers, the generation prompt). For one identical
#: 9-word prompt, models were billed between 3 and 16 tokens more than its
#: text, so this errs high. An allowance, not a count.
CHAT_TEMPLATE_TOKEN_ALLOWANCE = 20

#: Prompt-token allowance per message for its role header and separators.
CHAT_MESSAGE_TOKEN_ALLOWANCE = 4

#: Prompt-token allowance for the tool-calling instructions a chat template
#: adds once per request when ``tools`` are sent, on top of the tool JSON
#: itself (counted at :data:`SCHEMA_JSON_CHARS_PER_TOKEN`). Measured on
#: five chat models from five providers, Venice system prompt off: a 256-character tool definition added 43 to 270 prompt tokens, a
#: 1,410-character one 250 to 741, and both together 283 to 864. Splitting
#: each model's cost into a fixed part and a per-character part puts the
#: fixed part between 0 and about 170 tokens. This allowance sits above the
#: largest. Together with the per-character rate, the tools allowance came
#: out at least 1.2x the billed overhead on every model measured, and up to
#: about 7.8x for a small tool on the most compact template. An allowance,
#: not a count.
TOOLS_TEMPLATE_TOKEN_ALLOWANCE = 200

#: Characters of compact JSON per prompt token used to size the ``tools`` and
#: ``response_format`` allowance. In the same measurements, tool
#: JSON cost one token per 2.45 to 3.4 characters on open-weight chat
#: templates (JSON tokenizes worse than prose, and templates re-render it) and
#: less on a proprietary one. A ``json_schema`` response format added no
#: prompt tokens on three models, which enforce it while decoding, and one
#: token per 4.4 characters on a model that sends the schema in the prompt.
#: Two characters per token sits below every measured rate.
SCHEMA_JSON_CHARS_PER_TOKEN = 2

_MILLION = Decimal("1000000")
_ZERO = Decimal("0.00")


class _Rates(NamedTuple):
    """Per-million USD rates in effect for one request."""

    input: Decimal
    output: Decimal
    cache_input: Decimal
    cache_write: Decimal


def _usd(tier: PricingTier | None) -> Decimal | None:
    if tier is None or tier.usd is None:
        return None
    return Decimal(str(tier.usd))


def _rates_for(model_pricing: ModelPricing, prompt_tokens: int) -> _Rates:
    """Resolve the rates that bill a request with *prompt_tokens* prompt tokens.

    Standard rates apply unless the model publishes ``extended`` pricing and
    the total prompt (cached and cache-written tokens included) exceeds
    ``extended.context_token_threshold``; the extended rates then apply to
    the whole request. A missing cache rate falls back to the input rate of
    the same tier, and a missing extended input/output rate falls back to the
    standard one.
    """
    input_usd = _usd(model_pricing.input) or _ZERO
    output_usd = _usd(model_pricing.output) or _ZERO
    # Optional rates are read defensively so pricing-shaped objects that only
    # carry ``input`` / ``output`` still price as a standard-tier model.
    cache_input_usd = _usd(getattr(model_pricing, "cache_input", None))
    cache_write_usd = _usd(getattr(model_pricing, "cache_write", None))

    extended = getattr(model_pricing, "extended", None)
    threshold = extended.context_token_threshold if extended is not None else None
    if extended is not None and threshold is not None and prompt_tokens > threshold:
        extended_input = _usd(extended.input)
        extended_output = _usd(extended.output)
        input_usd = input_usd if extended_input is None else extended_input
        output_usd = output_usd if extended_output is None else extended_output
        cache_input_usd = _usd(extended.cache_input)
        cache_write_usd = _usd(extended.cache_write)

    return _Rates(
        input=input_usd,
        output=output_usd,
        cache_input=input_usd if cache_input_usd is None else cache_input_usd,
        cache_write=input_usd if cache_write_usd is None else cache_write_usd,
    )


def _token_count(*candidates: object) -> int:
    """First candidate that is a real positive ``int``; ``0`` if none is.

    Anything else (``None``, a bool, a float, a mock attribute) counts as
    absent, so a partially populated usage object never skews the bill.
    """
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 0


def _per_million(tokens: int, rate: Decimal) -> Decimal:
    return (Decimal(tokens) / _MILLION) * rate


def calculate_completion_cost(
    completion: ChatCompletion, model_pricing: ModelPricing | None
) -> dict[str, Decimal]:
    """
    Calculate the actual cost of a completed chat completion request.

    This function analyzes a ChatCompletion response and calculates the precise
    cost based on actual token usage reported by the API. It handles both input
    and output token pricing.

    The calculation uses the token usage data from the completion response and
    applies the current pricing structure for the specific model used. This
    provides accurate post-request cost tracking for billing and analytics.

    Token Cost Calculation:
        ``usage.prompt_tokens`` is the whole prompt, including tokens read
        from and written to the prompt cache. It is split into:

        * **Cache reads** (``prompt_tokens_details.cached_tokens``, or the
          top-level ``cache_read_input_tokens`` mirror) at ``cache_input``
        * **Cache writes** (``prompt_tokens_details.cache_creation_input_tokens``,
          or the top-level ``cache_creation_input_tokens`` mirror) at
          ``cache_write``
        * **Uncached remainder** at ``input``

        Completion tokens bill at ``output``. A model without a published
        ``cache_input`` / ``cache_write`` rate bills those tokens at ``input``.
        When the model publishes ``extended`` pricing and ``prompt_tokens``
        exceeds ``extended.context_token_threshold``, the extended rates
        apply to the entire request.

    Args:
        completion: The completed ChatCompletion response containing actual
                   token usage data from the API. Must include usage information
                   with prompt_tokens and completion_tokens counts.
        model_pricing: Current pricing information for the model that was used.
                      Contains input and output costs per million tokens. If None,
                      returns zero cost.

    Returns:
        Dictionary with cost breakdown containing:
        - 'usd': Total cost in US Dollars as a Decimal with exact precision

    Note:
        If the completion lacks usage data or model_pricing is None, the function
        returns zero cost rather than raising an exception to maintain robust
        operation in production environments.

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import calculate_completion_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Create a chat completion
        >>> completion = await client.chat.completions.create(
        ...     model=await client.models.resolve_chat(),
        ...     messages=[{"role": "user", "content": "Hello world!"}]
        ... )
        >>>
        >>> # Get current model pricing
        >>> entry = await client.models.get(completion.model)
        >>> model_pricing = entry.model_spec.pricing
        >>>
        >>> # Calculate actual costs
        >>> costs = calculate_completion_cost(completion, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD")
        >>> print(f"Tokens: {completion.usage.total_tokens} total")
    """
    if not model_pricing or not hasattr(completion, "usage") or not completion.usage:
        return {"usd": Decimal("0.00")}

    usage = completion.usage
    prompt_tokens = _token_count(usage.prompt_tokens)
    completion_tokens = _token_count(usage.completion_tokens)

    # The nested breakdown and the top-level fields mirror the same counts;
    # take one of them, never the sum.
    details = getattr(usage, "prompt_tokens_details", None)
    cached = _token_count(
        getattr(details, "cached_tokens", None),
        getattr(usage, "cache_read_input_tokens", None),
    )
    written = _token_count(
        getattr(details, "cache_creation_input_tokens", None),
        getattr(usage, "cache_creation_input_tokens", None),
    )
    cached = min(cached, prompt_tokens)
    written = min(written, prompt_tokens - cached)
    uncached = prompt_tokens - cached - written

    rates = _rates_for(model_pricing, prompt_tokens)
    usd_cost = (
        _per_million(uncached, rates.input)
        + _per_million(cached, rates.cache_input)
        + _per_million(written, rates.cache_write)
        + _per_million(completion_tokens, rates.output)
    )
    return {"usd": usd_cost}


def calculate_embedding_cost(
    embedding_response: Any, model_pricing: ModelPricing | None
) -> dict[str, Decimal]:
    """
    Calculate the actual cost of a completed embedding request.

    This function analyzes an embedding response and calculates the cost based on
    the total tokens processed during the embedding generation. Unlike chat
    completions, embeddings typically use only input token pricing since they
    don't generate variable-length outputs.

    Embedding Cost Calculation:
        * **Input Processing**: total_tokens × input_cost_per_million_tokens
        * **Fixed Output**: Embeddings have fixed output dimensions
        * **Total Cost**: Primarily based on input token processing

    Args:
        embedding_response: The completed embedding response containing usage
                           data. Must include a usage object with total_tokens
                           count from the embedding operation.
        model_pricing: Current pricing information for the embedding model.
                      Contains input costs per million tokens. If None,
                      returns zero cost.

    Returns:
        Dictionary with cost breakdown containing:
        - 'usd': Total cost in US Dollars as a Decimal with exact precision

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import calculate_embedding_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Create embeddings
        >>> response = await client.embeddings.create(
        ...     model=await client.models.resolve_embedding(),
        ...     input="Hello, world! This is a sample text."
        ... )
        >>>
        >>> # Get current model pricing
        >>> entry = await client.models.get(response.model)
        >>> model_pricing = entry.model_spec.pricing
        >>>
        >>> # Calculate actual costs
        >>> costs = calculate_embedding_cost(response, model_pricing)
        >>> print(f"Cost: ${costs['usd']:.6f} USD")
        >>> print(f"Tokens processed: {response.usage.total_tokens}")
    """
    if (
        not model_pricing
        or not hasattr(embedding_response, "usage")
        or not embedding_response.usage
    ):
        return {"usd": Decimal("0.00")}

    total_tokens = embedding_response.usage.total_tokens

    # New pricing structure uses nested PricingTier objects
    # Convert to Decimal for exact monetary calculations
    input_usd = Decimal(str(model_pricing.input.usd or 0.0))

    # Calculate USD cost with exact decimal precision
    usd_cost = (Decimal(str(total_tokens)) / Decimal("1000000")) * input_usd

    return {"usd": usd_cost}


def estimate_completion_cost(
    prompt: str,
    estimated_completion_tokens: int,
    model_pricing: ModelPricing | None,
    tokens_per_word: float = 1.3,
    *,
    include_venice_system_prompt: bool = True,
    template_overhead_tokens: int = CHAT_TEMPLATE_TOKEN_ALLOWANCE + CHAT_MESSAGE_TOKEN_ALLOWANCE,
) -> dict[str, Decimal]:
    """
    Estimate the cost of a chat completion before making the API request.

    This function provides pre-request cost estimation based on prompt analysis
    and expected completion length. It uses heuristic token counting to estimate
    input costs and user-provided estimates for output costs, enabling budget
    planning and cost-aware request optimization.

    The estimation is particularly useful for:
    * Budget planning and cost control
    * Optimizing prompts for cost efficiency
    * Batch processing cost estimation
    * User-facing cost previews

    Estimation Methodology:
        * **Input Tokens**: Estimated from word count using configurable ratio,
          plus ``template_overhead_tokens`` for the chat template the model
          wraps the prompt in (one request with one message by default)
        * **Venice System Prompt**: Unless ``include_venice_system_prompt`` is
          ``False``, :data:`VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE` tokens are
          added for the system prompt Venice prepends. They are priced at
          ``cache_input`` (the injected prompt is served from the prompt
          cache), falling back to ``input``
        * **Output Tokens**: User-provided estimate based on expected response length
        * **Pricing**: Applied using current model pricing structure, including
          ``extended`` rates when the estimated prompt exceeds the model's
          long-context threshold
        * **Accuracy**: Approximation only - actual costs may vary

    Args:
        prompt: The input text to estimate token costs for. This is analyzed
               for word count and converted to estimated tokens using the
               tokens_per_word ratio.
        estimated_completion_tokens: Expected number of tokens in the model's
                                   response. This should be estimated based on
                                   the desired response length and complexity.
        model_pricing: Current pricing information for the target model.
                      Contains input and output costs per million tokens.
                      If None, returns zero cost.
        tokens_per_word: Conversion ratio from words to tokens. Default of 1.3
                        is optimized for English text. Adjust for other contexts:
                        - English text: ~1.3 tokens/word (default)
                        - Japanese/Chinese: ~2.0 tokens/word
                        - Code/technical: ~1.5-2.0 tokens/word
                        - Mixed content: Adjust based on composition
        include_venice_system_prompt: Whether the request will carry the Venice
                        system prompt. Pass ``False`` when the request sets
                        ``venice_parameters.include_venice_system_prompt=False``;
                        the default matches the server default (``True``).
        template_overhead_tokens: Allowance for the chat template, billed at the
                        input rate. Defaults to one request with one message
                        (:data:`CHAT_TEMPLATE_TOKEN_ALLOWANCE` +
                        :data:`CHAT_MESSAGE_TOKEN_ALLOWANCE`).

    Returns:
        Dictionary with estimated cost breakdown containing:
        - 'usd': Estimated total cost in US Dollars as a Decimal with exact precision

    Accuracy Notes:
        * Token estimation is heuristic and may not match exact tokenization
        * Actual costs depend on precise tokenizer behavior
        * Output token count is user-estimated and may vary significantly
        * Different models may have different tokenization patterns

    Example:
        >>> from venice_ai import VeniceClient
        >>> from venice_ai.costs import estimate_completion_cost
        >>>
        >>> client = VeniceClient(api_key="your-api-key")
        >>>
        >>> # Get current model pricing for the model you plan to call
        >>> model_id = await client.models.resolve_chat()
        >>> model_pricing = (await client.models.get(model_id)).model_spec.pricing
        >>>
        >>> # Estimate costs for different scenarios
        >>> prompt = "Write a detailed explanation of quantum computing"
        >>>
        >>> # Short response estimate
        >>> short_cost = estimate_completion_cost(
        ...     prompt=prompt,
        ...     estimated_completion_tokens=200,
        ...     model_pricing=model_pricing
        ... )
        >>>
        >>> # Long response estimate
        >>> long_cost = estimate_completion_cost(
        ...     prompt=prompt,
        ...     estimated_completion_tokens=1000,
        ...     model_pricing=model_pricing
        ... )
        >>>
        >>> print(f"Short response: ${short_cost['usd']:.6f} USD")
        >>> print(f"Long response: ${long_cost['usd']:.6f} USD")
        >>> print(f"Cost difference: ${long_cost['usd'] - short_cost['usd']:.6f} USD")
    """
    if not model_pricing:
        return {"usd": Decimal("0.00")}

    prompt_tokens = int(len(prompt.split()) * tokens_per_word) + template_overhead_tokens
    system_prompt_tokens = (
        VENICE_SYSTEM_PROMPT_TOKEN_ALLOWANCE if include_venice_system_prompt else 0
    )
    prompt_cost, completion_cost = _estimate_costs(
        model_pricing,
        prompt_tokens=prompt_tokens,
        system_prompt_tokens=system_prompt_tokens,
        completion_tokens=estimated_completion_tokens,
    )
    return {"usd": prompt_cost + completion_cost}


def _estimate_costs(
    model_pricing: ModelPricing,
    *,
    prompt_tokens: int,
    system_prompt_tokens: int,
    completion_tokens: int,
) -> tuple[Decimal, Decimal]:
    """Price a pre-flight estimate as ``(prompt_cost, completion_cost)``.

    *prompt_tokens* are the caller's own tokens, billed at ``input``;
    *system_prompt_tokens* is the Venice system-prompt allowance, billed at
    ``cache_input`` (falling back to ``input``). The rate tier is chosen from
    their sum, the same way the server chooses it from the billed prompt.
    """
    rates = _rates_for(model_pricing, prompt_tokens + system_prompt_tokens)
    prompt_cost = _per_million(prompt_tokens, rates.input) + _per_million(
        system_prompt_tokens, rates.cache_input
    )
    return prompt_cost, _per_million(completion_tokens, rates.output)


# ---------------------------------------------------------------------------
# Stateful cost tracking — CostTracker / BudgetManager
# ---------------------------------------------------------------------------


class CostRecord(BaseModel):
    """One per-request cost-tracking entry."""

    timestamp: datetime = Field(..., description="UTC timestamp when the request was tracked")
    model: str = Field(..., description="Model id used for the request")
    prompt_tokens: int = Field(..., description="Input/prompt token count from the response")
    completion_tokens: int = Field(
        ..., description="Output/completion token count (0 for embeddings)"
    )
    total_tokens: int = Field(..., description="Total tokens billed for the request")
    cost_usd: Decimal = Field(..., description="Computed cost in USD")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Free-form metadata supplied by the caller"
    )


class CostSummary(BaseModel):
    """Aggregate stats produced by :meth:`CostTracker.summary`."""

    total_requests: int
    total_cost_usd: Decimal
    total_tokens: int
    average_cost_usd: Decimal = Field(
        ..., description="Mean USD cost per tracked request (0 when none)"
    )
    average_tokens: float = Field(
        ..., description="Mean tokens per tracked request (0.0 when none)"
    )


class BudgetRemaining(BaseModel):
    """Remaining-budget snapshot returned by :meth:`BudgetManager.remaining`."""

    daily_remaining_usd: Decimal | None = Field(
        default=None, description="USD remaining against the daily cap (None if no daily cap)"
    )
    daily_used_pct: float | None = Field(
        default=None, description="Daily-cap usage as a 0–100 percentage (None if no daily cap)"
    )
    monthly_remaining_usd: Decimal | None = Field(
        default=None, description="USD remaining against the monthly cap (None if no monthly cap)"
    )
    monthly_used_pct: float | None = Field(
        default=None, description="Monthly-cap usage as a 0–100 percentage (None if no monthly cap)"
    )


class CostTracker:
    """Stateful, async-safe accumulator for per-request API costs.

    Wraps the existing :func:`calculate_completion_cost` and
    :func:`calculate_embedding_cost` helpers. Three integration paths:

    * **Manual** — call :meth:`track` on each response yourself.
    * **Wired-on-client** — pass to ``VeniceClient(cost_tracker=tracker)``;
      the SDK calls :meth:`track` automatically on every chat / embeddings
      response.
    * **From-client factory** — :meth:`from_client` builds a tracker
      pre-populated with the live pricing map.

    All mutating operations take a single :class:`asyncio.Lock` so concurrent
    in-flight requests can update state safely.
    """

    def __init__(self, pricing_map: dict[str, ModelPricing] | None = None) -> None:
        """:param pricing_map: ``{model_id: LLMModelPricing}``. Models absent
        from the map produce zero-cost records (the underlying helpers
        gracefully handle missing pricing)."""
        self.pricing_map: dict[str, ModelPricing] = dict(pricing_map or {})
        self.requests: list[CostRecord] = []
        self.total_cost_usd: Decimal = Decimal("0.00")
        self.total_tokens: int = 0
        self._lock = asyncio.Lock()

    @classmethod
    async def from_client(cls, client: VeniceClient) -> CostTracker:
        """Build a tracker pre-populated with the live chat-pricing map."""
        catalog = await client.models.list(type="chat")
        pricing_map: dict[str, ModelPricing] = {}
        for entry in catalog.data:
            spec = entry.model_spec
            if spec and spec.pricing and isinstance(spec.pricing, ModelPricing):
                pricing_map[entry.id] = spec.pricing
        return cls(pricing_map=pricing_map)

    async def track(
        self,
        response: ChatCompletion | EmbeddingsResponse,
        *,
        model: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Decimal:
        """Record one response and return its USD cost.

        :param response: A :class:`ChatCompletionResponse` or
            :class:`EmbeddingsResponse`.
        :param model: Override the model id used to look up pricing. Defaults
            to ``response.model``.
        :param metadata: Free-form metadata stored on the resulting
            :class:`CostRecord`.
        :raises TypeError: For unsupported response types.
        """
        if isinstance(response, ChatCompletion):
            model_id = model or response.model
            pricing = self.pricing_map.get(model_id)
            cost = calculate_completion_cost(response, pricing)["usd"]
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
        elif isinstance(response, EmbeddingsResponse):
            model_id = model or response.model
            pricing = self.pricing_map.get(model_id)
            cost = calculate_embedding_cost(response, pricing)["usd"]
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = 0
            total_tokens = response.usage.total_tokens
        else:
            raise TypeError(
                f"CostTracker.track() does not support {type(response).__name__}; "
                f"expected ChatCompletionResponse or EmbeddingsResponse."
            )

        record = CostRecord(
            timestamp=datetime.now(UTC),
            model=model_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            metadata=dict(metadata or {}),
        )
        async with self._lock:
            self.requests.append(record)
            self.total_cost_usd += cost
            self.total_tokens += total_tokens
        return cost

    async def summary(self) -> CostSummary:
        """Aggregate stats across all tracked requests."""
        async with self._lock:
            n = len(self.requests)
            if n == 0:
                return CostSummary(
                    total_requests=0,
                    total_cost_usd=Decimal("0.00"),
                    total_tokens=0,
                    average_cost_usd=Decimal("0.00"),
                    average_tokens=0.0,
                )
            return CostSummary(
                total_requests=n,
                total_cost_usd=self.total_cost_usd,
                total_tokens=self.total_tokens,
                average_cost_usd=self.total_cost_usd / Decimal(n),
                average_tokens=self.total_tokens / n,
            )

    async def by_model(self) -> dict[str, Decimal]:
        """USD cost grouped by model id."""
        async with self._lock:
            costs: dict[str, Decimal] = {}
            for rec in self.requests:
                costs[rec.model] = costs.get(rec.model, Decimal("0.00")) + rec.cost_usd
            return costs

    async def reset(self) -> None:
        """Clear all tracked state."""
        async with self._lock:
            self.requests.clear()
            self.total_cost_usd = Decimal("0.00")
            self.total_tokens = 0


class BudgetManager:
    """Daily / monthly USD-cap enforcement layered on a :class:`CostTracker`.

    Either ``daily_usd`` or ``monthly_usd`` may be ``None`` to disable that
    cap. The tracker is shared, not owned — ``BudgetManager`` does not call
    :meth:`CostTracker.reset`; callers manage rollover themselves.
    """

    def __init__(
        self,
        *,
        tracker: CostTracker,
        daily_usd: Decimal | None = None,
        monthly_usd: Decimal | None = None,
    ) -> None:
        if daily_usd is None and monthly_usd is None:
            raise ValueError("BudgetManager needs at least one of daily_usd or monthly_usd")
        self.tracker = tracker
        self.daily_usd = daily_usd
        self.monthly_usd = monthly_usd

    async def can_afford(self, estimated_cost_usd: Decimal) -> bool:
        """``True`` if adding *estimated_cost_usd* keeps both caps satisfied."""
        summary = await self.tracker.summary()
        projected = summary.total_cost_usd + estimated_cost_usd
        return not (
            (self.daily_usd is not None and projected > self.daily_usd)
            or (self.monthly_usd is not None and projected > self.monthly_usd)
        )

    async def remaining(self) -> BudgetRemaining:
        """Snapshot of remaining headroom and usage percentages."""
        summary = await self.tracker.summary()
        spent = summary.total_cost_usd

        daily_remaining: Decimal | None = None
        daily_pct: float | None = None
        if self.daily_usd is not None:
            daily_remaining = max(self.daily_usd - spent, Decimal("0"))
            daily_pct = float(spent / self.daily_usd * 100) if self.daily_usd > 0 else 0.0

        monthly_remaining: Decimal | None = None
        monthly_pct: float | None = None
        if self.monthly_usd is not None:
            monthly_remaining = max(self.monthly_usd - spent, Decimal("0"))
            monthly_pct = float(spent / self.monthly_usd * 100) if self.monthly_usd > 0 else 0.0

        return BudgetRemaining(
            daily_remaining_usd=daily_remaining,
            daily_used_pct=daily_pct,
            monthly_remaining_usd=monthly_remaining,
            monthly_used_pct=monthly_pct,
        )


async def _maybe_track_response(tracker: CostTracker, response: Any) -> None:
    """Internal hook: feed a parsed response into *tracker* if it's a tracked type.

    Called from :class:`VeniceClient._request` when a ``cost_tracker`` is wired
    on the client. Silently ignores untracked response types and swallows any
    tracking-side exception so an observability bug never masks a successful
    request.
    """
    if not isinstance(response, ChatCompletion | EmbeddingsResponse):
        return
    try:
        await tracker.track(response)
    except Exception:  # noqa: BLE001 — observability must never break the request
        import logging

        logging.getLogger(__name__).warning("cost_tracker.track() raised; ignoring", exc_info=True)
