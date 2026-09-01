"""
Adaptive rate-limiter saturation benchmark.

Two scenarios:
  (a) Single-model: saturate the cheapest text model in each quartile
      (XS/S/M/L) of the combined input+output USD-per-million price ladder.
  (b) Multi-model isolation: heavy load on the L-quartile model, light load
      on the XS-quartile model in parallel - the XS p50 must hold within
      50 percent of its solo baseline.

Budget cap: $5 (input + output tokens, all runs).

Run with::

    VENICE_API_KEY=...
    VENICE_REDIS_URL=redis://localhost:6379/0
    poetry run pytest tests/benchmarks/test_adaptive_saturation.py \\
        -m benchmark -s --no-cov

Outputs:
  ``docs/superpowers/audits/saturation-results.md`` is rewritten with the
  measured table.

Why "quartile" instead of an L/M/S/XS field on the model spec?
  ``ModelSpec`` (and its ``TextModelSpec`` subclass) does not expose a
  size-class field. RPM/TPM live only in the response headers, not in
  ``/v1/models``. So we approximate "size class" by sorting on the price
  ladder (cheap -> expensive == small -> large) and bucketing by quartile,
  then discover the per-model RPM/TPM from a one-shot probe call.

Why ``VeniceClientFactory`` instead of ``VeniceClient(rate_limiter=...)``?
  ``VeniceClient.__init__`` takes a ``RateLimiterProtocol`` *instance*, not
  a ``RateLimiterConfig`` dataclass. The supported wiring path for
  ADAPTIVE mode is::

    config = VeniceAIConfig(
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE, redis_url=...,
        ),
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(redis_url=...),
        ),
    )
    client = VeniceClientFactory.create_client(config, api_key=...)

  which is what this benchmark uses.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from venice_ai import UserMessage
from venice_ai.core.config import VeniceAIConfig
from venice_ai.core.config.backends import (
    BackendConfig,
    BackendType,
    RedisBackendConfig,
)
from venice_ai.factory import VeniceClientFactory
from venice_ai.rate_limiting import RateLimiterConfig, RateLimiterMode
from venice_ai.types.api.models import LLMModelPricing, TextModelSpec

RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "superpowers"
    / "audits"
    / "saturation-results.md"
)
SIZE_CLASSES = ("L", "M", "S", "XS")
DURATION_S = 60.0
LIGHT_LOAD_RPS = 1.0
HEAVY_DRIVE_FACTOR = 1.10  # drive 10% above advertised RPM to force saturation
TOKEN_BUDGET_USD = 5.00
PROMPT = "Reply with a single token: ok"
ESTIMATED_PROMPT_TOKENS = 8  # PROMPT tokenizes to ~6-8 tokens for most BPE tokenizers
ESTIMATED_OUTPUT_TOKENS = 4  # max_completion_tokens=4 cap below
MAX_OUTPUT_TOKENS = 4


@dataclass
class ModelChoice:
    """Bundle of selection + saturation parameters for one chosen model."""

    size: str
    id: str
    input_per_m_usd: float  # USD per million input tokens
    output_per_m_usd: float  # USD per million output tokens
    rpm: int  # discovered from response headers
    tpm: int  # discovered from response headers (0 if header missing)


@dataclass
class RunResult:
    """Aggregated metrics for one saturation run."""

    model: ModelChoice
    advertised_rpm: int
    achieved_rps: float
    error_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    spent_usd: float
    latencies_ms: list[float] = field(default_factory=list)


def _llm_combined_price_per_million(spec: TextModelSpec) -> float | None:
    """Return ``input_usd + output_usd`` (per million tokens) or ``None``.

    Returns ``None`` when pricing is missing or not the LLM shape, so the
    caller can skip the model rather than guess a price.
    """
    pricing = spec.pricing
    if not isinstance(pricing, LLMModelPricing):
        return None
    return float(pricing.input.usd) + float(pricing.output.usd)


async def _probe_rate_limits(client, model_id: str) -> tuple[int, int]:
    """Send one chat call and read RPM/TPM out of response headers.

    Returns ``(rpm, tpm)`` with ``tpm=0`` when the API doesn't expose a
    token budget. ``rpm=0`` means the probe failed - caller should treat
    the model as unusable.
    """
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[UserMessage(content=PROMPT)],
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception:
        return 0, 0
    info = response.response_rate_limits
    if info is None:
        return 0, 0
    rpm = int(info.limit_requests or 0)
    tpm = int(info.limit_tokens or 0)
    return rpm, tpm


async def select_models_by_price_quartile(client) -> dict[str, ModelChoice]:
    """Pick one text model per price-quartile bucket (XS/S/M/L).

    1. List ``type='text'`` models, keep only those with LLM pricing.
    2. Sort ascending by ``input_usd + output_usd`` (per-million).
    3. Quartile-split: cheapest 25 percent -> XS, then S, M, L (most
       expensive).
    4. Within each bucket pick the cheapest member.
    5. Probe each pick for RPM/TPM via response headers.
    """
    listing = await client.models.list(type="text")
    priced: list[tuple[float, str, TextModelSpec]] = []
    for model in listing.data:
        spec = model.model_spec
        if not isinstance(spec, TextModelSpec):
            continue
        price = _llm_combined_price_per_million(spec)
        if price is None or price <= 0:
            continue
        priced.append((price, str(model.id), spec))

    if len(priced) < len(SIZE_CLASSES):
        raise RuntimeError(f"Need >= {len(SIZE_CLASSES)} priced text models; found {len(priced)}.")

    priced.sort(key=lambda triple: triple[0])
    n = len(priced)
    # Cheapest -> XS, most expensive -> L.
    bucket_order = ("XS", "S", "M", "L")
    edges = [int(round(n * (i + 1) / 4)) for i in range(4)]

    out: dict[str, ModelChoice] = {}
    start = 0
    for bucket, end in zip(bucket_order, edges, strict=True):
        end = max(end, start + 1)
        # Cheapest in the bucket.
        price, model_id, spec = priced[start:end][0]
        pricing = spec.pricing
        assert isinstance(pricing, LLMModelPricing)  # narrowed above
        rpm, tpm = await _probe_rate_limits(client, model_id)
        if rpm <= 0:
            # Fall back to the next entry if the probe failed.
            for alt in priced[start + 1 : end]:
                rpm, tpm = await _probe_rate_limits(client, alt[1])
                if rpm > 0:
                    price, model_id, spec = alt
                    pricing = spec.pricing
                    assert isinstance(pricing, LLMModelPricing)
                    break
            else:
                raise RuntimeError(f"Could not discover RPM for any model in bucket {bucket!r}.")
        out[bucket] = ModelChoice(
            size=bucket,
            id=model_id,
            input_per_m_usd=float(pricing.input.usd),
            output_per_m_usd=float(pricing.output.usd),
            rpm=rpm,
            tpm=tpm,
        )
        start = end

    missing = [s for s in SIZE_CLASSES if s not in out]
    if missing:
        raise RuntimeError(f"No models found for size classes: {missing!r}")
    return out


async def saturate_single(
    client,
    choice: ModelChoice,
    duration_s: float = DURATION_S,
) -> RunResult:
    """Drive ``choice.id`` at 110 percent of advertised RPM for ``duration_s``."""
    target_rps = (choice.rpm / 60.0) * HEAVY_DRIVE_FACTOR
    start = time.monotonic()
    end_at = start + duration_s
    latencies: list[float] = []
    errors = 0
    requests = 0
    sem = asyncio.Semaphore(int(target_rps) + 4)

    async def one_call() -> None:
        nonlocal errors, requests
        async with sem:
            t0 = time.monotonic()
            try:
                await client.chat.completions.create(
                    model=choice.id,
                    messages=[UserMessage(content=PROMPT)],
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception:
                errors += 1
            else:
                latencies.append((time.monotonic() - t0) * 1000)
            requests += 1

    interval = 1.0 / target_rps if target_rps > 0 else 1.0
    pending: set[asyncio.Task[None]] = set()
    next_at = start
    while time.monotonic() < end_at:
        now = time.monotonic()
        if now >= next_at:
            pending.add(asyncio.create_task(one_call()))
            next_at += interval
        done = {t for t in pending if t.done()}
        pending -= done
        await asyncio.sleep(0.001)
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    elapsed = time.monotonic() - start
    achieved = (requests - errors) / elapsed if elapsed else 0.0
    err_rate = errors / requests if requests else 0.0
    spent = (
        choice.input_per_m_usd * requests * ESTIMATED_PROMPT_TOKENS / 1_000_000
        + choice.output_per_m_usd * requests * ESTIMATED_OUTPUT_TOKENS / 1_000_000
    )
    return RunResult(
        model=choice,
        advertised_rpm=choice.rpm,
        achieved_rps=achieved,
        error_rate=err_rate,
        p50_ms=statistics.median(latencies) if latencies else 0.0,
        p95_ms=(statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else 0.0),
        p99_ms=(statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else 0.0),
        spent_usd=spent,
        latencies_ms=latencies,
    )


async def isolation_run(
    client,
    heavy: ModelChoice,
    light: ModelChoice,
    duration_s: float = DURATION_S,
) -> tuple[RunResult, list[float], list[float]]:
    """Compare light-model latency alone vs. while heavy-model saturates.

    Returns ``(heavy_result, control_latencies_ms, contended_latencies_ms)``.

    Step A: control - run light alone for ``duration_s/2`` and record latencies.
    Step B: contention - run heavy + light concurrently for ``duration_s/2``
            and record the light latencies under contention.
    """
    half = duration_s / 2

    async def light_only(t_end: float, lat: list[float]) -> None:
        while time.monotonic() < t_end:
            t0 = time.monotonic()
            try:
                await client.chat.completions.create(
                    model=light.id,
                    messages=[UserMessage(content=PROMPT)],
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                )
            except Exception:
                pass
            else:
                lat.append((time.monotonic() - t0) * 1000)
            await asyncio.sleep(1.0 / LIGHT_LOAD_RPS)

    control: list[float] = []
    end_a = time.monotonic() + half
    await light_only(end_a, control)

    contended: list[float] = []
    end_b = time.monotonic() + half
    heavy_task = asyncio.create_task(saturate_single(client, heavy, duration_s=half))
    await light_only(end_b, contended)
    heavy_result = await heavy_task

    return heavy_result, control, contended


def write_saturation_report(
    singles: dict[str, RunResult],
    choices: dict[str, ModelChoice],
    ctl_p50: float,
    cnt_p50: float,
    delta_pct: float,
    iso_pass: bool,
    total_spent: float,
) -> None:
    """Render results into ``docs/superpowers/audits/saturation-results.md``."""
    rows_models = "\n".join(
        f"| {s} | `{c.id}` | {c.input_per_m_usd:.2f} | {c.output_per_m_usd:.2f} | "
        f"{c.rpm} | {c.tpm} |"
        for s, c in choices.items()
    )
    rows_single_lines: list[str] = []
    for size, r in singles.items():
        sat_pct = (r.achieved_rps * 60 / r.advertised_rpm * 100) if r.advertised_rpm else 0.0
        rows_single_lines.append(
            f"| {size} | `{r.model.id}` | {r.advertised_rpm} | {r.achieved_rps:.2f} | "
            f"{sat_pct:.1f}% | {r.error_rate:.2%} | "
            f"{r.p50_ms:.0f} | {r.p95_ms:.0f} | {r.p99_ms:.0f} |"
        )
    rows_single = "\n".join(rows_single_lines)
    body = (
        "# Adaptive Rate-Limiter Saturation Results\n\n"
        f"Total spend: ${total_spent:.2f} (budget cap ${TOKEN_BUDGET_USD:.2f})\n\n"
        "## Model selection\n\n"
        "| Bucket | Cheapest model | Input $/M | Output $/M | RPM | TPM |\n"
        "|--------|----------------|----------:|-----------:|----:|----:|\n"
        f"{rows_models}\n\n"
        "## Single-model saturation\n\n"
        "| Size | Model | Adv. RPM | Achieved RPS | Sat % | Err | p50 | p95 | p99 |\n"
        "|------|-------|---------:|-------------:|------:|----:|----:|----:|----:|\n"
        f"{rows_single}\n\n"
        "## Multi-model isolation\n\n"
        "| Heavy | Light | Light p50 (control) | Light p50 (contended) | Δ % | Pass? |\n"
        "|-------|-------|--------------------:|----------------------:|----:|------:|\n"
        f"| `{choices['L'].id}` | `{choices['XS'].id}` | {ctl_p50:.0f} ms | "
        f"{cnt_p50:.0f} ms | {delta_pct:+.1f}% | {'YES' if iso_pass else 'NO'} |\n"
    )
    # The audits directory is not tracked, so it may not exist on a fresh
    # checkout. Create it rather than losing a completed live run — this
    # write happens after every request has already been paid for.
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(body, encoding="utf-8")


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_saturation_full_run() -> None:
    """End-to-end saturation benchmark.

    Writes results into ``docs/superpowers/audits/saturation-results.md``.

    Manual run only - requires ``VENICE_API_KEY`` and ``VENICE_REDIS_URL``
    (the adaptive rate limiter uses Redis for cross-process coordination).
    """
    pytest.importorskip("adaptive_rate_limiter")
    redis_url = os.environ.get("VENICE_REDIS_URL")
    if not redis_url:
        pytest.skip(
            "VENICE_REDIS_URL not set; ADAPTIVE rate limiter requires Redis. "
            "Set VENICE_REDIS_URL=redis://localhost:6379/0 to run."
        )
    if not os.environ.get("VENICE_API_KEY"):
        pytest.skip("VENICE_API_KEY not set; live API calls are required.")

    # backend_type must be set explicitly: it defaults to MEMORY, and a
    # rate_limiter redis_url on its own does not change it. Without this the
    # run silently uses the in-process backend and measures nothing about the
    # cross-process coordination the adaptive limiter exists to provide.
    config = VeniceAIConfig(
        rate_limiter=RateLimiterConfig(
            mode=RateLimiterMode.ADAPTIVE,
            redis_url=redis_url,
        ),
        backend=BackendConfig(
            backend_type=BackendType.REDIS,
            redis=RedisBackendConfig(redis_url=redis_url),
        ),
    )
    client = VeniceClientFactory.create_client(config)
    try:
        choices = await select_models_by_price_quartile(client)
        single_results: dict[str, RunResult] = {}
        spent = 0.0
        for size in SIZE_CLASSES:
            result = await saturate_single(client, choices[size])
            single_results[size] = result
            spent += result.spent_usd
            assert spent < TOKEN_BUDGET_USD, f"budget exceeded after bucket {size}: ${spent:.2f}"

        heavy = choices["L"]
        light = choices["XS"]
        heavy_res, ctl, cnt = await isolation_run(client, heavy, light)
        spent += heavy_res.spent_usd
        ctl_p50 = statistics.median(ctl) if ctl else 0.0
        cnt_p50 = statistics.median(cnt) if cnt else 0.0
        delta_pct = ((cnt_p50 - ctl_p50) / ctl_p50 * 100) if ctl_p50 else 0.0
        # Pass criterion: contended p50 within 50 percent of control p50.
        isolation_pass = delta_pct < 50.0
    finally:
        await client.close()

    write_saturation_report(
        single_results, choices, ctl_p50, cnt_p50, delta_pct, isolation_pass, spent
    )

    # Asserted after the report is written so a failing run still leaves its
    # diagnostics on disk.
    #
    # The emptiness checks are not incidental: with no control samples ctl_p50
    # is 0.0, which pins delta_pct to 0.0 and makes isolation_pass true. A run
    # that measured nothing has to fail rather than report success.
    assert ctl, "isolation run collected no control latencies; nothing was measured"
    assert cnt, "isolation run collected no contended latencies; nothing was measured"
    assert isolation_pass, (
        f"adaptive limiter did not isolate the light model from the heavy one: "
        f"contended p50 {cnt_p50:.3f}s vs control p50 {ctl_p50:.3f}s "
        f"({delta_pct:+.1f}%, criterion <50%)"
    )
