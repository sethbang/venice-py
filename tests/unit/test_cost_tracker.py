"""Unit tests for CostTracker, BudgetManager, and the auto-track hook."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from venice_ai.costs import (
    BudgetManager,
    BudgetRemaining,
    CostRecord,
    CostSummary,
    CostTracker,
    _maybe_track_response,
)
from venice_ai.types.api.chat import (
    ChatChoice,
    ChatCompletionResponse,
    ChatMessage,
    ChatUsage,
)
from venice_ai.types.api.embeddings import (
    EmbeddingObject,
    EmbeddingsResponse,
    EmbeddingUsage,
)
from venice_ai.types.api.models import (
    LLMModelPricing,
    ModelResponse,
    ModelsListResponse,
    PricingTier,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

# Fake test models — never reach the API, deterministic ids are safe.
_MODEL_A = "fake-test-model-a"
_MODEL_B = "fake-test-model-b"


def _pricing(input_usd: float, output_usd: float) -> LLMModelPricing:
    return LLMModelPricing(
        input=PricingTier(usd=input_usd, diem=input_usd),
        output=PricingTier(usd=output_usd, diem=output_usd),
        cache_input=None,
    )


def _chat_response(
    *,
    model: str = _MODEL_A,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="resp-test",
        object="chat.completion",
        created=1000000,
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content="hello"),
                finish_reason="stop",
                stop_reason=None,
            )
        ],
        usage=ChatUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=None,
        ),
        prompt_logprobs=None,
        venice_parameters=None,
        service_tier=None,
        system_fingerprint=None,
        kv_transfer_params=None,
    )


def _embedding_response(*, model: str = _MODEL_A, total_tokens: int = 30) -> EmbeddingsResponse:
    return EmbeddingsResponse(
        object="list",
        model=model,
        data=[
            EmbeddingObject(object="embedding", index=0, embedding=[0.1, 0.2], encoding_format=None)
        ],
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
        id=None,
        created=None,
    )


# ---------------------------------------------------------------------------
# CostTracker — track()
# ---------------------------------------------------------------------------


class TestCostTrackerTrack:
    @pytest.mark.asyncio
    async def test_chat_response_cost_math(self):
        # 100 prompt @ $3/M = 0.0003; 50 completion @ $6/M = 0.0003 → total 0.0006
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(3.0, 6.0)})
        cost = await tracker.track(_chat_response())
        assert cost == Decimal("0.0006")
        assert tracker.total_cost_usd == Decimal("0.0006")
        assert tracker.total_tokens == 150
        assert len(tracker.requests) == 1

    @pytest.mark.asyncio
    async def test_embedding_response_cost_math(self):
        # 30 tokens @ $0.10/M = 0.000003
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(0.10, 0.10)})
        cost = await tracker.track(_embedding_response(total_tokens=30))
        assert cost == Decimal("0.000003")
        assert tracker.total_tokens == 30

    @pytest.mark.asyncio
    async def test_record_metadata_preserved(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await tracker.track(_chat_response(), metadata={"caller": "unit-test"})
        rec = tracker.requests[0]
        assert isinstance(rec, CostRecord)
        assert rec.metadata == {"caller": "unit-test"}
        assert rec.model == _MODEL_A
        assert rec.prompt_tokens == 100
        assert rec.completion_tokens == 50
        assert rec.total_tokens == 150

    @pytest.mark.asyncio
    async def test_model_kwarg_overrides_response_model(self):
        tracker = CostTracker(pricing_map={"override-id": _pricing(1.0, 1.0)})
        cost = await tracker.track(_chat_response(model=_MODEL_A), model="override-id")
        assert cost > Decimal("0")
        assert tracker.requests[0].model == "override-id"

    @pytest.mark.asyncio
    async def test_unknown_model_yields_zero_cost(self):
        # Pricing missing for response.model — tracker does NOT raise; cost is 0.
        tracker = CostTracker(pricing_map={})
        with pytest.warns(UserWarning, match="no pricing for model"):
            cost = await tracker.track(_chat_response())
        assert cost == Decimal("0.00")
        # Record is still created with zero cost.
        assert len(tracker.requests) == 1
        assert tracker.requests[0].cost_usd == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_unpriced_model_warns_once_per_model(self):
        tracker = CostTracker()
        with pytest.warns(UserWarning, match=f"no pricing for model '{_MODEL_A}'") as caught:
            await tracker.track(_chat_response(model=_MODEL_A))
            await tracker.track(_chat_response(model=_MODEL_A))
        assert len(caught) == 1

        with pytest.warns(UserWarning, match=f"no pricing for model '{_MODEL_B}'"):
            await tracker.track(_chat_response(model=_MODEL_B))

    @pytest.mark.asyncio
    async def test_unpriced_embedding_model_warns(self):
        tracker = CostTracker()
        response = _embedding_response(model="embed-x")
        with pytest.warns(UserWarning, match="no pricing for model 'embed-x'"):
            cost = await tracker.track(response)
        assert cost == Decimal("0.00")

    @pytest.mark.asyncio
    async def test_priced_model_does_not_warn(self, recwarn: pytest.WarningsRecorder):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await tracker.track(_chat_response(model=_MODEL_A))
        assert not [w for w in recwarn if "no pricing" in str(w.message)]

    @pytest.mark.asyncio
    async def test_warning_returns_after_reset(self):
        tracker = CostTracker()
        with pytest.warns(UserWarning, match="no pricing"):
            await tracker.track(_chat_response(model=_MODEL_A))
        await tracker.reset()
        with pytest.warns(UserWarning, match="no pricing"):
            await tracker.track(_chat_response(model=_MODEL_A))

    @pytest.mark.asyncio
    async def test_from_client_keeps_a_model_priced_without_diem(self):
        entry = ModelResponse.model_validate(
            {
                "id": "no-diem",
                "object": "model",
                "created": None,
                "owned_by": "venice.ai",
                "type": "text",
                "model_spec": {
                    "name": "no-diem",
                    "pricing": {"input": {"usd": 2.0}, "output": {"usd": 4.0}},
                },
            }
        )
        client = MagicMock()
        client.models.list = AsyncMock(return_value=MagicMock(data=[entry]))
        tracker = await CostTracker.from_client(client)
        cost = await tracker.track(_chat_response(model="no-diem"))
        assert "no-diem" in tracker.pricing_map
        assert cost > Decimal("0")

    @pytest.mark.asyncio
    async def test_from_client_prices_embedding_models(self):
        entry = _model_entry("embed-x", _pricing(1.0, 0.0))

        async def listing(*, type: str) -> MagicMock:
            return MagicMock(data=[entry] if type == "embedding" else [])

        client = MagicMock()
        client.models.list = AsyncMock(side_effect=listing)
        tracker = await CostTracker.from_client(client)
        cost = await tracker.track(_embedding_response(model="embed-x", total_tokens=1_000_000))
        assert cost == Decimal("1")

    @pytest.mark.asyncio
    async def test_unsupported_response_type_raises(self):
        tracker = CostTracker()
        with pytest.raises(TypeError, match="does not support"):
            await tracker.track("not a response object")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_concurrent_tracks_are_safe(self):
        # Many concurrent track() calls should accumulate exactly N records and N×cost.
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        n = 50
        await asyncio.gather(*(tracker.track(_chat_response()) for _ in range(n)))
        summary = await tracker.summary()
        assert summary.total_requests == n
        # Per-call cost is 100/M*1 + 50/M*1 = 0.00015; n=50 → 0.0075
        assert summary.total_cost_usd == Decimal("0.00750")
        assert summary.total_tokens == 150 * n


# ---------------------------------------------------------------------------
# CostTracker — summary / by_model / reset
# ---------------------------------------------------------------------------


class TestCostTrackerAggregations:
    @pytest.mark.asyncio
    async def test_summary_empty(self):
        tracker = CostTracker()
        s = await tracker.summary()
        assert isinstance(s, CostSummary)
        assert s.total_requests == 0
        assert s.total_cost_usd == Decimal("0.00")
        assert s.total_tokens == 0
        assert s.average_cost_usd == Decimal("0.00")
        assert s.average_tokens == 0.0

    @pytest.mark.asyncio
    async def test_summary_averages(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await tracker.track(_chat_response(prompt_tokens=100, completion_tokens=50))
        await tracker.track(_chat_response(prompt_tokens=200, completion_tokens=100))
        s = await tracker.summary()
        assert s.total_requests == 2
        assert s.total_tokens == 450  # 150 + 300
        assert s.average_tokens == 225.0

    @pytest.mark.asyncio
    async def test_by_model_groups_correctly(self):
        tracker = CostTracker(
            pricing_map={
                _MODEL_A: _pricing(1.0, 1.0),
                _MODEL_B: _pricing(2.0, 2.0),
            }
        )
        await tracker.track(_chat_response(model=_MODEL_A))
        await tracker.track(_chat_response(model=_MODEL_A))
        await tracker.track(_chat_response(model=_MODEL_B))
        breakdown = await tracker.by_model()
        assert set(breakdown.keys()) == {_MODEL_A, _MODEL_B}
        assert breakdown[_MODEL_A] == Decimal("0.00030")  # 2x at 0.00015
        assert breakdown[_MODEL_B] == Decimal("0.00030")  # 1x at 0.0003

    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await tracker.track(_chat_response())
        assert tracker.total_cost_usd > Decimal("0")
        await tracker.reset()
        assert tracker.total_cost_usd == Decimal("0.00")
        assert tracker.total_tokens == 0
        assert tracker.requests == []


# ---------------------------------------------------------------------------
# CostTracker.from_client
# ---------------------------------------------------------------------------


class TestCostTrackerFromClient:
    @pytest.mark.asyncio
    async def test_pre_populates_pricing_map(self):
        # Mock client.models.list(type="chat") returning two priced models.
        listing = ModelsListResponse(
            object="list",
            type="text",
            data=[
                _model_entry(_MODEL_A, _pricing(1.0, 2.0)),
                _model_entry(_MODEL_B, _pricing(3.0, 4.0)),
                _model_entry("no-pricing", None),  # filtered out
            ],
        )
        mock_client = MagicMock()
        mock_client.models.list = AsyncMock(return_value=listing)

        tracker = await CostTracker.from_client(mock_client)
        assert _MODEL_A in tracker.pricing_map
        assert _MODEL_B in tracker.pricing_map
        assert "no-pricing" not in tracker.pricing_map
        assert [c.kwargs for c in mock_client.models.list.await_args_list] == [
            {"type": "chat"},
            {"type": "embedding"},
        ]


# ---------------------------------------------------------------------------
# VeniceClient.attach_cost_tracker
# ---------------------------------------------------------------------------


class TestAttachCostTracker:
    """Post-construction wiring solves the chicken-and-egg between
    :meth:`CostTracker.from_client` (needs an open client) and
    :class:`VeniceClient`'s ``cost_tracker=`` constructor kwarg."""

    @pytest.mark.asyncio
    async def test_attach_hydrates_pricing_map(self):
        from venice_ai._client import VeniceClient

        # Build a no-network client (we never call any HTTP method).
        client = VeniceClient.__new__(VeniceClient)
        client._cost_tracker = None  # type: ignore[attr-defined]
        listing = ModelsListResponse(
            object="list",
            type="text",
            data=[_model_entry(_MODEL_A, _pricing(1.0, 2.0))],
        )
        client.models = MagicMock()  # type: ignore[attr-defined]
        client.models.list = AsyncMock(return_value=listing)

        tracker = CostTracker()
        assert tracker.pricing_map == {}

        await VeniceClient.attach_cost_tracker(client, tracker)
        assert _MODEL_A in tracker.pricing_map
        assert client._cost_tracker is tracker  # type: ignore[attr-defined]
        assert [c.kwargs for c in client.models.list.await_args_list] == [
            {"type": "chat"},
            {"type": "embedding"},
        ]

    @pytest.mark.asyncio
    async def test_attach_skip_pricing_avoids_network(self):
        from venice_ai._client import VeniceClient

        client = VeniceClient.__new__(VeniceClient)
        client._cost_tracker = None  # type: ignore[attr-defined]
        client.models = MagicMock()  # type: ignore[attr-defined]
        client.models.list = AsyncMock(side_effect=AssertionError("must not be called"))

        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(0.5, 1.0)})
        await VeniceClient.attach_cost_tracker(client, tracker, populate_pricing=False)
        assert client._cost_tracker is tracker  # type: ignore[attr-defined]
        client.models.list.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_preserves_existing_pricing_entries(self):
        """User-provided pricing wins over hydrated values."""
        from venice_ai._client import VeniceClient

        client = VeniceClient.__new__(VeniceClient)
        client._cost_tracker = None  # type: ignore[attr-defined]
        listing = ModelsListResponse(
            object="list",
            type="text",
            data=[_model_entry(_MODEL_A, _pricing(1.0, 2.0))],
        )
        client.models = MagicMock()  # type: ignore[attr-defined]
        client.models.list = AsyncMock(return_value=listing)

        custom = _pricing(99.0, 99.0)
        tracker = CostTracker(pricing_map={_MODEL_A: custom})
        await VeniceClient.attach_cost_tracker(client, tracker)
        assert tracker.pricing_map[_MODEL_A] is custom  # not overwritten


def _model_entry(model_id: str, pricing: LLMModelPricing | None) -> ModelResponse:
    # Use ``model_validate`` so ``model_spec`` is dispatched to ``TextModelSpec``
    # by the spec-hierarchy router; type-specific fields (e.g.
    # ``availableContextTokens``) live on the subclass.
    return ModelResponse.model_validate(
        {
            "id": model_id,
            "object": "model",
            "created": None,
            "owned_by": "venice.ai",
            "type": "text",
            "model_spec": {
                "name": model_id,
                "availableContextTokens": 8192.0,
                "pricing": pricing.model_dump() if pricing is not None else None,
            },
        }
    )


# ---------------------------------------------------------------------------
# BudgetManager
# ---------------------------------------------------------------------------


class TestBudgetManager:
    @pytest.mark.asyncio
    async def test_requires_at_least_one_cap(self):
        with pytest.raises(ValueError, match="at least one"):
            BudgetManager(tracker=CostTracker(), daily_usd=None, monthly_usd=None)

    @pytest.mark.asyncio
    async def test_can_afford_under_daily_cap(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        bm = BudgetManager(tracker=tracker, daily_usd=Decimal("1.00"))
        assert await bm.can_afford(Decimal("0.50")) is True

    @pytest.mark.asyncio
    async def test_can_afford_blocks_over_daily(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        # Pre-load some spend
        for _ in range(100):
            await tracker.track(_chat_response())
        # total spend is 100 * 0.00015 = 0.015
        bm = BudgetManager(tracker=tracker, daily_usd=Decimal("0.02"))
        assert await bm.can_afford(Decimal("0.001")) is True  # 0.016 < 0.02
        assert await bm.can_afford(Decimal("0.010")) is False  # 0.025 > 0.02

    @pytest.mark.asyncio
    async def test_remaining_with_both_caps(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        for _ in range(10):
            await tracker.track(_chat_response())
        # spent = 10 * 0.00015 = 0.0015
        bm = BudgetManager(
            tracker=tracker,
            daily_usd=Decimal("0.01"),
            monthly_usd=Decimal("1.00"),
        )
        rem = await bm.remaining()
        assert isinstance(rem, BudgetRemaining)
        assert rem.daily_remaining_usd == Decimal("0.0085")
        assert rem.daily_used_pct is not None and rem.daily_used_pct == pytest.approx(15.0)
        assert rem.monthly_remaining_usd == Decimal("0.9985")
        assert rem.monthly_used_pct is not None and rem.monthly_used_pct == pytest.approx(0.15)

    @pytest.mark.asyncio
    async def test_remaining_caps_at_zero(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(100.0, 100.0)})
        await tracker.track(_chat_response())  # 0.015 cost
        bm = BudgetManager(tracker=tracker, daily_usd=Decimal("0.001"))
        rem = await bm.remaining()
        assert rem.daily_remaining_usd == Decimal("0")  # clamped, not negative


# ---------------------------------------------------------------------------
# _maybe_track_response — internal hook used by VeniceClient._request
# ---------------------------------------------------------------------------


class TestMaybeTrackResponse:
    @pytest.mark.asyncio
    async def test_tracks_chat_response(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await _maybe_track_response(tracker, _chat_response())
        assert len(tracker.requests) == 1

    @pytest.mark.asyncio
    async def test_tracks_embeddings_response(self):
        tracker = CostTracker(pricing_map={_MODEL_A: _pricing(1.0, 1.0)})
        await _maybe_track_response(tracker, _embedding_response())
        assert len(tracker.requests) == 1

    @pytest.mark.asyncio
    async def test_ignores_unknown_response_type(self):
        tracker = CostTracker()
        # Should silently skip — no raise, no record added.
        await _maybe_track_response(tracker, {"some": "dict"})
        await _maybe_track_response(tracker, None)
        assert len(tracker.requests) == 0

    @pytest.mark.asyncio
    async def test_swallows_tracker_errors(self):
        # Tracker exceptions must NEVER mask successful requests.
        broken = MagicMock()
        broken.track = AsyncMock(side_effect=RuntimeError("boom"))
        # Use an actual chat response so isinstance() check passes.
        await _maybe_track_response(broken, _chat_response())
        # No exception leaked. broken.track was called.
        broken.track.assert_awaited_once()


# ---------------------------------------------------------------------------
# Top-level re-exports
# ---------------------------------------------------------------------------


def test_top_level_exports():
    import venice_ai

    for name in (
        "CostTracker",
        "BudgetManager",
        "CostRecord",
        "CostSummary",
        "BudgetRemaining",
    ):
        assert hasattr(venice_ai, name), f"{name} missing from top-level"
        assert name in venice_ai.__all__, f"{name} not in __all__"
