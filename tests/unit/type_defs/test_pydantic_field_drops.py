"""Regression tests for Pydantic silent field-drop fixes.

Each test pins a wire-shaped payload (captured live) or a static construction
that previously lost data, and asserts the data now survives the typed model.
"""

from venice_ai.types.api.api_keys import ApiKey, CreatedApiKey
from venice_ai.types.api.billing import UsageAnalyticsResponse
from venice_ai.types.api.chat import ChatMessage
from venice_ai.types.api.models import ModelResponse
from venice_ai.types.api.requests.api_keys import (
    CreateApiKeyRequest,
    UpdateApiKeyRequest,
    Web3CreateApiKeyRequest,
)
from venice_ai.types.api.requests.chat import (
    AssistantMessage,
    ChatCompletionRequest,
    UserMessage,
)


# ---------------------------------------------------------------------------
# ChatCompletionRequest must not silently drop unmodeled kwargs
# ---------------------------------------------------------------------------
def test_chat_completion_request_preserves_forward_compat_kwargs():
    req = ChatCompletionRequest(
        model="x",
        messages=[UserMessage(content="hi")],
        future_param=1,  # type: ignore[call-arg]
    )
    dumped = req.model_dump(exclude_none=True)
    assert dumped.get("future_param") == 1


# ---------------------------------------------------------------------------
# response ChatMessage must keep reasoning_details, and
# AssistantMessage.from_response must carry it back into history.
# ---------------------------------------------------------------------------
def test_chat_message_preserves_reasoning_details():
    details = [{"type": "reasoning.text", "text": "thinking..."}]
    msg = ChatMessage.model_validate(
        {"role": "assistant", "content": "answer", "reasoning_details": details}
    )
    assert msg.reasoning_details == details


def test_assistant_message_from_response_carries_reasoning_details():
    details = [{"type": "reasoning.text", "text": "thinking..."}]
    response = type(
        "FakeResponse",
        (),
        {
            "choices": [
                type(
                    "FakeChoice",
                    (),
                    {
                        "message": ChatMessage.model_validate(
                            {
                                "role": "assistant",
                                "content": "answer",
                                "reasoning_details": details,
                            }
                        )
                    },
                )()
            ]
        },
    )()
    am = AssistantMessage.from_response(response)  # type: ignore[arg-type]
    assert am.reasoning_details == details


# ---------------------------------------------------------------------------
# image-model resolution pricing with quality/upscale keys must
# survive through the real ModelResponse.model_validate(...) path.
# ---------------------------------------------------------------------------
def test_image_model_pricing_preserves_quality_and_upscale():
    gpt_image_2 = {
        "id": "gpt-image-2",
        "type": "image",
        "object": "model",
        "created": 0,
        "owned_by": "venice",
        "model_spec": {
            "name": "GPT Image 2",
            "pricing": {
                "resolutions": {"1024x1024": {"usd": 0.04, "diem": 0.4}},
                "quality": {"high": {"usd": 0.08, "diem": 0.8}},
                "upscale": {
                    "2x": {"usd": 0.02, "diem": 0.2},
                    "4x": {"usd": 0.04, "diem": 0.4},
                },
            },
        },
    }
    parsed = ModelResponse.model_validate(gpt_image_2)
    pricing = parsed.model_spec.pricing
    assert pricing is not None
    dumped = pricing.model_dump(by_alias=True, exclude_none=True)
    assert "quality" in dumped, f"quality dropped: keys={list(dumped)}"
    assert "upscale" in dumped, f"upscale dropped: keys={list(dumped)}"


# ---------------------------------------------------------------------------
# ApiKey wire-shaped dict must retain limitPeriod,
# currentPeriodUsage, and usage.trailingSevenDays.vcu.
# ---------------------------------------------------------------------------
def _api_key_wire() -> dict:
    return {
        "apiKeyType": "INFERENCE",
        "createdAt": "2026-05-19T02:04:06.442Z",
        "description": "smart-contract",
        "expiresAt": None,
        "id": "2c9d04bd-aaaa-bbbb-cccc-dddddddddddd",
        "last6Chars": "76QpKw",
        "consumptionLimits": {"diem": None, "usd": 150, "vcu": None},
        "limitPeriod": "LIFETIME",
        "lastUsedAt": "2026-06-04T09:32:19.821Z",
        "usage": {"trailingSevenDays": {"usd": "61.9948", "vcu": "0.0000", "diem": "3.1667"}},
        "currentPeriodUsage": {"usd": "113.8246", "diem": "13.7806"},
        "modelPrivacy": "PRIVATE_TEXT",
    }


def test_api_key_preserves_limit_period():
    ak = ApiKey.model_validate(_api_key_wire())
    assert ak.limitPeriod == "LIFETIME"


def test_api_key_preserves_model_privacy():
    ak = ApiKey.model_validate(_api_key_wire())
    assert ak.modelPrivacy == "PRIVATE_TEXT"


def test_api_key_preserves_current_period_usage():
    ak = ApiKey.model_validate(_api_key_wire())
    assert ak.currentPeriodUsage is not None
    dumped = ak.currentPeriodUsage.model_dump(exclude_none=True)
    assert dumped.get("usd") == "113.8246"
    assert dumped.get("diem") == "13.7806"


def test_trailing_seven_days_usage_preserves_vcu():
    ak = ApiKey.model_validate(_api_key_wire())
    assert ak.usage is not None
    assert ak.usage.trailingSevenDays.vcu == "0.0000"


# ---------------------------------------------------------------------------
# CreatedApiKey (create + web3 create response) must retain
# limitPeriod, which is swagger-required on the response envelope.
# ---------------------------------------------------------------------------
def test_created_api_key_preserves_limit_period():
    created = CreatedApiKey.model_validate(
        {
            "id": "x",
            "apiKey": "sk-abc123",
            "apiKeyType": "INFERENCE",
            "description": "d",
            "consumptionLimit": {"usd": 50, "diem": 10, "vcu": None},
            "limitPeriod": "MONTH",
            "expiresAt": None,
        }
    )
    assert created.limitPeriod == "MONTH"


# ---------------------------------------------------------------------------
# create/update request models must be able to set limitPeriod.
# ---------------------------------------------------------------------------
def test_create_api_key_request_accepts_limit_period():
    req = CreateApiKeyRequest(apiKeyType="INFERENCE", description="k", limitPeriod="MONTH")
    assert req.model_dump(exclude_none=True).get("limitPeriod") == "MONTH"


def test_update_api_key_request_accepts_limit_period():
    req = UpdateApiKeyRequest(id="k", limitPeriod="MONTH")
    assert req.model_dump(exclude_none=True).get("limitPeriod") == "MONTH"


def test_web3_create_api_key_request_accepts_limit_period():
    req = Web3CreateApiKeyRequest(
        apiKeyType="INFERENCE",
        address="0xabc",
        signature="sig",
        token="tok",
        limitPeriod="MONTH",
    )
    assert req.model_dump(exclude_none=True).get("limitPeriod") == "MONTH"


# ---------------------------------------------------------------------------
# UsageAnalyticsResponse must keep byKeyDailyUsd / byModelDailyUsd.
# ---------------------------------------------------------------------------
def test_usage_analytics_preserves_usd_daily_charts():
    wire = {
        "lookback": "7d",
        "byDate": [],
        "byModel": [],
        "byModelDaily": [{"date": 1, "Llama 3.3 70B": 1.0}],
        "byModelDailyUsd": [{"date": 1, "Llama 3.3 70B": 0.5}],
        "topModels": ["Llama 3.3 70B"],
        "byKey": [],
        "byKeyDaily": [{"date": 1, "Web App": 1.0}],
        "byKeyDailyUsd": [{"date": 1, "Web App": 0.5}],
        "topKeyNames": ["Web App"],
    }
    resp = UsageAnalyticsResponse.model_validate(wire)
    assert resp.byKeyDailyUsd == [{"date": 1, "Web App": 0.5}]
    assert resp.byModelDailyUsd == [{"date": 1, "Llama 3.3 70B": 0.5}]


# ---------------------------------------------------------------------------
# RateLimitLogEntry.rateLimitType is a documented five-value enum, not a
# free-form string: RPM/TPM/RPD are throughput limits, FAILED_REQUESTS counts
# non-success responses in the window, and UNSUPPORTED_FEATURE_REQUESTS counts
# requests asking a model for a feature it does not have.
# ---------------------------------------------------------------------------
def _rate_limit_log_wire(rate_limit_type: str) -> dict:
    return {
        "apiKeyId": "2c9d04bd-aaaa-bbbb-cccc-dddddddddddd",
        "modelId": "zai-org-glm-5-2",
        "rateLimitTier": "paid",
        "rateLimitType": rate_limit_type,
        "timestamp": "2026-09-01T12:00:00.000Z",
    }


def test_rate_limit_type_enum_covers_the_documented_values():
    """Callers compare against these instead of hand-writing magic strings.

    Mirrors how ``VeniceAPIErrorCode`` is used against ``APIError.code``: the
    field stays ``str`` so an unrecognised server value still parses, and the
    enum is the set of known constants.
    """
    from venice_ai.types.api.api_keys import RateLimitType

    assert {member.value for member in RateLimitType} == {
        "RPD",
        "RPM",
        "TPM",
        "FAILED_REQUESTS",
        "UNSUPPORTED_FEATURE_REQUESTS",
    }


def test_rate_limit_type_compares_equal_to_the_wire_string():
    from venice_ai.types.api.api_keys import RateLimitLogEntry, RateLimitType

    entry = RateLimitLogEntry.model_validate(_rate_limit_log_wire("UNSUPPORTED_FEATURE_REQUESTS"))
    assert entry.rateLimitType == RateLimitType.UNSUPPORTED_FEATURE_REQUESTS


def test_rate_limit_log_entry_tolerates_an_unknown_type():
    """A sixth server-side value must not fail the whole log listing."""
    from venice_ai.types.api.api_keys import RateLimitLogEntry

    entry = RateLimitLogEntry.model_validate(_rate_limit_log_wire("CONCURRENCY"))
    assert entry.rateLimitType == "CONCURRENCY"
