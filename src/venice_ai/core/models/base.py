"""
Base models for the Venice AI SDK.

This module defines ``VeniceBaseModel`` and ``TimestampMixin`` — the
foundational types that all other core models inherit from.  It has **no**
dependencies on sibling submodules at import time; header-model types used
in property return annotations are guarded by ``TYPE_CHECKING`` and
resolved at runtime via lazy imports inside each property method.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from .headers import (
        BalanceInfo,
        ContentSafetyInfo,
        DeprecationInfo,
        ModelInfo,
        PaginationInfo,
        RateLimitInfo,
    )


# ============================================================================
# Base Models
# ============================================================================


class VeniceBaseModel(BaseModel):
    """Base model for all Venice AI Pydantic models."""

    model_config = ConfigDict(
        # Allow field names to be used as aliases (renamed in v2)
        validate_by_name=True,
        # Validate assignment
        validate_assignment=True,
        # Use enum values
        use_enum_values=True,
        # Extra fields are forbidden by default
        extra="forbid",
    )

    # Private attribute to store the raw HTTP response for header extraction
    _response: Any | None = PrivateAttr(default=None)

    @property
    def headers(self) -> dict[str, str] | None:
        """Access raw headers from the HTTP response."""
        if self._response and hasattr(self._response, "headers"):
            return dict(self._response.headers)
        return None

    @property
    def response_rate_limits(self) -> RateLimitInfo | None:
        """Extract rate limit information from headers."""
        from .headers import RateLimitInfo as _RateLimitInfo  # lazy

        headers = self.headers
        if not headers:
            return None

        # Create RateLimitInfo instance directly to avoid imports
        rate_limit_data = {
            "limit_requests": self._parse_int(headers.get("x-ratelimit-limit-requests")),
            "remaining_requests": self._parse_int(headers.get("x-ratelimit-remaining-requests")),
            "reset_requests": self._parse_timestamp(headers.get("x-ratelimit-reset-requests")),
            "limit_tokens": self._parse_int(headers.get("x-ratelimit-limit-tokens")),
            "remaining_tokens": self._parse_int(headers.get("x-ratelimit-remaining-tokens")),
            "reset_tokens": self._ms_to_seconds(
                self._parse_float(headers.get("x-ratelimit-reset-tokens"))
            ),
            "type": headers.get("x-ratelimit-type"),
        }

        # Only create if we have some rate limit data
        if any(v is not None for v in rate_limit_data.values()):
            # Use model_validate for proper Pydantic instantiation
            return _RateLimitInfo.model_validate(rate_limit_data)
        return None

    @property
    def pagination_info(self) -> PaginationInfo | None:
        """Get pagination info from response headers."""
        from .headers import PaginationInfo as _PaginationInfo  # lazy

        headers = self.headers
        if not headers:
            return None
        # Check if any pagination headers are present
        if not any(k.startswith("x-pagination") for k in headers):
            return None
        # Parse all four pagination header values
        limit = self._parse_int(headers.get("x-pagination-limit"))
        page = self._parse_int(headers.get("x-pagination-page"))
        total = self._parse_int(headers.get("x-pagination-total"))
        total_pages = self._parse_int(headers.get("x-pagination-total-pages"))
        # PaginationInfo requires all fields; only construct when all are present
        if limit is not None and page is not None and total is not None and total_pages is not None:
            return _PaginationInfo.model_validate(
                {
                    "limit": limit,
                    "page": page,
                    "total": total,
                    "total_pages": total_pages,
                }
            )
        return None

    @property
    def deprecation_info(self) -> DeprecationInfo | None:
        """Extract model deprecation information from headers."""
        from .headers import DeprecationInfo as _DeprecationInfo  # lazy

        headers = self.headers
        if not headers:
            return None

        warning = headers.get("x-venice-model-deprecation-warning")
        date_str = headers.get("x-venice-model-deprecation-date")

        if not warning and not date_str:
            return None

        # Use model_validate for proper Pydantic instantiation
        return _DeprecationInfo.model_validate(
            {
                "warning": warning,
                "date": self._parse_timestamp(date_str) if date_str else None,
            }
        )

    @property
    def balance_info(self) -> BalanceInfo | None:
        """What the calling API key could spend before this request, from headers.

        Parsed from ``x-venice-balance-usd`` / ``x-venice-balance-diem``. The
        value is key-scoped (see :class:`~venice_ai.core.models.headers.BalanceInfo`),
        not the account balance that ``client.billing.get_balance()`` reports.
        ``None`` when neither header is present.
        """
        from .headers import BalanceInfo as _BalanceInfo  # lazy

        headers = self.headers
        if not headers:
            return None

        diem_str = headers.get("x-venice-balance-diem")
        usd_str = headers.get("x-venice-balance-usd")

        if not any([diem_str, usd_str]):
            return None

        return _BalanceInfo.model_validate(
            {
                "diem": self._parse_float(diem_str),
                "usd": self._parse_float(usd_str),
            }
        )

    @property
    def x402_balance_remaining(self) -> float | None:
        """Remaining x402 credit balance in USD after this request.

        Only present when authenticating via x402 (wallet-based auth). Maps to
        the ``X-Balance-Remaining`` response header declared on the inference
        and augment endpoints (chat, image, audio, video, embeddings, augment,
        etc.). Returns ``None`` for Bearer-auth requests or when the header is
        absent.
        """
        headers = self.headers
        if not headers:
            return None
        return self._parse_float(headers.get("x-balance-remaining"))

    @property
    def venice_version(self) -> str | None:
        """Get Venice API version from headers."""
        headers = self.headers
        return headers.get("x-venice-version") if headers else None

    @property
    def request_id(self) -> str | None:
        """Get the Cloudflare CF-RAY request ID for debugging/support."""
        headers = self.headers
        return headers.get("cf-ray") if headers else None

    @property
    def content_safety_info(self) -> ContentSafetyInfo | None:
        """Extract content safety information from response headers."""
        from .headers import ContentSafetyInfo as _ContentSafetyInfo  # lazy

        headers = self.headers
        if not headers:
            return None

        def _parse_bool(val: str | None) -> bool | None:
            if val is None:
                return None
            return val.lower() == "true"

        info = _ContentSafetyInfo(
            is_blurred=_parse_bool(headers.get("x-venice-is-blurred")),
            is_content_violation=_parse_bool(headers.get("x-venice-is-content-violation")),
            is_adult_model_content_violation=_parse_bool(
                headers.get("x-venice-is-adult-model-content-violation")
            ),
            contains_minor=_parse_bool(headers.get("x-venice-contains-minor")),
        )
        # Only return if at least one field is set
        if any(
            v is not None
            for v in [
                info.is_blurred,
                info.is_content_violation,
                info.is_adult_model_content_violation,
                info.contains_minor,
            ]
        ):
            return info
        return None

    @property
    def model_info(self) -> ModelInfo | None:
        """Extract model information from response headers."""
        from .headers import ModelInfo as _ModelInfo  # lazy

        headers = self.headers
        if not headers:
            return None

        info = _ModelInfo(
            model_id=headers.get("x-venice-model-id"),
            model_name=headers.get("x-venice-model-name"),
            model_router=headers.get("x-venice-model-router"),
            deprecation_warning=headers.get("x-venice-model-deprecation-warning"),
            deprecation_date=headers.get("x-venice-model-deprecation-date"),
        )
        if any(v is not None for v in [info.model_id, info.model_name, info.model_router]):
            return info
        return None

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        """Parse string to int, returning ``None`` if invalid.

        Delegates to :func:`venice_ai.utils.parsing.safe_int`.
        """
        from venice_ai.utils.parsing import safe_int

        return safe_int(value)

    @staticmethod
    def _parse_float(value: str | None) -> float | None:
        """Parse string to float, returning ``None`` if invalid.

        Delegates to :func:`venice_ai.utils.parsing.safe_float`.
        """
        from venice_ai.utils.parsing import safe_float

        return safe_float(value)

    @staticmethod
    def _ms_to_seconds(value: float | None) -> float | None:
        """Normalize a possible ms-epoch numeric to seconds.

        Several Venice rate-limit reset headers (e.g.
        ``x-ratelimit-reset-requests`` / ``x-ratelimit-reset-tokens``) arrive
        as 13-digit *absolute Unix epoch milliseconds* (e.g.
        ``1780567876726``). Values at or above ``1e12`` are treated as
        milliseconds and divided by 1000; smaller values (already in seconds,
        e.g. a 10-digit epoch) pass through untouched. This makes ms/seconds
        handling symmetric across reset headers regardless of the consuming
        field's type (``datetime`` vs ``float``).
        """
        if value is None:
            return None
        if abs(value) >= 1e12:
            return value / 1000
        return value

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        """Parse timestamp string to datetime, returning None if invalid."""
        if value is None:
            return None
        try:
            # Handle Unix timestamp. The integer may be seconds or 13-digit
            # millisecond epoch; normalize ms→seconds before constructing.
            if value.isdigit():
                import datetime as dt

                seconds = self._ms_to_seconds(float(value))
                assert seconds is not None  # value.isdigit() guarantees non-None
                return dt.datetime.fromtimestamp(seconds, tz=dt.UTC)
            # Handle ISO format
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (ValueError, TypeError, OSError):
            return None


class TimestampMixin(BaseModel):
    """Mixin for models that include timestamps."""

    created_at: datetime | None = Field(default=None, description="Creation timestamp")
    updated_at: datetime | None = Field(default=None, description="Last update timestamp")


__all__ = [
    "VeniceBaseModel",
    "TimestampMixin",
]
