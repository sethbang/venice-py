"""
x402 endpoint models for Venice AI API.

Covers the three ``/x402/*`` endpoints: ``balance``, ``top-up``, and
``transactions``. All responses follow a ``{success, data}`` envelope.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class X402BalanceData(BaseModel):
    """Inner ``data`` for GET /x402/balance/{walletAddress}."""

    model_config = ConfigDict(extra="allow")

    walletAddress: str = Field(..., description="The wallet address queried.")
    balanceUsd: float = Field(..., description="Current prepaid USDC balance in USD.")
    canConsume: bool = Field(
        ..., description="Whether the balance is sufficient for the minimum API request."
    )
    minimumTopUpUsd: float | None = Field(
        default=None, description="Suggested minimum top-up amount when balance is low."
    )
    suggestedTopUpUsd: float | None = Field(
        default=None, description="Suggested default top-up amount."
    )
    diemBalanceUsd: float | None = Field(
        default=None, description="Diem (Venice internal) balance, if applicable."
    )


class X402BalanceResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Literal[True]
    data: X402BalanceData


class X402TopUpData(BaseModel):
    """Inner ``data`` for POST /x402/top-up."""

    model_config = ConfigDict(extra="allow")

    walletAddress: str
    amountCredited: float = Field(..., description="Amount credited in USD.")
    newBalance: float = Field(..., description="New balance after the top-up (USD).")
    paymentId: str = Field(..., description="Server-assigned payment identifier.")


class X402TopUpResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Literal[True]
    data: X402TopUpData


class X402Transaction(BaseModel):
    """A single ledger entry returned by GET /x402/transactions/{walletAddress}."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Ledger entry ID.")
    amount: float = Field(
        ..., description="Signed USD amount (negative for debit, positive for top-up)."
    )
    balanceAfter: float = Field(..., description="Balance immediately after this entry.")
    type: str = Field(..., description='Entry type, e.g. "TOP_UP", "CHARGE".')
    createdAt: str = Field(..., description="ISO-8601 timestamp of the entry.")
    requestId: str | None = Field(default=None, description="Associated API request, if any.")
    modelId: str | None = Field(default=None, description="Model used, if the entry is usage.")


class X402TransactionsPagination(BaseModel):
    model_config = ConfigDict(extra="allow")

    limit: int
    offset: int
    hasMore: bool


class X402TransactionsData(BaseModel):
    model_config = ConfigDict(extra="allow")

    walletAddress: str
    currentBalance: float
    transactions: list[X402Transaction]
    pagination: X402TransactionsPagination


class X402TransactionsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    success: Literal[True]
    data: X402TransactionsData


__all__ = [
    "X402BalanceData",
    "X402BalanceResponse",
    "X402TopUpData",
    "X402TopUpResponse",
    "X402Transaction",
    "X402TransactionsData",
    "X402TransactionsPagination",
    "X402TransactionsResponse",
]
