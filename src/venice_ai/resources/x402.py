"""
Venice x402 API resources.

Wraps the three ``/x402/*`` endpoints documented at
``api-reference/endpoint/x402/``. These endpoints use wallet-based
Sign-In-With-X (SIWE / EIP-4361) authentication rather than Bearer tokens,
so calls go through a per-request ``X-Sign-In-With-X`` header built by
:class:`~venice_ai.auth.x402.X402Auth`.

``top_up`` uses Bearer auth (the standard ``VENICE_API_KEY``) and may
optionally carry an ``X-402-Payment`` header containing a signed payment
payload.

This module is intentionally import-time dependency-free; the ``X402Auth``
helper lives under :mod:`venice_ai.auth.x402` which is only imported lazily
by the caller. That keeps the core SDK installable without the
``eth-account`` / ``siwe`` dependencies when x402 is not used.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from .._pagination import DEFAULT_PAGE_SIZE, Paginator, _PageResult
from .._resource import APIResource
from ..exceptions import PaymentRequiredError
from ..types.api.x402 import (
    X402BalanceResponse,
    X402TopUpResponse,
    X402Transaction,
    X402TransactionsResponse,
)

if TYPE_CHECKING:
    from .._client import VeniceClient  # noqa: F401
    from ..auth.x402 import X402Auth
    from ..auth.x402_solana import SolanaX402Auth

logger = logging.getLogger(__name__)

__all__ = ["X402"]


class X402(APIResource["VeniceClient"]):
    """Provides access to Venice's x402 wallet-based billing endpoints."""

    async def balance(self, *, auth: X402Auth | SolanaX402Auth) -> X402BalanceResponse:
        """Get the current x402 prepaid USDC balance for a wallet.

        Wraps ``GET /api/v1/x402/balance/{wallet}``. Authentication uses
        the SIWE (EIP-4361) flow via the ``X-Sign-In-With-X`` header
        derived from ``auth``; standard Bearer auth is not used here.

        Args:
            auth: An :class:`~venice_ai.auth.x402.X402Auth` instance
                built from the wallet's private key. The wallet address
                is derived automatically.

        Returns:
            :class:`X402BalanceResponse` with ``success`` and ``data``
            (containing ``balanceUsd`` among other fields).

        Raises:
            AuthenticationError: If the SIWE signature is rejected or
                expired.
            PermissionDeniedError: If the wallet is not authorized for
                x402.
            NotFoundError: If the wallet has no x402 ledger entry.
            APIError: For other HTTP-level failures.

        Example:

            .. code-block:: python

                import os
                from venice_ai import VeniceClient
                from venice_ai.auth.x402 import X402Auth

                async with VeniceClient() as client:
                    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
                    bal = await client.x402.balance(auth=auth)
                    print(bal.data.balanceUsd)
        """
        wallet = auth.wallet_address
        return await self._client.get(
            f"x402/balance/{wallet}",
            cast_to=X402BalanceResponse,
            siwe_auth=auth,
        )

    async def transactions(
        self,
        *,
        auth: X402Auth | SolanaX402Auth,
        limit: int | None = None,
        offset: int | None = None,
    ) -> X402TransactionsResponse:
        """List x402 ledger entries (usage + top-ups) for the wallet.

        Wraps ``GET /api/v1/x402/transactions/{wallet}``. SIWE-signed via
        ``auth`` (no Bearer auth).

        Args:
            auth: :class:`~venice_ai.auth.x402.X402Auth` for the wallet to
                query.
            limit: Optional maximum number of transactions to return
                (server allows 1-100; defaults to 50 when omitted).
            offset: Optional number of transactions to skip for
                pagination (server default 0).

        Returns:
            :class:`X402TransactionsResponse` with ``data.transactions``
            and ``data.pagination`` (carries ``hasMore`` for paging).

        Raises:
            AuthenticationError: If the SIWE signature is rejected.
            PermissionDeniedError: If the wallet is not authorized for
                x402.
            InvalidRequestError: If ``limit`` / ``offset`` are out of
                range server-side.
            APIError: For other HTTP-level failures.
        """
        wallet = auth.wallet_address
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.get(
            f"x402/transactions/{wallet}",
            cast_to=X402TransactionsResponse,
            siwe_auth=auth,
            params=params or None,
        )

    def iter_transactions(
        self,
        *,
        auth: X402Auth | SolanaX402Auth,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_items: int | None = None,
    ) -> Paginator[X402Transaction]:
        """Lazily iterate every x402 transaction for the wallet.

        Wraps :meth:`transactions` for unbounded enumeration. Termination
        uses the response's ``data.pagination.hasMore`` flag, so iteration
        stops when the server signals no more pages. Each page hits
        ``GET /api/v1/x402/transactions/{wallet}``.

        Args:
            auth: :class:`X402Auth` for the wallet to query.
            page_size: Server page size (default 100, max 100).
            max_items: Optional cap on total items yielded.

        Returns:
            A :class:`~venice_ai._pagination.Paginator` over
            :class:`X402Transaction`. Iteration is async; HTTP errors
            from underlying :meth:`transactions` calls propagate when the
            paginator is consumed.

        Example::

            async for tx in client.x402.iter_transactions(auth=auth):
                print(tx.id, tx.amount)
        """

        async def _fetch_page(page_index: int) -> _PageResult[X402Transaction]:
            response = await self.transactions(
                auth=auth,
                limit=page_size,
                offset=page_index * page_size,
            )
            items = list(response.data.transactions)
            has_more = response.data.pagination.hasMore
            return _PageResult(items=items, has_more=has_more)

        return Paginator(_fetch_page, page_size=page_size, max_items=max_items)

    async def top_up(self, *, payment_header: str | None = None) -> X402TopUpResponse:
        """Top up the wallet balance via the x402 payment channel.

        Wraps ``POST /api/v1/x402/top-up``. Uses the standard Bearer auth
        (``VENICE_API_KEY``). Optionally accepts a pre-built
        ``X-402-Payment`` header containing the signed payment payload -
        when omitted, an empty POST triggers a 402 response containing
        structured payment-requirement details.

        Args:
            payment_header: Optional signed x402 payment payload sent in
                the ``X-402-Payment`` header.

        Returns:
            :class:`X402TopUpResponse` on success.

        Raises:
            PaymentRequiredError: When the server returns 402 with payment
                instructions (the structured payment requirements are
                attached to the error body).
            AuthenticationError: If the API key is missing or invalid.
            InvalidRequestError: If the supplied payment header is
                malformed.
            APIError: For other HTTP-level failures.
        """
        headers: dict[str, str] | None = None
        if payment_header is not None:
            headers = {"X-402-Payment": payment_header}
        return cast(
            X402TopUpResponse,
            await self._client.post(
                "x402/top-up",
                json_data={},
                cast_to=X402TopUpResponse,
                headers=headers,
            ),
        )

    async def top_up_with(
        self,
        *,
        auth: X402Auth,
        amount_usdc: float,
        max_amount_usdc: float | None = None,
    ) -> X402TopUpResponse:
        """Top up the prepaid ledger from ``auth``'s wallet in one call.

        Implements the full x402 v2 probe-sign-submit flow so callers don't
        have to handle the 402 challenge cycle manually:

        1. POST ``/x402/top-up`` with no payment header → catches
           :class:`~venice_ai.exceptions.PaymentRequiredError`.
        2. Picks the first ``"exact"`` requirement on Base mainnet
           (``eip155:8453``).
        3. Validates ``amount_usdc`` against the server's required amount
           and the optional ``max_amount_usdc`` cap.
        4. Builds the EIP-3009 ``X-402-Payment`` header via
           :meth:`venice_ai.auth.x402.X402Auth.build_payment_header`.
        5. Re-POSTs ``/x402/top-up`` with the signed header; returns the
           settlement response.

        Currently supports **only USDC on Base mainnet**. Other tokens /
        chains require manual signing via
        :meth:`X402Auth.build_payment_header` with explicit overrides.

        Args:
            auth: An :class:`~venice_ai.auth.x402.X402Auth` whose private
                key signs the EIP-3009 transfer authorization. The wallet
                must hold enough USDC on-chain to cover the payment.
            amount_usdc: The intended top-up amount in USD (Venice's x402
                pricing is in USD; USDC is the settlement asset). Must
                meet or exceed the server's minimum (currently $5).
            max_amount_usdc: Optional safety cap. Defaults to
                ``amount_usdc``. Raises :class:`ValueError` before signing
                if the server's required amount exceeds this cap.

        Returns:
            :class:`~venice_ai.types.api.x402.X402TopUpResponse` from the
            settled top-up. Inspect ``.data.amountCredited`` for the USD
            amount credited to the prepaid ledger and
            ``.data.paymentId`` for the settlement record.

        Raises:
            ValueError: If ``amount_usdc`` is below the server's minimum,
                or if the server's required amount exceeds
                ``max_amount_usdc``, or if the 402 body is malformed.
            ImportError: If the ``[x402]`` extra is not installed.
            RuntimeError: If the initial probe call (no payment header)
                unexpectedly succeeds, or the 402 body lacks any
                acceptable ``"exact"`` requirement on Base mainnet.
            PaymentRequiredError: If the signed payment is rejected
                server-side (e.g., insufficient on-chain balance, expired
                signature, replayed nonce).
            APIError: For other HTTP-level failures.

        Example:

            .. code-block:: python

                import os
                from venice_ai import VeniceClient
                from venice_ai.auth.x402 import X402Auth

                async with VeniceClient() as client:
                    auth = X402Auth(private_key=os.environ["WALLET_PRIVATE_KEY"])
                    result = await client.x402.top_up_with(
                        auth=auth,
                        amount_usdc=5.0,
                    )
                    print(f"Credited ${result.data.amountCredited}; "
                          f"new balance ${result.data.newBalance}")
        """
        if amount_usdc <= 0:
            raise ValueError(f"amount_usdc must be positive, got {amount_usdc}")
        cap_usdc = max_amount_usdc if max_amount_usdc is not None else amount_usdc
        if cap_usdc < amount_usdc:
            raise ValueError(
                f"max_amount_usdc {cap_usdc} cannot be less than amount_usdc {amount_usdc}"
            )
        cap_units = int(round(cap_usdc * 1_000_000))

        # 1. Probe: empty top_up returns 402 with structured requirements.
        requirement: dict[str, Any] | None = None
        try:
            unexpected = await self.top_up()
        except PaymentRequiredError as exc:
            body = exc.body or {}
            accepts = body.get("accepts") if isinstance(body, dict) else None
            if not accepts:
                raise RuntimeError(
                    f"x402.top_up() probe returned 402 with no 'accepts' list: {body!r}"
                ) from exc
            for entry in accepts:
                if not isinstance(entry, dict):
                    continue
                if entry.get("network") == "eip155:8453" and entry.get("scheme", "exact") in (
                    "exact",
                    "evm-exact",
                    "evm/exact",
                ):
                    requirement = entry
                    break
            if requirement is None:
                raise RuntimeError(
                    "x402.top_up() probe 402 has no 'exact' requirement on "
                    f"eip155:8453: {accepts!r}"
                ) from exc
        else:
            raise RuntimeError(
                f"x402.top_up() with no payment_header succeeded unexpectedly; "
                f"expected 402: {unexpected!r}"
            )

        # 2. Validate against caller's amount_usdc and cap.
        server_units_raw = requirement.get("amount", "0")
        try:
            server_units = int(server_units_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"x402 requirement has non-integer amount {server_units_raw!r}"
            ) from exc
        requested_units = int(round(amount_usdc * 1_000_000))
        if server_units > cap_units:
            raise ValueError(
                f"Server requires ${server_units / 1_000_000} USDC (>{cap_usdc} cap); "
                "refusing to sign."
            )
        if requested_units < server_units:
            raise ValueError(
                f"amount_usdc=${amount_usdc} is below server minimum "
                f"${server_units / 1_000_000}; pass a larger amount."
            )

        # 3. Sign the EIP-3009 payment.
        payment_header = auth.build_payment_header(
            requirement,
            max_amount_units=cap_units,
        )

        # 4. Submit the signed payment.
        return await self.top_up(payment_header=payment_header)

    async def top_up_with_solana(
        self,
        *,
        auth: SolanaX402Auth,
        amount_usdc: float,
        max_amount_usdc: float | None = None,
        rpc_url: str | None = None,
    ) -> X402TopUpResponse:
        """Top up the prepaid ledger from a Solana wallet in one call.

        Implements the full x402 v2 probe-sign-submit flow for the Solana
        ("exact" SVM settlement) path, mirroring :meth:`top_up_with` for EVM:

        1. POST ``/x402/top-up`` with no payment header → catches
           :class:`~venice_ai.exceptions.PaymentRequiredError`.
        2. Picks the ``accepts`` entry whose ``network`` identifies Solana
           mainnet (the bare string ``"solana"`` or the CAIP-2 id
           ``"solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"``).
        3. Validates ``amount_usdc`` against the server's required amount and
           the optional ``max_amount_usdc`` cap.
        4. Fetches the live transaction context (recent blockhash, mint
           decimals, token program) from a Solana RPC via
           :func:`venice_ai.auth.x402_solana.fetch_solana_tx_context`.
        5. Builds the base64 ``X-402-Payment`` envelope (a partially-signed
           ``VersionedTransaction``) via
           :meth:`venice_ai.auth.x402_solana.SolanaX402Auth.build_payment_header`.
        6. Re-POSTs ``/x402/top-up`` with the signed header; returns the
           settlement response.

        Args:
            auth: A :class:`~venice_ai.auth.x402_solana.SolanaX402Auth` whose
                base58 secret signs the SPL transfer authorization. The wallet
                must hold enough USDC on-chain to cover the payment (the
                facilitator pays the gas / network fees).
            amount_usdc: The intended top-up amount in USD. Must meet or exceed
                the server's minimum (currently $5).
            max_amount_usdc: Optional safety cap. Defaults to ``amount_usdc``.
                Raises :class:`ValueError` before signing if the server's
                required amount exceeds this cap.
            rpc_url: Optional Solana JSON-RPC endpoint. Defaults to the
                ``VENICE_X402_SOLANA_RPC_URL`` environment variable, then to
                ``https://api.mainnet-beta.solana.com``.

        Returns:
            :class:`~venice_ai.types.api.x402.X402TopUpResponse` from the
            settled top-up.

        Raises:
            ValueError: If ``amount_usdc`` is below the server's minimum, or
                if the server's required amount exceeds ``max_amount_usdc``,
                or if the 402 body is malformed.
            ImportError: If the ``[x402-solana]`` extra is not installed.
            RuntimeError: If the initial probe call succeeds unexpectedly, the
                402 body lacks a Solana mainnet requirement, or RPC fails.
            PaymentRequiredError: If the signed payment is rejected
                server-side.
            APIError: For other HTTP-level failures.

        Example:

            .. code-block:: python

                import os
                from venice_ai import VeniceClient
                from venice_ai.auth.x402_solana import SolanaX402Auth

                async with VeniceClient() as client:
                    auth = SolanaX402Auth(private_key=os.environ["SOLANA_SECRET"])
                    result = await client.x402.top_up_with_solana(
                        auth=auth,
                        amount_usdc=5.0,
                    )
                    print(f"Credited ${result.data.amountCredited}")
        """
        import os

        import aiohttp

        from ..auth.x402_solana import (
            DEFAULT_SOLANA_RPC_URL,
            SOLANA_MAINNET_CAIP2,
            fetch_solana_tx_context,
            is_solana_mainnet,
        )

        if amount_usdc <= 0:
            raise ValueError(f"amount_usdc must be positive, got {amount_usdc}")
        cap_usdc = max_amount_usdc if max_amount_usdc is not None else amount_usdc
        if cap_usdc < amount_usdc:
            raise ValueError(
                f"max_amount_usdc {cap_usdc} cannot be less than amount_usdc {amount_usdc}"
            )
        cap_units = int(round(cap_usdc * 1_000_000))

        # 1. Probe: empty top_up returns 402 with structured requirements.
        requirement: dict[str, Any] | None = None
        try:
            unexpected = await self.top_up()
        except PaymentRequiredError as exc:
            body = exc.body or {}
            accepts = body.get("accepts") if isinstance(body, dict) else None
            if not accepts:
                raise RuntimeError(
                    f"x402.top_up() probe returned 402 with no 'accepts' list: {body!r}"
                ) from exc
            for entry in accepts:
                if isinstance(entry, dict) and is_solana_mainnet(entry.get("network")):
                    requirement = entry
                    break
            if requirement is None:
                raise RuntimeError(
                    "x402.top_up() probe 402 has no Solana mainnet requirement "
                    f"(expected network 'solana' or {SOLANA_MAINNET_CAIP2!r}): {accepts!r}"
                ) from exc
        else:
            raise RuntimeError(
                f"x402.top_up() with no payment_header succeeded unexpectedly; "
                f"expected 402: {unexpected!r}"
            )

        # 2. Validate against caller's amount_usdc and cap.
        server_units_raw = requirement.get("amount", "0")
        try:
            server_units = int(server_units_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"x402 requirement has non-integer amount {server_units_raw!r}"
            ) from exc
        requested_units = int(round(amount_usdc * 1_000_000))
        if server_units > cap_units:
            raise ValueError(
                f"Server requires ${server_units / 1_000_000} USDC (>{cap_usdc} cap); "
                "refusing to sign."
            )
        if requested_units < server_units:
            raise ValueError(
                f"amount_usdc=${amount_usdc} is below server minimum "
                f"${server_units / 1_000_000}; pass a larger amount."
            )

        # 3. Fetch live tx context from the Solana RPC.
        effective_rpc = (
            rpc_url or os.environ.get("VENICE_X402_SOLANA_RPC_URL") or DEFAULT_SOLANA_RPC_URL
        )
        mint = str(requirement.get("asset"))
        async with aiohttp.ClientSession() as session:
            recent_blockhash, mint_decimals, token_program = await fetch_solana_tx_context(
                effective_rpc, mint, http=session
            )

        # 4. Build the partially-signed payment envelope.
        payment_header = auth.build_payment_header(
            requirement=requirement,
            recent_blockhash=recent_blockhash,
            mint_decimals=mint_decimals,
            token_program=token_program,
            max_amount_units=cap_units,
        )

        # 5. Submit the signed payment.
        return await self.top_up(payment_header=payment_header)


# Used by Pydantic's forward references in places that import the x402 module
# without the optional auth extras installed. Safe because resources only
# import this class for typing within method signatures (TYPE_CHECKING).
_: Any = None
