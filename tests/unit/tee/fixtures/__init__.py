"""Offline TEE attestation test corpus.

Four real ``GET /tee/attestation`` responses captured live from the Venice API
on 2026-06-04 (capture epoch :data:`CAPTURE_EPOCH`), one per ``e2ee-*`` model in
:data:`FIXTURE_SLUGS`. Each ``attestation_<slug>.json`` wraps the full server
response under ``"attestation"`` plus the exact ``"_client_nonce"`` the client
sent (the REPORTDATA binding needs it) and the ``"_capture_epoch"`` the
collateral was fetched at.

A fifth capture, ``attestation_live_evidence.json``, uses the current
``attestation.evidence`` wire shape and has its own capture epoch
:data:`EVIDENCE_CAPTURE_EPOCH`.

Each capture has its **own** ``collateral_<slug>.json`` snapshot
(:class:`dcap_qvl.QuoteCollateralV3` ``to_json()``). All four quotes share FMSPC
``90C06F000000`` / ``ca=platform`` / TDX v4, yet their PCK certificate chains differ
(different physical platforms). A snapshot carries the PCK chain of the machine it
was fetched from. ``DcapTdxVerifier`` drops that chain and uses the one in the quote,
so one snapshot can verify any of the four; handed straight to
``dcap_qvl.verify_with_root_ca``, it fails QE-report signature verification on the
other machines' quotes.

Freeze ``now_secs`` to :data:`CAPTURE_EPOCH` in every test: the TCB-info /
QE-identity collateral carries a validity window, and a real wall-clock
``now_secs`` would make the committed snapshots "expire".
"""

from __future__ import annotations

import json
from pathlib import Path

#: Wall-clock epoch (seconds) at which the captures + collateral were taken.
#: Freeze ``now_secs`` to this in offline tests so validity windows pass.
CAPTURE_EPOCH = 1780598249

#: Model slugs with committed ``attestation_<slug>.json`` + ``collateral_<slug>.json``.
#: ``glm_5_1`` is the near-ai provider (``tee_provider != "phala"``) carrying the
#: identical dstack structure; ``gpt_oss_120b_p`` and ``glm_5_1`` are the two
#: where ``intel_quote != quote``.
FIXTURE_SLUGS = (
    "gemma_3_27b_p",
    "gpt_oss_120b_p",
    "qwen3_vl_30b_a3b_p",
    "glm_5_1",
)

#: Slug for the capture using the current ``attestation.evidence`` wire shape.
#: Kept out of :data:`FIXTURE_SLUGS` because the dstack-shaped tests assert
#: dstack-only fields (``app_compose`` / ``compose_hash``) it does not carry.
EVIDENCE_SLUG = "live_evidence"

#: Capture epoch for :data:`EVIDENCE_SLUG` (its collateral has its own window).
#: Must equal the fixture's embedded ``_capture_epoch``.
EVIDENCE_CAPTURE_EPOCH = 1784573023

_DIR = Path(__file__).parent


def load_attestation(slug: str) -> dict:
    """Return the raw captured payload (``attestation`` + ``_client_nonce`` + ``_capture_epoch``)."""
    return json.loads((_DIR / f"attestation_{slug}.json").read_text())


def load_collateral_json(slug: str) -> str:
    """Return the committed ``QuoteCollateralV3.to_json()`` text for ``slug``."""
    return (_DIR / f"collateral_{slug}.json").read_text()
