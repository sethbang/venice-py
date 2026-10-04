#!/usr/bin/env bash
# Re-record VCR cassettes in cost-ordered batches, pausing for your input
# before each batch. Every batch is first recorded live against the Venice API
# (VENICE_VCR_RECORD=all), then replayed offline (VENICE_CI_MODE=true) to verify
# the freshly recorded cassettes load and match. Cheap/metadata batches run
# first so you can validate the workflow before spending on generation.
#
# The test fixtures delete each target cassette before an ALL-mode record, so a
# PASSING re-record overwrites cleanly with no stale-interaction accumulation
# (vcrpy 8.2 ALL-mode otherwise appends, and replay serves the oldest match).
# A FAILED call writes no cassette at all (the VCR config sets
# record_on_exception=False), so API drift fails loudly and you can fix it and
# retry the batch.
#
# Usage:
#   VENICE_API_KEY=sk-... tools/rerecord_cassettes.sh [START_BATCH]
#     START_BATCH — 1-based batch number to start from (default 1; for resuming
#                   after a fix). Use -l/--list to print the batch plan.
#
# Env:
#   VENICE_API_KEY  — required; your live Venice API key
#   NONINTERACTIVE  — if set, runs every batch without pausing (unattended)
#
# Prerequisites: Redis on localhost:6379 (the test client uses db 15), and the
# project deps installed (`make install`). Benchmarks are intentionally excluded.

set -uo pipefail   # not -e: we handle pytest exit codes ourselves

cd "$(dirname "$0")/.." || exit 1

if [[ -t 1 ]]; then
  G=$'\033[0;32m'; Y=$'\033[1;33m'; R=$'\033[0;31m'; B=$'\033[1;34m'; D=$'\033[2m'; N=$'\033[0m'
else
  G=""; Y=""; R=""; B=""; D=""; N=""
fi

# Each entry: "Name|cost|space-separated test files"
BATCHES=(
  "Metadata — account, keys, billing, models, characters|free|tests/integration/test_account_vcr.py tests/integration/test_api_keys_vcr.py tests/integration/test_billing_vcr.py tests/integration/test_models_vcr.py tests/integration/test_model_selection_vcr.py tests/integration/test_characters_vcr.py"
  "Client & infra — http, scheduler, rate limiting, resilience|free/cheap|tests/integration/test_base_client_vcr.py tests/integration/test_http_client_vcr.py tests/integration/test_observability_vcr.py tests/integration/test_scheduler_vcr.py tests/integration/test_rate_limiter_vcr.py tests/integration/test_resilience_vcr.py tests/integration/test_circuit_breaker_recovery.py tests/integration/test_concurrent_requests.py tests/integration/test_rate_limit_edge_cases.py tests/integration/test_429_handling.py tests/integration/test_shared_state_verification.py"
  "Text inference — chat, embeddings, responses, augment, decisions|cheap|tests/integration/test_chat_completions_vcr.py tests/integration/test_embeddings_vcr.py tests/integration/test_responses_vcr.py tests/integration/test_augment_vcr.py tests/integration/test_decisions_vcr.py"
  "Image — generate + upscale|moderate \$|tests/integration/test_image_vcr.py"
  "Audio — TTS/STT|moderate \$|tests/integration/test_audio_resource_vcr.py tests/e2e/test_audio_e2e.py tests/e2e/test_audio_helpers_e2e.py"
  "Music — generation (async jobs)|expensive \$\$|tests/integration/test_music_vcr.py"
  "Video — generation (async jobs)|expensive \$\$\$|tests/e2e/test_video_e2e.py"
  "x402 — wallet / on-chain billing|special (needs wallet)|tests/integration/test_x402_vcr.py"
)

TOTAL=${#BATCHES[@]}
PYTEST_ARGS=(--no-cov -p no:cacheprovider -q --show-capture=no)

# Tests that cannot be recorded/replayed from a cassette offline because they
# rely on live network behaviour in normal operation — e.g. requests that time
# out client-side by design (no HTTP response ever comes back to record). Real
# CI runs these live anyway (`make test-ci` uses --disable-vcr), so they are
# excluded from both the record and the offline replay-verify steps here.
DESELECT=(
  "tests/integration/test_billing_vcr.py::test_billing_get_usage_empty_date_range"
)
DESELECT_ARGS=()
for _t in "${DESELECT[@]}"; do DESELECT_ARGS+=(--deselect "$_t"); done

print_plan() {
  echo "${B}Re-record plan (${TOTAL} batches, cheap → expensive):${N}"
  local i
  for (( i = 1; i <= TOTAL; i++ )); do
    IFS='|' read -r name cost files <<< "${BATCHES[$((i-1))]}"
    local -a farr; read -ra farr <<< "$files"
    printf "  ${Y}%2d${N}  %-58s ${D}%-22s %2d file(s)${N}\n" "$i" "$name" "$cost" "${#farr[@]}"
  done
}

# Record a batch live, then verify it replays offline. Returns 0 if both pass.
run_batch() {
  local files="$1"
  echo "${B}── Recording live (VENICE_VCR_RECORD=all, auto-retry transients) ─${N}"
  # --reruns/--reruns-delay (pytest-rerunfailures) gives the RECORD step a
  # delayed-retry so a fresh re-record isn't derailed by live-API flakiness:
  # a failed live call (timeout, 429, 500, or a randomly-drawn task-incapable
  # model) re-fires up to 2x with a 10s backoff. Each rerun re-draws the random
  # test model, so a wrong-model 400 often clears on retry too. Only the RECORD
  # step retries — the replay-verify below stays strict (a flaky replay is a
  # real bug, not transient). backup-restore makes reruns safe: a failed attempt
  # never destroys the prior cassette. (Genuinely non-transient failures — e.g.
  # 402 insufficient balance, or a persistent provider/API bug — still fail
  # after the retries, which is correct: they need a real fix, not a retry.)
  # shellcheck disable=SC2086
  VENICE_CI_MODE=false VENICE_VCR_RECORD=all \
    poetry run pytest $files "${PYTEST_ARGS[@]}" "${DESELECT_ARGS[@]}" --reruns 2 --reruns-delay 10
  local rec=$?

  echo
  echo "${B}── Verifying replay (VENICE_CI_MODE=true, no network) ──${N}"
  # CI mode forces RecordMode.NONE regardless of VENICE_VCR_RECORD, so replay
  # is safe even if VENICE_VCR_RECORD is exported in the caller's environment.
  # shellcheck disable=SC2086
  VENICE_CI_MODE=true \
    poetry run pytest $files "${PYTEST_ARGS[@]}" "${DESELECT_ARGS[@]}"
  local rep=$?

  (( rec == 0 && rep == 0 ))
}

case "${1:-}" in
  -h|--help) sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  -l|--list) print_plan; exit 0 ;;
esac

if [[ -z "${VENICE_API_KEY:-}" ]]; then
  echo "${R}ERROR: VENICE_API_KEY is not set.${N}"
  echo "Usage: VENICE_API_KEY=sk-... $0 [START_BATCH]"
  exit 1
fi

START="${1:-1}"
if ! [[ "$START" =~ ^[0-9]+$ ]] || (( START < 1 || START > TOTAL )); then
  echo "${R}ERROR: START_BATCH must be between 1 and ${TOTAL}.${N}"
  exit 1
fi

print_plan
echo
echo "${D}A failed call writes no cassette (record_on_exception=False) — fix drift and retry.${N}"

declare -a STATUS
for (( j = 0; j < TOTAL; j++ )); do STATUS[j]="·"; done

i=$START
while (( i <= TOTAL )); do
  IFS='|' read -r name cost files <<< "${BATCHES[$((i-1))]}"
  read -ra file_arr <<< "$files"
  echo
  echo "${Y}════════════════════════════════════════════════════════${N}"
  echo "${Y} Batch ${i}/${TOTAL}: ${name}${N}"
  echo "${Y} Cost: ${cost}  ·  ${#file_arr[@]} test file(s)${N}"
  echo "${Y}════════════════════════════════════════════════════════${N}"

  if [[ -z "${NONINTERACTIVE:-}" ]]; then
    while true; do
      printf "%s" "${B}Record this batch? [y]es · [s]kip · [q]uit > ${N}"
      read -r choice </dev/tty || choice="q"
      case "$choice" in
        y|Y|"") break ;;
        s|S) STATUS[i-1]="skip"; echo "${D}Skipped.${N}"; choice="skip"; break ;;
        q|Q) echo "Stopping at batch ${i}."; choice="quit"; break ;;
        *) echo "Please enter y, s, or q." ;;
      esac
    done
    [[ "$choice" == "skip" ]] && { (( i++ )); continue; }
    [[ "$choice" == "quit" ]] && break
  fi

  if run_batch "$files"; then
    STATUS[i-1]="${G}pass${N}"
    echo "${G}✓ Batch ${i} recorded and replay-verified.${N}"
    (( i++ ))
  else
    STATUS[i-1]="${R}FAIL${N}"
    echo "${R}✗ Batch ${i} had failures (see output above). No cassette written for the failed call(s).${N}"
    if [[ -n "${NONINTERACTIVE:-}" ]]; then
      (( i++ )); continue
    fi
    while true; do
      printf "%s" "${B}[r]etry this batch · [c]ontinue anyway · [q]uit > ${N}"
      read -r choice </dev/tty || choice="q"
      case "$choice" in
        r|R|"") break ;;                       # re-run same batch
        c|C) (( i++ )); break ;;
        q|Q) echo "Stopping at batch ${i}."; i=$((TOTAL + 1)); break ;;
        *) echo "Please enter r, c, or q." ;;
      esac
    done
  fi
done

echo
echo "${B}Summary:${N}"
for (( j = 0; j < TOTAL; j++ )); do
  IFS='|' read -r name cost files <<< "${BATCHES[$j]}"
  printf "  %2d  %-58s [%b]\n" "$((j+1))" "$name" "${STATUS[$j]}"
done
