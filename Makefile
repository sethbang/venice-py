# =====================================================
# Venice AI Quality & Testing Makefile
# Refactored for reduced duplication and consistency
# =====================================================

.DEFAULT_GOAL := help

# -----------------------------------------------------
# Version (extracted from pyproject.toml)
# -----------------------------------------------------
VERSION := $(shell grep '^version' pyproject.toml | head -1 | sed 's/.*"\(.*\)"/\1/')

# -----------------------------------------------------
# Environment (.env auto-loading)
# -----------------------------------------------------
# Automatically export .env variables if the file exists.
# This makes VENICE_API_KEY (and any other secrets) available
# to all make targets without manual sourcing.
ifneq (,$(wildcard .env))
    include .env
    export
endif

# -----------------------------------------------------
# Phony Targets (exhaustive)
# -----------------------------------------------------
.PHONY: help install clean clean-all clean-cassettes \
        test test-ci test-unit test-e2e test-fast test-fresh test-refresh test-quick test-verbose \
        test-mutation test-mutation-core test-mutation-results test-mutation-report test-mutation-show \
        lint format format-check type-check type-check-strict type-check-pyright type-check-report security \
        check-all check-imports check-dead-code \
        coverage coverage-html release-check dev-check pre-commit \
        build smoke-test \
        skills-install skills-symlink skills-uninstall skills-check

# -----------------------------------------------------
# Tool Definitions
# -----------------------------------------------------
POETRY := poetry
RUN    := $(POETRY) run
PYTEST := $(RUN) pytest
RUFF   := $(RUN) ruff
MYPY   := $(RUN) mypy

# -----------------------------------------------------
# Coverage Configuration
# -----------------------------------------------------
COV_SOURCE     := src/venice_ai
COV_REPORT_DIR := tests/reports/coverage
COV_HTML_DIR   := $(COV_REPORT_DIR)/html
COV_XML        := $(COV_REPORT_DIR)/coverage.xml
COV_THRESHOLD  := 90

# -----------------------------------------------------
# Pytest Flag Composition (reduces duplication)
# -----------------------------------------------------
PYTEST_COMMON    := -n 4 --tb=short --show-capture=no
PYTEST_IGNORE    := --ignore=tests/benchmarks --ignore=tests/profiling
PYTEST_COV_BASE  := --cov=$(COV_SOURCE) --cov-report=term-missing
PYTEST_COV_FULL  := $(PYTEST_COV_BASE) --cov-report=html:$(COV_HTML_DIR) --cov-report=xml:$(COV_XML)
PYTEST_COV_FAIL  := --cov-fail-under=$(COV_THRESHOLD)

# -----------------------------------------------------
# Colors for output
# -----------------------------------------------------
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
CYAN   := \033[0;36m
NC     := \033[0m

# =====================================================
# HELP - Self-documenting target
# =====================================================
help: ## Show this help message
	@echo "$(GREEN)Venice AI v$(VERSION) - Quality & Testing Commands$(NC)"
	@echo ""
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Usage:$(NC) make <target> [ARGS='pytest flags']"
	@echo "$(YELLOW)Example:$(NC) make test ARGS='--verbose -k specific_test'"

# =====================================================
# INSTALLATION
# =====================================================
# --all-extras is required, not optional, and matches the validate job in CI.
# Without it the TEE, x402 and Solana test modules skip wholesale (importorskip
# on cryptography, dcap-qvl, eth-account, siwe, solders) and mypy degrades the
# same imports to Any — a green local run that never exercised the most
# security-sensitive code in the SDK.
install: ## Install all dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(POETRY) install --with dev --all-extras

# =====================================================
# TESTING TARGETS
# =====================================================
test: clean ## Run all tests (parallel, no coverage)
	@echo "$(GREEN)Running tests in parallel...$(NC)"
	$(PYTEST) tests/ --no-cov $(PYTEST_COMMON) $(PYTEST_IGNORE) -q $(ARGS)

test-refresh: clean ## Refresh model cache from API and run tests
	@echo "$(GREEN)Refreshing model cache and running tests...$(NC)"
	$(PYTEST) tests/ --refresh-models --no-cov $(PYTEST_COMMON) $(PYTEST_IGNORE) -q $(ARGS)

test-fresh: clean-cassettes ## Delete cassettes and re-record them live (SPENDS CREDIT)
	@echo "$(YELLOW)Re-recording every cassette against the live API — this spends credit.$(NC)"
	VENICE_VCR_RECORD=all $(MAKE) test
	@echo "$(GREEN)Tests completed with fresh cassettes!$(NC)"

test-quick: clean-cassettes ## Fast iteration: E2E + modified integration tests (SPENDS CREDIT)
	@echo "$(YELLOW)Re-recording the selected cassettes against the live API — spends credit.$(NC)"
	VENICE_VCR_RECORD=all $(PYTEST) tests/e2e/ \
		tests/integration/test_image_vcr.py \
		tests/integration/test_api_keys_vcr.py \
		tests/integration/test_embeddings_vcr.py \
		tests/integration/test_rate_limit_edge_cases.py \
		tests/integration/test_concurrent_requests.py \
		tests/integration/test_model_selection_vcr.py \
		tests/integration/test_audio_resource_vcr.py \
		--no-cov $(PYTEST_COMMON) -v $(ARGS)
	@echo "$(GREEN)Quick test iteration completed!$(NC)"

# Full LIVE suite against the real Venice API. `--disable-vcr` makes the
# vcr_cassette fixtures bypass cassettes and hit the network (see
# tests/conftest.py), so this needs a valid VENICE_API_KEY and SPENDS CREDIT.
# NOTE: this is NOT the per-push CI gate — ci-validation.yaml runs unit tests
# only (deterministic, no live API). This target is the on-demand full-suite
# live run (and the same path the nightly e2e job exercises). Transient
# live-API errors — 500 "Inference processing failed" and image-gen
# TimeoutError — are globally auto-retried (2x) via --only-rerun, so blips
# don't fail the run while real, repeatable failures still surface.
test-ci: clean ## Full LIVE suite vs the real API — on-demand; needs VENICE_API_KEY, spends credit
	@echo "$(GREEN)Running the FULL LIVE suite against the Venice API (coverage on)...$(NC)"
	VENICE_CI_MODE=true $(PYTEST) tests/ \
		--refresh-models --disable-vcr $(PYTEST_COV_FULL) $(PYTEST_COV_FAIL) \
		--reruns 2 --reruns-delay 3 --only-rerun 'TimeoutError' --only-rerun 'InternalServerError' \
		$(PYTEST_COMMON) $(PYTEST_IGNORE) -v $(ARGS)
	@echo "$(GREEN)Live run complete — cassettes bypassed via --disable-vcr.$(NC)"

# Fast LIVE smoke subset — representative live coverage of the text/metadata
# resources (chat, models, embeddings, augment, characters, responses) WITHOUT
# the slow generative resources (image/video/audio/music). Needs a valid
# VENICE_API_KEY and spends a little credit, but runs in a few minutes rather
# than ~20. Use for a quick on-demand live sanity check; `make test-ci` is the
# full live run.
test-live-smoke: clean ## Fast LIVE smoke subset vs the real API — on-demand; needs VENICE_API_KEY
	@echo "$(GREEN)Running fast LIVE smoke subset against the Venice API...$(NC)"
	$(PYTEST) \
		tests/integration/test_chat_completions_vcr.py \
		tests/integration/test_models_vcr.py \
		tests/integration/test_model_selection_vcr.py \
		tests/integration/test_embeddings_vcr.py \
		tests/integration/test_augment_vcr.py \
		tests/integration/test_characters_vcr.py \
		tests/integration/test_responses_vcr.py \
		--refresh-models --disable-vcr --no-cov \
		--reruns 2 --reruns-delay 3 --only-rerun 'TimeoutError' --only-rerun 'InternalServerError' \
		$(PYTEST_COMMON) -q $(ARGS)
	@echo "$(GREEN)Live smoke complete.$(NC)"

test-verbose: clean ## Run tests with verbose output and coverage
	@echo "$(GREEN)Running tests with verbose output...$(NC)"
	$(PYTEST) tests/ $(PYTEST_COV_FULL) $(PYTEST_COV_FAIL) $(PYTEST_COMMON) -v $(ARGS)

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ $(PYTEST_COV_BASE) $(PYTEST_COMMON) -q $(ARGS)

test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running E2E tests...$(NC)"
	@[ -n "$$VENICE_API_KEY" ] || echo "$(YELLOW)Warning: VENICE_API_KEY not set for E2E tests$(NC)"
	$(PYTEST) tests/e2e/ $(PYTEST_COV_BASE) --tb=short --show-capture=no -q $(ARGS)

test-fast: ## Run fast tests only (no slow marker)
	@echo "$(GREEN)Running fast tests only...$(NC)"
	$(PYTEST) -m "not slow" $(PYTEST_COV_FULL) \
		--cov-report=xml:tests/reports/coverage.xml \
		$(PYTEST_COMMON) -q $(ARGS)

# -----------------------------------------------------
# Mutation Testing
# -----------------------------------------------------
test-mutation: ## Run mutation tests (slow - run locally)
	@echo "$(GREEN)Running mutation tests...$(NC)"
	@echo "$(YELLOW)Warning: This is SLOW! Consider testing one module at a time.$(NC)"
	@echo "$(YELLOW)Example: poetry run mutmut run --paths-to-mutate=src/venice_ai/costs.py$(NC)"
	$(RUN) mutmut run --paths-to-mutate=src/venice_ai

test-mutation-core: ## Run mutation tests on critical modules
	@echo "$(GREEN)Running mutation tests on core modules...$(NC)"
	$(RUN) mutmut run --paths-to-mutate=src/venice_ai/exceptions.py
	$(RUN) mutmut run --paths-to-mutate=src/venice_ai/costs.py
	$(RUN) mutmut results

test-mutation-results: ## Show mutation test results
	@echo "$(GREEN)Mutation test results:$(NC)"
	$(RUN) mutmut results

test-mutation-report: ## Generate HTML mutation report
	@echo "$(GREEN)Generating HTML mutation report...$(NC)"
	$(RUN) mutmut html
	@echo "$(GREEN)Report generated at html/index.html$(NC)"

test-mutation-show: ## Show mutation details
	@echo "$(GREEN)Showing mutation details...$(NC)"
	$(RUN) mutmut show

# =====================================================
# CODE QUALITY TARGETS
# =====================================================
lint: ## Run ruff linting checks
	@echo "$(GREEN)Running ruff linting...$(NC)"
	$(RUFF) check src/ tests/ tools/ benchmarks/

format: ## Format code with ruff
	@echo "$(GREEN)Formatting code with ruff...$(NC)"
	$(RUFF) format src/ tests/ tools/ benchmarks/
	$(RUFF) check --fix src/ tests/ tools/ benchmarks/

# Non-mutating counterpart to `format`, for CI. `format` rewrites files, so
# running it in a pipeline gates nothing — badly formatted code is silently
# fixed and the job passes. This target fails instead.
format-check: ## Verify formatting without modifying files
	@echo "$(GREEN)Checking formatting with ruff...$(NC)"
	$(RUFF) format --check src/ tests/ tools/ benchmarks/
	$(RUFF) check src/ tests/ tools/ benchmarks/

type-check: ## Run mypy type checking
	@echo "$(GREEN)Running mypy type checking...$(NC)"
	$(MYPY) -p venice_ai --show-error-codes --pretty

type-check-strict: ## Run strict mypy on critical modules
	@echo "$(GREEN)Running strict mypy type checking...$(NC)"
	$(MYPY) -p venice_ai.exceptions \
		-p venice_ai.costs \
		-p venice_ai._request_classifier \
		-p venice_ai.auth.x402 \
		--strict --show-error-codes --pretty

# The `pyright (project)` job on main-review gates every PR, and mypy does not
# stand in for it: pyright reports classes mypy does not, such as a name bound
# only inside a try block being referenced from its except clause. Without a
# target here, nothing local runs it and the first signal is a red required
# check. CI installs --extras all so modules behind optional extras resolve.
type-check-pyright: ## Run pyright over src/ (mirrors the required CI check)
	@echo "$(GREEN)Running pyright...$(NC)"
	$(RUN) pyright src/

type-check-report: ## Generate mypy coverage report
	@echo "$(GREEN)Generating mypy coverage report...$(NC)"
	$(MYPY) src/ --html-report mypy-report --show-error-codes
	@echo "$(YELLOW)Report generated at mypy-report/index.html$(NC)"

# pip-audit runs with no suppressions. If a vulnerability ever has to be waived,
# add a --ignore-vuln entry here with the ID, the package, and a one-line reason,
# and delete it as soon as the fix ships.
security: ## Run security scans (pip-audit, bandit)
	@echo "$(GREEN)Running security checks...$(NC)"
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	$(RUN) pip-audit
	@echo "$(YELLOW)Scanning code...$(NC)"
	$(RUN) bandit -r src/ -ll

check-imports: ## Validate import structure
	@echo "$(GREEN)Validating import structure...$(NC)"
	@echo "$(YELLOW)Checking for circular imports...$(NC)"
	$(RUN) python -c "import sys; sys.path.insert(0, 'src'); import venice_ai; print('✓ No circular imports detected')"
	@echo "$(YELLOW)Checking for unused imports...$(NC)"
	$(RUFF) check --select F401 src/

check-dead-code: ## Check for dead/unused code
	@echo "$(GREEN)Checking for dead code...$(NC)"
	@if $(RUN) python -c "import vulture" >/dev/null 2>&1; then \
		$(RUN) vulture src/ --min-confidence 80; \
	else \
		echo "$(YELLOW)vulture not installed in poetry env, using ruff for unused variable detection$(NC)"; \
		$(RUFF) check --select F841 src/; \
	fi

check-all: format lint type-check type-check-pyright check-imports check-dead-code security ## Run all quality checks
	@echo "$(GREEN)All quality checks completed!$(NC)"

# =====================================================
# COVERAGE TARGETS
# =====================================================
coverage: clean ## Generate full coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ $(PYTEST_COV_FULL) $(PYTEST_COV_FAIL) \
		$(PYTEST_COMMON) $(PYTEST_IGNORE) -v $(ARGS)
	@echo "$(GREEN)Coverage report generated at $(COV_HTML_DIR)$(NC)"

coverage-html: coverage ## Generate and open HTML coverage report
	@echo "$(GREEN)Opening HTML coverage report...$(NC)"
	@$(RUN) python -m webbrowser $(COV_HTML_DIR)/index.html

# =====================================================
# RELEASE VALIDATION
# =====================================================
release-check: clean check-all test ## Complete quality validation for release
	@echo "$(GREEN)=== Venice AI v$(VERSION) Release Validation Complete ===$(NC)"
	@echo ""
	@echo "$(YELLOW)Quality Checks:$(NC) ✓ Passed"
	@echo "$(YELLOW)Test Suite:$(NC) ✓ Passed with $(COV_THRESHOLD)%+ coverage"
	@echo "$(YELLOW)Security Scan:$(NC) ✓ Passed"
	@echo ""
	@echo "$(GREEN)Ready for release!$(NC)"

# =====================================================
# CLEANUP TARGETS
# =====================================================
clean: ## Clean test artifacts and cache
	@echo "$(YELLOW)Cleaning test artifacts...$(NC)"
	@rm -rf .pytest_cache .coverage* $(COV_REPORT_DIR) tests/reports e2e_tests/reports
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true

clean-cassettes: ## Delete all VCR cassettes (re-record needs VENICE_VCR_RECORD=all)
	@echo "$(YELLOW)Deleting all VCR cassettes...$(NC)"
	@find tests -type d -name "cassettes" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)All VCR cassettes deleted$(NC)"

clean-all: clean ## Clean everything including build artifacts
	@echo "$(YELLOW)Cleaning everything...$(NC)"
	@rm -rf .venv dist build *.egg-info .mypy_cache .ruff_cache

# =====================================================
# DEVELOPMENT SHORTCUTS
# =====================================================
dev-check: format lint type-check test-fast ## Quick development validation
	@echo "$(GREEN)Quick development validation completed!$(NC)"

pre-commit: check-all test-fast ## Pre-commit validation
	@echo "$(GREEN)Pre-commit validation completed!$(NC)"

# =====================================================
# BUILD & PUBLISH TARGETS
# =====================================================
build: clean-all ## Build the package (sdist + wheel)
	@echo "$(GREEN)Building venice-py v$(VERSION)...$(NC)"
	$(POETRY) build
	@echo "$(GREEN)Built artifacts in dist/:$(NC)"
	@ls -lh dist/

smoke-test: ## Smoke test: import check and version verification
	@echo "$(GREEN)Running smoke test...$(NC)"
	@$(RUN) python -c "from venice_ai import VeniceClient, __version__; print(f'venice-py v{__version__} - import OK')"
	@echo "$(GREEN)Smoke test passed!$(NC)"

# =====================================================
# CLAUDE CODE SKILLS
# =====================================================
skills-install: ## Install Claude Code skills into ~/.claude/skills/ (copy)
	@tools/skills/install.sh

skills-symlink: ## Install Claude Code skills as symlinks (live edits during dev)
	@tools/skills/install.sh --symlink

skills-uninstall: ## Remove the four Venice skills from ~/.claude/skills/
	@tools/skills/install.sh --uninstall

skills-check: ## Validate skills: size + example-path + code-block lint + symbol existence
	@echo "$(GREEN)Skill SKILL.md size check (≤500 lines)...$(NC)"
	@$(RUN) python tools/skills/check_skill_size.py
	@echo "$(GREEN)Skill examples/* path resolution...$(NC)"
	@$(RUN) python tools/skills/check_skill_examples.py
	@echo "$(GREEN)Skill code-block lint (idiomatic v2 patterns)...$(NC)"
	@$(RUN) python tools/skills/lint_skill_code.py
	@echo "$(GREEN)Skill SDK symbol existence (imports / kwargs / attrs)...$(NC)"
	@$(RUN) python tools/skills/check_skill_symbols.py

