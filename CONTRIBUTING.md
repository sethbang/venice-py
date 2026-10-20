# Contributing to the Venice AI Python Client

Thank you for your interest in contributing to the Venice AI Python Client!

- Questions about using the SDK: see [SUPPORT.md](SUPPORT.md).
- Security vulnerabilities: report them privately as described in [SECURITY.md](SECURITY.md), never in a public issue.
- Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Maintainers cutting a release: see [RELEASING.md](RELEASING.md).

## Reporting Issues

If you encounter a bug or have an issue with the library:

1. **Search Existing Issues:** Before submitting a new issue, please check if a similar one has already been reported by searching on GitHub under [Issues](https://github.com/sethbang/venice-py/issues).
2. **Open a New Issue:** If you can't find an existing issue that addresses your problem, please [open a new one](https://github.com/sethbang/venice-py/issues/new).
   - Provide a **clear and descriptive title**.
   - Include a **detailed description** of the issue.
   - If applicable, provide a **code sample or an executable test case** that demonstrates the problem.
   - Mention your **Python version, library version, and operating system**.

## Development Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management and requires **Python ≥3.11**.

```bash
# Install dependencies, dev tools and all extras
make install

# Activate the virtual environment (Poetry 2.0+)
eval "$(poetry env activate)"
```

## Running Tests

Use the Makefile targets rather than calling `pytest` directly. They set the parallelism, coverage and ignore flags CI uses. `make help` lists every target.

```bash
make install     # install dependencies, dev tools and all extras
make test-unit   # unit tests (tests/unit/)
make test-fast   # every test not marked slow
```

The CI gate on every pull request also runs `tests/cli/`, `tests/property/`, `tests/provider/` and `tests/presets/`. Pass them through `ARGS` to run the same set locally:

```bash
make test-unit ARGS="tests/cli/ tests/property/ tests/provider/ tests/presets/"
```

The project's 90% coverage threshold applies to the full suite, so a unit-only run can exit non-zero on coverage alone. Read pytest's pass/fail summary line, not just the exit code.

**Integration and end-to-end tests** (`tests/integration/`, `tests/e2e/`) replay recorded VCR cassettes. The cassettes are gitignored, so on a fresh clone those tests skip or fail, and they are not part of the PR gate. Recording or refreshing cassettes calls the live Venice API with your `VENICE_API_KEY` and **spends real credit**. So do the live targets `make test-ci`, `make test-live-smoke`, `make test-quick` and `make test-fresh`. You do not need to run any of them to contribute. If a change can only be shown live, use the cheapest model and the smallest request that demonstrates it, and note the calls and approximate cost in the PR.

## Code Style

This project enforces code style with **ruff** and type correctness with **mypy** and **pyright**.

```bash
make format              # ruff format + ruff check --fix
make lint                # ruff check
make type-check          # mypy over the venice_ai package
make type-check-strict   # strict mypy on critical modules (a required CI check)
make type-check-pyright  # pyright over src/ (a required CI check)
make check-all           # format, lint, both type checkers, import check, dead-code check, security scan
```

Run `make check-all` and `make type-check-strict` before opening a PR. `check-all` does not include the strict mypy pass, which is a separate required check. It starts with `make format`, so commit any files it rewrites.

## Pull Request Guidelines

- Branch naming: use descriptive names such as `fix/issue-description` or `feat/short-feature-name`.
- Keep PRs focused — one logical change per PR makes review easier.
- Add or update tests to cover your changes.
- Ensure `make check-all`, `make type-check-strict` and `make test-unit` pass locally before opening the PR. Five CI checks are required to merge: `validate (3.13)`, `Build sdist + wheel`, `mypy --strict (critical modules)`, `pyright (project)` and `Vulture (dead-code gate)`.
- Update `CHANGELOG.md` under the `[Unreleased]` section if your change is user-visible.

## Questions?

If you have general questions about using the library that aren't bug reports, you can open an issue for discussion.

---

Your contributions are valuable!
