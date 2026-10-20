# Releasing venice-py

This runbook is for maintainers with write access to `sethbang/venice-py` and approval rights on its `pypi` and `testpypi` environments. Contributors do not need it.

**Publishing a GitHub Release is what publishes to PyPI.** [`release.yaml`](.github/workflows/release.yaml) runs on the `release: published` event. Pushing a tag does nothing on its own, and merging to `main` does nothing either. PyPI never lets a version number be reused, even after a delete, so every step before you publish the Release exists to catch mistakes while they are still free.

## 1. Prepare the release PR

On a branch off an up-to-date `main` (for example `release/2.7.0`):

1. **Bump the version in `pyproject.toml`, and only there.** `venice_ai.__version__` is read from the installed package metadata, and the Makefile reads `VERSION` from `pyproject.toml`, so neither needs editing. Do not add a `__version__` literal. Run `make install`, then `make smoke-test`, and check that the version it prints matches.
2. **Update `CHANGELOG.md`** ([Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and SemVer):
   - Rename the `## [Unreleased]` entries to `## [X.Y.Z] - YYYY-MM-DD`, using the date you will publish.
   - Keep the usual subsections (`### Added`, `### Changed`, `### Fixed`, `### Removed`, plus `### Upgrading` notes for anything breaking or behavior-changing).
   - At the bottom of the file, point `[Unreleased]` at `compare/vX.Y.Z...HEAD` and add `[X.Y.Z]: https://github.com/sethbang/venice-py/compare/vPREV...vX.Y.Z`.
3. **Run the local gates:** `make check-all` (it formats first, so commit what it rewrites), `make type-check-strict`, `make test-unit`, and finally `make build` (sdist and wheel in `dist/`). `make build` runs `clean-all`, which also deletes `.venv`, so run `make install` again afterwards. `make release-check` runs `check-all` and `make test` together. `make test` replays the integration and e2e cassettes, which are gitignored and exist only on a machine that recorded them. On any other machine those tests skip (no `VENICE_API_KEY`) or fail (key set, no cassette). Do not run the live targets (`test-ci`, `test-live-smoke`, `test-quick`, `test-fresh`) unless the release needs them: they spend real Venice credit.
4. **Open a PR to `main`.** The `main-review` ruleset requires a PR and these five checks, matched by display name:
   - `validate (3.13)`
   - `Build sdist + wheel`
   - `mypy --strict (critical modules)`
   - `pyright (project)`
   - `Vulture (dead-code gate)`

   Some checks can take a minute to appear in the list. Wait for all five by name, not for "everything currently listed is green". Required approvals are 0, because GitHub does not let a PR's author approve it and the project has one maintainer. Admins can bypass the ruleset, but releases should not: go through the PR and the checks. The `main-integrity` ruleset blocks force-pushes and deletion of `main` for everyone.
5. **Merge the PR.** Nothing is published yet.

## 2. Optional: dry run on TestPyPI

Do this for a release that changes packaging, dependencies, extras, the Python floor, or either publish workflow.

1. On a branch, set the version to a pre-release such as `X.Y.Zrc1`. [`publish-testpypi.yaml`](.github/workflows/publish-testpypi.yaml) refuses any version that is not an `rc`, `a` or `b` pre-release.
2. In **Actions → Publish to TestPyPI → Run workflow**, choose that branch and type `publish-rc` in the confirmation field.
3. Approve the `testpypi` environment deployment when prompted.
4. Install from TestPyPI in a clean virtualenv using the command the workflow prints, and import the package.
5. TestPyPI versions cannot be reused either. Use `rc2`, `rc3` and so on for each retry. Do not merge the rc version bump to `main`.

## 3. Publish the GitHub Release

1. **Releases → Draft a new release.**
2. **Tag:** `vX.Y.Z`, created on publish, targeting `main` at the merged release commit. The leading `v` is required. The `pypi` environment only accepts deployments from tags matching `v*`. The `version-check` job strips the `v` and fails if the result differs from `pyproject.toml`.
3. **Title:** `vX.Y.Z — <one-line headline>`, matching earlier releases.
4. **Notes:** a short summary followed by the CHANGELOG section, with an **Upgrading** section when anything changed behavior.
5. **Save as draft** first and read it through. Publishing is the trigger, so the draft is your last chance to catch a problem.
6. **Publish.** Ticking *Set as a pre-release* does **not** stop the PyPI upload, because pre-releases also fire `published`.

Never create the tag by pushing it, and never run `git push --tags`. A pushed tag publishes nothing, and it can leave a tag pointing at the wrong commit for the Release to reuse.

## 4. Approve and verify

1. In **Actions → Publish to PyPI**, check that the `version-check` job (*Verify pyproject.version matches release tag*) passed, then approve the `pypi` environment deployment. The job waits for that approval.
2. When the job finishes, confirm:
   - <https://pypi.org/project/venice-py/> shows the new version, with both an sdist and a wheel.
   - In a fresh virtualenv, `pip install "venice-py==X.Y.Z"` and then `python -c "import venice_ai; print(venice_ai.__version__)"` prints `X.Y.Z`.
   - The docs site (<https://venice-docs.sbang.dev/>) was redeployed by the `Docs` workflow on the merge to `main`, if `website/` or `src/` changed.

## If something goes wrong

- **`version-check` failed:** the tag and `pyproject.toml` disagree, and nothing was uploaded. Delete the Release and its tag, fix whichever is wrong through a PR if the version needs changing, and publish again.
- **The upload failed with a Trusted Publishing error:**
  - A *token exchange* error means no publisher matched. Check every field against the header of `release.yaml`. The workflow is `release.yaml`, not `release.yml`: this repo uses both extensions, and a publisher recorded with the wrong one never matches.
  - A *403 "OIDC scoped token is not valid for project 'venice-py'"* means a publisher matched but is registered on a different PyPI project. Register it from the `venice-py` project's own Publishing settings page.
  - A publisher is proven only by a successful upload. After a repository rename, publishers registered under the old name stop matching and must be re-added.
- **The job was skipped, not failed:** both publish workflows only run when `github.repository_owner == 'sethbang'`, so forks never publish. If they skip in this repository, check the owner.
- **A broken release reached PyPI:** fix forward. A version cannot be re-uploaded, so:
  1. Yank the bad version on PyPI (**Manage project → Releases → Options → Yank**, which requires the PyPI owner account). Yanking keeps exact pins working but stops new resolutions from choosing that version. Give a reason that points to the fixed version.
  2. Ship `X.Y.Z+1` through this runbook.
  3. Edit the bad GitHub Release's notes to point to the fix. Deleting the Release or its tag does not remove anything from PyPI.
  4. For a security issue, publish a GitHub Security Advisory as described in [SECURITY.md](SECURITY.md).

## The `venice-ai` bridge package

The project was published as `venice-ai` before it was renamed `venice-py`. `venice-ai` 2.1.1 is a metadata-only bridge: it ships no code and depends on `venice-py>=2.2.0`, so `pip install venice-ai` keeps working. It is never yanked. Its source is not in this repository, and it is not part of a normal release.

Republish the bridge only when its metadata must change. The usual reason is a change to venice-py's supported Python range. The bridge's `Requires-Python` must not be narrower than venice-py's. If it is, pip skips the bridge on the excluded Python versions and falls back to the old v1 `venice-ai` line. Publish the bridge only after the `venice-py` version it depends on is live on PyPI.
