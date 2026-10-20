# Maintainer tools

Scripts in this directory support development and CI. They are not part of the published package.

## API drift scan (`api_drift.py`)

`api_drift.py` compares the live Venice OpenAPI spec (`https://api.venice.ai/api/v1/swagger.yaml`, public, no API key) with the committed snapshot `api_drift_snapshot.json`. The snapshot is a normalized summary of the spec:

- every endpoint (`METHOD /path`)
- its request parameters (`query.*`, `path.*`, `header.*`, `body.*`)
- its success-response field paths and content types
- the members of every enum

Body and response fields are dotted paths: `[]` marks array items and `{}` marks map values. The script follows `$ref` and merges `allOf` / `anyOf` / `oneOf` branches, so a field that exists in only one branch of a union still counts.

```bash
poetry run python tools/api_drift.py                      # fetch the live spec and compare
poetry run python tools/api_drift.py --spec swagger.yaml  # compare a local copy
poetry run python tools/api_drift.py --update-snapshot    # rewrite the snapshot from the live spec
```

| Exit code | Meaning |
|---|---|
| 0 | No drift |
| 1 | Drift found. The Markdown report goes to stdout, and also to a file with `--report PATH`. |
| 2 | The spec or snapshot could not be read. Causes: a non-200 response, an HTML or empty body, invalid YAML, or a document with no `paths`. |

The [`API Drift`](../.github/workflows/api-drift.yml) workflow runs the scan daily and on manual dispatch. When it finds drift, it opens a single issue labelled `api-drift`, or updates the one that is already open. It posts a comment only when the drift itself changes. If the spec cannot be fetched or parsed, the run fails and the issue is left alone.

### Handling an `api-drift` issue

1. **Read the report** in the issue (the full version is in the linked run's summary). Decide whether each change affects the SDK:
   - **Added endpoints:** add a resource method.
   - **Removed endpoints:** deprecate or remove the method.
   - **New params or enum members:** expose them in the request models.
   - **Removed or renamed fields:** handle them in the response models.
   - **Documentation-only reshuffles:** nothing to change in the SDK.

   The spec and the API do not always agree, so confirm anything doubtful before you change the SDK.
2. **Update the SDK** with tests, as for any other change.
3. **Refresh the snapshot in the same PR:**

   ```bash
   poetry run python tools/api_drift.py --update-snapshot
   git add tools/api_drift_snapshot.json
   ```

   The snapshot diff shows reviewers exactly which spec changes the PR accepts.
4. Merging the PR does not close the issue on its own, so add `Closes #<issue>` to the PR description. If the issue is still open when Venice changes the spec again, the next run updates it.

If some drift is deliberately not mirrored in the SDK, refresh the snapshot anyway, so that the drift is not reported every day, and say why in the PR description.
