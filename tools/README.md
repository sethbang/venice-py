# Maintainer tools

Scripts in this directory support development and CI. They are not part of the published package.

## API drift scan (`api_drift.py`)

`api_drift.py` compares the live Venice OpenAPI spec (`https://api.venice.ai/api/v1/swagger.yaml`, public, no API key) with the committed snapshot `api_drift_snapshot.json`. The snapshot is a normalized summary of the spec:

- every endpoint (`METHOD /path`)
- its request parameters (`query.*`, `path.*`, `header.*`, `body.*`)
- its success-response field paths and content types
- the members of every enum

Body and response fields are dotted paths: `[]` marks array items and `{}` marks map values. The script follows `$ref` and merges `allOf` / `anyOf` / `oneOf` branches, so a field that exists in only one branch of a union still counts.

Each change in the report is sorted by its impact on venice-py, using the SDK's own request models (`api_drift_impact.py`):

| Group | Meaning |
|---|---|
| **Needs SDK work** | A gap the SDK provably has: a new or removed endpoint, a request field no SDK request model has, or a new value on a request field the SDK restricts to a fixed set. |
| **Review** | Anything the models cannot decide: a removed field the SDK may still send, a content-type or status-code change, query parameters, an endpoint without a request model, or a field that a passthrough request model (`extra="allow"`) can send but does not type. |
| **No SDK change** | Only what the SDK provably handles already: a new value on a request field typed as an open string, a field the request model already has, and every new response field or value. Response models accept those by construction, enforced by `tests/unit/type_defs/test_response_models_tolerant.py`. |

Request fields are resolved against each model's `model_fields` and aliases, through nested models, lists and maps. Endpoints are mapped to request models in `MODEL_MAP`; `UNMODELLED` lists endpoints with a body but no model, and `KNOWN_OMISSIONS` lists fields left out on purpose. A test fails when an endpoint with a body is in neither map. Classification needs venice-py importable. If it is not, or classification fails, every change is listed under **Review** with a note, and the exit code is unchanged.

```bash
poetry run python tools/api_drift.py                      # fetch the live spec and compare
poetry run python tools/api_drift.py --spec swagger.yaml  # compare a local copy
poetry run python tools/api_drift.py --update-snapshot    # rewrite the snapshot from the live spec
poetry run python tools/api_drift.py --title title.txt    # also write the issue title
```

| Exit code | Meaning |
|---|---|
| 0 | No drift |
| 1 | Drift found. The Markdown report goes to stdout, and also to a file with `--report PATH`. |
| 2 | The spec or snapshot could not be read. Causes: a non-200 response, an HTML or empty body, invalid YAML, or a document with no `paths`. |

The [`API Drift`](../.github/workflows/api-drift.yml) workflow runs the scan daily and on manual dispatch. When it finds drift, it opens a single issue labelled `api-drift`, or updates the one that is already open. The title says how many changes need SDK work or review (`Venice API drift: 1 needs SDK work`, or `no SDK impact`). It posts a comment only when the drift itself changes. If the spec cannot be fetched or parsed, the run fails and the issue is left alone.

### Handling an `api-drift` issue

1. **Read the report** in the issue (the full version is in the linked run's summary). Each line says why it landed in its group:
   - **Needs SDK work:** add the resource method, request field or value it names.
   - **Review:** decide each one. A removed field may need removing from a request model; an untyped passthrough field may be worth typing.
   - **No SDK change:** nothing to change in the SDK, though a new model or value may be worth a docs or skill update.

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
