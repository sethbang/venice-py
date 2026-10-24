"""Unit tests for the OpenAPI drift scanner (``tools/api_drift.py``).

Every test uses small inline specs and ``tmp_path``; nothing touches the
network or the committed snapshot.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import re
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "api_drift.py"


def _load():
    spec = importlib.util.spec_from_file_location("api_drift", TOOL_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


drift = _load()


BASE_SPEC: dict[str, Any] = {
    "openapi": "3.0.0",
    "info": {"title": "test", "version": "1"},
    "paths": {
        "/chat/completions": {
            "post": {
                "requestBody": {
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/ChatRequest"}}
                    }
                },
                "responses": {
                    "200": {"$ref": "#/components/responses/ChatOk"},
                    "400": {
                        "description": "bad",
                        "content": {
                            "application/json": {
                                "schema": {"properties": {"error": {"type": "string"}}}
                            }
                        },
                    },
                },
            }
        },
        "/models": {
            "get": {
                "parameters": [
                    {
                        "name": "type",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["text", "image"]},
                    }
                ],
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "data": {
                                            "type": "array",
                                            "items": {"properties": {"id": {"type": "string"}}},
                                        }
                                    },
                                }
                            }
                        },
                    }
                },
            }
        },
    },
    "components": {
        "responses": {
            "ChatOk": {
                "description": "ok",
                "content": {
                    "application/json": {
                        "schema": {
                            "allOf": [
                                {"properties": {"id": {"type": "string"}}},
                                {"properties": {"choices": {"type": "array"}}},
                            ]
                        }
                    }
                },
            }
        },
        "schemas": {
            "ChatRequest": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "reasoning_effort": {"type": "string", "enum": ["low", "medium", "high"]},
                    "messages": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"$ref": "#/components/schemas/UserMessage"},
                                {"properties": {"role": {"type": "string"}, "tool_call_id": {}}},
                            ]
                        },
                    },
                },
            },
            "UserMessage": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user"]},
                    "content": {"type": "string"},
                    "parent": {"$ref": "#/components/schemas/UserMessage"},
                },
            },
        },
    },
}


def _write_spec(tmp_path: Path, spec: dict[str, Any], name: str = "spec.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


def _snapshot_of(tmp_path: Path, spec: dict[str, Any]) -> Path:
    snapshot = tmp_path / "snapshot.json"
    spec_path = _write_spec(tmp_path, spec, "base.yaml")
    assert (
        drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot), "--update-snapshot"])
        == 0
    )
    return snapshot


def _run(tmp_path: Path, snapshot: Path, spec: dict[str, Any]) -> int:
    spec_path = _write_spec(tmp_path, spec, "new.yaml")
    return drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot)])


# ---- normalization ----


def test_summary_follows_refs_and_merges_union_branches():
    summary = drift.summarize(copy.deepcopy(BASE_SPEC))
    chat = summary["endpoints"]["POST /chat/completions"]
    assert "body.messages[].content" in chat["params"]  # via $ref inside anyOf
    assert "body.messages[].tool_call_id" in chat["params"]  # inline anyOf branch
    assert "body.messages[].parent" in chat["params"]
    assert not any(p.startswith("body.messages[].parent.") for p in chat["params"])  # cycle stops
    assert chat["responses"]["200"]["fields"] == ["choices", "id"]  # allOf via components/responses
    assert "400" not in chat["responses"]
    assert summary["endpoints"]["GET /models"]["params"] == ["query.type"]
    assert summary["enums"]["schemas/ChatRequest.reasoning_effort"] == [
        '"high"',
        '"low"',
        '"medium"',
    ]
    assert summary["enums"]["GET /models query.type"] == ['"image"', '"text"']


def test_snapshot_is_deterministic(tmp_path):
    first = _snapshot_of(tmp_path, BASE_SPEC).read_text()
    shuffled = copy.deepcopy(BASE_SPEC)
    shuffled["components"]["schemas"]["ChatRequest"]["properties"]["reasoning_effort"][
        "enum"
    ].reverse()
    second = _snapshot_of(tmp_path, shuffled).read_text()
    assert first == second


def test_non_string_enum_members_are_serializable(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    spec["components"]["schemas"]["UserMessage"]["properties"]["flag"] = {
        "enum": [True, 1, None, "x"]
    }
    summary = drift.summarize(spec)
    assert summary["enums"]["schemas/UserMessage.flag"] == ['"x"', "1", "null", "true"]


def test_unquoted_scalars_follow_yaml_1_2(tmp_path):
    text = (
        "paths:\n"
        "  /video/queue:\n"
        "    post:\n"
        "      requestBody:\n"
        "        content:\n"
        "          application/json:\n"
        "            schema:\n"
        "              properties:\n"
        "                aspect_ratio:\n"
        "                  type: string\n"
        "                  enum: [1:1, 16:9, 2:1, auto]\n"
        "                mode:\n"
        "                  enum: [on, off, yes, 2026-01-01, 12, true]\n"
        "      responses: {}\n"
    )
    summary = drift.summarize(drift.parse_spec(text, "inline"))
    assert summary["enums"]["POST /video/queue body.aspect_ratio"] == [
        '"16:9"',
        '"1:1"',
        '"2:1"',
        '"auto"',
    ]
    assert summary["enums"]["POST /video/queue body.mode"] == [
        '"2026-01-01"',
        '"off"',
        '"on"',
        '"yes"',
        "12",
        "true",
    ]


# ---- comparison ----


def test_identical_spec_exits_zero(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    assert _run(tmp_path, snapshot, BASE_SPEC) == 0
    assert "No drift" in capsys.readouterr().out


def test_added_enum_member_is_drift(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    spec["components"]["schemas"]["ChatRequest"]["properties"]["reasoning_effort"]["enum"].append(
        "max"
    )
    assert _run(tmp_path, snapshot, spec) == 1
    out = capsys.readouterr().out
    assert '`schemas/ChatRequest.reasoning_effort` members added: `"max"`' in out


def test_report_neutralizes_backticks_and_newlines_in_spec_values(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    spec["components"]["schemas"]["ChatRequest"]["properties"]["reasoning_effort"]["enum"].append(
        "x` @someone\n::warning::injected"
    )
    assert _run(tmp_path, snapshot, spec) == 1
    out = capsys.readouterr().out
    assert '`"x\' @someone\\n::warning::injected"`' in out
    assert not any(line.startswith("::") for line in out.splitlines())
    assert drift._code("a`b\nc") == "`a'b c`"


def test_removed_endpoint_is_drift(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    del spec["paths"]["/models"]
    assert _run(tmp_path, snapshot, spec) == 1
    out = capsys.readouterr().out
    assert "## Removed endpoints" in out
    assert "- `GET /models`" in out


def test_new_body_field_inside_anyof_is_drift(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    branch = spec["components"]["schemas"]["ChatRequest"]["properties"]["messages"]["items"][
        "anyOf"
    ][1]
    branch["properties"]["cache_hint"] = {"type": "string"}
    assert _run(tmp_path, snapshot, spec) == 1
    out = capsys.readouterr().out
    assert "### `POST /chat/completions`" in out
    assert "params added: `body.messages[].cache_hint`" in out


def test_new_response_field_and_removed_param_are_drift(tmp_path, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    models = spec["paths"]["/models"]["get"]
    models["parameters"] = []
    schema = models["responses"]["200"]["content"]["application/json"]["schema"]
    schema["properties"]["data"]["items"]["properties"]["owned_by"] = {"type": "string"}
    assert _run(tmp_path, snapshot, spec) == 1
    out = capsys.readouterr().out
    assert "params removed: `query.type`" in out
    assert "response 200 fields added: `data[].owned_by`" in out
    assert "enum removed `GET /models query.type`" in out


def test_report_file_is_written(tmp_path):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec = copy.deepcopy(BASE_SPEC)
    del spec["paths"]["/models"]
    report = tmp_path / "report.md"
    spec_path = _write_spec(tmp_path, spec, "new.yaml")
    code = drift.main(
        ["--spec", str(spec_path), "--snapshot", str(snapshot), "--report", str(report)]
    )
    assert code == 1
    assert "`GET /models`" in report.read_text()


# ---- unreadable input must never read as "no drift" ----


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<!DOCTYPE html>\n<html><body>Just a moment...</body></html>\n", "is HTML"),
        ("", "is empty"),
        ("   \n\n", "is empty"),
        ("openapi: 3.0.0\ninfo: {title: x, version: '1'}\n", "no 'paths'"),
        ("openapi: 3.0.0\npaths: {}\n", "no 'paths'"),
        ("paths: [unclosed\n  - : :\n", "not valid YAML"),
        ("- just\n- a list\n", "not a YAML mapping"),
        ("paths:\n  /x: {summary: no operations}\n", "no operations"),
    ],
)
def test_unreadable_spec_exits_two(tmp_path, capsys, body, message):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec_path = tmp_path / "bad.yaml"
    spec_path.write_text(body, encoding="utf-8")
    assert drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot)]) == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "No drift" not in captured.out


@pytest.mark.parametrize(
    "body",
    [
        "paths:\n  /x:\n    get:\n      responses: [1, 2]\n",
        "paths:\n  /x:\n    get:\n      responses:\n        '200':\n          content:\n            application/json: just-a-string\n",
        "components: []\npaths:\n  /x:\n    get:\n      responses: {}\n",
        "paths:\n  /x:\n    get:\n      parameters: oops\n      responses: {}\n",
    ],
)
def test_malformed_spec_exits_two_not_one(tmp_path, capsys, body):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec_path = tmp_path / "odd.yaml"
    spec_path.write_text(body, encoding="utf-8")
    assert drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot)]) == 2
    assert "api_drift: ERROR" in capsys.readouterr().err


def test_unexpected_exception_exits_two(tmp_path, monkeypatch, capsys):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec_path = _write_spec(tmp_path, BASE_SPEC)

    def boom(_spec):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(drift, "summarize", boom)
    assert drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot)]) == 2
    err = capsys.readouterr().err
    assert "unexpected failure" in err
    assert "kaboom" in err


def test_missing_snapshot_exits_two(tmp_path, capsys):
    spec_path = _write_spec(tmp_path, BASE_SPEC)
    assert drift.main(["--spec", str(spec_path), "--snapshot", str(tmp_path / "nope.json")]) == 2
    assert "--update-snapshot" in capsys.readouterr().err


class _FakeResponse:
    def __init__(self, status: int, body: bytes, content_type: str) -> None:
        self.status = status
        self.headers = {"Content-Type": content_type}
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _patch_urlopen(monkeypatch, outcome):
    def fake_urlopen(request, timeout):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(drift.urllib.request, "urlopen", fake_urlopen)


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (
            urllib.error.HTTPError(drift.SPEC_URL, 503, "Service Unavailable", {}, io.BytesIO()),  # type: ignore[arg-type]
            "HTTP 503",
        ),
        (urllib.error.URLError("Name or service not known"), "Name or service not known"),
        (_FakeResponse(204, b"", "application/yaml"), "HTTP 204"),
        (_FakeResponse(200, b"<html>login</html>", "text/html; charset=utf-8"), "returned HTML"),
        (_FakeResponse(200, b"", "application/yaml"), "is empty"),
    ],
)
def test_bad_fetch_exits_two(tmp_path, monkeypatch, capsys, outcome, message):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    _patch_urlopen(monkeypatch, outcome)
    assert drift.main(["--snapshot", str(snapshot)]) == 2
    assert message in capsys.readouterr().err


def test_good_fetch_is_compared(tmp_path, monkeypatch):
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    body = yaml.safe_dump(BASE_SPEC).encode()
    _patch_urlopen(monkeypatch, _FakeResponse(200, body, "application/yaml; charset=utf-8"))
    assert drift.main(["--snapshot", str(snapshot)]) == 0


def test_committed_snapshot_is_populated():
    data = json.loads(drift.DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    assert len(data["endpoints"]) > 20
    assert data["enums"]
    assert "POST /chat/completions" in data["endpoints"]


# ---- SDK impact classification ----

IMPACT_PATH = REPO_ROOT / "tools" / "api_drift_impact.py"


def _load_impact():
    spec = importlib.util.spec_from_file_location("api_drift_impact", IMPACT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


impact = _load_impact()


def _report(tmp_path: Path, spec: dict[str, Any]) -> tuple[int, str, str]:
    snapshot = _snapshot_of(tmp_path, BASE_SPEC)
    spec_path = _write_spec(tmp_path, spec, "new.yaml")
    report, title = tmp_path / "report.md", tmp_path / "title.txt"
    code = drift.main(
        [
            "--spec",
            str(spec_path),
            "--snapshot",
            str(snapshot),
            "--report",
            str(report),
            "--title",
            str(title),
        ]
    )
    return code, report.read_text(), title.read_text().strip()


def _section(report: str, title: str) -> str:
    """The lines under ``## <title>`` up to the next heading."""
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |^<details>|\Z)", report, re.S | re.M)
    return match.group(1) if match else ""


def _chat_props(spec: dict[str, Any]) -> dict[str, Any]:
    return spec["components"]["schemas"]["ChatRequest"]["properties"]


def test_field_no_request_model_has_needs_sdk_work(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    props = _chat_props(spec)["messages"]["items"]["anyOf"][1]["properties"]
    props["brand_new_knob"] = {"type": "string"}
    code, report, title = _report(tmp_path, spec)
    assert code == 1
    needs = _section(report, "Needs SDK work")
    assert "`body.messages[].brand_new_knob` added" in needs
    assert "ChatCompletionRequest" in needs
    assert title == "Venice API drift: 1 needs SDK work"


def test_untyped_field_on_a_passthrough_request_model_needs_review(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    _chat_props(spec)["brand_new_knob"] = {"type": "string"}
    _, report, title = _report(tmp_path, spec)
    review = _section(report, "Review")
    assert "`body.brand_new_knob` added" in review
    assert "passes unknown fields through" in review
    assert title == "Venice API drift: 1 to review"


def test_field_the_sdk_already_models_needs_no_change(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    _chat_props(spec)["temperature"] = {"type": "number"}
    _, report, title = _report(tmp_path, spec)
    assert "`body.temperature` added" in _section(report, "No SDK change")
    assert title == "Venice API drift: no SDK impact"


def test_new_value_outside_a_closed_request_literal_needs_sdk_work(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    _chat_props(spec)["reasoning_effort"]["enum"] += ["max", "turbo"]
    _, report, _ = _report(tmp_path, spec)
    needs = _section(report, "Needs SDK work")
    assert "`schemas/ChatRequest.reasoning_effort` gained" in needs
    assert '"turbo"' in needs.split("does not include", 1)[1]
    assert '"max"' not in needs.split("does not include", 1)[1]


def test_new_value_on_an_open_request_field_needs_no_change(tmp_path):
    base = copy.deepcopy(BASE_SPEC)
    _chat_props(base)["model"] = {"type": "string", "enum": ["a"]}
    snapshot = _snapshot_of(tmp_path, base)
    spec = copy.deepcopy(base)
    _chat_props(spec)["model"]["enum"].append("b")
    report = tmp_path / "report.md"
    spec_path = _write_spec(tmp_path, spec, "new.yaml")
    assert (
        drift.main(["--spec", str(spec_path), "--snapshot", str(snapshot), "--report", str(report)])
        == 1
    )
    no_change = _section(report.read_text(), "No SDK change")
    assert '`schemas/ChatRequest.model` gained `"b"`' in no_change
    assert "accepts any value" in no_change


def test_response_changes_need_no_change_and_removals_need_review(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    models = spec["paths"]["/models"]["get"]
    models["parameters"] = []
    schema = models["responses"]["200"]["content"]["application/json"]["schema"]
    schema["properties"]["data"]["items"]["properties"]["owned_by"] = {"type": "string"}
    _, report, title = _report(tmp_path, spec)
    assert "fields added: `data[].owned_by`" in _section(report, "No SDK change")
    review = _section(report, "Review")
    assert "`query.type` removed" in review
    assert "enum `GET /models query.type` removed" in review
    assert title == "Venice API drift: 2 to review"


def test_endpoint_changes_need_sdk_work(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    del spec["paths"]["/models"]
    spec["paths"]["/brand/new"] = {"get": {"responses": {}}}
    _, report, _ = _report(tmp_path, spec)
    needs = _section(report, "Needs SDK work")
    assert "`GET /brand/new` added" in needs
    assert "`GET /models` removed" in needs


def test_every_change_lands_in_exactly_one_group(tmp_path):
    spec = copy.deepcopy(BASE_SPEC)
    _chat_props(spec)["brand_new_knob"] = {"type": "string"}
    _chat_props(spec)["temperature"] = {"type": "number"}
    _chat_props(spec)["reasoning_effort"]["enum"].append("turbo")
    del spec["paths"]["/models"]
    result = drift.diff(drift.summarize(copy.deepcopy(BASE_SPEC)), drift.summarize(spec))
    items, note = drift.classify(result, spec)
    assert not note
    assert len(items) == len(drift.drift_items(result))
    assert all(item.group in drift.GROUP_TITLES and item.reason for item in items)
    counts = drift.impact_counts(items)
    assert sum(counts.values()) == len(items)
    report = drift.render_report(result, "src", "1", items)
    listed = sum(
        len(re.findall(r"^- ", _section(report, t), re.M)) for t in drift.GROUP_TITLES.values()
    )
    assert listed == len(items)


def test_nested_new_fields_fold_under_their_parent():
    folded = drift._fold_nested(["images", "images[].base64", "images[].content_type", "other.x"])
    assert folded == [("images", 2), ("other.x", 0)]


@pytest.mark.parametrize("failure", ["import", "drops-items"])
def test_classifier_failure_lists_everything_under_review(tmp_path, monkeypatch, failure):
    if failure == "import":
        monkeypatch.setitem(sys.modules, "api_drift_impact", None)
    else:
        monkeypatch.setattr(impact, "classify", lambda items, spec: items[:-1])
    spec = copy.deepcopy(BASE_SPEC)
    _chat_props(spec)["temperature"] = {"type": "number"}
    del spec["paths"]["/models"]
    code, report, title = _report(tmp_path, spec)
    assert code == 1
    assert "every change is listed under Review" in report
    assert not _section(report, "No SDK change") and not _section(report, "Needs SDK work")
    assert "`body.temperature` added" in _section(report, "Review")
    assert title == "Venice API drift: 3 to review"


def test_no_drift_writes_no_impact_section(tmp_path):
    code, report, title = _report(tmp_path, copy.deepcopy(BASE_SPEC))
    assert code == 0
    assert "SDK impact" not in report
    assert title == "Venice API drift: no SDK impact"


def test_resolve_walks_nested_models_lists_and_aliases():
    from venice_ai.types.api.requests.api_keys import CreateApiKeyRequest
    from venice_ai.types.api.requests.decisions import CreateDecisionRequest

    assert impact.resolve([CreateDecisionRequest], "images[].base64") is not None
    assert impact.resolve([CreateDecisionRequest], "images[].url") is None
    assert impact.resolve([CreateDecisionRequest], "questions{}.instructions") is not None
    assert impact.resolve([CreateApiKeyRequest], "consumptionLimit.usd") is not None
    assert impact.resolve([CreateDecisionRequest], "nope") is None


def test_model_map_covers_every_endpoint_with_a_body():
    data = json.loads(drift.DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    with_body = {
        ep
        for ep, entry in data["endpoints"].items()
        if any(p.startswith("body") for p in entry["params"])
    }
    mapped = set(impact.MODEL_MAP) | set(impact.UNMODELLED)
    assert not sorted(with_body - mapped), "map these endpoints in tools/api_drift_impact.py"
    assert not sorted(mapped - set(data["endpoints"])), "stale entries in tools/api_drift_impact.py"
    assert not set(impact.MODEL_MAP) & set(impact.UNMODELLED)


def test_mapped_request_models_cover_the_committed_snapshot():
    """Every body field in the snapshot resolves, apart from listed omissions
    and known gaps, so a new field is reported because it is new."""
    known_gaps = {
        ("PATCH /api_keys", "consumptionLimit.vcu"),
        ("POST /api_keys", "consumptionLimit.vcu"),
        ("POST /api_keys/generate_web3_key", "consumptionLimit.vcu"),
        ("POST /chat/completions", "messages[].thought_signature"),
    }
    data = json.loads(drift.DEFAULT_SNAPSHOT.read_text(encoding="utf-8"))
    unresolved = set()
    for ep, models in impact.MODEL_MAP.items():
        for param in data["endpoints"][ep]["params"]:
            path = param.removeprefix("body.")
            if param == path or impact.KNOWN_OMISSIONS.get(ep, {}).get(impact._top(path)):
                continue
            if impact.resolve(models, path) is None:
                unresolved.add((ep, path))
    assert unresolved == known_gaps
