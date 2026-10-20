"""Unit tests for the OpenAPI drift scanner (``tools/api_drift.py``).

Every test uses small inline specs and ``tmp_path``; nothing touches the
network or the committed snapshot.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import json
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
