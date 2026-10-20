#!/usr/bin/env python3
"""Detect drift between the live Venice OpenAPI spec and a committed snapshot.

The spec is normalized into a stable summary:

- ``endpoints``: one entry per ``METHOD /path`` with its request parameters
  (``query.*``, ``path.*``, ``header.*``, ``body.*``), request content types,
  and, per success status code, the response content types and field paths.
  Field paths are dotted, ``[]`` marks array items and ``{}`` marks map values.
  ``$ref`` is followed and ``allOf`` / ``anyOf`` / ``oneOf`` branches are merged.
- ``enums``: every enum's member set, keyed by where it is declared, e.g.
  ``schemas/ChatCompletionRequest.reasoning_effort`` for a component schema or
  ``POST /chat/completions body.foo`` for an inline operation schema.

The summary is compared with ``tools/api_drift_snapshot.json`` and the
differences are printed as a Markdown report.

Exit codes:
    0  no drift
    1  drift found (report printed)
    2  the spec or snapshot could not be fetched, read, or parsed

Usage:
    python tools/api_drift.py                      # fetch live spec, compare
    python tools/api_drift.py --spec swagger.yaml  # compare a local spec
    python tools/api_drift.py --update-snapshot    # rewrite the snapshot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

SPEC_URL = "https://api.venice.ai/api/v1/swagger.yaml"
DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "api_drift_snapshot.json"
USER_AGENT = "venice-py-api-drift/1 (+https://github.com/sethbang/venice-py)"
HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
MAX_DEPTH = 12
FETCH_TIMEOUT = 60

EXIT_OK = 0
EXIT_DRIFT = 1
EXIT_ERROR = 2


class SpecError(Exception):
    """The spec or snapshot could not be obtained or understood."""


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core-schema scalars.

    PyYAML follows YAML 1.1, which reads unquoted ``16:9`` as the sexagesimal
    integer 969, ``yes``/``on`` as booleans and ISO dates as ``datetime``.
    The spec uses unquoted aspect ratios in string enums, so 1.1 rules would
    record the wrong members.
    """


_YAML11_TAGS = {
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
}
_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, regex) for tag, regex in resolvers if tag not in _YAML11_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9]*)$"),
    list("-+0123456789"),
)
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?[0-9]+[eE][-+]?[0-9]+"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def fetch_spec_text(url: str = SPEC_URL, timeout: float = FETCH_TIMEOUT) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/yaml, text/yaml, */*"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status: int = response.status
            content_type: str = response.headers.get("Content-Type", "")
            body: bytes = response.read()
    except urllib.error.HTTPError as exc:
        raise SpecError(f"fetching {url} failed: HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SpecError(f"fetching {url} failed: {exc}") from exc
    if status != 200:
        raise SpecError(f"fetching {url} failed: HTTP {status}")
    if "text/html" in content_type.lower():
        raise SpecError(f"fetching {url} returned HTML (Content-Type: {content_type})")
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecError(f"{url} did not return UTF-8 text: {exc}") from exc


def parse_spec(text: str, source: str) -> dict[str, Any]:
    stripped = text.lstrip()
    if not stripped:
        raise SpecError(f"{source} is empty")
    head = stripped[:512].lower()
    if stripped.startswith("<") or "<!doctype" in head or "<html" in head:
        raise SpecError(f"{source} is HTML, not an OpenAPI document")
    try:
        spec = yaml.load(text, Loader=_Yaml12Loader)
    except yaml.YAMLError as exc:
        raise SpecError(f"{source} is not valid YAML: {exc}") from exc
    if not isinstance(spec, dict):
        raise SpecError(f"{source} is not a YAML mapping (got {type(spec).__name__})")
    paths = spec.get("paths")
    if not isinstance(paths, dict) or not paths:
        raise SpecError(f"{source} has no 'paths'; refusing to compare an empty API")
    return spec


def load_spec(spec_path: str | None) -> tuple[dict[str, Any], str]:
    if spec_path is None:
        return parse_spec(fetch_spec_text(), SPEC_URL), SPEC_URL
    path = Path(spec_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise SpecError(f"cannot read {path}: {exc}") from exc
    return parse_spec(text, str(path)), str(path)


def load_snapshot(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(
            f"snapshot {path} does not exist; create it with --update-snapshot"
        ) from exc
    except (OSError, ValueError) as exc:
        raise SpecError(f"cannot read snapshot {path}: {exc}") from exc
    if not isinstance(data, dict) or not data.get("endpoints"):
        raise SpecError(f"snapshot {path} has no endpoints")
    return data


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


def _scalar(value: Any) -> str:
    """Stable string form of an enum member (``"str"``, ``1``, ``true``, ``null``)."""
    return json.dumps(value, default=str, sort_keys=True)


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _resolve(ref: str, spec: dict[str, Any]) -> tuple[str, Any]:
    if not ref.startswith("#/"):
        raise SpecError(f"unsupported external $ref {ref!r}")
    node: Any = spec
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"unresolvable $ref {ref!r}")
        node = node[part]
    return ref, node


def _deref(node: Any, spec: dict[str, Any], seen: frozenset[str]) -> tuple[Any, frozenset[str]]:
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if ref in seen:
            return None, seen
        seen = seen | {ref}
        _, node = _resolve(ref, spec)
    return node, seen


def schema_fields(
    schema: Any,
    spec: dict[str, Any],
    prefix: str = "",
    seen: frozenset[str] = frozenset(),
    depth: int = 0,
) -> set[str]:
    """Return every dotted field path reachable from ``schema``.

    ``$ref`` is resolved (with a cycle guard) and composition keywords are
    merged, so a property that exists only in one ``anyOf`` branch is still
    reported.
    """
    schema, seen = _deref(schema, spec, seen)
    if not isinstance(schema, dict) or depth > MAX_DEPTH:
        return set()
    fields: set[str] = set()
    for keyword in ("allOf", "anyOf", "oneOf"):
        for branch in schema.get(keyword) or []:
            fields |= schema_fields(branch, spec, prefix, seen, depth + 1)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, sub in properties.items():
            path = _join(prefix, str(name))
            fields.add(path)
            fields |= schema_fields(sub, spec, path, seen, depth + 1)
    items = schema.get("items")
    if isinstance(items, dict):
        fields |= schema_fields(items, spec, f"{prefix}[]", seen, depth + 1)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict) and additional:
        fields |= schema_fields(additional, spec, f"{prefix}{{}}", seen, depth + 1)
    return fields


def collect_enums(schema: Any, prefix: str, out: dict[str, set[str]], depth: int = 0) -> None:
    """Record enums declared inline in ``schema`` without following ``$ref``.

    Referenced component schemas are recorded under their own name, so each
    enum appears exactly once in the summary.
    """
    if not isinstance(schema, dict) or "$ref" in schema or depth > MAX_DEPTH:
        return
    enum = schema.get("enum")
    if isinstance(enum, list):
        out.setdefault(prefix, set()).update(_scalar(v) for v in enum)
    for keyword in ("allOf", "anyOf", "oneOf"):
        for branch in schema.get(keyword) or []:
            collect_enums(branch, prefix, out, depth + 1)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, sub in properties.items():
            collect_enums(sub, _join(prefix, str(name)), out, depth + 1)
    items = schema.get("items")
    if isinstance(items, dict):
        collect_enums(items, f"{prefix}[]", out, depth + 1)
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        collect_enums(additional, f"{prefix}{{}}", out, depth + 1)


def _mapping(node: Any, where: str) -> dict[str, Any]:
    if node is None:
        return {}
    if not isinstance(node, dict):
        raise SpecError(f"{where} is a {type(node).__name__}, expected a mapping")
    return node


def _sequence(node: Any, where: str) -> list[Any]:
    if node is None:
        return []
    if not isinstance(node, list):
        raise SpecError(f"{where} is a {type(node).__name__}, expected a list")
    return node


def _operation_parameters(
    path_item: dict[str, Any], operation: dict[str, Any], spec: dict[str, Any], where: str
) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    raws = _sequence(path_item.get("parameters"), f"{where} parameters") + _sequence(
        operation.get("parameters"), f"{where} parameters"
    )
    for raw in raws:
        param, _ = _deref(raw, spec, frozenset())
        param = _mapping(param, f"{where} parameter")
        if "name" not in param or "in" not in param:
            raise SpecError(f"{where} has a parameter without 'name' and 'in'")
        merged[(str(param["in"]), str(param["name"]))] = param
    return list(merged.values())


def summarize(spec: dict[str, Any]) -> dict[str, Any]:
    endpoints: dict[str, Any] = {}
    enums: dict[str, set[str]] = {}

    components = _mapping(spec.get("components"), "components")
    for name, schema in _mapping(components.get("schemas"), "components.schemas").items():
        collect_enums(schema, f"schemas/{name}", enums)

    for raw_path, path_item in spec["paths"].items():
        path_item = _mapping(path_item, f"paths.{raw_path}")
        for method in HTTP_METHODS:
            if method not in path_item:
                continue
            key = f"{method.upper()} {raw_path}"
            operation = _mapping(path_item[method], key)
            params: set[str] = set()

            for param in _operation_parameters(path_item, operation, spec, key):
                location = f"{param['in']}.{param['name']}"
                params.add(location)
                collect_enums(param.get("schema"), f"{key} {location}", enums)

            request_types: set[str] = set()
            body, _ = _deref(operation.get("requestBody"), spec, frozenset())
            body = _mapping(body, f"{key} requestBody")
            content = _mapping(body.get("content"), f"{key} requestBody.content")
            for media_type, media in content.items():
                request_types.add(str(media_type))
                schema = _mapping(media, f"{key} requestBody.{media_type}").get("schema")
                params |= {
                    f"body{f}" if f.startswith(("[", "{")) else f"body.{f}"
                    for f in schema_fields(schema, spec)
                }
                collect_enums(schema, f"{key} body", enums)

            responses: dict[str, Any] = {}
            operation_responses = _mapping(operation.get("responses"), f"{key} responses")
            for status, raw_response in operation_responses.items():
                status = str(status)
                if not status.startswith("2"):
                    continue
                response, _ = _deref(raw_response, spec, frozenset())
                response = _mapping(response, f"{key} response {status}")
                content_types: set[str] = set()
                fields: set[str] = set()
                where = f"{key} response {status}"
                for media_type, media in _mapping(response.get("content"), where).items():
                    content_types.add(str(media_type))
                    schema = _mapping(media, f"{where} {media_type}").get("schema")
                    fields |= schema_fields(schema, spec)
                    collect_enums(schema, f"{key} response.{status}", enums)
                responses[status] = {
                    "content_types": sorted(content_types),
                    "fields": sorted(fields),
                }

            endpoints[key] = {
                "params": sorted(params),
                "request_content_types": sorted(request_types),
                "responses": dict(sorted(responses.items())),
            }

    if not endpoints:
        raise SpecError("spec has 'paths' but no operations")

    return {
        "endpoints": dict(sorted(endpoints.items())),
        "enums": {k: sorted(v) for k, v in sorted(enums.items())},
    }


# --------------------------------------------------------------------------- #
# Comparison and reporting
# --------------------------------------------------------------------------- #


def _set_diff(old: list[str], new: list[str]) -> tuple[list[str], list[str]]:
    return sorted(set(new) - set(old)), sorted(set(old) - set(new))


def diff(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_eps: dict[str, Any] = old.get("endpoints", {})
    new_eps: dict[str, Any] = new.get("endpoints", {})
    added_eps, removed_eps = _set_diff(list(old_eps), list(new_eps))

    changed: dict[str, dict[str, tuple[list[str], list[str]]]] = {}
    for key in sorted(set(old_eps) & set(new_eps)):
        before, after = old_eps[key], new_eps[key]
        entry: dict[str, tuple[list[str], list[str]]] = {}
        for label, field in (
            ("params", "params"),
            ("request content types", "request_content_types"),
        ):
            added, removed = _set_diff(before.get(field, []), after.get(field, []))
            if added or removed:
                entry[label] = (added, removed)
        old_resp = before.get("responses", {})
        new_resp = after.get("responses", {})
        added, removed = _set_diff(list(old_resp), list(new_resp))
        if added or removed:
            entry["success status codes"] = (added, removed)
        for status in sorted(set(old_resp) & set(new_resp)):
            for label, field in (("content types", "content_types"), ("fields", "fields")):
                added, removed = _set_diff(
                    old_resp[status].get(field, []), new_resp[status].get(field, [])
                )
                if added or removed:
                    entry[f"response {status} {label}"] = (added, removed)
        if entry:
            changed[key] = entry

    old_enums: dict[str, list[str]] = old.get("enums", {})
    new_enums: dict[str, list[str]] = new.get("enums", {})
    added_enums, removed_enums = _set_diff(list(old_enums), list(new_enums))
    changed_enums: dict[str, tuple[list[str], list[str]]] = {}
    for key in sorted(set(old_enums) & set(new_enums)):
        added, removed = _set_diff(old_enums[key], new_enums[key])
        if added or removed:
            changed_enums[key] = (added, removed)

    return {
        "added_endpoints": added_eps,
        "removed_endpoints": removed_eps,
        "changed_endpoints": changed,
        "added_enums": {k: new_enums[k] for k in added_enums},
        "removed_enums": {k: old_enums[k] for k in removed_enums},
        "changed_enums": changed_enums,
    }


def has_drift(result: dict[str, Any]) -> bool:
    return any(bool(v) for v in result.values())


def _code(value: object) -> str:
    """Render a spec-derived value as an inline code span that cannot break out
    of the span or the line."""
    text = str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")
    return f"`{text}`"


def _code_list(values: list[str]) -> str:
    return ", ".join(_code(v) for v in values)


def render_report(result: dict[str, Any], source: str, spec_version: str) -> str:
    lines = ["# Venice API drift report", ""]
    lines.append(f"Spec: {_code(source)} (info.version {_code(spec_version)})")
    lines.append("")
    if not has_drift(result):
        lines.append("No drift: the spec matches `tools/api_drift_snapshot.json`.")
        return "\n".join(lines) + "\n"

    counts = [
        ("endpoints added", len(result["added_endpoints"])),
        ("endpoints removed", len(result["removed_endpoints"])),
        ("endpoints changed", len(result["changed_endpoints"])),
        ("enums added", len(result["added_enums"])),
        ("enums removed", len(result["removed_enums"])),
        ("enums changed", len(result["changed_enums"])),
    ]
    lines.append("| Change | Count |")
    lines.append("|---|---|")
    lines.extend(f"| {label} | {count} |" for label, count in counts if count)
    lines.append("")

    if result["added_endpoints"]:
        lines.append("## Added endpoints")
        lines.append("")
        lines.extend(f"- {_code(ep)}" for ep in result["added_endpoints"])
        lines.append("")
    if result["removed_endpoints"]:
        lines.append("## Removed endpoints")
        lines.append("")
        lines.extend(f"- {_code(ep)}" for ep in result["removed_endpoints"])
        lines.append("")
    if result["changed_endpoints"]:
        lines.append("## Changed endpoints")
        lines.append("")
        for ep, entry in result["changed_endpoints"].items():
            lines.append(f"### {_code(ep)}")
            lines.append("")
            for label, (added, removed) in entry.items():
                if added:
                    lines.append(f"- {label} added: {_code_list(added)}")
                if removed:
                    lines.append(f"- {label} removed: {_code_list(removed)}")
            lines.append("")
    if result["changed_enums"] or result["added_enums"] or result["removed_enums"]:
        lines.append("## Enums")
        lines.append("")
        for key, (added, removed) in result["changed_enums"].items():
            if added:
                lines.append(f"- {_code(key)} members added: {_code_list(added)}")
            if removed:
                lines.append(f"- {_code(key)} members removed: {_code_list(removed)}")
        for key, members in result["added_enums"].items():
            lines.append(f"- new enum {_code(key)}: {_code_list(members)}")
        for key, members in result["removed_enums"].items():
            lines.append(f"- enum removed {_code(key)} (was {_code_list(members)})")
        lines.append("")
    return "\n".join(lines)


def write_snapshot(summary: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec", help=f"read the spec from a local file instead of {SPEC_URL}")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="snapshot file to compare against (default: tools/api_drift_snapshot.json)",
    )
    parser.add_argument(
        "--update-snapshot",
        action="store_true",
        help="rewrite the snapshot from the spec instead of comparing",
    )
    parser.add_argument("--report", type=Path, help="also write the Markdown report to this file")
    args = parser.parse_args(argv)

    try:
        return _run(args)
    except SpecError as exc:
        print(f"api_drift: ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except Exception:
        print("api_drift: ERROR: unexpected failure", file=sys.stderr)
        traceback.print_exc()
        return EXIT_ERROR


def _run(args: argparse.Namespace) -> int:
    spec, source = load_spec(args.spec)
    summary = summarize(spec)
    if args.update_snapshot:
        write_snapshot(summary, args.snapshot)
        print(
            f"Wrote {args.snapshot} ({len(summary['endpoints'])} endpoints, {len(summary['enums'])} enums)"
        )
        return EXIT_OK
    snapshot = load_snapshot(args.snapshot)
    result = diff(snapshot, summary)
    info = spec.get("info")
    spec_version = str(info.get("version", "unknown")) if isinstance(info, dict) else "unknown"
    report = render_report(result, source, spec_version)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    return EXIT_DRIFT if has_drift(result) else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
