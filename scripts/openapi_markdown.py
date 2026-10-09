"""Render an OpenAPI 3 schema as one Markdown page.

Pure functions over the schema dict; no network, no imports from localm.
``render(schema)`` returns the page body: the operations grouped by path
prefix, each with its parameters, request body and responses, followed by the
component schemas as property tables.
"""

import re

_METHODS = ("get", "put", "post", "delete", "patch", "head", "options")
_GROUPS = (
    ("/v1/", "OpenAI-compatible and model API (`/v1`)"),
    ("/api/", "Application API (`/api`)"),
)
_OTHER_GROUP = "Other"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _cell(text: str) -> str:
    """Make *text* safe inside one Markdown table cell."""
    return " ".join(str(text).split()).replace("|", "\\|")


def _ref_name(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def _ref_link(name: str) -> str:
    return f"[{name}](#schema-{_slug(name)})"


def type_of(schema: dict | None) -> str:
    """Return a short Markdown description of the type *schema* declares."""
    if not schema:
        return "any"
    if "$ref" in schema:
        return _ref_link(_ref_name(schema["$ref"]))
    for key, joiner in (("anyOf", " or "), ("oneOf", " or "), ("allOf", " and ")):
        if key in schema:
            return joiner.join(type_of(s) for s in schema[key])
    if "enum" in schema:
        return " / ".join(f"`{v}`" for v in schema["enum"])
    if "const" in schema:
        return f"`{schema['const']}`"
    kind = schema.get("type", "any")
    if isinstance(kind, list):
        return " or ".join(str(k) for k in kind)
    if kind == "array":
        return f"array of {type_of(schema.get('items'))}"
    if kind == "object" and isinstance(schema.get("additionalProperties"), dict):
        return f"map of {type_of(schema['additionalProperties'])}"
    fmt = schema.get("format")
    return f"{kind} ({fmt})" if fmt else str(kind)


def _description(obj: dict) -> str:
    return (obj.get("description") or "").strip()


def _property_table(schema: dict) -> list[str]:
    props = schema.get("properties") or {}
    if not props:
        return []
    required = set(schema.get("required") or [])
    lines = ["| Field | Type | Required | Description |", "|---|---|---|---|"]
    for name, prop in props.items():
        desc = _description(prop) or prop.get("title", "")
        lines.append(
            f"| `{name}` | {_cell(type_of(prop))} | {'yes' if name in required else 'no'} "
            f"| {_cell(desc)} |")
    return lines


def _parameters(params: list[dict]) -> list[str]:
    if not params:
        return []
    lines = ["**Parameters**", "", "| Name | In | Type | Required | Description |",
             "|---|---|---|---|---|"]
    for p in params:
        lines.append(
            f"| `{p['name']}` | {p['in']} | {_cell(type_of(p.get('schema')))} "
            f"| {'yes' if p.get('required') else 'no'} | {_cell(_description(p))} |")
    return lines + [""]


def _body(content: dict) -> list[str]:
    return [f"- `{ctype}`: {type_of(media.get('schema'))}"
            for ctype, media in content.items()]


def _operation(method: str, path: str, op: dict) -> list[str]:
    anchor = _slug(f"{method} {path}")
    lines = [f"### {method.upper()} `{path}` {{ #{anchor} }}", ""]
    if op.get("deprecated"):
        lines += ["!!! warning", "    Deprecated.", ""]
    desc = _description(op) or op.get("summary", "")
    if desc:
        lines += [desc, ""]
    lines += _parameters(op.get("parameters") or [])
    request = (op.get("requestBody") or {}).get("content")
    if request:
        lines += ["**Request body**", ""] + _body(request) + [""]
    responses = op.get("responses") or {}
    if responses:
        lines += ["**Responses**", ""]
        for status, resp in responses.items():
            content = resp.get("content") or {}
            kinds = ", ".join(f"`{c}` {type_of(m.get('schema'))}" for c, m in content.items())
            text = _description(resp)
            lines.append(f"- `{status}` {text}" + (f": {kinds}" if kinds else ""))
        lines.append("")
    return lines


def _group_of(path: str) -> str:
    for prefix, title in _GROUPS:
        if path.startswith(prefix):
            return title
    return _OTHER_GROUP


def _operations_section(paths: dict) -> list[str]:
    grouped: dict[str, list[tuple[str, str, dict]]] = {}
    for path in sorted(paths):
        for method in _METHODS:
            if method in paths[path]:
                grouped.setdefault(_group_of(path), []).append((method, path, paths[path][method]))
    order = [t for _, t in _GROUPS] + [_OTHER_GROUP]
    lines: list[str] = []
    for title in order:
        if title not in grouped:
            continue
        lines += [f"## {title}", ""]
        for method, path, op in grouped[title]:
            lines += _operation(method, path, op)
    return lines


def _schemas_section(schemas: dict) -> list[str]:
    if not schemas:
        return []
    lines = ["## Schemas", ""]
    for name in sorted(schemas):
        schema = schemas[name]
        lines += [f"### {name} {{ #schema-{_slug(name)} }}", ""]
        desc = _description(schema)
        if desc:
            lines += [desc, ""]
        table = _property_table(schema)
        if table:
            lines += table + [""]
        else:
            lines += [f"Type: {type_of(schema)}", ""]
    return lines


def render(schema: dict) -> str:
    """Return the API reference body for *schema* as Markdown: a version line,
    then the operations and the component schemas as ``##`` sections."""
    info = schema.get("info", {})
    lines: list[str] = []
    if info.get("version"):
        lines += [f"Schema version {info['version']} (OpenAPI {schema.get('openapi', '3')}).", ""]
    lines += _operations_section(schema.get("paths") or {})
    lines += _schemas_section((schema.get("components") or {}).get("schemas") or {})
    return "\n".join(lines).rstrip() + "\n"
