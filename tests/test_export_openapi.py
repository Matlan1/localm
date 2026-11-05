"""The OpenAPI export and the docs site that renders it.

The export runs as a real subprocess of scripts/export_openapi.py, exactly as
the documentation build runs it. The renderer is driven both with a small
hand-built schema and with the exported real one.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


openapi_markdown = _load("openapi_markdown")


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("openapi_export")
    home = root / "caller_home"
    home.mkdir()
    out = root / "out" / "openapi.json"
    env = dict(os.environ, LOCALM_HOME=str(home))
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "export_openapi.py"), "--out", str(out)],
        capture_output=True, text=True, env=env, timeout=120, cwd=root)
    return proc, out, home


def test_export_runs_and_writes_valid_json(exported):
    proc, out, _ = exported
    assert proc.returncode == 0, proc.stderr
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert schema["openapi"].startswith("3.")
    assert schema["paths"]


def test_export_contains_the_chat_completions_route(exported):
    _, out, _ = exported
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert "post" in schema["paths"]["/v1/chat/completions"]


def test_export_stamps_the_installed_version(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import localm; localm.__version__ = '9.8.7'; "
        "import export_openapi; "
        "print(export_openapi.build_schema()['info']['version'])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPTS)], capture_output=True, text=True,
        cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO)), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "9.8.7"


def test_export_in_log_mode_closes_the_server_session_log(tmp_path):
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import export_openapi; export_openapi.build_schema(); "
        "from localm.inference import http_server; "
        "print(http_server._session_recorder.current()[0].__class__.__name__, "
        "http_server._session_recorder._closed)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPTS)], capture_output=True, text=True,
        cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO), LOCALM_MODE="log",
                           LOCALM_HOME=str(tmp_path)), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "NullAuditLog True"


def test_export_leaves_the_callers_home_untouched(exported):
    _, _, home = exported
    assert list(home.iterdir()) == []


_INTERNAL_MARKERS = re.compile(
    r"(?<![A-Za-z0-9])_{1,2}[A-Za-z]"
    r"|\bdeliberately\b|\bon purpose\b|\bthe fix\b|\bused to\b|\bpreviously\b"
    r"|\bwhich is why\b|\bwas the defect\b|\bthe trap\b|\bMEASURED\b"
    r"|#\d+"
    r"|\b(?:REG|AUDIT|BUG|SEC|ADR|DIST|VOX)(?:-[A-Za-z]+)?-\d+"
    r"|\bAGENTS\.md\b|\bdev-notes\b|\bpath_formatter\b",
    re.IGNORECASE)


def _internal_text_hits(node, where="$"):
    """Every (location, marker) where a summary or description in *node* carries
    a private identifier or a rationale marker."""
    hits = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("description", "summary") and isinstance(value, str):
                hits += [(f"{where}.{key}", m.group(0)) for m in _INTERNAL_MARKERS.finditer(value)]
            else:
                hits += _internal_text_hits(value, f"{where}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hits += _internal_text_hits(value, f"{where}[{i}]")
    return hits


def test_exported_descriptions_carry_no_private_names_or_design_rationale(exported):
    _, out, _ = exported
    schema = json.loads(out.read_text(encoding="utf-8"))
    assert _internal_text_hits(schema) == []


@pytest.mark.parametrize("text", [
    "Goes through _resolve_unnamed_model_name first.",
    "This is deliberately not scoped.",
    "Uses the fix for the handover.",
    "It used to skip the last model.",
    "Previously this read the default.",
    "The duplication was the defect.",
    "See issue #1234.",
    "Tracked as AUDIT-CRIT-1 and BUG-9b.",
    "Calls http_server.path_formatter on the reply.",
])
def test_the_description_scan_flags_internal_text(text):
    schema = {"paths": {"/x": {"get": {"description": text}}}}
    assert _internal_text_hits(schema), text


def test_the_description_scan_accepts_plain_endpoint_text():
    text = ("Returns ``{available, include_log}`` for the model named in `model`; "
            "answers 404 for an unknown model. See POST /api/bug-report/send "
            "and GET /v1/models/{model_id}.")
    schema = {"paths": {"/x": {"get": {"description": text, "summary": "Get X"}}}}
    assert _internal_text_hits(schema) == []


def test_render_documents_every_exported_operation(exported):
    _, out, _ = exported
    schema = json.loads(out.read_text(encoding="utf-8"))
    page = openapi_markdown.render(schema)
    assert "### POST `/v1/chat/completions` { #post-v1-chat-completions }" in page
    for path, ops in schema["paths"].items():
        for method in ops:
            assert f"### {method.upper()} `{path}`" in page


def test_render_builds_tables_and_links_for_a_small_schema():
    schema = {
        "openapi": "3.1.0",
        "info": {"title": "T", "version": "9.9"},
        "paths": {
            "/v1/things/{thing_id}": {
                "put": {
                    "summary": "Put Thing",
                    "description": "Replace one thing.",
                    "parameters": [{"name": "thing_id", "in": "path", "required": True,
                                    "schema": {"type": "string"}}],
                    "requestBody": {"content": {"application/json": {
                        "schema": {"$ref": "#/components/schemas/Thing"}}}},
                    "responses": {"200": {"description": "OK"},
                                  "422": {"description": "Bad | input"}},
                }
            },
            "/elsewhere": {"get": {"summary": "Elsewhere", "responses": {}}},
        },
        "components": {"schemas": {"Thing": {
            "type": "object", "required": ["name"],
            "properties": {
                "name": {"type": "string", "description": "A | name"},
                "tags": {"type": "array", "items": {"type": "string"}},
            }}}},
    }
    page = openapi_markdown.render(schema)
    assert "Schema version 9.9" in page
    assert "| `thing_id` | path | string | yes |" in page
    assert "- `application/json`: [Thing](#schema-thing)" in page
    assert "| `tags` | array of string | no |" in page
    assert page.index("## Other") > page.index("## OpenAI-compatible")
    assert "### Thing { #schema-thing }" in page


def test_render_escapes_a_pipe_inside_a_table_cell():
    schema = {"components": {"schemas": {"Thing": {
        "type": "object", "properties": {"name": {"type": "string", "description": "A | name"}}}}}}
    assert "| `name` | string | no | A \\| name |" in openapi_markdown.render(schema)


def test_render_marks_a_deprecated_operation():
    schema = {"paths": {"/api/old": {"get": {"summary": "Old", "deprecated": True,
                                              "responses": {}}}}}
    assert "Deprecated." in openapi_markdown.render(schema)


@pytest.fixture(scope="module")
def mkdocs_config():
    return yaml.safe_load((REPO / "mkdocs.yml").read_text(encoding="utf-8"))


def _nav_files(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for item in node:
            yield from _nav_files(item)
    elif isinstance(node, dict):
        for value in node.values():
            yield from _nav_files(value)


def test_nav_lists_only_files_that_exist(mkdocs_config):
    missing = [f for f in _nav_files(mkdocs_config["nav"]) if not (REPO / "docs" / f).is_file()]
    assert missing == []


def test_every_doc_page_is_in_the_nav_or_excluded(mkdocs_config):
    in_nav = set(_nav_files(mkdocs_config["nav"]))
    excluded = set(mkdocs_config["exclude_docs"].split())
    pages = {p.name for p in (REPO / "docs").glob("*.md")}
    assert pages - in_nav - excluded == set()


def test_the_api_reference_page_carries_the_marker():
    hooks = _load_hooks_marker()
    text = (REPO / "docs" / "api-reference.md").read_text(encoding="utf-8")
    assert hooks in text.splitlines()


def _load_hooks_marker() -> str:
    source = (SCRIPTS / "mkdocs_hooks.py").read_text(encoding="utf-8")
    line = next(ln for ln in source.splitlines() if ln.startswith("MARKER = "))
    return line.split("=", 1)[1].strip().strip('"')


def test_the_docs_extra_pins_mkdocs_below_its_next_major():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"]["optional-dependencies"]["docs"]
    mkdocs = [r for r in extras if r.startswith("mkdocs>=")]
    material = [r for r in extras if r.startswith("mkdocs-material>=")]
    assert len(mkdocs) == 1 and mkdocs[0].endswith(",<2")
    assert len(material) == 1 and material[0].endswith(",<10")
