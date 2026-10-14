# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binds enumerations published in the CLI, API, install and feature docs to
the code they describe.

Each test derives the real set from the code (a module constant, a registry,
the live app, a workflow file) and compares it with the literal list in the
document, so a status, key, tool, module or backend added without a doc update
fails here instead of being found by an audit.
"""

import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute

from localm import scopes
from localm.inference import anthropic_protocol, ollama_protocol, responses_protocol
from localm.inference.http_server import _BACKEND_ERROR_STATUS, create_app

_ROOT = Path(__file__).resolve().parents[1]


def _text(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8")


def _collapsed(rel: str) -> str:
    return re.sub(r"\s+", " ", _text(rel))


def _between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


@pytest.fixture
def app(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    return create_app(None)


# --------------------------------------------------------------------------- #
#  docs/server-api.md and docs/ollama-api.md                                   #
# --------------------------------------------------------------------------- #

def test_server_api_generation_failure_row_lists_every_backend_error_status():
    doc = _collapsed("docs/server-api.md")
    row = _between(doc, "The request is refused before generation starts",
                   "| Generation started and then failed")
    documented = {int(s) for s in re.findall(r"`(\d{3})`", row)}
    assert documented == {status for _, status in _BACKEND_ERROR_STATUS}


def _documented_error_types(section: str) -> dict[str, set[int]]:
    """``{type: {statuses}}`` from prose like ``permission_error` (403), ...``."""
    out: dict[str, set[int]] = {}
    for kind, statuses in re.findall(r"`([a-z_]+)` \((\d{3}[^)]*)\)", section):
        out.setdefault(kind, set()).update(int(s) for s in re.findall(r"\b\d{3}\b", statuses))
    return out


def test_server_api_anthropic_error_types_match_the_status_map():
    doc = _collapsed("docs/server-api.md")
    section = _between(doc, "The `type` follows the status: `authentication_error`",
                       "### `POST /v1/messages/count_tokens`")
    documented = _documented_error_types(section)
    named = {s for statuses in documented.values() for s in statuses}
    assert named == set(anthropic_protocol._ERROR_TYPES) - {400, 422}
    assert anthropic_protocol.error_type(400) == anthropic_protocol.error_type(422) == "invalid_request_error"
    for kind, statuses in documented.items():
        for status in statuses:
            assert anthropic_protocol.error_type(status) == kind
    assert anthropic_protocol.error_type(418) == "invalid_request_error"
    assert anthropic_protocol.error_type(500) == "api_error"


def test_server_api_responses_error_types_match_the_status_map():
    doc = _collapsed("docs/server-api.md")
    section = _between(doc, "Errors use OpenAI's shape",
                       "### `POST /v1/embeddings`")
    documented = _documented_error_types(section)
    assert set(documented) == {"authentication_error", "permission_error",
                               "not_found_error"}
    for kind, statuses in documented.items():
        for status in statuses:
            body = responses_protocol.error_body(status, "x")
            assert body["error"]["type"] == kind
    assert responses_protocol.error_body(400, "x")["error"]["type"] == "invalid_request_error"
    assert responses_protocol.error_body(500, "x")["error"]["type"] == "server_error"


def test_ollama_api_auth_table_names_only_mounted_paths_and_every_constant_path(app):
    doc = _text("docs/ollama-api.md")
    table = _between(doc, "| Route | Needs |", "In open mode")
    documented = {p.rstrip("/") for p in re.findall(r"`(?:\w+ )?(/api/[A-Za-z/{}]+)`", table)}
    for token in re.findall(r"`(?:POST|GET|DELETE)? ?(/api/[A-Za-z]+)`", table):
        documented.add(token)
    routes = {r.path for r in app.routes if isinstance(r, APIRoute)}
    documented = {"/api/blobs/{digest}" if p.startswith("/api/blobs") else p
                  for p in documented}
    assert documented <= routes, sorted(documented - routes)
    required = (set(ollama_protocol.INFERENCE_POST_PATHS)
                | set(ollama_protocol.READ_GET_PATHS)
                | {ollama_protocol.SHOW_PATH, ollama_protocol.COPY_PATH})
    assert required <= documented, sorted(required - documented)


# --------------------------------------------------------------------------- #
#  docs/cli.md                                                                 #
# --------------------------------------------------------------------------- #

def _tool_table_rows() -> dict[str, tuple[str, str]]:
    doc = _text("docs/cli.md")
    section = doc[doc.index("### Built-in tools"):]
    section = section[:section.index("\n## ")] if "\n## " in section else section
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"`\w+`", cells[0]):
            rows[cells[0].strip("`")] = (cells[2], cells[3])
    return rows


def test_cli_md_builtin_tool_table_matches_the_tool_registry():
    from localm.plugins.coder.tools.registry import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY

    rows = _tool_table_rows()
    assert set(rows) == set(TOOL_REGISTRY)
    assert f"The agent ships {len(TOOL_REGISTRY)} built-in tools" in _collapsed("docs/cli.md")
    for name, (confirms, restricted) in rows.items():
        tool = TOOL_REGISTRY[name]
        assert (confirms == "yes") == bool(tool.destructive or tool.ask_by_default), name
        assert (restricted == "yes") == (name in SAFE_RESTRICTED_TOOLS), name


def test_cli_md_privileged_scope_list_matches_the_code():
    doc = _collapsed("docs/cli.md")
    sentence = _between(doc, "Privileged scopes (", ") are refused by `key create`")
    documented = set(re.findall(r"`([a-z:]+)`", sentence))
    assert documented == set(scopes.PRIVILEGED_SCOPES)


def test_cli_md_plugin_setup_defaults_match_the_code():
    from localm.cli.plugins import _SETUP_DEFAULTS

    doc = _collapsed("docs/cli.md")
    line = _between(doc, "localm plugin setup --defaults", "(also:")
    named = set(re.findall(r"\b(coder|rag|web|tts)\b", line))
    assert named == set(_SETUP_DEFAULTS)


# --------------------------------------------------------------------------- #
#  Install guides                                                              #
# --------------------------------------------------------------------------- #

def test_gpu_setup_backend_commands_cover_every_selectable_backend():
    from localm.setup_llama.cli import BACKENDS

    doc = _text("docs/gpu-setup.md")
    block = _between(doc, "## Choosing a backend", "\n| ")
    documented = set(re.findall(r"setup-llama --backend (\S+)", block))
    assert documented == set(BACKENDS) - {"auto"}


def test_docker_md_backend_tags_match_the_build_matrix():
    workflow = _text(".github/workflows/docker.yml")
    matrix = re.search(r"matrix:\s*\n\s*backend:\s*\[([^\]]*)\]", workflow).group(1)
    backends = {b.strip() for b in matrix.split(",")}
    doc = _collapsed("docs/docker.md")
    for backend in backends - {"cpu"}:
        assert f"`<version>-{backend}`, `{backend}`" in doc, backend
    sentence = _between(doc, "`latest`, `cpu`, `vulkan`", "move only")
    assert set(re.findall(r"`([a-z0-9]+)`", sentence)) == backends | {"latest"}


# --------------------------------------------------------------------------- #
#  Feature pages                                                               #
# --------------------------------------------------------------------------- #

def test_plugins_md_manifest_reference_lists_exactly_the_known_keys():
    from localm.plugins.contract import KNOWN_PLUGIN_KEYS, KNOWN_SURFACE_KEYS

    doc = _text("docs/plugins.md")
    block = _between(doc, "## `plugin.toml` reference", "An unknown or misspelled key")
    plugin_part, surface_part = block.split("[surface]")
    plugin_keys = set(re.findall(r"^(\w+)\s*=", plugin_part, flags=re.M))
    surface_keys = set(re.findall(r"^(\w+)\s*=", surface_part, flags=re.M))
    assert plugin_keys == set(KNOWN_PLUGIN_KEYS) - {"entry"}
    assert surface_keys == set(KNOWN_SURFACE_KEYS)


def test_jobs_md_task_and_schedule_kinds_match_the_store():
    from localm.plugins.builtin.jobs.store import SCHEDULE_KINDS, TASK_KINDS

    doc = _collapsed("docs/jobs.md")
    body = _between(doc, "A create/update body carries:", "`schedule`")
    assert set(re.findall(r"`(chat|coder|memory|rag)`", body)) == set(TASK_KINDS)
    assert set(re.findall(r"`(interval|cron)`", body)) == set(SCHEDULE_KINDS)


def test_skills_md_size_caps_match_the_loader():
    from localm.plugins.coder import skills

    doc = _collapsed("docs/skills.md")
    sentence = _between(doc, "The instruction body is cut at", "bundled file at")
    assert f"{skills._MAX_BODY:,}" in sentence
    assert f"{skills._MAX_FILE:,}" in _collapsed("docs/skills.md")


def test_memory_md_limits_match_the_store():
    import localm.memory as memory
    from localm.memory import store

    doc = _collapsed("docs/memory.md")
    assert f"holds at most {store.N_MAX} records, each capped at {store.MAX_TEXT_LEN} characters" in doc
    assert f"Up to {memory.MAX_INJECT} facts are injected" in doc
    assert f"a {memory.INJECT_BLOCK_CHARS:,}-character block" in doc
    assert f"holds {store.N_MAX} records" in doc


def test_llamacpp_binding_module_table_lists_every_module_in_the_package():
    doc = _text("docs/llamacpp-binding.md")
    table = _between(doc, "| `_loader.py`", "## DLL Loading")
    documented = set(re.findall(r"^\| `([\w.]+\.py)` \|", "| `_loader.py`" + table[len("| `_loader.py`"):], flags=re.M))
    package = _ROOT / "localm" / "inference" / "backends" / "llamacpp"
    assert documented == {p.name for p in package.glob("*.py")}


def test_architecture_md_route_modules_match_the_routes_package():
    doc = _collapsed("docs/architecture.md")
    sentence = _between(doc, "`inference/routes/` modules (", "compatibility surfaces")
    documented = set(re.findall(r"\b(chat|models|config|keys|session|admin|system|gpu|peer_routing|ollama|anthropic|responses)\b", sentence))
    package = _ROOT / "localm" / "inference" / "routes"
    modules = {p.stem for p in package.glob("*.py") if not p.stem.startswith("_")}
    assert documented == modules
