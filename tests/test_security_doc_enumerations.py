# SPDX-License-Identifier: AGPL-3.0-or-later
"""Binds the enumerations SECURITY.md, docs/server-api.md, docs/network.md,
docs/privacy.md and README.md publish to the code they describe.

Each test derives the real set from the code (the live app, the settings
schema, pyproject.toml) and compares it with the literal list in the document,
so a route, setting or extra added without a doc update fails here instead of
being found by an audit.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from localm import settings_schema
from localm.config import DEFAULT_CONFIG
from localm.inference import anthropic_protocol, ollama_protocol, responses_protocol
from localm.inference.http_server import create_app

from tests.test_cross_origin_exempt_prefix_contract import _live_cross_origin_ok
from tests.test_kernel_routes_scope_contract import (
    _BESPOKE_GATED_ROUTES,
    _PUBLIC_ROUTES,
)

_ROOT = Path(__file__).resolve().parents[1]


def _collapsed(rel: str) -> str:
    """The doc with every run of whitespace collapsed to one space, so a
    hard-wrapped sentence is not decided by where the line breaks."""
    return re.sub(r"\s+", " ", (_ROOT / rel).read_text(encoding="utf-8"))


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
#  Routes that take no bearer key                                              #
# --------------------------------------------------------------------------- #

def _keyless_routes() -> set[tuple[str, str]]:
    return set(_PUBLIC_ROUTES) | set(_BESPOKE_GATED_ROUTES)


def test_security_md_lists_exactly_the_routes_that_take_no_bearer_key():
    section = _between(_collapsed("SECURITY.md"),
                       "The routes that take no bearer key are",
                       "*State-changing endpoints* below")
    documented = set(re.findall(r"`(GET|POST) (/[^`\s]*)`", section))
    assert documented == _keyless_routes()


def test_server_api_md_names_every_keyless_route_except_the_api_landing_root():
    doc = _collapsed("docs/server-api.md")
    section = _between(doc, "session cookie) except", "Most authenticated")
    documented = set(re.findall(r"`(GET|POST) (/[^`\s]*)`", section))
    assert documented == _keyless_routes() - {("GET", "/")}


# --------------------------------------------------------------------------- #
#  Cross-origin (CSRF) exempt paths                                            #
# --------------------------------------------------------------------------- #

def test_security_md_lists_exactly_the_cross_origin_exempt_paths(app):
    live = (set(_live_cross_origin_ok(app))
            | set(ollama_protocol.CROSS_ORIGIN_OK_PATHS)
            | set(anthropic_protocol.CROSS_ORIGIN_OK_PATHS)
            | set(responses_protocol.CROSS_ORIGIN_OK_PATHS))
    section = _between(_collapsed("SECURITY.md"),
                       "Two kinds of route are exempt",
                       'A configured `"cors_origins"`')
    documented = set(re.findall(r"`(/(?:v1|api)/[^`\s]*)`", section))
    assert documented == live


def test_security_md_says_exempt_paths_are_exact_matches():
    doc = _collapsed("SECURITY.md")
    assert "matched as an exact route, not a prefix" in doc


def test_a_route_under_an_exempt_path_is_not_exempt_by_inheritance(app):
    @app.post("/v1/embeddings/child")
    async def _child():
        return {"reached": True}

    r = TestClient(app).post(
        "/v1/embeddings/child", json={},
        headers={"Origin": "http://localhost:9999"})
    assert r.status_code == 403
    assert "cross-origin" in r.json()["detail"].lower()


# --------------------------------------------------------------------------- #
#  Open-mode management-read exemptions                                        #
# --------------------------------------------------------------------------- #

def _normalise(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{id}", path)


def _open_mode_exempt_reads(app) -> set[str]:
    """GET paths under /api and /v1 that a cross-origin, token-less request
    reaches the handler for, found by probing every live GET route."""
    client = TestClient(app)
    exempt = set()
    for route in app.routes:
        if not isinstance(route, APIRoute) or "GET" not in route.methods:
            continue
        if not route.path.startswith(("/api/", "/v1/")):
            continue
        probe = re.sub(r"\{[^}]+\}", "x", route.path)
        r = client.get(probe, headers={"Origin": "http://evil.example"})
        detail = ""
        if r.headers.get("content-type", "").startswith("application/json"):
            body = r.json()
            detail = str(body.get("detail", "")) if isinstance(body, dict) else ""
        if r.status_code == 403 and (
                detail.startswith("Open-mode management requires")
                or "cross-origin" in detail.lower()):
            continue
        exempt.add(_normalise(route.path))
    return exempt


def test_security_md_lists_exactly_the_open_mode_exempt_reads(app):
    live = _open_mode_exempt_reads(app) - set(_live_cross_origin_ok(app))
    section = _between(_collapsed("SECURITY.md"),
                       "The exempt read paths", "never enter this gate")
    documented = {_normalise(p) for p in
                  re.findall(r"`GET (/(?:v1|api)/[^`\s]*)`", section)}
    assert documented == live


# --------------------------------------------------------------------------- #
#  Owner-only configuration keys                                               #
# --------------------------------------------------------------------------- #

def test_server_api_md_lists_exactly_the_owner_only_config_keys():
    doc = _collapsed("docs/server-api.md")
    section = _between(doc, "These settings widen a trust boundary",
                       "Two further keys")
    known = set(DEFAULT_CONFIG) | {f.key for f in settings_schema.CORE_FIELDS}
    documented = {t for t in re.findall(r"`([a-z][a-z0-9_]*)`", section)
                  if t in known}
    assert documented == set(settings_schema.admin_only_keys())


# --------------------------------------------------------------------------- #
#  pip extras                                                                  #
# --------------------------------------------------------------------------- #

def test_readme_extras_table_lists_exactly_the_pyproject_extras():
    import tomllib
    with open(_ROOT / "pyproject.toml", "rb") as fh:
        extras = set(tomllib.load(fh)["project"]["optional-dependencies"])
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    table = _between(text, "The defined extras are:", "Not every plugin needs an extra")
    documented = set(re.findall(r"^\| `([a-z0-9_-]+)` \|", table, re.M))
    assert documented == extras


# --------------------------------------------------------------------------- #
#  net_mode = off is not an absolute                                           #
# --------------------------------------------------------------------------- #

_ABSOLUTE_OFF_CLAIMS = (
    "every outbound request",
    "with no exception",
    "refuses it unconditionally",
    "always governs",
    "is a real kill switch",
    "one documented exception",
)


@pytest.mark.parametrize("rel", ["README.md", "SECURITY.md", "docs/network.md",
                                 "docs/privacy.md"])
def test_docs_do_not_call_net_mode_off_an_absolute(rel):
    doc = _collapsed(rel).lower()
    found = [p for p in _ABSOLUTE_OFF_CLAIMS if p in doc]
    assert not found


def test_net_allow_model_downloads_is_admin_only_and_documented_where_off_is():
    assert "net_allow_model_downloads" in settings_schema.admin_only_keys()
    for rel in ("SECURITY.md", "docs/network.md", "docs/privacy.md"):
        assert "net_allow_model_downloads" in _collapsed(rel), rel


# --------------------------------------------------------------------------- #
#  Registry model types, plugin nav icons, CUDA driver floor                   #
# --------------------------------------------------------------------------- #

def test_cli_md_model_type_lists_match_the_registry():
    from localm.model_manager.registry import MODEL_TYPES
    doc = _collapsed("docs/cli.md")
    prose = _between(doc, "The types are", ".")
    in_prose = set(re.findall(r"`([a-z-]+)`", prose))
    comment = _between(doc, "# types: ", " ```")
    in_comment = set(comment[len("# types: "):].split())
    assert in_prose == set(MODEL_TYPES)
    assert in_comment == set(MODEL_TYPES)


def test_plugins_md_nav_icons_match_the_gui_icon_map():
    js = (_ROOT / "localm/plugins/gui/static/app/settings-perf.js").read_text(
        encoding="utf-8")
    body = re.search(r"export const NAV_ICON = \{([^}]*)\}", js).group(1)
    live = set(re.findall(r"(\w+):", body))
    doc = (_ROOT / "docs/plugins.md").read_text(encoding="utf-8")
    block = _between(doc, "# one of a fixed set:", "anything else falls back")
    documented = set(re.findall(r"[a-z]+", block.split(":", 1)[1])) - {"anything", "else"}
    assert documented == live


def test_cuda_driver_floor_in_docs_matches_setup_llama():
    from localm.setup_llama.cuda import _MIN_DRIVER_CUDA
    major, minor = _MIN_DRIVER_CUDA["cuda-13"]
    stated = set()
    for rel in ("README.md", "docs/gpu-setup.md", "docs/docker.md"):
        stated |= set(re.findall(r"CUDA (?:>= )?(13\.\d+)", _collapsed(rel)))
    assert stated == {f"{major}.{minor}"}


def test_server_api_md_status_codes_match_the_protocol_map():
    from localm.inference.protocol import STATUS_CODE_BY_TEXT
    doc = _collapsed("docs/server-api.md")
    section = _between(doc, "(a stable id):",
                       "A streaming request that is compacted")
    documented = set(re.findall(r"`([a-z_]+)`", section))
    assert documented == set(STATUS_CODE_BY_TEXT.values())
