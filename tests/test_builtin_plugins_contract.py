# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract & client-asset guards over EVERY shipped builtin plugin.

These are invariant guards, not feature tests. They enumerate the builtins from
the bundled store and assert the cross-cutting rules that are easy to break and
invisible from any single file:

  * every builtin's manifest parses, targets the supported api_version, and
    declares a capability scope + a register entry (contract conformance); and a
    declared client_entry lives under a declared assets_dir and exists on disk;
  * every builtin whose [surface] declares a client_entry actually SERVES that
    module at /plugins/<name>/<entry> with a JS MIME once enabled, and 404s once
    disabled, so a client_entry plugin can never silently 404.

The lists are built at import time by parsing each builtin's plugin.toml (no
module import), so a new builtin - or a new client_entry plugin - is covered
automatically the moment it ships, with no edit here.

Open-mode (no API key) so the mounted, auto-scoped routes are reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _builtin_root() -> Path:
    import localm.plugins.engine as eng
    return Path(eng.__file__).resolve().parent / "builtin"


def _builtin_specs():
    from localm.plugins.engine import parse_spec
    return [parse_spec(m.parent, builtin=True)
            for m in sorted(_builtin_root().glob("*/plugin.toml"))]


_SPECS = _builtin_specs()
_CLIENT_ENTRY = [s for s in _SPECS if s.surface and s.surface.client_entry]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


# --------------------------------------------------------------------------- #
#  Contract conformance over ALL builtins (manifest-level, no module import)   #
# --------------------------------------------------------------------------- #

def test_builtins_are_discoverable():
    """Sanity: the store shelf is non-empty (so the parametrized guards below
    actually run against real plugins, not an empty list)."""
    assert _SPECS, "no builtin plugins found in the store"


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_builtin_manifest_conforms_to_contract(spec):
    assert spec.compatible(), (
        f"{spec.name}: api_version {spec.api_version} unsupported by the engine")
    assert spec.scope, f"{spec.name}: no capability scope declared"
    assert spec.register_entry, f"{spec.name}: no register entry declared"
    # A declared client_entry must live under a declared assets_dir and exist.
    if spec.surface and spec.surface.client_entry:
        assert spec.surface.assets_dir, (
            f"{spec.name}: declares client_entry but no assets_dir")
        entry = (_builtin_root() / spec.name / spec.surface.assets_dir
                 / spec.surface.client_entry)
        assert entry.is_file(), (
            f"{spec.name}: client_entry file does not exist: {entry}")


# --------------------------------------------------------------------------- #
#  Every client_entry builtin actually SERVES its module once enabled          #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _CLIENT_ENTRY,
                    reason="no builtin declares a client_entry")
@pytest.mark.parametrize("spec", _CLIENT_ENTRY, ids=[s.name for s in _CLIENT_ENTRY])
def test_builtin_client_entry_is_served(env, spec):
    """An enabled client_entry builtin serves its module at /plugins/<name>/...
    with a JS module MIME, and stops serving it once disabled. This is the
    outcome guard for the silent-404 bug class, regardless of whether the asset
    is mounted explicitly by the plugin or auto-mounted by the engine."""
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)                          # default roots = the real builtins
    url = f"/plugins/{spec.name}/{spec.surface.client_entry}"
    with TestClient(app) as c:
        r = c.post(f"/api/plugins/{spec.name}/install")
        assert r.status_code == 200, (
            f"install {spec.name}: {r.status_code} {r.text}")

        js = c.get(url)
        assert js.status_code == 200, (
            f"{url} -> {js.status_code}: client_entry declared but not served")
        assert js.headers["content-type"].startswith("text/javascript"), (
            f"{url} served as {js.headers.get('content-type')!r}, "
            "not a JS module MIME (ES import() would be blocked)")

        assert c.post(f"/api/plugins/{spec.name}/disable").status_code == 200
        assert c.get(url).status_code == 404, (
            f"{url} still served after disable (asset not unmounted)")
