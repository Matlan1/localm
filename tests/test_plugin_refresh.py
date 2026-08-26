# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refreshing stale installed builtin-plugin copies on upgrade.

Enabling a first-party plugin copies it from the bundled store into the data
dir. A copy that is never refreshed keeps SHADOWING the newer store source after
a localm upgrade, so the user silently runs stale plugin code (including missing
bug and security fixes). These tests cover the content-hash + provenance-marker
refresh: a stale copy is re-synced, user config survives, an externally-installed
plugin is never clobbered, and an up-to-date copy is left alone. They also pin
the sys.modules submodule purge that lets a reload pick up freshly written code.
"""

import textwrap

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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


def _valued_plugin(root, name, value, *, with_backend=True):
    """A minimal plugin whose GET /api/<name>/val returns *value*. When
    *with_backend* the value comes from a sibling ``backend.py`` imported as
    ``from . import backend`` - exercising the submodule import path that the
    real image plugin uses."""
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n',
        encoding="utf-8")
    if with_backend:
        (pdir / "backend.py").write_text(
            f"def val():\n    return {value}\n", encoding="utf-8")
        plug = f'''
            from fastapi import APIRouter
            from . import backend as _b
            _r = APIRouter()

            @_r.get("/api/{name}/val")
            def _val():
                return {{"val": _b.val()}}

            def register(host):
                host.mount_router(_r)

            def unregister():
                pass
        '''
    else:
        plug = f'''
            from fastapi import APIRouter
            _r = APIRouter()

            @_r.get("/api/{name}/val")
            def _val():
                return {{"val": {value}}}

            def register(host):
                host.mount_router(_r)

            def unregister():
                pass
        '''
    (pdir / "plug.py").write_text(textwrap.dedent(plug), encoding="utf-8")
    return pdir


def _enable(name):
    from localm.config import load_config, save_config
    cfg = load_config()
    cfg["plugins_enabled"] = sorted(set(cfg.get("plugins_enabled", [])) | {name})
    save_config(cfg)


# --------------------------------------------------------------------------- #
#  Core: a stale builtin is refreshed on launch                               #
# --------------------------------------------------------------------------- #

def test_stale_builtin_refreshed_on_load(env):
    """load_enabled re-copies an installed builtin whose store source changed,
    and the LIVE route then serves the fresh code. Negative-testable: without
    the refresh the stale v1 copy loads and /val returns 1."""
    from localm.plugins.engine import (
        PluginManager, _dir_content_hash, _read_marker)
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref1", 2)            # bundled store: fresh (v2)
    _valued_plugin(inst, "ref1", 1)             # installed: stale (v1), no marker
    _enable("ref1")

    app = FastAPI()
    PluginManager(app, store_root=store, installed_root=inst).load_enabled()

    # on-disk copy now matches the store, byte for byte (marker excluded)
    assert _dir_content_hash(inst / "ref1") == _dir_content_hash(store / "ref1")
    assert (inst / "ref1" / "backend.py").read_text() == \
        (store / "ref1" / "backend.py").read_text()
    m = _read_marker(inst / "ref1")
    assert m and m["source"] == "store"
    # and the running plugin serves the refreshed code
    with TestClient(app) as c:
        assert c.get("/api/ref1/val").json()["val"] == 2


def test_refresh_preserves_user_config(env):
    """Per-plugin user config lives in config.json (NOT the plugin dir), so a
    dir re-copy must leave it intact."""
    from localm.config import load_config, save_config
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref2", 2)
    _valued_plugin(inst, "ref2", 1)
    cfg = load_config()
    cfg.setdefault("plugins", {})["ref2"] = {"backend": "custom", "knob": 7}
    save_config(cfg)
    _enable("ref2")

    PluginManager(FastAPI(), store_root=store, installed_root=inst).load_enabled()

    assert load_config()["plugins"]["ref2"] == {"backend": "custom", "knob": 7}
    assert (inst / "ref2" / "backend.py").read_text().strip().endswith("return 2")


# --------------------------------------------------------------------------- #
#  Safety: never clobber a third-party (externally-installed) plugin          #
# --------------------------------------------------------------------------- #

def test_third_party_install_never_refreshed(env):
    """A plugin with no bundled-store counterpart is never a refresh target."""
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    src = env / "thirdparty"
    _valued_plugin(src, "tp1", 9, with_backend=False)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.set_installed_from_dir(src / "tp1")     # external install (marker=external)
    before = (inst / "tp1" / "plug.py").read_text()

    assert mgr.refresh() == []                  # nothing to refresh
    assert (inst / "tp1" / "plug.py").read_text() == before


def test_external_marker_blocks_refresh_even_for_store_name(env):
    """Even when a store plugin of the SAME name exists, an installed copy that
    declares external provenance is left untouched (defends a user who shadowed a
    builtin name with their own plugin)."""
    from localm.plugins.engine import (
        PluginManager, _dir_content_hash, _write_marker)
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref3", 2)            # a builtin of this name exists
    pdir = _valued_plugin(inst, "ref3", 1)      # but the install is "external"
    _write_marker(pdir, "external", _dir_content_hash(pdir))
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)

    assert mgr.refresh() == []
    assert (inst / "ref3" / "backend.py").read_text().strip().endswith("return 1")


# --------------------------------------------------------------------------- #
#  Idempotence: an up-to-date copy is not re-copied                           #
# --------------------------------------------------------------------------- #

def test_uptodate_builtin_not_recopied(env):
    """A fresh install records the source hash; a later refresh is a no-op."""
    from localm.plugins.engine import PluginManager, _read_marker
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "up1", 5)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.install("up1")
    m = _read_marker(inst / "up1")
    assert m and m["source"] == "store"

    assert mgr.refresh() == []                  # whole-set refresh: nothing changed
    assert mgr.refresh("up1") == []             # targeted refresh: still a no-op


def test_legacy_identical_copy_adopts_marker_without_recopy(env):
    """A pre-feature install (no marker) that already matches the store adopts
    the provenance marker WITHOUT a re-copy."""
    from localm.plugins.engine import PluginManager, _read_marker
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "leg1", 5)
    _valued_plugin(inst, "leg1", 5)             # identical content, no marker
    assert _read_marker(inst / "leg1") is None
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)

    assert mgr.refresh() == []                  # identical -> not re-copied
    m = _read_marker(inst / "leg1")
    assert m and m["source"] == "store"         # but provenance is now recorded


# --------------------------------------------------------------------------- #
#  Robustness: a reload after on-disk change picks up the new code            #
# --------------------------------------------------------------------------- #

def test_unload_purges_submodules_from_sys_modules(env):
    """_unload must drop the WHOLE plugin namespace - the top-level module AND
    its submodules (e.g. '<uniq>.backend'). Popping only the top-level name
    leaves a stale '.backend' cached that shadows a later load."""
    import sys
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "rel1", 1)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.install("rel1")
    assert "_localm_plugin_rel1.backend" in sys.modules   # submodule cached on load

    mgr._unload("rel1")
    assert "_localm_plugin_rel1" not in sys.modules
    assert "_localm_plugin_rel1.backend" not in sys.modules   # purged, not just top


def test_runtime_refresh_reloads_fresh_code(env):
    """End-to-end runtime refresh: a newer store version is detected, the live
    plugin is re-copied AND reloaded, and the route serves the new code. This
    passes only because both the on-disk re-copy and the sys.modules submodule
    purge work together (a stale cached '.backend' would otherwise shadow it)."""
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "rel2", 1)
    app = FastAPI()
    mgr = PluginManager(app, store_root=store, installed_root=inst)
    mgr.install("rel2")
    with TestClient(app) as c:
        assert c.get("/api/rel2/val").json()["val"] == 1

    _valued_plugin(store, "rel2", 2)            # ship a newer store version
    assert mgr.refresh("rel2") == ["rel2"]
    assert (inst / "rel2" / "backend.py").read_text() == \
        (store / "rel2" / "backend.py").read_text()
    with TestClient(app) as c:
        assert c.get("/api/rel2/val").json()["val"] == 2


# --------------------------------------------------------------------------- #
#  HTTP surface + targeted-refresh errors                                     #
# --------------------------------------------------------------------------- #

def test_refresh_endpoint_over_real_builtins(env, monkeypatch):
    """The /api/plugins/{name}/refresh management route: a just-installed plugin
    reports up-to-date; an unknown name 404s."""
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    from localm.plugins.engine import attach_engine
    app = FastAPI()
    attach_engine(app)                          # default roots = the real builtins
    with TestClient(app) as c:
        assert c.post("/api/plugins/voice/install").status_code == 200
        r = c.post("/api/plugins/voice/refresh")
        assert r.status_code == 200
        assert r.json()["status"] in ("refreshed", "up-to-date")
        assert c.post("/api/plugins/ghost/refresh").status_code == 404


def test_refresh_unknown_name_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(None, store_root=env / "store", installed_root=env / "inst")
    with pytest.raises(KeyError):
        mgr.refresh("ghost")                    # no such builtin
