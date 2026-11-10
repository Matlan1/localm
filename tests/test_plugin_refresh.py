# SPDX-License-Identifier: AGPL-3.0-or-later
"""Installed first-party plugins run from the store, and old copied installs
are converted.

Installing a first-party plugin writes a marker folder (manifest + provenance
marker) into the data dir; its code is read from the store. An install made by
an older localm is a full copy of the store folder: it is never imported, and
``refresh`` / ``load_enabled`` / ``enable`` delete its copied code. These tests
cover that conversion, that user config survives, that an externally-installed
plugin is never touched, and the sys.modules purge for plugins loaded under a
synthetic name.
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
#  An old copied install is converted, and the store code is what runs         #
# --------------------------------------------------------------------------- #

def test_old_copy_is_converted_on_load_and_the_store_code_runs(env):
    """load_enabled converts an installed full copy into a marker-only install
    and the LIVE route serves the store's code, not the copy's."""
    from localm.plugins.engine import PluginManager, _read_marker
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref1", 2)            # bundled store: v2
    _valued_plugin(inst, "ref1", 1)             # installed: an old v1 copy, no marker
    _enable("ref1")

    app = FastAPI()
    PluginManager(app, store_root=store, installed_root=inst).load_enabled()

    assert sorted(p.name for p in (inst / "ref1").iterdir()) == [
        ".localm-source.json", "plugin.toml"]
    assert _read_marker(inst / "ref1") == {"source": "store", "package": True}
    with TestClient(app) as c:
        assert c.get("/api/ref1/val").json()["val"] == 2


def test_conversion_preserves_user_config(env):
    """Per-plugin user config lives in config.json (NOT the plugin dir), so the
    conversion leaves it intact."""
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
    assert not (inst / "ref2" / "backend.py").exists()


# --------------------------------------------------------------------------- #
#  Safety: never touch a third-party (externally-installed) plugin            #
# --------------------------------------------------------------------------- #

def test_third_party_install_never_converted(env):
    """A plugin with no bundled-store counterpart is never a conversion target."""
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    src = env / "thirdparty"
    _valued_plugin(src, "tp1", 9, with_backend=False)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.set_installed_from_dir(src / "tp1")     # external install (marker=external)
    before = (inst / "tp1" / "plug.py").read_text()

    assert mgr.refresh() == []
    assert (inst / "tp1" / "plug.py").read_text() == before


def test_external_marker_keeps_its_own_code_even_for_store_name(env):
    """Even when a store plugin of the SAME name exists, an installed copy that
    declares external provenance is left untouched and its own code runs."""
    from localm.plugins.engine import (
        PluginManager, _dir_content_hash, _write_marker)
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref3", 2)            # a builtin of this name exists
    pdir = _valued_plugin(inst, "ref3", 1)      # but the install is "external"
    _write_marker(pdir, "external", _dir_content_hash(pdir))
    app = FastAPI()
    mgr = PluginManager(app, store_root=store, installed_root=inst)

    assert mgr.refresh() == []
    assert (inst / "ref3" / "backend.py").read_text().strip().endswith("return 1")
    mgr.enable("ref3")
    with TestClient(app) as c:
        assert c.get("/api/ref3/val").json()["val"] == 1


# --------------------------------------------------------------------------- #
#  A marker-only install needs no refresh                                     #
# --------------------------------------------------------------------------- #

def test_fresh_install_is_a_marker_and_refresh_is_a_no_op(env):
    from localm.plugins.engine import PluginManager, _read_marker
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "up1", 5)
    mgr = PluginManager(FastAPI(), store_root=store, installed_root=inst)
    mgr.install("up1")
    assert _read_marker(inst / "up1") == {"source": "store", "package": True}
    assert not (inst / "up1" / "plug.py").exists()

    assert mgr.refresh() == []
    assert mgr.refresh("up1") == []


def test_a_changed_store_takes_effect_on_reload_without_a_refresh(env):
    """The store is the code: after the store changes, a disable + enable serves
    the new code with no refresh step."""
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "rel2", 1)
    app = FastAPI()
    mgr = PluginManager(app, store_root=store, installed_root=inst)
    mgr.install("rel2")
    with TestClient(app) as c:
        assert c.get("/api/rel2/val").json()["val"] == 1

    _valued_plugin(store, "rel2", 2)            # ship a newer store version
    mgr.disable("rel2")
    mgr.enable("rel2")
    with TestClient(app) as c:
        assert c.get("/api/rel2/val").json()["val"] == 2


def test_unload_purges_submodules_from_sys_modules(env):
    """_unload drops the WHOLE namespace of a plugin loaded under a synthetic
    name: the top-level module AND its submodules (e.g. '<uniq>.backend')."""
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
        assert r.json()["status"] == "up-to-date"
        assert c.post("/api/plugins/ghost/refresh").status_code == 404


def test_refresh_unknown_name_raises(env):
    from localm.plugins.engine import PluginManager
    mgr = PluginManager(None, store_root=env / "store", installed_root=env / "inst")
    with pytest.raises(KeyError):
        mgr.refresh("ghost")                    # no such builtin


def _undeletable(monkeypatch, path):
    """Make Path.unlink fail for *path* only."""
    from pathlib import Path
    real = Path.unlink

    def unlink(self, *a, **kw):
        if Path(self) == path:
            raise PermissionError("locked")
        return real(self, *a, **kw)
    monkeypatch.setattr(Path, "unlink", unlink)


def test_a_failed_conversion_is_reported_not_called_up_to_date(env, monkeypatch, caplog):
    import logging

    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref9", 2)
    _valued_plugin(inst, "ref9", 1)
    _undeletable(monkeypatch, inst / "ref9" / "backend.py")
    mgr = PluginManager(None, store_root=store, installed_root=inst)

    with caplog.at_level(logging.WARNING, logger="localm.plugins"):
        with pytest.raises(RuntimeError, match="could not be fully removed"):
            mgr.refresh("ref9")
        assert mgr.refresh() == []
    assert "ref9" in mgr.last_refresh_failures
    assert any("backend.py" in r.getMessage() for r in caplog.records)
    assert (inst / "ref9" / "backend.py").is_file()


def test_refresh_routes_report_a_failed_conversion(env, monkeypatch):
    from localm.plugins.engine import attach_engine
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref9", 2)
    _valued_plugin(inst, "ref9", 1)
    _undeletable(monkeypatch, inst / "ref9" / "backend.py")
    monkeypatch.setattr("localm.plugins.engine._store_root", lambda: store)
    monkeypatch.setattr("localm.plugins.loader.plugins_dir", lambda: inst)
    app = FastAPI()
    attach_engine(app)
    with TestClient(app) as c:
        one = c.post("/api/plugins/ref9/refresh")
        assert one.status_code == 500 and "could not be fully removed" in one.json()["detail"]
        every = c.post("/api/plugins/refresh").json()
        assert every["status"] == "partial" and "ref9" in every["failed"]


def test_enable_runs_the_store_code_when_the_conversion_fails(env, monkeypatch):
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "ref9", 2)
    _valued_plugin(inst, "ref9", 1)
    _undeletable(monkeypatch, inst / "ref9" / "backend.py")
    app = FastAPI()
    mgr = PluginManager(app, store_root=store, installed_root=inst)
    mgr.discover()

    mgr.enable("ref9")

    with TestClient(app) as c:
        assert c.get("/api/ref9/val").json()["val"] == 2
    assert "could not be fully removed" in mgr.api_state()["errors"].get("ref9", "")


def _link_dir(link, target):
    """A directory link: a junction on Windows, a symlink elsewhere."""
    import os
    import sys
    if sys.platform == "win32":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


def test_conversion_never_follows_or_removes_a_linked_folder(env):
    """A folder in the old copy that links elsewhere (a junction or symlink) is
    kept as it is: nothing behind it is deleted, and the link stays."""
    from localm.plugins.engine import PluginManager
    store, inst = env / "store", env / "installed"
    _valued_plugin(store, "lnk1", 2)
    (store / "lnk1" / "static").mkdir()
    (store / "lnk1" / "static" / "a.js").write_text("shipped", encoding="utf-8")
    _valued_plugin(inst, "lnk1", 1)
    outside = env / "outside"
    (outside / "__pycache__").mkdir(parents=True)
    (outside / "a.js").write_text("mine", encoding="utf-8")
    (outside / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (outside / "empty").mkdir()
    _link_dir(inst / "lnk1" / "static", outside)

    assert PluginManager(None, store_root=store, installed_root=inst).refresh("lnk1") == ["lnk1"]

    assert (outside / "a.js").read_text(encoding="utf-8") == "mine"
    assert (outside / "__pycache__" / "x.pyc").is_file()
    assert (outside / "empty").is_dir()
    assert (inst / "lnk1" / "static" / "a.js").is_file()
    assert not (inst / "lnk1" / "plug.py").exists()
