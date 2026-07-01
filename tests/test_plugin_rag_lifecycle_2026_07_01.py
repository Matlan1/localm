# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-07-01 plugin/RAG-lifecycle backlog cluster.

  REC-ONFIRSTUSE      - on_first_use fires once, at first activation, persisted
  REC-PLUGIN-REQUIRES - enable refuses an unmet dep; uninstall cascade-disables dependents
  REC-RAG-EMBED-PARITY - `rag add --embed` embeds at index time (GUI parity)
"""

import textwrap

import pytest
from fastapi import FastAPI


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _make_plugin(root, name, body, *, toml_extra=""):
    pdir = root / name
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "plugin.toml").write_text(
        f'[plugin]\nname = "{name}"\nscope = "{name}"\nregister = "plug"\n{toml_extra}',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return pdir


_PLAIN = '''
    from fastapi import APIRouter
    _r = APIRouter()
    def register(host):
        host.mount_router(_r)
    def unregister():
        pass
'''

_WITH_FIRST_USE = '''
    from fastapi import APIRouter
    _r = APIRouter()
    def register(host):
        host.mount_router(_r)
    def unregister():
        pass
    def on_first_use():
        from localm.config import home_dir
        p = home_dir() / "gamma-fu.txt"
        n = int(p.read_text()) if p.exists() else 0
        p.write_text(str(n + 1), encoding="utf-8")
'''


def _mgr(env):
    from localm.plugins.engine import PluginManager
    store = env / "store"
    installed = env / "installed"
    store.mkdir(parents=True, exist_ok=True)
    installed.mkdir(parents=True, exist_ok=True)
    return PluginManager(FastAPI(), store_root=store, installed_root=installed), store


# --------------------------------------------------------------------------- #
#  REC-ONFIRSTUSE
# --------------------------------------------------------------------------- #

def test_on_first_use_fires_once_and_is_persisted(env):
    mgr, store = _mgr(env)
    _make_plugin(store, "gamma", _WITH_FIRST_USE)
    marker = env / "gamma-fu.txt"

    mgr.set_installed_state("gamma", True, enable=False)
    mgr.enable("gamma")                       # first activation -> on_first_use fires
    assert marker.exists() and marker.read_text() == "1"

    # A reload (disable -> enable) must NOT re-fire it (persisted).
    mgr.disable("gamma")
    mgr.enable("gamma")
    assert marker.read_text() == "1", "on_first_use must fire only ONCE per plugin"


# --------------------------------------------------------------------------- #
#  REC-PLUGIN-REQUIRES
# --------------------------------------------------------------------------- #

def test_enable_refuses_when_a_required_plugin_is_missing(env):
    mgr, store = _mgr(env)
    _make_plugin(store, "alpha", _PLAIN)
    _make_plugin(store, "beta", _PLAIN, toml_extra='requires = ["alpha"]\n')

    mgr.set_installed_state("beta", True, enable=False)   # install beta, NOT alpha
    with pytest.raises(ValueError, match="requires"):
        mgr.set_enabled_state("beta", True)
    with pytest.raises(ValueError, match="requires"):
        mgr.enable("beta")


def test_enable_succeeds_once_the_dependency_is_installed(env):
    mgr, store = _mgr(env)
    _make_plugin(store, "alpha", _PLAIN)
    _make_plugin(store, "beta", _PLAIN, toml_extra='requires = ["alpha"]\n')
    mgr.set_installed_state("alpha", True)
    mgr.set_installed_state("beta", True, enable=False)
    mgr.set_enabled_state("beta", True)                   # dep present -> allowed
    assert mgr.is_enabled("beta")


def test_uninstalling_a_dependency_cascades_disable_to_dependents(env):
    mgr, store = _mgr(env)
    _make_plugin(store, "alpha", _PLAIN)
    _make_plugin(store, "beta", _PLAIN, toml_extra='requires = ["alpha"]\n')
    mgr.set_installed_state("alpha", True)
    mgr.set_installed_state("beta", True)
    assert mgr.is_enabled("beta")

    mgr.uninstall("alpha")                                # removing the dep ...
    assert not mgr.is_enabled("beta"), \
        "a dependent must be disabled when its dependency is uninstalled"


# --------------------------------------------------------------------------- #
#  REC-RAG-EMBED-PARITY
# --------------------------------------------------------------------------- #

def test_rag_add_embed_flag_passes_embed_fn(env, monkeypatch):
    from click.testing import CliRunner
    from localm.cli import rag as ragcli

    captured = {}

    class _FakeColl:
        def __init__(self, name):
            pass

        def create(self):
            pass

        def add_paths(self, paths, *, force=False, embed_fn=None, on_progress=None):
            captured["embed_fn"] = embed_fn
            return {"added": 1, "updated": 0, "skipped": 0, "chunks": 3, "failed": []}

    monkeypatch.setattr(ragcli, "Collection", _FakeColl, raising=False)
    # Also patch the imported symbol inside the command (from ..rag import Collection).
    import localm.rag as ragpkg
    monkeypatch.setattr(ragpkg, "Collection", _FakeColl, raising=False)
    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn", lambda url: (lambda texts: []))

    (env / "f.txt").write_text("hello world", encoding="utf-8")
    runner = CliRunner()
    # without --embed: lexical (embed_fn is None)
    r2 = runner.invoke(ragcli.rag_add, ["coll", str(env / "f.txt")])
    assert captured["embed_fn"] is None, r2.output

    # with --embed: an embed_fn is threaded to add_paths (index-time embeddings)
    r3 = runner.invoke(ragcli.rag_add, ["coll", str(env / "f.txt"), "--embed"])
    assert callable(captured["embed_fn"]), r3.output
