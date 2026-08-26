# SPDX-License-Identifier: AGPL-3.0-or-later
"""Plugin and RAG lifecycle:

  - on_first_use fires once, at first activation, and is persisted
  - enable refuses an unmet dep; uninstall cascade-disables dependents
  - `rag add --embed` embeds at index time (GUI parity)
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
#  on_first_use
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
#  plugin requires
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
#  rag embed parity
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

        def add_paths(self, paths, *, force=False, embed_fn=None, on_progress=None,
                     model_name=None):
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


# --------------------------------------------------------------------------- #
#  `rag repair` must not SILENTLY strip embeddings
# --------------------------------------------------------------------------- #

def _fake_repair_collection(monkeypatch, ragcli, *, has_vectors, captured):
    class _FakeColl:
        def __init__(self, name):
            self.name = name

        @property
        def corrupt(self):
            return False

        def documents(self):
            return ["/x/a.txt"]

        def stats(self):
            return {"has_vectors": has_vectors}

        def add_paths(self, paths, *, force=False, embed_fn=None, on_progress=None,
                     model_name=None):
            captured["embed_fn"] = embed_fn
            captured["ran"] = True
            return {"added": 0, "updated": len(paths), "skipped": 0,
                    "chunks": len(paths), "failed": []}

    import localm.rag as ragpkg
    monkeypatch.setattr(ragcli, "Collection", _FakeColl, raising=False)
    monkeypatch.setattr(ragpkg, "Collection", _FakeColl, raising=False)


def test_repair_without_embed_on_hybrid_collection_asks_before_dropping_vectors(
        env, monkeypatch):
    from click.testing import CliRunner
    from localm.cli import rag as ragcli
    captured = {}
    _fake_repair_collection(monkeypatch, ragcli, has_vectors=True, captured=captured)
    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn", lambda url: (lambda texts: []))

    runner = CliRunner()
    # Declining the confirmation must NOT run add_paths at all.
    r = runner.invoke(ragcli.rag_repair, ["coll"], input="n\n")
    assert "has_vectors" not in captured, r.output   # never got as far as add_paths
    assert not captured.get("ran"), r.output
    assert r.exit_code != 0

    # Confirming proceeds, but WITHOUT an embedder (embed_fn is None) - the
    # user was warned and chose to drop semantic search anyway.
    r2 = runner.invoke(ragcli.rag_repair, ["coll"], input="y\n")
    assert captured.get("ran"), r2.output
    assert captured["embed_fn"] is None, r2.output


def test_repair_yes_flag_skips_confirmation(env, monkeypatch):
    from click.testing import CliRunner
    from localm.cli import rag as ragcli
    captured = {}
    _fake_repair_collection(monkeypatch, ragcli, has_vectors=True, captured=captured)

    runner = CliRunner()
    r = runner.invoke(ragcli.rag_repair, ["coll", "--yes"])
    assert r.exit_code == 0, r.output
    assert captured.get("ran"), r.output
    assert captured["embed_fn"] is None


def test_repair_with_embed_flag_needs_no_confirmation_and_keeps_vectors(env, monkeypatch):
    from click.testing import CliRunner
    from localm.cli import rag as ragcli
    captured = {}
    _fake_repair_collection(monkeypatch, ragcli, has_vectors=True, captured=captured)
    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn", lambda url: (lambda texts: []))

    runner = CliRunner()
    r = runner.invoke(ragcli.rag_repair, ["coll", "--embed"])
    assert r.exit_code == 0, r.output
    assert captured.get("ran"), r.output
    assert callable(captured["embed_fn"]), r.output


def test_repair_on_bm25_only_collection_needs_no_confirmation(env, monkeypatch):
    # Nothing to lose (no existing vectors) - repair proceeds without asking.
    from click.testing import CliRunner
    from localm.cli import rag as ragcli
    captured = {}
    _fake_repair_collection(monkeypatch, ragcli, has_vectors=False, captured=captured)

    runner = CliRunner()
    r = runner.invoke(ragcli.rag_repair, ["coll"])
    assert r.exit_code == 0, r.output
    assert captured.get("ran"), r.output
