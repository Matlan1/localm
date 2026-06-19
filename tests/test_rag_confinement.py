"""C2: confine API-driven RAG indexing to safe roots so a request (a loopback
browser page or a remote client) cannot index + serve back system files
(C:/Windows/win.ini, /etc/passwd), the localm keystore, or credential folders.

The CLI stays unconfined (a local user can already read their own files): the
confinement only engages when ``allowed_roots`` is passed, which the HTTP route
always does and the CLI never does.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag.store import Collection, confine_index_path, indexing_roots


@pytest.fixture
def home_env(tmp_path, monkeypatch):
    """A controlled user home (Path.home) with the localm data dir under it."""
    home = tmp_path / "userhome"
    (home / "docs").mkdir(parents=True)
    localm = home / ".localm"
    localm.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")
    return home, localm


# --------------------------------------------------------------------------- #
#  confine_index_path / indexing_roots (unit)                                  #
# --------------------------------------------------------------------------- #

class TestConfineIndexPath:
    def test_in_home_ok(self, home_env):
        home, _ = home_env
        f = home / "docs" / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, [home]) == f.resolve()

    def test_outside_roots_rejected(self, home_env, tmp_path):
        home, _ = home_env
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(ValueError, match="outside"):
            confine_index_path(outside, [home])

    def test_system_file_rejected(self, home_env):
        home, _ = home_env
        target = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
        with pytest.raises(ValueError):
            confine_index_path(target, [home])

    def test_localm_data_dir_rejected_even_inside_home(self, home_env):
        # NEGATIVE-ish: the data dir is *under* the allowed home root, yet must
        # still be refused - it holds the API key + registry.
        home, localm = home_env
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="data director"):
            confine_index_path(keyfile, [home])

    def test_credential_dir_rejected(self, home_env):
        home, _ = home_env
        ssh = home / ".ssh"
        ssh.mkdir()
        key = ssh / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ValueError, match="credential"):
            confine_index_path(key, [home])

    def test_secrets_denied_even_without_roots(self, home_env, tmp_path):
        # The function denies secrets regardless of roots; with roots=None an
        # ordinary user path is allowed (the unconfined CLI contract).
        home, localm = home_env
        ok = tmp_path / "anywhere.txt"
        ok.write_text("x", encoding="utf-8")
        assert confine_index_path(ok, None) == ok.resolve()
        with pytest.raises(ValueError):
            confine_index_path(localm / "registry.json", None)


class TestIndexingRoots:
    def test_includes_home_and_cwd(self, home_env):
        home, _ = home_env
        roots = indexing_roots()
        assert home.resolve() in roots
        assert Path.cwd().resolve() in roots

    def test_config_extra_roots(self, home_env, tmp_path, monkeypatch):
        extra = tmp_path / "shared"
        extra.mkdir()
        import localm.config as cfg
        monkeypatch.setattr(
            cfg, "load_config", lambda: {"rag_indexing_roots": [str(extra)]})
        assert extra.resolve() in indexing_roots()


# --------------------------------------------------------------------------- #
#  add_paths(allowed_roots=...) (unit)                                         #
# --------------------------------------------------------------------------- #

class TestAddPathsConfinement:
    def test_in_home_indexes(self, home_env, tmp_path):
        home, _ = home_env
        (home / "docs" / "a.txt").write_text(
            "rocm gfx1030 runtime dll", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([home / "docs"], allowed_roots=[home])
        assert res["added"] == 1

    def test_out_of_root_raises(self, home_env, tmp_path):
        # NEGATIVE: pre-fix add_paths had no allowed_roots and indexed anything.
        home, _ = home_env
        outside = tmp_path / "out"
        outside.mkdir()
        (outside / "x.txt").write_text("secret", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        with pytest.raises(ValueError):
            c.add_paths([outside], allowed_roots=[home])

    def test_nested_secret_is_skipped_not_indexed(self, home_env, tmp_path):
        # Indexing the whole home dir must drop nested credential / data-dir
        # files (the symlink-escape defense, exercised via real secret dirs).
        home, localm = home_env
        (home / "docs" / "good.txt").write_text(
            "ordinary indexable document", encoding="utf-8")
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa.txt").write_text("PRIVATE KEY MATERIAL", encoding="utf-8")
        (localm / "registry.json").write_text(
            '{"secret_model": true}', encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([home], allowed_roots=[home])
        sources = " ".join(d["path"] for d in c.docs())
        assert "good.txt" in sources
        assert "id_rsa" not in sources
        assert "registry.json" not in sources


# --------------------------------------------------------------------------- #
#  HTTP route /api/rag/collections/{name}/add (C2 route enforcement)           #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_route_app(tmp_path, monkeypatch):
    """Minimal GUI app with the builtin rag plugin enabled; Path.home == tmp_path
    so docs placed under tmp_path are inside the allowed root."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui
    home = tmp_path
    localm = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


class TestRagAddRouteConfinement:
    def test_in_home_path_accepted(self, rag_route_app):
        app, home = rag_route_app
        docs = home / "kdocs"
        docs.mkdir()
        (docs / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(docs)], "embed": False})
            assert r.status_code == 200

    def test_out_of_root_path_rejected(self, rag_route_app, tmp_path_factory):
        # NEGATIVE: pre-fix this returned 200 and indexed the file for exfil.
        app, _ = rag_route_app
        outside = tmp_path_factory.mktemp("outside_home")   # sibling of home
        secret = outside / "secret.txt"
        secret.write_text("TOPSECRET", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(secret)], "embed": False})
            assert r.status_code == 400
            assert "outside" in r.json()["detail"].lower()

    def test_localm_data_dir_path_rejected(self, rag_route_app):
        app, home = rag_route_app
        localm = home / ".localm"
        localm.mkdir(exist_ok=True)
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(keyfile)], "embed": False})
            assert r.status_code == 400
