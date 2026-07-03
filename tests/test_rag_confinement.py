# SPDX-License-Identifier: AGPL-3.0-or-later
"""C2 confinement: the RAG indexing API must not be trickable into reading and
serving back system files (C:/Windows/win.ini, /etc/passwd), the localm keystore,
or credential folders.

The confinement is MODE-based (whitelist / blacklist) with an always-on HARD FLOOR
(the localm data dir + credential folders, refused in every mode). The CLI stays
unconfined except the hard floor - a local user can already read their own files -
so the mode confinement engages only when a policy is passed, which the HTTP route
always does and the CLI never does. A whitelist MISS is offered back to the owner
as 'add and continue' (409), not a dead-end error.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag.store import (Collection, ConfinementError, confine_index_path,
                              indexing_policy)


def _wl(*allowed):
    """A whitelist policy allowing (home + cwd, always) plus *allowed*."""
    return {"mode": "whitelist", "allowed": [Path(a) for a in allowed], "denied": []}


def _bl(*denied):
    """A blacklist policy denying *denied* (everything else allowed)."""
    return {"mode": "blacklist", "allowed": [], "denied": [Path(d) for d in denied]}


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
#  Whitelist mode                                                             #
# --------------------------------------------------------------------------- #

class TestWhitelist:
    def test_in_home_ok(self, home_env):
        home, _ = home_env
        f = home / "docs" / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, _wl()) == f.resolve()   # home always allowed

    def test_added_root_ok(self, home_env, tmp_path):
        extra = tmp_path / "shared"
        extra.mkdir()
        f = extra / "a.txt"
        f.write_text("hi", encoding="utf-8")
        assert confine_index_path(f, _wl(extra)) == f.resolve()

    def test_outside_rejected_with_reason(self, home_env, tmp_path):
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(outside, _wl())
        assert ei.value.reason == "outside_allowed"
        assert "outside" in str(ei.value).lower()

    def test_system_file_rejected(self, home_env):
        target = "C:/Windows/win.ini" if os.name == "nt" else "/etc/passwd"
        with pytest.raises(ConfinementError):
            confine_index_path(target, _wl())

    def test_ordinary_dotdir_under_home_indexable(self, home_env):
        # Regression: a non-credential dotted folder (.github) must not be blocked.
        home, _ = home_env
        wf = home / "repo" / ".github" / "ci.yml"
        wf.parent.mkdir(parents=True)
        wf.write_text("on: push", encoding="utf-8")
        assert confine_index_path(wf, _wl()) == wf.resolve()


# --------------------------------------------------------------------------- #
#  Blacklist mode                                                             #
# --------------------------------------------------------------------------- #

class TestBlacklist:
    def test_outside_home_allowed(self, home_env, tmp_path):
        # blacklist: anywhere not denied is fine, even far outside home.
        outside = tmp_path / "elsewhere" / "a.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("x", encoding="utf-8")
        assert confine_index_path(outside, _bl()) == outside.resolve()

    def test_denied_root_rejected(self, home_env, tmp_path):
        denied = tmp_path / "secret"
        deep = denied / "deep"
        deep.mkdir(parents=True)
        f = deep / "a.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(f, _bl(denied))
        assert ei.value.reason == "denied"

    def test_hard_floor_still_applies(self, home_env):
        # The data dir + credential folders are refused even in blacklist mode.
        home, localm = home_env
        (localm / "registry.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(localm / "registry.json", _bl())
        assert ei.value.reason == "data_dir"
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id").write_text("k", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei2:
            confine_index_path(ssh / "id", _bl())
        assert ei2.value.reason == "credential"


# --------------------------------------------------------------------------- #
#  Hard floor (both modes + the unconfined CLI)                               #
# --------------------------------------------------------------------------- #

class TestHardFloor:
    def test_data_dir_rejected_even_in_home(self, home_env):
        home, localm = home_env
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(keyfile, _wl())
        assert ei.value.reason == "data_dir"

    def test_credential_dir_rejected(self, home_env):
        home, _ = home_env
        ssh = home / ".ssh"
        ssh.mkdir()
        key = ssh / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ConfinementError) as ei:
            confine_index_path(key, _wl())
        assert ei.value.reason == "credential"

    def test_nested_credential_under_home_rejected(self, home_env):
        home, _ = home_env
        nested = home / "proj" / ".ssh"
        nested.mkdir(parents=True)
        key = nested / "id_rsa"
        key.write_text("PRIVATE KEY", encoding="utf-8")
        with pytest.raises(ConfinementError, match="credential"):
            confine_index_path(key, _wl())

    def test_credential_name_case_insensitive(self, home_env):
        home, _ = home_env
        d = home / "docs" / ".SSH"
        d.mkdir(parents=True)
        f = d / "key"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ConfinementError, match="credential"):
            confine_index_path(f, _wl())

    def test_cli_unconfined_but_hard_floor_holds(self, home_env, tmp_path):
        # policy=None: an ordinary path anywhere is allowed (the CLI contract),
        # but the hard floor still denies the data dir / credential folders.
        home, localm = home_env
        ok = tmp_path / "anywhere.txt"
        ok.write_text("x", encoding="utf-8")
        assert confine_index_path(ok, None) == ok.resolve()
        with pytest.raises(ConfinementError):
            confine_index_path(localm / "registry.json", None)


# --------------------------------------------------------------------------- #
#  indexing_policy()                                                          #
# --------------------------------------------------------------------------- #

class TestIndexingPolicy:
    def test_default_mode_is_whitelist(self, home_env):
        assert indexing_policy()["mode"] == "whitelist"

    def test_reads_mode_and_both_lists(self, home_env, tmp_path, monkeypatch):
        a = tmp_path / "a"
        d = tmp_path / "d"
        a.mkdir()
        d.mkdir()
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config", lambda: {
            "rag_indexing_mode": "blacklist",
            "rag_allowed_roots": [str(a)],
            "rag_denied_roots": [str(d)]})
        pol = indexing_policy()
        assert pol["mode"] == "blacklist"
        assert a.resolve() in pol["allowed"]
        assert d.resolve() in pol["denied"]

    def test_bad_mode_falls_back_to_whitelist(self, home_env, monkeypatch):
        import localm.config as cfg
        monkeypatch.setattr(cfg, "load_config",
                            lambda: {"rag_indexing_mode": "bogus"})
        assert indexing_policy()["mode"] == "whitelist"


# --------------------------------------------------------------------------- #
#  add_paths(policy=...) (unit)                                               #
# --------------------------------------------------------------------------- #

class TestAddPathsConfinement:
    def test_in_home_indexes(self, home_env, tmp_path):
        home, _ = home_env
        (home / "docs" / "a.txt").write_text(
            "rocm gfx1030 runtime dll", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        res = c.add_paths([home / "docs"], policy=_wl())
        assert res["added"] == 1

    def test_out_of_root_raises(self, home_env, tmp_path):
        home, _ = home_env
        outside = tmp_path / "out"
        outside.mkdir()
        (outside / "x.txt").write_text("secret", encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        with pytest.raises(ValueError):
            c.add_paths([outside], policy=_wl())

    def test_nested_secret_is_skipped_not_indexed(self, home_env, tmp_path):
        home, localm = home_env
        (home / "docs" / "good.txt").write_text(
            "ordinary indexable document", encoding="utf-8")
        ssh = home / ".ssh"
        ssh.mkdir()
        (ssh / "id_rsa.txt").write_text("PRIVATE KEY MATERIAL", encoding="utf-8")
        (localm / "registry.json").write_text(
            '{"secret_model": true}', encoding="utf-8")
        c = Collection("kb", base=tmp_path / "rag").create()
        c.add_paths([home], policy=_wl())
        sources = " ".join(d["path"] for d in c.docs())
        assert "good.txt" in sources
        assert "id_rsa" not in sources
        assert "registry.json" not in sources


# --------------------------------------------------------------------------- #
#  HTTP route /api/rag/collections/{name}/add                                 #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_route_app(tmp_path, monkeypatch):
    """Minimal GUI app with the builtin rag plugin enabled; Path.home == tmp_path
    so docs placed under tmp_path are inside the whitelist. Open mode -> the
    caller is the owner."""
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


class TestRagAddRoute:
    def test_in_home_accepted(self, rag_route_app):
        app, home = rag_route_app
        docs = home / "kdocs"
        docs.mkdir()
        (docs / "a.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(docs)], "embed": False})
            assert r.status_code == 200

    def test_out_of_whitelist_offers_consent_to_owner(self, rag_route_app,
                                                      tmp_path_factory):
        # NEW: an out-of-whitelist path is OFFERED to the owner (409 needs_consent),
        # not hard-blocked, so they can add it and continue.
        app, _ = rag_route_app
        outside = tmp_path_factory.mktemp("outside_home")
        secret = outside / "secret.txt"
        secret.write_text("TOPSECRET", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(secret)], "embed": False})
            assert r.status_code == 409, r.text
            body = r.json()
            assert body["needs_consent"] is True
            assert any("secret.txt" in a for a in body["addable"])

    def test_data_dir_is_hard_blocked_not_offered(self, rag_route_app):
        app, home = rag_route_app
        localm = home / ".localm"
        localm.mkdir(exist_ok=True)
        keyfile = localm / "keys.json"
        keyfile.write_text("{}", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(keyfile)], "embed": False})
            assert r.status_code == 400   # hard floor: refused, never offered

    def test_blacklist_allows_outside_but_denies_listed(self, rag_route_app,
                                                        tmp_path_factory):
        app, _ = rag_route_app
        denied = tmp_path_factory.mktemp("denied_zone")
        (denied / "s.txt").write_text("x", encoding="utf-8")
        free = tmp_path_factory.mktemp("free_zone")
        (free / "ok.md").write_text("rocm gfx1030 dll", encoding="utf-8")
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg["rag_indexing_mode"] = "blacklist"
        cfg["rag_denied_roots"] = [str(denied)]
        save_config(cfg)
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            # outside home, not denied -> indexed (blacklist allows).
            r1 = client.post("/api/rag/collections/kb/add",
                             json={"paths": [str(free)], "embed": False})
            assert r1.status_code == 200, r1.text
            # a denied folder -> hard 400 (an explicit deny is not offered).
            r2 = client.post("/api/rag/collections/kb/add",
                             json={"paths": [str(denied / "s.txt")], "embed": False})
            assert r2.status_code == 400, r2.text
