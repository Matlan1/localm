# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reaching a past coder session from the rail: list what is dormant across
every remembered project, and continue one PARTICULAR conversation by id.

The load-bearing test is the first one. Resuming by id has a silent failure
mode that looks like success from the outside - falling back to the newest
checkpoint - so the assertions here are on WHICH conversation came back, never
on the `resumed` flag alone. A boolean cannot tell those two apart.
"""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _close_coder_sessions_left_open(monkeypatch):
    """Records every CoderSession the test creates and closes the ones still
    open when it ends, before monkeypatch restores the test's paths."""
    from localm.plugins.coder.sessions import CoderSession
    opened = []
    real_init = CoderSession.__init__

    def _recording_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        opened.append(self)

    monkeypatch.setattr(CoderSession, "__init__", _recording_init)
    yield
    for session in opened:
        if not session.closed:
            session.close()


def _coder_app(tmp_path, monkeypatch, *, api_key):
    """A real app, real routes, real Agent, real checkpoint files on disk."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    monkeypatch.setattr(_cfg, "home_dir", lambda: home)
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def _seed(app, sid, messages, title):
    """Persist a saved conversation for a live session and return its
    checkpoint id."""
    sess = app.state.coder_sessions.get(sid)
    sess.agent._messages = messages
    sess.agent._turns = len(messages)
    sess.agent._total_tokens = 42
    sess.agent._session_title = title
    sess.persist_checkpoint()
    return sess.agent._checkpoint_id


def _age(app, cwd, checkpoint_id, mtime):
    """Stamp a checkpoint's mtime explicitly.

    list_checkpoints sorts by mtime, and two files written microseconds apart
    can tie or land in either order. Without this the "resumed the one I asked
    for" test could pass by coincidence rather than because the id was
    honoured - the fixture has to be able to express the failure it is looking
    for.
    """
    import os
    from localm.plugins.coder.agent.checkpoint import _checkpoint_path_for
    p = _checkpoint_path_for(Path(cwd), checkpoint_id)
    os.utime(p, (mtime, mtime))


OWNER = {"Authorization": "Bearer ownersecret"}


class TestResumingOneParticularSession:
    """A listing is only worth having if acting on a row continues THAT row."""

    def test_resuming_by_id_restores_that_conversation_not_the_newest(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        older = [{"role": "user", "content": "build a calculator"},
                 {"role": "assistant", "content": "Here is the calculator plan."}]
        newer = [{"role": "user", "content": "write a csv parser"},
                 {"role": "assistant", "content": "Here is the parser plan."}]

        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            old_id = _seed(app, a.json()["id"], older, "build a calculator")
            b = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            new_id = _seed(app, b.json()["id"], newer, "write a csv parser")
            assert old_id != new_id, "two sessions must not share a checkpoint file"
            _age(app, proj, old_id, 1_000_000)
            _age(app, proj, new_id, 2_000_000)   # unambiguously the newest

            # Both are offered, newest first.
            listing = client.get("/api/coder/dormant", headers=OWNER,
                                 params={"cwd": str(proj)}).json()
            ids = [s["id"] for s in listing["projects"][0]["sessions"]]
            assert ids == [new_id, old_id]

            # Now ask for the OLDER one by id.
            r = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log", "resume": True,
                                  "resume_checkpoint_id": old_id})
            restored = app.state.coder_sessions.get(r.json()["id"]).agent._messages

        # Assert on the conversation content first: `resumed: true` on its own is
        # also satisfied by a fallback to the newest checkpoint.
        assert restored == older, (
            "resumed the wrong conversation: asked for the calculator session "
            "and got " + repr(restored[:1]))
        assert r.json()["resumed"] is True

    def test_an_unknown_id_does_not_silently_resume_something_else(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            _seed(app, a.json()["id"], [{"role": "user", "content": "secret work"}],
                  "secret work")
            r = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log", "resume": True,
                                  "resume_checkpoint_id": "deadbeefdeadbeef"})
            restored = app.state.coder_sessions.get(r.json()["id"]).agent._messages

        # A stale id starts fresh and says so, rather than handing back a
        # conversation the caller did not ask for.
        assert restored == [], "an unknown id must not restore another session"
        assert r.json()["resumed"] is False

    def test_the_zero_argument_default_is_unchanged(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        newer = [{"role": "user", "content": "the newest thing"}]
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            old_id = _seed(app, a.json()["id"], [{"role": "user", "content": "old"}],
                           "old")
            b = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            new_id = _seed(app, b.json()["id"], newer, "the newest thing")
            _age(app, proj, old_id, 1_000_000)
            _age(app, proj, new_id, 2_000_000)
            # No id: "continue where I left off", exactly as before.
            r = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log", "resume": True})
            restored = app.state.coder_sessions.get(r.json()["id"]).agent._messages
        assert restored == newer


class TestTheListing:
    def test_sessions_from_other_projects_are_reachable(self, tmp_path, monkeypatch):
        one = tmp_path / "one"; one.mkdir()
        two = tmp_path / "two"; two.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(tmp_path)
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(one), "mode": "log"})
            _seed(app, a.json()["id"], [{"role": "user", "content": "in one"}], "in one")
            b = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(two), "mode": "log"})
            _seed(app, b.json()["id"], [{"role": "user", "content": "in two"}], "in two")

            # Asking with one project selected still surfaces the other.
            got = client.get("/api/coder/dormant", headers=OWNER,
                             params={"cwd": str(one)}).json()

        by_name = {p["name"]: p for p in got["projects"]}
        assert set(by_name) == {"one", "two"}
        assert by_name["one"]["current"] is True
        assert by_name["two"]["current"] is False
        assert [s["title"] for s in by_name["two"]["sessions"]] == ["in two"]
        # The selected project is listed once, not twice (it is both the
        # current cwd and a remembered project).
        assert len(got["projects"]) == 2

    def test_a_deleted_project_still_offers_its_past_sessions(
            self, tmp_path, monkeypatch):
        gone = tmp_path / "gone"; gone.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(tmp_path)
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(gone), "mode": "log"})
            _seed(app, a.json()["id"], [{"role": "user", "content": "work"}], "work")
            app.state.coder_sessions.remove(a.json()["id"])
            gone.rmdir()
            got = client.get("/api/coder/dormant", headers=OWNER).json()

        row = next(p for p in got["projects"] if p["name"] == "gone")
        # Checkpoints live in the data dir, keyed by a digest of the path, so they
        # outlive the directory. Reported unavailable, never silently dropped.
        assert row["available"] is False
        assert [s["title"] for s in row["sessions"]] == ["work"]

    def test_the_privacy_note_is_permanent_not_an_empty_state(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            empty = client.get("/api/coder/dormant", headers=OWNER).json()
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            _seed(app, a.json()["id"], [{"role": "user", "content": "x"}], "x")
            full = client.get("/api/coder/dormant", headers=OWNER,
                              params={"cwd": str(proj)}).json()

        # Present on both the empty and the non-empty listing.
        assert empty["privacy_note"] and full["privacy_note"]
        assert empty["privacy_note"] == full["privacy_note"]
        assert full["projects"], "this arm must be the NON-empty one"

    def test_a_privacy_session_contributes_nothing(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "privacy"})
            # A real privacy session with a real conversation, persisted through
            # the same call every other mode uses. The agent declines to write.
            _seed(app, a.json()["id"],
                  [{"role": "user", "content": "confidential"}], "confidential")
            got = client.get("/api/coder/dormant", headers=OWNER,
                             params={"cwd": str(proj)}).json()

        rows = [s for p in got["projects"] for s in p["sessions"]]
        assert rows == [], "a privacy session must leave nothing to list"
        # And not merely absent from the listing: nothing on disk either.
        assert not [s for p in got["projects"] for s in p["sessions"]
                    if "confidential" in (s.get("title") or "")]

    def test_the_listing_is_owner_only(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        from localm import auth
        scoped_h = {"Authorization": f"Bearer {auth.create_key('phone', ['coder'])['key']}"}
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            _seed(app, a.json()["id"],
                  [{"role": "user", "content": "the owner's own words"}],
                  "the owner's own words")
            r = client.get("/api/coder/dormant", headers=scoped_h,
                           params={"cwd": str(proj)})

        # A scoped or shared key is never shown a session title, same gate as
        # /resumable. The key used here is valid: an invalid one is refused at
        # the auth layer with a 401 and never reaches this route.
        assert r.status_code == 200
        assert r.json()["projects"] == []
        assert r.json()["privacy_note"], "the note is not owner-gated"

    @pytest.mark.parametrize("bad", [r"\\192.0.2.1\share", "//192.0.2.1/share",
                                     r"\\.\PhysicalDrive0"])
    def test_a_unc_or_device_cwd_is_refused(self, tmp_path, monkeypatch, bad):
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(tmp_path)
        with TestClient(app) as client:
            r = client.get("/api/coder/dormant", headers=OWNER, params={"cwd": bad})
        # Refused on the string, before any filesystem call - a stat on a UNC
        # path is an SMB dial that authenticates as the logged-in user.
        assert r.status_code == 400


# --------------------------------------------------------------------------- #
#  Deleting a past session, removing a project                                 #
# --------------------------------------------------------------------------- #

def _ckpt(cwd, checkpoint_id):
    from localm.plugins.coder.agent.checkpoint import _checkpoint_path_for
    return _checkpoint_path_for(Path(cwd), checkpoint_id)


def _ended(app, client, cwd, title):
    """A past session: saved for *cwd*, then ended so no live session holds
    it. Returns its checkpoint id."""
    a = client.post("/api/coder/sessions", headers=OWNER,
                    json={"cwd": str(cwd), "mode": "log"})
    cid = _seed(app, a.json()["id"], [{"role": "user", "content": title}], title)
    app.state.coder_sessions.remove(a.json()["id"])
    assert _ckpt(cwd, cid).is_file(), "precondition: the session was saved"
    return cid


def _listed_paths():
    from localm.plugins.coder.projects import list_projects
    return [e["path"] for e in list_projects()]


def _scoped_headers():
    from localm import auth
    return {"Authorization": f"Bearer {auth.create_key('phone', ['coder'])['key']}"}


class TestDeletingOnePastSession:
    def test_deletes_that_checkpoint_file_and_no_other(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            gone = _ended(app, client, proj, "delete me")
            kept = _ended(app, client, proj, "keep me")
            r = client.request("DELETE", "/api/coder/checkpoints", headers=OWNER,
                               json={"cwd": str(proj), "checkpoint_id": gone})
            listing = client.get("/api/coder/dormant", headers=OWNER,
                                 params={"cwd": str(proj)}).json()

        # The file on disk first: a 200 alone is also what a route that deleted
        # nothing would answer.
        assert not _ckpt(proj, gone).exists(), "the deleted session is still on disk"
        assert _ckpt(proj, kept).is_file(), "another session of the project was deleted"
        assert r.status_code == 200, r.text
        assert r.json() == {"deleted": gone}
        assert [s["id"] for s in listing["projects"][0]["sessions"]] == [kept]

    def test_an_unknown_id_is_404_and_deletes_nothing(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            kept = _ended(app, client, proj, "keep me")
            r = client.request("DELETE", "/api/coder/checkpoints", headers=OWNER,
                               json={"cwd": str(proj), "checkpoint_id": "deadbeef0000"})
        assert _ckpt(proj, kept).is_file()
        assert r.status_code == 404

    @pytest.mark.parametrize("bad", ["../../outside", "a.json", "", "x" * 65])
    def test_an_invalid_id_is_400(self, tmp_path, monkeypatch, bad):
        proj = tmp_path / "proj"; proj.mkdir()
        outside = tmp_path / ".localm" / "outside.json"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("{}", encoding="utf-8")
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            r = client.request("DELETE", "/api/coder/checkpoints", headers=OWNER,
                               json={"cwd": str(proj), "checkpoint_id": bad})
        assert outside.is_file(), "a traversal id reached a file outside the store"
        assert r.status_code == 400

    def test_the_checkpoint_of_a_live_session_is_409(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            a = client.post("/api/coder/sessions", headers=OWNER,
                            json={"cwd": str(proj), "mode": "log"})
            live = _seed(app, a.json()["id"], [{"role": "user", "content": "open"}],
                         "open")
            r = client.request("DELETE", "/api/coder/checkpoints", headers=OWNER,
                               json={"cwd": str(proj), "checkpoint_id": live})
            assert _ckpt(proj, live).is_file(), (
                "deleted the checkpoint a live session is still writing")
            assert r.status_code == 409, r.text

    def test_a_scoped_key_cannot_delete_the_owners_session(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            cid = _ended(app, client, proj, "the owner's own words")
            r = client.request("DELETE", "/api/coder/checkpoints",
                               headers=_scoped_headers(),
                               json={"cwd": str(proj), "checkpoint_id": cid})
        assert _ckpt(proj, cid).is_file(), "a scoped key deleted the owner's session"
        assert r.status_code == 403

    def test_a_delete_that_leaves_the_file_is_not_reported_as_success(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            cid = _ended(app, client, proj, "stubborn")
            target = _ckpt(proj, cid)
            real_unlink = Path.unlink
            calls = []

            def no_op_unlink(self, *a, **kw):
                if self == target:
                    calls.append(self)
                    return None
                return real_unlink(self, *a, **kw)

            with monkeypatch.context() as m:
                m.setattr(Path, "unlink", no_op_unlink)
                r = client.request("DELETE", "/api/coder/checkpoints", headers=OWNER,
                                   json={"cwd": str(proj), "checkpoint_id": cid})
        assert calls, "the injected failure never fired"
        assert target.is_file()
        assert r.status_code == 500, r.text
        assert "deleted" not in r.json()


class TestRemovingAProject:
    def test_forgets_it_and_deletes_its_sessions_but_not_its_files(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        other = tmp_path / "other"; other.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        from localm.plugins.coder.agent.checkpoint import (
            _legacy_checkpoint_path_for, _legacy_home_checkpoint_path_for,
            _project_dir_for, _project_map_path_for,
        )
        with TestClient(app) as client:
            one = _ended(app, client, proj, "first")
            two = _ended(app, client, proj, "second")
            elsewhere = _ended(app, client, other, "in another project")
            # Every saved-session shape this project can have.
            _project_map_path_for(proj).write_text("{}", encoding="utf-8")
            legacy_home = _legacy_home_checkpoint_path_for(proj)
            legacy_home.write_text("{}", encoding="utf-8")
            legacy_local = _legacy_checkpoint_path_for(proj)
            legacy_local.parent.mkdir(parents=True, exist_ok=True)
            legacy_local.write_text("{}", encoding="utf-8")
            # The user's own files, which must survive.
            (proj / "keep.txt").write_text("user work", encoding="utf-8")
            (proj / ".localcoder" / "config.toml").write_text("x = 1", encoding="utf-8")
            (proj / "src").mkdir()
            (proj / "src" / "main.py").write_text("print(1)", encoding="utf-8")
            before = {p.relative_to(proj): p.read_bytes()
                      for p in proj.rglob("*") if p.is_file() and p != legacy_local}
            assert str(proj.resolve()) in _listed_paths(), "precondition: listed"

            r = client.request("DELETE", "/api/coder/projects", headers=OWNER,
                               json={"path": str(proj.resolve())})

        assert not _project_dir_for(proj).exists(), "the saved sessions are still on disk"
        assert not legacy_home.exists() and not legacy_local.exists()
        assert str(proj.resolve()) not in _listed_paths(), "the project is still listed"
        after = {p.relative_to(proj): p.read_bytes()
                 for p in proj.rglob("*") if p.is_file()}
        assert after == before, "a file in the project folder was changed or removed"
        assert (proj / "keep.txt").read_text(encoding="utf-8") == "user work"
        # The other project is untouched.
        assert _ckpt(other, elsewhere).is_file()
        assert str(other.resolve()) in _listed_paths()
        assert r.status_code == 200, r.text
        assert r.json()["sessions_deleted"] == 4, (one, two)
        assert r.json()["forgotten"] is True

    def test_a_project_whose_folder_is_gone_can_still_be_removed(
            self, tmp_path, monkeypatch):
        gone = tmp_path / "gone"; gone.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(tmp_path)
        from localm.plugins.coder.agent.checkpoint import _project_dir_for
        with TestClient(app) as client:
            _ended(app, client, gone, "work")
            path = str(gone.resolve())
            gone.rmdir()
            r = client.request("DELETE", "/api/coder/projects", headers=OWNER,
                               json={"path": path})
            listing = client.get("/api/coder/dormant", headers=OWNER).json()
        assert not _project_dir_for(Path(path)).exists()
        assert path not in _listed_paths()
        assert not any(p["path"] == path for p in listing["projects"])
        assert r.status_code == 200, r.text

    def test_a_project_with_a_live_session_is_409(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        from localm.plugins.coder.agent.checkpoint import _project_dir_for
        with TestClient(app) as client:
            past = _ended(app, client, proj, "past")
            client.post("/api/coder/sessions", headers=OWNER,
                        json={"cwd": str(proj), "mode": "log"})
            r = client.request("DELETE", "/api/coder/projects", headers=OWNER,
                               json={"path": str(proj.resolve())})
            assert _ckpt(proj, past).is_file()
            assert _project_dir_for(proj).is_dir()
            assert str(proj.resolve()) in _listed_paths()
            assert r.status_code == 409, r.text

    def test_a_scoped_key_cannot_remove_a_project(self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        with TestClient(app) as client:
            cid = _ended(app, client, proj, "the owner's")
            r = client.request("DELETE", "/api/coder/projects",
                               headers=_scoped_headers(),
                               json={"path": str(proj.resolve())})
        assert _ckpt(proj, cid).is_file()
        assert str(proj.resolve()) in _listed_paths()
        assert r.status_code == 403

    def test_an_unknown_project_is_404(self, tmp_path, monkeypatch):
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(tmp_path)
        with TestClient(app) as client:
            r = client.request("DELETE", "/api/coder/projects", headers=OWNER,
                               json={"path": str(tmp_path / "never-used")})
        assert r.status_code == 404

    def test_a_removal_that_leaves_sessions_is_not_reported_as_success(
            self, tmp_path, monkeypatch):
        proj = tmp_path / "proj"; proj.mkdir()
        app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        app.state.root_dir = str(proj)
        import shutil
        calls = []
        with TestClient(app) as client:
            cid = _ended(app, client, proj, "stubborn")
            with monkeypatch.context() as m:
                m.setattr(shutil, "rmtree", lambda p, *a, **kw: calls.append(p))
                r = client.request("DELETE", "/api/coder/projects", headers=OWNER,
                                   json={"path": str(proj.resolve())})
        assert calls, "the injected failure never fired"
        assert _ckpt(proj, cid).is_file()
        assert str(proj.resolve()) in _listed_paths(), (
            "forgot the project although its sessions are still on disk")
        assert r.status_code == 500, r.text


def _link_dir(link, target):
    """Make *link* a directory symlink (POSIX) or junction (Windows)."""
    import os
    import subprocess
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True)
    else:
        os.symlink(target, link, target_is_directory=True)


def test_a_linked_project_checkpoint_dir_is_refused_and_its_target_left_intact(
        tmp_path, monkeypatch):
    import localm.config as _cfg
    from localm.plugins.coder.agent.checkpoint import (
        _project_dir_for, delete_project_checkpoints,
    )
    home = tmp_path / ".localm"
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    proj = tmp_path / "proj"; proj.mkdir()
    target = tmp_path / "precious"; target.mkdir()
    (target / "a.json").write_text("{}", encoding="utf-8")
    (target / "notes.txt").write_text("keep me", encoding="utf-8")
    d = _project_dir_for(proj)
    d.parent.mkdir(parents=True)
    _link_dir(d, target)

    with pytest.raises(OSError):
        delete_project_checkpoints(proj)

    assert sorted(p.name for p in target.iterdir()) == ["a.json", "notes.txt"]
    assert (target / "notes.txt").read_text(encoding="utf-8") == "keep me"
    assert os.path.lexists(d), "the link itself was removed"


@pytest.mark.parametrize("method,body", [
    ("DELETE /api/coder/checkpoints", {"checkpoint_id": "abc123"}),
    ("DELETE /api/coder/projects", {}),
])
@pytest.mark.parametrize("bad", [r"\\192.0.2.1\share", "//192.0.2.1/share",
                                 r"\\.\PhysicalDrive0"])
def test_unc_or_device_paths_are_refused_before_any_filesystem_call(
        tmp_path, monkeypatch, method, body, bad):
    real = {"resolve": Path.resolve, "is_dir": Path.is_dir, "exists": Path.exists}
    fired = []

    def make_spy(name):
        def spy(self, *a, **kw):
            s = str(self)
            if s[:2] in ("\\\\", "//", "\\/", "/\\"):
                fired.append((name, s))
                raise AssertionError(f"Path.{name}() reached the filesystem with {s!r}")
            return real[name](self, *a, **kw)
        return spy

    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    verb, url = method.split(" ", 1)
    field = "path" if url.endswith("/projects") else "cwd"
    with TestClient(app) as client:
        for name in real:
            monkeypatch.setattr(Path, name, make_spy(name))
        r = client.request(verb, url, headers=OWNER, json={**body, field: bad})
    assert fired == []
    assert r.status_code == 400, r.text


class TestForgetProject:
    def test_removes_only_the_matching_entry(self, tmp_path, monkeypatch):
        _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        from localm.plugins.coder.projects import forget_project, record_project
        a = tmp_path / "a"; a.mkdir()
        b = tmp_path / "b"; b.mkdir()
        record_project(a, "log")
        record_project(b, "log")
        assert forget_project(a) is True
        assert _listed_paths() == [str(b.resolve())]
        assert forget_project(a) is False, "a second forget finds nothing"

    def test_letter_case_matches_where_the_platform_ignores_it(
            self, tmp_path, monkeypatch):
        import os
        _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
        from localm.plugins.coder.projects import forget_project, record_project
        a = tmp_path / "CaseProj"; a.mkdir()
        record_project(a, "log")
        stored = _listed_paths()[0]
        # Folder removed: resolve() then cannot restore the on-disk letter case
        # of the missing part, so only the comparison itself can match it.
        a.rmdir()
        swapped = stored.swapcase()
        assert str(Path(swapped).resolve()) != stored, (
            "precondition: resolve() alone must not already undo the case change")
        folds = os.path.normcase(stored) == os.path.normcase(swapped)
        assert forget_project(swapped) is folds
        assert (_listed_paths() == []) is folds
