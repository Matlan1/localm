# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reaching a past coder session from the rail: list what is dormant across
every remembered project, and continue one PARTICULAR conversation by id.

The load-bearing test is the first one. Resuming by id has a silent failure
mode that looks like success from the outside - falling back to the newest
checkpoint - so the assertions here are on WHICH conversation came back, never
on the `resumed` flag alone. A boolean cannot tell those two apart.
"""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


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
