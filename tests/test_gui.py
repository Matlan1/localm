"""Tests for the GUI plugin: coder sessions, agent event hooks, web endpoints."""

import json
import queue
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.gui.sessions import CoderSession, SessionManager
from localm.plugins.gui.web import attach_gui


# ------------------------------------------------------------------ #
#  Fake LLM backends                                                  #
# ------------------------------------------------------------------ #

class ScriptedBackend:
    """Yields one scripted response per chat call, then repeats the last."""

    model_id = "fake-model"
    last_usage = {"total_tokens": 7}
    native_tools = False
    supports_grammar = False

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def _next(self):
        i = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return self.responses[i]

    def chat(self, messages, **kw):
        return self._next()

    def chat_stream(self, messages, **kw):
        text = self._next()
        # Stream in two pieces to exercise token assembly
        mid = max(1, len(text) // 2)
        yield text[:mid]
        yield text[mid:]


def _write_call(path, content):
    return (
        "Writing the file now.\n<tool_call>\n"
        + json.dumps({"name": "write_file", "args": {"path": path, "content": content}})
        + "\n</tool_call>"
    )


def _drain(session, *, until_types, timeout=10.0):
    """Collect events from the session queue until one of until_types appears."""
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            ev = session.events.get(timeout=0.2)
        except queue.Empty:
            continue
        events.append(ev)
        if ev["type"] in until_types:
            return events
    raise TimeoutError(f"No {until_types} event within {timeout}s: {events}")


# ------------------------------------------------------------------ #
#  CoderSession                                                       #
# ------------------------------------------------------------------ #

class TestCoderSession:
    def test_plain_answer_emits_tokens_and_final(self, tmp_path):
        session = CoderSession(
            tmp_path, ScriptedBackend(["All done, nothing to do."]),
            auto_approve=True,
        )
        assert session.send_message("say hi")
        events = _drain(session, until_types={"final"})
        types = [e["type"] for e in events]
        assert "token" in types
        assert types[-1] == "final"
        final = events[-1]
        assert "All done" in final["text"]
        assert final["ok"] is True

    def test_busy_rejects_second_message(self, tmp_path):
        class SlowBackend(ScriptedBackend):
            def chat_stream(self, messages, **kw):
                time.sleep(0.5)
                yield "done"

        session = CoderSession(tmp_path, SlowBackend(["done"]), auto_approve=True)
        assert session.send_message("one")
        assert session.send_message("two") is False
        _drain(session, until_types={"final"})

    def test_tool_call_events_flow(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("hello.txt", "hi"),
            "File written, task complete.",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=True)
        session.send_message("write hello.txt")
        events = _drain(session, until_types={"final"})
        types = [e["type"] for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        call = next(e for e in events if e["type"] == "tool_call")
        assert call["tool"] == "write_file"
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is True
        assert (tmp_path / "hello.txt").read_text() == "hi"

    def test_confirm_reject_blocks_write(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("guarded.txt", "nope"),
            "Understood, not writing.",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write guarded.txt")

        events = _drain(session, until_types={"confirm_request"})
        req = events[-1]
        assert req["tool"] == "write_file"
        assert req["diff"]                      # diff preview included
        assert "+nope" in req["diff"]

        assert session.answer_confirm(req["confirm_id"], approved=False)
        events = _drain(session, until_types={"final"})
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is False
        assert not (tmp_path / "guarded.txt").exists()

    def test_confirm_approve_writes(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("approved.txt", "yes"),
            "Done.",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write approved.txt")
        events = _drain(session, until_types={"confirm_request"})
        session.answer_confirm(events[-1]["confirm_id"], approved=True)
        _drain(session, until_types={"final"})
        assert (tmp_path / "approved.txt").read_text() == "yes"

    def test_wrong_confirm_id_rejected(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("x.txt", "x"),
            "ok",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write")
        events = _drain(session, until_types={"confirm_request"})
        assert session.answer_confirm("bogus", approved=True) is False
        session.answer_confirm(events[-1]["confirm_id"], approved=False)
        _drain(session, until_types={"final"})

    def test_stop_unblocks_pending_confirm(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("y.txt", "y"),
            "stopped",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write")
        _drain(session, until_types={"confirm_request"})
        session.stop()
        events = _drain(session, until_types={"final"})
        assert not (tmp_path / "y.txt").exists()
        assert any(e["type"] == "final" for e in events)

    def test_close_emits_closed(self, tmp_path):
        session = CoderSession(tmp_path, ScriptedBackend(["hi"]), auto_approve=True)
        session.close()
        events = _drain(session, until_types={"closed"})
        assert events[-1]["type"] == "closed"
        assert session.send_message("too late") is False


class TestConfirmResolution:
    """Every answered confirmation must leave a confirm_resolved event in the
    stream AND the replay buffer — otherwise a reloaded page replays the
    confirm_request with live approve/reject buttons."""

    def _request(self, tmp_path, fname):
        backend = ScriptedBackend([_write_call(fname, "x"), "Done."])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write")
        events = _drain(session, until_types={"confirm_request"})
        return session, events[-1]

    def test_approve_emits_confirm_resolved(self, tmp_path):
        session, req = self._request(tmp_path, "a.txt")
        session.answer_confirm(req["confirm_id"], approved=True)
        events = _drain(session, until_types={"final"})
        resolved = next(e for e in events if e["type"] == "confirm_resolved")
        assert resolved["confirm_id"] == req["confirm_id"]
        assert resolved["approved"] is True
        assert resolved["timed_out"] is False

    def test_reject_emits_confirm_resolved(self, tmp_path):
        session, req = self._request(tmp_path, "b.txt")
        session.answer_confirm(req["confirm_id"], approved=False)
        events = _drain(session, until_types={"final"})
        resolved = next(e for e in events if e["type"] == "confirm_resolved")
        assert resolved["approved"] is False
        assert resolved["timed_out"] is False

    def test_stop_emits_confirm_resolved_rejected(self, tmp_path):
        session, req = self._request(tmp_path, "c.txt")
        session.stop()
        events = _drain(session, until_types={"final"})
        resolved = next(e for e in events if e["type"] == "confirm_resolved")
        assert resolved["confirm_id"] == req["confirm_id"]
        assert resolved["approved"] is False

    def test_replay_buffer_contains_resolution(self, tmp_path):
        session, req = self._request(tmp_path, "d.txt")
        session.answer_confirm(req["confirm_id"], approved=True)
        _drain(session, until_types={"final"})
        types = [e["type"] for e in session.history]
        i_req = types.index("confirm_request")
        i_res = types.index("confirm_resolved")
        assert i_req < i_res          # replay rebuilds the card, then resolves it
        assert session.history[i_res]["confirm_id"] == req["confirm_id"]


class TestSessionManager:
    def test_create_get_remove(self, tmp_path):
        mgr = SessionManager()
        s = CoderSession(tmp_path, ScriptedBackend(["x"]), auto_approve=True)
        mgr.create(s)
        assert mgr.get(s.id) is s
        assert mgr.remove(s.id) is s
        assert mgr.get(s.id) is None
        assert mgr.remove(s.id) is None


# ------------------------------------------------------------------ #
#  Web endpoints                                                      #
# ------------------------------------------------------------------ #

@pytest.fixture
def gui_app(tmp_path):
    app = FastAPI()
    switched = []

    async def switch_model(name):
        switched.append(name)

    attach_gui(
        app,
        self_url="http://127.0.0.1:9/v1",   # never dialled in these tests
        switch_model=switch_model,
        active_model=lambda: switched[-1] if switched else "model-a",
    )
    return app, switched


_FAKE_REGISTRY = {
    "model-a": {"path": "C:/nonexistent/a.gguf", "source": "local"},
    "model-b": {"path": "C:/nonexistent/b.gguf", "source": "hf"},
}


class TestModelEndpoints:
    def test_models_lists_registry(self, gui_app):
        app, _ = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                data = client.get("/api/models").json()
        names = [m["name"] for m in data["models"]]
        assert names == ["model-a", "model-b"]
        assert data["active"] == "model-a"
        active = next(m for m in data["models"] if m["active"])
        assert active["name"] == "model-a"

    def test_load_unknown_model_404(self, gui_app):
        app, _ = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/models/load", json={"model": "nope"})
        assert r.status_code == 404

    def test_load_active_model_is_noop(self, gui_app):
        app, switched = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/models/load", json={"model": "model-a"})
        assert r.json()["status"] == "already_active"
        assert switched == []

    def test_load_switches_model(self, gui_app):
        app, switched = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/models/load", json={"model": "model-b"})
        assert r.json()["status"] == "loaded"
        assert switched == ["model-b"]


class TestCoderEndpoints:
    def test_create_session_rejects_bad_cwd(self, gui_app):
        app, _ = gui_app
        with TestClient(app) as client:
            r = client.post("/api/coder/sessions",
                            json={"cwd": "Z:/definitely/not/here"})
        assert r.status_code == 400

    def test_create_and_delete_session(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            r = client.post("/api/coder/sessions", json={"cwd": str(tmp_path)})
            assert r.status_code == 200
            sid = r.json()["id"]
            assert r.json()["cwd"] == str(tmp_path.resolve())

            r = client.post(f"/api/coder/sessions/{sid}/message",
                            json={"text": ""})
            assert r.status_code == 400      # empty message

            r = client.post(f"/api/coder/sessions/{sid}/confirm",
                            json={"confirm_id": "none", "approved": True})
            assert r.status_code == 409      # nothing pending

            assert client.delete(f"/api/coder/sessions/{sid}").status_code == 200
            assert client.delete(f"/api/coder/sessions/{sid}").status_code == 404

    def test_unknown_session_404(self, gui_app):
        app, _ = gui_app
        with TestClient(app) as client:
            assert client.post("/api/coder/sessions/zzz/message",
                               json={"text": "hi"}).status_code == 404
            assert client.post("/api/coder/sessions/zzz/stop").status_code == 404
            assert client.delete("/api/coder/sessions/zzz").status_code == 404

    def test_event_stream_ends_on_closed(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            # Closing poisons the queue with a "closed" event → stream terminates
            client.delete(f"/api/coder/sessions/{sid}")

            # Recreate to test the live stream path with a pre-poisoned queue
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            from localm.plugins.gui import web as _web  # noqa: F401
            # Reach the manager through the route closure is awkward; instead
            # drive the stream by closing the session from another thread.
            def _close_soon():
                time.sleep(0.3)
                client.delete(f"/api/coder/sessions/{sid}")
            t = threading.Thread(target=_close_soon, daemon=True)
            t.start()
            collected = []
            with client.stream("GET", f"/api/coder/sessions/{sid}/events") as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        collected.append(json.loads(line[6:]))
                        if collected[-1]["type"] == "closed":
                            break
            t.join()
        assert collected[-1]["type"] == "closed"


class TestModelLessServer:
    """The GUI starts with no engine on a fresh install (empty registry); the
    user adds a model from the Models page. The server must not crash."""

    @pytest.fixture
    def app_no_engine(self):
        from localm.inference.http_server import create_app
        app = create_app(None)
        with TestClient(app) as client:
            yield client

    def test_v1_models_empty_when_no_engine(self, app_no_engine):
        data = app_no_engine.get("/v1/models").json()
        assert data == {"object": "list", "data": []}

    def test_health_503_when_no_engine(self, app_no_engine):
        assert app_no_engine.get("/health").status_code == 503

    def test_gui_models_lists_registry_without_engine(self):
        """/api/models reads the registry, not the engine — works model-less."""
        app = FastAPI()

        async def switch_model(name):
            pass

        attach_gui(app, self_url="http://127.0.0.1:9/v1",
                   switch_model=switch_model, active_model=lambda: "")
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                data = client.get("/api/models").json()
        assert data["active"] == ""
        assert [m["name"] for m in data["models"]] == ["model-a", "model-b"]
        assert all(m["active"] is False for m in data["models"])


class TestStaticFiles:
    def test_index_served(self, gui_app):
        app, _ = gui_app
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 200
            assert "localm" in r.text
            assert client.get("/app.js").status_code == 200
            assert client.get("/style.css").status_code == 200
            assert client.get("/vendor/marked.min.js").status_code == 200


class TestSessionExtras:
    def test_session_list_and_info(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            assert client.get("/api/coder/sessions").json()["sessions"] == []
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            sessions = client.get("/api/coder/sessions").json()["sessions"]
            assert [s["id"] for s in sessions] == [sid]
            assert sessions[0]["busy"] is False
            assert sessions[0]["mode"] == "privacy"
            client.delete(f"/api/coder/sessions/{sid}")

    def test_undo_with_nothing_to_undo_is_409(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            assert client.post(f"/api/coder/sessions/{sid}/undo").status_code == 409
            assert client.post(f"/api/coder/sessions/{sid}/compact").status_code == 409
            client.delete(f"/api/coder/sessions/{sid}")

    def test_log_404_in_privacy_mode(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            assert client.get(f"/api/coder/sessions/{sid}/log").status_code == 404
            client.delete(f"/api/coder/sessions/{sid}")

    def test_create_with_unknown_model_404(self, gui_app, tmp_path):
        app, _ = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/coder/sessions",
                                json={"cwd": str(tmp_path), "model": "ghost"})
        assert r.status_code == 404

    def test_create_with_model_switches_engine(self, gui_app, tmp_path):
        app, switched = gui_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/coder/sessions",
                                json={"cwd": str(tmp_path), "model": "model-b"})
                assert r.status_code == 200
                client.delete(f"/api/coder/sessions/{r.json()['id']}")
        assert switched == ["model-b"]

    def test_replay_rebuilds_history(self, gui_app, tmp_path):
        app, _ = gui_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            # Drive some history directly through the session object
            # (no model behind these tests)
            import localm.plugins.gui.web  # noqa: F401
            # fetch via the manager closure: reach through the route list
            # is brittle — instead use the documented API shape: events with
            # replay after pushing through a message is covered by the
            # CoderSession unit tests; here we just check the marker frame.
            collected = []
            def _close_soon():
                time.sleep(0.3)
                client.delete(f"/api/coder/sessions/{sid}")
            t = threading.Thread(target=_close_soon, daemon=True)
            t.start()
            with client.stream(
                "GET", f"/api/coder/sessions/{sid}/events?replay=true") as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        collected.append(json.loads(line[6:]))
                        if collected[-1]["type"] in ("closed",):
                            break
            t.join()
        assert collected[0]["type"] == "replay_done"


class TestPlatformEndpoints:
    """The /v1/plugins, /v1/config, /v1/models/{id} endpoints live on the
    inference app; build one with a stub engine."""

    @pytest.fixture
    def v1_client(self):
        from localm.inference.http_server import create_app
        engine = type("E", (), {"display_name": "model-a", "loaded": False})()
        app = create_app(engine)
        with TestClient(app) as client:
            yield client

    def test_model_detail_404(self, v1_client):
        with patch("localm.config.load_registry", return_value={}):
            assert v1_client.get("/v1/models/nope").status_code == 404

    def test_model_detail_fields(self, v1_client, tmp_path):
        f = tmp_path / "a.gguf"
        f.write_bytes(b"x" * 128)
        registry = {
            "model-a": {"path": str(f), "source": "local", "sha256": "ab" * 32},
            "alias-a": {"path": str(f), "source": "local"},
        }
        with patch("localm.config.load_registry", return_value=registry):
            data = v1_client.get("/v1/models/model-a").json()
        assert data["size_bytes"] == 128
        assert data["aliases"] == ["alias-a"]
        assert data["sha256"] == "ab" * 32
        assert data["active"] is True

    def test_config_roundtrip(self, v1_client, tmp_path):
        cfg_file = tmp_path / "config.json"
        with patch("localm.config.CONFIG_FILE", cfg_file), \
             patch("localm.config.HOME_DIR", tmp_path), \
             patch("localm.config.MODELS_DIR", tmp_path / "models"):
            data = v1_client.get("/v1/config").json()
            assert data["n_ctx"] == 4096
            r = v1_client.patch("/v1/config", json={"n_ctx": 8192})
            assert r.json()["n_ctx"] == 8192
            assert json.loads(cfg_file.read_text())["n_ctx"] == 8192

    def test_config_rejects_unknown_keys(self, v1_client):
        r = v1_client.patch("/v1/config", json={"hax": 1})
        assert r.status_code == 400

    def test_plugins_list_empty(self, v1_client, tmp_path):
        with patch("localm.plugins.loader.plugins_dir", return_value=tmp_path):
            data = v1_client.get("/v1/plugins").json()
        assert data["plugins"] == []
        assert data["errors"] == []

    def test_plugin_install_and_remove(self, v1_client, tmp_path):
        src = tmp_path / "src" / "myplug"
        src.mkdir(parents=True)
        (src / "plugin.toml").write_text(
            '[plugin]\nname = "myplug"\nentry = "mod:main"\n', encoding="utf-8")
        (src / "mod.py").write_text("main = None\n", encoding="utf-8")
        plugdir = tmp_path / "installed"
        with patch("localm.plugins.loader.plugins_dir", return_value=plugdir):
            r = v1_client.post("/v1/plugins/install", json={"source": str(src)})
            assert r.status_code == 200
            assert r.json()["name"] == "myplug"
            assert (plugdir / "myplug" / "plugin.toml").is_file()
            assert v1_client.delete("/v1/plugins/myplug").status_code == 200
            assert v1_client.delete("/v1/plugins/myplug").status_code == 404

    def test_plugin_install_invalid_source(self, v1_client):
        r = v1_client.post("/v1/plugins/install", json={"source": "Z:/nope"})
        assert r.status_code == 400


class TestJobs:
    def test_cli_job_streams_lines_and_ends(self):
        from localm.plugins.gui.jobs import JobManager
        mgr = JobManager()
        # Use python -m localm --help via start_cli's own python: cheap + real
        job = mgr.start_cli("pull", ["--help"])
        events = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                ev = job.events.get(timeout=0.5)
            except queue.Empty:
                continue
            events.append(ev)
            if ev["type"] == "end":
                break
        assert events[-1]["type"] == "end"
        assert events[-1]["status"] == "done"
        assert any("localm" in e.get("text", "") for e in events if e["type"] == "line")

    def test_fn_job_success_and_failure(self):
        from localm.plugins.gui.jobs import JobManager
        mgr = JobManager()

        ok_job = mgr.start_fn("imagine", lambda job: True)
        fail_job = mgr.start_fn("imagine", lambda job: False)
        boom_job = mgr.start_fn("imagine", lambda job: (_ for _ in ()).throw(RuntimeError("x")))

        for job, status in ((ok_job, "done"), (fail_job, "failed"), (boom_job, "failed")):
            end = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    ev = job.events.get(timeout=0.5)
                except queue.Empty:
                    continue
                if ev["type"] == "end":
                    end = ev
                    break
            assert end is not None
            assert end["status"] == status


# ------------------------------------------------------------------ #
#  Conversation store (server-side chat persistence)                  #
# ------------------------------------------------------------------ #

@pytest.fixture
def persist_app(tmp_path, monkeypatch):
    """GUI app whose data dir lives under tmp_path."""
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, tmp_path / ".localm" / "chats"


class TestConversationStore:
    def test_privacy_mode_disables_store(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        app, chats = persist_app
        with TestClient(app) as client:
            data = client.get("/api/conversations").json()
            assert data == {"enabled": False, "conversations": []}
            r = client.put("/api/conversations/abc",
                           json={"title": "x", "messages": []})
            assert r.status_code == 403
            assert client.delete("/api/conversations/abc").status_code == 403
        assert not chats.exists()       # privacy: not even an empty directory

    def test_upsert_list_delete_roundtrip(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, chats = persist_app
        with TestClient(app) as client:
            r = client.put("/api/conversations/abc123", json={
                "title": "My chat", "updated_at": 5,
                "messages": [{"role": "user", "content": "hi"}]})
            assert r.status_code == 200
            assert (chats / "abc123.json").is_file()

            data = client.get("/api/conversations").json()
            assert data["enabled"] is True
            assert [c["id"] for c in data["conversations"]] == ["abc123"]
            assert data["conversations"][0]["messages"][0]["content"] == "hi"

            assert client.delete(
                "/api/conversations/abc123").json()["status"] == "deleted"
            assert client.get("/api/conversations").json()["conversations"] == []
            assert client.delete(
                "/api/conversations/abc123").json()["status"] == "absent"

    def test_list_sorted_newest_first(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = persist_app
        with TestClient(app) as client:
            client.put("/api/conversations/old",
                       json={"title": "old", "updated_at": 1, "messages": []})
            client.put("/api/conversations/new",
                       json={"title": "new", "updated_at": 2, "messages": []})
            data = client.get("/api/conversations").json()
        assert [c["id"] for c in data["conversations"]] == ["new", "old"]

    @pytest.mark.parametrize("bad_id", [
        "..%5Cevil",        # ..\evil
        "a b",              # whitespace
        "x" * 65,           # too long
        "sp%C3%A4t",        # non-ASCII
    ])
    def test_invalid_ids_rejected(self, persist_app, monkeypatch, bad_id):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, chats = persist_app
        with TestClient(app) as client:
            r = client.put(f"/api/conversations/{bad_id}",
                           json={"title": "x", "messages": []})
            assert r.status_code == 400
            assert client.delete(f"/api/conversations/{bad_id}").status_code == 400
        assert not chats.exists()

    def test_corrupt_file_skipped(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, chats = persist_app
        chats.mkdir(parents=True)
        (chats / "broken.json").write_text("{nope", encoding="utf-8")
        with TestClient(app) as client:
            client.put("/api/conversations/ok",
                       json={"title": "ok", "updated_at": 1, "messages": []})
            data = client.get("/api/conversations").json()
        assert [c["id"] for c in data["conversations"]] == ["ok"]


# ------------------------------------------------------------------ #
#  Network tool gating (net_mode policy in the agent)                 #
# ------------------------------------------------------------------ #

class TestNetworkToolGating:
    @staticmethod
    def _fetch_call():
        return ("Fetching.\n<tool_call>\n"
                + json.dumps({"name": "fetch_url",
                              "args": {"url": "https://example.com/x"}})
                + "\n</tool_call>")

    def test_net_off_blocks_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "off")
        backend = ScriptedBackend([self._fetch_call(), "Understood."])
        session = CoderSession(tmp_path, backend, auto_approve=True)
        session.send_message("fetch the page")
        events = _drain(session, until_types={"final"})
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is False
        assert "network policy" in result["summary"]

    def test_net_ask_routes_through_approval(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "ask")
        backend = ScriptedBackend([self._fetch_call(), "Understood."])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("fetch the page")
        events = _drain(session, until_types={"confirm_request"})
        req = events[-1]
        assert req["tool"] == "fetch_url"
        assert "example.com" in json.dumps(req["args"])
        session.answer_confirm(req["confirm_id"], approved=False)
        events = _drain(session, until_types={"final"})
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is False          # rejected, nothing fetched

    def test_net_allow_runs_without_confirmation(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_NET_MODE", "allow")
        monkeypatch.setattr("localm.netpolicy.fetch_text",
                            lambda url, **kw: (url, "FETCHED BODY"))
        backend = ScriptedBackend([self._fetch_call(), "Done."])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("fetch the page")
        events = _drain(session, until_types={"final"})
        types = [e["type"] for e in events]
        assert "confirm_request" not in types
        result = next(e for e in events if e["type"] == "tool_result")
        assert result["ok"] is True


# ------------------------------------------------------------------ #
#  Web endpoints (/api/web/*)                                         #
# ------------------------------------------------------------------ #

class TestWebEndpoints:
    def test_search_success_and_empty_query(self, gui_app, monkeypatch):
        app, _ = gui_app
        monkeypatch.setattr(
            "localm.netpolicy.web_search",
            lambda q, max_results=5: [
                {"title": "T", "url": "https://t/", "snippet": "s"}])
        with TestClient(app) as client:
            data = client.post("/api/web/search", json={"query": "x"}).json()
            assert data["results"][0]["title"] == "T"
            assert client.post("/api/web/search",
                               json={"query": "  "}).status_code == 400

    def test_search_policy_refusal_is_403(self, gui_app, monkeypatch):
        from localm.netpolicy import NetworkPolicyError
        app, _ = gui_app

        def deny(q, max_results=5):
            raise NetworkPolicyError("Network access is disabled (net_mode=off).")
        monkeypatch.setattr("localm.netpolicy.web_search", deny)
        with TestClient(app) as client:
            r = client.post("/api/web/search", json={"query": "x"})
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"]

    def test_fetch_truncation_and_failure(self, gui_app, monkeypatch):
        app, _ = gui_app
        monkeypatch.setattr("localm.netpolicy.fetch_text",
                            lambda url, **kw: (url, "y" * 2000))
        with TestClient(app) as client:
            data = client.post("/api/web/fetch",
                               json={"url": "https://e/", "max_chars": 500}).json()
            assert data["truncated"] is True
            assert len(data["text"]) == 500

        def boom(url, **kw):
            raise RuntimeError("connection refused")
        monkeypatch.setattr("localm.netpolicy.fetch_text", boom)
        with TestClient(app) as client:
            assert client.post("/api/web/fetch",
                               json={"url": "https://e/"}).status_code == 502


# ------------------------------------------------------------------ #
#  Model discovery endpoints (/api/discover/*)                         #
# ------------------------------------------------------------------ #

class TestDiscoverEndpoints:
    def test_search_returns_results_and_vram(self, gui_app, monkeypatch):
        app, _ = gui_app
        monkeypatch.setattr(
            "localm.discover.hf_search",
            lambda q, limit=20: [{"id": "org/m", "downloads": 1,
                                  "likes": 0, "updated": ""}])
        monkeypatch.setattr("localm.discover.vram_info",
                            lambda: {"total": 16_000_000_000})
        with TestClient(app) as client:
            data = client.get("/api/discover/search?q=llama").json()
        assert data["results"][0]["id"] == "org/m"
        assert data["vram"]["total"] == 16_000_000_000

    def test_files_get_fit_badges(self, gui_app, monkeypatch):
        app, _ = gui_app
        monkeypatch.setattr(
            "localm.discover.hf_gguf_files",
            lambda repo: [{"file": "m-Q4_K_M.gguf", "quant": "Q4_K_M",
                           "size_bytes": 4_000_000_000, "n_parts": 1}])
        monkeypatch.setattr("localm.discover.vram_info",
                            lambda: {"total": 16_000_000_000})
        with TestClient(app) as client:
            data = client.get("/api/discover/files?repo=org/m").json()
        assert data["files"][0]["fit"] == "fits"

    def test_net_off_is_403(self, gui_app, monkeypatch):
        from localm.discover import DiscoverError
        app, _ = gui_app

        def blocked(q, limit=20):
            raise DiscoverError("Network access is disabled (net_mode=off).")
        monkeypatch.setattr("localm.discover.hf_search", blocked)
        with TestClient(app) as client:
            r = client.get("/api/discover/search?q=x")
        assert r.status_code == 403

    def test_hf_unreachable_is_502(self, gui_app, monkeypatch):
        from localm.discover import DiscoverError
        app, _ = gui_app

        def down(repo):
            raise DiscoverError("HuggingFace request failed: timeout")
        monkeypatch.setattr("localm.discover.hf_gguf_files", down)
        with TestClient(app) as client:
            assert client.get("/api/discover/files?repo=a/b").status_code == 502


# ------------------------------------------------------------------ #
#  Knowledge endpoints (/api/rag/*)                                    #
# ------------------------------------------------------------------ #

class TestRagEndpoints:
    """persist_app monkeypatches Path.home → collections land under tmp."""

    @staticmethod
    def _wait_job(client, job_id, timeout=30):
        """Stream a job's SSE events until the end frame; return all lines."""
        import time as _time
        lines, end = [], None
        deadline = _time.monotonic() + timeout
        with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for raw in r.iter_lines():
                if _time.monotonic() > deadline:
                    break
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                if ev["type"] == "line":
                    lines.append(ev["text"])
                if ev["type"] == "end":
                    end = ev
                    break
        return end, lines

    def test_create_list_detail_delete(self, persist_app):
        app, _ = persist_app
        with TestClient(app) as client:
            assert client.get("/api/rag/collections").json() == {"collections": []}
            r = client.post("/api/rag/collections", json={"name": "kb1"})
            assert r.status_code == 200
            assert r.json()["name"] == "kb1"
            # duplicate
            assert client.post("/api/rag/collections",
                               json={"name": "kb1"}).status_code == 409
            # invalid name
            assert client.post("/api/rag/collections",
                               json={"name": "a b"}).status_code == 400
            data = client.get("/api/rag/collections").json()
            assert [c["name"] for c in data["collections"]] == ["kb1"]
            detail = client.get("/api/rag/collections/kb1").json()
            assert detail["n_docs"] == 0 and detail["docs"] == []
            assert client.delete("/api/rag/collections/kb1").status_code == 200
            assert client.delete("/api/rag/collections/kb1").status_code == 404

    def test_add_and_query_roundtrip(self, persist_app, tmp_path):
        app, _ = persist_app
        docs = tmp_path / "kdocs"
        docs.mkdir()
        (docs / "gpu.md").write_text(
            "ROCm needs the gfx1030 runtime DLLs.", encoding="utf-8")
        with TestClient(app) as client:
            client.post("/api/rag/collections", json={"name": "kb"})
            # unknown collection / bad path validation
            assert client.post("/api/rag/collections/ghost/add",
                               json={"paths": [str(docs)]}).status_code == 404
            assert client.post("/api/rag/collections/kb/add",
                               json={"paths": ["Z:/nope"]}).status_code == 400

            r = client.post("/api/rag/collections/kb/add",
                            json={"paths": [str(docs)], "embed": False})
            assert r.status_code == 200
            end, lines = self._wait_job(client, r.json()["job_id"])
            assert end and end["status"] == "done"
            assert any("1 added" in l for l in lines)

            q = client.post("/api/rag/collections/kb/query",
                            json={"query": "ROCm runtime DLLs"})
            assert q.status_code == 200
            hits = q.json()["hits"]
            assert hits and "gpu.md" in hits[0]["source"]

            assert client.post("/api/rag/collections/kb/query",
                               json={"query": "  "}).status_code == 400
            assert client.post(
                "/api/rag/collections/kb/remove-doc",
                json={"path": "Z:/never"}).status_code == 404
            src = q.json()["hits"][0]["source"]
            assert client.post("/api/rag/collections/kb/remove-doc",
                               json={"path": src}).status_code == 200

    def test_extract_endpoint(self, persist_app):
        import base64
        app, _ = persist_app
        with TestClient(app) as client:
            b64 = base64.b64encode("hello attachment".encode()).decode()
            r = client.post("/api/rag/extract",
                            json={"filename": "note.txt", "content_b64": b64})
            assert r.status_code == 200
            assert r.json()["text"] == "hello attachment"
            assert r.json()["truncated"] is False
            # invalid base64
            assert client.post("/api/rag/extract",
                               json={"filename": "x.txt",
                                     "content_b64": "!!not-b64!!"}).status_code == 400
            # unsupported type
            exe = base64.b64encode(b"\x00\x01").decode()
            assert client.post("/api/rag/extract",
                               json={"filename": "x.exe",
                                     "content_b64": exe}).status_code == 422

    def test_extract_writes_nothing_to_disk(self, persist_app, tmp_path):
        """Privacy guarantee: attachment extraction is in-memory only."""
        import base64
        app, _ = persist_app
        home = tmp_path / ".localm"
        before = {str(p) for p in home.rglob("*")} if home.exists() else set()
        with TestClient(app) as client:
            b64 = base64.b64encode(b"secret content").decode()
            client.post("/api/rag/extract",
                        json={"filename": "secret.txt", "content_b64": b64})
        after = {str(p) for p in home.rglob("*")} if home.exists() else set()
        assert after == before


# ------------------------------------------------------------------ #
#  Coder session history (past audit logs)                            #
# ------------------------------------------------------------------ #

class TestCoderHistory:
    @staticmethod
    def _fake_log(sessions_dir, name="2026-01-01_000000_1_coder.jsonl"):
        sessions_dir.mkdir(parents=True, exist_ok=True)
        log = sessions_dir / name
        entry = {"t": 1, "turn": 0, "type": "user", "data": {"content": "hi"}}
        log.write_text(json.dumps(entry) + "\nnot json\n", encoding="utf-8")
        return log, entry

    def test_history_lists_and_reads_logs(self, gui_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        sessions_dir = tmp_path / "sessions"
        log, entry = self._fake_log(sessions_dir)
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", sessions_dir)
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = gui_app
        with TestClient(app) as client:
            data = client.get("/api/coder/history").json()
            assert data["enabled"] is True
            assert [l["name"] for l in data["logs"]] == [log.name]
            parsed = client.get(f"/api/coder/history/{log.name}").json()
        assert parsed["entries"] == [entry]     # malformed line skipped

    def test_history_enabled_false_in_privacy(self, gui_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", tmp_path / "none")
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        app, _ = gui_app
        with TestClient(app) as client:
            data = client.get("/api/coder/history").json()
        assert data == {"enabled": False, "logs": []}

    def test_history_rejects_bad_names(self, gui_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        sessions_dir = tmp_path / "sessions"
        self._fake_log(sessions_dir)
        # plant a sibling the traversal would reach
        (tmp_path / "secret.jsonl").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", sessions_dir)
        app, _ = gui_app
        with TestClient(app) as client:
            assert client.get("/api/coder/history/notes.txt").status_code == 400
            r = client.get("/api/coder/history/..%5Csecret.jsonl")
            assert r.status_code in (400, 404)


# ------------------------------------------------------------------ #
#  Agent hooks (direct)                                               #
# ------------------------------------------------------------------ #

class TestAgentHooks:
    def test_on_event_receives_token_stream(self, tmp_path):
        from localm.plugins.coder.agent import Agent
        events = []
        agent = Agent(
            ScriptedBackend(["Short answer."]),
            cwd=tmp_path,
            on_event=events.append,
        )
        out = agent.run_task("hello")
        assert out == "Short answer."
        tokens = [e for e in events if e["type"] == "token"]
        assert "".join(t["text"] for t in tokens) == "Short answer."

    def test_broken_sink_does_not_crash(self, tmp_path):
        from localm.plugins.coder.agent import Agent

        def explode(_):
            raise RuntimeError("sink failure")

        agent = Agent(
            ScriptedBackend(["Still fine."]),
            cwd=tmp_path,
            on_event=explode,
        )
        assert agent.run_task("hello") == "Still fine."

    def test_request_stop_before_run(self, tmp_path):
        from localm.plugins.coder.agent import Agent
        agent = Agent(ScriptedBackend(["unused"]), cwd=tmp_path)
        agent.request_stop()
        # A stale stop request must not kill the next task
        assert agent.run_task("hello") == "unused"


# ------------------------------------------------------------------ #
#  Image management endpoints                                         #
# ------------------------------------------------------------------ #

@pytest.fixture
def img_app(tmp_path, monkeypatch):
    """GUI app whose images dir lives under tmp_path."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    app = FastAPI()

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    images = tmp_path / ".localm" / "gui_images"
    images.mkdir(parents=True)
    return app, images


class TestImageManagement:
    @staticmethod
    def _make_image(images, name="img.png", meta=None):
        (images / name).write_bytes(b"\x89PNG fake")
        if meta is not None:
            (images / (name + ".json")).write_text(json.dumps(meta))

    def test_history_includes_path(self, img_app):
        app, images = img_app
        self._make_image(images, meta={"prompt": "a fox"})
        with TestClient(app) as client:
            data = client.get("/api/imagine/history").json()
        assert data["images"][0]["name"] == "img.png"
        assert data["images"][0]["path"] == str(images / "img.png")
        assert data["images"][0]["meta"]["prompt"] == "a fox"

    def test_delete_removes_file_and_sidecar(self, img_app):
        app, images = img_app
        self._make_image(images, meta={"prompt": "x"})
        with TestClient(app) as client:
            r = client.delete("/api/imagine/file/img.png")
        assert r.status_code == 200
        assert not (images / "img.png").exists()
        assert not (images / "img.png.json").exists()

    def test_delete_missing_404(self, img_app):
        app, _ = img_app
        with TestClient(app) as client:
            assert client.delete("/api/imagine/file/nope.png").status_code == 404

    def test_delete_rejects_path_traversal(self, img_app):
        app, _ = img_app
        with TestClient(app) as client:
            r = client.delete("/api/imagine/file/..%5Cconfig.json")
        assert r.status_code in (400, 404)

    @pytest.mark.parametrize("name", [
        "..%5Cconfig.json",       # ..\config.json
        "..%2Fconfig.json",       # ../config.json (decodes to /, off-route)
        "C:evil.png",             # Windows drive-relative (blocklist bypass)
        "%2e%2e%5c%2e%2e%5cwin.ini",  # ..\..\win.ini
        "sub%2Ffile.png",         # nested subpath
    ])
    def test_delete_rejects_traversal_vectors(self, img_app, tmp_path, name):
        app, _ = img_app
        # plant a file the traversal would target; it must survive
        target = tmp_path / ".localm" / "config.json"
        target.write_text("{}")
        with TestClient(app) as client:
            r = client.delete(f"/api/imagine/file/{name}")
        assert not (200 <= r.status_code < 300)   # never a successful delete
        assert target.exists()

    def test_confine_blocks_drive_relative_serve(self, img_app, tmp_path):
        """A drive-relative name must never resolve outside the images dir."""
        app, _ = img_app
        # plant a file one level up that the bypass would have reached
        (tmp_path / ".localm" / "config.json").write_text("{}")
        with TestClient(app) as client:
            r = client.get("/api/imagine/file/..%5Cconfig.json")
        assert r.status_code in (400, 404)

    def test_move_relocates_file_and_sidecar(self, img_app, tmp_path):
        app, images = img_app
        self._make_image(images, meta={"prompt": "x"})
        dest = tmp_path / "kept"
        with TestClient(app) as client:
            r = client.post("/api/imagine/file/img.png/move",
                            json={"dest": str(dest)})
        assert r.status_code == 200
        assert (dest / "img.png").is_file()
        assert (dest / "img.png.json").is_file()
        assert not (images / "img.png").exists()

    def test_move_refuses_overwrite(self, img_app, tmp_path):
        app, images = img_app
        self._make_image(images)
        dest = tmp_path / "kept"
        dest.mkdir()
        (dest / "img.png").write_bytes(b"existing")
        with TestClient(app) as client:
            r = client.post("/api/imagine/file/img.png/move",
                            json={"dest": str(dest)})
        assert r.status_code == 409
        assert (images / "img.png").exists()
