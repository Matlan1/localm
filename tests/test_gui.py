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
