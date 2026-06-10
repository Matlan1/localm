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
