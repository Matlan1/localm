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

from localm.plugins.coder.sessions import CoderSession, SessionManager
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

    def test_busy_queues_second_message(self, tmp_path):
        """Sending mid-task no longer bounces - the message is queued as a
        steering note and surfaced in the feed with queued=True."""
        class SlowBackend(ScriptedBackend):
            def chat_stream(self, messages, **kw):
                time.sleep(0.5)
                yield "done"

        session = CoderSession(tmp_path, SlowBackend(["done"]), auto_approve=True)
        assert session.send_message("one") == "started"
        assert session.send_message("two") == "queued"
        events = _drain(session, until_types={"final"})
        queued = [e for e in events
                  if e["type"] == "user" and e.get("queued")]
        assert len(queued) == 1 and queued[0]["text"] == "two"

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
        assert session.send_message("too late") == "closed"


class TestConfirmResolution:
    """Every answered confirmation must leave a confirm_resolved event in the
    stream AND the replay buffer - otherwise a reloaded page replays the
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


class TestCoderQoL:
    """Coder QoL round: always-allow, changed-files surfaces, ctx meter,
    dry-run wiring, and the queued-message follow-up."""

    def test_always_allow_skips_second_confirmation(self, tmp_path):
        backend = ScriptedBackend([
            _write_call("a.txt", "1"),
            _write_call("b.txt", "2"),
            "Done.",
        ])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write two files")
        events = _drain(session, until_types={"confirm_request"})
        req = events[-1]
        assert session.answer_confirm(req["confirm_id"], approved=True,
                                      always_allow=True)
        events = _drain(session, until_types={"final"})
        # The second write_file must NOT raise another confirmation
        assert not any(e["type"] == "confirm_request" for e in events)
        auto = [e for e in events if e["type"] == "info"
                and "always-allow" in e.get("text", "")]
        assert auto, "expected the auto-approve info event"
        assert (tmp_path / "b.txt").is_file()
        assert "write_file" in session.info()["allowed_tools"]

    def test_always_allow_ignored_on_reject(self, tmp_path):
        backend = ScriptedBackend([_write_call("a.txt", "1"), "Done."])
        session = CoderSession(tmp_path, backend, auto_approve=False)
        session.send_message("write")
        req = _drain(session, until_types={"confirm_request"})[-1]
        session.answer_confirm(req["confirm_id"], approved=False,
                               always_allow=True)
        _drain(session, until_types={"final"})
        assert session.allowed_tools == set()
        assert not (tmp_path / "a.txt").exists()

    def test_changed_files_and_diff_surfaces(self, tmp_path):
        backend = ScriptedBackend([_write_call("hello.txt", "hi"), "Done."])
        session = CoderSession(tmp_path, backend, auto_approve=True)
        session.send_message("write hello")
        events = _drain(session, until_types={"final"})
        files = session.changed_files()
        assert [f["path"] for f in files] == ["hello.txt"]
        assert "+hi" in session.session_diff("hello.txt")
        # the final event names the changed files for the feed summary
        final = events[-1]
        assert final["changed_files"] == ["hello.txt"]
        assert session.info()["changed_files"] == 1

    def test_turn_events_carry_ctx_ratio(self, tmp_path):
        session = CoderSession(tmp_path, ScriptedBackend(["Done."]),
                               auto_approve=True)
        session.send_message("hi")
        events = _drain(session, until_types={"final"})
        turn = next(e for e in events if e["type"] == "turn")
        assert "ctx_ratio" in turn
        assert 0.0 <= turn["ctx_ratio"] <= 1.0

    def test_dry_run_wires_through_to_agent(self, tmp_path):
        backend = ScriptedBackend([_write_call("x.txt", "x"), "Done."])
        session = CoderSession(tmp_path, backend, auto_approve=True,
                               dry_run=True)
        assert session.info()["dry_run"] is True
        session.send_message("write")
        _drain(session, until_types={"final"})
        assert not (tmp_path / "x.txt").exists()

    def test_leftover_queued_message_runs_as_followup(self, tmp_path):
        """A message queued in the task's final moments becomes a follow-up
        task instead of sitting in the queue forever."""
        class SlowFinish(ScriptedBackend):
            def chat_stream(self, messages, **kw):
                time.sleep(0.4)
                yield self._next()

        session = CoderSession(tmp_path, SlowFinish(["Done.", "Follow-up done."]),
                               auto_approve=True)
        assert session.send_message("first") == "started"
        assert session.send_message("second") == "queued"
        _drain(session, until_types={"final"}, timeout=15)
        # Whether it was drained mid-task or ran as a follow-up task, the
        # agent must see the second message shortly after the first final.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if "second" in json.dumps(session.agent._messages):
                break
            time.sleep(0.05)
        assert "second" in json.dumps(session.agent._messages)


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


class TestStatsEndpoint:
    """The hardware-monitor stats feed."""

    def test_system_stats_never_raises_and_is_a_dict(self):
        from localm.sysstats import system_stats
        stats = system_stats()                 # must not raise on any box
        assert isinstance(stats, dict)
        # Whatever sections are present must have a sane shape.
        if "vram" in stats:
            assert isinstance(stats["vram"].get("total"), int)
        if "ram" in stats:
            assert isinstance(stats["ram"].get("total"), int)
        if "cpu" in stats:
            assert isinstance(stats["cpu"].get("percent"), (int, float))

    def test_stats_endpoint_returns_collected_stats(self, gui_app):
        app, _ = gui_app
        fake = {"cpu": {"percent": 12.5},
                "ram": {"used": 4, "total": 8, "percent": 50.0},
                "vram": {"used": 2, "total": 16, "percent": 12.5}}
        with patch("localm.sysstats.system_stats", return_value=fake):
            with TestClient(app) as client:
                r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() == fake

    def test_stats_endpoint_ok_when_nothing_measurable(self, gui_app):
        app, _ = gui_app
        with patch("localm.sysstats.system_stats", return_value={}):
            with TestClient(app) as client:
                r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.json() == {}


class TestVramEstimate:
    """The VRAM-estimate behind the Settings performance sliders."""

    def test_estimate_grows_with_context_and_offload(self):
        from localm.sysstats import estimate_vram
        gb = 1024 ** 3
        base = estimate_vram(4 * gb, n_ctx=4096, n_gpu_layers=99)
        more_ctx = estimate_vram(4 * gb, n_ctx=16384, n_gpu_layers=99)
        assert more_ctx["kv_cache"] > base["kv_cache"]
        assert more_ctx["needed"] > base["needed"]
        # Offloading fewer layers puts less weight on the GPU.
        half = estimate_vram(4 * gb, n_ctx=4096, n_gpu_layers=16, n_layers=32)
        assert half["weights"] < base["weights"]
        # Nothing to load -> nothing needed (no phantom overhead).
        assert estimate_vram(0, 0)["needed"] == 0

    def test_estimate_endpoint_shape_and_fit(self, gui_app):
        app, _ = gui_app
        reg = {"m": {"path": "C:/nonexistent/m.gguf", "source": "local"}}
        with patch("localm.config.load_registry", return_value=reg), \
             patch("localm.discover.vram_info",
                   return_value={"free": 20 * 1024 ** 3, "total": 24 * 1024 ** 3}):
            with TestClient(app) as client:
                r = client.get("/api/vram-estimate",
                               params={"model": "m", "n_ctx": 4096, "n_gpu_layers": 99})
        assert r.status_code == 200
        data = r.json()
        assert data["approximate"] is True
        assert {"weights", "kv_cache", "overhead", "needed", "free", "total", "fits"} <= data.keys()
        # 20 GB free, a (missing -> 0-byte) model: trivially fits.
        assert data["fits"] is True

    def test_estimate_fit_unknown_when_no_vram_reading(self, gui_app):
        app, _ = gui_app
        with patch("localm.config.load_registry", return_value={}), \
             patch("localm.discover.vram_info", return_value={}):
            with TestClient(app) as client:
                r = client.get("/api/vram-estimate")
        assert r.status_code == 200
        assert r.json()["fits"] is None        # can't claim fit without a reading


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

    def _capture_pull_args(self, monkeypatch):
        """Patch JobManager.start_cli to capture the argv it would run."""
        captured = {}

        class _FakeJob:
            id = "job-test"

        def fake_start_cli(self, kind, cli_args, **kw):
            captured["kind"] = kind
            captured["args"] = list(cli_args)
            return _FakeJob()

        monkeypatch.setattr(
            "localm.plugins.gui.jobs.JobManager.start_cli", fake_start_cli)
        return captured

    def test_pull_passes_spec_after_double_dash(self, gui_app, monkeypatch):
        """A flag-like spec (e.g. -h) must reach the CLI as the argument, not an
        option, so Click never dumps help text or a usage error."""
        app, _ = gui_app
        captured = self._capture_pull_args(monkeypatch)
        with TestClient(app) as client:
            r = client.post("/api/models/pull", json={"spec": "-h"})
        assert r.status_code == 200
        assert captured["args"] == ["pull", "--", "-h"]

    def test_pull_name_option_precedes_separator(self, gui_app, monkeypatch):
        app, _ = gui_app
        captured = self._capture_pull_args(monkeypatch)
        with TestClient(app) as client:
            r = client.post(
                "/api/models/pull", json={"spec": "owner/repo", "name": "alias1"})
        assert r.status_code == 200
        assert captured["args"] == ["pull", "--name", "alias1", "--", "owner/repo"]

    def test_pull_rejects_all_dash_spec(self, gui_app, monkeypatch):
        app, _ = gui_app
        self._capture_pull_args(monkeypatch)
        with TestClient(app) as client:
            for bad in ("--", "-", "   "):
                r = client.post("/api/models/pull", json={"spec": bad})
                assert r.status_code == 400


@pytest.fixture
def coder_app(tmp_path, monkeypatch):
    """GUI app with the builtin coder plugin installed; isolated home.
    coder became a plugin in Phase 3 - installed BEFORE attach_gui (production
    order: the engine mounts /api/coder routes first, then attach_gui publishes
    app.state.coder_sessions / switch_model, which the routes read lazily)."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    switched = []

    async def switch_model(name):
        switched.append(name)

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model,
               active_model=lambda: switched[-1] if switched else "model-a")
    return app, switched


class TestCoderEndpoints:
    def test_create_session_rejects_bad_cwd(self, coder_app):
        app, _ = coder_app
        with TestClient(app) as client:
            r = client.post("/api/coder/sessions",
                            json={"cwd": "Z:/definitely/not/here"})
        assert r.status_code == 400

    def test_create_and_delete_session(self, coder_app, tmp_path):
        app, _ = coder_app
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

    def test_files_endpoints_and_dry_run_flag(self, coder_app, tmp_path):
        app, _ = coder_app
        with TestClient(app) as client:
            info = client.post("/api/coder/sessions",
                               json={"cwd": str(tmp_path),
                                     "dry_run": True}).json()
            assert info["dry_run"] is True
            sid = info["id"]

            assert client.get(
                f"/api/coder/sessions/{sid}/files").json() == {"files": []}
            # full-session diff of an untouched session is empty, not an error
            r = client.get(f"/api/coder/sessions/{sid}/files/diff")
            assert r.status_code == 200 and r.json()["diff"] == ""
            # a specific never-touched path is a 404
            r = client.get(f"/api/coder/sessions/{sid}/files/diff",
                           params={"path": "never.txt"})
            assert r.status_code == 404

    def test_unknown_session_404(self, coder_app):
        app, _ = coder_app
        with TestClient(app) as client:
            assert client.post("/api/coder/sessions/zzz/message",
                               json={"text": "hi"}).status_code == 404
            assert client.post("/api/coder/sessions/zzz/stop").status_code == 404
            assert client.delete("/api/coder/sessions/zzz").status_code == 404

    def test_event_stream_ends_on_closed(self, coder_app, tmp_path):
        app, _ = coder_app
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
        """/api/models reads the registry, not the engine - works model-less."""
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

    def test_pwa_shell_served(self, gui_app):
        """The PWA shell is served with the right MIME types, and index.html
        links the manifest + registers the service worker - so the GUI installs
        as a companion app (and is reachable from a phone over the network)."""
        app, _ = gui_app
        with TestClient(app) as client:
            m = client.get("/manifest.webmanifest")
            assert m.status_code == 200
            assert m.headers["content-type"].startswith("application/manifest+json")
            body = m.json()
            assert body["start_url"] == "/" and body["display"] == "standalone"
            assert body["icons"][0]["src"] == "/icon.svg"

            sw = client.get("/sw.js")
            assert sw.status_code == 200
            assert sw.headers["content-type"].startswith("text/javascript")
            assert "localm-shell" in sw.text
            # the worker must never cache live API/model traffic
            assert "api|v1|plugins" in sw.text

            icon = client.get("/icon.svg")
            assert icon.status_code == 200
            assert icon.headers["content-type"].startswith("image/svg+xml")

            html = client.get("/").text
            assert 'rel="manifest"' in html
            assert "/sw.js" in html                     # registration present


class TestSessionExtras:
    def test_session_list_and_info(self, coder_app, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "privacy")  # hermetic: ignore ambient config mode
        app, _ = coder_app
        with TestClient(app) as client:
            assert client.get("/api/coder/sessions").json()["sessions"] == []
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            sessions = client.get("/api/coder/sessions").json()["sessions"]
            assert [s["id"] for s in sessions] == [sid]
            assert sessions[0]["busy"] is False
            assert sessions[0]["mode"] == "privacy"
            client.delete(f"/api/coder/sessions/{sid}")

    def test_undo_with_nothing_to_undo_is_409(self, coder_app, tmp_path):
        app, _ = coder_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            assert client.post(f"/api/coder/sessions/{sid}/undo").status_code == 409
            assert client.post(f"/api/coder/sessions/{sid}/compact").status_code == 409
            client.delete(f"/api/coder/sessions/{sid}")

    def test_log_404_in_privacy_mode(self, coder_app, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "privacy")  # hermetic: ignore ambient config mode
        app, _ = coder_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            assert client.get(f"/api/coder/sessions/{sid}/log").status_code == 404
            client.delete(f"/api/coder/sessions/{sid}")

    def test_create_with_unknown_model_404(self, coder_app, tmp_path):
        app, _ = coder_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/coder/sessions",
                                json={"cwd": str(tmp_path), "model": "ghost"})
        assert r.status_code == 404

    def test_create_with_model_switches_engine(self, coder_app, tmp_path):
        app, switched = coder_app
        with patch("localm.config.load_registry", return_value=_FAKE_REGISTRY):
            with TestClient(app) as client:
                r = client.post("/api/coder/sessions",
                                json={"cwd": str(tmp_path), "model": "model-b"})
                assert r.status_code == 200
                client.delete(f"/api/coder/sessions/{r.json()['id']}")
        assert switched == ["model-b"]

    def test_replay_rebuilds_history(self, coder_app, tmp_path):
        app, _ = coder_app
        with TestClient(app) as client:
            sid = client.post("/api/coder/sessions",
                              json={"cwd": str(tmp_path)}).json()["id"]
            # Drive some history directly through the session object
            # (no model behind these tests)
            import localm.plugins.gui.web  # noqa: F401
            # fetch via the manager closure: reach through the route list
            # is brittle - instead use the documented API shape: events with
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
        # H5: /v1/config + /v1/plugins are management routes; in open mode they
        # need the loopback shell token (the GUI carries it).
        with TestClient(
            app, headers={"Authorization": f"Bearer {app.state.shell_token}"}
        ) as client:
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

    def test_config_accepts_plugins_enabled(self, v1_client, tmp_path):
        """plugins_enabled is a real config key (managed by the engine); it must
        not be rejected as unknown - that was B3 ('Unknown config keys:
        plugins_enabled' on every settings save)."""
        cfg_file = tmp_path / "config.json"
        with patch("localm.config.CONFIG_FILE", cfg_file), \
             patch("localm.config.HOME_DIR", tmp_path), \
             patch("localm.config.MODELS_DIR", tmp_path / "models"):
            data = v1_client.get("/v1/config").json()
            assert data["plugins_enabled"] == []          # default present now
            r = v1_client.patch("/v1/config", json={"plugins_enabled": ["chat"]})
            assert r.status_code == 200, r.text
            assert r.json()["plugins_enabled"] == ["chat"]
            assert json.loads(cfg_file.read_text())["plugins_enabled"] == ["chat"]

    def test_config_ignores_readonly_extras(self, v1_client, tmp_path):
        """The GET handler injects effective_* extras; echoing them back on a
        PATCH must not 400 and must not be persisted."""
        cfg_file = tmp_path / "config.json"
        with patch("localm.config.CONFIG_FILE", cfg_file), \
             patch("localm.config.HOME_DIR", tmp_path), \
             patch("localm.config.MODELS_DIR", tmp_path / "models"):
            r = v1_client.patch("/v1/config", json={
                "n_ctx": 8192, "effective_mode": "privacy",
                "effective_coder_mode": "log", "effective_ctx_max": 99999})
            assert r.status_code == 200, r.text
            stored = json.loads(cfg_file.read_text())
            assert stored["n_ctx"] == 8192
            assert "effective_mode" not in stored
            assert "effective_ctx_max" not in stored

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


class TestGuiNoModel:
    """`localm gui --no-model` opens model-less even when usable models exist."""

    def test_no_model_flag_skips_auto_selection(self, monkeypatch):
        from click.testing import CliRunner

        from localm.plugins.gui import cli as guicli

        # A populated registry that would normally auto-select a default model.
        monkeypatch.setattr(
            "localm.config.load_registry",
            lambda: {"some-model": {"path": "x.gguf", "source": "local"}})
        # get_model_info must NOT be consulted when --no-model is set.
        monkeypatch.setattr(
            "localm.model_manager.get_model_info",
            lambda name: (_ for _ in ()).throw(AssertionError("auto-selected a model")))
        monkeypatch.setattr("localm.winconsole.disable_quickedit", lambda: None)
        ran = {}
        monkeypatch.setattr("uvicorn.run", lambda app, **kw: ran.setdefault("ok", True))

        result = CliRunner().invoke(guicli.main, ["--no-model", "--no-browser"])
        assert result.exit_code == 0, result.output
        assert "no model loaded" in result.output.lower()
        assert ran.get("ok")


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

    def test_pull_flaglike_spec_fails_not_help(self):
        """End-to-end: `pull -- -h` treats -h as the spec (unknown) and exits
        non-zero, so the job is 'failed' and no Click help text is dumped."""
        from localm.plugins.gui.jobs import JobManager
        mgr = JobManager()
        # cli_args is the full argv after "-m localm" (the endpoint passes the
        # "pull" subcommand itself), so this runs: localm pull -- -h
        job = mgr.start_cli("pull", ["pull", "--", "-h"])
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
        assert events[-1]["status"] == "failed"
        text = " ".join(e.get("text", "") for e in events if e["type"] == "line")
        assert "Usage:" not in text          # help was NOT dumped
        assert "Unknown spec" in text        # treated as a (bad) model spec

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
    """GUI app with the builtin CHAT plugin installed; data dir under tmp_path.
    chat became the protected, default-enabled plugin #0 in Phase 3 - installed
    BEFORE attach_gui so /api/conversations|memory|prompts mount. The config
    constants are pinned (like music_app) so install()'s enable write stays in
    the throwaway home."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("chat")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home / "chats"


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

    def test_pinned_folder_branches_roundtrip(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = persist_app
        branches = [{"parent": "root", "current": 1,
                     "tails": [[{"role": "user", "content": "old", "id": "a-1"}],
                               None]}]
        with TestClient(app) as client:
            client.put("/api/conversations/org1", json={
                "title": "work", "updated_at": 1, "pinned": True,
                "folder": "projects", "branches": branches, "messages": []})
            client.put("/api/conversations/org2", json={
                "title": "plain", "updated_at": 2, "messages": []})
            convs = {c["id"]: c for c in
                     client.get("/api/conversations").json()["conversations"]}
        assert convs["org1"]["pinned"] is True
        assert convs["org1"]["folder"] == "projects"
        assert convs["org1"]["branches"] == branches
        assert convs["org2"]["pinned"] is False     # defaults applied
        assert convs["org2"]["folder"] is None
        assert convs["org2"]["branches"] == []

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

@pytest.fixture
def web_app(tmp_path, monkeypatch):
    """A bare app with the builtin web plugin loaded (open mode). web became a
    plugin in Phase 3, so its routes are mounted by enabling it - not attach_gui."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("web")
    return app


class TestWebEndpoints:
    def test_search_success_and_empty_query(self, web_app, monkeypatch):
        app = web_app
        monkeypatch.setattr(
            "localm.netpolicy.web_search",
            lambda q, max_results=5: [
                {"title": "T", "url": "https://t/", "snippet": "s"}])
        with TestClient(app) as client:
            data = client.post("/api/web/search", json={"query": "x"}).json()
            assert data["results"][0]["title"] == "T"
            assert client.post("/api/web/search",
                               json={"query": "  "}).status_code == 400

    def test_search_policy_refusal_is_403(self, web_app, monkeypatch):
        from localm.netpolicy import NetworkPolicyError
        app = web_app

        def deny(q, max_results=5):
            raise NetworkPolicyError("Network access is disabled (net_mode=off).")
        monkeypatch.setattr("localm.netpolicy.web_search", deny)
        with TestClient(app) as client:
            r = client.post("/api/web/search", json={"query": "x"})
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"]

    def test_fetch_truncation_and_failure(self, web_app, monkeypatch):
        app = web_app
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
#  Voice (/api/voice/transcribe)                                       #
# ------------------------------------------------------------------ #

@pytest.fixture
def voice_app(tmp_path, monkeypatch):
    """A bare app with the builtin voice plugin loaded (open mode). voice became
    a plugin in Phase 3, so its routes are mounted by enabling it - not attach_gui."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("voice")
    return app


class TestVoiceEndpoint:
    def test_success_path(self, voice_app, monkeypatch):
        import base64
        app = voice_app
        monkeypatch.setattr("localm.voice.transcribe_bytes",
                            lambda data, language=None: "hello world")
        with TestClient(app) as client:
            r = client.post("/api/voice/transcribe", json={
                "audio_b64": base64.b64encode(b"fake-webm").decode()})
        assert r.status_code == 200
        assert r.json()["text"] == "hello world"

    def test_missing_package_is_501(self, voice_app):
        """faster-whisper is not installed in the test venv - the real
        VoiceError install-hint path must surface as 501."""
        import base64
        try:
            import faster_whisper  # noqa: F401
            pytest.skip("faster-whisper installed - hint path unreachable")
        except ImportError:
            pass
        app = voice_app
        with TestClient(app) as client:
            r = client.post("/api/voice/transcribe", json={
                "audio_b64": base64.b64encode(b"fake").decode()})
        assert r.status_code == 501
        assert "localm[voice]" in r.json()["detail"]

    def test_bad_base64_is_400(self, voice_app):
        app = voice_app
        with TestClient(app) as client:
            r = client.post("/api/voice/transcribe",
                            json={"audio_b64": "!!nope!!"})
        assert r.status_code == 400

    def test_transcription_failure_is_422(self, voice_app, monkeypatch):
        import base64
        from localm.voice import VoiceError
        app = voice_app

        def boom(data, language=None):
            raise VoiceError("No speech detected in the recording")
        monkeypatch.setattr("localm.voice.transcribe_bytes", boom)
        with TestClient(app) as client:
            r = client.post("/api/voice/transcribe", json={
                "audio_b64": base64.b64encode(b"silence").decode()})
        assert r.status_code == 422


# ------------------------------------------------------------------ #
#  Assistant memory (/api/memory)                                      #
# ------------------------------------------------------------------ #

class TestAssistantMemory:
    def test_roundtrip_and_append(self, persist_app, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = persist_app
        with TestClient(app) as client:
            data = client.get("/api/memory").json()
            assert data["text"] == "" and data["writable"] is True

            assert client.put("/api/memory",
                              json={"text": "- likes terse answers"}).status_code == 200
            client.post("/api/memory/append", json={"text": "runs an RX 6800"})
            text = client.get("/api/memory").json()["text"]
            assert "- likes terse answers" in text
            assert "- runs an RX 6800" in text     # "- " prefix added
            assert (tmp_path / ".localm" / "chat-memory.md").is_file()

            # clearing removes the file entirely
            client.put("/api/memory", json={"text": ""})
            assert not (tmp_path / ".localm" / "chat-memory.md").exists()

    def test_privacy_blocks_writes_allows_reads(self, persist_app, tmp_path,
                                                monkeypatch):
        home = tmp_path / ".localm"
        home.mkdir(parents=True, exist_ok=True)
        (home / "chat-memory.md").write_text("- earlier fact\n", encoding="utf-8")
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        app, _ = persist_app
        with TestClient(app) as client:
            data = client.get("/api/memory").json()
            assert data["text"].strip() == "- earlier fact"   # read OK
            assert data["writable"] is False
            assert client.put("/api/memory",
                              json={"text": "x"}).status_code == 403
            assert client.post("/api/memory/append",
                               json={"text": "x"}).status_code == 403

    def test_empty_append_rejected(self, persist_app, monkeypatch):
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = persist_app
        with TestClient(app) as client:
            assert client.post("/api/memory/append",
                               json={"text": "  "}).status_code == 400


# ------------------------------------------------------------------ #
#  Prompt library (/api/prompts)                                       #
# ------------------------------------------------------------------ #

class TestPromptLibrary:
    def test_crud_roundtrip(self, persist_app, tmp_path):
        app, _ = persist_app
        with TestClient(app) as client:
            assert client.get("/api/prompts").json() == {"prompts": []}
            r = client.put("/api/prompts/Code%20reviewer", json={
                "system": "You review code tersely.",
                "params": {"temperature": 0.3, "max_tokens": 512}})
            assert r.status_code == 200
            client.put("/api/prompts/Poet", json={"system": "Rhyme."})

            data = client.get("/api/prompts").json()["prompts"]
            assert [p["name"] for p in data] == ["Code reviewer", "Poet"]
            reviewer = data[0]
            assert reviewer["system"] == "You review code tersely."
            assert reviewer["params"]["temperature"] == 0.3
            assert (tmp_path / ".localm" / "prompts.json").is_file()

            # upsert replaces
            client.put("/api/prompts/Poet", json={"system": "Haiku only."})
            data = client.get("/api/prompts").json()["prompts"]
            assert next(p for p in data
                        if p["name"] == "Poet")["system"] == "Haiku only."

            assert client.delete("/api/prompts/Poet").status_code == 200
            assert client.delete("/api/prompts/Poet").status_code == 404
            assert [p["name"] for p in
                    client.get("/api/prompts").json()["prompts"]] == ["Code reviewer"]

    @pytest.mark.parametrize("bad", ["%20%20", "x" * 65, "a%0Ab"])
    def test_invalid_names_rejected(self, persist_app, bad):
        app, _ = persist_app
        with TestClient(app) as client:
            assert client.put(f"/api/prompts/{bad}",
                              json={"system": "x"}).status_code == 400


class TestChatPlugin:
    """Chat-as-plugin-#0 specifics: first-run auto-provisioning, the protected
    refusal of disable/uninstall, and scope gating. The persistence behaviour
    itself is covered above via the install-based persist_app fixture."""

    @staticmethod
    def _isolate(tmp_path, monkeypatch):
        home = tmp_path / ".localm"
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", home)
        monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
        return home

    def test_auto_provisions_and_is_active(self, tmp_path, monkeypatch):
        """On first run the engine copies chat from the store into the installed
        dir and enables it (preinstalled + default_enabled) with no explicit
        install() call - so chat ships active out of the box."""
        self._isolate(tmp_path, monkeypatch)
        from localm.plugins.engine import PluginManager
        app = FastAPI()
        mgr = PluginManager(app, external_root=tmp_path / "noplugins")
        mgr.load_enabled()                       # first run
        assert mgr.is_active("chat")
        assert (tmp_path / "noplugins" / "chat" / "plugin.toml").is_file()
        with TestClient(app) as client:          # routes mounted by the engine
            monkeypatch.setenv("LOCALM_MODE", "privacy")
            r = client.get("/api/conversations")
            assert r.status_code == 200 and r.json()["enabled"] is False

    def test_protected_refuses_disable_and_uninstall(self, tmp_path, monkeypatch):
        self._isolate(tmp_path, monkeypatch)
        from localm.plugins.engine import PluginManager
        mgr = PluginManager(FastAPI(), external_root=tmp_path / "noplugins")
        mgr.load_enabled()
        assert mgr.is_active("chat")
        for op in ("disable", "uninstall"):
            with pytest.raises(ValueError):
                getattr(mgr, op)("chat")

    def test_routes_require_chat_scope(self, persist_app, monkeypatch):
        """Chat persistence routes are auto-scoped to the 'chat' capability."""
        from localm import auth, scopes as S
        app, _ = persist_app
        made = auth.create_key("reader", [S.MODELS_READ])      # lacks 'chat'
        with TestClient(app) as client:
            denied = client.get(
                "/api/conversations",
                headers={"Authorization": f"Bearer {made['key']}"})
            assert denied.status_code == 403
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")  # owner = admin
            ok = client.get(
                "/api/conversations",
                headers={"Authorization": "Bearer ownersecret"})
            assert ok.status_code == 200


# ------------------------------------------------------------------ #
#  Knowledge endpoints (/api/rag/*)                                    #
# ------------------------------------------------------------------ #

@pytest.fixture
def rag_app(tmp_path, monkeypatch):
    """GUI app (for the shared job manager + /api/jobs SSE) with the builtin rag
    plugin enabled. rag became a plugin in Phase 3; its handlers read the shared
    services (jobs / self_url / active_model) that attach_gui puts on app.state.
    The plugin is enabled BEFORE attach_gui so its routes sit ahead of the
    catch-all "/" static mount - the production order (plugins, then GUI)."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")

    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app, home


class TestRagEndpoints:
    """rag_app enables the rag plugin; collections land under tmp/.localm/rag."""

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

    def test_create_list_detail_delete(self, rag_app):
        app, _ = rag_app
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

    def test_add_and_query_roundtrip(self, rag_app, tmp_path):
        app, _ = rag_app
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

    def test_extract_endpoint(self, rag_app):
        import base64
        app, _ = rag_app
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

    def test_extract_writes_nothing_to_disk(self, rag_app, tmp_path):
        """Privacy guarantee: attachment extraction is in-memory only."""
        import base64
        app, _ = rag_app
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

    def test_history_lists_and_reads_logs(self, coder_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        sessions_dir = tmp_path / "sessions"
        log, entry = self._fake_log(sessions_dir)
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", sessions_dir)
        monkeypatch.setenv("LOCALM_MODE", "log")
        app, _ = coder_app
        with TestClient(app) as client:
            data = client.get("/api/coder/history").json()
            assert data["enabled"] is True
            assert [l["name"] for l in data["logs"]] == [log.name]
            parsed = client.get(f"/api/coder/history/{log.name}").json()
        assert parsed["entries"] == [entry]     # malformed line skipped

    def test_history_enabled_false_in_privacy(self, coder_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", tmp_path / "none")
        monkeypatch.setenv("LOCALM_MODE", "privacy")
        app, _ = coder_app
        with TestClient(app) as client:
            data = client.get("/api/coder/history").json()
        assert data == {"enabled": False, "logs": []}

    def test_history_rejects_bad_names(self, coder_app, tmp_path, monkeypatch):
        import localm.audit as audit_mod
        sessions_dir = tmp_path / "sessions"
        self._fake_log(sessions_dir)
        # plant a sibling the traversal would reach
        (tmp_path / "secret.jsonl").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(audit_mod, "_SESSIONS_DIR", sessions_dir)
        app, _ = coder_app
        with TestClient(app) as client:
            assert client.get("/api/coder/history/notes.txt").status_code == 400
            r = client.get("/api/coder/history/..%5Csecret.jsonl")
            assert r.status_code in (400, 404)


class TestCoderPlugin:
    """Coder-as-plugin specifics: 503 without the GUI, scope gating on the
    high-privilege routes, the CLI is_active gate, and the severed kernel
    coupling. Route behaviour itself is covered above via the install-based
    coder_app fixture."""

    @staticmethod
    def _isolate(tmp_path, monkeypatch):
        home = tmp_path / ".localm"
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", home)
        monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
        return home

    def test_coder_without_gui_is_503(self, tmp_path, monkeypatch):
        """The engine mounts /api/coder routes even with no GUI; they must 503
        (not 500) when attach_gui never published the session manager."""
        self._isolate(tmp_path, monkeypatch)
        from localm.plugins.engine import PluginManager
        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("coder")
        with TestClient(app) as client:        # NOTE: no attach_gui
            r = client.post("/api/coder/sessions", json={"cwd": str(tmp_path)})
            assert r.status_code == 503

    def test_routes_require_coder_scope(self, coder_app, monkeypatch):
        """The agentic coder is shell-exec + file-write; the engine gates every
        /api/coder route on the 'coder' capability scope."""
        from localm import auth, scopes as S
        app, _ = coder_app
        made = auth.create_key("reader", [S.CHAT])              # lacks 'coder'
        with TestClient(app) as client:
            denied = client.get(
                "/api/coder/sessions",
                headers={"Authorization": f"Bearer {made['key']}"})
            assert denied.status_code == 403
            monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")  # owner = admin
            ok = client.get(
                "/api/coder/sessions",
                headers={"Authorization": "Bearer ownersecret"})
            assert ok.status_code == 200

    def test_cli_gated_when_inactive(self, tmp_path, monkeypatch):
        """`localm coder` / `localcoder` refuse cleanly until the plugin is
        installed+enabled (mirrors the mcp server gate)."""
        self._isolate(tmp_path, monkeypatch)
        from click.testing import CliRunner
        from localm.plugins.coder.cli import main
        result = CliRunner().invoke(main, ["noop"])
        assert result.exit_code == 1
        assert "not active" in result.output.lower()
        assert "localm plugin install coder" in result.output

    def test_readline_privacy_util_does_not_import_coder(self):
        """The kernel chat REPL suppresses readline history via the kernel
        module, so importing it never drags the coder plugin in."""
        import subprocess as _sp
        import sys as _sys
        code = (
            "import importlib, sys\n"
            "importlib.import_module('localm.readline_privacy')\n"
            "bad = [m for m in sys.modules if m.startswith('localm.plugins.coder')]\n"
            "assert not bad, bad\n"
        )
        r = _sp.run([_sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


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
    """GUI app with the builtin image plugin enabled; images dir under tmp.
    image became a plugin in Phase 3 - its routes are mounted by enabling it
    (before attach_gui, the production order), not by attach_gui."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("image")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    images = home / "gui_images"
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


class TestImageGeneration:
    """The /api/imagine generation route, with the ComfyUI backend mocked at the
    shared-plumbing level (localm.image_gen.comfy). Exercises the image plugin's
    backend wiring + the background job -> result path."""

    @staticmethod
    def _wait_job(client, job_id, timeout=30):
        import time as _time
        end = None
        deadline = _time.monotonic() + timeout
        with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for raw in r.iter_lines():
                if _time.monotonic() > deadline:
                    break
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                if ev["type"] == "end":
                    end = ev
                    break
        return end

    def test_empty_prompt_is_400(self, img_app):
        app, _ = img_app
        with TestClient(app) as client:
            assert client.post("/api/imagine", json={"prompt": "   "}).status_code == 400

    def test_missing_input_image_is_400(self, img_app):
        app, _ = img_app
        with TestClient(app) as client:
            r = client.post("/api/imagine",
                            json={"prompt": "x", "input_image": "Z:/nope.png"})
        assert r.status_code == 400

    def test_imagine_without_gui_jobs_is_503(self, tmp_path, monkeypatch):
        """Image gen needs the GUI's job manager (app.state.jobs). With the plugin
        enabled on a bare app (no attach_gui), the route returns a clear 503 rather
        than a 500."""
        home = tmp_path / ".localm"
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", home)
        monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.plugins.engine import PluginManager
        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("image")
        with TestClient(app) as client:
            r = client.post("/api/imagine", json={"prompt": "a fox"})
        assert r.status_code == 503

    def test_imagine_job_generates_and_returns_result(self, img_app, monkeypatch):
        app, images = img_app
        import localm.image_gen.comfy as comfy
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "ComfyUI is running."))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)

        def fake_gen(prompt, out_path, **kw):
            Path(out_path).write_bytes(b"\x89PNG fake")
            return True, f"Image saved to {out_path}"
        monkeypatch.setattr(comfy, "generate_image", fake_gen)

        with TestClient(app) as client:
            r = client.post("/api/imagine", json={"prompt": "a fox in snow"})
            assert r.status_code == 200
            end = self._wait_job(client, r.json()["job_id"])
        assert end and end["status"] == "done"
        assert end.get("result", "").endswith(".png")
        assert list(images.glob("*.png"))      # the image landed in the plugin dir

    def test_backend_uses_per_plugin_api_url(self, img_app, monkeypatch):
        """A per-plugin comfy.api_url overrides the default and is what the
        shared plumbing is called with."""
        app, _ = img_app
        from localm.config import load_config, save_config
        cfg = load_config()
        cfg.setdefault("plugins", {})["image"] = {
            "backend": "comfy", "comfy": {"api_url": "http://127.0.0.1:9999"}}
        save_config(cfg)

        import localm.image_gen.comfy as comfy
        seen = {}
        monkeypatch.setattr(comfy, "ensure_comfy",
                            lambda api_url=None, **k: (seen.update(url=api_url), (True, "up"))[1])
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)
        monkeypatch.setattr(comfy, "generate_image",
                            lambda prompt, out_path, **kw: (Path(out_path).write_bytes(b"x"),
                                                            (True, "ok"))[1])
        with TestClient(app) as client:
            r = client.post("/api/imagine", json={"prompt": "x"})
            self._wait_job(client, r.json()["job_id"])
        assert seen.get("url") == "http://127.0.0.1:9999"


# ------------------------------------------------------------------ #
#  Music plugin (/api/music*)                                          #
# ------------------------------------------------------------------ #

@pytest.fixture
def music_app(tmp_path, monkeypatch):
    """GUI app with the builtin music plugin enabled; music dir under tmp.
    music became a plugin in Phase 3 - enabled before attach_gui (production
    order), reading its own per-plugin backend config."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("music")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    tracks = home / "gui_music"
    tracks.mkdir(parents=True)
    return app, tracks


class TestMusicPlugin:
    @staticmethod
    def _wait_job(client, job_id, timeout=30):
        import time as _time
        end = None
        deadline = _time.monotonic() + timeout
        with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for raw in r.iter_lines():
                if _time.monotonic() > deadline:
                    break
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                if ev["type"] == "end":
                    end = ev
                    break
        return end

    def test_empty_tags_is_400(self, music_app):
        app, _ = music_app
        with TestClient(app) as client:
            assert client.post("/api/music", json={"tags": "  "}).status_code == 400

    def test_bad_duration_is_400(self, music_app):
        app, _ = music_app
        with TestClient(app) as client:
            assert client.post("/api/music",
                               json={"tags": "lofi", "duration_seconds": 0}).status_code == 400
            assert client.post("/api/music",
                               json={"tags": "lofi", "duration_seconds": 99999}).status_code == 400

    def test_music_job_generates_and_returns_result(self, music_app, monkeypatch):
        app, tracks = music_app
        import localm.image_gen.comfy as comfy
        import localm.music_gen as music_gen
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "up"))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)

        def fake_gen(tags, out_path, **kw):
            Path(out_path).write_bytes(b"FLACfake")
            return True, f"Track saved to {out_path}"
        monkeypatch.setattr(music_gen, "generate_music", fake_gen)

        with TestClient(app) as client:
            r = client.post("/api/music", json={"tags": "synthwave", "duration_seconds": 30})
            assert r.status_code == 200
            end = self._wait_job(client, r.json()["job_id"])
        assert end and end["status"] == "done"
        assert end.get("result", "").endswith(".flac")
        assert list(tracks.glob("*.flac"))

    def test_history_delete_move(self, music_app, tmp_path):
        app, tracks = music_app
        (tracks / "t.flac").write_bytes(b"FLACfake")
        (tracks / "t.flac.json").write_text(json.dumps({"tags": "lofi"}))
        with TestClient(app) as client:
            data = client.get("/api/music/history").json()
            assert data["tracks"][0]["name"] == "t.flac"
            dest = tmp_path / "kept"
            r = client.post("/api/music/file/t.flac/move", json={"dest": str(dest)})
            assert r.status_code == 200
            assert (dest / "t.flac").is_file() and (dest / "t.flac.json").is_file()
            assert client.delete("/api/music/file/nope.flac").status_code == 404

    def test_music_without_gui_jobs_is_503(self, tmp_path, monkeypatch):
        home = tmp_path / ".localm"
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", home)
        monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.plugins.engine import PluginManager
        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("music")
        with TestClient(app) as client:
            r = client.post("/api/music", json={"tags": "lofi"})
        assert r.status_code == 503


# ------------------------------------------------------------------ #
#  Video generation endpoints                                          #
# ------------------------------------------------------------------ #

@pytest.fixture
def video_app(tmp_path, monkeypatch):
    """GUI app with the builtin video plugin enabled; video dir under tmp.
    video became a plugin in Phase 3 - enabled before attach_gui (production
    order), reading its own per-plugin backend config."""
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("video")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    videos = home / "gui_video"
    videos.mkdir(parents=True)
    return app, videos


class TestVideoEndpoints:
    @staticmethod
    def _make_clip(videos, name="clip.mp4", meta=None):
        (videos / name).write_bytes(b"fake mp4")
        if meta is not None:
            (videos / (name + ".json")).write_text(json.dumps(meta))

    @staticmethod
    def _wait_job(client, job_id, timeout=30):
        import time as _time
        end = None
        deadline = _time.monotonic() + timeout
        with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
            for raw in r.iter_lines():
                if _time.monotonic() > deadline:
                    break
                if not raw.startswith("data: "):
                    continue
                ev = json.loads(raw[6:])
                if ev["type"] == "end":
                    end = ev
                    break
        return end

    def test_empty_prompt_rejected(self, video_app):
        app, _ = video_app
        with TestClient(app) as client:
            r = client.post("/api/video", json={"prompt": "   "})
        assert r.status_code == 400

    @pytest.mark.parametrize("seconds", [0, -1, 21, 3600])
    def test_bad_duration_rejected(self, video_app, seconds):
        app, _ = video_app
        with TestClient(app) as client:
            r = client.post("/api/video",
                            json={"prompt": "a fox", "seconds": seconds})
        assert r.status_code == 400

    def test_bad_fps_rejected(self, video_app):
        app, _ = video_app
        with TestClient(app) as client:
            r = client.post("/api/video", json={"prompt": "a fox", "fps": 0})
        assert r.status_code == 400

    def test_missing_input_image_rejected(self, video_app, tmp_path):
        app, _ = video_app
        with TestClient(app) as client:
            r = client.post("/api/video", json={
                "prompt": "a fox",
                "input_image": str(tmp_path / "nope.png")})
        assert r.status_code == 400

    def test_history_lists_clips_with_meta(self, video_app):
        app, videos = video_app
        self._make_clip(videos, meta={"prompt": "a fox", "seconds": 5.0})
        with TestClient(app) as client:
            data = client.get("/api/video/history").json()
        assert data["videos"][0]["name"] == "clip.mp4"
        assert data["videos"][0]["meta"]["prompt"] == "a fox"
        assert data["videos"][0]["size_bytes"] > 0

    def test_history_empty(self, video_app):
        app, _ = video_app
        with TestClient(app) as client:
            assert client.get("/api/video/history").json() == {"videos": []}

    def test_serve_and_delete_clip(self, video_app):
        app, videos = video_app
        self._make_clip(videos, meta={"prompt": "x"})
        with TestClient(app) as client:
            r = client.get("/api/video/file/clip.mp4")
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("video/mp4")
            assert client.delete("/api/video/file/clip.mp4").status_code == 200
        assert not (videos / "clip.mp4").exists()
        assert not (videos / "clip.mp4.json").exists()

    @pytest.mark.parametrize("name", [
        "..%5Cconfig.json",       # ..\config.json
        "C:evil.mp4",             # Windows drive-relative
        "sub%2Ffile.mp4",         # nested subpath
    ])
    def test_confinement(self, video_app, tmp_path, name):
        app, _ = video_app
        target = tmp_path / ".localm" / "config.json"
        target.write_text("{}")
        with TestClient(app) as client:
            r = client.delete(f"/api/video/file/{name}")
        assert not (200 <= r.status_code < 300)
        assert target.exists()

    def test_move_relocates_clip_and_sidecar(self, video_app, tmp_path):
        app, videos = video_app
        self._make_clip(videos, meta={"prompt": "x"})
        dest = tmp_path / "kept"
        with TestClient(app) as client:
            r = client.post("/api/video/file/clip.mp4/move",
                            json={"dest": str(dest)})
        assert r.status_code == 200
        assert (dest / "clip.mp4").is_file()
        assert (dest / "clip.mp4.json").is_file()
        assert not (videos / "clip.mp4").exists()

    def test_video_job_generates_and_returns_result(self, video_app, monkeypatch):
        app, videos = video_app
        import localm.image_gen.comfy as comfy
        import localm.video_gen as video_gen
        monkeypatch.setattr(comfy, "ensure_comfy", lambda *a, **k: (True, "up"))
        monkeypatch.setattr(comfy, "free_comfy_vram", lambda *a, **k: False)

        def fake_gen(prompt, out_path, **kw):
            Path(out_path).write_bytes(b"fake mp4")
            return True, f"Clip saved to {out_path}"
        monkeypatch.setattr(video_gen, "generate_video", fake_gen)

        with TestClient(app) as client:
            r = client.post("/api/video", json={"prompt": "a fox runs", "seconds": 5})
            assert r.status_code == 200
            end = self._wait_job(client, r.json()["job_id"])
        assert end and end["status"] == "done"
        assert end.get("result", "").endswith(".mp4")
        assert list(videos.glob("*.mp4"))

    def test_video_without_gui_jobs_is_503(self, tmp_path, monkeypatch):
        home = tmp_path / ".localm"
        monkeypatch.setenv("LOCALM_HOME", str(home))
        monkeypatch.delenv("LOCALM_API_KEY", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "HOME_DIR", home)
        monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
        monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
        from localm.plugins.engine import PluginManager
        app = FastAPI()
        PluginManager(app, external_root=tmp_path / "noplugins").install("video")
        with TestClient(app) as client:
            r = client.post("/api/video", json={"prompt": "a fox"})
        assert r.status_code == 503
