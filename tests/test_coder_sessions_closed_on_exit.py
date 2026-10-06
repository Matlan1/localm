# SPDX-License-Identifier: AGPL-3.0-or-later
"""A server stop or restart closes every live GUI coder session.

``_do_shutdown`` ends at ``os._exit(0)`` and ``_do_restart`` at ``os.execv``.
Each live session still records its end there: its audit log ends with the
"session ended" record and its conversation is saved, and a session whose close
blocks or fails does not keep the stop from completing."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.inference import http_server as hs
from localm.plugins.coder import sessions as coder_sessions

_OWNER = {"Authorization": "Bearer ownersecret"}


def _coder_app(tmp_path: Path, monkeypatch) -> tuple[FastAPI, Path]:
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    import localm.audit as audit
    monkeypatch.setattr(audit, "_SESSIONS_DIR", home / "sessions")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app, home


def _open_session(client: TestClient, app: FastAPI, proj: Path):
    r = client.post("/api/coder/sessions", headers=_OWNER,
                    json={"cwd": str(proj), "mode": "log"})
    assert r.status_code == 200, r.text
    sess = app.state.coder_sessions.get(r.json()["id"])
    sess.agent._messages = [{"role": "user", "content": "marker task"}]
    sess.agent._turns = 1
    return sess


def _audit_records(home: Path, checkpoint_id: str) -> list[str]:
    logs = list((home / "sessions").glob(f"*_{checkpoint_id}_*.jsonl"))
    assert len(logs) == 1, f"expected one audit log for {checkpoint_id}: {logs}"
    return logs[0].read_text(encoding="utf-8").splitlines()


def _raise_exit(*_args):
    raise SystemExit(0)


@pytest.mark.parametrize("path", ["shutdown", "restart"])
def test_the_exit_paths_close_a_live_gui_coder_session(tmp_path, monkeypatch, path):
    app, home = _coder_app(tmp_path, monkeypatch)
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(os, "_exit", _raise_exit)
    monkeypatch.setattr(os, "execv", _raise_exit)

    with TestClient(app) as client:
        sess = _open_session(client, app, proj)
        checkpoint_id = sess.checkpoint_id
        exited = None
        try:
            if path == "shutdown":
                hs._do_shutdown()
            else:
                hs._do_restart()
        except SystemExit as e:
            exited = e

        records = _audit_records(home, checkpoint_id)
        assert '"session ended"' in records[-1], (
            f"the {path} path left the live coder session open: its audit log "
            f"never recorded its end (last record: {records[-1]})")
        saved = (home / "checkpoints").rglob(f"{checkpoint_id}.json")
        assert any("marker task" in p.read_text(encoding="utf-8") for p in saved)
        assert sess.closed
        assert exited is not None, f"_do_{path} did not reach its exit call"


def test_a_session_that_will_not_close_does_not_hang_the_stop(monkeypatch):
    release = threading.Event()

    class _Stuck:
        id = "stuck"

        def close(self):
            release.wait(60)

    manager = coder_sessions.SessionManager()
    manager._sessions["stuck"] = _Stuck()
    monkeypatch.setattr(coder_sessions, "_EXIT_CLOSE_BUDGET_S", 0.3)
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(os, "_exit", _raise_exit)

    started = time.monotonic()
    exited = None
    try:
        hs._do_shutdown()
    except SystemExit as e:
        exited = e
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert elapsed < 20, f"a session whose close blocks held the stop for {elapsed:.1f}s"
    assert exited is not None, "_do_shutdown did not reach its exit call"


def test_a_failing_close_does_not_keep_the_other_sessions_open():
    class _Boom:
        id = "boom"

        def close(self):
            raise RuntimeError("boom")

    class _Recording:
        id = "recording"
        closed = False

        def close(self):
            self.closed = True

    recording = _Recording()
    manager = coder_sessions.SessionManager()
    manager._sessions["boom"] = _Boom()
    manager._sessions["recording"] = recording

    closed = coder_sessions.close_all_for_exit(timeout_s=10)

    assert recording.closed, "a session after one whose close raised stayed open"
    assert closed >= 1
    assert manager.snapshot() == []


def test_a_failure_closing_sessions_does_not_block_the_stop(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(coder_sessions, "close_all_for_exit", _boom)
    monkeypatch.setattr(hs, "_engine", None)
    monkeypatch.setattr(os, "_exit", _raise_exit)

    exited = None
    try:
        hs._do_shutdown()
    except SystemExit as e:
        exited = e
    assert exited is not None, "a failure closing coder sessions blocked the stop"
