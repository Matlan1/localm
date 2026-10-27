# SPDX-License-Identifier: AGPL-3.0-or-later
"""POST /api/gui-error: a GUI error reaches the server log (the recent-activity
buffer behind /api/activity) as one sanitized, scrubbed, rate-limited line."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from localm.plugins.gui.routes import client_errors as ce


@pytest.fixture
def fresh_ring():
    """A clean ring buffer on the localm logger for one test; the logger's
    handlers, level and ring global are restored afterwards."""
    from localm import debuglog
    logger = debuglog.logger
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_ring = debuglog._ring_handler
    for h in list(logger.handlers):
        if isinstance(h, debuglog._RingBufferHandler):
            logger.removeHandler(h)
    debuglog._ring_handler = None
    assert debuglog.install_ring_buffer() is True
    yield
    logger.handlers[:] = saved_handlers
    logger.setLevel(saved_level)
    debuglog._ring_handler = saved_ring


@pytest.fixture
def app(tmp_path, monkeypatch, fresh_ring):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.inference.http_server import create_app
    from localm.plugins.gui.web import attach_gui
    a = create_app(None)
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda n: None, active_model=lambda: "model-a")
    return a


def _shell(app):
    return {"Authorization": f"Bearer {app.state.shell_token}"}


def _gui_lines():
    from localm.debuglog import recent_activity
    return [ln for ln in recent_activity() if " localm: gui: " in ln]


def test_a_report_is_logged_at_warning_with_the_gui_prefix(app):
    c = TestClient(app)
    r = c.post("/api/gui-error", headers=_shell(app),
               json={"message": "Could not load model x", "source": "toast",
                     "page": "#models", "time": "2026-10-27T10:00:00Z"})
    assert r.status_code == 200 and r.json() == {"logged": True}
    lines = _gui_lines()
    assert len(lines) == 1
    assert "WARNING" in lines[0]
    assert lines[0].endswith("gui: [toast #models] Could not load model x")
    assert "2026-10-27T10:00:00Z" not in lines[0]


def test_crlf_ansi_control_and_bidi_characters_are_removed(app):
    c = TestClient(app)
    msg = ("boom\r\n2026-01-01 00:00:00 ERROR   localm: forged line"
           "\x1b[31mred\x1b[0m\x1b]0;title\x07 tab\there \x00nul sep‮RTL\x85")
    assert c.post("/api/gui-error", headers=_shell(app), json={"message": msg}).status_code == 200
    (line,) = _gui_lines()
    assert "\n" not in line and "\r" not in line
    assert all(ch not in line for ch in "\x1b\x00\t ‮\x85")
    assert "red" in line and "title" not in line
    assert "boom 2026-01-01 00:00:00 ERROR localm: forged line" in line


def test_message_source_and_page_are_capped(app):
    c = TestClient(app)
    r = c.post("/api/gui-error", headers=_shell(app),
               json={"message": "m" * 5000, "source": "s" * 500, "page": "p" * 500})
    assert r.status_code == 200
    (line,) = _gui_lines()
    body = line.split(" localm: gui: ", 1)[1]
    msg = body.split("] ", 1)[1]
    assert len(msg) == ce.MAX_MESSAGE_CHARS and msg.endswith("...")
    src, page = body[1:].split("] ", 1)[0].split(" ")
    assert len(src) == ce.MAX_SOURCE_CHARS and len(page) == ce.MAX_PAGE_CHARS


def test_a_body_over_the_cap_is_refused_and_not_logged(app):
    c = TestClient(app)
    big = '{"message": "' + "x" * (ce.MAX_BODY_BYTES + 10) + '"}'
    r = c.post("/api/gui-error", headers={**_shell(app), "Content-Type": "application/json"},
               content=big)
    assert r.status_code == 413
    assert _gui_lines() == []


@pytest.mark.parametrize("body", ["not json", "[1, 2]", '{"message": ""}',
                                  '{"message": 42}', '{"message": "\\u0000\\r\\n"}', ""])
def test_a_malformed_or_empty_report_is_refused(app, body):
    c = TestClient(app)
    r = c.post("/api/gui-error", headers={**_shell(app), "Content-Type": "application/json"},
               content=body)
    assert r.status_code == 400
    assert _gui_lines() == []


def test_home_paths_and_credentials_are_scrubbed(app, tmp_path):
    c = TestClient(app)
    home = str(Path.home())
    msg = (f"failed to open {home}/models/secret-name.gguf with "
           "Authorization: Bearer abcdefghijklmnop1234 and localm-sk-ABCDEFGHIJKLMNOPQR")
    assert c.post("/api/gui-error", headers=_shell(app), json={"message": msg}).status_code == 200
    (line,) = _gui_lines()
    assert home not in line
    assert "abcdefghijklmnop1234" not in line
    assert "ABCDEFGHIJKLMNOPQR" not in line
    assert "secret-name.gguf" in line


def test_a_scrubber_failure_logs_no_text(app, monkeypatch):
    from localm.bugreport import scrub

    def _boom(text):
        raise RuntimeError("scrubber down")

    monkeypatch.setattr(scrub, "_scrub_secrets", _boom)
    c = TestClient(app)
    r = c.post("/api/gui-error", headers=_shell(app),
               json={"message": "token " + "sk-" + "SHOULDNOTAPPEAR123456"})
    assert r.status_code == 200 and r.json() == {"logged": False}
    lines = _gui_lines()
    assert len(lines) == 1 and "scrubber failed" in lines[0]
    assert "SHOULDNOTAPPEAR" not in "\n".join(lines)


def test_rate_limit_drops_extra_reports_and_counts_them(app, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(ce.time, "monotonic", lambda: clock[0])
    c = TestClient(app)
    for i in range(ce.RATE_LIMIT_COUNT):
        assert c.post("/api/gui-error", headers=_shell(app),
                      json={"message": f"e{i}"}).status_code == 200
    for i in range(5):
        r = c.post("/api/gui-error", headers=_shell(app), json={"message": f"over{i}"})
        assert r.status_code == 429 and r.json() == {"logged": False}
    lines = _gui_lines()
    assert len(lines) == ce.RATE_LIMIT_COUNT + 1
    assert "dropping further reports" in lines[-1]
    assert not any("over" in ln for ln in lines)
    clock[0] += ce.RATE_LIMIT_WINDOW_S
    assert c.post("/api/gui-error", headers=_shell(app),
                  json={"message": "after"}).status_code == 200
    tail = _gui_lines()[-2:]
    assert "5 GUI error report(s) were dropped" in tail[0]
    assert tail[1].endswith("gui: [gui] after")


def test_rate_limit_is_per_credential():
    lim = ce._RateLimiter(2, 60.0, 8)
    assert lim.admit("a", 0)[0] and lim.admit("a", 0)[0]
    assert lim.admit("a", 0)[0] is False
    assert lim.admit("b", 0)[0] is True


def test_rate_limiter_tracks_a_bounded_number_of_clients():
    lim = ce._RateLimiter(1, 60.0, 3)
    for k in "abcdef":
        lim.admit(k, 0)
    assert len(lim._state) == 3


def test_open_mode_without_the_shell_token_is_refused(app):
    c = TestClient(app)
    assert c.post("/api/gui-error", json={"message": "x"}).status_code == 403
    assert _gui_lines() == []


def test_a_cross_origin_post_is_refused(app):
    c = TestClient(app)
    r = c.post("/api/gui-error", headers={**_shell(app), "Origin": "http://evil.example"},
               json={"message": "x"})
    assert r.status_code == 403
    assert _gui_lines() == []


def test_a_keyed_server_refuses_a_request_without_a_key(app, monkeypatch):
    monkeypatch.setenv("LOCALM_API_KEY", "ownersecret")
    c = TestClient(app)
    assert c.post("/api/gui-error", json={"message": "x"}).status_code == 401
    assert c.post("/api/gui-error", headers={"Authorization": "Bearer wrong"},
                  json={"message": "x"}).status_code == 401
    assert _gui_lines() == []
    r = c.post("/api/gui-error", headers={"Authorization": "Bearer ownersecret"},
               json={"message": "keyed ok"})
    assert r.status_code == 200
    assert _gui_lines()[-1].endswith("keyed ok")



def test_the_record_reaches_the_status_window_log_source(app):
    """The status window's live log and "View logs" read debuglog.recent_activity
    (localm/plugins/gui/cli.py _start_app_face)."""
    from localm.plugins.gui import cli
    import inspect
    assert "get_log_lines=debuglog.recent_activity" in inspect.getsource(cli._start_app_face)
    c = TestClient(app)
    c.post("/api/gui-error", headers=_shell(app), json={"message": "visible in the window"})
    from localm.debuglog import recent_activity
    assert any(ln.endswith("gui: [gui] visible in the window") for ln in recent_activity())
