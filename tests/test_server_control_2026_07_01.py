# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the 2026-07-01 server-control backlog cluster.

  NEW-G            - restart re-exec marks fds non-inheritable (no fd 3/4 leak)
  REC-OPEN-GET-GATE - a keyless NETWORK bind gates GET data routes on the shell token

(SRV-CTRLC is deferred: it needs a portmux/serve refactor and a live Windows
Ctrl+C to verify; the /v1/server/shutdown route is the working stopgap.)
"""

import os

from fastapi.testclient import TestClient

from localm.inference import http_server


# --------------------------------------------------------------------------- #
#  NEW-G - fds are marked non-inheritable before os.execv
# --------------------------------------------------------------------------- #

def test_do_restart_marks_fds_non_inheritable(monkeypatch):
    calls = []

    class _Stop(Exception):
        pass

    monkeypatch.setattr(os, "set_inheritable", lambda fd, inh: calls.append((fd, inh)))

    def _fake_execv(exe, argv):
        raise _Stop()

    monkeypatch.setattr(os, "execv", _fake_execv)
    monkeypatch.setattr(http_server, "_restart_argv", lambda: ["python", "-m", "localm"])
    monkeypatch.setattr(http_server, "_engine", None)

    try:
        http_server._do_restart()
    except _Stop:
        pass

    assert calls, "no fds were marked non-inheritable before execv"
    assert all(inh is False for _, inh in calls), "fds must be marked NON-inheritable"
    assert all(fd >= 3 for fd, _ in calls), "stdin/stdout/stderr (0-2) must be left alone"


# --------------------------------------------------------------------------- #
#  REC-OPEN-GET-GATE - network bind + keyless gates GET data routes
# --------------------------------------------------------------------------- #

def _keyless_app(tmp_path, monkeypatch, bind_host):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    app = http_server.create_app(None)
    app.state.bind_host = bind_host
    return app


def test_network_bind_open_mode_get_data_route_gated(tmp_path, monkeypatch):
    app = _keyless_app(tmp_path, monkeypatch, bind_host="0.0.0.0")
    client = TestClient(app)
    # A data GET with no shell token is refused on a network bind ...
    assert client.get("/v1/config").status_code == 403
    # ... but the same GET with the loopback shell token passes the gate.
    ok = client.get("/v1/config",
                    headers={"Authorization": f"Bearer {app.state.shell_token}"})
    assert ok.status_code != 403


def test_network_bind_open_mode_health_stays_public(tmp_path, monkeypatch):
    app = _keyless_app(tmp_path, monkeypatch, bind_host="0.0.0.0")
    client = TestClient(app)
    assert client.get("/health").status_code in (200, 503)        # never gated
    # the OpenAI-compatible inference API stays network-callable by design
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code != 403


def test_loopback_bind_open_mode_get_unchanged(tmp_path, monkeypatch):
    app = _keyless_app(tmp_path, monkeypatch, bind_host="127.0.0.1")
    client = TestClient(app)
    # Loopback (the default local model) is completely unaffected by the gate.
    assert client.get("/v1/config").status_code != 403
