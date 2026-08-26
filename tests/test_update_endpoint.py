# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server endpoints for the issues tracker + updater: /api/issues,
/api/update/check (read-only), /api/update/apply (mutates -> CONFIG_WRITE). Apply is
explicit-only; failures roll back and are surfaced, never faked. All proxy calls are
monkeypatched (no network)."""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from localm import issue_tracker, updater
from localm.inference.http_server import create_app


def _engine():
    e = MagicMock()
    e.display_name = "m"
    e.loaded = True
    return e


def _get(app, path):
    with TestClient(app) as c:
        return c.get(path, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def _post(app, path):
    with TestClient(app) as c:
        return c.post(path, headers={"Authorization": f"Bearer {app.state.shell_token}"})


def _open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


# ------------------------------ issues ----------------------------------

def test_issues_endpoint_lists(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "list_issues",
                        lambda state="all": [{"number": 5, "state": "open", "title": "t"}])
    r = _get(create_app(_engine()), "/api/issues")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] and data["issues"][0]["number"] == 5


def test_issues_endpoint_unconfigured(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(issue_tracker, "available", lambda: False)
    r = _get(create_app(_engine()), "/api/issues")
    assert r.json()["available"] is False


def test_issues_endpoint_error_surfaced(monkeypatch):
    _open_mode(monkeypatch)
    from localm.bugreport import LocalmError
    monkeypatch.setattr(issue_tracker, "available", lambda: True)

    def boom(state="all"):
        raise LocalmError("could not reach the localm proxy", reason="timeout")

    monkeypatch.setattr(issue_tracker, "list_issues", boom)
    data = _get(create_app(_engine()), "/api/issues").json()
    assert "error" in data and "could not reach" in data["error"]


# --------------------------- update check -------------------------------

def test_update_check_endpoint(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "n", "asset": {"id": 3}})
    data = _get(create_app(_engine()), "/api/update/check").json()
    assert data["available"] and data["newer"] and data["latest"] == "v0.2.0"


def test_update_check_unconfigured(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "available", lambda: False)
    assert _get(create_app(_engine()), "/api/update/check").json()["available"] is False


def test_update_check_blocked_by_net_policy_is_not_up_to_date(monkeypatch):
    """A net-policy refusal goes through the route's error path, not through the
    success shape with newer=False (which the GUI reads as "up to date")."""
    _open_mode(monkeypatch)
    from localm.bugreport import LocalmError
    monkeypatch.setattr(updater, "available", lambda: True)

    def boom():
        raise LocalmError(
            "update checks are off because network access is set to off",
            reason='turn on "Check for updates even when network access is off" '
                   "in Settings, or set network access to ask or allow")

    monkeypatch.setattr(updater, "check", boom)
    data = _get(create_app(_engine()), "/api/update/check").json()
    assert data["available"] is True
    assert "network access is set to off" in data["error"]
    assert "newer" not in data


# --------------------------- update apply -------------------------------

def test_update_apply_endpoint(monkeypatch):
    _open_mode(monkeypatch)
    import localm.inference.http_server as hs
    monkeypatch.setattr(hs, "_request_restart", lambda *a, **k: None)  # do not really restart
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "notes": "", "asset": {"id": 3}})
    applied = {}

    def fake_apply(aid, **kw):
        applied["aid"] = aid
        return {"applied": True, "version": "0.2.0", "klass": "reboot", "backup": "b"}

    monkeypatch.setattr(updater, "apply", fake_apply)
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is True and applied["aid"] == 3 and data.get("restarting") is True


def test_update_apply_already_current(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.2.0", "latest": "0.2.0", "newer": False, "asset": None})
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is False


def test_update_apply_failure_is_surfaced(monkeypatch):
    _open_mode(monkeypatch)
    from localm.bugreport import LocalmError
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "asset": {"id": 3}})

    def boom(aid, **kw):
        raise LocalmError("the post-update step failed; rolled back", reason="uv exited 1")

    monkeypatch.setattr(updater, "apply", boom)
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is False and "error" in data


def test_update_apply_setup_class_does_not_restart(monkeypatch):
    _open_mode(monkeypatch)
    import localm.inference.http_server as hs
    restarted = []
    monkeypatch.setattr(hs, "_request_restart", lambda *a, **k: restarted.append(1))
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": "v0.2.0", "newer": True, "asset": {"id": 3}})
    monkeypatch.setattr(updater, "apply", lambda aid, **kw: {
        "applied": True, "version": "0.2.0", "klass": "setup", "backup": "b"})
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is True and not data.get("restarting")
    assert restarted == [], "a setup-class update must NOT auto-restart"


# ----------------------- post-update health watchdog -----------------------

def _monkeypatch_apply_ok(monkeypatch, version="0.2.0"):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": f"v{version}", "newer": True, "asset": {"id": 3}})
    monkeypatch.setattr(updater, "apply", lambda aid, **kw: {
        "applied": True, "version": version, "klass": "reboot", "backup": "b"})


def test_update_apply_builds_watchdog_from_app_state(monkeypatch):
    _open_mode(monkeypatch)
    import localm.inference.http_server as hs
    calls = []
    monkeypatch.setattr(hs, "_request_restart",
                        lambda *a, **k: calls.append(k.get("update_watchdog")))
    _monkeypatch_apply_ok(monkeypatch)
    app = create_app(_engine())
    app.state.bind_host = "0.0.0.0"          # wildcard bind -> probed via loopback
    app.state.instance_port = 9001
    app.state.instance_scheme = "http"
    data = _post(app, "/api/update/apply").json()
    assert data["applied"] is True
    assert calls == [{"host": "127.0.0.1", "port": 9001, "scheme": "http",
                      "expect_version": "0.2.0"}]


def test_update_apply_uses_concrete_bind_host_directly(monkeypatch):
    _open_mode(monkeypatch)
    import localm.inference.http_server as hs
    calls = []
    monkeypatch.setattr(hs, "_request_restart",
                        lambda *a, **k: calls.append(k.get("update_watchdog")))
    _monkeypatch_apply_ok(monkeypatch)
    app = create_app(_engine())
    app.state.bind_host = "192.168.1.5"      # a concrete, non-loopback bind
    app.state.instance_port = 9002
    app.state.instance_scheme = "http"
    _post(app, "/api/update/apply")
    assert calls[0]["host"] == "192.168.1.5"   # used as-is, NOT remapped to loopback


def test_update_apply_no_watchdog_when_port_missing(monkeypatch):
    """A bare create_app() that never advertised (instance_port unset) still
    restarts normally, with no watchdog."""
    _open_mode(monkeypatch)
    import localm.inference.http_server as hs
    calls = []
    monkeypatch.setattr(hs, "_request_restart",
                        lambda *a, **k: calls.append(k.get("update_watchdog")))
    _monkeypatch_apply_ok(monkeypatch)
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is True and data.get("restarting") is True
    assert calls == [None]


# ------------------------------- auth -----------------------------------

def test_update_apply_requires_management_auth(monkeypatch):
    """The mutating apply sits behind the management gate (shell-token /
    same-origin), so a no-token cross-origin caller cannot drive a self-update.
    The read-only check/issues follow the server's open-on-loopback read
    posture."""
    _open_mode(monkeypatch)
    with TestClient(create_app(_engine())) as c:
        assert c.post("/api/update/apply").status_code in (401, 403)
