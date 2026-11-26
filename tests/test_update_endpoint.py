# SPDX-License-Identifier: AGPL-3.0-or-later
"""The server endpoints for the issues tracker + updater: /api/issues,
/api/update/check (read-only), /api/update/apply (mutates -> CONFIG_WRITE). Apply is
explicit-only; an update that is not started is surfaced, never faked. All proxy calls
are monkeypatched (no network), and the build is never really prepared or handed over."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fastapi.testclient import TestClient

from localm import config, issue_tracker, updater
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


@pytest.fixture(autouse=True)
def prepared(monkeypatch):
    """``updater_handoff.prepare`` answers a prepared update to 0.2.0 with no other process
    of localm running, and records the asset id and arguments it was called with."""
    from localm import updater_handoff
    calls = []

    def prepare(asset_id, **kw):
        calls.append((asset_id, kw))
        return updater_handoff.Prepared(Path("."), "0" * 32, "0.1.0", "0.2.0", None, True,
                                        None, False, None)

    monkeypatch.setattr(updater_handoff, "prepare", prepare)
    return calls


def _open_mode(monkeypatch):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)


# ------------------------------ issues ----------------------------------

def test_issues_endpoint_lists(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "filed_status",
                        lambda **kw: [{"number": 5, "state": "open", "title": "t"}])
    r = _get(create_app(_engine()), "/api/issues")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] and data["issues"][0]["number"] == 5


def test_issues_endpoint_unconfigured(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(issue_tracker, "available", lambda: False)
    r = _get(create_app(_engine()), "/api/issues")
    assert r.json()["available"] is False


def test_issues_endpoint_row_error_surfaced(monkeypatch):
    _open_mode(monkeypatch)
    monkeypatch.setattr(issue_tracker, "available", lambda: True)
    monkeypatch.setattr(issue_tracker, "filed_status", lambda **kw: [
        {"number": 5, "title": "t", "error": "could not reach the localm proxy"}])
    data = _get(create_app(_engine()), "/api/issues").json()
    assert data["issues"][0]["error"] == "could not reach the localm proxy"
    assert "state" not in data["issues"][0]


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
        raise LocalmError(config.net_off_lead("no update check was made"),
                          reason=config.NET_OFF_UPDATE_REMEDY)

    monkeypatch.setattr(updater, "check", boom)
    data = _get(create_app(_engine()), "/api/update/check").json()
    assert data["available"] is True
    assert config.net_off_lead("no update check was made") in data["error"]
    assert "newer" not in data


# --------------------------- update apply -------------------------------

def _restarts(monkeypatch) -> list:
    """The keyword arguments of each restart the route asks for (none really happens)."""
    import localm.inference.http_server as hs
    calls: list = []
    monkeypatch.setattr(hs, "_request_restart", lambda *a, **k: calls.append(k))
    return calls


def _newer(monkeypatch, version="0.2.0"):
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.1.0", "latest": f"v{version}", "newer": True, "notes": "",
        "asset": {"id": 3}})


def test_update_apply_endpoint(monkeypatch, prepared):
    _open_mode(monkeypatch)
    restarts = _restarts(monkeypatch)
    _newer(monkeypatch)
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is True and data["handed_off"] is True
    assert data["restarting"] is True and data["version"] == "0.2.0"
    assert [aid for aid, _kw in prepared] == [3]
    assert len(restarts) == 1 and restarts[0]["update"].new == "0.2.0"


def test_update_apply_already_current(monkeypatch, prepared):
    _open_mode(monkeypatch)
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "check", lambda: {
        "current": "0.2.0", "latest": "0.2.0", "newer": False, "asset": None})
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is False
    assert prepared == []


def test_update_apply_with_an_unorderable_release_does_not_claim_up_to_date(monkeypatch,
                                                                            prepared):
    _open_mode(monkeypatch)
    from localm import _proxy, _version
    monkeypatch.setattr(updater, "available", lambda: True)
    monkeypatch.setattr(updater, "endpoint", lambda: ("https://proxy.invalid", None))
    monkeypatch.setattr(updater, "_net_policy_allows_update_check", lambda: True)
    monkeypatch.setattr(updater, "_prerelease_channel_enabled", lambda: False)
    monkeypatch.setattr(_version, "read_version", lambda: "0.2.0")
    monkeypatch.setattr(_proxy, "request", lambda *a, **kw: {
        "version": "nightly", "asset": {"id": 3}, "signature": "s"})
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert prepared == []
    assert data["applied"] is False
    assert "up to date" not in data["reason"]
    assert "could not tell" in data["reason"]
    assert data["latest"] == "nightly"


def test_update_apply_failure_is_surfaced(monkeypatch):
    _open_mode(monkeypatch)
    from localm import updater_handoff
    from localm.bugreport import LocalmError
    restarts = _restarts(monkeypatch)
    _newer(monkeypatch)

    def boom(aid, **kw):
        raise LocalmError("the update was not started; nothing was changed",
                          reason="its check ended with 1")

    monkeypatch.setattr(updater_handoff, "prepare", boom)
    data = _post(create_app(_engine()), "/api/update/apply").json()
    assert data["applied"] is False and "its check ended with 1" in data["error"]
    assert restarts == []


# ------------------- where the update program asks the new version -------------------

def test_update_apply_watches_the_loopback_of_a_wildcard_bind(monkeypatch):
    _open_mode(monkeypatch)
    restarts = _restarts(monkeypatch)
    _newer(monkeypatch)
    app = create_app(_engine())
    app.state.bind_host = "0.0.0.0"          # wildcard bind -> asked via loopback
    app.state.insecure_bind = True
    app.state.instance_port = 9001
    app.state.instance_scheme = "http"
    data = _post(app, "/api/update/apply").json()
    assert data["applied"] is True
    assert restarts[0]["update"].watch == {"host": "127.0.0.1", "port": 9001,
                                           "scheme": "http"}
    assert restarts[0]["port"] == 9001


def test_update_apply_uses_concrete_bind_host_directly(monkeypatch):
    _open_mode(monkeypatch)
    restarts = _restarts(monkeypatch)
    _newer(monkeypatch)
    app = create_app(_engine())
    app.state.bind_host = "192.168.1.5"      # a concrete, non-loopback bind
    app.state.insecure_bind = True
    app.state.instance_port = 9002
    app.state.instance_scheme = "http"
    _post(app, "/api/update/apply")
    assert restarts[0]["update"].watch["host"] == "192.168.1.5"   # used as-is


# ------------------------------- auth -----------------------------------

def test_update_apply_requires_management_auth(monkeypatch):
    """The mutating apply sits behind the management gate (shell-token /
    same-origin), so a no-token cross-origin caller cannot drive a self-update.
    The read-only check/issues follow the server's open-on-loopback read
    posture."""
    _open_mode(monkeypatch)
    with TestClient(create_app(_engine())) as c:
        assert c.post("/api/update/apply").status_code in (401, 403)
