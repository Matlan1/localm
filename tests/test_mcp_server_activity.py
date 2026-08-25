# SPDX-License-Identifier: AGPL-3.0-or-later
"""The MCP `server_activity` tool: what a running localm server is doing, and an honest answer when that cannot be determined."""

from __future__ import annotations

import time

import pytest

from localm.plugins.mcpserver.server import EngineCache, build_tools


@pytest.fixture
def tools(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "home_dir", lambda: home, raising=False)
    return build_tools(EngineCache(default_model=None), enable_images=False,
                       enable_coder=False)


def _call(tools):
    return tools["server_activity"]["handler"]({})["content"][0]["text"]


def _claims_idle(out: str) -> bool:
    """Whether the text POSITIVELY claims a server reported itself idle."""
    return "idle, nothing running" in out.lower()


def _patch_instances(monkeypatch, rows):
    from localm import instances
    monkeypatch.setattr(instances, "snapshot", lambda *a, **kw: rows)


def _patch_read(monkeypatch, state, payload):
    import localm.selfclient as sc
    monkeypatch.setattr(sc, "read_activity",
                        lambda scheme, port, instance_token=None, bind_host=None:
                        (state, payload))


_LIVE = [{"scheme": "http", "port": 1234, "alive": True}]


# ---------------------------------------------------------------- registered

def test_the_tool_is_registered_and_marked_read_only(tools):
    assert "server_activity" in tools
    assert tools["server_activity"]["annotations"]["readOnlyHint"] is True


# ------------------------------------------- the three states an agent needs

def test_no_server_running_is_not_reported_as_idle(tools, monkeypatch):
    """Nothing to ask."""
    _patch_instances(monkeypatch, [])
    out = _call(tools)
    assert "no localm server is running" in out.lower()
    assert not _claims_idle(out)


def test_a_registered_but_dead_server_is_unknown_not_idle(tools, monkeypatch):
    _patch_instances(monkeypatch, [{"scheme": "http", "port": 1234, "alive": False}])
    out = _call(tools)
    assert "unknown" in out.lower()
    assert not _claims_idle(out)


@pytest.mark.parametrize("state,payload", [
    ("unreachable", "ConnectionError"),
    ("unauthorized", 401),
    ("unsupported", 404),
    ("http", 500),
])
def test_no_failure_state_is_reported_as_idle(tools, monkeypatch, state, payload):
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, state, payload)
    out = _call(tools)
    assert "unknown" in out.lower(), f"{state} did not say activity is unknown: {out!r}"
    assert not _claims_idle(out), f"{state} was reported as idle: {out!r}"


def test_only_a_real_answer_reports_idle(tools, monkeypatch):
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "ok", {"now": time.time(), "operations": []})
    out = _call(tools)
    assert _claims_idle(out)


def test_unauthorized_matches_the_other_failure_branches_register(tools, monkeypatch):
    """#953 (grader 2): the pre-fix 'needs an API key this process does not have' wording read like an optional hardening tip rather than the same kind of failure 'could not be reached'/'could not be read' already are."""
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "unauthorized", 401)
    out = _call(tools)
    assert "could not be asked" in out.lower()
    assert "needs an api key" not in out.lower()


# ------------------------------------------------------------ the good case

def test_snapshot_called_with_include_token(tools, monkeypatch):
    """#953: this tool asks each discovered instance over HTTP, so it needs the attach token a genuinely open instance's middleware requires - unlike `localm ps`, which must never see it."""
    from localm import instances
    captured = {}

    def _spy_snapshot(*a, **kw):
        captured.update(kw)
        return []
    monkeypatch.setattr(instances, "snapshot", _spy_snapshot)
    _call(tools)
    assert captured.get("include_token") is True


def test_the_rows_token_reaches_read_activity(tools, monkeypatch):
    """The token snapshot() now returns per-row must actually reach read_activity, not just exist in the row - a real, working credential is useless if the call site never reads it."""
    _patch_instances(monkeypatch, [
        {"scheme": "http", "port": 1234, "alive": True, "token": "row-token",
         "host": "::1"}])
    captured = {}

    def _fake_read_activity(scheme, port, instance_token=None, bind_host=None):
        captured["instance_token"] = instance_token
        captured["bind_host"] = bind_host
        return "ok", {"now": time.time(), "operations": []}
    import localm.selfclient as sc
    monkeypatch.setattr(sc, "read_activity", _fake_read_activity)
    _call(tools)
    assert captured["instance_token"] == "row-token"
    # The row's BIND HOST must reach it too: an IPv6-bound instance is not
    # listening on the IPv4 loopback, so a call that dropped this would report
    # a healthy server as unreachable.
    assert captured["bind_host"] == "::1"


def test_running_operations_are_listed(tools, monkeypatch):
    now = time.time()
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "ok", {"now": now, "operations": [
        {"id": "a", "kind": "pull", "label": "Model pull owner/repo",
         "status": "running", "created_at": now - 90, "finished_at": None,
         "cancellable": True, "pct": 37.4}]})
    out = _call(tools)
    assert "Model pull owner/repo" in out
    assert "running" in out
    assert "37%" in out
    assert "90s elapsed" in out


def test_no_percentage_is_invented_when_none_was_reported(tools, monkeypatch):
    """R1: an operation that has reported no progress is at an unknown percentage, not at zero."""
    now = time.time()
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "ok", {"now": now, "operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": now - 5, "finished_at": None, "cancellable": True}]})
    assert "%" not in _call(tools)


def test_the_elapsed_time_uses_the_server_clock(tools, monkeypatch):
    """This process may not share the server's clock."""
    server_now = time.time() - 7200
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "ok", {"now": server_now, "operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": server_now - 30, "finished_at": None,
         "cancellable": True}]})
    out = _call(tools)
    assert "30s elapsed" in out, out


def test_an_age_is_omitted_when_the_server_sent_no_clock(tools, monkeypatch):
    now = time.time()
    _patch_instances(monkeypatch, _LIVE)
    _patch_read(monkeypatch, "ok", {"operations": [
        {"id": "a", "kind": "pull", "label": "P", "status": "running",
         "created_at": now - 500, "finished_at": None, "cancellable": True}]})
    out = _call(tools)
    assert "P" in out
    assert "elapsed" not in out, f"invented an age with no reference clock: {out!r}"
