# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resume a past coder session. A checkpoint saved for a cwd can be restored into
a new session (owner / coder:full only); the GET /api/coder/resumable probe is
owner-gated and reflects whether a checkpoint exists for a directory."""

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.plugins.coder.agent import Agent

# A non-routable RFC5737 documentation address, so nothing here can reach a real
# host.
_UNC = r"\\192.0.2.1\share"
_UNC_FWD = "//192.0.2.1/share"
_DEVICE = r"\\.\PhysicalDrive0"


def _is_unc_or_device(s: str) -> bool:
    """pathsafe.is_unc_or_device_path's forbidden-prefix check, judged by
    Windows rules on every host, not gated on os.name - cwd here is client
    (HTTP request) supplied, so the guard refuses `//host/share` everywhere,
    unlike the local-path (`reject_unsafe_path_string`) policy."""
    return s[:2] in ("\\\\", "//", "\\/", "/\\")


def _unc_calls(seen) -> list:
    """Filter, don't assert `seen == []`: an unrelated legitimate fs call
    elsewhere in request handling must not fail a test about the malicious
    string specifically."""
    return [s for s in seen if _is_unc_or_device(s)]


def _install_fs_spy(monkeypatch, method_name):
    """Record every path string that reaches Path.<method_name>(), and
    hard-fail the call when the string is UNC/device syntax - same discipline
    as test_admin_fs_routes.py's fs_spy: assert on the absence of the
    syscall, not merely on the returned status code, since the defect is the
    syscall (an SMB dial that auto-authenticates), not the response body."""
    seen: list = []
    real = getattr(Path, method_name)

    def spy(self, *a, **kw):
        s = str(self)
        seen.append(s)
        if _is_unc_or_device(s):
            raise AssertionError(
                f"Path.{method_name}() reached the filesystem with a UNC/device "
                f"string: {s!r} - this is the SMB dial (and the net-NTLMv2 "
                "leak), which happens before any status code is chosen")
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, method_name, spy)
    return seen


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _coder_app(tmp_path, monkeypatch, *, api_key):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setenv("LOCALM_API_KEY", api_key)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("coder")

    async def switch_model(name):
        pass

    from localm.plugins.gui.web import attach_gui
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


def _seed_checkpoint(app, sid, messages):
    """Put a saved conversation on a live session and persist it to disk."""
    sess = app.state.coder_sessions.get(sid)
    sess.agent._messages = messages
    sess.agent._turns = len(messages)
    sess.agent._total_tokens = 42
    sess.persist_checkpoint()
    return sess


def test_resume_restores_a_saved_conversation_for_the_owner(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    msgs = [{"role": "user", "content": "build a calculator"},
            {"role": "assistant", "content": "Here is the plan."}]

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        assert a.status_code == 200
        _seed_checkpoint(app, a.json()["id"], msgs)

        # The probe now sees a resumable checkpoint for this dir.
        probe = client.get("/api/coder/resumable",
                           headers=owner, params={"cwd": str(proj)}).json()
        assert probe["resumable"] is True
        assert probe["turns"] == 2 and probe["messages"] == 2

        # A fresh session with resume=true restores the conversation.
        b = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log", "resume": True})
        assert b.status_code == 200
        assert b.json()["resumed"] is True
        restored = app.state.coder_sessions.get(b.json()["id"]).agent._messages
        assert restored == msgs


def test_resume_false_does_not_restore(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], [{"role": "user", "content": "x"}])
        # Default (no resume) starts fresh.
        b = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        assert b.json().get("resumed") is False
        assert app.state.coder_sessions.get(b.json()["id"]).agent._messages == []


def test_resumable_and_resume_are_owner_only(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])
    sh = {"Authorization": f"Bearer {scoped['key']}"}

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], [{"role": "user", "content": "secret"}])

        # A scoped key is never told a resumable checkpoint exists.
        s = client.get("/api/coder/resumable", headers=sh,
                       params={"cwd": str(proj)}).json()
        assert s["resumable"] is False

        # And a restricted (scoped) session cannot resume - it starts fresh, NOT
        # loading the owner's prior conversation.
        b = client.post("/api/coder/sessions", headers=sh,
                        json={"cwd": str(proj), "resume": True})
        assert b.status_code == 200
        assert b.json().get("resumed") is False
        assert app.state.coder_sessions.get(b.json()["id"]).agent._messages == []


def test_restricted_session_does_not_clobber_the_owner_checkpoint(tmp_path, monkeypatch):
    proj = tmp_path / "proj"; proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    owner = {"Authorization": "Bearer ownersecret"}
    from localm import auth
    scoped = auth.create_key("phone", ["coder"])
    sh = {"Authorization": f"Bearer {scoped['key']}"}
    owner_msgs = [{"role": "user", "content": "owner work"}]

    with TestClient(app) as client:
        a = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log"})
        _seed_checkpoint(app, a.json()["id"], owner_msgs)

        # A restricted scoped session (forced to the project root) runs + persists.
        b = client.post("/api/coder/sessions", headers=sh,
                        json={"cwd": str(proj), "mode": "log"})
        sess = app.state.coder_sessions.get(b.json()["id"])
        assert sess.restricted is True
        sess.agent._messages = [{"role": "user", "content": "scoped work"}]
        sess.persist_checkpoint()                 # must be a NO-OP for restricted

        # The owner can still resume THEIR conversation, not the scoped one.
        c = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": str(proj), "mode": "log", "resume": True})
        assert c.json()["resumed"] is True
        assert app.state.coder_sessions.get(c.json()["id"]).agent._messages == owner_msgs


def test_resumable_validates_cwd(tmp_path, monkeypatch):
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        # Missing cwd -> 400.
        assert client.get("/api/coder/resumable", headers=owner).status_code == 400
        # A non-existent directory -> not resumable (not an error).
        r = client.get("/api/coder/resumable", headers=owner,
                       params={"cwd": str(tmp_path / "nope")}).json()
        assert r["resumable"] is False


@pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
def test_create_session_rejects_unc_and_device_cwd_without_touching_the_filesystem(
        tmp_path, monkeypatch, bad):
    """The owner/coder:full branch calls
    Path(req.cwd).expanduser().is_dir()/.resolve(), so it needs a lexical check
    first. The `restricted` branch just above it ignores req.cwd entirely (it
    uses root_dir), so this is the MORE-trusted branch, not the less-trusted
    one."""
    seen = _install_fs_spy(monkeypatch, "is_dir")
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": bad, "mode": "log"})
    assert r.status_code == 400, r.text
    assert bad not in r.json().get("detail", ""), (
        "the raw client string must not be echoed back unsanitised")
    assert _unc_calls(seen) == [], (
        "the UNC/device string reached Path.is_dir() - the whole finding is "
        "that this syscall happens before any status code is chosen")


@pytest.mark.parametrize("bad", [_UNC, _UNC_FWD, _DEVICE])
def test_resumable_rejects_unc_and_device_cwd_without_touching_the_filesystem(
        tmp_path, monkeypatch, bad):
    """coder_resumable is a GET route with no CSRF check (CSRF only applies to
    unsafe methods), so an unguarded cwd here is reachable via a plain
    cross-origin request from any page the user has open - no local foothold on
    the machine required."""
    seen = _install_fs_spy(monkeypatch, "is_dir")
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        r = client.get("/api/coder/resumable", headers=owner, params={"cwd": bad})
    assert r.status_code == 400, r.text
    assert bad not in r.json().get("detail", ""), (
        "the raw client string must not be echoed back unsanitised")
    assert _unc_calls(seen) == [], (
        "the UNC/device string reached Path.is_dir() - the whole finding is "
        "that this syscall happens before any status code is chosen")


@pytest.mark.skipif(
    os.name != "nt",
    reason="HOMEDRIVE/HOMEPATH take precedence over USERPROFILE and this "
           "repro clears them so USERPROFILE wins; POSIX expands ~ from the "
           "password database and ignores USERPROFILE entirely, so this "
           "mechanism is Windows-specific")
def test_create_session_rejects_a_cwd_that_expands_into_unc_via_userprofile(
        tmp_path, monkeypatch):
    """Regression pin for the expanduser-then-check ORDERING, not just the
    guard's existence: a raw `~`-prefixed cwd is NOT UNC-shaped as written, so a
    guard that checked the RAW string would pass it straight through - only
    AFTER expanduser() runs (resolving ~ against the server's own USERPROFILE)
    does it become a UNC string. That is a real, if unusual, Windows
    configuration (a roaming profile pointing the home directory at a network
    share), not attacker input - the SERVER's own environment, not something a
    client controls.

    The invariant is that the guard must check the string that actually reaches
    the syscall, not an earlier form of it. Every other UNC test in this file
    uses an already-UNC-shaped raw string, so none of them would fail if this
    ordering were reverted - this one is designed to."""
    monkeypatch.setenv("USERPROFILE", _UNC)
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    # Confirm the environment actually produces a UNC string.
    expanded = str(Path("~/proj").expanduser())
    assert _is_unc_or_device(expanded), (
        f"test setup did not produce a UNC path: expanduser() gave {expanded!r}")

    seen = _install_fs_spy(monkeypatch, "is_dir")
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        r = client.post("/api/coder/sessions", headers=owner,
                        json={"cwd": "~/proj", "mode": "log"})
    assert r.status_code == 400, r.text
    assert _unc_calls(seen) == []


@pytest.mark.skipif(
    os.name != "nt",
    reason="HOMEDRIVE/HOMEPATH take precedence over USERPROFILE and this "
           "repro clears them so USERPROFILE wins; POSIX expands ~ from the "
           "password database and ignores USERPROFILE entirely, so this "
           "mechanism is Windows-specific")
def test_resumable_rejects_a_cwd_that_expands_into_unc_via_userprofile(
        tmp_path, monkeypatch):
    """Same ordering-regression pin as create_session's version above, for
    coder_resumable's independent guard."""
    monkeypatch.setenv("USERPROFILE", _UNC)
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    expanded = str(Path("~/proj").expanduser())
    assert _is_unc_or_device(expanded), (
        f"test setup did not produce a UNC path: expanduser() gave {expanded!r}")

    seen = _install_fs_spy(monkeypatch, "is_dir")
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(tmp_path)
    owner = {"Authorization": "Bearer ownersecret"}
    with TestClient(app) as client:
        r = client.get("/api/coder/resumable", headers=owner, params={"cwd": "~/proj"})
    assert r.status_code == 400, r.text
    assert _unc_calls(seen) == []


def test_agent_persist_then_resume_roundtrip_offline(tmp_path, monkeypatch):
    # The persist/resume mechanism works without the HTTP layer (LOG mode).
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    from unittest.mock import patch
    from localm.plugins.coder.audit import SessionMode

    proj = tmp_path / "proj"; proj.mkdir()
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        a = Agent(_StubBackend(), cwd=proj, mode=SessionMode.LOG,
                  auto_approve=True, self_verify=False)
        a._messages = [{"role": "user", "content": "hi"}]
        a._turns = 1
        a.save_checkpoint()

        b = Agent(_StubBackend(), cwd=proj, mode=SessionMode.LOG,
                  auto_approve=True, self_verify=False)
        data = b.load_checkpoint()
        assert data is not None
        b.resume_checkpoint(data)
        assert b._messages == [{"role": "user", "content": "hi"}]


def test_resume_recap_strips_a_fenced_json_tool_call(tmp_path, monkeypatch):
    """A ```json-fenced call (one of the five shapes parse_tool_calls
    recognises, name-gated) must be removed from the recap exactly like an XML
    call is, leaving only the surviving prose - resume_from_checkpoint knowing
    only the <tool_call> XML wrapper leaves raw fence markers and JSON in."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()
    from localm.plugins.coder.sessions import CoderSession

    a = CoderSession(proj, _StubBackend(), auto_approve=True, auto_verify=False,
                     mode="log")
    a.agent._messages = [
        {"role": "user", "content": "read a.py please"},
        {"role": "assistant", "content": (
            'Sure, reading it now.\n'
            '```json\n{"name": "read_file", "args": {"path": "a.py"}}\n```'
        )},
    ]
    a.agent._turns = 2
    a.persist_checkpoint()

    b = CoderSession(proj, _StubBackend(), auto_approve=True, auto_verify=False,
                     mode="log")
    assert b.resume_from_checkpoint() is True

    texts = [e["text"] for e in b.history if e.get("type") == "history"]
    assert texts == ["read a.py please", "Sure, reading it now."]


def test_resume_recap_strips_a_bare_json_tool_call(tmp_path, monkeypatch):
    """The other shape that can leak raw: a bare top-level JSON object with no
    wrapper at all. A pure-call message (no prose) must vanish from the recap
    exactly like a pure XML tool-call message does; prose alongside one must
    survive while the JSON itself does not."""
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path / "home")
    proj = tmp_path / "proj"; proj.mkdir()
    from localm.plugins.coder.sessions import CoderSession

    a = CoderSession(proj, _StubBackend(), auto_approve=True, auto_verify=False,
                     mode="log")
    a.agent._messages = [
        {"role": "user", "content": "write it out"},
        {"role": "assistant", "content": (
            'Let me check that file.\n'
            '{"name": "read_file", "args": {"path": "a.py"}}'
        )},
        {"role": "assistant", "content": (
            '{"name": "write_file", "args": {"path": "out.py", "content": "x"}}'
        )},
    ]
    a.agent._turns = 3
    a.persist_checkpoint()

    b = CoderSession(proj, _StubBackend(), auto_approve=True, auto_verify=False,
                     mode="log")
    assert b.resume_from_checkpoint() is True

    texts = [e["text"] for e in b.history if e.get("type") == "history"]
    # The pure-call third message (no prose) contributes no row at all.
    assert texts == ["write it out", "Let me check that file."]
