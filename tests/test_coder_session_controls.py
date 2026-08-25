# SPDX-License-Identifier: AGPL-3.0-or-later
"""O6: the coder controls that existed only as REPL slash commands now have a
web form, per the standing rule that anything available in the CLI must be
available in SOME form in GUI mode.

  /approve /scope /verify   POST /api/coder/sessions/{id}/settings
  /cd                       POST /api/coder/sessions/{id}/cwd
  /memory /remember /forget GET + POST .../memory, POST .../memory/forget
  /bg                       GET  .../background

The workaround previously on record for the first group (start again with
resume) needs a checkpoint, and privacy mode never writes one - and privacy is
the DEFAULT on both surfaces, so a default GUI session had no route at all.

The properties pinned here are the ones that fail SILENTLY:

  * revoking auto-approve must reach a confirmation CHANNEL, not the
    fail-closed denial branch - a session that refuses everything looks, from
    the outside, exactly like a session that is asking;
  * a cwd change must take the session's checkpoint WITH it, or one
    conversation ends up with two resumable entries and the abandoned one is
    frozen forever while still being offered;
  * /bg must show only THIS session's jobs - the registry is process-wide and
    job labels are full command lines;
  * a memory write must RELOAD, or it silently takes effect only next session.
"""

import threading
from pathlib import Path

import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

# Non-routable RFC5737 (TEST-NET-1): guaranteed never to route anywhere, so even
# a total guard failure cannot dial a real host.
_UNC = "\\\\192.0.2.1\\share"


# --------------------------------------------------------------------------- #
#  Harness                                                                     #
# --------------------------------------------------------------------------- #

class _StubBackend:
    """Enough backend for a session, with no model behind it."""
    model_id = "stub-model"
    native_tools = False
    supports_native_tools = True
    supports_grammar = False

    def __init__(self):
        self.calls: list = []
        self.last_usage = {"prompt_tokens": 11, "total_tokens": 33}

    def set_tools(self, defs):
        pass

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return "ok"


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


def _owner(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    app = _coder_app(tmp_path, monkeypatch, api_key="ownersecret")
    app.state.root_dir = str(proj)
    return app, proj, {"Authorization": "Bearer ownersecret"}


def _start(client, headers, proj, **extra):
    r = client.post("/api/coder/sessions", headers=headers,
                    json={"cwd": str(proj), **extra})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _scoped(app):
    """Headers for a MINTED, non-owner coder-scoped key - the shareable session.

    Minted through localm.auth directly, the same way test_coder_safe_share.py
    does: the HTTP mint route has a lockout guard that behaves differently on a
    keystore's first key, which is not what any of these tests are about.
    """
    from localm import auth
    made = auth.create_key("shared", ["coder"])
    return {"Authorization": "Bearer " + made["key"]}


def _stub(app, sid):
    sess = app.state.coder_sessions.get(sid)
    sess.agent.backend = _StubBackend()
    return sess


# --------------------------------------------------------------------------- #
#  /approve and the confirmation CHANNEL it answers on                         #
# --------------------------------------------------------------------------- #

def test_revoking_auto_approve_installs_a_confirmation_channel(tmp_path, monkeypatch):
    """The flag alone is not the fix.

    A GUI session runs _loop(interactive=False). When a destructive tool needs
    confirmation and confirm_handler is None the agent takes its fail-closed
    branch and DENIES the call - correct as a default, useless as a revoke: the
    user gets no approval card and the session can no longer do anything at
    all. Flipping auto_approve without installing the handler would hand back a
    session that only refuses, which from the outside is indistinguishable from
    one that is waiting to be answered.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)

        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"auto_approve": False})
        assert r.status_code == 200, r.text
        assert r.json()["auto_approve"] is False
        assert r.json()["changed"] == ["auto_approve"]

        assert sess.agent.auto_approve is False
        # There is now a channel to ask on.
        assert sess.agent.confirm_handler == sess._confirm


def test_revoking_auto_approve_is_not_refused_while_the_agent_is_busy(
        tmp_path, monkeypatch):
    """The moment a user reaches for this is the moment the agent is mid-run
    doing something they want stopped. A busy guard would refuse the control in
    the only case it exists for - so, unlike the model route, there is none."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        sess.busy = True
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"auto_approve": False})
        assert r.status_code == 200, r.text
        assert sess.agent.auto_approve is False


def test_a_revoked_session_asks_instead_of_denying(tmp_path, monkeypatch):
    """End to end through the REAL dispatcher, not the wiring alone.

    Drives an actual destructive tool call after the revoke and asserts the
    confirmation was PUT to a human (and honoured), rather than short-circuited
    into "requires confirmation, but this run is non-interactive with no
    approval handler - denied".
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                    json={"auto_approve": False})

        asked = threading.Event()

        def answer():
            for _ in range(200):
                if sess._pending:
                    asked.set()
                    sess.answer_confirm(next(iter(sess._pending)), True)
                    return
                threading.Event().wait(0.02)

        threading.Thread(target=answer, daemon=True).start()

        from localm.plugins.coder.parser import ToolCall
        target = proj / "written.txt"
        res = sess.agent._execute_tool(
            ToolCall(name="write_file",
                     args={"path": "written.txt", "content": "hi"},
                     raw="", start=0, end=0),
            interactive=False)

        # Assert that a human was asked before checking the status flag.
        assert asked.is_set(), (
            "the revoke did not reach a confirmation channel - the call was "
            f"answered without asking anyone: {res.output!r}")
        assert res.ok, res.output
        assert target.read_text(encoding="utf-8") == "hi"


def test_interactive_confirm_at_creation_reaches_the_same_channel(
        tmp_path, monkeypatch):
    """A session created with auto-approve AND "still confirm shell commands"
    must ASK, not refuse.

    That combination sets always_confirm={run_shell, run_shell_background} while
    __init__ leaves confirm_handler None because auto_approve is on - so the
    confirmation gate reached the fail-closed branch and denied the command,
    under a checkbox whose own tooltip promises it "still stops for you". Same
    wiring as the revoke above, which is why one assignment covers both; pinned
    separately because it is reachable WITHOUT ever calling the settings route.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True,
                     interactive_confirm=True)
        sess = _stub(app, sid)
        assert "run_shell" in sess.agent.always_confirm

        asked = threading.Event()

        def answer():
            for _ in range(200):
                if sess._pending:
                    asked.set()
                    sess.answer_confirm(next(iter(sess._pending)), False)
                    return
                threading.Event().wait(0.02)

        threading.Thread(target=answer, daemon=True).start()

        from localm.plugins.coder.parser import ToolCall
        res = sess.agent._execute_tool(
            ToolCall(name="run_shell", args={"command": "echo hi"},
                     raw="", start=0, end=0),
            interactive=False)

        assert asked.is_set(), (
            "the shell command was decided without asking anyone: "
            f"{res.output!r}")
        # Rejected because the stand-in answered no, a HUMAN decision, not the
        # "no approval handler" denial.
        assert "Rejected by user" in (res.output or "")


# --------------------------------------------------------------------------- #
#  /scope and /verify, and absent-vs-null                                      #
# --------------------------------------------------------------------------- #

def test_absent_field_is_left_alone_and_explicit_null_clears(tmp_path, monkeypatch):
    """One call changes one control. A field the caller did not send must not be
    silently reset to its default, or every control would clobber the others."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, scope="src/**", verify="pytest -q")
        sess = _stub(app, sid)

        # Change ONLY auto_approve; scope and verify must survive untouched.
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"auto_approve": True})
        assert r.status_code == 200, r.text
        assert sess.agent.scope == "src/**"
        assert sess.agent.verify_cmd == "pytest -q"

        # An explicit null CLEARS - the other half of the same distinction.
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": None})
        assert r.status_code == 200, r.text
        assert sess.agent.scope is None
        assert sess.agent.verify_cmd == "pytest -q"      # still untouched

        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"verify": None})
        assert r.status_code == 200, r.text
        assert sess.agent.verify_cmd is None
        assert r.json()["verify"] is None


def test_the_settings_response_reports_the_scope_a_client_must_render(
        tmp_path, monkeypatch):
    """A control that can SET a value and never SHOW it is half a control.

    Every other test in this file asserts `sess.agent.scope` - the internal
    attribute - which a browser never sees. `info()` carried no `scope` key at
    all, so the settings panel would have rendered an empty box however many
    times the scope was set, and nothing here would have gone red. Found by the
    live end-to-end run, pinned here because the response a client reads back is
    the actual contract.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        _stub(app, sid)

        created = client.get("/api/coder/sessions", headers=owner).json()
        assert "scope" in created["sessions"][0], created["sessions"][0]
        assert "restricted" in created["sessions"][0]

        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": "src/**"})
        assert r.status_code == 200, r.text
        assert r.json()["scope"] == "src/**"

        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": None})
        assert r.json()["scope"] is None


def test_a_restricted_session_says_so_in_its_own_info(tmp_path, monkeypatch):
    """The GUI disables the directory and verification controls off this flag,
    so it can EXPLAIN the confinement instead of offering both and collecting a
    403 and a 409."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        shared = _scoped(app)
        sid = _start(client, shared, proj)
        _stub(app, sid)
        info = client.get("/api/coder/sessions", headers=shared).json()["sessions"][0]
        assert info["restricted"] is True

        owner_sid = _start(client, owner, proj)
        _stub(app, owner_sid)
        mine = [s for s in client.get("/api/coder/sessions", headers=owner)
                .json()["sessions"] if s["id"] == owner_sid][0]
        assert mine["restricted"] is False


def test_scope_can_be_set_and_confines_the_file_tools_immediately(
        tmp_path, monkeypatch):
    """Setting a scope is not bookkeeping: the very next file call must be
    refused if it falls outside. Asserted through the real dispatcher, because
    the attribute agreeing with the request proves only that a write landed."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    (proj / "src").mkdir()
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": "src/**"})
        assert r.status_code == 200, r.text
        assert r.json()["changed"] == ["scope"]

        from localm.plugins.coder.parser import ToolCall
        outside = sess.agent._execute_tool(
            ToolCall(name="write_file", args={"path": "nope.txt", "content": "x"},
                     raw="", start=0, end=0),
            interactive=False)
        assert not outside.ok
        assert "scope" in (outside.output or "").lower()
        assert not (proj / "nope.txt").exists()


def test_verify_auto_redetects_and_says_when_it_found_nothing(
        tmp_path, monkeypatch):
    """`/verify auto`. A re-detect that found no check must report None rather
    than leaving the previous command in place, or the caller reads a stale
    command back as a fresh detection."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, verify="pytest -q")
        sess = _stub(app, sid)
        # An empty project has no obvious check.
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"auto_verify": True})
        assert r.status_code == 200, r.text
        assert r.json()["verify"] is None
        assert sess.agent.verify_cmd is None


def test_verify_and_auto_verify_together_are_refused(tmp_path, monkeypatch):
    """Both asks cannot be honoured, so it is refused rather than resolved by an
    ordering nobody can see."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        _stub(app, sid)
        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"verify": "pytest", "auto_verify": True})
        assert r.status_code == 400


def test_a_restricted_session_cannot_be_given_a_verify_command(
        tmp_path, monkeypatch):
    """A restricted session has no process execution at all, and a verify
    command IS process execution - accepting one would hand a scoped key back
    exactly what the restriction removes. __init__ already refuses it at
    creation; this keeps the two paths agreeing."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        shared = _scoped(app)
        sid = _start(client, shared, proj)
        _stub(app, sid)
        assert app.state.coder_sessions.get(sid).restricted is True

        r = client.post(f"/api/coder/sessions/{sid}/settings", headers=shared,
                        json={"verify": "pytest -q"})
        assert r.status_code == 409
        assert "runs no commands" in r.json()["detail"]


# --------------------------------------------------------------------------- #
#  /cd - and the checkpoint rekey                                              #
# --------------------------------------------------------------------------- #

def test_cwd_moves_the_session_and_its_saved_checkpoint(tmp_path, monkeypatch):
    """THE REKEY DECISION.

    A checkpoint is filed under <digest(cwd)>/<checkpoint_id>.json, so changing
    the cwd changes where the next save lands. Leaving the old file behind gives
    ONE conversation TWO resumable entries, and the abandoned one is frozen at
    the moment of the move while still being offered by "continue last session"
    - a phantom that can never catch up. So it moves with the session.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    with TestClient(app) as client:
        # log mode, or a checkpoint is never written at all (privacy promises it).
        sid = _start(client, owner, proj, mode="log")
        sess = _stub(app, sid)
        sess.agent._messages = [{"role": "user", "content": "hello there"}]
        sess.agent.save_checkpoint()

        from localm.plugins.coder.agent.checkpoint import _checkpoint_path_for
        cid = sess.agent._checkpoint_id
        old_path = _checkpoint_path_for(proj, cid)
        assert old_path.is_file(), "precondition: a checkpoint was saved"

        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                        json={"cwd": str(other)})
        assert r.status_code == 200, r.text
        assert Path(r.json()["cwd"]) == other.resolve()
        assert sess.agent.cwd == other.resolve()

        new_path = _checkpoint_path_for(other, cid)
        assert new_path.is_file(), "the checkpoint did not follow the session"
        assert not old_path.exists(), (
            "a phantom copy was left behind under the old project - it would be "
            "offered by 'continue last session' forever, frozen at this moment")


def test_cwd_is_refused_for_a_restricted_session(tmp_path, monkeypatch):
    """create_session forces a shared-key session into the project root and
    ignores the cwd it was given. A route that moved it afterwards would hand
    back exactly what was taken away."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    with TestClient(app) as client:
        shared = _scoped(app)
        sid = _start(client, shared, proj)
        sess = _stub(app, sid)

        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=shared,
                        json={"cwd": str(other)})
        assert r.status_code == 403
        assert sess.agent.cwd == proj.resolve()

        # And not by the OWNER either: the containment belongs to the SESSION.
        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                        json={"cwd": str(other)})
        assert r.status_code == 403
        assert sess.agent.cwd == proj.resolve()


def test_cwd_refuses_unc_and_a_non_directory_and_a_busy_session(
        tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    other = tmp_path / "other"
    other.mkdir()
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        sess = _stub(app, sid)

        unc = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                          json={"cwd": _UNC})
        assert unc.status_code == 400
        assert "UNC" in unc.json()["detail"]

        missing = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                              json={"cwd": str(tmp_path / "nope")})
        assert missing.status_code == 400

        sess.busy = True
        busy = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                           json={"cwd": str(other)})
        assert busy.status_code == 409
        assert "busy" in busy.json()["detail"]
        assert sess.agent.cwd == proj.resolve()


# --------------------------------------------------------------------------- #
#  /memory /remember /forget                                                   #
# --------------------------------------------------------------------------- #

def test_remember_writes_the_file_and_reloads_it_into_the_prompt(
        tmp_path, monkeypatch):
    """THE RELOAD IS THE POINT, not the write.

    A GUI session loads and injects LOCALCODER.md but had no way to change it,
    and asking the agent to edit the file does not call reload_memory - so an
    edit made that way sits on disk without reaching the running session, taking
    effect only next session. Asserting the file exists would pass either way;
    the system prompt is what discriminates.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        sess = _stub(app, sid)

        before = client.get(f"/api/coder/sessions/{sid}/memory", headers=owner)
        assert before.status_code == 200, before.text
        assert before.json()["exists"] is False
        assert "npm test" not in sess.agent._system_prompt

        r = client.post(f"/api/coder/sessions/{sid}/memory", headers=owner,
                        json={"text": "the JS suite is run with npm test"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["exists"] is True
        assert "npm test" in body["text"]
        assert (proj / "LOCALCODER.md").is_file()

        # The live session is reading it NOW, not next session.
        assert "npm test" in sess.agent._memory
        assert "npm test" in sess.agent._system_prompt


def test_forget_distinguishes_no_file_from_no_match(tmp_path, monkeypatch):
    """Both leave the memory unchanged and they call for different next steps,
    so one number cannot stand for both."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        sess = _stub(app, sid)

        none = client.post(f"/api/coder/sessions/{sid}/memory/forget",
                           headers=owner, json={"pattern": "anything"})
        assert none.status_code == 200, none.text
        assert none.json() == {**none.json(), "had_file": False, "removed": 0}

        client.post(f"/api/coder/sessions/{sid}/memory", headers=owner,
                    json={"text": "keep me"})
        miss = client.post(f"/api/coder/sessions/{sid}/memory/forget",
                           headers=owner, json={"pattern": "absent"})
        assert miss.json()["had_file"] is True
        assert miss.json()["removed"] == 0
        assert "keep me" in miss.json()["text"]

        hit = client.post(f"/api/coder/sessions/{sid}/memory/forget",
                          headers=owner, json={"pattern": "KEEP"})
        assert hit.json()["removed"] == 1
        assert "keep me" not in hit.json()["text"]
        # And the running session stopped reading it.
        assert "keep me" not in sess.agent._system_prompt


def test_memory_rejects_an_empty_write(tmp_path, monkeypatch):
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        _stub(app, sid)
        assert client.post(f"/api/coder/sessions/{sid}/memory", headers=owner,
                           json={"text": "   "}).status_code == 400
        assert client.post(f"/api/coder/sessions/{sid}/memory/forget",
                           headers=owner,
                           json={"pattern": "  "}).status_code == 400


# --------------------------------------------------------------------------- #
#  /sessions + /resume <id>                                                    #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#  /bg                                                                         #
# --------------------------------------------------------------------------- #

def _fake_job_cls():
    """A registrable job with no process behind it.

    Built at call time rather than at import so this file does not import
    background.py at module scope, matching how the sibling suites keep the
    coder's heavier modules lazy.
    """
    from localm.plugins.coder.background import BackgroundJob

    class _Job(BackgroundJob):
        kind = "job"

        def _poll(self):
            return None          # never finishes on its own

        def _terminate(self, *, force: bool) -> None:
            return None          # nothing to signal; the base class would raise

    return _Job


def test_background_lists_only_this_sessions_jobs(tmp_path, monkeypatch):
    """The registry is PROCESS-WIDE and a GUI server runs many sessions in one
    process, so an unfiltered list would show one session another's work - and a
    job label is a full command line, so it would read the owner's commands out
    to whoever asked."""
    from localm.plugins.coder.background import get_registry

    Job = _fake_job_cls()
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        mine = _start(client, owner, proj)
        theirs = _start(client, owner, proj)
        s_mine = _stub(app, mine)
        s_theirs = _stub(app, theirs)

        reg = get_registry()
        reg.submit(lambda: Job("MY-SECRET-COMMAND",
                               owner=s_mine.agent.job_owner), kind="job")
        reg.submit(lambda: Job("THEIR-SECRET-COMMAND",
                               owner=s_theirs.agent.job_owner), kind="job")
        try:
            r = client.get(f"/api/coder/sessions/{mine}/background", headers=owner)
            assert r.status_code == 200, r.text
            labels = [j["label"] for j in r.json()["jobs"]]
            assert labels == ["MY-SECRET-COMMAND"], labels
            assert r.json()["supported"] is True
        finally:
            from localm.plugins.coder.background import reset_registry
            reset_registry()


def test_a_spawned_child_agents_jobs_belong_to_the_parent_session(
        tmp_path, monkeypatch):
    """A sub-agent is an implementation detail of the call that spawned it. If
    its background work were attributed to the child, the parent's /bg would
    silently stop listing work it started - and nothing else can query a child.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        sess = _stub(app, sid)
        from localm.plugins.coder.agent import Agent
        child = Agent(_StubBackend(), cwd=proj, parent=sess.agent,
                      mode=sess.agent.mode)
        try:
            assert child.job_owner == sess.agent.job_owner
        finally:
            child.close()


def test_background_says_a_restricted_session_can_never_have_any(
        tmp_path, monkeypatch):
    """"none yet" and "this session has no shell or sub-agent tools at all" are
    different answers, and an empty list alone cannot say which."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        shared = _scoped(app)
        sid = _start(client, shared, proj)
        _stub(app, sid)
        r = client.get(f"/api/coder/sessions/{sid}/background", headers=shared)
        assert r.status_code == 200, r.text
        assert r.json()["jobs"] == []
        assert r.json()["supported"] is False


def test_dropped_completions_are_counted_per_owner(tmp_path, monkeypatch):
    """A bounded table that says nothing about what it discarded is the silent
    truncation rule 5 forbids - and reporting the WHOLE process's losses inside
    one session would over-report to a session that lost nothing."""
    from localm.plugins.coder.background import JobRegistry

    Job = _fake_job_cls()
    reg = JobRegistry(keep_finished=1)
    for label, who in (("a1", "own-a"), ("a2", "own-a"), ("b1", "own-b")):
        job = reg.submit(lambda label=label, who=who: Job(label, owner=who),
                         kind="job")
        job.state = "done"
    # One more submit to force the prune now that three are finished, then park
    # it too so the atexit sweep has no live job to chase.
    reg.submit(lambda: Job("trigger", owner="own-b"), kind="job").state = "done"

    a = reg.dropped_for("own-a")
    b = reg.dropped_for("own-b")
    everyone = reg.dropped_for()
    assert sum(a.values()) + sum(b.values()) == sum(everyone.values())
    assert sum(everyone.values()) >= 1
    # Nobody is charged for a loss that was not theirs.
    assert sum(reg.dropped_for("own-nobody").values()) == 0


def test_background_route_is_not_reachable_for_another_principals_session(
        tmp_path, monkeypatch):
    """Session isolation still applies: a scoped key gets 404 (not 403) for a
    session it does not own, so it cannot even probe which ids exist."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        owners_session = _start(client, owner, proj)
        _stub(app, owners_session)
        shared = _scoped(app)
        for path in ("background", "memory"):
            r = client.get(f"/api/coder/sessions/{owners_session}/{path}",
                           headers=shared)
            assert r.status_code == 404, (path, r.status_code)
        r = client.post(f"/api/coder/sessions/{owners_session}/settings",
                        headers=shared, json={"auto_approve": True})
        assert r.status_code == 404


# --------------------------------------------------------------------------- #
#  A sub-agent is not a separate session: it follows the parent, live          #
# --------------------------------------------------------------------------- #

def _child_of(sess, proj):
    """A child Agent built exactly the way the spawn tools build one."""
    from localm.plugins.coder.agent import Agent
    from localm.plugins.coder.tools.agents import (
        _inherited_confirm_handler, inherited_child_kwargs,
    )
    return Agent(**inherited_child_kwargs(
        sess.agent, backend=_StubBackend(), cwd=proj, name="child",
        max_turns=5, confirm_handler=_inherited_confirm_handler(sess.agent)))


def test_revoking_auto_approve_reaches_a_sub_agent_already_running(
        tmp_path, monkeypatch):
    """The revoke has to reach DELEGATED work, or it is not the control the UI
    says it is.

    A child is built with the parent's settings copied in at construction, and
    nothing propagates a later change - so before this, revoking auto-approve
    on the session returned 200 while a spawned child carried on writing files
    without asking anyone. The panel says "it stops a run already under way",
    the route docstring says the same, and the changelog says it in public
    product text; this is the test that makes those true.

    A child cannot be addressed from outside - there is no route to it, no
    confirmation channel of its own, and Stop sets only the parent's flag - so
    the session that spawned it is the only thing a human can steer.
    """
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        child = _child_of(sess, proj)
        try:
            assert child.auto_approve is True        # inherited at spawn

            r = client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                            json={"auto_approve": False})
            assert r.status_code == 200, r.text
            assert child.auto_approve is False, (
                "the child kept auto-approving after the session revoked it")

            asked = threading.Event()

            def answer():
                for _ in range(200):
                    if sess._pending:
                        asked.set()
                        sess.answer_confirm(next(iter(sess._pending)), False)
                        return
                    threading.Event().wait(0.02)

            threading.Thread(target=answer, daemon=True).start()

            from localm.plugins.coder.parser import ToolCall
            target = proj / "child-wrote.txt"
            res = child._execute_tool(
                ToolCall(name="write_file",
                         args={"path": "child-wrote.txt", "content": "x"},
                         raw="", start=0, end=0),
                interactive=False)

            # Assert on the file first: a file the child wrote unasked is the loss.
            assert not target.exists(), (
                "a sub-agent wrote a file after the session revoked approval")
            assert asked.is_set(), (
                f"the child neither asked nor was stopped: {res.output!r}")
        finally:
            child.close()


def test_a_child_spawned_without_approval_is_not_re_approved_by_the_parent(
        tmp_path, monkeypatch):
    """The inheritance only ever NARROWS. A parent turning auto-approve back on
    must not silently re-approve a child deliberately spawned without it."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=False)
        sess = _stub(app, sid)
        child = _child_of(sess, proj)
        try:
            assert child.auto_approve is False
            client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"auto_approve": True})
            assert sess.agent.auto_approve is True
            assert child.auto_approve is False, (
                "the parent re-approved a child that was spawned unapproved")
        finally:
            child.close()


def test_tightening_the_scope_reaches_a_sub_agent_already_running(
        tmp_path, monkeypatch):
    """Same shape as the revoke: a child that inherited its confinement follows
    the parent's, so tightening mid-run reaches delegated work."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        # Spawn the child while the parent ALREADY has a scope. With the parent
        # unscoped at spawn, copying the value and following it live are
        # indistinguishable, since both leave the child at None.
        client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                    json={"scope": "wide/**"})
        child = _child_of(sess, proj)
        try:
            assert child.scope == "wide/**"      # inherited at spawn

            client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": "narrow/**"})
            assert child.scope == "narrow/**", (
                "the child kept the scope it was spawned with after the session "
                "tightened it")

            # Clearing it on the session releases the child too - it never had a
            # scope of its own to fall back on.
            client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": None})
            assert child.scope is None
        finally:
            child.close()


def test_a_child_given_its_own_scope_keeps_it(tmp_path, monkeypatch):
    """An explicit child scope is a deliberate NARROWING, so inheriting over it
    would WIDEN the child - the one direction that must never happen."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj)
        sess = _stub(app, sid)
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.tools.agents import inherited_child_kwargs
        child = Agent(**inherited_child_kwargs(
            sess.agent, backend=_StubBackend(), cwd=proj, name="child",
            max_turns=5, confirm_handler=None, scope="only/this/**"))
        try:
            client.post(f"/api/coder/sessions/{sid}/settings", headers=owner,
                        json={"scope": "everything/**"})
            assert child.scope == "only/this/**"
        finally:
            child.close()


# --------------------------------------------------------------------------- #
#  Moving a recording session into a project that declared itself private      #
# --------------------------------------------------------------------------- #

def test_cwd_refuses_to_move_a_recording_session_into_a_private_project(
        tmp_path, monkeypatch):
    """A session's persistence is fixed when it starts, and its transcript is
    written to whatever cwd it holds at close - so moving a `full` session into
    a project whose .localcoder/config.toml says `privacy` left a complete
    record inside a project that asked for none. The mode cannot be lowered to
    match (the audit log is already open), so the move is refused."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    private = tmp_path / "private"
    (private / ".localcoder").mkdir(parents=True)
    (private / ".localcoder" / "config.toml").write_text(
        'mode = "privacy"\n', encoding="utf-8")

    with TestClient(app) as client:
        sid = _start(client, owner, proj, mode="full")
        sess = _stub(app, sid)

        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                        json={"cwd": str(private)})
        assert r.status_code == 409, r.text
        assert "privacy" in r.json()["detail"]
        # The session did NOT move, so nothing can later be written there.
        assert sess.agent.cwd == proj.resolve()

        sess.agent._messages = [{"role": "user", "content": "secret work"}]
        sess.close()
        transcripts = private / ".localcoder" / "sessions"
        assert not transcripts.exists() or not list(transcripts.glob("*.md")), (
            "a project marked private received a session transcript")


def test_cwd_allows_a_move_into_an_equally_or_more_recording_project(
        tmp_path, monkeypatch):
    """The guard only blocks the LOSSY direction. A privacy session may move
    anywhere, and a move between equal modes is unaffected - otherwise the
    refusal would be a blanket ban dressed as a privacy control."""
    app, proj, owner = _owner(tmp_path, monkeypatch)
    # DECLARES a mode, and a more-recording one than the session's, so the rank
    # comparison is the line that decides. A destination that declares nothing
    # returns earlier and never reaches it.
    louder = tmp_path / "louder"
    (louder / ".localcoder").mkdir(parents=True)
    (louder / ".localcoder" / "config.toml").write_text(
        'mode = "full"' + chr(10), encoding="utf-8")
    plain = tmp_path / "plain"
    plain.mkdir()

    with TestClient(app) as client:
        sid = _start(client, owner, proj, mode="log")
        sess = _stub(app, sid)

        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                        json={"cwd": str(louder)})
        assert r.status_code == 200, r.text
        assert sess.agent.cwd == louder.resolve()

        # And a project that has declared NOTHING is not asserting privacy: the
        # global default is "privacy", so keying on effective_mode here would
        # refuse every ordinary directory.
        r = client.post(f"/api/coder/sessions/{sid}/cwd", headers=owner,
                        json={"cwd": str(plain)})
        assert r.status_code == 200, r.text
        assert sess.agent.cwd == plain.resolve()


# --------------------------------------------------------------------------- #
#  A checkpoint id is a filename component                                     #
# --------------------------------------------------------------------------- #

def test_a_checkpoint_id_cannot_escape_the_checkpoint_directory(tmp_path):
    """A checkpoint id arrives in an HTTP body and is concatenated into a path.

    A loaded id is also RETAINED (`self._checkpoint_id`), so the next
    save_checkpoint - and the cwd-change migration - would write wherever it
    pointed. Measured before the guard: `../../../../windows/win.ini` resolved
    clean outside the checkpoints tree.
    """
    from localm.plugins.coder.agent.checkpoint import (
        _checkpoint_path_for, is_valid_checkpoint_id,
    )
    for bad in ("../../../../windows/win.ini", "../other/x", "a/b",
                "a\\b", ".", "", "x" * 65):
        assert not is_valid_checkpoint_id(bad), bad
        with pytest.raises(ValueError):
            _checkpoint_path_for(tmp_path, bad)
    for good in ("deadbeefcafe", "abc_123-XYZ"):
        assert is_valid_checkpoint_id(good), good


# --------------------------------------------------------------------------- #
#  The /bg attribution mechanism itself                                        #
# --------------------------------------------------------------------------- #

def test_a_background_shell_job_is_stamped_with_the_running_sessions_owner(
        tmp_path, monkeypatch):
    """The per-session /bg list rests on an INJECTED hidden argument, and the
    other background tests construct jobs with an explicit owner= - so a
    regression in the injection itself would leave them green while every real
    job became unattributed and invisible to the session that started it. This
    drives the real dispatcher."""
    from localm.plugins.coder.background import get_registry, reset_registry
    from localm.plugins.coder.parser import ToolCall

    app, proj, owner = _owner(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sid = _start(client, owner, proj, auto_approve=True)
        sess = _stub(app, sid)
        try:
            res = sess.agent._execute_tool(
                ToolCall(name="run_shell_background",
                         args={"command": "python -c \"import time; time.sleep(5)\""},
                         raw="", start=0, end=0),
                interactive=False)
            assert res.ok, res.output

            jobs = get_registry().list_status()
            assert len(jobs) == 1, jobs
            # The job carries THIS agent's owner id...
            job = next(iter(get_registry()._jobs.values()))
            assert job.owner == sess.agent.job_owner

            # ...so the route lists it for this session.
            r = client.get(f"/api/coder/sessions/{sid}/background", headers=owner)
            assert r.status_code == 200, r.text
            assert [j["id"] for j in r.json()["jobs"]] == [job.id]

            # And a DIFFERENT session does not see it.
            other = _start(client, owner, proj)
            _stub(app, other)
            r2 = client.get(f"/api/coder/sessions/{other}/background", headers=owner)
            assert r2.json()["jobs"] == []
        finally:
            reset_registry()


def test_the_repl_cd_refuses_the_same_move(tmp_path, monkeypatch):
    """The CLI must not diverge from the web surface on this.

    The whole point of this unit is that the two agree, so fixing the privacy
    hole only on the route would have opened a NEW divergence in the same
    change. Both call the one helper; this drives the REPL's /cd handler.
    """
    from localm.plugins.coder.cli import repl as _repl

    proj = tmp_path / "proj"
    proj.mkdir()
    private = tmp_path / "private"
    (private / ".localcoder").mkdir(parents=True)
    (private / ".localcoder" / "config.toml").write_text(
        'mode = "privacy"\n', encoding="utf-8")

    class _Agent:
        def __init__(self):
            from localm.audit import SessionMode
            self.cwd = proj
            self.mode = SessionMode.FULL
            self.moved_to = None

        def set_cwd(self, d):
            self.moved_to = d

    agent = _Agent()
    warned: list = []
    monkeypatch.setattr(_repl, "print_warning", lambda m: warned.append(m))
    monkeypatch.setattr(_repl, "print_info", lambda m: None)

    _repl._handle_command(f"/cd {private}", agent)
    assert agent.moved_to is None, "the REPL moved a recording session anyway"
    assert warned and "privacy" in warned[-1], warned

    # And the harmless direction is still allowed, or the guard is a blanket ban.
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    _repl._handle_command(f"/cd {ordinary}", agent)
    assert agent.moved_to == ordinary
