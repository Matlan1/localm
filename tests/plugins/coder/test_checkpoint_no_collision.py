# SPDX-License-Identifier: AGPL-3.0-or-later
"""A checkpoint is keyed on (project, session id): ``HOME/checkpoints/<digest>/
<checkpoint-id>.json``, one file per session, so several interrupted sessions in
one project coexist rather than overwriting each other.

This file covers, bottom-up: the collision itself,
``list_checkpoints``/``checkpoint_info`` (the "pick any of them" and "resume the
latest" halves), title capture, migration of both legacy single-checkpoint
shapes, and checkpoint_info() as a side-effect-free probe that never claims an
id merely by being asked whether something exists.
"""

import json
from unittest.mock import patch

import pytest

from localm.plugins.coder.audit import SessionMode


class _Stub:
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def chat(self, messages, **kw):
        return "Done."

    def chat_stream(self, messages, **kw):
        yield "Done."


def _agent(cwd, **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / "home"
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    return h


# --------------------------------------------------------------------------- #
#  The collision itself                                                       #
# --------------------------------------------------------------------------- #

def test_two_sessions_in_the_same_project_no_longer_collide(home, tmp_path):
    """Session A is interrupted mid-task, then a SEPARATE session B (a different
    Agent, the shape of a second REPL/GUI session started in the same project)
    is interrupted on a different task. B's save must not touch A's file."""
    proj = tmp_path / "proj"
    proj.mkdir()

    a = _agent(proj)
    a._messages = [{"role": "user", "content": "task A: refactor the parser"}]
    a.save_checkpoint()
    path_a = a._checkpoint_path
    assert "task A" in path_a.read_text(encoding="utf-8")

    b = _agent(proj)
    b._messages = [{"role": "user", "content": "task B: fix the changelog"}]
    b.save_checkpoint()

    assert path_a != b._checkpoint_path, "two sessions must not share a file"
    assert "task A" in path_a.read_text(encoding="utf-8"), \
        "session A's checkpoint was destroyed by session B's save"
    assert "task B" in b._checkpoint_path.read_text(encoding="utf-8")


def test_resuming_and_re_saving_writes_back_to_the_same_file(home, tmp_path):
    """A SINGLE session's own resume-then-interrupt-again cycle must NOT mint a
    new file on every save."""
    proj = tmp_path / "proj"
    proj.mkdir()

    first = _agent(proj)
    first._messages = [{"role": "user", "content": "long task"}]
    first.save_checkpoint()
    original_path = first._checkpoint_path

    resumer = _agent(proj)
    data = resumer.load_checkpoint()
    resumer.resume_checkpoint(data)
    resumer._messages.append({"role": "assistant", "content": "more progress"})
    resumer.save_checkpoint()

    assert resumer._checkpoint_path == original_path
    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    assert len(list_checkpoints(proj)) == 1, "resume+save must not fork a new file"


# --------------------------------------------------------------------------- #
#  list_checkpoints / checkpoint_info: "any of them" vs "the latest, unaffected"
# --------------------------------------------------------------------------- #

def test_list_checkpoints_returns_every_session_newest_first(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    import time as _time

    a = _agent(proj)
    a._messages = [{"role": "user", "content": "older task"}]
    a._session_title = "older task"
    a.save_checkpoint()
    _time.sleep(0.02)   # mtime resolution
    b = _agent(proj)
    b._messages = [{"role": "user", "content": "newer task"}]
    b._session_title = "newer task"
    b.save_checkpoint()

    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    entries = list_checkpoints(proj)
    assert [e["title"] for e in entries] == ["newer task", "older task"]
    assert entries[0]["id"] == b._checkpoint_id
    assert entries[1]["id"] == a._checkpoint_id
    assert {e["id"] for e in entries} == {a._checkpoint_id, b._checkpoint_id}


def test_list_checkpoints_skips_a_corrupt_file_without_hiding_the_rest(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a._messages = [{"role": "user", "content": "good session"}]
    a._session_title = "good session"
    a.save_checkpoint()
    (a._checkpoint_path.parent / "garbage.json").write_text("{not json", encoding="utf-8")

    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    entries = list_checkpoints(proj)
    assert [e["title"] for e in entries] == ["good session"]


def test_list_checkpoints_empty_for_an_untouched_project(home, tmp_path):
    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    proj = tmp_path / "proj"
    proj.mkdir()
    assert list_checkpoints(proj) == []


def test_checkpoint_info_answers_the_most_recent_across_sessions(home, tmp_path):
    """The GUI's "resume?" probe (checkpoint_info) answers with the most recent
    session in the project, however many coexist."""
    proj = tmp_path / "proj"
    proj.mkdir()
    import time as _time

    a = _agent(proj)
    a._messages = [{"role": "user", "content": "first"}]
    a._session_title = "first"
    a.save_checkpoint()
    _time.sleep(0.02)
    b = _agent(proj)
    b._messages = [{"role": "user", "content": "second, more recent"}]
    b._session_title = "second, more recent"
    b.save_checkpoint()

    from localm.plugins.coder.agent.checkpoint import checkpoint_info
    info = checkpoint_info(proj)
    assert info["title"] == "second, more recent"


# --------------------------------------------------------------------------- #
#  Title capture: raw text, not the episode-wrapped one                       #
# --------------------------------------------------------------------------- #

def test_run_task_captures_the_raw_task_as_the_title(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a.run_task("refactor the auth module")
    assert a._session_title == "refactor the auth module"


def test_title_is_captured_before_episodic_preamble_wrapping(home, tmp_path):
    """_with_episodes prepends "relevant past lessons" text before the model
    ever sees the task - a title built from THAT would show boilerplate
    instead of what the user asked for."""
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a._episodic = True
    with patch.object(a, "_with_episodes",
                      return_value="Relevant past lessons:\n- ...\n\nfix the bug"):
        a.run_task("fix the bug")
    assert a._session_title == "fix the bug"
    assert "Relevant past lessons" not in a._session_title


def test_title_is_captured_once_not_overwritten_by_a_later_turn(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a.run_task("first task")
    a.chat("a completely different follow-up")
    assert a._session_title == "first task"


def test_saved_checkpoint_carries_a_truncated_single_line_title(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a._messages = [{"role": "user", "content": "line one\nline two   with   gaps"}]
    a._session_title = "line one\nline two   with   gaps"
    a.save_checkpoint()
    saved = json.loads(a._checkpoint_path.read_text(encoding="utf-8"))
    assert saved["title"] == "line one line two with gaps"


def test_untitled_session_gets_a_placeholder_not_an_empty_string(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a._messages = [{"role": "user", "content": "x"}]
    a.save_checkpoint()   # _session_title never set
    saved = json.loads(a._checkpoint_path.read_text(encoding="utf-8"))
    assert saved["title"] == "(untitled session)"


def test_resuming_restores_the_title_so_a_later_save_does_not_lose_it(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a.run_task("the original task")
    a.save_checkpoint()

    b = _agent(proj)
    b.resume_checkpoint(b.load_checkpoint())
    assert b._session_title == "the original task"
    b.save_checkpoint()
    saved = json.loads(b._checkpoint_path.read_text(encoding="utf-8"))
    assert saved["title"] == "the original task"


def test_reset_gives_a_cleared_conversation_a_fresh_identity(home, tmp_path):
    """reset() mints a fresh id, so the next interruption cannot overwrite the
    checkpoint of the session /clear just discarded."""
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a.run_task("session one")
    a.save_checkpoint()
    old_id, old_path = a._checkpoint_id, a._checkpoint_path

    a.reset()
    assert a._checkpoint_id != old_id
    assert a._session_title == ""
    a.run_task("session two")
    a.save_checkpoint()

    assert a._checkpoint_path != old_path
    assert "session one" in old_path.read_text(encoding="utf-8"), \
        "reset()'s new identity must not have clobbered the discarded session"


# --------------------------------------------------------------------------- #
#  Legacy migration: both shapes, not orphaned                                #
# --------------------------------------------------------------------------- #

def test_legacy_home_single_file_checkpoint_is_migrated_on_load(home, tmp_path):
    """The older shape: ONE file per project, HOME/checkpoints/<digest>.json,
    with no session id and no title."""
    proj = tmp_path / "proj"
    proj.mkdir()
    from localm.plugins.coder.agent.checkpoint import _legacy_home_checkpoint_path_for
    legacy = _legacy_home_checkpoint_path_for(proj)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({
        "version": 1, "turns": 3, "total_tokens": 50,
        "interrupted_at": "2026-07-01T00:00:00",
        "messages": [{"role": "user", "content": "an old pre-item-3 task"},
                     {"role": "assistant", "content": "ok"}],
    }), encoding="utf-8")

    a = _agent(proj)
    data = a.load_checkpoint()
    assert data is not None and data["turns"] == 3
    assert not legacy.exists(), "the legacy file must be migrated, not left behind"

    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    entries = list_checkpoints(proj)
    assert len(entries) == 1
    assert entries[0]["title"] == "an old pre-item-3 task", (
        "a legacy checkpoint predating the title field must be backfilled "
        "from its first message, not left untitled forever")

    a.resume_checkpoint(data)
    assert a._checkpoint_id == entries[0]["id"], (
        "load_checkpoint must adopt the migrated id so a later save lands "
        "in the SAME (new) file, not yet another one")


def test_legacy_in_project_checkpoint_is_also_migrated_on_load(home, tmp_path):
    """The oldest shape: <cwd>/.localcoder/checkpoint.json."""
    proj = tmp_path / "proj"
    proj.mkdir()
    legacy = proj / ".localcoder" / "checkpoint.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({
        "version": 1, "turns": 1, "total_tokens": 5,
        "messages": [{"role": "user", "content": "very old in-project checkpoint"}],
    }), encoding="utf-8")

    a = _agent(proj)
    data = a.load_checkpoint()
    assert data is not None
    assert not legacy.exists()
    from localm.plugins.coder.agent.checkpoint import list_checkpoints
    assert len(list_checkpoints(proj)) == 1


def test_migration_never_deletes_the_legacy_file_if_the_write_fails(home, tmp_path,
                                                                     monkeypatch):
    """A write failure while migrating must leave the ORIGINAL legacy file in
    place."""
    proj = tmp_path / "proj"
    proj.mkdir()
    from localm.plugins.coder.agent.checkpoint import _legacy_home_checkpoint_path_for
    legacy = _legacy_home_checkpoint_path_for(proj)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({
        "version": 1, "turns": 1, "total_tokens": 0,
        "messages": [{"role": "user", "content": "x"}],
    }), encoding="utf-8")

    from localm.plugins.coder.agent import checkpoint as ckpt_mod
    real_mkdir = ckpt_mod.Path.mkdir

    def _boom(self, *a, **kw):
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(ckpt_mod.Path, "mkdir", _boom)
    try:
        new_id = ckpt_mod.migrate_legacy_checkpoint(
            proj, legacy, json.loads(legacy.read_text(encoding="utf-8")))
    finally:
        monkeypatch.setattr(ckpt_mod.Path, "mkdir", real_mkdir)
    assert legacy.exists(), "the original must survive a failed migration write"
    assert new_id   # an id is still returned


# --------------------------------------------------------------------------- #
#  Resume by id                                                               #
# --------------------------------------------------------------------------- #

def test_load_checkpoint_by_explicit_id_picks_that_one_not_the_newest(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    import time as _time

    a = _agent(proj)
    a._messages = [{"role": "user", "content": "older"}]
    a.save_checkpoint()
    _time.sleep(0.02)
    b = _agent(proj)
    b._messages = [{"role": "user", "content": "newer"}]
    b.save_checkpoint()

    loader = _agent(proj)
    data = loader.load_checkpoint(a._checkpoint_id)
    assert data["messages"][0]["content"] == "older"
    assert loader._checkpoint_id == a._checkpoint_id


def test_load_checkpoint_by_unknown_id_returns_none(home, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    assert a.load_checkpoint("does-not-exist") is None


def test_explicit_id_lookup_never_falls_back_to_a_legacy_path(home, tmp_path):
    """An id names a per-session file. A legacy single-file checkpoint has no
    id, so it is never substituted when the requested id is not found."""
    proj = tmp_path / "proj"
    proj.mkdir()
    from localm.plugins.coder.agent.checkpoint import _legacy_home_checkpoint_path_for
    legacy = _legacy_home_checkpoint_path_for(proj)
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(json.dumps({
        "version": 1, "turns": 1, "total_tokens": 0,
        "messages": [{"role": "user", "content": "legacy"}],
    }), encoding="utf-8")

    a = _agent(proj)
    assert a.load_checkpoint("some-other-id") is None
    assert legacy.exists(), "an explicit-id miss must not trigger migration either"


# --------------------------------------------------------------------------- #
#  checkpoint_info() must be a true probe: no side effect on any Agent         #
# --------------------------------------------------------------------------- #

def test_checkpoint_info_probe_does_not_claim_a_checkpoint_id(home, tmp_path):
    """agent.load_checkpoint() sets self._checkpoint_id as a side effect, so a
    later save_checkpoint() writes back to the SAME file it just read. That is
    right for a caller about to resume_checkpoint() and wrong for one that only
    wants to know whether something exists; checkpoint_info() is the
    module-level, side-effect-free answer and touches no Agent state."""
    proj = tmp_path / "proj"
    proj.mkdir()
    existing = _agent(proj)
    existing._messages = [{"role": "user", "content": "an existing session"}]
    existing.save_checkpoint()
    existing_path = existing._checkpoint_path

    from localm.plugins.coder.agent.checkpoint import checkpoint_info
    info = checkpoint_info(proj)
    assert info is not None

    # A brand-new agent, as if a fresh session started and displayed the notice.
    fresh = _agent(proj)
    assert fresh._checkpoint_id != existing._checkpoint_id
    # Fresh agent typing a plain message: repl.py clears its own checkpoint first.
    fresh.clear_checkpoint()
    assert existing_path.read_text(encoding="utf-8"), (
        "the existing session's checkpoint must survive a fresh agent's own "
        "startup probe + clear-before-first-message sequence")
    fresh.run_task("a brand new, unrelated task")
    fresh.save_checkpoint()
    assert "an existing session" in existing_path.read_text(encoding="utf-8")
    assert fresh._checkpoint_path != existing_path


# --------------------------------------------------------------------------- #
#  REPL wiring: /sessions and /resume <id>                                    #
# --------------------------------------------------------------------------- #

def _printed(monkeypatch):
    """Capture every string printed via display.console.print during a test,
    as one big text blob - good enough to assert content without depending on
    rich's exact rendering."""
    from localm.plugins.coder import display
    lines = []
    monkeypatch.setattr(display.console, "print",
                        lambda *a, **kw: lines.append(" ".join(str(x) for x in a)))
    return lines


def test_sessions_command_lists_every_checkpoint_with_its_title(home, tmp_path,
                                                                  monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    a.run_task("investigate the flaky test")
    a.save_checkpoint()

    from localm.plugins.coder.cli.repl import _handle_command
    lines = _printed(monkeypatch)
    viewer = _agent(proj)
    _handle_command("/sessions", viewer)

    blob = "\n".join(lines)
    assert a._checkpoint_id in blob
    assert "investigate the flaky test" in blob


def test_sessions_command_on_an_empty_project_says_so(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    a = _agent(proj)
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_info", lambda msg: called.setdefault("msg", msg))
    repl_mod._handle_command("/sessions", a)
    assert "No saved sessions" in called["msg"]


def test_resume_with_no_arg_still_picks_the_newest(home, tmp_path, monkeypatch):
    """/resume with no argument resumes the most recent session."""
    proj = tmp_path / "proj"
    proj.mkdir()
    import time as _time

    a = _agent(proj)
    a.run_task("older")
    a.save_checkpoint()
    _time.sleep(0.02)
    b = _agent(proj)
    b.run_task("newer")
    b.save_checkpoint()

    from localm.plugins.coder.cli.repl import _handle_command
    resumer = _agent(proj)
    monkeypatch.setattr(resumer, "chat", lambda *a, **kw: "ok")
    _handle_command("/resume", resumer)
    assert resumer._checkpoint_id == b._checkpoint_id
    assert resumer._messages[0]["content"] == "newer"


def test_resume_with_an_explicit_id_picks_that_session(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    import time as _time

    a = _agent(proj)
    a.run_task("target session")
    a.save_checkpoint()
    _time.sleep(0.02)
    b = _agent(proj)
    b.run_task("decoy, more recent")
    b.save_checkpoint()

    from localm.plugins.coder.cli.repl import _handle_command
    resumer = _agent(proj)
    monkeypatch.setattr(resumer, "chat", lambda *a, **kw: "ok")
    _handle_command(f"/resume {a._checkpoint_id}", resumer)
    assert resumer._checkpoint_id == a._checkpoint_id
    assert resumer._messages[0]["content"] == "target session"


def test_resume_with_an_unknown_id_reports_it_cleanly(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_info", lambda msg: called.setdefault("msg", msg))
    a = _agent(proj)
    repl_mod._handle_command("/resume nope-not-real", a)
    assert "nope-not-real" in called["msg"]
    assert "/sessions" in called["msg"]


# --------------------------------------------------------------------------- #
#  cli/_main.py's startup notice: probe only, real end-to-end                 #
# --------------------------------------------------------------------------- #

def test_startup_notice_probe_leaves_an_existing_session_intact(home, tmp_path):
    """cli/_main.py calls checkpoint_info(work_dir), NOT agent.load_checkpoint(),
    for its "interrupted session found" notice, so a fresh REPL start never
    adopts an existing session's id just by checking whether one exists."""
    proj = tmp_path / "proj"
    proj.mkdir()
    existing = _agent(proj)
    existing.run_task("do not touch me")
    existing.save_checkpoint()
    existing_path = existing._checkpoint_path

    import localm.plugins.coder.cli._main as main_mod
    src = main_mod.__file__
    text = open(src, encoding="utf-8").read()
    assert "checkpoint_info(work_dir)" in text, (
        "the startup notice must use the side-effect-free probe, not "
        "agent.load_checkpoint() (which claims a checkpoint id as a side "
        "effect - fine for an actual resume, wrong for a mere existence check)")

    fresh = _agent(proj)
    from localm.plugins.coder.agent.checkpoint import checkpoint_info
    checkpoint_info(proj)          # the probe cli/_main.py actually calls
    assert fresh._checkpoint_id != existing._checkpoint_id, (
        "a probe must never mutate any agent's checkpoint identity")
    fresh.clear_checkpoint()
    assert "do not touch me" in existing_path.read_text(encoding="utf-8")
