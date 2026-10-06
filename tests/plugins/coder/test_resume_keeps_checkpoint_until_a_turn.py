# SPDX-License-Identifier: AGPL-3.0-or-later
"""A saved coder session stays on disk until a turn decides its fate. After
`localm coder --resume`, leaving the REPL before sending anything leaves the
session resumable; a turn interrupted with Ctrl-C rewrites the SAME checkpoint; a
turn, or a one-shot TASK, that finishes cleanly removes it; a turn that fails with
an error leaves the last saved state in place. The REPL's own /resume follows the
same lifetime, and the message printed on Ctrl-C matches what is on disk."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from localm.plugins.coder.agent import Agent
from localm.plugins.coder.agent.checkpoint import (
    _checkpoint_path_for,
    list_checkpoints,
)
from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.cli import _main
from localm.plugins.coder.cli import repl as repl_mod

_SAVED_TASK = "refactor the parser"


class _ScriptedBackend:
    """Answers every LLM call with one plain reply (no tool calls), or raises
    *fail_with* instead, and counts the calls."""

    model_id = "stub-model"
    native_tools = False
    supports_grammar = False

    def __init__(self, reply: str = "Done.", fail_with: Exception | None = None):
        self._reply = reply
        self._fail_with = fail_with
        self.calls = 0
        self.last_usage: dict = {}
        self.last_reasoning = ""

    def chat(self, messages, **kwargs):
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        return self._reply

    def chat_stream(self, messages, on_reasoning=None, **kwargs):
        self.calls += 1
        if self._fail_with is not None:
            raise self._fail_with
        yield self._reply

    def set_tools(self, tool_defs):
        pass

    def context_capacity(self):
        return None


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """An active coder plugin under a throwaway HOME, and an empty project."""
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    p = tmp_path / "proj"
    p.mkdir()
    return p


def _agent(proj: Path, backend: _ScriptedBackend | None = None) -> Agent:
    with patch("localm.plugins.coder.agent.ProjectMap") as pm, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        pm.build.return_value.file_count.return_value = 0
        pm.build.return_value.truncated = False
        return Agent(backend or _ScriptedBackend(), cwd=proj, mode=SessionMode.LOG,
                     auto_approve=True, self_verify=False)


def _saved_session(proj: Path, task: str = _SAVED_TASK) -> str:
    """Leave a checkpoint for *proj* the way a real session does, with a turn
    interrupted by Ctrl-C; returns its id."""
    agent = _agent(proj)
    with patch.object(Agent, "_call_llm", side_effect=KeyboardInterrupt), \
         pytest.raises(KeyboardInterrupt):
        agent.chat(task)
    agent.close()
    assert _checkpoint_path_for(proj, agent._checkpoint_id).is_file()
    return agent._checkpoint_id


def _cli(proj: Path, monkeypatch, *args, stdin: str = "",
         backend: _ScriptedBackend | None = None):
    backend = backend or _ScriptedBackend()
    monkeypatch.setattr(_main, "_build_backend", lambda *a, **k: backend)
    result = CliRunner().invoke(
        _main.main, ["-m", "stub-model", "--cwd", str(proj), *args],
        input=stdin, catch_exceptions=True)
    return result, backend


def _text(result) -> str:
    """The CLI output with line wrapping collapsed."""
    return " ".join(result.output.split())


def _ids(proj: Path) -> list:
    return [e["id"] for e in list_checkpoints(proj)]


# --------------------------------------------------------------------------- #
#  Leaving the REPL before any turn runs                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode_args", [[], ["--mode", "log"]],
                         ids=["default-mode", "log-mode"])
@pytest.mark.parametrize("stdin", ["exit\n", "/exit\n", ""],
                         ids=["exit", "slash-exit", "eof"])
def test_leaving_the_repl_right_after_resume_keeps_the_session(
        proj, monkeypatch, stdin, mode_args):
    ckpt_id = _saved_session(proj)

    result, backend = _cli(proj, monkeypatch, *mode_args, "--resume", stdin=stdin)

    assert "Resumed session" in result.output, result.output
    assert backend.calls == 0, "no turn may run between the resume and the exit"
    assert _checkpoint_path_for(proj, ckpt_id).is_file(), (
        "leaving the REPL before sending anything deleted the resumed session's "
        "checkpoint, so the conversation can no longer be resumed")
    assert _ids(proj) == [ckpt_id]
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_ctrl_c_at_the_first_prompt_after_resume_keeps_the_session(
        proj, monkeypatch):
    ckpt_id = _saved_session(proj)

    def _ctrl_c():
        raise KeyboardInterrupt

    monkeypatch.setattr(repl_mod, "_read_multiline", _ctrl_c)

    result, backend = _cli(proj, monkeypatch, "--resume")

    assert "Resumed session" in result.output, result.output
    assert backend.calls == 0
    assert _checkpoint_path_for(proj, ckpt_id).is_file(), (
        "Ctrl-C at the first prompt deleted the resumed session's checkpoint")
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_resuming_a_named_session_and_leaving_keeps_both_sessions(
        proj, monkeypatch):
    target = _saved_session(proj, "the target session")
    newer = _saved_session(proj, "a more recent session")

    result, _ = _cli(proj, monkeypatch, "--resume", target, stdin="exit\n")

    assert "the target session" in result.output, result.output
    assert _checkpoint_path_for(proj, target).is_file(), (
        "leaving the REPL deleted the explicitly resumed session's checkpoint")
    assert sorted(_ids(proj)) == sorted([target, newer])
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------- #
#  The first turn after the resume decides the checkpoint                     #
# --------------------------------------------------------------------------- #

def test_an_interrupted_first_turn_after_resume_rewrites_the_same_checkpoint(
        proj, monkeypatch):
    ckpt_id = _saved_session(proj)

    def _interrupted_turn(self, messages, interactive):
        raise KeyboardInterrupt

    monkeypatch.setattr(Agent, "_call_llm", _interrupted_turn)

    result, _ = _cli(proj, monkeypatch, "--mode", "log", "--resume",
                     stdin="now add tests\n")

    assert "Resumed session" in result.output, result.output
    assert _ids(proj) == [ckpt_id], (
        "the interrupted turn must rewrite the resumed session's own checkpoint, "
        "not leave a second one beside it")
    saved = _checkpoint_path_for(proj, ckpt_id).read_text(encoding="utf-8")
    assert _SAVED_TASK in saved and "now add tests" in saved
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_a_first_turn_after_resume_that_finishes_clears_the_checkpoint(
        proj, monkeypatch):
    ckpt_id = _saved_session(proj)

    result, backend = _cli(proj, monkeypatch, "--mode", "log", "--resume",
                           stdin="now add tests\n")

    assert "Resumed session" in result.output, result.output
    assert backend.calls >= 1, "the first message must have run a turn"
    assert not _checkpoint_path_for(proj, ckpt_id).exists(), (
        "a resumed session whose next turn finished cleanly left its checkpoint "
        "behind")
    assert _ids(proj) == []
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_resume_with_a_task_that_finishes_clears_the_checkpoint(proj, monkeypatch):
    ckpt_id = _saved_session(proj)

    result, backend = _cli(proj, monkeypatch, "now add tests", "--resume", "--yes")

    assert "Resumed session" in result.output, result.output
    assert backend.calls >= 1, "the one-shot task must have run"
    assert not _checkpoint_path_for(proj, ckpt_id).exists(), (
        "a resumed one-shot task that finished cleanly left its checkpoint behind")
    assert _ids(proj) == []
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------- #
#  The REPL's own /resume: same lifetime                                      #
# --------------------------------------------------------------------------- #

def test_repl_resume_whose_turn_finishes_clears_the_checkpoint(proj):
    ckpt_id = _saved_session(proj)
    resumer = _agent(proj)

    repl_mod._handle_command("/resume", resumer)

    assert resumer._checkpoint_id == ckpt_id
    assert _ids(proj) == []


def test_repl_resume_whose_turn_is_interrupted_rewrites_the_same_checkpoint(
        proj, monkeypatch):
    ckpt_id = _saved_session(proj)
    resumer = _agent(proj)

    def _interrupted_turn(self, messages, interactive):
        raise KeyboardInterrupt

    monkeypatch.setattr(Agent, "_call_llm", _interrupted_turn)

    repl_mod._handle_command("/resume", resumer)

    assert _ids(proj) == [ckpt_id]
    saved = _checkpoint_path_for(proj, ckpt_id).read_text(encoding="utf-8")
    assert _SAVED_TASK in saved and "Continue from where we left off." in saved


# --------------------------------------------------------------------------- #
#  A turn that fails with an error keeps the last saved state                 #
# --------------------------------------------------------------------------- #

_SERVER_DOWN = ConnectionRefusedError("model server refused the connection")


@pytest.mark.parametrize("mode_args", [[], ["--mode", "log"]],
                         ids=["default-mode", "log-mode"])
def test_a_failed_first_turn_after_resume_keeps_the_saved_session(
        proj, monkeypatch, mode_args):
    ckpt_id = _saved_session(proj)

    result, backend = _cli(proj, monkeypatch, *mode_args, "--resume",
                           stdin="now add tests\nexit\n",
                           backend=_ScriptedBackend(fail_with=_SERVER_DOWN))

    text = _text(result)
    assert "Resumed session" in text, result.output
    assert backend.calls >= 1 and "Agent error" in text, (
        "the first turn after the resume must have run and failed")
    assert _checkpoint_path_for(proj, ckpt_id).is_file(), (
        "a turn that failed with an error deleted the resumed session's "
        "checkpoint, so the session can no longer be resumed")
    assert _ids(proj) == [ckpt_id]
    saved = _checkpoint_path_for(proj, ckpt_id).read_text(encoding="utf-8")
    assert _SAVED_TASK in saved
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_repl_resume_whose_turn_fails_keeps_the_saved_session(proj):
    ckpt_id = _saved_session(proj)
    backend = _ScriptedBackend(fail_with=_SERVER_DOWN)
    resumer = _agent(proj, backend)

    repl_mod._handle_command("/resume", resumer)

    assert resumer._checkpoint_id == ckpt_id
    assert backend.calls >= 1, "the /resume continue turn must have run"
    assert _ids(proj) == [ckpt_id], (
        "a /resume whose continue turn failed deleted the saved session")


def test_a_failed_turn_after_an_interrupted_one_keeps_the_saved_progress(
        proj, monkeypatch):
    monkeypatch.setattr(Agent, "_call_llm",
                        MagicMock(side_effect=[KeyboardInterrupt, _SERVER_DOWN]))

    result, _ = _cli(proj, monkeypatch, "--mode", "log",
                     stdin="first task\nsecond task\nexit\n")

    text = _text(result)
    assert "progress saved" in text and "Agent error" in text, result.output
    ids = _ids(proj)
    assert len(ids) == 1, (
        "a turn that failed with an error deleted the progress the interrupted "
        "turn before it had saved")
    assert "first task" in _checkpoint_path_for(proj, ids[0]).read_text(
        encoding="utf-8")
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


# --------------------------------------------------------------------------- #
#  What Ctrl-C reports matches what is on disk                                #
# --------------------------------------------------------------------------- #

def test_ctrl_c_in_privacy_mode_after_resume_says_resume_still_works(
        proj, monkeypatch):
    ckpt_id = _saved_session(proj)
    monkeypatch.setattr(Agent, "_call_llm", MagicMock(side_effect=KeyboardInterrupt))

    result, _ = _cli(proj, monkeypatch, "--resume", stdin="now add tests\n")

    text = _text(result)
    assert "Resumed session" in text, result.output
    assert _checkpoint_path_for(proj, ckpt_id).is_file(), (
        "Ctrl-C during the first turn after a resume deleted the saved session")
    assert "/resume is unavailable" not in text, (
        "privacy mode said /resume is unavailable while the resumed session's "
        "checkpoint is still on disk")
    assert "/resume returns to the last saved point" in text, result.output
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_ctrl_c_in_privacy_mode_with_nothing_saved_says_resume_is_unavailable(
        proj, monkeypatch):
    monkeypatch.setattr(Agent, "_call_llm", MagicMock(side_effect=KeyboardInterrupt))

    result, _ = _cli(proj, monkeypatch, stdin="first task\n")

    assert _ids(proj) == []
    assert "no checkpoint saved; /resume is unavailable" in _text(result), (
        result.output)
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output


def test_ctrl_c_when_the_checkpoint_cannot_be_written_does_not_claim_it_was_saved(
        proj, monkeypatch):
    import localm.plugins.coder.agent.persistence as persistence

    def _disk_full(path, data, *args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(persistence, "atomic_write", _disk_full)
    monkeypatch.setattr(Agent, "_call_llm", MagicMock(side_effect=KeyboardInterrupt))

    result, _ = _cli(proj, monkeypatch, "--mode", "log", stdin="first task\n")

    text = _text(result)
    assert _ids(proj) == [], "the checkpoint write was supposed to fail"
    assert "progress saved" not in text, (
        "Ctrl-C reported the progress as saved when the write had failed")
    assert "progress could not be saved" in text, result.output
    assert result.exception is None, repr(result.exception)
    assert result.exit_code == 0, result.output
