# SPDX-License-Identifier: AGPL-3.0-or-later
"""The REPL's /review command: an on-demand second opinion from the existing
Reviewer, independent of whether coder_review is enabled for the automatic
end-of-session pass.

reviewer.py's own Reviewer/reviewer_for_agent logic (prompt building, JSON
parsing, the privacy/restricted gate) is covered by test_reviewer.py. This
file covers only the REPL wiring: does /review reuse an already-configured
reviewer, does it build one on demand (bypassing only the coder_review
on/off switch) when none exists, and does it surface the ReviewResult to
the user.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localm.plugins.coder.audit import SessionMode
from localm.plugins.coder.reviewer import Reviewer


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


def _proj(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    return p


def _backend_returning(text):
    b = MagicMock()
    b.chat.return_value = text
    b.model_id = "reviewer-model"
    return b


# --------------------------------------------------------------------------- #
#  Registration                                                                #
# --------------------------------------------------------------------------- #

def test_review_command_is_in_slash_command_list_and_help():
    from localm.plugins.coder.cli.repl import _SLASH_COMMANDS
    assert "/review" in _SLASH_COMMANDS
    from localm.plugins.coder.display import HELP_TEXT
    assert "/review" in HELP_TEXT


# --------------------------------------------------------------------------- #
#  No diff yet: no reviewer is even built                                     #
# --------------------------------------------------------------------------- #

def test_review_with_no_changes_reports_and_builds_no_reviewer(
        home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "")
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_info",
                        lambda msg: called.setdefault("msg", msg))
    with patch("localm.plugins.coder.reviewer.reviewer_for_agent") as rfa:
        repl_mod._handle_command("/review", agent)
    assert "No changes to review" in called["msg"]
    rfa.assert_not_called()


# --------------------------------------------------------------------------- #
#  coder_review already on: reuse agent._reviewer, never rebuild               #
# --------------------------------------------------------------------------- #

def test_review_reuses_the_agents_existing_reviewer(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    agent._reviewer = Reviewer(
        _backend_returning('{"approved": true, "blocking": []}'))
    from localm.plugins.coder.cli import repl as repl_mod
    with patch("localm.plugins.coder.reviewer.reviewer_for_agent") as rfa:
        repl_mod._handle_command("/review", agent)
    rfa.assert_not_called()
    agent._reviewer.backend.chat.assert_called_once()


# --------------------------------------------------------------------------- #
#  coder_review off: build one on demand, forcing PAST only that gate          #
# --------------------------------------------------------------------------- #

def test_review_builds_a_forced_reviewer_when_none_configured(
        home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    assert agent._reviewer is None          # default: coder_review is off
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    scripted = Reviewer(_backend_returning('{"approved": true, "blocking": []}'))
    from localm.plugins.coder.cli import repl as repl_mod
    with patch("localm.plugins.coder.reviewer.reviewer_for_agent",
               return_value=scripted) as rfa:
        repl_mod._handle_command("/review", agent)
    rfa.assert_called_once_with(agent.backend, agent.mode, agent.restricted,
                                force=True)
    scripted.backend.chat.assert_called_once()


def test_review_reports_no_reviewer_for_a_restricted_session(
        home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    from localm.plugins.coder.cli import repl as repl_mod
    called = {}
    monkeypatch.setattr(repl_mod, "print_warning",
                        lambda msg: called.setdefault("msg", msg))
    with patch("localm.plugins.coder.reviewer.reviewer_for_agent",
               return_value=None):
        repl_mod._handle_command("/review", agent)
    assert "No reviewer available" in called["msg"]


# --------------------------------------------------------------------------- #
#  The verdict is surfaced to the user                                        #
# --------------------------------------------------------------------------- #

def test_review_prints_approval_when_no_blocking_issues(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    agent._reviewer = Reviewer(
        _backend_returning('{"approved": true, "blocking": [], "notes": "clean"}'))
    from localm.plugins.coder.cli import repl as repl_mod
    calls = []
    monkeypatch.setattr(repl_mod, "print_success", lambda msg: calls.append(msg))
    repl_mod._handle_command("/review", agent)
    assert calls and "no blocking issues" in calls[0].lower()


def test_review_prints_blocking_issues_when_flagged(home, tmp_path, monkeypatch):
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    agent._reviewer = Reviewer(
        _backend_returning(
            '{"approved": false, "blocking": ["off-by-one in loop"], '
            '"notes": "fix it"}'),
        heterogeneous=True)
    from localm.plugins.coder.cli import repl as repl_mod
    warnings = []
    prints = []
    monkeypatch.setattr(repl_mod, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(repl_mod.console, "print", lambda msg="": prints.append(msg))
    repl_mod._handle_command("/review", agent)
    assert warnings and "1 blocking issue" in warnings[0]
    assert "separate reviewer model" in warnings[0]
    assert any("off-by-one in loop" in p for p in prints)
    assert any("fix it" in p for p in prints)


def test_review_surfaces_a_failed_review_as_a_warning(home, tmp_path, monkeypatch):
    """A crashed/unparseable review must read as a warning, not a silent
    approval - the same fail-open-but-visible contract as the automatic pass
    (test_reviewer.py's test_reviewer_fails_open_on_backend_error)."""
    agent = _agent(_proj(tmp_path))
    monkeypatch.setattr(agent, "session_diff", lambda *a, **k: "diff text")
    b = MagicMock()
    b.chat.side_effect = RuntimeError("backend down")
    agent._reviewer = Reviewer(b)
    from localm.plugins.coder.cli import repl as repl_mod
    warnings = []
    successes = []
    monkeypatch.setattr(repl_mod, "print_warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(repl_mod, "print_success", lambda msg: successes.append(msg))
    repl_mod._handle_command("/review", agent)
    assert not successes
    assert warnings and "self-review did NOT run" in warnings[0]
    assert "backend down" in warnings[0]
