# SPDX-License-Identifier: AGPL-3.0-or-later
"""REPL commands: /help lists every command, /save and /export write where they
say and ask in privacy mode, /export leaves the audit log open, and /goal runs
its iterations as terminal turns."""

from __future__ import annotations

import ast
import inspect
import json
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from localm.plugins.coder.audit import SessionMode


@pytest.fixture(autouse=True)
def _trusted_project(monkeypatch):
    monkeypatch.setattr("localm.plugins.coder.hook_trust.project_trusted",
                        lambda cwd: True)


@pytest.fixture
def home(tmp_path, monkeypatch):
    import localm.config as cfg
    h = tmp_path / "home"
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    return h


class _Scripted:
    """A backend that answers with the next reply of *script*, then "Done."."""
    model_id = "m"
    native_tools = False
    supports_grammar = False
    last_usage = {"total_tokens": 0}

    def __init__(self, script=()):
        self.script = list(script)

    def set_tools(self, defs):
        pass

    def _next(self):
        return self.script.pop(0) if self.script else "Done."

    def chat(self, messages, **kw):
        return self._next()

    def chat_stream(self, messages, **kw):
        yield self._next()


def _agent(cwd, script=(), **kw):
    from localm.plugins.coder.agent import Agent
    with patch("localm.plugins.coder.agent.ProjectMap") as PM, \
         patch("localm.plugins.coder.agent.make_audit_log"), \
         patch("localm.plugins.coder.agent.load_memory", return_value=""):
        PM.build.return_value.file_count.return_value = 0
        PM.build.return_value.truncated = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Scripted(script), cwd=cwd, self_verify=False, **kw)


def _proj(tmp_path):
    p = tmp_path / "proj"
    p.mkdir()
    return p


def _rendered(text: str) -> str:
    console = Console(record=True, width=200, highlight=False)
    console.print(text)
    return console.export_text()


def _handled_commands() -> set[str]:
    """Every command name _handle_command compares ``cmd`` against."""
    from localm.plugins.coder.cli import repl as repl_mod
    tree = ast.parse(inspect.getsource(repl_mod._handle_command))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id == "cmd"):
            continue
        for comp in node.comparators:
            if isinstance(comp, ast.Constant) and isinstance(comp.value, str):
                names.add(comp.value)
            elif isinstance(comp, (ast.Tuple, ast.List, ast.Set)):
                names.update(e.value for e in comp.elts
                             if isinstance(e, ast.Constant) and isinstance(e.value, str))
    return names


class TestHelpListsEveryCommand:
    def test_every_handled_command_is_in_help(self):
        help_text = _rendered(__import__(
            "localm.plugins.coder.display", fromlist=["HELP_TEXT"]).HELP_TEXT)
        handled = _handled_commands()
        assert "bg" in handled and "bye" in handled
        missing = sorted(c for c in handled if f"/{c}" not in help_text)
        assert missing == []

    def test_every_completion_entry_is_handled_and_in_help(self):
        from localm.plugins.coder.cli.repl import _SLASH_COMMANDS
        from localm.plugins.coder.display import HELP_TEXT
        help_text = _rendered(HELP_TEXT)
        handled = _handled_commands()
        assert sorted(c for c in _SLASH_COMMANDS if c[1:] not in handled) == []
        assert sorted(c for c in _SLASH_COMMANDS if c not in help_text) == []


class TestSave:
    def test_default_path_is_in_the_project(self, home, tmp_path, monkeypatch):
        proj = _proj(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        agent = _agent(proj)
        from localm.plugins.coder.cli import repl as repl_mod
        repl_mod._handle_command("/save", agent)
        assert (proj / "conversation.json").is_file()
        assert not (elsewhere / "conversation.json").exists()

    def test_missing_folder_is_created(self, home, tmp_path):
        proj = _proj(tmp_path)
        agent = _agent(proj)
        from localm.plugins.coder.cli import repl as repl_mod
        repl_mod._handle_command("/save sub/new.json", agent)
        saved = proj / "sub" / "new.json"
        assert isinstance(json.loads(saved.read_text(encoding="utf-8")), list)


class TestExport:
    def test_named_path_asks_in_privacy_mode(self, home, tmp_path, monkeypatch):
        proj = _proj(tmp_path)
        agent = _agent(proj, mode=SessionMode.PRIVACY)
        from localm.plugins.coder.cli import repl as repl_mod
        asked = []
        monkeypatch.setattr(repl_mod, "confirm",
                            lambda prompt: asked.append(prompt) or False)
        repl_mod._handle_command("/export notes.md", agent)
        assert len(asked) == 1 and "notes.md" in asked[0]
        assert not (proj / "notes.md").exists()
        assert not (proj / ".localcoder" / "sessions").exists()

    def test_export_leaves_the_audit_log_open(self, home, tmp_path):
        proj = _proj(tmp_path)
        agent = _agent(proj)
        agent._audit = MagicMock()
        from localm.plugins.coder.cli import repl as repl_mod
        repl_mod._handle_command("/export out/notes.md", agent)
        assert (proj / "out" / "notes.md").is_file()
        agent._audit.close.assert_not_called()


class TestRememberAndSystemText:
    def test_remembering_a_note_already_there_says_so(self, home, tmp_path, monkeypatch):
        proj = _proj(tmp_path)
        (proj / "LOCALCODER.md").write_text("# Project Memory\n\n- use tabs\n",
                                            encoding="utf-8")
        agent = _agent(proj)
        from localm.plugins.coder.cli import repl as repl_mod
        said = []
        monkeypatch.setattr(repl_mod, "print_success", lambda m: said.append(("ok", m)))
        monkeypatch.setattr(repl_mod, "print_info", lambda m: said.append(("info", m)))
        repl_mod._handle_command("/remember use tabs", agent)
        assert said and said[0][0] == "info" and "Already in" in said[0][1], said
        repl_mod._handle_command("/remember run ruff", agent)
        assert said[-1][0] == "ok" and "Remembered" in said[-1][1], said

    def test_an_over_long_system_flag_is_not_called_a_file(self, tmp_path):
        from localm.plugins.coder.memory import (
            _MAX_CUSTOM_INSTRUCTIONS_CHARS,
            cap_user_instructions,
            custom_instructions_warning,
        )
        text = "x " * _MAX_CUSTOM_INSTRUCTIONS_CHARS
        warning = custom_instructions_warning(tmp_path, override=text)
        assert "--system" in warning and "file" not in warning
        assert "whole file" not in cap_user_instructions(text)


class TestGoalInTheRepl:
    def test_a_write_is_asked_at_the_terminal_and_runs(self, home, tmp_path,
                                                       monkeypatch):
        proj = _proj(tmp_path)
        call = {"name": "write_file", "args": {"path": "a.txt", "content": "hi\n"}}
        agent = _agent(proj, script=["<tool_call>" + json.dumps(call) + "</tool_call>"],
                       auto_approve=False)
        agent.goal_cmd = "check"
        import localm.plugins.coder.agent as agent_pkg
        import localm.plugins.coder.cli as cli_pkg
        asked = []
        monkeypatch.setattr(agent_pkg, "confirm", lambda *a, **k: asked.append(a) or True)
        monkeypatch.setattr(agent_pkg, "confirm_diff",
                            lambda *a, **k: asked.append(a) or True)
        monkeypatch.setattr(cli_pkg, "_run_verify", lambda cmd, cwd: (0, ""))
        from localm.plugins.coder.cli import repl as repl_mod
        inputs = iter(["write the file", EOFError()])

        def fake_input(prompt=""):
            nxt = next(inputs)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        monkeypatch.setattr(repl_mod.console, "input", fake_input)
        repl_mod._repl(agent)
        assert asked, "the write was not asked at the terminal"
        assert (proj / "a.txt").read_text(encoding="utf-8") == "hi\n"
