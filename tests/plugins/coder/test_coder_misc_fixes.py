# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder fixes: blank diff context lines, undo after a failed edit, CRLF diffs and
patch files, Windows glob scope, doctor lines, Ctrl+C mid-reply, terminal approval
of a call emitted beside a read."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import click
from localm.plugins.coder.audit import SessionMode
import pytest

from localm.plugins.coder._patch import PatchError, apply_diff
from tests.plugins.coder.test_agent_qol import _make_agent, _make_call, _tc


# ---- CODB-015: an empty line in a hunk is a blank context line ----------

_ORIG = "a = 1\n\nb = 2\n"


def test_blank_context_line_written_as_empty_line_applies():
    diff = "@@ -1,3 +1,3 @@\n a = 1\n\n-b = 2\n+b = 3\n"
    assert apply_diff(_ORIG, diff) == "a = 1\n\nb = 3\n"


def test_blank_context_line_written_with_a_space_still_applies():
    diff = "@@ -1,3 +1,3 @@\n a = 1\n \n-b = 2\n+b = 3\n"
    assert apply_diff(_ORIG, diff) == "a = 1\n\nb = 3\n"


def test_trailing_empty_lines_after_the_last_hunk_are_not_context():
    diff = "@@ -3 +3 @@\n-b = 2\n+b = 3\n\n\n"
    assert apply_diff(_ORIG, diff) == "a = 1\n\nb = 3\n"


def test_empty_line_diff_agrees_with_git_apply(tmp_path):
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a = 1\n\n-b = 2\n+b = 3\n"
    (tmp_path / "f.py").write_bytes(_ORIG.encode())
    (tmp_path / "p.diff").write_bytes(diff.encode())
    out = subprocess.run(["git", "apply", "--check", "p.diff"], cwd=tmp_path,
                         capture_output=True)
    assert out.returncode == 0
    assert apply_diff(_ORIG, diff) == "a = 1\n\nb = 3\n"


def test_a_real_mismatch_still_raises():
    with pytest.raises(PatchError):
        apply_diff(_ORIG, "@@ -1,3 +1,3 @@\n zzz\n\n-b = 2\n+b = 3\n")


# ---- CODA-081: a failed edit leaves no undo entry -----------------------

def test_failed_edit_file_leaves_no_undo_entry(tmp_path):
    (tmp_path / "a.txt").write_text("one\n", encoding="utf-8")
    agent = _make_agent(tmp_path, [
        _tc("edit_file", path="a.txt", old_string="one", new_string="two"),
        _tc("edit_file", path="a.txt", old_string="ABSENT", new_string="x"),
        "Done."])
    agent.run_task("edit twice")
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "two\n"
    assert len(agent._undo_stack) == 1
    agent.undo()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert agent.undo() is None


# ---- CODA-043: /diff of a CRLF file shows only the change ---------------

def test_session_diff_of_a_crlf_file_shows_only_the_change(tmp_path):
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\nz = 3\r\n")
    agent = _make_agent(tmp_path, [
        _tc("edit_file", path="a.py", old_string="y = 2", new_string="y = 9"),
        "Done."])
    agent.run_task("change y")
    diff = agent.session_diff()
    changed = [l for l in diff.splitlines() if l[:1] in "+-" and l[:3] not in ("---", "+++")]
    assert changed == ["-y = 2", "+y = 9"]


# ---- CODA-030: a patch file is written without newline translation ------

def test_flush_patch_to_file_keeps_lf(tmp_path):
    agent = _make_agent(tmp_path, ["Done."])
    agent._patch_chunks.append("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n+b\n")
    out = tmp_path / "out.patch"
    agent.flush_patch(out)
    assert b"\r" not in out.read_bytes()


def test_main_writes_the_patch_file_without_crlf_source_pin():
    from localm.plugins.coder.cli import _main
    src = open(_main.__file__, encoding="utf-8").read()
    assert 'out.write_text(patch_content, encoding="utf-8", newline="")' in src


# ---- CODA-020/021: the localm group does not expand globs for coder -----

def test_root_group_does_not_expand_windows_globs_for_coder():
    from localm.cli._core import main as root
    seen = {}

    def fake_main(self, args=None, **extra):
        seen.update(extra)
        return 0

    with patch.object(click.Group, "main", fake_main):
        root.main(["coder", "--scope", "src/**/*.py"], standalone_mode=False)
        assert seen.get("windows_expand_args") is False
        seen.clear()
        root.main(["list"], standalone_mode=False)
        assert "windows_expand_args" not in seen


# ---- CODA-003: doctor with no -m does not refuse ------------------------

def test_doctor_without_a_model_does_not_demand_a_registered_one(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from localm.plugins.coder.cli import _main
    home = tmp_path / "home"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import PluginManager
    PluginManager(None).set_installed_state("coder", True)
    proj = tmp_path / "proj"
    proj.mkdir()
    res = CliRunner().invoke(_main.main, ["-c", str(proj), "-p", "1", "--no-server", "doctor"])
    assert "is not registered" not in res.output


# ---- CODA-048-2 / doctor tool-calls line --------------------------------

def test_doctor_verify_off_only_when_turned_off(tmp_path):
    from localm.plugins.coder import doctor
    agent = _make_agent(tmp_path, ["Done."])
    assert agent.verify_off is False
    c = doctor.check_verify(tmp_path, None, verify_off=agent.verify_off and agent.verify_cmd is None)
    assert "is off" not in c.detail


def test_doctor_tool_calls_line_says_the_grammar_is_lazy():
    from localm.plugins.coder import doctor
    from localm.plugins.coder.backends.http import make_localm_backend
    c = doctor.check_tools(make_localm_backend("m"))
    assert "opening tag" in c.detail and "grammar-constrained by" not in c.detail


# ---- CODA-052: Ctrl+C mid-reply reaches the run loop ---------------------

def test_ctrl_c_mid_reply_saves_a_checkpoint_and_ends_the_run(tmp_path):
    agent = _make_agent(tmp_path, ["partial"], mode=SessionMode.LOG)

    def stream(messages, **kw):
        yield "partial "
        raise KeyboardInterrupt()

    agent.backend.chat_stream = stream
    with patch("localm.plugins.coder.agent.context.print_reasoning_token"), \
         patch("localm.plugins.coder.agent.context.print_streaming_token"), \
         patch.object(type(agent), "save_checkpoint", return_value=True) as saved:
        with pytest.raises(KeyboardInterrupt):
            agent.run_task("long essay", interactive=True)
    assert saved.called


# ---- CODA-042: a gated call beside a read asks at the terminal ----------

def test_run_tests_beside_read_file_is_asked_not_denied(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    agent = _make_agent(tmp_path, ["Done."], auto_approve=False)
    asked = []

    def fake_confirm(call):
        asked.append(call.name)
        return False

    agent._confirm_tool = fake_confirm
    assert agent.confirm_handler is None
    calls = [_make_call("read_file", path="a.py"), _make_call("run_tests")]
    blocks = agent._execute_tools(calls, interactive=True)
    assert "run_tests" in asked
    assert not any("non-interactive with no approval handler" in b for b in blocks)
