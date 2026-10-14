# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder fixes from the 0.3.0 audit.

* A restricted write call walks the project for protected links once, however
  many paths the call names.
* A command the coder runs for the user sees the caches and Hugging Face login
  it would see outside localm, not the locations localm pinned for itself.
"""

import json
import os
import sys
from unittest.mock import patch

import pytest

from localm.audit import SessionMode
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import base as tools_base


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
        PM.build.return_value.dirty = False
        kw.setdefault("mode", SessionMode.LOG)
        kw.setdefault("auto_approve", True)
        return Agent(_Stub(), cwd=cwd, self_verify=False, **kw)


def _call(name, **args):
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _count_walks():
    calls = []
    real = tools_base.unwritable_redirects

    def counting(cwd_resolved):
        calls.append(cwd_resolved)
        return real(cwd_resolved)

    return calls, counting


class TestRestrictedWritesWalkTheProjectOnce:
    def _project(self, tmp_path, n=5):
        root = tmp_path / "proj"
        (root / "src").mkdir(parents=True)
        for i in range(n):
            (root / "src" / f"f{i}.txt").write_text("old\n", encoding="utf-8")
        return root

    def test_a_multi_path_edit_walks_once(self, tmp_path):
        root = self._project(tmp_path)
        agent = _agent(root, restricted=True)
        calls, counting = _count_walks()
        edits = [{"path": f"src/f{i}.txt", "old": "old", "new": "new"}
                 for i in range(5)]
        with patch.object(tools_base, "unwritable_redirects", counting):
            result = agent._execute_tool(_call("edit_files", edits=edits),
                                         interactive=False)
        assert result.ok, result.output
        assert len(calls) == 1
        for i in range(5):
            assert (root / "src" / f"f{i}.txt").read_text(encoding="utf-8") == "new\n"

    def test_each_call_still_walks_afresh(self, tmp_path):
        root = self._project(tmp_path, n=2)
        agent = _agent(root, restricted=True)
        calls, counting = _count_walks()
        with patch.object(tools_base, "unwritable_redirects", counting):
            for i in range(2):
                agent._execute_tool(
                    _call("edit_file", path=f"src/f{i}.txt", old="old", new="new"),
                    interactive=False)
        assert len(calls) == 2

    def test_a_protected_path_among_the_paths_is_still_refused(self, tmp_path):
        root = self._project(tmp_path, n=2)
        (root / ".localcoder").mkdir()
        (root / ".localcoder" / "config.toml").write_text("mode = 1\n", encoding="utf-8")
        agent = _agent(root, restricted=True)
        result = agent._execute_tool(_call("edit_files", edits=[
            {"path": "src/f0.txt", "old": "old", "new": "new"},
            {"path": ".localcoder/config.toml", "old": "1", "new": "2"}]),
            interactive=False)
        assert result.ok is False and ".localcoder" in result.output
        assert (root / "src" / "f0.txt").read_text(encoding="utf-8") == "old\n"

    def test_a_walk_that_fails_refuses_every_path_of_the_call(self, tmp_path):
        root = self._project(tmp_path, n=3)
        agent = _agent(root, restricted=True)
        calls = []

        def failing(cwd_resolved):
            calls.append(cwd_resolved)
            raise PermissionError("denied")

        edits = [{"path": f"src/f{i}.txt", "old": "old", "new": "new"}
                 for i in range(3)]
        with patch.object(tools_base, "unwritable_redirects", failing):
            result = agent._execute_tool(_call("edit_files", edits=edits),
                                         interactive=False)
        assert result.ok is False
        assert "could not be checked" in result.output
        assert len(calls) == 1
        for i in range(3):
            assert (root / "src" / f"f{i}.txt").read_text(encoding="utf-8") == "old\n"

    def test_the_memo_is_shared_between_direct_checks(self, tmp_path):
        root = self._project(tmp_path, n=3)
        calls, counting = _count_walks()
        memo = {}
        with patch.object(tools_base, "unwritable_redirects", counting):
            for i in range(3):
                assert tools_base.restricted_write_refusal(
                    root, f"src/f{i}.txt", walk_memo=memo) is None
            assert tools_base.restricted_write_refusal(root, "src/f0.txt") is None
        assert len(calls) == 2


_PRINT_ENV = ("import json, os; print(json.dumps({k: os.environ.get(k) for k in "
              "('HF_HOME', 'HF_TOKEN_PATH', 'LOCALM_PINNED_HF_HOME', "
              "'AMD_COMGR_CACHE_DIR', 'LOCALM_PINNED_AMD_COMGR_CACHE_DIR', "
              "'MIOPEN_USER_DB_PATH', 'MESA_SHADER_CACHE_DIR')}))")


def _localm_pinned_environ(monkeypatch, tmp_path):
    """Make the process environment the one localm leaves behind after import."""
    import localm.config as cfg
    for var in ("HF_HOME", "HF_TOKEN_PATH", "HF_HUB_CACHE", "XDG_CACHE_HOME",
                "LOCALM_PINNED_HF_HOME", "AMD_COMGR_CACHE_DIR",
                "MIOPEN_USER_DB_PATH", "MESA_SHADER_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cfg, "cache_dir", lambda: tmp_path / "data" / "cache")
    for key, value in cfg.contained_gpu_cache_env(cfg.contained_hf_env()).items():
        if key not in os.environ or os.environ[key] != value:
            monkeypatch.setenv(key, value)
    assert os.environ["HF_HOME"] == os.environ["LOCALM_PINNED_HF_HOME"]
    return cfg


def _env_seen_by(result):
    assert result.ok, (result.stdout, result.stderr, result.error)
    return json.loads(result.stdout)


class TestUserCommandsSeeTheirOwnCaches:
    def test_pinned_values_do_not_reach_run_subprocess(self, monkeypatch, tmp_path):
        _localm_pinned_environ(monkeypatch, tmp_path)
        seen = _env_seen_by(tools_base.run_subprocess(
            [sys.executable, "-c", _PRINT_ENV], tmp_path, timeout=60))
        assert all(v is None for v in seen.values()), seen

    def test_cancellable_form_is_scrubbed_too(self, monkeypatch, tmp_path):
        _localm_pinned_environ(monkeypatch, tmp_path)
        seen = _env_seen_by(tools_base.run_subprocess(
            [sys.executable, "-c", _PRINT_ENV], tmp_path, timeout=60,
            cancel=lambda: False))
        assert all(v is None for v in seen.values()), seen

    def test_an_explicit_env_is_scrubbed_too(self, monkeypatch, tmp_path):
        _localm_pinned_environ(monkeypatch, tmp_path)
        env = dict(os.environ)
        env["HISTFILE"] = ""
        seen = _env_seen_by(tools_base.run_subprocess(
            [sys.executable, "-c", _PRINT_ENV], tmp_path, timeout=60, env=env))
        assert all(v is None for v in seen.values()), seen

    def test_a_background_shell_job_is_scrubbed(self, monkeypatch, tmp_path):
        _localm_pinned_environ(monkeypatch, tmp_path)
        from localm.plugins.coder.background import ShellJob
        job = ShellJob([sys.executable, "-c", _PRINT_ENV], tmp_path, label="env")
        assert job._proc.wait(timeout=60) == 0
        assert job._drain(10)
        out, _err, _dropped = job.output()
        seen = json.loads(out)
        assert all(v is None for v in seen.values()), seen

    def test_a_value_the_user_set_is_left_as_set(self, monkeypatch, tmp_path):
        import localm.config as cfg
        for var in ("LOCALM_PINNED_HF_HOME", "AMD_COMGR_CACHE_DIR"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
        monkeypatch.setenv("HF_TOKEN_PATH", str(tmp_path / "mine" / "tok"))
        monkeypatch.setenv("AMD_COMGR_CACHE_DIR", str(tmp_path / "comgr"))
        env = cfg.user_command_env()
        assert env["HF_HOME"] == str(tmp_path / "mine")
        assert env["HF_TOKEN_PATH"] == str(tmp_path / "mine" / "tok")
        assert env["AMD_COMGR_CACHE_DIR"] == str(tmp_path / "comgr")

    def test_a_token_path_the_user_chose_survives_the_pinned_home(
            self, monkeypatch, tmp_path):
        cfg = _localm_pinned_environ(monkeypatch, tmp_path)
        chosen = str(tmp_path / "elsewhere" / "token")
        monkeypatch.setenv("HF_TOKEN_PATH", chosen)
        env = cfg.user_command_env()
        assert "HF_HOME" not in env
        assert env["HF_TOKEN_PATH"] == chosen

    def test_localm_itself_still_runs_with_its_pins(self, monkeypatch, tmp_path):
        cfg = _localm_pinned_environ(monkeypatch, tmp_path)
        cfg.user_command_env()
        assert os.environ["HF_HOME"] == os.environ["LOCALM_PINNED_HF_HOME"]
        assert "AMD_COMGR_CACHE_DIR" in os.environ

    @pytest.mark.parametrize("base", [{}, {"PATH": "x"}])
    def test_an_environment_without_pins_is_unchanged(self, base):
        import localm.config as cfg
        assert cfg.user_command_env(dict(base)) == base
