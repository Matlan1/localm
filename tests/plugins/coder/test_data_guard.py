"""An owner coder cannot reach localm's own keys or settings unapproved.

Under auto_approve (``--yes``, the GUI toggle, an unattended task) the coder's
shell and file tools used to delete auth.key or run ``localm key clear`` with no
one asked. A call reaching localm's credentials or settings now needs an
explicit approval of that exact call, and is denied when no one can be asked.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from localm import auth
from localm.config import home_dir
from localm.plugins.coder import data_guard
from localm.plugins.coder.agent import Agent
from localm.plugins.coder.parser import ToolCall
from localm.plugins.coder.tools import base as tools_base

STRONG_KEY = "w1-strong-test-key-0123456789abcdef"
_MARKER = "localm-data-guard-marker"


class _StubBackend:
    model_id = "stub-model"
    native_tools = False

    def set_tools(self, defs):
        pass


def _call(name: str, **args) -> ToolCall:
    return ToolCall(name=name, args=args, raw="", start=0, end=0)


def _rm(path: Path) -> str:
    return f'del "{path}"' if sys.platform == "win32" else f"rm '{path}'"


@pytest.fixture
def keyed(monkeypatch, tmp_path):
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    auth.set_api_key(STRONG_KEY)
    work = tmp_path / "proj"
    work.mkdir()
    return work


# --------------------------------------------------------------------------- #
#  data_guard.check                                                             #
# --------------------------------------------------------------------------- #

def _shell(cmd: str, cwd: Path, home: Path):
    return data_guard.check("run_shell", {"command": cmd}, cwd,
                            shell_args=("command",), home=home)


@pytest.mark.parametrize("cmd_t, label", [
    ("rm {h}/auth.key", "owner API key"),
    ("rm -f {h}/auth.json", "API key store"),
    ("del {hw}\\auth.key", "owner API key"),
    ("Remove-Item -Force {hw}\\sessions.json", "browser session store"),
    ("cat {h}/config.json", "settings"),
    ("type {hw}\\config.json.bak", "settings backup"),
    ("echo x > {h}/auth.kdf.json", "owner key records"),
    ("cp evil.json {h}/model_source_credentials.json", "model-source credentials"),
    ("rm -rf {h}/run", "running-instance registry"),
    ("cd {h} && rm auth.key", "owner API key"),
    ("rm -rf {h}", "owner API key"),
    ("mv {h} /tmp/elsewhere", "owner API key"),
    ("localm key clear --yes", "localm key"),
    ("echo y | localm key clear", "localm key"),
    ("localm key rm abc123 --yes", "localm key"),
    ("localm key show --reveal", "localm key"),
    ("python -m localm key generate", "localm key"),
    ("uv run localm config require_auth false", "localm config"),
    ("localm --debug config bind_host 0.0.0.0", "localm config"),
])
def test_protected_shell_commands_are_named(tmp_path, cmd_t, label):
    home = tmp_path / "home"
    cmd = cmd_t.format(h=home.as_posix(), hw=str(home).replace("/", "\\"))
    got = _shell(cmd, tmp_path / "proj", home)
    assert got is not None and label in got, (cmd, got)


@pytest.mark.parametrize("cmd", [
    "rm -rf build",
    "pytest tests/test_x.py -q",
    "localm key list",
    "localm status",
    "localm config",
    "git status",
    "cat README.md",
    "ls {h}",
    "grep -r hf_ {h}/models",
])
def test_ordinary_commands_are_not_flagged(tmp_path, cmd):
    home = tmp_path / "home"
    assert _shell(cmd.format(h=home.as_posix()), tmp_path / "proj", home) is None


def test_relative_path_into_the_data_folder_from_its_parent(tmp_path):
    home = tmp_path / "home"
    assert _shell("rm home/auth.key", tmp_path, home) is not None
    assert _shell("rm ../home/auth.key", tmp_path / "proj", home) is not None


def test_case_is_ignored(tmp_path):
    home = tmp_path / "Home"
    assert _shell(f"rm {home.as_posix().upper()}/AUTH.KEY", tmp_path, home) is not None


@pytest.mark.parametrize("name, args", [
    ("write_file", {"path": "{h}/auth.key"}),
    ("edit_file", {"path": "{h}/config.json"}),
    ("read_file", {"path": "{h}/auth.json"}),
    ("read_env", {"path": "{h}/model_source_credentials.json"}),
    ("search_replace", {"glob": "{h}/**/*.json"}),
])
def test_protected_file_tool_paths_are_named(tmp_path, name, args):
    home = tmp_path / "home"
    a = {k: v.format(h=home.as_posix()) for k, v in args.items()}
    assert data_guard.check(name, a, tmp_path, path_args=tuple(a), home=home)


def test_a_project_wide_search_replace_is_not_flagged(tmp_path):
    home = tmp_path / "home"
    for glob in ("**/*", "*.py", "./**/*.json"):
        assert data_guard.check("search_replace", {"glob": glob}, tmp_path,
                                path_args=("glob",), home=home) is None, glob
    assert data_guard.check("search_replace", {"glob": "home/**/*"}, tmp_path,
                            path_args=("glob",), home=home)


def test_patch_file_headers_are_checked(tmp_path):
    home = tmp_path / "home"
    diff = f"--- a/{home.as_posix()}/config.json\n+++ b/{home.as_posix()}/config.json\n"
    assert data_guard.check("patch_file", {"path": "x.py", "diff": diff}, tmp_path,
                            path_args=("path",), home=home)


def test_a_symlink_into_the_data_folder_is_resolved(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "auth.key").write_text("k", encoding="utf-8")
    link = tmp_path / "innocent.txt"
    try:
        link.symlink_to(home / "auth.key")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available to this account")
    assert data_guard.check("read_file", {"path": "innocent.txt"}, tmp_path,
                            path_args=("path",), home=home)


def test_ordinary_file_paths_are_not_flagged(tmp_path):
    home = tmp_path / "home"
    for path in ("src/main.py", "config.json", "auth.key", f"{home.as_posix()}/models/x.gguf"):
        assert data_guard.check("write_file", {"path": path}, tmp_path,
                                path_args=("path",), home=home) is None, path


# --------------------------------------------------------------------------- #
#  The agent dispatch path                                                      #
# --------------------------------------------------------------------------- #

def test_auto_approved_shell_cannot_delete_the_owner_key(keyed):
    key = home_dir() / "auth.key"
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    cmd = f"echo {_MARKER} > ran.txt && {_rm(key)}"
    result = agent._execute_tool(_call("run_shell", command=cmd), interactive=False)
    assert key.read_text(encoding="utf-8").strip() == STRONG_KEY
    assert not (keyed / "ran.txt").exists(), "the command ran"
    assert not result.ok
    assert "localm's owner API key" in result.output
    assert "denied" in result.output


def test_auto_approved_shell_cannot_run_localm_key_clear(keyed):
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    cmd = f"echo {_MARKER} > ran.txt && localm key clear --yes"
    result = agent._execute_tool(_call("run_shell", command=cmd), interactive=False)
    assert not (keyed / "ran.txt").exists(), "the command ran"
    assert not result.ok
    assert "localm key" in result.output
    assert auth.get_api_key() == STRONG_KEY


def test_auto_approved_read_does_not_return_the_key(keyed):
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    result = agent._execute_tool(
        _call("read_file", path=str(home_dir() / "auth.key")), interactive=False)
    assert not result.ok
    assert STRONG_KEY not in result.output


def test_auto_approved_write_cannot_change_the_settings(keyed):
    cfg = home_dir() / "config.json"
    cfg.write_text('{"require_auth": true}', encoding="utf-8")
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    result = agent._execute_tool(
        _call("write_file", path=str(cfg), content='{"require_auth": false}'),
        interactive=False)
    assert not result.ok
    assert cfg.read_text(encoding="utf-8") == '{"require_auth": true}'


def test_the_approval_handler_is_asked_for_that_exact_call(keyed):
    key = home_dir() / "auth.key"
    asked = []

    def deny(call, **_kw):
        asked.append((call.name, dict(call.args)))
        return False

    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set(),
                  confirm_handler=deny)
    cmd = _rm(key)
    result = agent._execute_tool(_call("run_shell", command=cmd), interactive=False)
    assert asked == [("run_shell", {"command": cmd})]
    assert not result.ok
    assert key.exists()


def test_an_approved_call_runs(keyed):
    key = home_dir() / "auth.key"
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set(),
                  confirm_handler=lambda call, **_kw: True)
    result = agent._execute_tool(_call("run_shell", command=_rm(key)), interactive=False)
    assert result.ok, result.output
    assert not key.exists()


def test_ordinary_work_still_runs_unasked(keyed):
    asked = []
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set(),
                  confirm_handler=lambda call, **_kw: asked.append(call) or True)
    result = agent._execute_tool(
        _call("run_shell", command=f"echo {_MARKER} > ran.txt"), interactive=False)
    assert result.ok, result.output
    assert (keyed / "ran.txt").exists()
    assert asked == []


def test_a_failing_check_still_requires_approval(keyed, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("probe failure")
    monkeypatch.setattr(data_guard, "check", boom)
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    result = agent._execute_tool(
        _call("run_shell", command=f"echo {_MARKER} > ran.txt"), interactive=False)
    assert not result.ok
    assert not (keyed / "ran.txt").exists()


def test_shell_subprocesses_get_no_stdin(monkeypatch, tmp_path):
    seen = {}
    real = subprocess.Popen

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(tools_base.subprocess, "Popen", spy)
    r = tools_base.run_subprocess([sys.executable, "-c", "print(1)"], tmp_path, timeout=60)
    assert r.ok
    assert seen.get("stdin") is subprocess.DEVNULL
