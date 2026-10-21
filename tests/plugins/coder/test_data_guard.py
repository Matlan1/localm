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
    ("cd {h} && rm auth.key", "data folder"),
    ("rm -rf {h}", "data folder"),
    ("mv {h} /tmp/elsewhere", "data folder"),
    ("rm -rf {h}/..", "owner API key"),
    ("localm key clear --yes", "localm key"),
    ("echo y | localm key clear", "localm key"),
    ("localm key rm abc123 --yes", "localm key"),
    ("localm key show --reveal", "localm key"),
    ("python -m localm key generate", "localm key"),
    ("uv run localm config require_auth false", "localm config"),
    ("localm --debug config bind_host 0.0.0.0", "localm config"),
    ("ls {h}", "data folder"),
    ("cat {h}/*", "data folder"),
    ("grep -r . {h}", "data folder"),
    ("tar czf out.tgz {h}", "data folder"),
    ("cat {h}/tls/ca.key", "CA private key"),
    ("cp {h}/tls/server.key /tmp/k", "TLS server private key"),
    ('bash -c "rm {h}/auth.key"', "owner API key"),
    ("sh -c 'cat {h}/config.json'", "settings"),
    ("cmd /c del {hw}\\auth.key", "owner API key"),
    ("powershell -Command Remove-Item {hw}\\auth.json", "API key store"),
    ('bash -c "localm key clear --yes"', "localm key"),
    ("rm $LOCALM_HOME/auth.key", "owner API key"),
    ("rm ${{LOCALM_HOME}}/auth.key", "owner API key"),
    ("del %LOCALM_HOME%\\auth.key", "owner API key"),
    ("cat {h}/auth.json.", "API key store"),
    ("type {hw}\\auth.key::$DATA", "owner API key"),
    ("pytest --basetemp={h}", "data folder"),
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
    "grep -r hf_ {h}/models",
    "ls {h}/models",
    'bash -c "pytest -q"',
    "rm $OTHER_VAR/auth.key",
])
def test_ordinary_commands_are_not_flagged(tmp_path, cmd):
    home = tmp_path / "home"
    assert _shell(cmd.format(h=home.as_posix()), tmp_path / "proj", home) is None


def test_a_continued_or_cmd_quoted_command_is_checked(tmp_path):
    home = tmp_path / "home"
    bs = chr(92)
    assert _shell("cat " + bs + "\n" + home.as_posix() + "/auth.key",
                  tmp_path / "proj", home)
    assert _shell("echo 'x & type " + str(home) + bs + "auth.key & '",
                  tmp_path / "proj", home)


def test_the_hook_trust_store_is_protected(tmp_path):
    home = tmp_path / "home"
    label = data_guard.check("write_file", {"path": f"{home.as_posix()}/coder-hook-trust.json"},
                             tmp_path, path_args=("path",), home=home)
    assert label is not None and "trust store" in label
    assert _shell(f"rm {home.as_posix()}/coder-hook-trust.json", tmp_path, home)


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
    """The data folder inside the project (a portable home in a repo): file
    tools reach it, so only the data guard stands in the way."""
    project = home_dir().parent
    agent = Agent(_StubBackend(), cwd=project, auto_approve=True, always_confirm=set())
    result = agent._execute_tool(
        _call("read_file", path=str(home_dir() / "auth.key")), interactive=False)
    assert not result.ok
    assert STRONG_KEY not in result.output


def test_auto_approved_write_cannot_change_the_settings(keyed):
    cfg = home_dir() / "config.json"
    cfg.write_text('{"require_auth": true}', encoding="utf-8")
    project = home_dir().parent
    agent = Agent(_StubBackend(), cwd=project, auto_approve=True, always_confirm=set())
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


# --------------------------------------------------------------------------- #
#  run_tests, the sweep tools and the GUI approval                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("args", [
    {"path": "{h}/auth.key"},
    {"path": "tests", "extra_args": "--basetemp={h}"},
    {"path": "tests", "extra_args": "--basetemp {h}"},
])
def test_run_tests_paths_and_flag_values_are_checked(keyed, args):
    h = home_dir().as_posix()
    a = {k: v.format(h=h) for k, v in args.items()}
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    assert agent._data_guard_verdict(_call("run_tests", **a)), a


def test_ordinary_run_tests_is_not_flagged(keyed):
    agent = Agent(_StubBackend(), cwd=keyed, auto_approve=True, always_confirm=set())
    assert agent._data_guard_verdict(
        _call("run_tests", path="tests", extra_args="-q -k smoke")) is None


def test_a_project_wide_grep_does_not_read_the_key(keyed):
    from localm.plugins.coder.tools.files import tool_grep
    project = home_dir().parent
    (project / "notes.txt").write_text(STRONG_KEY[:12] + " in notes\n", encoding="utf-8")
    r = tool_grep(project, STRONG_KEY[:12])
    assert r.ok, r.output
    assert "notes.txt" in r.output
    assert STRONG_KEY not in r.output
    assert "auth.key" not in r.output.split("[not searched")[0]
    assert "localm's own keys or settings" in r.output


def test_a_project_wide_search_replace_leaves_the_settings_alone(keyed):
    from localm.plugins.coder.tools.files import tool_search_replace
    project = home_dir().parent
    cfg = home_dir() / "config.json"
    cfg.write_text('{"require_auth": true}', encoding="utf-8")
    (project / "app.json").write_text('{"require_auth": true}', encoding="utf-8")
    r = tool_search_replace(project, "true", "false", glob="**/*.json")
    assert r.ok, r.output
    assert (project / "app.json").read_text(encoding="utf-8") == '{"require_auth": false}'
    assert cfg.read_text(encoding="utf-8") == '{"require_auth": true}'
    assert "localm's own keys or settings" in r.output


def test_protected_file(tmp_path):
    home = tmp_path / "home"
    assert data_guard.protected_file(home / "auth.key", home)
    assert data_guard.protected_file(home / "run" / "abc.json", home)
    assert data_guard.protected_file(home / "models" / "x.gguf", home) is None
    assert data_guard.protected_file(tmp_path / "auth.key", home) is None


def _gui_session(cwd):
    from localm.plugins.coder.sessions import CoderSession
    return CoderSession(cwd, _StubBackend(), auto_approve=True, auto_verify=False)


def _answer_when_asked(session, approved, always_allow=False):
    import threading

    def _run():
        waiter = threading.Event()
        for _ in range(1000):
            with session._lock:
                pending = list(session._pending)
            for cid in pending:
                session.answer_confirm(cid, approved, always_allow=always_allow)
            if pending:
                return
            waiter.wait(0.01)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def test_a_gui_always_allow_does_not_cover_the_data_folder(keyed):
    key = home_dir() / "auth.key"
    session = _gui_session(keyed)
    try:
        session.allowed_tools.add("run_shell")
        t = _answer_when_asked(session, approved=False)
        result = session.agent._execute_tool(_call("run_shell", command=_rm(key)),
                                             interactive=False)
        t.join(10)
        asked = [e for e in session.history if e.get("type") == "confirm_request"]
        assert len(asked) == 1, "the always-allow grant answered for the data folder"
        assert not result.ok
        assert key.exists()
    finally:
        session.close()


def test_a_gui_always_allow_answer_is_not_kept_for_the_data_folder(keyed):
    key = home_dir() / "auth.key"
    session = _gui_session(keyed)
    try:
        t = _answer_when_asked(session, approved=True, always_allow=True)
        result = session.agent._execute_tool(_call("run_shell", command=_rm(key)),
                                             interactive=False)
        t.join(10)
        assert result.ok, result.output
        assert not key.exists()
        assert "run_shell" not in session.allowed_tools
    finally:
        session.close()


def test_a_gui_always_allow_still_covers_ordinary_calls(keyed):
    session = _gui_session(keyed)
    try:
        session.allowed_tools.add("run_shell")
        assert session._confirm(_call("run_shell", command="echo hi")) is True
        assert not [e for e in session.history if e.get("type") == "confirm_request"]
    finally:
        session.close()
