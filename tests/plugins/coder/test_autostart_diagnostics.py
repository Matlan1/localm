import pytest
# SPDX-License-Identifier: AGPL-3.0-or-later
"""When the coder auto-starts a `localm gui` server and it dies fast (a busy
--port, or any other quick non-zero exit), the CLI must surface the real reason
instead of waiting out the full attach timeout and reporting the generic
"Failed to attach to the auto-started server." The child's own refusal message
prints to its own console/process, which this process never reads."""

import socket
import subprocess
import sys

from click.testing import CliRunner

import localm.instances as instances
import localm.plugins.coder.cli as ccli


@pytest.fixture
def open_gate(monkeypatch):
    from localm import remotegate
    monkeypatch.setattr(remotegate, "require_off_machine_allowed",
                        lambda *a, **k: None)


def _bypass_plugin_gate(monkeypatch):
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)


def _no_existing_instance(monkeypatch):
    # No localm running for this project dir, so main() takes the auto-start branch.
    monkeypatch.setattr(instances, "resolve_root_dir", lambda **kw: "/tmp/proj")
    monkeypatch.setattr(instances, "attach_target", lambda *a, **kw: None)


class TestExplicitBusyPortFailsFast:
    def test_reports_real_reason_without_spawning(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _no_existing_instance(monkeypatch)

        # A real bound socket occupies the port.
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]
        try:
            def fail_if_spawned(*a, **kw):
                raise AssertionError(
                    "subprocess should not be spawned when the requested port "
                    "is already busy")
            monkeypatch.setattr("subprocess.Popen", fail_if_spawned)

            result = CliRunner().invoke(
                ccli.main,
                ["--model", "m", "--port", str(busy_port), "hi"])

            assert result.exit_code == 1
            assert f"Port {busy_port} is already in use" in result.output
            assert "Failed to attach" not in result.output
        finally:
            sock.close()


class _DeadProc:
    """Stand-in for a subprocess.Popen handle that has already exited."""

    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


class TestAutoStartedServerDiesFast:
    def test_reports_exit_code_instead_of_generic_timeout(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _no_existing_instance(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda s: None)
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _DeadProc(1))

        result = CliRunner().invoke(
            ccli.main, ["--model", "m", "hi"])

        assert result.exit_code == 1
        assert "exited immediately (exit code 1)" in result.output
        assert "Failed to attach to the auto-started server" not in result.output


class TestHeadlessAutoStartSurfacesChildError:
    """Driven headlessly (MCP's run_coder_task, CI, a script - stdin is not a
    TTY, exactly how CliRunner invokes main), the auto-start must not route the
    child's output into a NEW CONSOLE WINDOW nobody can see: the child's real
    refusal (e.g. gui's "Model not found: X") must reach THIS process's error
    output, and no console window may be requested."""

    def test_child_error_reaches_the_error_message(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _no_existing_instance(monkeypatch)
        monkeypatch.setattr("time.sleep", lambda s: None)
        seen = {}

        def fake_popen(cmd, **kw):
            seen.update(kw)
            # Write the refusal to the captured stdout, as the gui child does.
            target = kw.get("stdout")
            assert target is not None, (
                "headless spawn must capture the child's output (a new console "
                "window is invisible to an MCP/CI caller)")
            target.write("Model not found: m\n")
            return _DeadProc(1)

        monkeypatch.setattr("subprocess.Popen", fake_popen)

        result = CliRunner().invoke(ccli.main, ["--model", "m", "hi"])

        assert result.exit_code == 1
        assert "exited immediately (exit code 1)" in result.output
        assert "Model not found: m" in result.output
        assert "console window" not in result.output
        if sys.platform == "win32":
            assert not (seen.get("creationflags", 0)
                        & subprocess.CREATE_NEW_CONSOLE), (
                "a headless caller has no desktop for a new console window")
            assert seen.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW
            assert not seen.get("creationflags", 0) & subprocess.DETACHED_PROCESS


class TestAutoStartedServerRunsInTheProjectDirectory:
    """The server registers itself against ITS OWN working directory, and the
    attach that follows looks for an instance registered against the project
    directory. Spawning it without a cwd leaves it registered against whatever
    directory the CLI happened to be run from, so the two never match and every
    attach times out - taking the whole session with it and leaving the server
    it started running.

    That is every run started with --cwd, which includes the launcher's coder
    mode: the launcher runs from the install folder and passes the folder the
    user picked."""

    def test_the_server_is_spawned_in_the_directory_it_will_be_looked_up_by(
            self, monkeypatch, tmp_path):
        _bypass_plugin_gate(monkeypatch)
        project = tmp_path / "myproject"
        project.mkdir()
        monkeypatch.setattr(instances, "resolve_root_dir",
                            lambda **kw: str(project))
        monkeypatch.setattr(instances, "attach_target", lambda *a, **kw: None)
        monkeypatch.setattr("time.sleep", lambda s: None)
        seen = {}

        def fake_popen(cmd, **kw):
            seen.update(kw)
            return _DeadProc(1)

        monkeypatch.setattr("subprocess.Popen", fake_popen)
        CliRunner().invoke(ccli.main, ["--cwd", str(project), "--model", "m", "hi"])

        assert "cwd" in seen, (
            "the server was spawned with no working directory, so it registers "
            "against the caller's directory and the attach can never match")
        assert str(seen["cwd"]) == str(project), seen["cwd"]


class TestWhereAnAutoStartedServersOutputGoes:
    """The detached server must never share the stdout of a run whose stdout
    carries a result document or event lines, or a pipe someone else reads."""

    def test_the_table(self):
        from localm.plugins.coder.cli._main import _captures_child_output as cap
        assert cap(False, True, "linux", False)
        assert cap(False, True, "win32", False)
        assert cap(True, True, "linux", True)
        assert cap(True, False, "linux", False)
        assert cap(True, False, "darwin", False)
        assert not cap(True, True, "linux", False)
        assert not cap(True, True, "win32", False)
        assert not cap(True, False, "win32", False)
        assert not cap(True, True, "win32", True)

    def test_a_json_run_passes_machine_output_to_the_backend_builder(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        seen = {}

        def fake_build(*a, **kw):
            seen.update(kw)
            raise SystemExit(0)

        monkeypatch.setattr("localm.plugins.coder.cli._main._build_backend", fake_build)
        CliRunner().invoke(ccli.main, ["--model", "m", "--output-format", "json", "hi"])
        assert seen.get("machine_output") is True
        seen.clear()
        CliRunner().invoke(ccli.main, ["--model", "m", "hi"])
        assert seen.get("machine_output") is False


@pytest.mark.usefixtures("open_gate")
def test_a_backend_built_from_url_assumes_a_window_and_a_preset_does_not():
    import click

    from localm.plugins.coder.backends.http import make_openai_backend
    from localm.plugins.coder.cli._main import _build_backend

    with click.Context(ccli.main):
        by_url = _build_backend(None, "http://127.0.0.1:1/v1", "m", "k", False, None,
                                False, False, ".", False)
        preset = _build_backend("openai", None, "gpt-4o", "k", False, None,
                                False, False, ".", False)
    assert by_url.assume_ctx_window is True
    assert preset.assume_ctx_window is False
    assert make_openai_backend().assume_ctx_window is False


class _LiveProc:
    """Stand-in for a server that is still running."""

    returncode = None

    def poll(self):
        return None


def _flat(captured):
    """Captured stdout and stderr with all whitespace removed, so a wrapped
    console line still matches."""
    return "".join((captured.out + captured.err).split())


def _autostart_logs(directory):
    from localm.plugins.coder.cli._main import AUTOSTART_LOG_PREFIX
    return sorted(p.name for p in directory.iterdir()
                  if p.name.startswith(AUTOSTART_LOG_PREFIX))


def _headless_autostart(monkeypatch, tmp_path, popen, attach_after=None,
                        session_mode=None):
    """Run the auto-start branch of _build_backend with the output captured to
    a log in *tmp_path*. *attach_after* is how many polls pass before the new
    server is found (None: never)."""
    import click
    import tempfile

    from localm.plugins.coder.cli import _main

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(_main, "_captures_child_output", lambda *a: True)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setattr(instances, "resolve_root_dir", lambda **kw: str(tmp_path))
    calls = {"n": 0}

    def attach_target(*a, **kw):
        if kw.get("exclude_ids") is None:
            return None
        calls["n"] += 1
        if attach_after is not None and calls["n"] > attach_after:
            return {"base_url": "http://127.0.0.1:1", "port": 1, "token": None}
        return None

    monkeypatch.setattr(instances, "attach_target", attach_target)
    monkeypatch.setattr("subprocess.Popen", popen)
    work = tmp_path / "proj"
    work.mkdir(exist_ok=True)
    with click.Context(ccli.main):
        try:
            return _main._build_backend(None, None, "m", "k", False, None, False,
                                        False, work, False,
                                        session_mode=session_mode), None
        except SystemExit as exc:
            return None, exc


@pytest.mark.usefixtures("open_gate")
class TestTheAutoStartLogIsNotLeftBehind:
    """The temp file a headless auto-start sends the server's output to must
    not stay on disk: removed after an early exit's tail is read, removed once
    the attach succeeds (while the server still runs), and kept after a failed
    attach only outside privacy mode or with keep diagnostics on."""

    def test_an_early_exit_reports_the_output_and_removes_the_file(
            self, monkeypatch, tmp_path, capsys):
        def popen(cmd, **kw):
            kw["stdout"].write("Model not found: m\n")
            return _DeadProc(1)

        _, exc = _headless_autostart(monkeypatch, tmp_path, popen)
        assert exc is not None and exc.code == 1
        assert "Modelnotfound:m" in _flat(capsys.readouterr())
        assert _autostart_logs(tmp_path) == []

    def test_a_successful_attach_removes_the_file_while_the_server_runs(
            self, monkeypatch, tmp_path):
        real_popen = subprocess.Popen
        children = []

        def popen(cmd, **kw):
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            child = real_popen(
                [sys.executable, "-c",
                 "import sys, time\n"
                 "print('started', flush=True)\n"
                 "for _ in range(60):\n"
                 "    time.sleep(0.1)\n"
                 "    if sys.stdin.read(1) == '': break\n"
                 "print('still writing', flush=True)\n"],
                stdin=subprocess.PIPE, stdout=kw["stdout"],
                stderr=subprocess.STDOUT, creationflags=flags)
            children.append(child)
            return child

        try:
            backend, exc = _headless_autostart(monkeypatch, tmp_path, popen,
                                               attach_after=1)
            assert exc is None and backend is not None
            assert children and children[0].poll() is None, (
                "the stand-in server must still be running at this point")
            assert _autostart_logs(tmp_path) == []
            children[0].stdin.close()
            assert children[0].wait(timeout=30) == 0, (
                "the server must keep writing to its output after the log is "
                "removed")
        finally:
            for child in children:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=30)
        assert _autostart_logs(tmp_path) == []

    def test_a_failed_attach_in_privacy_mode_removes_the_file(
            self, monkeypatch, tmp_path, capsys):
        import localm.config
        from localm.plugins.coder.audit import SessionMode

        monkeypatch.setattr(localm.config, "keep_diagnostics_enabled",
                            lambda: False)
        _, exc = _headless_autostart(monkeypatch, tmp_path,
                                     lambda cmd, **kw: _LiveProc(),
                                     session_mode=SessionMode.PRIVACY)
        assert exc is not None and exc.code == 1
        out = _flat(capsys.readouterr())
        assert "Failedtoattachtotheauto-startedserver" in out
        assert "wasnotkept" in out
        assert _autostart_logs(tmp_path) == []

    def test_a_failed_attach_with_keep_diagnostics_keeps_the_file(
            self, monkeypatch, tmp_path, capsys):
        import localm.config
        from localm.plugins.coder.audit import SessionMode

        monkeypatch.setattr(localm.config, "keep_diagnostics_enabled",
                            lambda: True)
        _, exc = _headless_autostart(monkeypatch, tmp_path,
                                     lambda cmd, **kw: _LiveProc(),
                                     session_mode=SessionMode.PRIVACY)
        assert exc is not None and exc.code == 1
        kept = _autostart_logs(tmp_path)
        assert len(kept) == 1
        assert kept[0] in _flat(capsys.readouterr())

    def test_a_failed_attach_outside_privacy_mode_keeps_the_file(
            self, monkeypatch, tmp_path, capsys):
        import localm.config
        from localm.plugins.coder.audit import SessionMode

        monkeypatch.setattr(localm.config, "keep_diagnostics_enabled",
                            lambda: False)
        _, exc = _headless_autostart(monkeypatch, tmp_path,
                                     lambda cmd, **kw: _LiveProc(),
                                     session_mode=SessionMode.LOG)
        assert exc is not None and exc.code == 1
        kept = _autostart_logs(tmp_path)
        assert len(kept) == 1
        assert kept[0] in _flat(capsys.readouterr())
