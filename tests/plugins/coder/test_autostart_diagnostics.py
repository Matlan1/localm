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
