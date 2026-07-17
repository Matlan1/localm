# SPDX-License-Identifier: AGPL-3.0-or-later
"""When the coder auto-starts a `localm gui` server and it dies fast (a busy
--port, or any other quick non-zero exit), the CLI must surface the real reason
instead of waiting out the full attach timeout and reporting the generic
"Failed to attach to the auto-started server." (see PR #740 follow-up: the
child's own refusal message prints to its own console/process, which this
process never reads)."""

import socket

from click.testing import CliRunner

import localm.instances as instances
import localm.plugins.coder.cli as ccli


def _bypass_plugin_gate(monkeypatch):
    monkeypatch.setattr("localm.plugins.engine.PluginManager.is_active",
                        lambda self, name: True)


def _no_existing_instance(monkeypatch):
    # No localm already running for this project dir, so main() takes the
    # auto-start branch instead of attaching to one.
    monkeypatch.setattr(instances, "resolve_root_dir", lambda **kw: "/tmp/proj")
    monkeypatch.setattr(instances, "attach_target", lambda *a, **kw: None)


class TestExplicitBusyPortFailsFast:
    def test_reports_real_reason_without_spawning(self, monkeypatch):
        _bypass_plugin_gate(monkeypatch)
        _no_existing_instance(monkeypatch)

        # A real bound socket, not a mock - the port is genuinely busy.
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
