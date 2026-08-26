# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-CODER-MANAGED-SERVER-STDERR: ManagedServer.start() used to spawn `localm
serve` with stderr to DEVNULL, so an early exit's real cause was unrecoverable.
The real subprocess/pipe/thread mechanism runs here; only the child COMMAND is
substituted for a controlled, fast-failing script."""

import subprocess
import sys

from localm.plugins.coder import server as server_mod


def _patch_popen_to_run(monkeypatch, script: str):
    """Redirect ManagedServer's Popen call to a controlled script, keeping
    every stdio/text kwarg it was actually called with."""
    real_popen = subprocess.Popen

    def fake_popen(_cmd, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(server_mod.subprocess, "Popen", fake_popen)


def test_exited_early_warning_includes_the_servers_stderr(monkeypatch, capsys):
    _patch_popen_to_run(
        monkeypatch,
        "import sys; sys.stderr.write('port already in use by another process\\n'); "
        "sys.exit(1)",
    )
    monkeypatch.setattr(server_mod, "_port_open", lambda host, port: False)

    srv = server_mod.ManagedServer("bogus-model", port=59999)
    ok = srv.start()

    assert ok is False
    out = capsys.readouterr().out
    assert "exited early" in out
    assert "port already in use by another process" in out


def test_exited_early_with_no_stderr_output_omits_the_colon(monkeypatch, capsys):
    """No tail to show must not print a dangling ':' with nothing after it."""
    _patch_popen_to_run(monkeypatch, "import sys; sys.exit(1)")
    monkeypatch.setattr(server_mod, "_port_open", lambda host, port: False)

    srv = server_mod.ManagedServer("bogus-model", port=59998)
    ok = srv.start()

    assert ok is False
    out = capsys.readouterr().out
    assert "exited early" in out
    assert "exited early (code 1):" not in out
