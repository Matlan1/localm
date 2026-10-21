# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm run` never loads the model in-process silently when it fails to attach
to a background server: it says why, or that none was found. --no-server stays
quiet."""

import os
import signal
import subprocess
import sys

import pytest

from localm.cli import _attach_fallback_note


def test_no_server_opt_out_is_silent():
    assert _attach_fallback_note(no_server=True, attach_error=None) is None
    # --no-server stays quiet even when attaching errored.
    assert _attach_fallback_note(no_server=True, attach_error=RuntimeError("x")) is None


def test_no_running_server_explains_fallback():
    note = _attach_fallback_note(no_server=False, attach_error=None)
    assert note is not None
    assert "this process" in note.lower()
    assert "localm serve" in note.lower()


def test_attach_error_is_surfaced():
    note = _attach_fallback_note(
        no_server=False, attach_error=RuntimeError("whoami timed out"))
    assert note is not None
    assert "could not attach" in note.lower()
    assert "whoami timed out" in note


def test_autostart_timeout_is_acknowledged():
    # After a background auto-start that timed out, the note acknowledges the
    # timeout.
    note = _attach_fallback_note(no_server=False, attach_error=None,
                                 autostart_attempted=True)
    assert note is not None
    low = note.lower()
    assert "background" in low            # acknowledges the auto-start
    assert "in time" in low               # names the timeout
    assert "this process" in low          # still explains the in-process fallback
    assert "no localm server is serving this directory" not in low   # no contradiction


def test_run_autostart_timeout_note(monkeypatch):
    # Drives the real `run` command so run() enters the auto-start block and the
    # poll times out. Only the subprocess and server discovery are stubbed.
    from unittest.mock import MagicMock

    from click.testing import CliRunner

    from localm.audit import SessionMode
    from localm.cli.chat import run

    monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: MagicMock())
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("localm.audit.effective_mode", lambda *a, **k: SessionMode.LOG)

    runner = CliRunner()
    result = runner.invoke(run, ["definitely-not-a-real-model-xyz", "-p", "hi"])
    # Collapse rich's line-wrapping into a single line.
    flat = " ".join(result.output.split()).lower()
    assert "starting one in the background" in flat          # auto-start was attempted
    assert "background server did not come up in time" in flat   # timeout acknowledged
    assert "no localm server is serving this directory" not in flat   # no contradiction


def test_chat_cli_sys_unbound_error(monkeypatch):
    from click.testing import CliRunner
    from localm.cli.chat import run
    from unittest.mock import MagicMock, patch
    
    fake_target = {"base_url": "http://127.0.0.1:8642", "token": "tok"}
    mock_engine = MagicMock()
    
    with patch("localm.instances.attach_target", return_value=fake_target), \
         patch("localm.inference.http_engine.remote_model_status", return_value=("ready", "gemma")), \
         patch("localm.inference.http_engine.HttpEngine", return_value=mock_engine), \
         patch("localm.config.load_config", return_value={"max_tokens": 10, "temperature": 0.7, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1}), \
         patch("localm.audit.effective_mode", return_value=MagicMock()):
        
        runner = CliRunner()
        result = runner.invoke(run, ["gemma", "-p", "hello"])
        assert result.exit_code == 0


@pytest.mark.skipif(os.name == "nt", reason="the POSIX detached-server launch")
def test_run_autostart_gives_the_server_no_shared_stdio(monkeypatch):
    from unittest.mock import MagicMock

    from click.testing import CliRunner

    from localm.audit import SessionMode
    from localm.cli.chat import run

    seen = {}

    def fake_popen(cmd, **kw):
        seen.update(kw)
        return MagicMock()

    monkeypatch.setattr("localm.instances.attach_target", lambda *a, **k: None)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)
    monkeypatch.setattr("localm.audit.effective_mode", lambda *a, **k: SessionMode.LOG)
    CliRunner().invoke(run, ["definitely-not-a-real-model-xyz", "-p", "hi"])
    assert seen["start_new_session"] is True
    for stream in ("stdin", "stdout", "stderr"):
        assert seen[stream] == subprocess.DEVNULL


@pytest.mark.skipif(os.name == "nt", reason="the POSIX detached-server launch")
def test_a_detached_server_does_not_hold_the_callers_pipe_open(tmp_path):
    pidfile = tmp_path / "pid"
    script = "; ".join([
        "import subprocess, sys",
        "from localm.cli.chat import _detached_stdio_kwargs",
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],"
        " **_detached_stdio_kwargs())",
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))",
        "print('reply')",
    ])
    proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE)
    try:
        out, _ = proc.communicate(timeout=30)
        assert out.strip() == b"reply"
    finally:
        proc.kill()
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass
