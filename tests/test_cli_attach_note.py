# SPDX-License-Identifier: AGPL-3.0-or-later
"""CLI-1: `localm run` must not silently load the model in-process when it fails
to attach to a background server - it should say why (or that none was found).
--no-server stays quiet (the user opted out)."""

from localm.cli import _attach_fallback_note


def test_no_server_opt_out_is_silent():
    assert _attach_fallback_note(no_server=True, attach_error=None) is None
    # even with an error, --no-server means the user chose in-process: stay quiet
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
