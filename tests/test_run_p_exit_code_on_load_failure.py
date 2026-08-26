# SPDX-License-Identifier: AGPL-3.0-or-later
"""NEW-RUN-P-EXIT-0-ON-LOAD-FAILURE: `localm run M -p ...` exited 0 even when
the model load hard-failed and nothing was produced. `_stream_once`'s
`except RuntimeError` arm (an attached server returning an error - no model
loaded, unreachable, a crashed load) printed the message and returned "",
byte-identical to a successful call that happened to produce no tokens. A
script doing `localm run M -p ... > out.txt && next_step` could not tell a
total failure from empty output.

Not a blanket miss: `localm run no-such-model-xyz -p hi` already exits 1
(chat.py's model-not-found branch). This closes the one path that did not.
"""

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock

from localm.audit import SessionMode
from localm.cli.chat import _stream_once, run


class _RaisingEngine:
    def __init__(self, exc: Exception):
        self._exc = exc

    def chat_stream(self, *a, **k):
        raise self._exc


class TestStreamOnceExitsNonZeroOnRuntimeError:
    def test_runtime_error_raises_system_exit_1(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _stream_once(_RaisingEngine(RuntimeError(
                "server error (HTTP 503): Failed to load qa-chat: ... "
                "crashed (exit code 3)")),
                [{"role": "user", "content": "Say OK"}])
        assert exc_info.value.code == 1
        text = capsys.readouterr().out
        assert "Failed to load qa-chat" in text, (
            "the real failure reason must still reach the user, not just a "
            "bare exit code")

    def test_image_decode_unavailable_still_exits_0(self):
        """The two OTHER early-return arms are unaffected - this is not a
        blanket "every _stream_once failure exits non-zero" change."""
        from localm.inference.backends.base import ImageDecodeUnavailable
        out = _stream_once(_RaisingEngine(ImageDecodeUnavailable("no pillow")),
                           [{"role": "user", "content": "hi"}])
        assert out == ""


@pytest.fixture
def patched_attach(monkeypatch):
    """Same seam `test_cli_run_model_conflict.py` stubs: attach to a fake
    server already serving the requested model, then hand the streamed
    request a RuntimeError instead of a reply."""
    fake_target = {"base_url": "http://127.0.0.1:8642/v1", "token": "tok"}
    engine_instance = MagicMock(name="HttpEngine_instance")
    engine_instance.chat_stream.side_effect = RuntimeError(
        "server error (HTTP 503): Failed to load qa-chat: worker process "
        "crashed (exit code 3)")
    engine_cls = MagicMock(name="HttpEngine", return_value=engine_instance)

    monkeypatch.setattr("localm.instances.attach_target",
                        lambda *a, **k: fake_target)
    monkeypatch.setattr("localm.instances.resolve_root_dir", lambda *a, **k: ".")
    monkeypatch.setattr("localm.inference.http_engine.HttpEngine", engine_cls)
    monkeypatch.setattr("localm.inference.http_engine.remote_model_status",
                        lambda *a, **k: ("loaded", "qa-chat"))
    monkeypatch.setattr("localm.audit.effective_mode",
                        lambda *a, **k: SessionMode.LOG)
    return engine_instance


class TestRunCommandEndToEnd:
    def test_run_p_exits_nonzero_on_a_load_failure(self, patched_attach):
        """The ticket's own repro shape, through the real `run` command:
        "localm run qa-chat -p Say OK" against an injected load fault."""
        result = CliRunner().invoke(run, ["qa-chat", "-p", "Say OK"])
        assert result.exit_code != 0, (
            f"a hard failure must not exit 0 (byte-identical to success); "
            f"output={result.output!r}")
        assert "Failed to load qa-chat" in result.output
