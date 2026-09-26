# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression coverage for the GPU-vision-encode-failed-retrying-on-CPU status
notice: the CLI's two on_status callbacks (_stream_once, _interactive in
chat.py) and the llamacpp backend's producer (_generate_image in llama.py)
must all agree on one shared string, base.VISION_CPU_FALLBACK_STATUS, rather
than each side hand-typing its own copy of the text."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from localm.audit import SessionMode
from localm.cli.chat import run
from localm.inference.backends.base import VISION_CPU_FALLBACK_STATUS
from tests._bare_llama import make_bare_llama


def _fake_cli_engine(statuses, tokens=("Hello",)):
    """A MagicMock engine whose chat_stream relays *statuses* through
    on_status before yielding *tokens*, for driving the CLI's on_status
    callbacks in _stream_once/_interactive."""
    engine = MagicMock()

    def _chat_stream(messages, on_status=None, **kwargs):
        if on_status:
            for s in statuses:
                on_status(s)
        for t in tokens:
            yield t

    engine.chat_stream.side_effect = _chat_stream
    engine.display_name = "test-vision-model"
    engine.count_tokens.return_value = 2
    engine.count_messages_tokens.return_value = 3
    engine.context_capacity.return_value = 4096
    return engine


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch):
    """Pin the CLI console wide so the fallback status, long enough to wrap
    at a narrow width, survives as one unbroken substring."""
    from tests.conftest import make_console_wide_and_plain
    make_console_wide_and_plain(monkeypatch, width="300")
    from localm.cli import _core
    monkeypatch.setattr(_core.console, "_width", 300)
    monkeypatch.setattr(_core.console, "_height", 25)
    monkeypatch.setattr(_core.err_console, "_width", 300)
    monkeypatch.setattr(_core.err_console, "_height", 25)


class TestStreamOnceVisionCpuFallbackWarning:
    def test_prints_the_status_to_stderr_not_stdout(self, capsys):
        """`localm run -p ... > answer.txt` must not capture this warning: it
        is a status notice, not part of the reply."""
        from localm.cli.chat import _stream_once
        engine = _fake_cli_engine(statuses=[VISION_CPU_FALLBACK_STATUS])

        _stream_once(engine, [{"role": "user", "content": "describe this"}])

        captured = capsys.readouterr()
        assert VISION_CPU_FALLBACK_STATUS in captured.err, (
            f"_stream_once must print the vision-CPU-fallback status to "
            f"stderr when the engine reports it via on_status: {captured.err!r}")
        assert VISION_CPU_FALLBACK_STATUS not in captured.out, (
            f"the vision-CPU-fallback status leaked into stdout, where a "
            f"piped -p reply is read from: {captured.out!r}")

    def test_other_statuses_are_not_printed(self, capsys):
        from localm.cli.chat import _stream_once
        engine = _fake_cli_engine(
            statuses=["Encoding image (GPU)...", "Generating response..."])

        _stream_once(engine, [{"role": "user", "content": "describe this"}])

        out = capsys.readouterr().out
        assert "Encoding image" not in out
        assert "Generating response" not in out


class TestInteractiveVisionCpuFallbackWarning:
    def test_prints_the_status_to_stderr_not_stdout(self, monkeypatch, capsys):
        from localm.cli import chat as chat_mod
        engine = _fake_cli_engine(statuses=[VISION_CPU_FALLBACK_STATUS])
        inputs = iter(["describe this image"])

        def _fake_input(*a, **kw):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError()
        monkeypatch.setattr(chat_mod.console, "input", _fake_input)

        chat_mod._interactive(engine, None, {})

        captured = capsys.readouterr()
        assert VISION_CPU_FALLBACK_STATUS in captured.err, (
            f"_interactive must print the vision-CPU-fallback status to "
            f"stderr when the engine reports it via on_status: {captured.err!r}")
        assert VISION_CPU_FALLBACK_STATUS not in captured.out, (
            f"the vision-CPU-fallback status leaked into stdout: {captured.out!r}")


class TestRunCommandEndToEnd:
    """The same property, driven through the real `run` command and a real
    CliRunner invocation rather than calling _stream_once directly - proves
    it through the full `localm run -p` dispatch, matching how a piped
    `localm run M -p ... > answer.txt` actually invokes the CLI."""

    @pytest.fixture
    def patched_attach(self, monkeypatch):
        """Same seam test_run_p_exit_code_on_load_failure.py stubs: attach to
        a fake server already serving the requested model."""
        fake_target = {"base_url": "http://127.0.0.1:8642/v1", "token": "tok"}
        engine_instance = MagicMock(name="HttpEngine_instance")

        def _chat_stream(messages, on_status=None, **kwargs):
            if on_status:
                on_status(VISION_CPU_FALLBACK_STATUS)
            yield "answer"
        engine_instance.chat_stream.side_effect = _chat_stream
        engine_instance.count_tokens.return_value = 1
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

    def test_piped_reply_excludes_the_warning_which_lands_on_stderr(self, patched_attach):
        result = CliRunner().invoke(run, ["qa-chat", "-p", "describe this"])
        assert result.exit_code == 0, result.output
        assert "answer" in result.stdout
        assert VISION_CPU_FALLBACK_STATUS not in result.stdout, (
            f"a piped `localm run -p` reply must not carry the status "
            f"warning: {result.stdout!r}")
        assert VISION_CPU_FALLBACK_STATUS in result.stderr, (
            f"the warning must still reach the user, on stderr: "
            f"{result.stderr!r}")


class TestLlamaCppEmitsTheSharedFallbackConstant:
    """_generate_image's GPU-vision-encode-failed CPU-retry path (llama.py)
    must emit base.VISION_CPU_FALLBACK_STATUS exactly, so a rewording of the
    shared constant cannot drift out of sync with what the CLI/SSE consumers
    compare against."""

    def test_cpu_retry_emits_exactly_the_constant(self):
        from localm.inference.backends.llamacpp.mtmd import MtmdGpuEncodeFailed

        class _StopAfterRetry(Exception):
            """Aborts the generator right after the retry eval_into call, so
            the test never reaches the native decode loop."""

        llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
        llm._mtmd = MagicMock()
        llm._mtmd.marker = "<image>"
        llm._mtmd.on_gpu = True
        llm._mtmd.retry_on_cpu.return_value = True
        llm._mtmd.count_tokens.return_value = 10
        llm._mtmd.eval_into.side_effect = [MtmdGpuEncodeFailed(), _StopAfterRetry()]

        mock_api = MagicMock()
        mock_api.llama_model_chat_template.return_value = None
        mock_api.has_memory_api.return_value = True

        messages = [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        statuses: list = []

        with patch("localm.inference.backends.llamacpp.llama.api", mock_api):
            gen = llm._generate_image(
                messages, max_new_tokens=8, temperature=0.8, top_k=40,
                top_p=0.95, repeat_penalty=1.1, on_status=statuses.append)
            with pytest.raises(_StopAfterRetry):
                next(gen)

        assert statuses == ["Encoding image (GPU)...", VISION_CPU_FALLBACK_STATUS], (
            f"the CPU-retry path must emit the shared VISION_CPU_FALLBACK_STATUS "
            f"constant via on_status, not a hand-typed literal: {statuses!r}")
