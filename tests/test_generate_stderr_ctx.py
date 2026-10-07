# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which stderr context LlamaCpp._generate wraps native generation output in,
and what a grammar-constrained generation then shows on the console."""

import contextlib
import os
from unittest.mock import MagicMock, patch

import pytest

import localm.inference.backends.llamacpp.llama as llama_mod
from localm import debuglog
from localm.debuglog import dedup_native_stderr
from localm.inference.backends.llamacpp.llama import _stderr_ctx_for_generate
from tests._bare_llama import make_bare_llama
from tests._fake_batch import fake_batch_init

_GRAMMAR = 'root ::= "x"'
_TRIGGER = r"(<tool_call>[\s\S]*)"


def _mock_native_api() -> MagicMock:
    """A mock api module for the KV-reuse prefill plus the decode loop."""
    mock_api = MagicMock()
    mock_api.has_memory_api.return_value = True
    mock_api.llama_get_memory.return_value = 333
    mock_api.llama_memory_seq_rm.return_value = True
    mock_api.llama_decode.return_value = 0
    mock_api.llama_batch_init.side_effect = fake_batch_init
    mock_api.llama_sampler_sample.return_value = 42
    return mock_api


def _run_generate(llm, mock_api, *, max_new_tokens, grammar, grammar_lazy):
    triggers = [_TRIGGER] if grammar_lazy else None
    with patch.object(llama_mod, "api", mock_api), \
         patch.object(llama_mod, "_build_sampler", return_value=999):
        return list(llm._generate(
            prompt_tokens=[1, 2, 3], max_new_tokens=max_new_tokens,
            temperature=0.8, top_k=40, top_p=0.95, repeat_penalty=1.1,
            grammar=grammar, grammar_lazy=grammar_lazy,
            grammar_triggers=triggers))


class _Console:
    """Stands in for the stderr duplicate dedup_native_stderr writes to."""

    def __init__(self) -> None:
        self.writes: list = []

    def write(self, text: str) -> None:
        self.writes.append(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def text(self) -> str:
        return "".join(self.writes)


def test_verbose_uses_nullcontext():
    assert _stderr_ctx_for_generate(True) is contextlib.nullcontext


def test_non_verbose_uses_dedup_native_stderr_with_the_stderr_lock():
    ctx = _stderr_ctx_for_generate(False)
    assert ctx.func is dedup_native_stderr
    assert ctx.keywords == {"swap_lock": llama_mod._stderr_lock}


@pytest.mark.parametrize(
    ("grammar", "grammar_lazy"),
    [(None, False), (None, True), (_GRAMMAR, False), (_GRAMMAR, True)],
    ids=["plain", "lazy-flag-without-grammar", "strict-grammar", "lazy-grammar"],
)
def test_generate_wraps_prefill_and_decode_in_dedup(monkeypatch, grammar, grammar_lazy):
    """Every non-verbose generation, grammar or not, runs its prefill and its
    decode loop inside dedup_native_stderr, with _stderr_lock as its swap lock,
    and never inside _quiet_stderr."""
    entered = []

    def _recorder(name):
        @contextlib.contextmanager
        def _ctx(swap_lock=None):
            entered.append((name, swap_lock is llama_mod._stderr_lock))
            yield
        return _ctx

    monkeypatch.setattr(debuglog, "dedup_native_stderr", _recorder("dedup"))
    monkeypatch.setattr(llama_mod, "_quiet_stderr", _recorder("quiet"))
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
    llm._tokenizer.is_eog.return_value = False

    tokens = _run_generate(llm, _mock_native_api(), max_new_tokens=2,
                           grammar=grammar, grammar_lazy=grammar_lazy)

    assert tokens == [42, 42]
    assert entered == [("dedup", True), ("dedup", True)], entered


# Pieces a lazy-grammar trace record carries: plain text, pieces that contain a
# newline (so the record spans lines), a piece ending in ")" before a newline,
# pieces with "`)" right before a newline, a backtick and an empty piece. "ZQX"
# marks generated text that must not leak.
_PIECES = ("I", " will", "\n", " ZQXalpha", "\n\n", ")\n", "}`)\n", "`", "",
           "ZQXbeta\nZQXgamma", "`)\n")


def test_grammar_generation_shows_native_lines_and_folds_the_trigger_trace(monkeypatch):
    """A lazy-grammar generation with LOCALM_DEBUG off: a native warning written
    during decode reaches the console and the ring buffer, the per-token
    "Grammar still awaiting trigger" records reach neither as lines, and the
    ring buffer holds one counted line for them without any generated text."""
    monkeypatch.delenv("LOCALM_DEBUG", raising=False)
    debuglog.install_ring_buffer()
    debuglog._ring_handler._buf.clear()
    console = _Console()
    monkeypatch.setattr(debuglog, "_stable_console_stream", lambda: console)

    calls = {"n": 0}

    def _sample(*_args):
        i = calls["n"]
        calls["n"] += 1
        if i == 1:
            os.write(2, b"ggml_vulkan: simulated device warning\n")
        end = "\r\n" if i == 2 else "\n"
        piece = _PIECES[i % len(_PIECES)]
        os.write(2, f"Grammar still awaiting trigger after token {9000 + i} "
                    f"(`{piece}`){end}".encode())
        return 42

    mock_api = _mock_native_api()
    mock_api.llama_sampler_sample.side_effect = _sample
    llm = make_bare_llama(_model_ptr=111, _ctx_ptr=222)
    llm._tokenizer.is_eog.return_value = False

    n = len(_PIECES)
    tokens = _run_generate(llm, mock_api, max_new_tokens=n,
                           grammar=_GRAMMAR, grammar_lazy=True)
    assert tokens == [42] * n

    shown = console.text()
    ring = "\n".join(debuglog.recent_activity())
    assert "ggml_vulkan: simulated device warning" in shown, shown
    assert "ggml_vulkan: simulated device warning" in ring, ring
    assert "Grammar" not in shown, shown
    assert "ZQX" not in shown and "ZQX" not in ring, (shown, ring)
    assert ring.count("Grammar still awaiting trigger") == 1, ring
    assert f"Grammar still awaiting trigger after {n} token(s)" in ring, ring
