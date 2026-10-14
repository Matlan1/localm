# SPDX-License-Identifier: AGPL-3.0-or-later
"""``LlamaCpp.check_grammar`` keeps the native parser's stderr (its error and
the whole grammar source) out of the debug log unless ``debug_content_enabled()``
allows chat content there. Only the native parse call is replaced: it writes to
fd 2 the way llama.cpp's parser does."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from localm import debuglog
from localm.inference.backends.base import InvalidGrammarError
from localm.inference.backends.llamacpp import llama as llama_mod

_CANARY = "GRAMMARCANARY_7Q4"


def _native_rejects(vocab, grammar_bytes, root):
    os.write(2, b"llama_grammar_init_impl: failed to parse grammar\n"
                + grammar_bytes + b"\n")
    return None


@pytest.fixture
def debug_log(tmp_path, monkeypatch):
    log = tmp_path / "debug.log"
    monkeypatch.setattr(debuglog, "native_stderr_target",
                        lambda: os.open(str(log), os.O_WRONLY | os.O_APPEND | os.O_CREAT))
    monkeypatch.setattr(llama_mod.api, "llama_sampler_init_grammar", _native_rejects,
                        raising=False)
    return log


def _backend():
    lc = object.__new__(llama_mod.LlamaCpp)
    lc._model_ptr = 1
    lc._tokenizer = SimpleNamespace(_vocab=None)
    return lc


def _check(lc) -> None:
    with pytest.raises(InvalidGrammarError):
        lc.check_grammar(f'root ::= "{_CANARY}" broken')


def test_grammar_text_stays_out_of_the_debug_log_without_content_permission(
        debug_log, monkeypatch):
    monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: False)
    _check(_backend())
    assert not debug_log.exists() or _CANARY not in debug_log.read_text(errors="replace")


def test_grammar_text_reaches_the_debug_log_when_content_is_allowed(debug_log, monkeypatch):
    monkeypatch.setattr(debuglog, "debug_content_enabled", lambda: True)
    _check(_backend())
    assert _CANARY in debug_log.read_text(errors="replace")


def test_quiet_stderr_restores_fd_2(capfd, debug_log):
    with llama_mod._quiet_stderr(to_log=False):
        os.write(2, b"hidden\n")
    os.write(2, b"visible\n")
    err = capfd.readouterr().err
    assert "visible" in err and "hidden" not in err
