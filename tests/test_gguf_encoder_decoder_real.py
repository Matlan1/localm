# SPDX-License-Identifier: AGPL-3.0-or-later
"""REAL end-to-end test of an encoder-decoder (T5) GGUF through GgufBackend and
its isolated worker: LaMini-Flan-T5-248M (Q4_K_M, about 221 MB), CPU only.

@integration so the default `pytest -m "not integration"` skips it. Skips when
the native runtime is not provisioned or the model cannot be fetched; once both
are on disk, a failure is a real failure. The reference comparison also needs
upstream llama.cpp's own completion tool, which some runtime packages ship as
llama-completion-impl; that one test skips without it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap

import pytest

from tests._real_gguf import fetch_gguf, native_runtime_lib_path, require_native_runtime

pytestmark = [pytest.mark.integration, pytest.mark.real_gguf]

_REPO = "Felladrin/gguf-LaMini-Flan-T5-248M"
_FILE = "LaMini-Flan-T5-248M.Q4_K_M.gguf"
_GREEDY = dict(max_tokens=48, temperature=0.0, repeat_penalty=1.0, seed=1)


@pytest.fixture(scope="module")
def model_path():
    require_native_runtime()
    return fetch_gguf(_REPO, _FILE)


@pytest.fixture(scope="module")
def backend(model_path):
    from localm.inference.backends.gguf import GgufBackend
    be = GgufBackend(model_path, n_ctx=4096, n_gpu_layers=0)
    be.load()
    yield be
    be.unload()


def _ask(backend, messages):
    return "".join(backend.chat_stream(messages, **_GREEDY))


def _user(text):
    return [{"role": "user", "content": text}]


def test_it_registers_as_a_chat_model(model_path):
    from pathlib import Path

    from localm.model_manager.registry import _detect_local_model_type
    mtype, meta = _detect_local_model_type(Path(model_path), is_gguf=True, is_hf=False)
    assert meta["architecture"] == "t5"
    assert mtype == "llm"


def test_it_is_recorded_as_not_formatting_tool_calls(model_path):
    from pathlib import Path

    from localm.model_manager.gguf import gguf_tool_use_signal
    assert gguf_tool_use_signal(Path(model_path)) is False


def test_it_loads_as_an_encoder_decoder_model(backend):
    assert backend.loaded and backend.encoder_decoder is True
    assert backend.effective_ctx_max == 2048
    assert backend.supports_images is False and backend.supports_mtp is False


def test_it_answers_a_question(backend):
    assert "Paris" in _ask(backend, _user("What is the capital of France?"))
    assert backend.last_finish_reason == "stop"


def test_requests_are_independent(backend):
    first = _ask(backend, _user("List three primary colors."))
    other = _ask(backend, _user("What is the capital of France?"))
    again = _ask(backend, _user("List three primary colors."))
    assert first == again and first != other


def test_a_conversation_is_answered_from_its_history(backend):
    reply = _ask(backend, [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "And what is the capital of Germany?"},
    ])
    assert "Berlin" in reply


def test_the_token_count_is_the_encoder_input(backend):
    assert backend.count_messages_tokens(_user("What is the capital of France?")) == 8


def test_an_over_long_prompt_is_refused_and_the_model_keeps_serving(backend):
    from localm.inference.backends.base import ContextCapacityExceededError
    with pytest.raises(ContextCapacityExceededError, match="at most 2048"):
        _ask(backend, _user("apple banana cherry " * 900))
    assert backend.loaded
    assert "Paris" in _ask(backend, _user("What is the capital of France?"))


_UPSTREAM = textwrap.dedent("""
    import ctypes, os, sys
    lib_dir, model, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
    os.add_dll_directory(lib_dir)
    dll = ctypes.CDLL(os.path.join(lib_dir, sys.argv[4]))
    fn = getattr(dll, sys.argv[5])
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    args = ["llama-completion", "-m", model, "-p", prompt, "-n", "48", "--temp", "0",
            "-ngl", "0", "--device", "none", "-no-cnv", "--no-display-prompt",
            "--seed", "0", "-c", "4096", "-b", "2048", "-ub", "2048", "--no-warmup"]
    argv = (ctypes.c_char_p * (len(args) + 1))(*[a.encode() for a in args], None)
    sys.stdout.flush()
    sys.exit(fn(len(args), argv))
""")


def _upstream_completion(model_path, prompt):
    """Upstream llama.cpp's own completion tool (tools/completion) on *prompt*,
    greedy, CPU only, through the runtime's llama-completion-impl library."""
    lib = native_runtime_lib_path()
    if sys.platform != "win32":
        pytest.skip("the upstream completion tool is looked up in the Windows runtime only")
    dll, symbol = "llama-completion-impl.dll", "?llama_completion@@YAHHPEAPEAD@Z"
    if not (lib.parent / dll).is_file():
        pytest.skip(f"this runtime does not ship upstream's completion tool ({dll})")
    run = subprocess.run(
        [sys.executable, "-I", "-c", _UPSTREAM, str(lib.parent), model_path, prompt, dll, symbol],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        env=dict(os.environ, PATH=str(lib.parent) + os.pathsep + os.environ.get("PATH", "")))
    assert run.returncode == 0, run.stderr[-2000:]
    out = re.sub(r"\x1b\[[0-9;]*m", "", run.stdout)
    marker = " [end of text]"
    assert marker in out, out
    return out[:out.index(marker)]


@pytest.mark.parametrize("prompt", [
    "What is the capital of France?",
    "Answer the following question. What is the boiling point of water in Celsius?",
    "Write a short sentence about cats.",
])
def test_the_reply_matches_upstream_llama_cpp(backend, model_path, prompt):
    expected = _upstream_completion(model_path, prompt)
    from localm.inference.backends.llamacpp import LlamaCpp
    llm = LlamaCpp(model_path=model_path, n_ctx=4096, n_gpu_layers=0)
    try:
        tokens = list(llm._generate_encoder_decoder(
            _user(prompt), max_new_tokens=48, temperature=0.0, top_k=40, top_p=0.95,
            repeat_penalty=1.0))
        assert llm.detokenize(tokens) == expected
    finally:
        llm.close()
