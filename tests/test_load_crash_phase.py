# SPDX-License-Identifier: AGPL-3.0-or-later
"""A native crash during a model load is reported with advice that matches
where it happened.

A crash while creating the context (``llama_init_from_model``) happens after
the runtime has already loaded the weights, so telling the user to repair the
runtime points them the wrong way. The runner reads the phase from the
captured faulthandler trace; ``gguf._load_failure_message`` builds the final
text from it.
"""

import os
import sys

import pytest

from localm.inference.backends import gguf as gguf_mod
from localm.inference.backends.llamacpp import _runner as runner_mod
from localm.inference.backends.llamacpp._runner import (
    ModelRunner, NativeLoadCrashError, crash_phase_from_trace)

# The crashing-thread block from a real report (Linux, localm 0.2.0), with the
# install path shortened. The main context's llama_init_from_model is on top.
_CONTEXT_TRACE = """Fatal Python error: Segmentation fault

Thread 0x00007f8b0612a6c0 (most recent call first):
  File "~/localm/.python/lib/python3.12/multiprocessing/queues.py", line 103 in get
  File "~/localm/localm/inference/backends/llamacpp/_runner.py", line 268 in _control_loop
  File "~/localm/.python/lib/python3.12/threading.py", line 1012 in run

Current thread 0x00007f8b16dcf780 (most recent call first):
  File "~/localm/localm/inference/backends/llamacpp/_api.py", line 169 in llama_init_from_model
  File "~/localm/localm/inference/backends/llamacpp/llama.py", line 1242 in __init__
  File "~/localm/localm/inference/backends/llamacpp/_worker.py", line 167 in load
  File "~/localm/localm/inference/backends/llamacpp/_runner.py", line 308 in _runner_main
  File "<string>", line 1 in <module>

Extension modules: regex._regex (total: 1)
"""

_WEIGHTS_TRACE = _CONTEXT_TRACE.replace(
    '_api.py", line 169 in llama_init_from_model',
    '_api.py", line 130 in llama_load_model_from_file')


class TestCrashPhaseFromTrace:
    def test_context_creation_crash_is_classified_as_context(self):
        assert crash_phase_from_trace(_CONTEXT_TRACE) == "context"

    def test_weight_load_crash_is_classified_as_weights(self):
        assert crash_phase_from_trace(_WEIGHTS_TRACE) == "weights"

    def test_known_frame_on_another_thread_does_not_count(self):
        trace = _CONTEXT_TRACE.replace(
            'queues.py", line 103 in get', 'x.py", line 1 in llama_load_model_from_file')
        assert crash_phase_from_trace(trace) == "context"
        only_other = trace.replace(
            '_api.py", line 169 in llama_init_from_model', '_api.py", line 1 in helper')
        assert crash_phase_from_trace(only_other) is None

    def test_trace_without_a_current_thread_block_is_unknown(self):
        assert crash_phase_from_trace("Fatal Python error: Aborted\n") is None
        assert crash_phase_from_trace("") is None

    def test_windows_trace_shape(self):
        trace = ("Windows fatal exception: access violation\n\n"
                 "Current thread 0x00001a2c (most recent call first):\n"
                 '  File "C:\\x\\_api.py", line 169 in llama_init_from_model\n'
                 '  File "C:\\x\\llama.py", line 1242 in __init__\n')
        assert crash_phase_from_trace(trace) == "context"


class TestLoadFailureMessage:
    def test_context_crash_gives_context_advice_and_no_runtime_repair(self):
        exc = NativeLoadCrashError(
            "The native model-loading process crashed (exit code -11 (killed "
            "by signal SIGSEGV)) while loading. The server stayed up."
            + runner_mod._load_crash_advice("context"), phase="context")
        msg = gguf_mod._load_failure_message(exc)
        assert "setup-llama" not in msg
        assert "LLAMA_CPP_LIB" not in msg
        assert "creating the context" in msg
        assert "n_ctx" in msg and "gpu_split_indices" in msg
        assert ".." not in msg
        assert msg.startswith("The model failed to load: ")

    def test_weight_crash_names_setup_llama_exactly_once(self):
        exc = NativeLoadCrashError(
            "The native model-loading process crashed (exit code -11) while "
            "loading." + runner_mod._load_crash_advice("weights"), phase="weights")
        msg = gguf_mod._load_failure_message(exc)
        assert msg.count("setup-llama") == 1
        assert ".." not in msg

    def test_null_context_is_not_blamed_on_the_runtime(self):
        msg = gguf_mod._load_failure_message(
            RuntimeError(gguf_mod._CONTEXT_FAILED_MSG))
        assert "setup-llama" not in msg
        assert "context" in msg and "n_ctx" in msg

    def test_null_context_message_matches_what_the_worker_raises(self):
        from pathlib import Path
        src = Path(sys.modules["localm.inference.backends.llamacpp._runner"].__file__
                   ).with_name("llama.py").read_text(encoding="utf-8")
        assert f'raise RuntimeError("{gguf_mod._CONTEXT_FAILED_MSG}")' in src

    @pytest.mark.parametrize("platform,lib", [
        ("linux", "libllama.so"), ("darwin", "libllama.dylib"), ("win32", "llama.dll")])
    def test_runtime_error_names_this_platforms_library(self, monkeypatch,
                                                        platform, lib):
        monkeypatch.setattr(sys, "platform", platform)
        msg = gguf_mod._load_failure_message(
            RuntimeError("could not load the native library"), " Extra hint.")
        assert f"working {lib})" in msg
        assert msg.count("setup-llama") == 1
        assert "library. Extra hint." in msg


class TestSpawnAndLoadReportsThePhase:
    """A real worker dies from a real native abort inside a frame named like
    the context-creation entry point; the parent reads the phase from the
    captured trace."""

    @pytest.fixture(autouse=True)
    def _clean_fault_env(self):
        os.environ.pop(runner_mod._FAULT_ENV, None)
        yield
        os.environ.pop(runner_mod._FAULT_ENV, None)

    def _load(self, monkeypatch, fault):
        monkeypatch.setenv(runner_mod._FAULT_ENV, fault)
        monkeypatch.setenv("LOCALM_MODE", "log")
        r = ModelRunner()
        try:
            with pytest.raises(NativeLoadCrashError) as ei:
                r.spawn_and_load(dict(model_path="does-not-matter.gguf"),
                                 timeout=60)
            return ei.value
        finally:
            r.shutdown(grace=0)

    def test_crash_in_context_creation_is_reported_as_context(self, monkeypatch):
        exc = self._load(monkeypatch, "abort-in-context")
        assert exc.phase == "context"
        assert "creating the context" in str(exc)
        assert "setup-llama" not in str(exc)

    def test_crash_elsewhere_keeps_the_runtime_advice(self, monkeypatch):
        exc = self._load(monkeypatch, "abort")
        assert exc.phase is None
        assert "setup-llama" in str(exc)
