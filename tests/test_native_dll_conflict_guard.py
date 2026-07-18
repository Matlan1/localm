# SPDX-License-Identifier: AGPL-3.0-or-later
"""A real, root-cause fix for the torch/ROCm DLL-identity conflict documented
in VramSizingMixin._torch_rocm_init_broken (llamacpp/_sizing.py): once
llama.cpp's own native runtime is loaded in a process (always true inside
GgufWorker or the embedder's isolated worker - each loaded it themselves to
run their own model), a LATER `import torch` on this project's Windows + AMD
ROCm build reliably hits an entry-point mismatch between torch's own ROCm
wheel and whatever DLL the OS loader already resolved for llama.cpp.

The PRE-EXISTING `_torch_rocm_init_broken` cache only stops a REPEAT of this
within one process - it does nothing for the first occurrence, and since every
worker is a FRESH process, that "first occurrence" happens on every single
worker, every time, including a real Windows modal dialog the OS shows before
the exception even reaches Python (confirmed live: the model still finished
loading and replying, but only after the box was manually dismissed).

These tests pin the actual fix: _loader.native_lib_loaded() lets
_free_total_vram_bytes() know AHEAD OF TIME that this exact process is the
risky one, so it skips the torch attempt entirely instead of triggering the
conflict and catching the aftermath - and suppress_native_error_dialogs()
stops the OS from ever presenting that blocking UI in the first place, for
any OTHER native DLL failure a worker process might hit.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._sizing import VramSizingMixin


@pytest.fixture(autouse=True)
def _reset_sizing_caches():
    # Process-wide caches must not leak between tests, or an earlier test that
    # sets them can silently defeat a later one's assertions.
    before = VramSizingMixin._torch_rocm_init_broken
    VramSizingMixin._torch_rocm_init_broken = False
    yield
    VramSizingMixin._torch_rocm_init_broken = before


class TestNativeLibLoaded:
    def test_false_before_any_load(self, monkeypatch):
        monkeypatch.setattr(_loader, "_loaded_lib", None)
        assert _loader.native_lib_loaded() is False

    def test_true_once_load_lib_has_set_it(self, monkeypatch):
        monkeypatch.setattr(_loader, "_loaded_lib", MagicMock())
        assert _loader.native_lib_loaded() is True


class TestFreeTotalVramBytesSkipsTorchWhenNativeLibLoaded:
    def test_never_touches_torch_once_native_lib_is_loaded(self, monkeypatch):
        """The actual regression this fix closes: a fresh worker process has
        ALWAYS just loaded llama.cpp's native lib (that is what a worker is
        for), so this condition is true on every single worker load, not a
        rare edge case."""
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)

        poisoned = MagicMock()
        poisoned.cuda.is_available.side_effect = AssertionError(
            "torch.cuda must never be touched once native_lib_loaded() is True")
        monkeypatch.setitem(sys.modules, "torch", poisoned)

        assert VramSizingMixin._free_total_vram_bytes() == (None, None)
        poisoned.cuda.is_available.assert_not_called()

    def test_still_attempts_torch_when_native_lib_not_loaded(self, monkeypatch):
        """Control: the parent process (GgufBackend, before any child is
        spawned) never loads llama.cpp's native lib itself, so its own
        preflight VRAM checks must keep using the fast in-process torch.cuda
        read exactly as before - this fix must not affect that path."""
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)

        probed = MagicMock()
        probed.cuda.is_available.return_value = False  # short-circuit before
        monkeypatch.setitem(sys.modules, "torch", probed)  # config/discover imports

        VramSizingMixin._free_total_vram_bytes()
        probed.cuda.is_available.assert_called_once()

    def test_still_short_circuits_on_the_pre_existing_broken_cache(self, monkeypatch):
        """The backstop this fix keeps: if _torch_rocm_init_broken was ALREADY
        set (some other path hit the conflict and cached it), this method must
        still skip torch even when native_lib_loaded() says False."""
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        VramSizingMixin._torch_rocm_init_broken = True

        poisoned = MagicMock()
        poisoned.cuda.is_available.side_effect = AssertionError("must not be reached")
        monkeypatch.setitem(sys.modules, "torch", poisoned)

        assert VramSizingMixin._free_total_vram_bytes() == (None, None)
        poisoned.cuda.is_available.assert_not_called()


class TestSuppressNativeErrorDialogs:
    @pytest.fixture(autouse=True)
    def _reset_flag(self):
        from localm import _mp_spawn
        before = _mp_spawn._native_error_dialogs_suppressed
        _mp_spawn._native_error_dialogs_suppressed = False
        yield
        _mp_spawn._native_error_dialogs_suppressed = before

    def test_noop_off_windows(self, monkeypatch):
        from localm import _mp_spawn
        monkeypatch.setattr(sys, "platform", "linux")
        assert _mp_spawn.suppress_native_error_dialogs() is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only mechanism")
    def test_applies_seterrormode_on_windows(self):
        # Real call, not mocked: suppressing the OS critical-error UI for the
        # current (test) process is harmless and exactly what a worker
        # process's entry point does for real - the same real-behavior
        # standard as the rest of this suite's llama.cpp/GPU tests.
        from localm import _mp_spawn
        assert _mp_spawn.suppress_native_error_dialogs() is True

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only mechanism")
    def test_idempotent_second_call_is_a_cheap_noop(self):
        from localm import _mp_spawn
        assert _mp_spawn.suppress_native_error_dialogs() is True
        assert _mp_spawn.suppress_native_error_dialogs() is True
