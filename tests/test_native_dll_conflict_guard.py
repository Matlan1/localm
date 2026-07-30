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

import importlib
import io
import sys
import types
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from localm.inference.backends.llamacpp import _loader
from localm.inference.backends.llamacpp._sizing import VramSizingMixin

doctor_mod = importlib.import_module("localm.cli.doctor")


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


class TestVramLevelsSkipsTorchWhenNativeLibLoaded:
    """VramSizingMixin._vram_levels() must get the SAME proactive skip as its
    sibling _free_total_vram_bytes() above: it also does a bare `import torch`
    under a broad except, so a resident native runtime makes it re-hit and
    re-trace the identical STATUS_ENTRYPOINT_NOT_FOUND fault on every call
    (observed live via tests/test_grammar_sampling.py::
    test_invalid_grammar_does_not_poison_later_valid_grammars, which runs
    after a real_gguf-marked test has loaded the native lib in-process)."""

    def test_never_touches_torch_once_native_lib_is_loaded(self, monkeypatch):
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)

        poisoned = MagicMock()
        poisoned.cuda.is_available.side_effect = AssertionError(
            "torch.cuda must never be touched once native_lib_loaded() is True")
        monkeypatch.setitem(sys.modules, "torch", poisoned)

        assert VramSizingMixin._vram_levels() == []
        poisoned.cuda.is_available.assert_not_called()

    def test_still_attempts_torch_when_native_lib_not_loaded(self, monkeypatch):
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)

        probed = MagicMock()
        probed.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", probed)

        assert VramSizingMixin._vram_levels() == []
        probed.cuda.is_available.assert_called_once()

    def test_skip_is_surfaced_at_debug_not_silenced(self, monkeypatch, caplog):
        """Rule 5 (AGENTS.md): a skip must be diagnosable, never silent."""
        import logging

        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            VramSizingMixin._vram_levels()
        assert "skipping the torch VRAM read" in caplog.text


def _torch_stub_with_poisoned_cuda(side_effect):
    """A torch stand-in for a FULL `doctor()` CLI run (not just the unit-level
    _check_vram_torch() call): a bare MagicMock() does not survive
    _check_packages()'s later ``importlib.import_module("torch")`` re-fetch of
    the already-cached module - first ``ValueError: torch.__spec__ is not
    set`` (a Mock has no real ``__spec__``), then, once given a real
    ``types.ModuleType``, ``ValueError: torch.__spec__ is None`` (a fresh
    ModuleType's ``__spec__`` defaults to None) - so a real ``ModuleSpec``
    must be set explicitly too, exactly as
    test_doctor_hf_backend_usable.py's ``_BrokenLazyModule.__init__`` already
    does for transformers, for the identical reason. ``cuda.is_available``
    stays a MagicMock so tests can still assert on whether it was called."""
    import importlib.machinery

    mod = types.ModuleType("torch")
    mod.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    cuda = MagicMock()
    cuda.is_available.side_effect = side_effect
    mod.cuda = cuda
    return mod


class TestCheckVramTorchSkipsTorchWhenNativeLibLoaded:
    """`localm.cli.doctor._check_vram_torch` is the FOURTH call site for the same
    DLL-identity conflict as the two classes above and
    discover._torch_gpu_probe_known_doomed: it also did a bare `import torch`
    under only a narrow `except ImportError`. Unlike its siblings, `doctor()`
    (cli/doctor.py) calls it with no try/except of its own, so an escaping
    exception here would silently truncate every later doctor check (GPU
    verdict, packages, HF backend, plugin deps, managed comfy) - not just
    re-trace noisily like the other three did before their fixes."""

    def test_never_touches_torch_once_native_lib_is_loaded(self, monkeypatch):
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)

        poisoned = MagicMock()
        poisoned.cuda.is_available.side_effect = AssertionError(
            "torch.cuda must never be touched once native_lib_loaded() is True")
        monkeypatch.setitem(sys.modules, "torch", poisoned)

        assert doctor_mod._check_vram_torch() is False
        poisoned.cuda.is_available.assert_not_called()

    def test_still_attempts_torch_when_native_lib_not_loaded(self, monkeypatch):
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)

        probed = MagicMock()
        probed.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", probed)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        assert doctor_mod._check_vram_torch() is False
        probed.cuda.is_available.assert_called_once()

    def test_skip_is_surfaced_at_debug_not_silenced(self, monkeypatch, caplog):
        """Rule 5 (AGENTS.md): a skip must be diagnosable, never silent."""
        import logging

        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        with caplog.at_level(logging.DEBUG, logger="localm"):
            doctor_mod._check_vram_torch()
        assert "skipping the torch VRAM probe" in caplog.text

    def test_non_importerror_torch_failure_does_not_escape(self, monkeypatch, caplog):
        """The other half of this fix: a torch import/probe failure that is NOT
        a plain ImportError (the doomed DLL-identity conflict itself, reached
        some other way, or any other native fault) must degrade
        torch_gpu_found to False rather than propagate out of
        _check_vram_torch() - the original bug this pins, since `doctor()`
        has no try/except of its own around this call."""
        import logging

        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)

        doomed = MagicMock()
        doomed.cuda.is_available.side_effect = OSError(
            "[WinError 127] The specified procedure could not be found")
        monkeypatch.setitem(sys.modules, "torch", doomed)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        with caplog.at_level(logging.DEBUG, logger="localm"):
            result = doctor_mod._check_vram_torch()

        assert result is False
        assert "torch GPU/VRAM probe failed" in caplog.text

    def test_doctor_cli_survives_the_doomed_combination_end_to_end(
            self, cli_runner, monkeypatch):
        """Wires the exact doctor() concern through the REAL CLI command: with
        native_lib_loaded() True, `doctor()` must still run every later check
        (proven by the managed-comfy hint, its very last print) instead of
        dying at the GPU verdict - and torch.cuda must still never be touched,
        proving the guard itself (not just the widened except) is what saves
        it."""
        from localm import cli as _cli

        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(_cli, "find_binary_dir", lambda: None)

        poisoned = _torch_stub_with_poisoned_cuda(AssertionError(
            "torch.cuda must never be touched once native_lib_loaded() is True"))
        monkeypatch.setitem(sys.modules, "torch", poisoned)

        result = cli_runner.invoke(_cli.doctor, [])

        assert result.exit_code == 0, result.output
        poisoned.cuda.is_available.assert_not_called()
        # The very last thing doctor() prints - proof nothing upstream truncated it.
        assert "ComfyUI" in result.output

    def test_doctor_cli_survives_a_non_importerror_torch_failure_end_to_end(
            self, cli_runner, monkeypatch):
        """The property that matters most about this fix, pinned at the CLI
        level and kept separate from the guard test above: a torch-probe
        failure that is NOT the DLL-conflict guard's precondition
        (native_lib_loaded() False here, so the guard does not fire at all)
        must still not stop doctor() from running and reporting every LATER
        check. This is the rule-5 half of the fix (AGENTS.md) - a diagnostic
        tool must never lose diagnostics silently - and it is exercised by the
        widened ``except Exception`` alone, with the guard out of the
        picture."""
        from localm import cli as _cli

        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr(_cli, "find_binary_dir", lambda: None)

        doomed = _torch_stub_with_poisoned_cuda(OSError(
            "[WinError 127] The specified procedure could not be found"))
        monkeypatch.setitem(sys.modules, "torch", doomed)

        result = cli_runner.invoke(_cli.doctor, [])

        assert result.exit_code == 0, result.output
        # Every later check ran to completion, not just a warning for this one.
        assert "ComfyUI" in result.output


class TestCheckPackagesSkipsTorchWhenNativeLibLoadedAndNotResident:
    """`localm.cli.doctor._check_packages` is a FIFTH call site for the same
    DLL-identity conflict, found live while verifying the fourth
    (_check_vram_torch above): its own bare
    ``importlib.import_module("torch")`` (under only ``except ImportError``)
    raised the real ``OSError: [WinError 127]`` when
    test_doctor_gpu_verdict.py::test_compute_devices_reports_real_devices_when_provisioned
    loaded the native runtime in-process earlier in the same pytest worker,
    then test_doctor_hf_backend_usable.py hit it via this exact call.

    Narrower than _check_vram_torch's blanket skip: torch already RESIDENT in
    sys.modules (a real success earlier in this process, or a test double)
    must still be returned, never discarded to None just because
    native_lib_loaded() is True - only a FRESH import is actually doomed. A
    first version of this fix skipped unconditionally and broke
    test_doctor_hf_backend_usable.py's own CLI end-to-end test, which
    pre-mocks a safe torch before invoking doctor() - caught by re-running
    that exact file paired with test_doctor_gpu_verdict.py after this fix."""

    def _no_torch_resident(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)

    def test_skips_a_fresh_import_when_native_lib_loaded(self, monkeypatch):
        self._no_torch_resident(monkeypatch)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        real_import_module = importlib.import_module

        def _poisoned(name, *a, **k):
            if name == "torch":
                raise AssertionError(
                    "torch must not be imported once native_lib_loaded() is "
                    "True and torch is not already resident")
            return real_import_module(name, *a, **k)

        monkeypatch.setattr(importlib, "import_module", _poisoned)

        modules = doctor_mod._check_packages()
        assert modules.get("torch") is None

    def test_keeps_an_already_resident_torch_even_if_native_lib_loaded(
            self, monkeypatch):
        """The regression this refinement guards against: an ALREADY-resident
        torch (real success earlier in this process, or a test double) must
        not be discarded just because native_lib_loaded() is True."""
        stub = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", stub)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        modules = doctor_mod._check_packages()
        assert modules.get("torch") is stub

    def test_still_imports_torch_when_native_lib_not_loaded(self, monkeypatch):
        self._no_torch_resident(monkeypatch)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        stub = types.ModuleType("torch")
        real_import_module = importlib.import_module

        def _returns_stub(name, *a, **k):
            if name == "torch":
                return stub
            return real_import_module(name, *a, **k)

        monkeypatch.setattr(importlib, "import_module", _returns_stub)

        modules = doctor_mod._check_packages()
        assert modules.get("torch") is stub

    def test_skip_is_surfaced_at_debug_not_silenced(self, monkeypatch, caplog):
        """Rule 5 (AGENTS.md): a skip must be diagnosable, never silent."""
        import logging

        self._no_torch_resident(monkeypatch)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))
        with caplog.at_level(logging.DEBUG, logger="localm"):
            doctor_mod._check_packages()
        assert "skipping the torch package import" in caplog.text

    def test_non_importerror_torch_failure_does_not_truncate_later_packages(
            self, monkeypatch):
        """The widened except half, independent of the guard
        (native_lib_loaded() False here): a torch import failure that is NOT
        a plain ImportError must not stop LATER packages (transformers, the
        last in the loop) from still being checked."""
        self._no_torch_resident(monkeypatch)
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
        monkeypatch.setattr(doctor_mod, "console",
                            Console(file=io.StringIO(), force_terminal=False))

        real_import_module = importlib.import_module

        def _doomed(name, *a, **k):
            if name == "torch":
                raise OSError(
                    "[WinError 127] The specified procedure could not be found")
            return real_import_module(name, *a, **k)

        monkeypatch.setattr(importlib, "import_module", _doomed)

        modules = doctor_mod._check_packages()
        assert modules.get("torch") is None
        assert "transformers" in modules


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
