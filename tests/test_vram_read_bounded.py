# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model load stalled forever, with no error and no timeout, whenever the
out-of-process torch GPU probe timed out. The console stopped after "Loading
<model> (backend: Gguf)", no native loader output ever appeared, and ``/health``
reported ``loaded: false`` indefinitely.

Root cause: ``VramSizingMixin._free_total_vram_bytes`` - which the load-sizing
preflight reaches before the native load's own bounded timeout can apply - did
an in-process ``import torch`` plus ``torch.cuda`` calls with no bound of its
own, AND did it even after ``discover`` had already latched the opposite
conclusion. That latch's own docstring states the rule this violated:
"retrying this import IN-PROCESS would reproduce the multi-minute startup hang
the isolation exists to prevent, so this one must never fall back that way."

Two independent guards, and both need their own control, because either one
passing alone would let the other rot:

- the latch (skip the attempt entirely), and
- the deadline (bound whatever attempt is still made).

Every "torch does not answer" fixture here wedges via a ``find_spec`` meta-path
finder or a blocking attribute call, i.e. it models a wait, not a raise - a
fixture that RAISED could not fail on this defect at all, since exceptions were
already handled.
"""
from __future__ import annotations

import importlib.abc
import sys
import threading
import time

import pytest

from localm.inference.backends.llamacpp._sizing import VramSizingMixin


@pytest.fixture(autouse=True)
def clean_sizing_state(monkeypatch):
    """Per-process latches on the mixin, reset per test - and the native-lib
    short-circuit pinned False, since a True there returns before any of the
    behaviour under test runs."""
    monkeypatch.setattr(VramSizingMixin, "_torch_rocm_init_broken", False)
    monkeypatch.setattr(VramSizingMixin, "_torch_vram_read_wedged", False)
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: False)
    yield


class WedgedTorchImport(importlib.abc.MetaPathFinder):
    """A ``torch`` whose IMPORT does not finish until the test releases it -
    the Windows loader-lock shape the isolation machinery exists for.

    ``find_spec``, never ``find_module``: the latter was removed in Python
    3.12, so a finder written that way is never consulted and "torch was not
    imported" would be true for something other than the code under test.

    RELEASABLE, and not merely for tidiness. An abandoned thread stuck
    mid-import holds CPython's per-module import lock for ``torch``, so a wedge
    that outlives its own test hangs the NEXT one that touches torch - on the
    test runner's main thread, where nothing bounds it. That is the same
    abandoned-import hazard ``gpu_usage.raw_reading_is_process_scoped``
    documents in the product. Raising on release, instead of falling through to
    the real finders, keeps the outcome deterministic and never imports real
    torch into the test process."""

    def __init__(self):
        self.attempted = threading.Event()
        self.release = threading.Event()

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch":
            self.attempted.set()
            self.release.wait(60)
            raise ImportError("wedged torch import, released by the test")
        return None


@pytest.fixture
def no_resident_torch(monkeypatch):
    """torch NOT in sys.modules - the state the server parent is in when a
    load begins, and the only state in which the latch is consulted."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)


@pytest.fixture
def short_deadline(monkeypatch):
    monkeypatch.setattr(VramSizingMixin, "_torch_vram_read_deadline",
                        staticmethod(lambda: 0.5))
    return 0.5


@pytest.fixture
def wedged_import(monkeypatch, no_resident_torch):
    finder = WedgedTorchImport()
    monkeypatch.setattr(sys, "meta_path", [finder] + list(sys.meta_path))
    yield finder
    # Release before teardown so no abandoned thread carries the torch import
    # lock into the next test.
    finder.release.set()
    time.sleep(0.05)


def _call_bounded(timeout: float = 20.0):
    """``_free_total_vram_bytes()`` on a helper thread, so a REGRESSION shows up
    as "did not return", not as a hung test run."""
    out: dict = {}
    t = threading.Thread(
        target=lambda: out.setdefault("v", VramSizingMixin._free_total_vram_bytes()),
        daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(timeout)
    return {"returned": not t.is_alive(),
            "elapsed": time.monotonic() - t0,
            "value": out.get("v")}


class TestTheLatchStopsTheInProcessImport:
    def test_a_proven_unavailable_torch_is_never_imported_here(
            self, wedged_import, monkeypatch):
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: True)

        r = _call_bounded()

        # Whether the doomed import was attempted at all, asserted before the
        # return value.
        assert r["returned"], (
            "the load-sizing VRAM read never came back - this is the stall "
            "itself, with no timeout and no error")
        assert not wedged_import.attempted.is_set(), (
            "torch was imported in-process even though the isolated probe had "
            "already PROVEN it cannot answer here; that import is the "
            "multi-minute hang the isolation exists to prevent")
        assert r["value"] == (None, None)
        assert r["elapsed"] < 1.0, r["elapsed"]

    def test_without_the_latch_the_import_IS_attempted(
            self, wedged_import, short_deadline, monkeypatch):
        """The control for the test above. Without it, a finder that could
        never fire for some unrelated reason would make that assertion pass on
        an instrument incapable of the opposite result."""
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)

        r = _call_bounded()

        assert wedged_import.attempted.is_set(), (
            "the import was not attempted even with the latch CLEAR, so this "
            "fixture cannot distinguish 'skipped by the latch' from 'never "
            "reachable'")
        assert r["returned"]

    def test_a_resident_torch_is_still_read_even_when_the_latch_is_set(
            self, monkeypatch):
        """Do not over-refuse. Once torch is in sys.modules it has, by
        definition, finished importing in this process, so the child probe's
        verdict says nothing about reading it - throwing away a working VRAM
        reading there would be its own defect."""
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: True)
        monkeypatch.setattr(VramSizingMixin, "_torch_free_total_uncapped",
                            staticmethod(lambda: (11, 22)))
        monkeypatch.setitem(sys.modules, "torch", sys.modules[__name__])

        assert VramSizingMixin._free_total_vram_bytes() == (11, 22)


class TestTheDeadlineBoundsWhatIsStillAttempted:
    def test_a_wedged_import_releases_the_caller_at_the_deadline(
            self, wedged_import, short_deadline, monkeypatch, caplog):
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)

        with caplog.at_level("WARNING", logger="localm.debug"):
            r = _call_bounded()

        assert r["returned"], (
            "a wedged torch import held the model-load VRAM read forever - "
            "the exact stall QA item 8 reported")
        assert r["value"] == (None, None)
        assert r["elapsed"] < short_deadline + 5.0, r["elapsed"]
        assert any("did not answer within" in rec.message for rec in caplog.records), (
            "the capability loss was not surfaced anywhere; a silent "
            "degrade is the rule-5 half of this defect")

    def test_a_wedged_read_is_not_retried_on_the_next_call(
            self, wedged_import, short_deadline, monkeypatch):
        """Latched, so the SECOND call costs nothing. Without this a load that
        reads VRAM several times pays the full deadline each time, and every
        overrun leaks another abandoned thread."""
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)

        first = _call_bounded()
        assert first["returned"]
        assert VramSizingMixin._torch_vram_read_wedged is True

        threads_before = threading.active_count()
        t0 = time.monotonic()
        again = VramSizingMixin._free_total_vram_bytes()
        elapsed = time.monotonic() - t0

        assert again == (None, None)
        assert elapsed < 0.2, elapsed
        assert threading.active_count() <= threads_before

    def test_the_deadline_is_the_gpu_probe_deadline(self):
        """Asserted as a RELATION, not as a literal: two independently chosen
        numbers for one property drift, and the direction that hurts is this
        one becoming the shorter, which would start reporting VRAM
        unmeasurable on exactly the cold boxes discover widened its own budget
        to tolerate."""
        from localm import discover
        assert (VramSizingMixin._torch_vram_read_deadline()
                == float(discover._GPU_PROBE_DEADLINE))


class TestTheHealthyReadIsUnchanged:
    def test_a_torch_that_answers_is_passed_straight_through(self, monkeypatch):
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)
        monkeypatch.setattr(VramSizingMixin, "_torch_free_total_uncapped",
                            staticmethod(lambda: (1234, 5678)))

        assert VramSizingMixin._free_total_vram_bytes() == (1234, 5678)
        assert VramSizingMixin._torch_vram_read_wedged is False

    def test_an_unmeasurable_answer_is_passed_straight_through(self, monkeypatch):
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)
        monkeypatch.setattr(VramSizingMixin, "_torch_free_total_uncapped",
                            staticmethod(lambda: (None, None)))

        assert VramSizingMixin._free_total_vram_bytes() == (None, None)
        assert VramSizingMixin._torch_vram_read_wedged is False, (
            "an honest 'unmeasurable' answer must not be mistaken for a wedge - "
            "that would disable torch for the rest of the process over a "
            "torch-less install answering correctly")

    def test_the_native_lib_short_circuit_still_wins(self, monkeypatch):
        """Inside the GGUF worker the bundled HIP runtime is resident and a
        torch import is the known-doomed DLL conflict. That skip must keep
        firing BEFORE any thread is spawned."""
        from localm.inference.backends.llamacpp import _loader
        monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)

        def _must_not_run():
            raise AssertionError("unreachable")
        monkeypatch.setattr(VramSizingMixin, "_torch_free_total_uncapped",
                            staticmethod(_must_not_run))

        assert VramSizingMixin._free_total_vram_bytes() == (None, None)

    def test_an_unexpected_error_is_not_swallowed_by_the_wrapper(self, monkeypatch):
        """The bound changes how long the caller waits, never what it sees. An
        exception escaping the raw read still reaches the caller rather than
        being converted into a silent 'unmeasurable' by the thread boundary."""
        from localm import discover
        monkeypatch.setattr(discover, "isolated_torch_unavailable", lambda: False)

        def _boom():
            raise ZeroDivisionError("driver said no")
        monkeypatch.setattr(VramSizingMixin, "_torch_free_total_uncapped",
                            staticmethod(_boom))

        with pytest.raises(ZeroDivisionError, match="driver said no"):
            VramSizingMixin._free_total_vram_bytes()


class TestTheRawReadKeepsItsOwnContract:
    def test_an_unimportable_torch_latches_the_rocm_flag(self, monkeypatch):
        """The pre-existing DLL-conflict latch is still driven by the raw read,
        not lost in the split.

        ``sys.modules["torch"] = None`` is the standard idiom for forcing
        ImportError on an otherwise-importable package without touching
        sys.path or meta_path - the lookup short-circuits there, so nothing
        else in the process is disturbed."""
        monkeypatch.setitem(sys.modules, "torch", None)

        assert VramSizingMixin._torch_free_total_uncapped() == (None, None)
        assert VramSizingMixin._torch_rocm_init_broken is True

    def test_a_torch_whose_cuda_read_raises_degrades_to_unmeasurable(
            self, monkeypatch):
        import types
        fake = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("HIP error: invalid argument")
        fake.cuda = _Cuda()
        monkeypatch.setitem(sys.modules, "torch", fake)

        assert VramSizingMixin._torch_free_total_uncapped() == (None, None)
        assert VramSizingMixin._torch_rocm_init_broken is False
