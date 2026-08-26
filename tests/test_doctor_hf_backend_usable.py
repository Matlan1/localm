# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` must prove the HF (transformers) backend is actually USABLE,
not merely importable.

transformers 5.14 hard-imports `distributed/fsdp.py` on the ordinary
`transformers.AutoTokenizer` attribute access (transformers is a LAZY module, so
`import transformers` alone never touches that path), and fsdp needs
`torch._C._distributed_c10d`, absent from the pinned ROCm/Windows torch build.
That makes EVERY HF model load die at "loading processor..." while a
version/presence probe reports both `torch` and `transformers` OK.

These tests exercise `_check_hf_backend_usable` directly (real success path,
against whatever transformers/torch is actually installed in this venv - skipped
if absent) and via a synthetic broken lazy-import chain, plus the full
`cli.doctor` wiring end to end.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import io
import sys
import types

import pytest
from rich.console import Console

import localm.cli as cli

doctor_mod = importlib.import_module("localm.cli.doctor")

_OK = "✓"
_FAIL = "✗"


class _BrokenLazyModule(types.ModuleType):
    """Stands in for transformers' `_LazyModule` when attribute resolution fails:
    a chain of `ModuleNotFoundError("Could not import module 'X'") from <next
    layer down>`, several layers deep, bottoming out in the real cause
    (`torch._C._distributed_c10d` missing). Matches the shape of the real
    `_LazyModule.__getattr__` (`raise ModuleNotFoundError(...) from e`) without
    needing a broken transformers/torch install.

    A REAL ModuleType subclass (not a bare object): a plain object's
    `__getattr__` would also intercept dunder lookups like `__spec__` that
    `importlib.import_module` itself needs when a name is already cached in
    `sys.modules`, raising before doctor's own code is reached. Setting a real
    `__spec__` here means only the Auto* names doctor actually touches fall
    through to `__getattr__`, the same way the real `_LazyModule` only
    intercepts names it does not already have as a normal attribute."""

    def __init__(self):
        super().__init__("transformers")
        self.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    def __getattr__(self, name):
        root = ModuleNotFoundError("No module named 'torch._C._distributed_c10d'")
        mid = ModuleNotFoundError(
            "Could not import module 'fsdp'. Are this object's requirements "
            "defined correctly?"
        )
        mid.__cause__ = root
        top = ModuleNotFoundError(
            f"Could not import module '{name}'. Are this object's requirements "
            "defined correctly?"
        )
        top.__cause__ = mid
        raise top


def _run_check_capturing_output(monkeypatch, torch_mod, transformers_mod):
    buf = io.StringIO()
    monkeypatch.setattr(doctor_mod, "console", Console(file=buf, force_terminal=False))
    doctor_mod._check_hf_backend_usable(torch_mod, transformers_mod)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
#  Real working combo -> OK                                                   #
# --------------------------------------------------------------------------- #

def test_reports_ok_for_the_real_installed_combo(monkeypatch):
    """Against whatever torch/transformers is genuinely installed in THIS venv,
    the check must actually resolve AutoTokenizer/AutoProcessor/
    AutoModelForCausalLM and report OK - not just that they import.

    Skips rather than crashes when llama.cpp's native runtime is already loaded
    in this process (test_doctor_gpu_verdict.py's own real compute-device probe
    does this in-process when run earlier in the same pytest worker): a FRESH
    `import torch` there is the known-doomed DLL-identity conflict
    (VramSizingMixin._free_total_vram_bytes's docstring), which
    `pytest.importorskip` cannot turn into a skip - it only catches ImportError,
    and this raises OSError: [WinError 127]. A targeted single-file run is
    unaffected."""
    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        pytest.skip("llama.cpp's native runtime is already loaded in this "
                     "process (a real compute-device probe ran earlier in "
                     "this same pytest worker) - a fresh torch import here "
                     "is the known-doomed DLL-identity conflict, not this "
                     "test's own subject")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    out = _run_check_capturing_output(monkeypatch, torch, transformers)
    assert _OK in out
    assert _FAIL not in out
    assert "AutoTokenizer" in out


# --------------------------------------------------------------------------- #
#  Broken lazy-import chain -> FAIL with the DUG-OUT root cause                #
# --------------------------------------------------------------------------- #

def test_reports_fail_and_digs_to_the_real_root_cause(monkeypatch):
    """transformers imports fine, but resolving AutoTokenizer dies several layers
    down. Doctor must report FAIL (not the silent OK a mere `import
    transformers` would give) and must surface the REAL bottom-of-chain cause,
    not just the generic top-level wrapper message."""
    torch_stub = object()
    out = _run_check_capturing_output(monkeypatch, torch_stub, _BrokenLazyModule())

    assert _FAIL in out
    assert "UNUSABLE" in out
    # The real root cause must be surfaced...
    assert "torch._C._distributed_c10d" in out
    # ...not merely the shallow, unhelpful wrapper message standing alone.
    assert "Are this object's requirements defined correctly" not in out


def test_root_cause_digging_stops_on_self_referencing_chain(monkeypatch):
    """Defensive: a pathological exception chain that cycles back on itself must
    not hang doctor in an infinite loop - the walk must terminate."""

    class _CyclicModule:
        def __getattr__(self, name):
            e1 = ModuleNotFoundError("layer one")
            e2 = ModuleNotFoundError("layer two")
            e1.__cause__ = e2
            e2.__cause__ = e1  # cycle
            raise e1

    out = _run_check_capturing_output(monkeypatch, object(), _CyclicModule())
    assert _FAIL in out
    assert "layer" in out


# --------------------------------------------------------------------------- #
#  Absent optional backend -> silent (not a fault)                            #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("torch_mod,transformers_mod", [
    (None, None),
    (None, object()),
    (object(), None),
])
def test_silent_when_either_package_not_installed(monkeypatch, torch_mod, transformers_mod):
    """torch/transformers are OPTIONAL; when either did not import at all,
    `_check_packages` already reported that (not installed) - this check must
    add nothing, not a spurious FAIL for a backend nobody opted into."""
    out = _run_check_capturing_output(monkeypatch, torch_mod, transformers_mod)
    assert out == ""


# --------------------------------------------------------------------------- #
#  Full CLI wiring: cli.doctor surfaces the same verdict end to end            #
# --------------------------------------------------------------------------- #

def _fake_working_torch():
    """A stand-in torch with no CUDA device - just enough for the unrelated
    `_check_vram_torch` probe elsewhere in `doctor()` to run without crashing;
    this test is not about torch's own state, only about wiring the HF-backend
    usability check through to the real transformers module."""
    import types

    mod = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    mod.cuda = _Cuda()
    return mod


def test_doctor_cli_surfaces_broken_hf_backend_end_to_end(cli_runner, monkeypatch):
    """Wire the broken-chain scenario through the REAL `localm doctor` command
    (not just the unit-level check), proving `_check_packages`'s returned module
    handles actually reach `_check_hf_backend_usable`."""
    import subprocess

    def _no_smi(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _no_smi)
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_working_torch())
    monkeypatch.setitem(sys.modules, "transformers", _BrokenLazyModule())
    monkeypatch.setenv("COLUMNS", "400")  # avoid mid-token soft-wrap in assertions

    out = cli_runner.invoke(cli.doctor, []).output
    assert "HF backend" in out
    assert "UNUSABLE" in out
    assert "torch._C._distributed_c10d" in out


# --------------------------------------------------------------------------- #
#  A broken lazy module must not be misreported as "not installed"             #
# --------------------------------------------------------------------------- #

def test_missing_dist_metadata_does_not_turn_a_broken_module_into_not_installed(
        monkeypatch):
    """_check_packages must keep the module HANDLE when only the VERSION lookup
    fails, or the usability check above never gets to run.

    The failure this pins is environment-dependent, which is exactly why it
    needs its own test. _check_packages reads the version from dist metadata
    first and falls back to ``getattr(m, "__version__", "")``. That fallback
    only runs when metadata is ABSENT - so on a machine with transformers
    genuinely installed it is never reached and the end-to-end test above passes
    regardless. Where transformers is NOT installed (CI, and any lean install),
    the fallback runs, _LazyModule.__getattr__ raises ModuleNotFoundError for
    __version__, getattr's default does not suppress it because it is not an
    AttributeError, and it lands in the `except ImportError` that means "not
    installed". doctor then reported an imported module as missing and said
    nothing at all about the breakage.

    Simulated here on EVERY platform by forcing PackageNotFoundError, so the
    path is covered whether or not this venv has transformers.
    """
    broken = _BrokenLazyModule()
    monkeypatch.setitem(sys.modules, "transformers", broken)

    # _check_packages imports importlib.metadata LOCALLY (as _ilm), so there is
    # no module attribute to patch - patch the real module it binds to. Only
    # transformers loses its metadata: blanking it for EVERY package would push
    # click onto the deprecated __version__ fallback and emit a warning.
    _real_version = importlib.metadata.version

    def _no_metadata(name):
        if name == "transformers":
            raise importlib.metadata.PackageNotFoundError(name)
        return _real_version(name)

    monkeypatch.setattr(importlib.metadata, "version", _no_metadata)
    buf = io.StringIO()
    monkeypatch.setattr(doctor_mod, "console",
                        Console(file=buf, force_terminal=False, width=400))

    modules = doctor_mod._check_packages()

    assert modules.get("transformers") is broken, (
        "the module handle was dropped because its VERSION could not be read - "
        "_check_hf_backend_usable can no longer see it, so a broken backend "
        "goes unreported")
    assert "transformers (HF backend) - not installed" not in buf.getvalue(), (
        "an imported-but-broken module was reported as not installed")
