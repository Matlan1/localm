# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` must prove the HF (transformers) backend is actually USABLE,
not merely importable - found during the 0.1.2 release verification: transformers
5.14 hard-imports `distributed/fsdp.py` on the ordinary
`transformers.AutoTokenizer` attribute access (transformers is a LAZY module, so
`import transformers` alone never touches that path), and fsdp needs
`torch._C._distributed_c10d`, absent from the pinned ROCm/Windows torch build. That
made EVERY HF model load die at "loading processor..." while `localm doctor`
reported both `torch` and `transformers` OK - a version/presence probe, not a
usability one (see tests/test_gpu_extra_pins.py for the version-pin guard this
backs up with a functional check of doctor's OWN diagnosis).

These tests exercise `_check_hf_backend_usable` directly (real success path,
against whatever transformers/torch is actually installed in this venv - skipped
if absent) and via a synthetic broken lazy-import chain (the regression shape,
reproduced without needing to actually install a broken transformers/torch
combo), plus the full `cli.doctor` wiring end to end.
"""

from __future__ import annotations

import importlib
import importlib.machinery
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
    """Stands in for transformers' `_LazyModule` when attribute resolution hits
    the exact 0.1.2 regression shape: a chain of `ModuleNotFoundError("Could not
    import module 'X'") from <next layer down>`, several layers deep, bottoming
    out in the real cause (`torch._C._distributed_c10d` missing). Reproduces the
    shape verified against transformers 5.13.1's actual `_LazyModule.__getattr__`
    source (`raise ModuleNotFoundError(...) from e`), without needing a real
    broken transformers/torch install.

    A REAL ModuleType subclass (not a bare object): a plain object's
    `__getattr__` would also intercept dunder lookups like `__spec__` that
    `importlib.import_module` itself needs when a name is already cached in
    `sys.modules`, raising before doctor's own code is ever reached. Setting a
    real `__spec__` here means only the Auto* names doctor actually touches
    fall through to `__getattr__` - the same way the real `_LazyModule` only
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
    AutoModelForCausalLM and report OK - not just that they import."""
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
    """The exact 0.1.2 regression: transformers imports fine, but resolving
    AutoTokenizer dies several layers down. Doctor must report FAIL (not the
    silent OK a mere `import transformers` would give) and must surface the
    REAL bottom-of-chain cause, not just the generic top-level wrapper message
    that hid this regression in the first place."""
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
