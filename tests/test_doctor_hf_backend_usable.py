# SPDX-License-Identifier: AGPL-3.0-or-later
"""`localm doctor` must prove the HF (transformers) backend is actually USABLE,
not merely importable, and usable where a model loads, not in doctor's own
process.

transformers 5.14 hard-imports `distributed/fsdp.py` on the ordinary
`transformers.AutoTokenizer` attribute access (transformers is a LAZY module, so
`import transformers` alone never touches that path), and fsdp needs
`torch._C._distributed_c10d`, absent from the pinned ROCm/Windows torch build.
That makes EVERY HF model load die at "loading processor..." while a
version/presence probe reports both `torch` and `transformers` OK.

And a model loads in a spawned worker, where an import can fail that works in
doctor's own process: an Intel XPU torch imported fine there and failed in every
HF worker with `[WinError 126]` (GitHub issue #1989). So the check imports and
resolves in a worker spawned the way a model load spawns one.

These tests exercise `_check_hf_backend_usable` and `check_hf_backend` directly
(real success path, against whatever transformers/torch is actually installed
in this venv - skipped if absent) and against stand-in packages written to disk
for the worker to import: a broken lazy-import chain, an import that fails only
in the worker, a worker that dies and one that hangs. Plus the full
`cli.doctor` wiring end to end.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import io
import multiprocessing as mp
import os
import sys
import time
import types

import pytest
from rich.console import Console

import localm.cli as cli
from localm import _mp_spawn, diagnostics

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
#  Stand-in packages, for the check's spawned worker to import                #
# --------------------------------------------------------------------------- #

_WORKING_TRANSFORMERS = "AutoTokenizer = AutoProcessor = AutoModelForCausalLM = object\n"

# _BrokenLazyModule's chain as package source: a module-level __getattr__ is how
# a lazy module intercepts the names it does not have yet.
_BROKEN_TRANSFORMERS = '''\
def __getattr__(name):
    root = ModuleNotFoundError("No module named 'torch._C._distributed_c10d'")
    mid = ModuleNotFoundError(
        "Could not import module 'fsdp'. Are this object's requirements "
        "defined correctly?")
    mid.__cause__ = root
    top = ModuleNotFoundError(
        f"Could not import module '{name}'. Are this object's requirements "
        "defined correctly?")
    top.__cause__ = mid
    raise top
'''


def _in_the_worker_only(statement: str) -> str:
    """Source for a stand-in package whose import runs *statement* in any other
    process, which here is the check's spawned worker, and succeeds in this
    one: the same import working here and failing there, as in #1989."""
    return ("import os\nimport time\n"
            f"if os.getpid() != {os.getpid()}:\n"
            f"    {statement}\n")


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    """Install stand-in ``torch`` and ``transformers`` packages from source.

    On disk rather than in ``sys.modules``: the check imports them in a fresh
    spawned interpreter, and what reaches that from this process is
    ``sys.path``, which multiprocessing hands over. They go first on it, ahead
    of any real install. Whatever this process holds under the two names is
    set aside for the test and put back after it, so nothing the test imports
    under them outlives it."""
    root = tmp_path / "site"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    for name in ("torch", "transformers"):
        monkeypatch.setitem(sys.modules, name, None)   # records what to restore
        monkeypatch.delitem(sys.modules, name)

    def install(torch_src: str = "", transformers_src: str = _WORKING_TRANSFORMERS):
        for name, src in (("torch", torch_src), ("transformers", transformers_src)):
            (root / name).mkdir()
            (root / name / "__init__.py").write_text(src, encoding="utf-8")
        importlib.invalidate_caches()

    return install


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

def test_reports_fail_and_digs_to_the_real_root_cause(monkeypatch, fake_hf):
    """transformers imports fine, but resolving AutoTokenizer dies several layers
    down. Doctor must report FAIL (not the silent OK a mere `import
    transformers` would give) and must surface the REAL bottom-of-chain cause,
    not just the generic top-level wrapper message.

    The broken chain is a package on disk because the worker imports
    transformers for itself. The handles passed in only say both packages are
    there, so plain objects do."""
    fake_hf(transformers_src=_BROKEN_TRANSFORMERS)
    out = _run_check_capturing_output(monkeypatch, object(), object())

    assert _FAIL in out
    assert "UNUSABLE" in out
    # The real root cause must be surfaced...
    assert "torch._C._distributed_c10d" in out
    # ...not merely the shallow, unhelpful wrapper message standing alone.
    assert "Are this object's requirements defined correctly" not in out


def test_root_cause_digging_stops_on_self_referencing_chain():
    """Defensive: a pathological exception chain that cycles back on itself must
    not send the walk to its root into an infinite loop - it must terminate, on
    the last link it had not seen."""
    e1 = ModuleNotFoundError("layer one")
    e2 = ModuleNotFoundError("layer two")
    e1.__cause__ = e2
    e2.__cause__ = e1  # cycle
    assert diagnostics._root_cause(e1) is e2


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
#  The imports happen in a spawned worker, as a model load's do (#1989)       #
# --------------------------------------------------------------------------- #

_WINERROR_126 = ('[WinError 126] The specified module could not be found. Error '
                 'loading "...\\torch\\lib\\c10_xpu.dll" or one of its dependencies.')


@pytest.mark.parametrize("caller", ["doctor", "standalone"])
def test_fails_when_the_worker_cannot_import_torch_that_this_process_can(
        fake_hf, caller):
    """GitHub issue #1989 as doctor saw it. With an Intel XPU torch, `import
    torch` worked in the process doctor ran in and failed in every HF worker,
    and the check, which imported torch in its own process, called the backend
    healthy while every real load failed.

    Here torch imports in this process and raises in any other one. Both call
    shapes must fail: doctor's, which hands in the handles it imported here,
    and the standalone one the GUI's isolated run makes, which passes nothing."""
    fake_hf(torch_src=_in_the_worker_only(f"raise OSError({_WINERROR_126!r})"))
    torch = importlib.import_module("torch")          # this process CAN
    transformers = importlib.import_module("transformers")

    if caller == "doctor":
        res = diagnostics.check_hf_backend(torch, transformers, resolved=True)
    else:
        res = diagnostics.check_hf_backend()

    assert res.status == diagnostics.FAIL
    assert "UNUSABLE" in res.summary
    assert f"OSError: {_WINERROR_126}" in res.summary
    # Named as the worker's import, so it cannot be read as a claim about here.
    assert "import torch, in a spawned model worker" in res.summary
    (finding,) = res.findings
    assert any("separate worker process" in h for h in finding.hints), finding.hints


def test_passes_without_importing_either_package_in_this_process(
        fake_hf, monkeypatch):
    """The OK path, established in the worker with nothing imported here.

    That includes a process already holding llama.cpp's native runtime, where
    importing torch would be the known-doomed DLL-identity conflict: the import
    happens in the worker, so the check still answers."""
    from localm.inference.backends.llamacpp import _loader
    monkeypatch.setattr(_loader, "native_lib_loaded", lambda: True)
    fake_hf()

    res = diagnostics.check_hf_backend()

    assert res.status == diagnostics.OK, res.summary
    assert "AutoTokenizer" in res.summary
    assert "in a spawned model worker" in res.summary
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_a_worker_that_dies_fails_the_check_and_names_the_step(fake_hf):
    """A worker that exits without a verdict (a native abort does this, and
    leaves no Python exception behind) fails the check, naming the step it
    died in and its exit code."""
    fake_hf(torch_src=_in_the_worker_only("os._exit(134)"))

    res = diagnostics.check_hf_backend()

    assert res.status == diagnostics.FAIL
    assert "died during import torch" in res.summary
    assert "exit code 134" in res.summary


def test_a_worker_that_hangs_is_stopped_at_the_deadline(fake_hf, monkeypatch):
    """The check never hangs doctor: a worker that does not answer in time is
    killed, and the check reports that it could not verify the backend. The
    step is not asserted, because a slow machine may still be starting the
    worker when the deadline comes."""
    fake_hf(torch_src=_in_the_worker_only("time.sleep(3600)"))
    monkeypatch.setattr(diagnostics, "HF_PROBE_TIMEOUT_S", 3.0)
    before = {p.pid for p in mp.active_children()}

    started = time.monotonic()
    res = diagnostics.check_hf_backend()
    elapsed = time.monotonic() - started

    assert res.status == diagnostics.WARN
    assert "not verified" in res.summary
    assert "within 3s and was stopped" in res.summary
    # The timeout path's own bound: the wait, then terminate and kill.
    assert elapsed < 3.0 + 2 * diagnostics.SPAWN_JOIN_TIMEOUT_S
    assert {p.pid for p in mp.active_children()} <= before, (
        "the hung worker was left running")


def test_the_worker_starts_the_way_the_hf_worker_does(fake_hf, monkeypatch):
    """The probe is evidence about the HF worker only while it starts the same
    way, and the part #1989 turned on is the venv's DLL directories being
    registered before anything imports torch.

    Run in this process so the order can be recorded. The process-wide steps
    are stubbed, because here they would act on the test process itself."""
    fake_hf()
    events = []
    monkeypatch.setattr(_mp_spawn, "install_parent_death_watchdog", lambda: None)
    monkeypatch.setattr(_mp_spawn, "ignore_interrupt_signals", lambda: None)
    monkeypatch.setattr(_mp_spawn, "suppress_native_error_dialogs", lambda: None)
    monkeypatch.setattr(_mp_spawn, "add_venv_dll_directories",
                        lambda: events.append("dll directories") or [])

    class _Conn:
        def send(self, msg):
            events.append(msg)

        def close(self):
            events.append("closed")

    diagnostics._hf_backend_probe(_Conn())

    assert events == [
        ("step", "worker startup"),
        "dll directories",
        ("step", "import torch"),
        ("step", "import transformers"),
        ("step", "transformers.AutoTokenizer"),
        ("step", "transformers.AutoProcessor"),
        ("step", "transformers.AutoModelForCausalLM"),
        ("ok",),
        "closed",
    ]


# --------------------------------------------------------------------------- #
#  Full CLI wiring: cli.doctor surfaces the same verdict end to end            #
# --------------------------------------------------------------------------- #

# A torch with no CUDA device - just enough for the unrelated `_check_vram_torch`
# probe elsewhere in `doctor()` to run; this test is about wiring the
# HF-backend check, not about torch's own state.
_TORCH_WITHOUT_A_GPU = '''\
class cuda:
    @staticmethod
    def is_available():
        return False
'''


def test_doctor_cli_surfaces_broken_hf_backend_end_to_end(
        cli_runner, monkeypatch, fake_hf):
    """Wire the broken-chain scenario through the REAL `localm doctor` command
    (not just the unit-level check), proving `_check_packages`'s returned module
    handles actually reach `_check_hf_backend_usable`, and the worker's verdict
    comes back out.

    torch is imported here first, as a torch already in ``sys.modules`` would
    be: `_check_packages` then takes it as present whatever an earlier test
    left loaded in this process, since its native-runtime guard only applies to
    a torch not imported yet."""
    import subprocess

    def _no_smi(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _no_smi)
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    fake_hf(torch_src=_TORCH_WITHOUT_A_GPU, transformers_src=_BROKEN_TRANSFORMERS)
    importlib.import_module("torch")
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
