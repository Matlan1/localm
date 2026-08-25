# SPDX-License-Identifier: AGPL-3.0-or-later
"""doctor-1: `localm doctor` must derive the GPU verdict from what localm will ACTUALLY use for inference (the provisioned llama.cpp backend / a real device probe), NOT from nvidia-smi / rocm-smi / torch alone."""

import importlib
import subprocess
import sys
import types

import pytest

import localm.cli as cli

# The doctor MODULE (localm.cli.doctor resolves to the re-exported click Command,
# so import the module object explicitly for monkeypatching its globals).
doctor_mod = importlib.import_module("localm.cli.doctor")


@pytest.fixture(autouse=True)
def _neutralise_native_lib_loaded():
    """_loader.native_lib_loaded() (added by #754) is True for the rest of ANY xdist worker in which a real_gguf-gated test has RUN (conftest.py's lazy resource gate - or the test itself - calls load_lib() at that test's setup, and _loaded_lib is deliberately never reset)."""
    from localm.inference.backends.llamacpp import _loader
    saved = _loader.native_lib_loaded
    _loader.native_lib_loaded = lambda: False
    yield
    _loader.native_lib_loaded = saved


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _fake_torch(gpu_names):
    mod = types.ModuleType("torch")

    class _Props:
        def __init__(self, name):
            self.name = name

    class _Cuda:
        @staticmethod
        def is_available():
            return bool(gpu_names)

        @staticmethod
        def device_count():
            return len(gpu_names)

        @staticmethod
        def get_device_properties(i):
            return _Props(gpu_names[i])

        @staticmethod
        def mem_get_info(i):
            return (8 * 1024**3, 16 * 1024**3)

    mod.cuda = _Cuda()
    return mod


def _no_smi(monkeypatch):
    """Make every smi subprocess probe fail (no nvidia-smi / rocm-smi)."""
    def _raise(*a, **k):
        raise FileNotFoundError("not found")

    monkeypatch.setattr(subprocess, "run", _raise)


def _install_torch(monkeypatch, gpu_names):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(gpu_names))
    # This fake torch stub lacks the real internals transformers needs, so if
    # the REAL transformers is installed in this venv, its lazy AutoTokenizer/
    # AutoProcessor/AutoModelForCausalLM resolution would genuinely fail against
    # it - a false "HF backend UNUSABLE" unrelated to the GPU verdict under test.
    monkeypatch.setattr(doctor_mod, "_check_hf_backend_usable", lambda *a, **k: None)


def _healthy_bindir(tmp_path, backend=None):
    """A binary dir with a plausibly sized llama.dll (so lib_healthy is True) and an optional .localm-backend marker written the real way (via setup_llama)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "llama.dll").write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB
    if backend is not None:
        from localm.setup_llama import _record_provisioned_backend
        _record_provisioned_backend(bindir, backend)
    return bindir


def _stub_probe(monkeypatch, result):
    """Control the subprocess device probe (real ggml load needs a real GPU runtime; the verdict logic under test is what consumes its result)."""
    monkeypatch.setattr(doctor_mod, "_probe_gpu_devices", lambda: result)


def _no_op_abi(monkeypatch):
    """Skip the ABI subprocess so these GPU-verdict tests stay fast/hermetic."""
    monkeypatch.setattr(doctor_mod, "_check_native_abi", lambda: None)


def _blind_probes(monkeypatch):
    """The exact audit condition: no smi AND torch sees no CUDA GPU."""
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, [])


# --------------------------------------------------------------------------- #
#  (1) Ground truth: a real device probe naming a GPU device                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend,gpu_dev", [
    ("vulkan", "Vulkan0"),     # Intel / NVIDIA / mixed via Vulkan
    ("metal", "Metal"),        # Apple Silicon (torch.cuda ALWAYS False on Mac)
    ("amd-rocm", "ROCm0"),     # AMD on Windows, bundled ROCm build, rocm-smi absent
])
def test_gpu_backend_with_device_probe_is_not_cpu_only(
    cli_runner, tmp_path, monkeypatch, backend, gpu_dev
):
    """smi + torch both blind, but the loaded runtime registers a GPU device -> doctor must report the GPU and NEVER say 'CPU mode only' (the doctor-1 bug)."""
    bindir = _healthy_bindir(tmp_path, backend=backend)
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {
        "loaded": True,
        "devices": [[gpu_dev, 1], ["CPU", 0]],  # 1 = GGML GPU, 0 = CPU
        "error": "",
    })

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" not in out
    assert gpu_dev in out
    assert "used for inference" in out


def test_only_cpu_device_is_reported_cpu_only(cli_runner, tmp_path, monkeypatch):
    """A genuine cpu build: the runtime loads but registers ONLY a CPU device -> 'CPU mode only' is the correct, honest verdict here."""
    bindir = _healthy_bindir(tmp_path, backend="cpu")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {"loaded": True, "devices": [["CPU", 0]], "error": ""})

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" in out


# --------------------------------------------------------------------------- #
#  (2) Fallback: provisioned-backend marker when the device probe is unusable  #
# --------------------------------------------------------------------------- #

def test_gpu_marker_used_when_probe_has_no_device_registry(
    cli_runner, tmp_path, monkeypatch
):
    """Older build: runtime loads (computes) but exposes no ggml_backend_dev_* registry (empty devices)."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {"loaded": True, "devices": [], "error": ""})

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" not in out
    assert "vulkan" in out
    assert "backend provisioned" in out
    assert "used for inference" in out


def test_gpu_marker_used_when_probe_unavailable(cli_runner, tmp_path, monkeypatch):
    """The probe could not run at all (None)."""
    bindir = _healthy_bindir(tmp_path, backend="cuda")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, None)

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" not in out
    assert "cuda" in out


def test_cpu_marker_fallback_is_cpu_only(cli_runner, tmp_path, monkeypatch):
    """A cpu-build marker with an unusable probe -> honest 'CPU mode only'."""
    bindir = _healthy_bindir(tmp_path, backend="cpu")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, None)

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" in out


def test_broken_gpu_runtime_not_claimed_as_working(cli_runner, tmp_path, monkeypatch):
    """A GPU marker but the probe says the runtime did NOT load (loaded False): doctor must NOT claim a working GPU from the marker alone - the marker is only trusted when the runtime is not known-broken."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {"loaded": False, "devices": [], "error": "boom"})

    out = cli_runner.invoke(cli.doctor, []).output
    assert "used for inference" not in out


def test_corrupt_lib_with_gpu_marker_is_not_claimed_as_working(
    cli_runner, tmp_path, monkeypatch
):
    """A truncated/corrupt llama.dll (lib_healthy False) next to a GPU marker: the device probe is deliberately skipped, so the marker must NOT be trusted as a working GPU - doctor just flagged the runtime as unloadable (AGENTS.md rule 5: never report success for a known-broken state)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "llama.dll").write_bytes(b"\x00")  # 1 byte -> corrupt/truncated
    from localm.setup_llama import _record_provisioned_backend
    _record_provisioned_backend(bindir, "vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    # A GPU probe result that WOULD claim a GPU - it must be ignored because the
    # lib is unhealthy and the real code never runs the probe in that case.
    _stub_probe(monkeypatch, {"loaded": True, "devices": [["Vulkan0", 1]], "error": ""})

    out = cli_runner.invoke(cli.doctor, []).output
    assert "used for inference" not in out          # no false GPU success
    assert "truncated" in out.lower() or "suspicious" in out.lower()  # broken lib flagged
    assert "CPU mode only" in out                    # honest fallback


def test_accel_only_device_is_not_labelled_gpu(cli_runner, tmp_path, monkeypatch):
    """A runtime that registers only a ggml ACCEL device (e.g. a custom BLAS/RPC build) plus CPU is NOT a GPU: it must not be reported as 'GPU: ... used for inference' (ACCEL is a CPU-side/remote accelerator, not a GPU)."""
    bindir = _healthy_bindir(tmp_path, backend="custom")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {
        "loaded": True,
        "devices": [["BLAS", 2], ["CPU", 0]],  # 2 = GGML ACCEL, 0 = CPU
        "error": "",
    })

    out = cli_runner.invoke(cli.doctor, []).output
    assert "used for inference" not in out
    assert "CPU mode only" in out


# --------------------------------------------------------------------------- #
#  (2b) The probe RAN AND FAILED: indeterminate, not "no GPU" (H1)             #
# --------------------------------------------------------------------------- #

# Short and unbroken on purpose: the reason is printed on an indented dim line,
# and rich wraps a non-tty console at 80 columns, so a long sentinel could fail
# this test by being FOLDED rather than by being absent.
_PROBE_SENTINEL = "VKPROBE-SENTINEL-9271"


def test_failed_probe_reports_the_reason_not_no_gpu(cli_runner, tmp_path, monkeypatch):
    """The probe ran and RAISED (the shape of a driver too old for the provisioned Vulkan build)."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {
        "loaded": False,
        "devices": [],
        "error": f"OSError('{_PROBE_SENTINEL}')",
    })

    out = cli_runner.invoke(cli.doctor, []).output
    # The captured reason reaches the user.
    assert _PROBE_SENTINEL in out
    assert "could not be determined" in out
    # ... and none of the three false claims the old path made.
    assert "No GPU detected" not in out
    assert "CPU mode only" not in out
    # The step-(3) provisioning advice specifically. Asserted on this phrase
    # rather than the bare substring "setup-llama": _check_runtime_build
    # legitimately prints 'localm setup-llama --force' in this same run when the
    # build tag is unrecorded, and a bare-substring assertion would have been
    # red for that unrelated, correct line.
    assert "to provision one" not in out
    assert "used for inference" not in out          # no false GPU success either


def test_failed_probe_is_reported_even_when_torch_sees_a_gpu(
    cli_runner, tmp_path, monkeypatch
):
    """A card torch CAN see plus a runtime that will not load is the case most worth naming: the user has a GPU and localm cannot use it."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, ["RTX 4090"])
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {
        "loaded": False, "devices": [], "error": f"OSError('{_PROBE_SENTINEL}')",
    })

    out = cli_runner.invoke(cli.doctor, []).output
    assert _PROBE_SENTINEL in out
    assert "could not be determined" in out


def test_probe_error_is_truncated_to_one_line(cli_runner, tmp_path, monkeypatch):
    """A native loader error repr can carry a whole traceback."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {
        "loaded": False,
        "devices": [],
        "error": f"{_PROBE_SENTINEL}\nSECONDLINE-8814\nTHIRDLINE-8815",
    })

    out = cli_runner.invoke(cli.doctor, []).output
    assert _PROBE_SENTINEL in out
    assert "SECONDLINE-8814" not in out


def test_probe_that_cleanly_reports_no_gpu_is_unchanged(cli_runner, tmp_path, monkeypatch):
    """The guard on the branch above: a probe that ran, did NOT raise, and simply reported the runtime does not compute is a MEASURED negative, not an indeterminate one."""
    bindir = _healthy_bindir(tmp_path, backend="vulkan")
    monkeypatch.setattr(cli, "find_binary_dir", lambda: bindir)
    _blind_probes(monkeypatch)
    _no_op_abi(monkeypatch)
    _stub_probe(monkeypatch, {"loaded": False, "devices": [], "error": ""})

    out = cli_runner.invoke(cli.doctor, []).output
    assert "could not be determined" not in out
    assert "CPU mode only" in out


# --------------------------------------------------------------------------- #
#  (3) No-runtime fallback: smi/torch, hedged - and never a false GPU claim    #
# --------------------------------------------------------------------------- #

def test_no_runtime_no_gpu_is_hedged_cpu_only(cli_runner, monkeypatch):
    """No runtime provisioned and smi/torch see nothing: the legacy 'CPU mode only' still appears, but it is HEDGED (points at Vulkan/Metal + setup-llama) rather than stated as a bare fact."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _blind_probes(monkeypatch)

    out = cli_runner.invoke(cli.doctor, []).output
    assert "CPU mode only" in out
    # Hedged: it must not claim smi/rocm-smi absence proves CPU-only.
    assert "setup-llama" in out
    assert "Vulkan" in out or "Metal" in out


def test_no_runtime_but_torch_sees_gpu_is_not_cpu_only(cli_runner, monkeypatch):
    """No runtime marker, but torch sees a CUDA/ROCm GPU -> positive proof, no 'CPU mode only' contradiction (regression guard for the pre-existing BUG-2)."""
    monkeypatch.setattr(cli, "find_binary_dir", lambda: None)
    _no_smi(monkeypatch)
    _install_torch(monkeypatch, ["RTX 4090"])

    out = cli_runner.invoke(cli.doctor, []).output
    assert "RTX 4090" in out
    assert "CPU mode only" not in out


# --------------------------------------------------------------------------- #
#  Real ctypes device enumeration (integration, no mocks) - skipped w/o runtime #
# --------------------------------------------------------------------------- #

def test_compute_devices_reports_real_devices_when_provisioned():
    """Exercise the REAL ggml device enumeration against a provisioned runtime."""
    from localm.inference.backends.llamacpp import _loader

    try:
        if not _loader.compute_backends_available():
            pytest.skip("no computing llama runtime provisioned on this machine")
    except Exception as e:
        pytest.skip(f"no loadable llama runtime on this machine: {e}")

    devices = _loader.compute_devices()
    if not devices:
        pytest.skip("runtime lacks the ggml_backend_dev_* registry symbols")

    names = [n for (n, _t) in devices]
    types_ = [t for (_n, t) in devices]
    assert any(t == _loader.GGML_DEV_TYPE_CPU for t in types_), (
        f"a computing runtime must register a CPU device; got {devices}"
    )
    # A RANGE, not an enumerated set of named constants. This used to assert
    # membership in (CPU, GPU, ACCEL) with ACCEL declared as 2 - and upstream
    # later inserted IGPU at 2, pushing ACCEL to 3. So on any current runtime
    # that assertion PASSED for an integrated GPU while believing it was an
    # accelerator, and would FAIL on a real accelerator (3) or a meta device
    # (4). Both halves wrong, and neither visible from the test.
    #
    # 4 is META, the highest member measured (ggml-backend.h at master and at
    # b9870, 2026-08-11). A failure here is therefore informative rather than
    # flaky: it means the enum GREW again, and _loader's device-type comment
    # needs re-reading against the header for the provisioned runtime.
    assert all(isinstance(t, int) and 0 <= t <= 4 for t in types_), (
        f"ggml device type outside the known enum range in {devices}; "
        "the enum may have grown - re-read ggml-backend.h"
    )
    assert all(isinstance(n, str) and n for n in names)


def test_only_stable_ggml_device_type_constants_are_declared():
    """CPU and GPU are the only ggml_backend_dev_type members safe to hardcode."""
    from localm.inference.backends.llamacpp import _loader

    assert _loader.GGML_DEV_TYPE_CPU == 0
    assert _loader.GGML_DEV_TYPE_GPU == 1

    declared = [n for n in dir(_loader) if n.startswith("GGML_DEV_TYPE_")]
    assert sorted(declared) == ["GGML_DEV_TYPE_CPU", "GGML_DEV_TYPE_GPU"], (
        f"only the version-stable members may be declared; found {declared}. "
        "Members past GPU have moved between llama.cpp releases - read the "
        "header for the provisioned runtime instead of hardcoding one."
    )
