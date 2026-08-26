# SPDX-License-Identifier: AGPL-3.0-or-later
import sys
from typing import Optional

from localm import diagnostics

from ._core import console, main

# ------------------------------------------------------------------ #
#  Doctor                                                              #
# ------------------------------------------------------------------ #

_OK_SYM   = "[green]✓[/green]"
_WARN_SYM = "[yellow]![/yellow]"
_FAIL_SYM = "[red]✗[/red]"
# Neutral info marker for discovery hints that are NOT pass/fail checks.
_HINT_SYM = "[cyan]i[/cyan]"

_SYM_FOR = {diagnostics.OK: _OK_SYM,
            diagnostics.WARN: _WARN_SYM,
            diagnostics.FAIL: _FAIL_SYM}


def _render(result) -> None:
    """Print one ``diagnostics.CheckResult``.

    The five ACTIVE probes live in ``localm/diagnostics.py``, which the GUI runs
    directly. Everything terminal-specific stays here - the symbols, the dim
    parenthetical, the seven-space hint indent.

    A result with NO findings prints nothing."""
    from rich.markup import escape

    for f in result.findings:
        # diagnostics.py builds text/note/hints from raw probe output and never
        # escapes; this renderer is where the markup escaping happens.
        note = f" [dim]({escape(f.note)})[/dim]" if f.note else ""
        console.print(f"  {_SYM_FOR.get(f.status, _WARN_SYM)}  {escape(f.text)}{note}")
        for hint in f.hints:
            console.print(f"       [dim]{escape(hint)}[/dim]")


def _check_python() -> None:
    """Matches pyproject.toml's ``requires-python`` pin exactly: 3.12 only. Any
    older or newer interpreter is not reported as OK."""
    import sys as _sys
    major, minor = _sys.version_info[:2]
    if (major, minor) == (3, 12):
        console.print(f"  {_OK_SYM}  Python {major}.{minor}")
    else:
        console.print(f"  {_FAIL_SYM}  Python {major}.{minor} - 3.12 required")


def _check_llama_lib(find_binary_dir) -> bool:
    """Print the llama.dll/.so health line(s); return True if a healthy lib is
    present. The probe itself (presence, 0-byte/truncated detection, and the
    rocBLAS/hipBLASLt kernel-data check) is diagnostics.check_llama_lib."""
    result = diagnostics.check_llama_lib(find_binary_dir)
    _render(result)
    return result.healthy


def _check_native_abi() -> None:
    """Print the native ABI self-check line. The probe is
    diagnostics.check_native_abi, which loads the provisioned runtime in a
    subprocess."""
    _render(diagnostics.check_native_abi())


def _check_worker_spawn() -> None:
    """Print the isolated-worker-spawn line. The probe (a real
    ``multiprocessing.get_context("spawn")`` round trip) is
    diagnostics.check_worker_spawn."""
    _render(diagnostics.check_worker_spawn())


def _check_venv_creation() -> None:
    """Print the nested-venv-creation line. The probe (a real ``-m venv`` plus a
    pip-landed check) is diagnostics.check_venv_creation."""
    _render(diagnostics.check_venv_creation())


def _check_gpu_driver() -> bool:
    """Probe nvidia-smi / rocm-smi; return True if a GPU driver was found.

    ONLY A CLEAN EXIT COUNTS AS A GPU: these tools report a broken driver by
    printing an error and exiting non-zero, so non-empty stdout on a non-zero
    exit is not a GPU.

    A tool that is absent stays silent. A tool that is PRESENT AND FAILING is
    surfaced as a warning naming what it said. Neither counts as a GPU; the
    return value feeds ``smi_or_torch_gpu`` only, and the real verdict comes
    from _check_gpu_verdict's device probe."""
    import subprocess
    from localm.debuglog import logger as _dbg
    from rich.markup import escape
    for cmd, label in [
        (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
         "NVIDIA"),
        (["rocm-smi", "--showproductname"],
         "AMD ROCm"),
    ]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            continue          # not installed
        except Exception as e:
            # A timeout or an OS-level spawn failure is NOT the same as absent:
            # the tool is there and did not answer. Logged, not escalated.
            _dbg.debug("doctor: %s probe (%s) did not complete: %r",
                       label, cmd[0], e)
            continue
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            said = (out or (r.stderr or "").strip() or "no output")
            _dbg.debug("doctor: %s probe (%s) exited %d: %s",
                       label, cmd[0], r.returncode, said)
            # `said` is nvidia-smi/rocm-smi's own stderr/stdout, so it is
            # escaped; `cmd[0]` and `label` are the literals above.
            console.print(
                f"  {_WARN_SYM}  {cmd[0]} is installed but failed "
                f"(exit {r.returncode}): {escape(said.splitlines()[0])}")
            continue
        if out:
            # `first_line` is hardware/driver text reported by
            # nvidia-smi/rocm-smi, so it is escaped.
            first_line = out.splitlines()[0]
            console.print(f"  {_OK_SYM}  {label} GPU: {escape(first_line)}")
            return True
    return False


def _check_vram_torch() -> bool:
    """Probe torch for GPU/VRAM; return True if torch sees a usable GPU.

    Runs BEFORE the "CPU mode only" verdict is decided: the smi tools can be
    absent while torch still sees a usable GPU (common on ROCm installs without
    rocm-smi on PATH), so torch's view vetoes the CPU-only warning.

    Skips the torch attempt entirely, returning False, once
    ``_loader.native_lib_loaded()`` is True: with llama.cpp's native runtime
    already loaded in this process, a later ``import torch`` on this project's
    Windows + AMD ROCm build hits STATUS_ENTRYPOINT_NOT_FOUND.

    Catches ``Exception``, not only the ``ImportError`` that means "torch is
    not installed": ``doctor()`` calls this with no try/except of its own, so
    any other torch import failure must not escape and truncate every later
    doctor check."""
    torch_gpu_found = False
    from rich.markup import escape

    from localm.inference.backends.llamacpp import _loader
    if _loader.native_lib_loaded():
        from localm.debuglog import logger as _dbg
        _dbg.debug(
            "doctor: skipping the torch VRAM probe - llama.cpp's native "
            "runtime is already loaded in this process, so `import torch` "
            "here is the known-doomed DLL-identity conflict (see "
            "VramSizingMixin._free_total_vram_bytes's docstring); "
            "torch_gpu_found stays False for this supplementary line")
        return torch_gpu_found
    try:
        import torch
        if torch.cuda.is_available():
            from localm import discover
            # _GPU_PROBE_CLI_DEADLINE is the cold-init-tolerant deadline,
            # passed explicitly so this call keeps it if the default ever
            # changes. Never raises; [] means no correction is available and
            # the raw readings below stand.
            #
            # return_status=True: without the status, list_gpus() can return a
            # served last-known-good list from an earlier probe when this call
            # times out, is busy or is inconclusive - a stale reading with
            # valid free/free_scope fields, indistinguishable from a fresh one.
            # The status is what makes `untrusted` below correct.
            try:
                _gpus_corrected, _corrected_status = discover.list_gpus(
                    deadline=discover._GPU_PROBE_CLI_DEADLINE, return_status=True)
            except Exception as e:
                from localm.debuglog import logger as _dbg
                _dbg.debug("doctor: device-global VRAM correction probe failed "
                           "(%s); showing the raw torch readings", type(e).__name__)
                _gpus_corrected = []
                _corrected_status = discover.GPU_PROBE_INCONCLUSIVE
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Driver-level free/total, CORRECTED for cross-process
                # blindness. torch.cuda.mem_get_info's raw "free" is not the
                # whole board's on every platform: on Windows + an AMD ROCm/HIP
                # build it reports total minus THIS process's own allocations
                # and is blind to every other process. discover.list_gpus()
                # applies the device-global correction where one exists and
                # tags what it could not correct.
                free_b, total_b = torch.cuda.mem_get_info(i)
                scope = None
                corrected = False
                for g in _gpus_corrected:
                    if g.get("index") == i and g.get("free") is not None:
                        free_b, scope = g["free"], g.get("free_scope")
                        corrected = True
                        break
                # Which reading `free_b` is decides whether it can be trusted:
                #   * corrected (a device-global entry was substituted in) -
                #     trustworthy only when that correction was fresh
                #     (status == GPU_PROBE_OK) and device-global, not
                #     PROCESS-scoped.
                #   * uncorrected (the raw, synchronously-fresh
                #     torch.cuda.mem_get_info() value) - trustworthy unless
                #     this platform is blind to other processes' VRAM
                #     (Windows + AMD ROCm/HIP), where the raw reading is always
                #     process-scoped.
                from localm import gpu_usage
                untrusted = (
                    (corrected and (
                        _corrected_status != discover.GPU_PROBE_OK
                        or scope != discover.FREE_SCOPE_DEVICE))
                    or (not corrected and gpu_usage.raw_reading_is_process_scoped())
                )
                # An untrusted free figure is not printed at all: the line
                # shows the total only, with a note in place of the figure.
                if not untrusted:
                    free_s = f"{free_b / 1024**3:.1f} GB free / "
                    note = ""
                else:
                    free_s = ""
                    note = "  [yellow](free VRAM reading unavailable on this platform)[/yellow]"
                # `props.name` is the CUDA/ROCm driver's own device name, so
                # it is escaped.
                console.print(
                    f"  {_OK_SYM}  GPU {i}: {escape(props.name)}  "
                    f"{free_s}{total_b / 1024**3:.1f} GB total"
                    f"{note}"
                )
                torch_gpu_found = True
        else:
            console.print(f"  {_WARN_SYM}  torch available but torch.cuda.is_available() = False")
    except ImportError:
        console.print(f"  {_WARN_SYM}  torch not installed - GPU VRAM check skipped")
    except Exception as e:
        from localm.debuglog import logger as _dbg
        _dbg.debug("doctor: torch GPU/VRAM probe failed with %s (not a plain "
                   "ImportError, possibly the DLL-identity conflict above "
                   "reached some other way); GPU VRAM check skipped for this "
                   "supplementary line - see _check_gpu_verdict for the real "
                   "verdict", type(e).__name__)
        # A dynamically-constructed exception class can carry an arbitrary
        # __name__, so it is escaped.
        console.print(f"  {_WARN_SYM}  torch GPU/VRAM probe failed "
                      f"({escape(type(e).__name__)}) - skipped")
    return torch_gpu_found


# The llama.cpp backends that run inference on the GPU. localm's DEFAULT setup
# provisions one of these on any GPU box (vulkan for NVIDIA/Intel/mixed, the
# self-contained amd-rocm build for AMD on Windows, metal on Apple Silicon), or
# the user pins cuda/sycl/hip. None of them needs nvidia-smi/rocm-smi on PATH,
# and torch.cuda is False for all but CUDA/ROCm-torch. "custom" is a
# user-supplied build of unknown class, so it is NOT assumed GPU by name; its
# capability comes from the real device probe instead.
_GPU_BACKENDS = frozenset({"vulkan", "cuda", "sycl", "hip", "metal", "amd-rocm"})

# Run in a FRESH interpreter: the loader mutates the DLL/lib search path and
# pulls in the native GPU runtime, so a broken or mismatched build cannot crash
# doctor, and the probe matches the real run environment. It prints one JSON
# line reporting whether the runtime computes and the exact ggml devices it
# registered.
_GPU_PROBE_CODE = """
import json
from localm.inference.backends.llamacpp import _loader
r = {"loaded": False, "devices": [], "error": ""}
try:
    r["loaded"] = bool(_loader.compute_backends_available())
    r["devices"] = [[n, t] for (n, t) in _loader.compute_devices()]
except Exception as e:
    r["error"] = repr(e)
print("GPU_PROBE:" + json.dumps(r))
"""


def _provisioned_backend_name(find_binary_dir) -> Optional[str]:
    """The llama.cpp backend localm has provisioned (read from the runtime dir's
    .localm-backend marker), or None when unprovisioned or unmarked (an old
    install predating the marker, or a hand-placed build)."""
    binary_dir = find_binary_dir()
    if not binary_dir:
        return None
    try:
        from localm.setup_llama import _provisioned_backend
        return _provisioned_backend(binary_dir)
    except Exception:
        return None


def _check_runtime_build(find_binary_dir) -> None:
    """Print WHICH llama.cpp build is provisioned, and whether it is pinned.

    Silent when nothing is provisioned at all. An unrecorded build is stated
    as unrecorded, never guessed."""
    if not find_binary_dir():
        return
    from rich.markup import escape

    try:
        from localm.setup_llama import installed_build, pinned_tag
        build = installed_build()
        pin = pinned_tag()
    except Exception as e:
        console.print(f"  {_WARN_SYM}  could not read the llama.cpp build "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    # `build` is the second token of a marker file that is not re-validated on
    # read, and `pin` is validated by pinned_tag()'s own is_safe_tag(); both are
    # escaped.
    if build:
        console.print(f"  {_OK_SYM}  llama.cpp build: {escape(build)}")
    else:
        console.print(f"  {_WARN_SYM}  llama.cpp build not recorded - this "
                      "install predates build recording")
        console.print("     [dim]'localm setup-llama --force' records it; "
                      "until then a bug report cannot say which build this "
                      "is[/dim]")
    if pin and build and pin != build:
        # The pin and the disk disagree: a pin was set but never provisioned
        # through (setup-llama not re-run, or it fell back to another backend).
        console.print(f"  {_WARN_SYM}  pinned to {escape(pin)} but {escape(build)} "
                      "is installed - run 'localm setup-llama --force' to apply "
                      "the pin")
    elif pin:
        console.print(f"     [dim]pinned to {escape(pin)}; 'localm setup-llama "
                      "--tag latest' resumes tracking upstream[/dim]")


def _probe_gpu_devices() -> Optional[dict]:
    """Load the provisioned runtime in a subprocess and enumerate the ggml
    compute devices it registers - the ground truth for whether localm runs
    inference on the GPU. Returns the parsed
    ``{"loaded": bool, "devices": [[name, type], ...], "error": str}`` or None
    when the probe could not run at all. Subprocess-isolated, so a broken GPU
    build never crashes doctor."""
    from .errors import _run_probe_subprocess
    return _run_probe_subprocess(_GPU_PROBE_CODE, "GPU_PROBE:")


def _check_gpu_verdict(find_binary_dir, lib_healthy: bool, smi_or_torch_gpu: bool) -> None:
    """Report GPU capability from what localm will ACTUALLY use for inference,
    not from nvidia-smi / rocm-smi / torch alone, which miss the Vulkan, Metal
    and bundled-ROCm paths.

    Order of truth:
      1) a real load probe of the provisioned runtime - the ggml compute DEVICES
         it registers (a GPU build registers a GPU/ACCEL device beside the CPU);
      2) failing that, the provisioned backend marker (vulkan/cuda/.../metal =>
         GPU, cpu => CPU-only);
      3) failing that, the smi/torch signal already printed above, hedged: a GPU
         they see is positive proof, their silence is not.

    A probe that RAN AND RAISED is none of those: it is a third outcome,
    reported as its own verdict between (2) and (3).

    "CPU mode only" is emitted only for a genuine cpu build, a runtime that
    loads with no GPU device, or a truly indeterminate no-signal case."""
    from rich.markup import escape

    backend = _provisioned_backend_name(find_binary_dir)
    probe = _probe_gpu_devices() if lib_healthy else None
    loaded = bool(probe.get("loaded")) if probe else False
    devices = (probe.get("devices") or []) if probe else []
    # _GPU_PROBE_CODE captures the exception that stopped it, which separates
    # "the runtime says there is no GPU" from "the runtime could not be asked".
    # str() normalises the JSON-round-tripped value; splitlines()[0] keeps only
    # the first line of a native loader repr, which can carry a whole traceback.
    probe_error = str((probe.get("error") or "") if probe else "").strip()
    if probe_error:
        # Logged here rather than inside the verdict branch below, so a
        # captured reason is never dropped on the paths that return earlier.
        from localm.debuglog import logger as _dbg
        _dbg.debug("doctor: GPU device probe reported an error: %s", probe_error)
    probe_error = probe_error.splitlines()[0] if probe_error else ""

    # Each verdict is a SHORT primary line that a narrow terminal cannot split
    # mid-phrase, with any elaboration on a separate dim hint line. `backend`
    # comes from the same unvalidated marker file as `build` above, so it is
    # escaped once here and every use of `tag` below is already safe.
    tag = f" ({escape(backend)})" if backend and backend != "custom" else ""

    # (1) Ground truth: the runtime loaded and reported its real compute devices.
    if loaded and devices:
        # Count only true GPU devices (ggml type GPU). ACCEL devices (BLAS / RPC)
        # are CPU-side or remote accelerators, NOT a GPU, so they must not be
        # labelled "GPU"; all of localm's GPU backends (Vulkan/Metal/CUDA/HIP/
        # SYCL/bundled-ROCm) register as GPU type, so this loses no real GPU.
        gpu = [name for (name, dtype) in devices
               if int(dtype) == _loader_gpu_type()]
        if gpu:
            # escape(): device names come from the native ggml compute-device
            # probe - hardware/driver-reported text (Vulkan/ROCm/CUDA device
            # names), same risk class as nvidia-smi/rocm-smi output.
            console.print(f"  {_OK_SYM}  GPU: {', '.join(escape(g) for g in gpu)}"
                          f"{tag} - used for inference")
        else:
            console.print(f"  {_WARN_SYM}  GPU: none in the loaded runtime{tag} - CPU mode only")
        return

    # (2) No device registry (older build) or the probe could not enumerate:
    #     trust the provisioned backend NAME, but only when the runtime is not
    #     known-broken - it loaded, OR the probe did not run for a benign reason
    #     (a healthy lib that was not load-tested here). A lib flagged as
    #     truncated/corrupt (lib_healthy False) is never reported as a working
    #     GPU from its marker alone.
    marker_trustworthy = loaded or (probe is None and lib_healthy)
    if marker_trustworthy and backend in _GPU_BACKENDS:
        console.print(f"  {_OK_SYM}  GPU: '{escape(backend)}' backend provisioned "
                      "- used for inference")
        return
    if marker_trustworthy and backend == "cpu":
        console.print(f"  {_WARN_SYM}  GPU: 'cpu' backend provisioned - CPU mode only")
        console.print("       [dim]run 'localm setup-llama --backend vulkan' to enable GPU[/dim]")
        return

    # (2b) The probe RAN AND FAILED: not "no GPU" but "could not find out", and
    #      the runtime being unable to load is itself the fault to report. This
    #      branch sits ABOVE the smi_or_torch_gpu early return, so a card those
    #      tools can see plus a runtime that will not load is still named.
    #      No line here points at the debug log: `doctor` takes no --debug flag.
    if probe_error:
        console.print(f"  {_WARN_SYM}  GPU: could not be determined{tag} - the runtime probe failed")
        # `probe_error` is repr(e) of whatever exception the native loader
        # raised, JSON round-tripped out of a subprocess, so it is escaped.
        console.print(f"       [dim]{escape(probe_error)}[/dim]")
        # No provisioning advice here: the runtime IS provisioned, and
        # _check_llama_lib / _check_native_abi have already said what to do
        # about a runtime that will not load.
        console.print("       [dim]the runtime is installed but did not load, so this is not "
                      "a statement about your hardware[/dim]")
        return

    # (3) No trustworthy runtime signal. A GPU seen by smi/torch is positive
    #     proof (already printed by those checks); their silence is NOT proof
    #     of CPU-only, so the line is hedged.
    if smi_or_torch_gpu:
        return
    console.print(f"  {_WARN_SYM}  No GPU detected (nvidia-smi / rocm-smi / torch) - CPU mode only")
    console.print("       [dim]those miss Vulkan/Metal/bundled-ROCm GPUs; run "
                  "'localm setup-llama' to provision one[/dim]")


def _loader_gpu_type() -> int:
    """The ggml GPU device-type value, read from the loader so doctor and the
    runtime never disagree on the enum. Falls back to 1
    (GGML_BACKEND_DEVICE_TYPE_GPU) if the loader cannot be imported."""
    try:
        from localm.inference.backends.llamacpp import _loader
        return int(_loader.GGML_DEV_TYPE_GPU)
    except Exception:
        return 1


def _check_packages() -> dict:
    """Print each package's presence/version line and return the imported
    module objects, keyed by import name, with None where the import failed.

    The returned modules let a caller run a deeper USABILITY check on top of a
    package this function only confirmed is importable."""
    import importlib
    import importlib.metadata as _ilm
    from rich.markup import escape
    packages = [
        ("fastapi",           "FastAPI (HTTP server)"),
        ("uvicorn",           "uvicorn (ASGI server)"),
        ("huggingface_hub",   "huggingface-hub (model downloads)"),
        ("requests",          "requests (HTTP client)"),
        ("rich",              "rich (terminal output)"),
        ("click",             "click (CLI)"),
    ]
    optional_pkgs = [
        ("torch",             "torch (HF backend / GPU info)"),
        ("transformers",      "transformers (HF backend)"),
    ]
    # Import name -> distribution name where they differ (for version lookup).
    _dist_names = {"huggingface_hub": "huggingface-hub"}
    modules: dict = {}
    for mod, label in packages + optional_pkgs:
        # With llama.cpp's native runtime already loaded in this process, a
        # later `import torch` is doomed on this project's Windows + AMD ROCm
        # build, so it is skipped rather than attempted.
        #
        # Narrower than a blanket native_lib_loaded() skip: a torch already
        # resident in sys.modules makes `import torch` here a cache hit rather
        # than a fresh preload, so the conflict cannot occur and the working
        # module handle is kept for _check_hf_backend_usable below.
        if mod == "torch" and "torch" not in sys.modules:
            from localm.inference.backends.llamacpp import _loader
            if _loader.native_lib_loaded():
                from localm.debuglog import logger as _dbg
                _dbg.debug(
                    "doctor: skipping the torch package import - llama.cpp's "
                    "native runtime is already loaded in this process, so "
                    "`import torch` here is the known-doomed DLL-identity "
                    "conflict (see VramSizingMixin._free_total_vram_bytes's "
                    "docstring)")
                modules[mod] = None
                console.print(f"  {_WARN_SYM}  {label} - skipped (native GPU "
                              "runtime already loaded in this process)")
                continue
        try:
            m = importlib.import_module(mod)
            modules[mod] = m
            # Read the version from installed distribution METADATA first,
            # not module.__version__: click removed __version__ in 9.1 and
            # accessing it emits a DeprecationWarning. Fall back to
            # __version__ only for a package whose dist metadata is missing.
            try:
                ver = _ilm.version(_dist_names.get(mod, mod))
            except _ilm.PackageNotFoundError:
                # A lazy module (transformers._LazyModule) raises
                # ModuleNotFoundError from __getattr__ for ANY name it cannot
                # resolve, __version__ included, and getattr()'s default only
                # suppresses AttributeError. Unguarded that would escape into
                # the `except ImportError` below, since ModuleNotFoundError IS
                # an ImportError, and report an importable package as "not
                # installed". The version string is dropped instead.
                try:
                    ver = getattr(m, "__version__", "")
                except Exception:
                    ver = ""
            sym = _OK_SYM
            # `ver` comes from a third-party package's distribution metadata
            # or __version__ attribute, so it is escaped.
            ver_str = f" {escape(ver)}" if ver else ""
        except ImportError:
            modules[mod] = None
            sym     = _WARN_SYM if (mod, label) in optional_pkgs else _FAIL_SYM
            ver_str = " - not installed"
        except Exception as e:
            # ONLY torch has the DLL-identity conflict above; any other
            # package raising something other than ImportError is re-raised
            # rather than masked.
            if mod != "torch":
                raise
            modules[mod] = None
            from localm.debuglog import logger as _dbg
            _dbg.debug("doctor: torch import failed with %s (not a plain "
                       "ImportError, possibly the DLL-identity conflict "
                       "reached some other way); reported as unavailable for "
                       "this run", type(e).__name__)
            sym = _WARN_SYM
            # An exception class name can be arbitrary, so it is escaped.
            ver_str = f" - import failed ({escape(type(e).__name__)})"
        console.print(f"  {sym}  {label}{ver_str}")
    return modules


def _check_hf_backend_usable(torch_mod, transformers_mod) -> None:
    """Print the HF-backend-usable line. The probe is
    diagnostics.check_hf_backend, which resolves transformers' LAZY Auto*
    classes for real, separating "installed" from "usable".

    The two module handles come from ``_check_packages`` above, which has
    already decided whether each is importable here, so ``resolved=True`` is
    passed: a None means absent, and the core does not re-import torch."""
    _render(diagnostics.check_hf_backend(torch_mod, transformers_mod,
                                         resolved=True))


def _check_plugin_deps() -> None:
    """Report enabled plugins whose declared pip extras are not installed, and
    point at the one-shot fix."""
    from rich.markup import escape

    try:
        from localm.plugins.engine import PluginManager
        missing = PluginManager(None).all_missing_deps(enabled_only=True)
    except Exception as e:
        console.print(f"  {_WARN_SYM}  plugin dependency check skipped "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    if not missing:
        console.print(f"  {_OK_SYM}  plugin dependencies: enabled plugins have theirs")
        return
    # `name`/`reqs` come from an installed plugin's own manifest, so they are
    # escaped. Quoted by hand rather than with !r/repr(): repr() applied after
    # escape() re-escapes escape()'s own backslash, and escape() applied to a
    # repr'd string is wrong in the other direction.
    for name, reqs in missing.items():
        console.print(f"  {_WARN_SYM}  plugin '{escape(name)}' is missing: "
                      f"{', '.join(escape(r) for r in reqs)}")
    console.print("       [dim]Install them with:  localm plugin install-deps --all[/dim]")


def _check_managed_comfy() -> None:
    """Discovery hint for the opt-in localm-managed ComfyUI.

    Purely informational: it installs nothing and never changes doctor's
    verdict. When no managed instance exists it points at `localm comfy setup`;
    when one IS installed it reports where that lives. The hint is an info
    line, not a warning.

    A probe fault is surfaced as a skipped line rather than muted or escalated
    into a doctor failure.
    """
    from rich.markup import escape

    try:
        from localm.media.managed_comfy import (
            is_managed_comfy_installed,
            managed_comfy_paths,
        )
        installed = is_managed_comfy_installed()
    except Exception as e:  # noqa: BLE001 - a hint must not fail doctor; surface why.
        console.print(f"  {_WARN_SYM}  managed-ComfyUI hint skipped "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    # soft_wrap keeps these copy-pasteable path/command lines from breaking
    # mid-token on a narrow terminal. It affects only these lines.
    if installed:
        # A real filesystem path under LOCALM_HOME, so it is escaped.
        console.print(
            f"  {_OK_SYM}  managed ComfyUI: installed at "
            f"{escape(str(managed_comfy_paths().root))}",
            soft_wrap=True,
        )
    else:
        console.print(
            f"  {_HINT_SYM}  localm can manage its own ComfyUI "
            "(isolated, patched, pinned): run 'localm comfy setup'",
            soft_wrap=True,
        )


@main.command()
def doctor():
    """Check system requirements and report any issues.

    \b
    Verifies:
      - Python version (3.12 required)
      - llama.dll / llama.so available on PATH or in expected locations
      - The isolated worker process (used by every GGUF model load and the
        voice/STT engine) can actually be spawned - a real subprocess.Popen
        probe passes even when this cannot
      - A nested venv can actually be created via `-m venv` - the same
        mechanism the managed-ComfyUI installer depends on
      - GPU inference capability, from the backend localm actually provisioned
        (Vulkan / Metal / bundled-ROCm / CUDA), not just nvidia-smi/rocm-smi/torch
      - Available VRAM
      - Required Python packages (huggingface-hub, torch, uvicorn, fastapi)
      - The HF (transformers) backend is not just installed but actually
        USABLE - AutoTokenizer/AutoProcessor/AutoModelForCausalLM really load
      - Enabled plugins have their pip extras installed
    Also surfaces a one-line discovery hint for the opt-in managed ComfyUI.
    """
    # Resolve find_binary_dir from the package at call time so tests that
    # monkeypatch localm.cli.find_binary_dir affect this call site.
    from localm import cli as _cli
    find_binary_dir = _cli.find_binary_dir

    _check_python()
    lib_healthy = _check_llama_lib(find_binary_dir)
    # Immediately after the lib line and BEFORE the ABI check.
    _check_runtime_build(find_binary_dir)
    # native ABI self-check only when a healthy lib is present.
    if lib_healthy:
        _check_native_abi()
    _check_worker_spawn()
    _check_venv_creation()
    # smi/torch are SUPPLEMENTARY detail lines, not the verdict: they miss
    # localm's default Vulkan/Metal/bundled-ROCm GPU paths entirely. The
    # verdict is derived from what localm will actually load.
    gpu_found = _check_gpu_driver()
    torch_gpu_found = _check_vram_torch()
    _check_gpu_verdict(find_binary_dir, lib_healthy, gpu_found or torch_gpu_found)
    modules = _check_packages()
    _check_hf_backend_usable(modules.get("torch"), modules.get("transformers"))
    _check_plugin_deps()
    _check_managed_comfy()
