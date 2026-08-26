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
# Neutral info marker for discovery hints that are not pass/fail checks.
_HINT_SYM = "[cyan]i[/cyan]"

_SYM_FOR = {diagnostics.OK: _OK_SYM,
            diagnostics.WARN: _WARN_SYM,
            diagnostics.FAIL: _FAIL_SYM}


def _render(result) -> None:
    """Print one ``diagnostics.CheckResult``.

    The probes themselves live in ``localm/diagnostics.py``; everything
    terminal-specific stays here - the symbols, the dim parenthetical, the
    seven-space hint indent.

    A result with NO findings prints nothing."""
    for f in result.findings:
        note = f" [dim]({f.note})[/dim]" if f.note else ""
        console.print(f"  {_SYM_FOR.get(f.status, _WARN_SYM)}  {f.text}{note}")
        for hint in f.hints:
            console.print(f"       [dim]{hint}[/dim]")


def _check_python() -> None:
    """Print the Python-version line. Only 3.12 passes, matching
    pyproject.toml's ``requires-python`` pin; every other version prints
    FAIL."""
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
    """Print the native ABI self-check line. The probe (a subprocess load of the
    provisioned runtime) is diagnostics.check_native_abi."""
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
    printing an error and exiting non-zero.

    A tool that is absent stays silent. A tool that is PRESENT AND FAILING is
    surfaced as a warning naming what it said. Neither counts as a GPU - the
    verdict comes from _check_gpu_verdict's device probe."""
    import subprocess
    from localm.debuglog import logger as _dbg
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
            # A timeout or an OS-level spawn failure: the tool is there and did
            # not answer. Logged, and the probe moves on.
            _dbg.debug("doctor: %s probe (%s) did not complete: %r",
                       label, cmd[0], e)
            continue
        out = (r.stdout or "").strip()
        if r.returncode != 0:
            said = (out or (r.stderr or "").strip() or "no output")
            _dbg.debug("doctor: %s probe (%s) exited %d: %s",
                       label, cmd[0], r.returncode, said)
            console.print(
                f"  {_WARN_SYM}  {cmd[0]} is installed but failed "
                f"(exit {r.returncode}): {said.splitlines()[0]}")
            continue
        if out:
            first_line = out.splitlines()[0]
            console.print(f"  {_OK_SYM}  {label} GPU: {first_line}")
            return True
    return False


def _check_vram_torch() -> bool:
    """Probe torch for GPU/VRAM; return True if torch sees a usable GPU.

    Runs BEFORE the "CPU mode only" verdict is decided, so a GPU only torch can
    see vetoes that warning.

    Skips the torch attempt entirely once ``_loader.native_lib_loaded()`` is
    True: with llama.cpp's own native runtime loaded in this process, a later
    ``import torch`` on this project's Windows + AMD ROCm build hits
    STATUS_ENTRYPOINT_NOT_FOUND. ``VramSizingMixin._vram_levels`` /
    ``_free_total_vram_bytes``, ``discover._list_gpus_probe`` and
    ``gpu_usage.raw_reading_is_process_scoped`` guard the same conflict.

    Never raises: the ``except Exception`` below covers any torch import
    failure beyond the plain ``ImportError``, so the later doctor checks still
    run."""
    torch_gpu_found = False
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
            # passed explicitly so this call cannot pick up a thinner default.
            # Never raises; [] means no correction is available and the raw
            # readings below stand. return_status=True is required: the bare
            # form can return a served last-known-good list from an earlier
            # probe, which without the status is indistinguishable from a fresh
            # reading.
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
                # blindness: on Windows + an AMD ROCm/HIP build
                # torch.cuda.mem_get_info's raw "free" counts only THIS
                # process's allocations and is blind to every other process.
                # discover.list_gpus() applies the device-global correction
                # where one exists and tags what it could not correct.
                free_b, total_b = torch.cuda.mem_get_info(i)
                scope = None
                corrected = False
                for g in _gpus_corrected:
                    if g.get("index") == i and g.get("free") is not None:
                        free_b, scope = g["free"], g.get("free_scope")
                        corrected = True
                        break
                # Trustworthy depends on which reading `free_b` actually is:
                #   * corrected (a device-global entry was substituted in) -
                #     trustworthy only when that correction was fresh
                #     (status == GPU_PROBE_OK) and device-global, not
                #     PROCESS-scoped.
                #   * uncorrected (the raw torch.cuda.mem_get_info() value) -
                #     trustworthy unless this platform is blind to other
                #     processes' VRAM (Windows + AMD ROCm/HIP), where the raw
                #     reading is always process-scoped.
                from localm import gpu_usage
                untrusted = (
                    (corrected and (
                        _corrected_status != discover.GPU_PROBE_OK
                        or scope != discover.FREE_SCOPE_DEVICE))
                    or (not corrected and gpu_usage.raw_reading_is_process_scoped())
                )
                # An untrusted reading prints the total only, with no free
                # figure beside it.
                if not untrusted:
                    free_s = f"{free_b / 1024**3:.1f} GB free / "
                    note = ""
                else:
                    free_s = ""
                    note = "  [yellow](free VRAM reading unavailable on this platform)[/yellow]"
                console.print(
                    f"  {_OK_SYM}  GPU {i}: {props.name}  "
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
        console.print(f"  {_WARN_SYM}  torch GPU/VRAM probe failed ({type(e).__name__}) - skipped")
    return torch_gpu_found


# The llama.cpp backends that run inference on the GPU. None of them needs
# nvidia-smi/rocm-smi on PATH, and torch.cuda is False for all but CUDA/ROCm
# torch. "custom" is a user-supplied build of unknown class and is absent from
# this set; its capability comes from the real device probe.
_GPU_BACKENDS = frozenset({"vulkan", "cuda", "sycl", "hip", "metal", "amd-rocm"})

# Run in a FRESH interpreter: the loader mutates the DLL/lib search path and
# pulls in the native GPU runtime. Prints one JSON line reporting whether the
# runtime computes and the exact ggml devices it registered.
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

    Silent when nothing is provisioned at all. An unrecorded build is stated as
    unrecorded, never guessed."""
    if not find_binary_dir():
        return
    try:
        from localm.setup_llama import installed_build, pinned_tag
        build = installed_build()
        pin = pinned_tag()
    except Exception as e:
        console.print(f"  {_WARN_SYM}  could not read the llama.cpp build "
                      f"[dim]({e})[/dim]")
        return
    if build:
        console.print(f"  {_OK_SYM}  llama.cpp build: {build}")
    else:
        console.print(f"  {_WARN_SYM}  llama.cpp build not recorded - this "
                      "install predates build recording")
        console.print("     [dim]'localm setup-llama --force' records it; "
                      "until then a bug report cannot say which build this "
                      "is[/dim]")
    if pin and build and pin != build:
        # The pin and the disk disagree: a pin was set but never provisioned
        # through (setup-llama not re-run, or it fell back to another backend).
        console.print(f"  {_WARN_SYM}  pinned to {pin} but {build} is installed"
                      " - run 'localm setup-llama --force' to apply the pin")
    elif pin:
        console.print(f"     [dim]pinned to {pin}; 'localm setup-llama --tag "
                      "latest' resumes tracking upstream[/dim]")


def _probe_gpu_devices() -> Optional[dict]:
    """Load the provisioned runtime in a subprocess and enumerate the ggml
    compute devices it registers. Returns the parsed
    ``{"loaded": bool, "devices": [[name, type], ...], "error": str}`` or None
    when the probe could not run at all."""
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

    A probe that RAN AND RAISED is a fourth outcome, reported as its own verdict
    between (2) and (3) - see the ``probe_error`` branch.

    "CPU mode only" is emitted only for a genuine cpu build, a runtime that
    loads with no GPU device, or a truly indeterminate no-signal case - never
    from the mere absence of the vendor CLIs."""
    backend = _provisioned_backend_name(find_binary_dir)
    probe = _probe_gpu_devices() if lib_healthy else None
    loaded = bool(probe.get("loaded")) if probe else False
    devices = (probe.get("devices") or []) if probe else []
    # _GPU_PROBE_CODE captures the exception that stopped it, which separates
    # "the runtime says there is no GPU" from "the runtime could not be asked".
    # str() across the subprocess/JSON boundary; splitlines()[0] because a repr
    # of a native loader error can carry a whole traceback.
    probe_error = str((probe.get("error") or "") if probe else "").strip()
    if probe_error:
        # Logged here rather than inside the verdict branch below, so a
        # captured reason is not dropped on the paths that return earlier.
        from localm.debuglog import logger as _dbg
        _dbg.debug("doctor: GPU device probe reported an error: %s", probe_error)
    probe_error = probe_error.splitlines()[0] if probe_error else ""

    # Each verdict is a SHORT primary line that a narrow terminal cannot wrap,
    # with any elaboration on a separate dim hint line.
    tag = f" ({backend})" if backend and backend != "custom" else ""

    # (1) Ground truth: the runtime loaded and reported its real compute devices.
    if loaded and devices:
        # Count only true GPU devices (ggml type GPU). ACCEL devices
        # (BLAS / RPC) are CPU-side or remote accelerators and are not counted;
        # every localm GPU backend (Vulkan/Metal/CUDA/HIP/SYCL/bundled-ROCm)
        # registers as GPU type.
        gpu = [name for (name, dtype) in devices
               if int(dtype) == _loader_gpu_type()]
        if gpu:
            console.print(f"  {_OK_SYM}  GPU: {', '.join(gpu)}{tag} - used for inference")
        else:
            console.print(f"  {_WARN_SYM}  GPU: none in the loaded runtime{tag} - CPU mode only")
        return

    # (2) No device registry (older build) or the probe could not enumerate:
    #     trust the provisioned backend NAME, but only when the runtime is not
    #     known-broken - it loaded, or the probe did not run while the lib is
    #     healthy. A lib flagged truncated/corrupt (lib_healthy False) is never
    #     reported as a working GPU from its marker alone.
    marker_trustworthy = loaded or (probe is None and lib_healthy)
    if marker_trustworthy and backend in _GPU_BACKENDS:
        console.print(f"  {_OK_SYM}  GPU: '{backend}' backend provisioned - used for inference")
        return
    if marker_trustworthy and backend == "cpu":
        console.print(f"  {_WARN_SYM}  GPU: 'cpu' backend provisioned - CPU mode only")
        console.print("       [dim]run 'localm setup-llama --backend vulkan' to enable GPU[/dim]")
        return

    # (2b) The probe RAN AND FAILED: reported as "could not be determined",
    #      never as "no GPU". Sits ABOVE the smi_or_torch_gpu early return,
    #      which would otherwise swallow the case of a card those tools CAN see
    #      plus a runtime that will not load. No printed line points at the
    #      debug log above: `doctor` takes no --debug flag.
    if probe_error:
        console.print(f"  {_WARN_SYM}  GPU: could not be determined{tag} - the runtime probe failed")
        console.print(f"       [dim]{probe_error}[/dim]")
        # No provisioning advice here: the runtime IS provisioned, and
        # _check_llama_lib / _check_native_abi have already said what to do
        # about a runtime that will not load.
        console.print("       [dim]the runtime is installed but did not load, so this is not "
                      "a statement about your hardware[/dim]")
        return

    # (3) No trustworthy runtime signal. A GPU seen by smi/torch is positive
    #     proof (already printed by those checks); their silence is not proof of
    #     CPU-only, so the line is hedged.
    if smi_or_torch_gpu:
        return
    console.print(f"  {_WARN_SYM}  No GPU detected (nvidia-smi / rocm-smi / torch) - CPU mode only")
    console.print("       [dim]those miss Vulkan/Metal/bundled-ROCm GPUs; run "
                  "'localm setup-llama' to provision one[/dim]")


def _loader_gpu_type() -> int:
    """The ggml GPU device-type value, read from the loader. Falls back to 1
    (the ggml GGML_BACKEND_DEVICE_TYPE_GPU constant) if the loader cannot be
    imported."""
    try:
        from localm.inference.backends.llamacpp import _loader
        return int(_loader.GGML_DEV_TYPE_GPU)
    except Exception:
        return 1


def _check_packages() -> dict:
    """Print each package's presence/version line and return the imported
    module objects, keyed by import name, with None where the import failed."""
    import importlib
    import importlib.metadata as _ilm
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
        # The DLL-identity conflict documented at _check_vram_torch: with
        # llama.cpp's native runtime already loaded in this process, importing
        # torch raises OSError [WinError 127] on this project's Windows + AMD
        # ROCm build, so the import is skipped before it is attempted. No other
        # package here shares this risk.
        #
        # A torch ALREADY resident in sys.modules is a plain cache hit, never a
        # fresh preload, so the conflict cannot occur and that module is kept -
        # _check_hf_backend_usable below needs the real module handle back.
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
            # Read the version from installed distribution METADATA first, not
            # module.__version__: click deprecated __version__ (removed in Click
            # 9.1) and accessing it emits a DeprecationWarning. Fall back to
            # __version__ only for a package whose dist metadata is missing.
            try:
                ver = _ilm.version(_dist_names.get(mod, mod))
            except _ilm.PackageNotFoundError:
                # A lazy module (transformers._LazyModule) raises
                # ModuleNotFoundError from __getattr__ for ANY name it cannot
                # resolve, __version__ included, and getattr()'s default only
                # suppresses AttributeError. Unguarded it escapes into the
                # `except ImportError` below (ModuleNotFoundError IS an
                # ImportError) and reports an importable package as "not
                # installed".
                try:
                    ver = getattr(m, "__version__", "")
                except Exception:
                    ver = ""
            sym = _OK_SYM
            ver_str = f" {ver}" if ver else ""
        except ImportError:
            modules[mod] = None
            sym     = _WARN_SYM if (mod, label) in optional_pkgs else _FAIL_SYM
            ver_str = " - not installed"
        except Exception as e:
            # ONLY torch has the DLL-identity conflict above; any other package
            # raising something other than ImportError is re-raised.
            if mod != "torch":
                raise
            modules[mod] = None
            from localm.debuglog import logger as _dbg
            _dbg.debug("doctor: torch import failed with %s (not a plain "
                       "ImportError, possibly the DLL-identity conflict "
                       "reached some other way); reported as unavailable for "
                       "this run", type(e).__name__)
            sym = _WARN_SYM
            ver_str = f" - import failed ({type(e).__name__})"
        console.print(f"  {sym}  {label}{ver_str}")
    return modules


def _check_hf_backend_usable(torch_mod, transformers_mod) -> None:
    """Print the HF-backend-usable line. The probe (resolving transformers'
    LAZY Auto* classes for real, which separates "installed" from "usable") is
    diagnostics.check_hf_backend.

    The two module handles come from ``_check_packages`` above, which has
    already decided whether each is importable here - hence ``resolved=True``:
    a None means absent and the core does not re-import torch."""
    _render(diagnostics.check_hf_backend(torch_mod, transformers_mod,
                                         resolved=True))


def _check_plugin_deps() -> None:
    """Report enabled plugins whose declared pip extras are not installed, and
    point at the one-shot fix."""
    try:
        from localm.plugins.engine import PluginManager
        missing = PluginManager(None).all_missing_deps(enabled_only=True)
    except Exception as e:
        console.print(f"  {_WARN_SYM}  plugin dependency check skipped [dim]({e})[/dim]")
        return
    if not missing:
        console.print(f"  {_OK_SYM}  plugin dependencies: enabled plugins have theirs")
        return
    for name, reqs in missing.items():
        console.print(f"  {_WARN_SYM}  plugin {name!r} is missing: {', '.join(reqs)}")
    console.print("       [dim]Install them with:  localm plugin install-deps --all[/dim]")


def _check_managed_comfy() -> None:
    """Discovery hint for the opt-in localm-managed ComfyUI.

    Purely informational: it installs nothing and never changes doctor's
    verdict. With no managed instance it points at `localm comfy setup`; with
    one installed it reports where it lives. The hint prints as an info line,
    not a warning, and a probe fault prints a skipped line instead of raising.
    """
    try:
        from localm.media.managed_comfy import (
            is_managed_comfy_installed,
            managed_comfy_paths,
        )
        installed = is_managed_comfy_installed()
    except Exception as e:  # noqa: BLE001 - a hint must not fail doctor; surface why.
        console.print(f"  {_WARN_SYM}  managed-ComfyUI hint skipped [dim]({e})[/dim]")
        return
    # soft_wrap: this line names a literal path/command and must not be broken
    # mid-token. Affects THIS line only; the rest of doctor's output still
    # wraps normally.
    if installed:
        console.print(
            f"  {_OK_SYM}  managed ComfyUI: installed at {managed_comfy_paths().root}",
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
        voice/STT engine) can actually be spawned
      - A nested venv can actually be created via `-m venv` - the mechanism
        the managed-ComfyUI installer depends on
      - GPU inference capability, from the backend localm actually provisioned
        (Vulkan / Metal / bundled-ROCm / CUDA), not just nvidia-smi/rocm-smi/torch
      - Available VRAM
      - Required Python packages (huggingface-hub, torch, uvicorn, fastapi)
      - The HF (transformers) backend is not just installed but actually
        USABLE - AutoTokenizer/AutoProcessor/AutoModelForCausalLM really load
      - Enabled plugins have their pip extras installed
    Also surfaces a one-line discovery hint for the opt-in managed ComfyUI.
    """
    # Resolve find_binary_dir from the package at call time, not at import time.
    from localm import cli as _cli
    find_binary_dir = _cli.find_binary_dir

    _check_python()
    lib_healthy = _check_llama_lib(find_binary_dir)
    _check_runtime_build(find_binary_dir)
    # native ABI self-check only when a healthy lib is present.
    if lib_healthy:
        _check_native_abi()
    _check_worker_spawn()
    _check_venv_creation()
    # smi/torch are SUPPLEMENTARY detail lines, not the verdict: they miss
    # localm's default Vulkan/Metal/bundled-ROCm GPU paths entirely.
    gpu_found = _check_gpu_driver()
    torch_gpu_found = _check_vram_torch()
    _check_gpu_verdict(find_binary_dir, lib_healthy, gpu_found or torch_gpu_found)
    modules = _check_packages()
    _check_hf_backend_usable(modules.get("torch"), modules.get("transformers"))
    _check_plugin_deps()
    _check_managed_comfy()
