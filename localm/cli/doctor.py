# SPDX-License-Identifier: AGPL-3.0-or-later
import multiprocessing as mp
import sys
from typing import Optional

from ._core import console, main

# ------------------------------------------------------------------ #
#  Doctor                                                              #
# ------------------------------------------------------------------ #

_OK_SYM   = "[green]✓[/green]"
_WARN_SYM = "[yellow]![/yellow]"
_FAIL_SYM = "[red]✗[/red]"
# A neutral info marker for discovery hints that are NOT pass/fail checks (an
# opt-in feature the user simply may not know about). Distinct from _WARN_SYM so
# it never reads as "something is wrong".
_HINT_SYM = "[cyan]i[/cyan]"


def _check_python() -> None:
    import sys as _sys
    major, minor = _sys.version_info[:2]
    if (major, minor) >= (3, 10):
        console.print(f"  {_OK_SYM}  Python {major}.{minor}")
    else:
        console.print(f"  {_FAIL_SYM}  Python {major}.{minor} - 3.10+ required")


def _check_llama_lib(find_binary_dir) -> bool:
    """Print the llama.dll/.so health line; return True if a healthy lib is present."""
    binary_dir = find_binary_dir()
    if not binary_dir:
        console.print(f"  {_FAIL_SYM}  llama binary dir not found - GGUF backend unavailable")
        return False
    dll_names = ["llama.dll", "llama.so", "libllama.so", "llama"]
    found_dll = next(
        (binary_dir / d for d in dll_names if (binary_dir / d).exists()),
        None,
    )
    if not found_dll:
        files = [f.name for f in binary_dir.iterdir() if f.is_file()][:8]
        console.print(
            f"  {_WARN_SYM}  binary dir found ({binary_dir}) but no llama .dll/.so - "
            f"contents: {files}"
        )
        return False
    # Existence alone is not health: a zeroed or truncated llama.dll exists but
    # cannot load. Check the file size, and flag a lib that is present but
    # implausibly small to be a real native library.
    try:
        size = found_dll.stat().st_size
    except OSError as e:
        console.print(
            f"  {_FAIL_SYM}  {found_dll.name} in {binary_dir} cannot be "
            f"read (corrupt?): {e}"
        )
        return False
    # A genuine llama.dll/.so is multiple MB. 64 KiB is a generous floor that
    # still rejects 0/1-byte stubs and tiny placeholders.
    TINY_LIB_BYTES = 64 * 1024
    if size == 0:
        console.print(
            f"  {_FAIL_SYM}  {found_dll.name} in {binary_dir} is empty "
            f"(0 bytes) - corrupt; re-run 'localm setup-llama'"
        )
        return False
    if size < TINY_LIB_BYTES:
        console.print(
            f"  {_WARN_SYM}  {found_dll.name} found in {binary_dir} but "
            f"is suspiciously small ({size} bytes, expected multiple MB) "
            f"- it may be truncated/corrupt"
        )
        return False
    console.print(f"  {_OK_SYM}  {found_dll.name} found in {binary_dir}")
    return True


def _check_native_abi() -> None:
    """Native ABI self-check (struct layout vs the actual DLL). Runs in a
    SUBPROCESS (like setup-llama's load test) so a broken/incompatible DLL can
    never crash doctor itself, and so the GPU runtime is loaded out-of-process."""
    from .errors import _run_probe_subprocess
    abi_code = (
        "import json;"
        "from localm.inference.backends.llamacpp._abi import abi_report;"
        "v=abi_report();"
        "print('ABI_RESULT:'+json.dumps("
        "{'status':v.status,'detail':v.detail,'failures':v.failures[:3]}))"
    )
    abi = _run_probe_subprocess(abi_code, "ABI_RESULT:") or {}
    status = abi.get("status", "unchecked")
    if status == "ok":
        console.print(f"  {_OK_SYM}  native ABI: struct layout matches this build")
    elif status == "mismatch":
        console.print(f"  {_FAIL_SYM}  native ABI MISMATCH - the runtime's struct "
                      "layout differs from this build; loading is refused to avoid "
                      "memory corruption. Run 'localm setup-llama --force'.")
        for f in abi.get("failures", []):
            console.print(f"       [dim]{f}[/dim]")
    elif status == "skipped":
        console.print(f"  {_WARN_SYM}  native ABI check skipped (LOCALM_SKIP_ABI_CHECK set)")
    else:
        console.print(f"  {_WARN_SYM}  native ABI not verified "
                      f"[dim]({abi.get('detail', 'runtime not loadable')})[/dim]")


def _worker_spawn_probe(conn) -> None:
    """Target of the spawn self-check below - runs ONLY in the spawned child.
    Module-level (not a closure): the "spawn" start method re-imports the
    target by its module path + name in the child, which only works for a
    plain top-level function. Does nothing but confirm it started; the point
    is proving the spawn ITSELF works, not anything the child does afterward."""
    try:
        conn.send("ok")
    finally:
        conn.close()


def _check_worker_spawn() -> None:
    """Verify localm can actually spawn its isolated worker process - the SAME
    ``multiprocessing.get_context("spawn")`` mechanism every GGUF model load and
    the voice/STT engine depend on (see localm/_mp_spawn.py, #617).

    The native-ABI and GPU-probe checks above isolate via a PLAIN subprocess
    (``_run_probe_subprocess``), a different code path - that proves the native
    library loads and computes correctly, but it does NOT exercise
    multiprocessing's own spawn machinery, which on Windows redirects the
    child's executable under conditions a renamed launcher (LocaLM.exe) can
    break. That gap is exactly why #617 (every GGUF load failing with
    "[WinError 2] The system cannot find the file specified") passed a doctor
    run showing everything green. This check would have caught it."""
    try:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = ctx.Process(target=_worker_spawn_probe, args=(child_conn,), daemon=True)
    except Exception as e:
        # Setup itself failed (e.g. the fix helper or Pipe() errored) - rarer
        # and genuinely different from a spawn failure, so it gets its own line.
        console.print(f"  {_FAIL_SYM}  background worker spawn check errored: {e}")
        return

    reply = None
    error_detail = None
    try:
        proc.start()
        got_reply = parent_conn.poll(20)
        reply = parent_conn.recv() if got_reply else None
        proc.join(5)
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
    except Exception as e:
        # This IS the #617 failure mode: proc.start() raises directly
        # (FileNotFoundError: [WinError 2] ...) rather than the child ever
        # running - treated the same as "no reply", not a separate error line,
        # so every way this can fail reads as one consistent, actionable verdict.
        error_detail = str(e)

    if reply == "ok":
        console.print(f"  {_OK_SYM}  background worker spawn: OK "
                      "(model loads and voice transcription use this)")
    else:
        detail = f" ({error_detail})" if error_detail else ""
        console.print(
            f"  {_FAIL_SYM}  background worker spawn FAILED{detail} - GGUF "
            "model loads and voice transcription will fail even though the "
            "runtime above checks out")


def _check_venv_creation() -> None:
    """Verify localm can actually create a nested venv via ``-m venv`` using
    ``real_base_python()`` - the SAME mechanism the managed-ComfyUI installer
    depends on (managed_comfy_fresh.py, #621). Creates and immediately discards a
    throwaway venv under a temp dir; never touches LOCALM_HOME or any real install.

    A DIFFERENT code path from the worker-spawn check above (that exercises
    multiprocessing's spawn machinery; this exercises stdlib venv's own basename-
    matching + mandatory ensurepip bootstrap): #621 (managed ComfyUI setup
    silently failing with "[WinError 2]") passed a doctor run showing everything
    green precisely because nothing probed this. This check would have caught it."""
    import subprocess
    import tempfile
    from pathlib import Path

    from localm._mp_spawn import real_base_python

    venv_python = real_base_python() or sys.executable
    ok = False
    detail = ""
    try:
        with tempfile.TemporaryDirectory(prefix="localm-doctor-venv-") as tmp:
            dest = Path(tmp) / "probe-venv"
            r = subprocess.run(
                [str(venv_python), "-m", "venv", str(dest)],
                capture_output=True, text=True, timeout=60)
            expected = dest / ("Scripts/python.exe" if sys.platform == "win32"
                               else "bin/python3")
            ok = r.returncode == 0 and expected.is_file()
            if not ok:
                tail = (r.stderr or r.stdout or "").strip().splitlines()
                detail = tail[-1] if tail else ""
    except Exception as e:
        console.print(f"  {_FAIL_SYM}  venv-creation check errored: {e}")
        return

    if ok:
        console.print(f"  {_OK_SYM}  venv creation: OK "
                      "(the managed-ComfyUI installer uses this)")
    else:
        console.print(
            f"  {_FAIL_SYM}  venv creation FAILED"
            + (f" ({detail})" if detail else "")
            + " - managed ComfyUI setup ('localm comfy setup') will fail even "
              "though the runtime above checks out")


def _check_gpu_driver() -> bool:
    """Probe nvidia-smi / rocm-smi; return True if a GPU driver was found."""
    import subprocess
    for cmd, label in [
        (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
         "NVIDIA"),
        (["rocm-smi", "--showproductname"],
         "AMD ROCm"),
    ]:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout.strip()
            if out:
                first_line = out.splitlines()[0]
                console.print(f"  {_OK_SYM}  {label} GPU: {first_line}")
                return True
        except Exception:
            continue
    return False


def _check_vram_torch() -> bool:
    """Probe torch for GPU/VRAM; return True if torch sees a usable GPU.

    Run the torch GPU probe BEFORE deciding the "CPU mode only" verdict: the smi
    tools can be absent while torch still sees a usable GPU (common on ROCm
    installs without rocm-smi on PATH). Printing both "CPU mode only" and a torch
    GPU/VRAM line in the same run contradicts itself, so let torch's view veto the
    CPU-only warning."""
    torch_gpu_found = False
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                # Driver-level free/total - torch's allocator counters miss
                # everything allocated outside torch (llama.dll, other apps)
                free_b, total_b = torch.cuda.mem_get_info(i)
                console.print(
                    f"  {_OK_SYM}  GPU {i}: {props.name}  "
                    f"{free_b / 1024**3:.1f} GB free / {total_b / 1024**3:.1f} GB total"
                )
                torch_gpu_found = True
        else:
            console.print(f"  {_WARN_SYM}  torch available but torch.cuda.is_available() = False")
    except ImportError:
        console.print(f"  {_WARN_SYM}  torch not installed - GPU VRAM check skipped")
    return torch_gpu_found


# The llama.cpp backends that run inference on the GPU. localm's DEFAULT setup
# provisions one of these on any GPU box (vulkan for NVIDIA/Intel/mixed, the
# self-contained amd-rocm build for AMD on Windows, metal on Apple Silicon), or
# the user pins cuda/sycl/hip. NONE of them needs nvidia-smi/rocm-smi on PATH,
# and torch.cuda is False for all but CUDA/ROCm-torch - which is exactly why the
# smi/torch probes above cannot be the GPU verdict (audit doctor-1). "custom" is
# a user-supplied build of unknown class, so it is NOT assumed GPU by name; its
# capability comes from the real device probe instead.
_GPU_BACKENDS = frozenset({"vulkan", "cuda", "sycl", "hip", "metal", "amd-rocm"})

# Run in a FRESH interpreter (like the ABI self-check): the loader mutates the
# DLL/lib search path and pulls in the native GPU runtime, so a broken or
# mismatched build can never crash doctor, and the probe matches the real run
# environment. It prints one JSON line reporting whether the runtime computes and
# the exact ggml devices it registered.
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


def _probe_gpu_devices() -> Optional[dict]:
    """Load the provisioned runtime in a subprocess and enumerate the ggml
    compute devices it registers - the ground truth for whether localm runs
    inference on the GPU. Returns the parsed
    ``{"loaded": bool, "devices": [[name, type], ...], "error": str}`` or None
    when the probe could not run at all (subprocess-isolated, so a broken GPU
    build never crashes doctor)."""
    from .errors import _run_probe_subprocess
    return _run_probe_subprocess(_GPU_PROBE_CODE, "GPU_PROBE:")


def _check_gpu_verdict(find_binary_dir, lib_healthy: bool, smi_or_torch_gpu: bool) -> None:
    """Report GPU capability from what localm will ACTUALLY use for inference,
    not from nvidia-smi / rocm-smi / torch alone (those miss the Vulkan, Metal
    and bundled-ROCm paths, so their silence is NOT proof of CPU-only - the
    audit doctor-1 false negative).

    Order of truth:
      1) a real load probe of the provisioned runtime - the ggml compute DEVICES
         it registers (a GPU build registers a GPU/ACCEL device beside the CPU);
      2) failing that, the provisioned backend marker (vulkan/cuda/.../metal =>
         GPU, cpu => CPU-only) - still far better than smi/torch;
      3) failing that, the smi/torch signal already printed above, hedged: a GPU
         they see is positive proof, their silence is not.

    "CPU mode only" is emitted only for a genuine cpu build, a runtime that loads
    with no GPU device, or a truly indeterminate no-signal case - never from the
    mere absence of the vendor CLIs the default GPU path never needs."""
    backend = _provisioned_backend_name(find_binary_dir)
    probe = _probe_gpu_devices() if lib_healthy else None
    loaded = bool(probe.get("loaded")) if probe else False
    devices = (probe.get("devices") or []) if probe else []

    # Each verdict is a SHORT primary line (the pass/fail phrase never wraps, so
    # a narrow terminal cannot split "CPU mode only" across a line break and mask
    # it), with any elaboration on a separate dim hint line - the doctor idiom.
    tag = f" ({backend})" if backend and backend != "custom" else ""

    # (1) Ground truth: the runtime loaded and reported its real compute devices.
    if loaded and devices:
        # Count only true GPU devices (ggml type GPU). ACCEL devices (BLAS / RPC)
        # are CPU-side or remote accelerators, NOT a GPU, so they must not be
        # labelled "GPU"; all of localm's GPU backends (Vulkan/Metal/CUDA/HIP/
        # SYCL/bundled-ROCm) register as GPU type, so this loses no real GPU.
        gpu = [name for (name, dtype) in devices
               if int(dtype) == _loader_gpu_type()]
        if gpu:
            console.print(f"  {_OK_SYM}  GPU: {', '.join(gpu)}{tag} - used for inference")
        else:
            console.print(f"  {_WARN_SYM}  GPU: none in the loaded runtime{tag} - CPU mode only")
        return

    # (2) No device registry (older build) or the probe could not enumerate:
    #     trust the provisioned backend NAME. Only when the runtime is not
    #     known-broken - it loaded, OR the probe did not run for a BENIGN reason
    #     (a healthy lib we simply chose not to load-test here). A lib doctor just
    #     flagged as truncated/corrupt (lib_healthy False) must NOT be reported as
    #     a working GPU from its marker alone (AGENTS.md rule 5: never report
    #     success for a known-broken state).
    marker_trustworthy = loaded or (probe is None and lib_healthy)
    if marker_trustworthy and backend in _GPU_BACKENDS:
        console.print(f"  {_OK_SYM}  GPU: '{backend}' backend provisioned - used for inference")
        return
    if marker_trustworthy and backend == "cpu":
        console.print(f"  {_WARN_SYM}  GPU: 'cpu' backend provisioned - CPU mode only")
        console.print("       [dim]run 'localm setup-llama --backend vulkan' to enable GPU[/dim]")
        return

    # (3) No trustworthy runtime signal. A GPU seen by smi/torch is positive
    #     proof (already printed by those checks); their silence is NOT proof of
    #     CPU-only, so hedge rather than state a false fact (audit doctor-1).
    if smi_or_torch_gpu:
        return
    console.print(f"  {_WARN_SYM}  No GPU detected (nvidia-smi / rocm-smi / torch) - CPU mode only")
    console.print("       [dim]those miss Vulkan/Metal/bundled-ROCm GPUs; run "
                  "'localm setup-llama' to provision one[/dim]")


def _loader_gpu_type() -> int:
    """The ggml GPU device-type value, read from the loader so doctor and the
    runtime never disagree on the enum. Falls back to 1 (the ggml
    GGML_BACKEND_DEVICE_TYPE_GPU constant) if the loader cannot be imported."""
    try:
        from localm.inference.backends.llamacpp import _loader
        return int(_loader.GGML_DEV_TYPE_GPU)
    except Exception:
        return 1


def _check_packages() -> None:
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
    for mod, label in packages + optional_pkgs:
        try:
            m = importlib.import_module(mod)
            # Read the version from installed distribution METADATA first, not
            # module.__version__: click deprecated __version__ (removed in Click
            # 9.1) and accessing it emits a DeprecationWarning, so metadata-first
            # avoids the warning and the eventual blank (AUD-CLICKVER). Fall back
            # to __version__ only for a package whose dist metadata is missing.
            try:
                ver = _ilm.version(_dist_names.get(mod, mod))
            except _ilm.PackageNotFoundError:
                ver = getattr(m, "__version__", "")
            sym = _OK_SYM
            ver_str = f" {ver}" if ver else ""
        except ImportError:
            sym     = _WARN_SYM if (mod, label) in optional_pkgs else _FAIL_SYM
            ver_str = " - not installed"
        console.print(f"  {sym}  {label}{ver_str}")


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
    """Discovery hint for the opt-in localm-managed ComfyUI (design decision 8).

    Purely informational: it installs nothing and never changes doctor's verdict.
    When no managed instance exists, nudge the user toward `localm comfy setup`;
    when one IS installed, report where it lives instead. The hint is an info
    line (not a warning) because a not-set-up opt-in feature is not a fault.

    A discovery hint must never break doctor, so a probe fault is SURFACED as a
    skipped line (AGENTS.md rule 5: surface, do not silence) rather than muted or
    escalated into a doctor failure.
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
    # soft_wrap: this line names a literal path/command a user may copy-paste; a
    # narrow terminal must never break it mid-token (only affects THIS line, not
    # the rest of doctor's output, which still wraps normally).
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
      - Python version (3.10+ required)
      - llama.dll / llama.so available on PATH or in expected locations
      - The isolated worker process (used by every GGUF model load and the
        voice/STT engine) can actually be spawned - a real subprocess.Popen
        probe passes even when this cannot (#617)
      - A nested venv can actually be created via `-m venv` - the same
        mechanism the managed-ComfyUI installer depends on (#621)
      - GPU inference capability, from the backend localm actually provisioned
        (Vulkan / Metal / bundled-ROCm / CUDA), not just nvidia-smi/rocm-smi/torch
      - Available VRAM
      - Required Python packages (huggingface-hub, torch, uvicorn, fastapi)
      - Enabled plugins have their pip extras installed
    Also surfaces a one-line discovery hint for the opt-in managed ComfyUI.
    """
    # Resolve find_binary_dir from the package at call time so tests that
    # monkeypatch localm.cli.find_binary_dir affect this call site.
    from localm import cli as _cli
    find_binary_dir = _cli.find_binary_dir

    _check_python()
    lib_healthy = _check_llama_lib(find_binary_dir)
    # native ABI self-check only when a healthy lib is present.
    if lib_healthy:
        _check_native_abi()
    _check_worker_spawn()
    _check_venv_creation()
    # smi/torch are SUPPLEMENTARY detail lines, not the verdict: they miss
    # localm's default Vulkan/Metal/bundled-ROCm GPU paths entirely. The verdict
    # is derived from what localm will actually load (audit doctor-1).
    gpu_found = _check_gpu_driver()
    torch_gpu_found = _check_vram_torch()
    _check_gpu_verdict(find_binary_dir, lib_healthy, gpu_found or torch_gpu_found)
    _check_packages()
    _check_plugin_deps()
    _check_managed_comfy()
