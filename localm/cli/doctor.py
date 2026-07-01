# SPDX-License-Identifier: AGPL-3.0-or-later
from ._core import console, main

# ------------------------------------------------------------------ #
#  Doctor                                                              #
# ------------------------------------------------------------------ #

_OK_SYM   = "[green]✓[/green]"
_WARN_SYM = "[yellow]![/yellow]"
_FAIL_SYM = "[red]✗[/red]"


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
    import json as _json
    import subprocess
    import sys as _sys
    abi_code = (
        "import json;"
        "from localm.inference.backends.llamacpp._abi import abi_report;"
        "v=abi_report();"
        "print('ABI_RESULT:'+json.dumps("
        "{'status':v.status,'detail':v.detail,'failures':v.failures[:3]}))"
    )
    try:
        r = subprocess.run([_sys.executable, "-c", abi_code],
                           capture_output=True, text=True, timeout=120)
        line = next((ln for ln in (r.stdout or "").splitlines()
                     if ln.startswith("ABI_RESULT:")), "")
        abi = _json.loads(line[len("ABI_RESULT:"):]) if line else {}
    except Exception:
        abi = {}
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


@main.command()
def doctor():
    """Check system requirements and report any issues.

    \b
    Verifies:
      - Python version (3.10+ required)
      - llama.dll / llama.so available on PATH or in expected locations
      - CUDA / ROCm GPU driver
      - Available VRAM
      - Required Python packages (huggingface-hub, torch, uvicorn, fastapi)
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
    gpu_found = _check_gpu_driver()
    torch_gpu_found = _check_vram_torch()
    if not gpu_found and not torch_gpu_found:
        console.print(f"  {_WARN_SYM}  No GPU driver found (nvidia-smi / rocm-smi) - CPU mode only")
    _check_packages()
