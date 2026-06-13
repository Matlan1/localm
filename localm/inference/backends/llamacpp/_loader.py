"""
DLL bootstrap for the llama.cpp native backend.

Resolves the native binary directory from project-local locations only - an
explicit override, the config, or the ``localm-llama-runtime`` wheel installed
in this venv - never a sibling folder elsewhere on disk. Adds the venv's
rocm-sdk runtime dirs (amdhip64/rocm_kpack/rocblas/…) to the DLL search path so
a llama build that bundles only llama.dll + ggml-*.dll still finds its runtime.

Provision the binaries with ``localm setup-llama``. The ``LLAMA_CPP_LIB``
env-var overrides the path to llama.dll for one-off use.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional

_loaded_lib: Optional[ctypes.CDLL] = None


def _candidate_dirs() -> List[Path]:
    """Directories that may hold llama.dll, in priority order."""
    dirs: List[Path] = []

    explicit = os.environ.get("LLAMA_CPP_LIB")
    if explicit:
        p = Path(explicit)
        dirs.append(p.parent if p.suffix else p)

    try:
        from localm.config import load_config
        bd = load_config().get("binary_dir")
        if bd:
            dirs.append(Path(bd))
    except Exception:
        pass

    # The self-contained location: binaries bundled in the venv via the
    # localm-llama-runtime wheel (populated by `localm setup-llama`).
    try:
        import localm_llama_runtime
        d = localm_llama_runtime.lib_dir()
        if d:
            dirs.append(Path(d))
    except Exception:
        pass

    return dirs


def runtime_binary_dir() -> Optional[Path]:
    """The directory llama.dll will be loaded from, or None if unprovisioned.
    Also used by the subprocess fallback to find llama-cli/llama-server."""
    for d in _candidate_dirs():
        try:
            if d and d.is_dir() and (d / "llama.dll").exists():
                return d
        except OSError:
            continue
    return None


def rocm_runtime_dirs() -> List[Path]:
    """ROCm runtime DLL directories inside this venv (the rocm-sdk wheels).

    These hold amdhip64_7.dll, rocm_kpack.dll, rocblas.dll, hipblas.dll, … -
    the libraries a HIP-linked llama.dll needs at load time. Globbed (not
    hardcoded) so any gfx target's package is picked up."""
    found: List[Path] = []
    roots = set()
    try:
        import site
        for p in site.getsitepackages():
            roots.add(Path(p))
        user = site.getusersitepackages()
        if user:
            roots.add(Path(user))
    except Exception:
        pass
    roots.add(Path(sys.prefix) / "Lib" / "site-packages")
    for root in roots:
        try:
            for d in root.glob("_rocm_sdk_*/bin"):
                if d.is_dir():
                    found.append(d)
        except OSError:
            continue
    return found


def _add_to_dll_path(directory: Path) -> None:
    """Make *directory* resolvable by the OS loader for transitive DLL deps."""
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        try:
            add(str(directory))
        except OSError:
            pass


def load_lib() -> ctypes.CDLL:
    """Load (and cache) the llama.dll shared library.

    Idempotent: subsequent calls return the already-loaded handle.
    Raises RuntimeError if the library cannot be found or loaded.
    """
    global _loaded_lib
    if _loaded_lib is not None:
        return _loaded_lib

    explicit = os.environ.get("LLAMA_CPP_LIB")
    binary_dir = runtime_binary_dir()
    if explicit and not binary_dir:
        # An explicit path that points straight at the file but whose parent
        # lacks the usual layout - still honour it.
        binary_dir = Path(explicit).parent
    if binary_dir is None or not (binary_dir / "llama.dll").exists():
        raise RuntimeError(
            "Cannot find llama.dll - the native inference runtime is not "
            "provisioned.\n"
            "Run:  localm setup-llama        (downloads a prebuilt into this venv)\n"
            "  or: localm setup-llama --from <your llama.cpp build dir>\n"
            "  or set LLAMA_CPP_LIB=/path/to/llama.dll for a one-off."
        )
    llama_dll = binary_dir / "llama.dll"

    # Make the binary dir AND the venv's ROCm runtime resolvable by the loader,
    # so a HIP-linked llama.dll finds amdhip64/rocm_kpack/rocblas even when the
    # binary dir bundles only llama.dll + ggml-*.dll.
    _add_to_dll_path(binary_dir)
    for d in rocm_runtime_dirs():
        _add_to_dll_path(d)

    # Pre-load ggml dependencies in dependency order (base < cpu < hip/vulkan <
    # ggml.dll, which sorts correctly because '-' < '.'). add_dll_directory
    # already lets the OS resolve them; this is belt-and-suspenders for builds
    # without a manifest.
    try:
        ggml = sorted(binary_dir.glob("ggml*.dll"))
    except OSError:
        ggml = []
    for path in ggml:
        try:
            ctypes.CDLL(str(path))
        except OSError:
            pass  # already loaded or not needed on this system

    try:
        _loaded_lib = ctypes.CDLL(str(llama_dll))
    except OSError as e:
        raise RuntimeError(f"Failed to load llama.dll from {llama_dll}: {e}") from e

    return _loaded_lib
