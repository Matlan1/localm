"""
DLL bootstrap for the llama.cpp native backend.

Loads ggml dependency DLLs in the correct order, then loads llama.dll from the
prebuilt binary directory (D:/projects/llama-gfx1030-prebuilt by default).

The LLAMA_CPP_LIB env-var can override the path to llama.dll.  The caller
should set LLAMA_CPP_DLL_DIR (or rely on auto-discovery) to locate the ggml
dependencies.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Optional

# Order matters: ggml → ggml-base → ggml-cpu → ggml-hip → llama
_GGML_DEPS = ["ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-hip.dll"]

_loaded_lib: Optional[ctypes.CDLL] = None


def _find_binary_dir() -> Optional[Path]:
    """Return the prebuilt binary directory, or None if not found."""
    candidates = [
        Path(r"D:\projects\llama-gfx1030-prebuilt"),
        Path(r"D:\projects\llama.cpp\build\bin"),
    ]
    for p in candidates:
        if p.is_dir() and (p / "llama.dll").exists():
            return p
    return None


def load_lib() -> ctypes.CDLL:
    """Load (and cache) the llama.dll shared library.

    Idempotent: subsequent calls return the already-loaded handle.
    Raises RuntimeError if the library cannot be found or loaded.
    """
    global _loaded_lib
    if _loaded_lib is not None:
        return _loaded_lib

    # Allow explicit override via environment variable
    explicit = os.environ.get("LLAMA_CPP_LIB")
    if explicit:
        llama_dll = Path(explicit)
        binary_dir = llama_dll.parent
    else:
        binary_dir = _find_binary_dir()
        if binary_dir is None:
            raise RuntimeError(
                "Cannot find llama.dll. "
                "Set LLAMA_CPP_LIB=/path/to/llama.dll or place llama.dll in "
                "D:\\projects\\llama-gfx1030-prebuilt\\."
            )
        llama_dll = binary_dir / "llama.dll"

    # Extend PATH so Windows can find HIP runtime DLLs (amdhip64_7.dll etc.)
    os.environ["PATH"] = str(binary_dir) + os.pathsep + os.environ.get("PATH", "")

    # Load ggml dependencies first so their symbols are resolved when llama.dll loads
    for name in _GGML_DEPS:
        path = binary_dir / name
        if path.exists():
            try:
                ctypes.CDLL(str(path))
            except OSError:
                pass  # already loaded or not required on this system

    # Finally load llama.dll
    try:
        _loaded_lib = ctypes.CDLL(str(llama_dll))
    except OSError as e:
        raise RuntimeError(f"Failed to load llama.dll from {llama_dll}: {e}") from e

    return _loaded_lib
