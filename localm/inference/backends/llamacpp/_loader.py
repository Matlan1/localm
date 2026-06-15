"""
Native-library bootstrap for the llama.cpp backend.

Resolves the native binary directory from project-local locations only - an
explicit override, the config, or the ``localm-llama-runtime`` wheel installed
in this venv - never a sibling folder elsewhere on disk.

The loaded library is platform-specific: ``llama.dll`` on Windows,
``libllama.so`` on Linux, ``libllama.dylib`` on macOS. On Windows the ROCm/ggml
deps are made resolvable via the DLL search path; on POSIX the ggml deps are
pre-loaded by absolute path with ``RTLD_GLOBAL`` so the main library resolves
them, and the build's own runtime deps (ROCm/CUDA) are expected to be resolvable
via the build's rpath, the system linker (ldconfig), or ``LD_LIBRARY_PATH``.

Provision the binaries with ``localm setup-llama``. ``LLAMA_CPP_LIB`` overrides
the path to the library file for one-off use.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional

_loaded_lib: Optional[ctypes.CDLL] = None


def lib_filename() -> str:
    """The loadable llama library filename for this platform."""
    if sys.platform == "win32":
        return "llama.dll"
    if sys.platform == "darwin":
        return "libllama.dylib"
    return "libllama.so"


def _ggml_glob() -> str:
    """Glob matching the bundled ggml dependency libraries for this platform."""
    if sys.platform == "win32":
        return "ggml*.dll"
    if sys.platform == "darwin":
        return "libggml*.dylib"
    return "libggml*.so*"          # also matches versioned libggml.so.N


def _candidate_dirs() -> List[Path]:
    """Directories that may hold the native library, in priority order."""
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
    """The directory the llama library will be loaded from, or None if unprovisioned."""
    name = lib_filename()
    for d in _candidate_dirs():
        try:
            if d and d.is_dir() and (d / name).exists():
                return d
        except OSError:
            continue
    return None


def rocm_runtime_dirs() -> List[Path]:
    """ROCm runtime library directories: the rocm-sdk wheels in this venv, plus
    ``/opt/rocm`` on Linux. These hold amdhip64/rocblas/hipblas/… - the libraries
    a HIP-linked llama build needs at load time. Globbed (not hardcoded) so any
    gfx target's package is picked up."""
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

    if sys.platform == "win32":
        roots.add(Path(sys.prefix) / "Lib" / "site-packages")
        subdirs = ("_rocm_sdk_*/bin",)
    else:
        roots.add(Path(sys.prefix) / "lib")
        subdirs = ("_rocm_sdk_*/lib", "_rocm_sdk_*/lib64")

    for root in roots:
        for pat in subdirs:
            try:
                for d in root.glob(pat):
                    if d.is_dir():
                        found.append(d)
            except OSError:
                continue

    if sys.platform.startswith("linux"):
        for sys_dir in (Path("/opt/rocm/lib"), Path("/opt/rocm/lib64")):  # hygiene-ok: generic ROCm system path
            try:
                if sys_dir.is_dir():
                    found.append(sys_dir)
            except OSError:
                continue
    return found


def _add_to_search_path(directory: Path) -> None:
    """Make *directory* resolvable by the OS loader for transitive deps."""
    if sys.platform == "win32":
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            try:
                add(str(directory))
            except OSError:
                pass
    else:
        # glibc reads LD_LIBRARY_PATH at startup, so mutating it here is only a
        # best-effort hint; the reliable mechanism is the RTLD_GLOBAL preloads
        # in load_lib() plus the build's own rpath.
        os.environ["LD_LIBRARY_PATH"] = (
            str(directory) + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""))


def _preload(path: Path) -> None:
    """Best-effort dlopen of a dependency. On POSIX use RTLD_GLOBAL so its
    symbols are visible to libraries loaded afterwards (libllama)."""
    try:
        if sys.platform == "win32":
            ctypes.CDLL(str(path))
        else:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # already loaded or not needed on this system


def load_lib() -> ctypes.CDLL:
    """Load (and cache) the native llama shared library.

    Idempotent: subsequent calls return the already-loaded handle.
    Raises RuntimeError if the library cannot be found or loaded.
    """
    global _loaded_lib
    if _loaded_lib is not None:
        return _loaded_lib

    name = lib_filename()
    explicit = os.environ.get("LLAMA_CPP_LIB")
    binary_dir = runtime_binary_dir()
    if explicit and not binary_dir:
        # An explicit path straight at the file whose parent lacks the usual
        # layout - still honour it.
        binary_dir = Path(explicit).parent
    if binary_dir is None or not (binary_dir / name).exists():
        raise RuntimeError(
            f"Cannot find {name} - the native inference runtime is not "
            "provisioned.\n"
            "Run:  localm setup-llama --from <your llama.cpp build dir>\n"
            "  (Windows: 'localm setup-llama' alone fetches a prebuilt)\n"
            f"  or set LLAMA_CPP_LIB=/path/to/{name} for a one-off."
        )
    lib_path = binary_dir / name

    # Make the binary dir AND the venv's ROCm runtime resolvable by the loader.
    _add_to_search_path(binary_dir)
    for d in rocm_runtime_dirs():
        _add_to_search_path(d)

    # Pre-load ggml deps (dependency order: base < cpu < hip/vulkan < ggml,
    # which sorts correctly since '-'/'.' precede the suffix) by absolute path,
    # so the main library resolves them even without a manifest.
    try:
        ggml = sorted(binary_dir.glob(_ggml_glob()))
    except OSError:
        ggml = []
    for path in ggml:
        _preload(path)

    try:
        if sys.platform == "win32":
            _loaded_lib = ctypes.CDLL(str(lib_path))
        else:
            _loaded_lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        raise RuntimeError(f"Failed to load {name} from {lib_path}: {e}") from e

    return _loaded_lib
