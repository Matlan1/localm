# SPDX-License-Identifier: AGPL-3.0-or-later
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


# ggml shared libraries that are NOT compute backends (do not register them).
_GGML_NON_BACKENDS = {"ggml-base", "ggml"}


def _register_ggml_backends(binary_dir: Path, lib: ctypes.CDLL) -> bool:
    """Register the ggml compute backends so a model can actually load.

    Newer llama.cpp ships each compute backend (CPU, Vulkan, CUDA, HIP, ...) as a
    separate ``ggml-*`` plugin library that must be registered at runtime.
    Without it, ``llama_load_model_from_file`` aborts with "no backends are
    loaded" - which is every current upstream cpu/vulkan/cuda prebuilt (i.e. all
    non-AMD users). Older self-contained builds (e.g. the bundled lemonade gfx103X
    build) statically register their backend and do NOT export the loader symbol,
    so this is best-effort and fully guarded: a build that lacks it is untouched.

    We register each ``ggml-*`` plugin EXPLICITLY by absolute path via
    ``ggml_backend_load`` (deterministic) rather than rely on directory discovery,
    which searches next to the host executable (python) and the binary-dir layout
    confuses it. Returns True if at least one backend registered."""
    # Find ggml_backend_load on whichever handle exports it (ggml-base, else lib).
    candidates: List[ctypes.CDLL] = [lib]
    try:
        for p in sorted(binary_dir.glob(_ggml_glob())):
            try:
                candidates.append(ctypes.CDLL(str(p)))
            except OSError:
                pass
    except OSError:
        pass

    load_fn = None
    for h in candidates:
        fn = getattr(h, "ggml_backend_load", None)
        if fn is not None:
            fn.restype = ctypes.c_void_p          # ggml_backend_reg_t (or NULL)
            fn.argtypes = [ctypes.c_char_p]
            load_fn = fn
            break
    if load_fn is None:
        return False   # old build: backends are statically registered, nothing to do

    loaded_any = False
    try:
        for p in sorted(binary_dir.glob(_ggml_glob())):
            stem = p.stem.lower()
            if stem.startswith("lib"):
                stem = stem[3:]
            if stem in _GGML_NON_BACKENDS:
                continue
            try:
                if load_fn(str(p).encode("utf-8")):
                    loaded_any = True
            except Exception:
                continue
    except OSError:
        pass
    return loaded_any


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
        raise RuntimeError(
            f"Failed to load {name} from {lib_path}: {e}\n"
            "This usually means the provisioned build does not match this "
            "machine - e.g. an AMD ROCm or NVIDIA CUDA build on a box without "
            "that GPU or its runtime. Re-provision a backend that fits:\n"
            "  localm setup-llama --backend vulkan --force   (any GPU, no vendor toolkit)\n"
            "  localm setup-llama --backend cpu --force       (no GPU)\n"
            "  localm setup-llama --backend cuda --force      (NVIDIA)\n"
            "  localm setup-llama --backend amd-rocm --force  (AMD RX 6000)"
        ) from e

    # Register the ggml compute backends. Newer llama.cpp loads them as separate
    # plugin DLLs that must be registered before any model loads ("no backends are
    # loaded" otherwise - the failure mode on every current upstream cpu/vulkan/
    # cuda build). Older self-contained builds auto-register and lack the entry
    # point, so this is best-effort and guarded.
    try:
        _register_ggml_backends(binary_dir, _loaded_lib)
    except Exception:
        pass

    # Validate the runtime's struct layout BEFORE any by-value struct crosses the
    # FFI boundary. A layout that differs from this build's ctypes structs would
    # otherwise corrupt memory silently; verify_abi refuses with a clear,
    # reportable error instead (and fails OPEN if its own probe cannot run).
    # A failed check resets the cache so a re-provision can retry cleanly.
    from ._abi import verify_abi
    try:
        verify_abi(_loaded_lib, str(lib_path))
    except Exception:
        _loaded_lib = None
        raise

    return _loaded_lib
