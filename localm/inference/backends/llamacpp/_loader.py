# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native-library bootstrap for the llama.cpp backend."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import List, Optional

from localm.debuglog import logger

_loaded_lib: Optional[ctypes.CDLL] = None
# Whether load_lib() saw at least one ggml COMPUTE BACKEND registered. None until
# load_lib() has run. A build can load cleanly yet register zero backends (the
# "no backends are loaded" case), so this is tracked separately from _loaded_lib
# and read by setup-llama's load-test: a runtime that loads but cannot compute is
# a FAILED provision, not a silent success (AGENTS.md rule 5).
_compute_backends_ok: Optional[bool] = None


def native_lib_loaded() -> bool:
    """True once load_lib() has successfully loaded llama.cpp's native library IN THIS PROCESS."""
    return _loaded_lib is not None


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
    except ImportError:
        pass   # the wheel is not installed yet - normal before `localm setup-llama`
    except Exception as e:
        # Anything other than "not installed" (e.g. an AttributeError from a
        # broken/incomplete install that resolves as an empty namespace
        # package with no lib_dir()) must not look identical to "no runtime
        # provisioned" - that would hide a real environment bug behind a
        # misleading "Cannot find llama.dll" message later on
        # (AGENTS.md rule 5, do-not-hide-problems).
        logger.warning(
            "localm_llama_runtime is installed but broken (%r); skipping it "
            "as a runtime candidate. Try reinstalling it: "
            "uv pip install -e ./runtime", e)

    return dirs


_warned_explicit_lib = False


def runtime_binary_dir() -> Optional[Path]:
    """The directory the llama library will be loaded from, or None if unprovisioned."""
    name = lib_filename()
    result: Optional[Path] = None
    for d in _candidate_dirs():
        try:
            if d and d.is_dir() and (d / name).exists():
                result = d
                break
        except OSError:
            continue
    # An explicit LLAMA_CPP_LIB override that does NOT actually yield the library
    # must not be silently ignored (do-not-hide-problems): the user pointed us at
    # a custom build and needs to know it was skipped and why. Warn once per
    # process, naming the bad path, before the fallback is used (REC-LLAMALIB-SILENT).
    global _warned_explicit_lib
    explicit = os.environ.get("LLAMA_CPP_LIB")
    if explicit and not _warned_explicit_lib:
        p = Path(explicit)
        explicit_dir = p.parent if p.suffix else p
        if result != explicit_dir:
            _warned_explicit_lib = True
            logger.warning(
                "LLAMA_CPP_LIB=%s does not contain %s; ignoring it and falling back to %s.",
                explicit, name,
                f"the provisioned runtime ({result})" if result
                else "no runtime (none found - run 'localm setup-llama')")
    return result


def rocm_runtime_dirs() -> List[Path]:
    """ROCm runtime library directories: the rocm-sdk wheels in this venv, plus ``/opt/rocm`` on Linux."""
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


def _ensure_rocblas_tensile_path() -> None:
    """Point rocBLAS at its Tensile library if the caller has not already."""
    if os.environ.get("ROCBLAS_TENSILE_LIBPATH"):
        return
    for d in rocm_runtime_dirs():
        lib = d / "rocblas" / "library"
        try:
            if lib.is_dir():
                os.environ["ROCBLAS_TENSILE_LIBPATH"] = str(lib)
                return
        except OSError:
            continue


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
    """Best-effort dlopen of a dependency."""
    try:
        if sys.platform == "win32":
            ctypes.CDLL(str(path))
        else:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass  # already loaded or not needed on this system


# ggml shared libraries that are NOT compute backends (do not register them).
_GGML_NON_BACKENDS = {"ggml-base", "ggml"}


def _ggml_dev_count(handles: "List[ctypes.CDLL]") -> Optional[int]:
    """Number of ggml backend DEVICES currently registered, queried from whichever handle exports ``ggml_backend_dev_count`` (ggml.dll), or None when no handle exports it (an older build without the registry query)."""
    for h in handles:
        fn = getattr(h, "ggml_backend_dev_count", None)
        if fn is not None:
            fn.restype = ctypes.c_size_t
            try:
                return int(fn())
            except Exception:
                return None
    return None


def _register_ggml_backends(binary_dir: Path, lib: ctypes.CDLL) -> bool:
    """Ensure the ggml compute backends (CPU, GPU, ...) are registered."""
    # Handles that may export the registry-query / loader symbols (they live in
    # ggml.dll on a split build, possibly the main lib on a monolithic one).
    candidates: List[ctypes.CDLL] = [lib]
    try:
        for p in sorted(binary_dir.glob(_ggml_glob())):
            try:
                candidates.append(ctypes.CDLL(str(p)))
            except OSError:
                pass
    except OSError:
        pass

    # Already registered (the bundled AMD build and any static build)? Done -
    # no probing, no spurious "failed to find ggml_backend_init" noise.
    dev_count = _ggml_dev_count(candidates)
    if dev_count is not None and dev_count > 0:
        return True

    # Nothing registered. Prefer the build's own backend discovery.
    for h in candidates:
        load_all = getattr(h, "ggml_backend_load_all", None)
        if load_all is not None:
            load_all.restype = None
            try:
                load_all()
            except Exception:
                pass
            after = _ggml_dev_count(candidates)
            if after is not None and after > 0:
                return True
            break

    # Last resort: explicit per-plugin load by absolute path (genuine DL-plugin
    # builds). On those the plugins export ggml_backend_init so this is silent;
    # if it still fails the model load will surface "no backends are loaded",
    # which is the correct, honest error for a build with no usable backend.
    load_fn = None
    for h in candidates:
        fn = getattr(h, "ggml_backend_load", None)
        if fn is not None:
            fn.restype = ctypes.c_void_p          # ggml_backend_reg_t (or NULL)
            fn.argtypes = [ctypes.c_char_p]
            load_fn = fn
            break
    if load_fn is None:
        return False   # no loader symbol: nothing more we can do

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
    # Trust the device REGISTRY over the raw load signal: ggml_backend_load
    # returning a non-null reg handle does not prove a usable compute device
    # actually registered. When the count symbol is queryable it is authoritative
    # (so False here honestly means "no backend"); only an older build without the
    # symbol falls back to "did any plugin load".
    final = _ggml_dev_count(candidates)
    if final is not None:
        return final > 0
    return loaded_any


_HOST_VIS_VAR = "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM"

# Values that mean "leave host-visible video memory alone". Same vocabulary as
# debuglog._HANG_OFF and settings_schema._FALSE, plus the empty string: `set VAR=`
# does NOT unset a variable, so "" is a user trying to clear it - and to ggml an
# empty string is still PRESENT, i.e. still disabled.
_HOST_VIS_OPT_OUT = frozenset({"0", "false", "off", "no", ""})


def _force_vulkan_dedicated_vram(binary_dir: Path) -> None:
    """On a Windows Vulkan build, keep model weights in DEDICATED VRAM."""
    if sys.platform != "win32":
        return
    try:
        is_vulkan = ((binary_dir / "ggml-vulkan.dll").exists()
                     or any(binary_dir.glob("*ggml-vulkan*")))
    except OSError:
        return
    if not is_vulkan:
        return
    raw = os.environ.get(_HOST_VIS_VAR)
    if raw is None:
        os.environ[_HOST_VIS_VAR] = "1"
    elif raw.strip().lower() in _HOST_VIS_OPT_OUT:
        # The user asked to KEEP host-visible memory. Unsetting is the only way to
        # say that to ggml; leaving "0" in place would silently mean the opposite.
        os.environ.pop(_HOST_VIS_VAR, None)
        logger.debug("%s=%r reads as an opt-out, and ggml switches on PRESENCE, "
                     "so the variable is unset to actually keep host-visible "
                     "video memory enabled", _HOST_VIS_VAR, raw)


def _warn_if_not_bundled(binary_dir: Path) -> None:
    """Say WHERE the native runtime is being loaded from when it is not the bundled ``localm-llama-runtime`` wheel directory."""
    global _warned_foreign_binary_dir
    if _warned_foreign_binary_dir:
        return
    try:
        import localm_llama_runtime
        bundled = localm_llama_runtime.lib_dir()
        # Compared WITHOUT Path.resolve(). resolve() touches the filesystem (on
        # Windows it calls _getfinalpathname), and this function only decides
        # whether to emit a log line - it has no business making a syscall on a
        # config-supplied directory, least of all in a unit whose whole point is
        # that a path check must not hand a caller-named path to the OS. normcase
        # + abspath is pure string work.
        # The cost is that a symlinked-but-equivalent directory now compares
        # UNEQUAL and warns spuriously. That is the harmless direction: an extra
        # line saying where the runtime came from, versus a missing warning about
        # an override nobody intended.
        if bundled and (os.path.normcase(os.path.abspath(bundled))
                        == os.path.normcase(os.path.abspath(binary_dir))):
            return                       # the normal, self-contained case
        why = ("the bundled runtime is overridden" if bundled
               else "the wheel is installed but not provisioned")
    except ImportError:
        why = "no localm-llama-runtime wheel is installed"
    except Exception:
        return    # _candidate_dirs already warned about a broken wheel
    _warned_foreign_binary_dir = True
    source = ("the LLAMA_CPP_LIB environment variable"
              if os.environ.get("LLAMA_CPP_LIB")
              else "the binary_dir setting (owner-only)")
    logger.warning(
        "loading the native llama runtime from %s, NOT the bundled wheel "
        "directory (%s; source: %s). Everything in that directory is loaded as "
        "native code into this process. If you did not choose it, clear "
        "binary_dir / LLAMA_CPP_LIB and run 'localm setup-llama'.",
        binary_dir, why, source)


_warned_foreign_binary_dir = False


def load_lib() -> ctypes.CDLL:
    """Load (and cache) the native llama shared library."""
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
    _warn_if_not_bundled(binary_dir)

    # Keep the Vulkan backend's model weights in DEDICATED VRAM (must be set before
    # ggml-vulkan initialises, i.e. before the preload below).
    _force_vulkan_dedicated_vram(binary_dir)

    # Make the binary dir AND the venv's ROCm runtime resolvable by the loader.
    _add_to_search_path(binary_dir)
    for d in rocm_runtime_dirs():
        _add_to_search_path(d)
    _ensure_rocblas_tensile_path()

    # Before anything below dlopens by directory glob: on POSIX, the runtime
    # ships one libggml-cpu-<tier>.so per x86 microarchitecture, and EVERY
    # tier's .so exports identically-named global C symbols (llamafile_sgemm,
    # ggml_backend_cpu_init, ...) - so if more than one is ever simultaneously
    # dlopen'd with RTLD_GLOBAL (which the preload below does unconditionally),
    # a call meant for the compatible tier can resolve into an incompatible
    # tier's copy of the same function and execute an illegal instruction.
    # Verified live: a real embed() call crashed with SIGILL inside a matmul
    # kernel gdb attributed to a tier ggml's own compatibility check had
    # already rejected. cpu_backend_select prunes every tier but the one
    # actually safe for this machine BEFORE the glob below can see them, so
    # there is nothing left to collide with. Windows has no equivalent hazard
    # (PE/DLL symbol resolution is per-import-table, not a flat global table -
    # see cpu_backend_select's own module docstring), so this is POSIX-only.
    if sys.platform != "win32":
        from localm.cpu_backend_select import ensure_cpu_tier_selected
        try:
            ensure_cpu_tier_selected(binary_dir)
        except Exception as e:
            # Best-effort: a failure here must not block a load that might
            # otherwise succeed (e.g. a single-tier install with nothing to
            # prune) - fall through and let the ordinary load path surface
            # whatever the real problem is.
            logger.warning("CPU backend tier selection failed in %s: %s",
                           binary_dir, e)

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

    # Ensure the ggml compute backends are registered before any model loads.
    # Builds that already self-register (the bundled AMD build, and any
    # statically-linked build) need nothing; only the upstream prebuilts that
    # ship separate backend plugins must be loaded ("no backends are loaded"
    # otherwise). _register_ggml_backends checks first and only acts when needed.
    # Capture the result: False here means even the explicit-load fallback
    # registered no compute backends, so record the root cause now (both as a
    # warning AND in _compute_backends_ok that setup's load-test reads) instead
    # of leaving only the opaque deferred native "no backends are loaded" error.
    global _compute_backends_ok
    try:
        _compute_backends_ok = bool(_register_ggml_backends(binary_dir, _loaded_lib))
    except Exception:
        # Registration itself faulted: a model load would fail, so record no
        # backends (fail honest) rather than leaving the flag unknown.
        _compute_backends_ok = False
    if not _compute_backends_ok:
        logger.warning(
            'no ggml compute backends registered; model load will fail '
            'with "no backends are loaded"'
        )

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


def compute_backends_available() -> bool:
    """True when load_lib() registered at least one ggml compute backend."""
    load_lib()
    return bool(_compute_backends_ok)


# ggml_backend_dev_type values (ggml-backend.h): the class of device the runtime
# will run ops on. A GPU build registers a GPU (or an accelerator) device
# ALONGSIDE the CPU one; a cpu build registers only CPU. So "any device whose
# type is not CPU" is the honest test for whether localm actually runs inference
# on the GPU - independent of nvidia-smi/rocm-smi/torch, which never see the
# Vulkan, Metal or bundled-ROCm paths (the doctor "CPU mode only" false-negative,
# audit doctor-1).
#
# ONLY CPU AND GPU ARE DECLARED, AND THAT IS DELIBERATE: THE REST OF THIS ENUM
# HAS MOVED, AND WE DO NOT PIN WHICH llama.cpp A USER HAS.
#
# MEASURED 2026-08-11 by fetching ggml/include/ggml-backend.h at several tags:
#
#     b6000                          CPU 0, GPU 1, ACCEL 2
#     b8100 .. b9870 .. master       CPU 0, GPU 1, IGPU 2, ACCEL 3, META 4
#
# IGPU was inserted AHEAD of ACCEL, so ACCEL is 2 on an older runtime and 3 on a
# current one. There is no single correct value to declare for it, and pinning
# the install did not change that: setup_llama.py installs setup_llama._PINNED_TAG
# by default, but `--tag latest` and `--tag <any release>` are both supported, and
# a box may still be running a runtime provisioned long ago under whatever rule
# applied then. This module previously
# declared GGML_DEV_TYPE_ACCEL = 2, which on any runtime since roughly b8100
# silently means INTEGRATED GPU - a name asserting something the value does not
# say. It was inert (every use compares against CPU), and it is removed rather
# than "corrected" because correcting the number just recreates the same trap
# pointing the other way, for older runtimes.
#
# CPU 0 and GPU 1 have held at every tag sampled, which is why those two are safe
# to declare and why discover.implicit_split_capacity ALLOWLISTS GPU rather than
# excluding iGPUs and accelerators by value. Anything needing another member must
# read the header for the runtime actually provisioned, not add a constant here.
GGML_DEV_TYPE_CPU = 0
GGML_DEV_TYPE_GPU = 1


def _ggml_dev_handles() -> "List[ctypes.CDLL]":
    """Every loaded handle that MIGHT export the ggml_backend_dev_* registry symbols: the main library on a monolithic build, ggml.dll / ggml-base.dll on a split one (the symbols are split across them)."""
    handles: List[ctypes.CDLL] = []
    if _loaded_lib is not None:
        handles.append(_loaded_lib)
    binary_dir = runtime_binary_dir()
    if binary_dir is not None:
        try:
            for p in sorted(binary_dir.glob(_ggml_glob())):
                try:
                    handles.append(ctypes.CDLL(str(p)))
                except OSError:
                    pass
        except OSError:
            pass
    return handles


def cpu_buffer_type() -> Optional[int]:
    """The ggml CPU backend's buffer type, as an opaque pointer, or None."""
    load_lib()
    handles = _ggml_dev_handles()
    cnt = _ggml_sym(handles, "ggml_backend_dev_count")
    get = _ggml_sym(handles, "ggml_backend_dev_get")
    name_fn = _ggml_sym(handles, "ggml_backend_dev_name")
    buft_fn = _ggml_sym(handles, "ggml_backend_dev_buffer_type")
    if not (cnt and get and name_fn and buft_fn):
        return None
    cnt.restype = ctypes.c_size_t
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.c_size_t]
    name_fn.restype = ctypes.c_char_p
    name_fn.argtypes = [ctypes.c_void_p]
    buft_fn.restype = ctypes.c_void_p
    buft_fn.argtypes = [ctypes.c_void_p]
    try:
        for i in range(int(cnt())):
            dev = get(i)
            if not dev:
                continue
            if (name_fn(dev) or b"").decode("utf-8", "replace").upper() == "CPU":
                return buft_fn(dev) or None
    except Exception as e:  # noqa: BLE001 - a probe must never break a load
        logger.debug("cpu_buffer_type() query failed (%s); skipping the "
                     "tensor-placement override", type(e).__name__)
    return None


def _ggml_sym(handles: "List[ctypes.CDLL]", name: str):
    for h in handles:
        fn = getattr(h, name, None)
        if fn is not None:
            return fn
    return None


def compute_devices() -> "List[tuple]":
    """The ggml compute DEVICES the provisioned runtime registers, as a list of ``(name, type)`` where *type* is a raw ``ggml_backend_dev_type`` value."""
    load_lib()
    handles = _ggml_dev_handles()

    def _sym(sym_name: str):
        return _ggml_sym(handles, sym_name)

    cnt = _sym("ggml_backend_dev_count")
    get = _sym("ggml_backend_dev_get")
    name_fn = _sym("ggml_backend_dev_name")
    type_fn = _sym("ggml_backend_dev_type")
    if not (cnt and get and name_fn and type_fn):
        return []
    cnt.restype = ctypes.c_size_t
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.c_size_t]
    name_fn.restype = ctypes.c_char_p
    name_fn.argtypes = [ctypes.c_void_p]
    type_fn.restype = ctypes.c_int
    type_fn.argtypes = [ctypes.c_void_p]

    devices: "List[tuple]" = []
    try:
        n = int(cnt())
    except Exception:
        return []
    for i in range(n):
        try:
            dev = get(i)
            raw = name_fn(dev)
            dname = raw.decode("utf-8", "replace") if raw else f"device{i}"
            devices.append((dname, int(type_fn(dev))))
        except Exception:
            # One unreadable device must not lose the others (fail honest per
            # device, not silent for the whole probe).
            continue
    return devices


def native_device_inventory() -> "Optional[list]":
    """``[{'index', 'name', 'description', 'type', 'free', 'total'}, ...]`` for every NON-CPU compute device the ACTIVE ggml runtime registers, or ``None`` when the core registry symbols are unavailable (an older build without ``ggml_backend_dev_*``)."""
    load_lib()
    handles = _ggml_dev_handles()
    cnt = _ggml_sym(handles, "ggml_backend_dev_count")
    get = _ggml_sym(handles, "ggml_backend_dev_get")
    type_fn = _ggml_sym(handles, "ggml_backend_dev_type")
    name_fn = _ggml_sym(handles, "ggml_backend_dev_name")
    if not (cnt and get and type_fn and name_fn):
        return None
    desc_fn = _ggml_sym(handles, "ggml_backend_dev_description")
    mem_fn = _ggml_sym(handles, "ggml_backend_dev_memory")
    cnt.restype = ctypes.c_size_t
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.c_size_t]
    type_fn.restype = ctypes.c_int
    type_fn.argtypes = [ctypes.c_void_p]
    name_fn.restype = ctypes.c_char_p
    name_fn.argtypes = [ctypes.c_void_p]
    if desc_fn is not None:
        desc_fn.restype = ctypes.c_char_p
        desc_fn.argtypes = [ctypes.c_void_p]
    if mem_fn is not None:
        mem_fn.restype = None
        mem_fn.argtypes = [ctypes.c_void_p,
                           ctypes.POINTER(ctypes.c_size_t),
                           ctypes.POINTER(ctypes.c_size_t)]
    out: "list" = []
    try:
        n = int(cnt())
    except Exception:
        return None
    for i in range(n):
        try:
            dev = get(i)
            dev_type = int(type_fn(dev))
            if dev_type == GGML_DEV_TYPE_CPU:
                continue
            raw = name_fn(dev)
            name = raw.decode("utf-8", "replace") if raw else f"device{i}"
            desc = ""
            if desc_fn is not None:
                try:
                    raw_d = desc_fn(dev)
                    desc = raw_d.decode("utf-8", "replace") if raw_d else ""
                except Exception:
                    desc = ""   # optional nicety - the name below still identifies it
            free = total = 0
            if mem_fn is not None:
                try:
                    f = ctypes.c_size_t(0)
                    t = ctypes.c_size_t(0)
                    mem_fn(dev, ctypes.byref(f), ctypes.byref(t))
                    free, total = int(f.value), int(t.value)
                except Exception:
                    free = total = 0   # memory unreadable; the device itself still counts
            out.append({"index": len(out), "name": name, "description": desc,
                        "type": dev_type, "free": free, "total": total})
        except Exception:
            # One unreadable device must not lose the others (fail honest per
            # device, not silent for the whole probe).
            continue
    return out


# Cache of (gpu_device_handle, bound ggml_backend_dev_memory) once resolved, or
# False when unavailable (no GPU / no symbol / multi-GPU). The native lib is loaded
# once for the process lifetime, so the device handle is stable.
_gpu_mem_cache = None


def _resolve_gpu_memory():
    """Resolve (single GPU device handle, bound ggml_backend_dev_memory fn), or None when there is not exactly one GPU device or the symbol is missing."""
    handles = _ggml_dev_handles()
    cnt = _ggml_sym(handles, "ggml_backend_dev_count")
    get = _ggml_sym(handles, "ggml_backend_dev_get")
    type_fn = _ggml_sym(handles, "ggml_backend_dev_type")
    mem_fn = _ggml_sym(handles, "ggml_backend_dev_memory")
    if not (cnt and get and type_fn and mem_fn):
        return None
    cnt.restype = ctypes.c_size_t
    get.restype = ctypes.c_void_p
    get.argtypes = [ctypes.c_size_t]
    type_fn.restype = ctypes.c_int
    type_fn.argtypes = [ctypes.c_void_p]
    mem_fn.restype = None
    mem_fn.argtypes = [ctypes.c_void_p,
                       ctypes.POINTER(ctypes.c_size_t),
                       ctypes.POINTER(ctypes.c_size_t)]
    gpus = []
    try:
        n = int(cnt())
    except Exception:
        return None
    for i in range(n):
        try:
            dev = get(i)
            if int(type_fn(dev)) != GGML_DEV_TYPE_CPU:
                gpus.append(dev)
        except Exception:
            continue
    # Exactly one GPU: an unambiguous device to budget against. Zero GPUs (CPU
    # build) or two+ (a tensor-split spans devices, and picking one would be
    # arbitrary) fall back to the torch path, which honours main_gpu_index.
    if len(gpus) != 1:
        return None
    return (gpus[0], mem_fn)


def gpu_memory() -> "Optional[tuple]":
    """(free, total) VRAM bytes of the GPU compute device as the ACTIVE ggml backend itself sees it (ggml_backend_dev_memory), or None when unavailable."""
    if _loaded_lib is None:
        return None
    global _gpu_mem_cache
    if _gpu_mem_cache is None:
        _gpu_mem_cache = _resolve_gpu_memory() or False
    if not _gpu_mem_cache:
        return None
    dev, mem_fn = _gpu_mem_cache
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    try:
        mem_fn(dev, ctypes.byref(free), ctypes.byref(total))
    except Exception:
        return None
    if total.value <= 0:
        return None
    return (int(free.value), int(total.value))


_ISOLATED_PROBE_TIMEOUT = 5.0     # per-QUERY timeout against the running daemon
# First query after a (re)spawn: cold DLL load. MEASURED live across several
# runs: 1.9s-7.9s (varies with OS file-cache/driver warm-up state - a repeat
# spawn shortly after a prior one was consistently faster than the very first
# spawn in a fresh process). Generous margin above the worst observed value, not
# a guess - do not shrink this without re-measuring on a genuinely cold system.
_ISOLATED_PROBE_SPAWN_TIMEOUT = 20.0

# The long-lived probe daemon (see _vram_probe.py): a subprocess.Popen once
# spawned and reused for the rest of this process's lifetime, or None before the
# first query / after a detected crash. Guarded by _PROBE_LOCK - every query
# (spawn-if-needed, write request, read response) happens under the lock so
# concurrent callers cannot interleave writes/reads on the same pipe. A daemon
# process, not a fresh subprocess per query: measured live, fresh-process-per-
# call cost 1.1-2.0s EVERY time (re-importing localm's llama.cpp binding +
# re-loading the ggml/HIP DLL from scratch), unacceptable for a fallback path
# that is the COMMON case (see gpu_memory()'s docstring) - a daemon pays that
# cost ONCE (measured 1.9-7.9s for the first spawn, varying with OS/driver
# warm-up state) or once per crash-triggered respawn; every later query against
# a live daemon is a near-instant (measured ~0ms) pipe round-trip.
import threading as _threading

_PROBE_PROC = None
_PROBE_LOCK = _threading.Lock()


def _spawn_probe_daemon():
    """Start the long-lived VRAM-probe daemon."""
    import subprocess
    from localm._mp_spawn import interpreter_for_localm_children
    return subprocess.Popen(
        [interpreter_for_localm_children(), "-u", "-m",
         "localm.inference.backends.llamacpp._vram_probe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1)


def _readline_with_timeout(stream, timeout: float) -> "Optional[str]":
    """stream.readline() with a timeout, so a daemon that hangs (rather than crashes outright - a different failure mode than the abort this whole module exists to guard, but one a query timeout must still not block on forever) cannot stall the caller."""
    import queue
    q: "queue.Queue" = queue.Queue(maxsize=1)

    def _reader():
        try:
            q.put(stream.readline())
        except Exception:
            q.put(None)

    t = _threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        line = q.get(timeout=timeout)
    except queue.Empty:
        return None
    if not line:
        return None                      # EOF - the daemon process ended
    return line


def gpu_memory_isolated() -> "Optional[tuple]":
    """Like gpu_memory(), but SAFE to call automatically: queries the long-lived VRAM-probe daemon (spawning or respawning it as needed) so a native abort there (see gpu_memory()'s RISK section) only kills that disposable subprocess, never this one."""
    def _parse(line: str):
        free_s, total_s = line.split()
        return int(free_s), int(total_s)
    return _probe_roundtrip("q\n", _parse)


def gpu_devices_isolated() -> "Optional[list]":
    """``native_device_inventory()``, read crash-isolated via the same probe daemon as ``gpu_memory_isolated()`` (the 'devices' request line - see _vram_probe.py's protocol)."""
    import json

    def _parse(line: str):
        devs = json.loads(line)
        if not isinstance(devs, list) or not all(
                isinstance(d, dict) and isinstance(d.get("index"), int)
                for d in devs):
            raise ValueError("device inventory reply has the wrong shape")
        return devs
    return _probe_roundtrip("devices\n", _parse)


def _probe_roundtrip(request: str, parse) -> "Optional[object]":
    """One request/reply round-trip against the long-lived probe daemon (spawning or respawning it as needed), entirely under ``_PROBE_LOCK`` so concurrent callers cannot interleave writes/reads on the shared pipe."""
    global _PROBE_PROC
    with _PROBE_LOCK:
        first_spawn = False
        if _PROBE_PROC is None or _PROBE_PROC.poll() is not None:
            try:
                if _PROBE_PROC is not None:
                    _PROBE_PROC.kill()
            except Exception:
                pass
            try:
                _PROBE_PROC = _spawn_probe_daemon()
                first_spawn = True
            except Exception as e:
                # Surfaced, not swallowed (rule 5): a silent None here is
                # indistinguishable from "no GPU" to every caller, which is
                # exactly how the worker's dead daemon went unnoticed.
                logger.debug("vram-probe: daemon spawn failed (%s)", e)
                _PROBE_PROC = None
                return None
        proc = _PROBE_PROC
        try:
            proc.stdin.write(request)
            proc.stdin.flush()
        except Exception as e:
            logger.debug("vram-probe: request write failed (%s); a fresh daemon "
                         "spawns on the next query", e)
            _kill_and_clear_probe()
            return None
        timeout = _ISOLATED_PROBE_SPAWN_TIMEOUT if first_spawn else _ISOLATED_PROBE_TIMEOUT
        line = _readline_with_timeout(proc.stdout, timeout)
        if line is None:
            logger.debug("vram-probe: no reply (timeout after %.1fs, or EOF - "
                         "daemon rc=%s, None means still running); killed, a "
                         "fresh daemon spawns on the next query",
                         timeout, proc.poll())
            _kill_and_clear_probe()
            return None
        line = line.strip()
        if line.startswith("ERR") or not line:
            # The daemon is alive and answered "genuinely unmeasurable". A load
            # failure on its side rides along as "ERR <cause>" so the reason
            # lands here rather than dying with its discarded stderr.
            logger.debug("vram-probe: daemon answered unmeasurable%s",
                         f" ({line[4:]})" if len(line) > 4 else "")
            return None
        try:
            return parse(line)
        except Exception as e:
            logger.debug("vram-probe: reply desync (%.80r: %s); daemon killed, "
                         "a fresh one spawns on the next query", line, e)
            _kill_and_clear_probe()      # protocol desync - do not trust this daemon again
            return None


def _kill_and_clear_probe() -> None:
    """Caller holds _PROBE_LOCK."""
    global _PROBE_PROC
    if _PROBE_PROC is not None:
        try:
            _PROBE_PROC.kill()
        except Exception:
            pass
    _PROBE_PROC = None
