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

from localm.debuglog import logger

_loaded_lib: Optional[ctypes.CDLL] = None
# Whether load_lib() saw at least one ggml COMPUTE BACKEND registered. None until
# load_lib() has run. A build can load cleanly yet register zero backends (the
# "no backends are loaded" case), so this is tracked separately from _loaded_lib
# and read by setup-llama's load-test: a runtime that loads but cannot compute is
# a FAILED provision, not a silent success (AGENTS.md rule 5).
_compute_backends_ok: Optional[bool] = None


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


def _ensure_rocblas_tensile_path() -> None:
    """Point rocBLAS at its Tensile library if the caller has not already.

    The bundled rocm-sdk wheel ships ``rocblas/library/*.dat`` (the gfx-specific
    GEMM kernels), but rocBLAS otherwise looks beside its own DLL - often an empty
    location in our layout - and aborts a GEMM with "Cannot read
    TensileLibrary.dat". Text matmuls have a fallback, but the multimodal/clip
    encode (mtmd, GGUF vision) needs Tensile, so set ``ROCBLAS_TENSILE_LIBPATH``
    best-effort. No-op when already set or when no such directory exists (non-ROCm
    builds, or a runtime without the rocm-sdk wheel)."""
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


def _ggml_dev_count(handles: "List[ctypes.CDLL]") -> Optional[int]:
    """Number of ggml backend DEVICES currently registered, queried from
    whichever handle exports ``ggml_backend_dev_count`` (ggml.dll), or None when
    no handle exports it (an older build without the registry query). A count > 0
    means the compute backends are already registered and there is nothing to do."""
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
    """Ensure the ggml compute backends (CPU, GPU, ...) are registered.

    Registration model differs by build, so we DETECT before acting:

    * The bundled AMD ROCm build and any statically-linked build self-register
      their backends the moment ``ggml.dll`` loads - verified: a GPU device is
      already present (``ggml_backend_dev_count() > 0``) with no extra step. For
      these we do NOTHING. Critically, we must NOT then call ``ggml_backend_load``
      on the ``ggml-*`` libraries: that is the generic *dynamic-plugin* loader,
      it looks for a ``ggml_backend_init`` entry point these non-plugin DLLs do
      not export, and every such call prints a scary-looking but meaningless
      ``load_backend: failed to find ggml_backend_init in ...ggml-cpu.dll`` to
      stderr. The backends are already there; the probe is pure noise.

    * Some upstream prebuilts ship the backends as real ``GGML_BACKEND_DL``
      plugins that are NOT auto-registered, so a model load aborts with "no
      backends are loaded". Only THEN do we register them - preferring the
      build's own ``ggml_backend_load_all`` discovery, falling back to loading
      each ``ggml-*`` plugin explicitly by absolute path (those genuine plugins
      DO export ``ggml_backend_init``, so the load succeeds silently).

    Returns True when backends are registered (already, or by us)."""
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


def _force_vulkan_dedicated_vram(binary_dir: Path) -> None:
    """On a Windows Vulkan build, keep model weights in DEDICATED VRAM.

    By default ggml-vulkan allocates the model into HOST-VISIBLE video memory (to
    skip a staging copy) whenever the device exposes a large host-visible +
    device-local heap - which resizable-BAR and UMA systems do. On Windows/WDDM
    that host-visible allocation is then physically backed by SHARED system RAM, so
    the ENTIRE model runs at PCIe speed: measured ~13x slower than dedicated VRAM on
    an RX 6900 XT (a 12B ran 2.5 tok/s in shared RAM vs 32.9 tok/s in dedicated; a
    7B, 4.7 vs 62.7). It is also the fragile path behind the large-model first-decode
    crash on Vulkan. Setting GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM makes ggml-vulkan
    allocate DEVICE_LOCAL (dedicated VRAM) + a staging buffer instead, so the model
    stays in VRAM and any compute-buffer overflow falls to host memory gracefully.

    Scope: only a Vulkan build (the var is ggml-vulkan-specific), only on Windows
    (the WDDM shared-memory backing is Windows-specific; on Linux/amdgpu host-visible
    VRAM is real device memory and disabling it would only add a needless staging
    copy), and only when the user has not set the var themselves (an explicit choice
    always wins - e.g. a true UMA/integrated GPU may legitimately prefer host-visible
    memory). setdefault respects an existing value, including an explicit "0"."""
    if sys.platform != "win32":
        return
    try:
        is_vulkan = ((binary_dir / "ggml-vulkan.dll").exists()
                     or any(binary_dir.glob("*ggml-vulkan*")))
    except OSError:
        return
    if is_vulkan:
        os.environ.setdefault("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1")


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

    # Keep the Vulkan backend's model weights in DEDICATED VRAM (must be set before
    # ggml-vulkan initialises, i.e. before the preload below).
    _force_vulkan_dedicated_vram(binary_dir)

    # Make the binary dir AND the venv's ROCm runtime resolvable by the loader.
    _add_to_search_path(binary_dir)
    for d in rocm_runtime_dirs():
        _add_to_search_path(d)
    _ensure_rocblas_tensile_path()

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
    """True when load_lib() registered at least one ggml compute backend.

    Loads the library first (idempotent). A False result means model inference
    will fail with "no backends are loaded", so setup-llama's load-test uses this
    to reject a build that loads but cannot compute - otherwise a provision that
    is actually broken would be reported as a success the user only discovers at
    the first model load, with the real cause (no registered backend) already
    lost. This is the AGENTS.md rule-5 gate for runtime provisioning."""
    load_lib()
    return bool(_compute_backends_ok)


# ggml_backend_dev_type values (ggml-backend.h): the class of device the runtime
# will run ops on. A GPU build registers a GPU (or ACCEL) device ALONGSIDE the
# CPU one; a cpu build registers only CPU. So "any device whose type is not CPU"
# is the honest test for whether localm actually runs inference on the GPU -
# independent of nvidia-smi/rocm-smi/torch, which never see the Vulkan, Metal or
# bundled-ROCm paths (the doctor "CPU mode only" false-negative, audit doctor-1).
GGML_DEV_TYPE_CPU = 0
GGML_DEV_TYPE_GPU = 1
GGML_DEV_TYPE_ACCEL = 2


def _ggml_dev_handles() -> "List[ctypes.CDLL]":
    """Every loaded handle that MIGHT export the ggml_backend_dev_* registry
    symbols: the main library on a monolithic build, ggml.dll / ggml-base.dll on a
    split one (the symbols are split across them). Same candidate set as
    _register_ggml_backends."""
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


def _ggml_sym(handles: "List[ctypes.CDLL]", name: str):
    for h in handles:
        fn = getattr(h, name, None)
        if fn is not None:
            return fn
    return None


def compute_devices() -> "List[tuple]":
    """The ggml compute DEVICES the provisioned runtime registers, as a list of
    ``(name, type)`` where *type* follows ggml_backend_dev_type (0=CPU, 1=GPU,
    2=ACCEL). Loads the library first (idempotent, which also registers the
    backends), so this reports the devices a real model load would actually use.

    Returns ``[]`` when the ggml device-registry symbols are unavailable (an
    older build without ``ggml_backend_dev_*``); the caller then has no
    device-level signal and must fall back to the provisioned-backend name."""
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


# Cache of (gpu_device_handle, bound ggml_backend_dev_memory) once resolved, or
# False when unavailable (no GPU / no symbol / multi-GPU). The native lib is loaded
# once for the process lifetime, so the device handle is stable.
_gpu_mem_cache = None


def _resolve_gpu_memory():
    """Resolve (single GPU device handle, bound ggml_backend_dev_memory fn), or
    None when there is not exactly one GPU device or the symbol is missing."""
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
    """(free, total) VRAM bytes of the GPU compute device as the ACTIVE ggml
    backend itself sees it (ggml_backend_dev_memory), or None when unavailable.

    DO NOT CALL THIS DIRECTLY from a process that must not crash - use
    gpu_memory_isolated() below instead, which runs this exact query in a
    throwaway subprocess. This function is kept as the ISOLATED PROBE'S
    implementation (see _vram_probe.py, which calls it precisely because a crash
    there only kills that disposable subprocess) and for direct use only when the
    caller has independently decided the crash risk is acceptable (none do, as
    of this writing - grep confirms the only caller left is _vram_probe.py).

    Historically this was GgufBackend._free_vram_bytes's preferred, IN-PROCESS
    signal (it matches how the model's OWN backend budgets its allocations - the
    Vulkan / CUDA / Metal / bundled-ROCm runtime actually running the model - and
    needs no torch, which the Vulkan/Metal builds ship without at all). Both
    properties are real and valuable, which is why gpu_memory_isolated() restores
    them safely rather than dropping this capability - see RISK below for why the
    IN-PROCESS form was retired from every automatic caller.

    RISK (RULE 5, surfaced not hidden - the reason this function itself must
    never be wired into an automatic, in-process caller again): the mem_fn()
    ctypes call below (ggml_backend_dev_memory) is NOT exception-safe against a
    transient driver failure. llama.cpp's own CUDA/HIP error macro treats ANY
    non-success return from the underlying hipMemGetInfo/cudaMemGetInfo as
    unrecoverable and calls abort() in C - a hard, whole-PROCESS crash ("Fatal
    Python error: Aborted") that no Python try/except, including this function's
    own, can catch. Confirmed live three times on this exact call in this
    environment; also a documented, recurring class of issue across other
    llama.cpp-based projects on ROCm/HIP (e.g. ollama/ollama#3840, and
    torch.cuda.mem_get_info hitting the identical "HIP error: invalid argument"
    under concurrent/containerized GPU access in ROCm/ROCm#5378) - i.e. this is
    an upstream driver/runtime behavior, not something specific to how this
    codebase resolves the device handle.

    Ollama - the most comparable llama.cpp-wrapping project - answers this exact
    class of risk (a native crash inside GPU-touching code must never take down
    the parent) by running ALL model-runner GPU work in a separate subprocess.
    gpu_memory_isolated() applies the identical principle narrowly, to just this
    one low-frequency call - the model LOAD itself (llama_load_model_from_file,
    and the same call class on every later context grow) hits the identical
    abort risk and is ALSO isolated this way, in a longer-lived worker that
    serves the model's whole lifecycle, not just one query - see gguf.py and
    llamacpp/_runner.py / _worker.py.

    Returns None when the native lib is not loaded yet (we do NOT force-load it
    just to measure), the build lacks the symbol, or there is not exactly one GPU
    device (see _resolve_gpu_memory). NOTE this still reflects the driver's
    committed budget, not necessarily physical dedicated VRAM: on Windows/WDDM a
    backend allocation that the driver pages to shared system memory still counts
    against this number - which is correct, because that same budget is what
    governs whether the backend's NEXT allocation fits or spills."""
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
    """Start the long-lived VRAM-probe daemon. Caller holds _PROBE_LOCK.
    Line-buffered stdin/stdout text pipes (bufsize=1) so a single print()/
    readline() on either side is exactly one protocol message - see
    _vram_probe.py for the request/response contract. stderr is discarded
    (the daemon logs nothing useful to a caller that only wants two numbers;
    a crash there is diagnosed via the direct gpu_memory() docstring instead)."""
    import subprocess
    import sys
    return subprocess.Popen(
        [sys.executable, "-u", "-m", "localm.inference.backends.llamacpp._vram_probe"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1)


def _readline_with_timeout(stream, timeout: float) -> "Optional[str]":
    """stream.readline() with a timeout, so a daemon that hangs (rather than
    crashes outright - a different failure mode than the abort this whole
    module exists to guard, but one a query timeout must still not block on
    forever) cannot stall the caller. Python's blocking file-object readline()
    has no native timeout, so the read runs in a daemon THREAD (the standard,
    portable pattern for this - a WINDOWS PIPE handle offers no select()-style
    wait) and this function waits on it with a timeout via a queue; on timeout
    the reader thread is abandoned (it will exit on its own once the read
    unblocks or the pipe closes) rather than joined, so this function itself
    never blocks past `timeout`. Returns None on timeout, EOF (empty read - the
    daemon exited/crashed), or any read error."""
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
    """Like gpu_memory(), but SAFE to call automatically: queries the long-lived
    VRAM-probe daemon (spawning or respawning it as needed) so a native abort
    there (see gpu_memory()'s RISK section) only kills that disposable
    subprocess, never this one. This is the function GgufBackend._free_vram_bytes
    actually calls when torch cannot answer.

    Returns None on ANY failure - spawn failure, a query timeout, the daemon
    replying "ERR" (its own load_lib()/query failed), or the daemon having
    crashed/exited (detected via poll() or a read returning EOF) - exactly
    gpu_memory()'s own "unmeasurable" contract, just crash-safe to invoke
    unconditionally. Never raises. A crashed or unresponsive daemon is killed
    and cleared so the NEXT call spawns a fresh one rather than repeatedly
    querying a dead process.

    Cost: near-instant after the first call in this process (a plain pipe
    round-trip - MEASURED live at ~0ms across repeated warm calls); the first
    call (or the first after a crash) pays the daemon's cold-start cost -
    MEASURED live at 1.9-7.9s across several runs (varies with OS/driver
    warm-up state) for the underlying import+load_lib()+query chain the daemon
    itself performs once. Not cached beyond that: each successful query
    reflects current, live VRAM state."""
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
            except Exception:
                _PROBE_PROC = None
                return None
        proc = _PROBE_PROC
        try:
            proc.stdin.write("q\n")
            proc.stdin.flush()
        except Exception:
            _kill_and_clear_probe()
            return None
        timeout = _ISOLATED_PROBE_SPAWN_TIMEOUT if first_spawn else _ISOLATED_PROBE_TIMEOUT
        line = _readline_with_timeout(proc.stdout, timeout)
        if line is None:
            _kill_and_clear_probe()
            return None
        line = line.strip()
        if line == "ERR" or not line:
            return None                  # daemon is alive and answered - genuinely unmeasurable
        try:
            free_s, total_s = line.split()
            return int(free_s), int(total_s)
        except ValueError:
            _kill_and_clear_probe()      # protocol desync - do not trust this daemon again
            return None


def _kill_and_clear_probe() -> None:
    """Caller holds _PROBE_LOCK. Best-effort kill of a dead/hung/desynced daemon
    so the NEXT gpu_memory_isolated() call spawns a fresh one."""
    global _PROBE_PROC
    if _PROBE_PROC is not None:
        try:
            _PROBE_PROC.kill()
        except Exception:
            pass
    _PROBE_PROC = None
