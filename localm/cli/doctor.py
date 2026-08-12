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
    """Matches pyproject.toml's ``requires-python`` pin exactly (3.12 only, see
    its comment there): 3.10/3.11 cannot even import localm.plugins.loader, and
    the AMD [gpu] wheels are cp312-only, so this check must not report an older
    or newer interpreter as OK."""
    import sys as _sys
    major, minor = _sys.version_info[:2]
    if (major, minor) == (3, 12):
        console.print(f"  {_OK_SYM}  Python {major}.{minor}")
    else:
        console.print(f"  {_FAIL_SYM}  Python {major}.{minor} - 3.12 required")


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
    return _check_blas_kernels(binary_dir)


def _check_blas_kernels(binary_dir) -> bool:
    """Report a ROCm/HIP install whose BLAS kernel data is missing; return health.

    A found-and-loadable llama.dll is NOT the same as a usable install. rocBLAS
    resolves its GPU-arch GEMM kernels at runtime from a data directory next to
    the DLL, so an install can pass every check above with ZERO kernels and then
    hard-crash the native process (uncatchable from Python) the first time a
    workload dispatches through Tensile - the embedder's batch encode.

    That state is reachable two ways, and this catches both: the original defect
    where the provision copied the library and dropped its data, and a provision
    interrupted part-way (a locked file on a machine with the runtime open).

    FAIL rather than WARN, because the failure it predicts is a hard process
    crash, and because the remedy is one command."""
    from localm.setup_llama import blas_kernel_problems
    problems = blas_kernel_problems(binary_dir)
    if not problems:
        return True
    for p in problems:
        console.print(
            f"  {_FAIL_SYM}  {p} - GPU matrix ops will crash the native "
            f"process; re-run 'localm setup-llama --force'"
        )
    return False


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
        "{'status':v.status,'detail':v.detail,'failures':v.failures[:3],"
        "'layout':v.layout,'context_layout':v.context_layout}))"
    )
    # Kept separately from the `or {}` fallback below: None means the PROBE never
    # ran (subprocess timed out, crashed, or printed no matching line - see
    # _run_probe_subprocess), which is a different fact from the probe running
    # and reporting that it could not check. The reason line below has to tell
    # those apart; the rest of this function does not care.
    abi_raw = _run_probe_subprocess(abi_code, "ABI_RESULT:")
    abi = abi_raw or {}
    status = abi.get("status", "unchecked")
    # WHICH of the two llama_model_params layouts was selected is worth showing:
    # upstream reordered that struct in place at an unchanged size, so this is
    # the only externally visible sign of which generation of runtime is
    # installed, and it is the first thing anyone diagnosing a wrong-GPU or
    # unexpected-memory-behaviour report needs. See llamacpp/_structs.py.
    layout = abi.get("layout") or ""
    context_layout = abi.get("context_layout") or ""
    layout_bits = ", ".join(
        s for s in (f"model params {layout}" if layout else "",
                    f"context params {context_layout}" if context_layout else "")
        if s)
    suffix = f" [dim]({layout_bits})[/dim]" if layout_bits else ""
    if status == "ok":
        console.print(f"  {_OK_SYM}  native ABI: struct layout matches this build"
                      + suffix)
    elif status == "mismatch":
        console.print(f"  {_FAIL_SYM}  native ABI MISMATCH - the runtime's struct "
                      "layout differs from this build; loading is refused to avoid "
                      "memory corruption. Run 'localm setup-llama --force'.")
        for f in abi.get("failures", []):
            console.print(f"       [dim]{f}[/dim]")
    elif status == "skipped":
        console.print(f"  {_WARN_SYM}  native ABI check skipped (LOCALM_SKIP_ABI_CHECK set)")
    else:
        # The verdict ("not verified") was always honest; the REASON was not.
        # abi_report() populates detail on every path it can return from
        # ("runtime not loadable: ...", "loader import failed: ..."), so the old
        # hardcoded 'runtime not loadable' default could ONLY ever be reached
        # when the probe did not run at all - precisely the case where that
        # reason is least likely to be true, and it sent the user to
        # 'setup-llama --force' for a subprocess timeout it cannot fix.
        # _check_gpu_verdict already distinguishes its own probe's None (see
        # marker_trustworthy); this line now does too.
        if abi_raw is None:
            detail = ("the ABI probe did not run - it timed out, crashed, or "
                      "printed no result")
        else:
            detail = abi.get("detail") or "no reason reported"
        console.print(f"  {_WARN_SYM}  native ABI not verified [dim]({detail})[/dim]")


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
    green precisely because nothing probed this. This check would have caught it.

    Also probes that pip actually landed inside the new venv (NEW-MANAGED-COMFY-
    VENV-MISSING-PIP): ``-m venv`` can report success - return code 0, the
    interpreter file present - while its own mandatory ensurepip bootstrap
    silently failed (a base Python with ensurepip stripped, or a broken
    install). The managed-ComfyUI installer pip-installs into a venv it just
    created with no ``--without-pip`` fallback, so a pip-less venv here would
    read as doctor-green right up until provisioning fails deep inside with an
    opaque "No module named pip"."""
    import subprocess
    import tempfile
    from pathlib import Path

    from localm._mp_spawn import real_base_python

    venv_python = real_base_python() or sys.executable
    ok = False
    detail = ""
    pip_ok = True
    pip_detail = ""
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
            else:
                pr = subprocess.run(
                    [str(expected), "-m", "pip", "--version"],
                    capture_output=True, text=True, timeout=30)
                pip_ok = pr.returncode == 0
                if not pip_ok:
                    tail = (pr.stderr or pr.stdout or "").strip().splitlines()
                    pip_detail = tail[-1] if tail else ""
    except Exception as e:
        console.print(f"  {_FAIL_SYM}  venv-creation check errored: {e}")
        return

    if ok and pip_ok:
        console.print(f"  {_OK_SYM}  venv creation: OK "
                      "(the managed-ComfyUI installer uses this)")
    elif ok and not pip_ok:
        console.print(
            f"  {_FAIL_SYM}  venv creation FAILED"
            + (f" ({pip_detail})" if pip_detail else "")
            + " - the venv was created but has no working pip; managed ComfyUI "
              "setup ('localm comfy setup') will fail even though the runtime "
              "above checks out")
    else:
        console.print(
            f"  {_FAIL_SYM}  venv creation FAILED"
            + (f" ({detail})" if detail else "")
            + " - managed ComfyUI setup ('localm comfy setup') will fail even "
              "though the runtime above checks out")


def _check_gpu_driver() -> bool:
    """Probe nvidia-smi / rocm-smi; return True if a GPU driver was found.

    ONLY A CLEAN EXIT COUNTS AS A GPU. This used to read ``.stdout`` and never
    look at the return code, so ANY non-empty stdout became both a green tick and
    the device NAME - and these tools report a broken driver by printing an error
    and exiting non-zero. The reported nvidia-smi form of that ("Failed to
    initialize NVML: Driver/library version mismatch", the usual state after a
    driver update with no reboot) would have rendered as:

        [green]OK[/green]  NVIDIA GPU: Failed to initialize NVML: ...

    a tick on an error string. That is the check answering "did the tool print
    something" while the reader hears "you have a working GPU driver", which is
    the one case worth reporting.

    It got worse downstream: this return value feeds ``smi_or_torch_gpu``, and
    ``_check_gpu_verdict``'s step (3) RETURNS EARLY when it is True. So a broken
    driver also SUPPRESSED the "No GPU detected ... CPU mode only" line - the
    check that would have told the user something is wrong.

    A tool that is absent stays silent (that is the normal case on nearly every
    box and is not a fault). A tool that is PRESENT AND FAILING is surfaced as a
    warning naming what it said, because "not installed" and "installed but
    broken" need opposite responses from the user, and only the second is
    actionable. Neither counts as a GPU: the real verdict comes from
    _check_gpu_verdict's device probe, so a miss here costs a supplementary
    detail line, while a false positive costs the whole warning."""
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
            continue          # not installed: the normal case, never a fault
        except Exception as e:
            # A timeout or an OS-level spawn failure is NOT the same as absent -
            # the tool is there and did not answer. Nothing here should escalate
            # a supplementary line into a doctor failure, but it must not vanish
            # without trace either.
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

    Run the torch GPU probe BEFORE deciding the "CPU mode only" verdict: the smi
    tools can be absent while torch still sees a usable GPU (common on ROCm
    installs without rocm-smi on PATH). Printing both "CPU mode only" and a torch
    GPU/VRAM line in the same run contradicts itself, so let torch's view veto the
    CPU-only warning.

    Skips the torch attempt entirely once ``_loader.native_lib_loaded()`` is
    True - the SAME precondition and SAME blanket guard
    ``VramSizingMixin._vram_levels`` / ``_free_total_vram_bytes``
    (llamacpp/_sizing.py) use for the identical DLL-identity conflict (see
    their docstrings for the root cause): once llama.cpp's own native runtime
    is loaded in this process, a later ``import torch`` on this project's
    Windows + AMD ROCm build reliably hits STATUS_ENTRYPOINT_NOT_FOUND. This is
    doctor's fourth call site for the same conflict (discover._list_gpus_probe,
    gpu_usage.raw_reading_is_process_scoped and _sizing.py's VramSizingMixin
    are the other three, already guarded). Costs nothing to skip here:
    _check_gpu_verdict's real verdict comes from a direct backend probe, this
    torch reading is only ever OR'd in as a last-resort hedge. Not reachable in
    a real standalone `localm doctor` run (the earlier native-code checks in
    doctor() load in a separate subprocess, never in-process here) - only a
    mixed pytest run that loads the native lib in-process before this call
    reaches it.

    The ``except Exception`` below (beyond the plain ``ImportError`` for
    "torch not installed") is deliberate: ``doctor()`` calls this with no
    try/except of its own, so any OTHER torch import failure - such as the
    doomed-combination conflict above reached some other way - must not
    escape and silently truncate every later doctor check."""
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
            # _GPU_PROBE_CLI_DEADLINE is an alias of the (cold-init-tolerant)
            # default; passed explicitly because doctor is exactly where a cold
            # box lands (driver init measured up to ~6.5s) and this call must
            # never regress to a thin deadline if the default ever changes (same
            # reason cli/models.py pins it). Never raises; [] just means no
            # correction is available and the raw readings below stand.
            #
            # return_status=True is load-bearing, not optional: list_gpus()'s
            # bare form (return_status=False) can STILL silently return a
            # served last-known-good list from an earlier probe when THIS call
            # times out / is busy / inconclusive (see its own docstring) - a
            # stale reading with valid free/free_scope fields, indistinguishable
            # from a fresh one without the status. Discarding it here would
            # reopen the exact "trust an untrustworthy free figure" gap this
            # function exists to close (AGENTS.md rule 5).
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
                # Driver-level free/total, CORRECTED for cross-process blindness.
                #
                # torch.cuda.mem_get_info's raw "free" is not the whole board's on
                # every platform: measured on Windows + an AMD ROCm/HIP build, it
                # reports total - THIS process's own allocations and is blind to
                # every other process (0.14 GB reported while 10.53 GB was genuinely
                # in use). doctor is a fresh, short-lived process holding no model,
                # so the raw number would show a nearly-empty card no matter what is
                # actually resident - telling a user with a full GPU that it is free,
                # which is the opposite of a diagnostic's job (AGENTS.md rule 5: never
                # report a wrong number as fact). discover.list_gpus() applies the
                # device-global correction where one exists and tags what it could
                # not correct. See dev-notes/vram-cross-process-blindness.md.
                free_b, total_b = torch.cuda.mem_get_info(i)
                scope = None
                corrected = False
                for g in _gpus_corrected:
                    if g.get("index") == i and g.get("free") is not None:
                        free_b, scope = g["free"], g.get("free_scope")
                        corrected = True
                        break
                # Trustworthy depends on which reading `free_b` actually is:
                #   * corrected (a device-global entry was substituted in) - only
                #     trustworthy if THAT correction was fresh (status ==
                #     GPU_PROBE_OK; a served last-known-good list from an
                #     earlier probe is not a current measurement even though it
                #     carries a valid free_scope) and device-global, not
                #     PROCESS-scoped.
                #   * uncorrected (still the raw, synchronously-fresh
                #     torch.cuda.mem_get_info() value - staleness does not apply
                #     to a call made this instant) - trustworthy unless this
                #     platform is known to be blind to other processes' VRAM
                #     (Windows + AMD ROCm/HIP), in which case the raw reading is
                #     ALWAYS process-scoped by construction.
                from localm import gpu_usage
                untrusted = (
                    (corrected and (
                        _corrected_status != discover.GPU_PROBE_OK
                        or scope != discover.FREE_SCOPE_DEVICE))
                    or (not corrected and gpu_usage.raw_reading_is_process_scoped())
                )
                # AGENTS.md rule 5: a caveat beside a wrong number is not a
                # correction. This used to print the untrusted free figure
                # anyway, with only a dim parenthetical beside it. Show
                # total-only instead, exactly like sysstats._vram() already does
                # for the GUI/MCP surface.
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


def _check_runtime_build(find_binary_dir) -> None:
    """Print WHICH llama.cpp build is provisioned, and whether it is pinned.

    Its own line rather than a suffix on the llama.dll line: the build is the
    first thing anyone needs when an upstream release breaks a machine, and
    triage previously had to infer it from versioned library filenames. Silent
    when nothing is provisioned at all - _check_llama_lib has already said so
    and repeating it adds noise to a report people read under stress.

    An unrecorded build is stated as unrecorded, never guessed: installs that
    predate tag recording exist and will keep existing."""
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
        # The pin and the disk disagree, which is a real state worth naming: a
        # pin was set but never provisioned through (setup-llama not re-run, or
        # it fell back to another backend). Saying only "pinned to X" here would
        # assert something the filesystem contradicts.
        console.print(f"  {_WARN_SYM}  pinned to {pin} but {build} is installed"
                      " - run 'localm setup-llama --force' to apply the pin")
    elif pin:
        console.print(f"     [dim]pinned to {pin}; 'localm setup-llama --tag "
                      "latest' resumes tracking upstream[/dim]")


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

    A probe that RAN AND RAISED is none of those: it is a third outcome, reported
    as its own verdict between (2) and (3) - see the ``probe_error`` branch.

    "CPU mode only" is emitted only for a genuine cpu build, a runtime that loads
    with no GPU device, or a truly indeterminate no-signal case - never from the
    mere absence of the vendor CLIs the default GPU path never needs."""
    backend = _provisioned_backend_name(find_binary_dir)
    probe = _probe_gpu_devices() if lib_healthy else None
    loaded = bool(probe.get("loaded")) if probe else False
    devices = (probe.get("devices") or []) if probe else []
    # _GPU_PROBE_CODE captures the exception that stopped it. Reading it is what
    # separates "the runtime says there is no GPU" from "the runtime could not be
    # asked", which used to be the same verdict here (see the branch below).
    # str() because this crosses a subprocess/JSON boundary and doctor's whole
    # contract is that it cannot crash; splitlines()[0] because a repr of a native
    # loader error can carry a whole traceback, and a wrapped verdict is a hidden
    # one (the _check_gpu_driver idiom).
    probe_error = str((probe.get("error") or "") if probe else "").strip()
    if probe_error:
        # Logged HERE rather than inside the verdict branch below, so a captured
        # reason is never dropped on the paths that return earlier. The live one:
        # compute_backends_available() succeeds and compute_devices() then raises,
        # which leaves loaded True, devices empty and an error set - step (2)
        # correctly reports the backend marker and returns, and before this line
        # that reason went nowhere at all.
        from localm.debuglog import logger as _dbg
        _dbg.debug("doctor: GPU device probe reported an error: %s", probe_error)
    probe_error = probe_error.splitlines()[0] if probe_error else ""

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

    # (2b) The probe RAN AND FAILED. Not "no GPU" - "we could not find out", and
    #      the runtime being unable to load is itself the fault to report. It
    #      reached (3) before this branch existed, so a driver too old for the
    #      provisioned Vulkan build rendered as "No GPU detected ... run 'localm
    #      setup-llama'": hardware the user has, described as absent, plus advice
    #      to provision a backend that IS provisioned, while the captured reason
    #      was discarded. Same split _check_abi makes for its own probe and
    #      _check_gpu_driver makes for absent-vs-present-and-failing.
    #
    #      Deliberately ABOVE the smi_or_torch_gpu early return: a card those
    #      tools CAN see plus a runtime that will not load is the case most worth
    #      naming, and returning early there would swallow it entirely.
    # No user-facing line points at the debug log above: `doctor` takes no --debug
    # flag (checked - it is a bare @main.command()), so promising one would be an
    # unverified claim on the very path that exists to stop making them.
    if probe_error:
        console.print(f"  {_WARN_SYM}  GPU: could not be determined{tag} - the runtime probe failed")
        console.print(f"       [dim]{probe_error}[/dim]")
        # No provisioning advice here on purpose. The runtime IS provisioned, and
        # _check_llama_lib / _check_native_abi have already run and already say
        # what to do about a runtime that will not load - repeating it adds noise
        # to a report people read under stress (the _check_runtime_build idiom).
        console.print("       [dim]the runtime is installed but did not load, so this is not "
                      "a statement about your hardware[/dim]")
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


def _check_packages() -> dict:
    """Print each package's presence/version line and return the imported
    module objects (keyed by import name; None where the import failed), so a
    caller can run a deeper USABILITY check on top of a package this function
    already confirmed is merely importable (see ``_check_hf_backend_usable``)."""
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
        # torch's own fifth call site for the DLL-identity conflict documented
        # at _check_vram_torch (llama.cpp's native runtime already loaded in
        # this process makes a later `import torch` doomed on this project's
        # Windows + AMD ROCm build): proactively skip it here too, before it
        # can even be attempted - found live via this exact
        # importlib.import_module("torch") raising OSError: [WinError 127]
        # when test_doctor_gpu_verdict.py's real compute-device probe loaded
        # the native runtime earlier in the same pytest worker. No other
        # package here shares this specific risk.
        #
        # Narrower than _vram_levels's blanket native_lib_loaded() skip (the
        # same trade-off discover._torch_gpu_probe_known_doomed weighs against
        # _sizing's blanket one): a torch ALREADY resident in sys.modules -
        # imported for real earlier in this same process, or a test double -
        # makes `import torch` here a plain cache hit, never a fresh preload,
        # so the conflict cannot occur and the working module must be kept.
        # Skipping unconditionally would cost real behavior here (unlike
        # _vram_levels's display-only reading): _check_hf_backend_usable below
        # depends on getting the real module handle back.
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
            # 9.1) and accessing it emits a DeprecationWarning, so metadata-first
            # avoids the warning and the eventual blank (AUD-CLICKVER). Fall back
            # to __version__ only for a package whose dist metadata is missing.
            try:
                ver = _ilm.version(_dist_names.get(mod, mod))
            except _ilm.PackageNotFoundError:
                # A lazy module (transformers._LazyModule is the live example)
                # raises ModuleNotFoundError from __getattr__ for ANY name it
                # cannot resolve, __version__ included - and getattr()'s default
                # only suppresses AttributeError. Unguarded, that escapes into
                # the `except ImportError` below (ModuleNotFoundError IS an
                # ImportError) and reports a package that imported perfectly
                # well as "not installed", which then makes
                # _check_hf_backend_usable return early and say NOTHING about
                # the breakage it exists to report. Losing the version STRING is
                # cosmetic; losing the module handle hides a real fault.
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
            # raising something other than ImportError is a genuinely new,
            # un-root-caused failure mode that doctor() must not silently mask.
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
    """Prove the HF (transformers) backend is actually USABLE, not merely
    importable. ``localm/inference/backends/hf.py`` loads models through
    ``transformers.AutoTokenizer`` / ``AutoProcessor`` / ``AutoModelForCausalLM``,
    which transformers resolves through a LAZY module: ``import transformers``
    only sets up that machinery, and a heavy submodule (e.g. distributed/fsdp)
    is imported for real only on the FIRST attribute access that needs it. So
    ``import transformers`` can succeed - and ``_check_packages`` above reports
    a clean version - while every one of those classes is dead.

    This exact gap shipped in 0.1.2: transformers 5.14 hard-imports fsdp on the
    tokenizer path, which needs ``torch._C._distributed_c10d`` - absent from the
    pinned ROCm/Windows torch build - so EVERY HF model load died at "loading
    processor..." while ``localm doctor`` printed both packages OK (found during
    the 0.1.2 release verification; see tests/test_gpu_extra_pins.py for the
    version-pin guard this backs up with a functional one).

    Runs only when both packages actually import; an absent OPTIONAL backend
    (reported above) is not a fault."""
    if torch_mod is None or transformers_mod is None:
        return
    try:
        for name in ("AutoTokenizer", "AutoProcessor", "AutoModelForCausalLM"):
            getattr(transformers_mod, name)
    except Exception as e:
        # transformers' lazy loader re-raises a failed submodule import as a
        # generic ModuleNotFoundError("Could not import module 'X'") chained
        # (`raise ... from e`) onto the real cause - and since resolving one
        # lazy attribute can walk through OTHER lazy submodules, that can
        # repeat several layers deep before reaching the actual error. Printing
        # only the top frame reproduces exactly the unhelpful message that hid
        # this regression; walk the chain to the true root instead.
        root = e
        seen = {id(root)}
        while True:
            nxt = root.__cause__ or root.__context__
            if nxt is None or id(nxt) in seen:
                break
            root = nxt
            seen.add(id(root))
        console.print(
            f"  {_FAIL_SYM}  HF backend (transformers) is installed but UNUSABLE "
            f"- every HF model load will fail: {type(root).__name__}: {root}"
        )
        return
    console.print(
        f"  {_OK_SYM}  HF backend (transformers): AutoTokenizer / AutoProcessor / "
        "AutoModelForCausalLM load OK"
    )


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
      - Python version (3.12 required)
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
    # Immediately after the lib line and BEFORE the ABI check, because "which
    # build" is the first question an ABI mismatch or a native crash raises.
    _check_runtime_build(find_binary_dir)
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
    modules = _check_packages()
    _check_hf_backend_usable(modules.get("torch"), modules.get("transformers"))
    _check_plugin_deps()
    _check_managed_comfy()
