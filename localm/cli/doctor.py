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
# A neutral info marker for discovery hints that are NOT pass/fail checks (an
# opt-in feature the user simply may not know about). Distinct from _WARN_SYM so
# it never reads as "something is wrong".
_HINT_SYM = "[cyan]i[/cyan]"

_SYM_FOR = {diagnostics.OK: _OK_SYM,
            diagnostics.WARN: _WARN_SYM,
            diagnostics.FAIL: _FAIL_SYM}


def _render(result) -> None:
    """Print one ``diagnostics.CheckResult`` the way doctor has always printed it.

    The five ACTIVE probes live in ``localm/diagnostics.py`` so the GUI can run
    the same code instead of parsing this output (ADR-0001's follow-up, which
    also named the alternative: parsing doctor's console output would be a
    facade). Everything terminal-specific stays here - the symbols, the dim
    parenthetical, the seven-space hint indent.

    A result with NO findings prints nothing, which is not an oversight: an
    absent optional backend has never produced a line, and a report people read
    under stress does not need one saying so."""
    from rich.markup import escape

    for f in result.findings:
        # escape(): diagnostics.py builds text/note/hints from real probe
        # output - a subprocess's stderr/stdout, an exception message, raw
        # filenames read off disk, a native ABI mismatch detail - none of it
        # is restricted to a safe charset, and diagnostics.py itself
        # deliberately never escapes (its own docstring: "nothing here
        # imports click or rich"), so this renderer is the one place that must.
        note = f" [dim]({escape(f.note)})[/dim]" if f.note else ""
        console.print(f"  {_SYM_FOR.get(f.status, _WARN_SYM)}  {escape(f.text)}{note}")
        for hint in f.hints:
            console.print(f"       [dim]{escape(hint)}[/dim]")


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
    """Print the llama.dll/.so health line(s); return True if a healthy lib is
    present. The probe itself (presence, 0-byte/truncated detection, and the
    rocBLAS/hipBLASLt kernel-data check) is diagnostics.check_llama_lib."""
    result = diagnostics.check_llama_lib(find_binary_dir)
    _render(result)
    return result.healthy


def _check_native_abi() -> None:
    """Print the native ABI self-check line. The probe (a subprocess load of the
    provisioned runtime, so a broken DLL can never crash doctor itself) is
    diagnostics.check_native_abi."""
    _render(diagnostics.check_native_abi())


def _check_worker_spawn() -> None:
    """Print the isolated-worker-spawn line. The probe (a real
    ``multiprocessing.get_context("spawn")`` round trip, #617) is
    diagnostics.check_worker_spawn."""
    _render(diagnostics.check_worker_spawn())


def _check_venv_creation() -> None:
    """Print the nested-venv-creation line. The probe (a real ``-m venv`` plus a
    pip-landed check, the mechanism the managed-ComfyUI installer depends on,
    #621) is diagnostics.check_venv_creation."""
    _render(diagnostics.check_venv_creation())


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
    from rich.markup import escape
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
            # escape(): `said` is nvidia-smi/rocm-smi's own stderr/stdout - an
            # external driver tool's text, not a string localm controls the
            # charset of. `cmd[0]` and `label` are the hardcoded literals above.
            console.print(
                f"  {_WARN_SYM}  {cmd[0]} is installed but failed "
                f"(exit {r.returncode}): {escape(said.splitlines()[0])}")
            continue
        if out:
            # escape(): `first_line` is the GPU name/memory nvidia-smi/rocm-smi
            # itself reported - hardware/driver text, not localm's own string.
            first_line = out.splitlines()[0]
            console.print(f"  {_OK_SYM}  {label} GPU: {escape(first_line)}")
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
    from rich.markup import escape

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
                # escape(): `props.name` is the CUDA/ROCm driver's own device
                # name string - hardware/driver-reported text, same risk class
                # as the nvidia-smi/rocm-smi output above.
                console.print(
                    f"  {_OK_SYM}  GPU {i}: {escape(props.name)}  "
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
        # escape(): a dynamically-constructed exception class can carry an
        # arbitrary __name__ - escaped for the same reason the collection
        # names in #1463 were, despite being normally identifier-shaped: a
        # no-op on the common case, real protection if that ever stops holding.
        console.print(f"  {_WARN_SYM}  torch GPU/VRAM probe failed "
                      f"({escape(type(e).__name__)}) - skipped")
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
    from rich.markup import escape

    try:
        from localm.setup_llama import installed_build, pinned_tag
        build = installed_build()
        pin = pinned_tag()
    except Exception as e:
        console.print(f"  {_WARN_SYM}  could not read the llama.cpp build "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    # escape(): `build` is the second token of a plain marker file localm's own
    # provisioning code writes but never re-validates on read (setup_llama.py's
    # _read_marker() just splits on whitespace). `pin` IS already restricted to
    # a safe tag charset by pinned_tag()'s own is_safe_tag() check, but is
    # escaped anyway rather than relying on that chain holding - same reasoning
    # #1463 used for rag.py's already-validated collection names.
    if build:
        console.print(f"  {_OK_SYM}  llama.cpp build: {escape(build)}")
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
        console.print(f"  {_WARN_SYM}  pinned to {escape(pin)} but {escape(build)} "
                      "is installed - run 'localm setup-llama --force' to apply "
                      "the pin")
    elif pin:
        console.print(f"     [dim]pinned to {escape(pin)}; 'localm setup-llama "
                      "--tag latest' resumes tracking upstream[/dim]")


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
    from rich.markup import escape

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
    # escape(): `backend` is read from the same unvalidated marker file as
    # `build` above (setup_llama.py's _read_marker(), no charset check on
    # read) - escaped once here so every use of `tag` below is already safe.
    tag = f" ({escape(backend)})" if backend and backend != "custom" else ""

    # (1) Ground truth: the runtime loaded and reported its real compute devices.
    if loaded and devices:
        # Count only true GPU devices (ggml type GPU). ACCEL devices (BLAS / RPC)
        # are CPU-side or remote accelerators, NOT a GPU, so they must not be
        # labelled "GPU"; all of localm's GPU backends (Vulkan/Metal/CUDA/HIP/
        # SYCL/bundled-ROCm) register as GPU type, so this loses no real GPU.
        gpu = [name for (name, dtype) in devices
               if int(dtype) == _loader_gpu_type()]
        if gpu:
            # escape(): device names come from the native ggml compute-device
            # probe - hardware/driver-reported text (Vulkan/ROCm/CUDA device
            # names), same risk class as nvidia-smi/rocm-smi output.
            console.print(f"  {_OK_SYM}  GPU: {', '.join(escape(g) for g in gpu)}"
                          f"{tag} - used for inference")
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
        console.print(f"  {_OK_SYM}  GPU: '{escape(backend)}' backend provisioned "
                      "- used for inference")
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
        # escape(): `probe_error` is repr(e) of whatever exception the native
        # loader raised, JSON round-tripped out of a subprocess - exactly the
        # "probe error text" class this sweep exists to cover.
        console.print(f"       [dim]{escape(probe_error)}[/dim]")
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
    from rich.markup import escape
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
            # escape(): `ver` is read from an installed package's own
            # distribution metadata (or its __version__ attribute) - a
            # third-party package's string, not localm's to control.
            ver_str = f" {escape(ver)}" if ver else ""
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
            # escape(): defense-in-depth, same reasoning as _check_vram_torch's
            # type(e).__name__ site above - a no-op for a normal class name.
            ver_str = f" - import failed ({escape(type(e).__name__)})"
        console.print(f"  {sym}  {label}{ver_str}")
    return modules


def _check_hf_backend_usable(torch_mod, transformers_mod) -> None:
    """Print the HF-backend-usable line. The probe (resolving transformers' LAZY
    Auto* classes for real, which is what separates "installed" from "usable" -
    the 0.1.2 regression) is diagnostics.check_hf_backend.

    The two module handles come from ``_check_packages`` above, which has already
    decided whether each is importable HERE - so ``resolved=True``: a None means
    absent, and the core must not go re-importing torch in a process that may
    have just been told not to."""
    _render(diagnostics.check_hf_backend(torch_mod, transformers_mod,
                                         resolved=True))


def _check_plugin_deps() -> None:
    """Report enabled plugins whose declared pip extras are not installed, and
    point at the one-shot fix."""
    from rich.markup import escape

    try:
        from localm.plugins.engine import PluginManager
        missing = PluginManager(None).all_missing_deps(enabled_only=True)
    except Exception as e:
        console.print(f"  {_WARN_SYM}  plugin dependency check skipped "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    if not missing:
        console.print(f"  {_OK_SYM}  plugin dependencies: enabled plugins have theirs")
        return
    # escape(): `name`/`reqs` come from an installed plugin's own manifest, not
    # a localm-controlled charset - a community plugin's declared name or pip
    # requirement string could contain anything. Quoted by hand rather than via
    # Python's !r/repr(): repr() applied AFTER escape() re-escapes escape()'s
    # own backslash (verified empirically - it re-breaks the bracketed span),
    # and escape() applied to a repr'd string is the wrong order too; a plain
    # manual quote sidesteps the interaction entirely.
    for name, reqs in missing.items():
        console.print(f"  {_WARN_SYM}  plugin '{escape(name)}' is missing: "
                      f"{', '.join(escape(r) for r in reqs)}")
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
    from rich.markup import escape

    try:
        from localm.media.managed_comfy import (
            is_managed_comfy_installed,
            managed_comfy_paths,
        )
        installed = is_managed_comfy_installed()
    except Exception as e:  # noqa: BLE001 - a hint must not fail doctor; surface why.
        console.print(f"  {_WARN_SYM}  managed-ComfyUI hint skipped "
                      f"[dim]({escape(str(e))})[/dim]")
        return
    # soft_wrap: this line names a literal path/command a user may copy-paste; a
    # narrow terminal must never break it mid-token (only affects THIS line, not
    # the rest of doctor's output, which still wraps normally).
    if installed:
        # escape(): a real filesystem path under LOCALM_HOME - the same risk
        # class as the document paths #1463 fixed in rag.py.
        console.print(
            f"  {_OK_SYM}  managed ComfyUI: installed at "
            f"{escape(str(managed_comfy_paths().root))}",
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
