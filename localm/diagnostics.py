# SPDX-License-Identifier: AGPL-3.0-or-later
"""Active self-checks, as a callable core no surface owns.

Five ACTIVE probes - checks that go and try the thing rather than read a version
string. This module holds the probes; ``localm/cli/doctor.py`` renders them for
a terminal and ``localm/plugins/gui/routes/doctor.py`` renders them for a
browser, so neither surface has to parse the other's output.

The five:

  llama_lib      a provisioned llama library that EXISTS but is 0 bytes or
                 truncated, plus rocBLAS/hipBLASLt kernel data missing next to
                 a library that loads fine (the silent one - chat works and the
                 crash arrives on the first GEMM)
  native_abi     the struct-layout self-check against the actual DLL
  worker_spawn   a real multiprocessing "spawn" round trip (a plain subprocess
                 probe passes when this is broken)
  venv           a real ``-m venv`` plus a pip-landed check
  hf_backend     transformers' lazy classes really resolve (`import
                 transformers` can succeed while every model load dies)

The narrower reads a surface already has - VRAM, the GPU list, the installed
backend, plugin pip extras, the Python version, package versions - are NOT here.
They have GUI equivalents already, and duplicating them would create a second
source of truth.

NOTHING IN THIS MODULE PRINTS, and nothing here imports click or rich: a caller
renders findings, and a caller that cannot render markup (a JSON route) must not
have to strip it. Findings carry plain text plus the two decorations the
terminal renderer needs (an inline ``note``, indented ``hints``), so the CLI
renders markup and the GUI shows the same sentences without it.

RUNNING THIS IN A SERVER PROCESS IS NOT THE SAME AS RUNNING IT IN A FRESH ONE.
``check_hf_backend`` imports torch and transformers, which on this project's
Windows + AMD ROCm build is the known-doomed DLL-identity conflict once
llama.cpp's native runtime is already loaded in the same process (see
VramSizingMixin._free_total_vram_bytes). ``run_report_isolated`` therefore runs
the whole set in a child interpreter, which is also what makes a GUI answer
comparable to a terminal ``localm doctor`` run - that is a fresh process too.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# Status vocabulary, shared by findings, checks and the aggregate verdict.
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIPPED = "skipped"
# Only ever the aggregate: the run itself could not be completed, which is a
# different fact from every check passing.
ERROR = "error"

# Worst-first, so the aggregate is a max over this. SKIPPED sits below OK: a
# check that did not run neither drags a clean report down nor lifts a failing
# one up.
_SEVERITY = {SKIPPED: 0, OK: 1, WARN: 2, FAIL: 3, ERROR: 4}

# The prefix ``run_report_isolated`` scans the child's stdout for: one parseable
# line, so nothing else the child prints is mistaken for the result.
JSON_PREFIX = "LOCALM_DIAGNOSTICS:"
# One line per check as the child starts it, so a surface can say which check is
# in flight. A separate prefix from the result, which arrives once at the end.
PROGRESS_PREFIX = "LOCALM_DIAGNOSTICS_PROGRESS:"


@dataclass(frozen=True)
class Finding:
    """One line a check produced.

    ``text`` is a whole sentence in plain prose. ``note`` is a short
    parenthetical the terminal prints dim on the same line (a struct layout, a
    captured reason); ``hints`` are extra lines printed dim underneath. Both are
    kept apart from ``text`` so a non-terminal surface can lay them out its own
    way instead of unpicking markup.
    """

    status: str
    text: str
    note: str = ""
    hints: tuple = ()

    def as_dict(self) -> dict:
        out: dict = {"status": self.status, "text": self.text}
        if self.note:
            out["note"] = self.note
        if self.hints:
            out["hints"] = list(self.hints)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        return cls(status=str(d.get("status") or WARN),
                   text=str(d.get("text") or ""),
                   note=str(d.get("note") or ""),
                   hints=tuple(str(h) for h in (d.get("hints") or ())))


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict plus every line it produced.

    ``findings`` may be EMPTY, and that is meaningful rather than a bug: a check
    that did not run has nothing to say to a terminal, which is exactly what
    ``localm doctor`` has always done for an absent optional backend. ``summary``
    still carries the reason, because a compact surface has room for one line and
    rendering nothing at all would be the least honest option.
    """

    key: str
    label: str
    status: str
    summary: str
    findings: tuple = field(default_factory=tuple)

    @property
    def healthy(self) -> bool:
        """True only for a clean pass. A warning is not a pass: callers use this
        to decide whether it is safe to load-test the runtime, and a truncated
        library must answer no."""
        return self.status == OK

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "status": self.status,
                "summary": self.summary,
                "findings": [f.as_dict() for f in self.findings]}

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        return cls(key=str(d.get("key") or ""), label=str(d.get("label") or ""),
                   status=str(d.get("status") or WARN),
                   summary=str(d.get("summary") or ""),
                   findings=tuple(Finding.from_dict(f)
                                  for f in (d.get("findings") or ())))


@dataclass(frozen=True)
class DiagnosticsReport:
    """Every check plus the one-word aggregate a compact surface leads with."""

    checks: tuple = field(default_factory=tuple)
    verdict: str = OK
    # Set only when verdict is ERROR: the run could not be completed, and this
    # says why. The only field that can contradict the checks.
    error: str = ""

    def as_dict(self) -> dict:
        out: dict = {"verdict": self.verdict,
                     "checks": [c.as_dict() for c in self.checks]}
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticsReport":
        return cls(checks=tuple(CheckResult.from_dict(c)
                                for c in (d.get("checks") or ())),
                   verdict=str(d.get("verdict") or WARN),
                   error=str(d.get("error") or ""))


def _worst(statuses) -> str:
    """The most severe status present, or SKIPPED when there is nothing."""
    worst = SKIPPED
    for s in statuses:
        if _SEVERITY.get(s, 0) > _SEVERITY.get(worst, 0):
            worst = s
    return worst


def _result(key: str, label: str, findings, *, summary: str = "",
            status: str = "") -> CheckResult:
    """Assemble a CheckResult from its findings.

    The summary defaults to the first finding that carries the check's OWN
    verdict, not simply the first finding: ``check_llama_lib`` reports a green
    "found" line and then the BLAS failures underneath it, so leading with
    findings[0] would hand a compact surface the reassuring half of a failure.
    """
    findings = tuple(findings)
    status = status or _worst(f.status for f in findings)
    if not summary:
        lead = next((f for f in findings if f.status == status),
                    findings[0] if findings else None)
        if lead is not None:
            summary = lead.text + (f" ({lead.note})" if lead.note else "")
    return CheckResult(key=key, label=label, status=status, summary=summary,
                       findings=findings)


# Every bound a single isolated run can spend. The outer deadline has to fit
# around their sum; test_diagnostics_core asserts that arithmetic rather than the
# literals.
PROBE_TIMEOUT_S = 120.0        # the ABI probe's own subprocess
VENV_TIMEOUT_S = 60.0          # `-m venv`
VENV_PIP_TIMEOUT_S = 30.0      # the follow-up `-m pip --version`
SPAWN_REPLY_TIMEOUT_S = 20.0   # waiting for the spawned child's one message
SPAWN_JOIN_TIMEOUT_S = 5.0     # joining it, twice (once after terminate)
# Headroom for everything with no timeout of its own: interpreter startup, the
# localm import, and `import torch` + `import transformers` on a cold filesystem.
UNBOUNDED_HEADROOM_S = 120.0


def worst_case_run_seconds() -> float:
    """Every bounded step's ceiling, plus headroom for the unbounded ones."""
    return (PROBE_TIMEOUT_S + VENV_TIMEOUT_S + VENV_PIP_TIMEOUT_S
            + SPAWN_REPLY_TIMEOUT_S + 2 * SPAWN_JOIN_TIMEOUT_S
            + UNBOUNDED_HEADROOM_S)


def run_probe_subprocess(code: str, prefix: str, *,
                         timeout: float = PROBE_TIMEOUT_S) -> Optional[dict]:
    """Run *code* in a fresh subprocess and parse the one stdout line starting
    with *prefix* as JSON, e.g. ``"GPU_PROBE:{...}"``.

    Returns None on ANY failure (timeout, crash, no matching line). That None is
    load-bearing and callers must keep it distinct from a parsed result: it means
    the probe never reported, which is a different fact from the probe running
    and saying it could not check. See ``check_native_abi``.
    """
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        line = next((ln for ln in (r.stdout or "").splitlines()
                     if ln.startswith(prefix)), "")
        return json.loads(line[len(prefix):]) if line else None
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  1. The provisioned llama library, and its BLAS kernel data          #
# ------------------------------------------------------------------ #

# A genuine llama.dll/.so is multiple MB. 64 KiB is a generous floor that still
# rejects 0/1-byte stubs and tiny placeholders.
TINY_LIB_BYTES = 64 * 1024

_LIB_LABEL = "llama.cpp library"


def _blas_findings(binary_dir) -> list:
    """Report a ROCm/HIP install whose BLAS kernel data is missing.

    A found-and-loadable llama.dll is NOT the same as a usable install. rocBLAS
    resolves its GPU-arch GEMM kernels at runtime from a data directory next to
    the DLL, so an install can pass every check above with ZERO kernels and then
    hard-crash the native process (uncatchable from Python) the first time a
    workload dispatches through Tensile - the embedder's batch encode.

    That state is reachable two ways, and this catches both: a provision that
    copied the library and dropped its data, and a provision interrupted
    part-way (a locked file on a machine with the runtime open).

    FAIL rather than WARN, because the failure it predicts is a hard process
    crash, and because the remedy is one command."""
    from localm.setup_llama import blas_kernel_problems
    return [Finding(FAIL, f"{p} - GPU matrix ops will crash the native "
                          f"process; re-run 'localm setup-llama --force'")
            for p in blas_kernel_problems(binary_dir)]


def check_llama_lib(find_binary_dir: Optional[Callable] = None) -> CheckResult:
    """Is a provisioned llama library present, non-empty and complete.

    Existence alone is not health: a zeroed or truncated llama.dll exists but
    cannot load, and a rocBLAS install can ship the library without the kernel
    data it resolves at runtime. Both are read off the filesystem, so this check
    loads nothing and cannot crash.

    ``find_binary_dir`` is injected rather than imported so ``doctor`` can pass
    the accessor it resolved from ``localm.cli`` at call time (which is what
    tests monkeypatch); the default is the real one.
    """
    if find_binary_dir is None:
        from localm.config import find_binary_dir as _fbd
        find_binary_dir = _fbd
    label = _LIB_LABEL
    binary_dir = find_binary_dir()
    if not binary_dir:
        return _result("llama_lib", label, [Finding(
            FAIL, "llama binary dir not found - GGUF backend unavailable")])
    dll_names = ["llama.dll", "llama.so", "libllama.so", "llama"]
    found_dll = next(
        (binary_dir / d for d in dll_names if (binary_dir / d).exists()),
        None,
    )
    if not found_dll:
        files = [f.name for f in binary_dir.iterdir() if f.is_file()][:8]
        return _result("llama_lib", label, [Finding(
            WARN, f"binary dir found ({binary_dir}) but no llama .dll/.so - "
                  f"contents: {files}")])
    try:
        size = found_dll.stat().st_size
    except OSError as e:
        return _result("llama_lib", label, [Finding(
            FAIL, f"{found_dll.name} in {binary_dir} cannot be "
                  f"read (corrupt?): {e}")])
    if size == 0:
        return _result("llama_lib", label, [Finding(
            FAIL, f"{found_dll.name} in {binary_dir} is empty "
                  f"(0 bytes) - corrupt; re-run 'localm setup-llama'")])
    if size < TINY_LIB_BYTES:
        return _result("llama_lib", label, [Finding(
            WARN, f"{found_dll.name} found in {binary_dir} but "
                  f"is suspiciously small ({size} bytes, expected multiple MB) "
                  f"- it may be truncated/corrupt")])
    findings = [Finding(OK, f"{found_dll.name} found in {binary_dir}")]
    findings.extend(_blas_findings(binary_dir))
    return _result("llama_lib", label, findings)


# ------------------------------------------------------------------ #
#  2. Native ABI self-check                                            #
# ------------------------------------------------------------------ #

_ABI_PROBE_CODE = (
    "import json;"
    "from localm.inference.backends.llamacpp._abi import abi_report;"
    "v=abi_report();"
    "print('ABI_RESULT:'+json.dumps("
    "{'status':v.status,'detail':v.detail,'failures':v.failures[:3],"
    "'layout':v.layout,'context_layout':v.context_layout}))"
)

_ABI_LABEL = "Native ABI"


def check_native_abi() -> CheckResult:
    """Native ABI self-check (struct layout vs the actual DLL). Runs in a
    SUBPROCESS (like setup-llama's load test) so a broken/incompatible DLL can
    never crash the caller, and so the GPU runtime is loaded out-of-process."""
    # Kept separately from the `or {}` fallback below: None means the probe never
    # ran, which is a different fact from the probe running and reporting that it
    # could not check. Only the reason line distinguishes them.
    abi_raw = run_probe_subprocess(_ABI_PROBE_CODE, "ABI_RESULT:")
    abi = abi_raw or {}
    status = abi.get("status", "unchecked")
    # Which of the two llama_model_params layouts was selected: upstream
    # reordered that struct in place at an unchanged size, so this is the only
    # externally visible sign of which runtime generation is installed.
    layout = abi.get("layout") or ""
    context_layout = abi.get("context_layout") or ""
    layout_bits = ", ".join(
        s for s in (f"model params {layout}" if layout else "",
                    f"context params {context_layout}" if context_layout else "")
        if s)
    if status == "ok":
        return _result("native_abi", _ABI_LABEL, [Finding(
            OK, "native ABI: struct layout matches this build",
            note=layout_bits)])
    if status == "mismatch":
        return _result("native_abi", _ABI_LABEL, [Finding(
            FAIL, "native ABI MISMATCH - the runtime's struct "
                  "layout differs from this build; loading is refused to avoid "
                  "memory corruption. Run 'localm setup-llama --force'.",
            hints=tuple(str(f) for f in abi.get("failures", [])))])
    if status == "skipped":
        return _result("native_abi", _ABI_LABEL, [Finding(
            WARN, "native ABI check skipped (LOCALM_SKIP_ABI_CHECK set)")])
    # abi_report() populates detail on every path it can return from, so a
    # hardcoded default reason is reachable only when the probe did not run at
    # all.
    if abi_raw is None:
        detail = ("the ABI probe did not run - it timed out, crashed, or "
                  "printed no result")
    else:
        detail = abi.get("detail") or "no reason reported"
    return _result("native_abi", _ABI_LABEL,
                   [Finding(WARN, "native ABI not verified", note=detail)])


# ------------------------------------------------------------------ #
#  3. The isolated worker process                                      #
# ------------------------------------------------------------------ #

_SPAWN_LABEL = "Worker process spawn"


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


def check_worker_spawn() -> CheckResult:
    """Verify localm can actually spawn its isolated worker process - the SAME
    ``multiprocessing.get_context("spawn")`` mechanism every GGUF model load and
    the voice/STT engine depend on (see localm/_mp_spawn.py).

    The native-ABI and GPU-probe checks isolate via a PLAIN subprocess
    (``run_probe_subprocess``), a different code path: that proves the native
    library loads and computes correctly, but it does NOT exercise
    multiprocessing's own spawn machinery, which on Windows redirects the
    child's executable under conditions a renamed launcher (LocaLM.exe) can
    break, so every GGUF load fails with "[WinError 2] The system cannot find
    the file specified" while every other check stays green."""
    label = _SPAWN_LABEL
    try:
        from localm._mp_spawn import ensure_spawn_uses_venv_python
        ensure_spawn_uses_venv_python()
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = mp.Pipe(duplex=False)
        proc = ctx.Process(target=_worker_spawn_probe, args=(child_conn,), daemon=True)
    except Exception as e:
        # Setup itself failed (e.g. the fix helper or Pipe() errored) - rarer
        # and genuinely different from a spawn failure, so it gets its own line.
        return _result("worker_spawn", label, [Finding(
            FAIL, f"background worker spawn check errored: {e}")])

    reply = None
    error_detail = None
    try:
        proc.start()
        got_reply = parent_conn.poll(SPAWN_REPLY_TIMEOUT_S)
        reply = parent_conn.recv() if got_reply else None
        proc.join(SPAWN_JOIN_TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            proc.join(SPAWN_JOIN_TIMEOUT_S)
    except Exception as e:
        # proc.start() can raise directly rather than the child ever running.
        # Treated the same as "no reply" so every failure reads as one verdict.
        error_detail = str(e)

    if reply == "ok":
        return _result("worker_spawn", label, [Finding(
            OK, "background worker spawn: OK "
                "(model loads and voice transcription use this)")])
    detail = f" ({error_detail})" if error_detail else ""
    return _result("worker_spawn", label, [Finding(
        FAIL, f"background worker spawn FAILED{detail} - GGUF "
              "model loads and voice transcription will fail even though the "
              "runtime above checks out")])


# ------------------------------------------------------------------ #
#  4. Nested venv creation                                             #
# ------------------------------------------------------------------ #

_VENV_LABEL = "Nested venv creation"


def check_venv_creation() -> CheckResult:
    """Verify localm can actually create a nested venv via ``-m venv`` using
    ``real_base_python()`` - the SAME mechanism the managed-ComfyUI installer
    depends on (managed_comfy_fresh.py). Creates and immediately discards a
    throwaway venv under a temp dir; never touches LOCALM_HOME or any real
    install.

    A DIFFERENT code path from the worker-spawn check above (that exercises
    multiprocessing's spawn machinery; this exercises stdlib venv's own
    basename-matching plus its mandatory ensurepip bootstrap), so a managed
    ComfyUI setup failing with "[WinError 2]" is invisible without it.

    Also probes that pip actually landed inside the new venv: ``-m venv`` can
    report success - return code 0, the interpreter file present - while its own
    mandatory ensurepip bootstrap silently failed (a base Python with ensurepip
    stripped, or a broken install). The managed-ComfyUI installer pip-installs
    into a venv it just created with no ``--without-pip`` fallback, so a
    pip-less venv reads as doctor-green right up until provisioning fails deep
    inside with an opaque "No module named pip"."""
    import tempfile
    from pathlib import Path

    from localm._mp_spawn import real_base_python

    label = _VENV_LABEL
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
                capture_output=True, text=True, timeout=VENV_TIMEOUT_S)
            expected = dest / ("Scripts/python.exe" if sys.platform == "win32"
                               else "bin/python3")
            ok = r.returncode == 0 and expected.is_file()
            if not ok:
                tail = (r.stderr or r.stdout or "").strip().splitlines()
                detail = tail[-1] if tail else ""
            else:
                pr = subprocess.run(
                    [str(expected), "-m", "pip", "--version"],
                    capture_output=True, text=True, timeout=VENV_PIP_TIMEOUT_S)
                pip_ok = pr.returncode == 0
                if not pip_ok:
                    tail = (pr.stderr or pr.stdout or "").strip().splitlines()
                    pip_detail = tail[-1] if tail else ""
    except Exception as e:
        return _result("venv", label, [Finding(
            FAIL, f"venv-creation check errored: {e}")])

    if ok and pip_ok:
        return _result("venv", label, [Finding(
            OK, "venv creation: OK (the managed-ComfyUI installer uses this)")])
    if ok and not pip_ok:
        return _result("venv", label, [Finding(
            FAIL, "venv creation FAILED"
                  + (f" ({pip_detail})" if pip_detail else "")
                  + " - the venv was created but has no working pip; managed ComfyUI "
                    "setup ('localm comfy setup') will fail even though the runtime "
                    "above checks out")])
    return _result("venv", label, [Finding(
        FAIL, "venv creation FAILED"
              + (f" ({detail})" if detail else "")
              + " - managed ComfyUI setup ('localm comfy setup') will fail even "
                "though the runtime above checks out")])


# ------------------------------------------------------------------ #
#  5. The HF (transformers) backend, actually usable                   #
# ------------------------------------------------------------------ #

_HF_LABEL = "HF (transformers) backend"


def _import_hf_modules():
    """Import torch and transformers for ``check_hf_backend``, or report why not.

    Returns ``(torch_mod, transformers_mod, reason)``; a None module always comes
    with a reason, because "the optional backend is not installed" and "importing
    it would crash this process" are different facts and only the second is
    something to act on.

    The native_lib_loaded() guard is the same one doctor applies at its own torch
    call sites: once llama.cpp's native runtime is resident, ``import torch`` on
    this project's Windows + AMD ROCm build reliably hits
    STATUS_ENTRYPOINT_NOT_FOUND. A torch ALREADY in sys.modules is a plain cache
    hit and cannot trigger it, so that case is kept."""
    torch_mod = None
    reason = ""
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
    else:
        try:
            from localm.inference.backends.llamacpp import _loader
            native_loaded = _loader.native_lib_loaded()
        except Exception:
            native_loaded = False
        if native_loaded:
            return (None, None,
                    "skipped - llama.cpp's native runtime is already loaded in "
                    "this process, so importing torch here is the known-doomed "
                    "DLL-identity conflict")
        try:
            import torch as _torch
            torch_mod = _torch
        except Exception as e:
            reason = f"torch is not usable here ({type(e).__name__}: {e})"
    if torch_mod is None:
        return None, None, reason or "torch is not installed"
    try:
        import transformers as _tf
    except Exception as e:
        return torch_mod, None, (f"transformers is not usable here "
                                 f"({type(e).__name__}: {e})")
    return torch_mod, _tf, ""


def check_hf_backend(torch_mod: Any = None, transformers_mod: Any = None, *,
                     skip_reason: str = "", resolved: bool = False
                     ) -> CheckResult:
    """Prove the HF (transformers) backend is actually USABLE, not merely
    importable. ``localm/inference/backends/hf.py`` loads models through
    ``transformers.AutoTokenizer`` / ``AutoProcessor`` / ``AutoModelForCausalLM``,
    which transformers resolves through a LAZY module: ``import transformers``
    only sets up that machinery, and a heavy submodule (e.g. distributed/fsdp)
    is imported for real only on the FIRST attribute access that needs it. So
    ``import transformers`` can succeed - and a package-version line reports a
    clean version - while every one of those classes is dead.

    Concretely: transformers 5.14 hard-imports fsdp on the tokenizer path, which
    needs ``torch._C._distributed_c10d`` - absent from the pinned ROCm/Windows
    torch build - so EVERY HF model load dies at "loading processor..." while a
    package-version check reports both packages OK (see
    tests/test_gpu_extra_pins.py for the version-pin guard this backs up with a
    functional one).

    *resolved* says the caller already resolved the two modules and a None means
    absent, so this must not go importing them itself. ``doctor`` passes the
    handles its own package check produced; a standalone run leaves it False and
    lets ``_import_hf_modules`` do the work.

    Produces NO findings when the backend is absent: an optional backend that is
    not installed is not a fault, and the terminal has always stayed silent about
    it. The reason still reaches ``summary`` for a surface that shows a row per
    check."""
    if not resolved and torch_mod is None and transformers_mod is None:
        torch_mod, transformers_mod, skip_reason = _import_hf_modules()
    if torch_mod is None or transformers_mod is None:
        return CheckResult(
            key="hf_backend", label=_HF_LABEL, status=SKIPPED,
            summary=skip_reason or ("not installed - the HF backend is optional "
                                    "(torch + transformers)"),
            findings=())
    try:
        for name in ("AutoTokenizer", "AutoProcessor", "AutoModelForCausalLM"):
            getattr(transformers_mod, name)
    except Exception as e:
        # transformers' lazy loader re-raises a failed submodule import as a
        # generic ModuleNotFoundError chained onto the real cause, and that can
        # repeat several layers deep. Walk the chain to the true root.
        root = e
        seen = {id(root)}
        while True:
            nxt = root.__cause__ or root.__context__
            if nxt is None or id(nxt) in seen:
                break
            root = nxt
            seen.add(id(root))
        return _result("hf_backend", _HF_LABEL, [Finding(
            FAIL, "HF backend (transformers) is installed but UNUSABLE "
                  f"- every HF model load will fail: {type(root).__name__}: {root}")])
    return _result("hf_backend", _HF_LABEL, [Finding(
        OK, "HF backend (transformers): AutoTokenizer / AutoProcessor / "
            "AutoModelForCausalLM load OK")])


# ------------------------------------------------------------------ #
#  The set                                                             #
# ------------------------------------------------------------------ #

# In the order a report reads best: the library first, then what it can be asked,
# then the two process-level probes, then the optional backend. The CLI prints
# them in this order.
CHECK_LABELS = {
    "llama_lib":    _LIB_LABEL,
    "native_abi":   _ABI_LABEL,
    "worker_spawn": _SPAWN_LABEL,
    "venv":         _VENV_LABEL,
    "hf_backend":   _HF_LABEL,
}
CHECK_KEYS = tuple(CHECK_LABELS)


def skipped_native_abi() -> CheckResult:
    """The ABI check's stand-in when there is no healthy library to check.

    Its own named result rather than an omission: a surface that renders one row
    per check must be able to say the row was not run and why, and a missing row
    reads as a check that passed."""
    return CheckResult(
        key="native_abi", label=_ABI_LABEL, status=SKIPPED,
        summary="not checked - there is no healthy llama.cpp library to check "
                "against",
        findings=())


def run_checks(on_check_start: Optional[Callable] = None) -> list:
    """Run all five active checks in THIS process and return their results.

    ``check_native_abi`` is skipped, not run, when the library is not healthy:
    load-testing a runtime already known to be truncated tells nobody anything
    and costs a 120s subprocess timeout. That is the same ordering ``doctor``
    has always used.

    *on_check_start*, when given, is called as
    ``(key, label, done, total)`` immediately BEFORE each check, where ``done``
    is how many have actually finished. Reported before rather than after so a
    watching surface can name the check that is currently taking the time, and
    so ``done`` is never a number nothing has earned yet. A callback that raises
    must not cost the caller its report, so it is guarded - but the failure is
    logged rather than swallowed.

    See the module docstring before calling this from a long-lived process: it
    imports torch and transformers.
    """
    results: list = []
    total = len(CHECK_KEYS)

    def _starting(index: int) -> None:
        if on_check_start is None:
            return
        key = CHECK_KEYS[index]
        try:
            on_check_start(key, CHECK_LABELS[key], len(results), total)
        except Exception as e:  # noqa: BLE001 - progress must never lose a report
            try:
                from localm.debuglog import logger as _dbg
                _dbg.debug("diagnostics: progress callback failed for %s: %r",
                           key, e)
            except Exception:
                pass

    _starting(0)
    lib = check_llama_lib()
    results.append(lib)
    _starting(1)
    results.append(check_native_abi() if lib.healthy else skipped_native_abi())
    _starting(2)
    results.append(check_worker_spawn())
    _starting(3)
    results.append(check_venv_creation())
    _starting(4)
    results.append(check_hf_backend())
    return results


def verdict(checks) -> str:
    """The aggregate for a surface that leads with one word.

    NOT a claim about the machine as a whole: it aggregates these five checks
    and nothing else, so a caller must say WHICH checks it ran rather than
    render it as "everything is fine"."""
    return _worst(c.status for c in checks) or OK


def build_report(checks) -> DiagnosticsReport:
    checks = tuple(checks)
    return DiagnosticsReport(checks=checks, verdict=verdict(checks))


def run_report(on_check_start: Optional[Callable] = None) -> DiagnosticsReport:
    """Run every check in THIS process and aggregate."""
    return build_report(run_checks(on_check_start))


# The child command. ``-c`` rather than ``-m localm.diagnostics``: multiprocessing's
# "spawn" re-imports the parent's __main__ in the child, and with ``-m`` that would
# re-run this module for every spawn the worker-spawn check performs. A ``-c`` main
# has no spec and no __file__, so multiprocessing skips that.
_CHILD_CODE = "import localm.diagnostics as d; d.main_json()"


def run_report_isolated(*, timeout: Optional[float] = None,
                        on_progress: Optional[Callable] = None
                        ) -> DiagnosticsReport:
    """Run the checks in a FRESH child interpreter and parse its one JSON line.

    This is what a server surface must use, for three reasons:

      * ``check_hf_backend`` imports torch and transformers. In a process that
        has already loaded llama.cpp's native runtime that is the known-doomed
        DLL-identity conflict, and the alternative (skip it) would mean the GUI
        can never answer the question the check exists to answer.
      * ``check_venv_creation`` takes up to 90 seconds and ``check_worker_spawn``
        starts a process; neither belongs on a request path.
      * a terminal ``localm doctor`` is itself a fresh process, so this is the
        only way the two surfaces can be expected to agree.

    *on_progress* is called as ``(key, label, done, total)`` each time the child
    starts a check, so a caller can report which one is in flight rather than
    show two minutes of nothing.

    A child that times out, crashes or prints nothing parseable yields an ERROR
    verdict naming what happened - never an empty report, which would render as
    a clean bill of health.

    Read line by line rather than with ``subprocess.run`` because progress that
    only arrives at the end is not progress. That costs the built-in timeout, so
    a watchdog timer kills the child at the deadline and the read loop ends when
    its stdout closes. NOTE what the kill does NOT reach: the child's own
    grandchildren (the ABI probe, the venv probe). Each of those carries its own
    shorter timeout (120s, 60s, 30s), so they self-terminate rather than leak
    indefinitely.
    """
    import threading

    if timeout is None:
        timeout = worst_case_run_seconds()
    try:
        # stderr is merged into stdout rather than given its own pipe: reading one
        # pipe to EOF while the other fills its buffer deadlocks. The result is
        # picked out by prefix, and non-prefixed lines are kept as the tail.
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD_CODE],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return DiagnosticsReport(
            checks=(), verdict=ERROR,
            error=f"the diagnostics run could not be started: {e}")

    timed_out = threading.Event()

    def _expire() -> None:
        timed_out.set()
        try:
            proc.kill()
        except Exception:
            pass

    watchdog = threading.Timer(timeout, _expire)
    watchdog.daemon = True
    watchdog.start()

    result_line = ""
    # A bounded tail of whatever else the child said, so a run that produced no
    # result can still be explained without holding an unbounded log in memory.
    tail: list = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            if line.startswith(JSON_PREFIX):
                result_line = line
            elif line.startswith(PROGRESS_PREFIX):
                if on_progress is None:
                    continue
                try:
                    ev = json.loads(line[len(PROGRESS_PREFIX):])
                    on_progress(ev.get("key", ""), ev.get("label", ""),
                                int(ev.get("done", 0)), int(ev.get("total", 0)))
                except Exception:
                    # A malformed progress line costs a progress update, never the
                    # report.
                    pass
            elif line.strip():
                tail.append(line.strip())
                if len(tail) > 8:
                    tail.pop(0)
    finally:
        watchdog.cancel()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    if timed_out.is_set():
        return DiagnosticsReport(
            checks=(), verdict=ERROR,
            error=f"the diagnostics run did not finish within {int(timeout)}s")
    if not result_line:
        # Surface what the child actually said, so a run that produced no result
        # is still diagnosable.
        why = tail[-1] if tail else "it printed no result"
        return DiagnosticsReport(
            checks=(), verdict=ERROR,
            error=f"the diagnostics run reported nothing usable (exit "
                  f"{proc.returncode}): {why}")
    try:
        return DiagnosticsReport.from_dict(
            json.loads(result_line[len(JSON_PREFIX):]))
    except Exception as e:
        return DiagnosticsReport(
            checks=(), verdict=ERROR,
            error=f"the diagnostics result could not be read: {e}")


def main_json() -> None:
    """Entry point for the isolated run: progress lines then one result line.

    Any exception becomes an ERROR report rather than a traceback and a silent
    parent, so the caller always gets a verdict it can render."""
    def _emit(key, label, done, total):
        sys.stdout.write(PROGRESS_PREFIX + json.dumps(
            {"key": key, "label": label, "done": done, "total": total}) + "\n")
        sys.stdout.flush()

    try:
        report = run_report(_emit)
    except Exception as e:  # noqa: BLE001 - the parent must get a verdict, not a crash
        report = DiagnosticsReport(
            checks=(), verdict=ERROR,
            error=f"the diagnostics run raised {type(e).__name__}: {e}")
    sys.stdout.write(JSON_PREFIX + json.dumps(report.as_dict()) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main_json()
