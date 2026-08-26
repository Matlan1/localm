# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Pick ONE CPU-tier ggml backend for this machine and hide every other one.

localm's native runtime ships one ``libggml-cpu-<tier>.so`` per x86
microarchitecture (alderlake, haswell, zen4, ...). Both localm's own loader
(``llamacpp/_loader.py``'s ``_preload()``) and ggml's native
``ggml_backend_load_all()`` dlopen EVERY tier present in the runtime directory,
which is how ggml discovers which one is usable. Every tier's ``.so`` exports
IDENTICALLY-NAMED global C symbols (``llamafile_sgemm``,
``ggml_backend_cpu_init``, etc) as independently compiled copies at different
addresses, and a rejected tier is never actually unmapped from the process. With
multiple tiers simultaneously mapped and globally visible, the dynamic linker
can resolve a call meant for the compatible tier into an incompatible one's copy
of the same function: on an AMD Zen 3 CPU with no AVX-512, a real ``embed()``
call crashes with SIGILL inside a matmul kernel belonging to
``libggml-cpu-alderlake.so``, the tier ggml's own ``ggml_backend_score()``
rejected as "not supported on this system" earlier in the same run.

Once only ONE tier's ``.so`` is present under the ``libggml-cpu-*.so`` name
ggml's directory scan looks for, there is nothing left for a symbol collision to
happen WITH, whatever ggml's own loading code does internally.

HOW A TIER IS JUDGED SAFE: each candidate's own exported
``ggml_backend_score()`` is called, in an ISOLATED subprocess per candidate (one
``.so`` loaded, no siblings present in that process, so a probe itself cannot
exhibit this same collision). That is ggml's own authoritative compatibility
check; localm has no CPU-feature or CPUID detection of its own. SCOPE:
compatibility is verified via ``ggml_backend_score()`` only, not by driving a
real quantized matmul through the winning tier.

NON-DESTRUCTIVE AND REVERSIBLE: rejected tiers are renamed in place (prefixed
with ``_unused-``), never deleted and never moved to a subdirectory. A
subdirectory would be invisible to ``setup_llama.py``'s ``_clear_target()``
(which only recurses into a fixed allowlist of subdirectory names) and to
``install_manifest.py``'s ``_bin_files()`` (a non-recursive directory listing),
so files placed there would survive a re-provision as stale leftovers and leak
past uninstall tracking forever. Renamed-in-place files stay flat and
``.so``-suffixed, so both of those mechanisms keep handling them correctly.

SCOPE: POSIX only. ``_loader.py``'s Windows preload path uses plain
``ctypes.CDLL`` with no ``RTLD_GLOBAL`` equivalent - PE/DLL symbol resolution is
per-import-table, not a flat global table, so this collision mechanism does not
apply there by construction. Do not call this on Windows.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from localm.debuglog import logger

# Marker recording a completed selection: a small JSON file, never loaded as
# code, read on every load_lib() call.
_MARKER_NAME = ".localm-cpu-tier"
_MARKER_SCHEMA = 1

# Prefix for a rejected tier. It does not start with "libggml", so it falls
# outside this module's candidate glob and _loader.py's _ggml_glob().
_UNUSED_PREFIX = "_unused-"

_CANDIDATE_GLOB = "libggml-cpu-*.so"

_LOCK_NAME = ".localm-cpu-tier.lock"
_LOCK_OWNER_FILE = "owner.json"
_LOCK_WAIT_SECONDS = 45.0
_LOCK_POLL_SECONDS = 0.5

# Isolated per-candidate probe: loads exactly one .so with no siblings visible in
# that process, calls its ggml_backend_score(), and reports the result on a
# "@@VERDICT@@<json>" last line.
_SCORE_PROBE = r'''
import ctypes, json, os, sys

out = {"score": None, "error": None}
path = os.environ["LOCALM_CPU_TIER_CANDIDATE"]

def emit():
    sys.stdout.write("\n@@VERDICT@@" + json.dumps(out) + "\n")
    sys.stdout.flush()
    raise SystemExit(0)

try:
    lib = ctypes.CDLL(path)
except OSError as e:
    out["error"] = "could not load: %s" % e
    emit()

score_fn = getattr(lib, "ggml_backend_score", None)
if score_fn is None:
    out["error"] = "no ggml_backend_score symbol"
    emit()

score_fn.restype = ctypes.c_int
try:
    out["score"] = int(score_fn())
except Exception as e:
    out["error"] = "ggml_backend_score() raised: %s" % e
emit()
'''


def _cpu_fingerprint() -> str:
    """A short, stable string identifying the CPU actually running this
    process. Not a security/uniqueness ID - only used to notice "this runtime
    directory was provisioned/pruned on different hardware" (a portable install
    copied to another machine, or a VM migrated) so a stale selection gets
    redone rather than silently kept. Best-effort: an unreadable/unusual host
    returns a constant placeholder, which simply means the fingerprint check
    can never distinguish machines on that host - safe, just less precise."""
    if sys.platform.startswith("linux"):
        try:
            vendor = model = ""
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("vendor_id") and not vendor:
                        vendor = line.split(":", 1)[1].strip()
                    elif line.startswith("model name") and not model:
                        model = line.split(":", 1)[1].strip()
                    if vendor and model:
                        break
            if vendor or model:
                return f"{vendor}|{model}"
        except OSError:
            pass
    return "unknown"


def _candidates(lib_dir: Path) -> List[Path]:
    """Every not-yet-pruned CPU-tier .so present, in a stable (sorted) order."""
    try:
        return sorted(lib_dir.glob(_CANDIDATE_GLOB))
    except OSError:
        return []


def _marker_path(lib_dir: Path) -> Path:
    return lib_dir / _MARKER_NAME


def _read_marker(lib_dir: Path) -> Optional[dict]:
    try:
        raw = _marker_path(lib_dir).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("schema") != _MARKER_SCHEMA:
        return None
    return data


def _write_marker(lib_dir: Path, tier_filename: str) -> None:
    data = {
        "schema": _MARKER_SCHEMA,
        "tier": tier_filename,
        "fingerprint": _cpu_fingerprint(),
    }
    try:
        _marker_path(lib_dir).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        # Best-effort, like _BACKEND_MARKER: a write failure only means the
        # NEXT load_lib() call redoes selection (slower, not wrong).
        logger.warning("could not write %s in %s", _MARKER_NAME, lib_dir)


def _marker_is_current(lib_dir: Path) -> bool:
    """True when a marker exists, names a tier that is still present (and no
    longer has un-pruned siblings to collide with), and matches this machine."""
    data = _read_marker(lib_dir)
    if data is None:
        return False
    tier = data.get("tier")
    if not tier or not (lib_dir / tier).exists():
        return False
    if data.get("fingerprint") != _cpu_fingerprint():
        return False
    # A marker is trustworthy only when nothing other than the recorded winner
    # still matches the candidate glob; a partially completed prune would
    # otherwise read as done.
    remaining = _candidates(lib_dir)
    return remaining == [lib_dir / tier]


@contextlib.contextmanager
def _lock(lib_dir: Path):
    """Cross-process mkdir-based lock, mirroring setup_llama.py's
    _provisioning_lock idiom (atomic os.mkdir, PID-liveness staleness via
    localm.instances.pid_alive - never elapsed time). Waits, bounded, rather
    than failing fast: selection here is a handful of small subprocess probes,
    and the alternative to waiting is proceeding on an unpruned,
    still-colliding directory."""
    from localm.instances import pid_alive

    lock = lib_dir / _LOCK_NAME
    owner_file = lock / _LOCK_OWNER_FILE
    deadline = time.monotonic() + _LOCK_WAIT_SECONDS
    acquired = False
    while time.monotonic() < deadline:
        try:
            os.mkdir(str(lock))
            acquired = True
            break
        except FileExistsError:
            pid = None
            with contextlib.suppress(OSError, json.JSONDecodeError, ValueError):
                pid = json.loads(owner_file.read_text(encoding="utf-8")).get("pid")
            if isinstance(pid, int) and not pid_alive(pid):
                with contextlib.suppress(OSError):
                    shutil.rmtree(str(lock))
                continue  # retry the atomic mkdir immediately
            if _marker_is_current(lib_dir):
                # Whoever holds the lock already finished; nothing left to do.
                yield False
                return
            time.sleep(_LOCK_POLL_SECONDS)
        except OSError:
            break
    if not acquired:
        # Timed out or could not take the lock: proceed unlocked rather than fail
        # the load. A later load_lib() call redoes the selection.
        logger.warning(
            "could not take the CPU-tier selection lock in %s; proceeding "
            "without it", lib_dir)
        yield False
        return
    try:
        with contextlib.suppress(OSError):
            owner_file.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        yield True
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(str(lock))


def _probe_score(candidate: Path, lib_dir: Path) -> Optional[int]:
    """The candidate's own ggml_backend_score() in an isolated subprocess, or
    None if it could not be measured (load failure, missing symbol, crash -
    all of which mean "not usable", same as a real score of 0)."""
    env = dict(os.environ)
    env["LOCALM_CPU_TIER_CANDIDATE"] = str(candidate)
    # So the candidate's own DT_NEEDED dependency on the base ggml library
    # resolves, exactly as it would inside the real process.
    env["LD_LIBRARY_PATH"] = str(lib_dir) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    try:
        r = subprocess.run([sys.executable, "-c", _SCORE_PROBE], env=env,
                           capture_output=True, text=True, timeout=30)
    except Exception as e:
        logger.debug("CPU-tier probe of %s could not run: %s", candidate, e)
        return None
    for line in reversed((r.stdout or "").splitlines()):
        if line.startswith("@@VERDICT@@"):
            try:
                verdict = json.loads(line[len("@@VERDICT@@"):])
            except (json.JSONDecodeError, ValueError):
                break
            if verdict.get("error"):
                logger.debug("CPU-tier probe of %s: %s", candidate, verdict["error"])
                return None
            score = verdict.get("score")
            return int(score) if isinstance(score, int) else None
    logger.debug("CPU-tier probe of %s emitted no verdict (exit %s)",
                 candidate, r.returncode)
    return None


def ensure_cpu_tier_selected(lib_dir: Path) -> Optional[str]:
    """Ensure exactly one ``libggml-cpu-*.so`` in *lib_dir* is selectable by
    ggml's directory scan, pruning the rest. Idempotent and safe to call on
    every ``load_lib()`` - the common case (a marker already matches) costs one
    file read. Returns the winning tier's filename, or None when there was
    nothing to prune (no candidates present, or every probe failed - in which
    case the directory is left untouched and the caller's own load attempt
    will surface whatever the real underlying problem is)."""
    if _marker_is_current(lib_dir):
        data = _read_marker(lib_dir)
        return data["tier"] if data else None

    with _lock(lib_dir) as did_acquire:
        if not did_acquire:
            # Another process may have finished, or we are unlocked: re-check
            # before doing any work.
            if _marker_is_current(lib_dir):
                data = _read_marker(lib_dir)
                return data["tier"] if data else None

        candidates = _candidates(lib_dir)
        if not candidates:
            return None
        if len(candidates) == 1:
            _write_marker(lib_dir, candidates[0].name)
            return candidates[0].name

        scored = [(c, _probe_score(c, lib_dir)) for c in candidates]
        usable = [(c, s) for c, s in scored if s is not None and s > 0]
        if not usable:
            logger.warning(
                "none of %d CPU-tier candidates in %s reported a usable "
                "ggml_backend_score(); leaving them all in place",
                len(candidates), lib_dir)
            return None

        winner, _ = max(usable, key=lambda pair: pair[1])
        for c, _ in scored:
            if c == winner:
                continue
            try:
                c.rename(c.with_name(_UNUSED_PREFIX + c.name))
            except OSError as e:
                logger.warning("could not prune CPU-tier candidate %s: %s", c, e)

        _write_marker(lib_dir, winner.name)
        logger.info("selected CPU backend tier %s for this machine "
                    "(%d other tier(s) pruned)", winner.name, len(usable) - 1)
        return winner.name
