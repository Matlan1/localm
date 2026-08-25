# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pick ONE CPU-tier ggml backend for this machine and hide every other one."""

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

# The marker recording a completed selection. Same sibling-dotfile convention as
# setup_llama.py's ".localm-backend" (_BACKEND_MARKER) - a tiny JSON file, never
# loaded as code, read back cheaply on every load_lib() call.
_MARKER_NAME = ".localm-cpu-tier"
_MARKER_SCHEMA = 1

# Rejected tiers are renamed with this prefix. Deliberately does NOT start with
# "libggml", so it falls outside both this module's own candidate glob AND
# _loader.py's _ggml_glob() ("libggml*.so*") - neither will ever find it again.
_UNUSED_PREFIX = "_unused-"

_CANDIDATE_GLOB = "libggml-cpu-*.so"

_LOCK_NAME = ".localm-cpu-tier.lock"
_LOCK_OWNER_FILE = "owner.json"
_LOCK_WAIT_SECONDS = 45.0
_LOCK_POLL_SECONDS = 0.5

# Isolated per-candidate probe: loads exactly ONE .so (no siblings visible to
# THIS process), calls its own ggml_backend_score(), reports the result. Mirrors
# scripts/confirm_llama_runtime.py's _run_probe shape (env-var-driven target,
# a "@@VERDICT@@<json>" last-line convention) rather than inventing a new one.
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
    """A short, stable string identifying the CPU actually running this process."""
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
    """True when a marker exists, names a tier that is still present (and no longer has un-pruned siblings to collide with), and matches this machine."""
    data = _read_marker(lib_dir)
    if data is None:
        return False
    tier = data.get("tier")
    if not tier or not (lib_dir / tier).exists():
        return False
    if data.get("fingerprint") != _cpu_fingerprint():
        return False
    # A marker is only trustworthy if nothing OTHER than the recorded winner
    # still matches the un-pruned candidate glob - otherwise a partially
    # completed prior prune (e.g. interrupted mid-run) would read as "done"
    # while the actual collision hazard is still present on disk.
    remaining = _candidates(lib_dir)
    return remaining == [lib_dir / tier]


@contextlib.contextmanager
def _lock(lib_dir: Path):
    """Cross-process mkdir-based lock, mirroring setup_llama.py's _provisioning_lock idiom (atomic os.mkdir, PID-liveness staleness via localm.instances.pid_alive - never elapsed time)."""
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
        # Timed out or could not take the lock at all. Proceed unlocked rather
        # than fail the load outright - worst case this run races another
        # selection and a later load_lib() call redoes/corrects it; that is a
        # smaller risk than refusing to load a model because a lock was busy.
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
    """The candidate's own ggml_backend_score() in an isolated subprocess, or None if it could not be measured (load failure, missing symbol, crash - all of which mean 'not usable', same as a real score of 0)."""
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
    """Ensure exactly one ``libggml-cpu-*.so`` in *lib_dir* is selectable by ggml's directory scan, pruning the rest."""
    if _marker_is_current(lib_dir):
        data = _read_marker(lib_dir)
        return data["tier"] if data else None

    with _lock(lib_dir) as did_acquire:
        if not did_acquire:
            # Either another process just finished (marker now current) or we
            # are proceeding unlocked as a last resort - either way, re-check
            # once more before doing any work, to avoid a redundant prune.
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
