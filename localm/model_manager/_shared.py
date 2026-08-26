# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, low-level state for the model_manager package: the Rich console, the
GUI progress sentinel, the progress emitter that writes on it, and the one-time
Windows stdout/stderr UTF-8 reconfigure (an import-time side effect).

``_emit_progress`` and ``_verify_digest`` live HERE rather than in ``pull.py``:
``pull`` imports from ``registry``, so a registry-side caller could not reach
them without an import cycle, and ``registry`` has its own multi-GB hash to
report (``_store_into_models_dir`` hashes a copy twice). This module imports
nothing from the package, which is what makes it safe for either to use.
"""

import json
import os
import sys
import time
from pathlib import Path

import localm.model_manager as _mm  # read package-patchable names at call time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console

console = Console()

# GUI download progress: the GUI runs ``localm pull`` as a subprocess and
# parses its stdout. When LOCALM_PROGRESS_JSON=1 the downloader streams
# structured progress lines (this sentinel + JSON) that the GUI renders as a
# progress bar; interactive CLI use keeps huggingface_hub's own tqdm bars.
PROGRESS_SENTINEL = "\x1flocalm-progress\x1f"


def _emit_progress(downloaded: int, total: int, *, phase: str = "download",
                   name: "str | None" = None, index: int = 0, count: int = 0,
                   zero_is_unknown: bool = False) -> None:
    """Write one progress frame on the sentinel channel.

    *zero_is_unknown* suppresses the percentage while NOTHING has landed yet, and
    belongs on every IN-FLIGHT emitter (the opening seed and the poll loop). Set
    it there, never on a terminal event: before the first byte arrives, "0%" is
    indistinguishable from a stalled download, and the GUI renders the
    suppressed case as a busy bar plus a running byte count instead
    (pages/models.js, the `ev.pct != null && ev.total` branch's else).

    Scoped to `downloaded == 0`, not to the seed wholesale: a RESUME seed
    carries a real measurement (429MB of 4.68GB is a true 9.2% before any new
    byte moves) and still reports its percentage.

    The terminal event keeps exact semantics: a failed pull that landed nothing
    genuinely IS at zero.
    """
    pct = round(downloaded * 100 / total, 1) if total else None
    if zero_is_unknown and not downloaded:
        pct = None
    payload = {"phase": phase, "downloaded": downloaded, "total": total, "pct": pct}
    # For a multi-file download (a split GGUF), tell the GUI which file is in
    # flight so it can show "file 2 of 3: <name>". Omitted for a single file.
    if count > 1:
        payload["count"] = count
        payload["index"] = index
        if name:
            payload["name"] = name
    sys.stdout.write(PROGRESS_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_outcome(status: str) -> None:
    """Write a terminal OUTCOME frame on the sentinel channel: this command's
    real work has definitively finished with *status* ("done" or "failed"),
    decided BEFORE any later display step that could itself crash.

    GUI mode only (LOCALM_PROGRESS_JSON=1), same gate as ``_emit_progress``:
    interactive CLI use has no consumer for this frame and must not print raw
    sentinel JSON to a real terminal.

    The GUI job runner (``localm/plugins/gui/jobs.py``) otherwise infers a
    job's outcome from the subprocess exit code alone, so an exception raised
    by a display step AFTER the real work already succeeded reports a completed
    operation as failed. Call this BEFORE that risky step, once real work
    (download, install, registry write, ...) is verifiably done; emitting it
    later than that has no effect.

    This is an internal producer -> job-runner signal, not a progress update:
    ``jobs.py`` consumes it to correct its own status decision and does not
    forward it to SSE subscribers.
    """
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        return
    sys.stdout.write(
        PROGRESS_SENTINEL + json.dumps({"type": "outcome", "status": status}) + "\n")
    sys.stdout.flush()


# How often the verify phase may emit. Matches _download_progress's 0.7s poll so
# both phases tick at the same visible rate. _sha256_file calls back after EVERY
# block, so without this a large file emits hundreds of events a second.
_VERIFY_EMIT_INTERVAL_S = 0.7


def _verify_digest(path: Path, *, purpose: str = "to verify the download") -> str:
    """SHA256 *path*, reporting a ``verify`` phase while it runs.

    Every caller hashes a file the user has just watched download, AFTER the
    download's own terminal event has already announced 100%. Hashing many GB
    takes minutes, and ``phase`` is what distinguishes that stage from the
    download.

    In GUI mode (LOCALM_PROGRESS_JSON=1) this streams progress events on the
    existing sentinel channel; otherwise the CLI's own bar renders it. Both are
    driven by the same ``_sha256_file`` callback.

    The 100% reported here is a claim about the HASHING reaching the end of the
    file, NEVER about the digest matching. A mismatch is the caller's to report,
    and it is reported as an error rather than as a percentage.
    """
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        # CLI: _hash_with_progress already owns the size threshold and the bar.
        # It returns None only for a directory, which no caller here passes; the
        # fallback keeps the return type honest.
        return _mm._hash_with_progress(path, purpose=purpose) or _mm._sha256_file(path)

    last = 0.0
    seen = (0, 0)                      # (done, total) as last seen from the hasher

    def _report(done: int, total: int) -> None:
        nonlocal last, seen
        seen = (done, total)
        now = time.monotonic()
        if now - last < _VERIFY_EMIT_INTERVAL_S:
            return
        last = now
        _emit_progress(done, total, phase="verify")

    digest = _mm._sha256_file(path, progress=_report)
    # Terminal event, unconditionally and OUTSIDE the throttle, mirroring what
    # the download context managers do from `finally`. It may repeat a tick that
    # just got through; progress is latched, so a duplicate is harmless, while a
    # missing terminal event strands every consumer short of the end.
    _emit_progress(*seen, phase="verify")
    return digest
