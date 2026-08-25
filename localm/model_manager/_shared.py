# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared, low-level state for the model_manager package: the Rich console, the GUI progress sentinel, the progress emitter that writes on it, and the one-time Windows stdout/stderr UTF-8 reconfigure (an import-time side effect preserved from the original module)."""

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
    """Write one progress frame on the sentinel channel."""
    pct = round(downloaded * 100 / total, 1) if total else None
    if zero_is_unknown and not downloaded:
        pct = None
    payload = {"phase": phase, "downloaded": downloaded, "total": total, "pct": pct}
    # R06: for a multi-file download (a split GGUF), tell the GUI which file is in
    # flight so it can show "file 2 of 3: <name>". Omitted for a single file so the
    # single-file progress UX is unchanged.
    if count > 1:
        payload["count"] = count
        payload["index"] = index
        if name:
            payload["name"] = name
    sys.stdout.write(PROGRESS_SENTINEL + json.dumps(payload) + "\n")
    sys.stdout.flush()


def _emit_outcome(status: str) -> None:
    """Write a terminal OUTCOME frame on the sentinel channel: this command's real work has definitively finished with *status* ('done' or 'failed'), decided BEFORE any later display step that could itself crash."""
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        return
    sys.stdout.write(
        PROGRESS_SENTINEL + json.dumps({"type": "outcome", "status": status}) + "\n")
    sys.stdout.flush()


# How often the verify phase may emit. Matches _download_progress's 0.7s poll so
# both phases tick at the same visible rate, and so a large file cannot turn one
# emit per 4 MiB block into hundreds of events a second: _sha256_file calls back
# after EVERY block, at a rate set by the hasher's throughput rather than by
# anything a user could perceive.
_VERIFY_EMIT_INTERVAL_S = 0.7


def _verify_digest(path: Path, *, purpose: str = "to verify the download") -> str:
    """SHA256 *path*, reporting a ``verify`` phase while it runs."""
    if os.environ.get("LOCALM_PROGRESS_JSON") != "1":
        # CLI: _hash_with_progress already owns the size threshold and the bar.
        # It returns None only for a directory, which no caller here passes; the
        # fallback keeps the return type honest rather than propagating a None.
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
    # the download context managers do from `finally`.
    #
    # An earlier version tried to exempt the final callback inside _report by
    # testing `done < total`. That is unrecognisable when total is 0 (the size
    # could not be stat'd), so every tick including the last one was throttled
    # away and a fast hash reported a stale count that never advanced to the
    # end. Worst possible case to lose: with no denominator there is no
    # percentage, so the byte count is the ONLY honest signal left.
    #
    # It may repeat a tick that just got through. That is deliberate and matches
    # the download path: progress is latched, so a duplicate is free, whereas a
    # missing terminal event strands every consumer short of the end.
    _emit_progress(*seen, phase="verify")
    return digest
