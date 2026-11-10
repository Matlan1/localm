"""Append-only log of management events, written in every session mode.

One JSON line per event is appended to ``<data dir>/logs/management.jsonl``; the caller passes the data directory.
A line holds the time, the event kind, and only these optional fields:
``principal`` (a key id or a fixed label, never a key), ``keys`` (the NAMES of
the config keys a save changed, never their values), ``plugin``,
``version_from``, ``version_to`` and ``loopback`` (a failed login's client
only). No conversation content and no secret is ever written, which is why the
file is written in privacy mode too.

The file is capped at ``MAX_BYTES``; at the cap it is renamed to
``management.jsonl.1`` (replacing the previous rotation) and a new file starts.
A failed write logs a WARNING and returns False; it never raises into the
operation being recorded. An unknown ``kind`` raises ValueError.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("localm")

KINDS = frozenset({
    "key_minted",
    "key_revoked",
    "config_saved",
    "login_failed",
    "update_applied",
    "plugin_installed",
    "plugin_uninstalled",
})

FILE_NAME = "management.jsonl"
MAX_BYTES = 1_000_000
_MAX_TEXT = 200
_LOCK = threading.Lock()


def log_path(home) -> Path:
    """The management log file under the data directory *home*."""
    return Path(home) / "logs" / FILE_NAME


def _text(value) -> str:
    return str(value)[:_MAX_TEXT]


def record(kind: str, *, home, principal: Optional[str] = None,
           keys: Optional[Iterable[str]] = None, plugin: Optional[str] = None,
           version_from: Optional[str] = None, version_to: Optional[str] = None,
           loopback: Optional[bool] = None) -> bool:
    """Append one event line under the data directory *home*. Returns True when
    written, False when the write failed.

    Raises ValueError for a *kind* outside ``KINDS``. Never raises for an I/O
    failure. Thread-safe; lines are appended whole.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown management event kind: {kind!r}")
    event: dict = {"ts": time.time(), "kind": kind}
    if principal is not None:
        event["principal"] = _text(principal)
    if keys is not None:
        event["keys"] = sorted({_text(k) for k in keys})
    if plugin is not None:
        event["plugin"] = _text(plugin)
    if version_from is not None:
        event["version_from"] = _text(version_from)
    if version_to is not None:
        event["version_to"] = _text(version_to)
    if loopback is not None:
        event["loopback"] = bool(loopback)
    try:
        line = json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n"
        path = log_path(home)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if path.stat().st_size >= MAX_BYTES:
                    os.replace(path, path.with_name(FILE_NAME + ".1"))
            except FileNotFoundError:
                pass
            with open(path, "ab") as f:
                f.write(line.encode("ascii"))
        return True
    except Exception as e:
        logger.warning("could not write the management event log (%s: %s); "
                       "the %s event was not recorded", type(e).__name__, e, kind)
        return False
