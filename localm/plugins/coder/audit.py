"""
Session audit log and session-mode control for localcoder.

Session modes
-------------
privacy (default)
    Nothing is written to disk automatically.  The user can still call
    ``/save`` at any time to dump the conversation to a JSON file.

log
    A JSONL audit trail is appended to
    ``~/.localm/sessions/<YYYY-MM-DD_HHMMSS>_<pid>_<label>.jsonl``
    for every session.  One event per line:
    ``{"t": unix_ms, "turn": int, "type": str, "data": any}``

full
    Everything in ``log`` mode, plus a human-readable Markdown transcript
    written to ``.localcoder/sessions/<YYYY-MM-DD_HHMMSS>.md`` inside the
    project directory when the session ends (``agent.close()``).

Usage
-----
The mode is set via:
  - ``--mode`` CLI flag
  - ``mode = "log"`` in ``.localcoder/config.toml``
  - Defaults to ``privacy`` if unset.

Switching mid-session is not supported (the JSONL file is opened at startup).
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Union


# ---------------------------------------------------------------------------
#  SessionMode
# ---------------------------------------------------------------------------

class SessionMode(str, Enum):
    """Controls what the agent persists automatically during a session."""
    PRIVACY = "privacy"   # nothing written automatically
    LOG     = "log"       # JSONL audit trail to ~/.localm/sessions/
    FULL    = "full"      # JSONL + markdown transcript in .localcoder/sessions/


_VALID_MODES = {m.value for m in SessionMode}


def parse_mode(value: str) -> SessionMode:
    """Convert a string to a SessionMode, raising ValueError on bad input."""
    v = value.strip().lower()
    if v not in _VALID_MODES:
        raise ValueError(
            f"Unknown session mode '{value}'. "
            f"Choose from: {', '.join(sorted(_VALID_MODES))}"
        )
    return SessionMode(v)


# ---------------------------------------------------------------------------
#  NullAuditLog  (privacy mode — no-op)
# ---------------------------------------------------------------------------

class NullAuditLog:
    """
    Drop-in replacement for AuditLog that writes nothing to disk.

    Every method is a silent no-op so the rest of the agent code is
    identical regardless of the active session mode.
    """

    @property
    def path(self) -> Path | None:
        return None

    def set_turn(self, turn: int) -> None:
        pass

    def user(self, content: str) -> None:
        pass

    def llm(self, content: str, tokens: int = 0) -> None:
        pass

    def tool_call(self, name: str, args: dict) -> None:
        pass

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
#  AuditLog  (log + full modes — JSONL writer)
# ---------------------------------------------------------------------------

_SESSIONS_DIR = Path.home() / ".localm" / "sessions"


def _sessions_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


class AuditLog:
    """
    Append-only JSONL session log.

    Each line is a self-contained JSON object:
    {"t": <unix_ms>, "turn": <int>, "type": <str>, "data": <any>}

    Used in ``log`` and ``full`` modes.  In ``privacy`` mode, a
    ``NullAuditLog`` is used instead.
    """

    def __init__(self, label: str = "") -> None:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        pid = os.getpid()
        suffix = f"_{label}" if label else ""
        filename = f"{ts}_{pid}{suffix}.jsonl"
        self._path = _sessions_dir() / filename
        self._turn = 0
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        self._write("system", {"msg": "session started"})

    @property
    def path(self) -> Path:
        return self._path

    def set_turn(self, turn: int) -> None:
        self._turn = turn

    def user(self, content: str) -> None:
        self._write("user", {"content": content[:2000]})

    def llm(self, content: str, tokens: int = 0) -> None:
        self._write("llm", {"content": content[:2000], "tokens": tokens})

    def tool_call(self, name: str, args: dict) -> None:
        safe_args = {k: (str(v)[:200] if isinstance(v, str) else v) for k, v in args.items()}
        self._write("tool_call", {"name": name, "args": safe_args})

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        self._write("tool_result", {"name": name, "ok": ok, "summary": summary[:200]})

    def close(self) -> None:
        try:
            self._write("system", {"msg": "session ended"})
            self._fh.close()
        except Exception:
            pass

    def _write(self, event_type: str, data: Any) -> None:
        try:
            record = {
                "t":    int(time.time() * 1000),
                "turn": self._turn,
                "type": event_type,
                "data": data,
            }
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            pass   # never crash the agent due to logging


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

AuditLogT = Union[AuditLog, NullAuditLog]


def make_audit_log(mode: SessionMode, label: str = "") -> AuditLogT:
    """
    Return an appropriate audit log object for the given session mode.

    ``privacy`` → NullAuditLog (no disk writes)
    ``log``     → AuditLog (JSONL in ~/.localm/sessions/)
    ``full``    → AuditLog (JSONL; the markdown is written by agent.close())
    """
    if mode == SessionMode.PRIVACY:
        return NullAuditLog()
    return AuditLog(label=label)
