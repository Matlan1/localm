"""
Session audit log — appends one JSON-lines event per agent action.

Log location:  ~/.localm/sessions/<YYYY-MM-DD_HHMMSS>_<pid>.jsonl

Event types: user | llm | tool_call | tool_result | system

Usage (in agent.py):
    from .audit import AuditLog
    self._audit = AuditLog()          # creates the file
    self._audit.user("hello")
    self._audit.llm("response text")
    self._audit.tool_call("read_file", {"path": "x"})
    self._audit.tool_result("read_file", ok=True, summary="x — 10 lines")
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


_SESSIONS_DIR = Path.home() / ".localm" / "sessions"


def _sessions_dir() -> Path:
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSIONS_DIR


class AuditLog:
    """
    Append-only JSONL session log.

    Each line is a self-contained JSON object:
    {"t": <unix_ms>, "turn": <int>, "type": <str>, "data": <any>}
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
