# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Session audit log and session-mode control for localm.

Session modes (apply to EVERY surface: terminal chat, API server, web GUI,
and the coder agent)
--------------------------------------------------------------------------
privacy (default)
    Nothing is written to disk automatically - no audit trail, no
    transcripts, no checkpoints, no generation sidecars.  Anything that
    would leave a trace must be explicitly toggled (e.g. /save, /export,
    --debug).

log
    A JSONL audit trail is appended to
    ``<data dir>/sessions/<YYYY-MM-DD_HHMMSS>_<pid>_<label>.jsonl``
    for every session.  One event per line:
    ``{"t": unix_ms, "turn": int, "type": str, "data": any}``

full
    Everything in ``log`` mode, plus a human-readable Markdown transcript:
    - coder:  ``.localcoder/sessions/<ts>.md`` in the project (agent.close())
    - chat:   ``<data dir>/sessions/<ts>_chat.md``
    - server: ``<data dir>/sessions/<ts>_server.md`` (per-exchange append)

Mode resolution (``effective_mode(surface)``)
---------------------------------------------
    coder:   --mode flag > .localcoder/config.toml > config "coder_mode"
             > config "mode" > privacy
    chat:    --mode flag > config "chat_mode" > config "mode" > privacy
    server:  --mode flag > config "mode" > privacy

CLI ``--mode`` flags set the ``LOCALM_MODE`` env var so child processes
inherit the override.  Switching mid-session is not supported (the JSONL
file is opened at startup).
"""

from __future__ import annotations

import json
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from localm.debuglog import logger


# ---------------------------------------------------------------------------
#  SessionMode
# ---------------------------------------------------------------------------

class SessionMode(str, Enum):
    """Controls what localm persists automatically during a session."""
    PRIVACY = "privacy"   # nothing written automatically
    LOG     = "log"       # JSONL audit trail to <data dir>/sessions/
    FULL    = "full"      # JSONL + markdown transcript


_VALID_MODES = {m.value for m in SessionMode}

# Env var carrying an explicit --mode override; inherited by child processes
MODE_ENV_VAR = "LOCALM_MODE"


def parse_mode(value: str) -> SessionMode:
    """Convert a string to a SessionMode, raising ValueError on bad input."""
    v = value.strip().lower()
    if v not in _VALID_MODES:
        raise ValueError(
            f"Unknown session mode '{value}'. "
            f"Choose from: {', '.join(sorted(_VALID_MODES))}"
        )
    return SessionMode(v)


def effective_mode(surface: str, cwd=None) -> SessionMode:
    """
    Resolve the active session mode for *surface* ("chat" | "coder" | "server").

    Precedence: LOCALM_MODE env (set by --mode flags) > (coder only, when *cwd* is
    given) a project-local ``.localcoder/config.toml`` ``mode`` > per-surface
    config key ("chat_mode" / "coder_mode") > global config "mode" > privacy.
    Invalid values fall through to the next level rather than crashing.
    """
    env = os.environ.get(MODE_ENV_VAR, "").strip().lower()
    if env in _VALID_MODES:
        return SessionMode(env)

    # A project-local .localcoder/config.toml mode overrides the global
    # coder_mode; it applies only when the caller passes cwd.
    if surface == "coder" and cwd is not None:
        try:
            from localm.plugins.coder.project_config import (
                ProjectConfigUnreadable,
                load_project_config,
            )
            pmode = load_project_config(Path(cwd)).get("mode")
            if isinstance(pmode, str) and pmode.strip().lower() in _VALID_MODES:
                return SessionMode(pmode.strip().lower())
        except ProjectConfigUnreadable:
            # The project file did not parse, so its mode is unknown. Fail safe
            # to privacy rather than write a transcript for a session that may
            # have been marked private.
            return SessionMode.PRIVACY
        except Exception:
            pass

    try:
        from localm.config import load_config
        cfg = load_config()
    except Exception:
        return SessionMode.PRIVACY

    if surface in ("chat", "coder"):
        per = cfg.get(f"{surface}_mode")
        if isinstance(per, str) and per.strip().lower() in _VALID_MODES:
            return SessionMode(per.strip().lower())

    glob = cfg.get("mode")
    if isinstance(glob, str) and glob.strip().lower() in _VALID_MODES:
        return SessionMode(glob.strip().lower())

    return SessionMode.PRIVACY


# ---------------------------------------------------------------------------
#  NullAuditLog  (privacy mode - no-op)
# ---------------------------------------------------------------------------

class NullAuditLog:
    """
    Drop-in replacement for AuditLog that writes nothing to disk.

    Every method is a silent no-op so calling code is identical regardless
    of the active session mode.
    """

    @property
    def path(self) -> Path | None:
        return None

    def set_turn(self, turn: int) -> None:
        pass

    def user(self, content: str) -> None:
        pass

    def llm(self, content: str, tokens: int = 0, reasoning: str = "") -> None:
        pass

    def tool_call(self, name: str, args: dict) -> None:
        pass

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        pass

    def notice(self, kind: str, message: str) -> None:
        pass

    def episodes_recalled(self, episodes: list) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
#  AuditLog  (log + full modes - JSONL writer)
# ---------------------------------------------------------------------------

# Module-level so tests can patch it (e.g. patch("localm.audit._SESSIONS_DIR"))
from localm.config import HOME_DIR as _HOME_DIR

_SESSIONS_DIR = _HOME_DIR / "sessions"


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
        self._warned_write_fail = False  # warn once, not per line, on write failure
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        self._write("system", {"msg": "session started"})

    @property
    def path(self) -> Path:
        return self._path

    def set_turn(self, turn: int) -> None:
        self._turn = turn

    def user(self, content: str) -> None:
        self._write("user", {"content": content[:2000]})

    def llm(self, content: str, tokens: int = 0, reasoning: str = "") -> None:
        """Record one LLM turn. ``reasoning`` (the separated
        ``reasoning_content``, when the caller's backend/consumer provides it) is
        stored in its OWN field, never appended to ``content``, so the
        visible-answer field stays exactly what was shown or resent."""
        self._write("llm", {
            "content": content[:2000],
            "reasoning": reasoning[:2000] if reasoning else "",
            "tokens": tokens,
        })

    def tool_call(self, name: str, args: dict) -> None:
        safe_args = {k: (str(v)[:200] if isinstance(v, str) else v) for k, v in args.items()}
        self._write("tool_call", {"name": name, "args": safe_args})

    def tool_result(self, name: str, ok: bool, summary: str) -> None:
        self._write("tool_result", {"name": name, "ok": ok, "summary": summary[:200]})

    def notice(self, kind: str, message: str) -> None:
        """Record a session-level condition that is neither a turn nor a tool call
        - a self-review that crashed, a scope that does not confine shell
        execution. ``kind`` groups them for later reading."""
        self._write("notice", {"kind": kind, "message": str(message)[:500]})
    def episodes_recalled(self, episodes: list) -> None:
        """Record WHICH past lessons were injected into this session (id + the
        lesson text, which the injected prompt already carries verbatim into the
        ``user`` entry above). Makes a lesson that steered a run badly traceable
        afterwards, and gives the user the id to forget it by."""
        self._write("episodes_recalled", {"episodes": [
            {"id": str(e.get("id", ""))[:64],
             "outcome": str(e.get("outcome", ""))[:32],
             "lesson": str(e.get("lesson", ""))[:200]}
            for e in (episodes or [])]})

    def close(self) -> None:
        try:
            self._write("system", {"msg": "session ended"})
            self._fh.close()
        except Exception as e:
            logger.warning("AuditLog close failed: %s", e)

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
        except Exception as e:
            # Never crash the host over logging; warn once so a disk-full or
            # permission failure does not truncate the trail with no signal.
            if not self._warned_write_fail:
                self._warned_write_fail = True
                logger.warning("audit log write failed: %s; trail may be incomplete", e)


# ---------------------------------------------------------------------------
#  MarkdownTranscript  (full mode - chat + server surfaces)
# ---------------------------------------------------------------------------

class MarkdownTranscript:
    """
    Append-only human-readable transcript for ``full`` mode.

    The coder agent writes its own richer transcript in ``agent.close()``;
    this lightweight variant serves the chat REPL and the HTTP server,
    appending one exchange at a time (the server has no clean session end).
    """

    def __init__(self, label: str = "chat") -> None:
        ts = time.strftime("%Y-%m-%d_%H%M%S")
        self._path = _sessions_dir() / f"{ts}_{os.getpid()}_{label}.md"
        self._wrote_header = False
        self._label = label

    @property
    def path(self) -> Path:
        return self._path

    def exchange(self, user: str, assistant: str) -> None:
        """Append one user/assistant exchange. Best-effort, never raises.

        The model's ``<think>`` reasoning is separated from the answer and written
        to a collapsed ``<details>`` block after it, so the transcript reads as
        the conversation while still preserving the reasoning instead of dumping
        the raw tags inline."""
        from localm.textnorm import split_think
        answer, reasoning = split_think(assistant)
        try:
            with open(self._path, "a", encoding="utf-8") as fh:
                if not self._wrote_header:
                    fh.write(f"# localm {self._label} transcript - "
                             f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self._wrote_header = True
                fh.write(f"\n## You\n\n{user}\n\n## Assistant\n\n{answer}\n")
                if reasoning.strip():
                    fh.write("\n<details>\n<summary>reasoning</summary>\n\n"
                             f"{reasoning.strip()}\n\n</details>\n")
        except Exception as e:
            logger.debug("MarkdownTranscript write failed: %s", e)


# ---------------------------------------------------------------------------
#  Factory
# ---------------------------------------------------------------------------

AuditLogT = Union[AuditLog, NullAuditLog]


def make_audit_log(mode: SessionMode, label: str = "") -> AuditLogT:
    """
    Return an appropriate audit log object for the given session mode.

    ``privacy`` → NullAuditLog (no disk writes)
    ``log``     → AuditLog (JSONL in <data dir>/sessions/)
    ``full``    → AuditLog (JSONL; markdown is the caller's concern)
    """
    if mode == SessionMode.PRIVACY:
        return NullAuditLog()
    return AuditLog(label=label)


def make_transcript(mode: SessionMode, label: str = "chat") -> Optional[MarkdownTranscript]:
    """MarkdownTranscript in ``full`` mode, else None."""
    if mode == SessionMode.FULL:
        return MarkdownTranscript(label=label)
    return None
