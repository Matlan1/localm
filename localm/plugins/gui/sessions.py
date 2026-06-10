"""
Coder agent sessions for the GUI.

Each session owns one Agent running in a worker thread. The agent's structured
events (tokens, tool calls, results) are pushed onto a per-session queue that
the web layer drains as an SSE stream. Destructive-tool confirmations block the
agent thread until the browser answers (or a timeout rejects them).

Everything here is plain threading — no asyncio. The web layer bridges the
queue into the event loop.
"""

from __future__ import annotations

import difflib
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# How long a confirmation may sit unanswered before it is auto-rejected.
_CONFIRM_TIMEOUT_S = 600

# Queue size: generous, but bounded so a disconnected client can't grow memory
# without limit. When full, oldest events are dropped (tokens are recoverable —
# the final event always carries the full text).
_QUEUE_MAX = 10_000


@dataclass
class _PendingConfirm:
    id: str
    tool: str
    args: dict
    diff: Optional[str]
    answered: threading.Event = field(default_factory=threading.Event)
    approved: bool = False


class CoderSession:
    """One GUI coder session: an Agent, its event queue, and its worker thread."""

    def __init__(
        self,
        cwd: Path,
        backend,
        *,
        auto_approve: bool = False,
        max_turns: int = 40,
        mode: str = "privacy",
        **gen_kwargs,
    ) -> None:
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.audit import parse_mode

        self.id = uuid.uuid4().hex[:12]
        self.cwd = cwd
        self.created_at = time.time()
        self.events: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        self.busy = False
        self.closed = False
        self._pending: Optional[_PendingConfirm] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self.agent = Agent(
            backend,
            cwd=cwd,
            auto_approve=auto_approve,
            max_turns=max_turns,
            mode=parse_mode(mode),
            on_event=self._push,
            confirm_handler=None if auto_approve else self._confirm,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Event plumbing                                                     #
    # ------------------------------------------------------------------ #

    def _push(self, event: dict) -> None:
        """Enqueue an event, dropping the oldest when the queue is full."""
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _confirm(self, call) -> bool:
        """
        Block the agent thread until the browser approves or rejects this
        destructive tool call. Sends a confirm_request event with a diff
        preview for file-writing tools.
        """
        pending = _PendingConfirm(
            id=uuid.uuid4().hex[:8],
            tool=call.name,
            args=dict(call.args),
            diff=self._diff_preview(call),
        )
        with self._lock:
            self._pending = pending
        self._push({
            "type": "confirm_request",
            "confirm_id": pending.id,
            "tool": pending.tool,
            "args": pending.args,
            "diff": pending.diff,
        })
        answered = pending.answered.wait(timeout=_CONFIRM_TIMEOUT_S)
        with self._lock:
            self._pending = None
        if not answered:
            self._push({"type": "info",
                        "text": f"Confirmation for {call.name} timed out — rejected."})
            return False
        return pending.approved

    def _diff_preview(self, call) -> Optional[str]:
        """Unified diff of what a write/edit/patch call would change, or None."""
        try:
            if call.name == "patch_file":
                return call.args.get("diff") or None
            path_arg = call.args.get("path", "")
            if not path_arg or call.name not in ("write_file", "edit_file"):
                return None
            abs_path = (self.cwd / path_arg).resolve()
            old_text = ""
            if abs_path.is_file():
                old_text = abs_path.read_text(encoding="utf-8", errors="replace")
            if call.name == "write_file":
                new_text = call.args.get("content", "")
            else:
                new_text = old_text.replace(
                    call.args.get("old", ""), call.args.get("new", ""), 1)
            diff = "".join(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{path_arg}", tofile=f"b/{path_arg}",
            ))
            return diff or None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Public API (called from web handlers)                              #
    # ------------------------------------------------------------------ #

    def send_message(self, text: str) -> bool:
        """Start a task in the worker thread. False if the agent is busy."""
        with self._lock:
            if self.busy or self.closed:
                return False
            self.busy = True

        def _run():
            try:
                final = self.agent.run_task(text)
                self._push({
                    "type": "final",
                    "text": final,
                    "ok": self.agent.last_run_ok,
                    "turns": self.agent.turns,
                    "total_tokens": self.agent.total_tokens,
                })
            except Exception as e:
                self._push({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                with self._lock:
                    self.busy = False

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return True

    def answer_confirm(self, confirm_id: str, approved: bool) -> bool:
        """Resolve a pending confirmation. False if id doesn't match."""
        with self._lock:
            pending = self._pending
        if pending is None or pending.id != confirm_id:
            return False
        pending.approved = approved
        pending.answered.set()
        return True

    def stop(self) -> None:
        """Request the agent to stop at the next safe point; unblock confirms."""
        self.agent.request_stop()
        with self._lock:
            pending = self._pending
        if pending is not None:
            pending.approved = False
            pending.answered.set()

    def close(self) -> None:
        """Terminate the session: stop the agent and poison the event queue."""
        self.closed = True
        self.stop()
        self._push({"type": "closed"})
        try:
            self.agent.close()
        except Exception:
            pass


class SessionManager:
    """Registry of live coder sessions, keyed by id."""

    def __init__(self) -> None:
        self._sessions: dict[str, CoderSession] = {}
        self._lock = threading.Lock()

    def create(self, session: CoderSession) -> CoderSession:
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[CoderSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> Optional[CoderSession]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return session

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()
