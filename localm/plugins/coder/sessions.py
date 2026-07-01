# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Coder agent sessions for the GUI.

Each session owns one Agent running in a worker thread. The agent's structured
events (tokens, tool calls, results) are pushed onto a per-session queue that
the web layer drains as an SSE stream. Destructive-tool confirmations block the
agent thread until the browser answers (or a timeout rejects them).

Everything here is plain threading - no asyncio. The web layer bridges the
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
# Overridable via the "coder_confirm_timeout" config key (seconds).
_CONFIRM_TIMEOUT_S = 600


def _confirm_timeout() -> float:
    try:
        from localm.config import load_config
        return float(load_config().get("coder_confirm_timeout")
                     or _CONFIRM_TIMEOUT_S)
    except Exception:
        return _CONFIRM_TIMEOUT_S

# Queue size: generous, but bounded so a disconnected client can't grow memory
# without limit. When full, oldest events are dropped (tokens are recoverable -
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
        scope: Optional[str] = None,
        dry_run: bool = False,
        disabled_tools: Optional[frozenset] = None,
        restricted: bool = False,
        custom_instructions: Optional[str] = None,
        **gen_kwargs,
    ) -> None:
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.audit import parse_mode

        self.id = uuid.uuid4().hex[:12]
        self.cwd = cwd
        self.created_at = time.time()
        # Caller identity that created this session (None = the owner). Set by the
        # /api/coder route so a scoped key can only see/steer the sessions IT made,
        # not the owner's full-capability sessions (session isolation).
        self.principal: Optional[str] = None
        self.restricted = restricted
        self.model = getattr(backend, "model_id", "")
        self.auto_approve = auto_approve
        self.mode = mode
        self.dry_run = dry_run
        self.events: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        # Bounded replay buffer so a reloaded page can rebuild the feed.
        self.history: list = []
        self.busy = False
        self.closed = False
        # Tools the user marked "always allow" on an approval card - those
        # skip the confirmation flow for the rest of this session.
        self.allowed_tools: set[str] = set()
        self._pending: Optional[_PendingConfirm] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        self.agent = Agent(
            backend,
            cwd=cwd,
            auto_approve=auto_approve,
            max_turns=max_turns,
            mode=parse_mode(mode),
            scope=scope,
            dry_run=dry_run,
            disabled_tools=disabled_tools,
            restricted=restricted,
            # None -> Agent reads .localcoder/system.md; a GUI-supplied string
            # (setup form) overrides it, mirroring the CLI --system flag (rec#584).
            custom_instructions=custom_instructions,
            on_event=self._on_agent_event,
            confirm_handler=None if auto_approve else self._confirm,
            **gen_kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Event plumbing                                                     #
    # ------------------------------------------------------------------ #

    def _on_agent_event(self, event: dict) -> None:
        """Agent event hook: enrich write/edit/patch tool calls with a diff
        preview so the GUI can render them even under auto-approve."""
        if event.get("type") == "tool_call" and \
                event.get("tool") in ("write_file", "edit_file", "patch_file"):
            from types import SimpleNamespace
            call = SimpleNamespace(name=event["tool"], args=event.get("args", {}))
            diff = self._diff_preview(call)
            if diff:
                event = {**event, "diff": diff}
        self._push(event)

    def _push(self, event: dict) -> None:
        """Enqueue an event, dropping the oldest when the queue is full."""
        self.history.append(event)
        if len(self.history) > _QUEUE_MAX:
            del self.history[: _QUEUE_MAX // 10]
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
        # Tools granted "always allow" earlier in the session skip the flow
        if call.name in self.allowed_tools:
            self._push({"type": "info",
                        "text": f"{call.name} auto-approved "
                                "(always-allow granted this session)"})
            return True
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
        answered = pending.answered.wait(timeout=_confirm_timeout())
        with self._lock:
            self._pending = None
        # Always record the outcome in the event stream. Without this, a
        # reloaded page replays the confirm_request and shows live
        # approve/reject buttons for a confirmation that was already answered
        # (or timed out, or was force-rejected by stop()).
        self._push({
            "type": "confirm_resolved",
            "confirm_id": pending.id,
            "tool": pending.tool,
            "approved": pending.approved if answered else False,
            "timed_out": not answered,
        })
        if not answered:
            self._push({"type": "info",
                        "text": f"Confirmation for {call.name} timed out - rejected."})
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

    def send_message(self, text: str, _echo: bool = True) -> str:
        """
        Deliver a user message.

        Returns "started" when a new task begins, "queued" when the agent is
        mid-task (the message is injected at the next turn boundary as a
        steering note), or "closed" when the session is gone.
        """
        with self._lock:
            if self.closed:
                return "closed"
            if self.busy:
                # Mid-task steering: hand the text to the running agent and
                # record it in the feed so every tab sees it was delivered.
                self.agent.queue_message(text)
                self._push({"type": "user", "text": text, "queued": True})
                return "queued"
            self.busy = True

        # In the event stream (and replay buffer) so reloaded pages see it too
        if _echo:
            self._push({"type": "user", "text": text})

        def _run():
            try:
                final = self.agent.run_task(text)
                self._push({
                    "type": "final",
                    "text": final,
                    "ok": self.agent.last_run_ok,
                    "turns": self.agent.turns,
                    "total_tokens": self.agent.total_tokens,
                    "changed_files": [f["path"] for f in
                                      self.agent.changed_files()],
                })
            except Exception as e:
                self._push({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                with self._lock:
                    self.busy = False
                # Save the conversation so it can be resumed later (CODER-2). The
                # agent clears the checkpoint on a clean finish, so re-persist the
                # current state here after every task (no-op in privacy mode).
                self.persist_checkpoint()
                # A message queued in the task's final moments would otherwise
                # sit until the user sends again - run it as a follow-up task.
                leftover = self.agent._drain_queued()
                if leftover and not self.closed:
                    self._push({"type": "info",
                                "text": "running queued message(s) as a follow-up"})
                    self.send_message("\n\n".join(leftover), _echo=False)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return "started"

    def undo(self) -> Optional[str]:
        """Revert the last undoable file operation. None when nothing to undo."""
        with self._lock:
            if self.busy:
                return None
        return self.agent.undo()

    def compact(self) -> bool:
        """Summarise old conversation history. False when nothing to compact
        or the agent is mid-task."""
        with self._lock:
            if self.busy:
                return False
        return self.agent.compact()

    def audit_log_path(self) -> Optional[Path]:
        """Path of the JSONL audit log (log/full modes), or None in privacy mode."""
        return getattr(self.agent._audit, "path", None)

    def changed_files(self) -> list:
        """Files the agent has changed this session (safe to call mid-task)."""
        return self.agent.changed_files()

    def session_diff(self, path: Optional[str] = None) -> str:
        """Cumulative diff of the session's changes (all files or one)."""
        return self.agent.session_diff(path)

    def persist_checkpoint(self) -> None:
        """Save the conversation so it can be resumed later (CODER-2). The agent
        no-ops in privacy mode and on an empty conversation, so this is safe to
        call after every task and on close; it never raises."""
        # Restricted (scoped-key) sessions are ephemeral and cannot be resumed
        # (resume is owner-only), and they all share the forced project-root cwd -
        # persisting them would clobber the OWNER's checkpoint for that root. Skip.
        if self.restricted:
            return
        try:
            if self.agent._messages:
                self.agent.save_checkpoint()
        except Exception:
            pass

    def resume_from_checkpoint(self) -> bool:
        """Load this cwd's saved conversation back into the agent and replay a
        readable recap into the feed (CODER-2). The model gets the FULL restored
        history; the feed rows are a visual summary. True when something was
        restored. Tool-call markup is stripped from the recap, and tool-result
        envelopes / steering notes are skipped."""
        import re
        try:
            data = self.agent.load_checkpoint()
        except Exception:
            data = None
        if not data:
            return False
        self.agent.resume_checkpoint(data)
        when = data.get("interrupted_at") or "earlier"
        self._push({"type": "info",
                    "text": f"Resumed your last session here (saved {when}, "
                            f"{data.get('turns', 0)} turns). Continue where you "
                            "left off."})
        for m in self.agent._messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            text = re.sub(r"<tool_call>.*?</tool_call>", "", content,
                          flags=re.DOTALL).strip()
            if not text or text.startswith("<tool_result") \
                    or "[user steering note" in text:
                continue
            self._push({"type": "history", "role": role, "text": text[:4000]})
        return True

    def info(self) -> dict:
        """Summary dict for the session list endpoint."""
        return {
            "id": self.id,
            "cwd": str(self.cwd),
            "model": self.model,
            "mode": self.mode,
            "auto_approve": self.auto_approve,
            "dry_run": self.dry_run,
            "busy": self.busy,
            "turns": self.agent.turns,
            "total_tokens": self.agent.total_tokens,
            "created_at": self.created_at,
            "pending_confirm": self._pending is not None,
            "allowed_tools": sorted(self.allowed_tools),
            "changed_files": len(self.agent.changed_files()),
        }

    def answer_confirm(self, confirm_id: str, approved: bool,
                       always_allow: bool = False) -> bool:
        """Resolve a pending confirmation. False if id doesn't match.

        ``always_allow`` (only honoured on approval) whitelists the tool for
        the rest of the session - later calls skip the confirmation flow."""
        with self._lock:
            pending = self._pending
        if pending is None or pending.id != confirm_id:
            return False
        if approved and always_allow:
            self.allowed_tools.add(pending.tool)
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
        # Save the conversation first so "Continue last session" works after a
        # graceful close or a server shutdown (CODER-2). Best-effort.
        self.persist_checkpoint()
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

    def list(self, *, principal: Optional[str] = None, is_owner: bool = True) -> list:
        """Session summaries. The owner sees all; a scoped caller (is_owner=False)
        sees only the sessions matching its *principal* - so a handed-out key
        cannot enumerate the owner's sessions."""
        with self._lock:
            sessions = list(self._sessions.values())
        if not is_owner:
            sessions = [s for s in sessions if s.principal == principal]
        return [s.info() for s in sorted(sessions, key=lambda s: s.created_at)]

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
