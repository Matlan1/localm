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

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .diffutil import compute_multifile_diff, compute_tool_diff, read_old_content
from .parser import strip_tool_calls
from .verify import VerifyCommand as _VerifyCommand, command_text as _verify_text

# How long a confirmation may sit unanswered before it is auto-rejected.
# Overridable via the "coder_confirm_timeout" config key (seconds).
_CONFIRM_TIMEOUT_S = 600


def _confirm_timeout() -> Optional[float]:
    """Seconds to wait for a confirmation before auto-rejecting it, or None
    to block forever. `threading.Event.wait(timeout=0)` does NOT block at
    all (it's a non-blocking poll), so a configured 0 - documented in
    settings_schema.py as "wait forever" - must map to None, not 0.0."""
    try:
        from localm.config import load_config
        val = load_config().get("coder_confirm_timeout")
        if val is None:
            return float(_CONFIRM_TIMEOUT_S)
        val = float(val)
        return None if val == 0 else val
    except Exception:
        return float(_CONFIRM_TIMEOUT_S)

# Queue size, bounded so a disconnected client cannot grow memory without limit.
# When full the oldest events are dropped; the final event always carries the
# full text.
_QUEUE_MAX = 10_000

# How long a session may sit with no activity (no message, no token, no tool
# call or result) before SessionManager.reap_idle() closes it.
_IDLE_REAP_SECONDS = 24 * 3600


class SessionUnavailable(RuntimeError):
    """The session refused a claim: it is already busy, or it is closed.

    A dedicated type, so a caller can tell a scheduling conflict ("try again in
    a moment", 409) apart from a failed model call ("something is broken", 502).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason          # "busy" | "closed"


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
        interactive_confirm: bool = False,
        patch_mode: bool = False,
        disabled_tools: Optional[frozenset] = None,
        restricted: bool = False,
        custom_instructions: Optional[str] = None,
        # A shell string OR an argv list, the same union the agent's verify_cmd
        # holds; auto-detection assigns a list to this field below.
        verify: Optional["_VerifyCommand"] = None,
        auto_verify: bool = True,
        verify_max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        from localm.plugins.coder.agent import Agent
        from localm.plugins.coder.audit import parse_mode

        self.id = uuid.uuid4().hex[:12]
        self.cwd = cwd
        self.created_at = time.time()
        # Bumped on every _push() (a user message, a token, a tool call or result).
        # SessionManager.reap_idle() reads it to find a session abandoned without
        # DELETE.
        self.last_activity_at = self.created_at
        # Caller identity that created this session (None = the owner). Set by the
        # /api/coder route, so a scoped key only sees and steers its own sessions.
        self.principal: Optional[str] = None
        self.restricted = restricted
        self.model = getattr(backend, "model_id", "")
        self.auto_approve = auto_approve
        self.mode = mode
        self.dry_run = dry_run
        # Auto-approve file writes but still prompt before shell execution. Only
        # has an effect together with auto_approve.
        self.interactive_confirm = interactive_confirm
        # Writes are captured as a unified diff and never reach disk; the diff
        # accumulates for download (see current_patch()).
        self.patch_mode = patch_mode
        # Whether the caller ASKED for the OpenAI native tools protocol, and
        # whether the connected server can honour it. Two fields, not one, so a
        # dropped request stays distinguishable from an applied one.
        self.native_tools_requested = bool(getattr(backend, "native_tools", False))
        self.native_tools = (self.native_tools_requested
                             and bool(getattr(backend, "supports_native_tools", True)))
        # The last finished task's machine-readable result, readable by a client
        # that is not holding the SSE stream open (GET .../result). None until a
        # task has finished.
        self.last_result: Optional[dict] = None
        self.events: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        # Bounded replay buffer so a reloaded page can rebuild the feed.
        self.history: list = []
        self.busy = False
        self.closed = False
        # Tools the user marked "always allow" on an approval card; they skip the
        # confirmation flow for the rest of this session.
        self.allowed_tools: set[str] = set()
        # Keyed by confirm id, not a single slot: a turn can dispatch two or more
        # non-destructive tool calls concurrently, each needing its own
        # confirmation in flight at once.
        self._pending: dict[str, _PendingConfirm] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # The exit-code oracle for this GUI session: an explicit command, else the
        # project's detected check. A RESTRICTED session gets none, because a
        # verify command is process execution and those sessions have none. The
        # Agent enforces this too.
        verify_cmd = None
        if not restricted:
            verify_cmd = verify or (
                self._detect_verify(cwd) if auto_verify else None)
        self.verify_cmd = verify_cmd

        # The shell-execution family keeps its confirmation even under
        # auto_approve. A RESTRICTED session has no such tools, so the set stays
        # empty there.
        always_confirm: set = set()
        if interactive_confirm and not restricted:
            from localm.plugins.coder.agent.constants import _SHELL_EXEC_TOOLS
            always_confirm |= set(_SHELL_EXEC_TOOLS)

        # Record WHERE this session ran, so the rail can offer it again later.
        # record_project refuses for privacy mode and never raises.
        try:
            from localm.plugins.coder.projects import record_project
            record_project(cwd, mode)
        except Exception as _e:   # pragma: no cover - defensive, see above
            from localm.debuglog import logger
            logger.debug("coder projects: not recorded (%s)", _e)

        self.agent = Agent(
            backend,
            cwd=cwd,
            auto_approve=auto_approve,
            always_confirm=always_confirm,
            max_turns=max_turns,
            mode=parse_mode(mode),
            scope=scope,
            dry_run=dry_run,
            patch_mode=patch_mode,
            disabled_tools=disabled_tools,
            restricted=restricted,
            # None -> Agent reads .localcoder/system.md; a GUI-supplied string
            # overrides it.
            custom_instructions=custom_instructions,
            on_event=self._on_agent_event,
            # Passed ALWAYS, even under auto_approve: the agent consults it only
            # once its gate has decided a call needs confirmation, which under
            # auto_approve still happens for a tool named in always_confirm and
            # for a LENIENT call recovered by the name-gated parser fallback.
            # A GUI session runs _loop(interactive=False), so None here would make
            # the agent fail closed and refuse those calls.
            confirm_handler=self._confirm,
            verify_cmd=verify_cmd,
            verify_max_retries=verify_max_retries,
            **gen_kwargs,
        )

    @staticmethod
    def _detect_verify(cwd: Path):
        """The project's obvious check, or None. Best-effort: a detection
        failure means no oracle for this session, never a failed session, and is
        surfaced as a warning in the log."""
        try:
            from .verify import detect_verify_command
            return detect_verify_command(cwd)
        except Exception:                                      # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning(
                "coder: could not detect a project check for %s; this session "
                "runs without exit-code verification", cwd, exc_info=True)
            return None

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
        self.last_activity_at = time.time()
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

    def _confirm(self, call, agent: Optional[str] = None) -> bool:
        """
        Block the agent thread until the browser approves or rejects this
        destructive tool call. Sends a confirm_request event with a diff
        preview for file-writing tools.

        *agent* names the sub-agent asking, or is None when this session's own
        agent is. It is the optional attribution keyword of the confirm-handler
        protocol (coder/confirm.py) and reaches the browser on the event, which
        is the only place that attribution appears: concurrent sub-agents share
        this one channel, and a child's own tool_call events are not forwarded
        here (children do not inherit on_event).
        """
        # Tools granted "always allow" earlier in the session skip the flow
        if call.name in self.allowed_tools:
            who = f"sub-agent '{agent}': " if agent else ""
            self._push({"type": "info",
                        "text": f"{who}{call.name} auto-approved "
                                "(always-allow granted this session)"})
            return True
        pending = _PendingConfirm(
            id=uuid.uuid4().hex[:8],
            tool=call.name,
            args=dict(call.args),
            diff=self._diff_preview(call),
        )
        with self._lock:
            self._pending[pending.id] = pending
        self._push({
            "type": "confirm_request",
            "confirm_id": pending.id,
            "tool": pending.tool,
            "args": pending.args,
            "diff": pending.diff,
            # Absent (None) for the session's own agent.
            "agent": agent,
        })
        answered = pending.answered.wait(timeout=_confirm_timeout())
        with self._lock:
            self._pending.pop(pending.id, None)
        # Always record the outcome in the event stream, so a reloaded page does
        # not replay the confirm_request with live approve/reject buttons.
        self._push({
            "type": "confirm_resolved",
            "confirm_id": pending.id,
            "tool": pending.tool,
            "approved": pending.approved if answered else False,
            "timed_out": not answered,
        })
        if not answered:
            who = f" (sub-agent '{agent}')" if agent else ""
            self._push({"type": "info",
                        "text": f"Confirmation for {call.name}{who} timed out "
                                "- rejected."})
            return False
        return pending.approved

    def _diff_preview(self, call) -> Optional[str]:
        """Unified diff of what a write/edit/patch call would change, or None."""
        try:
            if call.name == "patch_file":
                return compute_tool_diff(call.name, call.args, "")
            # edit_files spans several files, so it has no single old_content.
            if call.name == "edit_files":
                return compute_multifile_diff(self.cwd, call.args.get("edits"))
            path_arg = call.args.get("path", "")
            if not path_arg or call.name not in ("write_file", "edit_file"):
                return None
            old_text = read_old_content(self.cwd, path_arg)
            return compute_tool_diff(call.name, call.args, old_text)
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
                # Mid-task steering: hand the text to the running agent and record
                # it in the feed.
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
                payload = {
                    "type": "final",
                    "text": final,
                    "ok": self.agent.last_run_ok,
                    # "passed"/"failed"/"inconclusive"/null. Separate from "ok":
                    # a check that could not run is neither.
                    "verify_state": self.agent.last_verify_state,
                    "turns": self.agent.turns,
                    "total_tokens": self.agent.total_tokens,
                    "changed_files": [f["path"] for f in
                                      self.agent.changed_files()],
                }
                # Latch the same numbers for GET .../result before pushing, so a
                # client woken by the event cannot race ahead of the record it is
                # about to ask for. response and text both carry the final text.
                self.last_result = {k: v for k, v in payload.items()
                                    if k != "type"}
                self.last_result["response"] = final
                self._push(payload)
            except Exception as e:
                self._push({"type": "error", "text": f"{type(e).__name__}: {e}"})
            finally:
                with self._lock:
                    self.busy = False
                # Save the conversation so it can be resumed later. The agent
                # clears the checkpoint on a clean finish, so re-persist the
                # current state after every task. No-op in privacy mode.
                self.persist_checkpoint()
                # Run a message queued in the task's final moments as a follow-up
                # task.
                leftover = self.agent._drain_queued()
                if leftover and not self.closed:
                    self._push({"type": "info",
                                "text": "running queued message(s) as a follow-up"})
                    self.send_message("\n\n".join(leftover), _echo=False)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return "started"

    def run_estimate(self, task: str) -> dict:
        """One planning turn on this session, claimed like an ordinary task.

        Estimating is a second trigger for the shared backend alongside
        ``send_message``, and two concurrent turns would interleave on its
        per-call state - ``last_usage`` in particular, which this function reads
        after its own call returns. ``busy`` is taken under the same lock
        ``send_message`` uses, so the two are mutually exclusive, and a caller
        that loses the race raises rather than queueing.

        The drain in the ``finally`` re-runs a message queued while the estimate
        held ``busy``. No checkpoint is persisted: an estimate changes no
        conversation.
        """
        from .estimate import estimate_task
        with self._lock:
            if self.closed:
                raise SessionUnavailable("closed")
            if self.busy:
                raise SessionUnavailable("busy")
            self.busy = True
        try:
            result = estimate_task(self.agent, task)
        finally:
            with self._lock:
                self.busy = False
            leftover = self.agent._drain_queued()
            if leftover and not self.closed:
                self._push({"type": "info",
                            "text": "running queued message(s) as a follow-up"})
                self.send_message("\n\n".join(leftover), _echo=False)
        self.push_estimate(task, result)
        return result

    def push_estimate(self, task: str, result: dict) -> None:
        """Put an estimate into the session feed (and the replay buffer).

        A dedicated event type, not an ``info`` line, so a consumer can tell a
        plan that was never executed apart from a turn that was. It carries the
        task it answered, which the composer no longer shows by replay time."""
        self._push({
            "type": "estimate",
            "task": task,
            "text": result.get("estimate") or "",
            "prompt_tokens": result.get("prompt_tokens"),
            "total_tokens": result.get("total_tokens"),
        })

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

    # ------------------------------------------------------------------ #
    #  Mid-session settings (the REPL's /approve, /scope, /verify, /cd)    #
    # ------------------------------------------------------------------ #

    def set_auto_approve(self, value: bool) -> None:
        """Grant or REVOKE auto-approve on a live session (the REPL's /approve).

        NOT busy-guarded, unlike set_model(): revoking must work while the agent
        is mid-run. The writes are plain attribute assignments, read at the next
        tool dispatch, so there is nothing to tear.

        Also installs the confirm handler when the agent has none. A GUI session
        runs _loop(interactive=False), so a destructive tool that needs
        confirmation with confirm_handler None takes the fail-closed branch and
        DENIES the call outright, leaving a revoked session unable to do
        anything. __init__ installs the handler for every session, so this
        branch only covers a CoderSession whose agent was built or swapped some
        other way.
        """
        self.auto_approve = bool(value)
        self.agent.auto_approve = bool(value)
        if self.agent.confirm_handler is None:
            self.agent.confirm_handler = self._confirm

    def set_scope(self, scope: Optional[str]) -> Optional[str]:
        """Set or clear the file-tool glob confinement (the REPL's /scope).

        Not busy-guarded, like set_auto_approve: tightening a scope mid-run is a
        safety action. The scope never enters the system prompt
        (build_system_prompt takes no scope argument), so there is no prompt to
        rebuild.

        Newly setting a scope re-emits the "a scope does not confine the shell
        tools" notice.
        """
        scope = (scope or "").strip() or None
        had = self.agent.scope
        self.agent.scope = scope
        if scope and scope != had:
            try:
                self.agent._notify_scope_does_not_confine_shell()
            except Exception:                                      # noqa: BLE001
                # Best-effort: the notice is a courtesy on top of the scope change
                # already applied above.
                pass
        return scope

    def set_verify(self, command: Optional[str], *, detect: bool = False):
        """Set, re-detect, or turn off the exit-code oracle (the REPL's /verify).

        Returns the new command (a shell string or an argv list), or None for
        off. Raises SessionUnavailable for a RESTRICTED session: those have no
        process execution at all (SAFE_RESTRICTED_TOOLS), and a verify command
        IS process execution. __init__ refuses it at creation too.
        """
        if self.restricted:
            raise SessionUnavailable(
                "A restricted session runs no commands, so it has no "
                "verification check to set.")
        if detect:
            new = self._detect_verify(self.cwd)
        elif command is None:
            new = None
        else:
            new = command.strip() or None
        self.agent.verify_cmd = new
        self.verify_cmd = new
        return new

    def set_cwd(self, cwd: Path) -> bool:
        """Move a live session to another project directory (the REPL's /cd).

        False when the agent is mid-task. Busy-guarded where the settings above
        are not: this rebuilds the project map AND the system prompt.

        The CALLER owns the UNC/device refusal and running this off the event
        loop - it scans the project tree.
        """
        with self._lock:
            if self.busy:
                return False
        self.agent.set_cwd(cwd)
        self.cwd = cwd
        return True

    # ------------------------------------------------------------------ #
    #  Project memory (the REPL's /memory, /remember, /forget)             #
    # ------------------------------------------------------------------ #

    def memory(self) -> dict:
        """The project-memory file as this session sees it.

        TWO texts, not one, because they legitimately differ and collapsing them
        hides which you are looking at: ``text`` is what is on disk, ``injected``
        is what the model actually reads (capped at the injection budget, with a
        visible notice appended when the file was cut). ``warning`` carries the
        user-facing form of that cap plus the unreadable-file case, so a memory
        that could not be read never looks like a memory that is simply empty.
        """
        from localm.plugins.coder.memory import find_memory_file, memory_warning
        p = find_memory_file(self.cwd)
        text = ""
        unreadable = False
        if p is not None:
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                unreadable = True
        return {
            "path": str(p) if p is not None else None,
            "exists": p is not None,
            "unreadable": unreadable,
            "text": text,
            "injected": self.agent._memory,
            "warning": memory_warning(self.cwd) or None,
        }

    def remember(self, text: str) -> dict:
        """Append a bullet AND refresh the system prompt (agent.remember does
        both).

        Goes through the agent rather than writing the file directly: a plain
        file edit never calls reload_memory, so it only takes effect next
        session.
        """
        self.agent.remember(text)
        return self.memory()

    def forget(self, pattern: str) -> dict:
        """Drop matching bullets and refresh the prompt.

        ``removed`` and ``had_file`` are reported separately from the resulting
        memory so "there is no memory file" and "no entry matched" stay
        distinguishable - they are different situations calling for different
        next steps, and one number cannot say which happened.
        """
        p, n = self.agent.forget(pattern)
        return {"removed": n, "had_file": p is not None, **self.memory()}

    # ------------------------------------------------------------------ #
    #  Background work started by THIS session (the REPL's /bg)            #
    # ------------------------------------------------------------------ #

    def background(self) -> dict:
        """This session's background jobs, plus what has aged out of the table.

        Filtered by this session's own job-owner id, never the whole registry:
        get_registry() is PROCESS-WIDE, and job labels are full command lines.
        The CLI passes no owner and lists everything, which for a one-session
        process is the same answer.

        ``dropped`` reports what the bounded table evicted, so a lost sub-agent
        completion is visible rather than silently truncated.
        """
        from localm.plugins.coder.background import get_registry
        registry = get_registry()
        owner = getattr(self.agent, "job_owner", None)
        return {
            "jobs": registry.list_status(owner=owner),
            "dropped": registry.dropped_for(owner),
        }

    def set_model(self, model: str) -> bool:
        """Repoint this session's backend at a different model, in place -
        conversation history, tools and agent state are untouched (no new
        Agent/backend is built). The backend otherwise keeps sending the
        ORIGINAL model name on every request (see HTTPBackend._body()), so a
        model switch elsewhere in the app never reaches a running session.

        False when the agent is mid-task (the same busy guard undo() and
        compact() use) or the backend does not support being repointed (only
        HTTPBackend does today; a backend without set_model degrades to "not
        supported" rather than an AttributeError)."""
        with self._lock:
            if self.busy:
                return False
        set_model_fn = getattr(self.agent.backend, "set_model", None)
        if set_model_fn is None:
            return False
        set_model_fn(model)
        self.model = model          # keep info() truthful - see its docstring
        return True

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
        # Restricted (scoped-key) sessions are ephemeral, cannot be resumed, and
        # share the forced project-root cwd, so they are never persisted.
        if self.restricted:
            return
        try:
            if self.agent._messages:
                self.agent.save_checkpoint()
        except Exception:
            pass

    def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Load a saved conversation back into the agent and replay a readable
        recap into the feed. The model gets the FULL restored history; the feed
        rows are a visual summary. True when something was restored. Tool-call
        markup is stripped from the recap, and tool-result envelopes and
        steering notes are skipped.

        *checkpoint_id* is None (the default) to restore this cwd's MOST RECENT
        conversation. Pass an id from ``list_checkpoints()`` to continue one
        PARTICULAR past session.

        An id that no longer resolves returns False; it never falls back to the
        newest conversation."""
        try:
            data = self.agent.load_checkpoint(checkpoint_id)
        except Exception:
            data = None
        if not data:
            return False
        self.agent.resume_checkpoint(data)
        when = data.get("interrupted_at") or "earlier"
        which = ("your last session here" if checkpoint_id is None
                 else f"“{data.get('title') or 'a past session'}”")
        self._push({"type": "info",
                    "text": f"Resumed {which} (saved {when}, "
                            f"{data.get('turns', 0)} turns). Continue where you "
                            "left off."})
        # Live-attribute access, not a top-level import, so a test patching
        # agent.TOOL_REGISTRY is honoured. Local, because sessions.py reaches into
        # the agent package lazily, at the point of use.
        import localm.plugins.coder.agent as _agent
        tool_names = set(_agent.TOOL_REGISTRY) - self.agent.disabled_tools
        for m in self.agent._messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            # Same splitter as the Markdown transcript: recognises every shape
            # parse_tool_calls does, not just the XML wrapper. The calls and the
            # malformed count are discarded; this recap only surfaces surviving
            # prose.
            _calls, text, _malformed = strip_tool_calls(content, tool_names=tool_names)
            text = text.strip()
            if not text or text.startswith("<tool_result") \
                    or "[user steering note" in text:
                continue
            self._push({"type": "history", "role": role, "text": text[:4000]})
        return True

    def info(self) -> dict:
        """Summary dict for the session list endpoint.

        "model" is whatever set_model() last set (or the creation-time model,
        if it was never called) - always the name this session's NEXT request
        actually uses, not a snapshot that can go stale (see set_model())."""
        return {
            "id": self.id,
            "cwd": str(self.cwd),
            "model": self.model,
            "mode": self.mode,
            "auto_approve": self.auto_approve,
            # The LIVE glob, not the one passed at creation: it is settable
            # mid-session.
            "scope": self.agent.scope,
            # Whether this is a shared-key session: changing directory and setting
            # a verification command are unavailable on one.
            "restricted": self.restricted,
            "dry_run": self.dry_run,
            "interactive_confirm": self.interactive_confirm,
            "patch_mode": self.patch_mode,
            # EFFECTIVE, not requested; native_tools_requested sits alongside it.
            "native_tools": self.native_tools,
            "native_tools_requested": self.native_tools_requested,
            "busy": self.busy,
            "turns": self.agent.turns,
            "total_tokens": self.agent.total_tokens,
            "created_at": self.created_at,
            "pending_confirm": bool(self._pending),
            "allowed_tools": sorted(self.allowed_tools),
            "changed_files": len(self.agent.changed_files()),
            # What this session verifies with.
            "verify": (_verify_text(self.agent.verify_cmd)
                       if self.agent.verify_cmd is not None else None),
            # How many fix attempts the exit-code oracle gets before it reports
            # failure.
            "verify_max_retries": self.agent.verify_max_retries,
            # Whether there is anything to download from GET .../patch.
            "has_patch": self.patch_mode and self.agent.has_patch(),
        }

    def current_patch(self) -> str:
        """The accumulated patch-mode diff, WITHOUT consuming it.

        Never ``agent.flush_patch()``: that clears the buffer, so a second
        reader (a reloaded tab, a retry, a status poll) would get an empty
        patch and no error - the "a call made just to check is still a call"
        shape. Draining is a deliberate act and this is not it."""
        return self.agent.current_patch()

    def answer_confirm(self, confirm_id: str, approved: bool,
                       always_allow: bool = False) -> bool:
        """Resolve a pending confirmation. False if id doesn't match.

        ``always_allow`` (only honoured on approval) whitelists the tool for
        the rest of the session - later calls skip the confirmation flow."""
        with self._lock:
            pending = self._pending.get(confirm_id)
        if pending is None:
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
            pendings = list(self._pending.values())
        for pending in pendings:
            pending.approved = False
            pending.answered.set()

    def close(self) -> None:
        """Terminate the session: stop the agent and poison the event queue."""
        # Save the conversation first, best-effort, so it can be resumed later.
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
        self.reap_idle()
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

    def reap_idle(self, *, max_idle_s: float = _IDLE_REAP_SECONDS,
                 now: Optional[float] = None) -> list[str]:
        """Close and remove sessions that have been idle (no ``_push()``
        activity, and not BUSY) for more than *max_idle_s*. Returns the ids
        reaped.

        Reaping runs close(), which writes the "session ended" audit record
        (audit.py). Called opportunistically from list(), which the GUI already
        polls to render the sidebar, so it needs no thread or shutdown hook.

        A BUSY session is never reaped regardless of *last_activity_at*: a
        long-running task may not have pushed a new event recently."""
        now = time.time() if now is None else now
        with self._lock:
            idle_ids = [s.id for s in self._sessions.values()
                       if not s.busy and (now - s.last_activity_at) > max_idle_s]
        reaped = []
        for session_id in idle_ids:
            if self.remove(session_id) is not None:
                reaped.append(session_id)
        return reaped

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            s.close()
