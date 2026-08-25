# SPDX-License-Identifier: AGPL-3.0-or-later
"""Coder agent sessions for the GUI."""

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
    """Seconds to wait for a confirmation before auto-rejecting it, or None to block forever. `threading.Event.wait(timeout=0)` does NOT block at all (it's a non-blocking poll), so a configured 0 - documented in settings_schema.py as 'wait forever' - must map to None, not 0.0."""
    try:
        from localm.config import load_config
        val = load_config().get("coder_confirm_timeout")
        if val is None:
            return float(_CONFIRM_TIMEOUT_S)
        val = float(val)
        return None if val == 0 else val
    except Exception:
        return float(_CONFIRM_TIMEOUT_S)

# Queue size: generous, but bounded so a disconnected client can't grow memory
# without limit. When full, oldest events are dropped (tokens are recoverable -
# the final event always carries the full text).
_QUEUE_MAX = 10_000

# How long a session may sit with no activity (no message, no token, no tool
# call/result) before SessionManager.reap_idle() closes it. Generous on
# purpose: this is cleanup for a session the user is never coming back to (a
# closed tab, a killed browser), not a working-session timeout - a real user
# mid-conversation pushes an event and resets the clock long before this.
_IDLE_REAP_SECONDS = 24 * 3600


class SessionUnavailable(RuntimeError):
    """The session refused a claim: it is already busy, or it is closed."""

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
        # holds: auto-detection assigns a list to this very field below.
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
        # Bumped on every _push() (a user message, a token, a tool call/result -
        # anything happening in this session). SessionManager.reap_idle() reads
        # this to find a session the user abandoned without DELETE (a closed
        # tab, a killed browser) - otherwise close()'s "session ended" audit
        # record is never written for it at all.
        self.last_activity_at = self.created_at
        # Caller identity that created this session (None = the owner). Set by the
        # /api/coder route so a scoped key can only see/steer the sessions IT made,
        # not the owner's full-capability sessions (session isolation).
        self.principal: Optional[str] = None
        self.restricted = restricted
        self.model = getattr(backend, "model_id", "")
        self.auto_approve = auto_approve
        self.mode = mode
        self.dry_run = dry_run
        # Auto-approve file writes but still prompt before shell execution (the
        # CLI's --interactive-confirm). It only bites WITH auto_approve, which is
        # exactly what it carves an exception out of: without it every
        # destructive tool already prompts, so there is nothing to except.
        self.interactive_confirm = interactive_confirm
        # Writes are captured as a unified diff and never reach disk. The CLI's
        # --patch-mode names an output FILE; a browser has no such thing, so the
        # web form is "accumulate, then download" - see current_patch().
        self.patch_mode = patch_mode
        # Whether the caller ASKED for the OpenAI native tools protocol, and
        # whether the connected server can actually honour it. Two fields, not
        # one, because collapsing them is exactly how a request that was quietly
        # dropped becomes indistinguishable from one that was applied.
        self.native_tools_requested = bool(getattr(backend, "native_tools", False))
        self.native_tools = (self.native_tools_requested
                             and bool(getattr(backend, "supports_native_tools", True)))
        # The last finished task's machine-readable result: the CLI's
        # --output-format json payload, kept so a client that is not holding the
        # SSE stream open can still read it (GET .../result). None until a task
        # has actually finished.
        self.last_result: Optional[dict] = None
        self.events: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        # Bounded replay buffer so a reloaded page can rebuild the feed.
        self.history: list = []
        self.busy = False
        self.closed = False
        # Tools the user marked "always allow" on an approval card - those
        # skip the confirmation flow for the rest of this session.
        self.allowed_tools: set[str] = set()
        # Keyed by confirm id, not a single slot - a turn can dispatch 2+
        # non-destructive tool calls concurrently (e.g. two fetch_url/
        # web_search calls under net_mode=ask, agent/loop.py's
        # _execute_tools parallel batch), each needing its own confirmation
        # in flight at once. A single slot would let the second call's
        # _confirm() clobber the first's, silently orphaning it until timeout.
        self._pending: dict[str, _PendingConfirm] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # The exit-code oracle for this GUI session: an explicit command, else
        # the project's detected check. A RESTRICTED session gets none - those
        # sessions have no process execution at all (SAFE_RESTRICTED_TOOLS), and
        # a verify command is process execution, so auto-enabling one would hand
        # a scoped key exactly the capability the restriction removes. The Agent
        # enforces this too (core.py), belt and braces.
        verify_cmd = None
        if not restricted:
            verify_cmd = verify or (
                self._detect_verify(cwd) if auto_verify else None)
        self.verify_cmd = verify_cmd

        # The shell-execution family keeps its confirmation even under
        # auto_approve. A RESTRICTED session never has those tools at all, so
        # the set is left empty there rather than naming tools that do not
        # exist for it.
        always_confirm: set = set()
        if interactive_confirm and not restricted:
            from localm.plugins.coder.agent.constants import _SHELL_EXEC_TOOLS
            always_confirm |= set(_SHELL_EXEC_TOOLS)

        # Record WHERE this session ran, so the rail can offer it again later.
        # record_project refuses outright for privacy mode - that refusal lives in
        # the module rather than here on purpose, so every future caller inherits it
        # instead of each one having to remember. It never raises: a convenience
        # list must not be able to stop a session starting.
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
            # (setup form) overrides it, mirroring the CLI --system flag (rec#584).
            custom_instructions=custom_instructions,
            on_event=self._on_agent_event,
            # ALWAYS, even under auto_approve. The agent only consults this when
            # its gate has already decided a call needs confirmation, and under
            # auto_approve that happens in two live cases: a tool named in
            # always_confirm (interactive_confirm puts the shell family there),
            # and a LENIENT call - one recovered by the name-gated parser
            # fallback, which auto_approve deliberately must not wave through.
            #
            # Passing None for those left a web session with no channel to ask
            # on, so the agent took its fail-closed branch and REFUSED the call
            # outright: a GUI session runs _loop(interactive=False), so there is
            # no terminal to fall back to. That is right as a default and wrong
            # here - it made the "still confirm shell commands" checkbox refuse
            # the command its own tooltip promises will stop for you, and it
            # denied the human look the lenient gate exists to demand.
            confirm_handler=self._confirm,
            verify_cmd=verify_cmd,
            verify_max_retries=verify_max_retries,
            **gen_kwargs,
        )

    @staticmethod
    def _detect_verify(cwd: Path):
        """The project's obvious check, or None."""
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
        """Agent event hook: enrich write/edit/patch tool calls with a diff preview so the GUI can render them even under auto-approve."""
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
        """Block the agent thread until the browser approves or rejects this destructive tool call."""
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
            # Absent (None) for the session's own agent, so an ordinary prompt is
            # byte-for-byte the event it always was.
            "agent": agent,
        })
        answered = pending.answered.wait(timeout=_confirm_timeout())
        with self._lock:
            self._pending.pop(pending.id, None)
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
            # Without this the GUI approval card gets diff: null and falls back
            # to a raw JSON dump of the args - the one write tool asking for
            # blind approval.
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
        """Deliver a user message."""
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
                payload = {
                    "type": "final",
                    "text": final,
                    "ok": self.agent.last_run_ok,
                    # "passed"/"failed"/"inconclusive"/null. Separate from "ok"
                    # because a check that could not run is neither: ok alone
                    # would report an unverified task as a clean finish.
                    "verify_state": self.agent.last_verify_state,
                    "turns": self.agent.turns,
                    "total_tokens": self.agent.total_tokens,
                    "changed_files": [f["path"] for f in
                                      self.agent.changed_files()],
                }
                # Latch the same numbers for GET .../result before pushing, so a
                # client woken by the event cannot race ahead of the record it
                # is about to ask for. `response` mirrors the CLI's
                # --output-format json key name; `text` stays for the SSE
                # consumers that already read it.
                self.last_result = {k: v for k, v in payload.items()
                                    if k != "type"}
                self.last_result["response"] = final
                self._push(payload)
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

    def run_estimate(self, task: str) -> dict:
        """One planning turn on this session, claimed like an ordinary task."""
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
        """Put an estimate into the session feed (and the replay buffer)."""
        self._push({
            "type": "estimate",
            "task": task,
            "text": result.get("estimate") or "",
            "prompt_tokens": result.get("prompt_tokens"),
            "total_tokens": result.get("total_tokens"),
        })

    def undo(self) -> Optional[str]:
        """Revert the last undoable file operation."""
        with self._lock:
            if self.busy:
                return None
        return self.agent.undo()

    def compact(self) -> bool:
        """Summarise old conversation history."""
        with self._lock:
            if self.busy:
                return False
        return self.agent.compact()

    # ------------------------------------------------------------------ #
    #  Mid-session settings (the REPL's /approve, /scope, /verify, /cd)    #
    # ------------------------------------------------------------------ #

    def set_auto_approve(self, value: bool) -> None:
        """Grant or REVOKE auto-approve on a live session (the REPL's /approve)."""
        self.auto_approve = bool(value)
        self.agent.auto_approve = bool(value)
        if self.agent.confirm_handler is None:
            self.agent.confirm_handler = self._confirm

    def set_scope(self, scope: Optional[str]) -> Optional[str]:
        """Set or clear the file-tool glob confinement (the REPL's /scope)."""
        scope = (scope or "").strip() or None
        had = self.agent.scope
        self.agent.scope = scope
        if scope and scope != had:
            try:
                self.agent._notify_scope_does_not_confine_shell()
            except Exception:                                      # noqa: BLE001
                # Best-effort: the notice is a courtesy on top of an enforcement
                # change that has already been applied above. Losing the notice
                # must not lose the scope.
                pass
        return scope

    def set_verify(self, command: Optional[str], *, detect: bool = False):
        """Set, re-detect, or turn off the exit-code oracle (the REPL's /verify)."""
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
        """Move a live session to another project directory (the REPL's /cd)."""
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
        """The project-memory file as this session sees it."""
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
        """Append a bullet AND refresh the system prompt (agent.remember does both)."""
        self.agent.remember(text)
        return self.memory()

    def forget(self, pattern: str) -> dict:
        """Drop matching bullets and refresh the prompt."""
        p, n = self.agent.forget(pattern)
        return {"removed": n, "had_file": p is not None, **self.memory()}

    # ------------------------------------------------------------------ #
    #  Background work started by THIS session (the REPL's /bg)            #
    # ------------------------------------------------------------------ #

    def background(self) -> dict:
        """This session's background jobs, plus what has aged out of the table."""
        from localm.plugins.coder.background import get_registry
        registry = get_registry()
        owner = getattr(self.agent, "job_owner", None)
        return {
            "jobs": registry.list_status(owner=owner),
            "dropped": registry.dropped_for(owner),
        }

    def set_model(self, model: str) -> bool:
        """Repoint this session's backend at a different model, in place - conversation history, tools and agent state are untouched (no new Agent/backend is built)."""
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
        """Save the conversation so it can be resumed later (CODER-2)."""
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

    def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Load a saved conversation back into the agent and replay a readable recap into the feed (CODER-2)."""
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
        # Live-attribute access (not a top-level import) so a test patching
        # agent.TOOL_REGISTRY is honoured, matching every read site inside the
        # agent package itself; local because sessions.py otherwise only reaches
        # into the agent package lazily, at the point of use (see __init__).
        import localm.plugins.coder.agent as _agent
        tool_names = set(_agent.TOOL_REGISTRY) - self.agent.disabled_tools
        for m in self.agent._messages:
            role = m.get("role")
            content = m.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            # Same splitter as the Markdown transcript (parser.strip_tool_calls):
            # recognises every shape parse_tool_calls does, not just the XML
            # wrapper, so a persisted ```json-fenced or bare-JSON call is removed
            # here exactly like an XML one always was, instead of surviving into
            # the recap as raw fence markers or a raw JSON blob. _calls and the
            # malformed count are discarded, same as _calls always was: this
            # recap has never synthesised a tool-name summary line, only
            # surfaced surviving prose (or dropped the row entirely).
            _calls, text, _malformed = strip_tool_calls(content, tool_names=tool_names)
            text = text.strip()
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
            # The LIVE glob, not the one passed at creation: it is settable
            # mid-session now, so a creation-time snapshot would leave a client
            # unable to show what is actually in force - and a control that can
            # set a value but never display it is worse than no control.
            "scope": self.agent.scope,
            # Whether this is a shared-key session. Already implied by the
            # background route's "supported", and needed here for the same
            # reason: a client can then SAY that changing directory and setting
            # a verification command are unavailable, instead of offering both
            # and collecting a 403 and a 409.
            "restricted": self.restricted,
            "dry_run": self.dry_run,
            "interactive_confirm": self.interactive_confirm,
            "patch_mode": self.patch_mode,
            # EFFECTIVE, not requested. `native_tools_requested` alongside it so
            # a UI can say "you asked and did not get it" rather than silently
            # showing the box unticked (AGENTS.md rule 5).
            "native_tools": self.native_tools,
            "native_tools_requested": self.native_tools_requested,
            "busy": self.busy,
            "turns": self.agent.turns,
            "total_tokens": self.agent.total_tokens,
            "created_at": self.created_at,
            "pending_confirm": bool(self._pending),
            "allowed_tools": sorted(self.allowed_tools),
            "changed_files": len(self.agent.changed_files()),
            # What this session verifies with, so the GUI can show the real gate
            # instead of leaving the user to guess whether one is running.
            "verify": (_verify_text(self.agent.verify_cmd)
                       if self.agent.verify_cmd is not None else None),
            # How many fix attempts the exit-code oracle gets before it reports
            # failure - the web equivalent of the CLI's --goal-max-iters, which
            # had no web field at all.
            "verify_max_retries": self.agent.verify_max_retries,
            # Whether there is anything to download from GET .../patch. A count
            # would be a lie about "files"; this is a plain has-it-or-not.
            "has_patch": self.patch_mode and self.agent.has_patch(),
        }

    def current_patch(self) -> str:
        """The accumulated patch-mode diff, WITHOUT consuming it."""
        return self.agent.current_patch()

    def answer_confirm(self, confirm_id: str, approved: bool,
                       always_allow: bool = False) -> bool:
        """Resolve a pending confirmation."""
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
        """Session summaries."""
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
        """Close and remove sessions that have been idle (no ``_push()`` activity, and not BUSY) for more than *max_idle_s*."""
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
