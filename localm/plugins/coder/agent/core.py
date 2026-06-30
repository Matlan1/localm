# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Agent - the core agentic loop.

Flow per turn:
    1. Call the LLM with the current message history
    2. Parse the response for <tool_call> blocks
    3. If no tool calls -> final answer, break
    4. For each tool call: display, optionally confirm, execute, append result
    5. Repeat

The Agent class is assembled here from the concern mixins (loop / execution /
context / persistence / session); this module owns construction (__init__) and
the small shared-state accessors. The Agent class is used by the CLI (interactive
chat + single-task run_task) and by the spawn_agent tool (child agents).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import localm.plugins.coder.agent as _agent
from ..backends.base import BaseLLMBackend
from ..indexer import ProjectMap
from ..tools import SAFE_RESTRICTED_TOOLS
from ..audit import AuditLogT, SessionMode
from .loop import _LoopMixin
from .execution import _ExecutionMixin
from .context import _ContextMixin
from .persistence import _PersistenceMixin
from .session import _SessionMixin
from .tooldefs import _build_openai_tool_defs


class Agent(
    _LoopMixin,
    _ExecutionMixin,
    _ContextMixin,
    _PersistenceMixin,
    _SessionMixin,
):
    """
    Stateful agentic loop.

    Parameters
    ----------
    backend:
        LLM backend (local or remote).
    cwd:
        Working directory for all file/shell operations.
    name:
        Display name (shown in the terminal and sub-agent logs).
    max_turns:
        Hard cap on LLM calls per task to prevent infinite loops.
    verbose:
        Print full tool outputs (not just summaries).
    auto_approve:
        Skip confirmation prompts for destructive tools.
    always_confirm:
        Set of tool names that always prompt for confirmation, even when
        ``auto_approve=True``.  Typical use: ``{"run_shell"}`` to auto-approve
        file writes but still gate shell execution.
    parent:
        Parent Agent when this instance is a sub-agent.
    gen_kwargs:
        Extra kwargs forwarded to every LLM call (temperature, max_tokens, …).
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        cwd: Path,
        name: str = "localcoder",
        max_turns: int = 40,
        verbose: bool = False,
        auto_approve: bool = True,
        always_confirm: Optional[set] = None,
        dry_run: bool = False,
        parent: Optional["Agent"] = None,
        mode: SessionMode = SessionMode.PRIVACY,
        scope: Optional[str] = None,
        disabled_tools: Optional[frozenset] = None,
        restricted: bool = False,
        self_verify: bool = True,
        turn_budget: Optional[int] = None,
        on_event=None,
        confirm_handler=None,
        **gen_kwargs,
    ) -> None:
        # Live-attribute access so tests patching agent.load_memory /
        # make_audit_log are honoured (these names moved into this submodule when
        # agent.py became a package; the _init_* helpers below unpack the rest).
        load_memory = _agent.load_memory
        make_audit_log = _agent.make_audit_log
        self.backend        = backend
        self.cwd            = cwd
        self.name           = name
        self.max_turns      = max_turns
        self.verbose        = verbose
        self.auto_approve   = auto_approve
        self.always_confirm = always_confirm or set()
        self.dry_run        = dry_run
        self.patch_mode     = False        # set via Agent.enable_patch_mode()
        self._patch_chunks: list[str] = [] # accumulated diffs when patch_mode=True
        self.parent         = parent
        self.mode           = mode
        self.scope          = scope        # optional glob filter on file-access tools
        # Restricted = a shareable, non-owner coder session: locked to the
        # SAFE_RESTRICTED_TOOLS allowlist (read + confined edits, no execution,
        # network, env, or sub-agents) and given no external (MCP/plugin/skill)
        # tools. The effective disabled set is finalised after tool registration.
        self.restricted = restricted
        # Tools removed from THIS session: hidden from the model and hard-refused
        # at dispatch so a minted scoped key cannot run them (RCE / data exfil).
        self.disabled_tools = frozenset(disabled_tools or ())
        self.self_verify    = self_verify  # nudge agent to verify code changes before finishing
        # Per-task turn budget for uncertainty escalation. None → 2/3 of max_turns.
        self.turn_budget    = turn_budget if turn_budget is not None else max(3, (max_turns * 2) // 3)
        # Structured event sink (GUI/web sessions). Called with a dict per event:
        # token, tool_call, tool_result, turn, info. None → terminal-only display.
        self.on_event       = on_event
        # External approval hook: Callable[[ToolCall], bool]. When set it is used
        # for destructive-tool confirmation instead of the terminal prompt, in
        # both interactive and non-interactive runs.
        self.confirm_handler = confirm_handler
        self._stop_requested = False
        self.gen_kwargs     = gen_kwargs

        self._messages: list[dict] = []
        self._turns: int = 0
        self._total_tokens: int = 0
        self._last_turn_tokens: int = 0   # tokens used in the most recently completed turn
        self._consecutive_errors: dict[str, int] = {}  # tool_name → failure streak
        self._abort_streak_tool: Optional[str] = None  # set when the circuit breaker trips
        self._global_error_streak: int = 0     # consecutive failed tool calls (ANY tool)
        self._abort_no_progress: bool = False  # set when the no-progress breaker trips
        self._compact_warned: bool = False
        self._last_run_ok: bool = True    # False when the last _loop hit max_turns
        self._undo_stack: list[dict] = []
        self._unverified_writes: set[str] = set()  # code files changed since last test run
        # Changed-files tracker: rel path → {original: bytes|None, writes: int,
        # last_tool: str}. The first-seen original is kept so session_diff()
        # can show the cumulative change, not just the last edit.
        self._changed_files: dict[str, dict] = {}
        # Mid-task steering: messages queued (possibly from another thread)
        # while the loop runs, delivered at the next turn boundary.
        self._queued_messages: list[str] = []
        self._queue_lock = threading.Lock()
        self._model_name: str = getattr(backend, "model_id", "")
        # Per-model harness profile: fill gen-kwarg defaults the caller did not set
        # (e.g. a steadier temperature for a small model). Explicit caller values
        # always win. max_tokens is handled in the CLI, not here (see
        # harness_profiles for why).
        from ..harness_profiles import agent_gen_overrides
        self.gen_kwargs = {**agent_gen_overrides(self._model_name), **self.gen_kwargs}
        self._audit: AuditLogT = make_audit_log(mode, label=name)
        self._project_map: ProjectMap = self._build_project_map(cwd)
        self._memory: str = load_memory(cwd)

        self._init_episodic_memory(cwd)
        self._init_provenance()
        self._init_reviewer()

        # External tools (MCP, then plugins, then skills) are registered BEFORE the
        # system prompt is built so the model learns about them; each step warns and
        # continues on failure - external code must never break the agent.
        self._init_mcp_tools(cwd)
        self._init_plugin_tools()
        self._init_skill_tools(cwd)
        if self.restricted:
            self._apply_restricted_toolset()

        # Single source of truth for the system prompt (see _rebuild_system_prompt);
        # every later rebuild goes through the same helper so the kwargs - notably
        # the COMBINED mcp+plugin+skill tool docs - cannot drift.
        self._system_prompt: str = ""
        self._rebuild_system_prompt()

        # Register OpenAI-format tool definitions when the backend supports it
        # (excluding any tool disabled for this session).
        if getattr(backend, "native_tools", False):
            backend.set_tools([
                d for d in _build_openai_tool_defs()
                if d.get("function", {}).get("name") not in self.disabled_tools
            ])

    # ------------------------------------------------------------------ #
    #  Construction helpers (split out of __init__; see __init__)
    # ------------------------------------------------------------------ #

    def _init_episodic_memory(self, cwd: Path) -> None:
        # Episodic memory: recall lessons from past sessions on this project, and
        # (at session close) distil this session into a new lesson. Disabled for
        # restricted, shareable-key sessions - they must neither read the owner's
        # lessons nor write a trace - and the write half is additionally gated on
        # the privacy contract at close time. The store path resolves under the
        # localm home dir, never the project tree.
        self._episode_task: str = ""
        self._episode_store = None
        try:
            from localm.config import load_config
            _episodic_cfg = bool(load_config().get("coder_episodic_memory", True))
        except Exception:
            _episodic_cfg = True
        self._episodic: bool = _episodic_cfg and not self.restricted
        if self._episodic:
            try:
                from ..episodes import EpisodeStore
                self._episode_store = EpisodeStore(cwd)
            except Exception:
                self._episode_store = None
                self._episodic = False

    def _init_provenance(self) -> None:
        # Provenance tagging: re-frame results from untrusted (network / MCP)
        # tools as data-not-instructions and harden their boundary, so a fetched
        # page or external server cannot inject instructions into the model loop
        # (indirect prompt injection). Defense in depth - it blocks nothing.
        try:
            from localm.config import load_config
            self._untrusted_provenance: bool = bool(
                load_config().get("coder_untrusted_provenance", True))
        except Exception:
            self._untrusted_provenance = True

    def _init_reviewer(self) -> None:
        # Pre-done self-review: an optional reviewer model reads the diff before the
        # agent declares done and feeds blocking issues back. Off by default; a
        # network reviewer is gated off privacy/restricted (see reviewer_for_agent).
        self._review_task: str = ""
        try:
            from ..reviewer import reviewer_for_agent
            self._reviewer = reviewer_for_agent(
                self.backend, self.mode, self.restricted)
        except Exception:
            self._reviewer = None

    def _init_mcp_tools(self, cwd: Path) -> None:
        print_warning = _agent.print_warning
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        # MCP: start configured servers and register their tools BEFORE the
        # system prompt is built so the model learns about them. Failures
        # warn and continue - external servers must never break the agent.
        self._mcp_docs: str = ""
        try:
            from ..mcp import register_mcp_tools
            mcp_names, mcp_warnings = register_mcp_tools(cwd)
            for w in mcp_warnings:
                print_warning(w)
            if mcp_names:
                lines = [
                    f"- {n}: {TOOL_REGISTRY[n].description}"
                    for n in mcp_names if n in TOOL_REGISTRY
                ]
                self._mcp_docs = (
                    "EXTERNAL MCP TOOLS (call exactly like built-in tools)\n"
                    + "\n".join(lines)
                )
        except Exception as e:
            print_warning(f"MCP setup failed: {e}")

    def _init_plugin_tools(self) -> None:
        print_warning = _agent.print_warning
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        # External plugin tools: register any tools exported by installed
        # plugins, the same way as MCP and before the prompt is built. External
        # code defaults to "destructive" (needs confirmation). Failures warn.
        self._plugin_docs: str = ""
        try:
            from ..plugin_tools import register_plugin_tools
            plugin_names, plugin_warnings = register_plugin_tools()
            for w in plugin_warnings:
                print_warning(w)
            if plugin_names:
                lines = [
                    f"- {n}: {TOOL_REGISTRY[n].description}"
                    for n in plugin_names if n in TOOL_REGISTRY
                ]
                self._plugin_docs = (
                    "EXTERNAL PLUGIN TOOLS (call exactly like built-in tools)\n"
                    + "\n".join(lines)
                )
        except Exception as e:
            print_warning(f"Plugin tool setup failed: {e}")

    def _init_skill_tools(self, cwd: Path) -> None:
        print_warning = _agent.print_warning
        # Agent skills: discover SKILL.md folders and expose list_skills/use_skill,
        # the same way as MCP/plugins and before the prompt is built. Read-only
        # tools; a skill's prescribed actions still go through the usual confirm.
        self._skill_docs: str = ""
        try:
            from ..skills import register_skill_tools
            skill_names, skill_warnings = register_skill_tools(cwd)
            for w in skill_warnings:
                print_warning(w)
            if skill_names:
                self._skill_docs = (
                    "AGENT SKILLS: call list_skills to see available skills, then "
                    "use_skill(name) to load one's instructions and follow it."
                )
        except Exception as e:
            print_warning(f"Skill setup failed: {e}")

    def _apply_restricted_toolset(self) -> None:
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        # A shareable, non-owner session gets NO external (MCP/plugin/skill)
        # tools and ONLY the SAFE_RESTRICTED_TOOLS allowlist. Drop the external
        # docs and disable every tool not in the allowlist (run_shell, run_tests,
        # git_commit/push, fetch_url, generate_image, read_env, spawn_agent, and
        # any registered external tool) so the model is neither offered nor able
        # to execute them. Default-deny: a newly-added tool is disabled here too.
        self._mcp_docs = self._plugin_docs = self._skill_docs = ""
        self.disabled_tools = self.disabled_tools | (
            frozenset(TOOL_REGISTRY) - SAFE_RESTRICTED_TOOLS)

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def last_run_ok(self) -> bool:
        """False if the last run ended by hitting max_turns rather than completing normally."""
        return self._last_run_ok

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all LLM calls in this session (server estimate)."""
        return self._total_tokens

    def _emit(self, event_type: str, **data) -> None:
        """Send a structured event to the registered sink. Never raises."""
        if self.on_event is None:
            return
        try:
            self.on_event({"type": event_type, **data})
        except Exception:
            pass  # a broken sink must not kill the agent loop

    def request_stop(self) -> None:
        """Ask the loop to stop at the next safe point (turn or token boundary)."""
        self._stop_requested = True

    def queue_message(self, text: str) -> None:
        """
        Queue a steering message for delivery at the next turn boundary.

        Thread-safe - the GUI calls this from the request thread while the
        agent loop runs in its own thread. The message is injected into the
        conversation before the next LLM call, so the user can redirect a
        running task ("also add logging", "skip the tests") without stopping
        it. Messages queued after a task finishes are delivered at the start
        of the next one.
        """
        with self._queue_lock:
            self._queued_messages.append(text)

    def queued_count(self) -> int:
        with self._queue_lock:
            return len(self._queued_messages)

    def _drain_queued(self) -> list[str]:
        with self._queue_lock:
            msgs, self._queued_messages = self._queued_messages, []
        return msgs
