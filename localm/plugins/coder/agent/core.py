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
from .constants import expand_shell_disable
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
        patch_mode: bool = False,
        parent: Optional["Agent"] = None,
        mode: SessionMode = SessionMode.PRIVACY,
        scope: Optional[str] = None,
        scope_inherited: bool = False,
        disabled_tools: Optional[frozenset] = None,
        restricted: bool = False,
        role: Optional[str] = None,
        inherited_skill_tools: Optional[frozenset] = None,
        self_verify: bool = True,
        verify_cmd=None,
        verify_max_retries: int = 2,
        turn_budget: Optional[int] = None,
        on_event=None,
        confirm_handler=None,
        custom_instructions: Optional[str] = None,
        **gen_kwargs,
    ) -> None:
        # Live-attribute access, so a patched agent.load_memory /
        # load_custom_instructions / make_audit_log is honoured.
        load_memory = _agent.load_memory
        load_custom_instructions = _agent.load_custom_instructions
        cap_user_instructions = _agent.cap_user_instructions
        make_audit_log = _agent.make_audit_log
        self.backend        = backend
        self.cwd            = cwd
        self.name           = name
        self.max_turns      = max_turns
        self.verbose        = verbose
        self._auto_approve  = auto_approve
        self.always_confirm = always_confirm or set()
        self.dry_run        = dry_run
        # True only while _loop(interactive=True) runs, i.e. this session owns a
        # terminal a user can answer a confirmation on. Defaults False so a bare
        # Agent (or an unattended run_task) never claims a channel it lacks; set
        # by _loop. spawn_agent reads it to route a child's confirmations to the
        # parent's REAL channel instead of hard-denying them.
        self._interactive   = False
        # Capture writes as a unified diff instead of touching disk. Settable
        # here (the GUI session, which builds the Agent in one call) or assigned
        # afterwards (the CLI, whose --patch-mode carries a FILE path and so
        # checks truthiness of its own option, not this flag).
        self.patch_mode     = patch_mode
        self._patch_chunks: list[str] = [] # accumulated diffs when patch_mode=True
        self.parent         = parent
        self.mode           = mode
        self._scope         = scope        # optional glob filter on file-access tools
        # Whether _scope was INHERITED from the parent rather than chosen for
        # this agent. Only an inherited scope follows the parent's later
        # changes; an explicit one is a narrowing and is left alone. The
        # inherited VALUE is still copied above, so a child confines itself even
        # if the parent reference ever goes away: confinement must not depend on
        # an object outliving the child.
        self._scope_inherited = bool(scope_inherited)
        # Restricted = a shareable, non-owner coder session: locked to the
        # SAFE_RESTRICTED_TOOLS allowlist (read + confined edits, no execution,
        # network, env, or sub-agents) and given no external (MCP/plugin/skill)
        # tools. The effective disabled set is finalised after tool registration.
        self.restricted = restricted
        # Tools removed from THIS session: hidden from the model and hard-refused
        # at dispatch so a minted scoped key cannot run them (RCE / data exfil).
        self.disabled_tools = expand_shell_disable(frozenset(disabled_tools or ()))
        # Sub-agent role preset (reviewer / researcher / test-writer): a focused
        # mission plus a narrowed toolset. Resolved here so an unknown name fails
        # at construction rather than silently running a full-capability child.
        # The narrowing itself is applied after tool registration, in
        # _apply_role_toolset, and is strictly subtractive - see roles.py.
        from ..roles import resolve_role
        self.role: Optional[str] = None
        self._role_preset = resolve_role(role)
        if self._role_preset is not None:
            self.role = self._role_preset.name
        self.self_verify    = self_verify  # nudge agent to verify code changes before finishing
        # The exit-code oracle: a shell string or an argv list the HARNESS runs
        # at the pre-done boundary, judging the task solely by its exit code
        # (see verify.py and loop.py's _run_verify_gate). None disables it,
        # which is the default - the one-shot CLI keeps using its own outer
        # --until loop (cli/goal.py), so nothing runs the check twice. Set by
        # an interactive REPL/GUI session, OR by tools/agents.py's
        # _isolated_verify_cmd for a worktree-isolated child spawned via
        # spawn_agent_background/dispatch_parallel - that child's diff lands
        # in a tree neither of those other two ever sees, so it needs its own.
        # A RESTRICTED session must never set it: those sessions have no
        # process execution at all (SAFE_RESTRICTED_TOOLS), and an oracle would
        # hand it straight back.
        self.verify_cmd     = None if restricted else verify_cmd
        self.verify_max_retries = verify_max_retries
        # Per-task turn budget for uncertainty escalation. None -> 2/3 of max_turns.
        self.turn_budget    = turn_budget if turn_budget is not None else max(3, (max_turns * 2) // 3)
        # Structured event sink (GUI/web sessions). Called with a dict per event:
        # token, reasoning, tool_call, tool_result, turn, info. None -> terminal-only
        # display. "reasoning" is a thinking model's reasoning text, kept
        # separate from "token" - see _call_llm.
        self.on_event       = on_event
        # External approval hook: Callable[[ToolCall], bool]. When set it is used
        # for destructive-tool confirmation instead of the terminal prompt, in
        # both interactive and non-interactive runs.
        self.confirm_handler = confirm_handler
        self._stop_requested = False
        self.gen_kwargs     = gen_kwargs

        # Stable identity for THIS conversation's resume checkpoint: generated
        # fresh per Agent so two sessions in the same project never write the
        # same file, then overwritten by load_checkpoint() the moment this agent
        # resumes an existing one, so a later save_checkpoint() lands back in the
        # SAME file rather than minting a new one every turn.
        import uuid
        self._checkpoint_id: str = uuid.uuid4().hex[:12]
        # Stable identity for attributing BACKGROUND JOBS to this session (the
        # REPL's /bg, and the GUI's per-session background list). The job
        # registry is process-wide, so on a GUI server - many coder sessions in
        # one process - this is the only thing that can tell one session's
        # background work from another's.
        #
        # INHERITED from the parent, so a spawned sub-agent's background work
        # still belongs to the session that spawned it: otherwise a child's job
        # is attributed to a short-lived agent nobody can query and the parent's
        # /bg stops listing work it started.
        #
        # NOT _checkpoint_id: that one is overwritten by load_checkpoint() when a
        # conversation is resumed, so jobs started before and after a resume
        # would land under two different owners.
        self.job_owner: str = (getattr(parent, "job_owner", None)
                               or uuid.uuid4().hex[:12])
        # The raw, pre-episodic-preamble text of this session's first task/
        # message - captured once in loop.py's run_task/chat, restored by
        # resume_checkpoint() so it survives a pause/resume. save_checkpoint()
        # turns it into a short display title (checkpoint._derive_title) for
        # a resume listing; kept RAW here rather than pre-truncated.
        self._session_title: str = ""

        # --- active-skill restriction (see _activate_skill / _skill_gate_denial) --
        # Initialised BEFORE anything can assign _last_user_request, whose property
        # setter below bumps the sequence.
        self._user_request_seq: int = 0
        self._active_skill_tools: Optional[frozenset] = None  # None = no restriction
        self._active_skill_names: list[str] = []
        self._active_skill_seq: int = -1
        self._skill_lock = threading.Lock()
        # A spawned child's inheritance of the parent's live skill restriction.
        # Held as the ALLOWLIST, not as a pre-computed disabled set: the child
        # registers its own MCP / plugin / skill tools during this __init__, so a
        # set computed at the spawn site would miss every one of them.
        self._inherited_skill_tools: Optional[frozenset] = (
            frozenset(inherited_skill_tools)
            if inherited_skill_tools is not None else None)

        self._messages: list[dict] = []
        self._turns: int = 0
        self._total_tokens: int = 0
        self._last_turn_tokens: int = 0   # tokens used in the most recently completed turn
        self._consecutive_errors: dict[str, int] = {}  # tool_name -> failure streak
        self._abort_streak_tool: Optional[str] = None  # set when the circuit breaker trips
        self._global_error_streak: int = 0     # consecutive failed tool calls (ANY tool)
        self._abort_no_progress: bool = False  # set when the no-progress breaker trips
        self._last_response_fp: str = ""       # last LLM response (repeated-scaffold breaker)
        self._repeat_response_count: int = 0   # consecutive near-identical responses
        self._recent_finals: list[str] = []    # bounded history the breaker compares against
        self._compact_warned: bool = False
        # Sticky, session-wide: latched True once the SERVER has authoritatively
        # refused a grammar-bearing request (see context._disable_grammar_on_
        # unsupported). Distinct from _force_tool_grammar, which is a one-shot
        # per-turn flag the escalation ladder sets/clears - this one never
        # re-arms.
        self._grammar_confirmed_unsupported: bool = False
        # The NARROWER sibling of the flag above, latched when the server refused
        # the LAZY (trigger-gated) form specifically. A backend that cannot
        # enforce a grammar from a trigger may still enforce one from the first
        # token, so this must NOT disable the forced rung - see
        # context._disable_grammar_on_unsupported.
        self._lazy_grammar_confirmed_unsupported: bool = False
        # Per-run: False when the LAST _loop failed (max_turns, a circuit breaker,
        # a stop). _loop re-arms it to True at the start of every run, so one bad
        # turn in a multi-turn session (REPL / GUI) does not mislabel every later
        # turn as a failure - the GUI reports this per turn ("ok" on the final
        # event) and the CLI turns it into an exit code.
        self._last_run_ok: bool = True
        # Per-run outcome of the exit-code oracle: "passed", "failed",
        # "inconclusive", or None when no check ran. A THIRD state: _last_run_ok
        # is a boolean and "the check could not run" is neither of its two
        # answers. Consumers that want "was this verified" read this;
        # _last_run_ok keeps meaning "did the run itself complete".
        self._last_verify_state: Optional[str] = None
        # Session-level: True once ANY run this session failed. The close-time
        # episodic reflection (session.py) needs the session-wide answer and
        # reads this one, so _last_run_ok being per-run does not narrow "did this
        # session fail" to "did the last run fail".
        self._had_any_failure: bool = False
        # True when the last _loop ended because the USER stopped it (Ctrl-C, or
        # declining "keep going?"), as opposed to a genuine failure (max_turns, a
        # circuit breaker). Both clear _last_run_ok, but only the latter carries
        # a lesson worth a close-time reflection.
        self._user_stopped: bool = False
        # Non-destructive tool threads abandoned at a parallel-batch deadline, as
        # (future, tool name). SESSION state, not per-batch: a tool that outran the
        # 120s batch deadline usually outlives the whole TURN too (a real test
        # suite does), so a per-call list would be empty again on the very next
        # turn and a destructive tool would launch straight into the
        # still-running peer.
        self._abandoned_peers: list = []
        self._undo_stack: list[dict] = []
        self._unverified_writes: set[str] = set()  # code files changed since last test run
        # Changed-files tracker: rel path -> {original: bytes|None, writes: int,
        # last_tool: str}. The first-seen original is kept so session_diff()
        # can show the cumulative change, not just the last edit.
        self._changed_files: dict[str, dict] = {}
        # Work done by ISOLATED children (their own git worktree), which is NOT
        # in _changed_files: it never touched this tree, and
        # merging its keys would fabricate diffs (keys are relative to the writing
        # agent's cwd, and session_diff re-resolves them against ours). Surfaced
        # as a separate labelled section in the human-facing views only.
        self._delegated: list = []
        # Bounded trace of tool/command failures this session, fed into the
        # close-time episode reflection so it can capture what_failed. Newest
        # kept, capped at _MAX_ERROR_TRACE.
        self._error_trace: list[str] = []
        # git change-detection baseline for the close-time episode: the set of
        # dirty paths captured just BEFORE the first run_shell, so run_shell writes
        # (git apply, formatters, codegen) the write-tool tracker never records can
        # be attributed to THIS session at close, without misattributing a
        # pre-existing dirty tree. None until captured / when cwd is not a git
        # work tree.
        self._shell_baseline_captured: bool = False
        self._git_baseline: Optional[frozenset] = None
        # Mid-task steering: messages queued (possibly from another thread)
        # while the loop runs, delivered at the next turn boundary.
        self._queued_messages: list[str] = []
        self._queue_lock = threading.Lock()
        # The model's own task list (tools/tasks.py). Held HERE, not in
        # _messages, so compaction cannot summarise the plan away, and written
        # into the resume checkpoint so it survives pause/resume. The lock guards
        # it against the parallel non-destructive tool batch (loop.py).
        self._todos: list[dict] = []
        self._todos_lock = threading.Lock()
        self._model_name: str = getattr(backend, "model_id", "")
        # Family-detection identity: enrich the (possibly opaque) alias with its
        # registry source (e.g. "hf:google/gemma-4-4b") so family-specific prompt
        # tuning keys off the model's REAL identity, not the alias. Display and
        # logging still use the bare alias.
        self._family_id: str = self._model_name
        try:
            # The registry entry (not get_model_info, which returns a
            # (path, hint) tuple) carries "source".
            from localm.model_manager import load_registry
            _entry = load_registry().get(self._model_name) or {}
            _src = _entry.get("source", "") if isinstance(_entry, dict) else ""
            if isinstance(_src, str) and _src.strip():
                self._family_id = f"{self._model_name} {_src}"
        except Exception:
            pass
        # Per-model harness profile: fill gen-kwarg defaults the caller did not set
        # (e.g. a steadier temperature for a small model); explicit caller values win.
        # max_tokens is handled in the CLI, not here (see harness_profiles).
        from ..harness_profiles import agent_gen_overrides
        self.gen_kwargs = {**agent_gen_overrides(self._model_name), **self.gen_kwargs}
        self._audit: AuditLogT = make_audit_log(mode, label=name)
        self._project_map: ProjectMap = self._build_project_map(cwd)
        self._memory: str = load_memory(cwd)
        # User-authored custom instructions: an explicit string (CLI
        # --system) overrides the .localcoder/system.md file; None means "read the
        # file". The override is kept so it survives a later set_cwd (file re-read
        # from the new cwd). Distinct from project memory above.
        self._system_override: Optional[str] = custom_instructions
        self._custom_instructions: str = (
            cap_user_instructions(custom_instructions) if custom_instructions is not None
            else load_custom_instructions(cwd))
        # Say so at startup when either file was too big to inject whole (or could
        # not be read), so a capped prompt is never a silent surprise.
        self._warn_injected_file_limits()

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
        # AFTER restriction, and after every external tool has registered, so a
        # role narrows the COMPLETE registry (see _apply_role_toolset).
        if self._role_preset is not None:
            self._apply_role_toolset()
        # Same placement, same reason: a parent under an active skill's
        # allowed-tools must not be able to spawn a child that escapes it (see
        # _apply_inherited_skill_toolset).
        if self._inherited_skill_tools is not None:
            self._apply_inherited_skill_toolset()

        # After the toolset is final: tell the user, once, if their --scope does
        # not mean what they almost certainly think it means.
        self._notify_scope_does_not_confine_shell()

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
    #  Inherited-from-the-parent state: READ LIVE, never snapshotted      #
    # ------------------------------------------------------------------ #
    #
    # A SUB-AGENT IS NOT A SEPARATE SESSION. It has no user of its own, no
    # confirmation channel of its own, and nothing can address it from outside;
    # the session that spawned it is the only thing a human can steer. So the
    # two settings a human revokes MID-RUN are read through the parent at every
    # dispatch instead of copied into the child once, so revoking auto-approve
    # or tightening a scope reaches a run already under way.

    @property
    def auto_approve(self) -> bool:
        """Whether destructive tools skip confirmation.

        A child can only ever be NARROWER than its parent: once the parent's
        approval is revoked the child's own True stops counting. It cannot work
        the other way round - a parent turning auto-approve back ON does not
        silently re-approve a child that was spawned without it, because the
        child's own value still has to be True as well.
        """
        if not self._auto_approve:
            return False
        parent = getattr(self, "parent", None)
        # Recursive through the property, so a grandchild is bounded by the whole
        # ancestor chain rather than only by its immediate parent.
        if parent is not None and not getattr(parent, "auto_approve", True):
            return False
        return True

    @auto_approve.setter
    def auto_approve(self, value: bool) -> None:
        self._auto_approve = bool(value)

    @property
    def scope(self):
        """The glob confining the file tools, or None.

        A child that INHERITED its scope follows the parent's, live - so
        tightening a scope mid-run reaches work already in flight. A child given
        an EXPLICIT scope keeps it: an explicit child scope is a deliberate
        narrowing (see inherited_child_kwargs), and following the parent over it
        would widen the child, the one direction that must never happen.

        The inherited copy in ``_scope`` is the floor and is never discarded, so
        a child whose parent reference is gone still confines itself to whatever
        it inherited rather than silently becoming unscoped.
        """
        if self._scope_inherited:
            parent = getattr(self, "parent", None)
            if parent is not None:
                return parent.scope
        return self._scope

    @scope.setter
    def scope(self, value) -> None:
        self._scope = value

    #  Construction helpers (split out of __init__).

    def _init_episodic_memory(self, cwd: Path) -> None:
        # Episodic memory: recall lessons from past sessions on this project, and
        # (at session close) distil this session into a new lesson. Disabled for
        # restricted, shareable-key sessions (neither read the owner's lessons nor
        # write a trace). In privacy mode it is off too UNLESS the user opted into
        # read-only recall (memory_recall_in_privacy + ..._coder): past lessons are
        # RECALLED but the close-time write stays blocked (session.py gates it on
        # privacy), so no new trace is created. The store path resolves under the
        # localm home dir, never the project tree.
        self._episode_task: str = ""
        self._episode_store = None
        # WHICH past lessons this run actually recalled ({id, lesson, outcome}),
        # and why recall came back empty when it did, so injected text is
        # traceable and a bad lesson can be forgotten by id. The chat side keeps
        # the same record in ctx.state["memory_used"].
        self._episodes_used: list = []
        self._episodes_degrade_reason: str = ""
        try:
            from localm.config import load_config
            _cfg = load_config()
            _episodic_cfg = bool(_cfg.get("coder_episodic_memory", True))
            _recall_in_privacy = bool(
                _cfg.get("memory_recall_in_privacy")
                and _cfg.get("memory_recall_in_privacy_coder", True))
        except Exception:
            _episodic_cfg = True
            _recall_in_privacy = False
        _privacy_ok = self.mode != SessionMode.PRIVACY or _recall_in_privacy
        self._episodic: bool = _episodic_cfg and not self.restricted and _privacy_ok
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

    def _notify_scope_does_not_confine_shell(self) -> None:
        """Once per session: say plainly that an active scope does not confine the
        shell tools, when any of them is actually enabled.

        ``--scope`` reads as "this session can only touch these files", and for
        every file tool it is exactly that. run_shell / run_tests execute a
        process, which no path-arg check can confine, so they are left out
        (_INTENTIONALLY_UNSCOPED) and this notice states so at runtime."""
        print_warning = _agent.print_warning
        if not self.scope:
            return
        if self.parent is not None:
            return    # a sub-agent is not a separate session: the parent already said it
        from .constants import _SHELL_UNSCOPED_TOOLS
        enabled = sorted(_SHELL_UNSCOPED_TOOLS - set(self.disabled_tools))
        if not enabled:
            return    # e.g. a restricted session: no shell at all, so nothing to warn about
        msg = (
            f"scope '{self.scope}' confines the file tools only. "
            f"{'/'.join(enabled)} execute a process, which a path check cannot "
            "confine, so a command can still read and write outside the scope. "
            "Disable them for a hard boundary."
        )
        print_warning(msg)
        self._emit("info", text=msg)
        self._audit.notice("scope_shell_unconfined", msg)

    def _apply_restricted_toolset(self) -> None:
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        # A shareable, non-owner session gets NO external (MCP/plugin/skill) tools
        # and ONLY the SAFE_RESTRICTED_TOOLS allowlist. Drop the external docs and
        # disable every tool not in the allowlist (run_shell, run_tests, git_commit/
        # push, fetch_url, generate_image, read_env, spawn_agent, any external tool)
        # so the model is neither offered nor able to execute them. Default-deny: a
        # newly-added tool is disabled here too.
        self._mcp_docs = self._plugin_docs = self._skill_docs = ""
        self.disabled_tools = self.disabled_tools | (
            frozenset(TOOL_REGISTRY) - SAFE_RESTRICTED_TOOLS)

    def _apply_role_toolset(self) -> None:
        """Narrow this session to its role's allowlist. STRICTLY SUBTRACTIVE.

        The new disabled set is a UNION with what is already disabled, never an
        assignment, so a role can only ever REMOVE capability: it cannot hand back
        a tool the parent disabled, nor one a restricted (shareable, non-owner)
        session forbids. That ordering matters - this runs after
        _apply_restricted_toolset, so restricted-then-role composes to the
        intersection of both allowlists rather than whichever ran last.

        Subtracting from the LIVE registry (like the restricted path above, and
        for the same reason) means every dynamically registered MCP / plugin /
        skill tool is denied to a role by default: an allowlist cannot be
        outflanked by a tool that did not exist when the preset was written.
        """
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        assert self._role_preset is not None  # guarded at the call site
        # External tool docs go too: a narrowed child that cannot call any of
        # them does not read their documentation either.
        self._mcp_docs = self._plugin_docs = self._skill_docs = ""
        self.disabled_tools = self.disabled_tools | (
            frozenset(TOOL_REGISTRY) - self._role_preset.allowed_tools)

    # ------------------------------------------------------------------ #
    #  Active-skill restriction: SKILL.md allowed-tools, hard-enforced     #
    # ------------------------------------------------------------------ #

    def _apply_inherited_skill_toolset(self) -> None:
        """Carry a spawning parent's active skill restriction into this child.

        Without it, ``allowed-tools: read_file, spawn_agent`` is a one-line
        bypass of the whole gate: the skill delegates, and the child - a fresh
        Agent with no active skill - writes files the skill never declared.
        STRICTLY SUBTRACTIVE, a union with what is already disabled, applied
        after every dynamic tool has registered.

        The skill's own two tools stay reachable so a child can still read the
        skill's bundled files (see SKILL_META_TOOLS).
        """
        from ..skills import SKILL_META_TOOLS
        TOOL_REGISTRY = _agent.TOOL_REGISTRY
        assert self._inherited_skill_tools is not None   # guarded at the call site
        self.disabled_tools = self.disabled_tools | (
            frozenset(TOOL_REGISTRY)
            - self._inherited_skill_tools - SKILL_META_TOOLS)

    def active_skill_tools(self) -> Optional[frozenset]:
        """The live allowed-tools intersection, or None when nothing is active.

        Public because the spawn path (tools/agents.py) has to read it off the
        parent to hand it to a child. Expires the same way every other read does,
        so a stale restriction from an earlier turn is never inherited.
        """
        with self._skill_lock:
            self._expire_active_skill_locked()
            return self._active_skill_tools

    @property
    def _last_user_request(self) -> str:
        """The raw text of the most recent USER request (not a mid-run nudge).

        A property, rather than a plain attribute, so the SETTER can count user
        requests - that count is what retires an active skill's restriction (see
        _activate_skill). loop.py assigns this at exactly the three user entry
        points (run_task / continue_task / chat) and nowhere else, so observing
        the assignment gives the turn boundary exactly, with no cooperation
        needed from loop.py and no hook inside the agentic loop to keep in step.

        Observing the ASSIGNMENT and not the VALUE is the point: a user who
        repeats a request verbatim still starts a new turn, and comparing the
        strings would silently miss it.
        """
        return self.__dict__.get("_last_user_request_text", "")

    @_last_user_request.setter
    def _last_user_request(self, value: str) -> None:
        self.__dict__["_last_user_request_text"] = value
        # getattr default: a subclass or a restored checkpoint could conceivably
        # assign this before __init__ has run its state block.
        self._user_request_seq = getattr(self, "_user_request_seq", 0) + 1

    def _activate_skill(self, name: str, allowed_tools) -> None:
        """Arm ``name``'s allowed-tools restriction for the rest of this turn.

        THE RESTRICTION ONLY EVER NARROWS. It is intersected with any skill
        already active, and the dispatch gate is checked on top of (never
        instead of) ``disabled_tools``, so the tools that can actually run are
        ``(registry - disabled_tools) & every active skill's allowed-tools``.
        The same strictly-subtractive invariant roles.py states for role presets.

        THERE IS NO RELEASE THE MODEL CAN CALL, and a second use_skill intersects
        rather than replaces. A SKILL.md body is UNTRUSTED content (skills.py),
        so any widening the model can reach is a one-line bypass - "release the
        restriction, then write_file", or "load this other skill that declares
        nothing, then write_file". The only boundary the model cannot reach is
        the human's next request: hence the sequence check below.

        An absent or empty allowed-tools arms NOTHING - the field is optional in
        the format and most skills omit it. An unrestricted skill contributes no
        set to intersect, so loading one while a restricted skill is active
        leaves the restriction exactly as it was.
        """
        allowed = frozenset(t for t in (allowed_tools or ()) if t)
        if not allowed:
            return
        with self._skill_lock:
            self._expire_active_skill_locked()
            self._active_skill_seq = self._user_request_seq
            self._active_skill_tools = (
                allowed if self._active_skill_tools is None
                else self._active_skill_tools & allowed)
            if name not in self._active_skill_names:
                self._active_skill_names.append(name)

    def _expire_active_skill_locked(self) -> None:
        """Drop the restriction if it belongs to an earlier user request.

        Lazy rather than cleared at a turn boundary: the boundary lives in
        loop.py's ``_loop``, and expiring on READ needs nothing there at all.
        Caller must hold ``_skill_lock``.
        """
        if self._active_skill_seq != self._user_request_seq:
            self._active_skill_tools = None
            self._active_skill_names = []
            self._active_skill_seq = self._user_request_seq

    def _skill_gate_denial(self, tool_name: str) -> Optional[str]:
        """The refusal message when an active skill forbids ``tool_name``, else None.

        The enforcement half of ``allowed-tools``, kept beside the state it
        reads; the dispatcher owns only the branch that acts on the answer.
        """
        from ..skills import SKILL_META_TOOLS
        with self._skill_lock:
            self._expire_active_skill_locked()
            allowed = self._active_skill_tools
            names = ", ".join(self._active_skill_names)
        if allowed is None or tool_name in allowed or tool_name in SKILL_META_TOOLS:
            return None
        return (
            f"'{tool_name}' was not run: the active skill ({names}) declares "
            f"allowed-tools: {', '.join(sorted(allowed))}, and nothing outside "
            "that list may run while it is loaded. Do the task with the tools "
            "the skill declares, or tell the user this skill cannot do what "
            f"they asked without {tool_name}."
        )

    @property
    def turns(self) -> int:
        return self._turns

    @property
    def last_run_ok(self) -> bool:
        """False if the LAST run failed (max_turns, a circuit breaker, a stop)
        rather than completing normally.

        Per-run, not per-session: a fresh run re-arms it, so a later healthy turn
        in the same session reports ok. ``_had_any_failure`` is the session-wide
        answer."""
        return self._last_run_ok

    @property
    def last_verify_state(self) -> Optional[str]:
        """How the exit-code oracle ended for the LAST run.

        ``"passed"`` (exited 0), ``"failed"`` (still failing after the retries),
        ``"inconclusive"`` (the command could not run, or collected nothing), or
        None when no check ran at all. Per-run, like ``last_run_ok``."""
        return self._last_verify_state

    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all LLM calls in this session (server estimate)."""
        return self._total_tokens

    def get_todos(self) -> list[dict]:
        """The model's current task list, as a copy (see :attr:`_todos`)."""
        with self._todos_lock:
            return [dict(t) for t in self._todos]

    def set_todos(self, todos: list[dict]) -> None:
        """Replace the task list. Copies in, so the caller cannot mutate the
        stored list afterwards; the whole-list swap under the lock is what makes
        two set_todos calls in one parallel batch last-writer-wins instead of
        torn."""
        with self._todos_lock:
            self._todos = [dict(t) for t in todos]

    def _emit(self, event_type: str, **data) -> None:
        """Send a structured event to the registered sink. Never raises."""
        if self.on_event is None:
            return
        try:
            self.on_event({"type": event_type, **data})
        except Exception as e:
            # A broken sink must not kill the agent loop, but dropping the event
            # in total silence hides the breakage: a consumer wired before it is
            # ready loses every event with nothing to find later. Log, continue.
            from localm.debuglog import logger
            logger.debug("on_event sink raised on a %r event: %s", event_type, e)

    def _record_error(self, tool: str, output: str) -> None:
        """Append a tool/command failure to the bounded session error trace that
        feeds the close-time episode reflection. Each entry is collapsed to one
        trimmed line; the newest _MAX_ERROR_TRACE are kept."""
        from .constants import _MAX_ERROR_TRACE
        line = " ".join((output or "").split())[:200]
        if not line:
            return
        self._error_trace.append(f"{tool}: {line}")
        if len(self._error_trace) > _MAX_ERROR_TRACE:
            self._error_trace = self._error_trace[-_MAX_ERROR_TRACE:]

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
