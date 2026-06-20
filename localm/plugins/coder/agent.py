# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Agent - the core agentic loop.

Flow per turn:
    1. Call the LLM with the current message history
    2. Parse the response for <tool_call> blocks
    3. If no tool calls → final answer, break
    4. For each tool call:
       a. Display it
       b. Optionally confirm (destructive tools + auto_approve=False)
       c. Execute
       d. Append result to messages as a user turn
    5. Repeat

The Agent class is used by:
    - CLI in interactive chat mode
    - CLI in single-task mode (run_task)
    - spawn_agent tool (child agents)
"""

from __future__ import annotations

import datetime
import difflib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from .backends.base import BaseLLMBackend
from .indexer import ProjectMap
from .memory import load_memory, remember, forget
from .parser import (
    ToolCall, looks_like_tool_attempt, parse_tool_calls, split_response,
)
from .tools import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY, ToolResult
from .display import (
    confirm,
    confirm_diff,
    console,
    print_assistant_label,
    print_assistant_response,
    print_diff_preview,
    print_info,
    print_streaming_done,
    print_streaming_token,
    print_thinking,
    print_tool_call,
    print_tool_error,
    print_tool_result,
    print_turn_divider,
    print_warning,
)
from .audit import AuditLog, AuditLogT, NullAuditLog, SessionMode, make_audit_log
from .prompts import build_system_prompt

# Tools that mutate files - trigger a project map refresh after they run
_MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "run_shell"})

# Tools whose file changes can be undone (we snapshot before they run).
# These are also the tools recorded in the changed-files tracker.
_UNDOABLE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "patch_file", "edit_notebook_cell",
})

# File-access tools whose target path must match the active scope glob.
# The check keys on the `path` arg; for tools whose primary target is a
# `glob` or `output_path` arg instead (or as well), that arg is checked too
# (see _SCOPE_PATH_ARGS). run_shell is intentionally NOT scoped: it runs
# arbitrary commands, so a path-arg check cannot meaningfully confine it.
_SCOPED_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "patch_file",
    "list_dir", "tree",
    # FAC-8: the rest of the file-reading/writing tools.
    "grep", "search_files", "search_replace", "read_env",
    "edit_notebook_cell", "generate_image",
})

# For each scoped tool, the argument names that name a path/glob to enforce the
# scope against. Order matters only for which value is reported first; any
# present arg that falls outside the scope rejects the call. Tools default to
# checking "path"; entries here add (or replace with) the tool's real target.
_SCOPE_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "grep":           ("path", "glob"),
    "search_files":   ("path", "pattern"),
    "search_replace": ("glob",),
    "generate_image": ("output_path", "input_image"),
}

# Model-initiated network tools, governed by the net_mode policy
# (localm.netpolicy): off = fail fast, ask = approval flow, allow = run.
_NETWORK_TOOLS: frozenset[str] = frozenset({"fetch_url", "web_search"})

# Fraction of estimated context window at which compaction is triggered
_COMPACT_WARN_RATIO  = 0.70   # warn user in interactive mode
_COMPACT_AUTO_RATIO  = 0.90   # silently compact in non-interactive mode
_DEFAULT_CTX_TOKENS  = 4096   # fallback when n_ctx is unknown

# Code file extensions that should be verified (tests / syntax) after writes
_CODE_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php",
})

# run_shell commands containing one of these substrings count as verification
_TEST_COMMAND_MARKERS: tuple[str, ...] = ("pytest", "unittest", "npm test", "cargo test", "go test")


# ---------------------------------------------------------------------------
#  Scope matching (path-aware glob)
# ---------------------------------------------------------------------------

# Cache compiled scope patterns - the same scope is matched many times per run.
_SCOPE_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _glob_to_regex(pattern: str) -> "re.Pattern[str]":
    """
    Compile a path-aware glob into a regex anchored to a full relative path.

    Semantics (gitignore / pathspec style), unlike plain ``fnmatch``:
      - ``*``  matches any run of characters WITHIN one path segment - it does
        NOT cross ``/``. So ``src/*.py`` matches ``src/a.py`` but not
        ``src/a/b.py``.
      - ``**`` matches across segments. ``**/`` matches any number of leading
        directories (including none); a trailing ``**`` matches the rest.
      - ``?``  matches a single non-``/`` character.
      - all other characters are matched literally.
    """
    i, n = 0, len(pattern)
    out = ["(?s:"]
    while i < n:
        c = pattern[i]
        if c == "*":
            if pattern[i:i + 2] == "**":
                j = i
                while j < n and pattern[j] == "*":
                    j += 1
                if pattern[j:j + 1] == "/":
                    # '**/' -> zero or more leading directory segments
                    out.append("(?:[^/]+/)*")
                    i = j + 1
                else:
                    out.append(".*")
                    i = j
            else:
                out.append("[^/]*")
                i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "/":
            out.append("/")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    out.append(r")\Z")
    return re.compile("".join(out))


def _scope_pattern(scope: str) -> "re.Pattern[str]":
    rx = _SCOPE_RE_CACHE.get(scope)
    if rx is None:
        rx = _glob_to_regex(scope)
        _SCOPE_RE_CACHE[scope] = rx
    return rx


# ---------------------------------------------------------------------------
#  Native tool-calling helpers
# ---------------------------------------------------------------------------

def _build_openai_tool_defs() -> list:
    """
    Convert TOOL_REGISTRY into the OpenAI /v1/chat/completions ``tools`` format.

    Used when the backend has ``native_tools=True`` (e.g. the OpenAI API),
    so the model receives a validated schema instead of relying on text parsing.
    """
    defs = []
    for tool in TOOL_REGISTRY.values():
        properties: dict = {}
        required:   list = []
        for param_name, meta in tool.params.items():
            prop: dict = {"description": meta.get("description", "")}
            raw_type = meta.get("type", "string")
            # Map our shorthand types to JSON Schema types
            prop["type"] = {
                "int":   "integer",
                "float": "number",
                "bool":  "boolean",
                "array": "array",
            }.get(raw_type, "string")
            if raw_type == "array":
                prop["items"] = {"type": "string"}
            properties[param_name] = prop
            if meta.get("required"):
                required.append(param_name)
        defs.append({
            "type": "function",
            "function": {
                "name":        tool.name,
                "description": tool.description,
                "parameters": {
                    "type":       "object",
                    "properties": properties,
                    "required":   required,
                },
            },
        })
    return defs


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

class Agent:
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
        self._audit: AuditLogT = make_audit_log(mode, label=name)
        self._project_map: ProjectMap = ProjectMap.build(cwd)
        self._memory: str = load_memory(cwd)

        # MCP: start configured servers and register their tools BEFORE the
        # system prompt is built so the model learns about them. Failures
        # warn and continue - external servers must never break the agent.
        self._mcp_docs: str = ""
        try:
            from .mcp import register_mcp_tools
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

        # External plugin tools: register any tools exported by installed
        # plugins, the same way as MCP and before the prompt is built. External
        # code defaults to "destructive" (needs confirmation). Failures warn.
        self._plugin_docs: str = ""
        try:
            from .plugin_tools import register_plugin_tools
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

        # Agent skills: discover SKILL.md folders and expose list_skills/use_skill,
        # the same way as MCP/plugins and before the prompt is built. Read-only
        # tools; a skill's prescribed actions still go through the usual confirm.
        self._skill_docs: str = ""
        try:
            from .skills import register_skill_tools
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

        if self.restricted:
            # A shareable, non-owner session gets NO external (MCP/plugin/skill)
            # tools and ONLY the SAFE_RESTRICTED_TOOLS allowlist. Drop the external
            # docs and disable every tool not in the allowlist (run_shell, run_tests,
            # git_commit/push, fetch_url, generate_image, read_env, spawn_agent, and
            # any registered external tool) so the model is neither offered nor able
            # to execute them. Default-deny: a newly-added tool is disabled here too.
            self._mcp_docs = self._plugin_docs = self._skill_docs = ""
            self.disabled_tools = self.disabled_tools | (
                frozenset(TOOL_REGISTRY) - SAFE_RESTRICTED_TOOLS)

        self._system_prompt: str = build_system_prompt(
            cwd,
            agent_name=name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
            extra_tool_docs="\n\n".join(
                d for d in (self._mcp_docs, self._plugin_docs, self._skill_docs) if d
            ),
            disabled_tools=self.disabled_tools,
        )

        # Register OpenAI-format tool definitions when the backend supports it
        # (excluding any tool disabled for this session).
        if getattr(backend, "native_tools", False):
            backend.set_tools([
                d for d in _build_openai_tool_defs()
                if d.get("function", {}).get("name") not in self.disabled_tools
            ])

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

    # ------------------------------------------------------------------ #
    #  Changed-files tracking
    # ------------------------------------------------------------------ #

    def changed_files(self) -> list[dict]:
        """
        Files this session has written, with change counts.

        Each entry: ``{path, writes, created, exists, last_tool}`` where
        *created* means the file did not exist before this session touched it
        and *exists* is its current on-disk state (False = since deleted).
        """
        # Snapshot first - the GUI reads this from another thread while the
        # agent loop may be inserting entries.
        snapshot = dict(self._changed_files)
        out = []
        for key in sorted(snapshot):
            e = snapshot[key]
            abs_path = (self.cwd / key)
            out.append({
                "path": key,
                "writes": e["writes"],
                "created": e["original"] is None,
                "exists": abs_path.is_file(),
                "last_tool": e["last_tool"],
            })
        return out

    def session_diff(self, path: Optional[str] = None) -> str:
        """
        Cumulative unified diff of everything this session changed.

        Compares each tracked file's first-seen original content against its
        current on-disk state - so three successive edits to one file show as
        one combined diff. Pass *path* for a single file, None for all.
        Returns "" when nothing was changed (or the path is untracked).
        """
        snapshot = dict(self._changed_files)   # cross-thread read safety
        keys = [path] if path else sorted(snapshot)
        parts: list[str] = []
        for key in keys:
            entry = snapshot.get(key)
            if entry is None:
                continue
            original = entry["original"]
            old_text = (original.decode("utf-8", errors="replace")
                        if original is not None else "")
            abs_path = (self.cwd / key)
            try:
                new_text = (abs_path.read_text(encoding="utf-8", errors="replace")
                            if abs_path.is_file() else "")
            except Exception:
                new_text = ""
            diff = "".join(difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{key}" if original is not None else "/dev/null",
                tofile=f"b/{key}" if new_text else "/dev/null",
            ))
            if diff:
                parts.append(diff)
        return "\n".join(parts)

    def _record_changed_file(self, path_arg: str, old_content: bytes | None,
                             tool: str) -> None:
        """Track a successful file write in the changed-files map."""
        abs_path = (self.cwd / path_arg).resolve()
        try:
            key = abs_path.relative_to(self.cwd.resolve()).as_posix()
        except ValueError:
            key = str(abs_path)
        entry = self._changed_files.get(key)
        if entry is None:
            self._changed_files[key] = {
                "original": old_content, "writes": 1, "last_tool": tool,
            }
        else:
            entry["writes"] += 1
            entry["last_tool"] = tool

    def undo_list(self) -> list[dict]:
        """The undo stack, most recent first: ``[{tool, path}, ...]``."""
        return [{"tool": e["tool"], "path": str(e["path"])}
                for e in reversed(self._undo_stack)]

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    #  Checkpoint (interruption / resume)
    # ------------------------------------------------------------------ #

    @property
    def _checkpoint_path(self) -> Path:
        return self.cwd / ".localcoder" / "checkpoint.json"

    def save_checkpoint(self) -> None:
        """Persist current conversation state so it can be resumed later.

        No-op in privacy mode - the checkpoint contains the full
        conversation, which privacy mode promises never to write to disk."""
        if self.mode == SessionMode.PRIVACY:
            return
        data = {
            "version": 1,
            "interrupted_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "turns": self._turns,
            "total_tokens": self._total_tokens,
            "messages": self._messages,
        }
        p = self._checkpoint_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # never let checkpoint failure crash the session

    def clear_checkpoint(self) -> None:
        """Remove any saved checkpoint for this working directory."""
        try:
            self._checkpoint_path.unlink(missing_ok=True)
        except Exception:
            pass

    def load_checkpoint(self) -> dict | None:
        """
        Read the checkpoint file if it exists and is valid.

        Returns the parsed dict, or None if no checkpoint is found.
        """
        p = self._checkpoint_path
        if not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("version") == 1 and isinstance(data.get("messages"), list):
                return data
        except Exception:
            pass
        return None

    def resume_checkpoint(self, data: dict) -> None:
        """Restore agent state from a checkpoint dict."""
        self._messages     = data["messages"]
        self._turns        = data.get("turns", len(self._messages))
        self._total_tokens = data.get("total_tokens", 0)

    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Clear conversation history."""
        self._messages = []
        self._turns = 0
        self._total_tokens = 0
        self._last_turn_tokens = 0
        self._compact_warned = False
        self._consecutive_errors.clear()
        self._last_run_ok = True
        self._unverified_writes.clear()

    def set_cwd(self, cwd: Path) -> None:
        self.cwd = cwd
        self._project_map = ProjectMap.build(cwd)
        self._memory = load_memory(cwd)
        self._system_prompt = build_system_prompt(
            cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
            extra_tool_docs=self._mcp_docs,
            disabled_tools=self.disabled_tools,
        )

    def reindex(self) -> int:
        """Rebuild the full project map and regenerate the system prompt."""
        self._project_map = ProjectMap.build(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
            extra_tool_docs=self._mcp_docs,
            disabled_tools=self.disabled_tools,
        )
        return self._project_map.file_count()

    def reload_memory(self) -> str:
        """Re-read the memory file from disk and rebuild the system prompt."""
        self._memory = load_memory(self.cwd)
        self._system_prompt = build_system_prompt(
            self.cwd,
            agent_name=self.name,
            project_map=self._project_map,
            memory=self._memory,
            model_name=self._model_name,
            extra_tool_docs=self._mcp_docs,
            disabled_tools=self.disabled_tools,
        )
        return self._memory

    def remember(self, text: str) -> Path:
        """Append a bullet to the memory file and refresh the system prompt."""
        p = remember(self.cwd, text)
        self.reload_memory()
        return p

    def forget(self, pattern: str) -> tuple:
        """Remove matching bullets from the memory file and refresh the system prompt."""
        p, n = forget(self.cwd, pattern)
        if n:
            self.reload_memory()
        return p, n

    def run_task(self, task: str) -> str:
        """
        Run a single task to completion (non-interactive).

        Returns the agent's final text response.
        Used by spawn_agent and the CLI `run` command.
        """
        self._add_user(task)
        return self._loop(interactive=False)

    def chat(self, user_input: str) -> str:
        """
        Send one user message in an ongoing interactive session.

        History is preserved between calls.
        Returns the agent's final text response for this turn.
        """
        self._add_user(user_input)
        return self._loop(interactive=True)

    # ------------------------------------------------------------------ #
    #  Core loop
    # ------------------------------------------------------------------ #

    def _loop(self, interactive: bool) -> str:
        """
        Agentic loop: call LLM → parse tool calls → execute → repeat.
        Returns the final response text.
        """
        final_response = ""
        self._stop_requested = False       # a stale stop must not kill a new task
        start_turns = self._turns          # turns used by *this* task only
        verify_nudged = False              # self-verification fires at most once per task
        budget_escalated = False           # uncertainty escalation fires at most once per task
        repair_nudged = False              # tool-call reformat fires at most once per task

        try:
            while self._turns < self.max_turns:
                if self._stop_requested:
                    self._stop_requested = False
                    final_response = "[stopped by user]"
                    self._last_run_ok = False
                    break

                # Mid-task steering: deliver queued user messages before the
                # next LLM call so the agent reads them this turn.
                for queued in self._drain_queued():
                    self._add_user(
                        "[user steering note - read this before continuing, it "
                        f"overrides earlier instructions where they conflict]\n{queued}"
                    )
                    self._emit("info", text="steering note delivered to the agent")
                    if interactive:
                        print_info("(steering note delivered)")

                self._turns += 1
                prev_turn_tokens = self._last_turn_tokens
                self._last_turn_tokens = 0   # reset counter for this turn

                ctx_ratio = self._fill_ratio()
                if interactive:
                    print_turn_divider(self._turns, self._total_tokens,
                                       prev_turn_tokens, ctx_ratio=ctx_ratio)
                self._emit("turn", turn=self._turns,
                           total_tokens=self._total_tokens,
                           ctx_ratio=round(min(ctx_ratio, 1.0), 3))

                self._audit.set_turn(self._turns)

                # ---- uncertainty escalation ------------------------------
                # When the task exceeds its turn budget, stop guessing:
                # interactively ask the user whether to keep going; in
                # non-interactive mode tell the model to surface blockers.
                task_turns = self._turns - start_turns
                if not budget_escalated and task_turns > self.turn_budget:
                    budget_escalated = True
                    if interactive:
                        print_warning(
                            f"This task has used {task_turns} turns "
                            f"(budget: {self.turn_budget})."
                        )
                        if not confirm("  Keep going?"):
                            final_response = (
                                f"[stopped by user after {task_turns} turns - "
                                "task exceeded its turn budget]"
                            )
                            self._last_run_ok = False
                            break
                    else:
                        self._emit(
                            "info",
                            text=f"Turn budget exceeded ({task_turns}/{self.turn_budget}) - "
                                 "asking the agent to surface blockers instead of guessing.",
                        )
                        self._add_user(
                            f"[turn budget] You have used {task_turns} turns on this "
                            f"task (budget: {self.turn_budget}). If you are stuck or "
                            "uncertain, STOP guessing: summarise what you tried, state "
                            "exactly what is blocking you, and ask for guidance instead "
                            "of continuing to experiment. If you are genuinely close to "
                            "done, finish with the minimal remaining steps."
                        )

                # ---- context-budget check --------------------------------
                self._maybe_compact(interactive=interactive)

                # ---- call LLM -------------------------------------------
                messages = self._build_messages()
                response = self._call_llm(messages, interactive=interactive)

                if self._stop_requested:
                    # Stopped mid-generation: keep the partial text, run nothing
                    self._stop_requested = False
                    self._add_assistant(response)
                    final_response = response or "[stopped by user]"
                    self._last_run_ok = False
                    break

                # ---- parse tool calls ------------------------------------
                # Pass the known tool names so the lenient, name-gated formats
                # (bare JSON and ```json / bare fences) are recognised without
                # mistaking a JSON example in prose for a call.
                calls = parse_tool_calls(
                    response, tool_names=set(TOOL_REGISTRY) - self.disabled_tools)

                if not calls:
                    # Self-verification: don't accept a final answer while code
                    # changes sit unverified - nudge the agent to check its work.
                    # Fires at most once per task to avoid infinite loops.
                    if (
                        self.self_verify
                        and not verify_nudged
                        and self._unverified_writes
                        and self._turns < self.max_turns
                    ):
                        verify_nudged = True
                        files = ", ".join(sorted(self._unverified_writes))
                        self._add_assistant(response)
                        self._add_user(
                            f"[self-verification] You changed code files ({files}) "
                            "but have not verified them. Before giving your final "
                            "answer: run run_tests if this project has a test "
                            "suite, otherwise re-read the changed files to check "
                            "for mistakes. Then give your final answer."
                        )
                        if interactive:
                            print_info(
                                "(self-verification: asking agent to verify "
                                f"changes to {files})"
                            )
                        continue

                    # Repair turn: the response looks like a tool call that
                    # failed to parse (a marker/fence, or a name+args object the
                    # lenient parser still could not recover - malformed JSON,
                    # an unknown tool name, or Python-style tool_code). Re-prompt
                    # once with the exact format instead of printing the broken
                    # call as the final answer.
                    if (
                        not repair_nudged
                        and self._turns < self.max_turns
                        and looks_like_tool_attempt(response)
                    ):
                        repair_nudged = True
                        self._add_assistant(response)
                        self._add_user(
                            "[tool-call format] That looked like a tool call, "
                            "but I could not parse it. Re-emit it in EXACTLY "
                            "this format and nothing else:\n"
                            "<tool_call>\n"
                            '{"name": "TOOL_NAME", "args": {"PARAM": "VALUE"}}\n'
                            "</tool_call>\n"
                            "Use one of the available tools by its exact name. "
                            "If you did NOT mean to call a tool, ignore this and "
                            "give your final answer as plain text."
                        )
                        if interactive:
                            print_info(
                                "(re-prompting: tool call could not be parsed)"
                            )
                        continue

                    # No tool calls → this is the final answer
                    if not interactive and self.on_event is None:
                        print_assistant_response(response, name=self.name)
                    final_response = response
                    self._add_assistant(response)
                    break

                # ---- there are tool calls --------------------------------
                # Show the non-tool-call text parts first
                if interactive:
                    segments = split_response(response, calls)
                    for seg in segments:
                        if isinstance(seg, str) and seg.strip():
                            console.print(seg.strip())

                self._add_assistant(response)

                # Execute tools - run non-destructive batches in parallel
                result_blocks = self._execute_tools(calls, interactive=interactive)

                # Feed all results back as a user message, compressing large
                # outputs when the context is filling up
                result_blocks = self._compress_results(result_blocks)
                combined = "\n\n".join(result_blocks)
                self._add_user(combined)

                # Circuit breaker: a tool that keeps failing identically wastes
                # the whole turn budget - stop and hand control back instead.
                if self._abort_streak_tool:
                    tool = self._abort_streak_tool
                    self._abort_streak_tool = None
                    streak = self._consecutive_errors.get(tool, 0)
                    final_response = (
                        f"[circuit breaker: {tool} failed {streak} times in a "
                        "row - stopping so you can take a look instead of "
                        "burning more turns. The conversation is intact; "
                        "adjust the approach and continue.]"
                    )
                    print_warning(final_response)
                    self._emit("info", text=final_response)
                    self._last_run_ok = False
                    break

            else:
                msg = f"[max_turns={self.max_turns} reached]"
                print_warning(msg)
                final_response = msg
                self._last_run_ok = False

        except KeyboardInterrupt:
            if interactive and self._messages:
                if self.mode == SessionMode.PRIVACY:
                    print_info(
                        "(interrupted - privacy mode, no checkpoint saved; "
                        "/resume is unavailable.)"
                    )
                else:
                    self.save_checkpoint()
                    print_info(
                        "(interrupted - progress saved. "
                        "Type /resume to continue or start a new task.)"
                    )
            raise

        else:
            # Clean finish - discard any stale checkpoint
            self.clear_checkpoint()

        return final_response

    # ------------------------------------------------------------------ #
    #  Parallel tool dispatch
    # ------------------------------------------------------------------ #

    def _execute_tools(self, calls: list, interactive: bool) -> list[str]:
        """
        Execute a list of tool calls and return their XML result blocks.

        Groups consecutive non-destructive calls and runs each group in parallel
        with a ``ThreadPoolExecutor``.  Destructive calls are always run alone,
        in order, to avoid unintended interactions (file corruption, overlapping
        shell commands, etc.).

        The grouping strategy preserves the original ordering of destructive
        calls relative to non-destructive ones: given [read, read, write, read],
        the two leading reads run in parallel, then the write runs alone, then
        the final read runs alone.  This is conservative but safe.
        """
        result_blocks: list[str] = []

        # Split into segments: each segment is (is_destructive, [calls])
        segments: list[tuple[bool, list]] = []
        for call in calls:
            td = TOOL_REGISTRY.get(call.name)
            destructive = td.destructive if td else True
            if segments and segments[-1][0] == destructive:
                segments[-1][1].append(call)
            else:
                segments.append((destructive, [call]))

        for destructive, group in segments:
            if destructive or len(group) == 1:
                # Serial execution
                for call in group:
                    result = self._execute_tool(call, interactive=interactive)
                    result_blocks.append(result.to_xml(call.name))
            else:
                # Parallel execution for non-destructive batch. The pool is
                # shut down without waiting so one hung tool (network fetch,
                # slow disk) cannot block the whole batch past the deadline -
                # the stuck thread is abandoned and reported as a timeout.
                ordered: dict[int, str] = {}
                pool = ThreadPoolExecutor(max_workers=min(len(group), 8))
                futures = {
                    pool.submit(self._execute_tool, call, False): (i, call)
                    for i, call in enumerate(group)
                }
                try:
                    for fut in as_completed(futures,
                                            timeout=self._PARALLEL_BATCH_TIMEOUT_S):
                        i, call = futures[fut]
                        try:
                            result = fut.result()
                        except Exception as exc:
                            result = ToolResult.error(f"Parallel execution error: {exc}")
                        # Print results in original order when interactive
                        ordered[i] = result.to_xml(call.name)
                        if interactive:
                            print_tool_call(call.name, call.args)
                            print_tool_result(call.name, result, verbose=self.verbose)
                except TimeoutError:
                    for fut, (i, call) in futures.items():
                        if i not in ordered:
                            fut.cancel()
                            result = ToolResult.error(
                                f"{call.name} did not finish within "
                                f"{self._PARALLEL_BATCH_TIMEOUT_S}s (parallel "
                                "batch timeout) - try a narrower target."
                            )
                            ordered[i] = result.to_xml(call.name)
                            if interactive:
                                print_tool_error(call.name, result.output)
                finally:
                    pool.shutdown(wait=False)
                for i in range(len(group)):
                    result_blocks.append(ordered[i])

        return result_blocks

    # Wall-clock deadline for one parallel batch of non-destructive tools
    _PARALLEL_BATCH_TIMEOUT_S = 120

    # ------------------------------------------------------------------ #
    #  Compaction
    # ------------------------------------------------------------------ #

    def _ctx_window_tokens(self) -> int:
        """Estimated context window size in tokens."""
        try:
            from localm.config import load_config
            return load_config().get("n_ctx", _DEFAULT_CTX_TOKENS)
        except Exception:
            return _DEFAULT_CTX_TOKENS

    def _fill_ratio(self) -> float:
        """Fraction of estimated context window currently consumed (0.0 - 1.0+)."""
        estimated = self.context_chars() // 4
        return estimated / max(1, self._ctx_window_tokens())

    def undo(self) -> str | None:
        """
        Revert the last undoable file operation (write_file, edit_file, patch_file).

        Returns a human-readable summary of what was restored, or None if the
        undo stack is empty.
        """
        if not self._undo_stack:
            return None
        entry = self._undo_stack.pop()
        path: Path    = entry["path"]
        old: bytes | None = entry["old_content"]
        tool: str     = entry["tool"]
        try:
            if old is None:
                # File didn't exist before - delete it
                if path.exists():
                    path.unlink()
                return f"Undid {tool}: deleted {path} (file was new)"
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(old)
                lines = old.count(b"\n") + 1
                return f"Undid {tool}: restored {path} ({lines} lines)"
        except Exception as e:
            return f"Undo failed: {e}"

    def compact(self) -> bool:
        """
        Summarise old conversation history into a single condensed exchange.

        Keeps the 4 most recent messages verbatim (= last 2 full turns) so
        the agent retains immediate context.  Everything older is replaced by
        a summary produced by a direct backend call (no tools, no loop).

        Returns True if compaction happened, False if there was nothing to compact.
        """
        return self._compact_history()

    # GBNF grammar for structured compaction output.
    # Produces: {"summary":"...","changed_files":["..."],"open_tasks":["..."]}
    _COMPACT_GRAMMAR = r"""
root   ::= "{" ws "\"summary\"" ws ":" ws string ws "," ws "\"changed_files\"" ws ":" ws str-array ws "," ws "\"open_tasks\"" ws ":" ws str-array ws "}"
str-array ::= "[" ws (string ("," ws string)*)? ws "]"
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
ws     ::= [ \t\n\r]*
"""

    def _compact_history(self) -> bool:
        keep_n = 4   # last 4 messages kept verbatim (~2 turns)
        if len(self._messages) <= keep_n:
            return False   # not enough history to compact

        older  = self._messages[:-keep_n]
        recent = self._messages[-keep_n:]

        # Build a concise conversation excerpt for the summariser
        excerpt_parts = []
        for m in older:
            role    = m["role"].upper()
            content = m.get("content", "")
            if isinstance(content, list):          # multipart messages
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            excerpt_parts.append(f"{role}: {content[:600]}")
        excerpt = "\n\n".join(excerpt_parts)

        # When the backend supports GBNF grammar sampling, request a structured
        # JSON summary so the compacted message is always machine-parseable.
        use_json = getattr(self.backend, "supports_grammar", False)

        if use_json:
            summary_prompt = (
                "Summarise the following coding session as JSON with exactly three fields:\n"
                '  "summary": a concise narrative (≤200 words) of decisions, edits, and fixes\n'
                '  "changed_files": list of file paths that were created or modified\n'
                '  "open_tasks": list of tasks or problems still unresolved\n\n'
                "Respond with valid JSON only - no prose outside the JSON object.\n\n"
                f"{excerpt}"
            )
        else:
            summary_prompt = (
                "Produce a concise summary (≤300 words) of the following coding session. "
                "Focus on: decisions made, files created or edited, errors and fixes, "
                "and any open problems or next steps.\n\n"
                f"{excerpt}"
            )

        try:
            call_kwargs: dict = {"max_tokens": 400}
            if use_json:
                call_kwargs["grammar"] = self._COMPACT_GRAMMAR.strip()
            raw = self.backend.chat(
                [{"role": "user", "content": summary_prompt}],
                **call_kwargs,
            )
        except Exception:
            return False   # best-effort; don't crash on summary failure

        # Parse structured output if we requested JSON
        if use_json:
            try:
                data = json.loads(raw)
                summary_text = data.get("summary", raw)
                changed = data.get("changed_files", [])
                tasks   = data.get("open_tasks", [])
                summary_lines = [summary_text]
                if changed:
                    summary_lines.append("\nChanged files: " + ", ".join(changed))
                if tasks:
                    summary_lines.append("\nOpen tasks:\n" + "\n".join(f"- {t}" for t in tasks))
                summary = "\n".join(summary_lines)
            except Exception:
                summary = raw   # fall back to raw text on parse failure
        else:
            summary = raw

        self._messages = [
            {"role": "user",      "content": f"[Session summary]\n{summary}"},
            {"role": "assistant", "content": "Understood. Continuing from this context."},
            *recent,
        ]
        return True

    # Tool-result compression thresholds
    _COMPRESS_FILL_RATIO = 0.50   # only compress when context is half full
    _COMPRESS_MIN_CHARS  = 6000   # blocks smaller than this are never touched
    _COMPRESS_HEAD_CHARS = 3000   # kept from the start of a compressed block
    _COMPRESS_TAIL_CHARS = 1000   # kept from the end (errors usually live here)

    def _compress_results(self, blocks: list[str]) -> list[str]:
        """
        Shrink oversized tool-result blocks once the context window is more
        than half full. Keeps the head (context) and tail (errors, summaries)
        of each block and marks the elision, so the agent knows output was
        dropped and can re-read specific files if it needs the middle.
        """
        if self._fill_ratio() < self._COMPRESS_FILL_RATIO:
            return blocks
        compressed = []
        for block in blocks:
            if len(block) <= self._COMPRESS_MIN_CHARS:
                compressed.append(block)
                continue
            dropped = len(block) - self._COMPRESS_HEAD_CHARS - self._COMPRESS_TAIL_CHARS
            compressed.append(
                block[: self._COMPRESS_HEAD_CHARS]
                + f"\n[... {dropped} chars of tool output elided to save context - "
                "re-run the tool on a narrower target if you need the middle ...]\n"
                + block[-self._COMPRESS_TAIL_CHARS:]
            )
        return compressed

    def _maybe_compact(self, interactive: bool) -> None:
        """Check fill ratio and warn or auto-compact as appropriate."""
        ratio = self._fill_ratio()
        if interactive:
            if ratio >= _COMPACT_WARN_RATIO and not getattr(self, "_compact_warned", False):
                print_warning(
                    f"Context is ~{ratio:.0%} full. "
                    "Use [bold]/compact[/bold] to summarise old turns."
                )
                self._compact_warned = True   # warn once per session
        else:
            # Non-interactive (run_task): auto-compact silently at 90%
            if ratio >= _COMPACT_AUTO_RATIO:
                self._compact_history()

    # ------------------------------------------------------------------ #
    #  LLM call
    # ------------------------------------------------------------------ #

    # Tool-call wrapper markers, canonical and mangled finetune variants
    _TC_OPENERS = ("<tool_call", "<|tool_call")
    _TC_CLOSERS = ("</tool_call>", "<tool_call|>", "<|/tool_call>", "<|tool_call|>")

    @classmethod
    def _stream_hiding_tool_calls(cls, pieces):
        """
        Yield displayable text tokens from a stream, silently buffering
        tool-call blocks (canonical <tool_call>...</tool_call> and mangled
        <|tool_call>...<tool_call|> variants) so raw JSON never hits the
        terminal.

        Yields (token, is_hidden) pairs where is_hidden=True means the
        token belongs to a tool-call block and should not be displayed.
        """
        def _find_first(haystack, needles, offset=0):
            best = -1
            best_len = 0
            for needle in needles:
                idx = haystack.find(needle, offset)
                if idx != -1 and (best == -1 or idx < best):
                    best, best_len = idx, len(needle)
            return best, best_len

        def _partial_opener_at_end(haystack):
            """Length of a trailing fragment that could grow into an opener."""
            max_keep = max(len(n) for n in cls._TC_OPENERS) - 1
            for k in range(min(max_keep, len(haystack)), 0, -1):
                tail = haystack[-k:]
                if any(needle.startswith(tail) for needle in cls._TC_OPENERS):
                    return k
            return 0

        buf = ""
        in_call = False
        for piece in pieces:
            buf += piece
            while True:
                if not in_call:
                    start, _ = _find_first(buf, cls._TC_OPENERS)
                    if start == -1:
                        # Hold back a tail that might be a split opener
                        keep = _partial_opener_at_end(buf)
                        if len(buf) > keep:
                            yield buf[:len(buf) - keep], False
                            buf = buf[len(buf) - keep:]
                        break
                    if start > 0:
                        yield buf[:start], False
                    buf = buf[start:]
                    in_call = True
                else:
                    # Search past the opener so <|tool_call|> as an opener
                    # is not immediately matched as its own closer
                    end, end_len = _find_first(buf, cls._TC_CLOSERS, 2)
                    if end == -1:
                        break
                    end += end_len
                    yield buf[:end], True
                    buf = buf[end:]
                    in_call = False
        if buf:
            yield buf, in_call   # unclosed tag at stream end - display as-is

    def _call_llm(self, messages: list[dict], interactive: bool) -> str:
        if self.on_event is not None:
            # Event-sink mode (GUI/web session): stream tokens to the sink,
            # keep the server terminal quiet.
            full = ""
            for piece, hidden in self._stream_hiding_tool_calls(
                self.backend.chat_stream(messages, **self.gen_kwargs)
            ):
                full += piece
                if not hidden:
                    self._emit("token", text=piece)
                if self._stop_requested:
                    break
            self._accumulate_usage()
            self._audit.llm(full, tokens=self._total_tokens)
            return full
        if interactive:
            print_thinking()
            print_assistant_label(self.name)
            full = ""
            try:
                for piece, hidden in self._stream_hiding_tool_calls(
                    self.backend.chat_stream(messages, **self.gen_kwargs)
                ):
                    full += piece
                    if not hidden:
                        print_streaming_token(piece)
                print_streaming_done()
            except KeyboardInterrupt:
                print_streaming_done()
                print_info("(interrupted)")
            self._accumulate_usage()
            self._audit.llm(full, tokens=self._total_tokens)
            return full
        else:
            # Silent call - used by sub-agents and non-interactive mode
            result = self.backend.chat(messages, **self.gen_kwargs)
            self._accumulate_usage()
            self._audit.llm(result, tokens=self._total_tokens)
            return result

    def _accumulate_usage(self) -> None:
        """Pull token counts from the backend's last call and add to the session total."""
        usage = getattr(self.backend, "last_usage", {})
        n = usage.get("total_tokens", 0)
        if n:
            self._total_tokens += n
            self._last_turn_tokens += n

    # ------------------------------------------------------------------ #
    #  Scope enforcement
    # ------------------------------------------------------------------ #

    def _scope_rel(self, value: str) -> Optional[str]:
        """
        Resolve a path/glob arg to a cwd-relative POSIX string for scope
        matching, or return None if it escapes cwd.

        Relative paths are joined onto cwd; absolute paths are accepted only
        when they live inside cwd (an in-cwd absolute path that matches the
        scope must pass - BUG-6). Glob metacharacters in *value* (e.g.
        ``**/*.py`` for grep/search_replace) survive resolution: they are kept
        verbatim in the relative string and matched against the scope as-is.
        """
        raw = str(value).replace("\\", "/")
        p = Path(raw)
        cwd = self.cwd.resolve()
        if p.is_absolute():
            try:
                # No resolve(): the path may contain glob chars or not exist.
                rel = Path(raw).relative_to(cwd)
            except ValueError:
                # Try once more against the resolved abs form for symlinks etc.
                try:
                    rel = Path(raw).resolve().relative_to(cwd)
                except ValueError:
                    return None   # outside cwd
            return rel.as_posix()
        # Relative: collapse any leading ./ and reject cwd escapes (../).
        rel_posix = (Path(".") / raw).as_posix()
        if rel_posix.startswith("./"):
            rel_posix = rel_posix[2:]
        parts = [seg for seg in rel_posix.split("/") if seg not in ("", ".")]
        if ".." in parts:
            return None   # escapes cwd
        return "/".join(parts)

    def _scope_allows(self, value: str) -> bool:
        """True if *value* (a path or glob arg) is within the active scope."""
        rel = self._scope_rel(value)
        if rel is None:
            return False
        return _scope_pattern(self.scope).match(rel) is not None

    def _scope_violation(self, call: ToolCall) -> Optional[str]:
        """
        Return the first in-scope-checked arg value that falls outside the
        active scope, or None if the call is allowed.

        Defaults to checking the ``path`` arg; ``_SCOPE_PATH_ARGS`` overrides
        this for tools whose primary target is a ``glob`` or ``output_path``
        arg (and may add ``path`` alongside it).
        """
        arg_names = _SCOPE_PATH_ARGS.get(call.name, ("path",))
        for name in arg_names:
            value = call.args.get(name)
            if value:
                if not self._scope_allows(str(value)):
                    return str(value)
        return None

    # ------------------------------------------------------------------ #
    #  Tool execution
    # ------------------------------------------------------------------ #

    def _execute_tool(self, call: ToolCall, interactive: bool) -> ToolResult:
        # Hard gate: a tool disabled for this session (e.g. run_shell for a
        # restricted, shareable coder key) can never execute, whatever the model
        # emits. This is the security boundary - the prompt/parse exclusions below
        # are only there so the model does not waste turns trying.
        if call.name in self.disabled_tools:
            result = ToolResult.error(
                f"'{call.name}' is disabled for this session and was not run.")
            if interactive:
                print_tool_error(call.name, result.output)
            return result

        tool_def = TOOL_REGISTRY.get(call.name)

        if tool_def is None:
            result = ToolResult.error(
                f"Unknown tool '{call.name}'. "
                f"Available: {', '.join(TOOL_REGISTRY)}"
            )
            if interactive:
                print_tool_error(call.name, result.output)
            return result

        self._audit.tool_call(call.name, call.args)
        if interactive:
            print_tool_call(call.name, call.args)
        self._emit("tool_call", tool=call.name, args=call.args)

        # Patch-mode: intercept write tools, accumulate diffs, don't touch disk.
        # A write tool the interceptor can't express as a diff must NOT fall
        # through to a real disk write - patch-mode promises no changes.
        if self.patch_mode and call.name in _UNDOABLE_TOOLS:
            chunk = self._patch_mode_intercept(call)
            if chunk is not None:
                self._patch_chunks.append(chunk)
                result = ToolResult.success(
                    f"[patch-mode] diff captured for {call.args.get('path', '?')}",
                    summary=f"[patch-mode] {call.name}",
                )
                if interactive:
                    console.print("    [dim cyan][patch-mode] diff captured[/dim cyan]")
            else:
                result = ToolResult.error(
                    f"[patch-mode] {call.name} cannot be captured as a diff "
                    "(no change, or unsupported operation) - skipped. Use "
                    "write_file/edit_file/patch_file in patch mode."
                )
                if interactive:
                    console.print("    [dim yellow][patch-mode] skipped[/dim yellow]")
            return result

        # Scope check - reject file operations that fall outside the active glob
        if self.scope and call.name in _SCOPED_TOOLS:
            offending = self._scope_violation(call)
            if offending is not None:
                result = ToolResult.error(
                    f"'{offending}' is outside the active scope '{self.scope}'. "
                    "Only files matching this glob pattern can be accessed."
                )
                if interactive:
                    print_tool_error(call.name, result.output)
                return result

        # Dry-run: show destructive calls but don't execute them
        if self.dry_run and tool_def.destructive:
            result = ToolResult.success(
                f"[dry-run] {call.name} - skipped",
                summary=f"[dry-run] {call.name}",
            )
            if interactive:
                console.print("    [dim yellow][dry-run] skipped[/dim yellow]")
            return result

        # Network policy: model-initiated network tools are governed by
        # net_mode (off = fail fast, ask = approval flow, allow = run).
        net_mode = None
        if call.name in _NETWORK_TOOLS:
            from localm.netpolicy import network_mode
            net_mode = network_mode()
            if net_mode == "off":
                result = ToolResult.error(
                    "Network access is disabled (net_mode=off). The user can "
                    "enable it with: localm config net_mode ask"
                )
                if interactive:
                    print_tool_error(call.name, result.output)
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="blocked by network policy (net_mode=off)")
                return result

        # Confirmation for destructive tools (diff preview for write_file)
        # and for network tools when net_mode is "ask"
        needs_confirm = (
            (tool_def.destructive or net_mode == "ask") and (
                not self.auto_approve or call.name in self.always_confirm
            )
        )
        if needs_confirm and (interactive or self.confirm_handler is not None):
            if self.confirm_handler is not None:
                approved = self.confirm_handler(call)
            else:
                approved = self._confirm_tool(call)
            if not approved:
                result = ToolResult.error("Rejected by user.")
                if interactive:
                    print_tool_result(call.name, result, verbose=False)
                self._emit("tool_result", tool=call.name, ok=False,
                           summary="rejected by user")
                return result

        # Snapshot file content before undoable writes so /undo can restore
        # it and the changed-files tracker can diff against the original
        snapshot_old: bytes | None = None
        if call.name in _UNDOABLE_TOOLS:
            path_arg = call.args.get("path", "")
            if path_arg:
                abs_path = (self.cwd / path_arg).resolve()
                try:
                    snapshot_old = abs_path.read_bytes() if abs_path.is_file() else None
                except Exception:
                    snapshot_old = None
                self._undo_stack.append({
                    "path": abs_path,
                    "old_content": snapshot_old,
                    "tool": call.name,
                })

        # Inject hidden runtime args into specific tools
        args = dict(call.args)
        if call.name == "spawn_agent":
            args["_parent_agent"] = self
        if call.name in ("run_shell", "fetch_url", "web_search", "generate_image") \
                and self.mode == SessionMode.PRIVACY:
            args["_privacy"] = True

        try:
            result = tool_def.fn(self.cwd, **args)
        except TypeError as e:
            result = ToolResult.error(f"Bad arguments for {call.name}: {e}")
        except Exception as e:
            result = ToolResult.error(f"Tool error: {e}")

        # Track consecutive failures and inject escalating recovery hints;
        # at 4 identical failures the circuit breaker stops the task after
        # this batch (checked in _loop) instead of burning the turn budget.
        if not result.ok:
            streak = self._consecutive_errors.get(call.name, 0) + 1
            self._consecutive_errors[call.name] = streak
            if streak == 2:
                result = ToolResult.error(
                    result.output
                    + "\n\n[Hint: this tool has failed twice in a row. "
                    "Try a different approach - check paths, arguments, or preconditions.]"
                )
            elif streak >= 3:
                result = ToolResult.error(
                    result.output
                    + f"\n\n[Warning: {call.name} has failed {streak} times consecutively. "
                    "Step back and reconsider your strategy. "
                    "Consider reading the relevant files first, "
                    "or breaking the task into smaller steps.]"
                )
            if streak >= 4:
                self._abort_streak_tool = call.name
        else:
            self._consecutive_errors.pop(call.name, None)

        self._audit.tool_result(call.name, result.ok, result.summary)
        if interactive:
            print_tool_result(call.name, result, verbose=self.verbose)
        self._emit("tool_result", tool=call.name, ok=result.ok,
                   summary=result.summary, output=result.output[:4000])

        # Incremental map refresh after file-mutating tools
        if result.ok and call.name in _MUTATING_TOOLS:
            self._refresh_map_for_tool(call)

        # Self-verification bookkeeping: remember code files changed on disk,
        # forget them once the agent runs the test suite (or a test command)
        if result.ok and not self.dry_run and not self.patch_mode:
            if call.name in _UNDOABLE_TOOLS:
                path_arg = call.args.get("path", "")
                if path_arg:
                    self._record_changed_file(path_arg, snapshot_old, call.name)
                if path_arg and Path(path_arg).suffix.lower() in _CODE_EXTS:
                    self._unverified_writes.add(path_arg)
            elif call.name == "run_tests":
                self._unverified_writes.clear()
            elif call.name == "run_shell":
                cmd = str(call.args.get("command", "")).lower()
                if any(marker in cmd for marker in _TEST_COMMAND_MARKERS):
                    self._unverified_writes.clear()

        return result

    def _patch_mode_intercept(self, call: ToolCall) -> Optional[str]:
        """
        Compute a unified diff for a write/edit/patch call without touching disk.

        Returns the diff string, or None if the diff cannot be computed.
        """
        import difflib as _difflib

        path_arg = call.args.get("path", "")
        abs_path = (self.cwd / path_arg).resolve() if path_arg else None
        old_text = ""
        if abs_path and abs_path.is_file():
            try:
                old_text = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass

        if call.name == "write_file":
            new_text = call.args.get("content", "")
        elif call.name == "edit_file":
            old_str = call.args.get("old", "")
            new_str = call.args.get("new", "")
            new_text = old_text.replace(old_str, new_str, 1)
        elif call.name == "patch_file":
            # diff is already a unified diff - wrap it as-is
            diff = call.args.get("diff", "")
            return diff if diff else None
        else:
            return None

        diff_lines = list(_difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{path_arg}",
            tofile=f"b/{path_arg}",
        ))
        return "".join(diff_lines) if diff_lines else None

    def flush_patch(self, output_path: Optional[Path] = None) -> str:
        """
        Return the accumulated unified diff (and optionally write it to a file).

        Clears the internal patch buffer.
        """
        content = "\n".join(c for c in self._patch_chunks if c)
        self._patch_chunks.clear()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        return content

    def _confirm_tool(self, call: ToolCall) -> bool:
        """
        Ask the user to approve a destructive tool call.

        For *write_file*, shows a coloured unified diff of the proposed change
        before the prompt so the user can see exactly what will happen.
        For all other destructive tools, falls back to a plain y/N prompt.
        """
        if call.name == "write_file":
            path_arg = call.args.get("path", "")
            new_content = call.args.get("content", "")
            abs_path = (self.cwd / path_arg).resolve() if path_arg else None
            old_content = ""
            if abs_path and abs_path.is_file():
                try:
                    old_content = abs_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            print_diff_preview(old_content, new_content, path_label=path_arg)
            return confirm_diff(path_arg or "file")

        if call.name == "edit_file":
            path_arg    = call.args.get("path", "")
            old_string  = call.args.get("old", "")
            new_string  = call.args.get("new", "")
            abs_path    = (self.cwd / path_arg).resolve() if path_arg else None
            old_content = ""
            if abs_path and abs_path.is_file():
                try:
                    old_content = abs_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    pass
            new_content = old_content.replace(old_string, new_string, 1)
            print_diff_preview(old_content, new_content, path_label=path_arg)
            return confirm_diff(path_arg or "file")

        if call.name == "patch_file":
            path_arg = call.args.get("path", "")
            patch    = call.args.get("diff", "")
            # The patch is already a unified diff - display it directly
            from .display import console as _con
            from rich.syntax import Syntax
            _con.print()
            _con.print(Syntax(patch, "diff", theme="monokai", line_numbers=False))
            return confirm_diff(path_arg or "file")

        return confirm(f"  Allow {call.name}?")

    def _refresh_map_for_tool(self, call: ToolCall) -> None:
        """Update the project map for files touched by a write/edit tool call."""
        path_arg = call.args.get("path")
        if path_arg:
            abs_path = (self.cwd / path_arg).resolve()
            self._project_map.refresh_file(abs_path)
            # Regenerate system prompt with updated map
            self._system_prompt = build_system_prompt(
                self.cwd,
                agent_name=self.name,
                project_map=self._project_map,
                memory=self._memory,
                model_name=self._model_name,
                disabled_tools=self.disabled_tools,
            )

    # ------------------------------------------------------------------ #
    #  Message management
    # ------------------------------------------------------------------ #

    def _build_messages(self) -> list[dict]:
        """Build the full message list with system prompt prepended."""
        return [
            {"role": "system", "content": self._system_prompt},
            *self._messages,
        ]

    def _add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})
        # Only log human-originating messages (skip tool results, which are very long)
        if not content.startswith("<tool_result"):
            self._audit.user(content)

    def _add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    # ------------------------------------------------------------------ #
    #  Context stats
    # ------------------------------------------------------------------ #

    def context_chars(self) -> int:
        """Rough estimate of total characters in the current context."""
        total = len(self._system_prompt)
        for m in self._messages:
            content = m.get("content", "")
            if isinstance(content, list):
                total += sum(len(p.get("text", "")) for p in content if isinstance(p, dict))
            else:
                total += len(content)
        return total

    def save_history(self, path: Path) -> None:
        import json as _json
        path.write_text(
            _json.dumps(self._messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def close(self) -> Path | None:
        """
        Finalise the session.

        - Closes the audit log (``log`` and ``full`` modes).
        - Writes a Markdown transcript to ``.localcoder/sessions/`` in
          ``full`` mode.

        Returns the path of the Markdown file, or None.
        Called automatically by the CLI's ``finally`` block.
        """
        self._audit.close()
        if self.mode == SessionMode.FULL:
            return self._write_session_markdown()
        return None

    def _write_session_markdown(self) -> Path:
        """
        Write a human-readable Markdown transcript of the session to
        ``.localcoder/sessions/<YYYY-MM-DD_HHMMSS>.md`` inside the project
        working directory.

        Tool-result messages (which are large XML blobs) are skipped.
        Tool calls embedded in assistant messages are extracted and listed
        as bullet points.
        """
        import re as _re
        import time as _time

        ts_label = _time.strftime("%Y-%m-%d_%H%M%S")
        ts_human = _time.strftime("%Y-%m-%d %H:%M:%S")

        out_dir = self.cwd / ".localcoder" / "sessions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ts_label}.md"

        tokens_line = (
            f"**Tokens (billed est.)**: ~{self._total_tokens:,}  "
            if self._total_tokens
            else ""
        )

        lines: list[str] = [
            f"# localcoder Session - {ts_human}",
            "",
            f"**Model**: {self._model_name or 'unknown'}  ",
            f"**Working directory**: {self.cwd}  ",
            f"**Turns**: {self._turns}  ",
        ]
        if tokens_line:
            lines.append(tokens_line)
        lines += ["", "---", ""]

        _TC_RE = _re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>", _re.DOTALL
        )

        for msg in self._messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                # multipart - join text parts
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict)
                )

            # Skip tool-result feed-backs (huge XML blobs)
            if content.lstrip().startswith("<tool_result"):
                continue

            if role == "user":
                lines.append(f"**You**: {content[:2000]}")
                lines.append("")

            elif role == "assistant":
                # Strip tool_call blocks and extract summaries
                call_matches = _TC_RE.findall(content)
                clean = _TC_RE.sub("", content).strip()

                if clean:
                    lines.append(f"**{self.name}**: {clean[:2000]}")
                elif call_matches:
                    lines.append(f"**{self.name}**:")

                for raw_json in call_matches:
                    try:
                        import json as _json
                        obj  = _json.loads(raw_json)
                        tool = obj.get("name", "?")
                        args = obj.get("args", {})
                        # Show path/command arg if present, else first arg value
                        hint = (
                            args.get("path")
                            or args.get("command")
                            or args.get("url")
                            or (next(iter(args.values()), None) if args else None)
                        )
                        hint_str = f" `{str(hint)[:60]}`" if hint else ""
                        lines.append(f"  - `{tool}`{hint_str}")
                    except Exception:
                        lines.append(f"  - (tool call)")

                lines.append("")

            lines.append("---")
            lines.append("")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
