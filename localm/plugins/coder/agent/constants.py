# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-level constants shared across the Agent mixins: the tool-category sets
(mutating / undoable / scoped / network), the scope path-arg map, and the
compaction / repair / error-breaker thresholds. Extracted verbatim from the
former single-file agent.py."""

from __future__ import annotations

# Tools that mutate files - trigger a project map refresh after they run
# (execution.py's _refresh_map_for_tool, the only consumer of this set).
# search_replace is included, but its paths come from ToolResult.changes
# (post-call, its own dry_run-driven sweep), not from set membership alone -
# see _refresh_map_for_tool's *result* param.
#
# run_shell has no path-shaped arg at all (a `command` string only), so
# _call_target_paths() returns [] for it and no per-file refresh_file() call
# ahead of time is possible. _refresh_map_for_tool special-cases it: mark the
# whole map dirty (ProjectMap.mark_dirty) and let the next read reconcile it
# against the filesystem with a bounded stat-diff (ProjectMap._rescan_if_dirty,
# triggered from context._build_messages once per turn). No git and no
# subprocess on this path.
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "edit_files", "run_shell",
    "patch_file", "edit_notebook_cell", "search_replace",
})

# Tools whose file changes can be undone via a PRE-call snapshot: the target
# path(s) are read from the call args and their bytes snapshotted before the
# tool runs. That same pre-call snapshot feeds the changed-files tracker.
#
# search_replace is NOT here: its targets are a glob + regex sweep discovered
# only by running it, so there is no `path` arg (or _NESTED_PATH_TOOLS entry) to
# snapshot ahead of time and _call_target_paths() returns []. It is undoable and
# tracked through a separate, post-call path (ToolResult.changes, populated by
# the tool once it knows which files it touched) - see
# _PATCH_MODE_ELIGIBLE_TOOLS, _patch_mode_intercept, and _post_tool_success's
# search_replace branch.
_UNDOABLE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "edit_files", "patch_file", "edit_notebook_cell",
})

# Tools patch mode intercepts (capture a diff, never touch disk). Superset of
# _UNDOABLE_TOOLS: search_replace is patch-mode-eligible via its own dry_run,
# not via the pre-call snapshot path.
_PATCH_MODE_ELIGIBLE_TOOLS: frozenset[str] = _UNDOABLE_TOOLS | frozenset({
    "search_replace",
})

# Tools whose target paths are NESTED inside a collection arg instead of a
# top-level `path` arg. Maps tool name -> (collection arg, key within each item).
# Every path-consuming site (scope check, undo snapshot, changed-file tracker,
# map refresh) resolves paths through _call_target_paths(), so a tool listed here
# is confined and tracked exactly like a single-path one. A nested-path tool left
# out of this map passes the scope check by having no `path` arg at all.
_NESTED_PATH_TOOLS: dict[str, tuple[str, str]] = {
    "edit_files": ("edits", "path"),
}


def _call_target_paths(tool_name: str, args: dict) -> list[str]:
    """Every filesystem path a tool call targets, in call order.

    One `path` arg for most tools; for a tool in _NESTED_PATH_TOOLS, each
    item's path inside its collection arg. Malformed items are skipped here
    (the tool itself reports them); duplicates are preserved so a caller can
    count edits, and callers that need unique files de-duplicate.
    """
    nested = _NESTED_PATH_TOOLS.get(tool_name)
    if nested is None:
        value = args.get("path", "")
        return [str(value)] if value else []
    coll_arg, item_key = nested
    items = args.get(coll_arg)
    if not isinstance(items, list):
        return []
    paths = []
    for item in items:
        if isinstance(item, dict):
            value = item.get(item_key)
            if value:
                paths.append(str(value))
    return paths

# File-access tools whose target path must match the active scope glob. Keys on
# the `path` arg; a tool whose real target is a `glob`/`output_path` arg has that
# checked too (see _SCOPE_PATH_ARGS). run_shell is unscoped (see _INTENTIONALLY_UNSCOPED).
_SCOPED_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "edit_files", "patch_file",
    "list_dir", "tree",
    # The rest of the file-reading/writing tools.
    "grep", "search_files", "search_replace", "read_env",
    "edit_notebook_cell", "generate_image", "browser_screenshot",
})

# Tools NOT confined by the scope glob: git_diff / git_log take a git
# PATHSPEC (not a filesystem path), and run_tests / run_shell / the background-shell
# trio EXECUTE a process, which a path-arg check cannot confine. Any OTHER registry
# tool with a path-like arg MUST be in _SCOPED_TOOLS above; a contract test enforces
# that, so a new file tool is a test failure rather than unconfined by omission.
_INTENTIONALLY_UNSCOPED: frozenset[str] = frozenset({
    "run_shell", "run_tests", "git_diff", "git_log",
    "run_shell_background", "check_shell_job", "kill_shell_job",
})

# The subset of _INTENTIONALLY_UNSCOPED that EXECUTES a process: git_diff and
# git_log only read. These tools get a one-per-session notice plus a best-effort
# argv path check (see _ExecutionMixin._warn_shell_outside_scope). Warn, never
# block. run_shell_background is here for the same reason as run_shell: it runs
# the user's command line, it just does not wait for it.
_SHELL_UNSCOPED_TOOLS: frozenset[str] = frozenset({
    "run_shell", "run_tests", "run_shell_background",
})

# Args of the shell tools that can carry a path. run_shell has a whole command
# line to tokenise; run_tests takes a target path plus free-form extra args.
# check_shell_job / kill_shell_job take only a job id, so they start no new
# command and have no path to flag.
_SHELL_COMMAND_ARGS: dict[str, tuple[str, ...]] = {
    "run_shell": ("command",),
    "run_shell_background": ("command",),
    "run_tests": ("path", "extra_args"),
}

# Which of the args above the tool's own schema DECLARES to be a path, as opposed
# to a free-form command line. A declared path is checked whole, not tokenised,
# since a path may contain spaces. Everything else is a command line
# whose tokens _shell_paths_outside_scope classifies by syntax alone - it may
# never ask the filesystem which of them exists.
_SHELL_DECLARED_PATH_ARGS: dict[str, tuple[str, ...]] = {"run_tests": ("path",)}

# How many out-of-scope paths one warning names before it says "and N more".
_MAX_SHELL_SCOPE_FLAGS = 3

# Tools that execute an arbitrary user command, blocking or in the background.
# They are the same capability (RCE) and are gated identically everywhere:
# privacy-env injection, the episodic git baseline, and the CLI confirmation
# gates.
_SHELL_EXEC_TOOLS: frozenset[str] = frozenset({"run_shell", "run_shell_background"})

# Tools the shell reject-list inspects before anything else can run them. The
# shell pair takes a command line; git_push takes argv parts and appends its
# branch verbatim, so a "+ref" or ":ref" refspec reaches git as a force or a
# delete with no flag present.
_SHELL_GUARDED_TOOLS: frozenset[str] = _SHELL_EXEC_TOOLS | frozenset({"git_push"})

# The background job-control tools. Useless without a way to start a job, so they
# follow the shell-exec family wherever it is disabled.
_SHELL_JOB_TOOLS: frozenset[str] = frozenset({"check_shell_job", "kill_shell_job"})

# The delegation family, parallel to the shell one above: spawning a sub-agent
# grants a child unbounded write+shell in this cwd, and the background variant is
# that same capability minus the wait. A caller that disables spawn_agent (the
# restricted/shareable-key path does) must not be left with
# spawn_agent_background enabled.
#
# A SEPARATE set, not merged into _SHELL_EXEC_TOOLS: the helper keys on set
# intersection, so one combined family would mean disabling spawn_agent also
# disabled run_shell, and vice versa.
# dispatch_parallel belongs here too: it spawns children with the same inherited
# write+shell reach.
_AGENT_EXEC_TOOLS: frozenset[str] = frozenset(
    {"spawn_agent", "spawn_agent_background", "dispatch_parallel"})

# check_agent_job is useless without a way to start a job, so it follows the
# delegation family wherever that is disabled - same rule as _SHELL_JOB_TOOLS.
_AGENT_JOB_TOOLS: frozenset[str] = frozenset({"check_agent_job"})

# Tools that are handed the running Agent as a hidden ``_parent_agent`` argument
# by the dispatcher. Every one of them guards on it and returns an error when it
# is None, so a tool MISSING from this set is not degraded, it is dead: still
# advertised in every system prompt, still confirmed by the user, and failing on
# every real call. A named set plus the registry-wide dispatch test keeps it in
# step with the registry.
_PARENT_AGENT_TOOLS: frozenset[str] = frozenset(
    {"spawn_agent", "spawn_agent_background", "dispatch_parallel"})


def expand_shell_disable(disabled: frozenset) -> frozenset:
    """Disabling any tool in a capability family disables the whole family.

    A caller that passes ``{"run_shell"}`` means "this session must not execute
    arbitrary commands" (that is how the shareable-key path uses it). Honouring
    that tool-name by tool-name would leave ``run_shell_background`` - the same
    capability minus the wait - enabled.

    Two families, expanded INDEPENDENTLY: shell execution, and sub-agent
    delegation. A single merged family would make disabling ``spawn_agent`` also
    disable ``run_shell``, removing a capability the caller never asked to lose.

    MUST be applied at BOTH boundaries that consume a disabled set: the Agent
    (which hard-refuses at dispatch) and the prompt builders (which decide what
    the model is told exists). Applying it in only one leaves the other
    advertising or accepting a tool the caller meant to switch off.
    """
    out = frozenset(disabled)
    if out & _SHELL_EXEC_TOOLS:
        out = out | _SHELL_EXEC_TOOLS | _SHELL_JOB_TOOLS
    if out & _AGENT_EXEC_TOOLS:
        out = out | _AGENT_EXEC_TOOLS | _AGENT_JOB_TOOLS
    return out

# For each scoped tool, the arg names holding a path/glob to enforce scope against.
# Any present arg outside the scope rejects the call (order only sets which value is
# reported first). Tools default to "path"; entries here add/replace the real target.
_SCOPE_PATH_ARGS: dict[str, tuple[str, ...]] = {
    "grep":           ("path", "glob"),
    "search_files":   ("path", "pattern"),
    "search_replace": ("glob",),
    "generate_image": ("output_path", "input_image"),
    "browser_screenshot": ("output_path",),
}

# MCP (mcp_<server>_<tool>) and plugin (plugin_<plugin>_<export>) tools register
# dynamically with unknown arg schemas, so they are not in _SCOPED_TOOLS /
# _SCOPE_PATH_ARGS. When a scope is active it is still applied to their common
# path-like args, best-effort, so an unusual path-arg name is not caught.
# Restricted, shareable keys cannot reach either family at all (both are disabled
# for a restricted session), so this only tightens an owner's own --scope.
_MCP_SCOPE_PATH_ARGS: tuple[str, ...] = (
    "path", "file", "filename", "filepath", "file_path", "source", "source_path",
    "src", "target", "target_path", "dest", "destination", "dir", "directory",
    "folder", "output", "output_path", "glob",
)

# Model-initiated network tools, governed by the net_mode policy
# (localm.netpolicy): off = fail fast, ask = approval flow, allow = run.
_NETWORK_TOOLS: frozenset[str] = frozenset({"fetch_url", "web_search"})

# Task-list tools (tools/tasks.py). They read and write THIS session's todo
# state, so the dispatcher injects the Agent as a hidden `_session` arg.
_TODO_TOOLS: frozenset[str] = frozenset({"set_todos", "read_todos"})

# Skill tools that write THIS session's active-skill state (skills.py), and so
# are handed the Agent as the same hidden `_session` arg. Separate from
# _TODO_TOOLS: injected identically, but they touch unrelated state.
_SKILL_STATE_TOOLS: frozenset[str] = frozenset({"use_skill"})

# Tools that read THIS session's live ProjectMap (tools/references.py), so the
# dispatcher hands them the Agent as the same hidden `_session` arg. Separate
# set again: injected identically to _TODO_TOOLS/_SKILL_STATE_TOOLS, but reads
# project-index state rather than writing session state.
_PROJECT_MAP_TOOLS: frozenset[str] = frozenset({"find_references"})

# Browser tools (tools/browser.py). They act on the ONE browser session that
# belongs to this coder session, and each re-checks the capability off the
# Agent, so the dispatcher hands them the same hidden `_session` arg.
_BROWSER_TOOLS: frozenset[str] = frozenset({
    "browser_navigate", "browser_read", "browser_click", "browser_fill",
    "browser_screenshot", "browser_console", "browser_network",
    "browser_close",
})

# Fraction of estimated context window at which compaction is triggered
_COMPACT_WARN_RATIO  = 0.70   # warn user in interactive mode

_COMPACT_AUTO_RATIO  = 0.90   # silently compact in non-interactive mode

_DEFAULT_CTX_TOKENS  = 4096   # fallback when n_ctx is unknown

# Re-prompt count when a response looks like a tool call but cannot be parsed.
# After this the raw attempt is SURFACED, never finalised as a hidden
# <tool_call> block.
_MAX_TOOL_REPAIRS = 2

# Abort after this many tool calls fail in a row across ANY tools. The per-tool
# breaker only catches N IDENTICAL failures, so this one catches a spin on VARIED
# failing calls. Any success resets it.
_GLOBAL_ERROR_ABORT = 6

# --- zero-tool-call escalation ---------------------------------------------
#
# The ladder for a turn that emits NO tool call at all. Rung 1 is the same format
# re-prompt the malformed case gets, triggered by ABSENCE. Rung 2 re-runs the
# turn with the tool-call grammar engaged from the first token, so the sampler
# cannot emit anything but a valid call. Only after forcing has failed is the
# user told, as a report of failed enforcement, never as a suggestion to choose
# a different model.
_MAX_NOCALL_ESCALATIONS = 2

# Imperative verbs that make a request an ACTION instead of a question. Read
# verbs are included: "show me what is in config.py" needs read_file just as
# "write config.py" needs write_file.
_ACTION_VERBS: frozenset[str] = frozenset({
    "add", "adjust", "append", "apply", "build", "bump", "change", "check",
    "clean", "clear", "commit", "compile", "convert", "copy", "correct",
    "create", "debug", "delete", "deploy", "diff", "drop", "edit", "execute",
    "extract", "find", "fix", "format", "generate", "grep", "implement",
    "init", "insert", "inspect", "install", "lint", "list", "load", "look",
    "make", "merge", "migrate", "modify", "move", "open", "patch", "port",
    "print", "pull", "push", "read", "rebase", "refactor", "remove", "rename",
    "repair", "replace", "reproduce", "rerun", "resolve", "revert", "rewrite",
    "run", "save", "scaffold", "search", "set", "show", "sort", "split",
    "start", "stop", "swap", "test", "translate", "tweak", "uninstall",
    "update", "upgrade", "verify", "write",
})

# Signals that a request is about THIS project, not programming in the abstract:
# an explicit path, a file extension, or a workspace noun. Either a verb OR one of
# these is enough - see implies_action().
_WORKSPACE_HINT = (
    r"(?:[\w./\\-]+\.[A-Za-z0-9]{1,6}\b"          # something.ext
    r"|[~./\\][\w./\\-]+"                          # a path-looking token
    r"|\b(?:file|files|folder|directory|dir|repo|repository|project|codebase"
    r"|workspace|script|module|package|test|tests|suite|branch|commit"
    r"|readme|config|source|sources)\b)"
)

# Two finals at least this similar (difflib ratio) count as the model restating
# itself instead of progressing.
_REPEAT_SIMILARITY = 0.85

# How many earlier finals a turn is compared against. Bounded so a long session
# cannot make this quadratic; the newest are kept.
_REPEAT_HISTORY_MAX = 12

# Abort when the model emits essentially the SAME response this many times in a
# row. Catches non-failing repetition, which the error-streak breakers (FAILED
# calls) and the repair path (malformed call) miss. Kept above the error and
# repair thresholds so those more-specific guards fire first.
_REPEAT_RESPONSE_ABORT = 5

# Max entries kept in the per-session tool/command failure trace that feeds the
# close-time episode reflection. Bounded so a long spin-loop cannot grow it
# without limit; the NEWEST failures are kept.
_MAX_ERROR_TRACE = 20

# Code file extensions that should be verified (tests / syntax) after writes
_CODE_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php",
})

# run_shell commands containing one of these substrings count as verification
_TEST_COMMAND_MARKERS: tuple[str, ...] = ("pytest", "unittest", "npm test", "cargo test", "go test")
