# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-level constants shared across the Agent mixins: the tool-category sets
(mutating / undoable / scoped / network), the scope path-arg map, and the
compaction / repair / error-breaker thresholds. Extracted verbatim from the
former single-file agent.py."""

from __future__ import annotations

# Tools that mutate files - trigger a project map refresh after they run
_MUTATING_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "edit_files", "run_shell",
})

# Tools whose file changes can be undone (we snapshot before they run).
# These are also the tools recorded in the changed-files tracker.
_UNDOABLE_TOOLS: frozenset[str] = frozenset({
    "write_file", "edit_file", "edit_files", "patch_file", "edit_notebook_cell",
})

# Tools whose target paths are NESTED inside a collection arg rather than sitting
# in a top-level `path` arg. Maps tool name -> (collection arg, key within each
# item). Every path-consuming site (scope check, undo snapshot, changed-file
# tracker, map refresh) resolves paths through _call_target_paths(), so a tool
# listed here is confined and tracked exactly like a single-path one. Without
# this, a nested-path tool passes the scope check by having no `path` arg at all
# - a silent fail-OPEN, which is why the resolution lives in one shared helper.
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
    # FAC-8: the rest of the file-reading/writing tools.
    "grep", "search_files", "search_replace", "read_env",
    "edit_notebook_cell", "generate_image",
})

# Tools deliberately NOT confined by the scope glob: git_diff / git_log take a git
# PATHSPEC (not a filesystem path), and run_tests / run_shell / the background-shell
# trio EXECUTE a process (a path-arg check cannot confine arbitrary code). Any OTHER
# registry tool with a path-like arg MUST be in _SCOPED_TOOLS above; the contract test
# test_coder_scope_default_deny enforces this, so a new file tool is a test failure,
# not "unconfined by omission" (AUD-CODERTOOLS: default-deny at authoring time, not
# reliant on a human remembering).
_INTENTIONALLY_UNSCOPED: frozenset[str] = frozenset({
    "run_shell", "run_tests", "git_diff", "git_log",
    "run_shell_background", "check_shell_job", "kill_shell_job",
})

# The subset of _INTENTIONALLY_UNSCOPED that EXECUTES a process, and so is the
# part a user who set --scope can be genuinely misled by: git_diff / git_log only
# read. The trade-off itself is deliberate and stays (a path-arg check cannot
# confine arbitrary code), but it used to be documented ONLY here, in the source -
# a user running under --scope had no runtime signal at all that their shell was
# unconfined. These tools get a one-per-session notice plus a best-effort argv
# path check (see _ExecutionMixin._warn_shell_outside_scope). Warn, never block.
# run_shell_background belongs here for exactly the same reason as run_shell: it
# runs the user's command line, it just does not wait for it.
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
# to a free-form command line. A declared path needs no path-likeness heuristic
# at all (it IS a path by contract) and is checked whole rather than tokenised,
# since a path may legitimately contain spaces. Everything else is a command line
# whose tokens are ambiguous, and _shell_paths_outside_scope classifies those by
# syntax alone - it may never ask the filesystem which of them exists.
_SHELL_DECLARED_PATH_ARGS: dict[str, tuple[str, ...]] = {"run_tests": ("path",)}

# How many out-of-scope paths one warning names before it says "and N more".
_MAX_SHELL_SCOPE_FLAGS = 3

# NOTE: the command-line grammar tables that used to live here
# (_SHELL_SEPARATORS / _SHELL_SUBCOMMAND_RUNNERS / _SHELL_NESTED_RUNNER_WORDS /
# _SHELL_PATH_VALUE_FLAGS / _SHELL_TRANSPARENT_PREFIXES) existed for one purpose:
# to stop the exists-under-cwd filesystem probe in _shell_paths_outside_scope
# from reading "npm test" or "make docs" as paths. That probe is gone (it stat-ed
# whatever a model-supplied token named, anywhere on the machine), so tracking
# the program word, runners, and path-valued flags no longer decides anything.
# Removed rather than left dead.

# Tools that execute an arbitrary user command, blocking or in the background.
# They are the same capability (RCE) and so must be gated identically everywhere:
# privacy-env injection, the episodic git baseline, and the CLI confirmation gates.
# A background variant that any of those forgets is a bypass of that gate, not a
# missing nicety.
_SHELL_EXEC_TOOLS: frozenset[str] = frozenset({"run_shell", "run_shell_background"})

# The background job-control tools. Useless without a way to start a job, so they
# follow the shell-exec family wherever it is disabled.
_SHELL_JOB_TOOLS: frozenset[str] = frozenset({"check_shell_job", "kill_shell_job"})

# The delegation family, exactly parallel to the shell one above and for the same
# reason: spawning a sub-agent grants a child unbounded write+shell in this cwd,
# and the background variant is that same capability minus the wait. A caller that
# disables spawn_agent (the restricted/shareable-key path does) must not be left
# with spawn_agent_background still enabled - that would be an RCE escape by a
# tool added after the caller was written, which is the precise hole the shell
# family closes on its side.
#
# A SEPARATE set, deliberately not merged into _SHELL_EXEC_TOOLS: the helper keys
# on set intersection, so one combined family would mean disabling spawn_agent
# also disabled run_shell (and vice versa) - welding two unrelated capabilities
# together.
_AGENT_EXEC_TOOLS: frozenset[str] = frozenset(
    {"spawn_agent", "spawn_agent_background"})

# check_agent_job is useless without a way to start a job, so it follows the
# delegation family wherever that is disabled - same rule as _SHELL_JOB_TOOLS.
_AGENT_JOB_TOOLS: frozenset[str] = frozenset({"check_agent_job"})


def expand_shell_disable(disabled: frozenset) -> frozenset:
    """Disabling any tool in a capability family disables the whole family.

    A caller that passes ``{"run_shell"}`` means "this session must not execute
    arbitrary commands" (that is exactly how the shareable-key path uses it).
    Honouring that literally, tool-name by tool-name, would leave
    ``run_shell_background`` - the same capability minus the wait - enabled, so
    the safety choice would be silently defeated by a tool added after the
    caller was written. Expand the intent instead.

    Two families, expanded INDEPENDENTLY: shell execution, and sub-agent
    delegation. Keeping them separate matters - a single merged family would make
    disabling ``spawn_agent`` also disable ``run_shell``, silently removing a
    capability the caller never asked to lose.

    Applied at BOTH boundaries that consume a disabled set: the Agent (which
    hard-refuses at dispatch) and the prompt builders (which decide what the
    model is told exists). Applying it in only one leaves the other advertising
    or accepting a tool the caller meant to switch off. (The name is historical -
    it predates the second family - but every consuming site already calls it, so
    extending it here is what keeps both boundaries covered for free.)
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
}

# MCP (mcp_<server>_<tool>) and plugin (plugin_<plugin>_<export>) tools register
# dynamically with unknown arg schemas, so they are not in _SCOPED_TOOLS /
# _SCOPE_PATH_ARGS (the default-deny test cannot see them). When a scope is active
# we still apply it to their common path-like args (CHK-MCP-SCOPE + CHK-SCOPE-PLUGIN,
# defense-in-depth); best-effort, so an unusual path-arg name is not caught (its
# author is the owner's own MCP / plugin config). Restricted, shareable keys cannot
# reach either family at all (both disabled for a restricted session), so this only
# tightens an owner's own --scope, never a cross-trust boundary.
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

# Fraction of estimated context window at which compaction is triggered
_COMPACT_WARN_RATIO  = 0.70   # warn user in interactive mode

_COMPACT_AUTO_RATIO  = 0.90   # silently compact in non-interactive mode

_DEFAULT_CTX_TOKENS  = 4096   # fallback when n_ctx is unknown

# Re-prompt count when a response looks like a tool call but cannot be parsed.
# After this the raw attempt is SURFACED, never silently finalised as a hidden
# <tool_call> block (which rendered as an empty bubble + no file written).
_MAX_TOOL_REPAIRS = 2

# Abort after this many tool calls fail in a row across ANY tools. The per-tool
# breaker only catches N IDENTICAL failures; a weak model can spin on VARIED failing
# calls (e.g. git_show with invented hashes) and burn the budget. Any success resets.
_GLOBAL_ERROR_ABORT = 6

# Abort when the model emits the SAME response this many times in a row (the
# "Message 1..4 / I will now wait" scaffold-repetition: narrates without progress).
# Catches identical NON-failing repetition, which the error-streak breakers (FAILED
# calls) and the repair path (malformed call) miss. Kept above the error/repair
# thresholds so those more-specific guards fire first for a broken loop; this is the
# last-resort catch for a succeeding-but-pointless one (REC-CODER-LOOPBREAK).
_REPEAT_RESPONSE_ABORT = 5

# Max entries kept in the per-session tool/command failure trace that feeds the
# close-time episode reflection (audit cluster 13: reflection was evidence-starved,
# so failure lessons could not be captured). Bounded so a long spin-loop cannot grow
# it without limit; the NEWEST failures are kept (they include the ones that tripped
# the circuit breakers / ended the run incomplete).
_MAX_ERROR_TRACE = 20

# Code file extensions that should be verified (tests / syntax) after writes
_CODE_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php",
})

# run_shell commands containing one of these substrings count as verification
_TEST_COMMAND_MARKERS: tuple[str, ...] = ("pytest", "unittest", "npm test", "cargo test", "go test")
