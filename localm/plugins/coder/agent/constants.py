# SPDX-License-Identifier: AGPL-3.0-or-later
"""Module-level constants shared across the Agent mixins: the tool-category sets
(mutating / undoable / scoped / network), the scope path-arg map, and the
compaction / repair / error-breaker thresholds. Extracted verbatim from the
former single-file agent.py."""

from __future__ import annotations

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

# File-touching tools deliberately NOT confined by the scope glob: git_diff /
# git_log take a git PATHSPEC (repo history, not a filesystem path to confine),
# and run_tests / run_shell EXECUTE a process (a path-arg check cannot
# meaningfully confine arbitrary code). Any OTHER registry tool that exposes a
# path-like argument MUST appear in _SCOPED_TOOLS above; the contract test
# test_coder_scope_default_deny enforces this, so a newly-added file tool is a
# test failure rather than "unconfined by omission" (AUD-CODERTOOLS: the scope
# allowlist is default-deny at authoring time, not reliant on a human remembering).
_INTENTIONALLY_UNSCOPED: frozenset[str] = frozenset({
    "run_shell", "run_tests", "git_diff", "git_log",
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

# MCP tools (mcp_<server>_<tool>) are registered dynamically with unknown arg
# schemas, so they are not in _SCOPED_TOOLS / _SCOPE_PATH_ARGS. When a scope is
# active we still apply it to an MCP tool's common path-like args, so an owner's
# declared --scope is honoured by MCP file tools too (CHK-MCP-SCOPE, defense-in-
# depth). Best-effort: an MCP tool using an unusual path-arg name is not caught
# (its author is the owner's own MCP config).
_MCP_SCOPE_PATH_ARGS: tuple[str, ...] = (
    "path", "file", "filename", "filepath", "file_path", "source", "source_path",
    "src", "target", "target_path", "dest", "destination", "dir", "directory",
    "folder", "output", "output_path", "glob",
)

# Model-initiated network tools, governed by the net_mode policy
# (localm.netpolicy): off = fail fast, ask = approval flow, allow = run.
_NETWORK_TOOLS: frozenset[str] = frozenset({"fetch_url", "web_search"})

# Fraction of estimated context window at which compaction is triggered
_COMPACT_WARN_RATIO  = 0.70   # warn user in interactive mode

_COMPACT_AUTO_RATIO  = 0.90   # silently compact in non-interactive mode

_DEFAULT_CTX_TOKENS  = 4096   # fallback when n_ctx is unknown

# How many times to re-prompt when a response looks like a tool call but cannot be
# parsed. After this, the raw attempt is SURFACED (never silently finalised as a
# hidden <tool_call> block - which rendered as an empty bubble + no file written).
_MAX_TOOL_REPAIRS = 2

# Abort a task after this many tool calls fail in a row across ANY tools. The
# per-tool breaker only catches N IDENTICAL failures; a weak model can spin on
# VARIED failing calls (e.g. git_show with invented hashes) and burn the whole
# turn/token budget. Any successful tool call resets the streak.
_GLOBAL_ERROR_ABORT = 6

# Abort when the model emits the SAME response this many times in a row - the
# "Message 1..4 / I will now wait" scaffold-repetition where it narrates without
# making progress. The error-streak breakers above only catch FAILED tool calls
# and the repair path re-prompts a malformed call; this catches identical
# NON-failing repetition. Kept ABOVE _GLOBAL_ERROR_ABORT-adjacent thresholds and
# the repair cap so those more-specific guards fire first for a failing/broken
# loop; this is the last-resort catch for a succeeding-but-pointless one
# (REC-CODER-LOOPBREAK).
_REPEAT_RESPONSE_ABORT = 5

# Code file extensions that should be verified (tests / syntax) after writes
_CODE_EXTS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php",
})

# run_shell commands containing one of these substrings count as verification
_TEST_COMMAND_MARKERS: tuple[str, ...] = ("pytest", "unittest", "npm test", "cargo test", "go test")
