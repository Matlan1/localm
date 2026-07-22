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

# File-access tools whose target path must match the active scope glob. Keys on
# the `path` arg; a tool whose real target is a `glob`/`output_path` arg has that
# checked too (see _SCOPE_PATH_ARGS). run_shell is unscoped (see _INTENTIONALLY_UNSCOPED).
_SCOPED_TOOLS: frozenset[str] = frozenset({
    "read_file", "write_file", "edit_file", "patch_file",
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

# How many out-of-scope paths one warning names before it says "and N more".
_MAX_SHELL_SCOPE_FLAGS = 3

# Tools that execute an arbitrary user command, blocking or in the background.
# They are the same capability (RCE) and so must be gated identically everywhere:
# privacy-env injection, the episodic git baseline, and the CLI confirmation gates.
# A background variant that any of those forgets is a bypass of that gate, not a
# missing nicety.
_SHELL_EXEC_TOOLS: frozenset[str] = frozenset({"run_shell", "run_shell_background"})

# The background job-control tools. Useless without a way to start a job, so they
# follow the shell-exec family wherever it is disabled.
_SHELL_JOB_TOOLS: frozenset[str] = frozenset({"check_shell_job", "kill_shell_job"})


def expand_shell_disable(disabled: frozenset) -> frozenset:
    """Disabling any shell-execution tool disables the whole family.

    A caller that passes ``{"run_shell"}`` means "this session must not execute
    arbitrary commands" (that is exactly how the shareable-key path uses it).
    Honouring that literally, tool-name by tool-name, would leave
    ``run_shell_background`` - the same capability minus the wait - enabled, so
    the safety choice would be silently defeated by a tool added after the
    caller was written. Expand the intent instead.

    Applied at BOTH boundaries that consume a disabled set: the Agent (which
    hard-refuses at dispatch) and the prompt builders (which decide what the
    model is told exists). Applying it in only one leaves the other advertising
    or accepting a tool the caller meant to switch off.
    """
    if disabled & _SHELL_EXEC_TOOLS:
        return frozenset(disabled) | _SHELL_EXEC_TOOLS | _SHELL_JOB_TOOLS
    return frozenset(disabled)

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
