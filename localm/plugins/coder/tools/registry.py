# SPDX-License-Identifier: AGPL-3.0-or-later
"""The tool registry: the ``ToolDef`` schema, the ``SAFE_RESTRICTED_TOOLS``
allowlist, and the ``TOOL_REGISTRY`` mapping the agent dispatches through.

``TOOL_REGISTRY`` is the single mutable registry; mcp.py / plugin_tools.py /
skills.py add entries to it at runtime, so it must stay one shared dict object
(re-exported by name from the package ``__init__``, never copied)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .files import (
    tool_edit_file,
    tool_edit_files,
    tool_edit_notebook_cell,
    tool_grep,
    tool_list_dir,
    tool_patch_file,
    tool_read_file,
    tool_search_files,
    tool_search_replace,
    tool_tree,
    tool_write_file,
)
from .shell import (
    tool_check_shell_job,
    tool_kill_shell_job,
    tool_run_shell,
    tool_run_shell_background,
    tool_run_tests,
)
from .git import (
    tool_git_commit,
    tool_git_create_branch,
    tool_git_diff,
    tool_git_log,
    tool_git_push,
    tool_git_status,
)
from .env import tool_read_env
from .web import tool_fetch_url, tool_web_search
from .agents import (
    tool_check_agent_job,
    tool_spawn_agent,
    tool_spawn_agent_background,
)
from .parallel import tool_dispatch_parallel
from .media import tool_generate_image
from .rag import tool_rag_list_collections, tool_rag_search
from .references import tool_find_references
from .tasks import tool_read_todos, tool_set_todos

@dataclass
class ToolDef:
    name:        str
    fn:          Callable
    description: str
    params:      dict
    destructive: bool = False
    # Marks a tool whose output is external, attacker-influenceable content; read
    # by provenance.is_untrusted_tool. The built-in network tools and MCP tools are
    # detected by name instead, so this is the seam a plugin tool uses to flag
    # itself.
    untrusted_output: bool = False
    # A NON-mutating tool that should still ask before running, ORed into
    # execution.py's needs_confirm alongside destructive and the ask network mode.
    # Separate from destructive, which also triggers dry-run-skip and an undo
    # snapshot.
    ask_by_default: bool = False


# The ONLY tools a RESTRICTED (shareable, non-owner) coder session may use: an
# allowlist, so a newly-added tool is denied to a restricted key by default. These
# read the project or edit files WITHIN the confined cwd; none spawn a process, run
# code, hit the network, read the environment, or re-enter the agent. Everything
# else is disabled for a restricted session, including run_shell/run_tests,
# git_commit/git_push, web_search/fetch_url, generate_image, read_env, spawn_agent,
# and every dynamically-registered MCP, plugin or skill tool.
SAFE_RESTRICTED_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_dir", "tree", "grep", "search_files", "find_references",
    "write_file", "edit_file", "edit_files", "patch_file", "search_replace",
    "edit_notebook_cell",
    "git_status", "git_diff", "git_log",
    # The task list is agent state, not a capability: it spawns no process, runs no
    # code, reaches no network, reads no environment, re-enters no agent and touches
    # no project file. Its only disk trace is a key in the checkpoint the session
    # already writes.
    "set_todos", "read_todos",
})
# edit_files is edit_file's exact-string replacement applied to N files in one
# call, each path put through the same _confine() check, so it grants no capability
# the allowlist did not already grant; it only makes the same edit atomic.


def _spawn_role_help() -> str:
    """The spawn_agent ``role`` parameter description, built from the presets so
    the tool schema cannot drift out of step with the roles that actually exist."""
    from ..roles import role_catalogue
    return (
        "Optional role preset giving the sub-agent a focused mission and a "
        "narrowed toolset (it can never exceed your own). One of: "
        f"{role_catalogue()}. Omit to give it your full toolset."
    )


TOOL_REGISTRY: dict[str, ToolDef] = {
    "read_file": ToolDef(
        name="read_file",
        fn=tool_read_file,
        description=(
            "Read the contents of a file. Large files are truncated - "
            "re-read a specific region with offset/limit."
        ),
        params={
            "path":   {"type": "string", "description": "File path (relative to cwd)", "required": True},
            "offset": {"type": "int",    "description": "1-based start line (optional)", "required": False},
            "limit":  {"type": "int",    "description": "Max lines to read from offset (optional)", "required": False},
        },
    ),
    "write_file": ToolDef(
        name="write_file",
        fn=tool_write_file,
        description="Write or overwrite a file with new content.",
        params={
            "path":    {"type": "string", "description": "File path",         "required": True},
            "content": {"type": "string", "description": "Full file content", "required": True},
        },
        destructive=True,
    ),
    "edit_file": ToolDef(
        name="edit_file",
        fn=tool_edit_file,
        description="Replace the first occurrence of `old` text with `new` text in a file.",
        params={
            "path": {"type": "string", "description": "File path",            "required": True},
            "old":  {"type": "string", "description": "Exact text to replace","required": True},
            "new":  {"type": "string", "description": "Replacement text",     "required": True},
        },
        destructive=True,
    ),
    "edit_files": ToolDef(
        name="edit_files",
        fn=tool_edit_files,
        description=(
            "Apply the SAME exact-string edit_file replacement to several files "
            "in one call, all-or-nothing: if any edit fails, every file is rolled "
            "back and nothing changes. Use for a snippet that must change "
            "identically across files; use search_replace instead when you want a "
            "regex."
        ),
        params={
            "edits": {
                "type": "array",
                "description": (
                    "List of {path, old, new} objects. Each replaces the first "
                    "occurrence of `old` with `new` in `path`, matched exactly."
                ),
                "required": True,
                # The native tool schema (agent/tooldefs.py) reads this; without it
                # the array is described as an array of strings.
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (relative to cwd)"},
                        "old":  {"type": "string", "description": "Exact text to replace"},
                        "new":  {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["path", "old", "new"],
                },
            },
        },
        destructive=True,
    ),
    "patch_file": ToolDef(
        name="patch_file",
        fn=tool_patch_file,
        description=(
            "Apply a unified diff (patch -u format) to a file. "
            "More reliable than edit_file for multi-hunk or large changes. "
            "Always read_file first so line numbers are accurate."
        ),
        params={
            "path": {"type": "string", "description": "File path (relative to cwd)",              "required": True},
            "diff": {"type": "string", "description": "Unified diff string (patch -u format)",    "required": True},
        },
        destructive=True,
    ),
    "run_shell": ToolDef(
        name="run_shell",
        fn=tool_run_shell,
        description="Execute a shell command in the working directory.",
        params={
            "command": {"type": "string", "description": "Shell command",      "required": True},
            "timeout": {"type": "int",    "description": "Timeout in seconds", "required": False},
        },
        destructive=True,
    ),
    "run_shell_background": ToolDef(
        name="run_shell_background",
        fn=tool_run_shell_background,
        description=(
            "Start a shell command in the background and get a job id back "
            "immediately, without waiting for it to finish. Use for a dev server "
            "you then want to talk to, a long build, or a watcher; poll it with "
            "check_shell_job and stop it with kill_shell_job. Use run_shell "
            "instead when you just need the command's result."
        ),
        params={
            "command": {"type": "string", "description": "Shell command", "required": True},
        },
        destructive=True,
    ),
    "check_shell_job": ToolDef(
        name="check_shell_job",
        fn=tool_check_shell_job,
        description=(
            "Check a background job started by run_shell_background: whether it "
            "is still running, its exit code once finished, and the output "
            "captured so far."
        ),
        params={
            "job_id": {"type": "string", "description": "Job id from run_shell_background", "required": True},
        },
        # NOT destructive, unlike its two siblings: this only reads a status field
        # and an output buffer, and starts nothing, kills nothing, writes nothing.
        # Starting and killing are the gated capabilities.
    ),
    "kill_shell_job": ToolDef(
        name="kill_shell_job",
        fn=tool_kill_shell_job,
        description=(
            "Stop a background job and its whole process tree, and report its "
            "final state and output."
        ),
        params={
            "job_id": {"type": "string", "description": "Job id from run_shell_background", "required": True},
        },
        destructive=True,
    ),
    "list_dir": ToolDef(
        name="list_dir",
        fn=tool_list_dir,
        description="List the contents of a directory.",
        params={"path": {"type": "string", "description": "Directory path (default: .)", "required": False}},
    ),
    "tree": ToolDef(
        name="tree",
        fn=tool_tree,
        description="Recursive directory tree with file sizes. Skips common noise dirs (.git, __pycache__, node_modules, etc.).",
        params={
            "path":      {"type": "string", "description": "Root directory (default: .)", "required": False},
            "max_depth": {"type": "int",    "description": "How many levels deep to recurse (default: 3)", "required": False},
            "max_files": {"type": "int",    "description": "Stop after this many files (default: 300)", "required": False},
        },
    ),
    "edit_notebook_cell": ToolDef(
        name="edit_notebook_cell",
        fn=tool_edit_notebook_cell,
        description="Replace the source of a single cell in a Jupyter notebook (.ipynb). Use read_file first to see cell indices.",
        params={
            "path":       {"type": "string", "description": "Path to the .ipynb file.", "required": True},
            "cell_index": {"type": "int",    "description": "Zero-based index of the cell to edit.", "required": True},
            "source":     {"type": "string", "description": "New source code or markdown text for the cell.", "required": True},
            "cell_type":  {"type": "string", "description": "Override cell type: code, markdown, or raw (optional).", "required": False},
        },
        destructive=True,
    ),
    "search_files": ToolDef(
        name="search_files",
        fn=tool_search_files,
        description="Find files matching a glob pattern.",
        params={
            "pattern": {"type": "string", "description": "Glob pattern, e.g. **/*.py", "required": True},
            "path":    {"type": "string", "description": "Root directory to search",    "required": False},
        },
    ),
    "grep": ToolDef(
        name="grep",
        fn=tool_grep,
        description=(
            "Search file contents with a regex pattern. Skips noise directories "
            "(.git, node_modules, ...), binaries, and oversized files, and says "
            "so. Raise max_per_file/max_output_lines when a result says it was "
            "capped."
        ),
        params={
            "pattern": {"type": "string", "description": "Regex pattern",                "required": True},
            "path":    {"type": "string", "description": "File or directory to search",   "required": False},
            "glob":    {"type": "string", "description": "File filter, e.g. **/*.py",     "required": False},
            "context": {"type": "int",    "description": "Lines of context (default 2)",  "required": False},
            "max_per_file":     {"type": "int", "description": "Matches shown per file (0 = all; default 20)",       "required": False},
            "max_output_lines": {"type": "int", "description": "Output lines before stopping (0 = no cap; default 300)", "required": False},
            "max_file_bytes":   {"type": "int", "description": "Skip files larger than this (0 = no cap; default 4 MB)",  "required": False},
        },
    ),
    "find_references": ToolDef(
        name="find_references",
        fn=tool_find_references,
        description=(
            "Find call sites of a symbol in the indexed project - who calls "
            "this before you change its signature. Best-effort and "
            "single-repo (a regex reverse index, not a real call graph): a "
            "shadowed local of the same name, or a call reached only through "
            "an alias or attribute access, is not distinguished from a "
            "genuine hit. Use grep for an exhaustive search."
        ),
        params={
            "symbol": {"type": "string", "description": "Function/method/class name to find call sites of", "required": True},
        },
    ),
    "git_status": ToolDef(
        name="git_status",
        fn=tool_git_status,
        description="Show working-tree status (git status --short --branch).",
        params={},
    ),
    "git_diff": ToolDef(
        name="git_diff",
        fn=tool_git_diff,
        description="Show git diff (unstaged by default; pass staged=true for staged changes).",
        params={
            "path":   {"type": "string", "description": "Limit to this file/dir",    "required": False},
            "staged": {"type": "bool",   "description": "Show staged diff (default false)", "required": False},
        },
    ),
    "git_log": ToolDef(
        name="git_log",
        fn=tool_git_log,
        description="Show recent commits (oneline format).",
        params={
            "n":    {"type": "int",    "description": "Number of commits (default 10)", "required": False},
            "path": {"type": "string", "description": "Limit to this file/dir",         "required": False},
        },
    ),
    "read_env": ToolDef(
        name="read_env",
        fn=tool_read_env,
        description=(
            "Read the project's .env file and active environment variables. "
            "Secret-looking values (keys, tokens, passwords) are redacted."
        ),
        params={
            "path": {"type": "string", "description": "Optional env-format file to read instead of .env", "required": False},
        },
    ),
    "fetch_url": ToolDef(
        name="fetch_url",
        fn=tool_fetch_url,
        description="Fetch a URL and return its plain-text content (HTML stripped).",
        params={
            "url":       {"type": "string", "description": "Full URL to fetch",               "required": True},
            "max_chars": {"type": "int",    "description": "Truncate output (default 8000)",  "required": False},
        },
    ),
    "web_search": ToolDef(
        name="web_search",
        fn=tool_web_search,
        description=(
            "Search the web; returns numbered results with title, URL, and "
            "snippet. Use when you need current information (versions, docs, "
            "errors, facts). Follow up with fetch_url to read a full page."
        ),
        params={
            "query":       {"type": "string", "description": "Search query",                       "required": True},
            "max_results": {"type": "int",    "description": "How many results (default 5, max 10)", "required": False},
        },
    ),
    "spawn_agent": ToolDef(
        name="spawn_agent",
        fn=tool_spawn_agent,
        description="Spawn a focused sub-agent to handle a specific sub-task and return its result.",
        params={
            "task":      {"type": "string", "description": "What the sub-agent should do",     "required": True},
            "name":      {"type": "string", "description": "Short name for this sub-agent",     "required": False},
            "files":     {"type": "array",  "description": "Files to pre-load into sub-agent (the call fails if one cannot be read)", "required": False},
            "model":     {"type": "string", "description": "Override model for sub-agent",      "required": False},
            "max_turns": {"type": "int",    "description": "Max iterations (default 10)",       "required": False},
            "role":      {"type": "string", "description": _spawn_role_help(),                  "required": False},
        },
        # A child agent runs write_file/run_shell/git_* of its own in the PARENT's
        # cwd, so spawn_agent is at least as destructive as the tools it grants.
        # Marking it destructive gives it its own segment in _execute_tools, so two
        # spawn_agent calls in one model turn can never run children CONCURRENTLY in
        # the same cwd; keeps it out of the parallel-batch deadline, whose pool is
        # abandoned with shutdown(wait=False); and puts the write and shell grant
        # behind the same confirmation gate every destructive tool passes.
        destructive=True,
    ),
    "spawn_agent_background": ToolDef(
        name="spawn_agent_background",
        fn=tool_spawn_agent_background,
        description=(
            "Start a sub-agent in the background and get a job id back immediately, "
            "so you can keep working while it runs. Check it with check_agent_job. "
            "Use spawn_agent instead when you need the result before continuing."
        ),
        params={
            "task":      {"type": "string", "description": "What the sub-agent should do",  "required": True},
            "name":      {"type": "string", "description": "Short name for this sub-agent", "required": False},
            "files":     {"type": "array",  "description": "Files to pre-load into sub-agent (the call fails if one cannot be read)", "required": False},
            "model":     {"type": "string", "description": "Override model for sub-agent",  "required": False},
            "max_turns": {"type": "int",    "description": "Max iterations (default 10)",   "required": False},
            "role":      {"type": "string", "description": _spawn_role_help(),              "required": False},
        },
        # Destructive for the same reason spawn_agent is: it grants a child
        # unbounded write and shell access in this cwd, and it is not exempted from
        # the gate for returning quickly. Its serial segment serialises the DISPATCH,
        # a fast registry submit, not the child's run, so backgrounded children still
        # overlap. The ONE confirmation taken here covers everything the child does
        # for its entire run, since a backgrounded child cannot come back and ask.
        destructive=True,
    ),
    "check_agent_job": ToolDef(
        name="check_agent_job",
        fn=tool_check_agent_job,
        description=(
            "Check a background sub-agent started by spawn_agent_background: "
            "whether it is still running, and its result once finished."
        ),
        params={
            "job_id": {"type": "string", "description": "Job id from spawn_agent_background", "required": True},
        },
        # NOT destructive, matching check_shell_job: this only reads a status field.
        # destructive is the CONFIRMATION axis, not the capability axis; the
        # capability is gated by membership in the agent-exec disable family
        # (agent/constants.py).
        destructive=False,
    ),
    "generate_image": ToolDef(
        name="generate_image",
        fn=tool_generate_image,
        description=(
            "Generate or refine an image using the local FLUX model. "
            "Without input_image: generates from scratch (txt2img). "
            "With input_image: refines an existing image guided by the prompt (img2img)."
        ),
        params={
            "prompt":       {"type": "string", "description": "What to generate. For img2img, describe what to change rather than the full scene.", "required": True},
            "output_path":  {"type": "string", "description": "Path to save the result (default: output.png)", "required": False},
            "input_image":  {"type": "string", "description": "Path to an existing image to use as the starting point (img2img mode).", "required": False},
            "denoise":      {"type": "float",  "description": "img2img only - how much to change the input (0.0=no change, 1.0=completely new). Default 0.75.", "required": False},
            "seed":         {"type": "int",    "description": "Noise seed for reproducible output. Each result reports its seed; pass it back to reproduce or tweak.", "required": False},
            "guidance":        {"type": "float",  "description": "Guidance scale (default: 3.5). Lower values (2.5-3.0) improve photorealism.", "required": False},
            "negative_prompt": {"type": "string", "description": "Things to keep OUT of the image, e.g. 'blurry, watermark, text'. Applied as a real negative branch via classifier-free guidance (CFGGuider).", "required": False},
            "lora_name":          {"type": "string", "description": "LoRA filename to load (optional).", "required": False},
            "lora_strength_model":{"type": "float",  "description": "LoRA strength on the UNet (default: 1.0). Main lever for unlock/style LoRAs.", "required": False},
            "lora_strength_clip": {"type": "float",  "description": "LoRA strength on the text encoder (default: 0.5).", "required": False},
        },
    ),
    "rag_list_collections": ToolDef(
        name="rag_list_collections",
        fn=tool_rag_list_collections,
        description=(
            "List every indexed RAG (Knowledge) collection with its stats: "
            "name, document/chunk counts, and retrieval mode (hybrid or BM25). "
            "Metadata only, no document content. Use before rag_search if you "
            "do not already know the collection name."
        ),
        params={},
        # Not in SAFE_RESTRICTED_TOOLS: a restricted coder session never sees this
        # tool at all.
    ),
    "rag_search": ToolDef(
        name="rag_search",
        fn=tool_rag_search,
        description=(
            "Search a RAG (Knowledge) collection - the user's own indexed "
            "documents (manuals, notes, code, papers) - and return the top-k "
            "matching excerpts with their source and a relevance score. Use "
            "rag_list_collections first to see what collections exist."
        ),
        params={
            "collection": {"type": "string", "description": "Collection name (see rag_list_collections)", "required": True},
            "query":      {"type": "string", "description": "What to search for",                          "required": True},
            "k":          {"type": "int",    "description": "How many excerpts to return (default 4, max 20)", "required": False},
        },
        # Not destructive (it mutates nothing) but still asks by default: a
        # collection is not scoped to the coder's cwd, so its content can enter the
        # model's context unprompted. untrusted_output because retrieved document
        # text is external, attacker-influenceable content, the same class as
        # fetch_url/web_search (provenance.py).
        ask_by_default=True,
        untrusted_output=True,
    ),
    "run_tests": ToolDef(
        name="run_tests",
        fn=tool_run_tests,
        description=(
            "Run the project test suite. Auto-detects pytest, cargo test, go test, or npm/yarn test "
            "from project files. Returns pass/fail counts and failure output."
        ),
        params={
            "runner":     {"type": "string", "description": "Test runner: auto (default), pytest, cargo, go, npm, yarn", "required": False},
            "path":       {"type": "string", "description": "Limit run to a file or directory (default: whole project)", "required": False},
            "extra_args": {"type": "string", "description": "Extra arguments appended to the test command", "required": False},
        },
    ),
    "git_commit": ToolDef(
        name="git_commit",
        fn=tool_git_commit,
        description="Stage files and create a git commit. Stages all tracked changes when files is omitted.",
        params={
            "message": {"type": "string", "description": "Commit message", "required": True},
            "files":   {"type": "array",  "description": "Specific files to stage (omit to stage all changes)", "required": False},
        },
        destructive=True,
    ),
    "git_push": ToolDef(
        name="git_push",
        fn=tool_git_push,
        description="Push the current branch to a remote (default: origin).",
        params={
            "remote": {"type": "string", "description": "Remote name (default: origin)", "required": False},
            "branch": {"type": "string", "description": "Branch name (default: current branch)", "required": False},
        },
        destructive=True,
    ),
    "git_create_branch": ToolDef(
        name="git_create_branch",
        fn=tool_git_create_branch,
        description="Create a new git branch and optionally check it out.",
        params={
            "name":     {"type": "string", "description": "Branch name, e.g. feat/my-feature", "required": True},
            "checkout": {"type": "bool",   "description": "Switch to the new branch (default: true)", "required": False},
        },
        destructive=True,
    ),
    "search_replace": ToolDef(
        name="search_replace",
        fn=tool_search_replace,
        description=(
            "Search for a regex pattern across multiple files and replace all matches "
            "(file by file; a mid-way failure reports which files were already modified). "
            "Use dry_run=true to preview changes before committing."
        ),
        params={
            "pattern":     {"type": "string", "description": "Python regex pattern (re.MULTILINE)", "required": True},
            "replacement": {"type": "string", "description": "Replacement string (supports \\1 back-references)", "required": True},
            "glob":        {"type": "string", "description": "File filter relative to cwd, e.g. **/*.py (default: all files)", "required": False},
            "dry_run":     {"type": "bool",   "description": "Preview without modifying (default: false)", "required": False},
        },
        destructive=True,
    ),
    # The task list (tools/tasks.py). NOT destructive: it writes no user file, and
    # under auto_approve off or an unattended run the fail-closed branch in
    # _execute_tool would DENY the model its own bookkeeping. The flag's other
    # meaning, run alone and never in a parallel batch (agent/loop.py), is not needed
    # either: set_todos replaces the whole list under the agent's lock, so two calls
    # in one batch are last-writer-wins with no torn state, and read_todos only
    # reads.
    "set_todos": ToolDef(
        name="set_todos",
        fn=tool_set_todos,
        description=(
            "Record your task list for this session. Pass the COMPLETE list every "
            "time - it replaces the previous one. Write one task per item, marked "
            "'[ ]' not started, '[>]' in progress (keep at most one), '[x]' done. "
            "Use it for any task with more than a couple of steps, and update it as "
            "you finish each one: this list survives context compaction and a "
            "pause/resume, the conversation does not."
        ),
        params={
            "items": {"type": "array",
                      "description": "The complete task list, one task per entry, e.g. "
                                     '["[x] read the failing test", "[>] fix the parser", "[ ] run the suite"]',
                      "required": True},
        },
    ),
    "read_todos": ToolDef(
        name="read_todos",
        fn=tool_read_todos,
        description=(
            "Read back the task list you wrote with set_todos. Use it after "
            "resuming a session, or whenever you are unsure what is left to do."
        ),
        params={},
    ),
    "dispatch_parallel": ToolDef(
        name="dispatch_parallel",
        fn=tool_dispatch_parallel,
        description=(
            "Run up to 2 sub-tasks AT THE SAME TIME, each in its own isolated git "
            "worktree on its own branch. Use when two pieces of work are genuinely "
            "independent - the children can even edit the same file without "
            "interfering, and your working tree is never touched. Each child's diff "
            "is returned for you to review; nothing is ever merged automatically. "
            "For a single sub-task, use spawn_agent instead."
        ),
        params={
            "tasks":     {"type": "array",  "description": "1 or 2 sub-tasks: strings, or objects with task/name/model", "required": True},
            "max_turns": {"type": "int",    "description": "Per-child iteration cap (default 10)", "required": False},
            "timeout_s": {"type": "int",    "description": "Wall-clock budget in seconds for the whole batch, shared by every child (default 600)", "required": False},
        },
        # Destructive so _execute_tools runs it ALONE: the tool manages its own
        # internal 2-way concurrency, and the batch executor running it alongside
        # other tools would stack concurrency on top of that.
        destructive=True,
    ),
}
