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
from .shell import tool_run_shell, tool_run_tests
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
from .agents import tool_spawn_agent
from .media import tool_generate_image

@dataclass
class ToolDef:
    name:        str
    fn:          Callable
    description: str
    params:      dict
    destructive: bool = False
    # Opt-in marker for a tool whose output is external, attacker-influenceable
    # content (read by provenance.is_untrusted_tool). The built-in network tools
    # and MCP tools are detected by name, so this is the seam for a future plugin
    # tool that returns fetched content to flag itself as untrusted.
    untrusted_output: bool = False


# The ONLY tools a RESTRICTED (shareable, non-owner) coder session may use: an
# allowlist, so a newly-added tool is denied to a restricted key by default. These
# read the project or edit files WITHIN the confined cwd; NONE spawn a process, run
# code, hit the network, read the environment, or re-enter the agent. Everything
# else - run_shell/run_tests (RCE; a planted conftest runs), git_commit/git_push
# (git hooks run), web_search/fetch_url (network), generate_image, read_env
# (secret disclosure), spawn_agent, and every dynamically-registered MCP/plugin/
# skill tool - is disabled for a restricted session. So a key you hand out can read
# and edit this project, but cannot execute anything; you review and run.
SAFE_RESTRICTED_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_dir", "tree", "grep", "search_files",
    "write_file", "edit_file", "patch_file", "search_replace", "edit_notebook_cell",
    "git_status", "git_diff", "git_log",
})


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
        description="Search file contents with a regex pattern.",
        params={
            "pattern": {"type": "string", "description": "Regex pattern",                "required": True},
            "path":    {"type": "string", "description": "File or directory to search",   "required": False},
            "glob":    {"type": "string", "description": "File filter, e.g. **/*.py",     "required": False},
            "context": {"type": "int",    "description": "Lines of context (default 2)",  "required": False},
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
            "files":     {"type": "array",  "description": "Files to pre-load into sub-agent", "required": False},
            "model":     {"type": "string", "description": "Override model for sub-agent",      "required": False},
            "max_turns": {"type": "int",    "description": "Max iterations (default 10)",       "required": False},
        },
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
            "Search for a regex pattern across multiple files and replace all matches atomically. "
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
}
