# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tool implementations for the coding agent.

This package replaces the former single-file ``localm/plugins/coder/tools.py``.
It is split by concern (base primitives, file ops, shell, git, env, web,
sub-agents, media, and the registry) but keeps the exact public surface the rest
of the coder and the test suite import from ``localm.plugins.coder.tools``:
``ToolResult``, ``ToolDef``, ``TOOL_REGISTRY``, ``SAFE_RESTRICTED_TOOLS``, the
shared helpers, and every ``tool_*`` function. ``TOOL_REGISTRY`` is re-exported by
name (the same dict mcp/plugin/skill registration mutates in place)."""

# Preserve the monkeypatch surface tests rely on: the single-file tools.py exposed
# ``subprocess`` and ``os`` as module attributes, so tests patch
# ``localm.plugins.coder.tools.subprocess.run`` and
# ``localm.plugins.coder.tools.os.environ``. The submodules (shell/git/env/media)
# import the SAME global modules, so re-exposing them on the package keeps those
# patches effective. (F401: deliberately re-exported, not used in this file.)
import os  # noqa: F401
import subprocess  # noqa: F401

from .base import ToolResult, _MAX_OUTPUT, _truncate, _confine
from .files import (
    _closest_snippet,
    _line_count,
    _render_notebook,
    _verify_syntax,
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
from .shell import _detect_test_runner, _needs_shell, tool_run_shell, tool_run_tests
from .git import (
    _git,
    tool_git_commit,
    tool_git_create_branch,
    tool_git_diff,
    tool_git_log,
    tool_git_push,
    tool_git_status,
)
from .env import _redact_env_value, tool_read_env
from .web import tool_fetch_url, tool_web_search
from .agents import tool_spawn_agent
from .media import tool_generate_image
from .registry import SAFE_RESTRICTED_TOOLS, TOOL_REGISTRY, ToolDef

__all__ = [
    "ToolResult", "ToolDef", "TOOL_REGISTRY", "SAFE_RESTRICTED_TOOLS",
    "_confine", "_truncate", "_MAX_OUTPUT", "_render_notebook", "_line_count",
    "_closest_snippet", "_verify_syntax", "_needs_shell", "_detect_test_runner",
    "_git", "_redact_env_value",
    "tool_read_file", "tool_write_file", "tool_edit_file", "tool_patch_file",
    "tool_list_dir", "tool_tree", "tool_edit_notebook_cell", "tool_search_files",
    "tool_grep", "tool_search_replace", "tool_run_shell", "tool_run_tests",
    "tool_git_status", "tool_git_diff", "tool_git_log", "tool_git_commit",
    "tool_git_push", "tool_git_create_branch", "tool_read_env",
    "tool_fetch_url", "tool_web_search", "tool_spawn_agent", "tool_generate_image",
]
