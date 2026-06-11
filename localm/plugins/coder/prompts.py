"""
System prompt templates for localcoder agents.

Per-model-family tuning is applied by ``build_system_prompt`` based on the
``model_name`` parameter.  Families:

  gemma     — gemma / gemma2 / gemma3 / gemma4
              Informs the model that its native <|tool_call> format is also
              accepted (parser.py handles both XML and native).

  thinking  — deepseek-r1 / qwq / qwen3 (thinking/reasoning variants)
              Adds an explicit <think>…</think> scratchpad instruction before
              the tool-use section; these models produce better results when
              given an explicit reasoning channel.

  small     — phi / phi2 / phi3 / phi4 / phi-mini / tiny
              Compressed prompt: condensed tool list + 5-rule set.  Smaller
              context window means every token counts.

  default   — llama / mistral / qwen2 / codellama / and everything else
              Standard XML tool-call format with the full tool list.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .indexer import ProjectMap


# ---------------------------------------------------------------------------
#  Family detection
# ---------------------------------------------------------------------------

def detect_model_family(model_name: str) -> str:
    """
    Return a family tag from a model name string.

    Tags: "gemma" | "thinking" | "small" | "default"
    """
    n = model_name.lower()

    if n.startswith("gemma"):
        return "gemma"

    # Thinking / reasoning models (chain-of-thought fine-tunes)
    if any(p in n for p in ("deepseek-r1", "deepseek_r1", "qwq", "qwen3")):
        return "thinking"

    # Small / resource-constrained models
    if n.startswith("phi") or any(p in n for p in ("-tiny", "tiny-")):
        return "small"

    return "default"


# ---------------------------------------------------------------------------
#  Shared tool documentation
# ---------------------------------------------------------------------------

_TOOL_DOCS_FULL = """\
## read_file — Read a file
{"name": "read_file", "args": {"path": "src/main.py"}}

## write_file — Create or overwrite a file
{"name": "write_file", "args": {"path": "src/new.py", "content": "..."}}

## edit_file — Replace exact text in a file (read it first!)
{"name": "edit_file", "args": {"path": "src/main.py", "old": "def foo():", "new": "def foo(x: int):"}}

## patch_file — Apply a unified diff to a file (read it first; more reliable than edit_file for multi-hunk changes)
{"name": "patch_file", "args": {"path": "src/main.py", "diff": "--- a/src/main.py\\n+++ b/src/main.py\\n@@ -10,4 +10,5 @@\\n context line\\n-old line\\n+new line\\n+added line\\n"}}

## run_shell — Execute a shell command
{"name": "run_shell", "args": {"command": "python -m pytest tests/ -x"}}

## list_dir — List a directory
{"name": "list_dir", "args": {"path": "src/"}}

## search_files — Glob-search for files
{"name": "search_files", "args": {"pattern": "**/*.py", "path": "."}}

## grep — Regex-search file contents
{"name": "grep", "args": {"pattern": "def authenticate", "path": "src/", "glob": "**/*.py"}}

## git_status — Show working-tree status
{"name": "git_status", "args": {}}

## git_diff — Show unstaged (or staged) changes
{"name": "git_diff", "args": {"staged": false}}

## git_log — Show recent commits
{"name": "git_log", "args": {"n": 10}}

## fetch_url — Fetch a URL and return plain-text content (HTML stripped)
{"name": "fetch_url", "args": {"url": "https://docs.python.org/3/library/pathlib.html"}}

## spawn_agent — Delegate a focused task to a sub-agent
{"name": "spawn_agent", "args": {"task": "Analyse auth.py for SQL injection", "files": ["src/auth.py"], "name": "security-reviewer"}}

## generate_image — Generate a high-quality image locally using FLUX
{"name": "generate_image", "args": {"prompt": "A cute orange cat coding on a mechanical keyboard, photorealistic, 4k", "output_path": "images/cat.png"}}"""

# Condensed list for small/phi models — one line per tool, no JSON examples
_TOOL_DOCS_BRIEF = """\
read_file(path)                       — read a file
write_file(path, content)             — create or overwrite a file
edit_file(path, old, new)             — replace exact text (read first!)
patch_file(path, diff)                — apply a unified diff (read first!)
run_shell(command, [timeout])         — run a shell command
list_dir([path])                      — list a directory
search_files(pattern, [path])         — glob-search for files
grep(pattern, [path], [glob])         — regex-search file contents
fetch_url(url)                        — fetch a URL as plain text
spawn_agent(task, [files], [name])    — delegate a sub-task"""


# ---------------------------------------------------------------------------
#  Per-family sections
# ---------------------------------------------------------------------------

def _tool_call_block(family: str) -> str:
    """Return the TOOL USE / format instruction block for the given family."""
    if family == "gemma":
        return """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call a tool by writing EXACTLY one of these formats — nothing else:

Preferred (XML):
<tool_call>
{{"name": "TOOL_NAME", "args": {{...}}}}
</tool_call>

Also accepted (native Gemma format):
<|tool_call>call:TOOL_NAME{{"key": "value"}}<tool_call|>

You will receive the result in a <tool_result> block, then continue.
You may call multiple tools in sequence across turns."""

    if family == "small":
        return """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call a tool using this EXACT format:

<tool_call>
{{"name": "TOOL_NAME", "args": {{...}}}}
</tool_call>

Result arrives in <tool_result>, then continue."""

    # default + thinking — identical call format
    return """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL USE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call a tool by writing EXACTLY this format — nothing else:

<tool_call>
{{"name": "TOOL_NAME", "args": {{...}}}}
</tool_call>

You will receive the result in a <tool_result> block, then continue.
You may call multiple tools in sequence across turns."""


def _thinking_hint(family: str) -> str:
    """Return the reasoning scratchpad instruction for thinking models, else ''."""
    if family != "thinking":
        return ""
    return """\

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REASONING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Before calling a tool or composing your final answer, reason privately
inside <think>…</think> tags.  Your thinking is never shown to the user.
Use it to plan your approach, check assumptions, and decide which tool to
call next.
"""


def _rules_section(family: str) -> str:
    if family == "small":
        return """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Always read_file before edit_file or patch_file.
2. Use relative paths.
3. Run tests after changes (pytest / cargo test / npm test).
4. Summarise what you changed when done.
5. If a tool errors, diagnose and retry."""

    return """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Always read_file before edit_file or patch_file — you need the exact text.
2. Use relative paths unless an absolute path is necessary.
3. Prefer edit_file for single small changes; patch_file for multi-hunk edits; write_file for new files or complete rewrites.
4. Run tests after code changes (run_shell: pytest / cargo test / npm test / …).
5. For complex tasks, use spawn_agent to delegate focused sub-tasks.
6. When you are done, give a concise summary of what you changed and why.
7. If a tool returns an error, diagnose and retry before giving up.
8. Ask for clarification only if the task is genuinely ambiguous."""


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_system_prompt(
    cwd: Path,
    agent_name: str = "localcoder",
    project_map: "ProjectMap | None" = None,
    memory: str = "",
    model_name: str = "",
    extra_tool_docs: str = "",
) -> str:
    """
    Build the system prompt for the main agent.

    Parameters
    ----------
    cwd:
        Working directory shown to the model.
    agent_name:
        Display name used in the identity line.
    project_map:
        Codebase index injected as context; omitted when empty.
    memory:
        Content of LOCALCODER.md; injected under "## Project Memory".
    model_name:
        Used to select per-family prompt tuning (Gemma / thinking / small / default).
    extra_tool_docs:
        Additional tool documentation appended after the built-in tool list
        (e.g. dynamically registered MCP tools).
    """
    family = detect_model_family(model_name) if model_name else "default"

    map_section = ""
    if project_map is not None and project_map.file_count() > 0:
        map_section = f"\n{project_map.to_context_string()}\n"

    memory_section = ""
    if memory:
        memory_section = f"\n## Project Memory\n\n{memory}\n"

    tool_docs = _TOOL_DOCS_BRIEF if family == "small" else _TOOL_DOCS_FULL
    tool_block = _tool_call_block(family)
    think_hint = _thinking_hint(family)
    rules      = _rules_section(family)

    # Identity line — terser for small models
    if family == "small":
        identity = (
            f"You are {agent_name}, an AI coding assistant.\n"
            f"Working directory: {cwd}"
        )
    else:
        identity = (
            f"You are {agent_name}, an expert AI coding assistant running fully offline.\n"
            f"You help the user write, debug, refactor, and understand code.\n\n"
            f"Working directory: {cwd}"
        )

    extra_section = f"\n{extra_tool_docs}\n" if extra_tool_docs else ""

    return (
        f"{identity}"
        f"{map_section}"
        f"{memory_section}"
        f"\n{think_hint}"
        f"{tool_block}\n\n"
        f"AVAILABLE TOOLS\n"
        f"━━━━━━━━━━━━━━━\n\n"
        f"{tool_docs}\n"
        f"{extra_section}\n"
        f"{rules}\n"
    )


def build_subagent_system_prompt(
    cwd: Path,
    role: str,
    model_name: str = "",
) -> str:
    """Leaner prompt for sub-agents — focused on their specific role."""
    family = detect_model_family(model_name) if model_name else "default"
    think_hint = _thinking_hint(family)

    call_fmt: str
    if family == "gemma":
        call_fmt = (
            "XML format:  <tool_call>\n"
            '{{"name": "TOOL_NAME", "args": {{...}}}}\n'
            "</tool_call>\n\n"
            "Native format also accepted:\n"
            "<|tool_call>call:TOOL_NAME{{...}}<tool_call|>"
        )
    else:
        call_fmt = (
            "<tool_call>\n"
            '{{"name": "TOOL_NAME", "args": {{...}}}}\n'
            "</tool_call>"
        )

    return (
        f"You are a specialised coding sub-agent with the role: {role}.\n"
        f"Working directory: {cwd}\n"
        f"{think_hint}\n"
        f"You have the same tools as the main agent (read_file, write_file, edit_file,\n"
        f"patch_file, run_shell, list_dir, search_files, grep). Use them as needed.\n\n"
        f"Call tools with:\n{call_fmt}\n\n"
        f"Complete your assigned task, then return a clear summary of findings or changes.\n"
        f"Do not ask questions — make sensible decisions and document your reasoning.\n"
    )
