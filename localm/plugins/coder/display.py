# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Rich-based terminal display for localcoder.

Keeps all formatting in one place so the Agent class stays clean.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console(highlight=False)

_ESC = "\x1b"


def _strip_esc(s: str) -> str:
    """Remove raw ANSI escape bytes so externally-controlled text (model
    output, tool args/results, an exception message) can never carry a
    terminal control sequence (e.g. a screen clear) to the user."""
    return s.replace(_ESC, "") if _ESC in s else s


def _sanitized_text(s: str, *, style: str | None = None) -> Text:
    """Externally-controlled text wrapped so Rich can never parse it as
    markup (an unmatched tag - a leaked ``[/INST]`` control token, ``[/b]``,
    a markdown link's ``[label]`` - raises MarkupError from a plain
    console.print) and so a raw ANSI escape cannot reach the terminal."""
    return Text(_strip_esc(s), style=style)


# ---------------------------------------------------------------------------
#  Welcome banner
# ---------------------------------------------------------------------------

def print_banner(
    model: str,
    cwd: Path,
    agent_name: str = "localcoder",
    file_count: int = 0,
    session_mode: "str | None" = None,
) -> None:
    map_info  = f"  ·  [dim]indexed:[/dim] [dim]{file_count} files[/dim]" if file_count else ""
    mode_colours = {"privacy": "dim", "log": "yellow", "full": "green"}
    mode_info = ""
    if session_mode:
        colour   = mode_colours.get(str(session_mode).lower(), "dim")
        mode_info = f"  ·  [{colour}]mode: {session_mode}[/{colour}]"
    console.print(Panel(
        f"[bold cyan]{agent_name}[/bold cyan]  ·  "
        f"[dim]model:[/dim] [bold]{model}[/bold]  ·  "
        f"[dim]cwd:[/dim] [dim]{cwd}[/dim]{map_info}{mode_info}\n"
        "[dim]Type your task, or [bold]/help[/bold] for commands. "
        "Ctrl+C or [bold]/exit[/bold] to quit.[/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
#  Tool call display
# ---------------------------------------------------------------------------

def print_tool_call(tool_name: str, args: dict, index: int = 0) -> None:
    # The tool name and every argument key/value are model-generated text,
    # composed via Text so none of it is parsed as Rich markup.
    line = Text("  ")
    line.append("● ", style="bold yellow")
    line.append(_strip_esc(tool_name), style="bold")
    line.append("(")
    first = True
    for k, v in args.items():
        if k == "content":   # don't show full file contents inline
            continue
        if not first:
            line.append(", ")
        first = False
        line.append(_strip_esc(k), style="cyan")
        line.append("=")
        line.append(_strip_esc(repr(str(v)[:60])), style="dim")
    if "content" in args:
        # Show content length instead of dumping it
        n = len(args["content"].splitlines())
        if not first:
            line.append(", ")
        line.append("content", style="cyan")
        line.append(f"=<{n} lines>", style="dim")
    line.append(")")
    console.print(line)


def print_tool_result(tool_name: str, result, verbose: bool = False) -> None:
    """Print the one-line summary, optionally expanding the full output."""
    icon, icon_style = ("✓", "green") if result.ok else ("✗", "red")
    trunc = " (truncated)" if result.truncated else ""
    line = Text("    ")
    line.append(icon + " ", style=icon_style)
    line.append(_strip_esc(result.summary + trunc), style="dim")
    console.print(line)

    if verbose and result.ok and result.output:
        # Collapse tool result behind a divider for readability
        console.print(_sanitized_text("    " + "─" * 60, style="dim"))
        for out_line in result.output.splitlines()[:40]:
            console.print(_sanitized_text("    " + out_line, style="dim"))
        if len(result.output.splitlines()) > 40:
            console.print(_sanitized_text(
                "    ... (use /verbose to see full output)", style="dim"))


def print_tool_error(tool_name: str, message: str) -> None:
    line = Text("    ")
    line.append("✗ ", style="red")
    line.append(_strip_esc(f"{tool_name}: {message}"), style="dim")
    console.print(line)


# ---------------------------------------------------------------------------
#  Agent response
# ---------------------------------------------------------------------------

def print_thinking(label: str = "Thinking…") -> None:
    console.print(f"\n[dim]{label}[/dim]")


def print_assistant_label(name: str = "Agent") -> None:
    console.print(f"\n[bold blue]{name}[/bold blue]: ", end="")


def print_assistant_response(text: str, name: str = "Agent") -> None:
    """Render the final response text.  Uses Markdown if it looks like Markdown."""
    text = text.strip()
    if not text:
        return
    if any(marker in text for marker in ("```", "**", "##", "- ", "1. ")):
        try:
            console.print(Markdown(text))
            return
        except Exception:
            pass
    console.print(_sanitized_text(text))


def print_streaming_token(token: str) -> None:
    # Append-only streaming with end="": styling escapes only, never cursor
    # repositioning, an alt-screen switch, or a Live region, so the terminal
    # keeps ownership of scrolling.
    # See test_coder_stream_renderer_streams_without_viewport_control.
    #
    # Model-generated text; an unmatched closing tag raises MarkupError from a
    # plain console.print. See test_streaming_token_survives_a_leaked_control_token.
    console.print(_sanitized_text(token), end="", highlight=False)


def print_reasoning_token(token: str) -> None:
    """Stream a thinking model's reasoning dimmed, so it reads as an aside next
    to the visible answer rather than being indistinguishable from it."""
    console.print(_sanitized_text(token, style="dim"), end="", highlight=False)


def print_streaming_done() -> None:
    console.print()


# ---------------------------------------------------------------------------
#  Turn divider
# ---------------------------------------------------------------------------

def print_turn_divider(turn: int, total_tokens: int = 0, turn_tokens: int = 0,
                       ctx_ratio: float | None = None) -> None:
    parts = [f"── turn {turn}"]
    if turn_tokens:
        parts.append(f"~{turn_tokens:,} tok this turn")
    if total_tokens:
        parts.append(f"~{total_tokens:,} total")
    if ctx_ratio is not None and ctx_ratio > 0:
        pct = min(ctx_ratio, 1.0) * 100
        color = "red" if ctx_ratio >= 0.85 else ("yellow" if ctx_ratio >= 0.65 else "dim")
        parts.append(f"[{color}]ctx {pct:.0f}%[/{color}]")
    body = "  ·  ".join(parts)
    console.print(f"\n[dim]{body} ──────────────────────────────────────[/dim]")


# ---------------------------------------------------------------------------
#  Status messages
# ---------------------------------------------------------------------------

def print_info(msg: str) -> None:
    console.print(_sanitized_text(msg, style="dim"))


def print_warning(msg: str) -> None:
    console.print(_sanitized_text(msg, style="yellow"))


def print_error(msg: str) -> None:
    # msg can embed an exception's own text, which may itself carry markup, so
    # it is sanitized rather than printed directly. See
    # test_print_error_survives_a_message_that_would_itself_crash_markup.
    console.print(_sanitized_text(msg, style="red"))


def print_success(msg: str) -> None:
    console.print(_sanitized_text(msg, style="green"))


# ---------------------------------------------------------------------------
#  Server management
# ---------------------------------------------------------------------------

def print_server_starting(model: str, port: int) -> None:
    console.print(f"[dim]Starting localm serve {model} on port {port}…[/dim]")


def print_server_ready(port: int) -> None:
    console.print(f"[green]✓[/green] [dim]Model server ready on port {port}[/dim]")


def print_server_timeout(stderr_tail: str = "") -> None:
    msg = "Server did not start in time. Is localm installed?"
    if stderr_tail:
        msg += f"\n{stderr_tail}"
    console.print(_sanitized_text(msg, style="red"))


# ---------------------------------------------------------------------------
#  Diff preview
# ---------------------------------------------------------------------------

def print_diff_preview(
    old_text: str,
    new_text: str,
    path_label: str = "",
    max_lines: int = 200,
) -> None:
    """
    Print a coloured unified diff between *old_text* and *new_text*.

    Parameters
    ----------
    old_text:
        Current file content (empty string if the file doesn't exist yet).
    new_text:
        Proposed new content.
    path_label:
        Shown in the diff header (e.g. "src/main.py").
    max_lines:
        Truncate the displayed diff at this many lines.
    """
    from_label = f"a/{path_label}" if path_label else "a/current"
    to_label   = f"b/{path_label}" if path_label else "b/proposed"

    diff_lines = list(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=from_label,
        tofile=to_label,
    ))

    if not diff_lines:
        console.print("[dim]  (no changes)[/dim]")
        return

    diff_text = "".join(diff_lines[:max_lines])
    if len(diff_lines) > max_lines:
        diff_text += f"\n... ({len(diff_lines) - max_lines} more lines truncated)\n"

    console.print()
    console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))


def confirm_diff(path_label: str) -> bool:
    """Prompt 'Apply changes to <path>? [y/N]' and return True on yes."""
    try:
        # path_label is model-chosen text, composed via Text rather than
        # embedded in a markup string.
        prompt = Text("Apply changes to ", style="yellow")
        prompt.append(_strip_esc(path_label), style="bold yellow")
        prompt.append("? [y/N] ", style="yellow")
        answer = console.input(prompt).strip().lower()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ---------------------------------------------------------------------------
#  Confirmation prompt (for destructive tools)
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    try:
        line = _sanitized_text(f"{prompt} [y/N] ", style="yellow")
        answer = console.input(line).strip().lower()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ---------------------------------------------------------------------------
#  Help
# ---------------------------------------------------------------------------

HELP_TEXT = """\
[bold cyan]localcoder commands[/bold cyan]

  [bold]/help[/bold]                 this message
  [bold]/exit[/bold]                 quit (also: exit, quit, q)
  [bold]/clear[/bold]                clear conversation history
  [bold]/model[/bold]                show current model
  [bold]/mode[/bold]                 show session persistence mode (privacy/log/full)
  [bold]/cwd[/bold]                  show working directory
  [bold]/cd <path>[/bold]            change working directory
  [bold]/reindex[/bold]              rebuild the codebase index
  [bold]/verbose[/bold]              toggle verbose tool output
  [bold]/approve[/bold]              toggle auto-approve for destructive tools
  [bold]/history[/bold]              show turn count, context usage, and index size
  [bold]/changes[/bold]              list every file this session has changed
  [bold]/diff [path][/bold]          cumulative diff of session changes (all or one file)
  [bold]/undo[/bold]                 revert the last file write or edit
  [bold]/sessions[/bold]             list this project's saved (interrupted) sessions
  [bold]/resume [id][/bold]          resume the most recent interrupted session, or a
                        specific one by id from /sessions
  [bold]/compact[/bold]              summarise old turns to free context space
  [bold]/memory[/bold]               show current project memory
  [bold]/remember <text>[/bold]      append a note to LOCALCODER.md
  [bold]/forget <pattern>[/bold]     remove memory bullets matching pattern
  [bold]/save[/bold]                 save conversation to JSON
  [bold]/export [path][/bold]        export session transcript to Markdown
  [bold]/scope [glob][/bold]         show or set the file-access scope (e.g. src/**/*.py)
  [bold]/scope clear[/bold]          remove the active scope restriction
  [bold]/verify [cmd|auto|off][/bold] command that must exit 0 before a turn that
                        changed files may finish (run by the harness, not the model)
  [bold]/goal [cmd|auto|off][/bold] iterate the next task until this command exits 0,
                        re-running the whole task with each failure fed back
  [bold]/review[/bold]               a reviewer model checks the current diff for blocking
                        issues right now, even with coder_review off

[dim]Tab completes commands and project paths (where readline is available);
REPL history persists across sessions in log/full modes, never in privacy.[/dim]

[dim]Session modes (set with --mode at startup or mode = "..." in .localcoder/config.toml):
  privacy  nothing saved automatically (default)
             - readline history suppressed (~/.python_history)
             - subprocess shell history vars zeroed (HISTFILE, HISTSIZE, …)
             - PSReadLine / bash / zsh history files scrubbed of localcoder
               lines on exit  (cmd.exe: no persistent history anyway)
             - /save requires explicit confirmation
  log      JSONL audit trail → <data dir>/sessions/
  full     JSONL audit trail + markdown transcript → .localcoder/sessions/
  Note: privacy mode cannot suppress OS process-creation logs (Event Log /
  auditd) or DNS/network logs from fetch_url.
  Online providers (--online/--anthropic) are always explicit opt-in;
  privacy mode warns if both are active simultaneously.[/dim]
"""

def print_help() -> None:
    console.print(HELP_TEXT)
