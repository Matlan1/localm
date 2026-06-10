"""
Rich-based terminal display for localcoder.

Keeps all formatting in one place so the Agent class stays clean.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

console = Console(highlight=False)


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
    args_text = ", ".join(
        f"[cyan]{k}[/cyan]=[dim]{repr(str(v)[:60])}[/dim]"
        for k, v in args.items()
        if k not in ("content",)   # don't show full file contents inline
    )
    # Show content length instead of dumping it
    if "content" in args:
        n = len(args["content"].splitlines())
        args_text += f", [cyan]content[/cyan]=[dim]<{n} lines>[/dim]"

    console.print(f"  [bold yellow]●[/bold yellow] [bold]{tool_name}[/bold]({args_text})")


def print_tool_result(tool_name: str, result, verbose: bool = False) -> None:
    """Print the one-line summary, optionally expanding the full output."""
    icon  = "[green]✓[/green]" if result.ok else "[red]✗[/red]"
    trunc = " [dim](truncated)[/dim]" if result.truncated else ""
    console.print(f"    {icon} [dim]{result.summary}{trunc}[/dim]")

    if verbose and result.ok and result.output:
        # Collapse tool result behind a divider for readability
        console.print(f"    [dim]{'─' * 60}[/dim]")
        for line in result.output.splitlines()[:40]:
            console.print(f"    [dim]{line}[/dim]")
        if len(result.output.splitlines()) > 40:
            console.print("    [dim]... (use /verbose to see full output)[/dim]")


def print_tool_error(tool_name: str, message: str) -> None:
    console.print(f"    [red]✗[/red] [dim]{tool_name}: {message}[/dim]")


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
    console.print(text)


def print_streaming_token(token: str) -> None:
    console.print(token, end="", highlight=False)


def print_streaming_done() -> None:
    console.print()


# ---------------------------------------------------------------------------
#  Turn divider
# ---------------------------------------------------------------------------

def print_turn_divider(turn: int, total_tokens: int = 0, turn_tokens: int = 0) -> None:
    parts = [f"── turn {turn}"]
    if turn_tokens:
        parts.append(f"~{turn_tokens:,} tok this turn")
    if total_tokens:
        parts.append(f"~{total_tokens:,} total")
    body = "  ·  ".join(parts)
    console.print(f"\n[dim]{body} ──────────────────────────────────────[/dim]")


# ---------------------------------------------------------------------------
#  Status messages
# ---------------------------------------------------------------------------

def print_info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def print_warning(msg: str) -> None:
    console.print(f"[yellow]{msg}[/yellow]")


def print_error(msg: str) -> None:
    console.print(f"[red]{msg}[/red]")


def print_success(msg: str) -> None:
    console.print(f"[green]{msg}[/green]")


# ---------------------------------------------------------------------------
#  Server management
# ---------------------------------------------------------------------------

def print_server_starting(model: str, port: int) -> None:
    console.print(f"[dim]Starting localm serve {model} on port {port}…[/dim]")


def print_server_ready(port: int) -> None:
    console.print(f"[green]✓[/green] [dim]Model server ready on port {port}[/dim]")


def print_server_timeout() -> None:
    console.print("[red]Server did not start in time. Is localm installed?[/red]")


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
        prompt = f"Apply changes to [bold]{path_label}[/bold]? [y/N] "
        answer = console.input(f"[yellow]{prompt}[/yellow]").strip().lower()
        return answer in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        return False


# ---------------------------------------------------------------------------
#  Confirmation prompt (for destructive tools)
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    try:
        answer = console.input(f"[yellow]{prompt} [y/N][/yellow] ").strip().lower()
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
  [bold]/undo[/bold]                 revert the last file write or edit
  [bold]/resume[/bold]               resume an interrupted session
  [bold]/compact[/bold]              summarise old turns to free context space
  [bold]/memory[/bold]               show current project memory
  [bold]/remember <text>[/bold]      append a note to LOCALCODER.md
  [bold]/forget <pattern>[/bold]     remove memory bullets matching pattern
  [bold]/save[/bold]                 save conversation to JSON

[dim]Session modes (set with --mode at startup or mode = "..." in .localcoder/config.toml):
  privacy  nothing saved automatically (default)
             - readline history suppressed (~/.python_history)
             - subprocess shell history vars zeroed (HISTFILE, HISTSIZE, …)
             - PSReadLine / bash / zsh history files scrubbed of localcoder
               lines on exit  (cmd.exe: no persistent history anyway)
             - /save requires explicit confirmation
  log      JSONL audit trail → ~/.localm/sessions/
  full     JSONL audit trail + markdown transcript → .localcoder/sessions/
  Note: privacy mode cannot suppress OS process-creation logs (Event Log /
  auditd) or DNS/network logs from fetch_url.
  Online providers (--online/--anthropic) are always explicit opt-in;
  privacy mode warns if both are active simultaneously.[/dim]
"""

def print_help() -> None:
    console.print(HELP_TEXT)
