"""
Rich-based terminal display for localcoder.

Keeps all formatting in one place so the Agent class stays clean.
"""

from __future__ import annotations

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
    model: str, cwd: Path, agent_name: str = "localcoder", file_count: int = 0
) -> None:
    map_info = f"  ·  [dim]indexed:[/dim] [dim]{file_count} files[/dim]" if file_count else ""
    console.print(Panel(
        f"[bold cyan]{agent_name}[/bold cyan]  ·  "
        f"[dim]model:[/dim] [bold]{model}[/bold]  ·  "
        f"[dim]cwd:[/dim] [dim]{cwd}[/dim]{map_info}\n"
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

def print_turn_divider(turn: int, total_tokens: int = 0) -> None:
    tokens_str = f"  ·  ~{total_tokens:,} tokens" if total_tokens else ""
    console.print(f"\n[dim]── turn {turn}{tokens_str} ──────────────────────────────────────[/dim]")


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
  [bold]/cwd[/bold]                  show working directory
  [bold]/cd <path>[/bold]            change working directory
  [bold]/reindex[/bold]              rebuild the codebase index
  [bold]/verbose[/bold]              toggle verbose tool output
  [bold]/approve[/bold]              toggle auto-approve for destructive tools
  [bold]/history[/bold]              show turn count, context usage, and index size
  [bold]/memory[/bold]               show current project memory
  [bold]/remember <text>[/bold]      append a note to LOCALCODER.md
  [bold]/forget <pattern>[/bold]     remove memory bullets matching pattern
  [bold]/save[/bold]                 save conversation to JSON
"""

def print_help() -> None:
    console.print(HELP_TEXT)
