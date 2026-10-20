# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Rich-based terminal display for localcoder.

Keeps all formatting in one place so the Agent class stays clean.
"""

from __future__ import annotations

import contextlib
import contextvars
import difflib
import re
import threading
import time
from pathlib import Path

from localm.console import SafeConsole as Console, strip_controls
from rich.markdown import Markdown
from rich.markup import escape as _markup_escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

_SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_last_diff_preview: tuple[str, str, str, bool] | None = None
_last_confirm_always: bool = False
_last_reject_reason: str = ""
_offer_session_allow: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "coder_offer_session_allow", default=False)


def consume_confirm_always() -> bool:
    """Return and clear the session auto-approve flag set by the last confirmation prompt."""
    global _last_confirm_always
    val = _last_confirm_always
    _last_confirm_always = False
    return val


def consume_reject_reason() -> str:
    """Return and clear the optional rejection reason entered at the last confirmation prompt."""
    global _last_reject_reason
    val = _last_reject_reason
    _last_reject_reason = ""
    return val


class ActivitySpinner:
    """Single-line carriage-return activity indicator using SGR styling only. See test_coder_stream_renderer_streams_without_viewport_control."""

    def __init__(
        self,
        label: str = "Thinking...",
        *,
        console_obj: Console | None = None,
        interval: float = 0.1,
    ) -> None:
        self._console_override = console_obj
        self._label = label
        self._detail = ""
        self._interval = interval
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._last_len = 0
        self._frame_idx = 0

    def _active_console(self) -> Console:
        return self._console_override if self._console_override is not None else console

    def _is_interactive_tty(self, c: Console) -> bool:
        is_term = getattr(c, "is_terminal", None)
        return isinstance(is_term, bool) and is_term

    def start(self, label: str | None = None) -> ActivitySpinner:
        with self._lock:
            if label is not None:
                self._label = label
            self._detail = ""
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_evt.clear()
            self._start_time = time.monotonic()
            self._frame_idx = 0
            c = self._active_console()
            if not self._is_interactive_tty(c):
                return self
            self._render_frame_locked(c)
            self._thread = threading.Thread(
                target=self._run, name="coder-spinner", daemon=True
            )
            self._thread.start()
        return self

    def update(self, label: str | None = None, detail: str | None = None) -> None:
        with self._lock:
            if label is not None:
                self._label = label
            if detail is not None:
                self._detail = detail

    def _run(self) -> None:
        while not self._stop_evt.wait(self._interval):
            with self._lock:
                if self._stop_evt.is_set():
                    break
                c = self._active_console()
                if self._is_interactive_tty(c):
                    self._render_frame_locked(c)

    def _render_frame_locked(self, c: Console) -> None:
        elapsed = time.monotonic() - self._start_time
        frame = _SPINNER_FRAMES[self._frame_idx % len(_SPINNER_FRAMES)]
        self._frame_idx += 1
        max_w = max(20, (getattr(c, "width", 80) or 80) - 2)
        line = Text("  ")
        line.append(f"{frame} ", style="cyan")
        line.append(_strip_esc(self._label), style="dim")
        if self._detail:
            line.append(f" · {_strip_esc(self._detail)}", style="dim cyan")
        if elapsed >= 0.4:
            line.append(f" ({elapsed:.1f}s)", style="dim")
        if len(line) > max_w:
            line.truncate(max_w)
        plain_len = len(line.plain)
        if plain_len < self._last_len:
            line.append(" " * (self._last_len - plain_len))
        self._last_len = plain_len
        c.print(line, end="\r", soft_wrap=True, highlight=False)

    def stop(self) -> None:
        self._stop_evt.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=0.3)
        with self._lock:
            self._thread = None
            c = self._active_console()
            if self._last_len > 0 and self._is_interactive_tty(c):
                c.print(
                    Text(" " * self._last_len),
                    end="\r",
                    soft_wrap=True,
                    highlight=False,
                )
            self._last_len = 0

    def __enter__(self) -> ActivitySpinner:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def spinner_available(console_obj: Console | None = None) -> bool:
    """True when an ActivitySpinner would animate on *console_obj* (default:
    the coder console), i.e. it is a real terminal."""
    return ActivitySpinner(console_obj=console_obj)._is_interactive_tty(
        console_obj if console_obj is not None else console)


def _strip_esc(s: str) -> str:
    """Remove terminal control characters (ESC and every other C0/C1 control
    except newline and tab) so externally-controlled text (model output, tool
    args/results, an exception message) can never carry a terminal control
    sequence (e.g. a screen clear) to the user."""
    return strip_controls(s)


def _sanitized_text(s: str, *, style: str | None = None) -> Text:
    """Externally-controlled text wrapped so Rich can never parse it as
    markup (an unmatched tag - a leaked ``[/INST]`` control token, ``[/b]``,
    a markdown link's ``[label]`` - raises MarkupError from a plain
    console.print) and so a raw ANSI escape cannot reach the terminal."""
    return Text(_strip_esc(s), style=style)


def safe_markup(s) -> str:
    """*s* rendered inert for use inside a Rich markup string.

    Returns a ``str`` with every ANSI escape byte removed and every markup
    tag escaped, so the value renders as the literal text it is. Use this for
    any externally-controlled value (model output, a model-chosen path or
    label, a tool result, an exception message) that is interpolated into a
    markup string or passed to a bare ``console.print``.

    ``_sanitized_text`` is the equivalent for a value that is printed on its
    own and needs no surrounding markup; this one composes into an f-string.
    """
    return _markup_escape(_strip_esc(str(s)))


# ---------------------------------------------------------------------------
#  Welcome banner
# ---------------------------------------------------------------------------

def print_banner(
    model: str,
    cwd: Path,
    agent_name: str = "localcoder",
    file_count: int = 0,
    session_mode: str | None = None,
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
        "[dim]Type your task, [bold]@file[/bold] to attach context, or [bold]/help[/bold] for commands. "
        "Ctrl+C or [bold]/exit[/bold] to quit.[/dim]",
        border_style="dim cyan",
        padding=(0, 1),
    ))


# ---------------------------------------------------------------------------
#  Tool call display
# ---------------------------------------------------------------------------

def print_tool_call(tool_name: str, args: dict, index: int = 0) -> None:
    console.print(_tool_call_line(tool_name, args))


def print_progress(text: str) -> None:
    """One dim line on stderr naming what a run without a live display is
    doing."""
    err_console.print(_sanitized_text(f"  {text}", style="dim"))


def tool_call_text(tool_name: str, args: dict) -> str:
    """The tool call as one plain line, as print_tool_call shows it."""
    return _tool_call_line(tool_name, args).plain.strip()


def print_progress_tool_call(tool_name: str, args: dict) -> None:
    """The tool call a run without a live display is about to make, on
    stderr."""
    err_console.print(_tool_call_line(tool_name, args))


def _tool_call_line(tool_name: str, args: dict) -> Text:
    # The tool name and every argument key/value are model-generated text,
    # composed via Text so none of it is parsed as Rich markup.
    line = Text("  ")
    line.append("● ", style="bold yellow")
    line.append(_strip_esc(tool_name), style="bold")
    line.append("(")
    first = True
    for k, v in args.items():
        if k in ("content", "diff"):
            continue
        if not first:
            line.append(", ")
        first = False
        line.append(_strip_esc(k), style="cyan")
        line.append("=")
        if k in ("old", "new") and isinstance(v, str) and ("\n" in v or len(v) > 48):
            n = len(v.splitlines()) or 1
            line.append(f"<{n} lines>", style="dim")
        elif k == "edits" and isinstance(v, list):
            line.append(f"<{len(v)} edits>", style="dim")
        elif k == "items" and isinstance(v, list):
            line.append(f"<{len(v)} items>", style="dim")
        else:
            line.append(_strip_esc(repr(str(v)[:60])), style="dim")
    for bulk_key in ("content", "diff"):
        if bulk_key in args and isinstance(args[bulk_key], str):
            n = len(args[bulk_key].splitlines())
            if not first:
                line.append(", ")
            first = False
            line.append(bulk_key, style="cyan")
            line.append(f"=<{n} lines>", style="dim")
    line.append(")")
    return line


_XML_WRAPPER_TAG_RE = re.compile(
    r"^\s*(?:<exit_code>[^<]*</exit_code>|<(?:stdout|stderr|output)>|</(?:stdout|stderr|output)>)\s*$"
)


def _clean_output_lines(raw_output: str) -> list[str]:
    """Return non-wrapper lines from tool output for display."""
    lines = []
    for raw_line in raw_output.splitlines():
        if _XML_WRAPPER_TAG_RE.match(raw_line):
            continue
        lines.append(raw_line)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def print_tool_result(
    tool_name: str,
    result,
    verbose: bool = False,
    *,
    elapsed: float | None = None,
    args: dict | None = None,
) -> None:
    """Print the one-line summary, todo checklist, error excerpt on failure, or verbose output."""
    icon, icon_style = ("✓", "green") if result.ok else ("✗", "red")
    trunc = " (truncated)" if getattr(result, "truncated", False) else ""
    line = Text("    ")
    line.append(icon + " ", style=icon_style)
    line.append(_strip_esc(str(result.summary) + trunc), style="dim")
    if elapsed is not None and elapsed >= 0.05:
        line.append(f" ({elapsed:.1f}s)", style="dim")
    console.print(line)

    if tool_name == "set_todos" and result.ok and isinstance(args, dict):
        items = args.get("items")
        if isinstance(items, list):
            for item in items[:20]:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "pending")).lower()
                task_text = str(item.get("task", item.get("content", ""))).strip()
                if not task_text:
                    continue
                todo_line = Text("      ")
                if status in ("done", "completed"):
                    todo_line.append("[✓] ", style="green")
                    todo_line.append(_strip_esc(task_text), style="dim strike")
                elif status == "in_progress":
                    todo_line.append("[▸] ", style="bold cyan")
                    todo_line.append(_strip_esc(task_text), style="bold")
                else:
                    todo_line.append("[ ] ", style="dim")
                    todo_line.append(_strip_esc(task_text), style="dim")
                console.print(todo_line)

    raw_output = getattr(result, "output", "")
    if not isinstance(raw_output, str) or not raw_output.strip():
        return

    if not result.ok:
        cleaned = _clean_output_lines(raw_output)
        summary_str = str(result.summary).strip()
        if cleaned and not (len(cleaned) == 1 and cleaned[0].strip() == summary_str):
            cap = 40 if verbose else 12
            excerpt = cleaned[-cap:] if len(cleaned) > cap else cleaned
            if len(cleaned) > cap:
                console.print(_sanitized_text(
                    f"      ... ({len(cleaned) - cap} earlier lines omitted)", style="dim"
                ))
            for out_line in excerpt:
                console.print(_sanitized_text("      " + out_line, style="red dim"))
        return

    if verbose and result.ok:
        console.print(_sanitized_text("    " + "─" * 60, style="dim"))
        out_lines = raw_output.splitlines()
        for out_line in out_lines[:40]:
            console.print(_sanitized_text("    " + out_line, style="dim"))
        if len(out_lines) > 40:
            console.print(_sanitized_text(
                "    ... (output capped at 40 lines)", style="dim"))


def print_tool_error(tool_name: str, message: str) -> None:
    line = Text("    ")
    line.append("✗ ", style="red")
    line.append(_strip_esc(f"{tool_name}: {message}"), style="dim")
    console.print(line)


# ---------------------------------------------------------------------------
#  Agent response
# ---------------------------------------------------------------------------

def print_thinking(label: str = "Thinking...") -> None:
    console.print(f"\n[dim]{label}[/dim]")


def print_status(text: str) -> None:
    """One dim, indented line naming what the model is doing before it
    answers."""
    console.print(_sanitized_text(f"  {text}", style="dim"))


def print_reasoning_header() -> None:
    """Print a dim header before streaming reasoning tokens."""
    console.print("\n[dim]  💭 Thinking:[/dim]\n  ", end="")


def print_assistant_label(name: str = "Agent") -> None:
    console.print(f"\n[bold blue]{name}[/bold blue]: ", end="")


def print_assistant_response(text: str, name: str = "Agent") -> None:
    """Render the final response text. Uses Markdown if it looks like Markdown."""
    text = text.strip()
    if not text:
        return
    if any(marker in text for marker in ("```", "**", "##", "- ", "1. ")):
        try:
            console.print(Markdown(_strip_esc(text)))
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
#  Turn divider & task summary footer
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


def print_task_footer(
    turns_used: int,
    files_changed: list[str] | None = None,
    verify_status: str | None = None,
    ctx_ratio: float | None = None,
) -> None:
    """Print a one-line turn summary footer at the end of an interactive task."""
    parts: list[str] = [f"{turns_used} turn{'s' if turns_used != 1 else ''}"]
    if files_changed:
        n = len(files_changed)
        names = ", ".join(safe_markup(f) for f in files_changed[:3])
        if n > 3:
            names += f", +{n - 3} more"
        parts.append(f"{n} file{'s' if n != 1 else ''} changed ({names})")
    if verify_status:
        v_col = "green" if verify_status == "passed" else "yellow"
        parts.append(f"[{v_col}]verify: {safe_markup(verify_status)}[/{v_col}]")
    if ctx_ratio is not None and ctx_ratio > 0:
        pct = min(ctx_ratio, 1.0) * 100
        color = "red" if ctx_ratio >= 0.85 else ("yellow" if ctx_ratio >= 0.65 else "dim")
        parts.append(f"[{color}]ctx {pct:.0f}%[/{color}]")
    body = "  ·  ".join(parts)
    console.print(f"[dim]  ── {body} ──[/dim]")


# ---------------------------------------------------------------------------
#  Status messages
# ---------------------------------------------------------------------------

def print_info(msg: str) -> None:
    console.print(_sanitized_text(msg, style="dim"))


def print_warning(msg: str, *, to_stderr: bool = False) -> None:
    """Print *msg* as a yellow warning, on stderr when *to_stderr* is set so a
    machine-readable stdout stays a lone document."""
    (err_console if to_stderr else console).print(_sanitized_text(msg, style="yellow"))


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
    console.print(f"[dim]Starting localm serve {model} on port {port}...[/dim]")


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
    max_lines: int = 30,
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
    global _last_diff_preview
    from_label = f"a/{path_label}" if path_label else "a/current"
    to_label   = f"b/{path_label}" if path_label else "b/proposed"

    diff_lines = list(difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=from_label,
        tofile=to_label,
    ))

    if not diff_lines:
        _last_diff_preview = None
        console.print("[dim]  (no changes)[/dim]")
        return

    truncated = len(diff_lines) > max_lines
    _last_diff_preview = (old_text, new_text, path_label, truncated)

    added = sum(1 for ln in diff_lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in diff_lines if ln.startswith("-") and not ln.startswith("---"))
    header = Text("\n  ")
    if not old_text:
        header.append("+ new file ", style="bold green")
        header.append(_strip_esc(path_label or "proposed"), style="bold")
        header.append(f" ({added} lines)", style="dim")
    else:
        header.append("Δ ", style="bold cyan")
        header.append(_strip_esc(path_label or "proposed"), style="bold")
        header.append(f" (+{added} -{removed} lines)", style="dim")

    diff_text = "".join(diff_lines[:max_lines])
    if truncated:
        rem = len(diff_lines) - max_lines
        diff_text += f"\n... ({rem} more lines truncated - press [f] at prompt for full diff)\n"

    console.print(header)
    console.print(Syntax(diff_text, "diff", theme="monokai", line_numbers=False))


def _parse_confirm_answer(raw: str) -> tuple[str, str]:
    """Parse raw confirmation input into (action, reason)."""
    s = raw.strip()
    lower = s.lower()
    if lower in ("y", "yes"):
        return "yes", ""
    if lower in ("a", "always", "all"):
        return "always", ""
    if lower in ("f", "full", "d", "diff"):
        return "full", ""
    if lower in ("n", "no", ""):
        return "no", ""
    for prefix in ("n:", "no:", "n ", "no "):
        if lower.startswith(prefix):
            return "no", s[len(prefix):].strip()
    # Any other text is a rejection that tells the agent what to do instead.
    return "no", s


def reset_diff_preview() -> None:
    """Forget the stored preview so a later prompt cannot offer [f]ull for it."""
    global _last_diff_preview
    _last_diff_preview = None


@contextlib.contextmanager
def offering_session_allow(offer: bool):
    """Within the block, approval prompts offer the "a" (allow this session)
    answer only when *offer* is true. Prompts outside any block never offer it."""
    token = _offer_session_allow.set(bool(offer))
    try:
        yield
    finally:
        _offer_session_allow.reset(token)


def _answer_hint(full: bool = False) -> str:
    parts = ["y/N"]
    if _offer_session_allow.get():
        parts.append("a=allow this session")
    if full:
        parts.append("f=full diff")
    return "[" + "/".join(parts) + "] "


def _record_answer(action: str, reason: str) -> bool:
    global _last_confirm_always, _last_reject_reason
    if action == "always":
        _last_confirm_always = _offer_session_allow.get()
        return True
    if action == "yes":
        return True
    _last_reject_reason = reason
    return False


def confirm_diff(path_label: str) -> bool:
    """Prompt to apply changes to *path_label*. [f] reprints the full diff, and
    is offered only when the stored preview is this file's and was cut short."""
    global _last_confirm_always, _last_reject_reason
    _last_confirm_always = False
    _last_reject_reason = ""
    try:
        while True:
            preview = _last_diff_preview
            has_full = bool(preview and preview[3] and preview[2] == path_label)
            prompt = Text("Apply changes to ", style="yellow")
            prompt.append(_strip_esc(path_label), style="bold yellow")
            prompt.append(f"? {_answer_hint(full=has_full)}", style="yellow")
            raw = console.input(prompt)
            action, reason = _parse_confirm_answer(raw)
            if action == "full":
                if has_full and preview is not None:
                    old_t, new_t, lbl, _ = preview
                    print_diff_preview(old_t, new_t, path_label=lbl, max_lines=100_000)
                continue
            return _record_answer(action, reason)
    except (KeyboardInterrupt, EOFError):
        return False


# ---------------------------------------------------------------------------
#  Confirmation prompt (for destructive tools)
# ---------------------------------------------------------------------------

def confirm(prompt: str) -> bool:
    global _last_confirm_always, _last_reject_reason
    _last_confirm_always = False
    _last_reject_reason = ""
    try:
        line = _sanitized_text(f"{prompt} {_answer_hint()}", style="yellow")
        raw = console.input(line)
        action, reason = _parse_confirm_answer(raw)
        return _record_answer(action, reason)
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
  [bold]/model \\[name][/bold]         show or switch model
  [bold]/mode[/bold]                 show session persistence mode (privacy/log/full)
  [bold]/cwd[/bold]                  show working directory
  [bold]/cd <path>[/bold]            change working directory
  [bold]/reindex[/bold]              rebuild the codebase index
  [bold]/verbose[/bold]              toggle verbose tool output
  [bold]/approve[/bold]              toggle auto-approve for destructive tools
  [bold]/history[/bold]              show turn count, context usage, and index size
  [bold]/changes[/bold]              list every file this session has changed
  [bold]/diff \\[path][/bold]          cumulative diff of session changes (all or one file)
  [bold]/undo[/bold]                 revert the last file write or edit
  [bold]/sessions[/bold]             list this project's saved (interrupted) sessions
  [bold]/resume \\[id][/bold]          resume the most recent interrupted session, or a
                        specific one by id from /sessions
  [bold]/compact[/bold]              summarise old turns to free context space
  [bold]/memory[/bold]               show current project memory
  [bold]/remember <text>[/bold]      append a note to LOCALCODER.md
  [bold]/forget <pattern>[/bold]     remove memory bullets matching pattern
  [bold]/save[/bold]                 save conversation to JSON
  [bold]/export \\[path][/bold]        export session transcript to Markdown
  [bold]/scope \\[glob][/bold]         show or set the file-access scope (e.g. src/**/*.py)
  [bold]/scope clear[/bold]          remove the active scope restriction
  [bold]/verify \\[cmd|auto|off][/bold] command that must exit 0 before a turn that
                        changed files may finish (run by the harness, not the model)
  [bold]/goal \\[cmd|auto|off][/bold] iterate the next task until this command exits 0,
                        re-running the whole task with each failure fed back
  [bold]/review[/bold]               a reviewer model checks the current diff for blocking
                        issues right now, even with coder_review off
  [bold]/doctor[/bold]               check the model server, tool calls, context window,
                        git, verify command and MCP servers
  [bold]/hooks \\[trust|forget][/bold]  show this project's tool hooks, or trust / untrust them

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
