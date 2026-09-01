# SPDX-License-Identifier: AGPL-3.0-or-later
"""The interactive REPL: multiline input, readline/tab-completion setup, the read
loop, and the /command handler."""

from __future__ import annotations

import os
import time
from pathlib import Path

from ..agent import Agent
from ..audit import SessionMode
from ..backends.http import CoderAuthError
from ..display import (
    confirm,
    console,
    print_error,
    print_help,
    print_info,
    print_success,
    print_warning,
    safe_markup,
)
from ..review_guard import classify_sensitive_changes, render_warning
from .goal import _run_goal_loop

def _read_multiline() -> str:
    """
    Read one user message, supporting backslash line continuation.

    End a line with \\ to keep typing on the next line.  The backslash is
    stripped and the lines are joined with a newline before being sent.
    """
    lines: list[str] = []
    first = True
    while True:
        prompt = "\n[bold green]You[/bold green]: " if first else "[dim]...[/dim] "
        try:
            line = console.input(prompt)
        except (KeyboardInterrupt, EOFError):
            raise
        if line.endswith("\\"):
            lines.append(line[:-1])
            first = False
        else:
            lines.append(line)
            break
    return "\n".join(lines)


# Every REPL slash command, for tab completion and /help
_SLASH_COMMANDS = (
    "/help", "/exit", "/quit", "/clear", "/model", "/mode", "/cwd", "/cd",
    "/reindex", "/verbose", "/approve", "/history", "/undo", "/resume",
    "/sessions", "/compact", "/memory", "/remember", "/forget", "/save",
    "/export", "/scope", "/changes", "/diff", "/bg", "/verify", "/goal",
    "/review",
)


def _setup_readline(agent: Agent) -> None:
    """
    Tab completion (slash commands + project paths) and persistent REPL
    history.

    Best-effort: stock Windows CPython has no readline (pyreadline3 provides
    it when installed), so every step degrades silently. History persists to
    .localcoder/repl_history only outside privacy mode - privacy promises no
    traces, and suppress_readline_history() has already disabled saving there.
    """
    try:
        import readline
    except ImportError:
        return

    def complete(text: str, state: int):
        buf = readline.get_line_buffer()
        if buf.startswith("/") and " " not in buf:
            options = [c + " " for c in _SLASH_COMMANDS if c.startswith(buf)]
        else:
            # Complete the current token as a path relative to the project
            import glob as _g
            try:
                matches = _g.glob(str(agent.cwd / (text + "*")))
            except Exception:
                matches = []
            options = []
            for m in matches[:50]:
                rel = os.path.relpath(m, agent.cwd)
                options.append(rel + os.sep if os.path.isdir(m) else rel)
        try:
            return options[state]
        except IndexError:
            return None

    try:
        readline.set_completer(complete)
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("tab: complete")
    except Exception:
        return

    if agent.mode != SessionMode.PRIVACY:
        hist = agent.cwd / ".localcoder" / "repl_history"
        try:
            hist.parent.mkdir(parents=True, exist_ok=True)
            if hist.is_file():
                readline.read_history_file(str(hist))
            readline.set_history_length(500)
            import atexit

            def _save_history(path=str(hist)):
                try:
                    readline.write_history_file(path)
                except Exception:
                    pass

            atexit.register(_save_history)
        except Exception:
            pass


def _warn_goal_run_sensitive_changes(agent: Agent, before: dict) -> None:
    """Print review_guard's warning for the files this goal run touched:
    entries in agent.changed_files() whose write count exceeds *before*'s.
    Best-effort: never let this advisory break the REPL.

    See test_goal_run_warning_is_scoped_to_that_run_not_the_whole_session."""
    try:
        touched = [f for f in agent.changed_files()
                  if f["writes"] > before.get(f["path"], 0)]
        message = render_warning(classify_sensitive_changes(touched))
        if message:
            print_warning(message)
    except Exception:                                       # noqa: BLE001
        pass


def _repl(agent: Agent) -> None:
    _setup_readline(agent)
    while True:
        try:
            user_input = _read_multiline().strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", "bye"):
            console.print("[dim]Bye.[/dim]")
            break

        if user_input.startswith("/"):
            stop = _handle_command(user_input, agent)
            if stop:
                break
            continue

        try:
            # Starting a fresh task - discard any stale checkpoint
            agent.clear_checkpoint()
            if agent.goal_cmd is not None:
                before = {f["path"]: f["writes"] for f in agent.changed_files()}
                try:
                    _run_goal_loop(agent, user_input, agent.goal_cmd,
                                   agent.goal_max_iters, agent.cwd)
                finally:
                    _warn_goal_run_sensitive_changes(agent, before)
            else:
                agent.chat(user_input)
        except KeyboardInterrupt:
            # Checkpoint was already saved inside _loop; just swallow here
            pass
        except CoderAuthError:
            # Must raise here to bypass generic Exception block and bubble up to main
            raise
        except Exception as e:
            print_error(f"Agent error: {e}")
            import traceback
            if agent.verbose:
                traceback.print_exc()


def _handle_command(raw: str, agent: Agent) -> bool:
    """Handle a /command. Returns True if the REPL should exit."""
    parts = raw[1:].split(" ", 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit", "q", "bye"):
        console.print("[dim]Bye.[/dim]")
        return True

    elif cmd == "help":
        print_help()

    elif cmd == "clear":
        agent.reset()
        print_info("Conversation cleared.")

    elif cmd == "model":
        print_info(f"Model: {agent.backend.model_id}")

    elif cmd == "cwd":
        print_info(f"Working directory: {agent.cwd}")

    elif cmd == "cd":
        if not arg:
            print_info("Usage: /cd <path>")
        else:
            new_dir = (agent.cwd / arg).resolve()
            if not new_dir.is_dir():
                print_warning(f"Not a directory: {new_dir}")
            else:
                # A project that marked itself private must not receive this
                # session's transcript, and the mode cannot be lowered to match
                # once the audit log is open - see the helper for the full story.
                # Shared with the web cwd route so the two surfaces agree.
                from ..privacy import refuse_move_into_stricter_project
                refusal = refuse_move_into_stricter_project(
                    agent.mode.value, new_dir)
                if refusal:
                    print_warning(refusal)
                else:
                    agent.set_cwd(new_dir)
                    print_info(f"Working directory: {new_dir}")

    elif cmd == "verbose":
        agent.verbose = not agent.verbose
        print_info(f"Verbose: {'on' if agent.verbose else 'off'}")

    elif cmd == "approve":
        agent.auto_approve = not agent.auto_approve
        print_info(f"Auto-approve: {'on' if agent.auto_approve else 'off'}")

    elif cmd == "reindex":
        n = agent.reindex()
        print_info(f"Project map rebuilt - {n} files indexed.")

    elif cmd == "history":
        chars = agent.context_chars()
        ctx_tokens_est = chars // 4
        billed = agent.total_tokens
        billed_str = f"  ·  billed: ~{billed:,} tokens" if billed else ""
        console.print(
            f"[dim]Turns: {agent.turns}  ·  "
            f"Context: ~{ctx_tokens_est:,} tokens ({chars:,} chars){billed_str}  ·  "
            f"Map: {agent._project_map.file_count()} files[/dim]"
        )

    elif cmd == "sessions":
        from ..agent.checkpoint import list_checkpoints
        entries = list_checkpoints(agent.cwd)
        if not entries:
            print_info("No saved sessions for this project.")
        else:
            console.print(
                f"[dim]{len(entries)} saved session(s) for this project "
                "(newest first) - /resume <id> to continue one:[/dim]")
            for e in entries:
                changed = (f", {e['changed_files']} file(s) changed"
                          if e["changed_files"] else "")
                when = e["interrupted_at"] or "unknown time"
                console.print(
                    f"  [bold]{safe_markup(e['id'])}[/bold]  "
                    f"{safe_markup(e['title'])}  "
                    f"[dim]({e['turns']} turns{changed}, {safe_markup(when)})[/dim]")

    elif cmd == "resume":
        # No id -> the most recent (the zero-argument default); an id from
        # /sessions resumes that SPECIFIC one, so several interrupted sessions
        # can coexist in one project.
        ckpt = agent.load_checkpoint(arg or None)
        if ckpt is None:
            if arg:
                print_info(f"No saved session with id '{arg}'. Try /sessions "
                          "to list what is available.")
            else:
                print_info("No interrupted session found.")
        else:
            agent.resume_checkpoint(ckpt)
            agent.clear_checkpoint()
            ts    = ckpt.get("interrupted_at", "unknown time")
            turns = ckpt.get("turns", "?")
            title = ckpt.get("title")
            label = f' "{title}"' if title else ""
            print_success(f"Resumed session{label} ({turns} turns, interrupted {ts}).")
            try:
                agent.chat("Continue from where we left off.")
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print_error(f"Agent error: {e}")

    elif cmd == "undo":
        msg = agent.undo()
        if msg is None:
            print_info("Nothing to undo.")
        else:
            print_success(msg)
            remaining = len(agent.undo_list())
            if remaining:
                print_info(f"({remaining} more step(s) on the undo stack)")

    else:
        return _handle_command_extended(cmd, arg, agent)

    return False


def _handle_command_extended(cmd: str, arg: str, agent: Agent) -> bool:
    """The second half of the /command dispatch: the changed-files / diff /
    compact / memory / save / export / scope commands and the unknown-command
    fallback. Always returns False (none of these exit the REPL), keeping a
    uniform signature with _handle_command."""
    if cmd == "changes":
        files = agent.changed_files()
        # Delegated work lives in another tree and is absent from
        # changed_files(); the footer is where it shows up ("" if none).
        from ..delegated import footer_for
        delegated = footer_for(agent)
        if not files:
            print_info("No files changed this session."
                       if not delegated else "No files changed in this tree.")
        else:
            for f in files:
                badge = "new" if f["created"] else "edited"
                gone = "" if f["exists"] else "  [red](deleted since)[/red]"
                console.print(
                    f"  [cyan]{safe_markup(f['path'])}[/cyan]  [dim]{badge}, "
                    f"{f['writes']} write(s) via {safe_markup(f['last_tool'])}"
                    f"[/dim]{gone}"
                )
            print_info("Use /diff [path] for the cumulative changes.")
        if delegated:
            console.print(safe_markup(delegated))

    elif cmd == "diff":
        diff = agent.session_diff(arg or None)
        # Rendered separately from the diff document below, never appended into
        # it: Syntax(..., "diff") presents ONE applicable patch, and foreign hunks
        # inlined there would read as directly applicable when they are not.
        from ..delegated import footer_for
        delegated = footer_for(agent) if not arg else ""
        if not diff:
            print_info("No changes to show."
                       + (f" ('{arg}' was not changed this session)" if arg else ""))
        else:
            from rich.syntax import Syntax
            console.print(Syntax(diff, "diff", theme="monokai", line_numbers=False))
        if delegated:
            console.print(safe_markup(delegated))

    elif cmd == "bg":
        # "/bg", not "/jobs": an installed jobs plugin owns that noun for
        # SCHEDULED recurring jobs (localm job ..., the GUI Jobs tab,
        # /api/jobs), which are a disjoint list from this session's background
        # work.
        from ..background import get_registry
        registry = get_registry()
        rows = registry.list_status()
        if not rows:
            print_info("No background work this session. Start some with "
                       "run_shell_background or spawn_agent_background.")
        else:
            running = [r for r in rows if r["state"] == "running"]
            done    = [r for r in rows if r["state"] != "running"]
            if running:
                console.print("[bold]Running[/bold]")
                for r in running:
                    age = time.time() - r["started_at"]
                    console.print(
                        f"  [cyan]{safe_markup(r['id'])}[/cyan]  "
                        f"{safe_markup(r['kind']):<6} "
                        f"{safe_markup(r['label'])}  [dim]{age:.0f}s[/dim]")
            if done:
                console.print("[bold]Finished[/bold]")
                for r in done:
                    res = r.get("result") or {}
                    extra = ""
                    if r["kind"] == "shell" and "exit_code" in res:
                        extra = f"exit {res['exit_code']}"
                    elif r["kind"] == "agent":
                        extra = f"{res.get('turns', 0)} turn(s)"
                        if res.get("branch"):
                            extra += f", branch {res['branch']}"
                    flag = "[red]" if r["state"] in ("failed", "killed") else "[dim]"
                    console.print(
                        f"  [cyan]{safe_markup(r['id'])}[/cyan]  "
                        f"{safe_markup(r['kind']):<6} "
                        f"{safe_markup(r['label'])}  {flag}"
                        f"{safe_markup(r['state'])}"
                        f"{extra and ' - ' + safe_markup(extra)}[/]")
        # The table is bounded, so a long session outgrows it. What fell off is
        # reported per KIND.
        dropped = dict(registry.dropped_undrained_by_kind)
        lost_agents = dropped.pop("agent", 0)
        if lost_agents:
            # Real loss: absorption is drain-only, so an evicted sub-agent
            # completion takes its summary, branch and diff with it.
            print_warning(
                f"{lost_agents} background sub-agent result(s) were discarded "
                "before they could be collected, and are lost.")
        other = sum(dropped.values())
        if other:
            # Not a silent loss: these are polled by id, and asking for an aged-out
            # one answers "No background job with id ...", listing the ids that do
            # exist. So this is housekeeping, reported as such.
            print_info(
                f"{other} older finished job(s) have aged out of the list "
                "(it is capped per kind). Check a job before it ages out.")

    elif cmd == "compact":
        ratio = agent._fill_ratio()
        did_compact = agent.compact()
        if did_compact:
            new_ratio = agent._fill_ratio()
            print_success(
                f"Compacted history. Context: {ratio:.0%} → {new_ratio:.0%} full."
            )
        else:
            print_info("Nothing to compact (too few turns).")

    elif cmd == "remember":
        if not arg:
            print_info("Usage: /remember <text>")
        else:
            try:
                p = agent.remember(arg)
                print_success(f"Remembered → {p}")
            except Exception as e:
                print_error(f"remember failed: {e}")

    elif cmd == "forget":
        if not arg:
            print_info("Usage: /forget <pattern>")
        else:
            p, n = agent.forget(arg)
            if p is None:
                print_info("No memory file found.")
            elif n == 0:
                print_info(f"No entries matching '{arg}'.")
            else:
                print_success(f"Removed {n} entr{'y' if n == 1 else 'ies'} from {p}.")

    elif cmd == "memory":
        mem = agent._memory
        if mem:
            console.print(f"[dim]{safe_markup(mem)}[/dim]")
        else:
            print_info("No memory file. Use /remember <text> to create one.")

    elif cmd == "mode":
        if not arg:
            # Show current mode
            m = agent.mode.value
            notes = {
                "privacy": (
                    "nothing saved; readline + shell history scrubbed on exit. "
                    "cmd.exe: no history anyway. "
                    "Online providers are explicit opt-in - warning shown if active."
                ),
                "log":     "JSONL audit trail → <data dir>/sessions/",
                "full":    "JSONL audit trail + markdown transcript on exit",
            }
            print_info(f"Session mode: {m}  - {notes.get(m, '')}")
        else:
            print_info(
                "Session mode cannot be changed mid-session "
                "(audit log is already open or closed). "
                "Start a new session with --mode <privacy|log|full>."
            )

    elif cmd == "save":
        filepath = Path(arg) if arg else Path("conversation.json")
        if agent.mode == SessionMode.PRIVACY:
            console.print(
                "[yellow]⚠  Privacy mode is active. "
                "This will write the conversation to disk.[/yellow]"
            )
            if not confirm(f"  Save to {filepath}?"):
                print_info("Cancelled.")
                return False
        try:
            agent.save_history(filepath)
            print_success(f"Saved to {filepath}")
        except Exception as e:
            print_error(f"Save failed: {e}")

    elif cmd == "export":
        import time as _time

        if agent.mode == SessionMode.PRIVACY and not arg:
            console.print(
                "[yellow]⚠  Privacy mode is active. "
                "This will write the session transcript to disk.[/yellow]"
            )
            if not confirm("  Export session?"):
                print_info("Cancelled.")
                return False

        if arg:
            out_path = Path(arg)
        else:
            ts_label = _time.strftime("%Y-%m-%d_%H%M%S")
            out_path = agent.cwd / ".localcoder" / "sessions" / f"{ts_label}.md"

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Force a write via the agent's session-markdown writer
            agent._audit.close()   # flush any pending audit writes
            md_path = agent._write_session_markdown()
            if arg:
                # Move to the requested path
                import shutil as _shutil
                _shutil.move(str(md_path), str(out_path))
                md_path = out_path
            print_success(f"Session exported → {md_path}")
        except Exception as e:
            print_error(f"Export failed: {e}")

    elif cmd == "verify":
        from ..verify import command_text, detect_verify_command
        if not arg:
            current = (command_text(agent.verify_cmd)
                       if agent.verify_cmd is not None else "(off)")
            print_info(
                f"Verification: {current}\n"
                "  /verify <command>  check this instead\n"
                "  /verify auto       re-detect the project's check\n"
                "  /verify off        no exit-code check")
        elif arg == "off":
            agent.verify_cmd = None
            print_info("Verification off - turns finish without an exit-code "
                       "check. The model's own 'done' is the only gate again.")
        elif arg == "auto":
            detected = detect_verify_command(agent.cwd)
            agent.verify_cmd = detected
            if detected is None:
                print_warning(
                    "No obvious check found in this project (looked for a "
                    "verify key in .localcoder/config.toml, Cargo.toml, go.mod, "
                    "a package.json test script, and a pytest setup). "
                    "Set one with /verify <command>.")
            else:
                print_success(f"Verification: `{command_text(detected)}`")
        else:
            agent.verify_cmd = arg
            print_success(f"Verification: `{arg}` must exit 0 before a turn "
                          "that changed files finishes.")

    elif cmd == "goal":
        from ..verify import command_text, detect_verify_command
        if not arg:
            current = (command_text(agent.goal_cmd)
                       if agent.goal_cmd is not None else "(off)")
            print_info(
                f"Goal mode: {current}\n"
                "  /goal <command>  iterate the next task until this exits 0\n"
                "  /goal auto       re-detect the project's check\n"
                "  /goal off        plain-text messages get one turn again\n"
                "Unlike /verify, a failed check re-runs the WHOLE task with "
                f"the failure fed back, up to {agent.goal_max_iters} "
                "iteration(s), instead of ending the turn.")
        elif arg == "off":
            agent.goal_cmd = None
            print_info("Goal mode off - plain-text messages get a single "
                       "chat turn again.")
        elif arg == "auto":
            detected = detect_verify_command(agent.cwd)
            agent.goal_cmd = detected
            if detected is None:
                print_warning(
                    "No obvious check found in this project (looked for a "
                    "verify key in .localcoder/config.toml, Cargo.toml, go.mod, "
                    "a package.json test script, and a pytest setup). "
                    "Set one with /goal <command>.")
            else:
                print_success(
                    f"Goal mode: `{command_text(detected)}` must exit 0, up "
                    f"to {agent.goal_max_iters} iteration(s) per task.")
        else:
            agent.goal_cmd = arg
            print_success(
                f"Goal mode: `{arg}` must exit 0, up to "
                f"{agent.goal_max_iters} iteration(s) per task, with each "
                "failure's output fed back for another attempt.")

    elif cmd == "review":
        diff = agent.session_diff()
        if not diff.strip():
            print_info("No changes to review this session.")
        else:
            reviewer = agent._reviewer
            if reviewer is None:
                # No reviewer configured (coder_review is off) - build one now.
                from ..reviewer import reviewer_for_agent
                reviewer = reviewer_for_agent(
                    agent.backend, agent.mode, agent.restricted, force=True)
            if reviewer is None:
                print_warning(
                    "No reviewer available for this session "
                    "(restricted sessions run no reviewer).")
            else:
                result = reviewer.review(diff, agent._review_task)
                warning = reviewer.failure_warning(result)
                model = ("separate reviewer model" if reviewer.heterogeneous
                          else "review pass")
                if warning:
                    print_warning(warning)
                elif result.approved:
                    print_success(f"Approved - the {model} found no blocking issues.")
                    if result.notes:
                        console.print(f"[dim]{safe_markup(result.notes)}[/dim]")
                else:
                    n = len(result.blocking)
                    print_warning(
                        f"The {model} flagged {n} blocking issue"
                        f"{'s' if n != 1 else ''}:")
                    for b in result.blocking:
                        console.print(f"  - {safe_markup(b)}")
                    if result.notes:
                        console.print(f"[dim]{safe_markup(result.notes)}[/dim]")

    elif cmd == "scope":
        if not arg:
            current = agent.scope or "(none)"
            print_info(f"Scope: {current}")
        else:
            agent.scope = arg if arg != "clear" else None
            if agent.scope:
                print_success(f"Scope set to '{agent.scope}'")
            else:
                print_info("Scope cleared - all files accessible.")

    else:
        print_info(f"Unknown command: /{cmd}  (try /help)")

    return False
