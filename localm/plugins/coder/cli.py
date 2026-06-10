"""
localcoder CLI entry point.

Commands
--------
  localcoder                     Interactive chat (auto-starts localm serve)
  localcoder "task"              Single-shot task (non-interactive)
  localcoder --model X           Specify model
  localcoder --url URL           Point at any OpenAI-compatible server
  localcoder --no-server         Don't start localm serve (connect to existing)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from .backends.http import HTTPBackend, make_localm_backend, make_openai_backend
from .agent import Agent
from .audit import SessionMode, parse_mode
from .privacy import (
    clear_shell_history_traces,
    suppress_readline_history,
    warn_external_provider,
)
from .project_config import load_project_config
from .display import (
    confirm,
    console,
    print_banner,
    print_error,
    print_help,
    print_info,
    print_success,
    print_warning,
)
from .server import ManagedServer, find_free_port

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
#  CLI definition
# ---------------------------------------------------------------------------

def _complete_model(ctx, param, incomplete):
    """Shell completion callback: suggest registered localm model names."""
    try:
        from localm.model_manager import load_registry
        return [
            click.shell_completion.CompletionItem(name)
            for name in load_registry()
            if name.startswith(incomplete)
        ]
    except Exception:
        return []


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("task", default="", required=False, metavar="[TASK]")
@click.option("-m", "--model",      default=None,  envvar="LOCALCODER_MODEL",
              shell_complete=_complete_model,
              help="Model name (must be registered in localm).")
@click.option("-u", "--url",        default=None,  envvar="LOCALCODER_URL",
              help="OpenAI-compat base URL, e.g. http://127.0.0.1:8080/v1.")
@click.option("-k", "--api-key",    default="localm",
              help="API key (use any string for local servers).")
@click.option("-p", "--port",       default=None,  type=int,
              help="Port for the auto-started localm serve [default: first free in 8080–8199].")
@click.option("-c", "--cwd",        default=None,  type=click.Path(exists=True, file_okay=False),
              help="Working directory [default: current directory].")
@click.option("--no-server",        is_flag=True,
              help="Don't start localm serve — assume server is already running.")
@click.option("--max-turns",        default=None,  type=int,
              help="Max agent iterations per task [default: 40, or from .localcoder/config.toml].")
@click.option("--temperature",      default=None,  type=float,
              help="Sampling temperature.")
@click.option("--max-tokens",       default=None,  type=int,
              help="Max tokens per LLM response [default: 2048, or from .localcoder/config.toml].")
@click.option("--verbose",          is_flag=True,
              help="Print full tool outputs.")
@click.option("--yes", "-y",        is_flag=True,
              help="Auto-approve all destructive tool calls.")
@click.option("--interactive-confirm", "interactive_confirm", is_flag=True,
              help="Auto-approve file writes but still prompt before shell commands.")
@click.option("--dry-run",          is_flag=True,
              help="Show what the agent would do without executing destructive tools.")
@click.option("--patch-mode",       "patch_mode", default=None,
              metavar="FILE",
              help=(
                  "Capture all file writes as a unified diff instead of modifying files. "
                  "Writes the .patch to FILE (use '-' for stdout)."
              ))
@click.option("--native-tools",     "native_tools", is_flag=True,
              help=(
                  "Use the OpenAI-compatible native tools API for structured tool calls "
                  "(enabled automatically for --online and --anthropic; "
                  "use this flag with --url for servers that support it, e.g. Ollama)."
              ))
@click.option("--online", "provider", flag_value="openai",  default=None,
              help="Use OpenAI API instead of local model.")
@click.option("--anthropic",        "provider", flag_value="anthropic",
              help="Use Anthropic API (via ANTHROPIC_API_KEY).")
@click.option("--ci",               is_flag=True,
              help=(
                  "CI mode: auto-approve all, no colors, structured exit codes. "
                  "Exit 0 = success, 1 = task incomplete (max turns), 2 = startup error."
              ))
@click.option("--output-format",    "output_format", default="text",
              type=click.Choice(["text", "json"], case_sensitive=False),
              help="Output format for non-interactive runs (text or json).")
@click.option("--mode",             default=None,
              type=click.Choice(["privacy", "log", "full"], case_sensitive=False),
              help=(
                  "Session persistence mode [default: privacy]. "
                  "privacy = nothing saved automatically; "
                  "log = JSONL audit trail to ~/.localm/sessions/; "
                  "full = log + markdown transcript in .localcoder/sessions/."
              ))
def main(
    task, model, url, api_key, port, cwd,
    no_server, max_turns, temperature, max_tokens,
    verbose, yes, interactive_confirm, dry_run, patch_mode, ci, output_format,
    native_tools, provider, mode,
):
    """
    Offline AI coding agent powered by local LLMs.

    \b
    Quick start:
      localcoder --model gemma4-4b
      localcoder --model gemma4-4b "add type hints to utils.py"

    \b
    Point at any running OpenAI-compat server:
      localcoder --url http://localhost:11434/v1 --model llama3.2 --no-server

    \b
    Use OpenAI or Anthropic (requires API keys):
      localcoder --online --model gpt-4o "review this code"
      localcoder --anthropic --model claude-opus-4-5 "add tests"
    """
    work_dir = Path(cwd).resolve() if cwd else Path.cwd()

    # ------------------------------------------------------------------ #
    #  CI mode setup
    # ------------------------------------------------------------------ #
    if ci:
        yes = True          # never prompt
        if mode is None:
            mode = "log"    # always leave an audit trail in CI
        # Strip Rich colors so log output is plain text
        import os as _os
        _os.environ.setdefault("NO_COLOR", "1")
        _os.environ.setdefault("TERM", "dumb")

    # ------------------------------------------------------------------ #
    #  Project-level config (.localcoder/config.toml) — CLI flags override
    # ------------------------------------------------------------------ #
    proj_cfg = load_project_config(work_dir)
    if model is None:
        model = proj_cfg.get("model")
    if max_turns is None:
        max_turns = int(proj_cfg.get("max_turns", 40))
    if max_tokens is None:
        max_tokens = int(proj_cfg.get("max_tokens", 2048))
    if temperature is None and "temperature" in proj_cfg:
        temperature = float(proj_cfg["temperature"])
    # auto_approve: config applies only when --yes flag was NOT passed
    if not yes and proj_cfg.get("auto_approve"):
        yes = True
    # always_confirm: tools that prompt even under --yes
    # --interactive-confirm sets {"run_shell"}; config can extend the list
    always_confirm: set[str] = set()
    if interactive_confirm:
        always_confirm.add("run_shell")
    cfg_confirm = proj_cfg.get("always_confirm", [])
    if isinstance(cfg_confirm, list):
        always_confirm.update(cfg_confirm)
    # mode: CLI > config > default (privacy)
    if mode is None:
        mode = proj_cfg.get("mode", "privacy")
    try:
        session_mode = parse_mode(mode)
    except ValueError as exc:
        print_error(str(exc))
        sys.exit(2 if ci else 1)

    # Privacy-mode setup — suppress readline history as early as possible
    if session_mode == SessionMode.PRIVACY:
        suppress_readline_history()
    # Warn when privacy mode is requested but prompts leave the machine
    if session_mode == SessionMode.PRIVACY and provider in ("openai", "anthropic"):
        warn_external_provider(provider)

    gen_kw   = {k: v for k, v in [
        ("temperature", temperature),
        ("max_tokens",  max_tokens),
    ] if v is not None}

    # ------------------------------------------------------------------ #
    #  Build backend
    # ------------------------------------------------------------------ #
    server_ctx: Optional[ManagedServer] = None

    if provider == "openai":
        if not model:
            model = "gpt-4o"
        backend = make_openai_backend(model=model)

    elif provider == "anthropic":
        if not model:
            model = "claude-opus-4-5"
        import os
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        backend = HTTPBackend(
            "https://api.anthropic.com/v1", model, api_key=key, native_tools=True
        )

    elif url:
        # Explicit URL — no server management
        if not model:
            print_error("--url requires --model")
            sys.exit(2 if ci else 1)
        backend = HTTPBackend(url.rstrip("/"), model, api_key=api_key,
                              native_tools=native_tools)

    else:
        # Offline path — localm backend
        if not model:
            print_error(
                "Specify a model with --model, e.g.:\n"
                "  localcoder --model gemma4-4b\n"
                "Run `localm list` to see registered models."
            )
            sys.exit(2 if ci else 1)

        srv_port = port or find_free_port()

        if no_server:
            backend = make_localm_backend(model, port=srv_port)
        else:
            server_ctx = ManagedServer(model, port=srv_port)
            if not server_ctx.start():
                sys.exit(2 if ci else 1)
            backend = make_localm_backend(model, port=srv_port)

    # ------------------------------------------------------------------ #
    #  Create agent
    # ------------------------------------------------------------------ #
    agent = Agent(
        backend=backend,
        cwd=work_dir,
        name="localcoder",
        max_turns=max_turns,
        verbose=verbose,
        auto_approve=yes or (task != ""),
        always_confirm=always_confirm,
        dry_run=dry_run,
        mode=session_mode,
        **gen_kw,
    )

    if patch_mode:
        agent.patch_mode = True

    try:
        if task:
            # Non-interactive single-task mode
            response = agent.run_task(task)
            success  = agent.last_run_ok

            if output_format == "json":
                import json as _json
                result = {
                    "success":      success,
                    "response":     response,
                    "turns":        agent.turns,
                    "total_tokens": agent.total_tokens,
                }
                sys.stdout.write(_json.dumps(result, indent=2) + "\n")

            if ci and not success:
                sys.exit(1)
        else:
            # Interactive REPL
            print_banner(backend.model_id, work_dir,
                         file_count=agent._project_map.file_count(),
                         session_mode=session_mode)
            # Notify if an interrupted session is waiting
            ckpt = agent.load_checkpoint()
            if ckpt:
                ts = ckpt.get("interrupted_at", "unknown time")
                turns = ckpt.get("turns", "?")
                print_warning(
                    f"Interrupted session found ({turns} turns, {ts}). "
                    "Type /resume to continue."
                )
            _repl(agent)

    finally:
        # Flush patch output before closing
        if patch_mode and agent._patch_chunks:
            patch_content = agent.flush_patch()
            if patch_mode == "-":
                sys.stdout.write(patch_content)
            else:
                out = Path(patch_mode)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(patch_content, encoding="utf-8")
                print_info(f"Patch written to {out}")

        md_path = agent.close()
        if md_path:
            print_info(f"Session transcript saved → {md_path}")
        if session_mode == SessionMode.PRIVACY:
            clear_shell_history_traces()
        if server_ctx:
            server_ctx.stop()


# ---------------------------------------------------------------------------
#  Interactive REPL
# ---------------------------------------------------------------------------

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


def _repl(agent: Agent) -> None:
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
            # Starting a fresh task — discard any stale checkpoint
            agent.clear_checkpoint()
            agent.chat(user_input)
        except KeyboardInterrupt:
            # Checkpoint was already saved inside _loop; just swallow here
            pass
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
            print_info(f"Usage: /cd <path>")
        else:
            new_dir = (agent.cwd / arg).resolve()
            if new_dir.is_dir():
                agent.set_cwd(new_dir)
                print_info(f"Working directory: {new_dir}")
            else:
                print_warning(f"Not a directory: {new_dir}")

    elif cmd == "verbose":
        agent.verbose = not agent.verbose
        print_info(f"Verbose: {'on' if agent.verbose else 'off'}")

    elif cmd == "approve":
        agent.auto_approve = not agent.auto_approve
        print_info(f"Auto-approve: {'on' if agent.auto_approve else 'off'}")

    elif cmd == "reindex":
        n = agent.reindex()
        print_info(f"Project map rebuilt — {n} files indexed.")

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

    elif cmd == "resume":
        ckpt = agent.load_checkpoint()
        if ckpt is None:
            print_info("No interrupted session found.")
        else:
            agent.resume_checkpoint(ckpt)
            agent.clear_checkpoint()
            ts    = ckpt.get("interrupted_at", "unknown time")
            turns = ckpt.get("turns", "?")
            print_success(f"Resumed session ({turns} turns, interrupted {ts}).")
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
            console.print(f"[dim]{mem}[/dim]")
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
                    "Online providers are explicit opt-in — warning shown if active."
                ),
                "log":     "JSONL audit trail → ~/.localm/sessions/",
                "full":    "JSONL audit trail + markdown transcript on exit",
            }
            print_info(f"Session mode: {m}  — {notes.get(m, '')}")
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

    else:
        print_info(f"Unknown command: /{cmd}  (try /help)")

    return False
