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

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("task", default="", required=False, metavar="[TASK]")
@click.option("-m", "--model",      default=None,  envvar="LOCALCODER_MODEL",
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
@click.option("--max-turns",        default=40,    type=int,
              help="Max agent iterations per task [default: 40].")
@click.option("--temperature",      default=None,  type=float,
              help="Sampling temperature.")
@click.option("--max-tokens",       default=2048,  type=int,
              help="Max tokens per LLM response [default: 2048].")
@click.option("--verbose",          is_flag=True,
              help="Print full tool outputs.")
@click.option("--yes", "-y",        is_flag=True,
              help="Auto-approve all destructive tool calls.")
@click.option("--online", "provider", flag_value="openai",  default=None,
              help="Use OpenAI API instead of local model.")
@click.option("--anthropic",        "provider", flag_value="anthropic",
              help="Use Anthropic API (via ANTHROPIC_API_KEY).")
def main(
    task, model, url, api_key, port, cwd,
    no_server, max_turns, temperature, max_tokens,
    verbose, yes, provider,
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
        backend = HTTPBackend("https://api.anthropic.com/v1", model, api_key=key)

    elif url:
        # Explicit URL — no server management
        if not model:
            print_error("--url requires --model")
            sys.exit(1)
        backend = HTTPBackend(url.rstrip("/"), model, api_key=api_key)

    else:
        # Offline path — localm backend
        if not model:
            print_error(
                "Specify a model with --model, e.g.:\n"
                "  localcoder --model gemma4-4b\n"
                "Run `localm list` to see registered models."
            )
            sys.exit(1)

        srv_port = port or find_free_port()

        if no_server:
            backend = make_localm_backend(model, port=srv_port)
        else:
            server_ctx = ManagedServer(model, port=srv_port)
            if not server_ctx.start():
                sys.exit(1)
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
        **gen_kw,
    )

    try:
        if task:
            # Non-interactive single-task mode
            response = agent.run_task(task)
            if not verbose:
                # In non-interactive mode the loop already printed the answer
                pass
        else:
            # Interactive REPL
            print_banner(backend.model_id, work_dir, file_count=agent._project_map.file_count())
            _repl(agent)

    finally:
        if server_ctx:
            server_ctx.stop()


# ---------------------------------------------------------------------------
#  Interactive REPL
# ---------------------------------------------------------------------------

def _repl(agent: Agent) -> None:
    while True:
        try:
            user_input = console.input("\n[bold green]You[/bold green]: ").strip()
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
            agent.chat(user_input)
        except KeyboardInterrupt:
            console.print("\n[dim](interrupted)[/dim]")
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

    elif cmd == "save":
        filepath = Path(arg) if arg else Path("conversation.json")
        try:
            agent.save_history(filepath)
            print_success(f"Saved to {filepath}")
        except Exception as e:
            print_error(f"Save failed: {e}")

    else:
        print_info(f"Unknown command: /{cmd}  (try /help)")

    return False
