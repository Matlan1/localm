# SPDX-License-Identifier: AGPL-3.0-or-later
"""
localcoder CLI entry point.

Commands
--------
  localcoder                     Interactive chat (auto-starts localm serve)
  localcoder "task"              Single-shot task (non-interactive)
  localcoder --model X           Specify model
  localcoder --url URL           Point at any OpenAI-compatible server
  localcoder --no-server         Don't start localm serve (connect to existing)

This module owns the ``main`` Click command (the localcoder entry point). The
goal-mode loop, estimate mode, and the interactive REPL live in sibling modules
(goal / estimate / repl); the heavy phases of ``main`` are split into the
``_handle_episode_flags`` / ``_resolve_session_config`` / ``_build_backend``
helpers below.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click

import localm.plugins.coder.cli as _cli
from ..backends.http import (
    HTTPBackend,
    make_anthropic_backend,
    make_openai_backend,
    CoderAuthError,
)
from ..agent import Agent
from ..audit import SessionMode, parse_mode
from ..privacy import (
    clear_shell_history_traces,
    suppress_readline_history,
    warn_external_provider,
)
from ..project_config import load_project_config
from ..display import (
    console,
    print_banner,
    print_error,
    print_info,
    print_warning,
)
from ..server import ManagedServer, find_free_port
from .goal import _run_goal_loop
from .estimate import _run_estimate
from .repl import _repl

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
              help="OpenAI-compat base URL, e.g. http://127.0.0.1:8642/v1.")
@click.option("-k", "--api-key",    default="localm", envvar="LOCALM_API_KEY",
              help="API key for the localm/custom server (also $LOCALM_API_KEY). "
                   "A require_auth server needs the real key.")
@click.option("-p", "--port",       default=None,  type=click.IntRange(1, 65535),
              help="Port for the auto-started localm serve [default: first free in 8642-8741].")
@click.option("-c", "--cwd",        default=None,  type=click.Path(exists=True, file_okay=False),
              help="Working directory [default: current directory].")
@click.option("--no-server",        is_flag=True,
              help="Don't start localm serve - assume server is already running.")
@click.option("--new",  "force_new", is_flag=True,
              help="Start a dedicated server even if a localm is already running "
                   "for this project (by default localcoder attaches to it and "
                   "uses its loaded model, so chat + coder share one model in VRAM).")
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
@click.option("--estimate",         is_flag=True,
              help=(
                  "Plan only: one LLM turn that outlines the approach and rough "
                  "effort for TASK, executes nothing, then exits."
              ))
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
@click.option("--scope",            default=None,
              help=(
                  "Restrict all file-access tools to paths matching this glob, "
                  "e.g. 'src/**/*.py'.  Requests touching files outside the "
                  "scope are rejected."
              ))
@click.option("--system", "system_instructions", default=None, metavar="TEXT",
              help=(
                  "Custom system instructions for the agent (conventions, style, "
                  "constraints). Overrides .localcoder/system.md for this run; when "
                  "omitted, that file is used automatically if present."
              ))
@click.option("--episodes", "show_episodes", is_flag=True, default=False,
              help="List the episodic-memory lessons stored for this project and exit.")
@click.option("--forget-episodes", "forget_episodes", is_flag=True, default=False,
              help="Delete all stored episodic-memory lessons for this project and exit.")
@click.option("--until", "until_cmd", default=None, metavar="COMMAND",
              help="Goal mode: iterate on the TASK until this command exits 0 "
                   "(e.g. --until 'pytest -x'). Success is judged by the command, "
                   "not the model, so it cannot declare premature success. "
                   "Requires a TASK; exits non-zero if the goal is not met.")
@click.option("--goal-max-iters", "goal_max_iters", default=5,
              type=click.IntRange(1, 50),
              help="Max fix iterations for --until before giving up [default: 5].")
def main(
    task, model, url, api_key, port, cwd,
    no_server, force_new, max_turns, temperature, max_tokens,
    verbose, yes, interactive_confirm, dry_run, estimate, patch_mode, ci, output_format,
    native_tools, provider, mode, scope, system_instructions,
    show_episodes, forget_episodes,
    until_cmd, goal_max_iters,
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
    # The coder is an optional plugin (Phase 3); it ships uninstalled. Running
    # `localm coder` / `localcoder` before it is installed+enabled gets a clear
    # refusal rather than a half-working agent.
    from localm.plugins.engine import PluginManager
    if not PluginManager(None).is_active("coder"):     # installed (on disk) AND enabled
        click.echo(
            "The coder plugin is not active.\n"
            "Install it with:  localm plugin install coder",
            err=True,
        )
        raise SystemExit(1)

    work_dir = Path(cwd).resolve() if cwd else Path.cwd()

    # Episodic-memory management: list or clear the lessons stored for this
    # project, then exit (no model/server needed). Transparency: the user can
    # always see and wipe what the coder remembers about a project.
    if _handle_episode_flags(work_dir, show_episodes, forget_episodes):
        return

    # Goal mode needs a task to work on; fail fast before any server/model setup.
    if until_cmd and not task:
        print_error("--until requires a TASK to work on.")
        sys.exit(2 if ci else 1)

    # Resolve CI defaults, project config (.localcoder/config.toml), the session
    # mode + privacy setup, and the LLM gen kwargs (split out of main; see
    # _resolve_session_config).
    model, max_turns, yes, always_confirm, session_mode, gen_kw = (
        _resolve_session_config(
            work_dir, model, max_turns, max_tokens, temperature, yes,
            interactive_confirm, mode, ci, provider))

    # Build the LLM backend (and the managed server when one is auto-started;
    # split out of main; see _build_backend).
    backend, server_ctx = _build_backend(
        provider, url, model, api_key, native_tools, port, no_server,
        force_new, work_dir, ci)

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
        scope=scope,
        custom_instructions=system_instructions,
        **gen_kw,
    )

    if patch_mode:
        agent.patch_mode = True

    try:
        if estimate:
            if not task:
                print_error("--estimate requires a TASK to estimate.")
                sys.exit(2 if ci else 1)
            _run_estimate(agent, task, output_format)
            return

        if task:
            # Non-interactive single-task mode (optionally a verify-until-pass loop)
            if until_cmd:
                success, response = _run_goal_loop(
                    agent, task, until_cmd, goal_max_iters, work_dir)
            else:
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

            # Goal mode: a failed verification is a real non-zero exit (the whole
            # point is a pass/fail oracle), not only under --ci.
            if (until_cmd or ci) and not success:
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
    except CoderAuthError as e:
        print_error(str(e))
        sys.exit(2 if ci else 1)
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

        _warn_sensitive_changes(agent)
        md_path = agent.close()
        if md_path:
            print_info(f"Session transcript saved → {md_path}")
        if session_mode == SessionMode.PRIVACY:
            clear_shell_history_traces()
        if server_ctx:
            server_ctx.stop()


def _handle_episode_flags(work_dir: Path, show_episodes: bool,
                          forget_episodes: bool) -> bool:
    """List or clear the episodic-memory lessons for *work_dir*, returning True if
    the flags were handled (so main can exit). No model/server needed - the user
    can always see and wipe what the coder remembers. Split out of main."""
    if show_episodes or forget_episodes:
        from localm.plugins.coder.episodes import EpisodeStore
        store = EpisodeStore(work_dir)
        if forget_episodes:
            store.clear()
            click.echo(f"Cleared episodic memory for {work_dir}.")
            return True
        eps = store.all()
        if not eps:
            click.echo("No episodic-memory lessons stored for this project yet.")
            return True
        click.echo(f"{len(eps)} episode(s) for {work_dir}:")
        for e in eps:
            click.echo(f"  - [{e.outcome}] {e.lesson or e.summary}")
        return True
    return False


def _resolve_session_config(work_dir, model, max_turns, max_tokens, temperature,
                            yes, interactive_confirm, mode, ci, provider):
    """Resolve CI defaults, project config (.localcoder/config.toml, CLI flags
    override), the session mode + privacy setup, and the LLM gen kwargs. Returns
    (model, max_turns, yes, always_confirm, session_mode, gen_kw). Split out of
    main; exits (2 under --ci, else 1) on an invalid mode."""
    # CI mode setup
    if ci:
        yes = True          # never prompt
        if mode is None:
            mode = "log"    # always leave an audit trail in CI
        # Strip Rich colors so log output is plain text
        import os as _os
        _os.environ.setdefault("NO_COLOR", "1")
        _os.environ.setdefault("TERM", "dumb")

    # Project-level config (.localcoder/config.toml) - CLI flags override
    proj_cfg = load_project_config(work_dir)
    if model is None:
        model = proj_cfg.get("model")
    if max_turns is None:
        max_turns = int(proj_cfg.get("max_turns", 40))
    if max_tokens is None:
        _cfg_max_tokens = proj_cfg.get("max_tokens")
        if _cfg_max_tokens is not None:
            max_tokens = int(_cfg_max_tokens)
        else:
            # No explicit value: a per-model default (e.g. more room for a
            # thinking model's <think> + answer), else the baseline cap.
            from localm.plugins.coder.harness_profiles import cli_max_tokens
            max_tokens = cli_max_tokens(model)
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
    # mode: CLI > project config > global config (coder_mode/mode) > privacy
    if mode is None:
        mode = proj_cfg.get("mode")
    if mode is None:
        from localm.audit import effective_mode
        mode = effective_mode("coder").value
    try:
        session_mode = parse_mode(mode)
    except ValueError as exc:
        print_error(str(exc))
        sys.exit(2 if ci else 1)

    # Privacy-mode setup - suppress readline history as early as possible
    if session_mode == SessionMode.PRIVACY:
        suppress_readline_history()
    # Warn when privacy mode is requested but prompts leave the machine
    if session_mode == SessionMode.PRIVACY and provider in ("openai", "anthropic"):
        warn_external_provider(provider)

    gen_kw   = {k: v for k, v in [
        ("temperature", temperature),
        ("max_tokens",  max_tokens),
    ] if v is not None}
    return model, max_turns, yes, always_confirm, session_mode, gen_kw


def _build_backend(provider, url, model, api_key, native_tools, port, no_server,
                   force_new, work_dir, ci):
    """Build the LLM backend for the chosen provider / explicit URL / offline path.
    Returns (backend, server_ctx) where server_ctx is a ManagedServer when one was
    auto-started for the offline path (else None). Split out of main; exits
    (2 under --ci, else 1) on a missing required option or a failed server start."""
    # Live-attribute access so a test patching cli.make_localm_backend is honoured
    # (the name moved into this submodule when cli.py became a package).
    make_localm_backend = _cli.make_localm_backend
    server_ctx: Optional[ManagedServer] = None

    if provider == "openai":
        if not model:
            model = "gpt-4o"
        backend = make_openai_backend(model=model)

    elif provider == "anthropic":
        if not model:
            model = "claude-opus-4-5"
        # Anthropic Messages API (x-api-key + anthropic-version + /v1/messages).
        backend = make_anthropic_backend(model=model)

    elif url:
        # Explicit URL - no server management
        if not model:
            print_error("--url requires --model")
            sys.exit(2 if ci else 1)
        backend = HTTPBackend(url.rstrip("/"), model, api_key=api_key,
                              native_tools=native_tools)

    else:
        # Offline path - localm backend
        if not model:
            print_error(
                "Specify a model with --model, e.g.:\n"
                "  localcoder --model gemma4-4b\n"
                "Run `localm list` to see registered models."
            )
            sys.exit(2 if ci else 1)

        # H6 phase 6: attach to the localm already running for this project dir
        # instead of starting a second server that loads its own model. This is
        # the "one server handles chat + coder" fix - and it uses the running
        # instance's own token, so it no longer 401s with a guessed key.
        attached = None
        if not (force_new or no_server):
            try:
                from localm import instances
                from localm.config import home_dir
                _root = instances.resolve_root_dir(start=str(work_dir))
                _tgt = instances.attach_target(home_dir(), _root)
            except Exception:
                _tgt = None
            if _tgt:
                # Authenticate with the user's configured key (env / --api-key);
                # an open-mode instance accepts any bearer. (The per-instance
                # registry token as a keyless local-attach credential is a
                # follow-up - it needs the server to accept it as a bearer.)
                attached = HTTPBackend(_tgt["base_url"], model, api_key=api_key,
                                       native_tools=native_tools)
                console.print(
                    f"[dim]Using the localm already running for[/dim] "
                    f"[cyan]{_root}[/cyan] [dim](port {_tgt['port']}, "
                    f"mode {_tgt.get('mode')}) - sharing its loaded model.[/dim]")

        if attached is not None:
            backend = attached
        else:
            srv_port = port or find_free_port()
            if no_server:
                backend = make_localm_backend(model, port=srv_port, api_key=api_key)
            else:
                server_ctx = ManagedServer(model, port=srv_port)
                if not server_ctx.start():
                    sys.exit(2 if ci else 1)
                backend = make_localm_backend(model, port=srv_port, api_key=api_key)

    return backend, server_ctx


def _warn_sensitive_changes(agent: Agent) -> None:
    """Surface test / CI-config edits so a green check over rewritten tests is
    reviewed, not trusted (R19, agentic code review). Best-effort: never let this
    advisory break the session."""
    try:
        from ..review_guard import classify_sensitive_changes, render_warning
        message = render_warning(classify_sensitive_changes(agent.changed_files()))
        if message:
            print_warning(message)
    except Exception:                                       # noqa: BLE001
        pass


def console_main() -> None:
    """The ``localcoder`` console-script entry point (pyproject [project.scripts]).

    Guards that we are inside the project venv, then runs the coder command. Kept
    SEPARATE from ``main`` so the ``localm coder`` route and the test suite invoke
    the command directly, without the venv gate; only a stray global ``localcoder``
    (a separate ``pip install``) hits it (NEW-J / NEW-J-CODER)."""
    from localm._venvguard import require_venv
    require_venv()
    main()
