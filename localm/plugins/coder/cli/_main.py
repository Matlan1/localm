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

import click

import localm.plugins.coder.cli as _cli
from ..backends.http import (
    HTTPBackend,
    make_anthropic_backend,
    make_openai_backend,
    CoderAuthError,
)
from ..agent import Agent
from ..agent.constants import _SHELL_EXEC_TOOLS
from ..audit import SessionMode, parse_mode
from ..privacy import (
    clear_shell_history_traces,
    suppress_readline_history,
    warn_external_provider,
)
from ..project_config import ProjectConfigUnreadable, load_project_config
from ..display import (
    console,
    print_banner,
    print_error,
    print_info,
    print_warning,
)
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


def _warn_unfinished_background(agent) -> None:
    """Report background sub-agents this one-shot run is about to abandon.

    The turn-boundary drain only fires at the START of a turn, so a child that is
    still running (or that finished after the final turn) is never folded in, and
    a one-shot process then exits and takes its daemon threads with it. Exiting
    silently would drop work the user explicitly asked for. What survives is
    stated exactly: a committed branch does, a running child does not.
    """
    try:
        from ..background import get_registry
        registry = get_registry()
        running = [j for j in registry.list_status(kind="agent")
                   if j["state"] == "running"]
        pending = registry.drain_finished(kind="agent")
    except Exception:
        return

    # Completions evicted before any drain saw them. Silence here would look
    # exactly like "nothing else finished", which is the one thing this warning
    # exists to prevent. Its OWN try, after the drain: drain_finished CONSUMES,
    # so a failure folded into the same try would discard completions already
    # handed over.
    try:
        lost = registry.take_dropped_undrained("agent")
    except Exception:
        lost = 0

    if lost:
        print_warning(
            f"{lost} background sub-agent completion(s) were discarded before "
            "they could be collected, so their results are lost.")
    for st in pending:
        branch = (st.get("result") or {}).get("branch")
        where = (f" Its work is committed on branch '{branch}'."
                 if branch else "")
        print_warning(
            f"background sub-agent '{st.get('label')}' ({st.get('id')}) finished "
            f"after the last turn, so its result was not folded into this run."
            f"{where}")
    for st in running:
        print_warning(
            f"background sub-agent '{st.get('label')}' ({st.get('id')}) is STILL "
            "RUNNING and will be killed when this one-shot run exits. Use an "
            "interactive session for background delegation, or spawn_agent "
            "(synchronous) for a one-shot.")


@click.command("coder", context_settings={"help_option_names": ["-h", "--help"]})
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
@click.option("--seed",             default=None,  type=int,
              help="RNG seed for sampling: the same seed, model, prompt and "
                   "settings reproduce the same output. Measured bit-for-bit on "
                   "one AMD gfx1030 box with the bundled llama.cpp runtime "
                   "(including across a model reload); other hardware, "
                   "backends, llama.cpp builds and concurrent load were not "
                   "measured, so treat it as repeatable-here, not guaranteed "
                   "everywhere. Ignored by --anthropic (no such API param).")
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
                  "log = JSONL audit trail to <data dir>/sessions/; "
                  "full = log + markdown transcript in .localcoder/sessions/."
              ))
@click.option("--scope",            default=None,
              help=(
                  "Restrict all file-access tools to paths matching this glob, "
                  "e.g. 'src/**/*.py'.  Requests touching files outside the "
                  "scope are rejected.  Does NOT confine run_shell/run_tests: "
                  "they start a process, which a path check cannot bound, so a "
                  "command can still reach outside the scope (the session warns "
                  "about this).  Disable those tools for a hard boundary."
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
              help="Erase ALL episodic memory for this project, archive included, "
                   "and exit.")
@click.option("--forget-episode", "forget_episode_id", default=None, metavar="ID",
              help="Drop ONE lesson by id (see --episodes) from recall and exit. It "
                   "is archived, so --restore-episode can bring it back; use "
                   "--forget-episodes to erase everything for good.")
@click.option("--episodes-archive", "show_archive", is_flag=True, default=False,
              help="List the lessons this project has dropped (merged, evicted at "
                   "the cap, or forgotten) and exit. These are recoverable.")
@click.option("--restore-episode", "restore_episode_id", default=None, metavar="ID",
              help="Put an archived lesson (see --episodes-archive) back into "
                   "recall and exit.")
@click.option("--consolidate-episodes", "consolidate_episodes", is_flag=True,
              default=False,
              help="Ask the model to merge related stored lessons into one. "
                   "OPT-IN and manual only, never automatic: the originals are "
                   "archived, so a bad merge is reversible with --restore-episode.")
@click.option("--until", "until_cmd", default=None, metavar="COMMAND",
              help="Goal mode: iterate on the TASK until this command exits 0 "
                   "(e.g. --until 'pytest -x'). Success is judged by the command, "
                   "not the model, so it cannot declare premature success. "
                   "Requires a TASK; exits non-zero if the goal is not met.")
@click.option("--goal-max-iters", "goal_max_iters", default=5,
              type=click.IntRange(1, 50),
              help="Max fix iterations for --until before giving up [default: 5].")
@click.option("--verify", "verify_cmd", default=None, metavar="COMMAND",
              help="Interactive sessions: the command whose exit code decides "
                   "whether a turn that changed files is done, run by the "
                   "harness (not the model) before it may finish. Defaults to "
                   "the project's detected check (pytest / npm test / cargo / "
                   "go, or a 'verify' key in .localcoder/config.toml). "
                   "Use --no-verify to turn it off. For a one-shot TASK, use "
                   "--until instead.")
@click.option("--no-verify", "no_verify", is_flag=True,
              help="Do not run any exit-code check before an interactive turn "
                   "finishes, even if the project has an obvious one.")
def main(
    task, model, url, api_key, port, cwd,
    no_server, force_new, max_turns, temperature, max_tokens, seed,
    verbose, yes, interactive_confirm, dry_run, estimate, patch_mode, ci, output_format,
    native_tools, provider, mode, scope, system_instructions,
    show_episodes, forget_episodes, forget_episode_id, show_archive,
    restore_episode_id, consolidate_episodes,
    until_cmd, goal_max_iters, verify_cmd, no_verify,
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
    if _handle_episode_flags(work_dir, show_episodes, forget_episodes,
                             forget_episode_id, show_archive, restore_episode_id):
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
            work_dir, model, max_turns, max_tokens, temperature, seed, yes,
            interactive_confirm, mode, ci, provider))

    # The in-loop exit-code oracle is for INTERACTIVE sessions (this REPL and the
    # GUI). A one-shot TASK keeps its own outer loop (--until, below), so the two
    # never both run and the check is never executed twice per iteration.
    session_verify = None
    if not task and not no_verify:
        session_verify = verify_cmd or _detect_verify(work_dir)
    elif task and verify_cmd and not until_cmd:
        # Do not accept a flag and quietly not use it: say which one runs a
        # one-shot task's check.
        print_warning(
            "--verify applies to an interactive session; a one-shot TASK is "
            f"verified with --until. Re-run with: --until '{verify_cmd}'")

    # Build the LLM backend.
    backend = _build_backend(
        provider, url, model, api_key, native_tools, port, no_server,
        force_new, work_dir, ci)

    # --native-tools asked for a protocol the chosen server does not implement.
    # Say so instead of letting it look applied: localm's own /v1/chat/completions
    # declares no `tools`/`tool_choice` fields, so the request body carrying them
    # is accepted and the fields are dropped, and the run proceeds on the XML
    # tool-call convention exactly as if the flag had never been passed. Nothing
    # breaks, which is precisely why the silence was the problem - the user has
    # no way to tell an honoured flag from an ignored one. Not an error: the
    # session is fine, and localm's grammar-constrained tool calls are the
    # equivalent guarantee on this path (AGENTS.md rule 5).
    if native_tools and not getattr(backend, "supports_native_tools", True):
        print_warning(
            "--native-tools was ignored: this server does not implement the "
            "OpenAI tools API. The session uses localm's own tool-call "
            "convention (grammar-constrained where the loaded model supports "
            "it). Use --native-tools with --url against a server that does, "
            "e.g. Ollama.")

    # Opt-in episodic consolidation needs a model, so unlike the other episode
    # flags it runs after the backend is up. Still a one-shot: consolidate, report,
    # exit - it never piggybacks on a normal session.
    if consolidate_episodes:
        _run_episode_consolidation(work_dir, backend)
        return

    # R19a: an unattended one-shot (`localcoder "task"`) auto-approves file writes
    # so it can run without a TTY - but shell (arbitrary code execution) and the
    # network tools must NOT be silently auto-run. Require confirmation for
    # run_shell unless the user explicitly passed --yes; a non-interactive run has
    # no way to confirm, so the gate in execution.py fails CLOSED (denied). This
    # makes confirm-on-shell the effective default for the one-shot CLI.
    if task and not yes:
        # run_shell_background is the same capability as run_shell (arbitrary code
        # execution, just without the wait), so it MUST carry the same gate - a
        # background variant left out here would be an unconfirmed-RCE bypass of it.
        always_confirm = set(always_confirm) | set(_SHELL_EXEC_TOOLS)
        print_warning(
            "Unattended one-shot: run_shell is code execution and the web tools "
            "egress data. run_shell and run_shell_background now require --yes "
            "(else they are denied this run); network tools still obey net_mode "
            "(off/ask/allow). Pass --yes to auto-approve, or run interactively to "
            "confirm each call.")

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
        verify_cmd=session_verify,
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

            # A one-shot run has no NEXT turn, so a background sub-agent that is
            # still going (or that finished after the last turn) never reaches the
            # turn-boundary drain - and this process is about to exit, taking its
            # daemon threads with it. Say so plainly instead of exiting quietly on
            # work the user asked for: the committed branch survives, the running
            # child does not.
            _warn_unfinished_background(agent)

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
            # Notify if an interrupted session is waiting. checkpoint_info() is
            # a read-only probe (NOT agent.load_checkpoint(), which - since
            # NEW-CODER-RESUME-DESTROYS-SESSIONS keyed the checkpoint on this
            # agent's own id - would claim the most recent saved session's
            # identity as a side effect of merely checking it exists, so
            # typing a plain message afterward (never /resume) would delete
            # then silently overwrite THAT session instead of starting fresh
            # under this agent's own id).
            from ..agent.checkpoint import checkpoint_info
            info = checkpoint_info(work_dir)
            if info and info.get("unreadable"):
                print_warning(
                    "A saved session was found for this project but could not "
                    "be read; it will not appear in /sessions."
                )
            elif info:
                ts = info.get("interrupted_at", "unknown time")
                turns = info.get("turns", "?")
                print_warning(
                    f"Interrupted session found ({turns} turns, {ts}). "
                    "Type /resume to continue, or /sessions to see all of them."
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


def _handle_episode_flags(work_dir: Path, show_episodes: bool,
                          forget_episodes: bool, forget_episode_id=None,
                          show_archive: bool = False,
                          restore_episode_id=None) -> bool:
    """Inspect or edit the episodic memory for *work_dir*, returning True if a flag
    was handled (so main can exit). No model/server needed - the user can always
    see, prune, and recover what the coder remembers. Split out of main."""
    if not (show_episodes or forget_episodes or forget_episode_id
            or show_archive or restore_episode_id):
        return False
    from localm.plugins.coder.episodes import EpisodeStore
    store = EpisodeStore(work_dir)

    if forget_episodes:
        store.clear()
        click.echo(f"Cleared episodic memory for {work_dir} (archive included).")
        return True

    if forget_episode_id:
        if store.forget(forget_episode_id):
            click.echo(f"Forgot episode {forget_episode_id} (archived; restore it "
                       f"with --restore-episode {forget_episode_id}).")
        else:
            click.echo(f"No episode with id {forget_episode_id} for {work_dir}.",
                       err=True)
        return True

    if restore_episode_id:
        ep = store.restore(restore_episode_id)
        if ep is not None:
            click.echo(f"Restored episode {ep.id}: {ep.lesson or ep.summary}")
            if not store.last_restore_archive_ok:
                # The lesson IS live again, so this is a caveat and not a failure -
                # but staying quiet would hide that the archive still lists it.
                click.echo(
                    "Note: the archive could not be updated, so this lesson is "
                    "also still listed by --episodes-archive. Re-run this command "
                    "once the archive is writable to tidy that up.", err=True)
            if any(e.id == ep.id for e in store.last_evicted):
                # It came back and went straight out again at the cap: saying only
                # "Restored" would be a claim the next read contradicts.
                click.echo(
                    "Note: the store is at its episode cap and this lesson ranked "
                    "lowest, so it was dropped again immediately (a recovery copy "
                    "was kept). Forget a lesson you no longer need first, with "
                    "--forget-episode.", err=True)
        elif not store.last_forgotten_ok:
            # The archive EXISTS but could not be read, so "no such id" would be a
            # claim we cannot make: the episode may well be sitting in there,
            # recoverable, and a retry once the file is readable would find it.
            # Rule 5 - do not report a clean negative for a step that failed.
            click.echo(
                f"Could not read the episode archive for {work_dir}, so episode "
                f"{restore_episode_id} could not be looked up. The archive file "
                f"exists ({store.archive_path.name}); it may be locked by another "
                f"process. Nothing was changed - try again.", err=True)
            # Non-zero: the lookup did not happen. Exiting 0 would tell a script
            # the same thing a genuine "no such episode" does, which is the very
            # collapse this branch exists to undo.
            sys.exit(1)
        else:
            click.echo(f"No archived episode with id {restore_episode_id}.", err=True)
        return True

    if show_archive:
        rows = store.forgotten()
        if not rows and not store.last_forgotten_ok:
            # Same distinction as above: an unreadable archive is not an empty one.
            click.echo(
                f"Could not read the episode archive for {work_dir}. The archive "
                f"file exists ({store.archive_path.name}) but could not be read, "
                f"so this list would be INCOMPLETE - it may be locked by another "
                f"process. Try again.", err=True)
            sys.exit(1)          # not the same outcome as an empty archive
        if not rows:
            click.echo("No dropped episodes archived for this project.")
            return True
        click.echo(f"{len(rows)} archived episode(s) for {work_dir}:")
        for r in rows:
            click.echo("  - %s [%s] %s" % (r.get("id", "?"), r.get("reason", "?"),
                                           r.get("lesson") or r.get("summary") or ""))
        return True

    eps = store.all()
    if not eps:
        click.echo("No episodic-memory lessons stored for this project yet.")
        return True
    click.echo(f"{len(eps)} episode(s) for {work_dir}:")
    for e in eps:
        # The id is what --forget-episode / --restore-episode address.
        click.echo(f"  - {e.id} [{e.outcome}] {e.lesson or e.summary}")
    return True


def _run_episode_consolidation(work_dir: Path, backend) -> None:
    """Run the OPT-IN model merge of related lessons and REPORT what it did.

    Reporting is the point: memory that rewrites itself without saying so is how a
    bad merge becomes invisible. Groups whose merge came back unusable are left
    untouched and counted."""
    from localm.textnorm import strip_think
    from localm.plugins.coder.episodes import EpisodeStore, consolidate

    def _complete(prompt: str) -> str:
        return strip_think(backend.chat([{"role": "user", "content": prompt}],
                                        max_tokens=1024) or "")

    store = EpisodeStore(work_dir)
    rep = consolidate(store, complete=_complete)
    if not rep.get("groups"):
        click.echo("Nothing to consolidate: no related lessons found.")
        return
    click.echo("Consolidated %d group(s): %d lesson(s) merged into %d, %d archived "
               "(restore any with --restore-episode)."
               % (rep["groups"], rep["replaced"], rep["merged"], rep["archived"]))
    if rep.get("skipped"):
        click.echo("%d group(s) left untouched (the model returned no usable merge)."
                   % rep["skipped"])
    if rep.get("warning"):
        print_warning(rep["warning"])


def _detect_verify(work_dir: Path):
    """The project's obvious check command for an interactive session, or None.

    Best-effort: detection reads project files, and a failure to read them is a
    reason to run no oracle, not to refuse the session. It is reported rather
    than swallowed silently, because a check the user expected to run and that
    quietly did not is exactly the kind of hidden problem that wastes their
    time."""
    try:
        from ..verify import command_text, detect_verify_command
        cmd = detect_verify_command(work_dir)
    except Exception as exc:                                   # noqa: BLE001
        print_warning(f"Could not detect a project check ({exc}); "
                      "no exit-code verification will run this session.")
        return None
    if cmd is not None:
        print_info(f"Verification: `{command_text(cmd)}` must exit 0 before a "
                   "turn that changed files finishes (/verify off to disable).")
    return cmd


def _resolve_session_config(work_dir, model, max_turns, max_tokens, temperature,
                            seed, yes, interactive_confirm, mode, ci, provider):
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
    try:
        proj_cfg = load_project_config(work_dir)
    except ProjectConfigUnreadable as e:
        # REFUSE rather than start with the file's settings silently dropped.
        # Two of them are safety settings: always_confirm (below) is what makes
        # shell tools still prompt under --yes, and mode = "privacy" is what
        # keeps a transcript off disk. Starting anyway would run the session
        # WITHOUT protections the user believes they configured, so a typo would
        # silently disarm them. Refusing costs one edit and is recoverable.
        print_error(
            f"Project config could not be read: {e}\n"
            f"Fix the file or remove it, then run again. It is not ignored, "
            f"because it may set 'always_confirm' or 'mode' and starting "
            f"without them would silently drop protections you configured.")
        sys.exit(2 if ci else 1)
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
    if seed is None and "seed" in proj_cfg:
        seed = int(proj_cfg["seed"])
    if seed is not None and provider == "anthropic":
        # Say so instead of sending an unknown field to the Messages API and
        # letting the run look reproducible when nothing pinned it.
        print_warning(
            "--seed is ignored with --anthropic: the Anthropic Messages API has "
            "no seed parameter, so this run is not reproducible.")
        seed = None
    # auto_approve: config applies only when --yes flag was NOT passed
    if not yes and proj_cfg.get("auto_approve"):
        yes = True
    # always_confirm: tools that prompt even under --yes
    # --interactive-confirm sets the shell-execution tools; config can extend the list
    always_confirm: set[str] = set()
    if interactive_confirm:
        always_confirm.update(_SHELL_EXEC_TOOLS)
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
        ("seed",        seed),
    ] if v is not None}
    return model, max_turns, yes, always_confirm, session_mode, gen_kw


def _build_backend(provider, url, model, api_key, native_tools, port, no_server,
                   force_new, work_dir, ci):
    """Build the LLM backend for the chosen provider / explicit URL / offline path.
    Returns backend. Split out of main; exits
    (2 under --ci, else 1) on a missing required option or a failed server start."""
    # Live-attribute access so a test patching cli.make_localm_backend is honoured
    # (the name moved into this submodule when cli.py became a package).
    make_localm_backend = _cli.make_localm_backend

    # AUTH-ATTACH: whether the caller actually chose *api_key* (a real
    # -k/--api-key or $LOCALM_API_KEY) rather than it sitting at its literal
    # placeholder default "localm" - the same distinction _explicit() draws in
    # plugins/gui/cli.py, widened to also count ENVIRONMENT (the option's own
    # envvar) as explicit, not just COMMANDLINE. Needed below: an unset
    # --api-key must never override an explicit user choice (hard-won rule),
    # but it also must never be sent as-is to a keyed server, since "localm"
    # authenticates nothing real.
    from click.core import ParameterSource
    _api_key_explicit = (
        click.get_current_context().get_parameter_source("api_key")
        != ParameterSource.DEFAULT)

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
        # the "one server handles chat + coder" fix - see the credential
        # resolution just below (_api_key_explicit / resolve_bearer_token) for
        # how it authenticates without a guessed key.
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
                # Authenticate with the user's EXPLICIT key (--api-key /
                # $LOCALM_API_KEY) when they gave one; otherwise resolve the
                # same credential `localm run` uses for its own attach - the
                # owner key when this install has one configured, else the
                # discovered instance's own attach token in open mode
                # (auth.resolve_bearer_token). Without this, an unset
                # --api-key falls through to its literal placeholder
                # "localm", which authenticates nothing and 401s against any
                # keyed server (checkup 2026-08-11 item 12).
                attach_key = api_key
                if not _api_key_explicit:
                    from localm.auth import resolve_bearer_token
                    attach_key = resolve_bearer_token(_tgt.get("token")) or api_key
                attached = HTTPBackend(_tgt["base_url"], model, api_key=attach_key,
                                       native_tools=native_tools,
                                       localm_server=True)   # discovered localm instance
                console.print(
                    f"[dim]Using the localm already running for[/dim] "
                    f"[cyan]{_root}[/cyan] [dim](port {_tgt['port']}, "
                    f"mode {_tgt.get('mode')}) - sharing its loaded model.[/dim]")

        if attached is not None:
            backend = attached
        else:
            if no_server:
                srv_port = port or 8642
                # api_key's click default is the literal placeholder "localm"
                # (always truthy), so forwarding it as-is would short-circuit
                # make_localm_backend's own env/auth.key resolution before it
                # ever runs. Forward "" (its own "nothing explicit" sentinel)
                # when the user gave no real key, so a server keyed via
                # `localm key generate` (persisted to auth.key, no env var)
                # still authenticates (checkup 2026-08-11 item 12).
                attach_key = api_key if _api_key_explicit else ""
                backend = make_localm_backend(model, port=srv_port, api_key=attach_key)
            else:
                console.print("[dim]No server running. Starting one in the background...[/dim]")
                import subprocess
                import time
                from localm import instances
                from localm.config import PortInUseError, home_dir, pick_port

                # An explicit --port that's already busy is refused by `localm gui`
                # itself (it prints "Port N is already in use..." to its OWN console
                # and exits - see #740). This process never reads that console, so
                # check up front and surface the same real reason here instead of
                # waiting out the attach-timeout below.
                if port:
                    try:
                        pick_port(port, host="127.0.0.1")
                    except PortInUseError as exc:
                        print_error(f"Port {exc.port} is already in use. "
                                    "Free it, or choose another with -p/--port.")
                        sys.exit(2 if ci else 1)

                cmd = [sys.executable, "-m", "localm", "gui", "--no-browser", "--api-mode"]
                if model:
                    cmd.append(model)
                else:
                    cmd.append("--no-model")
                if port:
                    cmd.extend(["-p", str(port)])

                import os
                kwargs = {}
                env = os.environ.copy()
                env["LOCALM_OWN_CONSOLE"] = "1"
                kwargs["env"] = env
                # A HEADLESS caller (MCP's run_coder_task, CI, a script - stdin
                # is not a TTY) has no desktop for a new console window, so the
                # child's real startup error (e.g. gui's "Model not found: X",
                # reproduced live 2026-07-21) died in a window nobody could see
                # while this process could only report "exit code 1". Capture
                # the child's output to a LOG FILE instead - a PIPE would
                # deadlock a healthy long-lived server once the OS buffer fills
                # with its normal request logging, a file cannot - read its
                # tail back on early exit, and point at the file otherwise. A
                # TTY run keeps today's visible console window.
                headless = not sys.stdin.isatty()
                child_log_path = None
                child_log = None
                if headless:
                    import tempfile
                    fd, child_log_path = tempfile.mkstemp(
                        prefix="localm-coder-autostart-", suffix=".log")
                    child_log = os.fdopen(fd, "w", encoding="utf-8",
                                          errors="replace")
                    kwargs["stdout"] = child_log
                    kwargs["stderr"] = subprocess.STDOUT
                    if sys.platform == "win32":
                        # No console at all (nothing would be visible anyway);
                        # a fresh process group so the server outlives this
                        # CLI, mirroring start_new_session on POSIX.
                        kwargs["creationflags"] = (
                            subprocess.DETACHED_PROCESS
                            | subprocess.CREATE_NEW_PROCESS_GROUP)
                    else:
                        kwargs["start_new_session"] = True
                elif sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
                else:
                    kwargs["start_new_session"] = True

                def _child_log_tail(limit: int = 2000) -> str:
                    """Best-effort tail of the captured child output; '' when
                    nothing was captured (TTY mode, or the child wrote nothing
                    before dying). Read failures are non-fatal by design: the
                    log is diagnostic garnish on an error path that already
                    reports the exit code either way."""
                    if not child_log_path:
                        return ""
                    try:
                        with open(child_log_path, "r", encoding="utf-8",
                                  errors="replace") as f:
                            return f.read()[-limit:].strip()
                    except OSError:
                        return ""

                try:
                    proc = subprocess.Popen(cmd, **kwargs)
                    if child_log is not None:
                        # The child holds its own handle from here; closing
                        # ours flushes anything buffered so the early-exit
                        # read below sees everything written so far.
                        child_log.close()
                    _tgt = None
                    for _ in range(40):
                        time.sleep(0.5)
                        if proc.poll() is not None:
                            break
                        _root = instances.resolve_root_dir(start=str(work_dir))
                        _tgt = instances.attach_target(home_dir(), _root)
                        if _tgt:
                            break
                    if _tgt:
                        attach_key = api_key
                        if not _api_key_explicit:
                            from localm.auth import resolve_bearer_token
                            attach_key = (resolve_bearer_token(_tgt.get("token"))
                                         or api_key)
                        backend = HTTPBackend(_tgt["base_url"], model, api_key=attach_key,
                                              native_tools=native_tools,
                                              localm_server=True)
                        from localm.console import show_url
                        console.print(f"[dim]connected to newly started server at "
                                      f"{show_url(_tgt['base_url'])}[/dim]")
                    elif proc.poll() is not None:
                        tail = _child_log_tail()
                        if tail:
                            detail = f" Its output:\n{tail}"
                        elif headless:
                            detail = ""
                        else:
                            detail = (" Check the new console window it opened "
                                      "for the specific error.")
                        print_error(
                            f"The auto-started server exited immediately (exit code "
                            f"{proc.returncode}) instead of starting.{detail}"
                        )
                        sys.exit(2 if ci else 1)
                    else:
                        hint = (f" Its output log: {child_log_path}"
                                if child_log_path else "")
                        print_error(
                            f"Failed to attach to the auto-started server.{hint}")
                        sys.exit(2 if ci else 1)
                except Exception as e:
                    print_error(f"Failed to start server: {e}")
                    sys.exit(2 if ci else 1)

    return backend


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
