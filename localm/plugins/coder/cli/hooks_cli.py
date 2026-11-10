# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line handling of the commands a project config declares (tool hooks
and MCP servers): the trust decision at startup and the ``/hooks`` REPL
command."""

from __future__ import annotations

from localm.console import markup_safe

from ..display import console, print_info, print_success, print_warning


def _noun(hooks) -> str:
    """What *hooks* declares: "Tool hooks", "MCP servers" or both, or "Project
    settings" when it declares neither."""
    if not hooks.has_commands():
        return "Project settings"
    if not hooks.has_mcp():
        return "Tool hooks"
    if not hooks.has_hooks():
        return "MCP servers"
    return "Tool hooks and MCP servers"


def _show_hooks(hooks) -> None:
    where = hooks.config_path if hooks.has_commands() else hooks.trust_path.parent
    console.print(f"[bold]{_noun(hooks)}[/bold] in {markup_safe(str(where))}:")
    for line in (*hooks.describe(), *hooks.describe_mcp(), *hooks.describe_project()):
        console.print(f"  {markup_safe(line)}")


def _remember(hooks) -> bool:
    """Trust *hooks* and record it. Returns True when the record was written;
    on a failure the hooks stay trusted for this session and the failure is
    reported."""
    from ..hook_trust import TrustStoreUnreadable
    try:
        hooks.remember_trust()
    except (OSError, TrustStoreUnreadable) as e:
        hooks.trust_for_session()
        print_warning(f"{_noun(hooks)} are trusted for this session only: the trust "
                      f"record could not be saved ({e}).")
        return False
    return True


def _apply_project(agent) -> None:
    """Load the project's trust-gated files into *agent* now that they are
    trusted (Agent.apply_project_trust)."""
    apply = getattr(agent, "apply_project_trust", None)
    if apply is not None:
        apply()


def _start_mcp(agent, *, remembered: bool) -> None:
    """Start the MCP servers the session held back, now that they are trusted,
    and say how many tools they registered. When the trusted config is not the
    one the session took its MCP servers from (after /cd), say instead that
    they start in a later session if the trust was *remembered*, and that they
    do not start otherwise."""
    from ..hook_trust import config_key
    _apply_project(agent)
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.has_mcp():
        return
    started_from = getattr(agent, "_mcp_config", None)
    if started_from is None or config_key(started_from) != config_key(hooks.config_path):
        if remembered:
            print_info("This session keeps the MCP servers of the folder it started "
                       "in; the ones in this project start in its next session.")
        else:
            print_info("This session keeps the MCP servers of the folder it started "
                       "in, so the ones in this project are not started; trust them "
                       "with \"a\" (always) or /hooks trust to start them in later sessions.")
        return
    if getattr(agent, "_mcp_deferred", None) is not hooks:
        return
    names = agent.start_deferred_mcp()
    if getattr(agent, "_mcp_deferred", None) is None:
        print_info(f"MCP servers started: {len(names)} tool(s) registered.")


def ask_trust(hooks) -> str:
    """Ask whether to run *hooks* and start its MCP servers. Returns "session",
    "always" or "no"; an interrupted prompt counts as "no"."""
    _show_hooks(hooks)
    if not hooks.has_commands():
        console.print("These settings and files come from the project folder and "
                      "change what the agent does and what it may do without asking. "
                      "Only trust a project you know.")
        refused = "none of them is used"
    elif not hooks.has_mcp():
        console.print("These commands run on this computer around the agent's tool "
                      "calls. Only trust hooks you have read.")
        refused = "calls a pre_tool_use hook checks are then refused"
    else:
        console.print("These commands run on this computer: each MCP server is "
                      "started as a process whose tools the agent can call"
                      + (", and the hooks run around the agent's tool calls"
                         if hooks.has_hooks() else "")
                      + ". Only trust commands you have read.")
        refused = "no MCP server is started"
        if hooks.has_hooks():
            refused += "; calls a pre_tool_use hook checks are refused"
    if hooks.has_commands() and hooks.has_project_items():
        refused += "; the project settings and files listed are not used"
    try:
        answer = console.input(
            "[yellow]Run them? \\[y] this session, \\[a] always (until they "
            f"change), \\[N] no ({refused}): [/yellow]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "no"
    if answer in ("a", "always"):
        return "always"
    if answer in ("y", "yes"):
        return "session"
    return "no"


def _declined_message(hooks) -> str:
    if not hooks.has_commands():
        return ("Project settings and files not trusted: none of them is used "
                "(/hooks trust to change that).")
    if not hooks.has_mcp():
        return ("Tool hooks not trusted: none of them run, and a tool call a "
                "pre_tool_use hook would check is refused (/hooks trust to "
                "change that).")
    if not hooks.has_hooks():
        return ("MCP servers not trusted: none of them is started (/hooks trust "
                "to change that).")
    return ("Tool hooks and MCP servers not trusted: no MCP server is started, none "
            "of the hooks run, and a tool call a pre_tool_use hook would check is "
            "refused (/hooks trust to change that).")


def settle_project_trust(work_dir, *, trust_flag: bool, can_prompt: bool) -> None:
    """Settle the project's trust before the session's settings are resolved,
    so the answer covers its ``auto_approve``, ``verify``, ``model`` and
    ``mode`` and the files the agent reads at startup.

    Nothing to do when nothing needs trust, it is already trusted, or the
    project config cannot be read (resolving the settings reports that).
    ``trust_flag`` trusts and records it; otherwise an interactive terminal is
    asked. A refusal is kept for this process, and resolve_hook_trust reports
    it once the session is built."""
    from ..hooks import ToolHooks
    from ..project_config import ProjectConfigUnreadable
    try:
        hooks = ToolHooks.load(work_dir)
    except ProjectConfigUnreadable:
        return
    if hooks is None or not hooks.needs_trust() or hooks.trusted:
        return
    if trust_flag:
        if _remember(hooks):
            print_success(f"{_noun(hooks)} in {hooks.trust_path.parent} trusted.")
        return
    if not can_prompt:
        return
    choice = ask_trust(hooks)
    if choice == "always":
        if _remember(hooks):
            print_success(f"{_noun(hooks)} trusted and remembered until they change.")
    elif choice == "session":
        hooks.trust_for_session()
        print_info(f"{_noun(hooks)} trusted for this session.")
    else:
        from ..hook_trust import set_session_decision
        hooks.revoke()
        set_session_decision(hooks.trust_path, hooks.digest, False)


def resolve_hook_trust(agent, *, trust_flag: bool, can_prompt: bool) -> None:
    """Settle, at startup, whether the session's project hooks may run and its
    MCP servers may start.

    Nothing to do when there are none or they are already trusted.
    ``trust_flag`` (``--trust-hooks``) trusts and records them. Otherwise an
    interactive terminal is asked; without one they stay untrusted and that is
    reported once. MCP servers held back start as soon as they are trusted."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.needs_trust() or hooks.trusted:
        return
    if hooks.revoked and not trust_flag:
        if hooks.claim_untrusted_report():
            print_warning(_declined_message(hooks))
        return
    if trust_flag:
        remembered = _remember(hooks)
        if remembered:
            print_success(f"{_noun(hooks)} in {hooks.config_path} trusted.")
        _start_mcp(agent, remembered=remembered)
        return
    if not can_prompt:
        if hooks.claim_untrusted_report():
            print_warning(hooks.untrusted_message(), to_stderr=True)
        return
    choice = ask_trust(hooks)
    if choice == "always":
        remembered = _remember(hooks)
        if remembered:
            print_success(f"{_noun(hooks)} trusted and remembered until they change.")
        _start_mcp(agent, remembered=remembered)
    elif choice == "session":
        hooks.trust_for_session()
        print_info(f"{_noun(hooks)} trusted for this session.")
        _start_mcp(agent, remembered=False)
    else:
        hooks.revoke()
        hooks.claim_untrusted_report()
        print_warning(_declined_message(hooks))


def handle_hooks_command(arg: str, agent) -> None:
    """``/hooks`` shows the project's hooks and MCP servers and whether they
    are trusted; ``/hooks trust`` trusts and remembers them and starts MCP
    servers the session held back; ``/hooks forget`` drops the remembered trust
    and stops the hooks for this session (MCP servers it already started keep
    running until it ends)."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.needs_trust():
        if getattr(agent, "restricted", False):
            print_info("Tool hooks never run and MCP servers never start in a "
                       "restricted session.")
        else:
            print_info("No tool hooks or MCP servers are configured for this "
                       "project ([[hooks.pre_tool_use]] / [[hooks.post_tool_use]] "
                       "/ [mcp.servers.<name>] in .localcoder/config.toml).")
        return
    sub = arg.strip().lower()
    if sub == "trust":
        remembered = _remember(hooks)
        if remembered:
            print_success(f"{_noun(hooks)} trusted and remembered until they change.")
        _start_mcp(agent, remembered=remembered)
        return
    if sub == "forget":
        from ..hook_trust import TrustStoreUnreadable, forget_trust
        hooks.revoke()
        try:
            forget_trust(hooks.trust_path)
        except (OSError, TrustStoreUnreadable) as e:
            print_warning(f"Could not update the trust record ({e}); the hooks "
                          "are stopped for this session only.")
            return
        if not hooks.has_mcp():
            print_info("Tool hooks are no longer trusted: none of them run, and a tool "
                       "call a pre_tool_use hook would check is refused.")
        else:
            detail = ""
            if hooks.has_hooks():
                detail = (", none of the hooks run, and a tool call a pre_tool_use "
                          "hook would check is refused")
            print_info(f"{_noun(hooks)} are no longer trusted: later sessions start "
                       f"no MCP server{detail}. MCP servers this session already "
                       "started keep running until it ends.")
        return
    if sub:
        print_info("Usage: /hooks [trust | forget]")
        return
    _show_hooks(hooks)
    if not hooks.has_commands():
        state = ("trusted: they are used" if hooks.trusted else
                 "NOT trusted: none of them is used")
    elif not hooks.has_mcp():
        state = ("trusted: they run" if hooks.trusted else
                 "NOT trusted: none run, and calls a pre_tool_use hook checks are refused")
    else:
        state = ("trusted: MCP servers start and hooks run" if hooks.trusted else
                 "NOT trusted: no MCP server starts and no hook runs")
    console.print(f"  [dim]{state}. /hooks trust or /hooks forget to change it.[/dim]")
