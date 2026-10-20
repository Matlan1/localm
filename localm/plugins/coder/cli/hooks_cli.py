# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line handling of the commands a project config declares (tool hooks
and MCP servers): the trust decision at startup and the ``/hooks`` REPL
command."""

from __future__ import annotations

from localm.console import markup_safe

from ..display import console, print_info, print_success, print_warning


def _noun(hooks) -> str:
    """What *hooks* declares: "Tool hooks", "MCP servers" or both."""
    if not hooks.has_mcp():
        return "Tool hooks"
    if not hooks.has_hooks():
        return "MCP servers"
    return "Tool hooks and MCP servers"


def _show_hooks(hooks) -> None:
    console.print(f"[bold]{_noun(hooks)}[/bold] in {markup_safe(str(hooks.config_path))}:")
    for line in (*hooks.describe(), *hooks.describe_mcp()):
        console.print(f"  {markup_safe(line)}")


def _remember(hooks) -> bool:
    """Trust *hooks* and record it. Returns True when the record was written;
    on a failure the hooks stay trusted for this session and the failure is
    reported."""
    from ..hook_trust import TrustStoreUnreadable
    try:
        hooks.remember_trust()
    except (OSError, TrustStoreUnreadable) as e:
        print_warning(f"{_noun(hooks)} are trusted for this session only: the trust "
                      f"record could not be saved ({e}).")
        return False
    return True


def _start_mcp(agent) -> None:
    """Start the MCP servers the session held back, now that they are trusted,
    and say how many tools they registered. When the trusted config is not the
    one the session took its MCP servers from (after /cd), say that they start
    in the next session instead."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.has_mcp():
        return
    if getattr(agent, "_mcp_config", None) != hooks.config_path:
        print_info("This session keeps the MCP servers of the folder it started "
                   "in; the ones in this project start in its next session.")
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
    if not hooks.has_mcp():
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


def resolve_hook_trust(agent, *, trust_flag: bool, can_prompt: bool) -> None:
    """Settle, at startup, whether the session's project hooks may run and its
    MCP servers may start.

    Nothing to do when there are none or they are already trusted.
    ``trust_flag`` (``--trust-hooks``) trusts and records them. Otherwise an
    interactive terminal is asked; without one they stay untrusted and that is
    reported once. MCP servers held back start as soon as they are trusted."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.has_commands() or hooks.trusted:
        return
    if trust_flag:
        if _remember(hooks):
            print_success(f"{_noun(hooks)} in {hooks.config_path} trusted.")
        _start_mcp(agent)
        return
    if not can_prompt:
        if hooks.claim_untrusted_report():
            print_warning(hooks.untrusted_message())
        return
    choice = ask_trust(hooks)
    if choice == "always":
        if _remember(hooks):
            print_success(f"{_noun(hooks)} trusted and remembered until they change.")
        _start_mcp(agent)
    elif choice == "session":
        hooks.trust_for_session()
        print_info(f"{_noun(hooks)} trusted for this session.")
        _start_mcp(agent)
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
    if hooks is None or not hooks.has_commands():
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
        if _remember(hooks):
            print_success(f"{_noun(hooks)} trusted and remembered until they change.")
        _start_mcp(agent)
        return
    if sub == "forget":
        from ..hook_trust import TrustStoreUnreadable, forget_trust
        hooks.revoke()
        try:
            forget_trust(hooks.config_path)
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
    if not hooks.has_mcp():
        state = ("trusted: they run" if hooks.trusted else
                 "NOT trusted: none run, and calls a pre_tool_use hook checks are refused")
    else:
        state = ("trusted: MCP servers start and hooks run" if hooks.trusted else
                 "NOT trusted: no MCP server starts and no hook runs")
    console.print(f"  [dim]{state}. /hooks trust or /hooks forget to change it.[/dim]")
