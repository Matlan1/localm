# SPDX-License-Identifier: AGPL-3.0-or-later
"""Command-line handling of project tool hooks: the trust decision at startup
and the ``/hooks`` REPL command."""

from __future__ import annotations

from localm.console import markup_safe

from ..display import console, print_info, print_success, print_warning


def _show_hooks(hooks) -> None:
    console.print(f"[bold]Tool hooks[/bold] in {markup_safe(str(hooks.config_path))}:")
    for line in hooks.describe():
        console.print(f"  {markup_safe(line)}")


def _remember(hooks) -> bool:
    """Trust *hooks* and record it. Returns True when the record was written;
    on a failure the hooks stay trusted for this session and the failure is
    reported."""
    from ..hook_trust import TrustStoreUnreadable
    try:
        hooks.remember_trust()
    except (OSError, TrustStoreUnreadable) as e:
        print_warning(f"Tool hooks are trusted for this session only: the trust "
                      f"record could not be saved ({e}).")
        return False
    return True


def ask_trust(hooks) -> str:
    """Ask whether to run *hooks*. Returns "session", "always" or "no"; an
    interrupted prompt counts as "no"."""
    _show_hooks(hooks)
    console.print("These commands run on this computer around the agent's tool "
                  "calls. Only trust hooks you have read.")
    try:
        answer = console.input(
            "[yellow]Run them? [y] this session, [a] always (until they change), "
            "[N] no (calls a pre_tool_use hook checks are then refused): "
            "[/yellow]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "no"
    if answer in ("a", "always"):
        return "always"
    if answer in ("y", "yes"):
        return "session"
    return "no"


def resolve_hook_trust(agent, *, trust_flag: bool, can_prompt: bool) -> None:
    """Settle, at startup, whether the session's project hooks may run.

    Nothing to do when there are no hooks or they are already trusted.
    ``trust_flag`` (``--trust-hooks``) trusts and records them. Otherwise an
    interactive terminal is asked; without one the hooks stay untrusted and
    that is reported."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.has_hooks() or hooks.trusted:
        return
    if trust_flag:
        if _remember(hooks):
            print_success(f"Tool hooks in {hooks.config_path} trusted.")
        return
    if not can_prompt:
        print_warning(hooks.untrusted_message())
        hooks.claim_untrusted_report()
        return
    choice = ask_trust(hooks)
    if choice == "always":
        if _remember(hooks):
            print_success("Tool hooks trusted and remembered until they change.")
    elif choice == "session":
        hooks.trust_for_session()
        print_info("Tool hooks trusted for this session.")
    else:
        hooks.claim_untrusted_report()
        print_warning("Tool hooks not trusted: none of them run, and a tool call a "
                      "pre_tool_use hook would check is refused (/hooks trust to "
                      "change that).")


def handle_hooks_command(arg: str, agent) -> None:
    """``/hooks`` shows the project's hooks and whether they run;
    ``/hooks trust`` trusts and remembers them; ``/hooks forget`` drops the
    remembered trust and stops them for this session."""
    hooks = getattr(agent, "_tool_hooks", None)
    if hooks is None or not hooks.has_hooks():
        if getattr(agent, "restricted", False):
            print_info("Tool hooks never run in a restricted session.")
        else:
            print_info("No tool hooks are configured for this project "
                       "([[hooks.pre_tool_use]] / [[hooks.post_tool_use]] in "
                       ".localcoder/config.toml).")
        return
    sub = arg.strip().lower()
    if sub == "trust":
        if _remember(hooks):
            print_success("Tool hooks trusted and remembered until they change.")
        return
    if sub == "forget":
        from ..hook_trust import TrustStoreUnreadable, forget_trust
        hooks.trusted = False
        try:
            forget_trust(hooks.config_path)
        except (OSError, TrustStoreUnreadable) as e:
            print_warning(f"Could not update the trust record ({e}); the hooks "
                          "are stopped for this session only.")
            return
        print_info("Tool hooks are no longer trusted: none of them run, and a tool "
                   "call a pre_tool_use hook would check is refused.")
        return
    if sub:
        print_info("Usage: /hooks [trust | forget]")
        return
    _show_hooks(hooks)
    state = ("trusted: they run" if hooks.trusted else
             "NOT trusted: none run, and calls a pre_tool_use hook checks are refused")
    console.print(f"  [dim]{state}. /hooks trust or /hooks forget to change it.[/dim]")
