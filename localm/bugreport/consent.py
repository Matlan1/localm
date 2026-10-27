# SPDX-License-Identifier: AGPL-3.0-or-later
"""Telling the user what failed, saving the report, and offering to send it.
A report is uploaded only on the user's menu choice or an explicit send flag.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

from localm.bugreport._common import console, MAINTAINER_EMAIL
from localm.console import markup_safe
from localm.bugreport.errors import LocalmError, RateLimitedError
from localm.bugreport.assembly import build_report
from localm.bugreport.transport import mailto_url
import localm.bugreport as _br


def report_failure(*, summary: str, reason: str = "",
                   error: Optional[BaseException] = None,
                   context: Optional[dict] = None,
                   interactive: bool = True,
                   assume_yes: bool = False,
                   auto_send: bool = False,
                   as_failure: bool = True,
                   open_browser=webbrowser.open,
                   prompt=None) -> Optional[Path]:
    """Say sorry (what + why), save an editable report, and offer to send it.

    Returns the saved report path (or None). ``interactive`` False (or
    ``assume_yes``) saves and points at the file without prompting or opening a
    browser - the right behaviour for unattended / scripted runs. ``as_failure``
    False is for a user-initiated report (``localm bug-report``): neutral header,
    no apology. ``auto_send`` True uploads immediately via the account-less proxy
    channel (no menu, no browser) - the caller's own explicit consent (e.g. the
    ``--send`` flag) stands in for picking option [1] interactively, so this is
    still an explicit user action, not automatic reporting. ``open_browser`` and
    ``prompt`` are injectable for tests.

    When automatic diagnostics are not allowed (``_diagnostics_allowed()``
    False: privacy mode without ``keep_diagnostics``) and ``auto_send`` is
    False, no report is built or saved unless an interactive user answers yes
    to a save prompt; otherwise this prints how to file one and returns None."""
    console.print()
    if as_failure:
        console.print(f"[bold red]Sorry - {markup_safe(summary)}.[/bold red]")
        if reason:
            console.print(f"[red]Reason:[/red] {markup_safe(reason)}")
    else:
        console.print(f"[bold]Filing a bug report:[/bold] {markup_safe(summary)}")
        if reason:
            console.print(f"[dim]{markup_safe(reason)}[/dim]")

    if not auto_send and not _br._diagnostics_allowed():
        if not _consents_to_save(interactive=interactive, assume_yes=assume_yes,
                                 prompt=prompt):
            return None

    text = build_report(summary, reason=reason, error=error, context=context)
    path = _br.save_report(text)
    if path is not None:
        console.print(f"[dim]A bug report was saved (edit it before sending):[/dim] {markup_safe(path)}",
                      soft_wrap=True)
    else:
        console.print("[yellow]Could not save a report file; you can still copy the "
                      "details above.[/yellow]")

    offer_to_send(summary, path, text, interactive=interactive,
                   assume_yes=assume_yes, auto_send=auto_send,
                   open_browser=open_browser, prompt=prompt)
    return path


def _consents_to_save(*, interactive: bool, assume_yes: bool, prompt) -> bool:
    """Whether to save an automatic report while diagnostics are off (privacy
    mode without ``keep_diagnostics``). True only on an explicit yes at an
    interactive prompt; otherwise prints that nothing was saved and returns
    False. Nothing is built, read or written before this returns True."""
    if interactive and not assume_yes:
        ask = prompt
        if ask is None:
            import click

            def ask(text_):  # noqa: E306
                return click.prompt(text_, default="", show_default=False)
        console.print("[dim]Privacy mode does not save bug reports on its own.[/dim]")
        try:
            answer = (ask("  Save a bug report for this error? [y/N]") or "")
        except Exception:
            # An unanswerable prompt is a no.
            # See test_privacy_mode_save_prompt_that_cannot_be_answered_saves_nothing.
            answer = ""
        if answer.strip().lower() in ("y", "yes"):
            return True
    console.print("[dim]Privacy mode: no bug report was saved. Run[/dim] "
                  "[bold]localm bug-report[/bold] [dim]to file one, or turn on "
                  "\"Keep diagnostics for bug reports\" to keep these automatically.[/dim]")
    return False


def offer_to_send(summary: str, path: Optional[Path], text: str, *,
                   interactive: bool = True, assume_yes: bool = False,
                   auto_send: bool = False,
                   open_browser=webbrowser.open, prompt=None) -> None:
    """Offer to send an ALREADY-BUILT, already-saved report (upload / email /
    self), or send it immediately with ``auto_send``.

    Split out of report_failure so it is the ONE send flow shared by every
    producer - report_failure (automatic crash/LocalmError reports, built via
    build_report) and the ``localm bug-report`` CLI (user-composed reports,
    built via save_user_report) - rather than two copies that can drift on
    retry/rate-limit/failure handling. *text* is the in-memory report
    body (used when *path* is None because the save itself failed); when a
    file exists, the actually-sent body is RE-READ from it so a user's edits
    made before picking a channel are what gets sent, not the stale
    in-memory copy."""
    up_url, up_token = _br.upload_config()
    can_upload = up_url is not None

    if auto_send:
        _auto_send(summary, path, text, can_upload, up_url, up_token)
        return

    if not interactive or assume_yes:
        if can_upload:
            console.print(
                "[dim]Run[/dim] [bold]localm bug-report --send[/bold] [dim]to send it now "
                "(no GitHub account needed) - or email/Discord it yourself.[/dim]")
        else:
            console.print(f"[dim]Send it to the maintainer ({MAINTAINER_EMAIL}) by email or "
                          "Discord.[/dim]")
        return

    # Re-read the saved file so the user's edits (made before picking a channel)
    # are what actually gets sent - otherwise "edit it first" would be a lie.
    body = text
    if path is not None:
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            body = text

    console.print("How would you like to send it (edit the file first if you want)?")
    # Map the displayed number -> action, so an extra "send now" option when an
    # upload endpoint is configured does not renumber the always-present channels
    # (email is [2], self [3] regardless of the upload option).
    actions = {}
    if can_upload:
        console.print("  [1] Send to the maintainer now  "
                      "[dim](uploads the report - no GitHub account needed)[/dim]")
        actions["1"] = "upload"
    console.print("  [2] Email the maintainer  [dim](opens your mail app, works anywhere)[/dim]")
    console.print(f"  [3] I'll send it myself   [dim](Discord / email to {MAINTAINER_EMAIL})[/dim]")
    console.print("  [Enter] not now")
    actions.update({"2": "email", "3": "self"})

    ask = prompt
    if ask is None:
        import click
        def ask(text_):  # noqa: E306
            return click.prompt(text_, default="", show_default=False)
    choice = (ask("  Pick a number") or "").strip()
    action = actions.get(choice, "none")

    # When the file save failed (path is None) the channels still work off the
    # in-memory text, but messages must not claim a file exists - say where the
    # report actually is, honestly ("we do not hide problems").
    where = (str(path) if path is not None
             else "the text above (it could not be saved to a file)")

    try:
        if action == "upload":
            _upload_with_retries(summary, body, up_url, up_token, ask, where)
        elif action == "email":
            open_browser(mailto_url(summary, body))
            console.print(f"[green]Opened your mail app to {MAINTAINER_EMAIL}.[/green]")
        elif action == "self":
            console.print(f"[dim]Thanks. Send {markup_safe(where)} to {MAINTAINER_EMAIL} or on "
                          "Discord when you can.[/dim]", soft_wrap=True)
        else:
            console.print("[dim]No report sent. It is saved if you change your mind.[/dim]")
    except Exception:
        console.print("[yellow]Could not open that automatically - the report is at "
                      f"{markup_safe(where)}.[/yellow]", soft_wrap=True)


def _auto_send(summary: str, path: Optional[Path], text: str,
               can_upload: bool, up_url, up_token) -> None:
    """Send the report at *path* (*text* when there is no file) straight away,
    the ``auto_send`` branch of offer_to_send. Nothing is sent when no endpoint
    is configured; a rate limit is waited out and retried once; every failure
    says where the report is."""
    body = path.read_text(encoding="utf-8") if path is not None else text
    where = str(path) if path is not None else "the text above"
    if not can_upload:
        console.print("[yellow]No hosted send channel is configured - the report is "
                      f"saved at {markup_safe(where)}; email it to {MAINTAINER_EMAIL} "
                      "instead.[/yellow]", soft_wrap=True)
        return
    try:
        res = _br.upload_report(summary, body, url=up_url, token=up_token)
    except RateLimitedError as e:
        import time as _time
        console.print(f"[yellow]Rate limited. Retrying in {e.retry_after}s...[/yellow]")
        _time.sleep(e.retry_after)
        try:
            res = _br.upload_report(summary, body, url=up_url, token=up_token)
        except LocalmError as e2:
            console.print(f"[yellow]Still could not send it.[/yellow] "
                          f"{e2.hint or e2.reason}")
            console.print(f"[dim]The report is at {where} - email it to "
                          f"{MAINTAINER_EMAIL} instead.[/dim]")
            return
    except LocalmError as e:
        console.print(f"[yellow]Could not send it.[/yellow] {markup_safe(e.hint or e.reason)}")
        console.print(f"[dim]The report is at {where} - email it to "
                      f"{MAINTAINER_EMAIL} instead.[/dim]")
        return
    _print_sent(res)
    return


def _print_sent(res) -> None:
    """Confirm an upload, with the tracking link the proxy returned when there is one."""
    link = res.get("url") if isinstance(res, dict) else None
    console.print("[green]Sent to the maintainer.[/green]"
                  + (f" Tracking issue: {markup_safe(link)}" if link else ""))


def _upload_with_retries(summary: str, body: str, up_url, up_token, ask,
                         where: str) -> None:
    """Upload *body* for the menu's send choice. A rate limit is waited out and
    retried once; any other failure says where the report is and is retried
    only when the user answers yes, for at most three attempts in all."""
    attempt = 0
    while True:
        attempt += 1
        try:
            res = _br.upload_report(summary, body, url=up_url, token=up_token)
            _print_sent(res)
            break
        except RateLimitedError as e:
            # Rate limited: wait the server-advised delay and retry ONCE
            # automatically rather than making the tester re-run the command.
            import time as _time
            msg = e.hint or "The bug-report server is rate limiting."
            console.print(f"[yellow]{msg} Retrying in {e.retry_after}s...[/yellow]")
            _time.sleep(e.retry_after)
            if attempt >= 2:
                console.print(f"[yellow]Still rate limited.[/yellow] The report "
                              f"is at {where} - email it to {MAINTAINER_EMAIL}.")
                break
            continue
        except LocalmError as e:
            # A failed send must never look like success: say WHERE it
            # failed, keep the file, and offer to retry.
            console.print(f"[yellow]Could not send it.[/yellow] {e.hint or e.reason}")
            console.print(f"[dim]The report is saved at {where}.[/dim]")
            if attempt >= 3:
                console.print(f"[dim]Email it to {MAINTAINER_EMAIL} when you can.[/dim]")
                break
            again = (ask("  Retry sending now? [y/N]") or "").strip().lower()
            if again in ("y", "yes"):
                continue
            console.print(f"[dim]Not retried - email {where} to "
                          f"{MAINTAINER_EMAIL} when you can.[/dim]")
            break
