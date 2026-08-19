# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-management routes: shutdown, restart, bug report, issues, and updater.

Extracted verbatim from create_app(); behavior unchanged. The shutdown/restart
request helpers and the client-context sanitiser live on the http_server module
and are referenced via ``_hs.``.
"""

from __future__ import annotations

import asyncio
import re

from fastapi import Depends, FastAPI, HTTPException, Request

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.errors import format_localm_error


def _watchdog_probe_host(bind_host) -> str:
    """The address the post-update health watchdog should probe: a wildcard bind
    (0.0.0.0 / :: / unset) is not itself connectable, so it is mapped to the
    loopback it covers; a concrete single-interface bind is used AS-IS, since
    that is the only address it answers on.

    Now delegates to ``bindhost.self_connect_host`` so there is one mapping
    instead of three. That matters: this function returned ``127.0.0.1`` for
    ``::`` while ``_hang_alarm._probe_host`` returned ``::1`` for the same bind,
    and only one of those survives a ``::`` bind whose dual-stack upgrade did not
    take. ``mount_gui_surface`` (http_server.py, self_url) no longer hardcodes
    127.0.0.1 either - the note this docstring used to carry about that
    divergence is resolved rather than merely described."""
    from localm.bindhost import self_connect_host
    return self_connect_host(bind_host)


# Keep a Changelog's in-progress heading. Matched case-insensitively with flexible
# spacing because it is prose in a hand-edited file, but anchored to the start of a
# line and to the "## " level so it can only ever match a section heading.
_UNRELEASED_HEADING = re.compile(r"^##[ \t]*\[unreleased\]", re.IGNORECASE)

# Keep a Changelog also carries a link-reference definition per section, at the foot of
# the file: ``[Unreleased]: https://github.com/.../compare/vX...HEAD``. With the section
# gone that line is a dangling pointer to something the reader is not being served, so
# it goes too. Anchored to the LINE START and to the ``]:`` definition form, which is
# what keeps it from matching the same words used in prose - the header sentence
# explaining the convention, and (measured on the real file) a bullet INSIDE a released
# section that refers back to a correction made in the unreleased one. Both of those
# must survive: the second is part of the permanent public record of a shipped release.
_UNRELEASED_LINKDEF = re.compile(r"^\[unreleased\]:[ \t]", re.IGNORECASE)

# bugreport.save_user_report() does local disk I/O only (read up to
# bugreport._LOG_TAIL_READ_BYTES of the current run's log, digest/scrub it,
# write the report markdown) - generous over even a slow-disk worst case.
# _BUG_REPORT_SAVE_TIMEOUT_S - the value actually passed to
# run_in_threadpool_bounded - is 2x that ceiling, not just equal to it:
# save_user_report() acquires its own _SAVE_REPORT_LOCK INSIDE the call (never
# around this await, per diff-review-discipline.md item 15), so a request's
# own clock also covers however long it waits behind another concurrent save.
# Same reasoning as media_workflows.py's _WORKFLOW_RMW_TIMEOUT_S.
_BUG_REPORT_OWN_WORK_TIMEOUT_S = 10.0
_BUG_REPORT_SAVE_TIMEOUT_S = 2 * _BUG_REPORT_OWN_WORK_TIMEOUT_S


def _strip_unreleased(markdown: str) -> str:
    """Return *markdown* with the ``[Unreleased]`` section (and its link-reference
    definition) removed, or UNCHANGED when there is no such section.

    Deliberately a LINE SCAN rather than a regex span across the whole document. The
    file is ~3600 lines and the only realistic way to break this is a pattern that
    matches past the section's end and silently eats a real release - a changelog
    missing its newest shipped version is far worse than showing one extra section.
    A line scan cannot over-match: it removes exactly from the heading up to the next
    line beginning ``## ``, and touches nothing else.

    A section that runs to end-of-file (``[Unreleased]`` last or only, the legitimate
    shape of a project with no releases yet) is removed to EOF. Serving it instead
    would be serving precisely the unreleased content this exists to withhold."""
    lines = markdown.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if _UNRELEASED_HEADING.match(ln)), None)
    if start is None:
        return markdown
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")),
               len(lines))
    kept = lines[:start] + lines[end:]
    return "".join(ln for ln in kept if not _UNRELEASED_LINKDEF.match(ln))


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    _request_shutdown = _hs._request_shutdown
    _request_restart = _hs._request_restart
    _sanitize_client_context = _hs._sanitize_client_context

    @app.post("/v1/server/shutdown",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_shutdown_ep():
        """SRV-4: stop this server cleanly (owner / config-write scope). A direct
        method to shut down so the user is not left force-closing the window
        (which segfaults) or relying on a Ctrl+C that sometimes does nothing. The
        model is unloaded before exit. (A Settings button calls this - Lane E.)"""
        _request_shutdown(instance_id=getattr(app.state, "instance_id", None))
        return {"stopping": True}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """R18: restart this server in place (owner / config-write scope). The model
        is unloaded first, then the process re-execs the same command line and comes
        back on the same port - a Settings button calls this, and the GUI's reconnect
        overlay auto-reconnects when the fresh process is up.

        "The same port" only holds if we say which one: the re-exec'd process
        otherwise re-runs pick_port() and can bind elsewhere, leaving the reconnect
        overlay waiting forever on a port nothing is listening on (the same root
        cause as REG-605's false rollback, minus the watchdog)."""
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        return {"restarting": True}

    @app.post("/api/bug-report",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def file_bug_report_ep(body: dict):
        """R47: file a bug report from the GUI. The CLI has `localm bug-report`, but
        the GUI had no manual trigger. Saves an editable markdown report (a safe
        environment snapshot plus the user's note - never keys/config/chat data;
        with ``include_log`` the home-scrubbed tail of the current run's log) and
        returns its path so the GUI can point the user at the file to edit/send.
        Owner / config-write scoped and same-origin gated like the other management
        routes (a report can carry local diagnostics)."""
        from localm import bugreport
        # The GUI "Report a bug" button sends ``description``; ``message`` is
        # accepted as an alias so the documented payload and CLI-shaped callers
        # both work. Single canonical endpoint (the GUI router does not duplicate
        # it - that would shadow this one and drop the user's text + log flag).
        description = (body.get("description") or body.get("message") or "").strip()
        # #958: what the user expected vs what actually happened are DISTINCT,
        # optional fields (the form's two new textareas) - not a duplicate of
        # description. Both are optional so an old client sending only
        # ``description`` still works exactly as before.
        what_i_expected = (body.get("what_i_expected") or "").strip()
        what_happened = (body.get("what_happened") or "").strip()
        if not description and not what_happened:
            raise HTTPException(400, "Please describe the problem before sending.")
        # Optional browser context the GUI attaches (user agent, page, viewport,
        # recent JS console errors). Untrusted client input: take only known
        # fields, coerce to strings, and cap sizes so a crafted payload cannot
        # bloat the report. It is rendered as plain text (markdown code fence),
        # never executed.
        client = _sanitize_client_context(body.get("client"))
        # Off the event loop: measured loop_lag=0.67s on this route in the
        # field - a synchronous log read + scrub + file write on the loop
        # stalls every concurrent request at exactly the moment the user is
        # already having a problem, which is why they are filing. Bounded
        # rather than a bare run_in_threadpool: see _BUG_REPORT_SAVE_TIMEOUT_S
        # above for why the budget is 2x save_user_report()'s own ceiling.
        def _save():
            return bugreport.save_user_report(
                description, what_i_expected=what_i_expected, what_happened=what_happened,
                include_log=bool(body.get("include_log")), client=client)

        try:
            path = await run_in_threadpool_bounded(
                _save, timeout=_BUG_REPORT_SAVE_TIMEOUT_S)
        except ThreadCallTimeout as e:
            raise HTTPException(504, f"Saving the bug report timed out: {e}")
        if path is None:
            # A failed save must not report success (we do not hide problems).
            raise HTTPException(500, "Could not save the bug report to disk.")
        result = {"saved": True, "filename": path.name, "path": str(path),
                  "maintainer": bugreport.MAINTAINER_EMAIL}
        # Return the saved report's markdown so the GUI can offer a browser DOWNLOAD
        # for manual sending (a tester on a phone/LAN cannot open a server-side path).
        # It is the user's own report; best-effort (a read failure just hides the
        # download button, the file is still on disk).
        try:
            result["report_markdown"] = path.read_text(encoding="utf-8")
        except OSError:
            pass
        # Optional explicit upload: file the saved report as a GitHub issue via the
        # configured proxy. Always user-initiated (never automatic). A failed upload
        # is surfaced with a diagnosed stage + message (NOT a false success) so the
        # GUI can tell the user WHERE it failed and offer retry/download - the file
        # is still saved either way (we do not hide problems).
        if body.get("upload"):
            # Same title-derivation preference as save_user_report itself
            # (what happened makes a more useful issue title than what the
            # user was doing) - kept in sync so the uploaded issue title
            # matches the report body's own H1.
            title_source = what_happened or description
            title = (title_source.splitlines()[0] if title_source else "")[:120] \
                or "user-reported issue"
            report_text = result.get("report_markdown") or path.read_text(encoding="utf-8")
            try:
                up = bugreport.upload_report(title, report_text)
                result["uploaded"] = True
                if isinstance(up, dict) and up.get("url"):
                    result["issue_url"] = up["url"]
            except bugreport.RateLimitedError as e:
                # Rate limited: hand the GUI a structured signal so it can count down
                # and auto-retry, instead of a dead-end error (the file is still saved).
                result["uploaded"] = False
                result["rate_limited"] = True
                result["retry_after"] = e.retry_after
                result["upload_stage"] = e.stage or "rate_limited"
                result["upload_message"] = e.hint or f"rate limited; retry in {e.retry_after}s"
                result["upload_error"] = f"rate limited; retry in {e.retry_after}s"
            except bugreport.LocalmError as e:
                result["uploaded"] = False
                result["upload_stage"] = e.stage or "unknown"
                result["upload_message"] = e.hint or e.summary
                result["upload_error"] = format_localm_error(e)
        return result

    @app.get("/api/issues", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def issues_ep(state: str = "all"):
        """Read-only list of the project's issues via the proxy (no GitHub account
        needed), so a tester can see whether a filed bug is acknowledged or fixed.
        A proxy failure is surfaced as ``error`` (never a fake empty success)."""
        from localm import issue_tracker
        from localm.bugreport import LocalmError
        if not issue_tracker.available():
            return {"available": False, "issues": []}
        try:
            issues = await asyncio.to_thread(issue_tracker.list_issues, state)
            return {"available": True, "issues": issues}
        except LocalmError as e:
            return {"available": True, "issues": [], "error": format_localm_error(e)}

    @app.get("/api/update/check", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_check_ep():
        """Check the proxy for a newer release. Read-only; never applies. Returns
        ``{available, current, latest, newer, notes, asset}`` or ``{available:false}``
        / ``{error}``."""
        from localm import updater
        from localm.bugreport import LocalmError
        if not updater.available():
            return {"available": False}
        try:
            info = await asyncio.to_thread(updater.check)
            info["available"] = True
            return info
        except LocalmError as e:
            return {"available": True, "error": format_localm_error(e)}

    @app.get("/api/changelog", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def changelog_ep():
        """Serve the RELEASED history from CHANGELOG.md so the Settings "Show
        changelog" button can show it (newest first) in-app, without leaving for
        GitHub. Read-only, public build content; scoped like its Updates sibling. The
        path is resolved via updater.repo_root() so it is correct in dev AND in an
        installed release. Returns {available, version, markdown}, or {available:
        false} when the file is absent from this build - an honest signal, never a
        faked empty success (we do not hide problems).

        The in-progress ``[Unreleased]`` section is REMOVED before serving. It
        describes changes that are not in the running build, so showing it tells users
        about fixes they do not have - and on a security-fix day it describes those
        fixes in detail before they ship. Stripped HERE rather than in the GUI because
        this endpoint is the single serving point: a client-side filter would leave the
        raw section reachable over the API by anyone who asks. Published prereleases
        (0.1.5rc2 and the like) are NOT stripped - they are on GitHub, so they shipped."""
        import localm
        from localm import updater
        try:
            markdown = (updater.repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
        except OSError:
            return {"available": False}
        return {"available": True, "version": localm.__version__,
                "markdown": _strip_unreleased(markdown)}

    @app.post("/api/update/apply", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_apply_ep():
        """Apply the latest update (download + swap + class step), then restart in
        place so the new code loads. EXPLICIT only (a button calls this); localm never
        self-updates on its own. A failed apply rolls back and is surfaced honestly,
        never as success. The asset is re-derived from the latest release server-side
        (not taken from the client)."""
        from localm import updater
        from localm.bugreport import LocalmError
        if not updater.available():
            raise HTTPException(400, "The updater is not configured.")
        try:
            info = await asyncio.to_thread(updater.check)
        except LocalmError as e:
            raise HTTPException(502, format_localm_error(e))
        if not info.get("newer"):
            return {"applied": False, "reason": "already up to date",
                    "current": info.get("current")}
        asset = info.get("asset") or {}
        if not asset.get("id"):
            raise HTTPException(400, "This release has no downloadable build attached.")
        try:
            res = await asyncio.to_thread(
                updater.apply, asset["id"], signature=info.get("signature"))
        except LocalmError as e:
            # apply() already rolled back; report the failure, do not fake success.
            return {"applied": False, "error": format_localm_error(e)}
        res["available"] = True
        # Restart in place so the swapped (editable) code loads - except a setup-class
        # update, which needs setup.bat re-run by the user.
        if res.get("klass") != "setup":
            # LM-DA-011: this is the ONLY restart trigger that transitions
            # automatically with no user watching (the CLI's `localm update` tells
            # the user to relaunch by hand; the plain /v1/server/restart button is
            # unrelated to updates) - so it is the one that gets a post-restart
            # health watchdog. Built from app.state (set by advertise()); a bare
            # create_app() test harness that never advertised leaves instance_port
            # unset, so watchdog stays None and the restart proceeds unwatched,
            # exactly like today.
            watchdog = None
            port = getattr(app.state, "instance_port", None)
            new_version = res.get("version")
            if port and new_version:
                watchdog = {
                    "host": _watchdog_probe_host(getattr(app.state, "bind_host", None)),
                    "port": port,
                    "scheme": getattr(app.state, "instance_scheme", None) or "http",
                    "expect_version": new_version,
                }
            else:
                from localm.debuglog import logger
                logger.warning(
                    "update applied but the instance has no bind port/version to "
                    "probe (never fully advertised); restarting WITHOUT a health "
                    "watchdog")
            # Pin the port we are actually bound to into the re-exec, so the new
            # process comes back on the SAME port the watchdog above is about to
            # probe. Without it the restart re-runs pick_port() and can bind a
            # different one (this instance may have been auto-bumped off a busy
            # default that is free again by now), the watchdog polls a port
            # nothing answers on, and a perfectly healthy update is auto-rolled
            # back after its 90s timeout (REG-605).
            _request_restart(update_watchdog=watchdog, port=port,
                             instance_id=getattr(app.state, "instance_id", None))
            res["restarting"] = True
        return res

    # -----------------------------------------------------------------------
    #  CHK-UPDATE-ROLLBACK: why these two routes exist, why they are OWNER-only,
    #  and why they carry no signature or anti-rollback check
    # -----------------------------------------------------------------------
    # `localm update --rollback` (cli/maintenance.py) had no GUI form at all, while
    # the two rollback paths that DO exist cover the opposite situation: the
    # post-apply health watchdog is a FAILURE handler (the new build did not come
    # back), and rollback.bat / rollback.sh are for a build too broken to run at
    # all. Neither covers "it applied cleanly, it runs, and it is worse" - which is
    # the only case a user can actually judge, and the case in which they are
    # sitting in the GUI having just pressed Update now, with no terminal.
    #
    # WHY NO SIGNATURE CHECK, and this is not an omission: there is nothing to
    # verify. apply() verifies a downloaded build.zip against a pinned release key
    # BEFORE extracting it (updater.verify_signature). A rollback restores a
    # DIRECTORY that this install produced from its own files at the previous
    # apply. It is not a signed artifact, it never crossed the network, and signing
    # it locally would prove nothing against an attacker who can already write it.
    # The integrity question for a local backup is filesystem access, not a
    # signature - and anyone who can rewrite <home>/updates/backup can rewrite the
    # install directly, without going through this route.
    #
    # WHY NO ANTI-ROLLBACK CHECK: updater._refuse_downgrade exists so a validly
    # SIGNED but OLDER build cannot be replayed at an install by a compromised
    # release channel. Applying it here would refuse every rollback, because a
    # rollback IS a downgrade by definition. The freshness property it protects is
    # not the property this operation has.
    #
    # WHAT THE REAL CONTROL IS: the residual risk is a principal reverting a
    # security fix by restoring the previous build. config:write is privileged
    # (scopes.PRIVILEGED_SCOPES) but it is NOT the owner, and it is what the rest of
    # the Updates card is gated on - so gating a downgrade on it alone would hand a
    # delegated key a capability it has nowhere else. These routes therefore use the
    # same owner gate as the admin_only settings in routes/config.py: open mode
    # (caller_scopes None) is the trusted local owner and passes; any key must hold
    # scopes.ADMIN. That leaves the CLI and the GUI genuinely equivalent - both
    # require the owner - rather than the GUI being the weaker door.

    @app.get("/api/update/rollback",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_rollback_info_ep():
        """Whether a rollback is possible and which build it would restore.
        Read-only: it never rolls anything back, and never creates the updates dir.
        Returns ``{available, backup, version, current}``. Scoped like its
        /api/update/check sibling; the POST below is the owner-gated half."""
        from localm import updater
        return await asyncio.to_thread(updater.rollback_info)

    @app.post("/api/update/rollback",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_rollback_ep(request: Request):
        """Restore the previous build from the last update backup, then restart in
        place so it actually loads. OWNER-only (see CHK-UPDATE-ROLLBACK above).

        THE RESTART IS NOT A CONVENIENCE, it is what makes this correct on a server.
        rollback_last() replaces the running install's own source ON DISK, including
        the localm package. `localm update --rollback` gets away with printing
        "restart to load it" because that process exits moments later; a server does
        not, and localm imports lazily throughout, so every subsequent lazy import in
        this process would load OLD code into a NEW-code process. Re-exec ends that
        window instead of leaving the user in it.

        No update watchdog on this restart, unlike /api/update/apply's: that
        watchdog's failure action IS a rollback, so arming it here would answer a
        failed rollback with another one.

        HONEST LIMIT: this restores the previous build's FILES, which is exactly what
        the CLI does. It does not undo a deps-class update's package installs, and
        the class of the last apply is not recorded anywhere, so this cannot warn
        about that specific case rather than guess at it."""
        from localm import updater
        from localm.bugreport import LocalmError
        held = _hs.caller_scopes(request)
        if held is not None and scopes.ADMIN not in held:
            raise HTTPException(
                403, "Rolling the install back to the previous build requires an "
                "owner (admin) key: it replaces the running code with an earlier "
                "version, which can put back a fixed defect.")
        # Read the target version BEFORE restoring, so the reply can name what it
        # put back. rollback_info() is a genuine read-only probe (it never calls
        # rollback_last), so this is not the "a call made just to check is still a
        # call" hazard - the whole reason that probe exists as its own function.
        target_version = (await asyncio.to_thread(updater.rollback_info)).get("version")
        try:
            res = await asyncio.to_thread(updater.rollback_last)
        except LocalmError as e:
            # A precondition, so NOTHING was touched: either there is no backup, or
            # an update/rollback already holds the single-flight lock. Both are a
            # genuine 409 conflict, and both are kept distinct from the partial-
            # restore case below, which looks similar to a caller and is the
            # opposite situation (the install HAS been modified).
            raise HTTPException(409, format_localm_error(e))
        except Exception as e:
            # _apply_update.rollback reports a PARTIAL restore by raising, listing
            # which restores failed, and deliberately keeps the backup for manual
            # recovery. The install may now be half-restored: that is the one
            # outcome that must never read as a success or as the benign "nothing to
            # roll back" above, so it is surfaced verbatim AND logged (a 500 body can
            # be lost; the log is what a bug report carries). Broad on purpose - the
            # exception class does not change the user's situation, and swallowing
            # anything here would hide a half-applied install.
            from localm.debuglog import logger
            logger.error("update rollback failed partway; the install may be "
                         "half-restored and the backup is kept: %s", e)
            raise HTTPException(
                500, f"The rollback failed partway, so the install may be "
                     f"half-restored. The backup was kept for manual recovery. "
                     f"Details: {e}")
        # Only now: the restore completed, so re-exec into it. Same port pinning as
        # /v1/server/restart, so the GUI's reconnect overlay finds the new process.
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        res["restarting"] = True
        res["version"] = target_version
        return res
