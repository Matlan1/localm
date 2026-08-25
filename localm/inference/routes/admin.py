# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-management routes: shutdown, restart, bug report, issues, and updater."""

from __future__ import annotations

import asyncio
import re

from fastapi import Depends, FastAPI, HTTPException, Request

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference._threadpool_timeout import ThreadCallTimeout, run_in_threadpool_bounded
from localm.inference.errors import format_localm_error


def _watchdog_probe_host(bind_host) -> str:
    """The address the post-update health watchdog should probe: a wildcard bind (0.0.0.0 / :: / unset) is not itself connectable, so it is mapped to the loopback it covers; a concrete single-interface bind is used AS-IS, since that is the only address it answers on."""
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

# The optional UPLOAD is a separate, slower call and needs its own budget:
# bugreport.upload_report defaults to a 15s socket timeout, which urllib applies
# to the connect and to each read independently. 2x that pair, on the same
# reasoning as the save budget above - generously past the legitimate worst case
# so it only fires for a call genuinely wedged, never for a slow proxy that is
# still working.
_BUG_REPORT_UPLOAD_TIMEOUT_S = 60.0


def _strip_unreleased(markdown: str) -> str:
    """Return *markdown* with the ``[Unreleased]`` section (and its link-reference definition) removed, or UNCHANGED when there is no such section."""
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
        """SRV-4: stop this server cleanly (owner / config-write scope)."""
        _request_shutdown(instance_id=getattr(app.state, "instance_id", None))
        return {"stopping": True}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """R18: restart this server in place (owner / config-write scope)."""
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        return {"restarting": True}

    @app.post("/api/bug-report",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def file_bug_report_ep(body: dict):
        """R47: file a bug report from the GUI."""
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

            # OFF THE EVENT LOOP, exactly like the save above and for a strictly
            # worse case than the one that earned that offload. The save was
            # moved off for a measured loop_lag of 0.67s; this is a blocking
            # HTTPS POST to the upload proxy on upload_report's own 15s timeout,
            # so an unreachable proxy froze every other client for up to 15s -
            # and the offload comment explaining why that is unacceptable sat
            # three lines above the call that did it.
            def _upload():
                return bugreport.upload_report(title, report_text)

            try:
                up = await run_in_threadpool_bounded(
                    _upload, timeout=_BUG_REPORT_UPLOAD_TIMEOUT_S)
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
            except ThreadCallTimeout:
                # The offload's own budget, not upload_report's. Reported in the
                # SAME shape as every other upload failure rather than raised:
                # the report is on disk, the GUI must still offer the download
                # and the retry, and a 500 here would throw away a saved report
                # over a send that did not go through. A failed send is never
                # reported as success (rule 5) - it is reported as failed.
                result["uploaded"] = False
                result["upload_stage"] = "upload"
                result["upload_message"] = (
                    "The upload did not complete in time. The report is saved - "
                    "you can retry, or download it and send it by hand.")
                # NOT str(e): ThreadCallTimeout's message names the offloaded
                # callable by __qualname__ ("...<locals>._upload"), which is
                # internal detail the client has no use for - the GUI only ever
                # reads this field to decide that the send FAILED, and shows
                # upload_message instead. The budget and the callable are
                # already logged at WARNING by run_in_threadpool_bounded, so
                # nothing is hidden by keeping them out of the response.
                result["upload_error"] = (
                    f"the upload did not finish within {_BUG_REPORT_UPLOAD_TIMEOUT_S:.0f}s")
        return result

    @app.get("/api/issues", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def issues_ep(state: str = "all"):
        """Read-only list of the project's issues via the proxy (no GitHub account needed), so a tester can see whether a filed bug is acknowledged or fixed."""
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
        """Check the proxy for a newer release."""
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
        """Serve the RELEASED history from CHANGELOG.md so the Settings 'Show changelog' button can show it (newest first) in-app, without leaving for GitHub."""
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
        """Apply the latest update (download + swap + class step), then restart in place so the new code loads."""
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
        """Whether a rollback is possible and which build it would restore."""
        from localm import updater
        return await asyncio.to_thread(updater.rollback_info)

    @app.post("/api/update/rollback",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_rollback_ep(request: Request):
        """Restore the previous build from the last update backup, then restart in place so it actually loads."""
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
