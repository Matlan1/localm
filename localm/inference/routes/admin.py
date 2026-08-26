# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-management routes: shutdown, restart, bug report, issues, and updater.

The shutdown/restart request helpers and the client-context sanitiser live on
the http_server module and are referenced via ``_hs.``.
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

    Delegates to ``bindhost.self_connect_host``, the single mapping shared with
    ``_hang_alarm._probe_host`` and ``mount_gui_surface``'s self_url, so the
    three cannot disagree about which loopback a ``::`` bind answers on."""
    from localm.bindhost import self_connect_host
    return self_connect_host(bind_host)


# Keep a Changelog's in-progress heading. Case-insensitive with flexible spacing,
# anchored to the line start and the "## " level so it matches only a section heading.
_UNRELEASED_HEADING = re.compile(r"^##[ \t]*\[unreleased\]", re.IGNORECASE)

# The matching link-reference definition at the foot of the file. Anchored to the
# line start and to the ``]:`` definition form, so the same words in prose do not
# match.
_UNRELEASED_LINKDEF = re.compile(r"^\[unreleased\]:[ \t]", re.IGNORECASE)

# Budget for save_user_report(), which does local disk I/O only: read the current
# run's log tail, digest and scrub it, write the report markdown. The value passed
# to run_in_threadpool_bounded is 2x that ceiling, because save_user_report()
# acquires its own _SAVE_REPORT_LOCK inside the call, so this clock also covers
# waiting behind another concurrent save.
_BUG_REPORT_OWN_WORK_TIMEOUT_S = 10.0
_BUG_REPORT_SAVE_TIMEOUT_S = 2 * _BUG_REPORT_OWN_WORK_TIMEOUT_S

# Separate budget for the optional upload. bugreport.upload_report defaults to a 15s
# socket timeout, which urllib applies to the connect and to each read
# independently; this is 2x that pair.
_BUG_REPORT_UPLOAD_TIMEOUT_S = 60.0


def _strip_unreleased(markdown: str) -> str:
    """Return *markdown* with the ``[Unreleased]`` section (and its link-reference
    definition) removed, or UNCHANGED when there is no such section.

    A LINE SCAN, not a regex span across the whole document: it removes exactly
    from the heading up to the next line beginning ``## `` and touches nothing else,
    so it cannot over-match past the section's end into a real release.

    A section that runs to end-of-file (``[Unreleased]`` last or only, the legitimate
    shape of a project with no releases yet) is removed to EOF."""
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
        """Stop this server cleanly (owner / config-write scope). The model is
        unloaded before exit. A Settings button calls this."""
        _request_shutdown(instance_id=getattr(app.state, "instance_id", None))
        return {"stopping": True}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """Restart this server in place (owner / config-write scope). The model
        is unloaded first, then the process re-execs the same command line and comes
        back on the same port - a Settings button calls this, and the GUI's reconnect
        overlay auto-reconnects when the fresh process is up.

        The port is pinned into the re-exec: without it the new process re-runs
        pick_port() and can bind elsewhere, leaving the reconnect overlay waiting
        on a port nothing is listening on."""
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        return {"restarting": True}

    @app.post("/api/bug-report",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def file_bug_report_ep(body: dict):
        """File a bug report from the GUI. Saves an editable markdown report (a
        safe environment snapshot plus the user's note - never keys/config/chat
        data; with ``include_log`` the home-scrubbed tail of the current run's
        log) and returns its path so the GUI can point the user at the file to
        edit/send.
        Owner / config-write scoped and same-origin gated like the other management
        routes (a report can carry local diagnostics)."""
        from localm import bugreport
        # The GUI "Report a bug" button sends ``description``; ``message`` is
        # accepted as an alias so the documented payload and CLI-shaped callers both
        # work. This is the single canonical endpoint.
        description = (body.get("description") or body.get("message") or "").strip()
        # What the user expected and what actually happened are distinct optional
        # fields, not duplicates of description. A client sending only ``description``
        # still works.
        what_i_expected = (body.get("what_i_expected") or "").strip()
        what_happened = (body.get("what_happened") or "").strip()
        if not description and not what_happened:
            raise HTTPException(400, "Please describe the problem before sending.")
        # Optional browser context the GUI attaches (user agent, page, viewport,
        # recent JS console errors). Untrusted client input: only known fields are
        # taken, coerced to strings and size-capped. Rendered as plain text in a
        # markdown code fence, never executed.
        client = _sanitize_client_context(body.get("client"))
        # Off the event loop and bounded: a synchronous log read, scrub and file write
        # on the loop stalls every concurrent request. See _BUG_REPORT_SAVE_TIMEOUT_S.
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
            # A failed save must not report success.
            raise HTTPException(500, "Could not save the bug report to disk.")
        result = {"saved": True, "filename": path.name, "path": str(path),
                  "maintainer": bugreport.MAINTAINER_EMAIL}
        # Return the saved report's markdown so the GUI can offer a browser download
        # for manual sending. Best-effort: a read failure just hides the download
        # button, the file is still on disk.
        try:
            result["report_markdown"] = path.read_text(encoding="utf-8")
        except OSError:
            pass
        # Optional explicit upload: file the saved report as a GitHub issue via the
        # configured proxy. Only ever user-initiated. A failed upload is reported with
        # a diagnosed stage and message, never as success, and the file stays saved.
        if body.get("upload"):
            # Same title-derivation preference as save_user_report, so the uploaded
            # issue title matches the report body's own H1.
            title_source = what_happened or description
            title = (title_source.splitlines()[0] if title_source else "")[:120] \
                or "user-reported issue"
            report_text = result.get("report_markdown") or path.read_text(encoding="utf-8")

            # Off the event loop: a blocking HTTPS POST to the upload proxy on
            # upload_report's own 15s socket timeout.
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
                # and auto-retry. The file is still saved.
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
                # The offload's own budget, not upload_report's. Reported in the same
                # shape as every other upload failure rather than raised: the report is
                # on disk and the GUI still offers the download and the retry.
                result["uploaded"] = False
                result["upload_stage"] = "upload"
                result["upload_message"] = (
                    "The upload did not complete in time. The report is saved - "
                    "you can retry, or download it and send it by hand.")
                # Not str(e): ThreadCallTimeout's message names the offloaded callable
                # by __qualname__, which the client has no use for. The budget and the
                # callable are already logged at WARNING by run_in_threadpool_bounded.
                result["upload_error"] = (
                    f"the upload did not finish within {_BUG_REPORT_UPLOAD_TIMEOUT_S:.0f}s")
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
        faked empty success.

        The in-progress ``[Unreleased]`` section is REMOVED before serving: it
        describes changes that are not in the running build. Stripped at this
        endpoint, the single serving point, so the raw section is not reachable over
        the API either. Published prereleases (0.1.5rc2 and the like) are NOT
        stripped."""
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
            # This is the only restart trigger that transitions automatically with no
            # user watching, so it carries a post-restart health watchdog. Built from
            # app.state (set by advertise()); with no instance_port the watchdog stays
            # None and the restart proceeds unwatched.
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
            # Pin the port we are bound to into the re-exec, so the new process comes
            # back on the same port the watchdog probes. Without it the restart re-runs
            # pick_port() and can bind a different one.
            _request_restart(update_watchdog=watchdog, port=port,
                             instance_id=getattr(app.state, "instance_id", None))
            res["restarting"] = True
        return res

    # -----------------------------------------------------------------------
    #  Update rollback: restore the previous build (owner-only)
    # -----------------------------------------------------------------------
    # Both routes are gated on the same owner check as the admin_only settings in
    # routes/config.py: open mode (caller_scopes None) is the trusted local owner
    # and passes; a key must hold scopes.ADMIN. Neither route runs a signature or
    # an anti-rollback check.

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
        place so it actually loads. OWNER-only.

        The restart is required, not a convenience: rollback_last() replaces the
        running install's own source ON DISK, including the localm package, and
        localm imports lazily throughout, so every subsequent lazy import in this
        process would load OLD code into a NEW-code process. Re-exec ends that
        window.

        No update watchdog on this restart, unlike /api/update/apply's: that
        watchdog's failure action IS a rollback, so arming it here would answer a
        failed rollback with another one.

        LIMIT: this restores the previous build's FILES, the same as the CLI. It
        does not undo a deps-class update's package installs, and the class of the
        last apply is not recorded anywhere, so it cannot warn about that case."""
        from localm import updater
        from localm.bugreport import LocalmError
        held = _hs.caller_scopes(request)
        if held is not None and scopes.ADMIN not in held:
            raise HTTPException(
                403, "Rolling the install back to the previous build requires an "
                "owner (admin) key: it replaces the running code with an earlier "
                "version, which can put back a fixed defect.")
        # Read the target version before restoring, so the reply can name what it put
        # back. rollback_info() is read-only and never calls rollback_last.
        target_version = (await asyncio.to_thread(updater.rollback_info)).get("version")
        try:
            res = await asyncio.to_thread(updater.rollback_last)
        except LocalmError as e:
        # A precondition failure, so nothing was touched: either there is no backup,
        # or an update/rollback already holds the single-flight lock. Kept distinct
        # from the partial-restore case below, where the install HAS been modified.
            raise HTTPException(409, format_localm_error(e))
        except Exception as e:
        # _apply_update.rollback reports a PARTIAL restore by raising, listing which
        # restores failed, and keeps the backup for manual recovery. The install may
        # now be half-restored, so the error is surfaced verbatim and logged. Broad
        # catch: any exception here leaves the same half-applied install.
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
