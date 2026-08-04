# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-management routes: shutdown, restart, bug report, issues, and updater.

Extracted verbatim from create_app(); behavior unchanged. The shutdown/restart
request helpers and the client-context sanitiser live on the http_server module
and are referenced via ``_hs.``.
"""

from __future__ import annotations

import asyncio

from fastapi import Depends, FastAPI, HTTPException

import localm.inference.http_server as _hs
from localm import scopes
from localm.inference.errors import format_localm_error


def _watchdog_probe_host(bind_host) -> str:
    """The address the post-update health watchdog should probe: a wildcard bind
    (0.0.0.0 / :: / unset) is not itself connectable, mapped to loopback exactly
    like mount_gui_surface's own self-connect URL (http_server.py, self_url); a
    concrete single-interface bind is used AS-IS - unlike mount_gui_surface, which
    always hardcodes 127.0.0.1, that would be wrong here if the server is bound
    ONLY to a non-loopback interface (loopback would then be unreachable)."""
    h = (bind_host or "").strip()
    return "127.0.0.1" if h in ("", "0.0.0.0", "::") else h


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
        path = bugreport.save_user_report(
            description, what_i_expected=what_i_expected, what_happened=what_happened,
            include_log=bool(body.get("include_log")), client=client)
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
        """Serve the release CHANGELOG.md so the Settings "Show changelog" button can
        show the full version history (newest first) in-app, without leaving for
        GitHub. Read-only, public build content; scoped like its Updates sibling. The
        path is resolved via updater.repo_root() so it is correct in dev AND in an
        installed release. Returns {available, version, markdown}, or {available:
        false} when the file is absent from this build - an honest signal, never a
        faked empty success (we do not hide problems)."""
        import localm
        from localm import updater
        try:
            markdown = (updater.repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
        except OSError:
            return {"available": False}
        return {"available": True, "version": localm.__version__, "markdown": markdown}

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
