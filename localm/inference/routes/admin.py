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
        _request_shutdown()
        return {"stopping": True}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """R18: restart this server in place (owner / config-write scope). The model
        is unloaded first, then the process re-execs the same command line and comes
        back on the same port - a Settings button calls this, and the GUI's reconnect
        overlay auto-reconnects when the fresh process is up."""
        _request_restart()
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
        if not description:
            raise HTTPException(400, "Please describe the problem before sending.")
        # Optional browser context the GUI attaches (user agent, page, viewport,
        # recent JS console errors). Untrusted client input: take only known
        # fields, coerce to strings, and cap sizes so a crafted payload cannot
        # bloat the report. It is rendered as plain text (markdown code fence),
        # never executed.
        client = _sanitize_client_context(body.get("client"))
        path = bugreport.save_user_report(
            description, include_log=bool(body.get("include_log")), client=client)
        if path is None:
            # A failed save must not report success (we do not hide problems).
            raise HTTPException(500, "Could not save the bug report to disk.")
        result = {"saved": True, "filename": path.name, "path": str(path),
                  "maintainer": bugreport.MAINTAINER_EMAIL}
        # Optional explicit upload: file the saved report as a GitHub issue via the
        # configured proxy. Always user-initiated (never automatic). A failed upload
        # is surfaced as upload_error, NOT a false success - the file is still saved.
        if body.get("upload"):
            title = (description.splitlines()[0] if description else "")[:120] \
                or "user-reported issue"
            try:
                report_text = path.read_text(encoding="utf-8")
                up = bugreport.upload_report(title, report_text)
                result["uploaded"] = True
                if isinstance(up, dict) and up.get("url"):
                    result["issue_url"] = up["url"]
            except bugreport.LocalmError as e:
                result["uploaded"] = False
                result["upload_error"] = (f"{e.summary}: {e.reason}"
                                          .strip().strip(":").strip())
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
            return {"available": True, "issues": [],
                    "error": f"{e.summary}: {e.reason}".strip().strip(":").strip()}

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
            return {"available": True,
                    "error": f"{e.summary}: {e.reason}".strip().strip(":").strip()}

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
            raise HTTPException(502, f"{e.summary}: {e.reason}".strip().strip(":").strip())
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
            return {"applied": False,
                    "error": f"{e.summary}: {e.reason}".strip().strip(":").strip()}
        res["available"] = True
        # Restart in place so the swapped (editable) code loads - except a setup-class
        # update, which needs setup.bat re-run by the user.
        if res.get("klass") != "setup":
            _request_restart()
            res["restarting"] = True
        return res
