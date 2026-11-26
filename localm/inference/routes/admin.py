# SPDX-License-Identifier: AGPL-3.0-or-later
"""Server-management routes: shutdown, restart, bug report, issues, and updater.

The shutdown/restart request helpers and the client-context sanitiser live on
the http_server module and are referenced via ``_hs.``.
"""

from __future__ import annotations

import asyncio
import re

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

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

#: Seconds within which a second press of Update confirms the processes it ends; then
#: the update the first press prepared is discarded.
_UPDATE_CONFIRM_S = 600.0


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


async def _saved_report_response(save, show, what: str) -> dict:
    """Run *save* (blocking; returns the saved path or None) off the event loop and
    return the save routes' response: the file's name, path and text, with the
    digest the send route checks. When the file cannot be read back, its text and
    digest are left out. HTTPException 500 when nothing was saved, 504 when saving
    took too long, both naming *what* was being saved."""
    from localm import bugreport
    try:
        path = await run_in_threadpool_bounded(save, timeout=_BUG_REPORT_SAVE_TIMEOUT_S)
    except ThreadCallTimeout as e:
        raise HTTPException(504, f"Saving the {what} timed out: {e}") from e
    if path is None:
        # A failed save must not report success.
        raise HTTPException(500, f"Could not save the {what} to disk.")
    result = {"saved": True, "filename": path.name,
              "path": show(path),
              "maintainer": bugreport.MAINTAINER_EMAIL}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return result
    result["report_markdown"] = text
    result["report_sha256"] = bugreport.report_digest(text)
    return result


def register(app: FastAPI, ctx) -> None:
    require_scope = _hs.require_scope
    _request_shutdown = _hs._request_shutdown
    _request_restart = _hs._request_restart
    _sanitize_client_context = _hs._sanitize_client_context

    @app.post("/v1/server/shutdown",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_shutdown_ep():
        """Stop this server cleanly (owner / config-write scope). Loaded models
        are unloaded before the process exits. Returns ``{"stopping": true}``, with
        ``wait_s`` when a private generation is running: the seconds the stop waits
        at most until it ended."""
        from localm.media import held
        wait = await run_in_threadpool(held.private_run_wait)
        _request_shutdown(instance_id=getattr(app.state, "instance_id", None))
        return {"stopping": True, **({"wait_s": wait} if wait else {})}

    @app.post("/v1/server/restart",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def server_restart_ep():
        """Restart this server in place (owner / config-write scope). Loaded
        models are unloaded first, then the process restarts with the same command
        line and comes back on the same port. Returns ``{"restarting": true}``."""
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        return {"restarting": True}

    @app.get("/v1/server/stall-stacks",
             dependencies=[Depends(require_scope(scopes.ADMIN))])
    async def server_stall_stacks_ep():
        """This server's newest recorded stall with every thread's stack, as
        ``{"stall_stacks": text}`` (``recent_stall_stacks``; empty text when no
        such stall is recorded). For ``localm bug-report`` on the same machine:
        the owner key (or open mode), and 403 unless the server is bound to
        loopback (``app.state.bind_host``; unset counts as not loopback)."""
        if not _hs._is_loopback_host(getattr(app.state, "bind_host", None) or ""):
            raise HTTPException(
                403, "Stall stacks are served only by a server bound to this "
                     "machine (127.0.0.1 or ::1).")
        from localm.inference._stall_watch import recent_stall_stacks
        return {"stall_stacks": recent_stall_stacks()}

    @app.post("/api/bug-report",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def file_bug_report_ep(body: dict, request: Request):
        """Save a bug report from the GUI and return it for review. Never sends.

        Saves an editable markdown report (a safe environment snapshot plus the
        user's note - never keys/config/chat data; with ``include_log`` the
        scrubbed tail of the current run's log) and returns ``filename``,
        ``path``, ``report_markdown`` (the saved text) and ``report_sha256``
        (its digest). Sending is ``POST /api/bug-report/send`` with that
        filename and digest. A body carrying ``upload`` is refused with 400
        before anything is saved. For a caller other than the local operator
        the ``path`` field and the saved report text are path-scrubbed, so
        ``report_markdown`` and what a later send uploads hold no host path.
        Requires the owner or config-write scope, and a same-origin request."""
        from localm import bugreport
        if body.get("upload"):
            raise HTTPException(
                400, "A report is sent only after it has been reviewed: save it, "
                     "then send it with POST /api/bug-report/send.")
        # The GUI "Report a bug" button sends ``description``; ``message`` is
        # accepted as an alias so the documented payload and CLI-shaped callers both
        # work.
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
        show = _hs.path_formatter(request)
        # Off the event loop and bounded: a synchronous log read, scrub and file write
        # on the loop stalls every concurrent request. See _BUG_REPORT_SAVE_TIMEOUT_S.
        def _save():
            from localm.inference._stall_watch import recent_stall_stacks
            return bugreport.save_user_report(
                description, what_i_expected=what_i_expected, what_happened=what_happened,
                include_log=bool(body.get("include_log")), client=client,
                stall_stacks=recent_stall_stacks(),
                scrub=None if show is str else show)

        return await _saved_report_response(_save, show, "bug report")

    @app.post("/api/feedback",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def save_feedback_ep(body: dict, request: Request):
        """Save feedback or a suggestion from the GUI and return it for review.
        Never sends.

        *body* is ``{"kind": "feedback" | "suggestion", "text": ...}``. Saves an
        editable markdown file with that text and the localm version, nothing
        else, and returns ``filename``, ``path``, ``maintainer``,
        ``report_markdown`` and ``report_sha256`` as ``POST /api/bug-report``
        does; sending is ``POST /api/bug-report/send`` with that filename and
        digest. For a caller other than the local operator the text is
        path-scrubbed before it is saved. 400 for another kind or an empty text,
        500 when the file cannot be written, 504 when saving takes too long.
        Requires the owner or config-write scope, and a same-origin request."""
        from localm import bugreport
        kind, text = body.get("kind"), body.get("text")
        if not isinstance(kind, str) or kind not in bugreport.FEEDBACK_KINDS:
            raise HTTPException(400, 'kind must be "feedback" or "suggestion".')
        if not isinstance(text, str) or not bugreport.feedback_text(text):
            raise HTTPException(400, "Please write something before sending.")
        show = _hs.path_formatter(request)

        def _save():
            return bugreport.save_feedback(text, kind, scrub=None if show is str else show)

        return await _saved_report_response(_save, show, kind)

    @app.post("/api/bug-report/send",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def send_bug_report_ep(body: dict, request: Request):
        """Upload a report the user has reviewed, as a GitHub issue via the
        configured proxy.

        *body* names a report saved by ``POST /api/bug-report`` or ``POST
        /api/feedback``, whose kind is sent with it: ``filename``
        and ``sha256``, the ``report_sha256`` that route returned. The saved
        file is uploaded only when its current text has that digest, so what is
        sent is the text the user was shown, without its closing edit note;
        the issue title is the report's own first heading.
        ``send_anyway: true`` sends this one report while Network access is
        off, with the setting left as it is; only the owner (open mode, or an
        owner key or session) may send it, any other caller gets 403 before
        anything is read or sent. A refusal because Network access is off
        (``upload_stage`` ``network_off``) carries ``can_send_anyway``, True
        for the owner.
        400 for a missing or invalid filename or digest, 404 when the file
        cannot be read, 409 when its text no longer matches the digest. A
        failed upload returns 200 with ``uploaded`` False and a diagnosed
        ``upload_stage``/``upload_message``, never success, and the file stays
        saved."""
        from localm import bugreport
        is_owner = _hs.caller_is_operator(request)
        send_anyway = body.get("send_anyway") is True
        if send_anyway and not is_owner:
            raise HTTPException(
                403, "Only the owner can choose Send anyway for a bug report.")
        filename = body.get("filename")
        digest = body.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise HTTPException(400, "Missing the reviewed report's sha256.")
        try:
            text = bugreport.read_saved_report(filename)
        except ValueError as e:
            raise HTTPException(400, "Not a saved bug report file name.") from e
        except OSError as e:
            raise HTTPException(404, "That saved bug report could not be read.") from e
        if bugreport.report_digest(text) != digest.strip().lower():
            raise HTTPException(
                409, "The saved report changed after it was shown. Review it again "
                     "before sending.")
        title = bugreport.report_title_from_markdown(text)
        kind = bugreport.saved_report_kind(filename)
        result: dict = {"filename": filename, "maintainer": bugreport.MAINTAINER_EMAIL}

        # Off the event loop: a blocking HTTPS POST to the upload proxy on
        # upload_report's own 15s socket timeout.
        def _upload():
            return bugreport.upload_report(title, text, send_anyway=send_anyway, kind=kind)

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
            if e.stage == "network_off":
                result["can_send_anyway"] = is_owner
        except ThreadCallTimeout:
            # The offload's own budget, not upload_report's. Reported in the same
            # shape as every other upload failure rather than raised: the report is
            # on disk and the GUI still offers the download and the retry.
            result["uploaded"] = False
            result["upload_stage"] = "upload"
            result["upload_message"] = (
                "The upload did not complete in time. The report is saved - "
                "you can retry, or download it and send it by hand."
                if kind == "bug" else "The upload did not complete in time.")
            # Not str(e): ThreadCallTimeout's message names the offloaded callable
            # by __qualname__, which the client has no use for. The budget and the
            # callable are already logged at WARNING by run_in_threadpool_bounded.
            result["upload_error"] = (
                f"the upload did not finish within {_BUG_REPORT_UPLOAD_TIMEOUT_S:.0f}s")
        return result

    @app.get("/api/issues", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def issues_ep():
        """The reports this install filed, each looked up by number through
        the proxy. Rows are ``{number, title, state, closed_at, html_url}``; a row
        whose lookup failed, or that the lookup's time budget did not reach,
        carries ``error`` instead of ``state``. ``persisted`` is False when the
        record is kept in memory only (privacy mode)."""
        from localm import issue_tracker
        if not issue_tracker.available():
            return {"available": False, "issues": []}
        issues = await asyncio.to_thread(issue_tracker.filed_status)
        return {"available": True, "issues": issues,
                "persisted": issue_tracker.persists(),
                "issues_url": issue_tracker.ISSUES_URL}

    @app.get("/api/update/check", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_check_ep():
        """Check the proxy for a newer release. Read-only; never applies. Returns
        ``{available, current, latest, newer, comparable, notes, published_at, asset,
        signature}`` or ``{available:false}`` / ``{available:true, error}``."""
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
        """The released changelog of this build, newest first. Read-only.
        Returns ``{available, version, markdown}``, or ``{available: false}`` when
        the changelog file is absent from this build. The in-progress
        ``[Unreleased]`` section is removed from ``markdown``; sections of
        published prereleases are kept. Requires the config-read scope."""
        import localm
        from localm import updater
        try:
            markdown = (updater.repo_root() / "CHANGELOG.md").read_text(encoding="utf-8")
        except OSError:
            return {"available": False}
        return {"available": True, "version": localm.__version__,
                "markdown": _strip_unreleased(markdown)}

    @app.get("/api/update/releases", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_releases_ep():
        """The releases a user can choose to install. Read-only. Returns ``{available, current, releases}``, ``{available:false}``
        when the updater is not configured, or ``{available:true, error}``."""
        from localm import updater
        from localm.bugreport import LocalmError
        if not updater.available():
            return {"available": False}
        try:
            info = await asyncio.to_thread(updater.list_versions)
        except LocalmError as e:
            return {"available": True, "error": format_localm_error(e)}
        info["available"] = True
        return info

    def _probes_held(fn, *args, **kwargs):
        """Call *fn* with the GPU meter's torch probe children and runtime probe
        daemon held (``discover.torch_probes_held``, ``_loader.probe_daemon_held``):
        while an update or rollback replaces the install's files, a probe child
        would run from a half-replaced tree and hold PyTorch's libraries or the
        native runtime. Blocks: call it off the event loop."""
        from localm import discover
        from localm.inference.backends.llamacpp._loader import probe_daemon_held
        with discover.torch_probes_held(), probe_daemon_held():
            return fn(*args, **kwargs)

    def _restart_after_apply(res: dict) -> None:
        """Restart in place after a successful apply so the swapped code loads,
        except for a setup-class update, which needs the setup script
        (setup.bat or setup.sh) re-run by the user. Sets ``res["restarting"]`` when it restarts."""
        if res.get("klass") == "setup":
            return
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

    #: Held while an update is prepared through the program its build ships.
    _handing = asyncio.Lock()

    def _discard(prepared) -> None:
        """Delete what *prepared* holds; a failure is logged."""
        try:
            prepared.discard()
        except Exception as e:
            from localm.debuglog import logger as _dbg
            _dbg.warning("the prepared update to %s could not be deleted: %s",
                         prepared.new, e)

    def _expire(armed) -> None:
        """Disarm the owner's first press *armed* (``app.state.update_confirm``) once
        :data:`_UPDATE_CONFIRM_S` has passed and no press took it, and discard what it
        prepared, in a thread."""
        if getattr(app.state, "update_confirm", None) is armed:
            app.state.update_confirm = None
            asyncio.get_running_loop().run_in_executor(None, _discard, armed[3])

    async def _with_applier(request: Request, asset_id, **kwargs):
        """:func:`_hand_over` for one press at a time: 409 while another press is
        preparing an update or one was handed over, with the reply's notes, files and
        error shown to this caller as :func:`path_formatter` does."""
        if _handing.locked() or getattr(app.state, "update_handed_off", False):
            raise HTTPException(409, "An update of localm is already being started.")
        async with _handing:
            res = await _hand_over(request, asset_id, **kwargs)
        if res is not None:
            show = _hs.path_formatter(request)
            for key in ("notes", "edited"):
                res[key] = [show(v) for v in res.get(key) or ()]
            if res.get("error"):
                res["error"] = show(res["error"])
        return res

    async def _hand_over(request: Request, asset_id, **kwargs):
        """Install the build *asset_id* through the program it ships
        (``localm.updater_handoff``): prepare and check it, then restart into it. When
        other processes of localm run, an owner's first press answers ``{applied: false,
        needs_confirmation: "processes", processes, error}`` naming them and keeps the
        prepared update; the owner's second press for the same build within
        :data:`_UPDATE_CONFIRM_S` that finds the same processes is the agreement to end
        them, and uses what was prepared; once that window has passed, what the first press
        prepared is discarded (:func:`_expire`). Anyone else gets 403, and the owner's first press
        stays armed: only an owner's press uses or discards it. Returns None when the build
        ships no such program (``updater.apply`` installs it then). Raises LocalmError or
        ``updater.UnverifiedBuild``."""
        import time as _time

        from localm import updater_handoff
        target = (asset_id, kwargs.get("target_version"))
        owner = _hs.caller_is_operator(request)
        asked = getattr(app.state, "update_confirm", None) if owner else None
        if owner:
            app.state.update_confirm = None
        prepared = None
        if asked and asked[1] == target and _time.monotonic() - asked[2] < _UPDATE_CONFIRM_S:
            prepared = await asyncio.to_thread(updater_handoff.look_around, asked[3])
        elif asked:
            asked[3].discard()
        if prepared is None:
            prepared = await asyncio.to_thread(_probes_held, updater_handoff.prepare,
                                               asset_id, **kwargs)
        if prepared.legacy:
            prepared.discard()
            return None
        keys = sorted(p["key"] for p in prepared.processes)
        if keys:
            if not owner:
                prepared.discard()
                raise HTTPException(403, "Ending the other processes of localm for an update "
                                         "requires an owner (admin) key.")
            if not (asked and asked[3] is prepared and asked[0] == keys):
                armed = (keys, target, _time.monotonic(), prepared)
                app.state.update_confirm = armed
                asyncio.get_running_loop().call_later(_UPDATE_CONFIRM_S, _expire, armed)
                return {"applied": False, "needs_confirmation": "processes",
                        "processes": prepared.processes,
                        "error": f"The update to {prepared.new} restarts or ends these "
                                 "processes of localm: "
                                 f"{updater_handoff.describe(prepared.processes)}. A server "
                                 "restarts into the new version by itself; a chat or a command "
                                 "ends; an MCP server is started again by its client. The "
                                 "same request again restarts or ends them and updates."}
        port = getattr(app.state, "instance_port", None)
        prepared.watch = {"host": _watchdog_probe_host(getattr(app.state, "bind_host", None)),
                          "port": port,
                          "scheme": getattr(app.state, "instance_scheme", None) or "http"}
        app.state.update_handed_off = True
        kept = getattr(app.state, "update_confirm", None)
        if kept is not None and kept[3] is not prepared:
            app.state.update_confirm = None
            kept[3].discard()
        _request_restart(port=port, instance_id=getattr(app.state, "instance_id", None),
                         update=prepared)
        return {"applied": True, "version": prepared.new, "restarting": True,
                "handed_off": True, "verified": prepared.verified,
                "verify_reason": prepared.verify_reason, "downgrade": prepared.downgrade,
                "edited": prepared.edited, "notes": prepared.notes, "available": True}

    async def _apply_selected(request: Request, body: dict):
        """Install the release ``body["version"]``, OWNER only. Its asset and
        signature come from the proxy, never from the client. A build whose
        signature does not verify answers ``{applied: false, needs_confirmation:
        "unverified", version, reason, sha256}`` without installing anything; the
        client installs it by repeating the request with ``allow_unverified: true``
        and that ``sha256``. Only an owner reaches the install, so its reply is
        never path-scrubbed."""
        from localm import updater
        from localm.bugreport import LocalmError
        if not _hs.caller_is_operator(request):
            raise HTTPException(
                403, "Installing a chosen version requires an owner (admin) key: it can "
                "replace the running code with an earlier version.")
        version = body.get("version")
        allow_unverified = body.get("allow_unverified") is True
        sha256 = body.get("sha256")
        if not isinstance(version, str) or not version.strip() or len(version) > 64:
            raise HTTPException(400, "version must be a release version string.")
        if sha256 is not None and not (
                isinstance(sha256, str) and len(sha256) == 64
                and all(c in "0123456789abcdefABCDEF" for c in sha256)):
            raise HTTPException(400, "sha256 must be a 64-character hex string.")
        if allow_unverified and sha256 is None:
            raise HTTPException(400, "allow_unverified needs the sha256 of the build "
                                     "you confirmed.")
        try:
            rel = await asyncio.to_thread(updater.fetch_release, version.strip())
        except LocalmError as e:
            raise HTTPException(502, format_localm_error(e)) from e
        asset = rel.get("asset") or {}
        if not asset.get("id"):
            raise HTTPException(400, "This release has no downloadable build attached.")
        try:
            handed = await _with_applier(request, asset["id"], signature=rel.get("signature"),
                                         target_version=rel["version"],
                                         allow_unverified=allow_unverified,
                                         expect_sha256=sha256)
            if handed is not None:
                return handed
            res = await asyncio.to_thread(
                _probes_held, updater.apply, asset["id"], signature=rel.get("signature"),
                target_version=rel["version"], allow_unverified=allow_unverified,
                expect_sha256=sha256, defer_runtime=True)
        except updater.UnverifiedBuild as e:
            return {"applied": False, "needs_confirmation": "unverified",
                    "version": rel["version"], "reason": e.reason, "sha256": e.sha256}
        except LocalmError as e:
            return {"applied": False, "error": format_localm_error(e)}
        res["available"] = True
        _restart_after_apply(res)
        return res

    @app.post("/api/update/apply", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_apply_ep(request: Request):
        """Apply the latest update (download, swap, class step), then restart in
        place so the new code loads. Only an explicit request applies an update;
        localm never updates itself. A failed apply is rolled back and reported as
        an error, never as success. The build to install is chosen by the server
        from the latest release, not taken from the request. The ``backup``,
        ``runtime_note``, ``components_note`` and ``error`` fields are path-scrubbed
        for a caller other than the local operator. A native runtime re-provision
        the update needs runs during the restart, after the models are unloaded.

        A JSON body ``{"version": V}`` installs release V instead, older ones
        included. That form needs an owner (admin) key (403 otherwise) and answers
        400 for an invalid ``version`` or ``sha256``. A build whose signature does
        not verify answers ``{applied: false, needs_confirmation: "unverified",
        version, reason, sha256}`` without installing anything; repeating the
        request with ``allow_unverified: true`` and that ``sha256`` installs it.
        The reply of that form is never path-scrubbed."""
        import json as _json

        from localm import updater
        from localm.bugreport import LocalmError
        if not updater.available():
            raise HTTPException(400, "The updater is not configured.")
        raw = await request.body()
        if raw.strip():
            try:
                body = _json.loads(raw)
            except (ValueError, RecursionError):
                raise HTTPException(400, "The request body is not valid JSON.") from None
            if not isinstance(body, dict):
                raise HTTPException(400, "The request body must be a JSON object.")
            if "version" in body:
                return await _apply_selected(request, body)
        try:
            info = await asyncio.to_thread(updater.check)
        except LocalmError as e:
            raise HTTPException(502, format_localm_error(e)) from e
        if not info.get("newer"):
            if info.get("comparable", True) is False:
                return {"applied": False,
                        "reason": "could not tell whether the latest release is newer "
                                  "(unrecognized version format)",
                        "current": info.get("current"), "latest": info.get("latest")}
            return {"applied": False, "reason": "already up to date",
                    "current": info.get("current")}
        asset = info.get("asset") or {}
        if not asset.get("id"):
            raise HTTPException(400, "This release has no downloadable build attached.")
        try:
            handed = await _with_applier(request, asset["id"], signature=info.get("signature"))
            if handed is not None:
                return handed
            res = await asyncio.to_thread(
                _probes_held, updater.apply, asset["id"], signature=info.get("signature"),
                defer_runtime=True)
        except LocalmError as e:
            # The error says whether apply() changed nothing, rolled back, or needs a manual restore.
            return {"applied": False,
                    "error": _hs.path_formatter(request)(format_localm_error(e))}
        for key in ("backup", "runtime_note", "components_note"):
            if res.get(key) is not None:
                res[key] = _hs.path_formatter(request)(res[key])
        res["available"] = True
        _restart_after_apply(res)
        return res

    # -----------------------------------------------------------------------
    #  Installed parts: which version each part is at, and the repair
    # -----------------------------------------------------------------------

    @app.get("/api/update/parts", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_parts_ep(request: Request):
        """Which version each part of the install is at against the installed
        localm version: ``{app,
        install_record, install_record_matches, deps_problems, runtime,
        components, problems, consistent, last_steps}``. Read-only.
        ``problems`` and each ``last_steps`` note are path-scrubbed for a caller
        other than the local operator."""
        from localm import updater_parts
        parts = await asyncio.to_thread(updater_parts.installed_parts)
        fmt = _hs.path_formatter(request)
        parts["problems"] = [fmt(p) for p in parts["problems"]]
        for step in parts["last_steps"]:
            step["note"] = fmt(step["note"])
        return parts

    @app.post("/api/update/repair", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_repair_ep(request: Request):
        """Bring every part of the install to the installed localm version:
        the dependencies now, the native runtime and
        the components during the restart that follows. OWNER-only. Returns
        ``{repaired, notes, restarting}``, or ``{repaired: false, error}`` when a
        step failed (the dependencies are put back as they were)."""
        from localm import updater_parts
        from localm.bugreport import LocalmError
        if not _hs.caller_is_operator(request):
            raise HTTPException(
                403, "Repairing the install requires an owner (admin) key: it reinstalls "
                "localm's packages and native runtime.")
        try:
            res = await asyncio.to_thread(_probes_held, updater_parts.repair, defer=True)
        except LocalmError as e:
            return {"repaired": False, "error": format_localm_error(e)}
        out = {"repaired": True, "notes": res["notes"], "restarting": False}
        if res["restart_needed"]:
            _request_restart(port=getattr(app.state, "instance_port", None),
                             instance_id=getattr(app.state, "instance_id", None))
            out["restarting"] = True
        return out

    # -----------------------------------------------------------------------
    #  Integrity: localm's files against the installed release
    # -----------------------------------------------------------------------

    def _integrity_paths(body):
        paths = body.get("paths") if isinstance(body, dict) else None
        if paths is None:
            return None
        if not (isinstance(paths, list) and len(paths) <= 10000
                and all(isinstance(p, str) and 0 < len(p) <= 1024 for p in paths)):
            raise HTTPException(400, "paths must be a list of file paths.")
        return paths

    async def _integrity_body(request: Request) -> dict:
        import json as _json
        raw = await request.body()
        if not raw.strip():
            return {}
        try:
            body = _json.loads(raw)
        except (ValueError, RecursionError):
            raise HTTPException(400, "The request body is not valid JSON.") from None
        if not isinstance(body, dict):
            raise HTTPException(400, "The request body must be a JSON object.")
        return body

    def _integrity_owner(request: Request, what: str) -> None:
        if not _hs.caller_is_operator(request):
            raise HTTPException(403, f"{what} requires an owner (admin) key: it changes "
                                "which of localm's files count as the release's.")

    @app.get("/api/integrity", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def integrity_ep(request: Request):
        """The integrity check's settings and the last result in this server:
        ``{at_start, notice, pending, running, result}`` (``integrity.state``).
        Read-only; ``result`` is None until a check ran."""
        from localm import integrity
        out = await asyncio.to_thread(integrity.state)
        if out["result"] is not None and out["result"].get("reason"):
            out["result"] = {**out["result"],
                             "reason": _hs.path_formatter(request)(out["result"]["reason"])}
        return out

    @app.post("/api/integrity/check",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def integrity_check_ep(request: Request):
        """Run the integrity check now; the JSON body ``{full}`` reads every file in
        full. Returns the result (``integrity.check``)."""
        from localm import integrity
        body = await _integrity_body(request)
        res = await asyncio.to_thread(integrity.run, full=body.get("full") is True)
        if res.get("reason"):
            res = {**res, "reason": _hs.path_formatter(request)(res["reason"])}
        return res

    @app.post("/api/integrity/keep",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def integrity_keep_ep(request: Request):
        """Keep the differing files ``{paths}`` (default every differing one) as the
        user's own. OWNER-only. Returns ``{kept, result}``."""
        from localm import integrity
        _integrity_owner(request, "Keeping changed files")
        paths = _integrity_paths(await _integrity_body(request))
        if paths is None:
            found = await asyncio.to_thread(integrity.check)
            paths = found["differ"] + found["missing"]
        why = await asyncio.to_thread(integrity.refusal, paths)
        if why:
            raise HTTPException(400, _hs.path_formatter(request)(why))
        try:
            done = await asyncio.to_thread(integrity.keep, paths)
        except (integrity.IntegrityError, OSError) as e:
            from localm.debuglog import logger
            logger.warning("integrity: keeping %d file(s) failed: %s", len(paths), e)
            raise HTTPException(500, "Your changes could not be recorded; the log says "
                                "why.") from None
        return {"kept": done, "result": await asyncio.to_thread(integrity.run)}

    @app.post("/api/integrity/repair",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def integrity_repair_ep(request: Request):
        """Write the differing files ``{paths}`` (default every differing one) again
        from the signed build of the installed version; without ``paths``, also install
        the native runtime again when files of it differ
        (``setup_llama.repair.repair_runtime``).
        OWNER-only. Returns ``{repaired: true, files, failed, runtime, result}``
        (``runtime`` the runtime repair's result, its ``error`` and ``fix``
        path-scrubbed, or None with ``paths`` given) or ``{repaired: false, error}``, with
        ``runtime`` too when the release files could not be repaired."""
        from localm import integrity
        from localm.bugreport import LocalmError
        _integrity_owner(request, "Repairing files")
        paths = _integrity_paths(await _integrity_body(request))
        fmt = _hs.path_formatter(request)
        why = await asyncio.to_thread(integrity.refusal, paths, repairing=True)
        if why:
            return {"repaired": False, "error": fmt(why)}
        async def runtime_repair():
            if paths is not None:
                return None
            from localm.setup_llama.repair import repair_runtime
            rt = await asyncio.to_thread(_probes_held, repair_runtime)
            return {**rt, **{k: fmt(rt[k]) for k in ("error", "fix", "note")
                             if rt[k] is not None}}

        try:
            res = await asyncio.to_thread(integrity.repair, paths)
        except integrity.IntegrityError as e:
            from localm.debuglog import logger
            logger.warning("integrity: the repair was refused: %s", e)
            return {"repaired": False, "error": "Nothing was repaired; the log says why.",
                    "runtime": await runtime_repair()}
        except LocalmError as e:
            return {"repaired": False, "error": fmt(format_localm_error(e)),
                    "runtime": await runtime_repair()}
        failed = [{"path": f["path"], "reason": fmt(f["reason"])} for f in res["failed"]]
        return {"repaired": True, "files": res["repaired"], "failed": failed,
                "runtime": await runtime_repair(),
                "result": await asyncio.to_thread(integrity.run)}

    # -----------------------------------------------------------------------
    #  Update rollback: restore the previous build (owner-only)
    # -----------------------------------------------------------------------
    # Both routes are gated on the same owner check as the admin_only settings in
    # routes/config.py: open mode is the trusted local owner
    # and passes; a key must hold scopes.ADMIN. Neither route runs a signature or
    # an anti-rollback check.

    @app.get("/api/update/rollback",
             dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def update_rollback_info_ep(request: Request):
        """Whether a rollback is possible and which build it would restore.
        Read-only: it never rolls anything back, and never creates the updates dir.
        Returns ``{available, backup, version, current}``; ``backup`` is
        path-scrubbed for a caller other than the local operator. Requires the
        config-read scope; performing the rollback is ``POST /api/update/rollback``."""
        from localm import updater
        info = await asyncio.to_thread(updater.rollback_info)
        if info.get("backup") is not None:
            info["backup"] = _hs.path_formatter(request)(info["backup"])
        return info

    @app.post("/api/update/rollback",
              dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def update_rollback_ep(request: Request):
        """Restore the previous build from the last update backup, then restart in
        place so the restored code loads. Without a backup, after an update the program of
        its build carried out, the version it replaced is installed again as a chosen
        release (``updater.rollback_info``'s ``reinstall``), answered as ``/api/update/apply``
        with a version answers (``needs_confirmation`` "processes" or "unverified"); the
        optional JSON body ``{allow_unverified, sha256}`` of that form confirms an
        unverified build. A reply that restored or installed it says ``applied: true``.
        Needs an owner (admin) key (403
        otherwise). Restores every part the update replaced, the same as the CLI:
        the files, the virtual environment as it was before the update's
        dependency step, the native runtime and the components. The reply,
        including the ``backup`` path and error texts, is never path-scrubbed.
        Answers 409 when there is no usable backup or an update or rollback is
        already running, and 500 when the restore failed partway (the backup is
        kept)."""
        from localm import updater
        from localm.bugreport import LocalmError
        if not _hs.caller_is_operator(request):
            raise HTTPException(
                403, "Rolling the install back to the previous build requires an "
                "owner (admin) key: it replaces the running code with an earlier "
                "version, which can put back a fixed defect.")
        # Read the target version before restoring, so the reply can name what it put
        # back. rollback_info() is read-only and never calls rollback_last.
        back = await asyncio.to_thread(updater.rollback_info)
        if back.get("reinstall"):
            import json as _json
            raw = await request.body()
            try:
                body = _json.loads(raw) if raw.strip() else {}
            except (ValueError, RecursionError):
                raise HTTPException(400, "The request body is not valid JSON.") from None
            if not isinstance(body, dict):
                raise HTTPException(400, "The request body must be a JSON object.")
            res = await _apply_selected(request, {
                "version": back["version"],
                **{k: body[k] for k in ("allow_unverified", "sha256") if k in body}})
            if not res.get("applied") and not res.get("needs_confirmation"):
                raise HTTPException(500, res.get("error") or res.get("reason")
                                    or f"{back['version']} could not be installed")
            return res
        target_version = back.get("version")
        try:
            res = await asyncio.to_thread(_probes_held, updater.rollback_last)
        except LocalmError as e:
            if e.stage == updater.ROLLBACK_INCOMPLETE:
                from localm.debuglog import logger
                logger.error("update rollback failed partway; the install may be "
                             "half-restored and the backup is kept: %s: %s",
                             e.summary, e.reason)
                raise HTTPException(
                    500, f"The rollback failed partway, so the install may be "
                         f"half-restored. The backup was kept for manual recovery. "
                         f"Details: {format_localm_error(e)}") from e
        # A precondition failure, so nothing was touched: there is no backup, the backup
        # is incomplete, or an update/rollback already holds the single-flight lock. Kept distinct
        # from the partial-restore case below, where the install HAS been modified.
            raise HTTPException(409, format_localm_error(e)) from e
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
                     f"Details: {e}") from e
        # Only now: the restore completed, so re-exec into it. Same port pinning as
        # /v1/server/restart, so the GUI's reconnect overlay finds the new process.
        _request_restart(port=getattr(app.state, "instance_port", None),
                         instance_id=getattr(app.state, "instance_id", None))
        res["restarting"] = True
        res["applied"] = True
        res["version"] = target_version
        return res
