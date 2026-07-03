# SPDX-License-Identifier: AGPL-3.0-or-later
"""
GUI web layer - API routes and static file serving, attached to the
existing localm FastAPI inference app.

Routes (all under /api, bearer-protected when LOCALM_API_KEY is set):
  GET    /api/models                       registry + active model
  POST   /api/models/load                  switch the active engine

Coder routes (/api/coder/*) live in the builtin "coder" plugin
(localm/plugins/builtin/coder); attach_gui only publishes the shared
services they read via request.app.state.

The static frontend is mounted at / and must be attached AFTER all API
routes so it doesn't shadow them.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from localm.plugins.coder.sessions import SessionManager

STATIC_DIR = Path(__file__).parent / "static"

# SSE keepalive interval - must beat proxy/browser idle timeouts
_KEEPALIVE_S = 15


class _RevalidatingStatic(StaticFiles):
    """Serve the GUI assets with ``Cache-Control: no-cache`` so the browser
    REVALIDATES every load instead of silently serving a stale copy.

    SEAMLESS UPDATES: Starlette's StaticFiles sends an ``ETag`` but no
    ``Cache-Control``, so browsers fall back to HEURISTIC caching and can keep an
    old ``app.js`` / ``sw.js`` / ``style.css`` for a while - the user (or a tester)
    then has to clear the cache by hand to pick up new code. ``no-cache`` does NOT
    disable caching: the browser still caches and revalidates with the ETag, so an
    unchanged file is a cheap ``304`` and a changed one is fetched fresh. It also
    lets a phone's ``serviceWorker.register(...).update()`` actually see a new
    ``sw.js`` (the browser HTTP-caching sw.js is a known update-stickiness cause).
    The server, not the user, is now responsible for delivering current code."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers.setdefault("Cache-Control", "no-cache")
        return resp


def _is_loopback_host(host: str) -> bool:
    """True for a loopback bind/client host (127.0.0.0/8, ::1, localhost)."""
    if not host:
        return False
    if host == "localhost":
        return True
    import ipaddress
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _index_html_with_shell_token(token: str) -> str:
    """The SPA shell, optionally seeding the per-process open-mode *token* (the
    shell token) as a JS global so a loopback launch can still perform management
    when no API key is configured. The protected-mode API key is NOT injected
    here - the shell route sets it as an HttpOnly cookie instead, so it never
    reaches page JS / localStorage (S2). An empty *token* injects nothing.

    The token is embedded only in same-origin HTML served to a trusted loopback
    client and is a short-lived per-process secret, not the durable API key."""
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    if not token:
        return html
    # json.dumps escapes quotes/backslashes; also escape "<" so the value can
    # never break out of the <script> element (defence in depth).
    snippet = ("<script>window.__LOCALM_SHELL_TOKEN__="
               + json.dumps(token).replace("<", "\\u003c")
               + ";</script>")
    lower = html.lower()
    i = lower.find("<head>")
    if i != -1:
        cut = i + len("<head>")
        return html[:cut] + snippet + html[cut:]
    return snippet + html


def _set_session_cookies(response, key: str, *, secure: bool) -> None:
    """Establish the S2 auth cookie on *response* for a loopback owner: mint an
    OPAQUE server-side session for the current owner *key* and set the HttpOnly
    ``localm_session`` cookie to the SESSION ID (never the key, so it never touches
    page JS and rolling the key does not invalidate it). It carries SESSION_MAX_AGE
    so the session PERSISTS across a browser/PWA restart (SEAMLESS). No-op if *key*
    is not a valid key. The CSRF token is DERIVED from the session and fetched by
    the client from GET /api/session, so there is no separate CSRF cookie to set (or
    to fall out of sync with the session, the pre-rework 'missing CSRF token' bug)."""
    from localm import scopes as S, sessions
    from localm.auth import _hash_key, fs_access_for, verify
    from localm.inference.http_server import SESSION_COOKIE, SESSION_MAX_AGE
    held = verify(key)
    if held is None:
        return
    fs = "host" if S.ADMIN in held else fs_access_for(key, "none")
    sid = sessions.create(scopes=held, key_hash=_hash_key(key), fs_access=fs)
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, secure=secure,
                        samesite="strict", path="/", max_age=SESSION_MAX_AGE)


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class LoadModelRequest(BaseModel):
    model: str


class PullRequest(BaseModel):
    spec: str
    name: str | None = None
    mmproj: str | None = None


class RemoveModelRequest(BaseModel):
    model: str


class AliasRequest(BaseModel):
    model: str
    alias: str


class ShareClearRequest(BaseModel):
    ids: list[str] = []


class LogExportRequest(BaseModel):
    dest: str = ""


# Image types accepted from a phone share-sheet into the chat composer.
_SHARE_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".heic", ".heif"}


def _share_inbox() -> Path:
    """Transient inbox for files shared INTO localm from a phone (PWA share
    target). Lives under the data dir; entries are deleted once the app ingests
    them, so it never accumulates."""
    from localm.config import home_dir
    d = home_dir() / "share_inbox"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _multipart_boundary(content_type: str) -> "bytes | None":
    """The boundary token from a multipart/form-data Content-Type, or None."""
    if "multipart/form-data" not in (content_type or "").lower():
        return None
    for part in content_type.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            b = part[len("boundary="):].strip().strip('"')
            return b.encode("latin-1") if b else None
    return None


def _disp_param(disposition: bytes, key: bytes) -> "bytes | None":
    """Value of a Content-Disposition parameter, e.g. name= or filename=."""
    token = key + b'="'
    i = disposition.find(token)
    if i == -1:
        return None
    i += len(token)
    j = disposition.find(b'"', i)
    return disposition[i:j] if j != -1 else None


def _parse_multipart(body: bytes, boundary: bytes):
    """Minimal multipart/form-data parser - no python-multipart dependency, in
    keeping with localm's self-contained rule (it already hand-builds multipart
    for ComfyUI uploads). Returns (fields: dict[str,str], files: list of
    (filename, content_type, data))."""
    fields: dict = {}
    files: list = []
    for raw in body.split(b"--" + boundary):
        part = raw.strip(b"\r\n")
        if not part or part == b"--":
            continue                       # preamble / closing delimiter
        head, sep, data = part.partition(b"\r\n\r\n")
        if not sep:
            continue
        headers: dict = {}
        for line in head.split(b"\r\n"):
            k, _, v = line.partition(b":")
            headers[k.strip().lower()] = v.strip()
        disposition = headers.get(b"content-disposition", b"")
        name = _disp_param(disposition, b"name")
        filename = _disp_param(disposition, b"filename")
        if filename is not None:
            ctype = headers.get(b"content-type", b"application/octet-stream")
            files.append((filename.decode("utf-8", "replace"),
                          ctype.decode("latin-1"), data))
        elif name is not None:
            fields[name.decode("utf-8", "replace")] = data.decode("utf-8", "replace")
    return fields, files


# R37: a phone (or any browser) uploads files INTO a durable localm folder that
# models and tools can then read - distinct from transient chat attachments and
# the share_inbox. Capped per request because the hand-rolled parser reads the
# whole body into memory; large model weights use the model pull flow, not this.
_MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB / request


# Characters never allowed in an uploaded file's basename: the Windows-reserved
# set (a name like "x.txt:stream" would otherwise write to an NTFS alternate data
# stream that the list/delete routes cannot see) plus all control chars (a literal
# NUL would raise ValueError deep in pathlib and surface as a bare 500).
_BAD_NAME_CHARS = set('<>:"|?*\\/') | {chr(c) for c in range(32)}


def _name_is_safe(safe: str) -> bool:
    """True if *safe* (already a basename) is a usable, listable file name."""
    return bool(safe) and safe not in (".", "..") and not (set(safe) & _BAD_NAME_CHARS)


def _uploads_dir() -> Path:
    """<home>/uploads/, created on demand."""
    from localm.config import home_dir
    d = home_dir() / "uploads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _confined_upload_path(name: str) -> Path:
    """Resolve a user-supplied file name to a path INSIDE the uploads dir. Strips
    any directory components (basename only) and rejects anything that would
    resolve outside the uploads dir - so '..', absolute paths, encoded slashes, or
    a name with illegal/control chars cannot traverse out or crash. Raises
    HTTPException(400) on an unsafe name."""
    base = _uploads_dir()
    safe = Path(name or "").name            # basename only, drops any dir parts
    if not _name_is_safe(safe):
        raise HTTPException(400, "Invalid file name.")
    target = (base / safe).resolve()
    if not target.is_relative_to(base.resolve()):
        raise HTTPException(400, "Invalid file name.")
    return target


def _unique_upload_target(base: Path, safe_name: str) -> Path:
    """A non-clobbering path for *safe_name* in *base*: 'note.txt' -> 'note (1).txt'
    if it already exists (mirrors the log-export dedup), so an upload never
    silently overwrites an existing file."""
    target = base / safe_name
    if not target.exists():
        return target
    stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
    n = 1
    while True:
        cand = base / f"{stem} ({n}){suffix}"
        if not cand.exists():
            return cand
        n += 1


# ------------------------------------------------------------------ #
#  Attach                                                             #
# ------------------------------------------------------------------ #

def attach_gui(
    app: FastAPI,
    *,
    self_url: str,
    switch_model,
    active_model,
) -> SessionManager:
    """
    Add GUI routes and static serving to *app*.

    Parameters
    ----------
    self_url:
        Base URL of this server's own /v1 API - coder agents talk to the
        model through it (e.g. ``http://127.0.0.1:8642/v1``).
    switch_model:
        ``Callable[[str], Awaitable[None]]`` - swaps the active engine.
    active_model:
        ``Callable[[], str]`` - name of the currently served model.
    """
    manager = SessionManager()

    from .jobs import JobManager
    jobs = JobManager()

    # Shared services that converted plugins (rag/image/music/video) reach via
    # request.app.state: the background-job manager, this server's own /v1 base
    # URL (for self-embedding), and the active-model accessor. The /api/jobs/*
    # SSE endpoints stay in the kernel GUI (routes/jobs.py) so every plugin's
    # jobs stream through the one manager.
    app.state.jobs = jobs
    app.state.self_url = self_url
    app.state.active_model = active_model
    # The builtin "coder" plugin reads these to drive live sessions and
    # per-session model switches; its routes 503 when they're absent
    # (headless / no GUI). The manager is also returned for close_all().
    app.state.switch_model = switch_model
    app.state.coder_sessions = manager

    # Route groups (extracted to localm/plugins/gui/routes/*.py). The active-model
    # accessor, the model-switch callable, and the job manager are the only locals
    # the handlers close over, so they travel to the groups on ctx. Each group's
    # register() appends its routes; registration order does not matter (every
    # path is distinct) as long as the static mount below stays last.
    from types import SimpleNamespace
    ctx = SimpleNamespace(active_model=active_model, switch_model=switch_model,
                          jobs=jobs)
    from .routes import models as _routes_models
    _routes_models.register(app, ctx)
    from .routes import system as _routes_system
    _routes_system.register(app, ctx)
    from .routes import pairing as _routes_pairing
    _routes_pairing.register(app, ctx)
    from .routes import admin as _routes_admin
    _routes_admin.register(app, ctx)
    from .routes import jobs as _routes_jobs
    _routes_jobs.register(app, ctx)
    from .routes import share as _routes_share
    _routes_share.register(app, ctx)
    from .routes import uploads as _routes_uploads
    _routes_uploads.register(app, ctx)

    # R47: the "/api/bug-report" POST lives on the core server (http_server.py) so
    # it works in headless `localm serve` too; the GUI button targets that single
    # canonical route. Duplicating it here would shadow it (first route registered
    # wins) and silently drop the user's description + include_log flag.

    # Media generation (image /api/imagine*, music /api/music*, video /api/video*)
    # moved to standalone builtin plugins (localm/plugins/builtin/{image,music,
    # video}) in Phase 3; each ships disabled by default and reads its own
    # per-plugin backend config (the shared ComfyUI launch/reload helpers moved
    # into those plugins' backends).

    # Chat persistence (/api/conversations, /api/memory, /api/prompts) moved
    # to the builtin "chat" plugin (localm/plugins/builtin/chat) - the
    # preinstalled, protected, default-enabled plugin #0. The chat turn
    # itself (/v1/chat/completions) stays in the kernel inference server.

    # Knowledge / RAG (/api/rag/*) moved to the builtin "rag" plugin
    # (localm/plugins/builtin/rag) in Phase 3; it ships disabled by default and
    # reaches the shared job manager + self-embed URL via request.app.state.

    # ------------------------- static ----------------------------- #
    # Mounted last: API routes above take precedence over the SPA files.
    # Pin the MIME types the PWA relies on (some Windows registries map .js to
    # text/plain, and .webmanifest is unknown to mimetypes) so the service
    # worker, app scripts, manifest, and icon are served correctly.
    import mimetypes
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("application/manifest+json", ".webmanifest")
    mimetypes.add_type("image/svg+xml", ".svg")

    # Serve the SPA shell. On a loopback bind (or for a loopback client on a LAN
    # bind) seed the configured API key into the page so a fresh launch is not
    # locked out when require_auth is on (C1). Registered before the "/" static
    # mount so it wins for the shell document; every other asset hits the mount.
    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    async def _gui_index(request: Request):
        from localm import auth
        # Only a loopback BIND (the default `localm gui`, reachable solely from
        # this machine) auto-seeds anything. We deliberately do NOT trust
        # request.client.host: behind a same-host reverse proxy it reads as
        # loopback for REMOTE users, which would leak the secret. A non-loopback
        # bind (e.g. -H 0.0.0.0) seeds nothing - the user enters the key in the
        # page, which POSTs it to /api/session to set the session cookie.
        loopback = _is_loopback_host(
            getattr(request.app.state, "bind_host", "127.0.0.1"))
        key = auth.get_api_key() or ""
        # The shell must never be served stale: it carries the per-request shell
        # token and references the current assets, so always revalidate (a new
        # app.js / index.html is then picked up without the user clearing caches).
        headers = {"Cache-Control": "no-cache"}
        if key and loopback:
            # Protected mode on loopback: establish an HttpOnly session cookie so
            # the key never touches page JS / localStorage (S2). Only MINT a new
            # session when the browser has no valid one, so an ordinary reload does
            # not spawn a session each time - and, crucially, a browser whose
            # session is still valid after an owner-key ROLL stays signed in (the
            # session is decoupled from the key), instead of being bounced to the
            # key gate (the reported bug).
            from localm.inference.http_server import SESSION_COOKIE
            from localm import sessions as _sessions
            existing = (request.cookies.get(SESSION_COOKIE) or "").strip()
            resp = HTMLResponse(_index_html_with_shell_token(""), headers=headers)
            if not (existing and _sessions.lookup(existing) is not None):
                _set_session_cookies(resp, key, secure=request.url.scheme == "https")
            return resp
        if not key and loopback:
            # Open mode on loopback: seed the per-process shell token as a JS
            # global so the loopback SPA can still manage (H5). app.js sends it
            # as a bearer HEADER (the open-mode gate is header-based); it is
            # never persisted.
            token = getattr(request.app.state, "shell_token", "") or ""
            return HTMLResponse(_index_html_with_shell_token(token), headers=headers)
        return HTMLResponse(_index_html_with_shell_token(""), headers=headers)

    app.mount("/", _RevalidatingStatic(directory=str(STATIC_DIR), html=True), name="gui")

    # Single source of truth that the GUI surface is mounted on this app, so the
    # on-demand mount (phase 5 mount_gui_surface) is idempotent whether the GUI
    # was attached at startup (localm gui) or live on a running api instance.
    app.state.gui_mounted = True

    return manager
