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

import asyncio
import json
import os
import queue
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (HTMLResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from localm import scopes
from localm.inference.http_server import _require_auth, require_scope
from localm.plugins.coder.sessions import SessionManager

STATIC_DIR = Path(__file__).parent / "static"

# SSE keepalive interval - must beat proxy/browser idle timeouts
_KEEPALIVE_S = 15


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
    """Set the S2 auth cookies on *response*: the HttpOnly ``localm_session``
    cookie (the API key, unreadable by page JS) plus a readable ``localm_csrf``
    token for double-submit CSRF. Names match http_server's SESSION_COOKIE /
    CSRF_COOKIE."""
    import secrets as _secrets
    from localm.inference.http_server import CSRF_COOKIE, SESSION_COOKIE
    response.set_cookie(SESSION_COOKIE, key, httponly=True, secure=secure,
                        samesite="strict", path="/")
    response.set_cookie(CSRF_COOKIE, _secrets.token_urlsafe(32), httponly=False,
                        secure=secure, samesite="strict", path="/")


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class LoadModelRequest(BaseModel):
    model: str


class PullRequest(BaseModel):
    spec: str
    name: str | None = None


class RemoveModelRequest(BaseModel):
    model: str


class AliasRequest(BaseModel):
    model: str
    alias: str


class ShareClearRequest(BaseModel):
    ids: list[str] = []


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

    # -------------------------- models ---------------------------- #

    @app.get("/api/models", dependencies=[Depends(_require_auth)])
    async def gui_models():
        from localm.config import load_registry
        registry = load_registry()
        current = active_model()
        models = []
        for name, entry in sorted(registry.items()):
            path = Path(entry.get("path", ""))
            size = None
            try:
                if path.is_file():
                    size = path.stat().st_size
            except OSError:
                pass
            models.append({
                "name": name,
                "source": entry.get("source", ""),
                "size_bytes": size,
                "active": name == current,
            })
        return {"models": models, "active": current}

    @app.post("/api/models/load", dependencies=[Depends(_require_auth)])
    async def gui_load_model(req: LoadModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            return {"status": "already_active", "model": req.model}
        try:
            await switch_model(req.model)
        except Exception as e:
            raise HTTPException(500, f"Failed to load {req.model}: {e}")
        return {"status": "loaded", "model": req.model}

    @app.get("/api/stats", dependencies=[Depends(_require_auth)])
    async def gui_stats():
        """Live system load for the status-bar hardware monitor: CPU %, RAM,
        VRAM, and (NVIDIA only) GPU utilisation. Any section that cannot be
        measured on this box is simply absent - the frontend renders what it
        gets. Runs off-thread so a slow probe (e.g. nvidia-smi) never blocks
        the event loop."""
        from localm.sysstats import system_stats
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(None, system_stats)
        return stats

    @app.get("/api/vram-estimate", dependencies=[Depends(_require_auth)])
    async def vram_estimate(model: str = "", n_ctx: int = 4096, n_gpu_layers: int = 99):
        """Approximate VRAM needed to load *model* (defaults to the active one)
        at the given context + GPU-offload, vs free/total VRAM. Powers the live
        readout under the Settings performance sliders. Always 'approximate'."""
        from localm.config import load_registry
        from localm.discover import vram_info
        from localm.sysstats import estimate_vram
        name = model or active_model()
        model_bytes = 0
        entry = load_registry().get(name)
        if entry:
            try:
                p = Path(entry.get("path", ""))
                if p.is_file():
                    model_bytes = p.stat().st_size
            except OSError:
                pass
        est = estimate_vram(model_bytes, n_ctx, n_gpu_layers)
        info = vram_info()
        free, total = info.get("free"), info.get("total")
        fits = (est["needed"] <= free) if isinstance(free, int) else None
        return {"model": name, "model_bytes": model_bytes, **est,
                "free": free, "total": total, "fits": fits, "approximate": True}

    def _pairing_qr_svg(key: str) -> str:
        """DOMPurify-safe SVG QR encoding ``localm-key:<key>`` for device pairing.
        Hand-built from the module matrix (one black <path> over a white <rect>
        with a viewBox): the qrcode lib's SvgImage emits namespace-prefixed
        <svg:rect> in mm with no viewBox, which DOMPurify strips (blank box) and
        which never scales into the CSS box anyway. Plain <rect>/<path> with
        explicit fills survive sanitisation and render black-on-white on any theme."""
        import qrcode
        qr = qrcode.QRCode(
            error_correction=qrcode.constants.ERROR_CORRECT_M, border=4)
        qr.add_data(f"localm-key:{key}")
        qr.make(fit=True)
        matrix = qr.get_matrix()
        n = len(matrix)
        segments = [
            f"M{x} {y}h1v1h-1z"
            for y, row in enumerate(matrix)
            for x, dark in enumerate(row) if dark
        ]
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" '
            f'shape-rendering="crispEdges" role="img" '
            f'aria-label="localm pairing code">'
            f'<rect width="{n}" height="{n}" fill="#ffffff"/>'
            f'<path d="{"".join(segments)}" fill="#000000"/></svg>'
        )

    @app.get("/api/pairing/qr",
             dependencies=[Depends(require_scope(scopes.ADMIN))])
    async def pairing_qr():
        """SVG QR encoding the OWNER API key (``localm-key:<key>``) so a phone can
        scan it on the onboarding screen and SAVE the key - no typing. Owner scope
        only: it carries the key. 404 in open mode (no key -> nothing to pair).
        Rendered server-side; never cached. For a scoped/limited key the Keys &
        devices manager POSTs the freshly-minted key to the sibling endpoint."""
        from localm import auth
        key = auth.get_api_key()
        if not key:
            raise HTTPException(404, "No API key configured - nothing to pair.")
        return Response(content=_pairing_qr_svg(key), media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/pairing/qr",
              dependencies=[Depends(require_scope(scopes.ADMIN))])
    async def pairing_qr_for_key(body: dict):
        """Render a pairing QR for an ARBITRARY scoped key the owner JUST minted -
        the plaintext is passed in the BODY so it never lands in a URL / access
        log. Owner-gated, never persisted or cached: the key is rendered into the
        SVG and discarded. The phone scans it exactly like the owner-key QR,
        pairing that device with the LIMITED key instead of full admin."""
        key = (body or {}).get("key")
        if not isinstance(key, str) or not key.strip():
            raise HTTPException(400, "Provide the minted key plaintext as 'key'.")
        # Only render a QR for a key that actually exists and is current: doing it
        # for arbitrary input is pointless, and this rejects garbage / an already
        # expired key (verify() returns None for both).
        from localm import auth
        if auth.verify(key.strip()) is None:
            raise HTTPException(400, "Not a current localm key (mint one first).")
        return Response(content=_pairing_qr_svg(key.strip()),
                        media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/fs/dirs", dependencies=[Depends(_require_auth)])
    async def fs_dirs(path: str = ""):
        """Subdirectories of *path*, for the coder setup directory picker.

        An empty path lists drive roots on Windows (filesystem root
        elsewhere). Only directory names leave the server - no file
        names or contents. The GUI is localhost + bearer-auth, and the
        coder agent this picker feeds can read those directories anyway.
        """
        if not path:
            if os.name == "nt":
                import string
                roots = [f"{letter}:\\" for letter in string.ascii_uppercase
                         if Path(f"{letter}:\\").is_dir()]
                return {"path": "", "parent": None, "dirs": roots}
            path = "/"
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a directory: {path}")
        p = p.resolve()
        dirs = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                try:
                    if child.is_dir() and not child.name.startswith("."):
                        dirs.append(child.name)
                except OSError:
                    continue   # broken junction / reparse point
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {path}")
        at_root = p.parent == p
        return {"path": str(p),
                "parent": "" if at_root else str(p.parent),
                "dirs": dirs}

    # ----------------------- model ops + jobs --------------------- #

    from .jobs import JobManager
    jobs = JobManager()

    # Shared services that converted plugins (rag/image/music/video) reach via
    # request.app.state: the background-job manager, this server's own /v1 base
    # URL (for self-embedding), and the active-model accessor. The /api/jobs/*
    # SSE endpoints stay here in the kernel GUI so every plugin's jobs stream
    # through one manager.
    app.state.jobs = jobs
    app.state.self_url = self_url
    app.state.active_model = active_model
    # The builtin "coder" plugin reads these to drive live sessions and
    # per-session model switches; its routes 503 when they're absent
    # (headless / no GUI). The manager is also returned for close_all().
    app.state.switch_model = switch_model
    app.state.coder_sessions = manager

    @app.post("/api/models/pull", dependencies=[Depends(_require_auth)])
    async def model_pull(req: PullRequest):
        spec = req.spec.strip()
        if not spec or set(spec) <= {"-"}:
            raise HTTPException(
                400,
                "Enter a model spec: owner/repo, owner/repo:file.gguf, "
                "or an https URL.",
            )
        # Pass the spec after "--" so a value like "-h" or "--help" is treated as
        # the model argument, not parsed by the CLI as an option/help flag.
        args = ["pull"]
        if req.name:
            args += ["--name", req.name]
        args += ["--", spec]
        # Stream structured download progress; suppress huggingface_hub's own
        # tqdm bars (their \r output doesn't line-stream cleanly).
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {spec}")
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(_require_auth)])
    async def model_remove(req: RemoveModelRequest):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"])
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(_require_auth)])
    async def model_alias(req: AliasRequest):
        from localm.config import load_registry
        registry = load_registry()
        if req.model not in registry:
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.alias in registry:
            raise HTTPException(409, f"Name already taken: {req.alias}")
        from localm.model_manager import alias_model
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, alias_model, req.model, req.alias)
        except Exception as e:
            raise HTTPException(400, f"Alias failed: {e}")
        return {"status": "aliased", "model": req.model, "alias": req.alias}

    @app.get("/api/jobs/{job_id}/events", dependencies=[Depends(_require_auth)])
    async def job_events(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        loop = asyncio.get_running_loop()

        async def _stream():
            while True:
                try:
                    event = await loop.run_in_executor(
                        None, job.events.get, True, _KEEPALIVE_S)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") == "end":
                    return

        return StreamingResponse(
            _stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(_require_auth)])
    async def job_cancel(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"No such job: {job_id}")
        job.cancel()
        return {"status": "cancelling"}

    # Media generation (image /api/imagine*, music /api/music*, video /api/video*)
    # moved to standalone builtin plugins (localm/plugins/builtin/{image,music,
    # video}) in Phase 3; each ships disabled by default and reads its own
    # per-plugin backend config (the shared ComfyUI launch/reload helpers moved
    # into those plugins' backends).

    # ------------------------ model discovery --------------------- #
    # Search HuggingFace for GGUF models and show per-quant "fits your
    # VRAM" badges. User-initiated prelude to a pull (docs/network.md);
    # net_mode=off blocks it like everything else.

    def _discover_status(e: Exception) -> int:
        msg = str(e)
        if "net_mode" in msg:
            return 403          # blocked by the network kill switch
        if "request failed" in msg:
            return 502          # HF unreachable
        return 422              # bad repo / no GGUF files

    @app.get("/api/discover/search", dependencies=[Depends(_require_auth)])
    async def discover_search(q: str = "", limit: int = 20):
        from localm.discover import DiscoverError, hf_search, vram_info
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: hf_search(q, limit=limit))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        return {"query": q, "results": results, "vram": vram_info()}

    @app.get("/api/discover/files", dependencies=[Depends(_require_auth)])
    async def discover_files(repo: str):
        from localm.discover import (DiscoverError, fit_label, hf_gguf_files,
                                     vram_info)
        loop = asyncio.get_running_loop()
        try:
            files = await loop.run_in_executor(
                None, lambda: hf_gguf_files(repo))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        vram = vram_info()
        total = vram.get("total")
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
        return {"repo": repo.strip().strip("/"), "files": files, "vram": vram}

    # Chat persistence (/api/conversations, /api/memory, /api/prompts) moved
    # to the builtin "chat" plugin (localm/plugins/builtin/chat) - the
    # preinstalled, protected, default-enabled plugin #0. The chat turn
    # itself (/v1/chat/completions) stays in the kernel inference server.

    # Knowledge / RAG (/api/rag/*) moved to the builtin "rag" plugin
    # (localm/plugins/builtin/rag) in Phase 3; it ships disabled by default and
    # reaches the shared job manager + self-embed URL via request.app.state.

    # -------------------- Web Share Target (PWA) ------------------ #
    # The phone shares an image (or text/link) from any app INTO localm via the
    # OS share sheet (manifest "share_target"). The browser POSTs it to
    # /share-target; we stash it in a transient server inbox and bounce back to
    # the app, which ingests the images as chat attachments and clears the inbox.
    # This makes phone content actually reach the model, not just "open a link".

    @app.post("/share-target", dependencies=[Depends(_require_auth)],
              include_in_schema=False)
    async def share_target(request: Request):
        import uuid as _uuid
        boundary = _multipart_boundary(request.headers.get("content-type", ""))
        if boundary is None:
            raise HTTPException(400, "Expected a multipart/form-data share")
        body = await request.body()
        fields, files = _parse_multipart(body, boundary)
        inbox = _share_inbox()
        n = 0
        for filename, _ctype, data in files:
            if not data:
                continue
            # Only images are ingested (the chat vision path); ignore other types.
            if Path(filename or "").suffix.lower() not in _SHARE_IMAGE_EXTS:
                continue
            safe = Path(filename or "shared").name[:80] or "shared"
            (inbox / f"{_uuid.uuid4().hex}__{safe}").write_bytes(data)
            n += 1
        shared_text = (fields.get("text") or fields.get("url") or "").strip()
        if shared_text:
            (inbox / f"{_uuid.uuid4().hex}__shared.txt").write_text(
                shared_text[:20000], encoding="utf-8")
            n += 1
        # 303 so the browser GETs the app shell (a POST-redirect-GET); the app
        # reads ?shared and pulls the inbox.
        return RedirectResponse(url=f"/?shared={n}", status_code=303)

    @app.get("/api/share/pending", dependencies=[Depends(_require_auth)])
    async def share_pending():
        """Pending shared files as data URIs, for the app to ingest as chat
        attachments. Does not delete - the app calls /api/share/clear after it
        has the data, so a failed fetch does not lose the share."""
        import base64
        import mimetypes as _mt
        inbox = _share_inbox()
        items = []
        for p in sorted(inbox.glob("*__*")):
            if not p.is_file():
                continue
            fid, _, name = p.name.partition("__")
            mime = _mt.guess_type(name)[0] or "application/octet-stream"
            try:
                data = p.read_bytes()
            except OSError:
                continue
            items.append({
                "id": fid, "name": name, "type": mime,
                "data_uri": f"data:{mime};base64," + base64.b64encode(data).decode(),
            })
        return {"items": items}

    @app.post("/api/share/clear", dependencies=[Depends(_require_auth)])
    async def share_clear(req: ShareClearRequest):
        """Delete shared inbox entries the app has ingested. With no ids, clears
        all. The id is matched as a filename prefix (no path is built from it),
        so it cannot traverse out of the inbox."""
        inbox = _share_inbox()
        keep = set(req.ids)
        removed = 0
        for p in inbox.glob("*__*"):
            fid = p.name.partition("__")[0]
            if not req.ids or fid in keep:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        return {"removed": removed}

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
        if key and loopback:
            # Protected mode on loopback: establish the HttpOnly session cookie
            # directly so the key never touches page JS / localStorage (S2).
            resp = HTMLResponse(_index_html_with_shell_token(""))
            _set_session_cookies(resp, key, secure=request.url.scheme == "https")
            return resp
        if not key and loopback:
            # Open mode on loopback: seed the per-process shell token as a JS
            # global so the loopback SPA can still manage (H5). app.js sends it
            # as a bearer HEADER (the open-mode gate is header-based); it is
            # never persisted.
            token = getattr(request.app.state, "shell_token", "") or ""
            return HTMLResponse(_index_html_with_shell_token(token))
        return HTMLResponse(_index_html_with_shell_token(""))

    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="gui")

    # Single source of truth that the GUI surface is mounted on this app, so the
    # on-demand mount (phase 5 mount_gui_surface) is idempotent whether the GUI
    # was attached at startup (localm gui) or live on a running api instance.
    app.state.gui_mounted = True

    return manager
