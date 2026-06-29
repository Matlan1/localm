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
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import (HTMLResponse, RedirectResponse, Response,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from localm import scopes
from localm.inference.http_server import (_require_auth, caller_scopes,
                                          job_owner_ok, principal_id,
                                          require_scope)
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
    """Set the S2 auth cookies on *response*: the HttpOnly ``localm_session``
    cookie (the API key, unreadable by page JS) plus a readable ``localm_csrf``
    token for double-submit CSRF. Names match http_server's SESSION_COOKIE /
    CSRF_COOKIE; both carry SESSION_MAX_AGE so the key PERSISTS across a browser/
    PWA restart (SEAMLESS) instead of being dropped as a session cookie."""
    import secrets as _secrets
    from localm.inference.http_server import (CSRF_COOKIE, SESSION_COOKIE,
                                              SESSION_MAX_AGE)
    response.set_cookie(SESSION_COOKIE, key, httponly=True, secure=secure,
                        samesite="strict", path="/", max_age=SESSION_MAX_AGE)
    response.set_cookie(CSRF_COOKIE, _secrets.token_urlsafe(32), httponly=False,
                        secure=secure, samesite="strict", path="/",
                        max_age=SESSION_MAX_AGE)


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

    # -------------------------- models ---------------------------- #

    @app.get("/api/models", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
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

    @app.post("/api/models/load", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
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

    @app.get("/api/vram-estimate", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
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

    @app.get("/api/companion", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def gui_companion():
        """LAN / Tailscale address a phone should open to reach THIS server, for
        the Companion-app card. The card builds full URLs from these plus the
        browser's own scheme + port (the server listens on one port across every
        interface), so it never shows the meaningless loopback address. On the
        default loopback bind (``localm gui``) no phone can connect yet, so
        ``network_bind`` is False and the card explains how to bind to the
        network instead."""
        from localm import tls
        bind_host = getattr(app.state, "bind_host", "127.0.0.1")
        addrs = tls.companion_addresses()
        return {
            "network_bind": not _is_loopback_host(bind_host),
            "lan": addrs.get("lan") or "",
            "tailscale": addrs.get("tailscale") or "",
        }

    @app.get("/api/capabilities", dependencies=[Depends(_require_auth)])
    async def gui_capabilities(caller: Optional[set] = Depends(caller_scopes)):
        """Effective navigation entitlements for the CURRENT key, so the GUI shows
        ONLY the tabs this key can actually use - a capability the key's scopes do
        not grant is never rendered (no show-then-'no access').

        Baseline-gated (any valid key) on purpose: a key must be able to learn its
        OWN entitlements without holding plugins:read, otherwise a narrow key could
        never see the tabs it IS allowed. In open mode (no key configured) caller is
        None and everything is granted. chat is always present - chatting needs no
        scope; the 'chat' scope gates the chat-HISTORY plugin, not the chat turn."""
        held = caller                      # None => open mode / full access
        def granted(required: str) -> bool:
            return held is None or scopes.grants(held, required)
        core = {
            "chat":     True,
            "models":   granted(scopes.MODELS_READ),
            "plugins":  granted(scopes.PLUGINS_READ),
            "settings": granted(scopes.CONFIG_READ),
        }
        mgr = getattr(app.state, "plugin_manager", None)
        plugins, suggest = [], True
        if mgr is not None:
            state = mgr.api_state()
            suggest = bool(state.get("suggest_plugins", True))
            # Only surface plugins this key's scopes grant; renderNav still filters
            # to active+tab, and the command-hint map keeps inactive-but-granted
            # entries so "/cmd needs the X plugin" still works for a scoped key.
            for p in state.get("plugins", []):
                if granted(p.get("scope") or p.get("name") or ""):
                    plugins.append(p)
        return {"scopes": sorted(held) if held else [], "open": held is None,
                "core": core, "plugins": plugins, "suggest_plugins": suggest}

    # R47: the "/api/bug-report" POST lives on the core server (http_server.py) so
    # it works in headless `localm serve` too; the GUI button targets that single
    # canonical route. Duplicating it here would shadow it (first route registered
    # wins) and silently drop the user's description + include_log flag.

    @app.post("/api/logs/export", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def export_logs(req: LogExportRequest):
        """R30: copy every log of this running instance into a user-chosen folder
        (picked via the GUI's /api/fs/dirs browser). Writes a timestamped
        subfolder so repeated exports never clobber each other. Logs live under
        <home>/logs; a few (e.g. comfy-launch.log) sit in the home root, so we
        sweep both. Returns the counts and the destination path."""
        import shutil
        import time as _time
        from localm.config import home_dir
        from localm.debuglog import logs_dir
        dest = (req.dest or "").strip()
        if not dest:
            raise HTTPException(400, "Choose a destination folder first.")
        dest_dir = Path(dest).expanduser()
        if not dest_dir.is_dir():
            raise HTTPException(400, "That folder does not exist.")
        sources = []
        try:
            sources.append(logs_dir())
            sources.append(home_dir())               # home root for stray *.log
        except Exception:
            pass
        out = dest_dir / f"localm-logs-{_time.strftime('%Y%m%d-%H%M%S')}"
        seen: set = set()
        copied = 0
        try:
            out.mkdir(parents=True, exist_ok=True)
            used: set = set()
            for src in sources:
                if not src or not src.is_dir():
                    continue
                for p in src.glob("*.log"):
                    if p.resolve() in seen:
                        continue
                    seen.add(p.resolve())
                    # Two logs can share a basename across home/ and home/logs/;
                    # disambiguate so the second does not clobber the first.
                    target = p.name
                    n = 1
                    while target in used:
                        target = f"{p.stem}-{n}{p.suffix}"
                        n += 1
                    used.add(target)
                    try:
                        shutil.copy2(p, out / target)
                        copied += 1
                    except OSError:
                        pass
        except OSError as e:
            raise HTTPException(500, f"Could not write to that folder: {e}")
        if copied == 0:
            # Be honest: nothing was exported (no logs, or all copies failed).
            return {"copied": 0, "dest": str(out),
                    "message": "No log files were found to export."}
        return {"copied": copied, "dest": str(out)}

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

    @app.get("/api/fs/dirs", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def fs_dirs(path: str = "", include_files: bool = False):
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
                return {"path": "", "parent": None, "dirs": roots, "files": []}
            path = "/"
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a directory: {path}")
        p = p.resolve()
        dirs = []
        files = []
        try:
            for child in sorted(p.iterdir(), key=lambda c: c.name.lower()):
                try:
                    if not child.name.startswith("."):
                        if child.is_dir():
                            dirs.append(child.name)
                        elif include_files and child.is_file():
                            files.append(child.name)
                except OSError:
                    continue   # broken junction / reparse point
        except PermissionError:
            raise HTTPException(403, f"Permission denied: {path}")
        at_root = p.parent == p
        return {"path": str(p),
                "parent": "" if at_root else str(p.parent),
                "dirs": dirs,
                "files": files}

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

    @app.post("/api/comfyui/create-launcher", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def create_comfy_launcher(workdir: str):
        if not workdir:
            raise HTTPException(400, "Missing workdir parameter")
            
        from localm.config import load_config
        cfg = load_config()
        
        valid_workdirs = []
        if cfg.get("comfy_workdir"):
            valid_workdirs.append(cfg["comfy_workdir"])
            
        plugins_cfg = cfg.get("plugins", {})
        if isinstance(plugins_cfg, dict):
            for p_cfg in plugins_cfg.values():
                if isinstance(p_cfg, dict):
                    c_cfg = p_cfg.get("comfy", {})
                    if isinstance(c_cfg, dict) and c_cfg.get("workdir"):
                        valid_workdirs.append(c_cfg["workdir"])
                        
        if not valid_workdirs:
            raise HTTPException(400, "ComfyUI working directory is not configured")
            
        import os
        try:
            p = Path(workdir).resolve()
            
            resolved_valids = [Path(v).resolve() for v in valid_workdirs]
            if p not in resolved_valids:
                raise HTTPException(403, "workdir must match a configured comfy_workdir")
                
            if not p.is_dir():
                raise HTTPException(400, "workdir is not a valid directory")
                
            if os.name == "nt":
                script = p / "launch-comfyui.bat"
                if script.exists():
                    raise HTTPException(409, "Launcher script already exists")
                script.write_text("@echo off\r\ncd /d \"%~dp0\"\r\nif exist venv\\Scripts\\activate (call venv\\Scripts\\activate)\r\npython main.py\r\npause\r\n", encoding="utf-8")
            else:
                script = p / "launch-comfyui.sh"
                if script.exists():
                    raise HTTPException(409, "Launcher script already exists")
                script.write_text("#!/bin/bash\ncd \"$(dirname \"$0\")\"\n[ -f venv/bin/activate ] && source venv/bin/activate\npython3 main.py\n", encoding="utf-8")
                script.chmod(0o755)
        except PermissionError:
            raise HTTPException(403, "Permission denied while checking or writing the launcher script")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to create launcher: {e}")
            
        return {"status": "ok"}

    @app.post("/api/models/pull", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_pull(req: PullRequest, request: Request):
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
        if req.mmproj:
            args += ["--mmproj", req.mmproj]
        args += ["--", spec]
        # Stream structured download progress; suppress huggingface_hub's own
        # tqdm bars (their \r output doesn't line-stream cleanly).
        job = jobs.start_cli("pull", args, extra_env={
            "LOCALM_PROGRESS_JSON": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }, host_label=f"Model pull {spec}", owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/remove", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
    async def model_remove(req: RemoveModelRequest, request: Request):
        from localm.config import load_registry
        if req.model not in load_registry():
            raise HTTPException(404, f"Model not registered: {req.model}")
        if req.model == active_model():
            raise HTTPException(409, "Cannot remove the active model - switch first")
        job = jobs.start_cli("remove", ["rm", req.model, "--yes"],
                             owner=principal_id(request))
        return {"job_id": job.id}

    @app.post("/api/models/alias", dependencies=[Depends(require_scope(scopes.MODELS_WRITE))])
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
    async def job_events(job_id: str, request: Request):
        job = jobs.get(job_id)
        # A job is reachable only by the key that created it (or an admin/owner).
        # Return 404 - not 403 - on an ownership mismatch so a non-owner cannot even
        # confirm the (unguessable) id exists (KEY-SCOPE-2).
        if job is None or not job_owner_ok(request, job.owner):
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
    async def job_cancel(job_id: str, request: Request):
        job = jobs.get(job_id)
        # Same owner-binding as the events stream: only the creating key (or an
        # admin/owner) may cancel; others get an indistinguishable 404.
        if job is None or not job_owner_ok(request, job.owner):
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

    @app.get("/api/discover/search", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
    async def discover_search(q: str = "", limit: int = 20):
        from localm.discover import DiscoverError, hf_search, vram_info
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None, lambda: hf_search(q, limit=limit))
        except DiscoverError as e:
            raise HTTPException(_discover_status(e), str(e))
        return {"query": q, "results": results, "vram": vram_info()}

    @app.get("/api/discover/files", dependencies=[Depends(require_scope(scopes.MODELS_READ))])
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
        models = []
        mmprojs = []
        for f in files:
            f["fit"] = fit_label(f["size_bytes"], total)
            if "mmproj" in f["file"].lower():
                mmprojs.append(f)
            else:
                models.append(f)
        return {"repo": repo.strip().strip("/"), "files": models, "mmprojs": mmprojs, "vram": vram}

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

    # --------------------- uploads (R37) -------------------------- #

    @app.post("/api/upload", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def upload_files(request: Request):
        """R37: accept files from a phone/browser into <home>/uploads/ so models
        and tools can read them, beyond transient chat attachments. Multipart,
        parsed without python-multipart. CONFIG_WRITE (owner/companion-admin)
        because writing host files is privileged - a restricted shared key must
        not be able to drop files on the host. Capped at _MAX_UPLOAD_BYTES."""
        clen = request.headers.get("content-length", "")
        cap_mb = _MAX_UPLOAD_BYTES // (1024 * 1024)
        if clen.isdigit() and int(clen) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload too large (max {cap_mb} MB per request).")
        boundary = _multipart_boundary(request.headers.get("content-type", ""))
        if boundary is None:
            raise HTTPException(400, "Expected a multipart/form-data upload.")
        body = await request.body()
        if len(body) > _MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload too large (max {cap_mb} MB per request).")
        _fields, files = _parse_multipart(body, boundary)
        base = _uploads_dir()
        saved = []
        for filename, _ctype, data in files:
            if not data:
                continue
            safe = Path(filename or "").name[:255]
            if not _name_is_safe(safe):
                continue            # empty/.. or illegal/control chars (e.g. NTFS ADS)
            target = _unique_upload_target(base, safe)
            # Defense in depth: never write outside the uploads dir.
            if not target.resolve().is_relative_to(base.resolve()):
                continue
            try:
                target.write_bytes(data)
            except OSError as e:
                raise HTTPException(500, f"Could not save {safe}: {e}")
            saved.append({"name": target.name, "bytes": len(data)})
        if not saved:
            raise HTTPException(400, "No files were uploaded.")
        return {"uploaded": saved, "dir": str(base)}

    @app.get("/api/uploads", dependencies=[Depends(require_scope(scopes.CONFIG_READ))])
    async def list_uploads():
        """R37: list files in <home>/uploads/ (name, size, mtime) for the Settings
        'Uploaded files' list."""
        base = _uploads_dir()
        items = []
        for p in sorted(base.iterdir()):
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            items.append({"name": p.name, "bytes": st.st_size,
                          "mtime": int(st.st_mtime)})
        return {"items": items, "dir": str(base)}

    @app.delete("/api/uploads/{name}",
                dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def delete_upload(name: str):
        """R37: remove one uploaded file. The name is basename-confined to the
        uploads dir (no path is built from raw input), so it cannot traverse out."""
        target = _confined_upload_path(name)
        if not target.is_file():
            raise HTTPException(404, "No such uploaded file.")
        try:
            target.unlink()
        except OSError as e:
            raise HTTPException(500, f"Could not delete {target.name}: {e}")
        return {"removed": target.name}

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
            # Protected mode on loopback: establish the HttpOnly session cookie
            # directly so the key never touches page JS / localStorage (S2).
            resp = HTMLResponse(_index_html_with_shell_token(""), headers=headers)
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
