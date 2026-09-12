# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI Web Share Target routes (PWA).

The phone shares an image (or text/link) from any app INTO localm via the OS
share sheet (manifest "share_target"). The browser POSTs it to /share-target; we
stash it in a transient inbox and bounce back to the app, which ingests the
images as chat attachments and clears the inbox.

The inbox is staging for one redirect, not a store. In privacy mode it lives in
this process's memory (``app.state.share_memory_inbox``); in the log/full modes
it is the on-disk ``share_inbox`` directory. Either way an entry expires after
``_SHARE_TTL_SECONDS``: expired entries are dropped whenever the inbox is
touched, and stale disk entries are swept once at startup.

The multipart parser and the inbox helpers stay in ``web.py`` and are imported
by name.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

import localm.plugins.gui.web as _web
from localm.debuglog import logger
from localm.inference.http_server import _require_auth, job_owner_ok, principal_id
from localm.plugins.gui.web import ShareClearRequest

# How long a shared entry waits for the app to ingest it.
_SHARE_TTL_SECONDS = 15 * 60


def _memory_inbox(app) -> dict:
    """The privacy-mode inbox: entry name -> (bytes, expires_at), kept on
    ``app.state`` so it dies with the process."""
    store = getattr(app.state, "share_memory_inbox", None)
    if store is None:
        store = {}
        app.state.share_memory_inbox = store
    return store


def _privacy_mode() -> bool:
    from localm.audit import SessionMode, effective_mode
    return effective_mode("server") == SessionMode.PRIVACY


def _sweep_expired(app, inbox: Path, *, startup: bool = False) -> None:
    """Drop memory entries past their expiry and unlink disk entries older than
    the TTL. Never raises: a disk entry that cannot be removed is logged and
    left for the next sweep."""
    now = time.time()
    store = _memory_inbox(app)
    for name in [n for n, (_, exp) in store.items() if exp <= now]:
        store.pop(name, None)
    cutoff = now - _SHARE_TTL_SECONDS
    try:
        stale = [p for p in inbox.glob("*__*")
                 if p.is_file() and p.stat().st_mtime <= cutoff]
    except OSError:
        return
    for p in stale:
        fid = _web._parse_share_entry(p)[0]
        try:
            p.unlink()
        except OSError as e:
            logger.warning("share-inbox: could not remove expired entry %s: %s", fid, e)
            continue
        if startup:
            logger.info("share-inbox: removed expired entry %s left by an earlier run", fid)
        else:
            logger.debug("share-inbox: removed expired entry %s", fid)


def register(app: FastAPI, ctx) -> None:

    try:
        _sweep_expired(app, _web._share_inbox(create=False), startup=True)
    except Exception as e:
        logger.warning("share-inbox: startup sweep failed: %s", e)

    @app.post("/share-target", dependencies=[Depends(_require_auth)],
              include_in_schema=False)
    async def share_target(request: Request):
        import uuid as _uuid
        boundary = _web._multipart_boundary(request.headers.get("content-type", ""))
        if boundary is None:
            raise HTTPException(400, "Expected a multipart/form-data share")
        body = await request.body()
        fields, files = _web._parse_multipart(body, boundary)
        _sweep_expired(app, _web._share_inbox(create=False))
        owner = principal_id(request)
        # Names are all checked BEFORE anything is written, so a refused share
        # never leaves a partial inbox entry behind.
        accepted = []
        for filename, _ctype, data in files:
            if not data:
                continue
            # Only images are ingested (the chat vision path); ignore other types.
            if Path(filename or "").suffix.lower() not in _web._SHARE_IMAGE_EXTS:
                continue
            safe = Path(filename or "shared").name[:80] or "shared"
            # Path().name alone is NOT a sufficient sanitizer: it leaves
            # "photo:stream.png" intact, which on NTFS writes into an alternate
            # data stream that /api/share/pending cannot list, and leaves an
            # embedded NUL intact, which raises ValueError out of write_bytes as
            # a bare 500. Reuses /api/upload's guard. This route REFUSES where
            # /api/upload skips: it returns only a redirect carrying a count, so
            # a silent skip would be invisible to the caller.
            if not _web._name_is_safe(safe):
                raise HTTPException(400, "Invalid file name.")
            accepted.append((safe, data))
        shared_text = (fields.get("text") or fields.get("url") or "").strip()
        if shared_text:
            accepted.append(("shared.txt", shared_text[:20000].encode("utf-8")))
        # Privacy mode stages in memory; the log/full modes stage on disk.
        in_memory = _privacy_mode()
        store = _memory_inbox(app) if in_memory else None
        inbox = None if in_memory else _web._share_inbox()
        expires_at = time.time() + _SHARE_TTL_SECONDS
        n = 0
        for safe, data in accepted:
            name = _web._share_entry_name(owner, _uuid.uuid4().hex, safe)
            if in_memory:
                store[name] = (data, expires_at)
            else:
                (inbox / name).write_bytes(data)
            n += 1
        # 303 so the browser GETs the app shell (a POST-redirect-GET); the app
        # reads ?shared and pulls the inbox.
        return RedirectResponse(url=f"/?shared={n}", status_code=303)

    @app.get("/api/share/pending", dependencies=[Depends(_require_auth)])
    async def share_pending(request: Request):
        """Pending shared entries as data URIs, for the app to ingest as chat
        attachments: the in-memory entries first, then the disk inbox. Does
        not delete - the app calls /api/share/clear after it has the data, so
        a failed fetch does not lose the share.

        Scoped to the caller's own shares: an admin/owner sees all, and an
        entry with no recorded owner (open mode) stays visible to everyone, so
        one key cannot read another key's shared content."""
        import base64
        import mimetypes as _mt
        inbox = _web._share_inbox(create=False)
        _sweep_expired(app, inbox)
        items = []

        def _add(fid, name, data):
            mime = _mt.guess_type(name)[0] or "application/octet-stream"
            items.append({
                "id": fid, "name": name, "type": mime,
                "data_uri": f"data:{mime};base64," + base64.b64encode(data).decode(),
            })

        store = _memory_inbox(app)
        for entry, (data, _exp) in sorted(store.items()):
            fid, owner, name = _web._parse_share_entry(Path(entry))
            if not job_owner_ok(request, owner):
                continue
            _add(fid, name, data)
        for p in sorted(inbox.glob("*__*")):
            if not p.is_file():
                continue
            fid, owner, name = _web._parse_share_entry(p)
            if not job_owner_ok(request, owner):
                continue
            try:
                data = p.read_bytes()
            except OSError:
                continue
            _add(fid, name, data)
        return {"items": items}

    @app.post("/api/share/clear", dependencies=[Depends(_require_auth)])
    async def share_clear(req: ShareClearRequest, request: Request):
        """Delete shared inbox entries the app has ingested, in memory and on
        disk. With no ids, clears all of the CALLER's own (never another key's
        - same ownership scoping as /api/share/pending). The id is matched as a
        name prefix (no path is built from it), so it cannot traverse out of
        the inbox. ``failed`` counts disk entries that could not be deleted."""
        inbox = _web._share_inbox(create=False)
        keep = set(req.ids)
        removed = 0
        failed = 0
        store = _memory_inbox(app)
        for entry in list(store):
            fid, owner, _name = _web._parse_share_entry(Path(entry))
            if not job_owner_ok(request, owner):
                continue
            if not req.ids or fid in keep:
                store.pop(entry, None)
                removed += 1
        for p in inbox.glob("*__*"):
            fid, owner, _name = _web._parse_share_entry(p)
            if not job_owner_ok(request, owner):
                continue
            if not req.ids or fid in keep:
                try:
                    p.unlink()
                    removed += 1
                except OSError as e:
                    # A delete that FAILED is reported distinctly from an entry
                    # the caller never asked about: on a privacy-adjacent store
                    # a locked or permission-denied file must not read like a
                    # clean sweep. Non-fatal rather than a 500 - the clear is
                    # cleanup riding on a share ingest that has already
                    # succeeded client-side, and the client re-sends the same
                    # ids on the next ingest, so it self-heals.
                    failed += 1
                    logger.warning("share-inbox clear could not delete %s: %s", p.name, e)
        # chat.js reads `failed`, logs it and toasts the user; its share-clear
        # handler defaults it to 0 for a server that omits it.
        return {"removed": removed, "failed": failed}
