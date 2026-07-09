# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI Web Share Target routes (PWA).

The phone shares an image (or text/link) from any app INTO localm via the OS
share sheet (manifest "share_target"). The browser POSTs it to /share-target; we
stash it in a transient server inbox and bounce back to the app, which ingests the
images as chat attachments and clears the inbox. This makes phone content actually
reach the model, not just "open a link".

Extracted verbatim from attach_gui(); behavior unchanged. The multipart parser and
the transient inbox helper stay in ``web.py`` and are imported by name.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

import localm.plugins.gui.web as _web
from localm.inference.http_server import _require_auth, job_owner_ok, principal_id
from localm.plugins.gui.web import ShareClearRequest


def register(app: FastAPI, ctx) -> None:

    @app.post("/share-target", dependencies=[Depends(_require_auth)],
              include_in_schema=False)
    async def share_target(request: Request):
        import uuid as _uuid
        boundary = _web._multipart_boundary(request.headers.get("content-type", ""))
        if boundary is None:
            raise HTTPException(400, "Expected a multipart/form-data share")
        body = await request.body()
        fields, files = _web._parse_multipart(body, boundary)
        inbox = _web._share_inbox()
        owner = principal_id(request)
        n = 0
        for filename, _ctype, data in files:
            if not data:
                continue
            # Only images are ingested (the chat vision path); ignore other types.
            if Path(filename or "").suffix.lower() not in _web._SHARE_IMAGE_EXTS:
                continue
            safe = Path(filename or "shared").name[:80] or "shared"
            name = _web._share_entry_name(owner, _uuid.uuid4().hex, safe)
            (inbox / name).write_bytes(data)
            n += 1
        shared_text = (fields.get("text") or fields.get("url") or "").strip()
        if shared_text:
            name = _web._share_entry_name(owner, _uuid.uuid4().hex, "shared.txt")
            (inbox / name).write_text(shared_text[:20000], encoding="utf-8")
            n += 1
        # 303 so the browser GETs the app shell (a POST-redirect-GET); the app
        # reads ?shared and pulls the inbox.
        return RedirectResponse(url=f"/?shared={n}", status_code=303)

    @app.get("/api/share/pending", dependencies=[Depends(_require_auth)])
    async def share_pending(request: Request):
        """Pending shared files as data URIs, for the app to ingest as chat
        attachments. Does not delete - the app calls /api/share/clear after it
        has the data, so a failed fetch does not lose the share.

        Scoped to the caller's own shares (an admin/owner sees all; an entry
        with no recorded owner - open mode, or left over from before ownership
        was tracked - stays visible to everyone), so one key cannot read
        another key's shared content."""
        import base64
        import mimetypes as _mt
        inbox = _web._share_inbox()
        items = []
        for p in sorted(inbox.glob("*__*")):
            if not p.is_file():
                continue
            fid, owner, name = _web._parse_share_entry(p)
            if not job_owner_ok(request, owner):
                continue
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
    async def share_clear(req: ShareClearRequest, request: Request):
        """Delete shared inbox entries the app has ingested. With no ids, clears
        all of the CALLER's own (never another key's - same ownership scoping as
        /api/share/pending). The id is matched as a filename prefix (no path is
        built from it), so it cannot traverse out of the inbox."""
        inbox = _web._share_inbox()
        keep = set(req.ids)
        removed = 0
        for p in inbox.glob("*__*"):
            fid, owner, _name = _web._parse_share_entry(p)
            if not job_owner_ok(request, owner):
                continue
            if not req.ids or fid in keep:
                try:
                    p.unlink()
                    removed += 1
                except OSError:
                    pass
        return {"removed": removed}
