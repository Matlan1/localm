# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI upload routes: accept files into <home>/uploads/, list them, delete one.

The multipart parser, the uploads-dir confinement, the unique-target helper, and
the size cap stay in ``web.py``; they are reached via ``import ... as _web`` (not
imported by value) so that a test which reassigns e.g. ``web._MAX_UPLOAD_BYTES``
is still seen here - the same live-attribute access the inference routes use for
``http_server``.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request

import localm.plugins.gui.web as _web
from localm import scopes
from localm.inference.http_server import require_scope


def register(app: FastAPI, ctx) -> None:

    @app.post("/api/upload", dependencies=[Depends(require_scope(scopes.CONFIG_WRITE))])
    async def upload_files(request: Request):
        """R37: accept files from a phone/browser into <home>/uploads/ so models
        and tools can read them, beyond transient chat attachments. Multipart,
        parsed without python-multipart. CONFIG_WRITE (owner/companion-admin)
        because writing host files is privileged - a restricted shared key must
        not be able to drop files on the host. Capped at _MAX_UPLOAD_BYTES."""
        clen = request.headers.get("content-length", "")
        cap_mb = _web._MAX_UPLOAD_BYTES // (1024 * 1024)
        if clen.isdigit() and int(clen) > _web._MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload too large (max {cap_mb} MB per request).")
        boundary = _web._multipart_boundary(request.headers.get("content-type", ""))
        if boundary is None:
            raise HTTPException(400, "Expected a multipart/form-data upload.")
        body = await request.body()
        if len(body) > _web._MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload too large (max {cap_mb} MB per request).")
        _fields, files = _web._parse_multipart(body, boundary)
        base = _web._uploads_dir()
        saved = []
        for filename, _ctype, data in files:
            if not data:
                continue
            safe = Path(filename or "").name[:255]
            if not _web._name_is_safe(safe):
                continue            # empty/.. or illegal/control chars (e.g. NTFS ADS)
            target = _web._unique_upload_target(base, safe)
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
        base = _web._uploads_dir()
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
        target = _web._confined_upload_path(name)
        if not target.is_file():
            raise HTTPException(404, "No such uploaded file.")
        try:
            target.unlink()
        except OSError as e:
            raise HTTPException(500, f"Could not delete {target.name}: {e}")
        return {"removed": target.name}
