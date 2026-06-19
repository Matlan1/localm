# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG plugin: document collections for retrieval-augmented chat.

Routes (mounted by the engine, auto-scoped to the ``rag`` capability):
  GET    /api/rag/collections                  - list collections + stats
  POST   /api/rag/collections                  - create a collection
  GET    /api/rag/collections/{name}           - collection detail + docs
  DELETE /api/rag/collections/{name}           - delete a collection
  POST   /api/rag/collections/{name}/add       - index files/folders (job)
  POST   /api/rag/collections/{name}/query     - retrieve top-k chunks
  POST   /api/rag/collections/{name}/remove-doc - drop one doc
  POST   /api/rag/extract                       - attachment -> text (in memory)

Collections are explicit user data - indexing writes to <data dir>/rag/ in every
session mode, like generated images. /api/rag/extract is the exception: it
converts an uploaded attachment to text entirely in memory, so privacy-mode
chats can use documents without leaving traces.

Background indexing and self-embedding use the kernel's shared services
(``request.app.state.jobs`` / ``.self_url`` / ``.active_model``), set up by the
GUI; the job stream is served by the kernel's /api/jobs/* endpoints.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_router = APIRouter()


class RagCreateRequest(BaseModel):
    name: str


class RagAddRequest(BaseModel):
    paths: list[str]
    embed: bool = True            # try embeddings; degrades to lexical-only


class RagQueryRequest(BaseModel):
    query: str
    k: int = 4


class RagRemoveDocRequest(BaseModel):
    path: str


class RagExtractRequest(BaseModel):
    filename: str
    content_b64: str              # in-memory extraction - no disk writes
    max_chars: int = 24_000


def _make_self_embed(self_url: str, active_model):
    """Embed via this server's own /v1/embeddings - the endpoint holds the
    inference semaphore, so indexing never races a chat reply. Raises when the
    backend has no embedding support (GGUF ctypes binding); callers degrade to
    lexical-only."""
    def _self_embed(texts: list) -> list:
        import requests as _rq
        headers = {}
        key = os.environ.get("LOCALM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        r = _rq.post(f"{self_url}/embeddings",
                     json={"input": texts, "model": active_model() or "localm"},
                     headers=headers, timeout=600)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]
    return _self_embed


def _get_collection(name: str):
    from localm.rag import Collection
    try:
        coll = Collection(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not coll.exists():
        raise HTTPException(404, f"No such collection: {name}")
    return coll


@_router.get("/api/rag/collections")
async def rag_collections():
    from localm.rag import Collection, collection_names
    return {"collections": [Collection(n).stats()
                            for n in collection_names()]}


@_router.post("/api/rag/collections")
async def rag_create(req: RagCreateRequest):
    from localm.rag import Collection
    try:
        coll = Collection(req.name.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if coll.exists():
        raise HTTPException(409, f"Collection already exists: {coll.name}")
    coll.create()
    return coll.stats()


@_router.get("/api/rag/collections/{name}")
async def rag_detail(name: str):
    coll = _get_collection(name)
    return {**coll.stats(), "docs": coll.docs()}


@_router.delete("/api/rag/collections/{name}")
async def rag_delete(name: str):
    from localm.rag import delete_collection
    try:
        if not delete_collection(name):
            raise HTTPException(404, f"No such collection: {name}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "deleted", "name": name}


@_router.post("/api/rag/collections/{name}/add")
async def rag_add(name: str, req: RagAddRequest, request: Request):
    coll = _get_collection(name)
    paths = [Path(p).expanduser() for p in req.paths if p.strip()]
    if not paths:
        raise HTTPException(400, "No paths given")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(400, f"Not found: {', '.join(missing[:5])}")
    # Confine API-driven indexing to the user's home / working dir so a request
    # from a loopback browser page or a remote client cannot read system files
    # or credentials and serve them back (C2).
    from localm.rag.store import confine_index_path, indexing_roots
    roots = indexing_roots()
    try:
        for p in paths:
            confine_index_path(p, roots)
    except ValueError as e:
        raise HTTPException(400, str(e))
    embed = req.embed
    jobs = request.app.state.jobs
    self_embed = _make_self_embed(request.app.state.self_url,
                                  request.app.state.active_model)

    def _index(job):
        embed_fn = self_embed if embed else None
        try:
            result = coll.add_paths(
                paths, embed_fn=embed_fn, allowed_roots=roots,
                on_progress=lambda t: job.push({"type": "line", "text": t}))
        except ValueError as e:
            # e.g. an embedding-model dimension change (C3) - report, don't crash.
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        summary = (f"done: {result['added']} added, "
                   f"{result['updated']} updated, "
                   f"{result['skipped']} unchanged, "
                   f"{len(result['failed'])} failed - "
                   f"{result['chunks']} chunks total")
        job.push({"type": "line", "text": summary})
        for f in result["failed"][:10]:
            job.push({"type": "line",
                      "text": f"  failed: {f['path']}: {f['error']}"})
        return True

    job = jobs.start_fn("rag-index", _index)
    return {"job_id": job.id}


@_router.post("/api/rag/collections/{name}/query")
async def rag_query(name: str, req: RagQueryRequest, request: Request):
    coll = _get_collection(name)
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    k = max(1, min(req.k, 20))
    self_embed = _make_self_embed(request.app.state.self_url,
                                  request.app.state.active_model)
    loop = asyncio.get_running_loop()
    hits = await loop.run_in_executor(
        None, lambda: coll.query(req.query, k=k, embed_fn=self_embed))
    return {"collection": name, "query": req.query, "hits": hits}


@_router.post("/api/rag/collections/{name}/remove-doc")
async def rag_remove_doc(name: str, req: RagRemoveDocRequest):
    coll = _get_collection(name)
    if not coll.remove_doc(req.path):
        raise HTTPException(404, f"Not in this collection: {req.path}")
    return {"status": "removed", "path": req.path}


@_router.post("/api/rag/extract")
async def rag_extract(req: RagExtractRequest):
    """Uploaded chat attachment -> plain text, entirely in memory."""
    from localm.rag import ExtractError, extract_bytes
    try:
        data = base64.b64decode(req.content_b64, validate=True)
    except Exception:
        raise HTTPException(400, "content_b64 is not valid base64")
    if len(data) > 30_000_000:
        raise HTTPException(413, "Attachment too large (max 30 MB)")
    try:
        text = extract_bytes(data, req.filename)
    except ExtractError as e:
        raise HTTPException(422, str(e))
    max_chars = max(500, min(req.max_chars, 200_000))
    return {"filename": req.filename,
            "text": text[:max_chars],
            "chars": len(text),
            "truncated": len(text) > max_chars}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
