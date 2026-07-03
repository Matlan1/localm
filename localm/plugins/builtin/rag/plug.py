# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG plugin: document collections for retrieval-augmented chat.

Routes (mounted by the engine, auto-scoped to the ``rag`` capability):
  GET    /api/rag/collections                  - list collections + stats
  POST   /api/rag/collections                  - create a collection
  GET    /api/rag/collections/{name}           - collection detail + docs
  DELETE /api/rag/collections/{name}           - delete a collection
  POST   /api/rag/collections/{name}/add       - index server files/folders (job)
  POST   /api/rag/collections/{name}/upload    - index uploaded device files (job)
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

from localm.textguard import neutralise

_router = APIRouter()


def _neutralise_hits(hits: list) -> list:
    """Defang chat control / frame tokens in each retrieved chunk's text before it
    leaves the retrieval boundary (LM-DA-SEC-03, indirect prompt injection).

    A retrieved chunk is UNTRUSTED content: the owner indexed the file, but its
    CONTENT is not trusted - a crafted or malicious document could embed control
    tokens (``<|im_start|>system ...``) or frame markers to forge a role or inject
    instructions when the chunk is spliced into the chat prompt. Untrusted tool
    output (fetch_url / web_search / MCP) is already neutralised via the coder
    provenance layer; RAG retrieval had no equivalent gate. Neutralising here means
    EVERY consumer (the GUI's chat injection, the KB search view, any future tool)
    gets defanged content by construction. Non-text fields (source / pos / score)
    are metadata and left untouched."""
    for h in hits:
        if isinstance(h, dict) and isinstance(h.get("text"), str):
            h["text"] = neutralise(h["text"])
    return hits


class RagCreateRequest(BaseModel):
    name: str


class RagAddRequest(BaseModel):
    paths: list[str]
    embed: bool = True            # try embeddings; degrades to lexical-only
    reindex: bool = False         # force re-index even unchanged files (repair)


class RagQueryRequest(BaseModel):
    query: str
    k: int = 4


class RagRemoveDocRequest(BaseModel):
    path: str


class RagExtractRequest(BaseModel):
    filename: str
    content_b64: str              # in-memory extraction - no disk writes
    max_chars: int = 24_000


class RagUploadItem(BaseModel):
    filename: str
    content_b64: str


class RagUploadRequest(BaseModel):
    files: list[RagUploadItem]
    embed: bool = True            # try embeddings; degrades to lexical-only
    reindex: bool = False         # force re-index of an unchanged upload


class EmbeddingModelRequest(BaseModel):
    model: str                    # an internal key, a registered model name, or a GGUF path


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
        from localm import tls as _tls
        r = _rq.post(f"{self_url}/embeddings",
                     json={"input": texts, "model": active_model() or "localm"},
                     headers=headers, timeout=600,
                     verify=_tls.requests_verify(self_url))
        if not r.ok:
            # Surface the endpoint's actionable detail (e.g. "No embedding model
            # available. Run 'localm setup-embeddings'") instead of the bare HTTP
            # status. This message is shown to the user when indexing/querying
            # degrades to lexical-only, and "422 Unprocessable Entity for url ..."
            # tells them nothing about what to do or that the fix is one command
            # away (AGENTS.md rule 10 / do-not-hide-problems: errors actionable).
            detail = ""
            try:
                detail = (r.json() or {}).get("detail") or ""
            except Exception:
                detail = (r.text or "").strip()
            raise RuntimeError(detail or f"embeddings endpoint returned HTTP {r.status_code}")
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
    # Confine API-driven indexing under the owner's policy (whitelist/blacklist,
    # plus the always-denied localm data dir + credential folders), so a request
    # from a loopback browser page or a remote client cannot read system files or
    # credentials and serve them back (C2). A whitelist MISS is offered back to the
    # owner as "add these folders and continue" (409) instead of a dead-end error;
    # a hard denial (credential / data dir / an explicit deny-list entry) is always
    # refused (400). Only the owner may widen the allow-list, so a non-owner shared
    # key gets a plain 403 for a miss.
    from localm.rag.store import (confine_index_path, indexing_policy,
                                  ConfinementError)
    policy = indexing_policy()
    addable: list[str] = []
    blocked: list[str] = []
    for p in paths:
        try:
            confine_index_path(p, policy)
        except ConfinementError as e:
            (addable if e.reason == "outside_allowed" else blocked).append(
                str(e.path) if e.reason == "outside_allowed" else str(e))
    if blocked:
        raise HTTPException(400, "; ".join(blocked[:5]))
    if addable:
        import localm.inference.http_server as _hs
        from localm import scopes
        held = _hs.caller_scopes(request)
        if held is None or scopes.ADMIN in held:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=409, content={
                "needs_consent": True,
                "reason": "outside_allowed",
                "addable": sorted(set(addable)),
                "detail": "These folders are outside your allowed indexing "
                          "folders. Add them and index?"})
        raise HTTPException(
            403, "These folders are outside the allowed indexing folders, and "
            "only the owner can widen the list: "
            + ", ".join(sorted(set(addable))[:5]))
    embed = req.embed
    jobs = request.app.state.jobs
    self_embed = _make_self_embed(request.app.state.self_url,
                                  request.app.state.active_model)

    def _index(job):
        embed_fn = self_embed if embed else None
        try:
            result = coll.add_paths(
                paths, embed_fn=embed_fn, policy=policy, force=req.reindex,
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

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("rag-index", _index, owner=principal_id(request))
    return {"job_id": job.id}


@_router.post("/api/rag/collections/{name}/upload")
async def rag_upload(name: str, req: RagUploadRequest, request: Request):
    """Ingest documents UPLOADED from the caller's OWN DEVICE into the collection.

    Unlike /add, this reads NO server path - the bytes are in the request - so it
    needs no host filesystem access and no path confinement (whitelist/blacklist
    does not apply to the caller's own files). This is the per-device path for a
    client (a phone, a scoped key) that cannot browse the server disk. Held to the
    rag scope like the rest of the plugin. The whole request body is bounded up
    front (MAX_REQUEST_BODY_BYTES, from Content-Length before buffering); per-file
    and per-request caps are then checked on the base64 STRING length BEFORE
    decoding, so no oversized payload is ever materialized in memory; a zip bomb is
    caught during extraction."""
    coll = _get_collection(name)
    if not req.files:
        raise HTTPException(400, "No files given")
    if len(req.files) > 50:
        raise HTTPException(400, "Too many files in one upload (max 50)")
    # base64 is ~4/3 of the decoded size, so checking the STRING length bounds the
    # decoded size WITHOUT decoding first - that is what stops b64decode from
    # doubling peak memory (the whole request body is already bounded upstream by
    # MAX_REQUEST_BODY_BYTES; this bounds each file WITHIN it). validate=True means
    # the string is pure base64 alphabet, so the 4/3 ratio holds exactly.
    _B64_PER_FILE = 40_000_000      # ~30 MB decoded
    _B64_PER_REQUEST = 134_000_000  # ~100 MB decoded
    uploads: list = []
    b64_total = 0
    for item in req.files:
        b64_total += len(item.content_b64)
        if len(item.content_b64) > _B64_PER_FILE:
            raise HTTPException(413, f"File too large (max 30 MB): {item.filename}")
        if b64_total > _B64_PER_REQUEST:
            raise HTTPException(413, "Upload too large (max 100 MB per request)")
        try:
            data = base64.b64decode(item.content_b64, validate=True)
        except Exception:
            raise HTTPException(400, f"content_b64 is not valid base64: {item.filename}")
        uploads.append({"filename": item.filename, "data": data})

    embed = req.embed
    jobs = request.app.state.jobs
    self_embed = _make_self_embed(request.app.state.self_url,
                                  request.app.state.active_model)

    def _index(job):
        embed_fn = self_embed if embed else None
        try:
            result = coll.add_uploads(
                uploads, embed_fn=embed_fn, force=req.reindex,
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

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("rag-upload", _index, owner=principal_id(request))
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
    # Defang control/frame tokens in the untrusted chunk text before it can be
    # spliced into a chat prompt (indirect prompt injection - LM-DA-SEC-03).
    return {"collection": name, "query": req.query, "hits": _neutralise_hits(hits)}


@_router.get("/api/rag/embedding")
async def rag_embedding_status():
    """Current embedding-model config + availability, for the Knowledge page's
    embedding picker. Cheap: it never loads a model - `dim` is reported only if one
    is already loaded, and `error` carries why the last load failed (if any)."""
    from localm.config import load_config
    from localm.inference.embedder import (
        DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, last_error, loaded_dim,
        resolve_embedding_model_path)
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    installed = bool(resolve_embedding_model_path(allow_download=False))
    return {
        "model": model,
        "default": DEFAULT_EMBEDDING_MODEL,
        "internal": list(KNOWN_EMBEDDING_MODELS),
        "installed": installed,
        "dim": loaded_dim(),
        "error": last_error(),
        "status": "ready" if installed else "not_installed",
    }


@_router.post("/api/rag/embedding")
async def rag_embedding_set(req: EmbeddingModelRequest, request: Request):
    """Select the embedding model, install it if it is an internal key not yet on
    disk (one-click setup - no terminal), then load-and-probe it so the user gets a
    clear answer: ready with its dimension, or a SPECIFIC reason it failed (e.g. it
    is not an embedding model). Runs as a job so a download shows progress. Never
    silently swaps the user's choice - on failure the selection stands and the UI
    offers the internal default."""
    model = req.model.strip()
    if not model:
        raise HTTPException(400, "No model given")
    jobs = request.app.state.jobs

    def _setup(job):
        from localm.config import update_config
        from localm.inference.embedder import (
            get_embedder, last_error, reset_embedder, resolve_embedding_model_path)

        def line(t):
            job.push({"type": "line", "text": t})

        # Persist the choice and drop any cached embedder so the running server
        # switches to the new model without a restart.
        update_config(lambda c: c.__setitem__("embedding_model", model))
        reset_embedder()
        line(f"Selected embedding model: {model}")
        # Resolve, downloading an internal key if it is not on disk yet (the
        # one-click setup). A registered model / path already present is a no-op.
        line("Resolving (downloading if needed)…")
        try:
            path = resolve_embedding_model_path(allow_download=True)
        except Exception as e:                      # network / HF error
            line(f"error: could not fetch '{model}' ({e}). Check your network "
                 "settings, or switch to the internal default (bge-small-en-v1.5).")
            return False
        if not path:
            line(f"error: '{model}' is not a known embedding key, a registered "
                 "model, or a GGUF path - or the download was blocked by the "
                 "network policy (net_mode). Pick a model from the list, or the "
                 "internal default (bge-small-en-v1.5).")
            return False
        # Load + probe via the shared embedder (so it stays loaded for real use).
        # get_embedder swallows the load error but records it; surface it verbatim
        # so the user learns exactly why a wrong pick failed.
        line("Loading and testing the model…")
        emb = get_embedder()
        if emb is None:
            why = last_error() or "the model could not be loaded"
            line(f"error: '{model}' could not produce embeddings ({why}). It may "
                 "not be an embedding model - pick an embedding model from the "
                 "list, or switch to the internal default (bge-small-en-v1.5).")
            return False
        try:
            vecs = emb.embed(["localm embedding self-test"])
            dim = len(vecs[0]) if vecs and vecs[0] else 0
        except Exception as ex:
            line(f"error: '{model}' failed to embed a test string ({ex}). Switch "
                 "to the internal default (bge-small-en-v1.5).")
            return False
        if dim <= 0:
            line(f"error: '{model}' returned empty vectors - it is likely not an "
                 "embedding model. Switch to the internal default "
                 "(bge-small-en-v1.5).")
            return False
        line(f"Ready: {model} ({dim}-dim). Semantic search is on. Re-index a "
             "collection (add its docs again) to add vectors to existing documents.")
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("embed-setup", _setup, owner=principal_id(request))
    return {"job_id": job.id}


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
    # Reject from the base64 STRING length before decoding, so a huge attachment
    # cannot double peak memory during b64decode (see rag_upload).
    if len(req.content_b64) > 40_000_000:      # ~30 MB decoded
        raise HTTPException(413, "Attachment too large (max 30 MB)")
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
