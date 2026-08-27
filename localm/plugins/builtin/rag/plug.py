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
  POST   /api/rag/collections/{name}/repair    - rebuild a damaged index (job)
  POST   /api/rag/extract                       - attachment -> text (in memory)

Collections are explicit user data - indexing writes to <data dir>/rag/ in every
session mode, like generated images. /api/rag/extract is the exception: it
converts an uploaded attachment to text entirely in memory, so privacy-mode
chats can use documents without leaving traces.

Background indexing uses the kernel's background-job registry
(``request.app.state.jobs``), created by ``attach_engine``, so a bare ``localm
serve`` has one too and indexing, upload and embedding-model setup all run as
streamed background jobs there exactly as they do under the GUI. The headless
response shape for add/upload is ``{"job_id": ...}`` like the GUI's.

Self-embedding derives its URL from the kernel's own advertised bind coordinates
plus the live engine (``_kernel_self_services``) when the GUI never published
``.self_url`` / ``.active_model``. Query never needs a job; with no embedder
reachable it falls back to lexical-only search (embed_fn=None), the same degrade
path used when embed=False or the embedder itself is unavailable. The background
job stream is served by the kernel's /api/jobs/* endpoints.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from localm.debuglog import logger
from localm.executor import get_plugin_executor
from localm.textguard import neutralise

_router = APIRouter()


def _neutralise_hits(hits: list) -> list:
    """Defang chat control / frame tokens in each retrieved chunk's text before it
    leaves the retrieval boundary (indirect prompt injection).

    A retrieved chunk is UNTRUSTED content: the owner indexed the file, but its
    CONTENT is not trusted - a crafted or malicious document could embed control
    tokens (``<|im_start|>system ...``) or frame markers to forge a role or inject
    instructions when the chunk is spliced into the chat prompt. Neutralising here
    means EVERY consumer (the GUI's chat injection, the KB search view, any future
    tool) gets defanged content by construction. Non-text fields (source / pos /
    score) are metadata and left untouched."""
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
    # None = the whole file. A number requests an excerpt of that many characters.
    max_chars: int | None = None


class RagUploadItem(BaseModel):
    filename: str
    content_b64: str


class RagUploadRequest(BaseModel):
    files: list[RagUploadItem]
    embed: bool = True            # try embeddings; degrades to lexical-only
    reindex: bool = False         # force re-index of an unchanged upload


class EmbeddingModelRequest(BaseModel):
    model: str                    # an internal key, a registered model name, or a GGUF path
    # False: report what switching might invalidate, without writing config or
    # resetting the embedder. True: perform the switch.
    confirm: bool = False


class RagRepairRequest(BaseModel):
    # Recompute embeddings while repairing. If no embedder is available, or this
    # is False, and the collection currently has vectors, repairing drops it to
    # lexical-only; that case requires `confirm`.
    embed: bool = True
    confirm: bool = False


def _make_self_embed(self_url: str, active_model):
    """Embed via this server's own /v1/embeddings - the endpoint holds the
    inference semaphore, so indexing never races a chat reply. Raises when the
    backend has no embedding support (GGUF ctypes binding); callers degrade to
    lexical-only.

    Sends the EMBEDDING model's registered name (from the ``embedding_model``
    config key), not the chat model name. /v1/embeddings recognises a registry
    entry with model_type="embedding" and routes it directly to embed_texts()
    without trying to load a chat engine - so embedding works even when no chat
    model is loaded."""
    def _self_embed(texts: list) -> list:
        from localm.selfclient import self_request
        from localm.config import load_config as _lc
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        _cfg = _lc()
        emb_name = str(_cfg.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
        r = self_request("POST", "/embeddings",
                         json={"input": texts, "model": emb_name or "localm"},
                         timeout=600, base_url=self_url)
        if not r.ok:
            # Surface the endpoint's actionable detail instead of the bare HTTP status.
            detail = ""
            try:
                detail = (r.json() or {}).get("detail") or ""
            except Exception:
                detail = (r.text or "").strip()
            raise RuntimeError(detail or f"embeddings endpoint returned HTTP {r.status_code}")
        return [d["embedding"] for d in r.json()["data"]]
    return _self_embed


def _make_self_classify(self_url: str, active_model):
    """A gated LLM tie-break for a document's format label, via this server's own
    /v1/chat/completions. Used ONLY when the free structural heuristic is unsure
    (see rag.extract.classify_format).

    Short-circuits to None when NO chat model is loaded (``active_model()`` is
    falsy): indexing is frequently embedding-only, and firing a chat request with
    no model would just burn the 10s request timeout per unknown extension for
    nothing. The caller then falls back to the plain "text" label - no stall."""
    def _self_classify(text_snippet: str) -> Optional[str]:
        model = active_model()
        if not model:
            return None
        from localm.selfclient import self_request
        prompt = (
            "You are a file format classifier. Respond ONLY with a single lowercase word "
            "identifying the format of the code/configuration/text snippet below "
            "(e.g., json, yaml, csv, python, javascript, html, markdown, ini, xml, or text). "
            "Do not include any extra words, formatting, markdown formatting, or punctuation.\n\n"
            f"Snippet:\n{text_snippet[:1000]}"
        )
        try:
            r = self_request("POST", "/chat/completions",
                             json={
                                 "model": model,
                                 "messages": [{"role": "user", "content": prompt}],
                                 "temperature": 0.0,
                                 "max_tokens": 10,
                             },
                             timeout=10, base_url=self_url)
            if r.ok:
                body = r.json()["choices"][0]
                if body.get("finish_reason") == "error":
                    # The backend caught a generation crash and rendered it as a
                    # normal 200 (http_server._complete's design, for the chat
                    # UI) - the content is an internal error message, not a real
                    # classification. Fall back to the heuristic label instead of
                    # trusting it.
                    from localm.debuglog import logger
                    logger.debug("rag classify: tie-break backend errored mid-"
                                 "generation, falling back to heuristic label: %s",
                                 body["message"]["content"].strip())
                    return None
                choice = body["message"]["content"].strip().lower()
                choice = choice.replace("`", "").replace(".", "")
                return choice
            # Non-ok HTTP response: self_request does not raise on a non-2xx, so log
            # the tie-break failure before falling back to the heuristic label.
            from localm.debuglog import logger
            logger.debug("rag classify: tie-break returned HTTP %s, falling back "
                         "to heuristic label", getattr(r, "status_code", "?"))
        except Exception as e:
            # Best-effort tie-break: log the failure, the caller falls back to the
            # "text" label.
            from localm.debuglog import logger
            logger.debug("rag classify: self-classification tie-break failed, "
                         "falling back to heuristic label: %s", e)
        return None
    return _self_classify


def _make_self_describe_image(self_url: str, active_model):
    """Describe image via this server's own /chat/completions (vision support)."""
    def _self_describe_image(image_bytes: bytes, mime_type: str) -> Optional[str]:
        from localm.selfclient import self_request
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64}"
        prompt = "Describe this image in detail. Extract any visible text, handwriting, diagram structure, or code verbatim."
        r = self_request("POST", "/chat/completions",
                         json={
                             "model": active_model() or "localm",
                             "messages": [{
                                 "role": "user",
                                 "content": [
                                     {"type": "text", "text": prompt},
                                     {"type": "image_url", "image_url": {"url": data_url}}
                                 ]
                             }],
                             "temperature": 0.2,
                             "max_tokens": 1000,
                         },
                         timeout=60, base_url=self_url)
        if r.ok:
            body = r.json()["choices"][0]
            if body.get("finish_reason") == "error":
                # The backend caught a generation crash (e.g. the vision worker
                # died on undecodable image bytes) and rendered it as a normal
                # 200 (http_server._complete's design, for the chat UI) - the
                # content is an internal error message, never a real
                # description. Raise so extract_bytes records a clean per-file
                # failure instead of indexing the crash text as content.
                raise RuntimeError(body["message"]["content"].strip())
            # "" means the model answered with an empty description. A request
            # failure propagates below.
            return body["message"]["content"].strip()
        # Not OK: surface the endpoint's real error. A transport error from
        # self_request propagates rather than being swallowed to None.
        err_detail = ""
        try:
            err_detail = r.json().get("detail") or ""
        except Exception:
            # Best-effort extraction of the endpoint's error detail; the raise below
            # always surfaces the failure, falling back to the HTTP status.
            pass
        if ("cannot accept image input" in err_detail
                or "UnsupportedInputError" in err_detail
                or "vision" in err_detail):
            raise RuntimeError(
                "Active model does not support vision (load a vision "
                "model/projector to index images).")
        raise RuntimeError(err_detail or f"HTTP {r.status_code}")
    return _self_describe_image


def _get_collection(name: str):
    from localm.rag import Collection
    try:
        coll = Collection(name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if not coll.exists():
        raise HTTPException(404, f"No such collection: {name}")
    return coll


def _require_rag_confinement(name: str, request: Request) -> None:
    """Raise 403 when the caller's key carries a per-key rag_roots allowlist
    and collection *name* holds any host-filesystem document indexed from
    outside those roots (``Collection.confined_to``). A no-op for a caller
    with no rag_roots allowlist (the owner, open mode, or a key that never
    had one set)."""
    from localm.rag import Collection
    from localm.inference.http_server import effective_rag_roots
    key_roots = effective_rag_roots(request)
    if key_roots and not Collection.confined_to(name, key_roots):
        raise HTTPException(
            403, "This key's RAG access is confined to specific folders, "
            "and this collection includes documents from outside them.")


def _dim_mismatch(stats: dict, active_dim) -> "bool | None":
    """Best-effort: does *stats* (a ``stats()``-shaped dict) disagree with
    *active_dim* (the currently RESIDENT embedder's dimension, from
    ``embedder.loaded_dim()`` - or None when nothing is loaded)?

    None whenever an honest comparison cannot be made - no vectors to compare,
    this collection's own dimension is unknown, or no embedder happens to be
    loaded right now - never folded into a false "matches". Mirrors
    ``_collection_dim_report``'s own three-way split, answered from whatever is
    already resident instead of loading the target model, which is what makes
    this cheap enough to run on every listing or detail request."""
    dim = stats.get("vector_dim")
    if not stats.get("has_vectors") or dim is None or active_dim is None:
        return None
    return dim != active_dim


def _collection_dim_report(target_dim: int) -> dict:
    """Compare every existing collection's currently stored vector dimension
    against *target_dim* (a newly selected embedding model's own dimension), so
    switching models can name exactly what it is about to invalidate instead of a
    generic "click reindex" pointer that says which collections need it for no
    one.

    ``/api/rag/collections`` reports ``has_vectors`` as a purely OFFLINE fact (are
    >=80% of this collection's chunks embedded), never compared against the model
    actually active right now, so a collection built under a now-replaced model
    still shows "hybrid" with no badge and only an actual query discovers the
    dimension mismatch and quietly drops to BM25 (see Collection._vector_scores'
    own guard, which this mirrors by reading the same ``vector_dim()``).

    Three buckets; a collection with no vectors at all lands in none of them -
    there is nothing for a model switch to invalidate:
      "degrades": has vectors at a KNOWN dimension that no longer matches
                  target_dim - falls back to BM25/lexical the moment it is next
                  queried, unless re-embedded first.
      "unknown":  has vectors, but the stored dimension cannot be established (a
                  legacy index, or one _load() already found unusable) - NEVER
                  folded into "fine" or "degrades", since neither is known.
    Collections whose dimension already matches are only counted
    (``unaffected``), not named.

    Best-effort per collection: one that fails to even construct (a stale
    directory mid-delete, an OS error) is counted into "unknown" rather than
    aborting the model switch this report is secondary to."""
    from localm.rag import Collection, collection_names
    degrades: list = []
    unknown: list = []
    unaffected = 0
    for name in collection_names():
        try:
            coll = Collection(name)
            stats = coll.stats()
            if not stats["has_vectors"]:
                continue
            dim = coll.vector_dim()
        except Exception as e:
            unknown.append({"name": name,
                             "reason": f"could not be read ({type(e).__name__}: {e})"})
            continue
        if dim is None:
            # Not reached under the current has_vectors/vector_dim coupling; degrades
            # to "unknown" rather than to "fine".
            unknown.append({"name": name,
                             "reason": stats.get("vector_degrade_reason")
                                       or "dimension unknown"})
        elif dim != target_dim:
            degrades.append({"name": name, "dim": dim, "n_chunks": stats["n_chunks"]})
        else:
            unaffected += 1
    return {"degrades": degrades, "unknown": unknown, "unaffected": unaffected}


def _require_jobs(request: Request):
    """The background job manager.

    Present on any app built through ``attach_engine``, which creates it -
    including a bare ``localm serve``.

    The guard stays because it catches a CONSTRUCTION error (a router mounted on
    an app that never ran attach_engine), where a clean 503 beats an unguarded
    AttributeError turning into an opaque 500."""
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "This server has no background job registry, "
                                 "so indexing cannot be started.")
    return jobs


def _kernel_self_services(request: Request):
    """Derive ``(self_url, active_model)`` from the KERNEL's own state when the GUI
    shell never published them. ``attach_gui`` is the only setter of
    ``app.state.self_url`` / ``.active_model``, so a bare ``localm serve``
    (api-mode) would otherwise have no way to self-embed and every headless index
    would silently degrade to lexical-only. But ``localm serve`` still advertises
    its bind coordinates (``instance_scheme`` / ``instance_port``, set by
    ``instances.advertise`` before uvicorn accepts a request), and the live engine
    is ``http_server._engine`` - the same two sources ``mount_gui_surface`` uses to
    build these very callables. Returns ``(None, None)`` when the coordinates are
    absent (a bare ``create_app`` test, or before ``advertise``), so the caller
    degrades cleanly to lexical-only rather than dialling a bogus URL."""
    port = getattr(request.app.state, "instance_port", None)
    if not port:
        return None, None
    scheme = getattr(request.app.state, "instance_scheme", None) or "http"
    from localm.bindhost import self_connect_host, url_host
    _h = url_host(self_connect_host(getattr(request.app.state, "bind_host", None)))
    self_url = f"{scheme}://{_h}:{port}/v1"

    def _active() -> str:
        import localm.inference.http_server as _hs
        eng = getattr(_hs, "_engine", None)
        return eng.display_name if eng is not None else ""

    return self_url, _active


def _self_services(request: Request):
    """Best-effort self_embed/self_classify/self_describe helpers built from this
    server's own /v1/* endpoints, or a matching trio of ``None`` when neither the
    GUI shell nor the kernel's bind coordinates are available. The GUI shell
    publishes ``self_url`` / ``active_model`` via ``attach_gui``; a bare ``localm
    serve`` (api-mode) does not, so they are derived from the kernel's own
    advertised coordinates instead (see ``_kernel_self_services``) rather than
    degrading every headless index to lexical-only. ``embed_fn=None`` etc. are
    already-supported degrade paths in the store layer (lexical-only search, no
    format tie-break, no image description) - the same fallback used when
    embed=False or the embedder itself is unavailable, so this never crashes the
    request."""
    self_url = getattr(request.app.state, "self_url", None)
    active_model = getattr(request.app.state, "active_model", None)
    if not self_url or active_model is None:
        derived_url, derived_active = _kernel_self_services(request)
        self_url = self_url or derived_url
        if active_model is None:
            active_model = derived_active
    if not self_url or active_model is None:
        return None, None, None
    return (_make_self_embed(self_url, active_model),
            _make_self_classify(self_url, active_model),
            _make_self_describe_image(self_url, active_model))


def _log_progress(text: str) -> None:
    """Route an indexing progress line to the debug logger.

    The line that matters is the "embeddings unavailable ... indexing
    lexical-only" degrade warning (store.py add_paths/add_uploads): a doc that
    fell back to lexical-only otherwise looks like an ordinary success. The
    logger surfaces it - printed when ``--debug`` is on, and always captured in
    the always-on in-memory activity ring buffer (see debuglog.py) so it shows
    up in a bug report even without --debug.

    Paired with the job stream push rather than replaced by it - see
    ``_job_progress``."""
    logger.warning("rag index: %s", text)


def _job_progress(job):
    """``on_progress`` for an indexing job: each line goes to the job's event
    stream AND to the log, and any call that also carries done/total (reembed's
    batch loop) additionally reports it through ``Job.progress``.

    Both, not either. The stream is what a watching client sees live, but it is
    ephemeral, per-job and bounded; the log is what a bug report carries.

    The structured branch reuses the SAME ``done``/``total``/``unit`` the text was
    already formatted from - store.py builds both from one set of numbers in a
    single call - rather than a second, independent computation here that could
    drift from the line a viewer reads."""
    def _cb(text: str, *, phase=None, done=None, total=None, unit=None) -> None:
        job.push({"type": "line", "text": text})
        _log_progress(text)
        if done is not None or total is not None:
            job.progress(phase=phase, done=done, total=total, unit=unit)
    return _cb


async def _write_off_loop(call):
    """Run a blocking collection WRITE on the plugin pool and map a lock refusal
    to 409.

    Every write waits a bounded time for any other process holding the collection
    (localm.rag.collection_lock), so a load-modify-save can legitimately sit for
    seconds. On the single-worker event loop that would stall the whole server, so
    these calls go to the same pool /extract and the headless index path use. 409
    (not 500) because "someone else is writing this collection, try again shortly"
    is exactly a conflict, and the message names the holder."""
    from localm.rag import CollectionLockedError
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(get_plugin_executor(), call)
    except CollectionLockedError as e:
        raise HTTPException(409, str(e))


@_router.get("/api/rag/collections")
async def rag_collections(request: Request):
    """List every collection's stats.

    ``Collection.peek_stats()`` answers from meta.json alone (see its docstring
    and the cache _save() writes), which is both cheap AND correct - not a coarser
    approximation, the SAME numbers a full load would produce, just already known.
    That is what every collection uses once it has been saved even once under this
    code. Calling ``Collection(n).stats()`` directly instead would run ``_load()``
    for every collection, parsing ALL of chunks.jsonl and vectors.json just to
    report counts, and freeze every other in-flight request on the single-worker
    loop for seconds; this endpoint has no write to protect, so nothing here needs
    that cost.

    A collection that predates the cache (never resaved under this code) falls
    back to the real, full stats(), run OFF the loop so it no longer stalls the
    loop completely. That fallback is not free even off the loop: json.loads on a
    large vectors.json is CPU-bound C code that does not release the GIL, so it
    still slows other requests through GIL contention.

    The cold path goes through ``Collection.load_and_maybe_backfill()`` rather
    than a plain ``Collection(n)`` construction - same full load, same cost this
    call was already paying, plus an opportunistic attempt (under a non-blocking
    ``collection_write_lock(..., timeout=0)``, skipping quietly if busy) to write
    the ``_stats_cache`` block from what THIS load just read, so the NEXT listing
    of the same collection is cheap. Without it a collection that is written once
    and only ever LISTED afterwards would stay on the fallback indefinitely. See
    that method's own docstring for why the lock is acquired before the load
    rather than after, which is what makes the backfilled values provably
    consistent with disk."""
    from localm.inference.embedder import loaded_dim
    from localm.rag import Collection, collection_names
    from localm.inference.http_server import effective_rag_roots
    loop = asyncio.get_running_loop()
    names = collection_names()
    key_roots = effective_rag_roots(request)
    if key_roots:
        # confined_to returns None for a collection meta.json cannot be
        # read from; both None and False are excluded here.
        names = [n for n in names if Collection.confined_to(n, key_roots)]
    peeked = {n: Collection.peek_stats(n) for n in names}
    cold = [n for n, s in peeked.items() if s is None]
    if cold:
        fresh = await loop.run_in_executor(
            get_plugin_executor(),
            lambda: {n: Collection.load_and_maybe_backfill(n).stats() for n in cold})
        peeked.update(fresh)
    # loaded_dim() answers from whatever embedder is already resident and never
    # loads one. Executor-offloaded because it blocks on the same embedder lock a
    # concurrent model load can hold for its full duration. It is None when nothing
    # is loaded, which leaves every collection's dim_mismatch None.
    active_dim = await loop.run_in_executor(get_plugin_executor(), loaded_dim)
    for n in names:
        peeked[n]["dim_mismatch"] = _dim_mismatch(peeked[n], active_dim)
    return {"collections": [peeked[n] for n in names]}


@_router.post("/api/rag/collections")
async def rag_create(req: RagCreateRequest):
    from localm.rag import Collection
    try:
        coll = Collection(req.name.strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    if coll.exists():
        raise HTTPException(409, f"Collection already exists: {coll.name}")
    # Off the loop like every other write: create() takes the collection write
    # lock (it can race a delete of the same name), so it can block.
    await _write_off_loop(coll.create)
    return coll.stats()


@_router.get("/api/rag/collections/{name}")
async def rag_detail(name: str, request: Request):
    """Same shape as rag_collections (``stats()`` fields plus ``docs``), with the
    same cheap-versus-fallback split and the same
    ``load_and_maybe_backfill()`` cold path - see its docstring. ``docs()`` is
    meta.json-only and cheap; it is ``stats()``'s eager ``Collection(name)``
    construction that would pay a full-corpus read just for this page."""
    from localm.inference.embedder import loaded_dim
    from localm.rag import Collection
    loop = asyncio.get_running_loop()
    peeked = Collection.peek_detail(name)
    if peeked is None:
        def load():
            try:
                coll = Collection.load_and_maybe_backfill(name)
            except ValueError as e:
                raise HTTPException(400, str(e))
            if not coll.exists():
                raise HTTPException(404, f"No such collection: {name}")
            return {**coll.stats(), "docs": coll.docs()}
        peeked = await loop.run_in_executor(get_plugin_executor(), load)
    _require_rag_confinement(name, request)
    # Best-effort, never a load.
    active_dim = await loop.run_in_executor(get_plugin_executor(), loaded_dim)
    peeked["dim_mismatch"] = _dim_mismatch(peeked, active_dim)
    return peeked


@_router.delete("/api/rag/collections/{name}")
async def rag_delete(name: str, request: Request):
    from localm.rag import delete_collection
    _get_collection(name)
    _require_rag_confinement(name, request)
    try:
        if not await _write_off_loop(lambda: delete_collection(name)):
            raise HTTPException(404, f"No such collection: {name}")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"status": "deleted", "name": name}


@_router.post("/api/rag/collections/{name}/add")
async def rag_add(name: str, req: RagAddRequest, request: Request):
    from localm.rag import CollectionLockedError
    coll = _get_collection(name)
    paths = [Path(p).expanduser() for p in req.paths if p.strip()]
    if not paths:
        raise HTTPException(400, "No paths given")
    if len(paths) > 50:
        # Bounds the number of top-level paths, not the files within a directory
        # tree. Mirrors /upload's 50-file cap.
        raise HTTPException(400, "Too many paths in one request (max 50)")
    # Confinement runs before anything touches the filesystem, so an out-of-policy
    # path gets the same answer whether or not it exists. Confines API-driven
    # indexing under the owner's policy (whitelist/blacklist, plus the always-denied
    # localm data dir and credential folders). A whitelist miss is offered back to
    # the owner as "add these folders and continue" (409); a hard denial (credential
    # folder, data dir, or an explicit deny-list entry) is refused (400). Only the
    # owner may widen the allow-list, so a non-owner shared key gets 403 for a miss.
    #
    # A caller whose key carries its own rag_roots allowlist is confined to exactly
    # those folders instead of the global policy; effective_rag_roots resolves to []
    # for the owner/ADMIN and for any key that never had one set.
    from localm.rag.store import (confine_index_path, indexing_policy,
                                  ConfinementError)
    from localm.inference.http_server import effective_rag_roots
    policy = indexing_policy(key_roots=effective_rag_roots(request))
    addable: list[str] = []
    blocked: list[str] = []
    confined: list[Path] = []
    for p in paths:
        try:
            confined.append(confine_index_path(p, policy))
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
    # Everything below acts on the confined, resolved paths the check returned,
    # never on the caller's original strings. The store's own confinement re-runs
    # under the collection lock and resolves identically.
    paths = confined
    # Existence is only probed on paths that passed confinement.
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(400, f"Not found: {', '.join(missing[:5])}")
    embed = req.embed
    self_embed, self_classify, self_describe = _self_services(request)
    embed_fn = self_embed if embed else None
    # The same configured-name lookup _make_self_embed sends over the wire, recorded
    # so the collection has embedding_model() on record. None (embed off) is never
    # reached without embed_fn.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    # The jobs kernel is always present, so a headless server also gets the streamed
    # background job and a {"job_id": ...} response.
    jobs = _require_jobs(request)

    def _index(job):
        try:
            result = coll.add_paths(
                paths, embed_fn=embed_fn, classify_fn=self_classify,
                model_name=model_name,
                describe_image_fn=self_describe,
                policy=policy, force=req.reindex,
                on_progress=_job_progress(job))
        except (ValueError, CollectionLockedError) as e:
            # e.g. an embedding-model dimension change, or another process still
            # writing this collection: report on the stream rather than crash.
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        # The indexing work itself is done: mark the outcome before the reporting
        # tail below, so a failure there cannot misreport a completed index.
        job.mark_outcome("done")
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
    from localm.rag import CollectionLockedError
    coll = _get_collection(name)
    if not req.files:
        raise HTTPException(400, "No files given")
    if len(req.files) > 50:
        raise HTTPException(400, "Too many files in one upload (max 50)")
    # base64 is ~4/3 of the decoded size, so the string length bounds the decoded
    # size without decoding first. validate=True keeps that ratio exact.
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
    # The whole request is already decoded here, so the job reports a real
    # denominator from its first event. add_uploads has no per-file progress signal,
    # so this is a single t=0 report of the total.
    upload_bytes_total = sum(len(u["data"]) for u in uploads)

    embed = req.embed
    self_embed, self_classify, self_describe = _self_services(request)
    embed_fn = self_embed if embed else None
    # Record which model this upload embeds with.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    # An upload streams as a background job.
    jobs = _require_jobs(request)

    def _index(job):
        job.progress(phase="uploading", done=0, total=len(uploads), unit="files",
                     total_bytes=upload_bytes_total)
        try:
            result = coll.add_uploads(
                uploads, embed_fn=embed_fn, classify_fn=self_classify,
                model_name=model_name,
                describe_image_fn=self_describe,
                force=req.reindex,
                on_progress=_job_progress(job))
        except (ValueError, CollectionLockedError) as e:
            # A dimension change, or another process still writing this collection:
            # reported on the stream rather than a crash.
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        # The upload work itself is done: mark the outcome before the reporting
        # tail below.
        job.mark_outcome("done")
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
    _require_rag_confinement(name, request)
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    k = max(1, min(req.k, 20))
    self_embed, _, _ = _self_services(request)
    loop = asyncio.get_running_loop()
    # Defang control/frame tokens in the untrusted chunk text before it can be
    # spliced into a chat prompt. Runs inside the executor, with the query, so
    # unbounded CPU over hostile text does not stall the event loop.
    hits = await loop.run_in_executor(
        get_plugin_executor(),
        lambda: _neutralise_hits(coll.query(req.query, k=k, embed_fn=self_embed)))
    return {"collection": name, "query": req.query, "hits": hits}


@_router.post("/api/rag/collections/{name}/reembed")
async def rag_reembed(name: str, request: Request):
    """Recompute this collection's vectors with the CURRENT embedding model.

    The GUI answer to "I changed the embedding model and now my collection refuses
    everything". Works from the chunk text already in chunks.jsonl, so no source
    file has to still exist - which is what separates it from the reindex button
    (add with force=True), that re-reads the originals AND trips the very dimension
    guard the user is trying to get past.

    A background job like add/upload: re-embedding a large collection is minutes of
    model work, so it streams progress rather than blocking the request.
    """
    coll = _get_collection(name)
    _require_rag_confinement(name, request)
    jobs = _require_jobs(request)
    self_embed, _, _ = _self_services(request)
    if self_embed is None:
        raise HTTPException(
            400, "No embedding model is available, so there is nothing to "
                 "re-embed with. Set one on the Knowledge page, or run "
                 "'localm setup-embeddings'.")

    from localm.config import load_config
    from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
    from localm.rag import CollectionLockedError
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()

    def _run(job):
        try:
            result = coll.reembed(
                embed_fn=self_embed, model_name=model,
                on_progress=_job_progress(job))
        except (ValueError, RuntimeError, CollectionLockedError) as e:
            # reembed only swaps the index in after the whole set is computed and
            # validated, so the previous one is intact.
            job.push({"type": "line",
                      "text": f"error: {e} - the previous index was left untouched"})
            return False
        # The new index is swapped in: mark the outcome before the reporting tail
        # below.
        job.mark_outcome("done")
        job.push({"type": "line",
                  "text": (f"done: {result['chunks']} chunks re-embedded at "
                           f"{result['dim']} dimensions with {model}")})
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("rag-reembed", _run, owner=principal_id(request))
    return {"job_id": job.id}


@_router.post("/api/rag/collections/{name}/repair")
async def rag_repair(name: str, req: RagRepairRequest, request: Request):
    """The GUI's answer to a "needs repair" badge: re-index every rebuildable
    document in *name*, exactly like ``localm rag repair`` (add with force=True
    from ``coll.documents()``). Job-backed and collection-locked the same way as
    add/upload/reembed - ``coll.add_paths`` takes both locks itself.

    Two things this mirrors from the CLI rather than reinvents:

    Embeddings-loss guard. Repairing without embeddings REMOVES them for every
    re-indexed document. *embed* defaults True so the common case (an embedder is
    loaded) never loses anything silently; when that would still drop existing
    vectors (no embedder available, or *embed* is False, and the collection
    currently has vectors), this returns a ``needs_confirm`` dry-run response
    instead of starting a job - the GUI shows it and re-POSTs with
    ``confirm: true``, the same two-step shape ``rag_embedding_set`` uses for its
    own data-risk confirmation.

    Upload-only documents cannot be rebuilt. The uploaded BYTES are never retained
    (see ``add_uploads``'s docstring), so an ``upload:<name>`` key has no source to
    re-extract from - add_paths() silently drops it (``Path('upload:x').is_file()``
    is always False). A collection with NO rebuildable document at all is refused
    up front with an honest reason, instead of running a job that reports
    "0 re-indexed" as if it had fixed something; a MIXED collection proceeds on
    what it can rebuild and the job log names how many upload-only documents were
    left untouched.
    """
    coll = _get_collection(name)
    _require_rag_confinement(name, request)
    docs = coll.documents()
    if not docs:
        detail = (f"'{name}' index is corrupt and has no indexed documents to "
                   "rebuild from." if coll.corrupt else
                   f"'{name}' has no indexed documents.")
        raise HTTPException(400, detail)
    repairable = [d for d in docs if not d.startswith("upload:")]
    upload_only = len(docs) - len(repairable)
    if not repairable:
        raise HTTPException(
            400,
            f"Every document in '{name}' was added via upload, so localm has "
            "no server-side copy to rebuild from - repair cannot fix it here. "
            "Re-upload the affected file(s) to restore them.")

    self_embed, self_classify, self_describe = _self_services(request)
    embed_fn = self_embed if req.embed else None
    would_lose_embeddings = embed_fn is None and bool(coll.stats().get("has_vectors"))
    if would_lose_embeddings and not req.confirm:
        return {
            "needs_confirm": True,
            "detail": (
                f"'{name}' currently has semantic (hybrid) search. Repairing "
                + ("without an embedding model available " if req.embed
                   else "without embeddings ")
                + "will remove the existing embeddings for every re-indexed "
                  "document (it goes back to BM25/lexical-only until "
                  "re-embedded)."),
        }

    jobs = _require_jobs(request)
    model_name = None
    if embed_fn is not None:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()

    def _run(job):
        from localm.rag import CollectionLockedError
        if upload_only:
            job.push({"type": "line", "text": (
                f"{upload_only} document(s) here were added via upload and "
                "have no server-side source - they cannot be rebuilt and are "
                "left as-is.")})
        try:
            result = coll.add_paths(
                repairable, force=True, embed_fn=embed_fn,
                classify_fn=self_classify, describe_image_fn=self_describe,
                model_name=model_name, on_progress=_job_progress(job))
        except (ValueError, CollectionLockedError) as e:
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        job.mark_outcome("done")
        summary = (f"repaired: {result['updated']} re-indexed, "
                   f"{result['added']} added - {result['chunks']} chunks total")
        job.push({"type": "line", "text": summary})
        for f in result["failed"][:10]:
            job.push({"type": "line",
                      "text": f"  failed: {f['path']}: {f['error']}"})
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("rag-repair", _run, owner=principal_id(request))
    return {"job_id": job.id}


@_router.get("/api/rag/embedding")
async def rag_embedding_status(request: Request):
    """Current embedding-model config + availability, for the Knowledge page's
    embedding picker. Cheap: it never loads a model - `dim` is reported only if one
    is already loaded, `error` carries why the last load failed (if any), and
    `gpu_fallback_reason` carries why the loaded embedder dropped to CPU after a
    native GPU crash (if it did).

    `installed` is a FILE-EXISTENCE answer, so a NON-OWNER caller only gets it for
    a localm-managed identity: a KNOWN_EMBEDDING_MODELS key or a registered model
    name. When `embedding_model` is a bare filesystem path, this reports
    `installed: null` / `status: "unknown"` and withholds the path, because both
    the boolean and the path itself are owner-only information: `embedding_model`
    is admin_only, so GET /v1/config already strips its value for a non-owner, and
    an absolute path also discloses the OS user's directory layout. The owner
    (open mode, or an ADMIN key) sees everything, unchanged.

    `can_download` tells the GUI whether to offer the one-time "download now"
    action (POST /api/rag/embedding/download): the configured model is an
    internal key that is not on disk (only those are fetchable - a registered
    model or a path has nothing to download), net_mode is not "off" (or
    net_allow_model_downloads exempts it), and the caller could authorize it -
    open mode, or a key granting config:write, the same scope that governs
    net_mode itself. UI hint only; the download route re-checks everything
    server-side."""
    import localm.inference.http_server as _hs
    from localm import scopes
    from localm.config import load_config, load_registry
    from localm.inference._threadpool_timeout import (
        ThreadCallTimeout, run_in_threadpool_bounded)
    from localm.inference.embedder import (
        DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, READ_TIMEOUT_S,
        gpu_fallback_reason, last_error, loaded_dim, resolve_embedding_model_path)
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    held = _hs.caller_scopes(request)
    is_owner = held is None or scopes.ADMIN in held
    # A known key is a dict lookup and a registered name is a registry lookup;
    # neither probes a caller-influenced path. Anything else is a raw path and is
    # not stat'd for a non-owner.
    may_answer = is_owner or model in KNOWN_EMBEDDING_MODELS or model in load_registry()
    if may_answer:
        installed = bool(resolve_embedding_model_path(allow_download=False))
        status = "ready" if installed else "not_installed"
    else:
        # No stat, no path, and no last_error: a load failure message quotes the
        # model spec that is withheld here.
        installed = None
        status = "unknown"
        model = "(set by the owner)"
    can_download = False
    if installed is False and model in KNOWN_EMBEDDING_MODELS:
        from localm.netpolicy import downloads_allowed_when_off, network_mode
        if network_mode() != "off" or downloads_allowed_when_off():
            can_download = held is None or scopes.grants(held, scopes.CONFIG_WRITE)

    # The three embedder readers run off the event loop together: each does
    # `with _LOCK:`, and get_embedder holds that same lock across a process spawn
    # plus a native load (up to 300s). These values are this route's answer, so the
    # call waits for a running load rather than timing out short.
    def _embedder_state():
        return (loaded_dim(), last_error() if may_answer else None,
                gpu_fallback_reason())

    try:
        dim, error, fallback = await run_in_threadpool_bounded(
            _embedder_state, timeout=READ_TIMEOUT_S)
    except ThreadCallTimeout as e:
        # Past READ_TIMEOUT_S the lock is wedged: report that, rather than returning
        # dim: null / error: null, which reads as "nothing loaded, nothing wrong".
        raise HTTPException(504, f"Could not read the embedder state: {e}")
    return {
        "model": model,
        "default": DEFAULT_EMBEDDING_MODEL,
        "internal": list(KNOWN_EMBEDDING_MODELS),
        "installed": installed,
        "dim": dim,
        "error": error,
        "gpu_fallback_reason": fallback,
        "status": status,
        "can_download": can_download,
    }


@_router.post("/api/rag/embedding/download")
async def rag_embedding_download(request: Request):
    """One-time download of the CURRENTLY CONFIGURED embedding model, for when
    the network policy does not download it automatically (net_mode=ask blocks
    the lazy fetch - see embedder._download_known).

    Writes NOTHING: unlike POST /api/rag/embedding this never touches the
    `embedding_model` config key (which is why config:write suffices here while
    the model SWITCH stays owner-only - selecting a different file this process
    opens widens a trust boundary; fetching the one already selected does not),
    and the one-download authorization is a call argument
    (``resolve_embedding_model_path(allow_download=True)``) consumed by the job
    - net_mode itself stays exactly as configured for every other network path.
    Gated on config:write, the same scope that could change net_mode itself, so
    a key that could not lift the policy cannot bypass it here either; open
    mode is the trusted local owner. net_mode=off refuses by default, and only
    a real config change lifts it - either net_mode itself, or
    net_allow_model_downloads exempting explicit downloads specifically. Only
    an internal KNOWN_EMBEDDING_MODELS key is fetchable this way - a
    registered model or a
    filesystem path has nothing to download, so those get an honest 409 (the
    model name is quoted only for callers GET /api/rag/embedding would answer,
    same disclosure rule)."""
    import localm.inference.http_server as _hs
    from localm import scopes
    from localm.config import load_config, load_registry
    from localm.inference.embedder import (
        DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS,
        resolve_embedding_model_path)
    held = _hs.caller_scopes(request)
    if held is not None and not scopes.grants(held, scopes.CONFIG_WRITE):
        raise HTTPException(
            403, "Downloading the embedding model needs the config:write scope "
                 "(the same permission that governs the network policy).")
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    if model not in KNOWN_EMBEDDING_MODELS:
        is_owner = held is None or scopes.ADMIN in held
        shown = model if (is_owner or model in load_registry()) else "(set by the owner)"
        raise HTTPException(
            409, f"The configured embedding model '{shown}' is not one of the "
                 "internal downloadable models, so there is nothing to fetch "
                 "here. Use the embedding picker to select and set up a model "
                 "instead.")
    if resolve_embedding_model_path(allow_download=False):
        return {"status": "already_installed", "model": model}
    from localm.netpolicy import downloads_allowed_when_off, network_mode
    if network_mode() == "off" and not downloads_allowed_when_off():
        raise HTTPException(
            409, "Network access is disabled (net_mode=off), which blocks even "
                 "an explicitly requested model download. Set net_mode to ask "
                 "or allow, or turn on \"Allow model downloads while network "
                 "access is off\", first.")
    jobs = _require_jobs(request)

    def _run(job):
        from localm.inference.embedder import last_error
        job.push({"type": "line",
                  "text": f"Downloading embedding model '{model}' (one-time)..."})
        try:
            path = resolve_embedding_model_path(allow_download=True)
        except Exception as e:
            job.push({"type": "line", "text": f"error: download failed ({e})"})
            return False
        if not path:
            job.push({"type": "line", "text":
                      f"error: {last_error() or 'the download was blocked or failed'}"})
            return False
        job.push({"type": "line",
                  "text": f"Ready: '{model}' is installed - semantic search is "
                          "on for new indexing. No settings were changed."})
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("embedding-model-download", _run,
                        owner=principal_id(request))
    return {"job_id": job.id, "model": model}


@_router.post("/api/rag/embedding")
async def rag_embedding_set(req: EmbeddingModelRequest, request: Request):
    """Select the embedding model, install it if it is an internal key not yet on
    disk (one-click setup - no terminal), then load-and-probe it so the user gets a
    clear answer: ready with its dimension, or a SPECIFIC reason it failed (e.g. it
    is not an embedding model). Runs as a job so a download shows progress. Never
    silently swaps the user's choice - on failure the selection stands and the UI
    offers the internal default.

    Refuses (409) when an ``embed-setup`` job is already running: a second one
    cannot proceed anyway (both block on the embedder's unbounded load locks), and
    queueing it silently leaves two jobs stuck at "Loading and testing the
    model..." with no error. See the comment above ``start_fn``.

    Two-step by *confirm*. Without it (the default), this is a DRY RUN: no config
    write, no embedder reset, no job - just
    ``localm.rag.collection_provenance_report()``'s honest "these collections have
    semantic search today and may be invalidated" answered synchronously and fast
    (see that function's docstring for what it does not assert about the NEW
    dimension). The same report backs the identical dry-run/confirm gate on
    ``PATCH /v1/config`` and ``localm setup-embeddings``, the other two writers of
    this key. With ``confirm: true``, the switch is made. The caller (the GUI) is
    expected to show the dry-run report, let the user confirm, and only then
    re-POST with ``confirm: true``, so the warning lands BEFORE the switch takes
    effect.

    OWNER-ONLY. This writes the `embedding_model` config key, which names a FILE
    THIS PROCESS OPENS and is flagged admin_only in the schema. The route's own
    mount gate is the plugin's `rag` scope, and `rag` is NOT in
    scopes.PRIVILEGED_SCOPES - `--scope chat --scope rag` is offered in docs/cli.md
    as the canonical restricted key - so without this check the plugin route is a
    back door around the owner gate on PATCH /v1/config. Same shape and placement
    as the rag_allowed_roots widening check above: open mode is the trusted local
    owner (caller_scopes None) and passes. The dry-run branch is gated identically:
    it names collections and chunk counts, the same information the confirmed
    switch discloses."""
    model = req.model.strip()
    if not model:
        raise HTTPException(400, "No model given")
    import localm.inference.http_server as _hs
    from localm import scopes
    held = _hs.caller_scopes(request)
    if held is not None and scopes.ADMIN not in held:
        raise HTTPException(
            403, "Changing the embedding model requires an owner (admin) key: it "
            "selects a file this process loads, so it widens a trust boundary. "
            "The rag scope alone is not enough.")

    if not req.confirm:
        from localm.rag import collection_provenance_note, collection_provenance_report
        affected = collection_provenance_report()
        note = collection_provenance_note(model, affected)
        return {"needs_confirm": True, "model": model,
                "collections": affected, "note": note}

    jobs = _require_jobs(request)

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
        # Load and probe via the shared embedder, so it stays loaded for real use.
        # get_embedder records the load error rather than raising it; it is surfaced
        # verbatim below.
        line("Loading and testing the model…")
        # on_progress=line forwards get_embedder's stage announcements to the job,
        # since this call can run for minutes. _emit_stage swallows anything the
        # sink raises, so a push failure cannot fail a working load.
        emb = get_embedder(on_progress=line)
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
        line(f"Ready: {model} ({dim}-dim). Semantic search is on.")
        # The switch is done and verified (config written, embedder loaded, self-test
        # passed): mark the outcome before the impact report below, which reads every
        # collection off disk.
        job.mark_outcome("done")
        # Name what this switch just invalidated. The config write and embedder reset
        # already took effect above, so this is reported, not confirmed up front.
        _MAX_NAMED = 8
        report = _collection_dim_report(dim)
        degrades, unknown = report["degrades"], report["unknown"]
        if degrades:
            shown = degrades[:_MAX_NAMED]
            detail = ", ".join(f"{c['name']} ({c['dim']}-dim, {c['n_chunks']} chunks)"
                               for c in shown)
            if len(degrades) > _MAX_NAMED:
                detail += f", and {len(degrades) - _MAX_NAMED} more"
            line(f"{len(degrades)} existing collection(s) will fall back to "
                 f"BM25/lexical-only search until re-embedded, because their "
                 f"stored vectors are not {dim}-dim: {detail}. Click 're-embed' "
                 "on each to restore semantic search.")
        if unknown:
            shown = unknown[:_MAX_NAMED]
            names = ", ".join(c["name"] for c in shown)
            if len(unknown) > _MAX_NAMED:
                names += f", and {len(unknown) - _MAX_NAMED} more"
            line(f"{len(unknown)} collection(s) could not be checked against "
                 f"the new model ({names}): their stored vector dimension "
                 "could not be determined. This does not mean they are fine - "
                 "click 're-embed' on each to find out.")
        line("Adding the same documents again does not re-embed unchanged "
             "files - use 're-embed' on a collection below for that.")
        return True

    from localm.inference.http_server import principal_id
    # Refuse a second concurrent setup instead of queueing it behind the first:
    # get_embedder's `with _LOAD_LOCK:` / `with _LOCK:` are bare acquires with no
    # timeout, so the loser would wait silently for as long as the winner takes.
    #
    # This check-then-act is atomic only because it sits in an `async def` on the
    # single event loop with no `await` between the check and start_fn below.
    # Adding an await in this window reopens the race.
    #
    # It does not serialise against another process (a terminal
    # `localm setup-embeddings`, or a second server on the same LOCALM_HOME).
    if jobs.has_running("embed-setup"):
        raise HTTPException(
            409, "An embedding-model setup is already running. Wait for it to "
                 "finish (the Knowledge page shows its progress), then try again.")
    job = jobs.start_fn("embed-setup", _setup, owner=principal_id(request))
    return {"job_id": job.id}


@_router.post("/api/rag/collections/{name}/remove-doc")
async def rag_remove_doc(name: str, req: RagRemoveDocRequest, request: Request):
    coll = _get_collection(name)
    _require_rag_confinement(name, request)
    if not await _write_off_loop(lambda: coll.remove_doc(req.path)):
        raise HTTPException(404, f"Not in this collection: {req.path}")
    return {"status": "removed", "path": req.path}


@_router.post("/api/rag/extract")
async def rag_extract(req: RagExtractRequest):
    """Uploaded chat attachment -> plain text, entirely in memory."""
    from localm.rag import ExtractError, extract_bytes
    # Reject from the base64 string length before decoding, so a huge attachment
    # cannot double peak memory during b64decode.
    if len(req.content_b64) > 40_000_000:      # ~30 MB decoded
        raise HTTPException(413, "Attachment too large (max 30 MB)")
    try:
        data = base64.b64decode(req.content_b64, validate=True)
    except Exception:
        raise HTTPException(400, "content_b64 is not valid base64")
    if len(data) > 30_000_000:
        raise HTTPException(413, "Attachment too large (max 30 MB)")
    # Extraction walks an archive's members and can take tens of seconds on a large
    # or crafted upload, so it runs on the plugin pool rather than inline on the
    # event loop. The plugin pool is kept off the inference pool so extraction
    # requests cannot starve chat completions.
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(
            get_plugin_executor(), extract_bytes, data, req.filename)
    except ExtractError as e:
        raise HTTPException(422, str(e))
    # No cap unless one was asked for; the 30 MB byte guard above is the memory bound.
    if req.max_chars is None:
        return {"filename": req.filename, "text": text,
                "chars": len(text), "truncated": False}
    max_chars = max(500, min(req.max_chars, 200_000))
    return {"filename": req.filename,
            "text": text[:max_chars],
            "chars": len(text),
            "truncated": len(text) > max_chars}


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
