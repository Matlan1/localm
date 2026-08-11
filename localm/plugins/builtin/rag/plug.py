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

Background indexing uses the kernel's background-job registry
(``request.app.state.jobs``), which since ADR-0008 is created by ``attach_engine``
rather than ``attach_gui`` - so a bare ``localm serve`` has one too, and indexing,
upload and embedding-model setup all run as streamed background jobs there exactly
as they do under the GUI. They previously did not: ``jobs`` was absent headless, so
add/upload ran synchronously on the plugin pool and setup returned a 503 telling
the caller to "run `localm gui`". Both of those are gone, and the headless response
shape for add/upload is now ``{"job_id": ...}`` like the GUI's.

Self-embedding still derives its URL from the kernel's own advertised bind
coordinates + live engine (``_kernel_self_services``) when the GUI never published
``.self_url`` / ``.active_model``. Query never needed a job; with no embedder
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
                choice = r.json()["choices"][0]["message"]["content"].strip().lower()
                choice = choice.replace("`", "").replace(".", "")
                return choice
            # Non-ok HTTP response: a real tie-break failure. self_request never
            # raises on a non-2xx, so this path was the silent one - surface it
            # before falling back to the heuristic label (AGENTS.md rule 5).
            from localm.debuglog import logger
            logger.debug("rag classify: tie-break returned HTTP %s, falling back "
                         "to heuristic label", getattr(r, "status_code", "?"))
        except Exception as e:
            # Best-effort tie-break only (the caller falls back to the "text"
            # label), but the failure should be discoverable, not silent
            # (AGENTS.md rule 5).
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
            # A "" here means the model answered but with nothing - a genuine
            # empty-but-successful description, which extract_bytes honestly
            # reports as an empty description. That is distinct from a REQUEST
            # failure, which propagates below.
            return r.json()["choices"][0]["message"]["content"].strip()
        # Not OK: surface the endpoint's REAL error so extract_bytes wraps its true
        # cause ("Image description failed: ...") instead of masking it as an empty
        # description (AGENTS.md rule 5). A real transport error from self_request
        # (timeout / connection reset) likewise propagates rather than being
        # swallowed to None - the old string-match ("vision") only re-raised the
        # no-vision case and hid everything else as a false "empty" result.
        err_detail = ""
        try:
            err_detail = r.json().get("detail") or ""
        except Exception:
            # Best-effort extraction of the endpoint's error detail; safe to skip
            # because the mandatory raise below ALWAYS surfaces the failure (it
            # falls back to the HTTP status when no JSON detail is available).
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


def _dim_mismatch(stats: dict, active_dim) -> "bool | None":
    """Best-effort: does *stats* (a ``stats()``-shaped dict) disagree with
    *active_dim* (the currently RESIDENT embedder's dimension, from
    ``embedder.loaded_dim()`` - or None when nothing is loaded)?

    None whenever an honest comparison cannot be made - no vectors to compare,
    this collection's own dimension is unknown, or no embedder happens to be
    loaded right now - never folded into a false "matches" (AGENTS rule 5).
    Mirrors ``_collection_dim_report``'s own three-way split, just answered
    from whatever is already resident instead of loading the target model,
    which is what makes this cheap enough to run on every listing/detail
    request instead of only at switch time."""
    dim = stats.get("vector_dim")
    if not stats.get("has_vectors") or dim is None or active_dim is None:
        return None
    return dim != active_dim


def _collection_dim_report(target_dim: int) -> dict:
    """Compare every existing collection's currently stored vector dimension
    against *target_dim* (a newly selected embedding model's own dimension),
    so switching models can name exactly what it is about to invalidate
    instead of a generic "click reindex" pointer that says which collections
    need it for no one (NEW-RAG-DIM-NO-REEMBED item 3).

    ``/api/rag/collections`` reports ``has_vectors`` as a purely OFFLINE fact
    (are >=80% of this collection's chunks embedded), never compared against
    the model actually active right now - so a collection built under a
    now-replaced model still shows "hybrid" with no badge; only an actual
    query discovers the dimension mismatch and quietly drops to BM25 (see
    Collection._vector_scores' own guard, which this mirrors by reading the
    same ``vector_dim()``). That silent gap is what this closes.

    Three buckets; a collection with no vectors at all lands in none of
    them - there is nothing for a model switch to invalidate:
      "degrades": has vectors at a KNOWN dimension that no longer matches
                  target_dim - falls back to BM25/lexical the moment it is
                  next queried, unless re-embedded first.
      "unknown":  has vectors, but the stored dimension cannot be
                  established (a legacy index, or one _load() already found
                  unusable) - NEVER folded into "fine" or "degrades", since
                  neither is actually known.
    Collections whose dimension already matches are only counted
    (``unaffected``), not named - there is nothing actionable to say about
    them.

    Best-effort per collection: one that fails to even construct (a stale
    directory mid-delete, an OS error) is counted into "unknown" rather than
    aborting the model switch this report is secondary to (AGENTS rule 5:
    best-effort is fine here as long as the failure is named, not folded into
    a false "fine")."""
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
            # Not reachable under the current has_vectors/vector_dim coupling
            # (see vector_dim()'s docstring) - kept so a future change to that
            # coupling degrades to "unknown", never silently to "fine".
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

    Present on any app built through ``attach_engine``, which since ADR-0008
    creates it - including a bare ``localm serve``. It used to be created by
    ``attach_gui`` alone, so this raised a 503 telling the caller that indexing
    "needs the localm GUI server (run `localm gui`)"; that sentence stopped
    being true the moment the registry moved to kernel level, so it is gone
    rather than reworded, along with its *needs* parameter.

    The guard itself stays, because it now catches a CONSTRUCTION error (a
    router mounted on an app that never ran attach_engine) and a clean 503
    still beats the unguarded AttributeError -> opaque 500 of audit item 8."""
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "This server has no background job registry, "
                                 "so indexing cannot be started.")
    return jobs


def _kernel_self_services(request: Request):
    """Derive ``(self_url, active_model)`` from the KERNEL's own state when the GUI
    shell never published them. ``attach_gui`` is the only setter of
    ``app.state.self_url`` / ``.active_model``, so a bare ``localm serve`` (api-mode)
    would otherwise have no way to self-embed - every headless index silently
    degrading to lexical-only (memory-audit cluster 24). But ``localm serve`` still
    advertises its bind coordinates (``instance_scheme`` / ``instance_port``, set by
    ``instances.advertise`` before uvicorn accepts a request), and the live engine
    is ``http_server._engine`` - the same two sources ``mount_gui_surface`` uses to
    build these very callables. Returns ``(None, None)`` when the coordinates are
    absent (a bare ``create_app`` test, or before ``advertise``), so the caller
    degrades cleanly to lexical-only rather than dialling a bogus URL."""
    port = getattr(request.app.state, "instance_port", None)
    if not port:
        return None, None
    scheme = getattr(request.app.state, "instance_scheme", None) or "http"
    self_url = f"{scheme}://127.0.0.1:{port}/v1"

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
    serve`` (api-mode) does not, so we fall back to deriving them from the kernel's
    own advertised coordinates (see ``_kernel_self_services``) rather than degrading
    every headless index to lexical-only (memory-audit cluster 24). ``embed_fn=None``
    etc. are already-supported degrade paths in the store layer (lexical-only search,
    no format tie-break, no image description) - the same fallback used when
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
    """Route an indexing progress line to the debug logger (LM-DA-015).

    The line that matters is the "embeddings unavailable ... indexing
    lexical-only" degrade warning (store.py add_paths/add_uploads): a doc that
    fell back to lexical-only otherwise looks like an ordinary success. The
    logger surfaces it - printed when ``--debug`` is on, and always captured in
    the always-on in-memory activity ring buffer (see debuglog.py) so it shows
    up in a bug report even without --debug.

    Was headless-only, when headless indexed synchronously with no job to stream
    into. Since ADR-0008 every mode indexes through a job, so this is paired
    with the stream push instead of replaced by it - see ``_job_progress``."""
    logger.warning("rag index: %s", text)


def _job_progress(job):
    """``on_progress`` for an indexing job: each line goes to the job's event
    stream AND to the log, and any call that also carries done/total (reembed's
    batch loop) additionally reports it through ``Job.progress`` (ADR-0009 P6).

    Both, not either. The stream is what a watching client sees live, but it is
    ephemeral, per-job and bounded; the log is what a bug report carries.
    Headless used to log the degrade and not stream it, the GUI streamed it and
    did not log it, so each mode was missing the other's copy. Folding headless
    onto the job path would have dropped the logged copy entirely, which is the
    regression LM-DA-015 exists to prevent - so the job path now does both, and
    the GUI gains the bug-report copy it never had.

    The structured branch reuses the SAME ``done``/``total``/``unit`` the text
    was already formatted from - store.py builds both from one set of numbers
    in a single call - rather than a second, independent computation here that
    could drift from the line a viewer reads."""
    def _cb(text: str, *, phase=None, done=None, total=None, unit=None) -> None:
        job.push({"type": "line", "text": text})
        _log_progress(text)
        if done is not None or total is not None:
            job.progress(phase=phase, done=done, total=total, unit=unit)
    return _cb


async def _write_off_loop(call):
    """Run a blocking collection WRITE on the plugin pool and map a lock refusal
    to 409.

    Every write now waits a bounded time for any other process holding the
    collection (localm.rag.collection_lock), so what used to be a quick
    load-modify-save can legitimately sit for seconds. On the single-worker event
    loop that would stall the whole server, so these calls go to the same pool
    /extract and the headless index path already use. 409 (not 500) because
    "someone else is writing this collection, try again shortly" is exactly a
    conflict, and the message names the holder."""
    from localm.rag import CollectionLockedError
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(get_plugin_executor(), call)
    except CollectionLockedError as e:
        raise HTTPException(409, str(e))


@_router.get("/api/rag/collections")
async def rag_collections():
    """List every collection's stats.

    ``Collection(n).stats()`` used to be called directly here, on the event
    loop, for every collection: construction runs ``_load()``, which parses
    ALL of chunks.jsonl and vectors.json just to report counts - measured at
    2.7-3.8s on a real tree, freezing every other in-flight request for that
    long on the single-worker loop (this endpoint has no write to protect, so
    nothing here needed that cost in the first place).

    ``Collection.peek_stats()`` answers from meta.json alone (see its
    docstring and the cache _save() writes), which is both cheap AND correct -
    not a coarser approximation, the SAME numbers a full load would produce,
    just already known. This is the real fix, and it is what every collection
    uses once it has been saved even once under this code (measured: 6
    collections / 3500 chunks each, eager 3.28s -> cached 0.003s, byte-
    identical output).

    A collection that predates this cache (never resaved under this code)
    falls back to the real, full stats() - moved off the loop, so it no
    longer stalls the loop COMPLETELY the way every collection used to
    (measured live, real server + an independent /health poller: the old
    direct call starves the poller entirely, zero responses for the full
    span; the executor-wrapped fallback still lets ~4 through instead of the
    ~25 an unaffected 50ms poller would get - a real server measurement, not
    just the in-process one). It is NOT a complete fix for that fallback case
    by itself: json.loads on a large vectors.json is CPU-bound C code that
    does not release the GIL, so even off the loop it still visibly slows
    other requests via GIL contention (measured up to 641ms for a single
    /health round trip while it ran) - much better than a total freeze, not
    as good as the cache path.

    FORMERLY A KNOWN GAP, now closed: ``_load()`` (which builds the full
    ``stats()``) never calls ``_save()``, so a collection that is written
    once and only ever LISTED after that (a common RAG pattern: index once,
    query many times) used to stay on this fallback on every single listing,
    indefinitely, until something else wrote to it. The cold path below now
    goes through ``Collection.load_and_maybe_backfill()`` instead of a plain
    ``Collection(n)`` construction - same full load, same cost this call was
    already paying, plus an opportunistic attempt (under a non-blocking
    ``collection_write_lock(..., timeout=0)``, skipping quietly if busy) to
    write the ``_stats_cache`` block from what THIS load just read, so the
    NEXT listing of the same collection is cheap. See that method's own
    docstring for why the lock is acquired before the load rather than
    after, which is what makes the backfilled values provably consistent
    with disk rather than a snapshot that could have been overtaken by a
    concurrent real write."""
    from localm.inference.embedder import loaded_dim
    from localm.rag import Collection, collection_names
    loop = asyncio.get_running_loop()
    names = collection_names()
    peeked = {n: Collection.peek_stats(n) for n in names}
    cold = [n for n, s in peeked.items() if s is None]
    if cold:
        fresh = await loop.run_in_executor(
            get_plugin_executor(),
            lambda: {n: Collection.load_and_maybe_backfill(n).stats() for n in cold})
        peeked.update(fresh)
    # Best-effort, NEVER a load: loaded_dim() answers from whatever embedder
    # already happens to be resident (the common case for anyone actively
    # using RAG) and is documented as safe for exactly this - a cheap status
    # probe with no side effect. Still executor-offloaded, not called directly
    # on this coroutine: it blocks on the SAME embedder lock a concurrent
    # model load can hold for its full duration (see http_server.py's own
    # loaded_dim() call site), so a synchronous call here could freeze this
    # whole event loop for that long, not just this one request. When nothing
    # is loaded this is None and every collection's dim_mismatch below is
    # correctly None too - "cannot tell" rendered honestly, never folded into
    # a false "matches". This does NOT replace _collection_dim_report (the
    # switch-time report, which loads and test-embeds the NEW model
    # deliberately, on the one occasion that is worth the cost) - it closes
    # the gap that report's own docstring names: a collection visited well
    # after that one-shot job log has scrolled by still showed a bare
    # "hybrid" with no hint the active model had moved on.
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
async def rag_detail(name: str):
    """Same shape as before (``stats()`` fields plus ``docs``), same cheap-vs-
    fall-back split as rag_collections above - see its docstring, including
    the same ``load_and_maybe_backfill()`` cold path (this route had the
    identical gap: a collection only ever viewed here, never listed, was
    equally cold and equally never backfilled by the old plain-load
    fallback). ``docs()`` was already meta.json-only and cheap; it was
    ``stats()``'s eager ``Collection(name)`` construction that paid the
    full-corpus read just for this page."""
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
    # See rag_collections()'s own comment: best-effort, never a load.
    active_dim = await loop.run_in_executor(get_plugin_executor(), loaded_dim)
    peeked["dim_mismatch"] = _dim_mismatch(peeked, active_dim)
    return peeked


@_router.delete("/api/rag/collections/{name}")
async def rag_delete(name: str):
    from localm.rag import delete_collection
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
        # Mirrors /upload's 50-file cap (LM-DA-018). Since #593 this runs on the
        # shared, bounded plugin ThreadPoolExecutor also used by /extract, /query,
        # web fetch, voice transcription, and coder session management in
        # headless api-mode - an unbounded path list is a cheap way to tie up
        # worker slots. This bounds the number of top-level paths, not the files
        # within a directory tree, which is the same partial-mitigation shape as
        # /upload's per-item cap.
        raise HTTPException(400, "Too many paths in one request (max 50)")
    # CONFINEMENT RUNS FIRST, before anything touches the filesystem (CodeQL 59).
    # This used to start with an `p.exists()` sweep that 400'd "Not found: <path>"
    # for an absent path while an existing one fell through to the confinement
    # verdict below - so the two answers differed, and any rag-scoped caller could
    # ask about ANY absolute path on the server (a system credential store, say)
    # and read existence off the status code. Confining first makes an out-of-policy path
    # get the SAME answer whether or not it is there, and the existence check
    # below then only ever runs on paths the caller is already allowed to index.
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
    # Everything below acts on the CONFINED, resolved paths the check returned,
    # never on the caller's original strings: probing existence on a value that
    # was not the one validated is how a check-then-use gap opens, and the
    # store's own confinement (which re-runs under the collection lock) resolves
    # identically, so the two stages can no longer disagree about which file
    # they mean.
    paths = confined
    # Only now, on paths that PASSED confinement: a caller entitled to index
    # here is entitled to know the file is not there, and rule 5 says tell them
    # the real reason rather than failing vaguely later.
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise HTTPException(400, f"Not found: {', '.join(missing[:5])}")
    embed = req.embed
    self_embed, self_classify, self_describe = _self_services(request)
    embed_fn = self_embed if embed else None
    # Same configured-name lookup _make_self_embed itself sends over the wire
    # (see there) - recorded so a collection built the ordinary way through
    # this route ends up with embedding_model() on record, not only reembed()
    # (FIX4). None (embed off) is fine: never reached without embed_fn.
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    # Kernel-level since ADR-0008, so a bare `localm serve` has one too. This
    # used to branch on "jobs is None" and index SYNCHRONOUSLY for a headless
    # server; that server now gets the same streamed background job the GUI
    # does, so the branch is deleted rather than left as unreachable code.
    # NOTE this changes the headless response shape from the inline index result
    # to {"job_id": ...}, which is the point: headless callers can now follow
    # progress instead of blocking on one long request.
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
            # e.g. an embedding-model dimension change (C3), or another process
            # still writing this collection - report, don't crash. The stream is
            # the only place the user sees this run, so the reason has to land
            # there, not just in a log (AGENTS rule 5).
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        # add_paths already returned successfully - the indexing work itself is
        # done. Mark it before the reporting tail below (formatting + a push
        # loop over its own result dict), so a defect there can no longer
        # misreport a completed index as failed (jobs.py start_fn's
        # mark_outcome contract - the in-process sibling of #1126's CLI-side
        # outcome sentinel).
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
    # Known BEFORE any indexing work starts - the whole request is already
    # decoded at this point - so the job can report a real denominator from its
    # very first event instead of going quiet until the first file finishes
    # (ADR-0009 P7). add_uploads itself has no per-file progress signal to hook
    # (see _job_progress: it only sees the "indexed <name>" line, no index), so
    # this is a single t=0 report of what is already known, not a fabricated
    # per-file percentage.
    upload_bytes_total = sum(len(u["data"]) for u in uploads)

    embed = req.embed
    self_embed, self_classify, self_describe = _self_services(request)
    embed_fn = self_embed if embed else None
    # See rag_add: record which model this upload actually embeds with (FIX4).
    model_name = None
    if embed:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    # See rag_add: kernel-level job registry, so the headless synchronous branch
    # is gone and an upload streams as a background job here too.
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
            # As in rag_add's _index: a dimension change, or another process still
            # writing this collection. Reported on the stream, not a crash.
            job.push({"type": "line", "text": f"error: {e}"})
            return False
        # add_uploads already returned successfully - see rag_add's _index for
        # why this is marked here, before the reporting tail below.
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
    if not req.query.strip():
        raise HTTPException(400, "Empty query")
    k = max(1, min(req.k, 20))
    self_embed, _, _ = _self_services(request)
    loop = asyncio.get_running_loop()
    # Defang control/frame tokens in the untrusted chunk text before it can be
    # spliced into a chat prompt (indirect prompt injection - LM-DA-SEC-03).
    # Done INSIDE the executor, with the query: the chunk text is attacker-
    # authored (a crafted indexed document), and unbounded CPU over hostile text
    # on the event loop stalls the whole server rather than one request.
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
            # validated, so the previous one is intact - say so, because the user's
            # next question is always whether they just lost the collection.
            job.push({"type": "line",
                      "text": f"error: {e} - the previous index was left untouched"})
            return False
        # reembed already returned successfully (the new index is swapped in) -
        # see rag_add's _index for why this is marked before the reporting tail.
        job.mark_outcome("done")
        job.push({"type": "line",
                  "text": (f"done: {result['chunks']} chunks re-embedded at "
                           f"{result['dim']} dimensions with {model}")})
        return True

    from localm.inference.http_server import principal_id
    job = jobs.start_fn("rag-reembed", _run, owner=principal_id(request))
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
    (open mode, or an ADMIN key) sees everything, unchanged."""
    import localm.inference.http_server as _hs
    from localm import scopes
    from localm.config import load_config, load_registry
    from localm.inference.embedder import (
        DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, gpu_fallback_reason,
        last_error, loaded_dim, resolve_embedding_model_path)
    model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    held = _hs.caller_scopes(request)
    is_owner = held is None or scopes.ADMIN in held
    # A known key is a dict lookup and a registered name is a registry lookup -
    # neither probes a caller-influenced path, so resolving those is safe to
    # answer. Anything else is a raw path: do not stat it for a non-owner.
    may_answer = is_owner or model in KNOWN_EMBEDDING_MODELS or model in load_registry()
    if may_answer:
        installed = bool(resolve_embedding_model_path(allow_download=False))
        status = "ready" if installed else "not_installed"
    else:
        # No stat, no path, and no last_error either - a load failure message
        # quotes the spec it failed on, so it would re-leak what is withheld here.
        installed = None
        status = "unknown"
        model = "(set by the owner)"
    return {
        "model": model,
        "default": DEFAULT_EMBEDDING_MODEL,
        "internal": list(KNOWN_EMBEDDING_MODELS),
        "installed": installed,
        "dim": loaded_dim(),
        "error": last_error() if may_answer else None,
        "gpu_fallback_reason": gpu_fallback_reason(),
        "status": status,
    }


@_router.post("/api/rag/embedding")
async def rag_embedding_set(req: EmbeddingModelRequest, request: Request):
    """Select the embedding model, install it if it is an internal key not yet on
    disk (one-click setup - no terminal), then load-and-probe it so the user gets a
    clear answer: ready with its dimension, or a SPECIFIC reason it failed (e.g. it
    is not an embedding model). Runs as a job so a download shows progress. Never
    silently swaps the user's choice - on failure the selection stands and the UI
    offers the internal default.

    OWNER-ONLY. This writes the `embedding_model` config key, which names a FILE
    THIS PROCESS OPENS and is flagged admin_only in the schema. The route's own
    mount gate is the plugin's `rag` scope, and `rag` is NOT in
    scopes.PRIVILEGED_SCOPES - `--scope chat --scope rag` is offered in
    docs/cli.md as the canonical restricted key - so without this check the
    plugin route is a back door around the owner gate on PATCH /v1/config. Same
    shape and placement as the rag_allowed_roots widening check above: open mode
    is the trusted local owner (caller_scopes None) and passes."""
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
        line(f"Ready: {model} ({dim}-dim). Semantic search is on.")
        # The switch itself is done and verified (config written, embedder
        # loaded, self-test passed) - mark it before the impact report below,
        # which reads every existing collection off disk (_collection_dim_report
        # -> collection_names()/Collection()) and must never be able to turn an
        # already-successful model switch into a reported failure (jobs.py
        # start_fn's mark_outcome contract - the in-process sibling of #1126's
        # CLI-side outcome sentinel).
        job.mark_outcome("done")
        # Name exactly what this switch just invalidated, rather than a generic
        # pointer to a button (NEW-RAG-DIM-NO-REEMBED item 3) - reported, not
        # confirmed up front: the config write and embedder reset above already
        # took effect before this line runs (unconditionally, even on the
        # failure returns above), so there is no earlier point in this job
        # where "are you sure" would still be honest about what is about to
        # happen. A pre-switch confirmation would need a separate dry-run
        # endpoint that computes this same report BEFORE writing config - a
        # bigger, two-step redesign that item 3 does not ask for.
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
    job = jobs.start_fn("embed-setup", _setup, owner=principal_id(request))
    return {"job_id": job.id}


@_router.post("/api/rag/collections/{name}/remove-doc")
async def rag_remove_doc(name: str, req: RagRemoveDocRequest):
    coll = _get_collection(name)
    if not await _write_off_loop(lambda: coll.remove_doc(req.path)):
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
    # Extraction walks an archive's members and can take 8-30s+ on a crafted or
    # large upload. This is a single-worker server, so running it inline on this
    # coroutine would freeze the event loop - every route, for every user - for
    # the duration. rag_upload already offloads the same call via a background
    # job (jobs.start_fn -> a worker thread); this route returns the text
    # directly rather than streaming a job, so it offloads to the plugin pool
    # instead (get_plugin_executor - kept off the inference pool so a burst of
    # extraction requests can never starve chat completions, see executor.py).
    loop = asyncio.get_running_loop()
    try:
        text = await loop.run_in_executor(
            get_plugin_executor(), extract_bytes, data, req.filename)
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
