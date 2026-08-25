# SPDX-License-Identifier: AGPL-3.0-or-later
"""RAG plugin: document collections for retrieval-augmented chat."""

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
    """Defang chat control / frame tokens in each retrieved chunk's text before it leaves the retrieval boundary (LM-DA-SEC-03, indirect prompt injection)."""
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
    # None = the WHOLE file, and that is the default because it is what
    # "attach this file" means. It used to default to 24_000, which silently
    # turned every attachment over ~24k characters into a PREVIEW: neither
    # chat.js nor coder.js ever sent this field, so both got the cap without
    # asking for it, and a user who attached a real document and asked about
    # its later pages got a confident answer drawn from the first few pages
    # only. Images were never affected - they take a different path entirely
    # (data URI -> image_url), which is why "attachments work" and "attachments
    # are truncated" were both true at once and the bug survived.
    #
    # A caller that genuinely wants an excerpt (a preview pane, a cheap
    # classification) can still ask for one by passing a number.
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
    # False (default): report what switching MIGHT invalidate and stop there -
    # no config write, no embedder reset. True: actually make the switch. See
    # rag_embedding_set's docstring for why this two-step split exists.
    confirm: bool = False


class RagRepairRequest(BaseModel):
    # Try to recompute embeddings while repairing (default: yes, matching the
    # CLI's own no-silent-data-loss stance - see rag_repair's docstring). If no
    # embedder is actually available, or this is False, and the collection
    # currently has vectors, repairing would drop it to lexical-only; that
    # case needs `confirm` (mirrors cli/rag.py's --embed / --yes / the
    # non-interactive confirm prompt it guards).
    embed: bool = True
    confirm: bool = False


def _make_self_embed(self_url: str, active_model):
    """Embed via this server's own /v1/embeddings - the endpoint holds the inference semaphore, so indexing never races a chat reply."""
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
    """A gated LLM tie-break for a document's format label, via this server's own /v1/chat/completions."""
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
    """Best-effort: does *stats* (a ``stats()``-shaped dict) disagree with *active_dim* (the currently RESIDENT embedder's dimension, from ``embedder.loaded_dim()`` - or None when nothing is loaded)?"""
    dim = stats.get("vector_dim")
    if not stats.get("has_vectors") or dim is None or active_dim is None:
        return None
    return dim != active_dim


def _collection_dim_report(target_dim: int) -> dict:
    """Compare every existing collection's currently stored vector dimension against *target_dim* (a newly selected embedding model's own dimension), so switching models can name exactly what it is about to invalidate instead of a generic 'click reindex' pointer that says which collections need it for no o..."""
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
    """The background job manager."""
    jobs = getattr(request.app.state, "jobs", None)
    if jobs is None:
        raise HTTPException(503, "This server has no background job registry, "
                                 "so indexing cannot be started.")
    return jobs


def _kernel_self_services(request: Request):
    """Derive ``(self_url, active_model)`` from the KERNEL's own state when the GUI shell never published them. ``attach_gui`` is the only setter of ``app.state.self_url`` / ``.active_model``, so a bare ``localm serve`` (api-mode) would otherwise have no way to self-embed - every headless index silently de..."""
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
    """Best-effort self_embed/self_classify/self_describe helpers built from this server's own /v1/* endpoints, or a matching trio of ``None`` when neither the GUI shell nor the kernel's bind coordinates are available."""
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
    """Route an indexing progress line to the debug logger (LM-DA-015)."""
    logger.warning("rag index: %s", text)


def _job_progress(job):
    """``on_progress`` for an indexing job: each line goes to the job's event stream AND to the log, and any call that also carries done/total (reembed's batch loop) additionally reports it through ``Job.progress`` (ADR-0009 P6)."""
    def _cb(text: str, *, phase=None, done=None, total=None, unit=None) -> None:
        job.push({"type": "line", "text": text})
        _log_progress(text)
        if done is not None or total is not None:
            job.progress(phase=phase, done=done, total=total, unit=unit)
    return _cb


async def _write_off_loop(call):
    """Run a blocking collection WRITE on the plugin pool and map a lock refusal to 409."""
    from localm.rag import CollectionLockedError
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(get_plugin_executor(), call)
    except CollectionLockedError as e:
        raise HTTPException(409, str(e))


@_router.get("/api/rag/collections")
async def rag_collections():
    """List every collection's stats."""
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
    """Same shape as before (``stats()`` fields plus ``docs``), same cheap-vs- fall-back split as rag_collections above - see its docstring, including the same ``load_and_maybe_backfill()`` cold path (this route had the identical gap: a collection only ever viewed here, never listed, was equally cold and e..."""
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
    #
    # A caller whose KEY carries its own rag_roots allowlist (auth.create_key's
    # per-key field) is confined to exactly those folders instead of the global
    # policy - effective_rag_roots resolves to [] (no per-key restriction) for the
    # owner/ADMIN and for any key that never had one set, so this is a no-op change
    # for every existing key.
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
    """Ingest documents UPLOADED from the caller's OWN DEVICE into the collection."""
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
    """Recompute this collection's vectors with the CURRENT embedding model."""
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


@_router.post("/api/rag/collections/{name}/repair")
async def rag_repair(name: str, req: RagRepairRequest, request: Request):
    """The GUI's answer to a 'needs repair' badge: re-index every rebuildable document in *name*, exactly like ``localm rag repair`` (add with force=True from ``coll.documents()``)."""
    coll = _get_collection(name)
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
    """Current embedding-model config + availability, for the Knowledge page's embedding picker."""
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
    can_download = False
    if installed is False and model in KNOWN_EMBEDDING_MODELS:
        from localm.netpolicy import network_mode
        if network_mode() != "off":
            can_download = held is None or scopes.grants(held, scopes.CONFIG_WRITE)

    # The three embedder readers go OFF THE EVENT LOOP, together. The docstring
    # above is right that this route never LOADS a model, and that is the trap:
    # each of loaded_dim / last_error / gpu_fallback_reason does `with _LOCK:`,
    # and get_embedder holds that same lock across a process spawn plus a native
    # load (up to 300s). This is the Knowledge page's poll, so on a cold server
    # it lands exactly while a first load is running - the same defect the hang
    # alarm caught on POST /api/embedding/warmup, at a second call site.
    #
    # READ, not PEEK: unlike the warm-up route, these values ARE this route's
    # answer, so there is nothing useful to do with a short timeout and no
    # honest reply to invent. Waiting for a load to finish was always this
    # route's behaviour and is kept exactly; what changes is that the wait now
    # costs the ONE request instead of the whole server. Every field is left as
    # it was, so no client contract moves with this fix.
    def _embedder_state():
        return (loaded_dim(), last_error() if may_answer else None,
                gpu_fallback_reason())

    try:
        dim, error, fallback = await run_in_threadpool_bounded(
            _embedder_state, timeout=READ_TIMEOUT_S)
    except ThreadCallTimeout as e:
        # Past READ_TIMEOUT_S the lock is not busy, it is wedged. Say so instead
        # of returning `dim: null, error: null`, which reads as "nothing loaded,
        # nothing wrong" - a could-not-check reported as a clean answer is the
        # rule-5 shape this whole class keeps producing.
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
    """One-time download of the CURRENTLY CONFIGURED embedding model, for when the network policy does not download it automatically (net_mode=ask blocks the lazy fetch - see embedder._download_known)."""
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
    from localm.netpolicy import network_mode
    if network_mode() == "off":
        raise HTTPException(
            409, "Network access is disabled (net_mode=off), which blocks even "
                 "an explicitly requested model download. Set net_mode to ask "
                 "or allow first.")
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
    """Select the embedding model, install it if it is an internal key not yet on disk (one-click setup - no terminal), then load-and-probe it so the user gets a clear answer: ready with its dimension, or a SPECIFIC reason it failed (e.g. it is not an embedding model)."""
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
        # Load + probe via the shared embedder (so it stays loaded for real use).
        # get_embedder swallows the load error but records it; surface it verbatim
        # so the user learns exactly why a wrong pick failed.
        line("Loading and testing the model…")
        # on_progress=line, not a bare get_embedder(): this call is the one that
        # can legitimately run for minutes (a VRAM-eviction wait plus the
        # isolated child's spawn and native load, each with its own 300s
        # window), and without the sink the job emitted NOTHING for that whole
        # time. That silence is half of what QA 2026-08-20 item 7 reported -
        # "no further event and no error" - and it is indistinguishable from a
        # wedge to anyone watching. get_embedder has announced its stages since
        # ADR-0004 Unit B; the warm-up route (/api/embedding/warmup) already
        # consumes them, and this route was simply throwing them away.
        # _emit_stage swallows anything the sink raises, so a push failure can
        # never turn a working load into a failed one.
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
    # Refuse a SECOND concurrent setup instead of queueing it behind the first
    # (QA 2026-08-20 item 7). Two of these jobs both call get_embedder(), whose
    # `with _LOAD_LOCK:` / `with _LOCK:` are BARE acquires with no timeout, so
    # the loser sits at its last emitted line ("Loading and testing the
    # model...") with no further event and no error for as long as the winner
    # takes - and, because each job also runs the pre-load VRAM swap check
    # (which round-trips through the event loop), the pair can hold unrelated
    # reads off the server the whole time. An unbounded silent wait is the
    # failure AGENTS.md rule 5 forbids; a 409 the caller can read is the same
    # answer, immediately. This is the shape every sibling long job on this
    # server already uses - runtime-update, comfy-setup/update, doctor.
    #
    # Check-then-act IS atomic here, and only because of where it sits: this is
    # an `async def` on the server's single event loop and there is NO `await`
    # between this check and start_fn below, so no other request coroutine can
    # interleave. Keep it that way - adding an await in this window silently
    # reopens the race (diff-review-discipline item 26).
    #
    # It does NOT serialise against another PROCESS (a terminal `localm
    # setup-embeddings`, or a second server on the same LOCALM_HOME); that is
    # the same, deliberately stated limit runtime.py's own 409 carries, and the
    # in-process case is the one the GUI can actually produce by double-click.
    if jobs.has_running("embed-setup"):
        raise HTTPException(
            409, "An embedding-model setup is already running. Wait for it to "
                 "finish (the Knowledge page shows its progress), then try again.")
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
    # No cap unless one was ASKED for. The 30 MB byte guard above is the real
    # memory bound and it already ran; a second character cap here only served
    # to hand the model a fraction of the document it was told it had. If the
    # text does not fit the model's context that is the engine's problem to
    # report honestly, not a reason to quietly shorten the input.
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
