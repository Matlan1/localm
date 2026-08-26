# SPDX-License-Identifier: AGPL-3.0-or-later
"""Execute a single job and return a result record.

``run_job(job, *, engine=None)`` runs the job's prompt and returns a dict:
    {status: "ok"|"error", output: str, error: str|None,
     started: float, finished: float}

It NEVER raises - any failure is caught and reported as an ``error`` result, so
a scheduler tick can safely run many jobs in a row.

task_kind "chat":  the prompt is run against the inference engine. A passed-in
    ``engine`` is reused; otherwise one is loaded via the model manager from the
    job's ``model`` (or the active/first registered model).
task_kind "coder": a coder Agent runs the prompt in the job's ``cwd`` with the
    job's ``scope`` and the current privacy mode. The coder path is best-effort:
    a full agentic run needs the coder extra installed and a working backend.
task_kind "rag":   the job's ``collection`` is re-synced against the folders it
    was indexed from (``Collection.resync``), picking up files added or changed
    since and flagging ones that vanished. Loads no chat model.

Results are explicit user data (the store saves them in every privacy mode), but
any session TRACE a run would leave (audit JSONL, transcripts) still honours
``effective_mode`` - the coder Agent is constructed with the resolved mode, and
the chat path writes no trace of its own.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

from localm.debuglog import logger
from localm.plugins.builtin.jobs.store import Job, JobStore

# Upper bound on how long the runner blocks its worker thread waiting for the server
# loop to complete a guarded shared-engine unload. Must stay above unload_one_model's
# own vram.wait_for_vram_release window, so reaching this bound means the server loop
# is wedged rather than an unload merely running long. On timeout the caller degrades
# to loading alongside rather than hanging.
_EVICT_TIMEOUT_S = 60.0

# Shared between the log line and the job's own output note below. Hedged, because
# this re-check cannot tell a revoked or expired key apart from an owner key that was
# rolled since the job was created.
_SHELL_DOWNGRADE_REASON = (
    "its authorization could not be re-confirmed at run time (the owning key "
    "may have been revoked or expired, or - if it was originally the owner "
    "key - simply rolled since this job was created)")


def run_job(job: Job, *, engine=None) -> dict:
    """Run *job* and return a result record. Dispatches on task_kind and never
    raises.

    When no *engine* is passed for a chat/memory job, the runner loads one itself
    and UNLOADS it again afterwards, so a sequence of headless runs does not stack
    model loads in VRAM. A shared engine is NEVER unloaded here - neither one passed
    in by the live server, nor the live server's shared engine that _load_engine may
    REUSE even for an engine=None call; only a genuinely fresh, runner-loaded engine
    is freed in the finally."""
    started = time.time()
    owned_engine = None
    try:
        eng = engine
        if job.task_kind in ("chat", "memory") and eng is None:
            # _load_engine reports whether it REUSED the live server's shared engine
            # (http_server._engine) or loaded a FRESH one. Only a fresh engine is
            # owned (and unloaded in the finally); a reused shared engine belongs to
            # the host. The verdict is taken from _load_engine rather than re-read
            # from _engine.
            eng, reused = _load_engine(job.model)   # may raise (model not found) -> caught below
            if eng is not None and not reused:
                owned_engine = eng          # we loaded it, so we unload it after the run
        if job.task_kind == "chat":
            output = _run_chat(job, engine=eng)
        elif job.task_kind == "coder":
            output = _run_coder(job, engine=engine)
        elif job.task_kind == "memory":
            output = _run_memory(job, engine=eng)
        elif job.task_kind == "rag":
            output = _run_rag(job)
        else:
            raise ValueError(f"unknown task_kind: {job.task_kind!r}")
        return {
            "status": "ok",
            "output": output,
            "error": None,
            "prompt": job.prompt,
            "task_kind": job.task_kind,
            "model": job.model,
            "started": started,
            "finished": time.time(),
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error": f"{type(e).__name__}: {e}",
            "prompt": job.prompt,
            "task_kind": job.task_kind,
            "model": job.model,
            "started": started,
            "finished": time.time(),
        }
    finally:
        if owned_engine is not None:
            _unload_engine(owned_engine)


def _evict_shared_engine_for_media(live) -> str:
    """Free the shared live-server chat engine to make VRAM room for this job's
    model, through the SAME guarded path the server's own unload uses.

    The engine belongs to the running server, so a raw ``live.unload()`` on this
    worker thread is unsafe two ways:
      * it ignores the in-flight-request PIN (``active_requests``): a chat may be
        generating on it right now, and unloading frees VRAM out from under that
        request while racing the native free; and
      * it runs OFF the event loop, so a concurrent request's ``get_engine()`` fast
        path can hand the being-freed engine back and pin it (the gguf backend
        clears ``.loaded`` only AFTER ``_llm.close()``, so the engine still reads
        loaded mid-free).
    ``unload_one_model`` closes both: it SKIPS a pinned engine (``active_requests``
    > 0 -> ``"in_use"``) and, run ON the loop, serializes with ``get_engine`` and
    the synchronous ``_pin``. So it is submitted to the server loop via
    ``run_coroutine_threadsafe`` and this thread blocks on the result. It also does
    its own VRAM-release wait on the loop, so no separate wait here.

    Returns the unload status (``"unloaded"`` / ``"in_use"`` / ``"already_unloaded"``),
    or ``"skipped"`` when the server loop is unreachable - in which case the shared
    engine is NOT raw-unloaded and is left resident: the job model loads alongside
    it. Every degrade is logged, not silent."""
    from localm.debuglog import logger as _dbg
    from localm.inference import http_server as _hs

    loop = getattr(_hs, "_server_loop", None)
    name = getattr(live, "display_name", None)
    if loop is None or not loop.is_running() or not name:
        _dbg.debug("jobs: cannot reach the server loop to evict the shared chat "
                   "engine safely; leaving it resident and loading the job model "
                   "alongside it (may be tight on VRAM)")
        return "skipped"
    try:
        fut = asyncio.run_coroutine_threadsafe(_hs.unload_one_model(name), loop)
        res = fut.result(timeout=_EVICT_TIMEOUT_S)
    except Exception as e:
        # The guarded unload could not complete (loop wedged, unload raised, or the
        # wait timed out). No raw-unload fallback: report, and let the caller load
        # alongside the still-resident engine.
        _dbg.debug("jobs: guarded shared-engine unload did not complete (%s); "
                   "loading the job model without evicting the live engine", e)
        return "error"
    status = res.get("status", "unloaded") if isinstance(res, dict) else "unloaded"
    if status == "in_use":
        _dbg.debug("jobs: the shared chat engine is serving a request (pinned), so "
                   "it was not evicted for this job; loading the job model alongside it")
    return status


def _unload_engine(eng) -> None:
    """Release a model the runner loaded itself, freeing VRAM for the next run.
    Best-effort, but a failure is surfaced (not silenced): a leaked model would
    accumulate across scheduled runs while the job still reported success."""
    unload = getattr(eng, "unload", None)
    if not callable(unload):
        return
    try:
        unload()
    except Exception as e:
        logger.warning("jobs: failed to unload the run's own engine: %s", e)


# --------------------------------------------------------------------------- #
#  chat                                                                        #
# --------------------------------------------------------------------------- #

def _run_chat(job: Job, *, engine=None) -> str:
    """Run the prompt through the inference engine and return the reply text.

    Scheduled chat jobs get the same web-search tool the interactive chat has; the
    bounded tool loop and the net_mode gating live in :mod:`webtool`. The chat path
    leaves no session trace of its own (no audit or transcript writes here), so it
    is privacy-safe regardless of mode; the explicit result is saved by the store
    like any other generated artifact."""
    eng = engine
    if eng is None:
        raise RuntimeError(
            "no inference engine available (pass one, or register a model)")
    from localm.plugins.builtin.jobs import webtool
    # Pin for the WHOLE tool-calling loop (run_chat_with_web can drive several
    # rounds), not per-round.
    from localm.inference.http_server import driving_engine
    with driving_engine(eng):
        return webtool.run_chat_with_web(eng, job.prompt)


def _load_engine(model: Optional[str]) -> "tuple[Optional[object], bool]":
    """Resolve an inference Engine for *model* (or the active/first registered
    model). Returns ``(engine, reused)``:

      * ``reused=True`` only when the returned engine IS the live server's shared
        engine (``http_server._engine``), which the runner must never unload. The
        fact is decided HERE, at the reuse branch, so ``run_job`` never re-derives
        ownership with a second, racy read of ``_engine``.
      * ``reused=False`` for a genuinely fresh engine the runner loaded itself (safe
        to unload after the run) - it is never registered in the server's engine
        table, so nothing else can reach or pin it.
      * ``(None, False)`` when no model can be resolved."""
    from localm.config import load_config, load_registry
    from localm.inference.engine import Engine
    from localm.model_manager import get_model_info, unregistered_model_error

    name = model

    # Re-check the job's model name at RUN time, not only at the API write: rows
    # persisted by an older build are still on disk, and the scheduler runs them
    # unattended.
    #
    # Runs FIRST, ahead of the live-engine reuse and VRAM branch below, which can
    # call _evict_shared_engine_for_media and unload the live chat engine before any
    # name is resolved.
    _bad = unregistered_model_error(name)
    if _bad:
        raise RuntimeError(_bad)

    try:
        from localm.inference.http_server import _engine as _live
        if _live is not None and _live.loaded:
            if not name or _live.display_name == name:
                return _live, True    # reuse the shared engine - no VRAM cost, no load
            
            # VRAM gate: unload the live engine if VRAM is tight. Uses
            # vram_capacity() (combined free across a configured multi-GPU split,
            # else the single-GPU vram_info() number), the same ceiling
            # switch_engine's eviction gate uses.
            from localm import vram as _vram
            from localm.discover import vram_capacity
            free = vram_capacity().get("free")
            est = _vram.media_estimate_bytes("chat")
            if _vram.should_swap_for_media(free, est):
                # Route the eviction through the guarded server-loop path instead of
                # a raw _live.unload() on this worker thread: unload_one_model honors
                # the in-flight pin and serializes with get_engine, and does its own
                # VRAM-release wait on the loop.
                _evict_shared_engine_for_media(_live)
    except Exception as e:
        # Best-effort live-engine reuse plus VRAM gate. Any failure here falls
        # through to loading a fresh engine below, with the cause logged.
        from localm.debuglog import logger as _dbg
        _dbg.debug("jobs: live-engine reuse / VRAM gate skipped (%s); "
                   "loading a fresh engine instead", e)

    if not name:
        cfg = load_config()
        name = cfg.get("default_model") or cfg.get("model")
    if not name:
        from localm.model_manager import is_auto_chat_eligible
        reg = load_registry()
        # Auto-pick the first chat-eligible model; skip a type='unknown' model. It
        # stays runnable when a job explicitly configures default_model/model above.
        name = next((n for n in sorted(reg) if is_auto_chat_eligible(reg[n])), None)
    if not name:
        return None, False
    info = get_model_info(name)
    if info is None:
        raise RuntimeError(f"model not found: {name}")
    model_path, display_hint = info
    eng = Engine(str(model_path), display_name=(name if model else display_hint))
    eng.load()
    return eng, False   # freshly loaded by the runner - owned, safe to unload after


# --------------------------------------------------------------------------- #
#  memory (auto-synthesis)                                                     #
# --------------------------------------------------------------------------- #

def _run_memory(job: Job, *, engine=None) -> str:
    """Distil durable user facts from recent sessions into the assistant memory
    file, using the model. The privacy gate lives inside synthesize_memory (it
    skips with a clear status in privacy mode, never a silent success). Returns a
    human-readable summary saved as the job result."""
    # Import the memory plugin's synthesizer directly, which resolves the
    # bundled-store source even when the memory plugin is not installed or enabled;
    # the privacy and write gates inside synthesize_memory still apply. Valid only
    # while synthesize_memory stays module-level-stateless (fresh store per call,
    # shared state on disk/config). Guarded, so a memory module that cannot import
    # degrades to a clear job result.
    try:
        from localm.plugins.builtin.memory.plug import synthesize_memory
    except Exception as e:
        return f"Memory is unavailable, so nothing was consolidated: {e}"
    eng = engine
    if eng is None:
        raise RuntimeError(
            "no inference engine available (pass one, or register a model)")

    from localm.textnorm import strip_think

    # Track when a reply was ALL reasoning (empty after the strip), so the no-facts
    # result carries a caveat instead of reading as a clean no-op.
    state = {"empty_replies": 0}

    def complete(prompt: str) -> str:
        raw = "".join(
            eng.chat_stream([{"role": "user", "content": prompt}])).strip()
        # strip_think: memory must never ingest the reasoning channel.
        text = strip_think(raw).strip()
        if raw and not text:
            state["empty_replies"] += 1
        return text

    # Pin the engine busy and touch its activity clock for the WHOLE synthesis pass
    # (synthesize_memory can call complete() several times), so idle-unload cannot
    # unload it mid-run.
    from localm.inference.http_server import driving_engine
    with driving_engine(eng):
        result = synthesize_memory(complete)
    if result.get("status") == "skipped":
        return f"memory synthesis skipped ({result.get('reason')})"
    # A contradiction to a saved (user-typed) fact waits as a suggested correction to
    # review in the memory panel, never silently applied or dropped. Reports the TOTAL
    # pending, not just this run's new ones.
    pending = result.get("pending", result.get("proposed", 0))
    suffix = ("\n%d suggested correction(s) to your saved facts await review in the "
              "memory panel" % pending) if pending else ""
    facts = result.get("facts") or []
    if not facts:
        if state["empty_replies"]:
            return ("memory synthesis: no facts extracted - the model produced "
                    "only reasoning output (%d reply/replies were empty after "
                    "removing the think channel; likely truncated by the "
                    "completion limit)" % state["empty_replies"]) + suffix
        return "memory synthesis: no new durable facts found" + suffix
    return ("memory synthesis: added %d fact(s):\n" % result["added"]) + \
           "\n".join(f"- {f}" for f in facts) + suffix


# --------------------------------------------------------------------------- #
#  rag (scheduled folder re-sync)                                              #
# --------------------------------------------------------------------------- #

def _rag_embed_fn():
    """The embedding callable a re-sync indexes with, or None when no embedding
    model is available (the collection then indexes lexical-only, exactly like
    ``rag add`` without ``--embed``).

    Resolved in-process from the shared embedder singleton rather than over HTTP,
    so a re-sync works under ``localm job run`` with no server up. ``embed_texts``
    is NOT used: it returns None when unavailable, and ``add_paths`` cannot consume
    that.

    Runs on the runner's worker thread, never the event loop: resolving the
    embedder can trigger a VRAM swap, which must not block the loop."""
    try:
        from localm.inference.embedder import get_embedder
        emb = get_embedder()
        return emb.embed if emb is not None else None
    except Exception as e:
        logger.debug("jobs: rag embedder resolution failed (%s); "
                     "re-syncing lexical-only", e)
        return None


def _run_rag(job: Job) -> str:
    """Re-sync a knowledge collection against the folders it was indexed from.

    Reuses the incremental index wholesale (``Collection.resync`` -> the same
    ``add_paths`` hash-skip path an interactive add uses), so an unchanged file
    costs a hash and nothing else, and reports what the run actually did.

    No CHAT model is loaded: a re-sync needs neither the format tie-break nor
    image description. It does resolve the shared EMBEDDER, which on a tight card
    can itself evict a resident chat model (embedder._maybe_swap_for_embedder ->
    vram.evict_chat_for_embedder), so this must run off the event loop.

    Confinement: the run always passes ``indexing_policy()``, so it never indexes a
    path an interactive add would refuse, including a root that has since fallen
    outside the owner's allowed folders. Deletion is non-destructive by design (see
    ``Collection.resync``).

    Privacy: a collection is explicit user data and is written in every session
    mode; this path adds no session trace of its own."""
    from localm.rag import Collection, CollectionLockedError
    from localm.rag.store import indexing_policy

    name = (job.collection or "").strip()
    if not name:
        raise RuntimeError("this rag job has no collection configured")
    coll = Collection(name)
    if not coll.exists():
        raise RuntimeError(f"no such collection: {name}")
    if not coll.roots() and not coll.documents():
        return (f"'{name}' has nothing to re-sync: no indexed folders and no "
                f"documents. Index a folder first "
                f"(localm rag add {name} <folder>).")

    had_vectors = bool(coll.stats().get("has_vectors"))
    embed_fn = _rag_embed_fn()
    # The same configured-name lookup the CLI's `rag add`/`rag resync` use, so a
    # scheduled re-sync that embeds also leaves the model on record.
    model_name = None
    if embed_fn is not None:
        from localm.config import load_config
        from localm.inference.embedder import DEFAULT_EMBEDDING_MODEL
        model_name = str(load_config().get("embedding_model")
                          or DEFAULT_EMBEDDING_MODEL).strip()
    lines: list = []
    try:
        result = coll.resync(embed_fn=embed_fn, policy=indexing_policy(),
                             model_name=model_name,
                             on_progress=lines.append)
    except CollectionLockedError as e:
        # Another localm process (or a hand-run `localm rag add|resync`) holds this
        # collection. The scheduled tick waits a bounded time and then stands down,
        # recorded as an error rather than a quiet success. The next tick re-walks the
        # same folders.
        raise RuntimeError(f"{e} The next scheduled run will pick this up.") from e
    return _format_rag_result(name, result, lines,
                              embedded=embed_fn is not None,
                              had_vectors=had_vectors)


def _format_rag_result(name: str, result: dict, lines: list, *,
                       embedded: bool, had_vectors: bool) -> str:
    """Render a re-sync result as the job's output.

    Every degrade is stated, not implied: a skipped root, a flagged-missing
    document, a per-file failure, a vectors.json the store found corrupt or stale,
    and new documents indexed WITHOUT embeddings into a collection that had
    semantic search, which pushes vector coverage down and can drop the whole
    collection to BM25."""
    out = [f"re-synced '{name}': {result['added']} added, "
           f"{result['updated']} updated, {result['skipped']} unchanged - "
           f"{result['chunks']} chunks over {len(result['roots'])} folder(s)"]
    if result["missing"]:
        out.append(f"{len(result['missing'])} document(s) are no longer on disk. "
                   f"They are FLAGGED, not removed, so nothing is lost if this "
                   f"was temporary; remove them from the collection yourself when "
                   f"you are sure:")
        out.extend(f"  missing: {p}" for p in result["missing"][:10])
    if result["restored"]:
        out.append(f"{len(result['restored'])} previously missing document(s) "
                   f"are back.")
    for r in result["unavailable_roots"] + result["blocked_roots"]:
        out.append(f"skipped folder {r['root']}: {r['reason']} "
                   f"(nothing under it was indexed, flagged, or removed)")
    if result["failed"]:
        out.append(f"{len(result['failed'])} file(s) failed:")
        out.extend(f"  {f['path']}: {f['error']}" for f in result["failed"][:10])
    if result.get("vector_degrade_reason"):
        out.append(
            f"NOTE: semantic search is degraded on this collection: "
            f"{result['vector_degrade_reason']}. The stored vector index was "
            f"left in place, not deleted - rebuild it with "
            f"'localm rag repair {name} --embed'.")
    if not embedded and had_vectors and (result["added"] or result["updated"]):
        out.append(
            "NOTE: no embedding model was available, so the newly indexed "
            "documents have no vectors while the rest of this collection does. "
            "Semantic search degrades as that gap grows - run "
            "'localm setup-embeddings' and re-sync again to close it.")
    # The per-file progress lines carry store-level degrades (an embedder that raised
    # mid-run, a non-finite vector) that would otherwise be dropped.
    degrades = [t for t in lines if t.startswith("embeddings")]
    out.extend(f"NOTE: {t}" for t in dict.fromkeys(degrades))
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#  coder (best-effort)                                                         #
# --------------------------------------------------------------------------- #

def _shell_still_authorized(job: Job) -> bool:
    """Re-validate a shell-opt-in job's authorization at RUN time, so a revoked or
    expired key cannot keep an unattended scheduled job running with elevated access
    forever.

    ``job.owner`` is the sha256 key hash ``principal_id()`` stamps at creation
    (store.py) - the same value ``auth.key_hash_live()`` checks for a cookie
    session. Two cases need no re-check: a job with no owner (no privileged key was
    needed at creation, so there is no key whose liveness matters), and a job
    created by the OWNER key itself (the owner key is not a keystore entry and is
    not revocable or expirable the way a scoped key is). Any other owner hash must
    still resolve to a live (unrevoked, unexpired) keystore key, or the run is
    downgraded to restricted.

    The owner case is decided by the ``owner_is_owner_key`` flag STAMPED AT
    CREATION, not by comparing key values here: once the owner ROTATES the key, a
    value comparison against the new key fails and ``key_hash_live`` of the old hash
    says "not live", which would strip elevated access from the owner's own
    scheduled jobs. The distinction cannot be recovered at run time: revocation
    deletes the keystore record, so a revoked scoped key and a rotated-away owner
    key both hash to nothing. The key-value comparison is kept only as the
    back-compat path for jobs persisted before the flag existed (they load with it
    False)."""
    if job.owner is None:
        return True
    if getattr(job, "owner_is_owner_key", False):
        return True
    from localm.auth import _hash_key, _legacy_owner_identity, get_api_key, key_hash_live
    owner_key = get_api_key()
    # Both the CURRENT derived identity and the LEGACY unsalted digest count as a
    # match. This is an IDENTITY comparison against an already-resolved owner key, not
    # an authentication step: the owner key itself is verified by plaintext compare
    # against auth.key. A match stamps owner_is_owner_key below, so each job pays this
    # once.
    if owner_key and job.owner in (_hash_key(owner_key),
                                   _legacy_owner_identity(owner_key)):
        # The owner key still holds the value the job was stamped with, so this run
        # proves the job is the owner's. Recorded now; after this the flag
        # short-circuits above, making it a one-time write per job.
        _remember_owner_key_job(job)
        return True
    return key_hash_live(job.owner)


def _cwd_trusted(job: Job) -> bool:
    """True when this job's CREATOR was allowed to choose an arbitrary working
    directory, i.e. the owner (or a ``coder:full`` / ADMIN key) rather than a plain
    ``jobs``-scoped key.

    Re-derived at RUN time rather than stamped, so a key whose scopes are narrowed,
    or which is revoked or expires, loses its arbitrary-cwd freedom on the next
    tick. Four positives, in cost order:

    - no owner at all: a tokenless / open-mode creation, which IS the loopback
      owner (``_caller_can_allow_shell`` returns True with no key configured), so
      there is no lesser principal to confine;
    - the ``owner_is_owner_key`` stamp: the owner key or an owner session created
      it. This is the one case re-derivation cannot reach, because after a key roll
      the recorded hash matches nothing;
    - the recorded principal still IS the owner key by value (covers a job created
      before the stamp existed, while the key has not rolled);
    - the recorded principal is a live keystore key holding ADMIN or coder:full.

    Anything else - notably a live ``jobs``-only key, and any hash that resolves to
    nothing - is untrusted, and ``_run_coder`` confines it to the project root.
    Fails CLOSED: ``scopes_for_key_hash`` returns None when the keystore cannot be
    read, and None is not a grant."""
    if job.owner is None:
        return True
    if getattr(job, "owner_is_owner_key", False):
        return True
    from localm import scopes as S
    from localm.auth import (_hash_key, _legacy_owner_identity, get_api_key,
                             scopes_for_key_hash)
    owner_key = get_api_key()
    if owner_key and job.owner in (_hash_key(owner_key),
                                   _legacy_owner_identity(owner_key)):
        return True
    held = scopes_for_key_hash(job.owner)
    return held is not None and (S.ADMIN in held or S.CODER_FULL in held)


def _remember_owner_key_job(job: Job) -> None:
    """Persist ``owner_is_owner_key`` on a legacy job just proven to be the owner's
    (best-effort).

    Failing to persist is not fatal - this run is authorized either way, and the
    next run re-proves it the same way - so it must not break the job. It is not
    silenced either: a job that keeps failing to persist stays exposed on the next
    key roll, and that has to be discoverable.
    """
    job.owner_is_owner_key = True
    try:
        JobStore().update(job.id, owner_is_owner_key=True)
    except (KeyError, OSError, RuntimeError, ValueError) as e:
        logger.debug("jobs: could not persist the owner-key stamp for job %s "
                     "(%s); it will be re-derived on the next run", job.id, e)


def _run_coder(job: Job, *, engine=None) -> str:
    """Run a coder Agent for the prompt in the job's cwd. Best-effort: requires
    the coder plugin and a reachable backend. Honours the current privacy mode
    for any session trace the agent would write.

    The agent always talks to an OpenAI-compatible HTTP endpoint, so a job run
    needs a localm server (``self_url``) reachable; without one this raises and
    the run is recorded as an error (never crashing the tick)."""
    from localm.audit import effective_mode
    from localm.plugins.builtin.jobs.store import cwd_unc_error

    # AUTHORITATIVE check, re-validated at RUN time and not only at the write: a row
    # persisted by an older build is still on disk, and the autonomous scheduler tick
    # runs it unattended. Shares cwd_unc_error's wording with the write-time checks.
    # Checked on the RAW string, before any Path object is built: Path.is_dir() and
    # .resolve() below dial SMB and auto-authenticate for a UNC target on Windows.
    _bad_cwd = cwd_unc_error(job.cwd)
    if _bad_cwd:
        raise RuntimeError(_bad_cwd)

    # WHO chose this cwd, not just what SHAPE it is. cwd_unc_error above rejects
    # UNC/device syntax for everyone and says nothing about authority. As in the coder
    # route for a restricted caller, the requested cwd is ignored and the project root
    # forced.
    cwd_confined = not _cwd_trusted(job)
    if cwd_confined:
        from localm.instances import resolve_root_dir
        # The same project root the coder route uses via app.state.root_dir; the
        # runner has no request or app to read it from, so it resolves it the same way
        # that value was produced.
        requested, job_cwd = job.cwd, resolve_root_dir()
    else:
        requested, job_cwd = job.cwd, job.cwd

    cwd = Path(job_cwd or ".").expanduser()
    if not cwd.is_dir():
        raise RuntimeError(f"coder cwd is not a directory: {job_cwd}")
    if cwd_confined and requested and Path(requested).expanduser() != cwd:
        # Never a silent degrade: the job asked to run somewhere it is not allowed to,
        # so the substitution is logged and carried in the job's own output.
        logger.warning(
            "job %r (%s) requested a working directory its creating key is not "
            "allowed to choose; running in the project root instead. Re-create "
            "the job with the owner key or a coder:full key to choose a "
            "directory.", job.name, job.id)

    backend = _coder_backend(job)
    mode = effective_mode("coder")

    # Safe-by-default: an unattended scheduled run has nobody to approve a destructive
    # tool, so it runs RESTRICTED (read plus confined edit, no run_shell, no network,
    # no sub-agents) unless the owner explicitly opted this job into the full
    # shell-capable coder. Restricted hard-refuses run_shell and fetch_url at
    # dispatch. A stored allow_shell flag is re-validated against the OWNING key's
    # live state on every run.
    allow_shell = bool(getattr(job, "allow_shell", False))
    downgraded = allow_shell and not _shell_still_authorized(job)
    restricted = not allow_shell or downgraded
    if downgraded:
        # Never a silent degrade: the downgrade is reported in the log AND in the
        # run's own output. The run still proceeds restricted; a downgrade does not
        # fail the whole job.
        logger.warning(
            "job %r (%s) opted into shell, but %s; running RESTRICTED (no "
            "run_shell). Trigger a manual run or any edit as the owner to "
            "repair it automatically, or re-create it with a live key.",
            job.name, job.id, _SHELL_DOWNGRADE_REASON)

    from localm.plugins.coder.agent import Agent
    agent = Agent(
        backend,
        cwd.resolve(),
        auto_approve=True,        # unattended scheduled run: no interactive prompts
        mode=mode,
        scope=job.scope,
        restricted=restricted,
    )
    try:
        out = (agent.run_task(job.prompt) or "").strip()
        notes = []
        if downgraded:
            notes.append(f"[jobs] This job opted into shell execution, but "
                         f"{_SHELL_DOWNGRADE_REASON}, so it ran RESTRICTED: no "
                         f"run_shell. Trigger a manual run or any edit as the "
                         f"owner to repair it automatically, or re-create it "
                         f"with a live key.")
        if cwd_confined and requested:
            # Reported the same way as the note above. The requested path is echoed
            # back only in this job's own result, to the job's owner, never onto a
            # shared surface.
            notes.append("[jobs] This job requested a working directory its "
                         "creating key is not allowed to choose, so it ran in "
                         "the project root instead. Re-create it with the owner "
                         "key or a coder:full key to choose a directory.")
        if notes:
            joined = "\n\n".join(notes)
            out = f"{joined}\n\n{out}" if out else joined
        return out
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception as e:
                # Surface, rather than silence, a failed cleanup. Best-effort: the job
                # result is unaffected.
                logger.warning("coder agent cleanup failed: %s", e)


def _coder_backend(job: Job):
    """Build the HTTP backend the coder Agent talks to. Points at this machine's
    localm server.

    URL resolution, most-authoritative first: LOCALM_SELF_URL (the live server
    publishes its OWN bind coordinates here at scheduler start, so an auto-bumped
    port is honoured), else the configured port. In open mode any api_key is
    accepted; in keyed mode the launcher injects LOCALM_API_KEY, so that is
    preferred over the open-mode placeholder."""
    import os

    from localm.plugins.coder.backends.http import HTTPBackend

    self_url = os.environ.get("LOCALM_SELF_URL")
    if not self_url:
        # The instance registry knows the address AND the port the server is really
        # on; the configured port is only a guess, and the IPv4 loopback is wrong for
        # an IPv6-bound server. The guess is used only when there is no registry entry.
        try:
            from localm import instances
            from localm.config import home_dir
            entry = instances.attach_target(home_dir(),
                                            instances.resolve_root_dir())
            self_url = entry.get("base_url") if entry else None
        except Exception:
            self_url = None
    if not self_url:
        from localm.config import load_config
        port = load_config().get("port", 8642)
        self_url = f"http://127.0.0.1:{port}/v1"
    api_key = os.environ.get("LOCALM_API_KEY") or "localm"
    # self-connection: grammar sampling available
    return HTTPBackend(self_url, model=job.model or "localm", api_key=api_key,
                       localm_server=True)
