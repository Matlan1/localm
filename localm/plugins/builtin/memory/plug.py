# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory plugin: durable chat memory (extracted from the chat plugin).

A small structured store of durable facts about the user (localm/memory), recalled
by recency + importance + relevance and injected server-side into the system
message by the inlet hook, plus unattended background consolidation that grows the
store after a turn. Owns:

  GET/PUT/POST/PATCH/DELETE /api/memory[...]   - the memory manager surface
  POST /api/memory/consolidate                 - manual "distil now" trigger
  GET /api/memory/forgotten                    - list archived/forgotten records
  POST /api/memory/forgotten/{id}/restore       - recover one back into the store

This is an OPT-IN plugin (off by default): install + enable it to turn memory on.
Disabling it removes the recall + consolidation hooks and 404s the routes, so chat
runs fine without it - the only effect of "memory off" is no recall and no growth.

Privacy: memory is FULLY OFF in privacy mode - no recall AND no writes. This is a
stronger contract than the rest of the app ("no new traces"): in privacy mode the
inlet injects nothing at all, so past-session facts never reach the model. Every
write (add/edit/delete/consolidation/migration) is gated too. The two knobs
`memory_enabled` (recall) and `memory_auto_consolidate` (auto-grow) are the finer
controls beneath the plugin's own enable/disable master switch.

Chat memory is OWNER-scoped in v1: the kernel chat pipeline carries no principal
and localm is single-user, so all chat turns share the "owner" namespace (matching
the old global chat-memory.md). The memory library supports full (principal, agent,
scope_key) isolation, exercised by the coder and the library tests.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading as _threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_router = APIRouter()

_MEMORY_MAX = 64_000                 # characters - keep injection bounded
_OWNER = "owner"

# The plugin host, stashed at register(). The manual POST /api/memory/consolidate
# route and the auto-consolidate pass resolve a LIVE engine handle on every use via
# host.engine(), so a later model switch is picked up instead of pinning to whatever
# was loaded at register() time. None until wired (headless / api-mode).
_HOST = None


def _live_engine():
    """The inference engine, resolved LIVE via the stashed host on every call - see
    _HOST above. None when no plugin host is wired, or no engine is currently live."""
    return _HOST.engine() if _HOST is not None else None


# ------------------------------------------------------------------ #
#  Memory store helper                                               #
# ------------------------------------------------------------------ #

def _chat_store(principal: str | None = None):
    from localm import memory as _mem
    return _mem.open_store(principal, "chat", "", root=_memory_root())


def _ctx_principal(ctx) -> str | None:
    """The memory-namespace principal for a chat-pipeline ctx, mirroring
    memory_principal() on the request-based paths: an ADMIN-scoped (owner) caller
    collapses to the shared "owner" namespace (None -> principal_of maps to
    "owner"); a non-owner scoped key keeps its own key-hash namespace.

    Every WRITE path (memory_get/put/append/patch/delete + memory_consolidate) and
    the auto-consolidate OUTLET collapse ADMIN->owner, so the recall INLET MUST
    resolve the SAME namespace here: reading the raw ctx.principal (the key hash)
    would, in protected mode, have the owner write into "owner" while recall reads
    the per-key-hash namespace and injects none of the owner's saved memories.
    Tolerates a missing/None ctx: getattr defaults keep it owner-scoped."""
    from localm import scopes as _scopes
    if _scopes.ADMIN in (getattr(ctx, "scopes", ()) or ()):
        return None
    return getattr(ctx, "principal", None)


def _request_principal(request: Request | None) -> str | None:
    """The principal-resolution half of the request/store snippet shared by
    every /api/memory* route."""
    from localm.inference.http_server import memory_principal
    return memory_principal(request) if request is not None else None


def _request_store(request: Request | None):
    """The principal-resolution + store-open snippet shared verbatim by every
    mutating /api/memory* route."""
    return _chat_store(_request_principal(request))


def _require_writable() -> None:
    """The shared '_persist_enabled() guard -> 403' check every mutating
    /api/memory* route applies, with one wording for all five."""
    if not _persist_enabled():
        raise HTTPException(
            403, "Memory writes are off in privacy mode (no new traces). "
            "Set mode/chat_mode to 'log' or 'full' to enable them.")


class MemoryUpdate(BaseModel):
    text: str


class MemoryAppend(BaseModel):
    text: str


class MemoryPatch(BaseModel):
    text: str | None = None
    importance: float | None = None


# ------------------------------------------------------------------ #
#  Shared helpers (paths resolved at request time)                    #
# ------------------------------------------------------------------ #

def _home() -> Path:
    from localm.config import home_dir
    return home_dir()


def _persist_enabled() -> bool:
    """Memory recall AND writes are off in privacy mode (the default). Unlike the
    rest of the app, memory treats privacy as FULLY off: this gates recall too, so
    no past facts are injected in privacy mode (not just "no new traces")."""
    from localm.audit import SessionMode, effective_mode
    return effective_mode("chat") != SessionMode.PRIVACY


def _recall_enabled() -> bool:
    """The `memory_enabled` knob: recall on/off while the plugin is enabled."""
    try:
        from localm.config import load_config
        return bool(load_config().get("memory_enabled", True))
    except Exception:
        return True                                # config unreadable -> default on


def _recall_in_privacy(surface: str) -> bool:
    """Whether the user opted into READ-ONLY memory recall in privacy mode for
    *surface* ('chat' or 'coder'). Off by default so privacy stays fully inert; the
    master switch AND the per-surface switch must both be on. Never permits a
    WRITE - only reading existing memories into the prompt."""
    try:
        from localm.config import load_config
        cfg = load_config()
        return bool(cfg.get("memory_recall_in_privacy")
                    and cfg.get(f"memory_recall_in_privacy_{surface}", True))
    except Exception:
        return False


def _memory_root() -> Path:
    return _home() / "memory"




def _embed_fn():
    """The embedding callable memory uses for semantic (vector) recall, or None
    when no embedding model is available (recall + consolidation then fall back to
    lexical BM25). Cheap after the first call - the embedder is a cached, shared
    singleton (localm.inference.embedder)."""
    try:
        from localm.inference.embedder import get_embedder
        emb = get_embedder()
        return emb.embed if emb is not None else None
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory _embed_fn resolution failed: %s", e)
        return None


def _embedder_download_status(request: Request | None) -> dict:
    """Whether semantic (vector) recall is degraded to lexical-only for lack of
    an installed embedding model, and whether THIS caller could fetch it with
    one click via the SAME one-time action the Knowledge page offers (POST
    /api/rag/embedding/download - reused here, not duplicated). Mirrors
    rag.plug.rag_embedding_status's can_download gate (a known internal key,
    not yet on disk, net_mode not off unless downloads are exempted while off,
    caller holds config:write) so the two surfaces never disagree about
    whether the button would work - PLUS two checks that gate is not itself
    exposed to: the download route lives on the rag PLUGIN's router, which
    is (a) absent entirely (404) whenever rag is not installed/enabled, and
    (b) gated on the "rag" SCOPE at the router-mount level, checked before
    the route's own config:write check ever runs - a caller scoped to
    "memory"+"config:write" but not "rag" would 403 there regardless of
    holding config:write. Withholds the model name (returns embedder_model:
    None) whenever can_download is False, so a caller who could not act on
    it is not told what it is either."""
    try:
        from localm import scopes
        from localm.config import load_config
        from localm.inference.embedder import (
            DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS,
            resolve_embedding_model_path)
        from localm.netpolicy import downloads_allowed_when_off, network_mode
        import localm.inference.http_server as _hs
        model = str(load_config().get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        if (model in KNOWN_EMBEDDING_MODELS
                and not resolve_embedding_model_path(allow_download=False)
                and (network_mode() != "off" or downloads_allowed_when_off())):
            rag_mounted = False
            if request is not None:
                from starlette.routing import NoMatchFound
                try:
                    request.app.router.url_path_for("rag_embedding_download")
                    rag_mounted = True
                except NoMatchFound:
                    rag_mounted = False
            held = _hs.caller_scopes(request) if request is not None else None
            if (rag_mounted
                    and (held is None or scopes.grants(held, "rag"))
                    and (held is None or scopes.grants(held, scopes.CONFIG_WRITE))):
                return {"can_download_embedder": True, "embedder_model": model}
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory embedder-download hint skipped: %s", e)
    return {"can_download_embedder": False, "embedder_model": None}


async def _off_loop(fn):
    """Run a blocking store operation OFF the server event loop, mapping a
    contended namespace to 409.

    Memory writes take a cross-process lock, so another process (a `localm memory
    ...` command, another instance) can legitimately hold the namespace. That is a
    recoverable conflict - the store refused and changed NOTHING - so it surfaces
    as 409 with the holder named, the same way the rag routes surface it.

    A memory write resolves the shared embedder (via _embed_fn -> get_embedder),
    which can trigger a VRAM swap (vram.evict_chat_for_embedder). That swap must
    NOT run on the event-loop thread: it blocks on a coroutine the loop itself has
    to execute. Offloading to the default executor keeps the loop free and lets the
    eviction complete while the store write / embedder load runs.
    """
    from localm.rag.collection_lock import CollectionLockedError
    try:
        return await asyncio.get_running_loop().run_in_executor(None, fn)
    except CollectionLockedError as e:
        raise HTTPException(409, str(e))


def _legacy_memory_file() -> Path:
    return _home() / "chat-memory.md"


def _read_memory() -> str:
    """The legacy flat chat-memory.md text (migration source).

    Returns "" ONLY when the file is genuinely ABSENT. Raises OSError when it
    EXISTS but cannot be read (locked / IO error): the caller MUST distinguish
    the two, or ``_migrate_legacy`` would write its permanent
    ``.legacy-imported`` marker having imported nothing."""
    p = _legacy_memory_file()
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.warning(
            "legacy chat-memory.md exists but could not be read (%s); NOT treating "
            "it as empty - migration retries next start, recall falls back", e)
        raise


def _strip_bullet(line: str) -> str:
    """Strip any leading list markers (a model may emit '- ', '* ', or even a
    nested '- - ') and surrounding whitespace, leaving the bare fact."""
    return re.sub(r"^[\s\-*]+", "", line).strip()


def _legacy_bullets() -> list:
    """Bare fact lines parsed out of the legacy chat-memory.md (bullets stripped)."""
    return [_strip_bullet(ln) for ln in _read_memory().splitlines()
            if _strip_bullet(ln)]


def _migrate_legacy(store) -> None:
    """Import the legacy flat chat-memory.md into the structured store ONCE.

    Gated on the chat session mode (privacy skips - a migration materialises new
    durable records, which is a write) and on a per-namespace marker file so it
    never re-imports. Best-effort: a failure is logged, never fatal, and never
    reported as a success it did not perform.

    This batches several ``add(..., save=False)`` calls into ONE save, so it must
    hold the namespace lock across the WHOLE batch (a reload + loop + one final
    save). The per-call add()/delete() lock is not enough: it serialises each
    individual append, not the read-then-decide-then-write-once shape of a
    batch."""
    marker = store.path.with_suffix(".legacy-imported")
    if marker.exists() or not _persist_enabled():
        return
    try:
        from localm.memory import MemoryRecord
        ef = _embed_fn()
        with store.lock():
            store._load()
            existing = {r.text.casefold() for r in store.all()}
            added = False
            for bullet in _legacy_bullets():
                if bullet.casefold() in existing:
                    continue
                store.add(MemoryRecord(text=bullet, kind="semantic",
                                       source="import", importance=0.7),
                          embed_fn=ef, save=False)
                existing.add(bullet.casefold())
                added = True
            if added:
                store._save()                    # one batch write
    except Exception as e:                        # never break chat on migration
        from localm.debuglog import logger
        logger.debug("chat memory legacy migration failed: %s", e)
        return                                    # do NOT mark done - retry next start
    # The import ran (or there was nothing new). Mark it done so it is not re-scanned
    # every start. A marker-write failure is logged rather than reported as 'skipped';
    # the casefold dedup above makes a re-run next start harmless.
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("1", encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.debug(
            "chat memory legacy migration %s but its completion marker could not "
            "be written (%s); it will re-run harmlessly (dedup) until it persists",
            "imported records" if added else "found nothing new", e)


def _item(rec) -> dict:
    return {"id": rec.id, "text": rec.text, "importance": rec.importance,
            "source": rec.source, "kind": rec.kind, "uses": rec.uses,
            "updated": rec.updated}


def _correction_item(c, target) -> dict:
    """A pending supersede proposal rendered for the memory modal: what it wants to
    change and how stale the targeted fact is (its last-confirmed timestamp), so the
    user can accept or reject it."""
    return {"id": c.id, "action": c.action, "proposed_text": c.proposed_text,
            "target_id": c.target_id, "target_text": c.target_text,
            "confidence": c.confidence, "created": c.created,
            "target_updated": getattr(target, "updated", None)}


def _corrections_payload(store) -> list:
    """Pending corrections for GET /api/memory, each paired with its (still-present)
    target record so the modal can show the was/now and the fact's staleness."""
    corrs = store.corrections()                 # already filtered to live targets
    by_id = {r.id: r for r in store.all()}
    return [_correction_item(c, by_id.get(c.target_id)) for c in corrs]


def _forgotten_item(entry: dict) -> dict:
    """A forgotten/archived record rendered for the recovery surface."""
    return {"id": entry.get("id"), "text": entry.get("text", ""),
            "importance": entry.get("importance"), "source": entry.get("source"),
            "kind": entry.get("kind"), "forgotten_at": entry.get("forgotten_at")}


def _rendered_text(store) -> str:
    """The store's facts as a markdown bullet list (what the memory modal textarea
    shows). Falls back to the legacy flat file when the structured store is still
    empty so existing memory stays visible."""
    recs = store.all()
    if recs:
        return "\n".join(f"- {r.text}" for r in recs)
    try:
        return _read_memory().strip()
    except OSError:
        # Unreadable legacy file (already WARN-logged in _read_memory): show an
        # empty textarea rather than 500 the memory modal. The migration path still
        # refuses to mark done.
        return ""


# ------------------------------------------------------------------ #
#  Memory manager routes (/api/memory*)                               #
# ------------------------------------------------------------------ #

@_router.get("/api/memory")
async def memory_get(request: Request = None):
    store = _request_store(request)
    writable = _persist_enabled()
    if writable:
        # Off the loop: _migrate_legacy resolves the embedder, never on-loop.
        await _off_loop(lambda: _migrate_legacy(store))
    # Corrections are only surfaced when writes are allowed: accept/reject need a
    # write, and store.corrections() prunes stale entries as a side effect, which must
    # never run in privacy mode. Privacy mode returns an empty list without touching
    # the sidecar.
    return {"text": _rendered_text(store), "writable": writable,
            "items": [_item(r) for r in store.all()],
            "corrections": _corrections_payload(store) if writable else [],
            "path": str(store.path),
            **_embedder_download_status(request)}


@_router.put("/api/memory")
async def memory_put(req: MemoryUpdate, request: Request = None):
    """Bulk-edit: the modal textarea is the authoritative user memory. DIFF-AWARE:
    a line matching an existing record's text keeps that record as-is; only new
    lines become new user records; omitted lines are deletes.

    The existing/records diff below is a snapshot-then-decide-then-write sequence,
    so this route holds the namespace lock across the whole thing: a concurrent
    POST /api/memory/append (e.g. a second browser tab) landing between the
    snapshot and store.replace() would otherwise be discarded by replace()'s
    whole-namespace overwrite."""
    _require_writable()
    from localm.memory import MAX_TEXT_LEN, N_MAX, MemoryRecord
    if len(req.text) > _MEMORY_MAX:
        raise HTTPException(413, "Memory too large (max 64k chars)")
    facts = [_strip_bullet(ln)[:MAX_TEXT_LEN]
             for ln in req.text.splitlines() if _strip_bullet(ln)]
    # Reject over-cap writes at the door rather than accept facts the next prune would
    # hard-delete.
    if len(facts) > N_MAX:
        raise HTTPException(
            413, f"Too many memory records ({len(facts)}); the store keeps at "
            f"most {N_MAX}. Trim the list before saving.")
    store = _request_store(request)

    # The whole snapshot-decide-write block runs off the loop: store.replace ->
    # _embed_fn -> get_embedder must not resolve the embedder on the loop thread.
    # Keeping migrate + lock + diff + replace in one executor call preserves the
    # atomicity the lock provides (a concurrent append cannot land mid-diff).
    def _do_put():
        _migrate_legacy(store)
        with store.lock():
            store._load()
            existing = {r.text: r for r in store.all()}
            seen: set = set()
            records: list = []
            for f in facts:
                if f in seen:
                    continue                  # collapse duplicate lines
                seen.add(f)
                keep = existing.get(f)
                records.append(keep if keep is not None else MemoryRecord(
                    text=f, kind="semantic", source="user", importance=0.8))
            store.replace(records, embed_fn=_embed_fn())
        return records

    records = await _off_loop(_do_put)
    return {"status": "saved", "count": len(records)}


@_router.post("/api/memory/append")
async def memory_append(req: MemoryAppend, request: Request = None):
    _require_writable()
    fact = _strip_bullet(req.text)
    if not fact:
        raise HTTPException(400, "Nothing to remember")
    store = _request_store(request)

    from localm.memory import N_MAX, MemoryRecord

    # Off the loop: _migrate_legacy and store.add both resolve the embedder
    # (_embed_fn -> get_embedder), which must not run on the event-loop thread.
    def _do_append():
        _migrate_legacy(store)
        # Refuse to append past the cap rather than accept a fact the next prune
        # would evict.
        if len(store) >= N_MAX:
            raise HTTPException(
                413, f"Memory is at its {N_MAX}-record cap; delete a fact before "
                     "adding another.")
        return store.add(MemoryRecord(text=fact, kind="semantic", source="user",
                                      importance=0.8), embed_fn=_embed_fn())

    rec = await _off_loop(_do_append)
    return {"status": "appended", "id": rec.id}


@_router.patch("/api/memory/{mem_id}")
async def memory_patch(mem_id: str, req: MemoryPatch, request: Request = None):
    _require_writable()
    store = _request_store(request)

    fields = {}
    if req.text is not None:
        fields["text"] = req.text
    if req.importance is not None:
        fields["importance"] = req.importance
    # Off the loop: store.update -> _embed_fn -> get_embedder.
    rec = (await _off_loop(lambda: store.update(mem_id, embed_fn=_embed_fn(), **fields))
           if fields else store.get(mem_id))
    if rec is None:
        raise HTTPException(404, "No such memory")
    return {"status": "saved", "item": _item(rec)}


@_router.delete("/api/memory/{mem_id}")
async def memory_delete(mem_id: str, request: Request = None):
    _require_writable()
    store = _request_store(request)

    # Off the loop like every other mutating route here: the write waits on a
    # cross-process lock, which must not be awaited on the event-loop thread.
    deleted = await _off_loop(lambda: store.delete(mem_id))
    return {"status": "deleted" if deleted else "absent", "id": mem_id}


@_router.post("/api/memory/corrections/{cid}/accept")
async def memory_correction_accept(cid: str, request: Request = None):
    """Apply a proposed supersession of a trusted fact: replace its text (or delete
    it), archiving the old value to the recoverable .forgotten sidecar. Only this
    route applies one - a synth candidate never auto-overwrites a user fact."""
    # Off the loop: resolve_correction -> _embed_fn -> get_embedder.
    return await _off_loop(lambda: _apply_correction(cid, True, request))


@_router.post("/api/memory/corrections/{cid}/reject")
async def memory_correction_reject(cid: str, request: Request = None):
    """Dismiss a proposed supersession: keep the fact as-is, reset its
    last-confirmed staleness, and remember the dismissal so consolidation does not
    re-propose the same change on the next pass."""
    # Off the loop: resolve_correction -> _embed_fn -> get_embedder.
    return await _off_loop(lambda: _apply_correction(cid, False, request))


@_router.get("/api/memory/forgotten")
async def memory_forgotten(request: Request = None):
    """List archived (forgotten) records for the caller's own namespace, read back
    from the recoverable archive ``_archive_forgotten()`` writes. Read-only (no
    side effect), so available in privacy mode too, matching how memory_get returns
    existing records regardless of mode."""
    store = _request_store(request)
    return {"items": [_forgotten_item(e) for e in store.forgotten()]}


@_router.post("/api/memory/forgotten/{mem_id}/restore")
async def memory_forgotten_restore(mem_id: str, request: Request = None):
    """Recover one archived record back into the live store."""
    _require_writable()
    store = _request_store(request)
    # Off the loop: restore_forgotten -> _embed_fn -> get_embedder.
    rec = await _off_loop(lambda: store.restore_forgotten(mem_id, embed_fn=_embed_fn()))
    if rec is None:
        raise HTTPException(
            404, "No such forgotten record (or it is already restored)")
    return {"status": "restored", "item": _item(rec)}


def _apply_correction(cid: str, accept: bool, request):
    _require_writable()
    store = _request_store(request)
    out = store.resolve_correction(cid, accept, embed_fn=_embed_fn())
    if out is None:
        raise HTTPException(404, "No such correction")
    if out.get("status") == "archive_failed":
        # The old value could not be archived; resolve_correction left the record and
        # the correction intact. Surfaced as an error, never as an applied correction.
        raise HTTPException(
            500, "Could not archive the old value, so the correction was not "
            "applied (your saved fact is unchanged). Please try again.")
    return out


@_router.post("/api/memory/consolidate")
async def memory_consolidate(request: Request = None):
    """Manually distil durable facts from recent sessions into the store (the
    opt-in consolidation trigger). Gated on privacy; needs a loaded model."""
    _require_writable()
    principal = _request_principal(request)

    # Off the loop: synthesize_memory drives a full blocking LLM generation per
    # candidate (complete(), below) plus _embed_fn() resolution, which can itself
    # trigger a VRAM swap lasting minutes.
    #
    # eng is resolved HERE, inside the offloaded call, so complete() closes over a
    # live engine, and the whole request uses that one resolution throughout.
    def _do_consolidate():
        eng = _live_engine()
        if eng is None or not getattr(eng, "loaded", False):
            raise HTTPException(503, "Load a model first to consolidate memory")

        def complete(prompt: str) -> str:
            # strip_think: memory must never ingest the reasoning channel.
            from localm.textnorm import strip_think
            return strip_think("".join(eng.chat_stream(
                [{"role": "user", "content": prompt}]))).strip()

        # driving_engine pins the engine busy and touches its activity clock for the
        # WHOLE synthesis pass, not per-completion, so idle-unload cannot evict the
        # model in a gap between candidates.
        from localm.inference.http_server import driving_engine
        with driving_engine(eng):
            return synthesize_memory(complete, principal=principal)

    return await _off_loop(_do_consolidate)


# ------------------------------------------------------------------ #
#  Consolidation (distil durable facts from finished sessions)        #
# ------------------------------------------------------------------ #
# Read recent session logs and fold durable user facts into the store via the
# ADD/UPDATE/DELETE/NO_OP consolidation loop. Runs out of band (the jobs "memory"
# task, POST /api/memory/consolidate, or the debounced auto pass), never in the chat
# hot path. Only log/full sessions exist to read, and the write path is gated on
# _persist_enabled().

def _recent_sessions_text(max_chars: int = 8000) -> str:
    """User+assistant content from the newest session JSONL logs (written only in
    log/full mode), newest file first, capped to *max_chars*. Empty when none.
    Prioritises the most recent turns within each file."""
    sdir = _home() / "sessions"
    if not sdir.is_dir():
        return ""
    files = sorted(sdir.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[str] = []
    total = 0
    stop = False
    for f in files:
        if stop:
            break
        try:
            raw = f.read_text(encoding="utf-8")
        except OSError as e:
            from localm.debuglog import logger
            logger.debug("memory consolidation: skipping unreadable session %s: %s",
                         f, e)
            continue
        file_pieces = []
        for line in reversed(raw.splitlines()):
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(rec, dict) or rec.get("type") not in ("user", "llm"):
                continue
            data = rec.get("data") or {}
            content = data.get("content", "") if isinstance(data, dict) else ""
            if not isinstance(content, str) or not content.strip():
                continue
            who = "User" if rec.get("type") == "user" else "Assistant"
            if who == "Assistant":
                # Session logs keep the assistant's reasoning channel; the fact
                # extractor sees only the visible answer.
                from localm.textnorm import strip_think
                content = strip_think(content)
                if not content.strip():
                    continue
            piece = f"{who}: {content.strip()}"
            added_len = len(piece) + (1 if out or file_pieces else 0)
            if total + added_len > max_chars:
                stop = True
                break
            file_pieces.insert(0, piece)
            total += added_len
        if file_pieces:
            out = file_pieces + out
    return "\n".join(out)


def synthesize_memory(complete, *, principal: str | None = None, max_facts: int = 12,
                      max_chars: int = 8000) -> dict:
    """Distil durable user facts from recent sessions into the structured store.

    *complete* is an injected ``(prompt: str) -> str`` model call (the jobs runner
    binds it to the engine; tests pass a fake). Delegates to the memory
    consolidation loop (ADD/UPDATE/DELETE/NO_OP + decay).

    Privacy: gated on _persist_enabled(); in privacy mode returns
    ``{"status": "skipped", "reason": "privacy", "added": 0}`` and NEVER calls the
    model. Returns ``{status, added, updated, deleted, facts}``, the shape the jobs
    "memory" runner consumes."""
    if not _persist_enabled():
        return {"status": "skipped", "reason": "privacy", "added": 0}
    sessions = _recent_sessions_text(max_chars=max_chars)
    if not sessions.strip():
        return {"status": "skipped", "reason": "no_sessions", "added": 0}
    from localm.memory import run_consolidation
    store = _chat_store(principal)
    _migrate_legacy(store)
    # Newly added facts come from diffing record ids: store.replace reuses the same
    # record objects, so an UPDATE keeps its id and only true ADDs appear.
    before = {r.id for r in store.all()}
    embed_fn = _embed_fn()
    if embed_fn is None:
        # No embedder resolvable: recall and consolidation fall back to lexical BM25,
        # and this round's records get no vector. Logged so it is distinguishable from
        # a per-record embed failure.
        from localm.debuglog import logger
        logger.debug("memory synthesize_memory: no embedder resolved, "
                     "this round's records will have no vector")
    res = run_consolidation(store, sessions, complete, embed_fn=embed_fn,
                            surface="chat", max_candidates=max_facts)
    new_facts = [r.text for r in store.all() if r.id not in before]
    # Episodic capture is per-session and watermarked: one episode per NEW session,
    # not one blob summary over all of them.
    episodic = _store_episodes(store, complete, embed_fn=embed_fn)
    # Backfill vectors for records stored before an embedder was available, so
    # semantic recall turns on retroactively. Bounded per pass; a large store fills
    # over several passes. No-op when no embedder.
    if embed_fn is not None:
        try:
            filled = store.backfill_vectors(embed_fn)
            if filled:
                res["backfilled"] = filled
        except Exception as e:
            from localm.debuglog import logger
            logger.debug("memory vector backfill skipped: %s", e)
    return {"status": res.get("status", "ok"), "added": res.get("added", 0),
            "updated": res.get("updated", 0), "deleted": res.get("deleted", 0),
            "proposed": res.get("proposed", 0),
            # TOTAL pending corrections awaiting review, not just this run's new ones,
            # so outstanding earlier suggestions are still reported.
            "pending": len(store.corrections()),
            "episodic": episodic, "facts": new_facts}


def _episodic_watermark_path(store) -> Path:
    """Sidecar next to the episodic store holding the newest session mtime already
    turned into an episode, so a processed session is never re-summarised."""
    p = store.path
    return p.with_name(p.stem + ".episodic-watermark.json")


def _read_episodic_watermark(store) -> float:
    try:
        data = json.loads(_episodic_watermark_path(store).read_text(encoding="utf-8"))
        return float(data.get("last_mtime", 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0                               # absent/corrupt -> process from scratch


def _read_episodic_stems(store) -> set:
    """The session stems already summarised AT EXACTLY last_mtime. mtime alone is
    not a unique cursor: a bulk LOCALM_HOME restore, or a coarse-granularity volume
    (FAT/exFAT/SMB, 1-2s mtime resolution), gives many files one identical mtime, so
    a strict `st_mtime > watermark` filter would permanently skip the tied files the
    per-run cap left unprocessed. Pairing the watermark mtime with the stems already
    done at that mtime makes the cursor tie-safe: a tied file is re-picked iff its
    stem is not yet recorded. Absent/old sidecar -> empty set."""
    try:
        data = json.loads(_episodic_watermark_path(store).read_text(encoding="utf-8"))
        stems = data.get("stems", [])
        return set(stems) if isinstance(stems, list) else set()
    except (OSError, ValueError, TypeError, AttributeError):
        return set()


def _write_episodic_watermark(store, mtime: float, stems=()) -> None:
    p = _episodic_watermark_path(store)
    try:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"last_mtime": float(mtime),
                                   "stems": sorted(stems)}), encoding="utf-8")
        tmp.replace(p)
    except OSError as e:
        from localm.debuglog import logger
        logger.debug("episodic watermark write skipped: %s", e)


def _session_text(path: Path, max_chars: int = 6000) -> str:
    """User+assistant content of ONE session file, chronological, capped (keeping the
    most recent turns). Assistant reasoning (<think>) is stripped so a summary never
    ingests scratchpad. Empty when the file has no usable turns."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.debug("memory consolidation: unreadable session %s skipped "
                     "(excluded from this summary): %s", path, e)
        return ""
    pieces: list[str] = []
    total = 0
    for line in reversed(raw.splitlines()):        # newest-first, so the cap keeps recent
        try:
            rec = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(rec, dict) or rec.get("type") not in ("user", "llm"):
            continue
        data = rec.get("data") or {}
        content = data.get("content", "") if isinstance(data, dict) else ""
        if not isinstance(content, str) or not content.strip():
            continue
        who = "User" if rec.get("type") == "user" else "Assistant"
        if who == "Assistant":
            from localm.textnorm import strip_think
            content = strip_think(content)
            if not content.strip():
                continue
        piece = f"{who}: {content.strip()}"
        if total + len(piece) + 1 > max_chars:
            break
        pieces.insert(0, piece)                    # rebuild chronological order
        total += len(piece) + 1
    return "\n".join(pieces)


_UNSET = object()

# Cap on real model generations per episodic pass. The backlog drains over several
# runs; the watermark advances only past files actually processed.
EPISODIC_MAX_PER_RUN = 5
# Only a SETTLED session (untouched for at least this long) is summarised. Matches
# the auto-consolidate debounce cadence (MEMORY_AUTO_MIN_INTERVAL).
EPISODIC_SETTLE_SECONDS = 900.0


def _store_episodes(store, complete, embed_fn=_UNSET, now=None) -> int:
    """Store one EPISODIC summary PER NEW session file (past the watermark), so N
    sessions become N episodes instead of collapsing into <=1 blob summary. Each
    episode is tagged with its source session id + mtime. Deduped against existing
    episodics (0.85). Best-effort; the caller confirmed writes are allowed
    (privacy). Returns the number of episodes stored.

    BOUNDED + SETTLED + ONE-PER-SESSION:
      - at most EPISODIC_MAX_PER_RUN real summaries per run, so a large first-pass
        backlog drains over several runs instead of one unbounded serial burst;
      - only SETTLED sessions (idle >= EPISODIC_SETTLE_SECONDS) are summarised, so a
        conversation is never distilled mid-flight from partial content;
      - the watermark advances only past a file actually processed (summarised, or
        seen-but-empty), never past an unsettled file nor past files the cap deferred;
      - a session that is RESUMED (grows after being summarised, so its mtime
        re-crosses the watermark) SUPERSEDES its own earlier episode via the
        meta["session"] stem tag, instead of accumulating a second record for the
        same conversation. Net: exactly ONE episode per session stem, always the
        fullest summary of it.

    *embed_fn*: reuse an already-resolved embedder (synthesize_memory calls this
    right after run_consolidation, in the same round, and passes its own
    embed_fn). Omit to resolve get_embedder() independently.

    *now*: injected wall-clock for the settle check (defaults to time.time())."""
    ef = _embed_fn() if embed_fn is _UNSET else embed_fn
    try:
        import time as _time
        from difflib import SequenceMatcher

        from localm.memory import MemoryRecord, summarize_session
        sdir = _home() / "sessions"
        if not sdir.is_dir():
            return 0
        now = _time.time() if now is None else now
        watermark = _read_episodic_watermark(store)
        wm_stems = _read_episodic_stems(store)     # stems already done AT watermark (tie-safe)
        # A file is unprocessed if it is strictly newer than the watermark, or ties the
        # watermark mtime but its stem was not yet summarised. Sorted by (mtime, stem)
        # for a total, stable oldest-first order even when mtimes tie.
        new_files = sorted(
            (f for f in sdir.glob("*.jsonl")
             if f.stat().st_mtime > watermark
             or (f.stat().st_mtime == watermark and f.stem not in wm_stems)),
            key=lambda p: (p.stat().st_mtime, p.stem))
        if not new_files:
            return 0
        stored = 0
        gens = 0                                   # real model generations this run (bounded)
        newest = watermark
        newest_stems = set(wm_stems)               # carry forward stems recorded at `watermark`
        # This session's OWN existing episodes, keyed by source stem, so a resumed
        # session supersedes its earlier record instead of adding a duplicate. A stem
        # can map to several records, so this keeps a list and the supersede branch
        # collapses them. Built once: each run sees each stem at most once (glob yields
        # distinct paths), and add/update/delete reload internally.
        prior_by_stem: dict = {}
        for _r in store.all():
            if getattr(_r, "kind", None) != "episodic":
                continue
            _meta = getattr(_r, "meta", None)
            if isinstance(_meta, dict) and _meta.get("session"):
                prior_by_stem.setdefault(_meta["session"], []).append(_r)

        def _advance(mt: float, stem: str) -> None:
            """Move the (mtime, stems-at-mtime) cursor past a processed file. A
            file that ties `newest` ADDS its stem; a strictly-newer file resets the
            stem set to just itself (nothing else at that new mtime is done yet)."""
            nonlocal newest, newest_stems
            if mt > newest:
                newest, newest_stems = mt, {stem}
            elif mt == newest:
                newest_stems.add(stem)

        for f in new_files:
            mt = f.stat().st_mtime
            # Only summarise a SETTLED session, and do NOT advance the watermark past
            # an unsettled one, so it is summarised exactly once after it goes quiet.
            # Files are oldest-first, so the first unsettled one means every later file
            # is unsettled too.
            if now - mt < EPISODIC_SETTLE_SECONDS:
                break
            text = _session_text(f)
            if not text.strip():
                _advance(mt, f.stem)               # no usable turns: seen, skip forever
                continue
            # Bound real generations per run. At the cap, leave this file, the rest,
            # and the cursor for the next run.
            if gens >= EPISODIC_MAX_PER_RUN:
                break
            summ = summarize_session(complete, text)
            gens += 1
            _advance(mt, f.stem)                    # a real attempt was made -> advance past it
            if not summ:
                continue
            lo = summ.lower()
            # ONE episode PER SESSION: a resumed session's later summary supersedes its
            # earlier record rather than adding a second one. _session_text reads the
            # whole file, so the later summary is the fuller record and the earlier one
            # is the partial.
            priors = prior_by_stem.get(f.stem) or []
            if priors:
                # Collapse to ONE record for this conversation, keeping the newest and
                # deleting the rest. Only the stem being processed is collapsed, never
                # a global sweep of the store.
                priors.sort(key=lambda r: (getattr(r, "meta", None) or {}).get(
                    "session_mtime") or 0.0)
                keep = priors[-1]
                for extra in priors[:-1]:
                    store.delete(extra.id)
                if SequenceMatcher(None, lo, keep.text.lower()).ratio() > 0.85:
                    # Substantively the same story: keep the text, re-stamp which state
                    # of the session it reflects (no re-embed needed).
                    store.update(keep.id,
                                 meta={"session": f.stem, "session_mtime": mt})
                    continue
                # update() re-embeds on a text change, so the vector cannot go stale
                # against the superseded text.
                store.update(keep.id, text=summ, embed_fn=ef,
                             meta={"session": f.stem, "session_mtime": mt})
                continue
            # Cross-stem dedup: a DIFFERENT session whose summary near-duplicates an
            # existing episode adds nothing.
            if any(r.kind == "episodic"
                   and SequenceMatcher(None, lo, r.text.lower()).ratio() > 0.85
                   for r in store.all()):
                continue                           # already have this episode
            store.add(MemoryRecord(text=summ, kind="episodic", source="synth",
                                   importance=0.4,
                                   meta={"session": f.stem, "session_mtime": mt}),
                      embed_fn=ef)
            stored += 1
        _write_episodic_watermark(store, newest, newest_stems)
        return stored
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("episodic capture skipped: %s", e)
        return 0


# ------------------------------------------------------------------ #
#  Automatic (unattended) memory formation                            #
# ------------------------------------------------------------------ #
# After a turn completes (log/full mode only), a debounced background pass distils
# durable facts into the store. Writes stay gated, so nothing runs in privacy mode.

MEMORY_AUTO_MIN_INTERVAL = 900.0     # >= 15 min between auto-consolidation runs
_auto_lock = _threading.Lock()       # guards the marker read + in-progress flag
_auto_running = False                 # True while a background pass is in flight


def _auto_marker() -> "Path":
    return _memory_root() / ".auto_consolidate"


def _auto_last_run() -> float:
    """Epoch of the last auto-consolidation, persisted so the debounce survives a
    restart. 0.0 when never run or unreadable (treated as 'due')."""
    try:
        return float(_auto_marker().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0.0


def _auto_stamp(now: float) -> None:
    try:
        m = _auto_marker()
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(str(now), encoding="utf-8")
    except OSError as e:
        # Non-fatal: a failed stamp just means the next turn may re-run sooner.
        from localm.debuglog import logger
        logger.debug("memory auto-consolidate: could not stamp marker: %s", e)


def _auto_consolidate_bg(principal: str | None = None) -> None:
    """Run one consolidation pass in the background, binding the model to the
    stashed engine. Never raises (a daemon thread); clears the in-progress flag and
    stamps the marker when done."""
    global _auto_running
    from localm.debuglog import logger
    try:
        # Resolved once here, at thread start, and reused for the whole pass, so the
        # engine cannot swap out from under consolidation while it is mid-run.
        eng = _live_engine()
        if eng is None or not getattr(eng, "loaded", False):
            return
        from localm.textnorm import strip_think

        def complete(prompt: str) -> str:
            return strip_think("".join(
                eng.chat_stream([{"role": "user", "content": prompt}]))).strip()

        # Pin for the whole pass, not per-completion, so a gap between candidates
        # cannot be evicted into.
        from localm.inference.http_server import driving_engine
        with driving_engine(eng):
            res = synthesize_memory(complete, principal=principal)
        added = res.get("added", 0)
        if added:
            logger.info("memory auto-consolidate: added %d fact(s)", added)
    except Exception as e:
        logger.warning("memory auto-consolidate failed: %s", e)
    finally:
        import time as _time
        _auto_stamp(_time.time())
        with _auto_lock:
            _auto_running = False


def _maybe_auto_consolidate(principal: str | None = None) -> None:
    """Best-effort trigger for the debounced background consolidation. Cheap when
    not due (a timestamp compare under a lock); spawns at most one daemon thread.
    Gated on config + privacy + model-loaded so it never runs a write the mode
    forbids and never blocks the turn."""
    global _auto_running
    try:
        from localm.config import load_config
        cfg = load_config()
        if not cfg.get("memory_auto_consolidate", True):
            return
        if not cfg.get("memory_enabled", True):
            return
    except Exception as e:
        # Config unreadable: stay disabled this turn, and log why, so a persistently
        # broken config is not an invisible "consolidation never runs".
        from localm.debuglog import logger
        logger.debug("memory auto-consolidate: config read failed, skipping this "
                     "turn: %s", e)
        return
    if not _persist_enabled():
        return                                # privacy: no new traces
    if not getattr(_live_engine(), "loaded", False):
        # This outlet only fires after a completed chat turn, so no loaded engine here
        # means memory is fully off, or a swap landed between generation and this
        # point. Logged rather than skipped silently.
        from localm.debuglog import logger
        logger.debug("memory auto-consolidate: no engine currently loaded, "
                     "skipping this turn")
        return                                # no model to distil with
    import os
    import time as _time
    # The debounce interval is overridable via env; malformed values fall back to the
    # default.
    try:
        interval = float(os.environ.get(
            "LOCALM_MEMORY_AUTO_INTERVAL", MEMORY_AUTO_MIN_INTERVAL))
    except (TypeError, ValueError):
        interval = MEMORY_AUTO_MIN_INTERVAL
    now = _time.time()
    with _auto_lock:
        if _auto_running:
            return
        if now - _auto_last_run() < interval:
            return
        _auto_running = True
        # Stamp immediately so a burst of concurrent turns cannot each spawn a pass
        # before the first finishes (the finally-stamp refreshes it).
        _auto_stamp(now)
    _threading.Thread(target=lambda: _auto_consolidate_bg(principal), daemon=True).start()


# ------------------------------------------------------------------ #
#  Automatic (unattended) vector backfill                             #
# ------------------------------------------------------------------ #
# backfill_all (memory.backfill) walks EVERY namespace under the memory root to
# completion; the manual `setup-embeddings` CLI command calls it too. The
# debounced auto-consolidate pass above only backfills the ONE store for
# whichever principal just chatted, bounded to 64 records per pass
# (MemoryStore.backfill_vectors), so an unrelated namespace, or a backlog
# bigger than that bound, does not converge through that path. A record with no
# vector is invisible to the semantic gate and recall falls back to the
# importance-ordered path. This sweep needs only the EMBEDDER, not a loaded
# chat model, and runs independently of memory_auto_consolidate and of whether
# a model is currently loaded.

BACKFILL_SWEEP_MIN_INTERVAL = 3600.0  # >= 1h between full-root backfill sweeps
_sweep_lock = _threading.Lock()       # guards the marker read + in-progress flag
_sweep_running = False                 # True while a background sweep is in flight


def _sweep_marker() -> "Path":
    return _memory_root() / ".backfill_sweep"


def _sweep_last_run() -> float:
    """Epoch of the last backfill sweep, persisted so the debounce survives a
    restart. 0.0 when never run or unreadable (treated as 'due')."""
    try:
        return float(_sweep_marker().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0.0


def _sweep_stamp(now: float) -> None:
    try:
        m = _sweep_marker()
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(str(now), encoding="utf-8")
    except OSError as e:
        # Non-fatal: a failed stamp just means the next turn may re-check sooner.
        from localm.debuglog import logger
        logger.debug("memory backfill sweep: could not stamp marker: %s", e)


def _backfill_sweep_bg() -> None:
    """Run one backfill_all pass over EVERY namespace in the background,
    mirroring _auto_consolidate_bg's shape. Never raises (a daemon thread);
    clears the in-progress flag and stamps the marker when done. A missing
    embedder is a normal, expected outcome (backfill_all(..., None) is a
    documented no-op), not an error."""
    global _sweep_running
    from localm.debuglog import logger
    try:
        from localm.memory.backfill import backfill_all
        res = backfill_all(_memory_root(), _embed_fn())
        if res["embedded"]:
            logger.info("memory backfill sweep: embedded %d record(s) across "
                        "%d namespace(s)", res["embedded"], res["namespaces"])
        if res["remaining"] or res["unreadable"]:
            logger.warning(
                "memory backfill sweep: %d record(s) still lack a vector, %d "
                "namespace(s) unreadable - will retry next sweep",
                res["remaining"], res["unreadable"])
    except Exception as e:
        logger.warning("memory backfill sweep failed: %s", e)
    finally:
        import time as _time
        _sweep_stamp(_time.time())
        with _sweep_lock:
            _sweep_running = False


def _maybe_sweep_backfill() -> None:
    """Best-effort trigger for the debounced full-root vector backfill sweep.
    Cheap when not due or nothing is pending (a local vectorless scan - no
    embedder/network touch on this thread; that stays deferred to the
    background pass, same discipline _maybe_auto_consolidate uses to keep
    engine work off the calling turn); spawns at most one daemon thread."""
    global _sweep_running
    try:
        from localm.config import load_config
        if not load_config().get("memory_enabled", True):
            return
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory backfill sweep: config read failed, skipping "
                     "this turn: %s", e)
        return
    if not _persist_enabled():
        return                                # privacy: nothing to fill in
    import os
    import time as _time
    try:
        interval = float(os.environ.get(
            "LOCALM_MEMORY_BACKFILL_INTERVAL", BACKFILL_SWEEP_MIN_INTERVAL))
    except (TypeError, ValueError):
        interval = BACKFILL_SWEEP_MIN_INTERVAL
    now = _time.time()
    with _sweep_lock:
        if _sweep_running:
            return
        if now - _sweep_last_run() < interval:
            return
        from localm.memory.backfill import vectorless_scan
        pending, unreadable_ns = vectorless_scan(_memory_root())
        if not pending and not unreadable_ns:
            _sweep_stamp(now)         # checked and clean: don't rescan every
            return                    # turn until the interval elapses again
        _sweep_running = True
        # Stamp immediately, same burst guard as auto-consolidate above; the
        # finally-stamp in _backfill_sweep_bg refreshes it on completion.
        _sweep_stamp(now)
    _threading.Thread(target=_backfill_sweep_bg, daemon=True).start()


# ------------------------------------------------------------------ #
#  Chat hooks: recall injection (inlet) + consolidation/backfill outlet
# ------------------------------------------------------------------ #

def _user_msg_text(m) -> str:
    """The text of one user message (multimodal text parts joined)."""
    content = m.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _recall_query(messages, *, max_chars: int = 400, max_user_turns: int = 3) -> str:
    """The recall QUERY: the most recent user message plus a short window of prior
    user turns, so an anaphoric follow-up ("yes, do that") still carries the earlier
    turn's topic. Newest-first and truncated to max_chars so the latest turn is
    never dropped. Only steers
    relevance ranking + the eligibility gate; recalled memories are neutralised before
    injection."""
    texts = []
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        t = _user_msg_text(m).strip()
        if t:
            texts.append(t)
        if len(texts) >= max_user_turns:
            break
    if not texts:
        return ""
    return "\n".join(texts)[:max_chars].strip()      # texts[0] is the newest turn


def _memory_outlet(text, messages, ctx):
    """After a completed turn, opportunistically grow the memory in the background
    (debounced). Side-effect only: returns the text unchanged. Any failure is
    contained so it can never affect the reply."""
    try:
        # Owner (ADMIN scope) collapses to the shared "owner" namespace, matching
        # memory_principal on the request-based paths and the recall inlet. Shared with
        # _memory_inlet via _ctx_principal, so reads and writes use one namespace.
        _maybe_auto_consolidate(_ctx_principal(ctx))
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory outlet skipped: %s", e)
    try:
        # A separate try/except from the consolidation trigger above: the two
        # are independent capabilities, so a failure in one must not skip the
        # other.
        _maybe_sweep_backfill()
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory backfill sweep trigger skipped: %s", e)
    return text


def _stash_memory_used(ctx, records, diag) -> None:
    """Record, in the per-request ``ctx.state``, WHICH memories the inlet injected
    and WHY recall degraded, so the chat route can surface a
    "used N memories" affordance + the degrade reason in a response header. Only
    metadata + the already-injected memory text is stashed (in-memory, per request,
    returned to the same authenticated user) - never written to any log; the debug
    line carries the COUNT and reason only, no memory content (privacy). Best-effort
    and side-effect free: a stash failure never affects the reply. No-op without a
    ctx (e.g. a pipeline-less test call)."""
    if ctx is None:
        return
    try:
        from localm.memory import INJECT_LINE_CHARS
        items = []
        for r in records:
            if isinstance(r, dict):
                rid, text = r.get("id"), r.get("text", "")
                source, kind = r.get("source"), r.get("kind")
            else:
                rid = getattr(r, "id", None)
                text = getattr(r, "text", "") or ""
                source, kind = getattr(r, "source", None), getattr(r, "kind", None)
            item = {"text": (text or "").strip()[:INJECT_LINE_CHARS]}
            if rid:
                item["id"] = rid
            if source:
                item["source"] = source
            if kind:
                item["kind"] = kind
            items.append(item)
        ctx.state["memory_used"] = items
        ctx.state["memory_degrade_reason"] = diag.get("degrade_reason")
        from localm.debuglog import logger
        logger.debug("memory recall: injected %d record(s), degrade=%s",
                     len(items), diag.get("degrade_reason"))
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory stash skipped: %s", e)


def _memory_inlet(messages, ctx):
    """Inject recalled memories into the system message. Off when the `memory_enabled`
    recall knob is off. In privacy mode it is off too UNLESS the user opted into
    read-only recall for chat (`memory_recall_in_privacy` + ..._chat) - and even
    then it only READS: no reinforcement, no migration, no write. Best-effort: any
    failure is logged at debug and skipped (the pipeline also isolates it)."""
    if ctx is not None and ctx.state.get("client_id") == "coder":
        return None
    if not _recall_enabled():
        return None
    writes_ok = _persist_enabled()
    # Privacy mode: fully off unless the user opted into read-only recall for chat.
    if not writes_ok and not _recall_in_privacy("chat"):
        return None
    try:
        from localm import memory as _mem
        query = _recall_query(messages)
        if not query.strip():
            return None
        # Resolve the SAME namespace the write path and the outlet write to
        # (ADMIN/owner -> "owner"), so an owner's saved memories are recalled in
        # protected mode.
        store = _chat_store(_ctx_principal(ctx))

        if writes_ok:
            _migrate_legacy(store)                 # migration is a write
        diag: dict = {}
        block_records = store.recall(query, k=_mem.MAX_INJECT,
                                     embed_fn=_embed_fn(), reinforce=writes_ok,
                                     diagnostics=diag)
        if not block_records and not writes_ok:
            # Privacy-recall opt-in with an un-migrated store: read the legacy flat
            # file, strictly read-only (no migration, no write).
            block_records = [{"text": b}
                             for b in _legacy_bullets()[:_mem.MAX_INJECT]]
        # Stash what recall selected and why it degraded BEFORE the empty-block early
        # return, so a zero-recall or degraded turn is still visible to the client.
        _stash_memory_used(ctx, block_records, diag)
        block = _mem.render_memories(block_records)
        if not block:
            return None
        from localm.textguard import compose
        for m in messages:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = compose(block, "\n\n", m["content"])
                return messages
        messages.insert(0, {"role": "system", "content": block})
        return messages
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory inlet skipped: %s", e)
        return None


async def _memory_inlet_hook(messages, ctx):
    """The REGISTERED inlet hook: runs _memory_inlet OFF the event loop.

    _memory_inlet's body is blocking, on the highest-frequency path there is (every
    chat turn): it _load()s the records JSONL AND the .vec.json embedding sidecar,
    re-_save()s both under the namespace lock when reinforcing (store.py's
    recall(reinforce=True)), and resolves the shared embedder. chat.py awaits
    run_inlet, which calls a hook INLINE (chat_pipeline.py), so a sync hook would do
    all of that ON the uvicorn event loop - multiple MB of JSON parsed and rewritten
    per turn once embeddings are on, stalling every other request.

    _embed_fn -> get_embedder can also trigger a VRAM swap, and vram.py SKIPS the
    guarded chat-model eviction when it detects it is running on the loop thread,
    because completing it there would deadlock. Off-loop, that eviction can run.

    A thin wrapper, so _memory_inlet stays sync and directly unit-testable;
    run_inlet awaits an awaitable hook.
    """
    return await _off_loop(lambda: _memory_inlet(messages, ctx))


def register(host) -> None:
    global _HOST
    host.mount_router(_router)
    host.register_chat_hook("inlet", _memory_inlet_hook)
    # Outlet: after each turn, grow the memory unattended (debounced, background,
    # log/full mode only). Disabling this plugin removes both hooks.
    host.register_chat_hook("outlet", _memory_outlet)
    # Stash the HOST, not host.engine()'s current return value: the engine is resolved
    # fresh via _live_engine() at every use site, so a later model switch is picked up
    # rather than pinned to whatever was loaded at register() time.
    _HOST = host


def unregister() -> None:
    pass
