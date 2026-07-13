# SPDX-License-Identifier: AGPL-3.0-or-later
"""Memory plugin: durable chat memory (extracted from the chat plugin).

A small structured store of durable facts about the user (localm/memory), recalled
by recency + importance + relevance and injected server-side into the system
message by the inlet hook, plus unattended background consolidation that grows the
store after a turn. Owns:

  GET/PUT/POST/PATCH/DELETE /api/memory[...]   - the memory manager surface
  POST /api/memory/consolidate                 - manual "distil now" trigger

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

import json
import re
import threading as _threading
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

_router = APIRouter()

_MEMORY_MAX = 64_000                 # characters - keep injection bounded
_OWNER = "owner"

# The inference engine handle, stashed at register() for the manual
# POST /api/memory/consolidate route and the auto-consolidate pass (both need a
# model). None until wired.
_ENGINE = None


# ------------------------------------------------------------------ #
#  Memory store helper                                               #
# ------------------------------------------------------------------ #

def _chat_store(principal: str | None = None):
    from localm import memory as _mem
    return _mem.open_store(principal, "chat", "", root=_memory_root())


def _request_principal(request: Request | None) -> str | None:
    """MEMORY-1: the principal-resolution half of the request/store snippet
    every /api/memory* route repeated."""
    from localm.inference.http_server import memory_principal
    return memory_principal(request) if request is not None else None


def _request_store(request: Request | None):
    """MEMORY-1: the principal-resolution + store-open snippet every mutating
    /api/memory* route repeated verbatim."""
    return _chat_store(_request_principal(request))


def _require_writable() -> None:
    """MEMORY-2: the '_persist_enabled() guard -> 403' shape every mutating
    /api/memory* route repeated, with wording that had drifted between call
    sites - this deliberately picks memory_put's more helpful wording for all
    five (an intentional behavior change, not an oversight)."""
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
    except Exception:
        return None


def _legacy_memory_file() -> Path:
    return _home() / "chat-memory.md"


def _read_memory() -> str:
    """The legacy flat chat-memory.md text (migration source), or empty string."""
    p = _legacy_memory_file()
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


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
    reported as a success it did not perform (RULE 5).

    CHK-MEM-LOCK: this batches several ``add(..., save=False)`` calls into ONE
    save, so it must hold the namespace lock across the WHOLE batch (a reload +
    loop + one final save), the same way rag's ``_add_paths_locked`` reloads once
    before its per-file loop rather than once per file - the per-call add()/
    delete() lock is not enough here, it would only serialise each individual
    append, not the read-then-decide-then-write-once shape of a batch."""
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
    # The import ran (or there was nothing new). Mark it done so we do not re-scan
    # every start. A marker-write failure must NOT be reported as 'skipped' when the
    # records were actually imported (the false-success RULE 5 forbids) - log it
    # honestly; the casefold dedup above makes a re-run next start harmless.
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
    user can accept or reject it (memory-audit 2026-07-02 [9])."""
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


def _rendered_text(store) -> str:
    """The store's facts as a markdown bullet list (what the memory modal textarea
    shows). Falls back to the legacy flat file when the structured store is still
    empty so existing memory stays visible."""
    recs = store.all()
    if recs:
        return "\n".join(f"- {r.text}" for r in recs)
    return _read_memory().strip()


# ------------------------------------------------------------------ #
#  Memory manager routes (/api/memory*)                               #
# ------------------------------------------------------------------ #

@_router.get("/api/memory")
async def memory_get(request: Request = None):
    store = _request_store(request)
    writable = _persist_enabled()
    if writable:
        _migrate_legacy(store)
    # Corrections are only surfaced when writes are allowed: they are useless
    # read-only (accept/reject need a write), and store.corrections() prunes stale
    # entries as a side effect, which must never run in privacy mode (no new
    # traces). So privacy mode returns an empty list without touching the sidecar.
    return {"text": _rendered_text(store), "writable": writable,
            "items": [_item(r) for r in store.all()],
            "corrections": _corrections_payload(store) if writable else [],
            "path": str(store.path)}


@_router.put("/api/memory")
async def memory_put(req: MemoryUpdate, request: Request = None):
    """Bulk-edit: the modal textarea is the authoritative user memory. DIFF-AWARE:
    a line matching an existing record's text keeps that record as-is; only new
    lines become new user records; omitted lines are deletes.

    CHK-MEM-LOCK: the existing/records diff below is a snapshot-then-decide-then-
    write sequence, same shape as consolidation's - a concurrent POST
    /api/memory/append (e.g. a second browser tab) landing between the snapshot
    and store.replace() would be silently discarded by replace()'s whole-namespace
    overwrite if this route didn't hold the lock across the whole thing."""
    _require_writable()
    from localm.memory import MAX_TEXT_LEN, N_MAX, MemoryRecord
    if len(req.text) > _MEMORY_MAX:
        raise HTTPException(413, "Memory too large (max 64k chars)")
    facts = [_strip_bullet(ln)[:MAX_TEXT_LEN]
             for ln in req.text.splitlines() if _strip_bullet(ln)]
    # Reject over-cap writes at the door instead of accepting facts the next prune
    # would silently hard-delete (audit F4). N_MAX is generous.
    if len(facts) > N_MAX:
        raise HTTPException(
            413, f"Too many memory records ({len(facts)}); the store keeps at "
            f"most {N_MAX}. Trim the list before saving.")
    store = _request_store(request)

    _migrate_legacy(store)
    with store.lock():
        store._load()
        existing = {r.text: r for r in store.all()}
        seen: set = set()
        records: list = []
        for f in facts:
            if f in seen:
                continue                      # collapse duplicate lines
            seen.add(f)
            keep = existing.get(f)
            records.append(keep if keep is not None else MemoryRecord(
                text=f, kind="semantic", source="user", importance=0.8))
        store.replace(records, embed_fn=_embed_fn())
    return {"status": "saved", "count": len(records)}


@_router.post("/api/memory/append")
async def memory_append(req: MemoryAppend, request: Request = None):
    _require_writable()
    fact = _strip_bullet(req.text)
    if not fact:
        raise HTTPException(400, "Nothing to remember")
    store = _request_store(request)

    _migrate_legacy(store)
    from localm.memory import N_MAX, MemoryRecord
    # Refuse to append past the cap rather than accept a fact the next prune would
    # silently evict (audit F4).
    if len(store) >= N_MAX:
        raise HTTPException(
            413, f"Memory is at its {N_MAX}-record cap; delete a fact before "
                 "adding another.")
    rec = store.add(MemoryRecord(text=fact, kind="semantic", source="user",
                                 importance=0.8), embed_fn=_embed_fn())
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
    rec = (store.update(mem_id, embed_fn=_embed_fn(), **fields)
           if fields else store.get(mem_id))
    if rec is None:
        raise HTTPException(404, "No such memory")
    return {"status": "saved", "item": _item(rec)}


@_router.delete("/api/memory/{mem_id}")
async def memory_delete(mem_id: str, request: Request = None):
    _require_writable()
    store = _request_store(request)

    return {"status": "deleted" if store.delete(mem_id) else "absent", "id": mem_id}


@_router.post("/api/memory/corrections/{cid}/accept")
async def memory_correction_accept(cid: str, request: Request = None):
    """Apply a proposed supersession of a trusted fact: replace its text (or delete
    it), archiving the old value to the recoverable .forgotten sidecar. The user is
    the one deciding - a synth candidate never auto-overwrites a user fact ([9])."""
    return _apply_correction(cid, True, request)


@_router.post("/api/memory/corrections/{cid}/reject")
async def memory_correction_reject(cid: str, request: Request = None):
    """Dismiss a proposed supersession: keep the fact as-is and reset its
    last-confirmed staleness so it is not immediately re-proposed."""
    return _apply_correction(cid, False, request)


def _apply_correction(cid: str, accept: bool, request):
    _require_writable()
    store = _request_store(request)
    out = store.resolve_correction(cid, accept, embed_fn=_embed_fn())
    if out is None:
        raise HTTPException(404, "No such correction")
    return out


@_router.post("/api/memory/consolidate")
async def memory_consolidate(request: Request = None):
    """Manually distil durable facts from recent sessions into the store (the
    opt-in consolidation trigger). Gated on privacy; needs a loaded model."""
    _require_writable()
    if _ENGINE is None or not getattr(_ENGINE, "loaded", False):
        raise HTTPException(503, "Load a model first to consolidate memory")

    def complete(prompt: str) -> str:
        # strip_think: memory must never ingest the reasoning channel (audit C1
        # store-poisoning; the /v1 routes strip it for clients, internal consumers
        # must do the same).
        from localm.inference.textnorm import strip_think
        return strip_think("".join(_ENGINE.chat_stream(
            [{"role": "user", "content": prompt}]))).strip()

    principal = _request_principal(request)
    return synthesize_memory(complete, principal=principal)


# ------------------------------------------------------------------ #
#  Consolidation (distil durable facts from finished sessions)        #
# ------------------------------------------------------------------ #
# Read recent session logs and fold durable user facts into the store via the
# ADD/UPDATE/DELETE/NO_OP consolidation loop. Runs OUT OF BAND (jobs "memory" task,
# the POST /api/memory/consolidate route, or the debounced auto pass) - never in
# the chat hot path. Privacy: only log/full sessions exist to read, AND the write
# path is gated on _persist_enabled().

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
        except OSError:
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
                # extractor must see only the visible answer, or scratchpad text
                # pollutes the extraction prompt (audit C1).
                from localm.inference.textnorm import strip_think
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
    # Report which facts were newly added by diffing record ids (stable across
    # consolidation - store.replace reuses the same record objects), so an UPDATE
    # reuses its id (not counted) and only true ADDs appear.
    before = {r.id for r in store.all()}
    embed_fn = _embed_fn()
    res = run_consolidation(store, sessions, complete, embed_fn=embed_fn,
                            surface="chat", max_candidates=max_facts)
    new_facts = [r.text for r in store.all() if r.id not in before]
    # Episodic capture is per-session + watermarked (memory-audit [14]): one episode
    # per NEW session, not one blob summary over all of them.
    episodic = _store_episodes(store, complete)
    # Backfill vectors for records stored before an embedder was available, so
    # semantic recall turns on RETROACTIVELY after 'localm setup-embeddings'. Bounded
    # per pass; a large store fills over several passes. No-op when no embedder.
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
            "proposed": res.get("proposed", 0), "episodic": episodic,
            "facts": new_facts}


def _episodic_watermark_path(store) -> Path:
    """Sidecar next to the episodic store holding the newest session mtime already
    turned into an episode, so a processed session is never re-summarised
    (memory-audit 2026-07-02 [14])."""
    p = store.path
    return p.with_name(p.stem + ".episodic-watermark.json")


def _read_episodic_watermark(store) -> float:
    try:
        data = json.loads(_episodic_watermark_path(store).read_text(encoding="utf-8"))
        return float(data.get("last_mtime", 0.0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0.0                               # absent/corrupt -> process from scratch


def _write_episodic_watermark(store, mtime: float) -> None:
    p = _episodic_watermark_path(store)
    try:
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps({"last_mtime": float(mtime)}), encoding="utf-8")
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
    except OSError:
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
            from localm.inference.textnorm import strip_think
            content = strip_think(content)
            if not content.strip():
                continue
        piece = f"{who}: {content.strip()}"
        if total + len(piece) + 1 > max_chars:
            break
        pieces.insert(0, piece)                    # rebuild chronological order
        total += len(piece) + 1
    return "\n".join(pieces)


def _store_episodes(store, complete) -> int:
    """Store one EPISODIC summary PER NEW session file (past the watermark), so N
    sessions become N episodes instead of collapsing into <=1 blob summary
    (memory-audit 2026-07-02 [14]). Each episode is tagged with its source session id
    + mtime. The watermark advances to the newest session SEEN (even one that yields
    no usable summary), so nothing is re-summarised or retried forever. Deduped
    against existing episodics (0.85). Best-effort; the caller confirmed writes are
    allowed (privacy). Returns the number of episodes stored."""
    try:
        from difflib import SequenceMatcher

        from localm.memory import MemoryRecord, summarize_session
        sdir = _home() / "sessions"
        if not sdir.is_dir():
            return 0
        watermark = _read_episodic_watermark(store)
        new_files = sorted(
            (f for f in sdir.glob("*.jsonl") if f.stat().st_mtime > watermark),
            key=lambda p: p.stat().st_mtime)       # oldest-first: watermark advances monotonically
        if not new_files:
            return 0
        stored = 0
        newest = watermark
        for f in new_files:
            mt = f.stat().st_mtime
            newest = max(newest, mt)
            text = _session_text(f)
            if not text.strip():
                continue
            summ = summarize_session(complete, text)
            if not summ:
                continue
            lo = summ.lower()
            if any(r.kind == "episodic"
                   and SequenceMatcher(None, lo, r.text.lower()).ratio() > 0.85
                   for r in store.all()):
                continue                           # already have this episode
            store.add(MemoryRecord(text=summ, kind="episodic", source="synth",
                                   importance=0.4,
                                   meta={"session": f.stem, "session_mtime": mt}),
                      embed_fn=_embed_fn())
            stored += 1
        _write_episodic_watermark(store, newest)
        return stored
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("episodic capture skipped: %s", e)
        return 0


# ------------------------------------------------------------------ #
#  Automatic (unattended) memory formation                            #
# ------------------------------------------------------------------ #
# After a turn completes (log/full mode only), a debounced background pass distils
# durable facts into the store, so memory accumulates with no manual step. Privacy:
# writes stay gated, so nothing runs in privacy mode.

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
        # Surfaced, not hidden (rule 5).
        from localm.debuglog import logger
        logger.debug("memory auto-consolidate: could not stamp marker: %s", e)


def _auto_consolidate_bg(principal: str | None = None) -> None:
    """Run one consolidation pass in the background, binding the model to the
    stashed engine. Never raises (a daemon thread); clears the in-progress flag and
    stamps the marker when done."""
    global _auto_running
    from localm.debuglog import logger
    try:
        eng = _ENGINE
        if eng is None or not getattr(eng, "loaded", False):
            return
        from localm.inference.textnorm import strip_think

        def complete(prompt: str) -> str:
            return strip_think("".join(
                eng.chat_stream([{"role": "user", "content": prompt}]))).strip()

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
    except Exception:
        return
    if not _persist_enabled():
        return                                # privacy: no new traces
    if _ENGINE is None or not getattr(_ENGINE, "loaded", False):
        return                                # no model to distil with
    import os
    import time as _time
    # The debounce interval is overridable via env (power users / tests that cannot
    # wait the default 15 min); malformed values fall back to the default.
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
#  Chat hooks: recall injection (inlet) + consolidation trigger (outlet)
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
    turn's topic (memory-audit 2026-07-02 [58]: a last-message-only query recalls
    poorly and, with the [10] relevance gate, would wrongly fall silent). Newest-first
    and truncated to max_chars so the latest turn is never dropped. Only steers
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
        # memory_principal on the request-based paths (AUDIT-MED-14).
        from localm import scopes as _scopes
        prin = getattr(ctx, "principal", None)
        if _scopes.ADMIN in (getattr(ctx, "scopes", ()) or ()):
            prin = None
        _maybe_auto_consolidate(prin)
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory outlet skipped: %s", e)
    return text


def _stash_memory_used(ctx, records, diag) -> None:
    """Record, in the per-request ``ctx.state``, WHICH memories the inlet injected
    and WHY recall degraded (F11 observability), so the chat route can surface a
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
        store = _chat_store(getattr(ctx, "principal", None))

        if writes_ok:
            _migrate_legacy(store)                 # migration is a write
        diag: dict = {}
        block_records = store.recall(query, k=_mem.MAX_INJECT,
                                     embed_fn=_embed_fn(), reinforce=writes_ok,
                                     diagnostics=diag)
        if not block_records and not writes_ok:
            # Privacy-recall opt-in with an un-migrated store: read the legacy flat
            # file, strictly read-only (no migration/write).
            block_records = [{"text": b}
                             for b in _legacy_bullets()[:_mem.MAX_INJECT]]
        # F11 observability: stash what recall selected + why it degraded BEFORE the
        # empty-block early return, so a zero-recall / degraded turn is still visible
        # to the client (rule 5: no silent recall).
        _stash_memory_used(ctx, block_records, diag)
        block = _mem.render_memories(block_records)
        if not block:
            return None
        for m in messages:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = block + "\n\n" + m["content"]
                return messages
        messages.insert(0, {"role": "system", "content": block})
        return messages
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory inlet skipped: %s", e)
        return None


def register(host) -> None:
    global _ENGINE
    host.mount_router(_router)
    host.register_chat_hook("inlet", _memory_inlet)
    # Outlet: after each turn, grow the memory unattended (debounced, background,
    # log/full mode only). Disabling this plugin removes both hooks.
    host.register_chat_hook("outlet", _memory_outlet)
    try:
        _ENGINE = host.engine()
    except Exception:
        _ENGINE = None


def unregister() -> None:
    pass
