# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat plugin: the built-in, protected, default-enabled plugin #0.

This is the reference implementation of the plugin contract - the cleanest,
most ordinary use of register(host)/mount_router that third-party plugins copy.
It owns the chat EXPERIENCE persistence (mounted by the engine, auto-scoped to
the ``chat`` capability):

  GET/PUT/DELETE /api/conversations[/{id}]   - server-side conversation store
  GET/PUT/POST   /api/memory[/append]        - assistant memory (chat-memory.md)
  GET/PUT/DELETE /api/prompts[/{name}]       - prompt library / personas

The actual LLM turn stays in the kernel: the SPA POSTs to /v1/chat/completions
on the inference server directly. Chat ships installed + enabled and cannot be
uninstalled or disabled (catalog: preinstalled + protected; manifest:
default_enabled), so the engine auto-provisions it on first run.

All paths resolve from the data dir at request time, so the plugin needs no
shared services from attach_gui. Persistence is gated on the chat surface's
session mode: privacy mode (the default) keeps conversations in the browser only
and blocks server writes - "no new traces", not amnesia, so reads stay allowed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

_router = APIRouter()


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class ConversationUpsert(BaseModel):
    title: str = "Untitled"
    updated_at: float = 0
    pinned: bool = False
    folder: str | None = None
    branches: list = []           # parked message-branch tails (fork points)
    messages: list = []


class PromptUpsert(BaseModel):
    system: str = ""
    params: dict = {}             # sampling defaults (temperature, top_p, ...)


class MemoryUpdate(BaseModel):
    text: str


class MemoryAppend(BaseModel):
    text: str


# ------------------------------------------------------------------ #
#  Shared helpers (paths resolved at request time)                    #
# ------------------------------------------------------------------ #

_CONV_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CONV_MAX_BYTES = 16 * 1024 * 1024   # data-URI images make these large
_MEMORY_MAX = 64_000                 # characters - keep injection bounded


def _home() -> Path:
    from localm.config import home_dir
    return home_dir()


def _persist_enabled() -> bool:
    """Chat persistence/writes are off in privacy mode (the default)."""
    from localm.audit import SessionMode, effective_mode
    return effective_mode("chat") != SessionMode.PRIVACY


# ------------------------------------------------------------------ #
#  Conversation store                                                 #
# ------------------------------------------------------------------ #
# Server-side persistence for GUI chat conversations so they survive browser
# reloads, profile wipes, and other devices on the LAN.

def _conv_path(conv_id: str) -> Path:
    if not _CONV_ID.match(conv_id):
        raise HTTPException(400, "Invalid conversation id")
    return _home() / "chats" / f"{conv_id}.json"


# R40: cache the projected index row per file keyed by (mtime, size), so the
# sidebar listing does not re-parse every chat JSON (with its embedded data-URI
# images) on every request. A changed file re-parses; the cache is bounded by the
# number of chat files. It only ever holds the lightweight meta row, never bodies.
_META_CACHE: dict = {}


def _conv_meta(p: Path) -> dict:
    """Lightweight index row for one chat file (no message bodies / images)."""
    st = p.stat()
    key = (st.st_mtime, st.st_size)
    cached = _META_CACHE.get(str(p))
    if cached and cached[0] == key:
        return dict(cached[1])
    data = json.loads(p.read_text(encoding="utf-8"))
    meta = {
        "id": p.stem,
        "title": data.get("title", "Untitled"),
        "updated_at": data.get("updated_at", 0),
        "pinned": bool(data.get("pinned", False)),
        "folder": data.get("folder"),
        "n_messages": len(data.get("messages") or []),
    }
    _META_CACHE[str(p)] = (key, meta)
    return dict(meta)


@_router.get("/api/conversations")
async def conversations_list(meta: bool = False, limit: int = 0, offset: int = 0):
    """List conversations newest-first.

    R40: pass ``meta=true`` for a lightweight index (id/title/updated_at/pinned/
    folder/n_messages) without the heavy message bodies and data-URI images - the
    GUI sidebar uses this and lazy-loads each conversation's messages on open via
    ``GET /api/conversations/{id}``. ``limit``/``offset`` paginate. Default (no
    params) keeps the historical full-payload, 200-item behaviour for back-compat.
    """
    if not _persist_enabled():
        return {"enabled": False, "conversations": []}
    chats_dir = _home() / "chats"
    rows = []
    if chats_dir.is_dir():
        for p in chats_dir.glob("*.json"):
            try:
                if meta:
                    rows.append(_conv_meta(p))
                else:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    data["id"] = p.stem
                    rows.append(data)
            except Exception:
                continue   # corrupt file - skip, never block the list
    rows.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
    total = len(rows)
    if limit and limit > 0:
        rows = rows[offset:offset + limit]
    elif offset:
        rows = rows[offset:]
    else:
        rows = rows[:200]   # historical default cap when unpaginated
    return {"enabled": True, "conversations": rows, "total": total}


@_router.get("/api/conversations/{conv_id}")
async def conversation_get(conv_id: str):
    """The full body of one conversation (R40 lazy load). Path validated by
    _conv_path; 404 when absent so the client can fall back to its local copy."""
    if not _persist_enabled():
        raise HTTPException(403, "Chat persistence is off (privacy mode)")
    path = _conv_path(conv_id)
    if not path.is_file():
        raise HTTPException(404, "No such conversation")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(500, "Conversation file is unreadable")
    data["id"] = conv_id
    return data


@_router.put("/api/conversations/{conv_id}")
async def conversation_upsert(conv_id: str, req: ConversationUpsert):
    if not _persist_enabled():
        raise HTTPException(
            403, "Chat persistence is off (privacy mode). "
                 "Set mode/chat_mode to 'log' or 'full' to enable it.")
    path = _conv_path(conv_id)
    payload = json.dumps(
        {"id": conv_id, "title": req.title,
         "updated_at": req.updated_at,
         "pinned": req.pinned, "folder": req.folder,
         "branches": req.branches,
         "messages": req.messages},
        ensure_ascii=False)
    if len(payload.encode("utf-8")) > _CONV_MAX_BYTES:
        raise HTTPException(413, "Conversation too large to persist")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return {"status": "saved", "id": conv_id}


@_router.delete("/api/conversations/{conv_id}")
async def conversation_delete(conv_id: str):
    if not _persist_enabled():
        raise HTTPException(403, "Chat persistence is off (privacy mode)")
    path = _conv_path(conv_id)
    if path.is_file():
        path.unlink()
        return {"status": "deleted", "id": conv_id}
    return {"status": "absent", "id": conv_id}


# ------------------------------------------------------------------ #
#  Assistant memory (structured, via localm/memory)                   #
# ------------------------------------------------------------------ #
# The chat "memory" is a small structured store of durable facts about the user
# (localm/memory), recalled by recency+importance+relevance and injected server-
# side by the inlet hook below. It supersedes the old flat chat-memory.md blob;
# that file is migrated in once (see _migrate_legacy) and then left alone.
# Privacy semantics are unchanged: "no new traces", not amnesia - READING/recall
# of memory from earlier non-privacy sessions is allowed, but every WRITE (add,
# edit, delete, consolidation, migration) is gated on the chat session mode.
#
# Chat memory is OWNER-scoped in v1: the kernel chat pipeline carries no principal
# and localm is single-user, so all chat turns share the "owner" namespace (this
# matches the old global chat-memory.md - no regression). The memory library fully
# supports (principal, agent, scope_key) isolation and is exercised there by the
# coder (per project) and by the library tests.

_OWNER = "owner"


def _memory_root() -> Path:
    return _home() / "memory"


def _chat_store():
    from localm import memory as _mem
    return _mem.open_store(_OWNER, "chat", "", root=_memory_root())


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
    """The legacy flat chat-memory.md text (migration source + privacy-mode read
    fallback), or empty string."""
    p = _legacy_memory_file()
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _legacy_bullets() -> list:
    """Bare fact lines parsed out of the legacy chat-memory.md (bullets stripped)."""
    return [_strip_bullet(ln) for ln in _read_memory().splitlines()
            if _strip_bullet(ln)]


def _migrate_legacy(store) -> None:
    """Import the legacy flat chat-memory.md into the structured store ONCE.

    Gated on the chat session mode (privacy skips - a migration materialises new
    durable records, which is a write) and on a per-namespace marker file so it
    never re-imports. Best-effort: a failure is logged, never fatal, and never
    reported as a success it did not perform (RULE 5)."""
    marker = store.path.with_suffix(".legacy-imported")
    if marker.exists() or not _persist_enabled():
        return
    try:
        from localm.memory import MemoryRecord
        ef = _embed_fn()
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
            store._save()                        # one batch write
    except Exception as e:                        # never break chat on migration
        from localm.debuglog import logger
        logger.debug("chat memory legacy migration failed: %s", e)
        return                                    # do NOT mark done - retry next start
    # The import ran (or there was nothing new). Mark it done so we do not re-scan
    # every start. A marker-write failure must NOT be reported as 'skipped' when the
    # records were actually imported (the false-success RULE 5 forbids, and which
    # this function's own docstring promises against) - log it honestly; the
    # casefold dedup above makes a re-run next start harmless until the marker sticks.
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("1", encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.debug(
            "chat memory legacy migration %s but its completion marker could not "
            "be written (%s); it will re-run harmlessly (dedup) until it persists",
            "imported records" if added else "found nothing new", e)


class MemoryPatch(BaseModel):
    text: str | None = None
    importance: float | None = None


def _item(rec) -> dict:
    return {"id": rec.id, "text": rec.text, "importance": rec.importance,
            "source": rec.source, "kind": rec.kind, "uses": rec.uses,
            "updated": rec.updated}


def _rendered_text(store) -> str:
    """The store's facts as a markdown bullet list (what the existing memory modal
    textarea shows). Falls back to the legacy flat file when the structured store
    is still empty (e.g. privacy mode blocked migration) so existing memory stays
    visible/read-only."""
    recs = store.all()
    if recs:
        return "\n".join(f"- {r.text}" for r in recs)
    legacy = _read_memory().strip()
    return legacy


@_router.get("/api/memory")
async def memory_get():
    store = _chat_store()
    if _persist_enabled():
        _migrate_legacy(store)
    return {"text": _rendered_text(store), "writable": _persist_enabled(),
            "items": [_item(r) for r in store.all()],
            "path": str(store.path)}


@_router.put("/api/memory")
async def memory_put(req: MemoryUpdate):
    """Bulk-edit: the modal textarea is the authoritative user memory. The old
    PUT re-minted EVERY line as a fresh user record, destroying the id, kind,
    source, importance, reinforcement, and timestamps of untouched records (so
    editing one line reset the whole store and made every fact immortal;
    memory-audit 2026-07-02). This is now DIFF-AWARE: a line matching an
    existing record's text keeps that record as-is; only genuinely new lines
    become new user records; omitted lines are deletes."""
    if not _persist_enabled():
        raise HTTPException(
            403, "Memory writes are off in privacy mode (no new traces). "
                 "Set mode/chat_mode to 'log' or 'full' to enable them.")
    from localm.memory import MAX_TEXT_LEN, N_MAX, MemoryRecord
    if len(req.text) > _MEMORY_MAX:
        raise HTTPException(413, "Memory too large (max 64k chars)")
    facts = [_strip_bullet(ln)[:MAX_TEXT_LEN]
             for ln in req.text.splitlines() if _strip_bullet(ln)]
    # Reject over-cap writes at the door instead of accepting facts that the
    # next prune would silently hard-delete (audit F4). N_MAX is generous.
    if len(facts) > N_MAX:
        raise HTTPException(
            413, f"Too many memory records ({len(facts)}); the store keeps at "
                 f"most {N_MAX}. Trim the list before saving.")
    store = _chat_store()
    _migrate_legacy(store)
    existing = {r.text: r for r in store.all()}
    seen: set = set()
    records: list = []
    for f in facts:
        if f in seen:
            continue                          # collapse duplicate lines
        seen.add(f)
        keep = existing.get(f)
        records.append(keep if keep is not None else MemoryRecord(
            text=f, kind="semantic", source="user", importance=0.8))
    store.replace(records, embed_fn=_embed_fn())
    return {"status": "saved", "count": len(records)}


@_router.post("/api/memory/append")
async def memory_append(req: MemoryAppend):
    if not _persist_enabled():
        raise HTTPException(
            403, "Memory writes are off in privacy mode (no new traces)")
    fact = _strip_bullet(req.text)
    if not fact:
        raise HTTPException(400, "Nothing to remember")
    from localm.memory import N_MAX, MemoryRecord
    store = _chat_store()
    _migrate_legacy(store)
    # Refuse to append past the cap rather than accept a fact the next prune
    # would silently evict (audit F4).
    if len(store) >= N_MAX:
        raise HTTPException(
            413, f"Memory is at its {N_MAX}-record cap; delete a fact before "
                 "adding another.")
    rec = store.add(MemoryRecord(text=fact, kind="semantic", source="user",
                                 importance=0.8), embed_fn=_embed_fn())
    return {"status": "appended", "id": rec.id}


@_router.patch("/api/memory/{mem_id}")
async def memory_patch(mem_id: str, req: MemoryPatch):
    if not _persist_enabled():
        raise HTTPException(403, "Memory writes are off in privacy mode")
    store = _chat_store()
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
async def memory_delete(mem_id: str):
    if not _persist_enabled():
        raise HTTPException(403, "Memory writes are off in privacy mode")
    store = _chat_store()
    return {"status": "deleted" if store.delete(mem_id) else "absent", "id": mem_id}


@_router.post("/api/memory/consolidate")
async def memory_consolidate():
    """Manually distil durable facts from recent sessions into the store (the
    opt-in consolidation trigger). Gated on privacy; needs a loaded model."""
    if not _persist_enabled():
        raise HTTPException(403, "Memory writes are off in privacy mode")
    if _ENGINE is None or not getattr(_ENGINE, "loaded", False):
        raise HTTPException(503, "Load a model first to consolidate memory")

    def complete(prompt: str) -> str:
        # strip_think: memory must never ingest the reasoning channel (audit C1
        # store-poisoning; the /v1 routes strip it for clients, internal
        # consumers must do the same).
        from localm.inference.textnorm import strip_think
        return strip_think("".join(_ENGINE.chat_stream(
            [{"role": "user", "content": prompt}]))).strip()

    return synthesize_memory(complete)


# ------------------------------------------------------------------ #
#  Memory consolidation (distil durable facts from finished sessions) #
# ------------------------------------------------------------------ #
# Read recent session logs and fold durable user facts into the structured store
# via the ADD/UPDATE/DELETE/NO_OP consolidation loop (localm/memory/consolidate).
# Runs OUT OF BAND: as the jobs "memory" task (the runner binds `complete` to the
# model) or the POST /api/memory/consolidate route - never in the chat hot path.
# Privacy: only log/full sessions exist to read, AND the write path is gated on
# _persist_enabled() - in privacy mode this SKIPS and says so (RULE 5: a
# privacy-blocked write must never report success; the model is never called).


def _recent_sessions_text(max_chars: int = 8000) -> str:
    """User+assistant content from the newest session JSONL logs (written only in
    log/full mode), newest file first, capped to *max_chars*. Empty when none."""
    sdir = _home() / "sessions"
    if not sdir.is_dir():
        return ""
    files = sorted(sdir.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[str] = []
    total = 0
    for f in files:
        try:
            raw = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in raw.splitlines():
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
            out.append(piece)
            total += len(piece)
        if total >= max_chars:
            break
    return "\n".join(out)[:max_chars]


def _strip_bullet(line: str) -> str:
    """Strip any leading list markers (a model may emit '- ', '* ', or even a
    nested '- - ') and surrounding whitespace, leaving the bare fact."""
    return re.sub(r"^[\s\-*]+", "", line).strip()


def synthesize_memory(complete, *, max_facts: int = 12,
                      max_chars: int = 8000) -> dict:
    """Distil durable user facts from recent sessions into the structured store.

    *complete* is an injected ``(prompt: str) -> str`` model call (the jobs runner
    binds it to the engine; tests pass a fake). Delegates to the memory
    consolidation loop (ADD/UPDATE/DELETE/NO_OP + decay), so the store stays small
    and non-contradictory rather than an ever-growing flat list.

    Privacy: gated on _persist_enabled(); in privacy mode returns
    ``{"status": "skipped", "reason": "privacy", "added": 0}`` and NEVER calls the
    model - it never reports a success it did not perform. Returns
    ``{status, added, updated, deleted, facts}`` (facts = the texts newly added),
    the shape the jobs "memory" runner consumes.
    """
    if not _persist_enabled():
        return {"status": "skipped", "reason": "privacy", "added": 0}
    sessions = _recent_sessions_text(max_chars=max_chars)
    if not sessions.strip():
        return {"status": "skipped", "reason": "no_sessions", "added": 0}
    from localm.memory import run_consolidation
    store = _chat_store()
    _migrate_legacy(store)
    # Report which facts were newly added by diffing record ids. Ids are stable
    # across consolidation (store.replace reuses the same record objects - it
    # never mints new ids or duplicates a record), so an UPDATE reuses its id
    # (not counted here) and only true ADDs appear. Vectors: replace embeds ids
    # that have none, and run_consolidation invalidates an UPDATEd record's old
    # vector first so its embedding tracks the NEW text (it used to stay stale
    # forever; memory-audit 2026-07-02).
    before = {r.id for r in store.all()}
    res = run_consolidation(store, sessions, complete, embed_fn=_embed_fn(),
                            surface="chat", max_candidates=max_facts)
    new_facts = [r.text for r in store.all() if r.id not in before]
    episodic = _store_episode(store, sessions, complete)
    return {"status": res.get("status", "ok"), "added": res.get("added", 0),
            "updated": res.get("updated", 0), "deleted": res.get("deleted", 0),
            "episodic": episodic, "facts": new_facts}


def _store_episode(store, sessions: str, complete) -> int:
    """Store a one-line EPISODIC summary of the session (what was discussed) so
    chat recalls past topics, not only durable facts. Deduped against existing
    episodic records; best-effort. Returns 1 if one was stored, else 0. The caller
    has already confirmed writes are allowed (privacy)."""
    try:
        from localm.memory import MemoryRecord, summarize_session
        summ = summarize_session(complete, sessions)
        if not summ:
            return 0
        from difflib import SequenceMatcher
        lo = summ.lower()
        for r in store.all():
            if r.kind == "episodic" and \
                    SequenceMatcher(None, lo, r.text.lower()).ratio() > 0.85:
                return 0                         # already have this episode
        store.add(MemoryRecord(text=summ, kind="episodic", source="synth",
                               importance=0.4), embed_fn=_embed_fn())
        return 1
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("episodic summary skipped: %s", e)
        return 0


# ------------------------------------------------------------------ #
#  Prompt library (personas)                                          #
# ------------------------------------------------------------------ #
# Named personas: a system prompt plus sampling defaults. Explicit user assets
# (like knowledge collections), stored in prompts.json in every session mode.

def _prompts_file() -> Path:
    return _home() / "prompts.json"


def _check_prompt_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 64 or any(c in name for c in "\n\r\t"):
        raise HTTPException(
            400, "Persona names must be 1-64 characters on one line")
    return name


def _load_prompts() -> dict:
    prompts_file = _prompts_file()
    if prompts_file.is_file():
        try:
            return json.loads(prompts_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_prompts(data: dict) -> None:
    prompts_file = _prompts_file()
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = prompts_file.with_name(prompts_file.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(prompts_file)


@_router.get("/api/prompts")
async def prompts_list():
    data = _load_prompts()
    return {"prompts": [
        {"name": name, **entry} for name, entry in sorted(data.items())
    ]}


@_router.put("/api/prompts/{name}")
async def prompt_upsert(name: str, req: PromptUpsert):
    name = _check_prompt_name(name)
    data = _load_prompts()
    data[name] = {"system": req.system, "params": req.params}
    _save_prompts(data)
    return {"status": "saved", "name": name}


@_router.delete("/api/prompts/{name}")
async def prompt_delete(name: str):
    name = _check_prompt_name(name)
    data = _load_prompts()
    if name not in data:
        raise HTTPException(404, f"No such persona: {name}")
    del data[name]
    _save_prompts(data)
    return {"status": "deleted", "name": name}


# ------------------------------------------------------------------ #
#  CHAT-2b: thinking-model <think> nudge for regular chat             #
# ------------------------------------------------------------------ #

_THINK_INSTRUCTION = (
    "If reasoning helps, think step by step inside <think> and </think> tags "
    "before your final answer. Everything inside <think>...</think> is your "
    "private scratchpad and is not shown to the user."
)


def _thinking_inlet(messages, ctx):
    """Nudge a thinking/reasoning model to emit <think> markers in regular chat.

    Coder sessions already carry this instruction in their system prompt, but
    plain chat never did (CHAT-2b), so a model that needs the explicit nudge
    produced no reasoning channel. Inject only for thinking-family models, and
    never twice: skip when a system message already steers <think> (the coder
    case, or a persona that already does it). Appends to the first system
    message - chat templates commonly honour only the first - otherwise inserts
    one. The kernel pipeline isolates any exception this raises."""
    from localm.inference.model_family import is_thinking_model

    if not is_thinking_model(getattr(ctx, "model_id", "") or ""):
        return None
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str) \
                and "<think>" in m["content"]:
            return None                          # already instructed; don't double up
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            m["content"] = m["content"].rstrip() + "\n\n" + _THINK_INSTRUCTION
            return messages
    messages.insert(0, {"role": "system", "content": _THINK_INSTRUCTION})
    return messages


# ------------------------------------------------------------------ #
#  Server-side memory injection (the single injection point)          #
# ------------------------------------------------------------------ #
# Recall the user's durable memories relevant to the latest message and inject
# them, neutralised + fenced as data-not-instructions, into the system message.
# Runs for EVERY /v1/chat/completions client (GUI, API, coder-via-localm), which
# is why the SPA no longer prepends memory client-side (no double injection).

# The inference engine handle, stashed at register() for the manual
# POST /api/memory/consolidate route (which needs a model). None until wired.
_ENGINE = None


def _last_user_text(messages) -> str:
    """The most recent user message's text, used only as the recall QUERY.

    For multimodal content (a list of parts) the text parts are joined - this is a
    best-effort query, not a reconstruction of what the model saw, so exact phrasing
    does not matter; it only steers relevance ranking (and the recalled memories are
    neutralised before injection regardless)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text")
    return ""


# --- automatic (unattended) memory formation ------------------------------- #
# The chat model learns from conversations WITHOUT any manual step: after a turn
# completes (log/full mode only), a debounced background pass distils durable
# facts into the store. This is the capability the audit found missing - the
# consolidation loop existed but only ran via a manual route or a hand-created
# jobs task, so a default install never accumulated a single memory
# (memory-audit 2026-07-02, F7). Privacy is unchanged: writes stay gated, so
# nothing runs in privacy mode.

import threading as _threading

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


def _auto_consolidate_bg() -> None:
    """Run one consolidation pass in the background, binding the model to the
    stashed engine. Never raises (a daemon thread); clears the in-progress flag
    and stamps the marker when done."""
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

        res = synthesize_memory(complete)
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


def _maybe_auto_consolidate() -> None:
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
    # The debounce interval is overridable via env (power users / tests that
    # cannot wait the default 15 min); malformed values fall back to the default.
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
        # Stamp immediately so a burst of concurrent turns cannot each spawn a
        # pass before the first finishes (the finally-stamp refreshes it).
        _auto_stamp(now)
    _threading.Thread(target=_auto_consolidate_bg, daemon=True).start()


def _memory_outlet(text, messages, ctx):
    """After a completed turn, opportunistically grow the memory in the
    background (debounced). Side-effect only: returns the text unchanged. Any
    failure is contained so it can never affect the reply."""
    try:
        _maybe_auto_consolidate()
    except Exception as e:
        from localm.debuglog import logger
        logger.debug("memory outlet skipped: %s", e)
    return text


def _memory_inlet(messages, ctx):
    """Inject recalled memories into the system message. Gated on ``memory_enabled``
    config; reinforcement (a write) only when the session mode allows it. In
    privacy mode with an empty structured store, the legacy flat memory is still
    injected READ-ONLY (recall is not amnesia). Best-effort: any failure is logged
    at debug and skipped - never breaks the turn (the pipeline also isolates it)."""
    try:
        from localm.config import load_config
        if not load_config().get("memory_enabled", True):
            return None
    except Exception:
        pass                                       # config unreadable -> default on
    try:
        from localm import memory as _mem
        query = _last_user_text(messages)
        if not query.strip():
            return None
        store = _chat_store()
        allow = _mem.writes_allowed("chat")
        if allow:
            _migrate_legacy(store)
        block_records = store.recall(query, k=_mem.MAX_INJECT,
                                     embed_fn=_embed_fn(), reinforce=allow)
        if not block_records and not allow:
            block_records = [{"text": b}
                             for b in _legacy_bullets()[:_mem.MAX_INJECT]]
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
    host.register_chat_hook("inlet", _thinking_inlet)
    host.register_chat_hook("inlet", _memory_inlet)
    # Outlet: after each turn, grow the memory unattended (debounced, background,
    # log/full mode only). This is what makes memory accumulate with no manual
    # step (memory-audit 2026-07-02, F7).
    host.register_chat_hook("outlet", _memory_outlet)
    try:
        _ENGINE = host.engine()
    except Exception:
        _ENGINE = None


def unregister() -> None:
    pass
