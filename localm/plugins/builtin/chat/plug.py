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
#  Assistant memory                                                   #
# ------------------------------------------------------------------ #
# A plain markdown file the user can read and edit, injected into the system
# prompt when the drawer toggle is on. Privacy semantics: "no new traces", not
# amnesia - READING memory from earlier non-privacy sessions is allowed, but
# WRITES (which persist conversation-derived facts) return 403 under privacy.

def _memory_file() -> Path:
    return _home() / "chat-memory.md"


def _read_memory() -> str:
    memory_file = _memory_file()
    if memory_file.is_file():
        try:
            return memory_file.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def _write_memory(text: str) -> None:
    text = text.strip()
    if len(text) > _MEMORY_MAX:
        raise HTTPException(413, "Memory file too large (max 64k chars)")
    memory_file = _memory_file()
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    if not text:
        memory_file.unlink(missing_ok=True)
        return
    tmp = memory_file.with_name(memory_file.name + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(memory_file)


@_router.get("/api/memory")
async def memory_get():
    return {"text": _read_memory(), "writable": _persist_enabled(),
            "path": str(_memory_file())}


@_router.put("/api/memory")
async def memory_put(req: MemoryUpdate):
    if not _persist_enabled():
        raise HTTPException(
            403, "Memory writes are off in privacy mode (no new traces). "
                 "Set mode/chat_mode to 'log' or 'full' to enable them.")
    _write_memory(req.text)
    return {"status": "saved", "chars": len(req.text.strip())}


@_router.post("/api/memory/append")
async def memory_append(req: MemoryAppend):
    if not _persist_enabled():
        raise HTTPException(
            403, "Memory writes are off in privacy mode (no new traces)")
    fact = req.text.strip()
    if not fact:
        raise HTTPException(400, "Nothing to remember")
    current = _read_memory().strip()
    line = fact if fact.startswith("-") else f"- {fact}"
    _write_memory((current + "\n" + line) if current else line)
    return {"status": "appended"}


# ------------------------------------------------------------------ #
#  Memory auto-synthesis (A2)                                         #
# ------------------------------------------------------------------ #
# Distil durable user facts from finished sessions into chat-memory.md so the
# model "remembers" across chats without the user typing /remember. Runs on a
# schedule as a jobs "memory" task (the jobs runner binds `complete` to the
# model). Privacy: only log/full sessions exist to read, AND writes are gated on
# _persist_enabled() - in privacy mode this SKIPS and says so (RULE 5: a
# privacy-blocked write must never report success).

_SYNTH_PREFIX = (
    "You maintain a long-term memory of durable facts about a user, built from "
    "their past conversations with an AI assistant. Below are recent exchanges.\n\n"
    "Extract ONLY durable, reusable facts about the user: their name, role, "
    "projects, tools, stable preferences, and recurring goals. IGNORE one-off "
    "questions, transient details, and anything that will not matter next week.\n"
    "Output one fact per line, each starting with '- '. Output nothing if there "
    "is nothing worth remembering.\n\n=== recent conversations ===\n"
)
_SYNTH_SUFFIX = "\n=== end ===\n\nDurable facts:"


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


def _existing_memory_lines() -> set:
    """Casefolded set of current memory fact lines (for dedupe)."""
    return {_strip_bullet(ln).casefold()
            for ln in _read_memory().splitlines() if ln.strip()}


def synthesize_memory(complete, *, max_facts: int = 12,
                      max_chars: int = 8000) -> dict:
    """Distil durable user facts from recent sessions into chat-memory.md.

    *complete* is an injected ``(prompt: str) -> str`` model call (the jobs runner
    binds it to the engine; tests pass a fake), so the deterministic logic here is
    unit-testable without a model.

    Privacy: writes are gated on _persist_enabled(); in privacy mode this returns
    ``{"status": "skipped", "reason": "privacy", "added": 0}`` - it never reports
    a success it did not perform.
    """
    if not _persist_enabled():
        return {"status": "skipped", "reason": "privacy", "added": 0}
    sessions = _recent_sessions_text(max_chars=max_chars)
    if not sessions.strip():
        return {"status": "skipped", "reason": "no_sessions", "added": 0}
    raw = complete(_SYNTH_PREFIX + sessions + _SYNTH_SUFFIX) or ""
    have = _existing_memory_lines()
    new_facts: list[str] = []
    for line in str(raw).splitlines():
        fact = _strip_bullet(line)
        if not fact or fact.casefold() in have:
            continue
        have.add(fact.casefold())
        new_facts.append(fact)
        if len(new_facts) >= max_facts:
            break
    if not new_facts:
        return {"status": "ok", "added": 0, "facts": []}
    current = _read_memory().strip()
    block = "\n".join(f"- {f}" for f in new_facts)
    _write_memory((current + "\n" + block) if current else block)
    return {"status": "ok", "added": len(new_facts), "facts": new_facts}


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


def register(host) -> None:
    host.mount_router(_router)
    host.register_chat_hook("inlet", _thinking_inlet)


def unregister() -> None:
    pass
