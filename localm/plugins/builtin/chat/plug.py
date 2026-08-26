# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat plugin: the built-in, protected, default-enabled plugin #0.

This is the reference implementation of the plugin contract - the cleanest,
most ordinary use of register(host)/mount_router that third-party plugins copy.
It owns the chat EXPERIENCE persistence (mounted by the engine, auto-scoped to
the ``chat`` capability):

  GET/PUT/DELETE /api/conversations[/{id}]   - server-side conversation store
  GET/PUT/DELETE /api/prompts[/{name}]       - prompt library / personas

Durable chat MEMORY (recall + consolidation + the /api/memory* routes) is a
SEPARATE, opt-in plugin now (localm/plugins/builtin/memory); chat no longer
depends on it. With memory disabled, chat simply runs without recall.

The actual LLM turn stays in the kernel: the SPA POSTs to /v1/chat/completions
on the inference server directly. Chat ships installed + enabled and cannot be
uninstalled or disabled (catalog: preinstalled + protected; manifest:
default_enabled), so the engine auto-provisions it on first run.

All paths resolve from the data dir at request time, so the plugin needs no
shared services from attach_gui. Persistence is gated on the chat surface's
session mode (see _persist_enabled): in privacy mode (the default) the store is
off entirely - writes 403 and the list/get routes return empty/403 - so a
conversation lives only in the browser tab for the current session and is gone
on reload ("no new traces", by design; the GUI also wipes its localStorage copy
when it confirms privacy mode). In log/full mode conversations are stored under
the data dir and survive reloads and other devices on the LAN.
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


# ------------------------------------------------------------------ #
#  Shared helpers (paths resolved at request time)                    #
# ------------------------------------------------------------------ #

_CONV_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CONV_MAX_BYTES = 16 * 1024 * 1024   # data-URI images make these large

# Windows reserved device names, matched regardless of extension: a
# conversation id of "nul" would target <home>/chats/nul.json -> the NUL device
# on Windows, not a real file (writes discard, reads fail). Not a path escape -
# _CONV_ID's charset already confines this to a flat basename inside "chats".
# Same enumeration as rag/store.py's _RESERVED_NAMES, kept local.
_RESERVED_NAMES = {"con", "prn", "aux", "nul",
                   *(f"com{i}" for i in range(1, 10)),
                   *(f"lpt{i}" for i in range(1, 10))}


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
    if not _CONV_ID.match(conv_id) or conv_id.lower() in _RESERVED_NAMES:
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


class _PromptsUnreadable(Exception):
    """prompts.json EXISTS but could not be read or parsed."""


def _load_prompts() -> dict:
    """The persona library, or ``{}`` when the user genuinely has none.

    Raises _PromptsUnreadable when the file EXISTS but cannot be read or
    parsed, which must NOT collapse into ``{}``: every writer below does
    read-modify-write, so the next save would replace the WHOLE library with the
    single entry being written. Same absent-vs-unreadable split as
    conversation_get above.
    """
    prompts_file = _prompts_file()
    if not prompts_file.is_file():
        return {}                       # genuinely absent: an empty library
    try:
        data = json.loads(prompts_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise _PromptsUnreadable(str(e)) from e
    if not isinstance(data, dict):
        # Well-formed JSON that is not an object (a list, a bare string) parses
        # cleanly and is still not a library, so it is refused like a parse error.
        raise _PromptsUnreadable("prompts.json is not a JSON object")
    return data


def _prompts_or_refuse() -> dict:
    """``_load_prompts()``, turning an unreadable library into a 500 that
    REFUSES the request.

    No caller may reach _save_prompts on a failed load. The HTTP message is
    path-free because this is a network surface; the concrete reason goes to the
    log."""
    try:
        return _load_prompts()
    except _PromptsUnreadable as e:
        from localm.debuglog import logger
        logger.warning(
            "prompts.json exists but could not be read (%s); refusing the "
            "request rather than overwriting the persona library", e)
        raise HTTPException(500, "Prompt library is unreadable")


def _save_prompts(data: dict) -> None:
    prompts_file = _prompts_file()
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = prompts_file.with_name(prompts_file.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(prompts_file)


@_router.get("/api/prompts")
async def prompts_list():
    data = _prompts_or_refuse()
    return {"prompts": [
        {"name": name, **entry} for name, entry in sorted(data.items())
    ]}


@_router.put("/api/prompts/{name}")
async def prompt_upsert(name: str, req: PromptUpsert):
    name = _check_prompt_name(name)
    data = _prompts_or_refuse()
    data[name] = {"system": req.system, "params": req.params}
    _save_prompts(data)
    return {"status": "saved", "name": name}


@_router.delete("/api/prompts/{name}")
async def prompt_delete(name: str):
    name = _check_prompt_name(name)
    data = _prompts_or_refuse()
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
