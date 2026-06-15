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


@_router.get("/api/conversations")
async def conversations_list():
    if not _persist_enabled():
        return {"enabled": False, "conversations": []}
    chats_dir = _home() / "chats"
    items = []
    if chats_dir.is_dir():
        for p in chats_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                data["id"] = p.stem
                items.append(data)
            except Exception:
                continue   # corrupt file - skip, never block the list
    items.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
    return {"enabled": True, "conversations": items[:200]}


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


def register(host) -> None:
    host.mount_router(_router)


def unregister() -> None:
    pass
