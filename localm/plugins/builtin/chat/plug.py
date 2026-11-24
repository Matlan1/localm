# SPDX-License-Identifier: AGPL-3.0-or-later
"""Chat plugin: the built-in, protected, default-enabled plugin #0.

This is the reference implementation of the plugin contract - the cleanest,
most ordinary use of register(host)/mount_router that third-party plugins copy.
It owns the chat EXPERIENCE persistence (mounted by the engine, auto-scoped to
the ``chat`` capability):

  GET/PUT/DELETE /api/conversations[/{id}]   - server-side conversation store
  GET/PUT/DELETE /api/prompts[/{name}]       - prompt library / personas
  POST           /api/chat/extract           - attachment -> text (in memory)

Durable chat MEMORY (recall + consolidation + the /api/memory* routes) is a
SEPARATE, opt-in plugin (localm/plugins/builtin/memory); chat does not depend on
it. With memory disabled, chat simply runs without recall.

The actual LLM turn stays in the kernel: the SPA POSTs to /v1/chat/completions
on the inference server directly. Chat ships installed + enabled and cannot be
uninstalled or disabled (catalog: preinstalled + protected; manifest:
default_enabled), so the engine auto-provisions it on first run.

All paths resolve from the data dir at request time, so the plugin needs no
shared services from attach_gui. Persistence is gated on the chat surface's
session mode (see _persist_enabled): in privacy mode (the default) the store is
off entirely - writes 403 and the list/get routes return empty/403 - so a
conversation lives only in the browser tab for the current session and is gone
on reload (no new traces; the GUI also wipes its localStorage copy when it
confirms privacy mode). In log/full mode conversations are stored under
the data dir and survive reloads and other devices on the LAN.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import secrets
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from localm import model_ids
from localm.config import model_index
from localm.storekit import (_REPLACE_RETRIES, NamespaceLockRegistry, _is_transient_permission_error,
                             _transient_backoff, atomic_write)

_router = APIRouter()


# ------------------------------------------------------------------ #
#  Request models                                                     #
# ------------------------------------------------------------------ #

class ConversationUpsert(BaseModel):
    title: str = "Untitled"
    updated_at: float = 0
    pinned: bool = False
    folder: str | None = None
    # The model every turn of this conversation is answered by, whatever it
    # needs; None when the conversation is not pinned to one.
    pinned_model: str | None = Field(None, max_length=256)
    branches: list = []           # parked message-branch tails (fork points)
    messages: list = []
    # The conversation's own system prompt and sampling fields, by field name.
    # Omitted, the stored ones are kept.
    params: dict = Field(default_factory=dict)
    # The stored revision this save replaces; null for a conversation the caller
    # has never seen stored. Omitted, see conversation_upsert.
    base_rev: str | None = Field(None, max_length=64)
    # A value the caller made up for this save; the index row of the stored
    # conversation names the one that wrote it.
    save_id: str | None = Field(None, max_length=64)
    # The install and the key the caller means to save under (see _store_owner).
    instance: str | None = Field(None, max_length=128)
    principal: str | None = Field(None, max_length=64)


class AttachmentExtractRequest(BaseModel):
    filename: str
    content_b64: str              # in-memory extraction - no disk writes
    # None = the whole file. A number requests an excerpt of that many characters.
    max_chars: int | None = None


class PromptUpsert(BaseModel):
    system: str = ""
    params: dict = {}             # sampling defaults (temperature, top_p, ...)


# ------------------------------------------------------------------ #
#  Shared helpers (paths resolved at request time)                    #
# ------------------------------------------------------------------ #

_CONV_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}\Z")
_CONV_MAX_BYTES = 96 * 1024 * 1024   # data-URI images and base64 audio clips make these large

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


def _conv_with_current_pin(data: dict) -> dict:
    """Saved conversation *data* with ``pinned_model`` holding the pinned model's
    current name and without the stored ``model_refs``."""
    if not isinstance(data, dict):
        return data
    refs = data.pop(model_ids.REFS_KEY, None)
    if data.get("pinned_model"):
        data["pinned_model"] = model_ids.stored_to_current(
            data["pinned_model"], refs, model_index())
    return data


# Serialisation scope: readers and writers inside this process (request handlers
# and the executor thread running relabel_pinned_model) take one lock per target
# file.
# The CLI and other processes are not covered; relabel_pinned_model re-checks
# the file's updated_at before it replaces it.
_LOCKS = NamespaceLockRegistry()


def _riding_out(work, *args):
    """``work(*args)``, a file operation, retried like storekit.replace_atomic
    while a transient Windows sharing violation refuses it."""
    attempt = 0
    while True:
        try:
            return work(*args)
        except PermissionError as e:
            attempt += 1
            if attempt == _REPLACE_RETRIES or not _is_transient_permission_error(e):
                raise
            _transient_backoff(attempt - 1)


def _load_json(path: Path):
    return json.loads(_riding_out(path.read_text, "utf-8"))


def _principal_dir(request: Request) -> str | None:
    """The per-caller storage namespace: None for the owner (the shared files),
    otherwise the caller's key hash."""
    from localm.inference.http_server import memory_principal
    prin = memory_principal(request)
    if prin is None:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prin):
        raise HTTPException(403, "Unusable caller identity")
    return prin


def _chats_root(principal: str | None) -> Path:
    base = _home() / "chats"
    return base if principal is None else base / "_principals" / principal / "chats"


def _persist_enabled() -> bool:
    """Chat persistence/writes are off in privacy mode (the default)."""
    from localm.audit import automatic_write_allowed
    return automatic_write_allowed("chat")


# ------------------------------------------------------------------ #
#  Conversation store                                                 #
# ------------------------------------------------------------------ #
# Server-side persistence for GUI chat conversations so they survive browser
# reloads, profile wipes, and other devices on the LAN.

def _conv_path(conv_id: str, principal: str | None = None) -> Path:
    if not _CONV_ID.match(conv_id) or conv_id.lower() in _RESERVED_NAMES:
        raise HTTPException(400, "Invalid conversation id")
    return _chats_root(principal) / f"{conv_id}.json"


# Caches the projected index row per file keyed by (mtime, size), so the sidebar
# listing does not re-parse every chat JSON (with its embedded data-URI images)
# on every request. A changed file re-parses; the cache is bounded by the number
# of chat files. It only ever holds the lightweight meta row, never bodies.
_META_CACHE: dict = {}


# The revision of a file saved before conversations had one.
_NO_REV = "0"


def _plain_text(value, default=None):
    """*value* when it is text, with what UTF-8 cannot hold replaced, else
    *default*."""
    return value.encode("utf-8", "replace").decode("utf-8") if isinstance(value, str) else default


def _rev_of(data) -> str:
    """The stored revision of conversation *data*: an opaque string, replaced on
    every save."""
    rev = data.get("rev") if isinstance(data, dict) else None
    return _plain_text(rev) or _NO_REV


def _meta_row(conv_id: str, data: dict) -> dict:
    """The index row of conversation *data* stored under *conv_id*."""
    at = data.get("updated_at", 0)
    messages = data.get("messages")
    return {
        "id": conv_id,
        "title": _plain_text(data.get("title"), "Untitled"),
        "updated_at": at if isinstance(at, int) or (isinstance(at, float) and math.isfinite(at)) else 0,
        "pinned": bool(data.get("pinned", False)),
        "folder": _plain_text(data.get("folder")),
        "n_messages": len(messages) if isinstance(messages, list) else 0,
        "rev": _rev_of(data),
        "save_id": _plain_text(data.get("save_id")),
    }


_FIELD_NAME = re.compile(r"[A-Za-z_]{1,32}\Z")


def _not_storable(value, where: str = "") -> str | None:
    """What in *value* a conversation file cannot hold, None when nothing: a
    number that is not finite, text UTF-8 cannot hold, or a text part of a message
    whose text is not text. The answer names the place, never the content."""
    todo = [(where, "", value)]
    while todo:
        at, key, v = todo.pop()
        if isinstance(v, str):
            try:
                v.encode("utf-8")
            except UnicodeEncodeError:
                return f"{at} is not valid text"
        elif isinstance(v, float):
            if not math.isfinite(v):
                return f"{at} is not a finite number"
        elif isinstance(v, list):
            todo.extend((f"{at}[{i}]", key, x) for i, x in enumerate(v))
        elif isinstance(v, dict):
            if key == "content" and v.get("type") == "text" and not isinstance(v.get("text") or "", str):
                return f"{at}.text is not text"
            for k, x in v.items():
                name = k if _FIELD_NAME.match(k) else "?"
                todo.append((f"{at}.{name}".lstrip("."), k, x))
                todo.append((f"a field name in {at or 'it'}", "", k))
    return None


def _conv_meta(p: Path) -> dict:
    """Lightweight index row for one chat file (no message bodies / images)."""
    st = p.stat()
    key = (st.st_mtime, st.st_size)
    cached = _META_CACHE.get(str(p))
    if cached and cached[0] == key:
        return dict(cached[1])
    meta = _meta_row(p.stem, json.loads(p.read_text(encoding="utf-8")))
    _META_CACHE[str(p)] = (key, meta)
    return dict(meta)


def _remember_meta(p: Path, data: dict) -> None:
    """Put the index row of *data*, just written to *p*, in the cache."""
    try:
        st = p.stat()
    except OSError:
        _META_CACHE.pop(str(p), None)
        return
    _META_CACHE[str(p)] = ((st.st_mtime, st.st_size), _meta_row(p.stem, data))


def _stored_row(path: Path) -> dict | None:
    """The index row of the conversation stored at *path*, None when no file is
    there. A file that cannot be read or parsed has revision ``_NO_REV``."""
    try:
        return _riding_out(_conv_meta, path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError, AttributeError, TypeError):
        return {"id": path.stem, "rev": _NO_REV}


def _stored_data(path: Path) -> dict | None:
    """The conversation stored at *path*, None when it cannot be read."""
    try:
        data = _load_json(path)
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


# Fields of a message part that only say how it is shown or was answered.
_DISPLAY_ONLY = {"refusedBy", "notSent", "name", "seconds"}


def _media_of(m: dict) -> str:
    """A fingerprint of everything in message *m* beside its text: pictures,
    audio and video; empty when it has none."""
    content = m.get("content")
    parts = [{k: v for k, v in p.items() if k not in _DISPLAY_ONLY}
             for p in (content if isinstance(content, list) else ())
             if isinstance(p, dict) and p.get("type") != "text"]
    parts += [m[k] for k in ("audio", "video") if m.get(k)]
    if not parts:
        return ""
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()


def _turn_keys(messages, out: Counter) -> Counter:
    """Count in *out* one key per message of *messages*, those archived by a
    compaction included: who wrote it, its text and its media (a search or page
    read: what it asked for, or the text of one migrated from an older format)."""
    for m in messages if isinstance(messages, list) else ():
        if not isinstance(m, dict):
            continue
        if isinstance(m.get("text"), str):
            out[("user", m["text"], "")] += 1
        elif m.get("kind"):
            out[(str(m.get("kind")), str(m.get("query") or m.get("url") or ""), "")] += 1
        else:
            out[(str(m.get("role")), _message_text(m), _media_of(m))] += 1
        _turn_keys(m.get("compacted"), out)
    return out


def _all_turn_keys(messages, branches) -> Counter:
    """The counted keys of a conversation's timelines: the live one and every
    parked tail. The slot of a fork that belongs to the timeline on screen
    repeats it and is left out."""
    out = _turn_keys(messages, Counter())
    for rec in branches if isinstance(branches, list) else ():
        if not isinstance(rec, dict):
            continue
        tails = rec.get("tails")
        for i, tail in enumerate(tails if isinstance(tails, list) else ()):
            if i != rec.get("current"):
                _turn_keys(tail, out)
    return out


def _would_lose_turns(stored: dict | None, req: ConversationUpsert) -> bool:
    """True when the *stored* conversation holds a message, or a repetition of
    one, that *req* does not, or cannot be read."""
    if stored is None:
        return True
    try:
        have = _all_turn_keys(stored.get("messages"), stored.get("branches"))
    except RecursionError:
        return True
    return bool(have - _all_turn_keys(req.messages, req.branches))


_NOT_THIS_STORE = {"error": "owner",
                   "message": "This conversation belongs to another key or install."}


def _store_owner(principal: str | None) -> dict:
    """Which store a caller's conversations are in: this install's id and a tag
    for the key behind the request ("owner" for the owner)."""
    from localm.config import instance_id
    tag = "owner" if principal is None else hashlib.sha256(
        b"localm-chat-store:" + principal.encode("utf-8")).hexdigest()[:16]
    return {"instance": instance_id(), "principal": tag}


# ---- ids of deleted conversations -------------------------------------- #
# The one proof that a conversation that is not stored was deleted: a browser
# that still holds it must not store it again. One id per line, appended before
# the conversation's file is removed; a line "!<id>" takes an earlier one back,
# because the file could not be removed.

_DELETED_MAX = 5000


class _DeletedIdsUnreadable(Exception):
    pass


def _deleted_path(principal: str | None) -> Path:
    return _chats_root(principal) / "deleted-ids.txt"


def _deleted_ids(principal: str | None) -> list[str]:
    """The ids of conversations deleted from this store, oldest first; a line
    that is not an id is skipped. Raises _DeletedIdsUnreadable when the list
    exists and cannot be read."""
    path = _deleted_path(principal)
    try:
        with _LOCKS.get(str(path)):
            raw = _riding_out(path.read_bytes)
    except FileNotFoundError:
        return []
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("chat store: the list of deleted conversations could not "
                       "be read (%s)", type(e).__name__)
        raise _DeletedIdsUnreadable from e
    ids: dict[str, None] = {}
    for line in raw.decode("ascii", errors="replace").split("\n"):
        if _CONV_ID.match(line[1:] if line.startswith("!") else line):
            ids.pop(line.lstrip("!"), None)
            if not line.startswith("!"):
                ids[line] = None
    return list(ids)


def _is_deleted(conv_id: str, principal: str | None) -> bool:
    """Whether *conv_id* was deleted from this store. Answers 503 when the list
    of deleted conversations cannot be read."""
    try:
        return conv_id in _deleted_ids(principal)
    except _DeletedIdsUnreadable as e:
        raise HTTPException(
            503, "The list of deleted conversations cannot be read, so this was "
                 "not saved.") from e


def _remember_deleted(conv_id: str, principal: str | None, *, take_back: bool = False) -> None:
    """Append *conv_id* to the list of deleted conversations, or with *take_back*
    the line that takes it back. Raises OSError when it cannot be written. A list
    that has grown past twice _DELETED_MAX is cut to its newest ids, when all of
    it can be read."""
    path = _deleted_path(principal)
    line = ("!" if take_back else "") + conv_id
    with _LOCKS.get(str(path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with _riding_out(open, path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            torn = False
            if f.tell():
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
            f.write((b"\n" if torn else b"") + line.encode("ascii") + b"\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            ids = _deleted_ids(principal)
        except _DeletedIdsUnreadable:
            return
        if len(ids) > 2 * _DELETED_MAX:
            try:
                atomic_write(path, "\n".join(ids[-_DELETED_MAX:]) + "\n")
            except OSError as e:
                from localm.debuglog import logger
                logger.warning("chat store: the list of deleted conversations could "
                               "not be shortened (%s)", type(e).__name__)


def _count_chat_files(principal: str | None) -> int:
    """How many conversation files the caller's store holds."""
    chats_dir = _chats_root(principal)
    if not chats_dir.is_dir():
        return 0
    return sum(1 for _ in chats_dir.glob("*.json"))


def _message_text(m) -> str:
    content = m.get("content") if isinstance(m, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p["text"] for p in content if isinstance(p, dict)
                       and p.get("type") == "text" and isinstance(p.get("text"), str))
    return ""


def _content_snippet(data: dict, term: str) -> str | None:
    """Short excerpt around the first message match of *term* (lowercase), or None."""
    messages = data.get("messages")
    for m in messages if isinstance(messages, list) else ():
        text = _message_text(m)
        idx = text.lower().find(term)
        if idx != -1:
            start = max(0, idx - 18)
            return _plain_text(("\u2026" if start > 0 else "")
                               + " ".join(text[start:idx + 50].split()))
    return None


_UNREADABLE_LOGGED: set = set()


def _log_unreadable(p: Path, err: Exception) -> None:
    """Warn once per (file, mtime, size) that a chat file could not be listed.
    The conversation id and error type are logged, never the title or body."""
    try:
        st = p.stat()
        key = (str(p), st.st_mtime, st.st_size)
    except OSError:
        key = (str(p), None, None)
    if key in _UNREADABLE_LOGGED:
        return
    _UNREADABLE_LOGGED.add(key)
    from localm.debuglog import logger
    logger.warning("chat store: conversation %s is unreadable and not listed (%s)",
                   p.stem, type(err).__name__)


@_router.get("/api/conversations")
def conversations_list(request: Request, meta: bool = False,
                       limit: int = 0, offset: int = 0, q: str = ""):
    """List conversations newest-first.

    Pass ``meta=true`` for a lightweight index (id, title, updated_at, pinned,
    folder, n_messages) without message bodies or embedded images; fetch a
    conversation's messages with ``GET /api/conversations/{id}``. ``limit`` and
    ``offset`` paginate. With no parameters, returns the full payload capped at
    200 items.

    A chat file that cannot be read or parsed is left out of ``conversations``
    and ``total`` and its id is listed in ``unreadable``; a warning naming the
    id is logged once per file version. Each row carries ``rev``, the stored
    revision. ``instance`` and ``principal`` name the store: this install and
    the key behind the request.

    In privacy mode the store is off: the answer is ``{"enabled": false,
    "conversations": [], "stored": n}``, where ``n`` counts the conversation
    files saved earlier that are still on disk. None of them is opened.

    ``q`` (with ``meta=true``) keeps only the conversations whose title or
    message text contains it, case-insensitively, and adds a ``snippet`` to the
    rows that match by content: the sidebar search must find chats whose bodies
    the browser has not loaded.
    """
    if not _persist_enabled():
        principal = _principal_dir(request)
        return {"enabled": False, "conversations": [],
                "stored": _count_chat_files(principal), **_store_owner(principal)}
    term = q.strip().lower() if meta else ""
    principal = _principal_dir(request)
    chats_dir = _chats_root(principal)
    rows = []
    unreadable: list[str] = []
    if chats_dir.is_dir():
        for p in chats_dir.glob("*.json"):
            lock = _LOCKS.get(str(p))
            try:
                if meta:
                    with lock:
                        row = _riding_out(_conv_meta, p)
                    if term and term not in str(row["title"]).lower():
                        with lock:
                            data = _load_json(p)
                        snippet = _content_snippet(data, term)
                        if snippet is None:
                            continue
                        row["snippet"] = snippet
                    rows.append(row)
                else:
                    with lock:
                        data = _load_json(p)
                    data["id"] = p.stem
                    data["rev"] = _rev_of(data)
                    data["updated_at"] = _meta_row(p.stem, data)["updated_at"]
                    json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
                    rows.append(_conv_with_current_pin(data))
            except FileNotFoundError:
                continue
            except Exception as e:
                unreadable.append(p.stem)
                _log_unreadable(p, e)
    unreadable.sort()
    rows.sort(key=lambda c: c.get("updated_at", 0), reverse=True)
    total = len(rows)
    if limit and limit > 0:
        rows = rows[offset:offset + limit]
    elif offset:
        rows = rows[offset:]
    else:
        rows = rows[:200]   # default cap when unpaginated
    return {"enabled": True, "conversations": rows, "total": total,
            "unreadable": unreadable, **_store_owner(principal)}


@_router.get("/api/conversations/{conv_id}")
def conversation_get(conv_id: str, request: Request):
    """The full body of one conversation. Answers 410 when it was deleted, 404
    when it is not stored and no deletion is known, and 403 when chat persistence
    is off (privacy mode)."""
    if not _persist_enabled():
        raise HTTPException(403, "Chat persistence is off (privacy mode)")
    principal = _principal_dir(request)
    path = _conv_path(conv_id, principal)
    with _LOCKS.get(str(path)):
        if not path.is_file():
            if _is_deleted(conv_id, principal):
                raise HTTPException(410, "This conversation was deleted")
            raise HTTPException(404, "No such conversation")
        try:
            data = _load_json(path)
        except (OSError, ValueError, RecursionError) as e:
            raise HTTPException(500, "Conversation file is unreadable") from e
    if not isinstance(data, dict):
        raise HTTPException(500, "Conversation file is unreadable")
    data["id"] = conv_id
    data["rev"] = _rev_of(data)
    try:
        return JSONResponse(_conv_with_current_pin(data))
    except (ValueError, RecursionError) as e:
        raise HTTPException(500, "Conversation file is unreadable") from e


def _stale(row: dict | None, message: str, *, deleted: bool = False) -> HTTPException:
    return HTTPException(409, {"error": "stale", "message": message, "row": row,
                               "rev": row["rev"] if row else None, "deleted": deleted})


@_router.put("/api/conversations/{conv_id}")
def conversation_upsert(conv_id: str, req: ConversationUpsert, request: Request):
    """Store one conversation, replacing the stored copy. Answers with the new
    revision as ``rev``; every save gets a new one.

    ``base_rev`` names the stored revision this save replaces, and null says the
    caller has never seen the conversation stored. When the store holds another
    revision (the conversation was saved or deleted by someone else since, or one
    is stored already), nothing is written and the answer is 409 with
    ``{"detail": {"error": "stale", "message": ..., "rev": <stored revision or
    null>, "row": <its index row or null>, "deleted": <bool>}}``. A null ``rev``
    means the conversation is not stored any more; ``deleted`` is true only when
    it was deleted.

    A request without ``base_rev`` is written when no conversation is stored
    under this id and none was deleted, or when every message of the stored one
    is in the request; otherwise it is refused with the same 409.

    A request whose ``instance`` or ``principal`` is not the store it reaches is
    refused with 403 ``{"detail": {"error": "owner"}}`` and nothing is written.
    503 when the list of deleted conversations has to be known and cannot be
    read. 403 in privacy mode, 413 when the conversation is larger than the store
    accepts, 422 naming the place when it holds something a conversation cannot
    (see _not_storable)."""
    if not _persist_enabled():
        raise HTTPException(
            403, "Chat persistence is off (privacy mode). "
                 "Set mode/chat_mode to 'log' or 'full' to enable it.")
    principal = _principal_dir(request)
    path = _conv_path(conv_id, principal)
    owner = _store_owner(principal)
    for field in ("instance", "principal"):
        if getattr(req, field) and getattr(req, field) != owner[field]:
            raise HTTPException(403, _NOT_THIS_STORE)
    record = {"id": conv_id, "title": req.title,
              "updated_at": req.updated_at,
              "pinned": req.pinned, "folder": req.folder,
              "pinned_model": req.pinned_model,
              "params": {k: v for k, v in req.params.items()
                         if isinstance(v, (str, int, float))
                         and not isinstance(v, bool)},
              "rev": secrets.token_hex(8),
              "save_id": req.save_id,
              "branches": req.branches,
              "messages": req.messages}
    wrong = _not_storable(record)
    if wrong:
        raise HTTPException(422, f"This conversation was not saved: {wrong}.")
    if req.pinned_model:
        index = model_index()
        record["pinned_model"] = model_ids.stored_to_current(req.pinned_model, None, index)
        model_ids.attach_refs(record, (record["pinned_model"],), index)
    with _LOCKS.get(str(path)):
        row = _stored_row(path)
        deleted = row is None and _is_deleted(conv_id, principal)
        if deleted:
            raise _stale(None, "This conversation was deleted.", deleted=True)
        if "base_rev" in req.model_fields_set:
            if (row["rev"] if row else None) != req.base_rev:
                raise _stale(row, "This conversation was changed or removed elsewhere.")
        elif row and _would_lose_turns(_stored_data(path), req):
            raise _stale(row, "The stored conversation holds messages this save would "
                              "remove. Send base_rev with the revision it replaces.")
        if row and "params" not in req.model_fields_set:
            stored = (_stored_data(path) or {}).get("params")
            record["params"] = stored if isinstance(stored, dict) else {}
        try:
            payload = json.dumps(record, ensure_ascii=False)
        except RecursionError as e:
            raise HTTPException(422, "This conversation was not saved: it is nested too deeply.") from e
        if len(payload.encode("utf-8")) > _CONV_MAX_BYTES:
            raise HTTPException(413, "Conversation too large to persist")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, payload)
        _remember_meta(path, record)
    return {"status": "saved", "id": conv_id, "rev": record["rev"]}


@_router.delete("/api/conversations/{conv_id}")
def conversation_delete(conv_id: str, request: Request):
    """Delete one conversation and record its id, also when none is stored under
    it (``absent``), so a save still on its way does not store it. Answers 500
    when the deletion cannot be recorded or the file cannot be removed; the
    conversation then stays, and does not count as deleted. A request whose
    ``X-LocaLM-Chat-Store`` header (``<instance> <principal>``, as the list names
    them) is not the store it reaches is refused with 403
    ``{"detail": {"error": "owner"}}``."""
    if not _persist_enabled():
        raise HTTPException(403, "Chat persistence is off (privacy mode)")
    principal = _principal_dir(request)
    path = _conv_path(conv_id, principal)
    meant = request.headers.get("x-localm-chat-store")
    if meant is not None and meant != " ".join(_store_owner(principal).values()):
        raise HTTPException(403, _NOT_THIS_STORE)
    with _LOCKS.get(str(path)):
        try:
            _remember_deleted(conv_id, principal)
        except OSError as e:
            from localm.debuglog import logger
            logger.warning("chat store: a deletion could not be recorded (%s); "
                           "nothing was deleted", type(e).__name__)
            raise HTTPException(
                500, "The deletion could not be recorded, so nothing was deleted.") from e
        if path.is_file():
            try:
                _riding_out(path.unlink)
            except OSError as e:
                from localm.debuglog import logger
                logger.warning("chat store: conversation %s could not be deleted (%s)",
                               conv_id, type(e).__name__)
                try:
                    _remember_deleted(conv_id, principal, take_back=True)
                except OSError as e2:
                    logger.warning("chat store: the record that conversation %s was deleted "
                                   "could not be taken back (%s)", conv_id, type(e2).__name__)
                raise HTTPException(500, "The conversation could not be deleted.") from e
            _META_CACHE.pop(str(path), None)
            return {"status": "deleted", "id": conv_id}
    return {"status": "absent", "id": conv_id}


def _all_chat_files() -> list:
    """Every stored conversation file: the owner's and each principal's."""
    base = _home() / "chats"
    if not base.is_dir():
        return []
    files = list(base.glob("*.json"))
    files.extend(base.glob("_principals/*/chats/*.json"))
    return sorted(files)


def _updated_at_of(p: Path):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data.get("updated_at") if isinstance(data, dict) else None


def relabel_pinned_model(old_name: str, new_name: str) -> tuple[list, list]:
    """Rewrite ``pinned_model`` from *old_name* to *new_name* in every stored
    conversation (the owner's and every principal's) that carries it.

    A rewritten conversation gets a fresh ``updated_at`` and keeps its ``rev``:
    a later save naming *old_name* is stored under *new_name* anyway. Each file
    is rewritten under its per-file lock, and a file whose
    ``updated_at`` changed since it was read is left as it is and listed in
    ``failed``. A rewritten conversation's ``model_refs`` is set for *new_name*.
    Returns ``(rewritten, failed)``: the ids rewritten, and the ids
    of files that name *old_name* but were not rewritten. A file that does not
    contain *old_name* is not parsed. Works in every session mode: it edits
    files that already exist and creates none."""
    needle = json.dumps(old_name, ensure_ascii=False)[1:-1]
    index = model_index()
    rewritten: list = []
    failed: list = []
    for p in _all_chat_files():
        try:
            with _LOCKS.get(str(p)):
                text = p.read_text(encoding="utf-8")
                if needle not in text:
                    continue
                data = json.loads(text)
                if not isinstance(data, dict) or data.get("pinned_model") != old_name:
                    continue
                read_stamp = data.get("updated_at")
                data["pinned_model"] = new_name
                model_ids.attach_refs(data, (new_name,), index)
                data["updated_at"] = int(time.time() * 1000)
                if _updated_at_of(p) != read_stamp:
                    failed.append(p.stem)
                    continue
                atomic_write(p, json.dumps(data, ensure_ascii=False))
                rewritten.append(p.stem)
        except FileNotFoundError:
            continue
        except (OSError, ValueError, RecursionError) as e:
            from localm.debuglog import logger
            logger.warning("rename: conversation %s could not be rewritten (%s)",
                           p.stem, type(e).__name__)
            failed.append(p.stem)
    return rewritten, failed


# ------------------------------------------------------------------ #
#  Prompt library (personas)                                          #
# ------------------------------------------------------------------ #
# Named personas: a system prompt plus sampling defaults. Explicit user assets
# (like knowledge collections), stored in prompts.json in every session mode.

def _prompts_file(principal: str | None = None) -> Path:
    if principal is None:
        return _home() / "prompts.json"
    return _home() / "chats" / "_principals" / principal / "prompts.json"


def _check_prompt_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > 64 or any(c in name for c in "\n\r\t"):
        raise HTTPException(
            400, "Persona names must be 1-64 characters on one line")
    return name


class _PromptsUnreadable(Exception):
    """prompts.json EXISTS but could not be read or parsed."""


def _load_prompts(principal: str | None = None) -> dict:
    """The persona library, or ``{}`` when the user genuinely has none.

    Raises _PromptsUnreadable when the file EXISTS but cannot be read or
    parsed, which must NOT collapse into ``{}``: every writer below does
    read-modify-write, so the next save would replace the WHOLE library with the
    single entry being written. Same absent-vs-unreadable split as
    conversation_get above.
    """
    prompts_file = _prompts_file(principal)
    if not prompts_file.is_file():
        return {}                       # genuinely absent: an empty library
    try:
        data = json.loads(prompts_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as e:
        raise _PromptsUnreadable(str(e)) from e
    if not isinstance(data, dict):
        # Well-formed JSON that is not an object (a list, a bare string) parses
        # cleanly and is still not a library, so it is refused like a parse error.
        raise _PromptsUnreadable("prompts.json is not a JSON object")
    return data


def _prompts_or_refuse(principal: str | None = None) -> dict:
    """``_load_prompts()``, turning an unreadable library into a 500 that
    REFUSES the request.

    No caller may reach _save_prompts on a failed load. The HTTP message is
    path-free because this is a network surface; the concrete reason goes to the
    log."""
    try:
        return _load_prompts(principal)
    except _PromptsUnreadable as e:
        from localm.debuglog import logger
        logger.warning(
            "prompts.json exists but could not be read (%s); refusing the "
            "request rather than overwriting the persona library", e)
        raise HTTPException(500, "Prompt library is unreadable") from e


def _save_prompts(data: dict, principal: str | None = None) -> None:
    prompts_file = _prompts_file(principal)
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(prompts_file, json.dumps(data, ensure_ascii=False, indent=2))


@_router.get("/api/prompts")
async def prompts_list(request: Request):
    data = _prompts_or_refuse(_principal_dir(request))
    return {"prompts": [
        {"name": name, **entry} for name, entry in sorted(data.items())
    ]}


@_router.put("/api/prompts/{name}")
async def prompt_upsert(name: str, req: PromptUpsert, request: Request):
    name = _check_prompt_name(name)
    principal = _principal_dir(request)
    with _LOCKS.get(str(_prompts_file(principal))):
        data = _prompts_or_refuse(principal)
        data[name] = {"system": req.system, "params": req.params}
        _save_prompts(data, principal)
    return {"status": "saved", "name": name}


@_router.delete("/api/prompts/{name}")
async def prompt_delete(name: str, request: Request):
    name = _check_prompt_name(name)
    principal = _principal_dir(request)
    with _LOCKS.get(str(_prompts_file(principal))):
        data = _prompts_or_refuse(principal)
        if name not in data:
            raise HTTPException(404, f"No such persona: {name}")
        del data[name]
        _save_prompts(data, principal)
    return {"status": "deleted", "name": name}


# ------------------------------------------------------------------ #
#  Attachments                                                        #
# ------------------------------------------------------------------ #

@_router.post("/api/chat/extract")
async def chat_extract(req: AttachmentExtractRequest):
    """Extract plain text from a chat or coder attachment. The attachment is
    processed in memory and nothing is written to disk."""
    from localm.executor import get_plugin_executor
    from localm.rag.attachment import AttachmentError, extract_attachment
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            get_plugin_executor(), extract_attachment,
            req.filename, req.content_b64, req.max_chars)
    except AttachmentError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


# ------------------------------------------------------------------ #
#  Shared capacity guard for the fixed-cost system nudges below       #
# ------------------------------------------------------------------ #

# The same "getting close to capacity" fraction routes/chat.py already uses
# to trigger compaction.
_NUDGE_CAPACITY_FRACTION = 0.10


OWN_CHAT_SURFACE = "chat"


def _own_chat_surface(ctx) -> bool:
    """True when the request came from localm's own chat (the GUI chat or
    ``localm run``), which send ``X-Client-Id: chat`` (``ctx.state["client_id"]``).
    False for every other caller, an API client with no such header included."""
    state = getattr(ctx, "state", None) or {}
    client_id = state.get("client_id") or ""
    return isinstance(client_id, str) and client_id.strip().lower() == OWN_CHAT_SURFACE


def _style_note_enabled() -> bool:
    """The ``chat_style_note`` setting (default off)."""
    from localm.config import load_config
    return bool(load_config().get("chat_style_note", False))


def _nudge_fits_capacity(instruction: str, ctx) -> bool:
    """True when appending *instruction* is safe for the model's known
    context capacity: ``ctx.state["capacity"]``, an int token ceiling, or
    absent/None when unknown (always safe). Character count stands in for
    token count, so a ceiling this clears is genuinely safe: it never
    undercounts what the instruction will actually cost."""
    capacity = ctx.state.get("capacity")
    if not isinstance(capacity, int) or capacity <= 0:
        return True
    return len(instruction) <= capacity * _NUDGE_CAPACITY_FRACTION


# ------------------------------------------------------------------ #
#  Thinking-model <think> nudge for regular chat                      #
# ------------------------------------------------------------------ #

_THINK_INSTRUCTION = (
    "If reasoning helps, think step by step inside <think> and </think> tags "
    "before your final answer. Everything inside <think>...</think> is your "
    "private scratchpad and is not shown to the user."
)


def _thinking_inlet(messages, ctx):
    """Nudge a thinking/reasoning model to emit <think> markers in regular chat.

    Injects only on localm's own chat surface (see _own_chat_surface), only for
    thinking-family models, and never twice: skips when a system
    message already steers <think> (a coder session, whose system prompt already
    carries the instruction, or a persona that already does it), and skips when
    the model's known context capacity is too small to absorb the instruction
    (see _nudge_fits_capacity). Appends to the first system message - chat
    templates commonly honour only the first - otherwise inserts one. The
    kernel pipeline isolates any exception this raises."""
    from localm.inference.model_family import is_thinking_model

    if not _own_chat_surface(ctx):
        return None
    if not is_thinking_model(getattr(ctx, "model_id", "") or ""):
        return None
    if not _nudge_fits_capacity(_THINK_INSTRUCTION, ctx):
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
#  Role-word self-reference nudge for regular chat                    #
# ------------------------------------------------------------------ #

_ROLE_WORD_INSTRUCTION = (
    "Answer every question fully and directly. Style note: when you mention "
    "yourself, say \"I\" or \"me\" rather than a third person role word such as "
    "\"the model\" or \"the assistant\", and never use an incorrect pronoun for "
    "yourself."
)


def _has_text(content) -> bool:
    return isinstance(content, str) and bool(content.strip())


def _is_summary_request(messages) -> bool:
    """True for a compaction-summary request: the client's own "Summarise the
    following conversation" prompt, which must reach the model unchanged."""
    from localm.inference.compact import SUMMARY_INSTRUCTION
    stem = SUMMARY_INSTRUCTION.split(" in under", 1)[0]
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            return isinstance(c, str) and c.lstrip().startswith(stem)
    return False


def _role_word_inlet(messages, ctx):
    """Steer the model to speak of itself in the first person rather than a
    bare third-person role word ("Model", "the assistant") or an incorrect
    pronoun for itself.

    Applies only on localm's own chat surface (see _own_chat_surface) while the
    ``chat_style_note`` setting is on. Skips when any system message with text
    exists (a user or persona prompt is never extended), for a compaction
    summary request, and when the model's known context capacity is too small to absorb the
    instruction (see _nudge_fits_capacity). Appends the instruction to the
    first system message, or inserts one when none exists. Exceptions raised
    here are isolated by the chat pipeline."""
    if not _own_chat_surface(ctx) or not _style_note_enabled():
        return None
    if not _nudge_fits_capacity(_ROLE_WORD_INSTRUCTION, ctx):
        return None
    if any(m.get("role") == "system" and _has_text(m.get("content")) for m in messages):
        return None
    if _is_summary_request(messages):
        return None
    messages.insert(0, {"role": "system", "content": _ROLE_WORD_INSTRUCTION})
    return messages


def rename_notes(old_name: str, new_name: str) -> list[str]:
    """The rename hook: re-pin saved chats from *old_name* to *new_name* and
    return the notes to show the user."""
    notes: list[str] = []
    migrated, failed = relabel_pinned_model(old_name, new_name)
    if migrated:
        notes.append(f"Updated {len(migrated)} saved chat(s) pinned to it")
    if failed:
        notes.append(
            "Could not update the saved chat(s) "
            f"{', '.join(failed)}: they are still pinned to '{old_name}'")
    return notes


def register(host) -> None:
    from localm.model_manager import register_rename_hook
    host.mount_router(_router)
    host.register_chat_hook("inlet", _role_word_inlet)
    host.register_chat_hook("inlet", _thinking_inlet)
    register_rename_hook("saved chats", rename_notes)


def unregister() -> None:
    from localm.model_manager import unregister_rename_hook
    unregister_rename_hook("saved chats")
