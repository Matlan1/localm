# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checkpoint location + indexing helpers, module-level so the GUI resume probe
can read a checkpoint (checkpoint_info) without constructing an Agent, and so the
startup-scan deadline (_index_deadline) is resolvable standalone."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from ..indexer import _BUILD_DEADLINE_S

def _project_digest(cwd) -> str:
    """Stable per-project id: sha256 of the case-normalised, resolved cwd, so
    each project's resume checkpoint gets its own file under HOME."""
    import hashlib
    import os
    key = os.path.normcase(str(Path(cwd).resolve()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def _project_dir_for(cwd) -> Path:
    """The directory holding EVERY saved checkpoint for this project - one file
    per session id, ``HOME/checkpoints/<project-digest>/``. HOME_DIR is imported
    lazily so tests that monkeypatch ``config.HOME_DIR`` are honoured."""
    from localm.config import HOME_DIR
    return HOME_DIR / "checkpoints" / _project_digest(cwd)

def _project_map_path_for(cwd) -> Path:
    """Where the cross-session project-map cache lives for *cwd*:
    ``HOME/checkpoints/<project-digest>/projectmap.json`` - a SIBLING of this
    project's per-session checkpoint files, not one of them: the map describes
    the filesystem, not a conversation, so every coder session in this project
    shares and refreshes the SAME cache file. See
    ``localm.plugins.coder.indexer.ProjectMap.load_cached_and_reconcile`` and
    ``save_cache`` for the read/write side."""
    return _project_dir_for(cwd) / "projectmap.json"

_ID_OK = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def is_valid_checkpoint_id(checkpoint_id: str) -> bool:
    """Is *checkpoint_id* safe to use as a filename component?

    Ids are minted as ``uuid4().hex[:12]``, so the real keyspace is hex; this
    accepts the slightly wider word-character set so a hand-migrated or
    future-format id is not rejected, while excluding every separator and dot.

    The id is concatenated into a path by _checkpoint_path_for and can arrive
    from an HTTP request body (the web resume_id): an id of
    ``../../../../windows/win.ini`` resolves clean outside the checkpoints tree,
    and load_checkpoint RETAINS the id it was given, so the next save_checkpoint
    - and the cwd-change migration - would mkdir and write there. Validated at
    the path helper rather than only at the route, so every caller is covered by
    construction.
    """
    return bool(_ID_OK.match(checkpoint_id or ""))


def _checkpoint_path_for(cwd, checkpoint_id: str) -> Path:
    """One session's resume checkpoint:
    ``HOME/checkpoints/<project-digest>/<checkpoint-id>.json``.

    Keyed on (project, session), not project alone: a path naming the project
    only has every session in it write the exact same file, so a second coder
    session started in the same project destroys the first session's saved
    conversation at its next save. *checkpoint_id* is generated once per Agent
    (``Agent._checkpoint_id`` in core.py) and carried across resume, so
    continuing a restored session keeps writing to the SAME file rather than
    minting a new one every save.

    Session DATA lives in HOME, not the project tree. Project-local config
    (.localcoder/config.toml), memory (LOCALCODER.md), and full-mode transcripts
    (.localcoder/sessions/) stay in the project; only the checkpoint is in
    HOME."""
    if not is_valid_checkpoint_id(checkpoint_id):
        raise ValueError(f"Invalid checkpoint id: {checkpoint_id!r}")
    return _project_dir_for(cwd) / (checkpoint_id + ".json")

def _legacy_checkpoint_path_for(cwd) -> Path:
    """The legacy in-project checkpoint path, still READ for back-compat so a
    session saved by an older build can still be resumed and cleaned up.

    The older single-file-per-project HOME path collapses onto this same
    read-only role once a session using it is next loaded:
    ``Agent.load_checkpoint()`` migrates it into the per-session layout - see
    ``_PersistenceMixin.load_checkpoint``."""
    return Path(cwd) / ".localcoder" / "checkpoint.json"

def _legacy_home_checkpoint_path_for(cwd) -> Path:
    """The legacy single checkpoint per project:
    ``HOME/checkpoints/<project-digest>.json`` (a FILE, sibling to the new
    per-project DIRECTORY of the same base name - the two can never collide on
    disk, a directory and a file cannot share one path). Read once for
    migration; never written again."""
    from localm.config import HOME_DIR
    return HOME_DIR / "checkpoints" / (_project_digest(cwd) + ".json")

def _read_checkpoint(p: Path) -> Optional[dict]:
    """Parse a checkpoint file, or None if absent / unreadable / wrong shape."""
    try:
        if not p.is_file():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("version") == 1 and isinstance(data.get("messages"), list):
        return data
    return None

#: Display titles are truncated to this many characters (a listing row, not a
#: transcript) - collapsed to one line first, since the source text is often
#: multi-line.
_TITLE_MAX_CHARS = 80

def _derive_title(raw: str) -> str:
    """A short, single-line display title from a session's first task/message
    text. *raw* must be the text BEFORE any episodic-memory preamble is
    prepended (see ``Agent._session_title`` capture in loop.py) - titling from
    the wrapped text would show boilerplate ("Relevant past lessons: ...")
    instead of what the user actually asked for."""
    collapsed = " ".join(raw.split())
    if not collapsed:
        return "(untitled session)"
    if len(collapsed) <= _TITLE_MAX_CHARS:
        return collapsed
    return collapsed[:_TITLE_MAX_CHARS - 1].rstrip() + "…"

def _entry_from_checkpoint(checkpoint_id: str, data: dict) -> dict:
    """The listing/probe row shape shared by ``list_checkpoints`` and
    ``checkpoint_info`` - one place so the two can never drift apart on which
    fields a row carries."""
    return {
        "id": checkpoint_id,
        "title": data.get("title") or "(untitled session)",
        "interrupted_at": data.get("interrupted_at"),
        "turns": data.get("turns", len(data["messages"])),
        "total_tokens": data.get("total_tokens", 0),
        "messages": len(data["messages"]),
        "changed_files": len(data.get("changed_files") or {}),
    }

def list_checkpoints(cwd) -> list:
    """Every saved checkpoint for *cwd*, NEWEST first, so a resume can choose
    among several interrupted sessions in one project.

    Sorted by file mtime, not ``interrupted_at``: mtime is stamped by the
    filesystem on every save, so it cannot be missing or hand-edited wrong the
    way a JSON field could, and it is what "newest" means here - when this file
    was last written, not when the interruption inside it happened.

    Best-effort per file: one unreadable or corrupt checkpoint is skipped, not
    raised."""
    d = _project_dir_for(cwd)
    if not d.is_dir():
        return []
    dated = []
    for p in d.glob("*.json"):
        data = _read_checkpoint(p)
        if data is None:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        dated.append((mtime, p.stem, data))
    dated.sort(key=lambda t: t[0], reverse=True)
    return [_entry_from_checkpoint(cid, data) for _mtime, cid, data in dated]

def checkpoint_info(cwd) -> Optional[dict]:
    """Summary of the MOST RECENT saved checkpoint for *cwd* across every
    session id, then the legacy single-file locations - used by the GUI resume
    probe without building an Agent. The zero-argument default answers "my last
    conversation here"; see ``list_checkpoints`` for choosing among all of
    them."""
    entries = list_checkpoints(cwd)
    if entries:
        return entries[0]
    for p in (_legacy_home_checkpoint_path_for(cwd), _legacy_checkpoint_path_for(cwd)):
        data = _read_checkpoint(p)
        if data is not None:
            return {
                "interrupted_at": data.get("interrupted_at"),
                "turns": data.get("turns", len(data["messages"])),
                "total_tokens": data.get("total_tokens", 0),
                "messages": len(data["messages"]),
            }
    return None

def _first_user_text(messages: list) -> str:
    """The first user message's raw text, or "" - the fallback title source
    ``resume_checkpoint`` uses for a checkpoint saved before "title" existed."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
    return ""

def migrate_legacy_checkpoint(cwd, legacy_path: Path, data: dict) -> str:
    """Adopt a checkpoint found at a legacy location into the per-session
    layout: mint a fresh id, write *data* under it (backfilling "title" when the
    legacy file predates that field too), and remove the legacy file so it is
    migrated ONCE, not re-read and re-copied on every future load. Returns the
    new checkpoint id.

    Never raises: on any I/O error it leaves *legacy_path* alone rather than
    deleting the only copy before the new one is confirmed, and the caller still
    gets usable in-memory data back."""
    import uuid
    new_id = uuid.uuid4().hex[:12]
    if not data.get("title"):
        data = {**data, "title": _derive_title(_first_user_text(data["messages"]))}
    new_path = _checkpoint_path_for(cwd, new_id)
    try:
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        legacy_path.unlink(missing_ok=True)
    except OSError:
        pass
    return new_id

def _index_deadline() -> Optional[float]:
    """Wall-clock cap (seconds) for the startup project scan. Reads the
    registered ``coder_index_timeout`` setting (``localm config
    coder_index_timeout N``, or the Settings page); a value <= 0 disables the
    deadline. Falls back to ``_BUILD_DEADLINE_S`` only if config itself is
    unreadable (this function must never raise)."""
    raw = None
    try:
        from localm.config import load_config
        raw = load_config().get("coder_index_timeout")
    except Exception:
        pass
    if raw is None:
        return _BUILD_DEADLINE_S
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _BUILD_DEADLINE_S
    return v if v > 0 else None
