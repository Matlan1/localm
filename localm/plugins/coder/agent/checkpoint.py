# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checkpoint location + indexing helpers, module-level so the GUI resume probe
can read a checkpoint (checkpoint_info) without constructing an Agent, and so the
startup-scan deadline (_index_deadline) is resolvable standalone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..indexer import _BUILD_DEADLINE_S

def _project_digest(cwd) -> str:
    """Stable per-project id: sha256 of the case-normalised, resolved cwd, so
    each project's resume checkpoint gets its own file under HOME (CODER-4)."""
    import hashlib
    import os
    key = os.path.normcase(str(Path(cwd).resolve()))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def _checkpoint_path_for(cwd) -> Path:
    """A project's resume checkpoint: ``HOME/checkpoints/<digest>.json``.

    Session DATA belongs in HOME, not the project tree (CODER-4) - the checkpoint
    used to land in ``<cwd>/.localcoder/checkpoint.json``, leaving a stray folder
    in the user's repo. Project-local config (.localcoder/config.toml), memory
    (LOCALCODER.md), and full-mode transcripts (.localcoder/sessions/) stay put;
    only the checkpoint moves. HOME_DIR is imported lazily so tests that
    monkeypatch ``config.HOME_DIR`` are honoured."""
    from localm.config import HOME_DIR
    return HOME_DIR / "checkpoints" / (_project_digest(cwd) + ".json")

def _legacy_checkpoint_path_for(cwd) -> Path:
    """The pre-CODER-4 in-project checkpoint path, still READ for back-compat so
    a session saved by an older build can still be resumed and cleaned up."""
    return Path(cwd) / ".localcoder" / "checkpoint.json"

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

def checkpoint_info(cwd) -> Optional[dict]:
    """Summary of a saved checkpoint for *cwd* (new HOME location, then legacy),
    or None - used by the GUI resume probe without building an Agent (CODER-2)."""
    for p in (_checkpoint_path_for(cwd), _legacy_checkpoint_path_for(cwd)):
        data = _read_checkpoint(p)
        if data is not None:
            return {
                "interrupted_at": data.get("interrupted_at"),
                "turns": data.get("turns", len(data["messages"])),
                "total_tokens": data.get("total_tokens", 0),
                "messages": len(data["messages"]),
            }
    return None

def _index_deadline() -> Optional[float]:
    """Wall-clock cap (seconds) for the startup project scan (CODER-1). Reads
    the registered ``coder_index_timeout`` setting (``localm config
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
