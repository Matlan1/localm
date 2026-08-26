# SPDX-License-Identifier: AGPL-3.0-or-later
"""The list of projects this instance has run a coder session in.

A small central file recording only WHERE work happened, never WHAT was done.
Neither session store can answer "what have I worked on": ``full`` mode writes to
``<project>/.localcoder/sessions/`` INSIDE each project, so enumerating from it
would mean scanning the disk, and ``log`` mode writes to ``<data dir>/sessions/``,
which indexes SESSIONS and never sees a ``full``-mode session at all.

PRIVACY MODE WRITES NOTHING HERE. THIS IS NOT CONFIGURABLE.

A consequence callers must surface rather than hide: this list is INCOMPLETE by
construction for anyone who uses privacy mode, and must never be presented as the
whole history.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from localm.debuglog import logger
from localm.pathsafe import is_unc_or_device_path

# The mode that must never be recorded. Compared case-insensitively because the CLI
# accepts the choice case-insensitively (cli/_main.py) and a differently-cased value
# reaching here must not silently become recordable.
PRIVACY_MODE = "privacy"

_FILENAME = "coder-projects.json"


def _store() -> Path:
    from localm.config import home_dir
    return home_dir() / _FILENAME


def _load() -> list:
    try:
        raw = json.loads(_store().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:
        # A corrupt store is reported but must not break a coder session: this list
        # is a convenience.
        logger.warning("coder projects: could not read %s (%s); starting a fresh "
                       "list. The old file is left in place.", _store().name, e)
        return []
    return raw if isinstance(raw, list) else []


def _remember_enabled() -> bool:
    try:
        from localm.config import load_config
        return bool(load_config().get("coder_remember_projects", True))
    except Exception:
        return True


def _limit() -> int:
    try:
        from localm.config import load_config
        n = int(load_config().get("coder_projects_remembered", 20))
    except Exception:
        return 20
    return max(0, n)


def record_project(cwd, mode: Optional[str]) -> bool:
    """Record that a session ran in *cwd*. Returns True when something was written.

    Refuses, in this order:

    1. ``mode`` is privacy. NOT configurable.
    2. The user turned remembering off.
    3. The limit is zero.

    Never raises: a failure to record is logged and the session continues.
    """
    if (mode or "").strip().lower() == PRIVACY_MODE:
        return False
    if not _remember_enabled():
        return False
    limit = _limit()
    if limit <= 0:
        return False
    try:
        path = str(Path(cwd).expanduser().resolve())
    except Exception as e:
        logger.debug("coder projects: unusable cwd %r (%s); not recorded", cwd, e)
        return False

    entries = [e for e in _load() if isinstance(e, dict) and e.get("path") != path]
    prior = next((e for e in _load()
                  if isinstance(e, dict) and e.get("path") == path), {})
    entries.insert(0, {
        "path": path,
        "name": Path(path).name or path,
        "last_used": time.time(),
        "sessions": int(prior.get("sessions", 0)) + 1,
    })
    del entries[limit:]
    try:
        p = _store()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("coder projects: could not record %s (%s)", path, e)
        return False
    return True


def list_projects() -> list:
    """Every remembered project, most recent first, each tagged ``available``.

    A path that no longer exists is RETURNED, marked unavailable, rather than
    filtered out, so the caller can report it as moved or deleted.
    """
    out = []
    for e in _load():
        if not isinstance(e, dict) or not e.get("path"):
            continue
        try:
            # A UNC or device path is reported unavailable WITHOUT the stat:
            # is_dir() on a UNC path is a network round trip that blocks until it
            # times out, and the store is a plain JSON file a user can edit.
            if is_unc_or_device_path(str(e["path"])):
                available = False
            else:
                available = Path(e["path"]).is_dir()
        except Exception:
            available = False
        out.append({**e, "available": available})
    return out


def forget_all() -> None:
    """Delete the whole list."""
    try:
        _store().unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("coder projects: could not clear the list (%s)", e)
