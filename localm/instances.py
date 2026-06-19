# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry (H6 server-rework, phase 3 - advertise only).

Each running localm server advertises itself so a future launch can DISCOVER and
attach to it (phase 4) instead of spinning a second server that double-loads the
model - the "why is the coder an extra CLI / why a second server" smell. This
phase only writes/reaps the registry and serves ``GET /whoami``; it changes no
behavior.

Registry: one JSON file per instance under ``<LOCALM_HOME>/run/<instance_id>.json``::

    {instance_id, pid, port, host, root_dir, mode, version, started, token}

Modeled on Jupyter's ``runtime/*.json`` + Ollama's identity handshake. Liveness is
ultimately a ``GET /whoami`` returning ``app == "localm"`` (added in phase 4); this
phase reaps by process liveness and ownership. ``mode`` here is the SURFACE mode
(``api`` = bare OpenAI API, ``full`` = API + GUI) - distinct from the session
persistence mode (privacy|log|full), which is a different concept that happens to
share the word "full". The per-instance ``token`` lets a local client authenticate
when it attaches (phase 4); it stays in the 0600 registry file and is NEVER served
by ``/whoami``.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

APP_NAME = "localm"

# Markers that identify a project root, walked up from cwd (decision 2). Mirrors
# the coder's existing .localcoder model; .git covers most repos.
_ROOT_MARKERS = (".git", ".localcoder")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("localm")
    except Exception:
        return "0.1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
#  Paths + ids                                                        #
# ------------------------------------------------------------------ #

def run_dir(home: Path) -> Path:
    """Directory holding this install's instance registry."""
    return Path(home) / "run"


def registry_path(home: Path, instance_id: str) -> Path:
    return run_dir(home) / f"{instance_id}.json"


def new_instance_id() -> str:
    """A random per-process instance id (not the PID - PIDs get reused)."""
    return secrets.token_hex(8)


def new_token() -> str:
    """A per-instance attach token (used by local clients in phase 4)."""
    return secrets.token_urlsafe(32)


def resolve_root_dir(start: Optional[str] = None, override: Optional[str] = None) -> str:
    """The project root that keys this instance (decision 2): the nearest ancestor
    of *start* (default cwd) containing a ``.git`` or ``.localcoder`` marker, else
    *start* itself. *override* (a future ``--project`` flag) wins outright."""
    if override:
        return str(Path(override).expanduser().resolve())
    try:
        cur = Path(start).resolve() if start else Path.cwd().resolve()
    except OSError:
        cur = Path(start) if start else Path(".")
    for d in (cur, *cur.parents):
        for marker in _ROOT_MARKERS:
            try:
                if (d / marker).exists():
                    return str(d)
            except OSError:
                pass
    return str(cur)


# ------------------------------------------------------------------ #
#  Identity payload                                                   #
# ------------------------------------------------------------------ #

def whoami_payload(instance_id: Optional[str], root_dir: Optional[str],
                   mode: Optional[str]) -> dict:
    """The public identity a discovering client reads to confirm this really is a
    localm instance (and which one). Deliberately omits the token and pid."""
    return {
        "app": APP_NAME,
        "version": _version(),
        "instance_id": instance_id,
        "root_dir": root_dir,
        "mode": mode,
    }


# ------------------------------------------------------------------ #
#  Registry read/write                                               #
# ------------------------------------------------------------------ #

def _lock_down_dir(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError:
        pass


def register_instance(home: Path, *, instance_id: str, port: int, host: str,
                      root_dir: str, mode: str, token: str,
                      started: Optional[str] = None) -> Path:
    """Write this process's registry entry atomically; return its path. The file
    is owner-only where the OS supports it (it carries the attach token)."""
    d = run_dir(home)
    _lock_down_dir(d)
    entry = {
        "instance_id": instance_id,
        "pid": os.getpid(),
        "port": port,
        "host": host,
        "root_dir": root_dir,
        "mode": mode,
        "version": _version(),
        "started": started or _now_iso(),
        "token": token,
    }
    path = registry_path(home, instance_id)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)   # atomic on Windows + POSIX
    return path


def unregister_instance(path) -> None:
    """Remove a registry entry - call only with the path THIS process wrote
    (ownership-tracked cleanup; never delete another instance's file)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def read_entry(path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def list_entries(home: Path) -> list[dict]:
    """All readable registry entries (each gains a ``_path`` key). Corrupt files
    are skipped, not raised on."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.json")):
        entry = read_entry(f)
        if entry is not None:
            entry["_path"] = str(f)
            out.append(entry)
    return out


# ------------------------------------------------------------------ #
#  Liveness + reaping                                                #
# ------------------------------------------------------------------ #

def pid_alive(pid: int) -> bool:
    """Best-effort: is *pid* a live process? Conservative - when we genuinely
    cannot tell (e.g. Windows without psutil), return True so reaping never
    deletes a live instance's entry. PID reuse is handled by phase 4's /whoami
    handshake, not here."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception:
            return True   # cannot determine -> assume alive (do not reap)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True       # exists, owned by someone else
    except OSError:
        return True


def reap_stale(home: Path, *, self_id: Optional[str] = None,
               is_alive: Optional[Callable[[dict], bool]] = None) -> list[str]:
    """Remove registry entries for processes that are gone (or whose file is
    corrupt). Never reaps *self_id* or a live entry. Returns the instance_ids
    removed. *is_alive* lets phase 4 swap in a /whoami handshake; the default is
    process liveness."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    alive = is_alive or (lambda e: pid_alive(int(e.get("pid", -1))))
    removed: list[str] = []
    for f in sorted(d.glob("*.json")):
        entry = read_entry(f)
        if entry is None:
            # Corrupt/unreadable -> remove (no owner can be established).
            try:
                f.unlink()
                removed.append(f.stem)
            except OSError:
                pass
            continue
        if entry.get("instance_id") == self_id:
            continue
        try:
            living = alive(entry)
        except Exception:
            living = True   # probe error -> keep (never reap on uncertainty)
        if not living:
            try:
                f.unlink()
                removed.append(entry.get("instance_id", f.stem))
            except OSError:
                pass
    return removed


# ------------------------------------------------------------------ #
#  Advertise: the surface-startup context manager                    #
# ------------------------------------------------------------------ #

@contextmanager
def advertise(app, home: Path, *, host: str, port: int, mode: str,
              project: Optional[str] = None):
    """Advertise *app* as a running instance for the duration of the server.

    Resolves the instance id + root dir + token, sets them on ``app.state`` (so
    ``GET /whoami`` can report them), reaps stale entries, writes this instance's
    registry file on enter, and removes ONLY that file on exit. Phase 3:
    advertise only - no attach/spawn decision is made here.
    """
    instance_id = new_instance_id()
    root_dir = resolve_root_dir(override=project)
    token = new_token()

    app.state.instance_id = instance_id
    app.state.root_dir = root_dir
    app.state.instance_mode = mode
    app.state.instance_token = token

    reap_stale(home, self_id=instance_id)
    path = register_instance(
        home, instance_id=instance_id, port=port, host=host,
        root_dir=root_dir, mode=mode, token=token,
    )
    try:
        yield {"instance_id": instance_id, "root_dir": root_dir,
               "port": port, "token": token, "path": str(path)}
    finally:
        unregister_instance(path)
