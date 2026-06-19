# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry (H6 server-rework, phases 3-4).

Each running localm server advertises itself (phase 3) so a future launch can
DISCOVER and attach to it (phase 4) instead of spinning a second server that
double-loads the model - the "why is the coder an extra CLI / why a second
server" smell. ``find_attachable`` returns the live, identity-verified instance
serving a project dir (or None to spawn); ``advertise`` writes/reaps the registry
and serves ``GET /whoami``.

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
                      scheme: str = "http",
                      started: Optional[str] = None) -> Path:
    """Write this process's registry entry atomically; return its path. The file
    is owner-only where the OS supports it (it carries the attach token).
    ``scheme`` (http|https) records whether this instance serves TLS, so a
    discovering client probes /whoami on the right protocol."""
    d = run_dir(home)
    _lock_down_dir(d)
    entry = {
        "instance_id": instance_id,
        "pid": os.getpid(),
        "port": port,
        "host": host,
        "scheme": scheme,
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
#  Discovery: attach-or-spawn (H6 phase 4)                           #
# ------------------------------------------------------------------ #

def _canon(p: str) -> str:
    """Normalize a path for comparison (resolve + OS-case-fold). Windows paths are
    case-insensitive, so root_dir matching must be too."""
    try:
        return os.path.normcase(str(Path(p).resolve()))
    except (OSError, ValueError):
        return os.path.normcase(str(p or ""))


def _try_whoami(scheme: str, port: int, instance_id: Optional[str],
                timeout: float) -> bool:
    """One handshake: GET <scheme>://127.0.0.1:<port>/whoami and confirm it is
    THIS localm instance (app==localm AND the advertised instance_id matches the
    registry file - the impostor guard). Always probes loopback: discovery is
    same-machine, even for a network-bound instance."""
    import requests
    url = f"{scheme}://127.0.0.1:{port}/whoami"
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except Exception:
        verify = False
    try:
        r = requests.get(url, timeout=timeout, verify=verify)
    except requests.RequestException:
        return False
    if r.status_code != 200:
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    return (data.get("app") == APP_NAME
            and data.get("instance_id") == instance_id)


def default_probe(entry: dict, *, timeout: float = 0.7) -> bool:
    """Liveness + identity for a registry entry. Uses the recorded scheme; an
    entry written before scheme existed is probed http-then-https."""
    port = entry.get("port")
    if not port:
        return False
    iid = entry.get("instance_id")
    scheme = entry.get("scheme")
    order = [scheme] if scheme in ("http", "https") else ["http", "https"]
    return any(_try_whoami(s, int(port), iid, timeout) for s in order)


def find_attachable(home: Path, root_dir: str,
                    probe: Optional[Callable[[dict], bool]] = None) -> Optional[dict]:
    """The live, identity-verified instance serving *root_dir*, or None.

    Attaches ONLY on a verified /whoami handshake (app==localm + matching
    instance_id), so an impostor on a reused port is never attached to. A same-dir
    entry is reaped here only when it is *confident-dead* - the handshake fails AND
    the process is gone - so a transient probe miss (a slow /whoami) can never
    remove a live instance's entry and trigger a duplicate server. Other dirs'
    entries are left to their own launch / the PID reaper.
    """
    probe = probe or default_probe
    target = _canon(root_dir)
    for entry in list_entries(home):
        if _canon(entry.get("root_dir", "")) != target:
            continue
        try:
            alive = probe(entry)
        except Exception:
            alive = False
        if alive:
            return entry
        # Not verified live. Reap only if the process is also gone; never delete a
        # live entry on a transient/identity probe miss (an impostor on a live PID
        # simply lingers, harmless - it is never attached to).
        try:
            pid_gone = not pid_alive(int(entry.get("pid", -1) or -1))
        except (TypeError, ValueError):
            pid_gone = True
        if pid_gone:
            unregister_instance(entry.get("_path"))
    return None


def attach_url(entry: dict) -> str:
    """The browser URL for an attachable instance (loopback - same machine)."""
    scheme = entry.get("scheme") or "http"
    return f"{scheme}://127.0.0.1:{entry.get('port')}/"


# ------------------------------------------------------------------ #
#  Advertise: the surface-startup context manager                    #
# ------------------------------------------------------------------ #

@contextmanager
def advertise(app, home: Path, *, host: str, port: int, mode: str,
              scheme: str = "http", project: Optional[str] = None,
              isolated: bool = False):
    """Advertise *app* as a running instance for the duration of the server.

    Resolves the instance id + root dir + token and sets them on ``app.state``
    (so ``GET /whoami`` reports them). Unless *isolated*, it reaps stale entries,
    writes this instance's registry file on enter, and removes ONLY that file on
    exit. ``--isolated`` (test safety) sets *isolated* so the server is invisible
    to discovery: nothing attaches to it and it is never attached to.
    """
    instance_id = new_instance_id()
    root_dir = resolve_root_dir(override=project)
    token = new_token()

    app.state.instance_id = instance_id
    app.state.root_dir = root_dir
    app.state.instance_mode = mode
    app.state.instance_token = token

    path = None
    if not isolated:
        reap_stale(home, self_id=instance_id)
        path = register_instance(
            home, instance_id=instance_id, port=port, host=host,
            root_dir=root_dir, mode=mode, token=token, scheme=scheme,
        )
    try:
        yield {"instance_id": instance_id, "root_dir": root_dir, "port": port,
               "token": token, "path": str(path) if path else None}
    finally:
        if path is not None:
            unregister_instance(path)
