# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry (H6 server-rework, phases 3-4)."""

from __future__ import annotations

import json
import os
import secrets
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from localm.debuglog import logger

APP_NAME = "localm"

# Markers that identify a project root, walked up from cwd (decision 2). Mirrors
# the coder's existing .localcoder model; .git covers most repos.
_ROOT_MARKERS = (".git", ".localcoder")


def _version() -> str:
    """The running version, advertised in the registry and served by /whoami."""
    try:
        from localm._version import read_version
        v = read_version()
        if v and v != "unknown":
            return v
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("localm")
    except Exception:
        # Both the live VERSION file and dist-info are unreadable. Return the
        # honest "unknown" sentinel (already used by read_version above), NOT a
        # hardcoded version literal: /whoami's version gates the post-update health
        # watchdog by EQUALITY (LM-DA-011), so a stale literal that ever equalled
        # the expected version would false-PASS a build whose version machinery is
        # actually broken. "unknown" can never equal a real target -> fails safe
        # (the watchdog rolls back).
        logger.debug("instances: version undetermined; reporting 'unknown'",
                     exc_info=True)
        return "unknown"


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
    """The project root that keys this instance (decision 2): the nearest ancestor of *start* (default cwd) containing a ``.git`` or ``.localcoder`` marker, else *start* itself. *override* (a future ``--project`` flag) wins outright."""
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
    """The public identity a discovering client reads to confirm this really is a localm instance (and which one)."""
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
    """Write this process's registry entry atomically; return its path."""
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
    # The entry carries the per-instance attach token (LM-DA-044/LM-DA-027):
    # restrict it the same Windows-aware way as auth.key, so the token is not
    # readable by another local account on Windows the way a bare chmod would
    # leave it. atomic_write_private restricts the TEMP file (os.replace carries
    # its ACL onto the destination) and retries the destination only if that
    # first attempt failed, and it CREATES the temp at 0600 via os.open rather
    # than write_text, closing the residual POSIX window where the token existed
    # at the umask default between the create and the chmod.
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(entry, indent=2))
    return path


def unregister_instance(path) -> None:
    """Remove a registry entry - call only with the path THIS process wrote (ownership-tracked cleanup; never delete another instance's file)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def set_mode(home: Path, instance_id: str, mode: str) -> bool:
    """Update the SURFACE mode of this instance's registry entry in place (e.g. ``api`` -> ``full`` after an on-demand GUI mount, phase 5), atomically."""
    if not instance_id:
        return False
    path = registry_path(home, instance_id)
    entry = read_entry(path)
    if entry is None:
        return False
    entry.pop("_path", None)
    entry["mode"] = mode
    try:
        # Same token-bearing file as register_instance above; same treatment,
        # including the create-at-0600 that closes the POSIX window.
        from localm.config import atomic_write_private
        atomic_write_private(path, json.dumps(entry, indent=2))
        return True
    except OSError:
        return False


def _describe_unreadable(path) -> str:
    """Best-effort (size, mtime) of *path* for the read_entry warning below - never raises (this runs inside the except branch of an already-failed read, so it must not itself fail), and 'stat also failed' is a real, reportable outcome rather than an exception escaping the log call."""
    try:
        st = Path(path).stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return f"size={st.st_size} mtime={mtime}"
    except OSError as e:
        return f"stat also failed: {e}"


def read_entry(path) -> Optional[dict]:
    """Read one registry entry, or None."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None  # missing entry is the normal case; do not log
    except (json.JSONDecodeError, OSError) as e:
        # Surface (not silence) a corrupt/permission-denied entry before it gets
        # treated as missing and potentially reaped while the process is live.
        logger.warning("registry entry %s unreadable: %s (%s)",
                       path, e, _describe_unreadable(path))
        return None


def list_entries(home: Path) -> list[dict]:
    """All readable registry entries (each gains a ``_path`` key)."""
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
    """Best-effort: is *pid* a live process? Conservative - when we genuinely cannot tell (e.g. Windows without psutil), return True so reaping never deletes a live instance's entry."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception as e:
            # Conservative: keep returning True (a broken psutil must never reap a
            # live instance), but log so --debug users can see WHY stale entries
            # are persisting rather than silently disabling reaping.
            logger.debug("psutil check failed (%s); assuming pid alive - reaping disabled", e)
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


def kill_pid(pid: int, *, timeout: float = 10.0) -> bool:
    """Direct-process fallback for ``localm stop`` when a graceful HTTP shutdown (POST /v1/server/shutdown) could not be confirmed - the server did not answer, took too long, or is an older build without that route."""
    if not isinstance(pid, int) or pid <= 0 or not pid_alive(pid):
        return True
    try:
        import psutil
        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                pass
    except psutil.NoSuchProcess:
        pass
    except Exception as e:
        logger.debug("kill_pid(%s) failed: %s", pid, e)
    return not pid_alive(pid)


def reap_stale(home: Path, *, self_id: Optional[str] = None,
               is_alive: Optional[Callable[[dict], bool]] = None) -> list[str]:
    """Remove registry entries for processes that are gone (or whose file is corrupt)."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    # `or -1` so a null/empty pid (a malformed entry) reads as a dead process and
    # is reaped, consistent with find_attachable(); without it int(None) raises a
    # TypeError that the probe-guard below swallows, KEEPING the corrupt entry
    # forever (it would never be cleaned and never be attachable).
    alive = is_alive or (lambda e: pid_alive(int(e.get("pid", -1) or -1)))
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
    """Normalize a path for comparison (resolve + OS-case-fold)."""
    try:
        return os.path.normcase(str(Path(p).resolve()))
    except (OSError, ValueError):
        return os.path.normcase(str(p or ""))


def _try_whoami(scheme: str, port: int, instance_id: Optional[str],
                timeout: float, bind_host: Optional[str] = None) -> bool:
    """One handshake: GET <scheme>://<loopback>:<port>/whoami and confirm it is THIS localm instance (app==localm AND the advertised instance_id matches the registry file - the impostor guard)."""
    import requests
    from localm.bindhost import self_connect_host, url_host
    host = url_host(self_connect_host(bind_host))
    url = f"{scheme}://{host}:{port}/whoami"
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        # CA file genuinely absent: verify=False is the intended best-effort path
        # on 127.0.0.1 (loopback, same machine), so keep it.
        verify = False
    except Exception as e:
        # Any OTHER error determining verification is unexpected: do NOT silently
        # downgrade to unverified HTTPS - treat the instance as unverifiable.
        logger.debug("could not determine TLS verification for %s: %s; skipping", url, e)
        return False
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
    """Liveness + identity for a registry entry."""
    port = entry.get("port")
    if not port:
        return False
    iid = entry.get("instance_id")
    scheme = entry.get("scheme")
    order = [scheme] if scheme in ("http", "https") else ["http", "https"]
    return any(_try_whoami(s, int(port), iid, timeout, entry.get("host"))
               for s in order)


def find_attachable(home: Path, root_dir: str,
                    probe: Optional[Callable[[dict], bool]] = None) -> Optional[dict]:
    """The live, identity-verified instance serving *root_dir*, or None."""
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
    from localm.bindhost import self_connect_host, url_host
    scheme = entry.get("scheme") or "http"
    host = url_host(self_connect_host(entry.get("host")))
    return f"{scheme}://{host}:{entry.get('port')}/"


def attach_target(home: Path, root_dir: str,
                  probe: Optional[Callable[[dict], bool]] = None) -> Optional[dict]:
    """The thin-client attach point for the instance serving *root_dir*, or None."""
    entry = find_attachable(home, root_dir, probe=probe)
    if not entry:
        return None
    from localm.bindhost import self_connect_host, url_host
    scheme = entry.get("scheme") or "http"
    _h = url_host(self_connect_host(entry.get("host")))
    return {
        "base_url": f"{scheme}://{_h}:{entry.get('port')}/v1",
        "token": entry.get("token"),
        "port": entry.get("port"),
        "scheme": scheme,
        "mode": entry.get("mode"),
    }


def snapshot(home: Path, probe: Optional[Callable[[dict], bool]] = None,
             reap: bool = True, *, include_token: bool = False) -> list[dict]:
    """Registry entries for ``localm ps``: each gains an ``alive`` flag from an identity-verified /whoami probe, with the attach ``token`` stripped (a listing must never surface a credential)."""
    probe = probe or default_probe
    if reap:
        reap_stale(home)
    rows: list[dict] = []
    for e in list_entries(home):
        try:
            alive = probe(e)
        except Exception:
            alive = False
        row = dict(e) if include_token else {k: v for k, v in e.items() if k != "token"}
        row["alive"] = alive
        rows.append(row)
    rows.sort(key=lambda r: r.get("started") or "")
    return rows


# ------------------------------------------------------------------ #
#  Advertise: the surface-startup context manager                    #
# ------------------------------------------------------------------ #

@contextmanager
def advertise(app, home: Path, *, host: str, port: int, mode: str,
              scheme: str = "http", project: Optional[str] = None,
              isolated: bool = False):
    """Advertise *app* as a running instance for the duration of the server."""
    instance_id = new_instance_id()
    root_dir = resolve_root_dir(override=project)
    token = new_token()

    app.state.instance_id = instance_id
    app.state.root_dir = root_dir
    app.state.instance_mode = mode
    app.state.instance_token = token
    # The instance's own bind coordinates, so it can build its loopback /v1
    # self-url when it mounts a surface on demand (phase 5 on-demand GUI mount).
    app.state.instance_port = port
    app.state.instance_scheme = scheme
    # Exposed so OTHER discovery-shaped features (e.g. the cross-install GPU/
    # VRAM coordination registry) can honour --isolated too: instance_id above
    # is set even when isolated (so /whoami still answers), but isolated means
    # "invisible to discovery" - a test/throwaway instance must not register
    # itself anywhere discoverable, in ANY registry, not just this one.
    app.state.instance_isolated = isolated

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
