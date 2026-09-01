# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry.

Each running localm server advertises itself so a later launch can DISCOVER and
attach to it instead of spinning a second server that double-loads the model.
``find_attachable`` returns the live, identity-verified instance serving a
project dir (or None to spawn); ``advertise`` writes/reaps the registry and
serves ``GET /whoami``.

Registry: one JSON file per instance under ``<LOCALM_HOME>/run/<instance_id>.json``::

    {instance_id, pid, port, host, root_dir, mode, version, started, token}

Liveness is ultimately a ``GET /whoami`` returning ``app == "localm"``; the
reaper here falls back to process liveness and ownership. ``mode`` is the
SURFACE mode (``api`` = bare OpenAI API, ``full`` = API + GUI), distinct from the
session persistence mode (privacy|log|full), which happens to share the word
"full". The per-instance ``token`` lets a local client authenticate when it
attaches; it stays in the 0600 registry file and is NEVER served by
``/whoami``.
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

from localm.debuglog import logger

APP_NAME = "localm"

# Markers that identify a project root, walked up from cwd. Mirrors the coder's
# .localcoder model; .git covers most repos.
_ROOT_MARKERS = (".git", ".localcoder")


def _version() -> str:
    """The running version, advertised in the registry and served by /whoami.

    Prefers the LIVE VERSION file (localm/_version.py:read_version()) over
    installed dist-info: the install is editable and .venv is never touched by a
    file-swap update (_apply_update.py's NEVER_TOUCH), so a reboot-class update
    changes VERSION on disk but does not refresh dist-info. /whoami's version is
    what the post-update health watchdog compares against the just-applied
    version."""
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
        # Both the live VERSION file and dist-info are unreadable, so report the
        # "unknown" sentinel rather than a literal: /whoami's version gates the
        # post-update health watchdog by equality, and "unknown" never matches.
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
    """A per-instance attach token, used by a local client when it attaches."""
    return secrets.token_urlsafe(32)


def resolve_root_dir(start: Optional[str] = None, override: Optional[str] = None) -> str:
    """The project root that keys this instance: the nearest ancestor of *start*
    (default cwd) containing a ``.git`` or ``.localcoder`` marker, else *start*
    itself. *override* wins outright."""
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
    localm instance (and which one). Omits the token and the pid."""
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
    # The entry carries the per-instance attach token, so it is restricted the
    # same Windows-aware way as auth.key. atomic_write_private restricts the temp
    # file before the rename and creates it at 0600.
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(entry, indent=2))
    return path


def unregister_instance(path) -> None:
    """Remove a registry entry - call only with the path THIS process wrote
    (ownership-tracked cleanup; never delete another instance's file)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def set_mode(home: Path, instance_id: str, mode: str) -> bool:
    """Update the SURFACE mode of this instance's registry entry in place
    (e.g. ``api`` -> ``full`` after an on-demand GUI mount), atomically.

    Best-effort: returns True if the entry was rewritten, False if it is missing
    or unwritable. Only the owning process should call this for its own id (it
    rewrites the same 0600 file ``advertise`` created)."""
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
    """Best-effort (size, mtime) of *path* for the read_entry warning below -
    never raises (this runs inside the except branch of an already-failed
    read, so it must not itself fail), and "stat also failed" is a real,
    reportable outcome rather than an exception escaping the log call."""
    try:
        st = Path(path).stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        return f"size={st.st_size} mtime={mtime}"
    except OSError as e:
        return f"stat also failed: {e}"


def read_entry(path) -> Optional[dict]:
    """Read one registry entry, or None. A MISSING file (FileNotFoundError) is
    normal and returns None silently; a CORRUPT or unreadable file
    (JSONDecodeError/OSError) ALSO collapses to None but is logged first, so a
    reap that deletes a live-but-unreadable entry is at least discoverable under
    --debug instead of looking identical to "no such file".

    The size/mtime in that log line separate a genuinely EMPTY entry (size=0)
    from a TRUNCATED write (size>0), which the read alone cannot distinguish.
    They cost nothing on the normal missing-file path, which returns before
    reaching them."""
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
    deletes a live instance's entry. PID reuse is handled by the /whoami
    handshake, not here."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception as e:
            # Conservative: a broken psutil must never reap a live instance, so
            # this keeps returning True and logs the failure at debug.
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
    """Direct-process fallback for ``localm stop`` when a graceful HTTP shutdown
    (POST /v1/server/shutdown) could not be confirmed - the server did not
    answer, took too long, or is an older build without that route. Sends a
    graceful terminate first (SIGTERM on POSIX; psutil maps this to
    TerminateProcess on Windows, where there is no finer-grained graceful
    signal for an unrelated process), force-kills if it has not exited within
    *timeout*. Returns whether the pid is confirmed gone afterward; never
    raises."""
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
    """Remove registry entries for processes that are gone (or whose file is
    corrupt). Never reaps *self_id* or a live entry. Returns the instance_ids
    removed. *is_alive* lets a caller swap in a /whoami handshake; the default is
    process liveness."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    # `or -1` so a null or empty pid reads as a dead process and is reaped;
    # without it int(None) raises and the corrupt entry is kept forever.
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
#  Discovery: attach-or-spawn                                        #
# ------------------------------------------------------------------ #

def _canon(p: str) -> str:
    """Normalize a path for comparison (resolve + OS-case-fold). Windows paths are
    case-insensitive, so root_dir matching must be too."""
    try:
        return os.path.normcase(str(Path(p).resolve()))
    except (OSError, ValueError):
        return os.path.normcase(str(p or ""))


def fetch_whoami(scheme: str, port: int, instance_id: Optional[str],
                 timeout: float, bind_host: Optional[str] = None) -> Optional[dict]:
    """One handshake: GET <scheme>://<loopback>:<port>/whoami, returning the
    verified identity payload, or None if it is not THIS localm instance
    (app==localm AND the advertised instance_id matches the registry file - the
    impostor guard). Always probes loopback: discovery is same-machine, even for
    a network-bound instance.

    WHICH loopback comes from the entry's recorded *bind_host*. "Loopback" is
    two addresses, not one: a server bound on ``::1`` has nothing listening on
    127.0.0.1, so hardcoding the IPv4 one reports a healthy IPv6-bound server as
    DEAD. Omitted, it still dials 127.0.0.1, which is what an entry with no
    recorded bind host wants.

    The payload carries ``version``, ``root_dir`` and ``mode``; a network-bound
    instance omits ``root_dir``."""
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
        return None
    try:
        r = requests.get(url, timeout=timeout, verify=verify)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("app") != APP_NAME or data.get("instance_id") != instance_id:
        return None
    return data


def _try_whoami(scheme: str, port: int, instance_id: Optional[str],
                timeout: float, bind_host: Optional[str] = None) -> bool:
    """Whether ``fetch_whoami`` verifies this entry."""
    return fetch_whoami(scheme, port, instance_id, timeout, bind_host) is not None


def default_probe(entry: dict, *, timeout: float = 0.7) -> bool:
    """Liveness + identity for a registry entry. Uses the recorded scheme; an
    entry written before scheme existed is probed http-then-https."""
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
        # Reap only when the process is also gone; never delete a live entry on a
        # transient or identity probe miss.
        try:
            pid_gone = not pid_alive(int(entry.get("pid", -1) or -1))
        except (TypeError, ValueError):
            pid_gone = True
        if pid_gone:
            unregister_instance(entry.get("_path"))
    return None


def attach_url(entry: dict) -> str:
    """The browser URL for an attachable instance (loopback - same machine).

    The loopback address is derived from the entry's recorded bind host, so an
    IPv6-bound instance gets ``http://[::1]:PORT/`` rather than an IPv4 loopback
    it is not listening on, correctly bracketed for a URL authority."""
    from localm.bindhost import self_connect_host, url_host
    scheme = entry.get("scheme") or "http"
    host = url_host(self_connect_host(entry.get("host")))
    return f"{scheme}://{host}:{entry.get('port')}/"


def attach_target(home: Path, root_dir: str,
                  probe: Optional[Callable[[dict], bool]] = None) -> Optional[dict]:
    """The thin-client attach point for the instance serving *root_dir*, or None.

    A CLI chat / coder client uses this to talk to the running instance's ``/v1``
    (with its registry ``token``) instead of starting its own server + model.
    Returns ``{base_url, token, port, scheme, mode}`` (loopback - same
    machine)."""
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
    """Registry entries for ``localm ps``: each gains an ``alive`` flag from an
    identity-verified /whoami probe, with the attach ``token`` stripped (a
    listing must never surface a credential). Process-gone entries are reaped
    first unless *reap* is False. Sorted by start time (oldest first).

    ``include_token``: keep the attach token in each row instead of stripping
    it. Default False (every DISPLAY caller, e.g. ``localm ps``). Set True only
    for an INTERNAL caller that needs the token to authenticate its OWN request
    to that instance (the MCP ``server_activity`` tool, which asks each
    discovered instance over HTTP) - never for anything a human reads."""
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


def list_machine_peers(home: Path, *, timeout: float = 0.7) -> list[dict]:
    """Live localm instances belonging to a DIFFERENT install, in ``snapshot``
    row shape, each carrying ``same_install: False``.

    ``snapshot`` reads ``<home>/run`` and so sees only instances sharing this
    install's data dir. This reads the machine-wide coordination registry
    (:mod:`localm.gpu_registry`), which every non-isolated server writes
    whatever its LOCALM_HOME, and returns the entries that are NOT in *home*'s
    own registry.

    A peer is listed only on a verified ``GET /whoami`` handshake, so a stale
    entry whose port got reused by an unrelated process is never listed;
    ``root_dir``, ``mode`` and ``version`` come from that handshake rather than
    from the coordination entry, which does not record them. A network-bound
    peer omits ``root_dir``. ``started`` is None: the coordination entry records
    a heartbeat time, not a start time.

    Returns no credential. The coordination entry's ``coordination_token`` is
    not an attach token and is dropped here, so a caller cannot use these rows
    to authenticate to another install.

    ``snapshot`` is deliberately NOT widened to cover these: callers that decide
    whether a model file is held elsewhere (``selfclient.read_model_file_hold``)
    treat a registry miss as a rule-out, which is only sound while every server
    they reach shares one registry.

    Best-effort: any failure yields the peers found so far rather than raising.
    """
    try:
        from localm import gpu_registry
        entries = gpu_registry.list_entries(gpu_registry.registry_dir())
    except Exception as e:
        logger.debug("cross-install peer lookup unavailable: %s", e)
        return []

    own = {str(e.get("instance_id")) for e in list_entries(home)}
    self_pid = os.getpid()
    rows: list[dict] = []
    for entry in entries:
        iid = entry.get("instance_id")
        port = entry.get("port")
        if not iid or not port or str(iid) in own:
            continue
        try:
            pid = int(entry.get("pid", -1) or -1)
        except (TypeError, ValueError):
            continue
        if pid == self_pid or not pid_alive(pid):
            continue
        scheme = entry.get("scheme") or "http"
        host = entry.get("host")
        try:
            ident = fetch_whoami(scheme, int(port), iid, timeout, host)
        except Exception as e:
            logger.debug("whoami probe failed for cross-install peer %s: %s", iid, e)
            continue
        if ident is None:
            continue
        rows.append({
            "instance_id": iid,
            "pid": pid,
            "port": port,
            "host": host,
            "scheme": scheme,
            "root_dir": ident.get("root_dir"),
            "mode": ident.get("mode"),
            "version": ident.get("version"),
            "started": None,
            "alive": True,
            "same_install": False,
        })
    rows.sort(key=lambda r: str(r.get("instance_id") or ""))
    return rows


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
    # The instance's own bind coordinates, so it can build its loopback /v1
    # self-url when it mounts a surface on demand.
    app.state.instance_port = port
    app.state.instance_scheme = scheme
    # Exposed so other discovery-shaped features can honour --isolated too.
    # instance_id is still set when isolated, so /whoami answers, but the
    # instance registers itself in no registry.
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
