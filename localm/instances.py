# SPDX-License-Identifier: AGPL-3.0-or-later
"""Instance discovery registry.

Each running localm server advertises itself so a later launch can DISCOVER and
attach to it instead of spinning a second server that double-loads the model.
``find_attachable`` returns the live, identity-verified instance serving a
project dir (or None to spawn); ``advertise`` writes/reaps the registry and
serves ``GET /whoami``.

Registry: one JSON file per instance under ``<LOCALM_HOME>/run/<instance_id>.json``::

    {instance_id, pid, port, host, root_dir, mode, version, started, token}

plus ``launch_tag`` when the process was started with ``LOCALM_LAUNCH_TAG`` set
(see :data:`LAUNCH_TAG_ENV`); readers treat a missing key as "no tag".

Liveness is ultimately a ``GET /whoami`` returning ``app == "localm"``; the
reaper here falls back to process liveness and ownership. ``mode`` is the
SURFACE mode (``api`` = bare OpenAI API, ``full`` = API + GUI), distinct from the
session persistence mode (privacy|log|full), which happens to share the word
"full". The per-instance ``token`` lets a local client authenticate when it
attaches; it stays in the 0600 registry file and is NEVER served by
``/whoami``.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import secrets
import sys
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import Callable, Collection, Optional

from localm.debuglog import logger
from localm.procid import (  # noqa: F401  (re-exported: callers use instances.<name>)
    darwin_host_uuid,
    host_id,
    legacy_pid_space_id,
    legacy_record_is_this_host,
    linux_machine_id,
    machine_guid,
    pid_alive,
    pid_liveness,
    pid_space_id,
    process_start_identity,
    start_identity_differs,
    start_identity_matches,
    thread_start_identity,
)

APP_NAME = "localm"

# Markers that identify a project root, walked up from cwd. Mirrors the coder's
# .localcoder model; .git covers most repos.
_ROOT_MARKERS = (".git", ".localcoder")

#: Environment variable a launcher sets on the server it spawns. ``advertise``
#: removes it from the environment and records its value as the entry's
#: ``launch_tag``, so the launcher can find exactly the server it started.
LAUNCH_TAG_ENV = "LOCALM_LAUNCH_TAG"
_LAUNCH_TAG_RE = re.compile(r"[0-9a-f]{16,64}")

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
    return datetime.now(UTC).isoformat()


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


def new_launch_tag() -> str:
    """A random tag for :data:`LAUNCH_TAG_ENV`."""
    return secrets.token_hex(16)


def take_launch_tag() -> Optional[str]:
    """Remove :data:`LAUNCH_TAG_ENV` from ``os.environ`` and return its value,
    or None when it is unset or not 16 to 64 lowercase hex characters (a
    malformed value is logged at debug and dropped). Child processes started
    after this call do not inherit it."""
    raw = os.environ.pop(LAUNCH_TAG_ENV, None)
    if raw is None:
        return None
    if _LAUNCH_TAG_RE.fullmatch(raw):
        return raw
    logger.debug("ignoring malformed %s (%d chars)", LAUNCH_TAG_ENV, len(raw))
    return None


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

def whoami_payload(instance_id: Optional[str], mode: Optional[str]) -> dict:
    """The public identity a discovering client reads to confirm this really is a
    localm instance (and which one). Omits the token, the pid and the project
    directory."""
    return {
        "app": APP_NAME,
        "version": _version(),
        "instance_id": instance_id,
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
                      started: Optional[str] = None,
                      launch_tag: Optional[str] = None) -> Path:
    """Write this process's registry entry atomically; return its path. The file
    is owner-only where the OS supports it (it carries the attach token).
    ``scheme`` (http|https) records whether this instance serves TLS, so a
    discovering client probes /whoami on the right protocol.

    The entry also carries ``pid_space`` (:func:`pid_space_id`) and
    ``start_identity`` (:func:`process_start_identity` of this process, or
    None when it cannot be read), which :func:`entry_process_state` compares
    with the live process before anything ends that pid.

    ``launch_tag``, when given, is written as the entry's ``launch_tag`` key;
    when None the key is omitted."""
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
        "pid_space": pid_space_id(),
        "start_identity": process_start_identity(os.getpid()),
    }
    if launch_tag:
        entry["launch_tag"] = launch_tag
    path = registry_path(home, instance_id)
    # The entry carries the per-instance attach token, so it is restricted the
    # same Windows-aware way as auth.key. atomic_write_private restricts the temp
    # file before the rename and creates it at 0600.
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(entry, indent=2))
    return path


def hold_marker_path(home: Path, instance_id: str) -> Path:
    return run_dir(home) / f"{instance_id}.hold"


def register_hold_marker(home: Path, *, instance_id: str, port: int,
                         host: str, token: str, scheme: str = "http") -> Path:
    """Write the hold-check-only marker of an ``--isolated`` server. It lets a
    CLI sharing this home ask the server which model files it holds, and is
    invisible to discovery: it is not a ``.json`` registry entry, so
    ``list_entries``, ``snapshot`` and attach never see it."""
    d = run_dir(home)
    _lock_down_dir(d)
    entry = {
        "instance_id": instance_id,
        "pid": os.getpid(),
        "port": port,
        "host": host,
        "scheme": scheme,
        "token": token,
    }
    path = hold_marker_path(home, instance_id)
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(entry, indent=2))
    return path


def list_hold_markers(home: Path) -> list[dict]:
    """Readable hold markers of LIVE isolated servers (each gains ``alive``).
    A marker whose process is gone, or that is unreadable, is removed."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    out: list[dict] = []
    for f in sorted(d.glob("*.hold")):
        entry = read_entry(f)
        living = False
        if entry is not None:
            try:
                living = pid_alive(int(entry.get("pid", -1) or -1))
            except (TypeError, ValueError):
                living = False
        if not living or entry is None:
            unregister_instance(f)
            continue
        entry["alive"] = True
        out.append(entry)
    return out


def unregister_instance(path) -> None:
    """Remove a registry entry - call only with the path THIS process wrote
    (ownership-tracked cleanup; never delete another instance's file)."""
    try:
        Path(path).unlink()
    except OSError:
        pass


#: Registry files written by advertise() in this process, by instance id.
_OWN_PATHS: dict = {}


def unregister_own(instance_id: Optional[str]) -> Optional[dict]:
    """Remove the registry entry advertise() wrote in this process for
    *instance_id*. For a process that leaves without advertise()'s exit
    (``os._exit``, ``os.execv``). Returns the removed entry, for
    :func:`restore_own`, or None when this process wrote none for that id.
    Never touches a file another process wrote."""
    if not instance_id:
        return None
    path = _OWN_PATHS.pop(instance_id, None)
    if path is None:
        return None
    entry = read_entry(path)
    if entry is not None:
        entry["_path"] = str(path)
    unregister_instance(path)
    return entry


def restore_own(entry: Optional[dict]) -> None:
    """Write back, at the path it was removed from, an entry
    :func:`unregister_own` removed, for a process that is still serving after
    all. No-op for None."""
    if not entry or not entry.get("_path") or not entry.get("instance_id"):
        return
    path = Path(entry["_path"])
    body = {k: v for k, v in entry.items() if k != "_path"}
    from localm.config import atomic_write_private
    atomic_write_private(path, json.dumps(body, indent=2))
    _OWN_PATHS[entry["instance_id"]] = path


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
        mtime = datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat()
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
    except UnicodeDecodeError:
        raise
    except (ValueError, OSError, RecursionError) as e:
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

_LINUX_LOCAL_FSTYPES = frozenset({
    "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "jfs", "reiserfs", "zfs",
    "bcachefs", "tmpfs", "ramfs", "overlay", "vfat", "msdos", "exfat", "ntfs",
    "ntfs3", "fuseblk", "hfsplus",
})


def _unescape_mountinfo(field: str) -> str:
    """A mountinfo path field with its octal escapes (``\\040`` and the like)
    decoded."""
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def _longest_mount(path: str, mounts) -> tuple | None:
    """The entry of *mounts* (``(mount point, ...)`` tuples, POSIX paths)
    whose mount point is the longest one containing *path* (the last such
    entry when several share it), or None."""
    best, best_len = None, -1
    for entry in mounts:
        mp = entry[0].rstrip("/") or "/"
        if mp == "/" or path == mp or path.startswith(mp + "/"):
            if len(mp) >= best_len:
                best, best_len = entry, len(mp)
    return best


_LINUX_MOUNTINFO = Path("/proc/self/mountinfo")


def linux_fstype_of(path: str, mountinfo: str) -> str | None:
    """The filesystem type of the mount holding absolute POSIX *path*,
    according to the text of a Linux ``/proc/<pid>/mountinfo``, or None when
    no mount in it holds *path*."""
    mounts = []
    for line in mountinfo.splitlines():
        head, sep, tail = line.partition(" - ")
        fields, rest = head.split(), tail.split()
        if not sep or len(fields) < 5 or not rest:
            continue
        mounts.append((_unescape_mountinfo(fields[4]), rest[0]))
    best = _longest_mount(path, mounts)
    return None if best is None else best[1]


def path_is_local(path) -> bool:
    """True only when directory *path* is on a filesystem of a disk attached
    to this machine: a fixed, removable or RAM drive on Windows, a mount the
    kernel flags local on macOS, a mount of a known local filesystem type on
    Linux. False for a network or virtual-machine share, a FUSE filesystem
    that is not block-device backed, an unknown type, and whenever it cannot
    be determined."""
    try:
        real = os.path.realpath(os.fspath(path))
    except (OSError, ValueError, TypeError):
        return False
    if sys.platform == "win32":
        import ctypes
        try:
            k = ctypes.WinDLL("kernel32", use_last_error=True)
            buf = ctypes.create_unicode_buffer(1024)
            if not k.GetVolumePathNameW(real, buf, len(buf)):
                return False
            return k.GetDriveTypeW(buf.value) in (2, 3, 6)
        except (OSError, AttributeError, ValueError, ctypes.ArgumentError):
            return False
    if sys.platform.startswith("linux"):
        try:
            text = _LINUX_MOUNTINFO.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return linux_fstype_of(real, text) in _LINUX_LOCAL_FSTYPES
    if sys.platform == "darwin":
        try:
            import psutil
            parts = psutil.disk_partitions(all=True)
        except Exception:
            return False
        best = _longest_mount(real, [(p.mountpoint, p.opts) for p in parts])
        return best is not None and "local" in best[1].split(",")
    return False


def entry_process_state(entry: dict) -> tuple[str, str]:
    """Whether the process at a registry entry's pid is the server the entry
    describes. Returns ``(state, reason)``:

    - ``"gone"``: the pid is invalid or no process has it.
    - ``"same"``: the live process's start identity matches the recorded
      ``start_identity`` in the same pid space.
    - ``"stale"``: the live process's start identity differs from the
      recorded one, so the pid now belongs to another process.
    - ``"unknown"``: neither can be proven: an entry with no recorded
      identity or pid space (written by an earlier localm), one recorded in
      another pid space, or a live process whose start identity cannot be
      read or compared.

    *reason* is a sentence for the person for ``"unknown"`` and ``"stale"``,
    empty otherwise. Never raises.
    """
    try:
        pid = int(entry.get("pid", -1) or -1)
    except (TypeError, ValueError):
        return "gone", ""
    try:
        if pid <= 0 or not pid_alive(pid):
            return "gone", ""
    except Exception as e:
        logger.debug("entry_process_state: liveness of pid %s unreadable (%s)", pid, e)
        return "unknown", (f"localm could not tell whether pid {pid} is still "
                           f"running.")
    recorded = entry.get("start_identity")
    space = entry.get("pid_space")
    if not isinstance(space, str):
        return "unknown", (f"this entry was written by an earlier localm that did "
                           f"not record when its server started, so localm cannot "
                           f"prove that pid {pid} is still that server.")
    if not isinstance(recorded, dict):
        return "unknown", (f"its server could not read its own start time when it "
                           f"registered (on Windows and macOS this needs the psutil "
                           f"package), so localm cannot prove that pid {pid} is "
                           f"still that server.")
    if space != pid_space_id():
        return "unknown", (f"this entry was written on another machine or "
                           f"container sharing this data folder, so pid {pid} "
                           f"here may be an unrelated program.")
    try:
        current = process_start_identity(pid)
    except Exception as e:
        logger.debug("entry_process_state: start identity of pid %s unreadable (%s)",
                     pid, e)
        current = None
    if start_identity_differs(recorded, current):
        return "stale", (f"its server has already exited and pid {pid} now "
                         f"belongs to another program.")
    if start_identity_matches(recorded, current):
        return "same", ""
    return "unknown", (f"localm cannot read when pid {pid} started, so it cannot "
                       f"prove that pid {pid} is still that server.")


def kill_pid(pid: int, *, timeout: float = 10.0,
             start_identity: Optional[dict] = None) -> bool:
    """Direct-process fallback for ``localm stop`` when a graceful HTTP shutdown
    (POST /v1/server/shutdown) could not be confirmed - the server did not
    answer, took too long, or is an older build without that route. Sends a
    graceful terminate first (SIGTERM on POSIX; psutil maps this to
    TerminateProcess on Windows, where there is no finer-grained graceful
    signal for an unrelated process), force-kills if it has not exited within
    *timeout*. Returns whether the pid is confirmed gone afterward; never
    raises.

    With *start_identity*, the process is ended only when its start identity,
    read after it is opened, matches that one (:func:`start_identity_matches`);
    otherwise nothing is signalled and the result is whether the pid is gone."""
    if not isinstance(pid, int) or pid <= 0 or not pid_alive(pid):
        return True
    # Imported before the try below, not inside it: psutil is an optional extra,
    # and an ImportError inside that block leaves the name unbound while the
    # `except psutil.NoSuchProcess` clause still has to evaluate it, raising
    # UnboundLocalError out of a function documented to never raise. See
    # test_kill_pid_without_psutil_does_not_raise.
    try:
        import psutil
    except Exception as e:
        logger.debug("kill_pid(%s): psutil unavailable (%s)", pid, e)
        return not pid_alive(pid)
    try:
        proc = psutil.Process(pid)
        if start_identity is not None and not start_identity_matches(
                start_identity, process_start_identity(pid)):
            logger.warning("kill_pid(%s): process start identity does not match "
                           "the recorded one; not ending it", pid)
            return not pid_alive(pid)
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
    removed. *is_alive* lets a caller swap in a /whoami handshake; the default
    keeps an entry unless :func:`entry_process_state` reports ``"gone"`` or
    ``"stale"`` (a live pid whose start identity differs from the recorded
    one)."""
    d = run_dir(home)
    if not d.is_dir():
        return []
    alive = is_alive or (lambda e: entry_process_state(e)[0] not in ("gone", "stale"))
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
    """One handshake: GET <scheme>://<this machine>:<port>/whoami, returning
    the verified identity payload, or None if it is not THIS localm instance
    (app==localm AND the advertised instance_id matches the registry file - the
    impostor guard). Only ever dials an address this machine holds: discovery
    is same-machine, even for a network-bound instance.

    WHICH address comes from the entry's recorded *bind_host*, via
    ``self_connect_host``: a wildcard or ``localhost`` dials 127.0.0.1, ``::``
    dials ``::1``, and a specific literal dials itself (a server bound on
    ``::1`` or on one interface's address has nothing listening on 127.0.0.1).
    Omitted, it dials 127.0.0.1, which is what an entry with no recorded bind
    host wants.

    A *bind_host* that is not an address this machine holds
    (``bindhost.is_own_address``: a hostname, a non-string, or a literal no
    interface here carries) returns None without sending anything. See
    test_a_forged_host_is_never_dialed_and_never_verified. The request is a
    single hop: a redirect answer is not followed and, not being a 200, reads
    as not verified. See test_a_redirect_from_a_local_listener_is_not_followed.

    The payload carries ``version`` and ``mode``."""
    data = fetch_any_whoami(scheme, port, timeout, bind_host)
    if data is None or data.get("instance_id") != instance_id:
        return None
    return data


def fetch_any_whoami(scheme: str, port: int, timeout: float,
                     bind_host: Optional[str] = None) -> Optional[dict]:
    """:func:`fetch_whoami` without the instance-id match: the identity payload
    of whatever localm instance answers ``GET /whoami`` on this machine's
    *port* (``app == "localm"``), or None. The caller learns the instance id
    from the payload; same dialing rules and single-hop behaviour as
    :func:`fetch_whoami`."""
    import requests
    from localm.bindhost import is_own_address, self_connect_host, url_host
    if bind_host is not None and not isinstance(bind_host, str):
        logger.warning("whoami probe refused: registry entry's host %r is not a "
                       "string; not dialing it", bind_host)
        return None
    dial = self_connect_host(bind_host)
    if not is_own_address(dial):
        logger.warning("whoami probe refused: registry entry's host %r is not "
                       "an address this machine holds; not dialing it", bind_host)
        return None
    host = url_host(dial)
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
        r = requests.get(url, timeout=timeout, verify=verify,
                         allow_redirects=False)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("app") != APP_NAME:
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
                    probe: Optional[Callable[[dict], bool]] = None,
                    *, exclude_ids: Collection[str] = (),
                    launch_tag: Optional[str] = None) -> Optional[dict]:
    """The live, identity-verified instance serving *root_dir*, or None.

    Entries whose ``instance_id`` is in *exclude_ids* are skipped (neither
    probed nor reaped). When *launch_tag* is given, every entry whose
    ``launch_tag`` is not exactly that value is skipped the same way, including
    entries without the key.

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
        if exclude_ids and entry.get("instance_id") in exclude_ids:
            continue
        if launch_tag is not None and entry.get("launch_tag") != launch_tag:
            continue
        try:
            alive = probe(entry)
        except Exception:
            alive = False
        if alive:
            return entry
        # Reap only when the process is also gone or its pid now names another
        # process; never delete a live entry on a transient or identity probe miss.
        if entry_process_state(entry)[0] in ("gone", "stale"):
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
                  probe: Optional[Callable[[dict], bool]] = None,
                  *, exclude_ids: Collection[str] = (),
                  launch_tag: Optional[str] = None) -> Optional[dict]:
    """The thin-client attach point for the instance serving *root_dir*, or None.

    A CLI chat / coder client uses this to talk to the running instance's ``/v1``
    (with its registry ``token``) instead of starting its own server + model.
    Returns ``{base_url, token, port, scheme, mode}`` (loopback - same
    machine). *exclude_ids* and *launch_tag* as for find_attachable."""
    entry = find_attachable(home, root_dir, probe=probe, exclude_ids=exclude_ids,
                            launch_tag=launch_tag)
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


def _threaded_map(fn: Callable[[object], object], items: list) -> list:
    """Apply *fn* to each of *items* on a bounded thread pool, returning the
    results in the same order as *items*."""
    items = list(items)
    if not items:
        return []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(items), 8)) as ex:
        return list(ex.map(fn, items))


def snapshot(home: Path, probe: Optional[Callable[[dict], bool]] = None,
             reap: bool = True, *, include_token: bool = False) -> list[dict]:
    """Registry entries for ``localm ps``: each gains an ``alive`` flag from an
    identity-verified /whoami probe, with the attach ``token`` stripped (a
    listing must never surface a credential). Process-gone entries are reaped
    first unless *reap* is False. Probes run concurrently. Sorted by start
    time (oldest first).

    ``include_token``: keep the attach token in each row instead of stripping
    it. Default False (every DISPLAY caller, e.g. ``localm ps``). Set True only
    for an INTERNAL caller that needs the token to authenticate its OWN request
    to that instance (the MCP ``server_activity`` tool, which asks each
    discovered instance over HTTP) - never for anything a human reads."""
    probe = probe or default_probe
    if reap:
        reap_stale(home)
    entries = list_entries(home)

    def _probe(e: dict) -> bool:
        try:
            return probe(e)
        except Exception:
            return False

    alive_flags = _threaded_map(_probe, entries)
    rows: list[dict] = []
    for e, alive in zip(entries, alive_flags, strict=True):
        row = dict(e) if include_token else {k: v for k, v in e.items() if k != "token"}
        row["alive"] = alive
        rows.append(row)
    rows.sort(key=lambda r: r.get("started") or "")
    return rows


def list_machine_peers(home: Path, *, timeout: float = 0.7) -> list[dict]:
    """Live localm instances belonging to a DIFFERENT install, in ``snapshot``
    row shape, each carrying ``same_install: False``.

    ``snapshot`` reads ``<home>/run`` and so sees only instances sharing this
    install's data dir. This detects the other running instances directly
    (:func:`localm.gpu_registry.list_gpu_peers`: listening ports in localm's range
    that answer ``GET /whoami`` as localm and ``GET /v1/instances/status`` as the
    same instance) and returns the ones that are NOT in *home*'s own registry.

    ``mode`` and ``version`` come from the instance's own ``/whoami`` answer;
    ``root_dir`` from its full status, read only by a verified sibling of the
    same OS user (None otherwise). A peer listening on a non-loopback
    interface only is not found. ``started`` is None: a live probe has no
    start time.

    Returns no credential. ``snapshot`` is deliberately NOT widened to cover
    these: callers that decide whether a model file is held elsewhere
    (``selfclient.read_model_file_hold``) treat a miss as a rule-out, which is
    only sound while every server they reach shares one registry.

    Best-effort: any failure yields the peers found so far rather than raising.
    """
    try:
        from localm import gpu_registry
        peers = gpu_registry.list_gpu_peers(timeout=timeout)
    except Exception as e:
        logger.debug("cross-install peer lookup unavailable: %s", e)
        return []

    own = {str(e.get("instance_id")) for e in list_entries(home)}
    rows: list[dict] = []
    for peer in peers:
        iid = peer.get("instance_id")
        if not iid or str(iid) in own:
            continue
        rows.append({
            "instance_id": iid,
            "pid": peer.get("pid"),
            "port": peer.get("port"),
            "host": peer.get("host"),
            "scheme": peer.get("scheme"),
            "root_dir": peer.get("root_dir"),
            "mode": peer.get("mode"),
            "version": peer.get("version"),
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
    exit. :data:`LAUNCH_TAG_ENV` is removed from the environment in both cases
    and, when not isolated, recorded as the entry's ``launch_tag``. ``--isolated`` (test safety) sets *isolated* so the server is invisible
    to discovery: nothing attaches to it and it is never attached to.
    """
    instance_id = new_instance_id()
    root_dir = resolve_root_dir(override=project)
    token = new_token()
    launch_tag = take_launch_tag()

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
            launch_tag=launch_tag,
        )
        _OWN_PATHS[instance_id] = path
    else:
        path = register_hold_marker(
            home, instance_id=instance_id, port=port, host=host,
            token=token, scheme=scheme)
        _OWN_PATHS[instance_id] = path
    try:
        yield {"instance_id": instance_id, "root_dir": root_dir, "port": port,
               "token": token, "path": str(path) if path else None}
    finally:
        if path is not None:
            _OWN_PATHS.pop(instance_id, None)
            unregister_instance(path)
