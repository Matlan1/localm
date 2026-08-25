# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cross-INSTALL GPU/VRAM coordination registry (multi-instance cooperation).

``localm/instances.py`` already discovers/attaches localm instances for the
SAME install (its registry lives under ``<LOCALM_HOME>/run/``, one per install
by the self-contained rule). That is the wrong rendezvous point for THIS
feature: two different install locations (two clones, or a packaged install
plus a dev checkout) have two different ``LOCALM_HOME``s and simply cannot see
each other under either one's ``run/`` dir - yet they can still be fighting
over the SAME physical GPU. This module adds the one thing that gap needs: a
MACHINE-WIDE (not per-install) rendezvous directory, resolved via the stdlib
temp dir (never a hardcoded absolute path, never inside any one install's
``HOME_DIR``), so every localm process on the box - regardless of which
install it came from - can see every other one.

Everything else deliberately REUSES instances.py's already-tested machinery
rather than re-implementing it: :func:`localm.instances.pid_alive` for
liveness, and the same ``GET /whoami`` identity-handshake shape (``app ==
"localm"`` AND the instance_id matches the registry entry) via
:func:`localm.instances._try_whoami`, so a stale entry whose port got reused
by an unrelated process is never trusted here either. The registry file
itself uses the same atomic temp-file + ``os.replace`` + 0600-permission
pattern as :func:`localm.instances.register_instance`.

Entry schema (one JSON file per instance, ``<dir>/<instance_id>.json``)::

    {instance_id, pid, port, host, scheme, model, vram_estimate_bytes,
     gpu_index, updated_at, coordination_token}

``coordination_token`` is a per-instance secret (``secrets.token_urlsafe(32)``,
minted fresh at startup) - a SEPARATE credential from the real API key / shell
token / instance attach token. It authenticates exactly one narrow action:
``POST /v1/instances/cooperate-unload`` on ITS OWN instance, which unloads
that instance's currently-loaded model(s). Handing this token to a sibling
process is deliberately safe because it grants nothing else.

Everything here is advisory / best-effort: a read or write failure must never
crash the caller's own request path (a model load/unload must keep working
even if cross-instance coordination is broken), so every public function here
either returns a safe default or raises nothing - failures are logged, never
silenced (RULE 5), and never escalated into a harder failure than "no peer
was found"."""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from localm.debuglog import logger
from localm.instances import pid_alive, read_entry as _read_entry, _try_whoami

APP_NAME = "localm"


# ------------------------------------------------------------------ #
#  Paths + ids                                                        #
# ------------------------------------------------------------------ #

def registry_dir() -> Path:
    """The machine-wide rendezvous directory, shared by every localm install
    on this machine (unlike ``instances.py``'s per-install ``run/`` under
    ``LOCALM_HOME``). Resolved via the stdlib temp dir - portable, and never a
    hardcoded absolute path or an install-specific location."""
    return Path(tempfile.gettempdir()) / "localm" / "gpu"


def entry_path(directory, instance_id: str) -> Path:
    return Path(directory) / f"{instance_id}.json"


def new_coordination_token() -> str:
    """A per-instance, single-purpose secret - NEVER the real API key, shell
    token, or instance attach token. Knowing it grants exactly one thing:
    telling THIS instance to unload its own model(s)."""
    return secrets.token_urlsafe(32)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def age_seconds(updated_at_iso: Optional[str]) -> Optional[float]:
    """Seconds since *updated_at_iso* (an entry's ``updated_at``), or None if
    it is missing/unparseable. Best-effort - used only for a human-readable
    diagnostic, never a liveness decision."""
    if not updated_at_iso:
        return None
    try:
        then = datetime.fromisoformat(updated_at_iso)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - then).total_seconds())
    except (TypeError, ValueError):
        return None


def _lock_down_dir(path: Path) -> None:
    # A mkdir failure surfaces: write_entry catches OSError, logs it, returns None.
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError as e:
        # 0700 is best-effort: chmod no-ops on Windows and can fail on some
        # filesystems. The registry still functions without it.
        logger.debug("gpu_registry: could not chmod 0700 %s: %s", path, e)


# ------------------------------------------------------------------ #
#  Registry read/write                                               #
# ------------------------------------------------------------------ #

def write_entry(directory, *, instance_id: str, pid: int, port: Optional[int],
                host: str, scheme: str, model: Optional[str],
                vram_estimate_bytes: Optional[int], gpu_index: int,
                coordination_token: str) -> Optional[Path]:
    """Atomically write/update this instance's coordination entry (temp file +
    ``os.replace``, 0600 - same pattern as ``instances.register_instance``).

    Best-effort: a write failure is logged and returns None rather than
    raising - this must never break the model load/unload it is piggybacking
    on (RULE 5: logged, not silenced)."""
    d = Path(directory)
    try:
        _lock_down_dir(d)
        entry = {
            "instance_id": instance_id,
            "pid": pid,
            "port": port,
            "host": host,
            "scheme": scheme,
            "model": model,
            "vram_estimate_bytes": vram_estimate_bytes,
            "gpu_index": gpu_index,
            "updated_at": _now_iso(),
            "coordination_token": coordination_token,
        }
        path = entry_path(d, instance_id)
        # The entry carries the coordination token, so it gets the same
        # Windows-aware restriction as instances.register_instance.
        from localm.config import atomic_write_private
        ok = atomic_write_private(path, json.dumps(entry, indent=2))
        if not ok:
            # Reported with this subsystem's own name on top of the warning
            # restrict_file_perms already emits, so a gpu-coordination bug
            # report says which writer hit it. Not fatal: the retry on the
            # destination has already run by here.
            logger.debug("gpu_registry: could not restrict perms on the temp "
                         "file for %s", path)
        return path
    except OSError as e:
        logger.debug("gpu_registry: failed to write entry for %s: %s", instance_id, e)
        return None


def remove_entry(path) -> None:
    """Best-effort delete of one entry - call only with a path THIS process
    owns (its own entry). A crash just leaves the file to be aged out by
    :func:`reap_stale` / the liveness check in :func:`list_gpu_peers`, matching
    instances.py's own philosophy for its registry."""
    try:
        Path(path).unlink()
    except OSError:
        pass


def list_entries(directory) -> list:
    """All readable registry entries under *directory* (each gains a
    ``_path`` key). A missing directory yields ``[]`` (the normal
    zero-daemon-required case - no peer has ever registered here); a corrupt
    file is skipped, not raised on."""
    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        entry = _read_entry(f)
        if entry is not None:
            entry["_path"] = str(f)
            out.append(entry)
    return out


# ------------------------------------------------------------------ #
#  Liveness + reaping                                                 #
# ------------------------------------------------------------------ #

def reap_stale(directory, *, self_id: Optional[str] = None) -> list:
    """Remove entries whose process is confirmed gone (or whose file is
    corrupt). Never reaps *self_id*. Liveness reuses
    :func:`localm.instances.pid_alive` (conservative: an undeterminable PID
    counts as alive, so a doubtful entry is never reaped) - PID reuse itself is
    handled by :func:`list_gpu_peers`'s /whoami handshake, not here, exactly as
    in instances.py's own two-layer design."""
    d = Path(directory)
    if not d.is_dir():
        return []
    removed = []
    for f in sorted(d.glob("*.json")):
        entry = _read_entry(f)
        if entry is None:
            try:
                f.unlink()
                removed.append(f.stem)
            except OSError:
                pass
            continue
        if entry.get("instance_id") == self_id:
            continue
        try:
            alive = pid_alive(int(entry.get("pid", -1) or -1))
        except (TypeError, ValueError):
            alive = False
        if not alive:
            try:
                f.unlink()
                removed.append(entry.get("instance_id", f.stem))
            except OSError:
                pass
    return removed


def list_gpu_peers(directory=None, *, exclude_self_id: Optional[str] = None,
                    timeout: float = 0.7) -> list:
    """Live, identity-verified peer instances (excluding *exclude_self_id*
    and, unconditionally, THIS process).

    Self-exclusion is by PID, not only by *exclude_self_id*: two different
    processes can never share a pid at the same instant, so a registry entry
    whose pid equals ``os.getpid()`` is unambiguously this process's own,
    regardless of whether the caller has an instance_id to pass. This
    matters because at least one caller (the backend's VRAM-sizing layer,
    ``llamacpp/_sizing.py::_vram_holder_hint``) has no reliable handle on its
    own instance_id without a backwards layering dependency on the HTTP
    server module, but every caller trivially knows its own pid. Without
    this, a "peer" that turned out to be the caller itself produced a
    low-VRAM warning blaming "another localm instance" for VRAM this same
    process was using - see :func:`own_entry` for finding that self entry
    back when the caller needs to name it (e.g. for a more honest message).

    A registry entry is NEVER trusted on file contents alone: a candidate must
    pass BOTH a process-liveness check (:func:`localm.instances.pid_alive`) AND
    a ``GET /whoami`` handshake confirming ``app == "localm"`` and the
    instance_id matches (:func:`localm.instances._try_whoami` - the same
    impostor guard instances.py's own discovery uses), so a stale entry whose
    port got reused by an unrelated process is never treated as a live peer.

    Best-effort / advisory only: any failure reading the directory yields
    ``[]`` rather than raising (logged at debug, not silenced)."""
    d = Path(directory) if directory is not None else registry_dir()
    try:
        entries = list_entries(d)
    except Exception as e:
        logger.debug("gpu_registry: failed to list peers under %s: %s", d, e)
        return []

    self_pid = os.getpid()
    peers = []
    for entry in entries:
        iid = entry.get("instance_id")
        if not iid or iid == exclude_self_id:
            continue
        try:
            pid = int(entry.get("pid", -1) or -1)
        except (TypeError, ValueError):
            continue
        if pid == self_pid:
            continue
        if not pid_alive(pid):
            continue
        port = entry.get("port")
        if not port:
            continue
        scheme = entry.get("scheme") or "http"
        try:
            verified = _try_whoami(scheme, int(port), iid, timeout)
        except Exception as e:
            logger.debug("gpu_registry: whoami probe failed for %s: %s", iid, e)
            verified = False
        if verified:
            peers.append(entry)
    return peers


def own_entry(directory=None) -> Optional[dict]:
    """This process's own registry entry, matched by pid (never by
    instance_id - a caller may not have one at hand; see
    :func:`list_gpu_peers`'s docstring), or None if this process has not
    registered one or the directory cannot be read.

    Unlike :func:`list_gpu_peers`, no liveness or ``/whoami`` check is
    needed: a pid equal to ``os.getpid()`` is this process, by definition,
    so nothing here needs corroborating. Exists so a caller that fails to
    find an external peer (e.g. ``llamacpp/_sizing.py::_vram_holder_hint``)
    can still tell the user their OWN model is what is holding the VRAM,
    rather than falling through to a fully generic guess.

    Best-effort: mirrors :func:`list_entries` - a read failure yields None
    rather than raising (logged at debug, not silenced)."""
    d = Path(directory) if directory is not None else registry_dir()
    try:
        entries = list_entries(d)
    except Exception as e:
        logger.debug("gpu_registry: failed to read own entry under %s: %s", d, e)
        return None
    self_pid = os.getpid()
    for entry in entries:
        try:
            pid = int(entry.get("pid", -1) or -1)
        except (TypeError, ValueError):
            continue
        if pid == self_pid:
            return entry
    return None


# ------------------------------------------------------------------ #
#  Cooperative unload request                                        #
# ------------------------------------------------------------------ #

def request_cooperative_unload(peer: dict, *, timeout: float = 5.0) -> bool:
    """Ask a verified peer to release its own VRAM via its own
    ``POST /v1/instances/cooperate-unload``, authenticated with the PEER's OWN
    ``coordination_token`` (never our real API key / shell token - a separate,
    single-purpose credential the peer itself minted and stored only in its
    own 0600 registry entry).

    Advisory/best-effort: any failure (missing port/token, network error,
    timeout, non-200, malformed body) returns False. The caller must treat
    False exactly like "no peer available" - never escalate a failed
    cooperation attempt into a harder failure than the pre-existing 503."""
    port = peer.get("port")
    token = peer.get("coordination_token")
    if not port or not token:
        return False
    scheme = peer.get("scheme") or "http"
    # Same machine, so loopback - but WHICH loopback depends on what the peer
    # bound: an IPv6-bound peer does not answer on 127.0.0.1.
    from localm.bindhost import self_connect_host, url_host
    _h = url_host(self_connect_host(peer.get("host")))
    url = f"{scheme}://{_h}:{int(port)}/v1/instances/cooperate-unload"

    import requests
    try:
        from localm.tls import requests_verify
        verify = requests_verify(url)
    except FileNotFoundError:
        # CA file genuinely absent: verify=False is the intended best-effort
        # path on 127.0.0.1 (loopback, same machine) - mirrors instances.py's
        # _try_whoami handling of the identical case.
        verify = False
    except Exception as e:
        logger.debug("gpu_registry: could not determine TLS verification for %s: %s", url, e)
        return False

    try:
        r = requests.post(url, json={"coordination_token": token},
                          headers={"X-LocalM-Coordination-Token": token},
                          timeout=timeout, verify=verify)
    except requests.RequestException as e:
        logger.debug("gpu_registry: cooperate-unload request to %s failed: %s", url, e)
        return False
    if r.status_code != 200:
        logger.debug("gpu_registry: cooperate-unload to %s returned %s", url, r.status_code)
        return False
    try:
        data = r.json()
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("status") in ("unloaded", "already_unloaded")
