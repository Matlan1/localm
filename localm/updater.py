# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-updater: check the localm proxy for a newer release and (on explicit user
action ONLY) apply it.

Checking is automatic (a quiet startup check + a manual button/command); APPLYING is
never automatic - it runs only from ``localm update`` (confirmed) or the GUI "Update
now" button. Most updates are code-only and need just a file swap + reboot (the
install is editable, so new source is live on restart); ``deps``/``runtime`` escalate
only when they actually change. The file-swap primitives live in
``localm/_apply_update.py``; ``apply()`` below backs up first and rolls back on a
failed swap or post-step. The server-initiated restart (the ONLY unattended
transition - the CLI always tells the user to relaunch by hand) is followed by a
DETACHED post-relaunch health watchdog (``spawn_health_watchdog()``
below) that polls the restarted instance's own ``/whoami`` for the applied
VERSION and automatically invokes the standalone ``scripts/rollback_update.py``
if it never comes up healthy within its timeout. Manual recovery is still always
available: ``localm update --rollback``, or ``rollback.bat``/``rollback.sh`` in
the install root.
"""

from __future__ import annotations

import contextlib
import json as _json
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from localm import _version

# Update classes, least -> most disruptive.
_ORDER = ("reboot", "deps", "runtime", "setup")

# ---------------------------------------------------------------------------
#  Pinned release-signing public key(s)
# ---------------------------------------------------------------------------
# Each release build.zip is signed with an Ed25519 private key whose public half
# is pinned here. apply() verifies the signature against these keys BEFORE
# anything is extracted. The list allows a key to be rotated in before an old one
# is retired.
#
# With a key pinned the updater enforces: a missing, unsigned, malformed or
# non-matching signature is refused before any swap. An EMPTY tuple fails open
# (transport trust only).
_UPDATE_PUBKEYS: tuple = (
    "3501b23eb1ab6ec245b5e9d9c6a70b522ca89d6250d13060a9642ce1c8868ecf",
)


def _load_update_pubkeys() -> list:
    """The pinned Ed25519 public keys as verifier objects; [] when none/invalid.

    A malformed pinned key is WARNED about and skipped, so a second, valid key
    still works. When NO valid key results, verify_signature below fails closed."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from localm.debuglog import logger
    keys = []
    for hexkey in _UPDATE_PUBKEYS:
        try:
            keys.append(Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(hexkey).strip())))
        except Exception as e:
            logger.warning(
                "pinned update key %.16s... does not parse as an Ed25519 public "
                "key hex (%s); skipping it", str(hexkey), e)
            continue
    return keys


def verify_signature(data: bytes, signature_b64) -> None:
    """Verify *signature_b64* (base64 Ed25519 signature) over *data* against the
    pinned public key(s), IF signing is configured.

    Signature verification is OPTIONAL hardening (see _UPDATE_PUBKEYS): fail-OPEN
    when unconfigured, enforce when configured.

    - No key pinned (the default): this ALLOWS the update (returns). Authenticity
      rests on the HTTPS-pinned transport plus the private release channel.
    - A key pinned but every pin UNPARSEABLE: configured-but-broken, NOT
      unconfigured -> FAIL CLOSED.
    - A key pinned and valid: REQUIRE a matching signature; a missing / malformed /
      non-matching signature refuses before any swap (a downgrade is checked
      separately, in _refuse_downgrade).

    Raises :class:`~localm.bugreport.LocalmError` on a refusal."""
    import base64

    from cryptography.exceptions import InvalidSignature
    from localm.bugreport import LocalmError
    keys = _load_update_pubkeys()
    if not keys:
        if _UPDATE_PUBKEYS:
            # Pins ARE configured but none parses -> fail closed.
            raise LocalmError(
                "refusing to apply an update: no pinned signing key is usable",
                reason=f"{len(_UPDATE_PUBKEYS)} key(s) are pinned in _UPDATE_PUBKEYS "
                       "but none parses as an Ed25519 public key hex")
        # No key pinned: allow the update on transport trust alone.
        return
    if not signature_b64:
        raise LocalmError(
            "refusing to apply an unsigned update",
            reason="a signing key is pinned but the release included no signature")
    try:
        sig = base64.b64decode(str(signature_b64), validate=True)
    except Exception:
        raise LocalmError("the update signature is malformed", reason="not valid base64")
    for key in keys:
        try:
            key.verify(sig, data)
            return   # a pinned key validated the build - authentic
        except InvalidSignature:
            continue
    raise LocalmError(
        "the update signature did not match the pinned key",
        reason="the build may be tampered with, or signed with an unknown key")


def _refuse_downgrade(new_version: str) -> None:
    """Raise unless *new_version* is strictly newer than the running version.

    The anti-rollback check: the signature proves authenticity, not freshness, and
    an OLD release's build.zip is still validly signed. This adds freshness."""
    from localm.bugreport import LocalmError
    current = _version.read_version()
    if not new_version:
        raise LocalmError("the update has no VERSION", reason="cannot confirm it is newer")
    if not _version.is_newer(new_version, current):
        raise LocalmError(
            f"refusing to 'update' to {new_version}: not newer than the installed {current}",
            reason="anti-rollback blocked a downgrade or a replayed old build")


def endpoint() -> tuple:
    """(base_url, token) for the update channel: ``update_url``/``update_token`` if
    set, else the shared ``bugreport_upload_url``/``_token`` (one Worker hosts report
    + issues + update). (None, None) when not configured. Never raises."""
    try:
        from localm.config import load_config
        cfg = load_config()
        url = (cfg.get("update_url") or cfg.get("bugreport_upload_url") or "").strip() or None
        token = (cfg.get("update_token") or cfg.get("bugreport_upload_token") or "").strip() or None
        return url, token
    except Exception:
        return None, None


def available() -> bool:
    """True when an update endpoint is configured."""
    return endpoint()[0] is not None


def repo_root() -> Path:
    """The installed repo root (one level above the package dir)."""
    return Path(__file__).resolve().parents[1]


def _prerelease_channel_enabled() -> bool:
    """Whether this install opted into the prerelease update channel
    (settings_schema.py's ``update_allow_prerelease``, admin_only, default
    False). Best-effort: an unreadable config falls back to stable-only."""
    try:
        from localm.config import load_config
        return bool(load_config().get("update_allow_prerelease", False))
    except Exception:
        return False


def _net_policy_allows_update_check() -> bool:
    """Whether the update check may reach the network: allowed whenever
    ``net_mode`` is not ``"off"``, or an admin explicitly exempted the update
    channel (settings_schema.py's ``update_ignore_net_policy``, admin_only,
    default False).

    Only the literal ``net_mode = off`` kill switch stops the check; "ask" and
    "allow" need no per-call confirmation here.

    Best-effort: an unreadable config resolves to blocked."""
    from localm.netpolicy import network_mode
    if network_mode() != "off":
        return True
    try:
        from localm.config import load_config
        return bool(load_config().get("update_ignore_net_policy", False))
    except Exception:
        return False


def check(*, opener=None) -> dict:
    """Ask the proxy for the latest release and compare to the running version.

    Returns ``{current, latest, newer, notes, published_at, asset}``. *latest* is
    None when there are no releases. Raises :class:`~localm.bugreport.LocalmError`
    when not configured, blocked by network policy, or the proxy fails - the
    caller decides whether to surface it (the manual command does) or swallow it
    (the startup check does).

    Network policy: gated on :func:`_net_policy_allows_update_check` BEFORE any
    request is built or attempted, so a blocked check never resolves DNS or
    opens a socket, and it RAISES rather than returning a dict.

    When the prerelease channel is opted into, appends ``?channel=prerelease`` to
    the request; the PROXY (not this client) decides which single candidate
    release to return for that channel, and is_newer() below orders that one
    candidate against the running version."""
    from localm import _proxy
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the updater is not configured",
                          reason="set bugreport_upload_url (or update_url) to enable it")
    if not _net_policy_allows_update_check():
        raise LocalmError(
            "update checks are off because network access is set to off",
            reason='turn on "Check for updates even when network access is off" '
                   "in Settings, or set network access to ask or allow")
    path = "/update?channel=prerelease" if _prerelease_channel_enabled() else "/update"
    data = _proxy.request(base, path, token=token, opener=opener)
    current = _version.read_version()
    latest = data.get("version") if isinstance(data, dict) else None
    newer = bool(latest) and _version.is_newer(latest, current)
    # Whether "not newer" means anything: is_newer() returns False both for a
    # genuine tie or older release and for a tag it could not parse as a version.
    comparable = (not latest) or _version.comparable(latest, current)
    return {
        "current": current,
        "latest": latest,
        "newer": newer,
        "comparable": comparable,
        "notes": (data.get("notes") or "") if isinstance(data, dict) else "",
        "published_at": data.get("published_at") if isinstance(data, dict) else None,
        "asset": data.get("asset") if isinstance(data, dict) else None,
        # The proxy serves the release's Ed25519 signature (base64) alongside the
        # asset; apply() verifies the downloaded build against the pinned key.
        "signature": data.get("signature") if isinstance(data, dict) else None,
    }


# ---------------- classification (run after download+extract) ------------

def _read_toml(path: Path) -> dict:
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _deps_set(root: Path) -> frozenset:
    """Declared dependencies + extras for the project at *root* (from pyproject.toml).
    Empty when unreadable."""
    data = _read_toml(root / "pyproject.toml")
    proj = data.get("project", {}) if isinstance(data, dict) else {}
    deps = set(proj.get("dependencies", []) or [])
    for extra, items in (proj.get("optional-dependencies", {}) or {}).items():
        for it in items or []:
            deps.add(f"{extra}:{it}")
    return frozenset(deps)


def _requires_python(root: Path) -> str:
    data = _read_toml(root / "pyproject.toml")
    proj = data.get("project", {}) if isinstance(data, dict) else {}
    return str(proj.get("requires-python", "") or "")


def _max_class(a: str, b: str) -> str:
    ia = _ORDER.index(a) if a in _ORDER else 0
    ib = _ORDER.index(b) if b in _ORDER else 0
    return _ORDER[max(ia, ib)]


def read_manifest(staged_dir) -> dict:
    """Optional ``update.json`` a release may include (``{version, needs, notes}``);
    {} if absent or unreadable. ``needs`` can ESCALATE the auto-detected class."""
    try:
        p = Path(staged_dir) / "update.json"
        if p.is_file():
            return _json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        # A corrupt or unreadable manifest; {} makes classify() rely on its own
        # tree auto-detection.
        pass
    return {}


def classify(staged_dir, installed_dir=None, manifest: Optional[dict] = None) -> str:
    """The lightest apply action for a staged update vs. the installed tree:

    - ``reboot``  - only source/assets changed -> swap + restart (editable install).
    - ``deps``    - pyproject dependencies/extras changed -> + ``uv pip install -e``.
    - ``runtime`` - the native llama.cpp build must be re-provisioned -> + ``setup-llama``.
      The binaries are not in the source tree, so this is taken from the release
      manifest's ``needs``, never auto-detected.
    - ``setup``   - ``requires-python`` changed -> re-run setup.

    Returns the MAX (most disruptive) of the manifest's declared ``needs`` and what
    is auto-detected, so a manifest can escalate but never silently downgrade."""
    staged = Path(staged_dir)
    installed = Path(installed_dir) if installed_dir else repo_root()
    klass = "reboot"
    if isinstance(manifest, dict) and manifest.get("needs") in _ORDER:
        klass = _max_class(klass, manifest["needs"])
    new_py, old_py = _requires_python(staged), _requires_python(installed)
    if new_py != old_py:   # any change INCLUDING removal (empty new) needs setup
        klass = _max_class(klass, "setup")
    if _deps_set(staged) != _deps_set(installed):
        klass = _max_class(klass, "deps")
    return klass


def class_summary(klass: str) -> str:
    """A one-line, user-facing description of what applying *klass* entails."""
    return {
        "reboot": "applies with just a restart",
        "deps": "reinstalls dependencies, then restarts",
        "runtime": "re-provisions the native runtime (re-downloads binaries), then restarts",
        "setup": "needs setup.bat re-run (a Python/venv-level change)",
    }.get(klass, "applies and restarts")


# ----------------------------- download ---------------------------------

def download(asset_id, dest, *, timeout: float = 120.0, opener=None) -> Path:
    """Stream the release asset (build zip) from the proxy to *dest*; return the path.

    Raises :class:`~localm.bugreport.LocalmError` on failure. *opener* is injectable
    for tests: it receives ``(url, headers, timeout, dest)`` and writes the file."""
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the updater is not configured", reason="no update endpoint")
    try:
        aid = int(asset_id)
    except (TypeError, ValueError):
        raise LocalmError("bad asset id", reason=str(asset_id))
    url = base.rstrip("/") + f"/update/download?id={aid}"
    headers = {"User-Agent": "localm"}
    if token:
        headers["X-Localm-Token"] = token
    dest = Path(dest)
    if opener is not None:
        opener(url, headers, timeout, dest)
        return dest
    import urllib.error
    import urllib.parse
    import urllib.request
    # A code-update download stays on HTTPS end to end: a non-HTTPS endpoint is
    # refused, and so is a redirect that downgrades to http. Redirects that stay
    # https are followed. The signature is verified separately in apply(), before
    # the build is extracted.
    if urllib.parse.urlparse(url).scheme != "https":
        raise LocalmError("refusing a non-HTTPS update download",
                          reason="the update endpoint must be https")

    # verified_urlopen follows the 302 to the GitHub release CDN and verifies both
    # hops, native cert store first, certifi as fallback. The https-only-redirect
    # guard is http_ssl.HttpsOnlyRedirect, installed by default.
    from localm.http_ssl import RedirectDowngradeRefused, verified_urlopen
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with verified_urlopen(req, timeout=timeout) as resp:
            if not (200 <= int(resp.status) < 300):
                raise LocalmError("the update download failed", reason=f"HTTP {resp.status}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        raise LocalmError("the update download failed", reason=f"HTTP {e.code}")
    except RedirectDowngradeRefused as e:
        # Must stay ahead of the URLError clause below, which it subclasses.
        raise LocalmError("the update download tried to downgrade to http",
                          reason=str(getattr(e, "reason", e)))
    except (urllib.error.URLError, OSError) as e:
        raise LocalmError("could not download the update", reason=str(getattr(e, "reason", e)))
    return dest


# ------------------------------ apply -----------------------------------

def _updates_dir(*, create: bool = True) -> Path:
    """The data dir holding update scratch, the manifest, and the stable backup.

    ``create=False`` is for READ-ONLY probes (:func:`rollback_info`): a status poll
    must not bring an ``updates/`` tree into existence on an install that has never
    updated."""
    from localm.config import home_dir
    d = home_dir() / "updates"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# Fallback-only threshold, used ONLY when a lock's holder PID could not be
# determined at all. Not the primary staleness signal - see _apply_lock_is_stale.
_APPLY_LOCK_STALE_S = 1800.0


def _lock_holder_pid(lock_dir: Path):
    """The PID recorded by whoever currently holds *lock_dir* (written by
    _apply_lock right after it acquires the lock), or None if unrecorded /
    unreadable - an older-format lock, or a crash in the gap between mkdir
    and the pid write."""
    try:
        return int((lock_dir / "pid").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _apply_lock_is_stale(lock_dir: Path) -> bool:
    """Whether *lock_dir* is an ORPHAN (its holder is gone) rather than a
    legitimately long-running apply.

    Primary signal: the PID _apply_lock recorded at acquisition. If
    instances.pid_alive() confirms it is dead, the lock is stale regardless of
    age. If the PID is alive, or pid_alive cannot tell (its own conservative
    default), the lock is NOT stale no matter how long it has been held.

    Fallback ONLY when no PID was recorded at all: age against
    _APPLY_LOCK_STALE_S."""
    from localm.instances import pid_alive
    pid = _lock_holder_pid(lock_dir)
    if pid is not None:
        return not pid_alive(pid)
    try:
        age = time.time() - lock_dir.stat().st_mtime
    except OSError:
        return True   # cannot even stat it -> treat as gone, do not block forever
    return age >= _APPLY_LOCK_STALE_S


@contextlib.contextmanager
def _apply_lock(what: str = "update"):
    """Cross-process, cross-thread single-flight guard for :func:`apply` and
    :func:`rollback_last`, which mutate the SAME install tree. ``mkdir`` is
    atomic, so two concurrent callers - two browser tabs' ``POST
    /api/update/apply`` (both landing in the SAME process via
    ``asyncio.to_thread``), or the CLI racing a live server in a SEPARATE
    process - cannot both proceed past this point.

    The SECOND caller fails FAST with a refusal rather than blocking. A lock
    whose holder is confirmed gone (see :func:`_apply_lock_is_stale`) is
    reclaimed."""
    import os
    lock_dir = _updates_dir() / "apply.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError:
        if _apply_lock_is_stale(lock_dir):
            # Reclaim an orphaned lock. Losing the race to reclaim it makes our own
            # mkdir below raise FileExistsError again, reported like a live lock.
            with contextlib.suppress(OSError):
                shutil.rmtree(lock_dir)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            from localm.bugreport import LocalmError
            raise LocalmError(
                f"another {what} is already being applied",
                reason="wait for it to finish, then try again")
    # Record who holds it, for a future caller's staleness check. Best-effort: a
    # failed write means the next check falls back to age.
    try:
        (lock_dir / "pid").write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(lock_dir)


def _new_run_dir() -> Path:
    """A fresh, unique-per-call scratch directory for ONE apply() run: the
    download, staging extraction, and swap-time backup for this run never share a
    path with any other run's."""
    d = _updates_dir() / "runs" / uuid.uuid4().hex
    d.mkdir(parents=True)
    return d


def _promote_backup(run_backup_dir: Path, backup_dir: Path) -> None:
    """Move *run_backup_dir* (this run's own, just-written backup) to the stable
    *backup_dir* location :func:`rollback_last` reads, replacing whatever was
    there.

    Must be called ONLY after ``swap_with_backup`` has already succeeded, so
    *run_backup_dir* is a complete, self-contained COPY of the pre-update install
    (``_apply_update.backup`` copies, never moves, the live tree).

    Best-effort: a failure does not undo the update but IS surfaced as a warning,
    since it degrades a later ``update --rollback``."""
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(run_backup_dir), str(backup_dir))
    except OSError as e:
        _apply_warn("update applied, but could not move this run's backup from %s "
                    "to %s: %s; a later `update --rollback` may not find it",
                    run_backup_dir, backup_dir, e)


def apply(asset_id, *, signature=None, installed=None, download_opener=None,
          runner=None) -> dict:
    """Download, VERIFY (signature + anti-rollback), and swap a build into the
    install, then run the class's deps/runtime step. Returns
    ``{applied, version, klass, backup, restart_needed}``.

    *signature* is the base64 Ed25519 signature from :func:`check` (the proxy serves
    it with the release). It is verified against the pinned public key(s) BEFORE the
    build is extracted or any of its code can run; a missing/invalid signature, an
    unconfigured key, or a downgrade all raise before anything is swapped (fail
    closed - we never install an unverified build).

    Single-flight (:func:`_apply_lock`): a second concurrent call is refused
    immediately rather than racing the first. Each call also downloads, extracts,
    and takes its swap-time backup in its OWN unique scratch directory
    (:func:`_new_run_dir`) and, once its swap has succeeded, moves that backup to
    the stable location :func:`rollback_last` reads (:func:`_promote_backup`).

    Rolls back on a swap or post-step failure (never a half-applied tree). Does NOT
    restart - the caller does (the CLI tells the user; the server re-execs). The file
    primitives live in :mod:`localm._apply_update`. *installed* defaults to the real
    repo root; tests pass a fake tree so apply never touches the live install.
    Injectables for tests: *download_opener* (download()'s opener) and *runner* (runs
    the post-swap command, returns an int exit code)."""
    import subprocess
    from localm import _apply_update as au
    from localm.bugreport import LocalmError
    target = Path(installed) if installed else repo_root()
    updir = _updates_dir()
    backup_dir = _backup_dir(create=True)   # the STABLE location rollback_last() reads

    with _apply_lock():
        run_dir = _new_run_dir()
        zip_path, staging, run_backup_dir = (
            run_dir / "build.zip", run_dir / "staging", run_dir / "backup")

        try:
            download(asset_id, zip_path, opener=download_opener)
            # SIGNATURE GATE - runs before verify_zip/extract/swap, i.e. before any
            # of the downloaded build's code can be extracted or executed.
            verify_signature(zip_path.read_bytes(), signature)
            au.verify_zip(zip_path)
            root = au.extract(zip_path, staging)
            vf = root / "VERSION"
            new_version = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
            # ANTI-ROLLBACK - a validly SIGNED but OLDER build must not be installable.
            _refuse_downgrade(new_version)
            klass = classify(root, target, read_manifest(root))
        except Exception:
            # No backup exists for this run yet, so the scratch dir can go.
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        # Remove any stale applied_names.json from a previous update before the
        # swap, so a failed write below degrades to the backup-dir listing rather
        # than leaving the prior update's names. Placed after
        # download/verify/extract so an early abort leaves the previous manifest
        # intact.
        manifest_path = updir / "applied_names.json"
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            _apply_warn("could not remove the previous update manifest %s: %s; if recording "
                        "the new one below also fails, a later rollback may use stale names",
                        manifest_path, e)

        # swap_with_backup backs up to run_backup_dir first, then swaps, rolling
        # back from it and re-raising on failure. run_dir is left in place on
        # failure, so run_backup_dir survives for a manual recovery.
        names = au.swap_with_backup(root, target, run_backup_dir)
        # Promote this run's backup to the stable path before the manifest write or
        # the post-swap command's own rollback touches `backup_dir`, so both use the
        # fixed path rollback_last() expects.
        _promote_backup(run_backup_dir, backup_dir)
        shutil.rmtree(run_dir, ignore_errors=True)   # zip/staging no longer needed

        # Record the full swap set, including brand-new top-level entries the backup
        # dir does not contain, so a later `update --rollback` removes them too.
        try:
            import json
            manifest_path.write_text(json.dumps(sorted(names)), encoding="utf-8")
        except OSError as e:
            _apply_warn("could not record the update manifest %s: %s; a later "
                        "`update --rollback` will fall back to the backup listing and cannot "
                        "remove brand-new top-level entries this update added", manifest_path, e)

        cmd = au.post_swap_command(klass, backend=_installed_backend())
        if cmd:
            run = runner or (lambda c: subprocess.run(c, cwd=str(target)).returncode)

            def _rollback_or_raise(why):
                # A failure of the recovery ITSELF raises rather than reporting a
                # clean rollback over a broken install.
                try:
                    au.rollback(backup_dir, target, names)
                except Exception as rb:
                    raise LocalmError(
                        "the post-update step failed AND rollback failed - manual recovery needed",
                        reason=f"{why}; rollback: {rb}; restore from {backup_dir}")

            try:
                rc = int(run(cmd))
            except Exception as e:
                _rollback_or_raise(f"post-update step crashed: {e}")
                raise LocalmError("the post-update step crashed; rolled back", reason=str(e))
            if rc != 0:
                _rollback_or_raise(f"{cmd[0]} exited {rc}")
                raise LocalmError("the post-update step failed; rolled back",
                                  reason=f"{cmd[0]} exited {rc}")
        return {"applied": True, "version": new_version, "klass": klass,
                "backup": str(backup_dir), "restart_needed": True}


def _installed_backend() -> str:
    """The backend for a runtime re-provision: whatever is ALREADY INSTALLED on
    this box, read from setup_llama's on-disk marker - never re-derived from
    the current hardware-recommendation policy. Falls back to
    recommended_install_backend() only when nothing is provisioned yet (no
    marker at all: a fresh install, or one predating the marker).

    An already-installed backend is preserved unconditionally, whether the user
    chose it deliberately or inherited it from an older default; an update never
    switches it."""
    try:
        from localm import setup_llama
        installed = setup_llama.installed_backend()
        if installed:
            return installed
    except Exception:
        pass
    try:
        from localm import hwdetect
        return hwdetect.recommended_install_backend() or "vulkan"
    except Exception:
        return "vulkan"


# ------------------------- post-restart health watchdog -------------------

# Defaults for the detached post-restart health watchdog. TIMEOUT budgets the
# whole restarted process coming back up and answering /whoami, which does not
# wait on model load. Runs unattended after the update button has returned 200.
WATCHDOG_TIMEOUT_S = 90.0
WATCHDOG_POLL_INTERVAL_S = 2.0
WATCHDOG_REQUEST_TIMEOUT_S = 3.0


def _watchdog_script() -> Path:
    return repo_root() / "scripts" / "update_watchdog.py"


def spawn_health_watchdog(*, host: str, port, scheme: str, expect_version: str,
                          timeout: float = WATCHDOG_TIMEOUT_S,
                          poll_interval: float = WATCHDOG_POLL_INTERVAL_S,
                          request_timeout: float = WATCHDOG_REQUEST_TIMEOUT_S,
                          popen=None) -> bool:
    """Spawn ``scripts/update_watchdog.py`` DETACHED to verify the build
    ``_do_restart`` is about to re-exec into actually comes back up, auto-rolling
    back if it does not. Returns True iff the spawn was attempted and succeeded -
    NEVER raises: a watchdog that fails to spawn must not block or fail the
    restart itself. *popen* is injectable, matching apply()'s
    download_opener/runner convention."""
    try:
        import os
        import subprocess
        import sys
        from localm.config import home_dir
        script = _watchdog_script()
        if not script.is_file():
            _watchdog_warn("update watchdog script missing at %s; this restart "
                           "will not be auto-verified/rolled back", script)
            return False
        home = home_dir()
        log_file = home / "updates" / "watchdog.log"
        argv = [sys.executable, str(script),
               "--host", str(host), "--port", str(port), "--scheme", str(scheme),
               "--expect-version", str(expect_version),
               "--install-root", str(repo_root()),
               "--timeout", str(timeout), "--poll-interval", str(poll_interval),
               "--request-timeout", str(request_timeout),
               "--log-file", str(log_file)]
        env = os.environ.copy()
        # So the watchdog's rollback call re-derives the same data home this server
        # was using, via rollback_update.py's own _detect_home() precedence.
        env["LOCALM_HOME"] = str(home)
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, env=env, close_fds=True)
        if sys.platform == "win32":
            # DETACHED_PROCESS (no console) + CREATE_NEW_PROCESS_GROUP so the
            # watchdog is not tied to this process's console/process group and
            # survives past the execv this call precedes. The literal fallbacks
            # keep this branch runnable on any platform.
            kwargs["creationflags"] = (
                getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200))
        else:
            # setsid: detaches from the parent's session/process group, so the
            # watchdog also survives a SIGHUP if the controlling terminal closes.
            kwargs["start_new_session"] = True
        (popen or subprocess.Popen)(argv, **kwargs)
        return True
    except Exception as e:
        _watchdog_warn("could not spawn the update health watchdog: %s", e)
        return False


def _watchdog_warn(msg, *args) -> None:
    try:
        from localm.debuglog import logger
        logger.warning(msg, *args)
    except Exception:
        pass   # logging must never be the thing that raises here


def _apply_warn(msg, *args) -> None:
    """Best-effort WARNING for apply()/rollback bookkeeping (the manifest
    write/prune). Never raises, even when the logger cannot be imported."""
    try:
        from localm.debuglog import logger
        logger.warning(msg, *args)
    except Exception:
        pass


def _backup_dir(*, create: bool = False) -> Path:
    """The ONE stable path :func:`apply` promotes each run's backup to and
    :func:`rollback_last` restores from."""
    return _updates_dir(create=create) / "backup"


def _backup_is_restorable(backup_dir: Path) -> bool:
    """True when *backup_dir* holds something :func:`rollback_last` could restore.

    ONE predicate, shared by the probe and the action. An unreadable directory
    answers False, and the action re-checks and reports the real reason if it is
    actually attempted."""
    try:
        return backup_dir.is_dir() and any(backup_dir.iterdir())
    except OSError:
        return False


def rollback_info() -> dict:
    """What :func:`rollback_last` would restore, WITHOUT restoring anything.

    Returns ``{available, backup, version, current}``. ``version`` is the backed-up
    build's VERSION, or None when the backup predates that file or it is unreadable
    (reported as unknown, never guessed from the running version).

    STRICTLY READ-ONLY: it restores nothing (:func:`rollback_last` MOVES THE
    INSTALL) and does not create ``updates/`` (see :func:`_updates_dir`), so a GUI
    status poll leaves an install that has never updated exactly as it was."""
    backup_dir = _backup_dir()
    available = _backup_is_restorable(backup_dir)
    version = None
    if available:
        try:
            version = (backup_dir / "VERSION").read_text(encoding="utf-8").strip() or None
        except OSError:
            version = None
    return {"available": available,
            "backup": str(backup_dir) if available else None,
            "version": version,
            "current": _version.read_version()}


def rollback_last(*, installed=None) -> dict:
    """Restore the install from the most recent update backup. Returns
    ``{rolled_back, backup}``. Raises LocalmError when there is no backup.

    NOT signature- or freshness-checked; the owner gate on its HTTP callers
    (``inference/routes/admin.py``) is the control on this operation. Use
    :func:`rollback_info` to ask whether a rollback is possible; calling this one
    to find out performs it."""
    from localm import _apply_update as au
    from localm.bugreport import LocalmError
    target = Path(installed) if installed else repo_root()
    updir = _updates_dir()
    backup_dir = _backup_dir(create=True)
    if not _backup_is_restorable(backup_dir):
        raise LocalmError("no update backup to roll back to", reason=str(backup_dir))
    # The SAME single-flight lock apply() takes, because both mutate the SAME
    # install tree. Cross-PROCESS: the contender can be the CLI in a terminal,
    # which no in-process lock could see.
    with _apply_lock("update or rollback"):
        # Prefer the recorded full swap set (includes brand-new top-level entries the update
        # added, which are NOT in the backup dir) so those are removed too; fall back to the
        # backup listing for an older backup written before the manifest existed. au.rollback
        # removes each name then restores whatever the backup holds, so a manifest-only (new)
        # name is removed-not-restored and a backed-up name is removed-then-restored.
        names = None
        manifest = updir / "applied_names.json"
        if manifest.is_file():
            try:
                import json
                loaded = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(loaded, list) and all(isinstance(x, str) for x in loaded):
                    names = loaded
            except (OSError, ValueError):
                names = None
            if names is None:
                # The manifest EXISTS but is unreadable or malformed (NOT the benign
                # never-written case): the rollback still runs from the backup dir,
                # but that listing lacks the brand-new top-level entries the update
                # ADDED, so those will not be removed.
                _apply_warn("applied_names.json at %s exists but is unreadable; new files "
                            "added by the update will not be removed by this rollback", manifest)
        if names is None:
            names = [p.name for p in backup_dir.iterdir()]
        au.rollback(backup_dir, target, names)
        return {"rolled_back": True, "backup": str(backup_dir)}
