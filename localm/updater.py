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
DETACHED post-relaunch health watchdog (LM-DA-011, ``spawn_health_watchdog()``
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
#  CHK-UPDATER-INTEGRITY (signature half): pinned release-signing key(s)
# ---------------------------------------------------------------------------
# Applying an update EXTRACTS and EXECUTES the downloaded build, so authenticity
# matters. Two layers protect it, and BOTH are on by default:
#   - the download is HTTPS-pinned (see download(); no http downgrade) and the release
#     channel is a PRIVATE repo behind a server-side token; AND
#   - each release build.zip is SIGNED with the Ed25519 private key whose PUBLIC half
#     is pinned below. apply() verifies the signature against these pinned key(s)
#     BEFORE anything is extracted, so a build served by a COMPROMISED release channel
#     (a leaked proxy/GitHub token, a tampered release, a same-channel checksum) is
#     rejected: the attacker controls the build but not the pinned key.
#
# The PRIVATE key is held by the release signer and used only to sign releases; it is
# never in the repo, the proxy, or CI. (A compromise of the signing machine itself is
# out of scope - at that point the whole toolchain is owned regardless.) The PUBLIC key
# ships in source: public data an attacker cannot forge, pinned so it cannot be swapped.
# It is a LIST so a key can be rotated in before an old one is retired.
#
# BEHAVIOUR (mirrors localm's auth model): with a key pinned the updater ENFORCES - a
# missing, unsigned, malformed, or non-matching signature is refused before any swap.
# If this tuple were ever EMPTY the updater fails OPEN (applies on transport trust) so
# an accidentally keyless build is not bricked, but the shipped default pins a key and
# test_updater_signature guards that it stays non-empty.
#
# Release flow: scripts/make_release.py -> build_release.py assembles the build.zip,
# sign_release.py signs it with the private key, and the update proxy serves the base64
# signature in its /update JSON (check() reads it -> apply() verifies it here).
_UPDATE_PUBKEYS: tuple = (
    "3501b23eb1ab6ec245b5e9d9c6a70b522ca89d6250d13060a9642ce1c8868ecf",
)


def _load_update_pubkeys() -> list:
    """The pinned Ed25519 public keys as verifier objects; [] when none/invalid.

    A malformed pinned key is skipped (so a second, valid key still works) rather
    than crashing the check - but if NO valid key results, verify_signature below
    fails closed, so a bad pin can never weaken verification into a silent pass.
    The skip is WARNED, never silent: a typo'd pin would otherwise surface as the
    misleading "not set up" refusal (missing != corrupt, AGENTS.md rule 5)."""
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

    Signature verification is OPTIONAL hardening (see _UPDATE_PUBKEYS); this mirrors
    localm's auth model - fail-OPEN when unconfigured, enforce when configured:

    - No key pinned (the default): signing is not set up, so this ALLOWS the update
      (returns). Authenticity rests on the HTTPS-pinned transport + private release
      channel; the self-updater must work out of the box with zero setup.
    - A key pinned but every pin UNPARSEABLE: configured-but-broken, NOT unconfigured
      -> FAIL CLOSED (missing != corrupt); a typo must never silently degrade into
      "no verification".
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
            # Pins ARE configured but none parses -> fail closed (missing != corrupt):
            # a typo'd pin must not weaken verification into a silent pass.
            raise LocalmError(
                "refusing to apply an update: no pinned signing key is usable",
                reason=f"{len(_UPDATE_PUBKEYS)} key(s) are pinned in _UPDATE_PUBKEYS "
                       "but none parses as an Ed25519 public key hex")
        # UNCONFIGURED (no key pinned): signing is optional hardening that is not
        # turned on, so allow the update - like auth.py with no key set. The default,
        # documented fail-open posture (transport-trusted, not signature-verified);
        # not a silenced failure.
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

    Anti-rollback: a build.zip for an OLD release is still validly SIGNED, so a
    MITM / compromised proxy could replay it to force a downgrade to a known-
    vulnerable version. The signature proves authenticity, not freshness; this adds
    freshness."""
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
    False - an rc build is signed and anti-rollback checked exactly like a
    stable one, but this is still an explicit opt-in, never a default-on
    channel). Best-effort: an unreadable config falls back to stable-only,
    the safe direction, never to silently offering prereleases."""
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

    Same bar as model_manager/pull.py's own ``network_mode() == "off"`` gate for
    an explicit-but-still-policy-governed network action (see netpolicy.py's
    module docstring: explicit user actions "still respect net_mode = off so
    one switch really does kill everything") - "ask"/"allow" need no per-call
    confirmation here, only the literal kill switch stops it.

    Best-effort like _prerelease_channel_enabled(): an unreadable config
    resolves to the safe direction (blocked), matching network_mode()'s own
    fail-closed behavior on a config-read failure (HON-2) rather than silently
    letting a network call through that a broken config cannot actually confirm
    was requested."""
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
    opens a socket - and, because it raises rather than returning a dict, it can
    never collapse into a false "up to date" the way a swallowed exception
    would (AGENTS.md rule 5).

    When the prerelease channel is opted into, appends ``?channel=prerelease`` to
    the request; the PROXY (not this client) decides which single candidate
    release to return for that channel (see tools/bugreport-proxy/worker.js's
    latestRelease) - is_newer() below just orders whatever ONE candidate comes
    back against the running version, unchanged either way."""
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
    # Whether "not newer" above actually means anything: is_newer() silently
    # returns False both for a genuine tie/older release AND for a tag it could
    # not parse as a version at all (see _version.comparable's docstring). No
    # candidate at all (latest falsy - "no releases published") is not this
    # function's problem to flag; the caller already branches on that first.
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
        # Absent is handled by the is_file() guard; this path is the corrupt or
        # unreadable manifest. Fall back to {} so classify() relies on its own
        # tree auto-detection (the conservative default). A bad manifest can only
        # LOSE a `needs` escalation hint, never weaken the update - the heavier
        # action is the escalation, so the fallback is the safe direction.
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
    # CHK-UPDATER-INTEGRITY (transport half): a code-update download must stay on
    # HTTPS end to end. Refuse a non-HTTPS endpoint, and refuse a redirect that
    # downgrades to http, so a MITM / redirect cannot serve the update in cleartext
    # (or pivot it). Redirects that STAY https are still followed (the proxy may
    # hand off to a release CDN). The signature half is separate and ENFORCED in
    # apply(): verify_signature() checks the build against the pinned Ed25519 key
    # (fail-closed) BEFORE it is extracted or run.
    if urllib.parse.urlparse(url).scheme != "https":
        raise LocalmError("refusing a non-HTTPS update download",
                          reason="the update endpoint must be https")

    # verified_urlopen (see localm/http_ssl.py): the download follows a 302 to the
    # GitHub release CDN and verifies both hops, native cert store first, certifi
    # as fallback. The https-only-redirect guard that used to be defined here is
    # now http_ssl.HttpsOnlyRedirect, installed by DEFAULT for every outbound
    # caller - this download was one of only two sites that had one.
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
        # Ahead of the URLError clause below (it is a URLError subclass): a
        # refused downgrade is a rejected attempt to serve the update in
        # cleartext, not a network problem, and must not be reported as one.
        raise LocalmError("the update download tried to downgrade to http",
                          reason=str(getattr(e, "reason", e)))
    except (urllib.error.URLError, OSError) as e:
        raise LocalmError("could not download the update", reason=str(getattr(e, "reason", e)))
    return dest


# ------------------------------ apply -----------------------------------

def _updates_dir() -> Path:
    from localm.config import home_dir
    d = home_dir() / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


# SEC-UPDATE-RACE: fallback-only threshold, used ONLY when a lock's holder PID
# could not be determined at all (an older-format lock, or a crash in the
# narrow window between mkdir and writing the pid file - see
# _apply_lock_is_stale). Generous on purpose, for the same reason it always
# was: a "runtime"-class apply re-provisions the native llama.cpp binaries
# (post_swap_command -> setup-llama), which can legitimately take several
# minutes on a slow connection. It is NOT the primary staleness signal - see
# _apply_lock_is_stale for why elapsed time alone cannot be trusted for that.
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

    An apply can hold this lock for a long time on a perfectly healthy run: it
    covers the DOWNLOAD (download() has no cap on TOTAL duration, only a
    per-socket-operation timeout, so a large build over a slow connection can
    legitimately take many minutes to fully arrive), and a "runtime"-class
    post-swap step is another real download on top of that. So elapsed time
    SINCE ACQUISITION cannot tell "still running, slowly" apart from "crashed
    without releasing the lock" - only whether the holder process is still
    alive can.

    Primary signal: the PID _apply_lock recorded at acquisition. If
    instances.pid_alive() confirms it is dead, the lock is stale regardless of
    age - reclaimed immediately rather than waiting out a timer. If the PID is
    alive, or pid_alive cannot tell (its own conservative default - see its
    docstring), the lock is NOT stale no matter how long it has been held:
    never falsely steal from a process that may still be genuinely working.

    Fallback ONLY when no PID was recorded at all: age against
    _APPLY_LOCK_STALE_S is the last resort, since there is no liveness signal
    to check in that case."""
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
def _apply_lock():
    """Cross-process, cross-thread single-flight guard for :func:`apply` (SEC-
    UPDATE-RACE, checkup 2026-08-11 item 7 - the only data-destroying finding in
    that run). ``mkdir`` is atomic, the same idiom this repo's own test-slot lock
    uses, so two concurrent callers - two browser tabs' ``POST /api/update/apply``
    (both landing in the SAME process via ``asyncio.to_thread``), or the CLI
    racing a live server in a SEPARATE process - cannot both proceed past this
    point.

    Without a lock, a second call's OWN backup step can run AFTER the first
    call's swap has already replaced the install, so the "pre-update" backup it
    writes actually contains the FIRST call's NEW build, not the true original.
    If that second call then needs to roll back, it restores the wrong state and
    the true pre-update install is gone for good.

    The SECOND caller fails FAST with a clear refusal rather than blocking: an
    apply can legitimately run for minutes, and a caller blocked that long is
    indistinguishable from a hang. A lock whose holder is confirmed gone (see
    :func:`_apply_lock_is_stale`) is reclaimed rather than stranding every
    future update forever after a crash."""
    import os
    lock_dir = _updates_dir() / "apply.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError:
        if _apply_lock_is_stale(lock_dir):
            # Reclaim an orphaned lock. If we lose a race to reclaim it (someone
            # else's mkdir wins first), our own mkdir below raises FileExistsError
            # again, and we report that exactly like a live lock - safe either way,
            # since that means a genuinely concurrent, non-stale caller now holds it.
            with contextlib.suppress(OSError):
                shutil.rmtree(lock_dir)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            from localm.bugreport import LocalmError
            raise LocalmError(
                "another update is already being applied",
                reason="wait for it to finish, then try again")
    # Record who holds it, so a future caller's staleness check can ask
    # instances.pid_alive() instead of guessing from elapsed time. Best-effort:
    # a failed write just means the next check falls back to age (see
    # _apply_lock_is_stale) - never the reason this apply itself fails.
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
    """A fresh, unique-per-call scratch directory for ONE apply() run (SEC-UPDATE-
    RACE, defense in depth alongside :func:`_apply_lock`): the download, staging
    extraction, and swap-time backup for this run never share a path with any
    other run's, so even if the lock were ever bypassed, two runs cannot corrupt
    each other's files on disk. uuid4 needs no coordination and will not collide
    in practice."""
    d = _updates_dir() / "runs" / uuid.uuid4().hex
    d.mkdir(parents=True)
    return d


def _promote_backup(run_backup_dir: Path, backup_dir: Path) -> None:
    """Move *run_backup_dir* (this run's own, just-written backup) to the stable
    *backup_dir* location :func:`rollback_last` reads, replacing whatever was
    there.

    Called ONLY after ``swap_with_backup`` has already succeeded, i.e.
    *run_backup_dir* is a complete, self-contained COPY of the pre-update install
    (``_apply_update.backup`` copies, never moves, the live tree) - relocating an
    already-independent copy cannot lose or corrupt it; this is not the "a backup
    you moved data out of is not a backup" hazard, because the live install was
    never the thing being moved.

    Best-effort: a failure here does not undo the update (the new build is
    already live and correct) but IS surfaced, never silently swallowed, since it
    degrades a later ``update --rollback`` (AGENTS.md rule 5)."""
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
    (:func:`_new_run_dir`) - defense in depth alongside the lock - and, once its
    swap has succeeded, moves that backup to the stable location
    :func:`rollback_last` reads (:func:`_promote_backup`). Without both, a second
    caller's "pre-update" backup could end up holding the FIRST caller's
    already-swapped NEW build instead of the true original, making the real
    pre-update install unrecoverable.

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
    backup_dir = updir / "backup"   # the STABLE location rollback_last() reads

    with _apply_lock():
        run_dir = _new_run_dir()
        zip_path, staging, run_backup_dir = (
            run_dir / "build.zip", run_dir / "staging", run_dir / "backup")

        try:
            download(asset_id, zip_path, opener=download_opener)
            # SIGNATURE GATE - authenticity, before verify_zip/extract/swap, i.e. before
            # any of the downloaded build's code can be extracted or executed. Fails closed.
            verify_signature(zip_path.read_bytes(), signature)
            au.verify_zip(zip_path)
            root = au.extract(zip_path, staging)
            vf = root / "VERSION"
            new_version = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
            # ANTI-ROLLBACK - a validly SIGNED but OLDER build must not be installable.
            _refuse_downgrade(new_version)
            klass = classify(root, target, read_manifest(root))
        except Exception:
            # Nothing precious was written yet (no backup exists for this run) -
            # safe to clean up unconditionally.
            shutil.rmtree(run_dir, ignore_errors=True)
            raise

        # A STALE applied_names.json from a PREVIOUS update would mis-describe THIS one: the
        # updates dir persists across updates and this file is only ever overwritten below,
        # so a failed write would leave the prior update's names and a later rollback would
        # then remove/restore the WRONG top-level set (stranding a file THIS build dropped -
        # silent install data loss reported as rolled_back:True). Remove it BEFORE the swap
        # so that, once backup/ is replaced, a stale manifest can never coexist with a fresh
        # backup: a failed write (or a crash) then degrades to the backup-dir listing
        # (correct for pre-existing names), never to stale data. Placed AFTER download/verify/
        # extract so an early abort there leaves the previous update's manifest intact.
        manifest_path = updir / "applied_names.json"
        try:
            manifest_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            _apply_warn("could not remove the previous update manifest %s: %s; if recording "
                        "the new one below also fails, a later rollback may use stale names",
                        manifest_path, e)

        # swap_with_backup backs up to run_backup_dir FIRST, then swaps - if the swap
        # itself fails it rolls back using run_backup_dir and re-raises. On failure we
        # deliberately do NOT touch run_dir: its error message may point a human at
        # run_backup_dir for manual recovery (if the rollback ALSO failed), and deleting
        # it here would destroy the one thing that message tells them to restore from.
        names = au.swap_with_backup(root, target, run_backup_dir)
        # This run's swap succeeded: promote its backup to the stable path BEFORE
        # anything downstream (the manifest write, the post-swap command's own
        # rollback) touches `backup_dir`, so both keep using the exact same fixed
        # path rollback_last() expects - unchanged from before this function grew
        # per-run scratch directories.
        _promote_backup(run_backup_dir, backup_dir)
        shutil.rmtree(run_dir, ignore_errors=True)   # zip/staging no longer needed

        # Record the FULL swap set (including brand-new top-level entries, which the backup
        # dir does NOT contain because they had nothing to back up) so a later manual
        # `update --rollback` removes them too, matching the pre-apply state. Without this,
        # rollback_last() sees only backed-up (pre-existing) names and would strand new files.
        try:
            import json
            manifest_path.write_text(json.dumps(sorted(names)), encoding="utf-8")
        except OSError as e:
            # Best-effort, but a failed write is now VISIBLE: the manifest was unlinked above,
            # so a later rollback falls back to the backup-dir listing (correct for pre-existing
            # names) and cannot remove brand-new top-level entries this update added. Say so.
            _apply_warn("could not record the update manifest %s: %s; a later "
                        "`update --rollback` will fall back to the backup listing and cannot "
                        "remove brand-new top-level entries this update added", manifest_path, e)

        cmd = au.post_swap_command(klass, backend=_installed_backend())
        if cmd:
            run = runner or (lambda c: subprocess.run(c, cwd=str(target)).returncode)

            def _rollback_or_raise(why):
                # If recovery ITSELF fails, say so loudly - never report a clean rollback
                # over a broken install (we do not hide problems).
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
    marker at all: a fresh install, or one predating the marker) - there is
    nothing to preserve in that case, so recommending is a first-time pick,
    not an override.

    THIS FUNCTION USED TO CALL recommended_install_backend() UNCONDITIONALLY.
    That was itself the fix for an earlier bug (#833: this function originally
    read the legacy Detection.recommended field, which can only ever be
    "vulkan" or "cpu" - see hwdetect.py). It was correct only as long as the
    recommendation policy never changed for hardware someone was already
    running, and it stopped being correct the moment it did: b8878c2b changed
    the NVIDIA-on-Linux recommendation from vulkan to cuda, which meant an
    NVIDIA-Linux user running vulkan - working fine - would have been silently
    re-provisioned onto cuda by their next "runtime"-class `localm update`,
    despite never asking for the switch.

    The maintainer's ruling: never override what a user is running, whether
    the choice was deliberate or just what an older default happened to
    install - detect and display it in Settings instead (the GUI /api/backend
    route), and at most offer a dismissable hint. So this now preserves
    unconditionally rather than trying to tell "deliberate" apart from
    "inherited default": the two need the same answer, which is to never
    change it out from under the user."""
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

# LM-DA-011: defaults for the detached post-restart health watchdog. TIMEOUT
# budgets the WHOLE restarted process coming back up and answering /whoami (which
# does NOT wait on model load) - importing localm's full chain plus unloading a
# previously-loaded engine plus rebinding the port can run into the tens of
# seconds on a cold disk / first import. 90s leaves comfortable margin without
# leaving a genuinely broken build undetected for long; this runs unattended in
# the background after the update button has already returned 200, so nothing
# user-facing is blocked waiting on it.
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
    back if it does not (LM-DA-011). Returns True iff the spawn was attempted and
    succeeded - NEVER raises: a watchdog that fails to spawn must not block or fail
    the restart itself (today there is no watchdog at all, so a failed spawn is a
    no-op, not a new regression). *popen* is injectable for tests, matching
    apply()'s download_opener/runner convention."""
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
        # So the watchdog's rollback call (on failure) re-derives the SAME data
        # home this server was actually using, via rollback_update.py's own
        # _detect_home() LOCALM_HOME-env precedence - no home-dir logic duplicated
        # here or in the watchdog itself.
        env["LOCALM_HOME"] = str(home)
        kwargs = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, env=env, close_fds=True)
        if sys.platform == "win32":
            # DETACHED_PROCESS (no console) + CREATE_NEW_PROCESS_GROUP so the
            # watchdog is not tied to this process's console/process group and
            # survives past the execv this call precedes. Literal fallback
            # constants let a test exercise this branch on any platform via a
            # monkeypatched sys.platform.
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
    """Best-effort WARNING for apply()/rollback bookkeeping (the manifest write/prune).
    A failure to LOG must never itself break or mask an update; when the logger is
    importable the real problem is surfaced through it."""
    try:
        from localm.debuglog import logger
        logger.warning(msg, *args)
    except Exception:
        pass


def rollback_last(*, installed=None) -> dict:
    """Restore the install from the most recent update backup. Returns
    ``{rolled_back, backup}``. Raises LocalmError when there is no backup."""
    from localm import _apply_update as au
    from localm.bugreport import LocalmError
    target = Path(installed) if installed else repo_root()
    updir = _updates_dir()
    backup_dir = updir / "backup"
    if not backup_dir.is_dir() or not any(backup_dir.iterdir()):
        raise LocalmError("no update backup to roll back to", reason=str(backup_dir))
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
            # never-written case): we can still roll back from the backup dir, but that
            # listing lacks the brand-new top-level entries the update ADDED, so those
            # will not be removed. Surface the degraded rollback rather than silently
            # doing a partial one (we do not hide problems).
            _apply_warn("applied_names.json at %s exists but is unreadable; new files "
                        "added by the update will not be removed by this rollback", manifest)
    if names is None:
        names = [p.name for p in backup_dir.iterdir()]
    au.rollback(backup_dir, target, names)
    return {"rolled_back": True, "backup": str(backup_dir)}
