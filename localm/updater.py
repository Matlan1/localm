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

import json as _json
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


def check(*, opener=None) -> dict:
    """Ask the proxy for the latest release and compare to the running version.

    Returns ``{current, latest, newer, notes, published_at, asset}``. *latest* is
    None when there are no releases. Raises :class:`~localm.bugreport.LocalmError`
    when not configured or the proxy fails - the caller decides whether to surface it
    (the manual command does) or swallow it (the startup check does)."""
    from localm import _proxy
    from localm.bugreport import LocalmError
    base, token = endpoint()
    if not base:
        raise LocalmError("the updater is not configured",
                          reason="set bugreport_upload_url (or update_url) to enable it")
    data = _proxy.request(base, "/update", token=token, opener=opener)
    current = _version.read_version()
    latest = data.get("version") if isinstance(data, dict) else None
    newer = bool(latest) and _version.is_newer(latest, current)
    return {
        "current": current,
        "latest": latest,
        "newer": newer,
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

    class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, rq, fp, code, msg, hdrs, newurl):
            if urllib.parse.urlparse(newurl).scheme != "https":
                raise LocalmError("the update download tried to downgrade to http",
                                  reason=f"blocked redirect to {newurl}")
            return super().redirect_request(rq, fp, code, msg, hdrs, newurl)

    # Verify against certifi, not the OS cert store (see localm/http_ssl.py): the
    # download follows a 302 to the GitHub release CDN, and a fresh Windows box
    # whose ROOT store has not cached that CA chain otherwise fails the hop with
    # CERTIFICATE_VERIFY_FAILED. The HTTPSHandler context applies to the CDN hop too.
    from localm.http_ssl import client_ssl_context
    built_opener = urllib.request.build_opener(
        _HttpsOnlyRedirect, urllib.request.HTTPSHandler(context=client_ssl_context()))
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with built_opener.open(req, timeout=timeout) as resp:
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
    except (urllib.error.URLError, OSError) as e:
        raise LocalmError("could not download the update", reason=str(getattr(e, "reason", e)))
    return dest


# ------------------------------ apply -----------------------------------

def _updates_dir() -> Path:
    from localm.config import home_dir
    d = home_dir() / "updates"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    zip_path, staging, backup_dir = updir / "build.zip", updir / "staging", updir / "backup"

    download(asset_id, zip_path, opener=download_opener)
    # SIGNATURE GATE - authenticity, before verify_zip/extract/swap, i.e. before any
    # of the downloaded build's code can be extracted or executed. Fails closed.
    verify_signature(zip_path.read_bytes(), signature)
    au.verify_zip(zip_path)
    root = au.extract(zip_path, staging)
    vf = root / "VERSION"
    new_version = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
    # ANTI-ROLLBACK - a validly SIGNED but OLDER build must not be installable.
    _refuse_downgrade(new_version)
    klass = classify(root, target, read_manifest(root))
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

    names = au.swap_with_backup(root, target, backup_dir)
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
    """Best-effort backend for a runtime re-provision: the hwdetect recommendation
    (the same universal-safe policy setup uses), defaulting to the vendor-neutral
    'vulkan'. The install manifest does not record the chosen backend, so this is a
    detection, not a lookup."""
    try:
        from localm import hwdetect
        return hwdetect.detect().recommended or "vulkan"
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
