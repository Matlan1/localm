# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-updater: check the localm proxy for a newer release and (on explicit user
action ONLY) apply it.

Checking is automatic (a quiet startup check + a manual button/command); APPLYING is
never automatic - it runs only from ``localm update`` (confirmed) or the GUI "Update
now" button. Most updates are code-only and need just a file swap + reboot (the
install is editable, so new source is live on restart); ``deps``/``runtime`` escalate
only when they actually change. The risky file-swap itself lives in
``localm/_apply_update.py`` (a detached helper with backup + health-checked rollback).
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
# Applying an update EXTRACTS and EXECUTES the downloaded build (it swaps into the
# editable install and reboots into the new code), so a forged build is arbitrary
# code execution. The transport is HTTPS-pinned (see download()), but transport
# alone does not prove AUTHENTICITY - a compromised proxy / release / token could
# still serve a malicious, well-formed build, and a checksum from the same channel
# proves nothing (the attacker controls both). So each release build.zip is signed
# with an Ed25519 private key the maintainer keeps OFFLINE, and apply() verifies the
# signature against the PUBLIC key(s) PINNED here before anything is extracted.
#
# This is a tuple of hex-encoded 32-byte Ed25519 PUBLIC keys. It is EMPTY by
# default, which makes the updater FAIL CLOSED: with no pinned key it refuses to
# apply any update (an unverifiable build must never be installed - AGENTS.md rule
# 5: a security step that cannot do its job fails, it does not report success). To
# enable self-update the maintainer:
#   1. generates a keypair offline:  python scripts/sign_release.py --generate-key
#   2. keeps the PRIVATE key offline (never in the repo / proxy / CI),
#   3. pastes the printed PUBLIC key hex into this tuple,
#   4. signs each release:  python scripts/sign_release.py sign build.zip --key ...
#      and serves that signature from the update proxy's /update JSON.
# It is a LIST so a new key can be added (rotation) before an old one is retired,
# without a flag day. Embedding a PUBLIC key in source is correct: it is public
# data, and pinning it in the shipped code is exactly what an attacker cannot forge.
_UPDATE_PUBKEYS: tuple = ()


def _load_update_pubkeys() -> list:
    """The pinned Ed25519 public keys as verifier objects; [] when none/invalid.

    A malformed pinned key is skipped (so a second, valid key still works) rather
    than crashing the check - but if NO valid key results, verify_signature below
    fails closed, so a bad pin can never weaken verification into a silent pass."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    keys = []
    for hexkey in _UPDATE_PUBKEYS:
        try:
            keys.append(Ed25519PublicKey.from_public_bytes(bytes.fromhex(str(hexkey).strip())))
        except Exception:
            continue
    return keys


def verify_signature(data: bytes, signature_b64) -> None:
    """Verify *signature_b64* (base64 Ed25519 signature) over *data* against the
    pinned public key(s). Raises :class:`~localm.bugreport.LocalmError` unless a
    pinned key validates it.

    FAIL CLOSED: refuses when no key is pinned OR no signature is supplied OR the
    signature does not match - applying an update executes its code, so an
    unverifiable build must never be installed."""
    import base64

    from cryptography.exceptions import InvalidSignature
    from localm.bugreport import LocalmError
    keys = _load_update_pubkeys()
    if not keys:
        raise LocalmError(
            "refusing to apply an update: no signing key is configured",
            reason="update signature verification is not set up (no pinned public key)")
    if not signature_b64:
        raise LocalmError(
            "refusing to apply an unsigned update",
            reason="the release did not include a signature")
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
    # hand off to a release CDN). The signature/checksum half is separate and
    # pending the release-signing scheme.
    if urllib.parse.urlparse(url).scheme != "https":
        raise LocalmError("refusing a non-HTTPS update download",
                          reason="the update endpoint must be https")

    class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, rq, fp, code, msg, hdrs, newurl):
            if urllib.parse.urlparse(newurl).scheme != "https":
                raise LocalmError("the update download tried to downgrade to http",
                                  reason=f"blocked redirect to {newurl}")
            return super().redirect_request(rq, fp, code, msg, hdrs, newurl)

    built_opener = urllib.request.build_opener(_HttpsOnlyRedirect)
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
    names = au.swap_with_backup(root, target, backup_dir)

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


def rollback_last(*, installed=None) -> dict:
    """Restore the install from the most recent update backup. Returns
    ``{rolled_back, backup}``. Raises LocalmError when there is no backup."""
    from localm import _apply_update as au
    from localm.bugreport import LocalmError
    target = Path(installed) if installed else repo_root()
    backup_dir = _updates_dir() / "backup"
    if not backup_dir.is_dir() or not any(backup_dir.iterdir()):
        raise LocalmError("no update backup to roll back to", reason=str(backup_dir))
    names = [p.name for p in backup_dir.iterdir()]
    au.rollback(backup_dir, target, names)
    return {"rolled_back": True, "backup": str(backup_dir)}
