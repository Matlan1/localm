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
#  Update signing trust
# ---------------------------------------------------------------------------
# Each release build.zip is signed by the keys of the trusted RELEASE set. The
# RECOVERY set never signs a build; its keys only sign a replacement key set.
# The release's `.sig` asset is a JSON envelope:
#
#   {"format": "localm-update-sig/2",
#    "signatures": [{"key": "<public key hex>", "sig": "<base64 Ed25519 over the zip>"}],
#    "keyset": {"seq": 2, "release": {"threshold": 1, "keys": [...]},
#               "recovery": {"threshold": 2, "keys": [...]},
#               "signatures": [{"key": ..., "sig": <base64 over canonical_keyset_bytes()>}]}}
#
# "keyset" is optional. apply() verifies the envelope BEFORE anything is
# extracted, and saves an accepted key set to <data dir>/updates/trust.json only
# after the whole update has verified.
_SIG_FORMAT = "localm-update-sig/2"
_SIG_MAX_CHARS = 4096
_TRUST_FILE = "trust.json"

_PINNED_KEYSET: dict = {
    "seq": 1,
    "release": {
        "threshold": 1,
        "keys": [
            "4faca1439ff49ca9269580061f17d1847e079c35c82e8372b221bc3abba6db40",
        ],
    },
    "recovery": {
        "threshold": 2,
        "keys": [
            "d5357ae966ce0eecc687f40ca3a68b93c4d8e1c246b9af7006ee03905f5507ee",
            "090193021f42f8514b60676b38c0365034e3bf312d5f4946e1434e328f4a78e8",
            "1a9a3fc808a1e2ab207b71fd4b6eeb09860b5ef17281d6f13c73dcf66f73f334",
        ],
    },
}

# Public keys that must never be trusted again, in any role and from any source
# (built in, saved trust file, or a key set offered by a signature envelope).
_REVOKED_KEYS: frozenset = frozenset({
    "3501b23eb1ab6ec245b5e9d9c6a70b522ca89d6250d13060a9642ce1c8868ecf",
})

_HEX_DIGITS = frozenset("0123456789abcdef")


class _BadSignature(ValueError):
    """An envelope or key set that is structurally invalid."""


def _strict_json(text: str):
    """``json.loads`` that refuses duplicate object fields and NaN/Infinity."""
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise _BadSignature(f"duplicate field {k!r}")
            out[k] = v
        return out

    def constant(name):
        raise _BadSignature(f"{name} is not allowed")

    return _json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _check_fields(obj, where: str, required: set, optional: frozenset = frozenset()) -> None:
    if not isinstance(obj, dict):
        raise _BadSignature(f"{where} is not an object")
    missing = required - obj.keys()
    if missing:
        raise _BadSignature(f"{where} lacks {', '.join(sorted(missing))}")
    extra = obj.keys() - required - optional
    if extra:
        raise _BadSignature(f"{where} has unknown field(s) {', '.join(sorted(map(str, extra)))}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _public_key(hexkey):
    """The Ed25519 public key for a 64-character lowercase hex string."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not (isinstance(hexkey, str) and len(hexkey) == 64 and set(hexkey) <= _HEX_DIGITS):
        raise _BadSignature(f"{str(hexkey)[:20]!r} is not a 64-character lowercase hex key")
    try:
        return Ed25519PublicKey.from_public_bytes(bytes.fromhex(hexkey))
    except ValueError:
        raise _BadSignature(f"{hexkey[:16]}... is not an Ed25519 public key") from None


def _parse_role(obj, where: str, min_threshold: int) -> dict:
    _check_fields(obj, where, {"threshold", "keys"})
    threshold, keys = obj["threshold"], obj["keys"]
    if not _is_int(threshold) or threshold < min_threshold:
        raise _BadSignature(f"{where} threshold must be an integer of at least {min_threshold}")
    if not isinstance(keys, list) or not keys:
        raise _BadSignature(f"{where} keys must be a non-empty list")
    for k in keys:
        _public_key(k)
        if k in _REVOKED_KEYS:
            raise _BadSignature(f"{where} contains the revoked key {k[:8]}")
    if len(set(keys)) != len(keys):
        raise _BadSignature(f"{where} keys are not distinct")
    if len(keys) < threshold:
        raise _BadSignature(f"{where} has fewer keys than its threshold")
    return {"threshold": threshold, "keys": list(keys)}


def _parse_signatures(obj, where: str) -> list:
    """Validated ``[{"key", "sig"}]`` entries; a "sig" must be base64 of 64 bytes."""
    import base64
    if not isinstance(obj, list):
        raise _BadSignature(f"{where} is not a list")
    out = []
    for entry in obj:
        _check_fields(entry, f"an entry of {where}", {"key", "sig"})
        if not isinstance(entry["key"], str) or not isinstance(entry["sig"], str):
            raise _BadSignature(f"an entry of {where} is not a pair of strings")
        try:
            raw = base64.b64decode(entry["sig"], validate=True)
        except ValueError:
            raise _BadSignature(f"a signature in {where} is not valid base64") from None
        if len(raw) != 64:
            raise _BadSignature(f"a signature in {where} is not 64 bytes")
        out.append({"key": entry["key"], "sig": entry["sig"]})
    return out


def _parse_keyset(obj, *, signed: bool) -> dict:
    """Validate a key set: ``seq`` >= 1, release threshold >= 1, recovery threshold
    >= 2, keys distinct valid Ed25519 hex, no key in both roles. *signed* requires
    (and keeps) its "signatures" list. Raises _BadSignature."""
    _check_fields(obj, "the key set",
                  {"seq", "release", "recovery"} | ({"signatures"} if signed else set()))
    if not _is_int(obj["seq"]) or obj["seq"] < 1:
        raise _BadSignature("the key set seq must be a positive integer")
    ks = {
        "seq": obj["seq"],
        "release": _parse_role(obj["release"], "the release set", 1),
        "recovery": _parse_role(obj["recovery"], "the recovery set", 2),
    }
    if set(ks["release"]["keys"]) & set(ks["recovery"]["keys"]):
        raise _BadSignature("a key is in both the release and the recovery set")
    if signed:
        ks["signatures"] = _parse_signatures(obj["signatures"], "the key set signatures")
    return ks


def canonical_keyset_bytes(keyset: dict) -> bytes:
    """The bytes recovery keys sign for *keyset*: its seq, release and recovery
    fields (never its signatures) as compact JSON with sorted keys."""
    body = {k: keyset[k] for k in ("seq", "release", "recovery")}
    return _json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _count_signers(signatures: list, trusted_keys: list, message: bytes) -> int:
    """How many DISTINCT keys of *trusted_keys* validly signed *message*. Unknown
    keys, repeats and invalid signatures add nothing."""
    import base64

    from cryptography.exceptions import InvalidSignature
    trusted, valid = set(trusted_keys), set()
    for entry in signatures:
        key = entry["key"]
        if key not in trusted or key in valid:
            continue
        try:
            _public_key(key).verify(base64.b64decode(entry["sig"]), message)
        except InvalidSignature:
            continue
        valid.add(key)
    return len(valid)


def _trust_path() -> Path:
    return _updates_dir(create=False) / _TRUST_FILE


def _load_trusted() -> dict:
    """The trusted key set: the one saved in the data dir when its seq is higher
    than the pinned one's, else the pinned one. A saved file that cannot be read,
    does not validate, or has the pinned seq with different keys raises
    :class:`~localm.bugreport.LocalmError` naming the file; it is never replaced by
    the pinned set."""
    from localm.bugreport import LocalmError
    pinned = _parse_keyset(_PINNED_KEYSET, signed=False)
    path = _trust_path()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return pinned
    except OSError as e:
        raise LocalmError("the saved update signing keys could not be read",
                          reason=f"{path}: {e}") from None
    try:
        saved = _parse_keyset(_strict_json(raw.decode("utf-8")), signed=True)
    except (ValueError, RecursionError) as e:
        raise LocalmError(
            "the saved update signing keys are corrupt",
            reason=f"{path}: {e}. Restore that file, or reinstall localm") from None
    if saved["seq"] == pinned["seq"] and (
            canonical_keyset_bytes(saved) != canonical_keyset_bytes(pinned)):
        raise LocalmError(
            "the saved update signing keys conflict with the built-in ones",
            reason=f"{path}: key set {saved['seq']} differs from the built-in key set "
                   f"{pinned['seq']}. Restore that file, or reinstall localm")
    return saved if saved["seq"] > pinned["seq"] else pinned


def _save_keyset(keyset: dict) -> None:
    """Atomically write the signed *keyset* to the data dir's trust file."""
    import os

    from localm.bugreport import LocalmError
    path = _updates_dir() / _TRUST_FILE
    tmp = path.with_name(f".{_TRUST_FILE}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(_json.dumps(keyset, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise LocalmError("could not save the update's new signing keys",
                          reason=f"{path}: {e}") from None


def verify_signature(data: bytes, signature, *, trusted: Optional[dict] = None):
    """Verify the signature envelope *signature* (the release's ``.sig`` text)
    over the build bytes *data*. Fails closed: every error raises.

    *trusted* is the key set to verify against; ``None`` loads it with
    :func:`_load_trusted`. An envelope "keyset" with a higher seq replaces it
    only when signed by at least the trusted recovery threshold of distinct
    trusted recovery keys; an equal seq must match it exactly; a lower seq is
    refused. The build then needs valid signatures from at least the release
    threshold of distinct keys of the (possibly replaced) release set.

    Returns the accepted replacement key set (with its signatures), or None when
    the envelope does not replace the trusted one. Never writes anything; the
    caller saves a returned key set with :func:`_save_keyset`.

    Raises :class:`~localm.bugreport.LocalmError` on a refusal, and
    ``ImportError`` when ``cryptography`` is not installed."""
    from localm.bugreport import LocalmError

    def refuse(reason):
        return LocalmError("refusing to apply the update: its signature does not verify",
                           reason=reason)

    if not isinstance(signature, str) or not signature.strip():
        raise LocalmError("refusing to apply an unsigned update",
                          reason="the release includes no signature")
    if len(signature) > _SIG_MAX_CHARS:
        raise refuse(f"the signature is longer than {_SIG_MAX_CHARS} characters")
    trusted = _load_trusted() if trusted is None else _parse_keyset(
        {k: trusted[k] for k in ("seq", "release", "recovery")}, signed=False)
    try:
        try:
            env = _strict_json(signature)
        except _BadSignature:
            raise
        except (ValueError, RecursionError):
            raise _BadSignature(
                f"it is not a {_SIG_FORMAT} signature; a release signed for localm "
                "0.2.0 or earlier is not accepted") from None
        _check_fields(env, "the signature", {"format", "signatures"}, frozenset({"keyset"}))
        if env["format"] != _SIG_FORMAT:
            raise _BadSignature(f"unsupported signature format {str(env['format'])[:40]!r}")
        build_sigs = _parse_signatures(env["signatures"], "the build signatures")
        offered = _parse_keyset(env["keyset"], signed=True) if "keyset" in env else None
    except _BadSignature as e:
        raise refuse(str(e)) from None

    effective, accepted = trusted, None
    if offered is not None:
        if offered["seq"] < trusted["seq"]:
            raise refuse(f"it carries key set {offered['seq']}, older than the trusted "
                         f"key set {trusted['seq']}")
        if offered["seq"] == trusted["seq"]:
            if canonical_keyset_bytes(offered) != canonical_keyset_bytes(trusted):
                raise refuse(f"it carries a different key set {offered['seq']} than the "
                             "trusted one")
        else:
            recovery = trusted["recovery"]
            if _count_signers(offered["signatures"], recovery["keys"],
                              canonical_keyset_bytes(offered)) < recovery["threshold"]:
                raise refuse(f"its new key set is not signed by {recovery['threshold']} "
                             "trusted recovery keys")
            effective = accepted = offered
    release = effective["release"]
    if _count_signers(build_sigs, release["keys"], data) < release["threshold"]:
        raise refuse("the build is not signed by a trusted release key; it may be "
                     "tampered with or signed with an unknown key")
    return accepted


# The oldest release signed by the current release key. Releases before it carry
# only signatures in a format this install does not accept.
_CURRENT_KEY_FLOOR = "0.3.0"


class UnverifiedBuild(Exception):
    """A selected build whose signature does not verify against the trusted key
    set, raised by :func:`apply` before anything is extracted when the caller has
    not passed ``allow_unverified``. Carries the requested *version*, a
    user-facing *reason*, and the build's *sha256*, which the caller passes back as
    ``expect_sha256`` to install exactly that build after the user agreed."""

    def __init__(self, version: str, reason: str, sha256: str):
        super().__init__(f"the build of {version} is not verified: {reason}")
        self.version = version
        self.reason = reason
        self.sha256 = sha256


def assess_signature(data: bytes, signature) -> dict:
    """Verify *signature* over *data* with :func:`verify_signature`, but report a
    failure instead of raising it.

    Returns ``{"verified": True, "reason": None, "keyset": <replacement key set or
    None>}`` when it verifies, else ``{"verified": False, "reason": <user-facing
    text>, "keyset": None}``. The reason says the build is unsigned, or why its
    signature does not verify, or why the check could not run. Never raises for
    the signature itself."""
    from localm.bugreport import LocalmError
    if not isinstance(signature, str) or not signature.strip():
        return {"verified": False, "keyset": None,
                "reason": "the release has no signature, so nothing shows who built it"}
    try:
        return {"verified": True, "reason": None,
                "keyset": verify_signature(data, signature)}
    except ImportError:
        return {"verified": False, "keyset": None,
                "reason": "the signature could not be checked: the cryptography "
                          "package is not installed"}
    except LocalmError as e:
        return {"verified": False, "keyset": None,
                "reason": f"{e.reason or e.summary}. It may be signed with an unknown "
                          "or retired key, or altered after it was signed"}


def _check_selected_version(new_version: str, requested: str) -> None:
    """Raise unless the build's VERSION *new_version* is the *requested* release.

    A selected install may go to any version, older ones included, but only the
    one asked for: a build carrying a different VERSION is refused."""
    from localm.bugreport import LocalmError
    if not new_version:
        raise LocalmError(f"refusing to install {requested}: the build has no VERSION",
                          reason="cannot confirm it is the version you chose")
    if _version.normalize(new_version) != _version.normalize(requested):
        raise LocalmError(
            f"refusing to install the build served for {requested}: it is {new_version}",
            reason="the update server sent a different version than the one you chose")


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


def is_site_packages_install(root=None) -> bool:
    """True when localm was pip-installed into a site-packages tree.

    :func:`apply` swaps a whole release tree into :func:`repo_root`. For a pip
    install that root IS site-packages, shared with every other installed
    package, so the swap would write a release tree over unrelated packages and
    leave pip with no record of what is on disk. Such an install is updated
    with pip, not by localm.
    """
    import sysconfig
    target = Path(root or repo_root()).resolve()
    try:
        paths = sysconfig.get_paths()
    except Exception:
        return False
    for key in ("purelib", "platlib"):
        raw = paths.get(key)
        if not raw:
            continue
        try:
            if target == Path(raw).resolve():
                return True
        except OSError:
            continue
    return False


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

    The request carries ``client=<running version>``; the proxy offers a request
    without it (a client from 0.3.1 or earlier) no release newer than its legacy
    ceiling. When the prerelease channel is opted into, it also carries
    ``channel=prerelease``; the PROXY (not this client) decides which single
    candidate release to return for that channel, and is_newer() below orders that
    one candidate against the running version."""
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
    import urllib.parse
    current = _version.read_version()
    query = {"client": current or "unknown"}
    if _prerelease_channel_enabled():
        query["channel"] = "prerelease"
    path = "/update?" + urllib.parse.urlencode(query)
    data = _proxy.request(base, path, token=token, opener=opener)
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
        # The proxy serves the release's signature envelope text alongside the
        # asset; apply() verifies the downloaded build with verify_signature().
        "signature": data.get("signature") if isinstance(data, dict) else None,
    }


# ---------------------- choosing a version to install ---------------------

def _selection_request(path: str, params: dict, *, opener=None) -> dict:
    """GET *path* with *params* on the update proxy, under the same configuration,
    network-policy and prerelease-channel rules as :func:`check`. A 404 means the
    proxy predates version selection and raises a LocalmError saying so."""
    import urllib.parse

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
    query = dict(params)
    if _prerelease_channel_enabled():
        query["channel"] = "prerelease"
    qs = urllib.parse.urlencode(query)
    try:
        data = _proxy.request(base, path + (f"?{qs}" if qs else ""), token=token,
                              opener=opener)
    except LocalmError as e:
        if str(e.reason).startswith("HTTP 404"):
            raise LocalmError(
                "the update server does not offer a choice of versions yet",
                reason="it needs the proxy version that serves /update/releases") from None
        raise
    if not isinstance(data, dict):
        raise LocalmError("the update server sent an unexpected reply", reason=str(data)[:200])
    return data


def _direction(version: str, current: str) -> str:
    """``"newer"``, ``"older"``, ``"same"`` or ``"unknown"`` (not comparable) for
    *version* relative to the running *current* version."""
    if not _version.comparable(version, current):
        return "unknown"
    if _version.is_newer(version, current):
        return "newer"
    if _version.is_newer(current, version):
        return "older"
    return "same"


def list_versions(*, opener=None) -> dict:
    """The releases a user can choose to install, newest first.

    Returns ``{current, releases}``; each release is ``{version, name,
    published_at, prerelease, notes, asset, signed, direction, installable,
    reason, warnings}``. ``direction`` is from :func:`_direction`. ``installable``
    is False only when the release has no build attached (``reason`` says so).
    ``warnings`` lists what the user should know before choosing it: a downgrade,
    a reinstall of the running version, an unrecognised version format, a release
    with no signature, or one from before ``_CURRENT_KEY_FLOOR`` (a signature
    format this version does not accept). The signature itself is checked when the build is installed, not
    here.

    Raises :class:`~localm.bugreport.LocalmError` like :func:`check`, and when the
    proxy has no version list."""
    data = _selection_request("/update/releases", {}, opener=opener)
    current = _version.read_version()
    out = []
    for rel in data.get("releases") or []:
        if not isinstance(rel, dict) or not isinstance(rel.get("version"), str):
            continue
        version = rel["version"]
        asset = rel.get("asset") if isinstance(rel.get("asset"), dict) else None
        signed = rel.get("signed") is True
        direction = _direction(version, current)
        warnings = []
        if direction == "older":
            warnings.append(f"older than the installed {current}: this is a downgrade")
        elif direction == "same":
            warnings.append("this is the installed version: it would be reinstalled")
        elif direction == "unknown":
            warnings.append("its version format is not recognised, so it cannot be "
                            "ordered against the installed one")
        pre_floor = (_version.comparable(version, _CURRENT_KEY_FLOOR)
                     and _version.is_newer(_CURRENT_KEY_FLOOR, version))
        if not signed:
            warnings.append("the release has no signature")
        elif pre_floor:
            warnings.append(f"released before {_CURRENT_KEY_FLOOR}: its signature is in a "
                            "format this version does not accept, so it will not verify")
        installable = bool(asset and asset.get("id"))
        out.append({
            "version": version, "name": rel.get("name"),
            "published_at": rel.get("published_at"),
            "prerelease": rel.get("prerelease") is True,
            "notes": str(rel.get("notes") or "")[:500],
            "asset": asset, "signed": signed, "direction": direction,
            "installable": installable,
            "reason": None if installable else "the release has no build attached",
            "warnings": warnings,
        })
    return {"current": current, "releases": out}


def fetch_release(version: str, *, opener=None) -> dict:
    """The proxy's meta for the release tagged *version* (with or without a leading
    ``v``): ``{version, notes, published_at, prerelease, asset, signature}``, the
    signature being the release's localm-update-sig/2 envelope when it has one.
    Raises :class:`~localm.bugreport.LocalmError` when there is no such release or
    the proxy answers with a different version than *version*, and like
    :func:`list_versions` otherwise."""
    from localm.bugreport import LocalmError
    want = str(version or "").strip()
    if not want:
        raise LocalmError("no version given", reason="name the version to install")
    data = _selection_request("/update/release", {"version": want}, opener=opener)
    if not data.get("found") or not data.get("version"):
        raise LocalmError(f"there is no release {want}",
                          reason="localm update --list shows the releases you can install")
    if _version.normalize(str(data["version"])) != _version.normalize(want):
        raise LocalmError(
            f"the update server answered {str(data['version'])[:40]} for {want}",
            reason="it sent a different release than the one you chose; nothing was installed")
    return {
        "version": data["version"],
        "notes": str(data.get("notes") or ""),
        "published_at": data.get("published_at"),
        "prerelease": data.get("prerelease") is True,
        "asset": data.get("asset") if isinstance(data.get("asset"), dict) else None,
        "signature": data.get("signature"),
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
        raise LocalmError("bad asset id", reason=str(asset_id)) from None
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
        raise LocalmError("the update download failed", reason=f"HTTP {e.code}") from None
    except RedirectDowngradeRefused as e:
        # Must stay ahead of the URLError clause below, which it subclasses.
        raise LocalmError("the update download tried to downgrade to http",
                          reason=str(getattr(e, "reason", e))) from None
    except (urllib.error.URLError, OSError) as e:
        raise LocalmError("could not download the update", reason=str(getattr(e, "reason", e))) from None
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
                reason="wait for it to finish, then try again") from None
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

    The native runtime snapshot and the components snapshot of the update whose
    backup this replaces (``_apply_update.runtime_snapshot_dir``,
    ``updater_components.snapshot_dir``) are removed with it.

    Best-effort: a failure does not undo the update but IS surfaced as a warning,
    since it degrades a later ``update --rollback``."""
    from localm import _apply_update as au
    try:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(run_backup_dir), str(backup_dir))
    except OSError as e:
        _apply_warn("update applied, but could not move this run's backup from %s "
                    "to %s: %s; a later `update --rollback` may not find it",
                    run_backup_dir, backup_dir, e)
    snapshot = au.runtime_snapshot_dir(backup_dir)
    try:
        if snapshot.exists():
            shutil.rmtree(snapshot)
    except OSError as e:
        _apply_warn("update applied, but could not remove the native runtime kept by the "
                    "previous update at %s: %s; a later `update --rollback` may put that "
                    "older runtime back", snapshot, e)
    from localm import updater_components
    updater_components.discard(updater_components.snapshot_dir(backup_dir))


# Data dir entries an older build may read without the fields a newer one added,
# and write back without them.
_DOWNGRADE_DATA = ("config.json", "registry.json", "checkpoints", "memory", "plugins",
                   "coder-hook-trust.json")


def _backup_data_for_downgrade(new_version: str) -> Path:
    """Copy the data dir entries in ``_DOWNGRADE_DATA`` that exist to
    ``<data dir>/updates/data-before-<new_version>-<stamp>/`` and return that
    directory. Raises :class:`~localm.bugreport.LocalmError` when a copy fails, so
    the downgrade does not go ahead without it."""
    from localm.bugreport import LocalmError
    from localm.config import home_dir
    home = home_dir()
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in new_version)[:40]
    dest = (_updates_dir() / f"data-before-{safe}-{time.strftime('%Y%m%d-%H%M%S')}"
            f"-{uuid.uuid4().hex[:8]}")
    try:
        dest.mkdir(parents=True)
    except OSError as e:
        raise LocalmError("could not back up your data before the downgrade",
                          reason=f"{dest}: {e}") from None
    try:
        for name in _DOWNGRADE_DATA:
            src = home / name
            if src.is_dir():
                shutil.copytree(src, dest / name, symlinks=True)
            elif src.is_file():
                shutil.copy2(src, dest / name)
    except OSError as e:
        shutil.rmtree(dest, ignore_errors=True)
        raise LocalmError("could not back up your data before the downgrade",
                          reason=f"{dest}: {e}") from None
    return dest


def _carry_trust_forward() -> None:
    """Save the key set this install trusts to the data dir's trust file when the
    file holds an older one (or none) and the trusted set is past the first, so an
    older build installed next reads it and can verify the releases this one can."""
    trusted = _load_trusted()
    if trusted["seq"] <= 1:
        return
    try:
        saved = _parse_keyset(_strict_json(_trust_path().read_text(encoding="utf-8")),
                              signed=True)
    except FileNotFoundError:
        saved = None
    if saved is not None and saved["seq"] >= trusted["seq"]:
        return
    _save_keyset({**{k: trusted[k] for k in ("seq", "release", "recovery")},
                  "signatures": trusted.get("signatures", [])})


_BRIDGE_MODULE = "_update_trust_bridge.py"


def _needs_trust_bridge(root: Path) -> bool:
    """True when the build at *root* has an updater that cannot verify a
    localm-update-sig/2 envelope (localm before 0.3.0) or does not refuse the
    revoked keys (localm before 0.3.2)."""
    try:
        text = (root / "localm" / "updater.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return _SIG_FORMAT not in text or "_REVOKED_KEYS" not in text


def _install_trust_bridge(root: Path) -> None:
    """Make the staged build at *root* verify its updates the way this build does.

    Writes this module's source to ``localm/_update_trust_bridge.py`` in *root*
    with ``_PINNED_KEYSET`` set to the key set trusted now, and appends to the
    staged ``localm/updater.py`` a rebinding of its ``verify_signature`` to the
    bridge's, which also saves a replacement key set an update carries. Raises
    :class:`~localm.bugreport.LocalmError` when a write fails."""
    from localm.bugreport import LocalmError
    trusted = _load_trusted()
    pinned = {k: trusted[k] for k in ("seq", "release", "recovery")}
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except OSError as e:
        raise LocalmError("could not prepare the older version to verify later updates",
                          reason=str(e)) from None
    bridge = (source.rstrip("\n")
              + "\n\n\n_PINNED_KEYSET = " + _json.dumps(pinned, sort_keys=True) + "\n")
    stanza = (
        "\n\n# Verifies updates with the update signing keys of the localm build that\n"
        "# installed this version (localm/_update_trust_bridge.py). A trusted= key\n"
        "# set passed in is not used: the bridge verifies against its own.\n"
        "def _localm_bridged_verify_signature(data, signature_b64=None, **_unused):\n"
        "    from localm import _update_trust_bridge as bridge\n"
        "    accepted = bridge.verify_signature(data, signature_b64)\n"
        "    if accepted is not None:\n"
        "        bridge._save_keyset(accepted)\n"
        "\n\n"
        "verify_signature = _localm_bridged_verify_signature\n")
    try:
        (root / "localm" / _BRIDGE_MODULE).write_text(bridge, encoding="utf-8")
        with open(root / "localm" / "updater.py", "a", encoding="utf-8") as f:
            f.write(stanza)
    except OSError as e:
        raise LocalmError("could not prepare the older version to verify later updates",
                          reason=str(e)) from None


def _needs_uv_bridge(root: Path) -> bool:
    """True when the build at *root* runs its dependency step with a ``uv``
    looked up on PATH (localm 0.3.1 and earlier)."""
    try:
        text = (root / "localm" / "_apply_update.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return 'return ["uv", "pip"' in text


def _install_uv_bridge(root: Path, uv: str, target: Path) -> None:
    """Make the staged build at *root* run the dependency step of its later
    updates with *uv* instead of a ``uv`` looked up on PATH.

    Appends to the staged ``localm/updater.py`` a rebinding of
    ``_apply_update.post_swap_command`` that replaces a bare ``uv`` first argument
    with *uv*: relative to the install root when *uv* is inside the install
    *target*, else the absolute path. Raises :class:`~localm.bugreport.LocalmError`
    when the write fails."""
    import os

    from localm.bugreport import LocalmError
    uv_path = Path(uv)
    try:
        rel = uv_path.resolve().relative_to(Path(target).resolve())
        uv_expr = ("_localm_os.path.join(_localm_os.path.dirname(_localm_os.path.dirname("
                   "_localm_os.path.abspath(__file__))), " + repr(str(rel)) + ")")
    except ValueError:
        uv_expr = repr(os.path.abspath(str(uv_path)))
    stanza = (
        "\n\n# Runs the dependency step of later updates with the uv the localm build\n"
        "# that installed this version found, instead of a uv looked up on PATH.\n"
        "import os as _localm_os\n"
        "from localm import _apply_update as _localm_apply_update\n"
        "_localm_unbridged_post_swap_command = _localm_apply_update.post_swap_command\n"
        "\n\n"
        "def _localm_bridged_post_swap_command(*args, **kwargs):\n"
        "    cmd = _localm_unbridged_post_swap_command(*args, **kwargs)\n"
        "    if cmd and cmd[0] == \"uv\":\n"
        f"        cmd = [{uv_expr}] + list(cmd[1:])\n"
        "    return cmd\n"
        "\n\n"
        "_localm_apply_update.post_swap_command = _localm_bridged_post_swap_command\n")
    try:
        with open(root / "localm" / "updater.py", "a", encoding="utf-8") as f:
            f.write(stanza)
    except OSError as e:
        raise LocalmError("could not prepare the older version to install later updates",
                          reason=str(e)) from None


def apply(asset_id, *, signature=None, target_version=None, allow_unverified=False,
          expect_sha256=None, installed=None, download_opener=None,
          runner=None, defer_runtime=False, say=None) -> dict:
    """Download, VERIFY (signature + anti-rollback), and swap a build into the
    install, then run the class's deps/runtime step. Returns
    ``{applied, version, klass, backup, restart_needed, verified, verify_reason,
    downgrade, data_backup, runtime, runtime_note, components, components_note,
    kept_for_updates}``.

    THE NATIVE RUNTIME. When the build pins a different llama.cpp build for the
    installed backend than the one installed (``updater_runtime.plan``), the
    class is ``runtime`` and a pending re-provision is recorded. The code and
    the runtime are updated both or neither: before the swap, this install's own
    processes that hold the runtime are stopped
    (``updater_runtime.stop_own_holders``; with *defer_runtime* only checked),
    and a holder that is not this install's refuses the update with nothing
    changed. Without *defer_runtime* the re-provision is carried out right after
    the swap (``updater_runtime.finish_pending``); when it fails, the previous
    runtime is put back, the code swap is rolled back and LocalmError is
    raised. With it (the server, which still runs its models) the caller's
    restart carries it out once its models are unloaded. ``runtime`` is that
    status (None when nothing was attempted) and ``runtime_note`` a one-line
    summary for the user (None when there is nothing to say). *say* receives
    each step of stopping processes and replacing the runtime as one line of
    text.

    THE OTHER COMPONENTS. Every installed component the build pins differently
    (``updater_components.plan``: the sd.cpp and KoboldCpp runtimes, the managed
    ComfyUI, the uv inside the install) is brought to the build's pin after the
    runtime, or with *defer_runtime* during the caller's restart, recorded as a
    pending component update. A uv running from inside the install refuses the
    update with nothing changed. When a component cannot be updated, the
    components are put back, the code swap (and the runtime) is rolled back and
    LocalmError is raised. ``components`` is that status (None when nothing ran)
    and ``components_note`` a one-line summary, naming a component the user
    chose themselves and kept (None when there is nothing to say).

    *target_version* selects a release by version (``localm update --to``): the
    anti-rollback check is replaced by :func:`_check_selected_version`, so an
    older release installs, but only when the build's VERSION is the requested
    one. Its signature is checked with :func:`assess_signature`; a build that does
    not verify raises :class:`UnverifiedBuild` before it is extracted, unless
    *allow_unverified* is True, which requires *expect_sha256* (the sha256 the
    UnverifiedBuild named) so only the build the user agreed to is installed. An
    unverified build never changes the trusted key set. Before a selected
    downgrade is swapped in, the data dir files an older build may not read are
    copied aside (:func:`_backup_data_for_downgrade`; ``data_backup`` is that
    directory or None). A selected install also saves the trusted key set for
    the build it installs (:func:`_carry_trust_forward`), and gives a build from
    before 0.3.2 this build's signature check (:func:`_install_trust_bridge`) so
    it can verify and install later releases and refuses the revoked keys;
    ``kept_for_updates`` names each such part, the only parts of this build an
    older one keeps (an empty list otherwise). Without
    *target_version* nothing of this applies: the signature must verify and the
    build must be newer.

    *expect_sha256*, when given, must equal the downloaded build's sha256.

    *signature* is the signature envelope text from :func:`check` (the proxy serves
    it with the release). :func:`verify_signature` checks it BEFORE the build is
    extracted or any of its code can run; a missing or invalid signature, a corrupt
    saved key set, or a downgrade all raise before anything is swapped (fail
    closed - we never install an unverified build). A replacement key set the
    envelope carries is saved (:func:`_save_keyset`) only after every check has
    passed and before the swap, so a refused update leaves the trusted keys
    unchanged; a failure to save it refuses the update. A swap or post-swap step
    that fails after that point rolls the files back but keeps the saved key set.

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
    if installed is None and is_site_packages_install(target):
        raise LocalmError(
            "this localm was installed with pip, so applying a release tree "
            "would overwrite unrelated packages in site-packages and leave "
            "pip unaware of what is installed",
            reason="run 'pip install --upgrade localm' instead")
    if allow_unverified and not expect_sha256:
        raise LocalmError("refusing to install an unverified build without its sha256",
                          reason="pass the sha256 the verification warning named")
    updir = _updates_dir()
    backup_dir = _backup_dir(create=True)   # the STABLE location rollback_last() reads

    with _apply_lock():
        run_dir = _new_run_dir()
        zip_path, staging, run_backup_dir = (
            run_dir / "build.zip", run_dir / "staging", run_dir / "backup")

        try:
            import hashlib
            download(asset_id, zip_path, opener=download_opener)
            data = zip_path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if expect_sha256 is not None and digest != str(expect_sha256).strip().lower():
                raise LocalmError(
                    "refusing to install: the build differs from the one you confirmed",
                    reason="the update server sent different bytes this time; "
                           "choose the version again")
            # SIGNATURE GATE - runs before verify_zip/extract/swap, i.e. before any
            # of the downloaded build's code can be extracted or executed.
            if target_version is None:
                new_keyset = verify_signature(data, signature)
                verified, verify_reason = True, None
            else:
                assessed = assess_signature(data, signature)
                new_keyset = assessed["keyset"]
                verified, verify_reason = assessed["verified"], assessed["reason"]
                if not verified and not allow_unverified:
                    raise UnverifiedBuild(str(target_version), verify_reason, digest)
            au.verify_zip(zip_path)
            root = au.extract(zip_path, staging)
            vf = root / "VERSION"
            new_version = vf.read_text(encoding="utf-8").strip() if vf.exists() else ""
            if target_version is None:
                # ANTI-ROLLBACK - a validly SIGNED but OLDER build must not be installable.
                _refuse_downgrade(new_version)
            else:
                _check_selected_version(new_version, str(target_version))
            downgrade = _version.is_newer(_version.read_version(), new_version)
            klass = classify(root, target, read_manifest(root))
            from localm import updater_runtime
            rt_plan = updater_runtime.plan(root, target)
            from localm import updater_components
            comp_plans = updater_components.plan(root, target, deep=False)
            comp_update = updater_components.needs_finish(comp_plans)
            if comp_update and klass != "setup":
                held = updater_components.blockers(comp_plans, target)
                if held:
                    raise LocalmError(
                        "a component this update replaces is in use, so nothing was updated",
                        reason=updater_runtime.in_use_message(held, "run the update again"))
            if (klass == "runtime" and rt_plan["source"] == "release" and rt_plan["target"]
                    and rt_plan["current"] is not None):
                rt_plan["reprovision"] = True
            if rt_plan["reprovision"]:
                klass = _max_class(klass, "runtime")
            if rt_plan["reprovision"] and klass != "setup":
                left = updater_runtime.stop_own_holders(
                    updater_runtime.runtime_lib(target), target, say=say,
                    stop=not defer_runtime)
                if left:
                    raise LocalmError(
                        "the native llama.cpp runtime is in use, so nothing was updated",
                        reason=updater_runtime.in_use_message(left, "run the update again"))
            deps_changed = klass != "setup" and _deps_set(root) != _deps_set(target)
            uv_exe = _uv_for_deps(target) if deps_changed else None
            post_cmd = au.post_swap_command("deps", uv=uv_exe) if deps_changed else None
            # The update has verified: save a key set it replaced, before the swap.
            if new_keyset is not None:
                _save_keyset(new_keyset)
            data_backup = None
            kept = []
            if target_version is not None:
                _carry_trust_forward()
                if _needs_trust_bridge(root):
                    _install_trust_bridge(root)
                    kept.append("this version's update signature check and trusted keys")
                if _needs_uv_bridge(root):
                    _install_uv_bridge(root, uv_exe or _uv_for_deps(target), target)
                    kept.append("the uv this version uses to apply later updates")
                if downgrade:
                    data_backup = _backup_data_for_downgrade(new_version)
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

        cmd = post_cmd
        if cmd:
            run = runner or (lambda c: subprocess.run(
                c, cwd=str(target), env=_deps_env(target)).returncode)

            def _rollback_or_raise(why):
                # A failure of the recovery ITSELF raises rather than reporting a
                # clean rollback over a broken install.
                try:
                    au.rollback(backup_dir, target, names)
                except Exception as rb:
                    raise LocalmError(
                        "the post-update step failed AND rollback failed - manual recovery needed",
                        reason=f"{why}; rollback: {rb}; restore from {backup_dir}") from None

            try:
                rc = int(run(cmd))
            except Exception as e:
                what = f"could not start {cmd[0]}: {e}" if isinstance(e, OSError) else str(e)
                _rollback_or_raise(f"post-update step crashed: {what}")
                raise LocalmError("the post-update step crashed; rolled back", reason=what) from None
            if rc != 0:
                _rollback_or_raise(f"{cmd[0]} exited {rc}")
                raise LocalmError("the post-update step failed; rolled back",
                                  reason=f"{cmd[0]} exited {rc}")
        runtime_status, runtime_note = None, rt_plan["note"]
        if not rt_plan["reprovision"]:
            try:
                updater_runtime.clear_pending()
            except OSError as e:
                _apply_warn("could not clear an earlier pending runtime update: %s", e)
        else:
            try:
                updater_runtime.write_pending(rt_plan, root=target, version=new_version)
            except OSError as e:
                _apply_warn("could not record the pending runtime update: %s", e)
                runtime_note = (f"the {rt_plan['backend']} runtime was NOT updated to "
                                f"{rt_plan['target']} (could not record it: {e}); to update it: "
                                f"localm setup-llama --backend {rt_plan['backend']} --force")
            else:
                if klass == "setup":
                    runtime_note = f"{rt_plan['note']} on the first start after setup"
                elif defer_runtime:
                    runtime_note = f"{rt_plan['note']} while localm restarts"
                else:
                    try:
                        runtime_status = updater_runtime.finish_pending(
                            root=target, runner=runner, env=_deps_env(target), say=say)
                    except Exception as e:
                        runtime_status = {"state": "failed", "note": str(e)}
                    if runtime_status is not None and runtime_status["state"] in (
                            "failed", "waiting"):
                        updater_runtime.clear_pending()
                        why = runtime_status["note"]
                        try:
                            au.rollback(backup_dir, target, names)
                        except Exception as rb:
                            raise LocalmError(
                                "the native runtime update failed AND rolling the code "
                                "back failed - manual recovery needed",
                                reason=f"{why}; rollback: {rb}; restore from {backup_dir}"
                            ) from None
                        raise LocalmError(
                            "the native runtime could not be updated, so the update was "
                            "rolled back", reason=why)
                    if runtime_status is not None:
                        runtime_note = runtime_status["note"]
        components_status, components_note = _apply_components(
            comp_plans, comp_update, klass=klass, target=target, version=new_version,
            defer=defer_runtime, runner=runner, say=say, backup_dir=backup_dir, names=names)
        return {"applied": True, "version": new_version, "klass": klass,
                "backup": str(backup_dir), "restart_needed": True,
                "verified": verified, "verify_reason": verify_reason,
                "downgrade": downgrade,
                "data_backup": str(data_backup) if data_backup else None,
                "runtime": runtime_status, "runtime_note": runtime_note,
                "components": components_status, "components_note": components_note,
                "kept_for_updates": kept}


def _apply_components(plans, needed: bool, *, klass: str, target: Path, version: str,
                      defer: bool, runner, say, backup_dir: Path, names) -> tuple:
    """The component half of :func:`apply`, after the swap: record the pending
    component update (``updater_components.write_pending``) and, unless
    *defer* or the ``setup`` class, carry it out now. When it fails, the code
    swap is rolled back (with the native runtime it re-provisioned) and
    LocalmError is raised. Returns ``(status, note)``: the status of the
    component update (None when none ran) and a one-line summary (None when
    there is nothing to say)."""
    from localm import _apply_update as au
    from localm import updater_components
    from localm.bugreport import LocalmError
    note = updater_components.summary(plans)
    if not needed:
        try:
            updater_components.clear_pending()
        except OSError as e:
            _apply_warn("could not clear an earlier pending component update: %s", e)
        return None, note
    try:
        updater_components.write_pending(plans, root=target, version=version)
    except OSError as e:
        _apply_warn("could not record the pending component update: %s", e)
        return None, (f"{note}; NOT done: the component update could not be recorded ({e}); "
                      "it is retried on the next start of localm")
    if klass == "setup":
        return None, (f"{note}, " if note else "checks the components ") + (
            "on the first start after setup")
    if defer:
        return None, (f"{note}, " if note else "checks the components ") + (
            "while localm restarts")
    try:
        status = updater_components.finish_pending(root=target, runner=runner,
                                                   env=_deps_env(target), say=say)
    except Exception as e:
        status = {"state": "failed", "note": f"the component update crashed: {e}"}
    if status is not None and status["state"] in ("failed", "waiting"):
        updater_components.clear_pending()
        why = status["note"]
        try:
            au.rollback(backup_dir, target, names)
        except Exception as rb:
            raise LocalmError(
                "a component update failed AND rolling the code back failed - manual "
                "recovery needed", reason=f"{why}; rollback: {rb}; restore from {backup_dir}"
            ) from None
        raise LocalmError("a component could not be updated, so the update was rolled back",
                          reason=why)
    return status, (status["note"] if status is not None else note)


def _uv_for_deps(target: Path) -> str:
    """The uv that reinstalls the dependencies of the install at *target*.

    Finds the install's own uv (:func:`localm.uv_tool.find_uv`); when there is
    none, installs the pinned uv into ``<target>/.uv``. Raises LocalmError,
    naming every place looked at and why the install failed, when neither
    works. Called before the swap, so a refusal changes nothing."""
    from localm import uv_tool
    from localm.bugreport import LocalmError
    found = uv_tool.find_uv(target)
    if found:
        return found
    searched = uv_tool.describe_search(target)
    try:
        installed = uv_tool.bootstrap_uv(target)
        _record_contained_uv(target, installed)
        return str(installed)
    except uv_tool.UvBootstrapError as e:
        raise LocalmError(
            "this update needs uv to reinstall the dependencies, and uv could not be found or installed",
            reason=f"looked for uv in: {searched}. Installing the pinned uv into "
                   f"{target / uv_tool.CONTAINED_DIR} failed: {e}") from None


def _record_contained_uv(target: Path, installed) -> None:
    """Note the uv just installed in the install record, so uninstall removes it.
    A record that cannot be written is logged and does not fail the update."""
    try:
        from localm import install_manifest
        install_manifest.record(target, uv_dir=str(Path(installed).parent))
    except (OSError, ValueError) as e:
        _apply_warn("installed uv into %s but could not note it in the install record: %s; "
                    "uninstall will not remove that folder", Path(installed).parent, e)


def _deps_env(target: Path) -> dict:
    """The environment for the dependency step: the current one, with uv's cache
    and Python folders pointed at the install's own when it recorded them as
    contained and the caller has not set them."""
    import os
    env = dict(os.environ)
    try:
        from localm import install_manifest
        m = install_manifest.load(target) or {}
    except Exception:
        m = {}
    if m.get("runtime_contained"):
        for var, key in (("UV_CACHE_DIR", "cache_dir"), ("UV_PYTHON_INSTALL_DIR", "python_dir")):
            if m.get(key) and var not in env:
                env[var] = str(m[key])
    return env


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
            # CREATE_NO_WINDOW (own hidden console, never a visible window) +
            # CREATE_NEW_PROCESS_GROUP so the watchdog is not tied to this
            # process's console/process group and survives past the execv this
            # call precedes. The literal fallbacks keep this branch runnable on
            # any platform.
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
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


def rollback_last(*, installed=None, say=None) -> dict:
    """Restore the install from the most recent update backup. Returns
    ``{rolled_back, backup}``. Raises LocalmError when there is no backup.

    The code and the native runtime the update replaced are restored both or
    neither. When the runtime differs from the one the update kept
    (``_apply_update.runtime_snapshot_dir``), this install's own processes that
    hold it are stopped first (``updater_runtime.stop_own_holders``); a holder
    that is not this install's, or a runtime that cannot be restored, raises
    LocalmError with nothing changed. The runtime is restored before the code,
    and put back as it was when the code restore fails. *say* receives each
    step as one line of text.

    The components the update replaced (``updater_components.snapshot_dir``)
    and that are still installed are restored first, with the same guarantee:
    a component that cannot be restored raises LocalmError with nothing
    changed, and a later failure of the runtime or code restore puts them back
    as they were.

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
            except (OSError, ValueError, RecursionError):
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
        from localm import updater_components
        comp_snap = updater_components.snapshot_dir(backup_dir)
        comp_names = updater_components.installed_names(
            updater_components.snapshot_names(comp_snap), target)
        comp_keep = None
        if comp_names:
            comp_keep = comp_snap.with_name(comp_snap.name + "-current")
            try:
                updater_components.snapshot(comp_names, comp_keep, root=target)
            except OSError as e:
                raise LocalmError("could not keep a copy of the current components, so "
                                  "nothing was rolled back", reason=str(e)) from None
            errors = updater_components.restore(comp_snap, root=target, say=say,
                                                names=comp_names)
            if errors:
                back = updater_components.restore(comp_keep, root=target, say=say)
                updater_components.discard(comp_keep)
                if back:
                    raise RuntimeError("rollback incomplete: the components could not be "
                                       "restored nor put back: " + "; ".join(errors + back))
                raise LocalmError("a component could not be restored, so nothing was "
                                  "rolled back", reason="; ".join(errors))
        try:
            from localm import updater_runtime
            lib = target / au.RUNTIME_LIB
            snapshot = au.runtime_snapshot_dir(backup_dir)
            keep = None
            if snapshot.is_dir() and au.runtime_differs(lib, snapshot):
                left = updater_runtime.stop_own_holders(lib, target, say=say)
                if left:
                    raise LocalmError(
                        "the native llama.cpp runtime is in use, so nothing was rolled back",
                        reason=updater_runtime.in_use_message(left, "run the rollback again"))
                keep = snapshot.with_name(snapshot.name + "-current")
                try:
                    au.snapshot_runtime(lib, keep)
                except OSError as e:
                    raise LocalmError("could not keep a copy of the current native runtime, so "
                                      "nothing was rolled back", reason=str(e)) from None
                if say:
                    say("Restoring the native runtime the update replaced...")
                errors = au.restore_runtime(snapshot, lib)
                if errors:
                    back = au.restore_runtime(keep, lib)
                    shutil.rmtree(keep, ignore_errors=True)
                    if back:
                        raise RuntimeError("rollback incomplete: the native runtime could not be "
                                           "restored nor put back: " + "; ".join(errors + back))
                    raise LocalmError("the native runtime could not be restored, so nothing was "
                                      "rolled back", reason="; ".join(errors))
            if say:
                say("Restoring the previous localm code...")
            try:
                au.rollback(backup_dir, target, names)
            except Exception as e:
                if keep is not None:
                    back = au.restore_runtime(keep, lib)
                    if back:
                        raise RuntimeError(f"{e}; putting the native runtime back failed too: "
                                           + "; ".join(back)) from None
                raise
            finally:
                if keep is not None:
                    shutil.rmtree(keep, ignore_errors=True)
        except BaseException as e:
            if comp_keep is not None:
                back = updater_components.restore(comp_keep, root=target, say=say)
                if back:
                    raise RuntimeError(f"{e}; putting the components back failed too: "
                                       + "; ".join(back)) from None
            raise
        finally:
            if comp_keep is not None:
                updater_components.discard(comp_keep)
        try:
            updater_components.clear_pending()
        except OSError as e:
            _apply_warn("could not clear a pending component update after the rollback: %s", e)
        try:
            updater_runtime.clear_pending()
        except OSError as e:
            _apply_warn("could not clear a pending runtime update after the rollback: %s", e)
        return {"rolled_back": True, "backup": str(backup_dir)}
