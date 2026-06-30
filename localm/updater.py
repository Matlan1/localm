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
    if new_py and new_py != old_py:
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
    import urllib.request
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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
