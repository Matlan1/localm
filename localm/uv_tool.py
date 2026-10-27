# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding, and if need be installing, the uv an install runs.

An install uses one of three uv setups: a contained copy in ``<root>/.uv``, the
user's own uv somewhere else, or none at all (uv was only ever needed by setup).
The updater's dependency step and the graphical installer both need the uv the
install actually uses, so the lookup lives here once.

Stdlib only: ``installer/gui.py`` imports this module before any virtual
environment exists.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

IS_WINDOWS = sys.platform == "win32"
UV_NAME = "uv.exe" if IS_WINDOWS else "uv"
CONTAINED_DIR = ".uv"

_PIN_SCRIPTS = ("setup.bat", "setup-gui.bat") if IS_WINDOWS else ("setup.sh", "setup-gui.sh")
_VERSION_RE = re.compile(r'^(?:set ")?UV_INSTALLER_VERSION=(?:")?([0-9][^"\r\n]*?)"?\s*$', re.M)
_SHA_RE = re.compile(r'^(?:set ")?UV_INSTALLER_SHA256=(?:")?([0-9a-fA-F]{64})"?\s*$', re.M)
_INSTALLER_NAME = "uv-installer.ps1" if IS_WINDOWS else "uv-installer.sh"


class UvBootstrapError(Exception):
    """The pinned uv could not be installed; the message says why."""


def _recorded(root: Path) -> tuple[Optional[Path], Optional[Path]]:
    """(contained uv, shared uv) as the install record names them."""
    try:
        from localm import install_manifest
        m = install_manifest.load(root) or {}
    except Exception:
        return None, None
    contained = Path(str(m["uv_dir"])) / UV_NAME if m.get("uv_dir") else None
    shared = Path(str(m["uv_exe"])) if m.get("uv_exe") else None
    return contained, shared


def search_places(root) -> tuple[list[Path], list[Path]]:
    """The places a uv is looked for, as (before PATH, after PATH). PATH itself
    is searched between the two lists. A contained copy in the folder comes
    first, then the shared uv the install record names."""
    root = Path(root)
    home = Path.home()
    contained, shared = _recorded(root)
    before = [p for p in (contained,) if p]
    before += [root / CONTAINED_DIR / UV_NAME, root / CONTAINED_DIR / "bin" / UV_NAME]
    before += [p for p in (shared,) if p]
    return before, [home / ".local" / "bin" / UV_NAME, home / ".cargo" / "bin" / UV_NAME]


def find_uv(root) -> Optional[str]:
    """Full path of the uv this install runs, or None.

    Order: the contained ``.uv`` copy, the shared uv the install record names, a
    uv on PATH, then Astral's default install directories (a shell started
    before the installer ran does not have those on PATH yet)."""
    before, after = search_places(root)
    for candidate in before:
        if candidate.is_file():
            return str(candidate)
    on_path = shutil.which("uv")
    if on_path:
        return on_path
    for candidate in after:
        if candidate.is_file():
            return str(candidate)
    return None


def describe_search(root) -> str:
    """What find_uv looked at, for an error message."""
    before, after = search_places(root)
    places = [str(p) for p in before] + ["every folder on PATH"] + [str(p) for p in after]
    return "; ".join(places)


def contained_uv(root) -> Optional[Path]:
    """The uv copy inside *root* (``.uv``), or None."""
    for rel in (Path(CONTAINED_DIR) / UV_NAME, Path(CONTAINED_DIR) / "bin" / UV_NAME):
        if (Path(root) / rel).is_file():
            return Path(root) / rel
    return None


def is_inside(path, root) -> bool:
    """Whether *path* lies inside *root* (symlinks resolved)."""
    try:
        Path(os.path.realpath(path)).relative_to(os.path.realpath(root))
    except ValueError:
        return False
    return True


def read_pin(root) -> tuple[str, str]:
    """(uv version, installer sha256) pinned by the setup script shipped in
    *root*. The pin lives in the setup scripts only; nothing here carries its
    own copy, so a pin bump needs no change to this module."""
    problems = []
    for name in _PIN_SCRIPTS:
        path = Path(root) / name
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            problems.append(f"{name}: {e}")
            continue
        version, sha = _VERSION_RE.search(text), _SHA_RE.search(text)
        if version and sha:
            return version.group(1), sha.group(1).lower()
        problems.append(f"{name}: no uv installer pin found")
    raise UvBootstrapError("the uv version this install pins could not be read ("
                           + "; ".join(problems) + ")")


def _download(url: str, dest: Path, timeout: float = 120.0) -> None:
    import urllib.request
    from localm.http_ssl import verified_urlopen
    req = urllib.request.Request(url, headers={"User-Agent": "localm"}, method="GET")
    with verified_urlopen(req, timeout=timeout) as resp:
        if not 200 <= int(resp.status) < 300:
            raise OSError(f"HTTP {resp.status}")
        dest.write_bytes(resp.read())


def _run_installer(script: Path, env: dict) -> int:
    if IS_WINDOWS:
        env = {k: v for k, v in env.items() if k.upper() != "PSMODULEPATH"}
        argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    else:
        argv = ["sh", str(script)]
    return subprocess.run(argv, env=env, stdin=subprocess.DEVNULL).returncode


def bootstrap_uv(root, *, download: Optional[Callable[[str, Path], None]] = None,
                 run_installer: Optional[Callable[[Path, dict], int]] = None,
                 emit: Callable[[str], None] = lambda s: None) -> Path:
    """Install the pinned uv into ``<root>/.uv`` and return its path.

    The installer script is downloaded from the pinned release and runs only if
    its sha256 equals the pin in the setup script; it is told to keep uv in
    ``.uv`` and to leave PATH and the user profile alone. Raises
    UvBootstrapError, naming the step that failed, and installs nothing on a
    failed download or a checksum mismatch."""
    root = Path(root)
    version, expected = read_pin(root)
    target = root / CONTAINED_DIR
    url = f"https://github.com/astral-sh/uv/releases/download/{version}/{_INSTALLER_NAME}"
    scratch = Path(tempfile.mkdtemp(prefix="uv-installer-"))
    try:
        script = scratch / _INSTALLER_NAME
        emit(f"Downloading the uv {version} installer ...")
        try:
            (download or _download)(url, script)
        except Exception as e:
            raise UvBootstrapError(f"could not download the uv {version} installer: {e}") from None
        import hashlib
        got = hashlib.sha256(script.read_bytes()).hexdigest()
        if got != expected:
            raise UvBootstrapError(
                f"the downloaded uv {version} installer did not match its expected "
                "checksum and was not run")
        env = dict(os.environ, UV_INSTALL_DIR=str(target), UV_UNMANAGED_INSTALL=str(target))
        emit(f"Installing uv {version} into {target} ...")
        try:
            rc = (run_installer or _run_installer)(script, env)
        except OSError as e:
            raise UvBootstrapError(f"could not run the uv installer: {e}") from None
        if rc != 0:
            raise UvBootstrapError(f"the uv installer exited {rc}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    found = contained_uv(root)
    if found is None:
        raise UvBootstrapError(f"the uv installer finished but no {UV_NAME} appeared in {target}")
    return found
