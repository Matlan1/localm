# SPDX-License-Identifier: AGPL-3.0-or-later
"""Finding, and if need be installing, the uv an install runs.

An install runs only a uv localm brought: a contained copy in ``<root>/.uv``
(Portable) or the copy in localm's Shared store (:func:`shared_store_dir`) its
install record names. A uv on PATH or in the user profile is never used. The
updater's dependency step, plugin installs and the graphical installer all need
the uv the install actually uses, so the lookup lives here once.

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


def install_root() -> Path:
    """The folder this localm runs from (one level above the package)."""
    return Path(__file__).resolve().parents[1]


def usable(path) -> bool:
    """Whether *path* is a file that can be run as uv (executable on POSIX)."""
    p = Path(path)
    return p.is_file() and (IS_WINDOWS or os.access(p, os.X_OK))


def shared_store_dir() -> Path:
    """The folder a Shared install keeps localm's own uv, managed Python and
    download cache in, one for every Shared localm install of this user:
    ``%LOCALAPPDATA%\\localm\\shared`` on Windows,
    ``${XDG_DATA_HOME:-~/.local/share}/localm/shared`` elsewhere."""
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "localm" / "shared"


def _recorded(root: Path) -> tuple[Optional[Path], list[Path]]:
    """(contained uv folder, shared store uvs) as the install record names them.
    A recorded contained folder outside *root* (the install was copied or moved)
    is ignored, and so is a recorded uv outside :func:`shared_store_dir`."""
    try:
        from localm import install_manifest
        m = install_manifest.load(root) or {}
    except Exception:
        return None, []
    contained = None
    store = []
    store_dir = shared_store_dir()
    if m.get("uv_dir"):
        if is_inside(m["uv_dir"], root):
            contained = Path(str(m["uv_dir"]))
        elif is_inside(m["uv_dir"], store_dir):
            store += [Path(str(m["uv_dir"])) / UV_NAME, Path(str(m["uv_dir"])) / "bin" / UV_NAME]
    if m.get("uv_exe") and is_inside(m["uv_exe"], store_dir):
        store.insert(0, Path(str(m["uv_exe"])))
    return contained, store


def search_places(root) -> list[Path]:
    """The places a uv is looked for, in order: the uv folder the install record
    names inside *root*, ``<root>/.uv`` (and its ``bin``), then the uv in the
    Shared store (:func:`shared_store_dir`) the install record names. PATH and
    the user profile are never searched."""
    root = Path(root)
    contained, store = _recorded(root)
    places = []
    if contained is not None:
        places += [contained / UV_NAME, contained / "bin" / UV_NAME]
    places += [root / CONTAINED_DIR / UV_NAME, root / CONTAINED_DIR / "bin" / UV_NAME]
    places += store
    return list(dict.fromkeys(places))


def find_uv(root) -> Optional[str]:
    """Full path of the uv this install runs (the first usable entry of
    :func:`search_places`), or None."""
    for candidate in search_places(root):
        if usable(candidate):
            return str(candidate)
    return None


def describe_search(root) -> str:
    """What find_uv looked at, for an error message."""
    return "; ".join(str(p) for p in search_places(root))


def _uv_in(folder) -> Optional[Path]:
    """The uv in *folder* (directly or in its ``bin``), or None."""
    for rel in (Path(UV_NAME), Path("bin") / UV_NAME):
        if usable(Path(folder) / rel):
            return Path(folder) / rel
    return None


def contained_uv(root) -> Optional[Path]:
    """The uv copy inside *root* (``.uv``), or None."""
    return _uv_in(Path(root) / CONTAINED_DIR)


def uv_version(path) -> Optional[str]:
    """The version the uv at *path* reports (``uv --version``), or None when it
    cannot be run or prints no version."""
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=30,
                             stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"^uv\s+([0-9][^\s]*)", out or "", re.M)
    return m.group(1) if m else None


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
        except (OSError, ValueError) as e:
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
                 emit: Callable[[str], None] = lambda s: None,
                 dest=None) -> Path:
    """Install the uv pinned by the setup scripts in *root* into ``<root>/.uv``
    (or into the folder *dest* when given) and return its path.

    The installer script is downloaded from the pinned release and runs only if
    its sha256 equals the pin in the setup script; it is told to keep uv in
    ``.uv`` and to leave PATH and the user profile alone. The installer script
    and every temporary file the installer writes (TEMP, TMP and TMPDIR point
    there) stay in a scratch folder under ``<root>/.localm-setup-tmp``, which
    is removed afterwards, together with ``.localm-setup-tmp`` when this call
    created it and it is empty. On Windows, when that scratch path contains
    ``[`` or ``]``, the installer's temporary files go to a fresh folder in the
    system temp folder instead, also removed afterwards: Astral's PowerShell
    installer cannot unpack under a path holding wildcard characters (see
    test_bootstrap_uv_avoids_a_bracketed_temp_folder_on_windows). Raises
    UvBootstrapError, naming the step that failed, and installs nothing on a
    failed download or a checksum mismatch."""
    from localm.install_manifest import SETUP_TMP_NAME
    root = Path(root)
    version, expected = read_pin(root)
    target = Path(dest) if dest is not None else root / CONTAINED_DIR
    url = f"https://github.com/astral-sh/uv/releases/download/{version}/{_INSTALLER_NAME}"
    parent = root / SETUP_TMP_NAME
    made_parent = not parent.exists()
    try:
        parent.mkdir(exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="uv-installer-", dir=parent))
        installer_tmp = scratch / "tmp"
        if IS_WINDOWS and ("[" in str(installer_tmp) or "]" in str(installer_tmp)):
            installer_tmp = Path(tempfile.mkdtemp(prefix="localm-uv-tmp-"))
        else:
            installer_tmp.mkdir()
    except OSError as e:
        raise UvBootstrapError(f"could not create a temporary folder in {parent}: {e}") from None
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
        env = dict(os.environ, UV_INSTALL_DIR=str(target), UV_UNMANAGED_INSTALL=str(target),
                   TEMP=str(installer_tmp), TMP=str(installer_tmp), TMPDIR=str(installer_tmp))
        emit(f"Installing uv {version} into {target} ...")
        try:
            rc = (run_installer or _run_installer)(script, env)
        except OSError as e:
            raise UvBootstrapError(f"could not run the uv installer: {e}") from None
        if rc != 0:
            raise UvBootstrapError(f"the uv installer exited {rc}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(installer_tmp, ignore_errors=True)
        if made_parent and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    found = _uv_in(target)
    if found is None:
        raise UvBootstrapError(f"the uv installer finished but no {UV_NAME} appeared in {target}")
    return found


def bootstrap_dest(root) -> Path:
    """Where :func:`ensure_uv` installs the pinned uv for the install at *root*:
    ``<root>/.uv`` for a Portable install (or one with no install record), the
    ``uv`` folder of :func:`shared_store_dir` for a Shared one."""
    try:
        from localm import install_manifest
        m = install_manifest.load(root)
    except Exception:
        m = None
    if m is not None and not m.get("runtime_contained"):
        return shared_store_dir() / "uv"
    return Path(root) / CONTAINED_DIR


def ensure_uv(root, *, emit: Callable[[str], None] = lambda s: None) -> str:
    """Full path of the uv the install at *root* runs (:func:`find_uv`); when
    there is none, installs the pinned uv into :func:`bootstrap_dest`, notes it in
    the install record, and returns it. Raises UvBootstrapError when the install
    fails."""
    found = find_uv(root)
    if found:
        return found
    dest = bootstrap_dest(root)
    installed = bootstrap_uv(root, emit=emit, dest=dest)
    try:
        from localm import install_manifest
        if install_manifest.load(root) is not None:
            if is_inside(dest, root):
                install_manifest.record(root, uv_dir=str(installed.parent))
            else:
                install_manifest.record(root, uv_exe=str(installed))
    except (OSError, ValueError) as e:
        emit(f"installed uv into {installed.parent} but could not note it in the install "
             f"record: {e}")
    return str(installed)


def uv_for_installs(root, *, emit: Callable[[str], None] = lambda s: None) -> Optional[str]:
    """The uv localm's own package installs run for the install at *root*: in an
    install setup made (it has an install record), :func:`ensure_uv`; elsewhere
    (a source checkout), :func:`find_uv`. None, with the reason passed to *emit*,
    when there is none."""
    try:
        from localm import install_manifest
        made_by_setup = install_manifest.load(root) is not None
    except Exception:
        made_by_setup = False
    if not made_by_setup:
        return find_uv(root)
    try:
        return ensure_uv(root, emit=emit)
    except UvBootstrapError as e:
        emit(f"uv could not be installed: {e}")
        return None
