# SPDX-License-Identifier: AGPL-3.0-or-later
"""The runtime lib dir: where it is, the backend marker it carries, installing
the runtime wheel, clearing it for a re-provision, and the cross-process
provisioning lock.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from localm import config
from localm.debuglog import logger
from localm.setup_llama._common import console
from localm.setup_llama.library_files import _BLAS_LIBRARY_DIRS
import localm.setup_llama as _sl

def _platform_key() -> str:
    if sys.platform == "win32":
        return "win32"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _lib_name() -> str:
    """The loadable llama library filename for this platform."""
    if sys.platform == "win32":
        return "llama.dll"
    if sys.platform == "darwin":
        return "libllama.dylib"
    return "libllama.so"


# A tiny marker file recording WHICH backend currently occupies the runtime lib
# dir. It exists so the "already provisioned" guard can be backend-aware: a later
# `setup-llama --backend cuda` on a box that already has a vulkan/cpu build must
# still fetch CUDA, instead of short-circuiting on the mere presence of a
# library. A dotfile, like the venv's .localm-venv marker; never loaded as code.
_BACKEND_MARKER = ".localm-backend"


def _record_provisioned_backend(target: Path, backend: str,
                                build: Optional[str] = None) -> None:
    """Record *backend* as the one now provisioned in *target*, optionally with
    the *build* tag it came from. Best-effort: the marker only optimises the
    guard, so a write failure is non-fatal (the guard then conservatively
    re-provisions an explicit pick rather than skipping it). Written AFTER
    provisioning because _clear_target wipes the dir's files.

    Format is ``<backend>`` or ``<backend> <build>``, whitespace-separated. The
    second token is OPTIONAL by design and is omitted whenever the tag is not
    known for free - see the call sites. A marker with no build reads back
    identically for the guard's purposes (see _provisioned_backend), so adding
    the tag needs no migration and no version detection."""
    line = (backend or "").strip()
    if build:
        line = f"{line} {str(build).strip()}"
    try:
        (target / _BACKEND_MARKER).write_text(line + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_marker(target: Path) -> Optional[list]:
    """The marker's whitespace-separated tokens, or None when there is no
    readable marker. One reader for both accessors below, so the two can never
    disagree about how the file is split."""
    try:
        raw = (target / _BACKEND_MARKER).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw.split() or None


def _provisioned_backend(target: Path) -> Optional[str]:
    """The backend last provisioned into *target*, or None if unknown (no marker
    - e.g. an install predating the marker, or a hand-placed build). 'Unknown'
    is treated conservatively by the guard: an explicit pick is re-provisioned.

    THE FIRST WHITESPACE TOKEN, never the whole file. That is what makes the
    optional build tag safe to add: "amd-rocm" and "amd-rocm b1307" both answer
    "amd-rocm", so the provision guard's ``have == want`` comparison is byte for
    byte the decision it always made. A bare .strip() of the whole file would
    have returned "amd-rocm b1307", matched no backend name, and re-provisioned
    on EVERY invocation - which on a shared box is precisely the destructive
    path the runtime-in-use refusal exists to stop. Backward and forward
    compatible by construction rather than by a version check, which is what a
    file written by releases you cannot revise needs."""
    parts = _read_marker(target)
    return parts[0] if parts else None


def _provisioned_build(target: Path) -> Optional[str]:
    """The build tag recorded alongside the backend, or None when the marker
    predates the two-token format or the tag was not knowable at provision time.

    ABSENCE IS NORMAL, never corruption: _record_provisioned_backend is
    best-effort and omits the tag whenever it is not free to obtain, so every
    reader must treat None as "not recorded" and say something honest rather
    than guess a version."""
    parts = _read_marker(target)
    return parts[1] if parts and len(parts) > 1 else None


def installed_backend() -> Optional[str]:
    """The backend actually provisioned on this box right now, or None when
    nothing is provisioned yet (a fresh install, or one that predates the
    marker).

    Public, read-only convenience for callers OUTSIDE this package that need
    "what is installed" rather than "what would be recommended fresh" -
    updater.py's own backend-preservation (an update must never silently swap
    a user to a different backend just because the hardware-recommendation
    policy changed) and the Settings page's backend display. Resolves the
    target directory the same way the provisioning code does
    (_repo_runtime_lib), so it always reads the marker the real install
    actually wrote."""
    return _sl._provisioned_backend(_sl._repo_runtime_lib())


def installed_build() -> Optional[str]:
    """The llama.cpp release tag actually provisioned on this box right now, or
    None when nothing is provisioned or the marker predates tag recording.

    Public, read-only, and the exact shape of installed_backend() above rather
    than a second mechanism: doctor and the bug reporter need "which build is on
    disk", and inferring it from library filenames is a guess, not a lookup.

    None is NORMAL and every caller must render it as "not recorded" rather than
    guessing a version. See _provisioned_build."""
    return _sl._provisioned_build(_sl._repo_runtime_lib())


def installed_runtime_identity() -> list:
    """What is provisioned in the runtime lib dir right now, as a list that
    differs whenever a provision ran there: the recorded backend and build
    (None when not recorded), then the size and modification time of the
    backend marker and of the runtime library (None for a file that cannot be
    read). Replacing one ``--from`` or ``--url`` build with another records the
    same backend and build, so the file entries are what tell them apart."""
    target = _sl._repo_runtime_lib()
    identity = [_sl._provisioned_backend(target), _sl._provisioned_build(target)]
    for path in (target / _BACKEND_MARKER, target / _sl._lib_name()):
        try:
            st = path.stat()
            identity += [st.st_size, st.st_mtime_ns]
        except OSError:
            identity += [None, None]
    return identity


def _repo_runtime_lib() -> Path:
    """The localm-llama-runtime wheel's lib/ dir."""
    try:
        import localm_llama_runtime
        return Path(localm_llama_runtime.LIB_DIR)
    except Exception as e:
        # The wheel is legitimately ABSENT before `setup-llama` installs it, so
        # the repo-relative fallback is correct then - do NOT hard-fail. But a
        # BROKEN install (import error other than not-found) would otherwise be
        # invisible and lead to loading stale binaries, so surface it at debug
        # level, without breaking the not-yet-installed path. Mirrors the
        # visible-fallback pattern in assets._auto_backend.
        logger.debug("localm_llama_runtime import failed (%s); "
                     "using the repo-relative runtime lib dir", e)
        repo_root = Path(__file__).resolve().parent.parent.parent
        return repo_root / "runtime" / "localm_llama_runtime" / "lib"


def _runtime_pkg_dir() -> Path:
    """The runtime wheel project dir (for `pip install -e`)."""
    return _sl._repo_runtime_lib().parent.parent


def _install_runtime_wheel(pkg_dir: Path) -> bool:
    """Install the runtime wheel editable into the active venv. Tries uv, then
    pip. Returns True on success.

    ``env`` pins uv's AND pip's caches inside the data dir (rule 4: self-contained),
    same as the plugin-extra installer (plugins/deps.py). An editable install of a
    local dir is a smaller leak than a full torch download, but build isolation still
    pulls the build backend (setuptools/wheel) into the tool's cache, and either tool
    would otherwise put it in a per-user location OUTSIDE the data dir. See
    ``config.contained_pip_env``."""
    # Already importable means the package ships in this install (a wheel built
    # with runtime/localm_llama_runtime in it), so its lib/ is provisioned in
    # place and there is nothing to install. Running the editable install here
    # would target site-packages itself, and `-m pip` is absent from a
    # uv-created venv. See test_runtime_wheel_install_skipped_when_importable.
    try:
        import localm_llama_runtime  # noqa: F401
        return True
    except Exception:
        pass
    env = config.contained_pip_env()
    last_err = ""
    for cmd in (["uv", "pip", "install", "-e", str(pkg_dir)],
                [sys.executable, "-m", "pip", "install", "-e", str(pkg_dir)]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if r.returncode == 0:
                return True
            # Keep the real pip/uv failure instead of discarding it, so the user
            # can see the actual cause (missing build tools, conflicting deps).
            last_err = (r.stderr or r.stdout or "").strip()
        except FileNotFoundError:
            continue
    # Surface the last failed attempt's output: full to the debug log, a trimmed
    # tail to stderr so the caller's "did not load" path has the real reason.
    if last_err:
        logger.debug("runtime wheel install failed: %s", last_err)
        tail = "\n".join(last_err.splitlines()[-8:])
        console.print(f"[yellow]Runtime wheel install failed:[/yellow]\n{tail}")
    return False


# Tracked git sentinel files living in the runtime lib dir (see
# runtime/localm_llama_runtime/lib/.gitignore, which keeps the downloaded
# native binaries out of version control). Unlike _BACKEND_MARKER, which
# _clear_target wipes and _record_provisioned_backend rewrites after every
# provision, these two are never regenerated by setup-llama: deleting them
# empties the .gitignore, so a later `git add -A`/`git add .` touching this
# directory would stage the freshly-downloaded DLLs straight into git.
_PRESERVED_TARGET_FILES = (".gitignore", ".gitkeep")


class RuntimeInUseError(Exception):
    """Something has the installed runtime open, so it cannot be replaced.

    NOT an ArtifactError and not a load failure. Those two mean a build is bad;
    this one means both builds are fine and a process is merely holding the
    files, so the response differs: a build that will not load earns the Vulkan
    fallback, this earns "close it and retry" with the existing install left
    completely intact."""

    def __init__(self, locked: list[Path], partial: bool = False):
        self.locked = list(locked)
        # True only when files were already deleted before the lock was hit (the
        # probe-to-unlink race). The install is then half cleared, and saying
        # "nothing was changed" would be a lie - so the two cases are tracked
        # apart and reported apart.
        self.partial = partial
        shown = ", ".join(sorted(p.name for p in self.locked[:6]))
        more = f" (+{len(self.locked) - 6} more)" if len(self.locked) > 6 else ""
        super().__init__(f"{len(self.locked)} runtime file(s) in use: {shown}{more}")


def _clearable_files(target: Path) -> list[Path]:
    """The files _clear_target WOULD delete, in deletion order."""
    out: list[Path] = []
    try:
        for f in target.iterdir():
            if f.is_file():
                if f.name in _PRESERVED_TARGET_FILES:
                    continue
                out.append(f)
            elif f.is_dir() and f.name in _BLAS_LIBRARY_DIRS:
                out.extend(p for p in f.rglob("*") if p.is_file())
    except OSError:
        pass
    return out


def _files_in_use(target: Path) -> list[Path]:
    """Of the files a provision would delete, those that cannot be replaced now.

    Probed with ``open(..., "r+b")``: it writes no bytes and creates nothing, and
    it asks the OS the exact question deletion asks. A DLL reports WRITABLE
    before ``ctypes.CDLL`` and PermissionError errno 13 after, the IDENTICAL
    error ``unlink()`` raises on the same handle, so the probe predicts the
    deletion rather than merely correlating with it.

    Naturally platform-correct with no platform test. Windows maps a loaded DLL
    without FILE_SHARE_WRITE/DELETE, so the probe refuses exactly when deletion
    would. POSIX has no mandatory locking and unlinking an open file SUCCEEDS
    (the directory entry goes, the inode lives until the last close), so there is
    no half-state to prevent there and the probe correctly finds nothing.

    An unprobeable file counts as NOT in use. This gate exists to prevent a
    destructive half-state, so an inconclusive answer must not become a new way
    to block a legitimate install."""
    locked: list[Path] = []
    for f in _clearable_files(target):
        try:
            with open(f, "r+b"):
                pass
        except PermissionError:
            locked.append(f)
        except OSError:
            # Not a "someone holds it" answer (gone, unreadable, a device). Let
            # the delete itself deal with it; _clear_target reports what remains.
            pass
    return locked


def _clear_target(target: Path) -> list[Path]:
    """Remove library files left by an earlier provision so a re-provision (or
    a fallback to a different backend) never mixes DLLs from two builds. Only
    touches files in the dir, plus the _BLAS_LIBRARY_DIRS subdirectories
    _copy_blas_library_dirs may have created (never any OTHER subdirectory) -
    and never _PRESERVED_TARGET_FILES, the tracked git sentinels.

    RETURNS THE FILES IT COULD NOT REMOVE, and the return value is load-bearing:
    every caller must treat a non-empty result as a failed provision. Swallowing
    the OSError instead would report success on a half-cleared directory, and
    the caller would then copy a new build over the survivors and produce the
    mixed-build state this function exists to prevent."""
    left: list[Path] = []
    try:
        for f in target.iterdir():
            if f.is_file():
                if f.name in _PRESERVED_TARGET_FILES:
                    continue
                try:
                    f.unlink()
                except OSError:
                    left.append(f)
            elif f.is_dir() and f.name in _BLAS_LIBRARY_DIRS:
                # ignore_errors keeps rmtree from raising part-way and stranding
                # the rest of the sweep; whatever survives is reported instead.
                shutil.rmtree(f, ignore_errors=True)
                if f.exists():
                    left.extend(p for p in f.rglob("*") if p.is_file())
    except OSError:
        pass
    return left


def _clear_target_or_refuse(target: Path) -> None:
    """Clear *target*, but REFUSE BEFORE DELETING ANYTHING if the runtime is in
    use. Raises RuntimeInUseError with the existing install untouched.

    Checking first is what makes the half-state unreachable rather than merely
    reported. Reporting alone (returning what could not be removed) is a real
    improvement over silence, but by the time it can report, it has already
    deleted everything it could - an honest error plus a runtime missing half its
    files. Probing first turns that into "could not update, something is using
    the runtime" with nothing lost.

    The post-clear check is not redundant: a process can open a file in the
    window between the probe and the unlink. That race leaves a half-state, which
    is why it raises the same error rather than continuing - it cannot be
    prevented here, but it must never be silent."""
    in_use = _sl._files_in_use(target)
    if in_use:
        raise RuntimeInUseError(in_use, partial=False)
    left = _sl._clear_target(target)
    if left:
        raise RuntimeInUseError(left, partial=True)


def _exit_runtime_in_use(e: RuntimeInUseError) -> None:
    """Report a refused provision and exit non-zero. Never falls back to another
    backend: the user's chosen backend is not the problem, so silently installing
    a different one would be exactly the never-override-the-user's-choice
    mistake, dressed as a recovery."""
    console.print(f"[red]Cannot replace the installed runtime: it is in use.[/red] {e}")
    if e.partial:
        console.print("[yellow]Some files were already removed before the lock "
                      "appeared, so this install is now incomplete - re-run the "
                      "same command once nothing is using it.[/yellow]")
    else:
        console.print("[green]Your existing install was left untouched.[/green]")
    console.print("[dim]Close anything using the runtime (a running `localm serve` "
                  "or `localm gui`, a Python session that imported localm, another "
                  "setup-llama) and run the same command again.[/dim]")
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Single-flight: only ONE process may PROVISION the runtime lib dir at once   #
# --------------------------------------------------------------------------- #
# Provisioning CLEARS then REFILLS `target` (_clear_target_or_refuse + a
# download/copy). Until the GUI grew its own standalone runtime-update button,
# this ran from effectively one caller at a time: a user's own CLI invocation,
# or `localm update`'s runtime-class post-swap step (itself serialized against
# OTHER `localm update` calls by updater.py's own _apply_lock, but that lock
# knows nothing about a bare `setup-llama` run in a terminal or a second,
# independent caller like the GUI route below). Two provisions racing on the
# SAME directory is not merely slow, it is a corrupted install: both clear and
# refill it, so one process's half-written file can be read - or deleted - by
# the other's _clear_target/_copy_binaries. Same hazard, same fix shape, as
# managed_comfy_update.py's update lock: do NOT copy
# managed_comfy._remove_lock (a threading.Lock) here either - the
# GUI route spawns `setup-llama` as a CHILD PROCESS, so the contenders are
# separate interpreters and a threading.Lock would guard nothing. mkdir is
# atomic; stat-then-create is not.
_PROVISION_LOCK_OWNER = "owner.json"


class ProvisioningBusyError(Exception):
    """Another process already holds the provisioning lock."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _provision_lock_path(target: Path) -> Path:
    """Where the provisioning lock lives: a SIBLING of the runtime lib dir,
    never inside it, so this command's own directory clear can never disturb
    the lock protecting it (same reasoning as managed_comfy_update.py's
    _update_lock_path)."""
    return target.parent / (target.name + ".setup.lock")


def _provision_lock_holder_pid(lock: Path) -> Optional[int]:
    try:
        data = json.loads((lock / _PROVISION_LOCK_OWNER).read_text(encoding="utf-8"))
        pid = data.get("pid") if isinstance(data, dict) else None
        return pid if isinstance(pid, int) else None
    except (OSError, ValueError, RecursionError):
        return None


@contextlib.contextmanager
def _provisioning_lock(target: Path):
    """Cross-process, fail-fast single-flight guard around a run that mutates
    *target*. FAILS FAST rather than blocking: a provision can legitimately run
    for minutes (a download over a slow link), and a caller blocked that long
    is indistinguishable from a hang - a GUI button would spin forever with no
    way to tell the user why.

    Staleness is judged by PID LIVENESS, never elapsed time, for the same
    reason managed_comfy_update.py's lock is: the operation is unbounded, so
    any fixed timeout would eventually reclaim a LIVE holder's lock and
    recreate the exact race this exists to prevent. ``pid_alive`` is
    conservative - when it genuinely cannot tell, it returns True - so an
    uncertain answer keeps the lock rather than stealing it.

    Raises :class:`ProvisioningBusyError` with an honest reason when the lock
    cannot be acquired. Always released in ``finally``."""
    from localm.instances import pid_alive
    lock = _provision_lock_path(target)
    acquired = False
    for attempt in (1, 2):
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            os.mkdir(str(lock))              # ATOMIC: creates or raises
            acquired = True
            break
        except FileExistsError as exc:
            pid = _provision_lock_holder_pid(lock)
            if pid is not None and not pid_alive(pid):
                # The holder is provably gone. Reclaim once, then retry the
                # atomic create - never assume the retry wins, another caller
                # may have taken it in the meantime.
                with contextlib.suppress(OSError):
                    shutil.rmtree(str(lock))
                if attempt == 1:
                    continue
                raise ProvisioningBusyError(
                    "Another setup-llama run is already provisioning the "
                    "runtime. Wait for it to finish, then try again.") from exc
            if pid is None:
                # Cannot tell who holds it: do NOT steal (that is how two
                # provisions end up interleaved). Say how to clear it by hand.
                raise ProvisioningBusyError(
                    f"A provisioning lock exists at {lock} but its owner could "
                    "not be read. If no setup-llama run is in progress, remove "
                    "that folder and try again.") from exc
            raise ProvisioningBusyError(
                f"Another setup-llama run is already provisioning the runtime "
                f"(process {pid}). Wait for it to finish, then try again.") from exc
        except OSError as e:
            raise ProvisioningBusyError(
                f"Could not take the provisioning lock at {lock}: {e}") from e
    if not acquired:
        raise ProvisioningBusyError(
            "Another setup-llama run is already provisioning the runtime. "
            "Wait for it to finish, then try again.")
    # Record who holds it, so a future caller's staleness check can ask
    # instances.pid_alive() instead of guessing from elapsed time.
    try:
        (lock / _PROVISION_LOCK_OWNER).write_text(
            json.dumps({"pid": os.getpid()}), encoding="utf-8")
    except OSError:
        pass
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            shutil.rmtree(str(lock))


def _exit_provisioning_busy(e: ProvisioningBusyError) -> None:
    """Report a refused provision and exit non-zero, mirroring
    _exit_runtime_in_use: the existing install is left completely untouched
    (the lock is taken before anything is cleared), so this is honest, not
    alarming."""
    console.print(f"[red]Cannot provision the runtime right now:[/red] {e.reason}")
    sys.exit(1)
