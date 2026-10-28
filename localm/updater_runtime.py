# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native llama.cpp runtime half of a localm update.

A release pins the runtime build it was made with (``_PINNED_TAG`` for the
upstream backends, ``_ROCM_BUILD`` for amd-rocm). Applying a release compares
what the staged tree pins for the INSTALLED backend with the build recorded in
the installed runtime's marker (:func:`plan`). When they differ, the update
records a pending re-provision (:func:`write_pending`) that
:func:`finish_pending` carries out once no process holds the runtime files:
right after the swap for ``localm update``, during the server's restart (after
its models and workers are stopped) for an update applied from the GUI, else on
the next start of a command that loads models.

A re-provision keeps a snapshot of the runtime it replaces beside the update
backup (``_apply_update.runtime_snapshot_dir``); a failed one restores it, and a
later rollback of the update restores it too.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Optional

from localm import _apply_update as au

PENDING_FILE = "pending-runtime.json"
STATUS_FILE = "runtime-update.json"

# Files that carried the runtime pins, newest layout first.
_PIN_FILES = ("localm/setup_llama/pins.py", "localm/setup_llama/__init__.py",
              "localm/setup_llama.py")

# Commands that load the native runtime, and so finish a pending re-provision
# before they start.
RUNTIME_COMMANDS = frozenset({"run", "serve", "gui", "chat", "coder", "bench-spec",
                              "rag", "rerank", "speak"})


def runtime_lib(root) -> Path:
    """The provisioned runtime directory of the install at *root*."""
    return Path(root) / au.RUNTIME_LIB


# ------------------------------- the decision --------------------------------

def _const(text: str, name: str) -> Optional[str]:
    m = re.search(rf'^{name}\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    return m.group(1) if m else None


def staged_pins(root) -> dict:
    """The runtime pins of the localm tree at *root*, read as text (never
    imported): ``{"pinned_tag", "rocm_tag", "rocm_cpu_tag", "rocm_build"}``, each
    None when that tree does not pin it. ``rocm_build`` is the build identity
    that tree's setup-llama records for amd-rocm: its ``_ROCM_BUILD``, or its
    ``_ROCM_TAG`` for a tree from before the SIMD CPU backend."""
    pins: dict[str, Optional[str]] = {"pinned_tag": None, "rocm_tag": None,
                                      "rocm_cpu_tag": None, "rocm_build": None}
    for rel in _PIN_FILES:
        try:
            text = (Path(root) / rel).read_text(encoding="utf-8")
        except OSError:
            continue
        if not re.search(r"^_PINNED_TAG\s*=", text, re.MULTILINE):
            continue
        pins["pinned_tag"] = _const(text, "_PINNED_TAG")
        pins["rocm_tag"] = _const(text, "_ROCM_TAG")
        pins["rocm_cpu_tag"] = _const(text, "_ROCM_CPU_TAG")
        literal = _const(text, "_ROCM_BUILD")
        composed = re.search(r'^_ROCM_BUILD\s*=\s*f"\{_ROCM_TAG\}-cpu-\{_ROCM_CPU_TAG\}"\s*$',
                             text, re.MULTILINE)
        if literal:
            pins["rocm_build"] = literal
        elif composed and pins["rocm_tag"] and pins["rocm_cpu_tag"]:
            pins["rocm_build"] = f"{pins['rocm_tag']}-cpu-{pins['rocm_cpu_tag']}"
        elif not re.search(r"^_ROCM_BUILD\s*=", text, re.MULTILINE):
            pins["rocm_build"] = pins["rocm_tag"]
        break
    return pins


def _is_current(lib: Path, backend: str, build: Optional[str], target: str, pins: dict) -> bool:
    if backend == "amd-rocm":
        from localm import setup_llama as sl
        return sl.rocm_build_is_current(lib, build, rocm_tag=pins.get("rocm_tag"),
                                        rocm_build_id=target,
                                        cpu_tag=pins.get("rocm_cpu_tag"))
    return build == target


def _manual_command(backend: str) -> str:
    return f"localm setup-llama --backend {backend} --force"


def plan(staged_root, installed_root) -> dict:
    """Whether applying the tree at *staged_root* over the install at
    *installed_root* must re-provision the native runtime.

    Returns ``{"reprovision", "backend", "current", "target", "source", "pins",
    "note"}``. ``reprovision`` is True only when a runtime is installed, its
    marker records its build, the build the staged release installs for that
    backend comes from the release (not from the user's pin or "latest"
    choice, see ``setup_llama.versions.runtime_target``), and it differs from
    the installed one. ``note`` is a one-line, user-facing sentence for the
    update summary, or None. Never raises: a failure to read the marker or the
    configuration answers ``reprovision`` False with a note saying so."""
    out = {"reprovision": False, "backend": None, "current": None, "target": None,
           "source": None, "pins": {}, "note": None}
    try:
        from localm import setup_llama as sl
        from localm.setup_llama.versions import runtime_target
        lib = runtime_lib(installed_root)
        backend = sl._provisioned_backend(lib)
        if not backend:
            return out
        current = sl._provisioned_build(lib)
        pins = staged_pins(staged_root)
        release_build = pins["rocm_build"] if backend == "amd-rocm" else pins["pinned_tag"]
        pin = sl.pinned_tag()
        target, source = runtime_target(backend, pin=pin, tracking=sl.tracks_latest(),
                                        pinned_tag=pins["pinned_tag"],
                                        rocm_build_id=pins["rocm_build"])
        out.update(backend=backend, current=current, target=target, source=source, pins=pins)
        if source == "pin":
            if release_build and release_build != pin:
                out["note"] = (f"kept your pinned llama.cpp build {pin}; this release ships "
                               f"{release_build}")
            return out
        if source == "latest":
            if release_build:
                out["note"] = ("kept your choice to follow upstream's newest llama.cpp build; "
                               f"this release ships {release_build} (to update the runtime: "
                               f"{_manual_command(backend)})")
            return out
        if not target:
            return out
        if current is None:
            out["note"] = (f"the installed {backend} runtime does not record its build, so it "
                           f"was left as it is; this release ships {target} (to install it: "
                           f"{_manual_command(backend)})")
            return out
        if _is_current(lib, backend, current, target, pins):
            return out
        out["reprovision"] = True
        out["note"] = f"re-provisions the {backend} runtime from {current} to {target}"
        return out
    except Exception as e:
        from localm.debuglog import logger
        logger.warning("update: could not compare the installed llama.cpp runtime with the "
                       "release's pin: %s", e)
        out["note"] = (f"could not compare the installed llama.cpp runtime with this release's "
                       f"pin ({e}); it was left as it is")
        return out


# ------------------------------ who holds it ---------------------------------

def _rm_holders(files: list) -> list:
    """Processes holding *files*, from the Windows Restart Manager:
    ``[{"pid", "name"}]``. Empty when it names none or cannot be asked."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes
    rm = ctypes.WinDLL("rstrtmgr")

    class _UniqueProcess(ctypes.Structure):
        _fields_ = [("dwProcessId", wintypes.DWORD), ("ProcessStartTime", wintypes.FILETIME)]

    class _ProcessInfo(ctypes.Structure):
        _fields_ = [("Process", _UniqueProcess), ("strAppName", ctypes.c_wchar * 256),
                    ("strServiceShortName", ctypes.c_wchar * 64),
                    ("ApplicationType", ctypes.c_int), ("AppStatus", wintypes.ULONG),
                    ("TSSessionId", wintypes.DWORD), ("bRestartable", wintypes.BOOL)]

    session = wintypes.DWORD()
    key = ctypes.create_unicode_buffer(33)
    if rm.RmStartSession(ctypes.byref(session), 0, key) != 0:
        return []
    try:
        names = (ctypes.c_wchar_p * len(files))(*[str(f) for f in files])
        if rm.RmRegisterResources(session, len(files), names, 0, None, 0, None) != 0:
            return []
        needed, count, reasons = wintypes.UINT(0), wintypes.UINT(0), wintypes.DWORD(0)
        rc = rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), None,
                          ctypes.byref(reasons))
        if rc not in (0, 234) or needed.value == 0:
            return []
        infos = (_ProcessInfo * needed.value)()
        count = wintypes.UINT(needed.value)
        if rm.RmGetList(session, ctypes.byref(needed), ctypes.byref(count), infos,
                        ctypes.byref(reasons)) != 0:
            return []
        return [{"pid": int(i.Process.dwProcessId), "name": i.strAppName}
                for i in infos[:count.value]]
    finally:
        rm.RmEndSession(session)


def _proc_holders(lib: Path) -> list:
    """Processes other than this one that map a file under *lib* (Linux):
    ``[{"pid", "name"}]``."""
    found = []
    for pid in sorted(au.mapped_by_others(lib)):
        try:
            name = Path(f"/proc/{pid}/comm").read_text(encoding="utf-8").strip()
        except OSError:
            name = "?"
        found.append({"pid": pid, "name": name})
    return found


def holders(lib) -> list:
    """The processes that hold the runtime at *lib*, so it cannot be replaced
    now: ``[{"pid", "name"}]`` (pid None when Windows reports files in use but
    cannot say by whom). Empty when nothing holds it. Never raises."""
    lib = Path(lib)
    try:
        if sys.platform == "win32":
            held = au.held_files(lib)
            if not held:
                return []
            try:
                found = _rm_holders(held)
            except Exception as e:
                from localm.debuglog import logger
                logger.warning("update: could not ask which process holds the runtime: %s", e)
                found = []
            return found or [{"pid": None, "name": f"an unidentified process ({held[0].name})"}]
        if sys.platform.startswith("linux") and lib.is_dir():
            return _proc_holders(lib)
        return []
    except Exception as e:
        from localm.debuglog import logger
        logger.warning("update: could not check whether the runtime is in use: %s", e)
    return []


def describe_holders(found: list) -> str:
    """*found* (from :func:`holders`) as text: ``name (pid N), ...``."""
    return ", ".join(f"{h['name']} (pid {h['pid']})" if h.get("pid") else str(h["name"])
                     for h in found)


# ------------------------------ pending marker -------------------------------

def _updates_dir() -> Path:
    from localm import updater
    return updater._updates_dir(create=True)


def write_pending(p: dict, *, root, version: str) -> None:
    """Record the re-provision :func:`plan` decided for the install at *root*,
    for :func:`finish_pending`. Raises OSError when it cannot be written."""
    data = {"root": str(Path(root).resolve()), "version": version,
            "backend": p["backend"], "from": p["current"], "target": p["target"],
            "pins": p["pins"]}
    path = _updates_dir() / PENDING_FILE
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def read_pending() -> Optional[dict]:
    """The pending re-provision, or None. A marker that cannot be parsed is
    reported once in the log and treated as none."""
    from localm import updater
    path = updater._updates_dir(create=False) / PENDING_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, RecursionError) as e:
        from localm.debuglog import logger
        logger.warning("update: ignoring an unreadable pending runtime update %s: %s", path, e)
        return None
    if not isinstance(data, dict) or not data.get("backend") or not data.get("target"):
        return None
    return data


def clear_pending() -> None:
    from localm import updater
    try:
        (updater._updates_dir(create=False) / PENDING_FILE).unlink()
    except FileNotFoundError:
        pass


def _write_status(status: dict) -> None:
    try:
        (_updates_dir() / STATUS_FILE).write_text(json.dumps(status), encoding="utf-8")
    except OSError as e:
        from localm.debuglog import logger
        logger.warning("update: could not record the runtime update result: %s", e)


def last_status() -> Optional[dict]:
    """The result of the last re-provision :func:`finish_pending` attempted, or None."""
    from localm import updater
    try:
        data = json.loads((updater._updates_dir(create=False) / STATUS_FILE)
                          .read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


# ------------------------------ the re-provision -----------------------------

def _command(root: Path, backend: str) -> list:
    cmd = [sys.executable, "-m", "localm", "setup-llama", "--backend", backend,
           "--force", "--yes", "--keep-choice"]
    text = ""
    for rel in ("localm/setup_llama/cli.py", "localm/setup_llama.py"):
        try:
            text += (root / rel).read_text(encoding="utf-8")
        except OSError:
            continue
    if '"--keep-choice"' not in text:
        cmd.remove("--keep-choice")
    return cmd


def finish_pending(*, root=None, runner: Optional[Callable] = None,
                   env: Optional[dict] = None) -> Optional[dict]:
    """Carry out the pending re-provision, if there is one for the install at
    *root* (default the running install).

    Returns None when nothing is pending, else a status ``{"state", "backend",
    "from", "to", "holders", "error", "note"}`` (also recorded for ``localm
    doctor``). ``state`` is:

    - ``"updated"``: the runtime now is the release's build;
    - ``"current"``: it already was, or the user has since chosen a build;
    - ``"waiting"``: a process holds the runtime files; nothing was changed and
      the re-provision stays pending for the next start;
    - ``"failed"``: the re-provision did not produce the release's build for the
      same backend; the previous runtime was put back (``error`` says what
      failed, including a restore that failed).

    *runner* runs the setup-llama argv and returns its exit code (tests);
    *env* is its environment."""
    import subprocess

    from localm import setup_llama as sl
    from localm import updater
    from localm.setup_llama.versions import runtime_target
    pending = read_pending()
    if pending is None:
        return None
    root = Path(root) if root else updater.repo_root()
    if Path(pending.get("root") or "").resolve() != root.resolve():
        return None
    lib = runtime_lib(root)
    backend, target, pins = pending["backend"], pending["target"], pending.get("pins") or {}
    status = {"state": None, "backend": backend, "from": pending.get("from"), "to": target,
              "holders": [], "error": None, "note": None}

    _t, source = runtime_target(backend, pin=sl.pinned_tag(), tracking=sl.tracks_latest(),
                                pinned_tag=pins.get("pinned_tag"),
                                rocm_build_id=pins.get("rocm_build"))
    installed = sl._provisioned_backend(lib)
    if source != "release" or installed != backend or _is_current(
            lib, backend, sl._provisioned_build(lib), target, pins):
        clear_pending()
        status.update(state="current", note=f"the {installed or backend} runtime was left as "
                                            "it is (it is current or was changed since)")
        _write_status(status)
        return status

    found = holders(lib)
    if found:
        who = describe_holders(found)
        status.update(state="waiting", holders=found, note=(
            f"the {backend} runtime is in use by {who}, so it was not changed; the update to "
            f"{target} finishes on the next start of localm once that process has closed "
            f"(or run: {_manual_command(backend)})"))
        _write_status(status)
        return status

    snapshot = au.runtime_snapshot_dir(updater._backup_dir(create=True))
    try:
        au.snapshot_runtime(lib, snapshot)
    except OSError as e:
        clear_pending()
        status.update(state="failed", error=f"could not keep a copy of the current runtime: {e}",
                      note=(f"the {backend} runtime was not re-provisioned because a copy of "
                            f"the current one could not be kept ({e}); to update it: "
                            f"{_manual_command(backend)}"))
        _write_status(status)
        return status

    cmd = _command(root, backend)
    if env is None:
        env = updater._deps_env(root)
    run = runner or (lambda c: subprocess.run(c, cwd=str(root), env=env).returncode)
    try:
        rc = int(run(cmd))
        problem = None if rc == 0 else f"setup-llama exited {rc}"
    except Exception as e:
        problem = f"could not run setup-llama: {e}"
    now_backend, now_build = sl._provisioned_backend(lib), sl._provisioned_build(lib)
    if problem is None and now_backend != backend:
        problem = f"setup-llama installed the {now_backend} runtime instead of {backend}"
    if problem is None and not _is_current(lib, backend, now_build, target, pins):
        problem = f"setup-llama installed {now_build or 'an unrecorded build'}, not {target}"
    clear_pending()
    if problem is None:
        status.update(state="updated", note=f"re-provisioned the {backend} runtime: "
                                            f"{pending.get('from')} -> {target}")
        _write_status(status)
        return status
    restore_errors = au.restore_runtime(snapshot, lib)
    kept = ("the previous runtime was put back" if not restore_errors
            else "putting the previous runtime back FAILED: " + "; ".join(restore_errors))
    status.update(state="failed", error=problem + ("; " + "; ".join(restore_errors)
                                                   if restore_errors else ""),
                  note=(f"the {backend} runtime was NOT updated to {target} ({problem}); "
                        f"{kept}. To retry: {_manual_command(backend)}"))
    from localm.debuglog import logger
    logger.warning("update: runtime re-provision failed: %s", status["error"])
    _write_status(status)
    return status
