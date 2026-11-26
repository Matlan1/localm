# SPDX-License-Identifier: AGPL-3.0-or-later
"""The repair of the native runtime that ``localm integrity --repair`` and the Settings
Repair carry out when files of it differ from its list (``localm.runtime_files``):
setup-llama installs the installed backend and build again, as an update re-provisions
the runtime, and the runtime is compared with its new list in full.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Optional

#: Lines of setup-llama's output a failed repair keeps for its reason.
_TAIL_LINES = 40
#: What a refusal tells the user to run once its cause is gone.
RETRY = "localm integrity --repair"


def differing(lib) -> list:
    """The files of the runtime in *lib* that differ from its list or are missing,
    every file read in full (``runtime_files.compare``)."""
    from localm import runtime_files
    rt = runtime_files.compare(lib, full=True)
    return rt["differ"] + rt["missing"]


def command(backend: str) -> list:
    """The setup-llama argv of an update's re-provision of *backend* in this release
    (``updater_runtime._command`` of this tree): every build fetched again, no prompt,
    the recorded backend choice kept, no other build or backend tried."""
    return [sys.executable, "-m", "localm", "setup-llama", "--backend", backend, "--force",
            "--yes", "--keep-choice", "--only"]


def _run(cmd: list, root: Path, say: Callable) -> tuple:
    """Run the setup-llama argv *cmd* in the install at *root* without a window, with the
    environment of an update's re-provision (``updater._deps_env``), passing each line it
    prints to *say*. Returns ``(exit code, last lines)``. Raises OSError when it cannot
    be started."""
    import subprocess

    from localm import updater
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    lines: list = []
    with subprocess.Popen(cmd, cwd=str(root), env=updater._deps_env(root),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", **flags) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line.strip():
                say(line)
                lines.append(line)
                del lines[:-_TAIL_LINES]
    return proc.returncode, lines


def _with_build(cmd: list, backend: str, lib: Path) -> tuple:
    """``(argv, choice before)``: *cmd* with ``--tag`` naming the build installed in *lib*
    when that is not the build setup-llama installs for *backend* now (the user follows
    upstream's newest, or an update kept the previous build), and then the stored choice
    of build (``llama_runtime_pin``) to put back afterwards; else *cmd* and None."""
    from localm import config
    import localm.setup_llama as sl
    from localm.setup_llama.versions import runtime_target
    build = sl._provisioned_build(lib)
    if backend == "amd-rocm" or not build or not sl.is_safe_tag(build):
        return cmd, None
    target, _source = runtime_target(backend, pin=sl.pinned_tag(),
                                     tracking=sl.tracks_latest())
    if target == build:
        return cmd, None
    return cmd + ["--tag", build], str(config.load_config().get("llama_runtime_pin") or "")


def _put_choice_back(value: str) -> None:
    """Store *value* again as the user's choice of llama.cpp build, saying in the log
    when it cannot be written."""
    import localm.setup_llama as sl
    try:
        sl.set_pinned_tag(value)
    except Exception as e:
        from localm.debuglog import logger
        logger.warning("runtime repair: the choice of llama.cpp build could not be put back "
                       "(%r): %s", value, e)


def repair_runtime(*, root=None, say: Optional[Callable] = None,
                   runner: Optional[Callable] = None) -> dict:
    """Install the native runtime of the install at *root* (default the running one)
    again when files of it differ from its list or are missing: setup-llama for the
    installed backend and build (:func:`command`, :func:`_with_build`), which tests the
    new copy in a folder of its own and leaves the installed one in place when it
    fails; then the runtime is compared with its new list in full.

    Returns ``{"needed", "repaired", "backend", "build", "files", "error", "fix",
    "note"}``: ``needed`` whether files differed, ``files`` those files, ``repaired``
    whether the runtime now matches its list, ``error`` why it was not repaired and
    ``fix`` what repairs it (both None when it was), ``note`` what is still to do after
    a repair, or None. setup-llama is not run when network access is off, the backend
    is not recorded or is a build the user supplied, an update of the install is in
    progress, or a program holds the runtime (setup-llama's own check, which stops this
    install's runtime probe daemons first). *say* receives each step and each line
    setup-llama prints; *runner* stands in for running the setup-llama argv:
    ``(argv, root, say) -> (exit code, last lines)``."""
    from localm import _apply_update as au
    from localm import updater
    from localm.bugreport import LocalmError
    import localm.setup_llama as sl
    from localm.setup_llama._common import _net_allowed
    from localm.setup_llama.runtime_dir import RuntimeInUseError
    from localm.setup_spec import VC_RUNTIME_ABSENT_EXIT
    from localm.update_mechanism import job
    say = say or (lambda _m: None)
    root = Path(root) if root is not None else updater.repo_root()
    lib = root.joinpath(*au.RUNTIME_LIB.split("/"))
    out = {"needed": False, "repaired": False, "backend": None, "build": None, "files": [],
           "error": None, "fix": None, "note": None}
    files = differing(lib)
    if not files:
        return out
    backend = sl._provisioned_backend(lib)
    build = sl._provisioned_build(lib)
    out.update(needed=True, files=files, backend=backend, build=build)

    def refuse(error: str, fix: str) -> dict:
        out.update(error=error, fix=fix)
        return out

    if not backend:
        return refuse("the native runtime does not record which backend it is",
                      "localm setup-llama --force")
    if backend == "custom":
        return refuse("it is a build you installed yourself (setup-llama --from or "
                      "--url), and no download holds it",
                      "localm setup-llama --from <build folder> --force")
    if not _net_allowed():
        from localm.config import NET_OFF_DOWNLOAD_REMEDY
        return refuse("network access is off, so the runtime cannot be downloaded",
                      f"{NET_OFF_DOWNLOAD_REMEDY}; then run: {RETRY}")
    if job.record_path(root).exists() or job.locked(root):
        return refuse("an update of this install is in progress",
                      f"once the update has finished, run: {RETRY}")
    try:
        sl.refuse_runtime_in_use(lib)
    except RuntimeInUseError as e:
        return refuse(f"the native runtime is in use ({e})",
                      "close the program that uses it (a localm with a model loaded, for "
                      f"one), then run: {RETRY}")
    cmd, choice_before = _with_build(command(backend), backend, lib)
    manual = f"localm setup-llama --backend {backend} --force"
    try:
        with updater._apply_lock("repair"):
            say(f"Installing the {backend} runtime{' ' + build if build else ''} again...")
            try:
                rc, tail = (runner or _run)(cmd, root, say)
            except OSError as e:
                return refuse(f"setup-llama could not be started: {e}", manual)
            finally:
                if choice_before is not None:
                    _put_choice_back(choice_before)
    except LocalmError as e:
        return refuse(f"{e.summary} ({e.reason})", f"once it has finished, run: {RETRY}")
    if rc not in (0, VC_RUNTIME_ABSENT_EXIT):
        last = f": {tail[-1][:300]}" if tail else ""
        return refuse(f"setup-llama exited {rc}{last}; the runtime that was installed "
                      "stays in place", manual)
    out["backend"] = sl._provisioned_backend(lib) or backend
    out["build"] = sl._provisioned_build(lib) or build
    left = differing(lib)
    if left:
        more = f" and {len(left) - 3} more" if len(left) > 3 else ""
        return refuse("after it was installed again, these files still differ from its "
                      "list: " + ", ".join(left[:3]) + more, manual)
    if rc == VC_RUNTIME_ABSENT_EXIT:
        out["note"] = ("no model can load until the Microsoft Visual C++ Redistributable is "
                       "installed: run 'localm setup-llama', which offers to install it")
    out["repaired"] = True
    return out
