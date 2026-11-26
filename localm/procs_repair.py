# SPDX-License-Identifier: AGPL-3.0-or-later
"""The repair of a start whose process record (:mod:`localm.procs`) could not be
imported.

``localm/__init__.py`` lets such a start through its update gate without the record
and then calls :func:`repair_and_enter` with the import error. It puts back the files
of the install that the error names, and ``localm/procs.py``, when they differ from
the installed release, from the signed build of that release
(:func:`localm.integrity.repair`). When the error is a Python package that is not
installed, it runs the repair of the installed parts (:func:`localm.updater_parts.repair`,
the one of Settings > System > Updates > Installed parts > Repair). After each repair
it imports the process record again and, once that works, records this process
(``procs.enter``) and says in one line on stderr what was repaired. When nothing could
be repaired, the start goes on without the record, and a warning on stderr and in the
debug log names the import error, why the repair could not run, and what repairs it.

Nothing of this runs while the process record imports."""

from __future__ import annotations

import functools
import logging
import os
import sys
from pathlib import Path

#: Seconds each network wait of a repair at a start may last: the request that names
#: the release (the update proxy's own timeout, ``_proxy.request``, which is this), each
#: read of the download of the signed build, and each request uv makes for a package
#: (``UV_HTTP_TIMEOUT``, when the environment does not set it).
NET_S = 15

#: How many repairs one start tries: each set of damaged files once, the installed
#: parts once.
_ROUNDS = 4

_PROCS = "localm/procs.py"
_PARTS = "Settings > System > Updates > Installed parts > Repair"


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _say(text: str, level: int = logging.WARNING) -> None:
    """Print *text* on stderr and queue it at *level* for the debug log."""
    if sys.stderr is not None:
        try:
            print(f"localm: {text}", file=sys.stderr, flush=True)
        except (OSError, ValueError):
            pass
    try:
        from localm.debuglog import defer_log
        defer_log(level, "procs repair: %s", text)
    except Exception as e:
        if sys.stderr is not None:
            try:
                print(f"localm: the debug log cannot record this: {e!r}", file=sys.stderr,
                      flush=True)
            except (OSError, ValueError):
                pass


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _rel(root: Path, path) -> str:
    """*path* as a posix path inside the install at *root*, or "" outside it."""
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (OSError, ValueError, TypeError, RuntimeError):
        return ""


def named_files(error: BaseException, root: Path) -> list:
    """The files of localm in the install at *root* that *error* names, then
    ``localm/procs.py``: the file of a missing ``localm`` module, the file of a
    ``SyntaxError``, and otherwise the last file of localm in the traceback (the one
    whose code raised it)."""
    import traceback
    out = []
    name = getattr(error, "name", None)
    if isinstance(error, ImportError) and isinstance(name, str) and name.startswith("localm."):
        out += [name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"]
    elif isinstance(error, SyntaxError):
        out += [_rel(root, error.filename)] if error.filename else []
    else:
        inside = [_rel(root, f.filename) for f in traceback.extract_tb(error.__traceback__)]
        out += [rel for rel in inside if rel.startswith("localm/")][-1:]
    out.append(_PROCS)
    return [rel for rel in dict.fromkeys(out) if rel.startswith("localm/")]


def missing_package(error: BaseException) -> str:
    """The top-level name of the package whose absence *error* is, or ""."""
    name = getattr(error, "name", None) if isinstance(error, ModuleNotFoundError) else None
    if not isinstance(name, str) or not name:
        return ""
    top = name.split(".", 1)[0]
    return "" if top == "localm" else top


def _damaged(files: list, root: Path) -> tuple:
    """``(damaged, kept, reason)``: those of *files* the release states whose content
    in the install is not the stated one, apart from those the user keeps as their own
    (``integrity.keep``), which are *kept*, and None; or ``[]``, ``[]`` and why there
    is no release to compare with."""
    from localm import integrity, runtime_files
    manifest, reason = integrity.statement(root)
    if manifest is None:
        return [], [], reason
    mine = integrity.kept()
    out, kept = [], []
    for rel in files:
        entry = manifest["files"].get(rel)
        if entry is None:
            continue
        p = root.joinpath(*rel.split("/"))
        try:
            same = p.stat().st_size == entry["size"] and runtime_files.sha256(p) == entry["sha256"]
        except OSError:
            same = False
        if not same:
            (kept if mine.get(rel) == entry["sha256"] else out).append(rel)
    return out, kept, None


def _import_procs():
    """``(procs module, None)``, or ``(None, the error)``."""
    import importlib
    import importlib.util
    sys.modules.pop("localm.procs", None)
    try:
        os.unlink(importlib.util.cache_from_source(str(_root() / "localm" / "procs.py")))
    except OSError:
        pass
    importlib.invalidate_caches()
    try:
        import localm.procs as procs
    except Exception as e:
        return None, e
    if not (callable(getattr(procs, "enter", None)) and callable(getattr(procs, "current", None))):
        return None, ImportError("localm.procs has no enter and current", name="localm.procs")
    return procs, None


def _repair_files(files: list) -> str:
    """Repair *files* from the signed release and return the line that says so.
    Raises IntegrityError or LocalmError with why nothing was repaired."""
    from localm import integrity
    from localm.bugreport import LocalmError
    res = integrity.repair(files, timeout=float(NET_S))
    if not res["repaired"]:
        raise LocalmError("no file was repaired", reason="; ".join(
            f"{f['path']}: {f['reason']}" for f in res["failed"]) or "nothing to write")
    tail = ("; not repaired: " + ", ".join(f["path"] for f in res["failed"])
            if res["failed"] else "")
    return (f"repaired {', '.join(res['repaired'])} from the signed release "
            f"{res['version']}{tail}")


def _repair_parts(package: str) -> str:
    """Run the repair of the installed parts when they lack a dependency, and return
    the line that says so once *package* can be imported. Raises LocalmError with why
    it was not installed: an update of the install in progress, parts that lack
    nothing, or a repair that did not bring it."""
    import importlib
    import importlib.util

    from localm import updater_parts
    from localm.bugreport import LocalmError
    from localm.update_mechanism import job
    root = _root()
    if job.record_path(root).exists() or job.locked(root):
        raise LocalmError(f"{package} was not installed",
                          reason="an update of this install is in progress")
    parts = updater_parts.installed_parts(root)
    if not parts["deps_problems"] and parts["install_record_matches"] is not False:
        raise LocalmError(f"the installed parts lack no package, so {package} was not "
                          "installed", reason="the dependencies of this localm do not name it")
    set_timeout = "UV_HTTP_TIMEOUT" not in os.environ
    if set_timeout:
        os.environ["UV_HTTP_TIMEOUT"] = str(NET_S)
    try:
        res = updater_parts.repair(defer=True)
    finally:
        if set_timeout:
            os.environ.pop("UV_HTTP_TIMEOUT", None)
    importlib.invalidate_caches()
    if importlib.util.find_spec(package) is None:
        raise LocalmError(f"the repair of the installed parts did not install {package}",
                          reason="; ".join(res["notes"]) or "it changed nothing")
    return f"installed the missing package {package} ({'; '.join(res['notes'])})"


def _fix(files: list, damaged: list, package: str, no_list, root: Path) -> str:
    """What the user runs to repair it: the repair of the installed parts for a
    missing *package* when the release states the named files as they are, the one
    that puts the files back when it states other content for them, and both when
    there is no release to compare with."""
    parts = f"{_PARTS}, or 'localm update --repair'"
    if package and not no_list and not damaged:
        return parts
    if damaged:
        package = ""
    names = " ".join(damaged or files)
    if (root / ".git").exists():
        put_back = f"'git checkout -- {names}' in {root}"
    elif no_list:
        put_back = f"unzip the release of this version over {root}"
    else:
        put_back = f"'localm integrity --repair {names}'"
    return f"{put_back}; when the package {package} is missing: {parts}" if package else put_back


def repair_and_enter(error: BaseException):
    """Repair what makes ``import localm.procs`` fail with *error*, import it again
    and record this process. Returns the process record module, or None, after the
    warning that says why, when it still cannot be imported. Raises nothing but
    KeyboardInterrupt and SystemExit."""
    root = _root()
    first, done, why = error, [], []
    tried: set = set()
    files: list = []
    package, no_list, damaged, kept = "", None, [], []
    for _round in range(_ROUNDS):
        files = list(dict.fromkeys([*files, *named_files(error, root)]))
        package = missing_package(error)
        try:
            damaged, kept, no_list = _damaged(files, root)
        except Exception as e:
            damaged, kept = [], []
            no_list = f"the files could not be compared with the release ({e})"
        if damaged and ("files", *damaged) not in tried:
            tried.add(("files", *damaged))
            step = functools.partial(_repair_files, damaged)
        elif package and not damaged and ("parts",) not in tried:
            tried.add(("parts",))
            step = functools.partial(_repair_parts, package)
        else:
            if kept:
                why.append(f"{', '.join(kept)} is kept as yours ('localm integrity --keep'), "
                           "so it was not written again")
                damaged = kept
            elif no_list:
                why.append(f"the files cannot be repaired: {no_list}")
            elif not damaged and not package:
                why.append("every file it names is the release's, and it is no missing "
                           "package")
            break
        try:
            done.append(step())
        except Exception as e:
            reason = getattr(e, "reason", "")
            why.append(f"{getattr(e, 'summary', None) or e}" + (f" ({reason})" if reason else ""))
        procs, again = _import_procs()
        if again is not None:
            error = again
        elif procs is not None:
            procs.enter()
            _say(("; ".join(done) or "localm/procs.py imports now (another start repaired "
                  "it)") + f" (it could not be imported: {_error_text(first)})", logging.INFO)
            return procs
    _say(f"the list of running processes (localm/procs.py) cannot be imported: "
         f"{_error_text(error)}. {'; '.join(done + why)}. This start goes on without it, "
         f"and a command that needs it stops with this error; to repair it: "
         f"{_fix(files, damaged, package, no_list, root)}.")
    return None
