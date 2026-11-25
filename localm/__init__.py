# SPDX-License-Identifier: AGPL-3.0-or-later
"""The first act of importing localm is to record this process (:mod:`localm.procs`).
Reading the record of an update comes after it and before anything else of localm: an
updater writes its record and then lists the processes, so one side sees the other.

While an update of this install is not finished, :func:`_gate` waits for it, or puts the
old version back when nobody is updating, or ends this process. Run as a program
(``<python> -I -S <this file>``) this file does only that and its exit status says whether
localm may start (0)."""
__version__ = "0.3.4b4"

import os
import sys

#: The Python versions localm runs on (``requires-python`` in pyproject.toml).
_PYTHONS = ((3, 12),)
#: Seconds to wait for a running update when nobody sits at a terminal.
_WAIT_S, _MCP_WAIT_S = 3600.0, 60.0


def _gate(me=None) -> bool:
    """Whether the files of localm may have been replaced while this waited. *me* is the
    :class:`localm.procs.Entry` of this process."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, ".localm-update")
    record = os.path.join(folder, "job.json")
    try:
        os.lstat(record)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as e:
        _say(f"{record} cannot be read: {e}")
    if sys.version_info[:2] not in _PYTHONS:
        _say(f"An update of LocaLM in {root} is not finished, and this Python "
             f"({sys.version.split()[0]}) cannot run LocaLM. Start LocaLM by its own launcher.")
        sys.exit(1)
    import time
    mark = me and os.path.join(folder, f"waiting-{me.pid}-{me.start}")
    end = None if _at_terminal() else time.monotonic() + (
        _MCP_WAIT_S if me and me.purpose == "mcp" else _WAIT_S)
    waited = False
    try:
        while os.path.lexists(record):
            update, phase = _job(record)
            held = _held(folder)
            if held and update and _let_through(folder, update, phase, me):
                return waited
            if mark:
                _touch(mark)
            if held:
                if not waited:
                    waited = True
                    _say("LocaLM is being updated. Waiting until that has ended...")
                if end is None or time.monotonic() < end:
                    time.sleep(0.2)
                    continue
            code = _put_back(root, folder, update)
            waited = True
            if code == 0:
                break
            if (code != 3 and not _held(folder)) or (
                    end is not None and time.monotonic() >= end):
                sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        if mark and os.path.lexists(mark):
            _touch(mark, remove=True)
    return waited


def _touch(mark: str, remove: bool = False) -> None:
    """Make or *remove* the file that says that this process waits here. The restore
    program, which ends what an update left running, does not end the waiting ones."""
    try:
        os.unlink(mark) if remove else open(mark, "ab").close()
    except OSError as e:
        _say(f"{mark}: {e}")


def _say(text: str) -> None:
    if sys.stderr is not None:
        print(text, file=sys.stderr, flush=True)


def _at_terminal() -> bool:
    return bool(sys.stderr and sys.stdin and sys.stderr.isatty() and sys.stdin.isatty())


def _is_id(name) -> bool:
    return isinstance(name, str) and len(name) == 32 and set(name) <= set("0123456789abcdef")


def _job(record: str) -> tuple:
    """The id and the phase that the record names, two empty strings when it is damaged."""
    import json
    try:
        with open(record, encoding="utf-8") as f:
            raw = json.loads(f.read(65536))
        return (raw["id"], str(raw["phase"])) if _is_id(raw["id"]) else ("", "")
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return "", ""


def _held(folder: str) -> bool:
    """Whether a process holds the lock of the update."""
    try:
        fd = os.open(os.path.join(folder, "lock"), os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return False
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def _let_through(folder: str, update: str, phase: str, me) -> bool:
    """Whether the running *update* started this process or chose it (it wrote its pid and
    start into ``<id>/started``), or such a process started this one."""
    if os.environ.get("LOCALM_UPDATE_ID") == update and phase in ("steps", "starting"):
        return True
    chosen = os.environ.get("LOCALM_UPDATE_STARTED") == update
    if phase == "starting" and me and not chosen:
        try:
            with open(os.path.join(folder, update, "started"), encoding="ascii") as f:
                chosen = f.read(100).strip() == f"{me.pid}-{me.start}"
        except (OSError, ValueError):
            return False
    if phase == "starting" and chosen:
        os.environ["LOCALM_UPDATE_STARTED"] = update
    return phase == "starting" and chosen


def _put_back(root: str, folder: str, update: str) -> int:
    """Run the restore program of *update* and return its exit status."""
    import subprocess
    if not update:
        ids = [n for n in os.listdir(folder) if _is_id(n)]
        update = ids[0] if len(ids) == 1 else ""
    places = [(".localm-update", update, "applier"), (".localm-update", update, "new", "localm")]
    for place in [*(places if update else ()), ("localm",)]:
        program = os.path.join(os.path.realpath(root), *place, "update_mechanism")
        if (os.path.isfile(os.path.join(program, "__main__.py"))
                and os.path.realpath(program) == program):
            break
    else:
        _say(f"An update of LocaLM in {root} was interrupted and nothing is left that can undo "
             f"it. Unzip LocaLM over that folder, delete {folder} and run setup.")
        return 1
    shown = sys.platform != "win32" or (sys.stderr and sys.stderr.isatty())
    hidden = 0 if shown else subprocess.CREATE_NO_WINDOW
    _say("An update of LocaLM was interrupted. Putting the previous version back...")
    return subprocess.run(
        [sys.executable, "-I", "-S", program, "restore", root], creationflags=hidden,
        stdin=None if _at_terminal() else subprocess.DEVNULL, stdout=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    _gate()
else:
    from localm import procs as _procs

    _procs.enter()
    _round = globals().get("_round", 0) + 1
    if _gate(_procs.current()) and _round < 3:
        # The files just put in place are read instead of the ones this began with.
        import importlib
        sys.modules.pop("localm.procs", None)
        globals().pop("procs", None)
        importlib.invalidate_caches()
        __spec__.loader.exec_module(sys.modules[__name__])
