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
#: Seconds to wait for a running update when nobody sits at a terminal, and those after
#: which the program that is updating is named.
_WAIT_S, _MCP_WAIT_S, _NAME_S = 3600.0, 60.0, 60.0


def _gate(me=None) -> bool:
    """Whether the files of localm may have been replaced while this waited. *me* is the
    :class:`localm.procs.Entry` of this process."""
    # A server that restarted itself (``LOCALM_RESTARTED_FROM``, ``<pid>-<start>`` of the
    # process it was) says so in the file that says it waits here; nothing it starts
    # gets the variable.
    restarted = os.environ.pop("LOCALM_RESTARTED_FROM", "")
    if not (restarted.isascii() and restarted.replace("-", "").replace(".", "").isalnum()):
        restarted = ""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, ".localm-update")
    record = os.path.join(folder, "job.json")
    try:
        os.lstat(record)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as e:
        _say(f"{record} cannot be read: {e}. Start LocaLM with the account that installed it.")
        sys.exit(1)
    if sys.version_info[:2] not in _PYTHONS:
        _say(f"An update of LocaLM in {root} is not finished, and this Python "
             f"({sys.version.split()[0]}) cannot run LocaLM. Start LocaLM by its own launcher.")
        sys.exit(1)
    import time
    mark = me and os.path.join(folder, f"waiting-{me.pid}-{me.start}")
    begun = time.monotonic()
    end = begun + (float("inf") if _tty(sys.stderr) and _tty(sys.stdin)
                   else _MCP_WAIT_S if me and me.purpose == "mcp" else _WAIT_S)
    name_at = min(begun + _NAME_S, end)
    waited = False
    try:
        while os.path.lexists(record):
            update, phase = _job(record)
            if me and phase == "setup" and os.environ.get("LOCALM_SETUP") == "1":
                return waited
            held = _held(folder)
            if held and update and _let_through(folder, update, phase, me):
                return waited
            if not held and os.environ.get("LOCALM_UNINSTALL") == "1":
                return waited
            if mark and not _touch(mark, update if os.environ.get(
                    "LOCALM_UPDATE_PRIMARY") == update else
                    f"restarted {restarted}" if restarted else ""):
                mark = None
            if held:
                if not waited:
                    waited = True
                    _say("LocaLM is being updated. Waiting until that has ended...")
                if time.monotonic() < name_at:
                    time.sleep(0.2)
                    continue
                name_at = end
            elif phase != "setup":
                _say("An update of LocaLM was interrupted. Putting the previous version back...")
            waited = True
            window = (_window(root, me) and not _chosen(folder, update, me)
                      and os.environ.get("LOCALM_UPDATE_PRIMARY") != update)
            code = _put_back(root, folder, update, window)
            if code == 0:
                break
            if code != 3 and _denied(folder):
                account = os.environ.get("USERNAME") or os.environ.get("USER") or "this account"
                _say(f"The account {account} may not open {os.path.join(folder, 'lock')}, so it "
                     f"cannot tell whether an update of LocaLM runs. Start LocaLM with the account "
                     f"that installed it.")
                sys.exit(1)
            if window or (code != 3 and not _held(folder)) or time.monotonic() >= end:
                sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        os.environ.pop("LOCALM_UPDATE_PRIMARY", None)
        if mark and os.path.lexists(mark):
            _touch(mark, remove=True)
    return waited


def _touch(mark: str, update: str = "", remove: bool = False) -> bool:
    """Make or *remove* the file that says that this process waits here; it holds *update*:
    the id of the update this process goes on with for the user
    (``LOCALM_UPDATE_PRIMARY``), ``restarted <pid>-<start>`` of the server this process
    restarted from, or nothing.
    The restore program, which ends what an update left running, does not end the
    waiting ones."""
    try:
        if remove:
            os.unlink(mark)
        else:
            with open(mark, "wb" if update else "ab") as f:
                f.write(update.encode("ascii"))
    except OSError as e:
        _say(f"{mark}: {e}")
        return False
    return True


def _say(text: str) -> None:
    """Print *text* on stderr, as long as that can be written to."""
    if sys.stderr is not None:
        try:
            print(text, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            sys.stderr = None


def _tty(stream) -> bool:
    try:
        return bool(stream and stream.isatty())
    except ValueError:
        return False


def _window(root: str, me) -> bool:
    """Whether this is a start of the GUI from a menu of the desktop, where nothing that
    is printed is seen."""
    return bool(me and me.purpose == "gui" and os.name == "posix"
                and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
                and not any(map(_tty, (sys.stdin, sys.stdout, sys.stderr)))
                and os.path.isfile(os.path.join(root, "scripts", "update_gate.pyw")))


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
    """Whether a process holds the lock of the update, or that cannot be told."""
    lock, flags = os.path.join(folder, "lock"), getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            fd = os.open(lock, os.O_RDWR | flags)
        except PermissionError:
            fd = os.open(lock, os.O_RDONLY | flags)
    except OSError as e:
        return isinstance(e, PermissionError)
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


def _denied(folder: str) -> bool:
    """Whether this account may not open the lock of the update at all."""
    try:
        os.close(os.open(os.path.join(folder, "lock"), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)))
    except PermissionError:
        return True
    except OSError:
        return False
    return False


def _chosen(folder: str, update: str, me) -> bool:
    """Whether the running *update* chose this process (it wrote its id and this one's pid
    and start into ``started``), or such a process started this one."""
    if os.environ.get("LOCALM_UPDATE_STARTED") == update:
        return True
    try:
        with open(os.path.join(folder, "started"), encoding="ascii") as f:
            return bool(me) and f.read(200).strip() == f"{update} {me.pid}-{me.start}"
    except (OSError, ValueError):
        return False


def _let_through(folder: str, update: str, phase: str, me) -> bool:
    """Whether the running *update* started this process or chose it."""
    if os.environ.get("LOCALM_UPDATE_ID") == update and phase in ("steps", "starting"):
        return True
    chosen = phase == "starting" and _chosen(folder, update, me)
    if chosen:
        os.environ["LOCALM_UPDATE_STARTED"] = update
    return chosen


def _put_back(root: str, folder: str, update: str, window: bool) -> int:
    """Run the restore program of *update*, or the *window* that does, and return its exit
    status."""
    import subprocess
    if window:
        return subprocess.call([sys.executable, "-I", "-S",
                                os.path.join(root, "scripts", "update_gate.pyw")])
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
             f"it. Unzip LocaLM over that folder, delete everything in {folder} except kept "
             f"(files of your own) and run setup.")
        return 1
    hidden = 0 if _tty(sys.stderr) else getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        [sys.executable, "-I", "-S", *(("-X", "utf8") if sys.flags.utf8_mode else ()), program,
         "restore", root], creationflags=hidden,
        stdin=None if _tty(sys.stdin) else subprocess.DEVNULL, stdout=subprocess.DEVNULL).returncode


if __name__ == "__main__":
    _gate()
else:
    try:
        from localm import procs as _procs
        _me = _procs.enter, _procs.current, _procs.__file__
    except (ImportError, AttributeError):
        # An update into a release without this gate puts its localm/procs.py in place
        # (or takes it out) just before this file: the gate decides without the record.
        _me = None
    if _me is not None:
        _me[0]()
    _round = globals().get("_round", 0) + 1
    if _gate(_me[1]() if _me is not None else None) and _round < 3:
        # The files just put in place are read instead of the ones this began with, whose
        # compiled copies may have the same size and time.
        import importlib.util
        for _file in (__file__, *(_me[2:] if _me is not None else ())):
            try:
                os.unlink(importlib.util.cache_from_source(_file))
            except OSError:
                pass
        sys.modules.pop("localm.procs", None)
        globals().pop("procs", None)
        importlib.invalidate_caches()
        __spec__.loader.exec_module(sys.modules[__name__])
