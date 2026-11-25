# SPDX-License-Identifier: AGPL-3.0-or-later
"""The restore program (``restore <root>``): puts the install at *root* back after an
update that was interrupted, or ends one that was done. Text goes to stderr. At a
terminal it asks what to do about what it cannot change; without one it waits its full
time and then says what is in the way."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from . import disk, job, mover, shared, steps, steps_update

#: Exit status: localm may start; it may not; another process is updating the install.
OK, FAILED, BUSY = 0, 1, 3
#: The phases in which nothing of the install was changed or the new version is in use.
_LEAVES_RUNNING = (job.ENDING, job.DONE, job.ENDED)
#: The phase in the record of an install whose setup has to install the packages before
#: anything of localm starts (``localm/__init__.py`` lets what setup starts through).
SETUP = "setup"
#: Seconds a process that has just started has until it says that it waits.
_ARRIVE_S = 1.0
_DOWNLOAD = "https://github.com/Matlan1/localm/releases"
_NOT_BACK = "The interrupted update could not be undone: "


def say(text: str) -> None:
    """Print *text* on stderr, as long as that can be written to."""
    if sys.stderr is not None:
        try:
            print(text, file=sys.stderr, flush=True)
        except (OSError, ValueError):
            sys.stderr = None


def at_terminal() -> bool:
    return bool(sys.stderr and sys.stdin and sys.stderr.isatty() and sys.stdin.isatty())


def _try_again() -> bool:
    """Ask at the terminal. Nothing typed in counts as cancel."""
    while True:
        say("  [T] Try again   [C] Cancel (LocaLM does not start)")
        answer = sys.stdin.readline()
        if answer.strip().lower()[:1] in ("t", "c"):
            return answer.strip().lower().startswith("t")
        if not answer:
            return False


def _ask(held: disk.Held) -> bool:
    say(f"{held}. Close that program, then try again.")
    return _try_again()


def run(root) -> int:
    """Put the install at *root* back and return the exit status."""
    try:
        while True:
            code, lines = _once(root)
            for line in lines:
                say(line)
            if code != FAILED or not at_terminal() or not _try_again():
                return code
    except KeyboardInterrupt:
        say("Stopped. The next start of LocaLM goes on with it.")
        return FAILED


def _once(root) -> tuple:
    """One attempt: the exit status and the lines for the user."""
    try:
        with job.Lock(root) as lock:
            if _waits_for_setup(root) and not in_setup():
                return FAILED, [f"LocaLM's setup did not finish. Run setup in {root} again."]
            notes: list = []
            try:
                if _is_another_release(root):
                    why = ["LocaLM was replaced by another release while an update of it was "
                           "not finished."]
                else:
                    running = _end_running(lock)
                    if running:
                        return FAILED, [_NOT_BACK + line for line in running]
                    result = steps_update.put_back(
                        lock, uv=find_uv(root), env=environment(root), say=say,
                        network=network(root), ask=_ask if at_terminal() else None)
                    notes, why = result.notes, [_NOT_BACK + line for line in result.errors]
                    if not why or result.held:
                        return (FAILED if why else OK), [*notes, *why]
                    if not in_setup():
                        return FAILED, [*notes, *why, *ways_out(root)]
            except mover.Unreadable as e:
                why = [_NOT_BACK + str(e)]
            aside = _set_aside(lock)
            return (OK, [*notes, aside]) if aside else (FAILED, [*notes, *why, *ways_out(root)])
    except job.Busy as e:
        by = ", ".join(f"{h['name']} (pid {h['pid']})"
                       for h in disk.holders_of(mover.update_folder(root) / "lock"))
        return BUSY, [f"{e}{by and ': ' + by}"]
    except PermissionError as e:
        return FAILED, [f"{_NOT_BACK}{e}. Close the program that holds it, or start LocaLM with "
                        f"the account that installed it."]
    except steps.STEP_ERRORS as e:
        return FAILED, [_NOT_BACK + str(e), *([] if isinstance(e, disk.Held) else ways_out(root))]


# ------------------------------------------------------------- what still runs

def _end_running(lock: job.Lock) -> list:
    """End what an interrupted update left running, except the processes that wait at
    the gate for this and the programs they started. Returns a line for each process of
    localm that still runs, the ones above this one among them."""
    procs, record = shared.also("procs"), job.current(lock.root)
    if procs is None or record is None or record.phase in _LEAVES_RUNNING:
        return []
    root, holders = disk.short_path(disk.real_folder(lock.root)), shared.apply_update.rm_holders
    seen: dict = {}

    def arrives(e) -> bool:
        return time.monotonic() - seen.setdefault(e.key, time.monotonic()) < _ARRIVE_S

    def spared(running) -> set:
        found = _waiting(lock.root, running)
        for e in running:
            if arrives(e) if e.recorded else e.parent and e.parent[0] in found:
                found.add(e.pid)
        return found

    try:
        while True:
            left = procs.end_others(root, spare=spared, holders=holders)
            running = procs.list(root, holders=holders)
            waiting = _waiting(lock.root, running)
            left = left or [e for e in running if e.recorded and e.pid not in waiting]
            if not any([arrives(e) for e in left]):
                break
            time.sleep(0.1)
    except OSError as e:
        return [str(e), *(_in_the_way(w) for w in getattr(e, "in_the_way", ()))]
    return [f"a process of LocaLM still runs ({e.purpose}): "
            + "; ".join(map(_in_the_way, e.in_the_way or ({"pid": e.pid, "name": "process"},)))
            for e in left]


def _in_the_way(what: dict) -> str:
    by = f"{what['name']} (pid {what['pid']})" if what.get("pid") else "the program"
    return f"close {by}, which holds {what['path']}" if what.get("path") else f"close {by}"


def _waiting(root, running: list) -> set:
    """The pids of *running* that wait at the gate. Deletes what ended ones left."""
    here = {f"waiting-{e.pid}-{e.start}": e.pid for e in running}
    folder, found = mover.update_folder(root), set()
    for name in os.listdir(folder):
        if name in here:
            found.add(here[name])
        elif name.startswith("waiting-"):
            disk.unlink(folder / name)
    return found


# ------------------------------------------------- an update that cannot be told

def ways_out(root) -> list:
    """What a user can do about an update that can be neither finished nor undone."""
    return [f"LocaLM does not start, because its files may be of two versions. The files the "
            f"update replaced are in {disk.short_path(mover.update_folder(root))}.",
            f"What helps: run setup in {root}. If it says this too, download LocaLM from "
            f"{_DOWNLOAD}, unzip it over {root} and run setup there (your chats, settings and "
            f"models stay). Setup with --uninstall removes LocaLM."]


def _is_another_release(root) -> bool:
    """Whether another release was put over the install: what tells one release from
    another is neither as the unfinished update found it nor as it leaves it."""
    record = job.current(root)
    if (record is None or record.phase in (job.DONE, job.ENDED)
            or not mover.has_plans(root, record.id)):
        return False
    ops = [op for plan in mover.read_plans(root, record.id) if plan.step == mover.FILES
           for op in plan.ops]
    for rel in (shared.update_package.NAME, "localm/__init__.py"):
        known = [i for op in ops if op.path == rel for i in (op.old, op.new, *op.release) if i]
        path = disk.real_folder(root) / rel
        if known and disk.lstat(path) is not None and not any(disk.same(path, i) for i in known):
            return True
    return False


def _waits_for_setup(root) -> bool:
    try:
        raw = disk.read_json(job.record_path(root), job.MAX_RECORD_BYTES)
    except ValueError:
        return False
    return isinstance(raw, dict) and raw.get("phase") == SETUP


def _set_aside(lock: job.Lock) -> str:
    """Put what an update that can be neither finished nor undone left away, when the
    files of the install are exactly those of one release. Under setup the record stays,
    in phase :data:`SETUP`, until setup has made a new Python environment; otherwise this
    does nothing when Python packages were being replaced. Returns the line that says so,
    or an empty string."""
    root = disk.real_folder(lock.root)
    ids = [n for n in job.ids(root) if job.over_of(root, n) is None]
    version = release_of(root)
    if not version or (not in_setup() and any(
            disk.lstat(mover.venv_folder(root, n)) is not None for n in ids)):
        return ""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = mover.update_folder(root)
    if in_setup():
        disk.write_json(job.record_path(root), {"id": "0" * 32, "phase": SETUP})
        kept = disk.real_folder(root / ".venv") / mover.VENV_KEPT
        if disk.lstat(kept) is not None:
            disk.move(kept, folder / "kept" / f"venv-{stamp}", disk.Wait())
    for name in ids:
        os.rename(folder / name, folder / f"aside-{name}-{stamp}")
    if not in_setup() and disk.lstat(job.record_path(root)) is not None:
        os.rename(job.record_path(root), folder / f"aside-{stamp}.json")
    return (f"The files of this install are those of LocaLM {version}. What an interrupted "
            f"update left is put aside in {disk.short_path(folder)}."
            + (" Setup installs its Python packages again." if in_setup() else ""))


def release_of(root) -> str:
    """The version of the release whose statement the install at *root* has, when every
    file it states is there as stated, else an empty string."""
    package = shared.update_package
    try:
        raw = disk.read_file(Path(root) / package.NAME, package.MAX_BYTES)
        stated = package.parse((raw or b"").decode("utf-8"))
        a = mover.area(root, "0" * 32, mover.FILES, 1)
        for rel, identity in stated["files"].items():
            if a.inside(rel):
                path, st = a.find(rel)
                if st is None or not disk.same(path, identity, disk.Wait(0)):
                    return ""
        return stated["version"]
    except (OSError, ValueError, disk.Unsafe, disk.Held):
        return ""


# ---------------------------------------------- what it needs of the install

def in_setup() -> bool:
    return os.environ.get("LOCALM_SETUP") == "1"


def _json(path: Path) -> dict:
    """What the JSON file *path* holds, an empty dict when it is not there. Raises
    OSError, ValueError or :class:`.disk.Held`."""
    raw = disk.read_json(path, 4 * 1024 * 1024)
    if raw is not None and not isinstance(raw, dict):
        raise ValueError(f"{path} is damaged")
    return raw or {}


def _installed(root) -> dict:
    """The record of what setup installed, empty when it cannot be read."""
    try:
        return _json(Path(root) / ".localm-install.json")
    except (OSError, ValueError, disk.Held, disk.Unsafe):
        return {}


def find_uv(root) -> Optional[str]:
    """The uv of the install at *root*: the one inside it, the one its record names, the
    one in its Python environment, or the one of the user. Never one from PATH."""
    root, name = Path(root), "uv.exe" if os.name == "nt" else "uv"
    home = Path(os.path.expanduser("~"))
    for path in (root / ".uv" / name, root / ".uv" / "bin" / name,
                 str(_installed(root).get("uv_exe") or ""),
                 root / ".venv" / ("Scripts" if os.name == "nt" else "bin") / name,
                 home / ".local" / "bin" / name, home / ".cargo" / "bin" / name):
        if (path and os.path.isabs(path) and os.path.isfile(path)
                and (os.name == "nt" or os.access(path, os.X_OK))):
            return str(path)
    return None


def environment(root) -> dict:
    """The environment for uv: this one, with the folders of a self-contained install."""
    env = dict(os.environ)
    if _installed(root).get("runtime_contained"):
        env.setdefault("UV_CACHE_DIR", str(Path(root) / ".cache"))
        env.setdefault("UV_PYTHON_INSTALL_DIR", str(Path(root) / ".python"))
    return env


def data_folder(root) -> Path:
    """The data folder of the install at *root*. Raises OSError, ValueError or
    LookupError when the file that names it cannot be read."""
    named = os.environ.get("LOCALM_HOME", "").strip()
    if named:
        return Path(os.path.expanduser(named)).absolute()
    cfg, reader = Path(root) / "localm-home.cfg", shared.also("_home_cfg")
    line = reader.read_line(cfg) if reader and cfg.is_file() else ""
    return reader.data_dir(line, root) if reader and line else Path(root) / "home"


def network(root) -> bool:
    """Whether localm may use the network here. Setup always may."""
    mode = os.environ.get("LOCALM_NET_MODE", "").strip().lower()
    if not mode and not in_setup():
        try:
            mode = str(_json(data_folder(root) / "config.json").get("net_mode", "")).lower()
        except (OSError, ValueError, LookupError, disk.Held, disk.Unsafe):
            mode = "off"
    return in_setup() or mode.strip() != "off"
