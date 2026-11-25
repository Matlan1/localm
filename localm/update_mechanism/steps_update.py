# SPDX-License-Identifier: AGPL-3.0-or-later
"""The steps of an update after its files are moved: Python packages, the uv inside
the install, the native runtime and the components in the data folder.

:func:`prepare` runs before anything changes, while the old version is in use, from
the new tree: it fetches everything the steps need and tests what can be tested.
:func:`run` carries the steps out in phase STEPS, without a network. Whatever of it
fails, ``job.restore`` puts every step back with the files of the update.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Optional

from . import disk, job, mover, steps, steps_packages
from .mover import Refused, Unreadable

Say = Callable[[str], None]
FORMAT = "localm-update-steps/1"
#: In the update's folder: what ``localm _update-prepare`` prepared, and the steps.
PREPARED, STEPS_FILE = "prepared.json", "steps.json"
#: Seconds a child of a step may print nothing.
SILENCE_S = 1800.0
#: Exit status of ``localm setup-llama --prove`` for a runtime that does not work here.
UNUSABLE_EXIT = 79
RUNTIME, FOLDERS, COMFYUI = "runtime", "folders", "comfyui"


# ----------------------------------------------------------------- children

def run_child(argv: list, *, cwd, env: Mapping, say: Say, silence_s: float = SILENCE_S) -> tuple:
    """Run *argv* without a window and give each line it prints to *say*. Returns
    ``(exit status, its last line)``; the status is None, and the line says why, when it
    could not start or was ended because it printed nothing for *silence_s* seconds."""
    flags: dict = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                   else {"start_new_session": True})
    try:
        proc = subprocess.Popen([str(a) for a in argv], cwd=str(cwd), env=dict(env),
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", bufsize=1, **flags)
    except OSError as e:
        return None, f"{argv[0]} could not be started: {e}"
    lines: queue.SimpleQueue = queue.SimpleQueue()

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    last, heard = "", time.monotonic()
    while True:
        try:
            line = lines.get(timeout=1.0)
        except queue.Empty:
            if time.monotonic() - heard > silence_s:
                _end(proc)
                return None, f"it printed nothing for {silence_s:.0f} s and was ended ({last})"
            if proc.poll() is not None and time.monotonic() - heard > 5:
                break
            continue
        if line is None:
            break
        heard = time.monotonic()
        text = line.rstrip()
        if text:
            last = text
            say(text)
    return proc.wait(), last


def _end(proc: subprocess.Popen) -> None:
    """End *proc* and, outside Windows, its process group."""
    if os.name != "nt":
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
    proc.kill()
    proc.wait()


def child_env(env: Mapping, tree, home, update_id: str) -> dict:
    """The environment of a localm child of an update: the code of *tree*, the data
    folder *home*, plain output."""
    out = {k: v for k, v in env.items() if k not in ("PYTHONPATH", "LOCALM_UPDATE_STEPS_DONE")}
    out.update(PYTHONPATH=str(tree), LOCALM_HOME=str(home), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUTF8="1", PYTHONUNBUFFERED="1", NO_COLOR="1", LOCALM_UPDATE_ID=update_id)
    return out


# --------------------------------------------------------------- preparing

def prepare(root, update_id: str, *, uv, home, env: Mapping, say: Say) -> dict:
    """Fetch and test everything the steps of the update *update_id* of the install at
    *root* need. Its new tree is ``<id>/new``; *home* is the data folder; *uv* the full
    path of the uv to run. Writes only below ``<id>`` and the same id's folder in the
    component folders it prepares, and the download caches. Returns the steps, also on
    disk for :func:`run`: ``{"sync", "torch", "uv", "parts", "kept", "notes", ...}``.

    Raises :class:`.mover.Refused` (the update cannot be made; nothing changed) and
    OSError. Call :func:`discard` when the update is not made."""
    work = mover.work_folder(root, update_id)
    (work / "plans").mkdir(parents=True, exist_ok=True)
    future = steps_packages.Future(root, update_id, uv, env=env, say=say)
    extras = steps_packages.used_extras(root, future.tree, uv, env=env)
    sync = future.sync(extras)
    tree, python = str(future.tree), future.python
    code, last = run_child(
        [python, "-m", "localm", "_update-prepare", "--root", disk.short_path(Path(root)),
         "--id", update_id, "--uv", uv], cwd=tree, say=say,
        env=child_env(env, tree, home, update_id))
    try:
        prepared = disk.read_json(work / PREPARED, job.MAX_RECORD_BYTES * 64)
    except ValueError as e:
        raise Refused(f"what the update prepared cannot be read: {e}") from None
    if code != 0 or not isinstance(prepared, dict):
        reason = prepared.get("refused") if isinstance(prepared, dict) else None
        raise Refused(str(reason or last or "the new version could not prepare its parts"))
    spec = {"format": FORMAT, "sync": sync, **prepared}
    _check(spec)
    disk.write_json(work / STEPS_FILE, spec)
    return spec


def _check(spec) -> dict:
    """*spec*, when it is the steps of an update. Raises :class:`Unreadable`."""
    lists = ("torch", "parts", "kept", "notes")
    if (not isinstance(spec, dict) or spec.get("format") != FORMAT
            or not all(isinstance(spec.get(k), list) for k in lists)
            or not isinstance(spec.get("sync"), (dict, type(None)))
            or not isinstance(spec.get("uv"), (str, type(None)))
            or not all(isinstance(p, dict) and isinstance(p.get("root"), str)
                       and p.get("kind") in (RUNTIME, FOLDERS, COMFYUI) for p in spec["parts"])):
        raise Unreadable(f"{STEPS_FILE} does not hold the steps of an update")
    return spec


def read(root, update_id: str) -> dict:
    """The steps :func:`prepare` wrote. Raises :class:`Unreadable`."""
    try:
        return _check(disk.read_json(mover.work_folder(root, update_id) / STEPS_FILE,
                                     job.MAX_RECORD_BYTES * 64))
    except ValueError as e:
        raise Unreadable(str(e)) from None


def part_root(root, part: Mapping) -> Path:
    """The component folder of *part*."""
    return Path(root) / part["root"]


def discard(root, update_id: str) -> None:
    """Delete what :func:`prepare` left outside the update's own folder; the staged
    packages go with ``job.discard``. Safe to call when nothing was prepared; what
    cannot be deleted is left for the next change."""
    work = mover.work_folder(root, update_id)
    for name in (STEPS_FILE, PREPARED):
        try:
            spec = disk.read_json(work / name, job.MAX_RECORD_BYTES * 64)
        except (ValueError, OSError):
            continue
        if isinstance(spec, dict):
            for part in spec.get("parts") or []:
                with contextlib.suppress(OSError, disk.Unsafe):
                    steps.discard(part_root(root, part), update_id)


# --------------------------------------------------------------- carried out

class Run:
    """The steps of one update being carried out; see :func:`run`."""

    def __init__(self, lock: job.Lock, update_id: str, *, uv, home, env: Mapping,
                 say: Say, ask: Optional[disk.Ask] = None) -> None:
        self.lock, self.id, self.uv, self.home = lock, update_id, str(uv), Path(home)
        self.root, self.env, self.say, self.ask = Path(lock.root), dict(env), say, ask
        #: Lines for the user, and the runtimes that stay because the new ones do not
        #: work here (:func:`record`).
        self.notes: list = []
        self.kept: list = []

    def _localm(self, *args: str) -> tuple:
        python = steps_packages.python_in(self.root / ".venv")
        return run_child([python, "-m", "localm", *args], cwd=self.root, say=self.say,
                         env=child_env(self.env, self.root, self.home, self.id))

    def packages(self, sync: Optional[Mapping], torch: list) -> None:
        self.say("Replacing Python packages...")
        steps_packages.run_packages(self.root, self.id, sync, torch, self.uv, env=self.env,
                                    ask=self.ask)
        if not torch:
            return
        python = steps_packages.python_in(self.root / ".venv")
        code, last = run_child([python, "-c", "import torch"], cwd=self.root, say=lambda _t: None,
                               env=steps_packages.packages._environment(self.env))
        if code != 0:
            raise Refused(f"the new PyTorch does not load: {last}")

    def own_uv(self, version: str) -> None:
        self.say(f"Replacing the uv inside localm by uv {version}...")
        old = steps.staged_files(disk.real_folder(self.root) / ".uv")
        steps.move_in(self.root, self.id, mover.UV, remove=old, ask=self.ask)
        uv = next((p for p in (self.root / ".uv" / _UV, self.root / ".uv" / "bin" / _UV)
                   if p.is_file()), None)
        code, out = (steps_packages.packages.run_hidden(
            [str(uv), "--version"], str(self.root), self.env)
            if uv else (1, "it is missing"))
        if code != 0 or out.split()[1:2] != [version]:
            raise Refused(f"the new uv does not report {version}: {out.strip()[-200:]}")

    def part(self, part: Mapping) -> None:
        folder = part_root(self.root, part)
        if part["kind"] == RUNTIME and not self._runtime_works(folder, part):
            return
        self.say(f"Replacing the {part['label']}...")
        held = job.add_part(self.lock, folder)
        if part["kind"] == RUNTIME:
            self._runtime(folder, part)
        elif part["kind"] == FOLDERS:
            steps.move_in(folder, self.id, mover.WHOLE, ask=self.ask)
        else:
            self._comfyui(folder, part)
        job.advance(held, job.STARTING)

    def _runtime_works(self, folder: Path, part: Mapping) -> bool:
        """Whether the prepared runtime works on this machine; when it does not and the
        installed one does with the new version, that one is kept."""
        if part.get("unproven"):
            return True
        code, last = self._localm("setup-llama", "--prove", str(folder / part["beside"]))
        if code == 0:
            return True
        if code == UNUSABLE_EXIT:
            kept, why = self._localm("setup-llama", "--prove", str(folder / part["name"]))
            if kept == 0:
                self.notes.append(f"kept the {part['backend']} runtime {part['from']}: "
                                  f"{part['to']} does not work here ({last})")
                self.kept.append({"backend": part["backend"], "kept": part["from"],
                                  "target": part["to"], "why": last})
                steps.discard(folder, self.id)
                return False
            last = f"{last}; the installed one does not work with this version either ({why})"
        raise Refused(f"the new {part['label']} could not be tested: {last}")

    def _runtime(self, folder: Path, part: Mapping) -> None:
        name = str(part["name"])
        disk.move(folder / part["beside"], steps.Place(folder, self.id).new(mover.WHOLE) / name,
                  disk.Wait())
        replaced = [rel for rel in steps.staged_files(disk.real_folder(folder / name))
                    if rel not in part["keep"]
                    and ("/" not in rel or rel.split("/", 1)[0] in part["folders"])]
        steps.move_in(folder, self.id, mover.WHOLE, ask=self.ask,
                      remove=[f"{name}/{rel}" for rel in replaced])

    def _comfyui(self, folder: Path, part: Mapping) -> None:
        steps.move_in(folder, self.id, mover.TREE, remove=part["removed"], ask=self.ask)
        if not part["packages"]:
            return
        steps_packages.run_pip(folder, self.id, ask=self.ask)
        env = {k: v for k, v in self.env.items() if k != "PYTHONPATH"}
        venv = mover.venv_of(folder, mover.VENV)
        if steps.shared.updater_components.dists(
                steps.shared.updater_components.site_packages(venv)).get("torch"):
            python = disk.short_path(steps_packages.python_in(venv))
            code, last = run_child([python, "-I", "-c", "import torch"], cwd=folder, env=env,
                                   say=lambda _t: None)
            if code != 0:
                raise Refused(f"the new PyTorch of the managed ComfyUI does not load: {last}")


_UV = "uv.exe" if os.name == "nt" else "uv"


def run(lock: job.Lock, update_id: str, *, uv, home, env: Mapping, say: Say,
        ask: Optional[disk.Ask] = None) -> Run:
    """Carry out the prepared steps of the update *update_id*, whose record is in phase
    STEPS and whose files are moved. *ask* is called about a file that stays held
    (:class:`.disk.Wait`). Returns the :class:`Run`. Raises
    :class:`.steps.Failed` with the reason; ``job.restore`` then puts back what was
    changed. The parts stay locked until *lock* is released."""
    carried = Run(lock, update_id, uv=uv, home=home, env=env, say=say, ask=ask)
    try:
        spec = read(lock.root, update_id)
        carried.notes += [str(n) for n in spec["notes"]]
        carried.kept += spec["kept"]
        if spec["sync"] or spec["torch"]:
            carried.packages(spec["sync"], spec["torch"])
        if spec["uv"]:
            carried.own_uv(spec["uv"])
        for part in spec["parts"]:
            carried.part(part)
    except steps.STEP_ERRORS as e:
        if isinstance(e, mover.StepFailed):
            raise steps.Failed(e.reason, e.errors, e.held) from e
        raise steps.Failed(str(e), held=[e] if isinstance(e, disk.Held) else None) from e
    return carried


def put_back(lock: job.Lock, *, uv, env: Mapping, say: Say, network: bool = True,
             wait_s: Optional[float] = None, ask: Optional[disk.Ask] = None) -> job.Result:
    """``job.restore`` of *lock*'s install. Python packages it cannot put back, because
    somebody changed them since, are installed again in their old versions with *uv*
    (:func:`.steps_packages.reinstall`), in the install and in its parts. Raises like
    ``job.restore``."""
    def reinstall(named: list) -> None:
        record = job.read(lock.root)
        if record is None:
            return
        spec: dict = {}
        with contextlib.suppress(Unreadable):
            spec = read(lock.root, record.id)
        for folder, step in sorted({(p.get("part", ""), p["step"]) for p in named}):
            wanted = [(p["name"], p["version"]) for p in named
                      if (p.get("part", ""), p["step"]) == (folder, step)]
            say(f"Installing {', '.join(f'{n} {v}' for n, v in wanted)} again...")
            parts = [p for p in spec.get("parts", ()) if p["root"] == folder]
            indexes = (parts[0].get("indexes") or () if parts else ()) if folder else (
                steps_packages.index_args(spec.get("torch", ())))
            extras = (spec.get("sync") or {}).get("extras") or (
                steps.shared.apply_update.DEPS_EXTRAS)
            steps_packages.reinstall(Path(lock.root) / folder, record.id, step, wanted, uv,
                                     env=env, indexes=indexes, extras=extras, network=network,
                                     wait_s=wait_s, ask=ask)
    repair = "Run localm's setup again, which repairs that."
    if not network:
        repair = ("They could not be fetched, because localm may not use the network. " + repair
                  + " It needs the network.")
    return steps.put_back(lambda: job.restore(lock, wait_s=wait_s, ask=ask), reinstall, repair)


def record(root, *, old: str, new: str, kept: list, home, env: Mapping, say: Say) -> Optional[str]:
    """After the update ended as done: let the new version record what changed (its
    install record, the runtime history, the runtimes *kept* of :class:`Run`). Returns
    None, or why that failed; the update stays done."""
    python = steps_packages.python_in(Path(root) / ".venv")
    code, last = run_child([python, "-m", "localm", "_update-record", "--from", old, "--to", new,
                            "--kept", json.dumps(kept)], cwd=root, say=say,
                           env=child_env(env, root, home, ""))
    return None if code == 0 else f"the update could not be recorded everywhere: {last}"
