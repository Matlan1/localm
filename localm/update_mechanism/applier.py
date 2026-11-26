# SPDX-License-Identifier: AGPL-3.0-or-later
"""The program that carries an update out, shipped with the release it installs.

The installed version extracts the release into ``<id>/new``, copies this package to
``<id>/applier`` and runs it twice with its base Python, ``-I -S``:

- ``check <root> <id>``, while localm runs: prepares the steps
  (:func:`.steps_update.prepare`) and lists the files of the user the update replaces.
  It writes only below ``<id>`` (and the download caches) and changes nothing of the
  install. Result in ``<id>/check.json``; exit status 0, else refused with the reason.
- ``apply <root> <id>``, as the last act of the installed version: reads the hand-off
  (:data:`HANDOFF_KEYS`) as one JSON line on standard input, takes the lock, writes the
  record in phase ENDING and prints ``ready`` on standard output. Then it waits until
  the process that goes on for the user (the server after its restart, or the waiting
  ``localm update``) reports at the gate of ``localm/__init__.py``, ends what was
  confirmed, and goes through MOVING, STEPS, STARTING and DONE. Any failure puts the old
  version back (:func:`.steps_update.put_back`). The outcome is written to
  :data:`RESULT` in the data folder's ``updates`` folder. Exit status 0 when the new
  version runs, 1 when the old one does or the install could not be put back, 3 when
  another process is updating the install.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import disk, job, mover, restoring, shared, steps, steps_data, steps_probe, steps_update

#: In the update's folder: what ``check`` found.
CHECK = "check.json"
#: In ``<data folder>/updates``: how the last update ended.
RESULT = "update-result.json"
#: The keys of the hand-off. ``mode``: ``server`` (a server restarts into the update
#: and is let through at STARTING) or ``command`` (``localm update`` waits for it).
#: ``starter``: ``<pid>-<start>`` of the process that starts this one; ``confirmed``:
#: the same of each process the user agreed to end; ``home``: the data folder;
#: ``old``, ``new``: the versions; ``watch``: ``{"host", "port", "scheme"}`` of the
#: server; ``relaunch``: its command line, to start the old version again when the
#: new one was let through and failed.
HANDOFF_KEYS = frozenset({"mode", "starter", "confirmed", "home", "old", "new", "watch",
                          "relaunch"})
SERVER, COMMAND = "server", "command"
#: Seconds the process that goes on for the user has to report at the gate.
ARRIVE_S = 120.0
#: Seconds a restarted server has to answer with the new version.
ANSWER_S = 180.0
_LINES = 40


class Abandoned(Exception):
    """The update was not started: nothing of the install was changed."""


# --------------------------------------------------------------------- check

def check(root, update_id: str, say: Callable[[str], None]) -> int:
    """``check``: see the module. Returns the exit status."""
    root = Path(root)
    work = mover.work_folder(root, update_id)
    out: dict = {"refused": None, "edited": [], "notes": []}
    try:
        statement = _statement(mover.work_folder(root, update_id) / "new")
        if statement is None:
            raise mover.Refused("the new version carries no list of its files")
        old = _statement(root)
        plan = mover.plan_files(root, update_id, statement["files"],
                                old["files"] if old else None, protect=_protect(root))
        out["edited"] = sorted(op.path for op in plan.ops if op.edited and op.old is not None)
        uv = restoring.find_uv(root)
        if uv is None:
            raise mover.Refused("the uv of this install cannot be found; run setup again")
        spec = steps_update.prepare(steps.Place(root, update_id), uv=uv,
                                    home=restoring.data_folder(root),
                                    env=restoring.environment(root), say=say)
        out["notes"] = [str(n) for n in spec["notes"]]
    except (*steps.STEP_ERRORS, ValueError, LookupError) as e:
        out["refused"] = str(e)
    disk.write_json(work / CHECK, out)
    if out["refused"]:
        say(f"The update cannot be made: {out['refused']}")
        return restoring.FAILED
    return restoring.OK


def _statement(folder: Path) -> Optional[dict]:
    package = shared.update_package
    raw = disk.read_file(folder / package.NAME, package.MAX_BYTES)
    return package.parse(raw.decode("utf-8")) if raw else None


def _protect(root: Path) -> list:
    """The data folder, when it is inside the install."""
    try:
        home = disk.real_folder(restoring.data_folder(root))
    except (OSError, ValueError, LookupError):
        return []
    whole = disk.real_folder(root)
    return [home.relative_to(whole).as_posix()] if whole in home.parents else []


# --------------------------------------------------------------------- apply

class Applier:
    """One update being carried out; see the module."""

    def __init__(self, root, update_id: str, handoff: dict) -> None:
        self.root, self.id, self.h = Path(root), update_id, handoff
        self.home = Path(handoff["home"])
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("LOCALM_UPDATE_",))}
        self.env["LOCALM_HOME"] = str(self.home)
        self.lines: list = []
        self.proot = disk.short_path(disk.real_folder(self.root))
        self.primary: Optional[str] = None
        self.let_through = False

    def say(self, text: str) -> None:
        self.lines = [*self.lines, text][-_LINES:]
        restoring.say(text)

    # ------------------------------------------------------------- the phases

    def run(self, lock: job.Lock) -> dict:
        """Carry the update out under *lock*, whose record is in phase ENDING. Returns
        the result (:data:`RESULT`)."""
        h, uv = self.h, restoring.find_uv(self.root)
        try:
            self.primary = self._wait_for_primary()
            self._end_others()
        except (Abandoned, OSError) as e:
            steps_update.discard(self.root, self.id)
            job.restore(lock)
            return self._result("old", f"The update to {h['new']} was not started: {e}")
        try:
            job.advance(lock, job.MOVING)
            self.say(f"Replacing the files of localm {h['old']} by those of {h['new']}...")
            steps_data.keep(self.root, self.id, self.home)
            new = _statement(mover.work_folder(self.root, self.id) / "new")
            old = _statement(self.root)
            if new is None:
                raise mover.Refused("the new version carries no list of its files")
            plan = mover.plan_files(self.root, self.id, new["files"],
                                    old["files"] if old else None, protect=_protect(self.root))
            mover.write_plan(self.root, self.id, plan)
            mover.forward(self.root, self.id, plan, ask=None)
            job.advance(lock, job.STEPS)
            carried = steps_update.run(lock, self.id, uv=uv, home=self.home, env=self.env,
                                       say=self.say)
            job.advance(lock, job.STARTING)
            self.say(f"Starting localm {h['new']}...")
            why = steps_probe.probe(steps_update.steps_packages.python_in(self.root / ".venv"),
                                    self.root, version=h["new"], home=self.home, env=self.env,
                                    update_id=self.id)
            if why is None and h["mode"] == SERVER:
                why = self._let_primary_through()
            if why:
                raise mover.Refused(why)
            done = job.finish(lock)
        except (*steps.STEP_ERRORS, steps.Failed) as e:
            return self._back(lock, uv, getattr(e, "reason", None) or str(e))
        except BaseException as e:
            self._back(lock, uv, f"{type(e).__name__}: {e}")
            raise
        why = steps_update.record(self.root, old=h["old"], new=h["new"], kept=carried.kept,
                                  home=self.home, env=self.env, say=self.say)
        return self._result("new", f"Updated to localm {h['new']}.",
                            notes=[*carried.notes, *done.notes, *([why] if why else [])])

    def _back(self, lock: job.Lock, uv, reason: str) -> dict:
        """Put the old version back after *reason*, and start it again when the new one
        was let through."""
        h = self.h
        self.say(f"The update to {h['new']} failed: {reason}. Putting localm {h['old']} back...")
        if self.let_through:
            self._end_primary()
        try:
            back = steps_update.put_back(lock, uv=uv, env=self.env, say=self.say,
                                         network=restoring.network(self.root))
        except steps.STEP_ERRORS as e:
            back = job.Result("failed", [str(e)])
        steps_update.discard(self.root, self.id)
        if back.state == "new":
            return self._result("new", f"Updated to localm {h['new']}.", notes=back.notes)
        if back.errors:
            return self._result(
                "failed", f"The update to {h['new']} failed ({reason}), and localm {h['old']} "
                f"could not be put back completely: " + "; ".join(back.errors),
                notes=[*back.notes, *restoring.ways_out(self.root)])
        if self.let_through:
            self._relaunch()
        return self._result("old", f"The update to {h['new']} failed and was undone: "
                            f"{reason}. localm {h['old']} runs.", notes=back.notes)

    # -------------------------------------------------- the processes of localm

    def _wait_for_primary(self) -> str:
        """``<pid>-<start>`` of the process that goes on for the user, once it waits at
        the gate and its starter has ended. Raises :class:`Abandoned`."""
        procs = shared.also("procs")
        if procs is None:
            raise Abandoned("this build cannot list the processes of localm")
        folder = mover.update_folder(self.root)
        deadline = time.monotonic() + ARRIVE_S
        while time.monotonic() < deadline:
            starter_runs = any(f"{e.pid}-{e.start}" == self.h["starter"]
                               for e in procs.list(self.proot))
            for name in sorted(os.listdir(folder)):
                if not name.startswith("waiting-") or starter_runs:
                    continue
                try:
                    said = (folder / name).read_text(encoding="ascii").strip()
                except (OSError, ValueError):
                    continue
                if said == self.id:
                    return name[len("waiting-"):]
            time.sleep(0.1)
        raise Abandoned(f"localm did not restart into the update within {ARRIVE_S:.0f} s")

    def _end_others(self) -> None:
        """End the processes of localm the user agreed to end and those the starter
        started. Raises :class:`Abandoned`, naming each process nobody agreed to end and
        each one that does not end."""
        procs = shared.also("procs")
        root = self.proot
        holders = shared.apply_update.rm_holders
        running = procs.list(root, holders=holders)
        waiting = restoring._waiting(self.root, running)
        agreed = set(self.h["confirmed"]) | {self.h["starter"]}
        by_key = {(e.pid, e.start): e for e in running}

        def agreed_to(e) -> bool:
            seen = set()
            while e is not None and (e.pid, e.start) not in seen:
                seen.add((e.pid, e.start))
                parent = tuple(e.parent) if e.parent else None
                if f"{e.pid}-{e.start}" in agreed or (
                        parent and f"{parent[0]}-{parent[1]}" in agreed):
                    return True
                e = by_key.get(parent) if parent else None
            return False

        def spare(entries) -> set:
            found = restoring._waiting(self.root, entries)
            return found | {e.pid for e in entries
                            if e.parent and e.parent[0] in found and not e.recorded}

        others = [e for e in running if e.recorded and e.pid not in waiting]
        refused = [e for e in others if not agreed_to(e)]
        if refused:
            raise Abandoned("these processes of localm run and nobody agreed to end them: "
                            + ", ".join(f"{e.purpose} (pid {e.pid})" for e in refused))
        if others:
            self.say("Ending " + ", ".join(f"{e.purpose} (pid {e.pid})" for e in others) + "...")
        try:
            left = procs.end_others(root, spare=spare, holders=holders)
        except OSError as e:
            raise Abandoned(str(e)) from None
        if left:
            raise Abandoned("these processes of localm did not end: " + "; ".join(
                restoring._in_the_way(w) for e in left
                for w in (e.in_the_way or ({"pid": e.pid, "name": e.purpose},))))

    def _let_primary_through(self) -> Optional[str]:
        """Let the restarted server through the gate and wait until it answers with the
        new version. None when it does, else why not."""
        h, watch = self.h, self.h.get("watch") or {}
        if not self.primary or not watch.get("port"):
            return None
        started = mover.update_folder(self.root) / "started"
        with open(started.with_name("started.tmp"), "w", encoding="ascii") as f:
            f.write(f"{self.id} {self.primary}")
            f.flush()
            os.fsync(f.fileno())
        os.replace(started.with_name("started.tmp"), started)
        self.let_through = True
        url = f"{watch.get('scheme') or 'http'}://{watch.get('host') or '127.0.0.1'}:{watch['port']}/whoami"
        deadline = time.monotonic() + ANSWER_S
        seen = ""
        while time.monotonic() < deadline:
            if not self._primary_runs():
                return f"localm {h['new']} ended right after it started ({seen or 'no answer'})"
            try:
                with urllib.request.urlopen(url, timeout=5) as r:   # noqa: S310 (own loopback)
                    version = str(json.loads(r.read(65536)).get("version") or "")
                if version == h["new"]:
                    return None
                seen = f"it answered version {version!r}"
            except (OSError, ValueError) as e:
                seen = str(e)
            time.sleep(0.5)
        return f"localm {h['new']} did not answer within {ANSWER_S:.0f} s ({seen})"

    def _primary_runs(self) -> bool:
        procs = shared.also("procs")
        return any(f"{e.pid}-{e.start}" == self.primary for e in procs.list(self.proot))

    def _end_primary(self) -> None:
        """End the server that was let through, with what it started."""
        procs = shared.also("procs")
        for e in procs.list(self.proot):
            if f"{e.pid}-{e.start}" == self.primary:
                for child in [*procs.descendants(e, self.proot), e]:
                    procs._end(child, True)

    def _relaunch(self) -> None:
        """Start the old version again with the server's command line."""
        argv = self.h.get("relaunch")
        if not argv:
            return
        import subprocess
        flags = ({"creationflags": subprocess.CREATE_NO_WINDOW
                  | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                 else {"start_new_session": True})
        try:
            subprocess.Popen([str(a) for a in argv], cwd=str(self.root), env=self.env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, **flags)
        except OSError as e:
            self.say(f"localm could not be started again: {e}. Start it yourself.")

    # ------------------------------------------------------------- the result

    def _result(self, state: str, text: str, notes: Optional[list] = None) -> dict:
        h = self.h
        result = {"state": state, "from": h["old"], "to": h["new"],
                  "version": h["new"] if state == "new" else h["old"], "note": text,
                  "notes": [str(n) for n in notes or ()], "lines": self.lines, "at": time.time()}
        for line in [text, *result["notes"]]:
            self.say(line)
        places = [self.home / "updates" / RESULT]
        if h["mode"] == COMMAND:
            places.append(mover.update_folder(self.root) / f"result-{self.id}.json")
        for path in places:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                disk.write_json(path, result)
            except (OSError, disk.Unsafe, disk.Held) as e:
                restoring.say(f"The result of the update could not be written to {path}: {e}")
        return result


def read_handoff(stream) -> dict:
    """The hand-off on *stream*. Raises ValueError."""
    raw = json.loads(stream.readline(1024 * 1024) or "null")
    if not isinstance(raw, dict) or set(raw) != HANDOFF_KEYS or raw["mode"] not in (
            SERVER, COMMAND) or not isinstance(raw["confirmed"], list):
        raise ValueError("the hand-off of the update is not complete")
    return raw


def apply(root, update_id: str) -> int:
    """``apply``: see the module. Returns the exit status."""
    try:
        handoff = read_handoff(sys.stdin)
        with job.Lock(root) as lock:
            job.start(lock, update_id, old=handoff["old"], new=handoff["new"])
            print("ready", flush=True)
            sys.stdout.close()
            sys.stdout = None   # type: ignore[assignment]
            result = Applier(root, update_id, handoff).run(lock)
    except job.Busy as e:
        restoring.say(f"The update cannot be made: {e}")
        return restoring.BUSY
    except (*steps.STEP_ERRORS, ValueError) as e:
        restoring.say(f"The update cannot be made: {e}")
        return restoring.FAILED
    return restoring.OK if result["state"] == "new" else restoring.FAILED
