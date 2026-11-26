# SPDX-License-Identifier: AGPL-3.0-or-later
"""The program that carries an update out, shipped with the release it installs.

The installed version extracts the release into ``<id>/new``, copies this package to
``<id>/applier`` and runs it twice with its base Python, ``-I -S``:

- ``check <root> <id>``, while localm runs: prepares the steps
  (:func:`.steps_update.prepare`) and lists the files of the user the update replaces.
  It writes only below ``<id>`` (and the download caches) and changes nothing of the
  install. Result in ``<id>/check.json``; exit status 0, else refused with the reason.
- ``apply <root> <id> [<command line>]``, as the last act of the installed version: reads
  the hand-off (:data:`HANDOFF_KEYS`) as one JSON line on standard input, takes the lock, writes the
  record in phase ENDING and prints ``ready`` on standard output. Then it waits until
  the process that goes on for the user (the server after its restart, or the waiting
  ``localm update``) reports at the gate of ``localm/__init__.py``, ends what was
  confirmed, and goes through MOVING, STEPS, STARTING and DONE. Any failure puts the old
  version back (:func:`.steps_update.put_back`). The outcome is written to
  :data:`RESULT` in the data folder's ``updates`` folder. Exit status 0 when the new
  version runs, 1 when the old one does or the install could not be put back, 3 when
  another process is updating the install.

In mode ``server`` the progress shows in a small window (:mod:`.window`), which also asks
"Try again" or "Cancel" about a held file or a process that does not end; where no window
can be shown the update goes on without it and says so in its log.

A release that ships no update program of its own (an older one, chosen with ``localm
update --to``) is installed by this program, copied there by the installed version
(:func:`is_older`): its files by its own list, its Python packages from its own lock, and
the rest by that release when it starts (``steps_update.prepare`` *older*).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from . import (disk, job, mover, restoring, shared, steps, steps_data, steps_probe, steps_update,
               window)

#: In the update's folder: what ``check`` found.
CHECK = "check.json"
#: In ``<data folder>/updates``: how the last update ended.
RESULT = "update-result.json"
#: The keys of the hand-off. ``mode``: ``server`` (a server restarts into the update
#: and is let through at STARTING) or ``command`` (``localm update`` waits for it).
#: ``starter``: ``<pid>-<start>`` of the process that starts this one; ``confirmed``:
#: the same of each process the user agreed to end; ``home``: the data folder;
#: ``old``, ``new``: the versions; ``watch``: ``{"host", "port", "scheme"}`` of the
#: server. The command line that starts the old version again, when the new one was let
#: through and failed, follows the id on the command line of ``apply``.
HANDOFF_KEYS = frozenset({"mode", "starter", "confirmed", "home", "old", "new", "watch"})
#: The keys a hand-off may carry besides :data:`HANDOFF_KEYS`. ``restarting``:
#: ``[<pid>-<start>, purpose]`` of each other server that was asked to restart into the
#: update.
#: ``ancestors``: ``<pid>-<start>`` of each process above the starter that goes on
#: running and is neither ended nor named: a :data:`LAUNCHER` (it shows the window of the
#: app), the ``localm update`` that waits for the update, and the programs between them
#: that have no file of their own. Any other process listed there is ended only when the
#: user agreed, like every other process.
#: ``ask``: true when, in mode ``command``, the user is at the terminal of ``localm
#: update`` and is asked "Try again" or "Cancel" about what is in the way.
MORE_KEYS = frozenset({"restarting", "ancestors", "ask"})
SERVER, COMMAND = "server", "command"
#: The purpose of the process that shows the window of the app.
LAUNCHER = "launcher"
#: In the update's folder: the JSON list of ``<pid>-<start>`` of each other server that
#: was asked to restart and did not; the update does not wait for it.
NOT_RESTARTING = "not-restarting"
#: Seconds the process that goes on for the user has to report at the gate.
ARRIVE_S = 120.0
#: Seconds another server that was asked to restart has to wait at the gate.
RESTART_S = 30.0
#: Seconds a restarted server has to answer with the new version.
ANSWER_S = 180.0
#: Seconds a process above this one that the user agreed to end has to end.
END_S = 20.0
_LINES = 40
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}")


class Abandoned(Exception):
    """The update was not started: nothing of the install was changed."""


def _procs():
    """The process record of this build (``localm/procs.py`` beside this package).
    Raises :class:`Abandoned` when the build has none."""
    procs = shared.also("procs")
    if procs is None:
        raise Abandoned("this build cannot list the processes of localm")
    return procs


# --------------------------------------------------------------------- check

def is_older(root, update_id: str) -> bool:
    """Whether the release extracted for the update *update_id* ships no update program
    of its own, so that this one, a copy of the installed version's, carries it out."""
    new = mover.work_folder(Path(root), update_id) / "new"
    return not (new / "localm" / "update_mechanism" / "applier.py").is_file()


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
                                    env=restoring.environment(root), say=say,
                                    older=is_older(root, update_id))
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

    def __init__(self, root, update_id: str, handoff: dict, relaunch=()) -> None:
        self.root, self.id, self.h = Path(root), update_id, handoff
        self.home = data_folder()
        self.relaunch = [str(a) for a in relaunch]
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("LOCALM_UPDATE_",))}
        self.env["LOCALM_HOME"] = str(self.home)
        self.lines: list = []
        self.proot = disk.short_path(disk.real_folder(self.root))
        self.primary: Optional[str] = None
        self.let_through = False
        self.older = is_older(self.root, update_id)
        self.window: Optional[window.Window] = None
        #: Asks the user about what is in the way (a held file, a process that does not
        #: end): True to try again, False to cancel. None: nobody is asked.
        self.ask: Optional[Callable[[str], bool]] = (
            terminal_ask() if handoff.get("ask") and handoff["mode"] == COMMAND else None)

    def _held(self, held) -> bool:
        return self.ask is not None and self.ask(f"{held}. Close that program, then try again.")

    def _held_back(self, held) -> bool:
        """:meth:`_held` while the old version is put back: cancel then leaves the
        install neither old nor new, which the question says with the ways out."""
        return self.ask is not None and self.ask(
            f"{held}. Close that program, then try again. Cancel leaves localm "
            f"{self.h['old']} not put back completely: " + " ".join(restoring.ways_out(self.root)))

    def say(self, text: str) -> None:
        self.lines = [*self.lines, text][-_LINES:]
        restoring.say(text)
        if self.window is not None:
            self.window.say(text)

    def show_window(self, program=None) -> None:
        """Show the progress in the small update window (*program*: :mod:`.window` when
        None), which from then on asks about what is in the way. Without a window the
        update goes on as before; the reason is said."""
        self.window = window.Window.start(restoring.say, program)
        if self.window is None:
            return
        self.ask = self.window.ask
        for line in self.lines:
            self.window.say(line)

    def close_window(self, text: str) -> None:
        """Show *text* in the window as the outcome; it then closes itself."""
        shown, self.window = self.window, None
        if shown is not None:
            shown.end(text)

    # ------------------------------------------------------------- the phases

    def run(self, lock: job.Lock) -> dict:
        """Carry the update out under *lock*, whose record is in phase ENDING. Returns
        the result (:data:`RESULT`)."""
        h, uv = self.h, restoring.find_uv(self.root)
        try:
            self.primary = self._wait_for_primary()
            self._wait_for_restarts()
            while True:
                try:
                    self._end_others()
                    break
                except Abandoned as e:
                    if self.ask is None or not self.ask(
                            f"{e}. Close them, then try again; cancel leaves localm "
                            f"{h['old']} as it is."):
                        raise
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
            # Into a release without the gate, the gate goes last (GATE_FILES_LAST).
            plan = mover.plan_files(self.root, self.id, new["files"],
                                    old["files"] if old else None, protect=_protect(self.root),
                                    **({"swap": mover.GATE_FILES_LAST, "last": True}
                                       if self.older else {}))
            mover.write_plan(self.root, self.id, plan)
            ask = self._held if self.ask is not None else None
            mover.forward(self.root, self.id, plan, ask=ask)
            job.advance(lock, job.STEPS)
            carried = steps_update.run(lock, self.id, uv=uv, home=self.home, env=self.env,
                                       say=self.say, ask=ask)
            job.advance(lock, job.STARTING)
            self.say(f"Starting localm {h['new']}...")
            why = steps_probe.probe(steps_update.steps_packages.python_in(self.root / ".venv"),
                                    self.root, version=h["new"], home=self.home, env=self.env,
                                    update_id=self.id, older=self.older)
            if why is None and h["mode"] == SERVER:
                why = self._let_primary_through()
            if why:
                raise mover.Refused(why)
            done = job.finish(lock, ask=ask)
        except (*steps.STEP_ERRORS, steps.Failed) as e:
            return self._back(lock, uv, getattr(e, "reason", None) or str(e))
        except BaseException as e:
            self._back(lock, uv, f"{type(e).__name__}: {e}")
            raise
        try:
            why = steps_update.record(self.root, old=h["old"], new=h["new"],
                                      kept=carried.kept, home=self.home, env=self.env,
                                      say=self.say, older=self.older)
        except (*steps.STEP_ERRORS, ValueError) as e:
            why = f"The record of what this update installed could not be written: {e}"
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
                                         network=restoring.network(self.root),
                                         ask=self._held_back if self.ask is not None else None)
        except steps.STEP_ERRORS as e:
            back = job.Result("failed", [str(e)])
        steps_update.discard(self.root, self.id)
        if back.state == "new":
            return self._result("new", f"Updated to localm {h['new']}.", notes=back.notes)
        if back.errors or back.state != "old":
            errors = back.errors or [f"the way back ended as {back.state!r}"]
            return self._result(
                "failed", f"The update to {h['new']} failed ({reason}), and localm {h['old']} "
                f"could not be put back completely: " + "; ".join(errors),
                notes=[*back.notes, *restoring.ways_out(self.root)])
        if self.let_through:
            self._relaunch()
        # One sentence says the outcome: the way back's own "was undone" note is left out.
        undone = job.undone_note(h["new"], h["old"])
        return self._result("old", f"The update to {h['new']} failed and was undone: "
                            f"{reason}. localm {h['old']} runs.",
                            notes=[n for n in back.notes if n != undone])

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
                # A starter that re-executed itself in place (POSIX execv) keeps its pid
                # and start time: it waits under its own key.
                if not name.startswith("waiting-") or (
                        starter_runs and name[len("waiting-"):] != self.h["starter"]):
                    continue
                try:
                    said = (folder / name).read_text(encoding="ascii").strip()
                except (OSError, ValueError):
                    continue
                if said == self.id:
                    return name[len("waiting-"):]
            time.sleep(0.1)
        raise Abandoned(f"localm did not restart into the update within {ARRIVE_S:.0f} s")

    def _wait_for_restarts(self) -> None:
        """Wait, at most :data:`RESTART_S`, until for each other server that was asked to
        restart (``restarting``, less those in :data:`NOT_RESTARTING`) a process waits at
        the gate: the same process after its re-exec in place, or the process its re-exec
        started (on Windows a new one), whose file at the gate names it
        (``restarted <pid>-<start>``, ``LOCALM_RESTARTED_FROM``). Each server that has none
        by then is named and ended with the rest (:meth:`_end_others`)."""
        wanted = [(str(k), str(p)) for k, p in self.h.get("restarting") or ()]
        refused = set(self._not_restarting())
        wanted = [(k, p) for k, p in wanted if k not in refused]
        procs = shared.also("procs")
        if not wanted or procs is None:
            return
        folder = mover.update_folder(self.root)

        def said(e) -> str:
            try:
                return (folder / f"waiting-{e.pid}-{e.start}").read_text(
                    encoding="ascii").strip()
            except (OSError, ValueError):
                return ""

        deadline = time.monotonic() + RESTART_S
        while True:
            running = procs.list(self.proot)
            waiting = restoring._waiting(self.root, running)
            at_gate = [(e, said(e)) for e in running if e.pid in waiting and e.recorded
                       and f"{e.pid}-{e.start}" != self.primary]
            taken: set = set()
            left = []
            for key, purpose in wanted:
                found = next((e for e, text in at_gate if e.key not in taken and (
                    f"{e.pid}-{e.start}" == key or text == f"restarted {key}")), None)
                if found is None:
                    left.append((key, purpose))
                else:
                    taken.add(found.key)
            if not left:
                return
            if time.monotonic() >= deadline:
                self.say("These servers of localm did not restart within "
                         f"{RESTART_S:.0f} s and are ended: " + ", ".join(
                             f"{p} (pid {k.split('-')[0]})" for k, p in left))
                return
            time.sleep(0.2)

    def _not_restarting(self) -> list:
        """The keys in :data:`NOT_RESTARTING`; none when it cannot be read."""
        try:
            keys = disk.read_json(mover.work_folder(self.root, self.id) / NOT_RESTARTING,
                                  1024 * 1024)
        except (OSError, ValueError, disk.Held, disk.Unsafe):
            return []
        return [k for k in keys if isinstance(k, str)] if isinstance(keys, list) else []

    def _end_others(self) -> None:
        """End the processes of localm the user agreed to end and those the starter
        started. Raises :class:`Abandoned`, naming each process nobody agreed to end and
        each one that does not end."""
        procs = _procs()
        root = self.proot
        holders = shared.apply_update.rm_holders
        running = procs.list(root, holders=holders)
        waiting = restoring._waiting(self.root, running)
        agreed = set(self.h["confirmed"]) | {self.h["starter"]}
        named_above = set(self.h.get("ancestors") or ())
        # Of the processes the hand-off names above the starter, only a launcher and a
        # program without a file of its own go on unasked.
        above = {f"{e.pid}-{e.start}" for e in running if f"{e.pid}-{e.start}" in named_above
                 and (e.purpose == LAUNCHER or not e.recorded)}
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
            found = restoring._waiting(self.root, entries) | {
                e.pid for e in entries if f"{e.pid}-{e.start}" in above}
            return found | {e.pid for e in entries
                            if e.parent and e.parent[0] in found and not e.recorded}

        others = [e for e in running if e.recorded and e.pid not in waiting
                  and e.pid != os.getpid() and f"{e.pid}-{e.start}" not in above]
        refused = [e for e in others if not agreed_to(e)]
        if refused:
            raise Abandoned("these processes of localm run and nobody agreed to end them: "
                            + ", ".join(f"{e.purpose} (pid {e.pid})" for e in refused))
        if others:
            self.say("Ending " + ", ".join(f"{e.purpose} (pid {e.pid})" for e in others) + "...")
        try:
            ended = procs.end_others(root, spare=spare, holders=holders)
        except OSError as e:
            raise Abandoned(str(e)) from None
        # end_others also spares the processes above this one (on POSIX, where the server
        # that started the update re-executes itself in place, what started that server);
        # those of them the user agreed to end are ended here.
        left = [*ended, *self._end_above(procs, [e for e in others if e not in ended])]
        if left:
            # One that reached the gate meanwhile goes on with the update.
            waiting = restoring._waiting(self.root, procs.list(root, holders=holders))
            left = [e for e in left if e.pid not in waiting]
        if left:
            raise Abandoned("these processes of localm did not end: " + "; ".join(
                restoring._in_the_way(w) for e in left
                for w in (e.in_the_way or ({"pid": e.pid, "name": e.purpose},))))

    def _end_above(self, procs, entries) -> list:
        """End each of *entries* that still runs and does not wait at the gate, as
        ``procs.end_others`` ends a process: asked first, ended after
        ``procs._POSIX_GRACE_S``. Returns those that still run after :data:`END_S`."""
        asked: dict = {}
        deadline = time.monotonic() + END_S
        while True:
            waiting = restoring._waiting(self.root, procs.list(self.proot))
            rest = [e for e in entries if e.pid not in waiting and procs.alive(e)]
            now = time.monotonic()
            if not rest or now >= deadline:
                return rest
            for entry in rest:
                first = asked.setdefault(entry.key, now)
                procs._end(entry, now - first >= procs._POSIX_GRACE_S)
            time.sleep(0.1)

    def _let_primary_through(self) -> Optional[str]:
        """Let the restarted server through the gate and wait until it answers with the
        new version. None when it does, else why not."""
        h, watch = self.h, self.h.get("watch") or {}
        if not self.primary or not watch.get("port"):
            return None
        if not self._primary_runs():
            self.say(f"The server ended before localm {h['new']} could be started in it; "
                     "the next start of localm runs it.")
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
                version = whoami(url)
                if version == h["new"]:
                    return None
                seen = f"it answered version {version!r}"
            except (OSError, ValueError) as e:
                seen = str(e)
            time.sleep(0.5)
        return f"localm {h['new']} did not answer within {ANSWER_S:.0f} s ({seen})"

    def _primary_runs(self) -> bool:
        procs = _procs()
        return any(f"{e.pid}-{e.start}" == self.primary for e in procs.list(self.proot))

    def _end_primary(self) -> None:
        """End the server that was let through, with what it started."""
        procs = _procs()
        for e in procs.list(self.proot):
            if f"{e.pid}-{e.start}" == self.primary:
                for child in [*procs.descendants(e, self.proot), e]:
                    procs._end(child, True)

    def _relaunch(self) -> None:
        """Start the old version again with the server's command line."""
        argv = self.relaunch
        if not argv:
            return
        import subprocess
        flags: dict = ({"creationflags": subprocess.CREATE_NO_WINDOW
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
        places = [(self.home / "updates" / RESULT, result)]
        if h["mode"] == COMMAND:
            places.append((mover.update_folder(self.root) / job.result_name(self.id),
                           {**result, job.RESULT_FOR: self.primary}))
        for path, value in places:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                disk.write_json(path, value)
            except (OSError, disk.Unsafe, disk.Held) as e:
                restoring.say(f"The result of the update could not be written to {path}: {e}")
        self.close_window(text)
        return result


def whoami(url: str) -> str:
    """The version the server at *url* (its own ``/whoami`` on this machine) answers,
    asked without a proxy and, over https, without checking its certificate: the server
    is this install's own, on a loopback address, and may use localm's own certificate.
    Raises OSError or ValueError."""
    import ssl
    handlers: list = [urllib.request.ProxyHandler({})]
    if url.startswith("https://"):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    with urllib.request.build_opener(*handlers).open(url, timeout=5) as r:
        try:
            said = json.loads(r.read(65536))
        except RecursionError:
            raise ValueError("the answer of /whoami is nested too deeply") from None
        return str(said.get("version") or "") if isinstance(said, dict) else ""


def try_again(text: str, answers, say: Callable[[str], None] = restoring.say) -> bool:
    """Say *text* and ask "Try again" or "Cancel", reading the answer from the lines of
    *answers*. True to try again; the end of *answers* counts as cancel."""
    say(text)
    return restoring._try_again(answers, then="", tell=say)


def terminal_ask() -> Optional[Callable[[str], bool]]:
    """:func:`try_again` at the terminal this program shares with the ``localm update``
    that waits for it; None when there is none. On Windows that is the console
    (``CONIN$``). Elsewhere this program runs in a session of its own, without a
    controlling terminal, so ``/dev/tty`` names none: the terminal is the one its
    standard error was given (``os.ttyname``), opened without making it the controlling
    terminal."""
    try:
        if sys.platform == "win32":
            answers = open("CONIN$", encoding="utf-8", errors="replace")
        else:
            fd = os.open(os.ttyname(2), os.O_RDONLY | os.O_NOCTTY)
            answers = open(fd, encoding="utf-8", errors="replace")
    except OSError:
        return None
    return lambda text: try_again(text, answers)


def data_folder() -> Path:
    """The data folder this program was started with (``LOCALM_HOME``). Raises
    ValueError."""
    named = os.environ.get("LOCALM_HOME", "").strip()
    if not named:
        raise ValueError("the update was started without its data folder (LOCALM_HOME)")
    return Path(os.path.expanduser(named)).absolute()


def read_handoff(stream) -> dict:
    """The hand-off on *stream*, each value of the type it has to have, its versions plain
    version strings and its data folder the one this program was started with (only
    :func:`data_folder` is used). Raises ValueError."""
    try:
        raw = json.loads(stream.readline(1024 * 1024) or "null")
    except RecursionError:
        raise ValueError("the hand-off of the update is nested too deeply") from None
    if not isinstance(raw, dict) or not HANDOFF_KEYS <= set(raw) <= HANDOFF_KEYS | MORE_KEYS:
        raise ValueError("the hand-off of the update is not complete")

    def keys(value) -> bool:
        return isinstance(value, list) and all(isinstance(k, str) for k in value)

    watch = raw["watch"]
    if raw["mode"] not in (SERVER, COMMAND) or not isinstance(raw["starter"], str) \
            or not keys(raw["confirmed"]) or not keys(raw.get("ancestors") or []) \
            or not all(isinstance(r, list) and len(r) == 2 and keys(r)
                       for r in raw.get("restarting") or ()) \
            or not all(isinstance(raw[v], str) and _VERSION.fullmatch(raw[v])
                       for v in ("old", "new")) \
            or not isinstance(raw["home"], str) or not isinstance(raw.get("ask", False), bool) \
            or not (watch is None or (isinstance(watch, dict) and all(
                isinstance(watch.get(k), t) or watch.get(k) is None
                for k, t in (("host", str), ("port", int), ("scheme", str))))):
        raise ValueError("the hand-off of the update is not what it has to be")
    if os.path.normcase(str(Path(os.path.expanduser(raw["home"])).absolute())) \
            != os.path.normcase(str(data_folder())):
        raise ValueError("the hand-off names another data folder than LOCALM_HOME")
    return raw


def apply(root, update_id: str, relaunch=()) -> int:
    """``apply``: see the module; *relaunch* is the command line that follows the id.
    Returns the exit status."""
    try:
        handoff = read_handoff(sys.stdin)
        with job.Lock(root) as lock:
            job.start(lock, update_id, old=handoff["old"], new=handoff["new"])
            print("ready", flush=True)
            sys.stdout.close()
            sys.stdout = None   # type: ignore[assignment]
            update = Applier(root, update_id, handoff, relaunch)
            if handoff["mode"] == SERVER:
                update.show_window()
            try:
                result = update.run(lock)
            finally:
                update.close_window("The update of localm ended.")
    except job.Busy as e:
        restoring.say(f"The update cannot be made: {e}")
        return restoring.BUSY
    except (*steps.STEP_ERRORS, ValueError) as e:
        restoring.say(f"The update cannot be made: {e}")
        return restoring.FAILED
    return restoring.OK if result["state"] == "new" else restoring.FAILED
