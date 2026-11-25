# SPDX-License-Identifier: AGPL-3.0-or-later
"""Carries out one step of an update, or one change of a component, on a small tree and
ends its own process at a chosen place (``test_update_steps_kills``).

    <python> _update_steps_driver.py <job json>

The job names the tree, the step and the place: none; ``{"record": "whole" | "part",
"phase": P}``, right after that phase is in the record; or ``{"back": bool, "path":
file, "move": n, "after": bool}``, right before or after the n-th rename of that file on
the way there or back. Without a place it prints what it did: the renames, as ``[back,
path]``, and the records, as ``[folder, phase]``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

JOB = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
sys.path.insert(0, JOB["repo"])

from localm.update_mechanism import (disk, job, mover, steps, steps_packages,  # noqa: E402
                                     steps_update)

KILL = JOB["kill"] or {}
moves: list = []
records: list = []
going_back = False


def _end() -> None:
    sys.stdout.flush()
    os._exit(9)


def _rel(path) -> str | None:
    """*path* relative to the tree, None for one in a folder of the mechanism."""
    rel = os.path.relpath(disk.short_path(Path(path)), JOB["top"]).replace(os.sep, "/")
    inner = set(rel.split("/"))
    return None if {".localm-update", ".localm-old", ".localm-kept"} & inner else rel


def _renaming(real):
    def call(src, dst, *rest):
        here = [rel for rel in (_rel(src), _rel(dst)) if rel is not None]
        path = here[0] if len(here) == 1 else None
        number = moves.count([going_back, path])
        at = path is not None and (
            KILL.get("path") == path and KILL["back"] == going_back and KILL["move"] == number)
        if at and not KILL["after"]:
            _end()
        done = real(src, dst, *rest)
        if at:
            _end()
        if path is not None:
            moves.append([going_back, path])
        return done
    return call


def _recording(real):
    def write_json(path, value):
        real(path, value)
        if Path(path).name != "job.json":
            return
        folder = disk.short_path(Path(path).parent.parent)
        name = "whole" if os.path.samefile(folder, JOB["root"]) else "part"
        if [name, value["phase"]] not in records:
            records.append([name, value["phase"]])
        if KILL.get("record") == name and KILL["phase"] == value["phase"]:
            _end()
    return write_json


def _restoring(real):
    def restore(*args, **kw):
        global going_back
        going_back = True
        return real(*args, **kw)
    return restore


def _nothing_left(*_args, **_kw):
    return 0, ""


def _update() -> bool:
    """The step as a part of an update; whether it ended as done."""
    root = Path(JOB["root"])
    real = steps_packages.run_packages
    steps_packages.run_packages = lambda *a, **kw: real(*a, run=_nothing_left, **kw)
    steps_packages.packages.run_hidden = lambda *_a, **_kw: (0, "uv 9.9")
    with job.Lock(root) as lock:
        job.start(lock, JOB["id"], old="1", new="2")
        job.advance(lock, job.MOVING)
        job.advance(lock, job.STEPS)
        try:
            steps_update.run(lock, JOB["id"], uv="uv", home=root, env=dict(os.environ),
                             say=lambda _t: None)
        except steps.Failed as e:
            print(f"FAILED {e}")
            job.restore(lock)
            return False
        job.advance(lock, job.STARTING)
        if JOB["fail"]:
            job.restore(lock)
            return False
        job.finish(lock)
        return True


def _alone() -> bool:
    """The step as a change of the component on its own; whether it ended as done."""
    root, place = Path(JOB["root"]), steps.Place(JOB["root"], JOB["id"])
    calls = [lambda: steps.move_in(root, JOB["id"], mover.WHOLE if JOB["whole"] else mover.TREE,
                                   remove=JOB["remove"])]
    if JOB["packages"]:
        calls.append(lambda: steps_packages.run_pip(root, JOB["id"]))
    try:
        steps.change(place, calls, prove=lambda: "it fails" if JOB["fail"] else None)
    except steps.Failed as e:
        print(f"FAILED {e}")
        return False
    return True


def main() -> None:
    disk.HELD_WAIT_S = 0.3
    os.rename, os.replace = _renaming(os.rename), _renaming(os.replace)
    disk.replace_through = _renaming(disk.replace_through)
    disk.write_json = _recording(disk.write_json)
    job.restore = _restoring(job.restore)
    done = _alone() if JOB["alone"] else _update()
    print("RESULT " + json.dumps({"done": done, "moves": moves, "records": records}))


if __name__ == "__main__":
    main()
