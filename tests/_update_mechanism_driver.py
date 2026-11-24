# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runs an update with the applier and ends its own process at a chosen point.

    <python> -I -S _update_mechanism_driver.py <applier folder> <job json> <kill at>

Every call the update mechanism makes to a function of ``os`` in :data:`CALLS` and to
``json.dump`` is one point; the process is killed right before the call number *kill at* (from 0; -1
never). With -1 the number of points is printed as ``POINTS <n>``. ``FINISH`` is
printed when the update starts to end as done. Exit status 0: the job ran through.
"""

import importlib.util
import json
import os
import signal
import sys

CALLS = ("rename", "replace", "fsync", "mkdir", "rmdir", "unlink")
count = 0


def _load(applier):
    path = os.path.join(applier, "__main__.py")
    spec = importlib.util.spec_from_file_location("applier_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.block_localm()
    mod.load_package()
    import update_mechanism.job
    import update_mechanism.mover
    import update_mechanism.packages
    return update_mechanism


def _arm(kill_at):
    def wrap(real):
        def call(*args, **kwargs):
            global count
            if count == kill_at:
                os.kill(os.getpid(), signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
            count += 1
            return real(*args, **kwargs)
        return call
    for name in CALLS:
        setattr(os, name, wrap(getattr(os, name)))
    json.dump = wrap(json.dump)


def _kill_after_uv_starts(um):
    real = um.packages.subprocess.Popen

    class Popen(real):
        def __init__(self, args, **kwargs):
            super().__init__(args, **kwargs)
            if "--dry-run" not in args:
                os.kill(os.getpid(), signal.SIGTERM if os.name == "nt" else signal.SIGKILL)
    um.packages.subprocess.Popen = Popen


def _update(um, spec):
    if spec.get("kill_after_uv_starts"):
        _kill_after_uv_starts(um)
    root, update_id = spec["root"], spec["id"]
    with um.job.Lock(root) as lock:
        um.job.start(lock, update_id, old="1", new="2")
        um.job.advance(lock, um.job.MOVING)
        if spec.get("new_files"):
            plan = um.mover.plan_files(root, update_id, spec["new_files"], spec["old_files"],
                                       swap=spec["swap"], protect=["mydata"])
            um.mover.write_plan(root, update_id, plan)
            um.mover.forward(root, update_id, plan)
        um.job.advance(lock, um.job.STEPS)
        if spec.get("uv"):
            plan = um.packages.plan_packages(root, update_id, spec["uv"], env=spec["env"])
            um.mover.write_plan(root, update_id, plan)
            um.packages.forward(root, update_id, plan, spec["uv"], env=spec["env"])
        um.job.advance(lock, um.job.STARTING)
        print("FINISH", flush=True)
        um.job.finish(lock)


def _restore(um, spec):
    with um.job.Lock(spec["root"]) as lock:
        result = um.job.restore(lock)
    if result.errors:
        sys.exit("\n".join(result.errors))


def _hold(um, spec):
    with um.job.Lock(spec["root"]):
        print("LOCKED", flush=True)
        sys.stdin.read()


def main():
    applier, job, kill_at = sys.argv[1], sys.argv[2], int(sys.argv[3])
    with open(job, encoding="utf-8") as f:
        spec = json.load(f)
    um = _load(applier)
    _arm(kill_at)
    {"update": _update, "restore": _restore, "hold": _hold}[spec["do"]](um, spec)
    if kill_at < 0:
        print(f"POINTS {count}", flush=True)


if __name__ == "__main__":
    main()
