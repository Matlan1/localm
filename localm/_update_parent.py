# SPDX-License-Identifier: AGPL-3.0-or-later
"""``localm update`` that installs a build: this process runs the command as a child,
and when the child has handed the update over (``updater_handoff.HANDED_OFF``) it waits
at the gate of ``localm/__init__.py`` until the update has ended, then says how it
ended. Its exit status is 0 only when the new version is installed. It imports nothing
outside the standard library and localm's gate, so it holds no file the update
replaces."""

import os
import subprocess
import sys

#: Set for the child, which then does the work itself.
WORK = "LOCALM_UPDATE_WORK"
#: The exit status of the child once the update is handed over.
HANDED_OFF = 75
#: Arguments with which ``localm update`` installs nothing.
_READ_ONLY = frozenset({"--check", "--list", "--help", "-h", "--repair"})
_PROGRAMS = frozenset({"localm", "localm.exe", "__main__.py", "localm-script.py"})


def wanted(argv) -> bool:
    """Whether *argv* (``sys.argv``) is a ``localm update`` that may install a build, in
    the process the user started."""
    return (len(argv) > 1 and argv[1] == "update" and not _READ_ONLY & set(argv[2:])
            and os.environ.get(WORK) != "1"
            and os.path.basename(argv[0]).lower() in _PROGRAMS)


def run(argv) -> int:
    """Run ``localm update`` *argv* as described above and return the exit status."""
    env = dict(os.environ)
    env[WORK] = "1"
    try:
        code = subprocess.call([sys.executable, "-m", "localm", *argv[1:]], env=env)
    except KeyboardInterrupt:
        return 130
    return wait() if code == HANDED_OFF else code


def _say(text: str) -> None:
    try:
        print(text, file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass


def wait() -> int:
    """Wait for the update the child handed over and return 0 when it installed the new
    version, else 1."""
    import json

    import localm
    from localm import procs
    root = os.path.dirname(os.path.dirname(os.path.abspath(localm.__file__)))
    folder = os.path.join(root, ".localm-update")
    update = localm._job(os.path.join(folder, "job.json"))[0]
    if update:
        os.environ["LOCALM_UPDATE_PRIMARY"] = update
        localm._gate(procs.current())
        import time
        deadline = time.monotonic() + 600
        while localm._held(folder) and time.monotonic() < deadline:
            time.sleep(0.2)
    path = os.path.join(folder, f"result-{update}.json")
    try:
        with open(path, encoding="utf-8") as f:
            result = json.loads(f.read(1024 * 1024))
        os.unlink(path)
    except (OSError, ValueError):
        result = None
    if not isinstance(result, dict):
        _say("The update ended without saying how. Run `localm update --check` to see "
             "which version is installed.")
        return 1
    return 0 if result.get("state") == "new" else 1
