# SPDX-License-Identifier: AGPL-3.0-or-later
"""The finishing step of an update to this build (:mod:`localm.updater_finish`).

The updater that applied the update runs it after the swap::

    <the install's python> -I -X utf8 -u <install>/localm/_finish_update.py OLD NEW CLASS RESULT

Outside Windows it gives the launch scripts of the install
(:data:`localm._apply_update.EXECUTABLES`) their execute permission, which an
updater that extracts without file modes (localm 0.3.3 to 0.3.4b3) leaves off.
It takes over no built-in step and writes no RESULT file, so the updater runs its
dependency, runtime and component steps as usual.

Exit status 0 in every case: a permission it cannot set is reported in a line of
output with the command that sets it, and the update itself stays applied.
Standard library only; ``_apply_update.py`` is loaded by file path from beside this
file, because ``-I`` keeps this file's folder off ``sys.path``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def _apply_update():
    """``localm/_apply_update.py`` of this build, loaded by file path."""
    path = Path(__file__).resolve().with_name("_apply_update.py")
    spec = importlib.util.spec_from_file_location("_localm_apply_update_finish", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv) -> int:
    """Run the finishing step for the install this file is in. *argv* is the
    command line after the script (OLD NEW CLASS RESULT); it is not used. Returns
    the exit status (always 0)."""
    del argv
    if os.name == "nt":
        return 0
    root = Path(__file__).resolve().parent.parent
    try:
        fixed = _apply_update().restore_executables(root)
    except Exception as e:
        print(f"Could not give the launch scripts their execute permission ({e}). "
              f"To do it by hand: chmod +x {root}/*.sh", flush=True)
        return 0
    if fixed:
        print("Gave the launch scripts their execute permission back: "
              + ", ".join(fixed), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
