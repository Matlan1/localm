# SPDX-License-Identifier: AGPL-3.0-or-later
"""``<python> -I -S <this folder> restore <root>``.

Puts the install at *root* back after an update that was not finished, or removes
what a finished one left. Text goes to stderr. Exit status: 0 nothing is left to
do, 1 the install could not be put back (each reason is named), 2 wrong
arguments, 3 another process is updating this install.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

NAME = "update_mechanism"


def block_localm() -> None:
    """Make every later ``import localm`` in this process raise ImportError."""
    for name in [n for n in sys.modules if n == "localm" or n.startswith("localm.")]:
        del sys.modules[name]
    sys.modules["localm"] = None  # type: ignore[assignment]


def load_package():
    """This folder imported as the package :data:`NAME`, whatever the folder is
    called, with the folder itself off ``sys.path``. Returns its ``job`` module."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != here]
    spec = importlib.util.spec_from_file_location(
        NAME, here / "__init__.py", submodule_search_locations=[str(here)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {here}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = package
    spec.loader.exec_module(package)
    return importlib.import_module(f"{NAME}.job")


def main(argv) -> int:
    """Run the command line *argv* (without the program) and return the exit
    status."""
    block_localm()
    job = load_package()
    if len(argv) != 2 or argv[0] != "restore":
        print("usage: <python> -I -S update_mechanism restore <root>", file=sys.stderr)
        return 2
    try:
        with job.Lock(argv[1]) as lock:
            result = job.restore(lock)
    except job.Busy as e:
        print(str(e), file=sys.stderr)
        return 3
    except (job.Unreadable, job.Unsafe, job.Held, OSError) as e:
        print(f"The interrupted update could not be undone: {e}", file=sys.stderr)
        return 1
    for line in result.notes:
        print(line, file=sys.stderr)
    for line in result.errors:
        print(f"The interrupted update could not be undone: {line}", file=sys.stderr)
    return 1 if result.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
