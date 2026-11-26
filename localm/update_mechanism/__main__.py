# SPDX-License-Identifier: AGPL-3.0-or-later
"""``<python> -I -S <this folder> restore <root>``.

Puts the install at *root* back after an update that was not finished, or removes
what a finished one left. Text goes to stderr. Exit status: 0 nothing is left to
do, 1 the install could not be put back (each reason is named), 2 wrong
arguments, 3 another process is updating this install. See :mod:`.restoring`.

``<python> -I -S <this folder> setup-done <root>``: setup has made the Python
environment again; deletes the record it kept for that (:func:`.restoring.end_setup`),
with the same exit statuses.

``<python> -I -S <this folder> check|apply <root> <id>``: prepare, or carry out, the
update *id* (:mod:`.applier`); after the id of ``apply``, the command line that starts
the old version again when the new one fails as a server.
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
    load_package()
    if (len(argv) == 3 or (len(argv) > 3 and argv[0] == "apply")) and argv[0] in (
            "check", "apply") and len(argv[2]) == 32 and set(argv[2]) <= set("0123456789abcdef"):
        applier = importlib.import_module(f"{NAME}.applier")
        if argv[0] == "apply":
            return applier.apply(argv[1], argv[2], argv[3:])
        return applier.check(argv[1], argv[2], say=lambda t: print(t, file=sys.stderr, flush=True))
    if len(argv) != 2 or argv[0] not in ("restore", "setup-done"):
        print("usage: <python> -I -S update_mechanism restore|setup-done <root>, or "
              "check|apply <root> <id>", file=sys.stderr)
        return 2
    restoring = importlib.import_module(f"{NAME}.restoring")
    return (restoring.run if argv[0] == "restore" else restoring.end_setup)(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
