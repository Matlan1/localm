# SPDX-License-Identifier: AGPL-3.0-or-later
"""The finishing step a build ships for its own update, as the updater of an older
localm runs it.

A build may contain ``localm/_finish_update.py`` (:data:`MODULE`). After its swap, the
updater of localm 0.3.3 to 0.3.4b4 (with its own copy of this module) runs it from the
tree the signature check verified::

    <the install's python> -I -X utf8 -u <install>/localm/_finish_update.py OLD NEW CLASS RESULT

with the install as the working directory and no console window. OLD and NEW are
the versions, CLASS the update class, RESULT the path of its result file
(:data:`RESULT_FILE` in the data folder's ``updates`` folder). It uses only the
standard library.

- Exit status 0: the update continues. Any other status, or being stopped at a
  limit below: that updater rolls the update back, with the last line of output as
  the reason.
- RESULT holds ``{"took_over": [...]}``, naming the built-in steps it carried out
  among :data:`STEPS`; that updater skips those and runs every other one as usual. No
  file, or a file that cannot be read, means nothing was taken over.
- Every line of output goes to the user the way the built-in steps' output does. A
  long step prints a line at least every :data:`SILENCE_LIMIT_S` seconds.
- It is stopped, with its child processes, after :data:`SILENCE_LIMIT_S` seconds
  without output or :data:`TOTAL_LIMIT_S` seconds in all.

This build installs an update with the program the build ships
(``localm.update_mechanism``) and runs no finishing step itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

MODULE = "localm/_finish_update.py"
RESULT_FILE = "finish-result.json"
STEPS = frozenset({"dependencies", "runtime", "components"})
SILENCE_LIMIT_S = 600.0
TOTAL_LIMIT_S = 7200.0


def interpreter(root) -> str:
    """The full path of the python of the install at *root*: its venv's, else the
    running one."""
    parts = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    own = Path(root).joinpath(".venv", *parts)
    return str(own) if own.is_file() else sys.executable
