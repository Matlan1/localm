# SPDX-License-Identifier: AGPL-3.0-or-later
"""Moving a folder and replacing an installed folder without losing it: the one
place the native media runtimes, the ComfyUI source and the uv inside the
install are swapped in.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("localm")


class ReplaceError(OSError):
    """An installed folder could not be replaced; it is still in place."""


def move_dir(src: Path, dst: Path) -> None:
    """Rename the folder *src* to *dst* (``os.replace``), retried with a short
    backoff while Windows refuses it with the sharing or access violation an
    open handle on a file inside causes (:func:`localm.storekit.replace_atomic`);
    each refused attempt is logged at WARNING with its number. Raises the last
    error when every attempt was refused, any other OSError at once."""
    from localm.storekit import replace_atomic

    def refused(attempt: int, error: OSError) -> None:
        logger.warning("moving %s to %s: attempt %d was refused (%s); retrying",
                       src, dst, attempt, error)
    replace_atomic(src, dst, on_retry=refused)


def replace_dir(staging: Path, dest: Path) -> None:
    """Put the folder *staging* in place of *dest* (same volume). An existing
    *dest* is first renamed to a hidden ``.<name>.previous-<pid>-<n>`` folder
    beside it, so a folder the system keeps open is refused whole and nothing
    in it is removed; when the move of *staging* then fails, that folder is put
    back. It is deleted once the new folder is in place (a failed delete is
    logged and leaves it on disk). Raises :class:`ReplaceError`: with *dest*
    unchanged, or, when putting it back fails too, naming the folder that
    still holds the installed files."""
    had_old = dest.exists()
    aside = dest.with_name(f".{dest.name}.previous-{os.getpid()}-{time.monotonic_ns()}")
    if had_old:
        try:
            move_dir(dest, aside)
        except OSError as e:
            raise ReplaceError(f"{dest} is in use and was left as it is ({e})") from e
    try:
        move_dir(staging, dest)
    except OSError as e:
        if had_old:
            try:
                move_dir(aside, dest)
            except OSError as back:
                logger.warning("replacing a folder: %s could not be put back from %s (%s)",
                               dest, aside, back)
                raise ReplaceError(
                    f"the new files could not be moved into {dest} ({e}), and the "
                    f"installed ones could not be put back ({back}); they are kept in "
                    f"{aside}") from e
        raise ReplaceError(f"the new files could not be moved into {dest} ({e}); the "
                           "installed ones were kept") from e
    if had_old:
        try:
            shutil.rmtree(aside)
        except OSError as e:
            logger.warning("replacing a folder: the replaced folder %s could not be "
                           "removed (%s); delete it by hand", aside, e)
