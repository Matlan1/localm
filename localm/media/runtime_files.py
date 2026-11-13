# SPDX-License-Identifier: AGPL-3.0-or-later
"""Steps shared by the installers of the native media runtimes
(stable-diffusion.cpp, KoboldCpp) and the ComfyUI source download: progress
that also reaches the debug log, moving and replacing an installed folder
without losing it, and the text of a refused download.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

Progress = Callable[[str], None]


class ReplaceError(OSError):
    """An installed folder could not be replaced; it is still in place."""


def refused_download_message(error) -> str:
    """The text for a ``netpolicy.NetworkPolicyError`` that refused a download:
    for the net_mode=off refusal (``error.off``) its own text, which names the
    Settings place and the command that turn network access on; else the
    policy's reason."""
    if getattr(error, "off", False):
        return f"the download was refused: {error}"
    return f"the download was refused by the network policy: {error}"


def logged(on_progress: Optional[Progress], tag: str) -> Progress:
    """*on_progress* that also writes each message to the debug log at INFO,
    prefixed with *tag*. *on_progress* may be None."""
    from localm.debuglog import logger

    def say(message: str) -> None:
        logger.info("%s: %s", tag, message)
        if on_progress is not None:
            on_progress(message)
    return say


def move_dir(src: Path, dst: Path) -> None:
    """Rename the folder *src* to *dst* (``os.replace``), retried with a short
    backoff while Windows refuses it with the sharing or access violation an
    open handle on a file inside causes (:func:`localm.storekit.replace_atomic`);
    each refused attempt is logged at WARNING with its number. Raises the last
    error when every attempt was refused, any other OSError at once."""
    from localm.debuglog import logger
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
    from localm.debuglog import logger
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
                logger.warning("media runtime: %s could not be put back from %s (%s)",
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
            logger.warning("media runtime: the replaced folder %s could not be removed "
                           "(%s); delete it by hand", aside, e)
