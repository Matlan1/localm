# SPDX-License-Identifier: AGPL-3.0-or-later
"""Steps shared by the installers of the native media runtimes
(stable-diffusion.cpp, KoboldCpp) and the ComfyUI source download: progress
that also reaches the debug log, replacing an installed folder without losing
it, and the remedy for a download refused because network access is off.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

Progress = Callable[[str], None]


class ReplaceError(OSError):
    """An installed folder could not be replaced; it is still in place."""


def net_off_remedy() -> str:
    """The remedy for a download refused because network access is off, for a
    GUI and a terminal reader alike."""
    return ("Network access is off. Turn it on, or allow model downloads while network "
            "access is off, in Settings > Server & network (or run: localm config "
            "net_mode ask).")


def refused_download_message(error) -> str:
    """The text for a ``netpolicy.NetworkPolicyError`` that refused a download:
    :func:`net_off_remedy` for the net_mode=off refusal (``error.off``), else
    the policy's own reason."""
    if getattr(error, "off", False):
        return f"the download was refused: {net_off_remedy()}"
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


def replace_dir(staging: Path, dest: Path, aside: Path) -> None:
    """Put the folder *staging* in place of *dest*. An existing *dest* is first
    renamed to *aside* (same volume, which must not exist), so a folder the
    system keeps open is refused whole and nothing in it is removed; when the
    move of *staging* then fails, *aside* is put back. The caller removes
    *aside* afterwards. Raises :class:`ReplaceError` with *dest* unchanged."""
    had_old = dest.exists()
    if had_old:
        try:
            os.replace(dest, aside)
        except OSError as e:
            raise ReplaceError(f"{dest} is in use and was left as it is ({e})") from e
    try:
        os.replace(staging, dest)
    except OSError as e:
        if had_old:
            os.replace(aside, dest)
        raise ReplaceError(f"the new files could not be moved into {dest} ({e}); the "
                           "installed ones were kept") from e
