# SPDX-License-Identifier: AGPL-3.0-or-later
"""Steps shared by the installers of the native media runtimes
(stable-diffusion.cpp, KoboldCpp) and the ComfyUI source download: progress
that also reaches the debug log and the text of a refused download. Moving and
replacing an installed folder is :mod:`localm.folder_swap`.
"""

from __future__ import annotations

from typing import Callable, Optional

Progress = Callable[[str], None]


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
