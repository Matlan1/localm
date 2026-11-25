# SPDX-License-Identifier: AGPL-3.0-or-later
"""Steps shared by the installers of the native media runtimes
(stable-diffusion.cpp, KoboldCpp) and the ComfyUI source download: progress
that also reaches the debug log, the text of a refused download, and putting a
prepared runtime in place (:mod:`localm.update_mechanism.steps`).
"""

from __future__ import annotations

from pathlib import Path
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


def put_runtime(root: Path, name: str, staging: Path, *, record: bool, error: type,
                busy: str) -> None:
    """Make the prepared folder *staging* the runtime folder *name* of *root*: with
    *record* under a record, so that a failure or a killed process leaves what was
    there, else by one rename to a name that is free. Raises *error*, saying *busy*
    when a file is held."""
    from localm.update_mechanism import disk, steps
    try:
        if record:
            steps.put_folder(root, name, staging)
        else:
            disk.move(staging, Path(root) / name, disk.Wait())
    except (steps.Failed, *steps.STEP_ERRORS) as e:
        held = isinstance(e, disk.Held) or isinstance(getattr(e, "__cause__", None), disk.Held)
        if getattr(e, "errors", None):
            raise error(str(e)) from e
        if not held:
            raise error(f"the new files could not be moved into {Path(root) / name} ({e}); the "
                        "installed ones were kept") from e
        raise error(f"{Path(root) / name} is in use and was left as it is ({e}); {busy}") from e


def settle_runtimes(root: Path, error: type) -> bool:
    """End what an interrupted install left in the runtimes folder *root*, or wait for
    one that moves its files. Returns whether its files are incomplete for good, so
    that the runtime has to be installed again. Raises *error* when a program has to
    end first."""
    from localm.debuglog import logger
    from localm.update_mechanism import steps
    try:
        broken = steps.mend(root)
    except (steps.Failed, *steps.STEP_ERRORS) as e:
        raise error(f"an interrupted install in {root} could not be undone: {e}") from e
    if broken:
        logger.warning("an interrupted install in %s cannot be undone; the runtime is "
                       "installed again", root)
    return broken


def logged(on_progress: Optional[Progress], tag: str) -> Progress:
    """*on_progress* that also writes each message to the debug log at INFO,
    prefixed with *tag*. *on_progress* may be None."""
    from localm.debuglog import logger

    def say(message: str) -> None:
        logger.info("%s: %s", tag, message)
        if on_progress is not None:
            on_progress(message)
    return say
