# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test helpers for the POSIX scripts' fixed system folders.

setup.sh, setup-gui.sh, report-issue.sh and rollback.sh set PATH to the folders
named on their ``LOCALM_SYSDIRS=`` line and find curl, wget, the sha256 tools,
perl and the GPU probes there only. A test that stands in a stub for one of
those programs copies the script and rewrites that one line to list the stub
folder first.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SYSDIRS_LINE = 'LOCALM_SYSDIRS="/usr/bin:/bin:/usr/sbin:/sbin"'
PINNED_UV = "0.13.6"


def bash_path(p: Path | str) -> str:
    """*p* as bash spells it (``/d/x`` for ``D:\\x`` on Windows)."""
    p = Path(p)
    if os.name == "nt" and p.drive:
        return "/" + p.drive[0].lower() + p.as_posix()[len(p.drive):]
    return p.as_posix()


def with_sysdirs(text: str, *first: Path | str) -> str:
    """*text* with its ``LOCALM_SYSDIRS`` line listing *first*, then /usr/bin and /bin."""
    assert text.count(SYSDIRS_LINE) == 1, "the LOCALM_SYSDIRS line moved"
    dirs = [bash_path(d) for d in first] + ["/usr/bin", "/bin"]
    return text.replace(SYSDIRS_LINE, 'LOCALM_SYSDIRS="' + ":".join(dirs) + '"')


def copy_with_sysdirs(src: Path, dest: Path, *first: Path | str) -> None:
    """Copies the script *src* to *dest* with :func:`with_sysdirs` applied."""
    text = src.read_bytes().decode("utf-8")
    dest.write_bytes(with_sysdirs(text, *first).encode("utf-8"))


def fake_uv_script(body: str = "exit 0\n", version: str = PINNED_UV) -> str:
    """A shell script standing in for uv: ``--version`` prints *version*, any other
    call runs *body*."""
    return ('#!/bin/sh\n'
            f'if [ "$1" = --version ]; then echo "uv {version} (stub)"; exit 0; fi\n'
            + body)


def sys_prog_function(text: str) -> str:
    """The ``sys_prog`` function of a script's text, as written there."""
    start = text.index("sys_prog() {")
    end = text.index("\n}\n", start) + 3
    return text[start:end]


def sysdirs_for(*programs: str) -> str:
    """A ``LOCALM_SYSDIRS=`` line listing the folders *programs* are found in on
    this machine's PATH, then /usr/bin and /bin."""
    dirs = []
    for name in programs:
        found = shutil.which(name)
        if found:
            dirs.append(bash_path(Path(found).parent))
    return 'LOCALM_SYSDIRS="' + ":".join([*dirs, "/usr/bin", "/bin"]) + '"\n'
