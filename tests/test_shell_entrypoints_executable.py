# SPDX-License-Identifier: AGPL-3.0-or-later
"""Every .sh a user is told to run must be executable in git.

A fresh `git clone` materialises the mode git recorded, so a shell entry point
stored 100644 cannot be started with `./name.sh` at all. setup-gui.sh shipped
that way: the changelog tells Linux and macOS users to run `./setup-gui.sh`,
and setup.sh only offers the graphical installer when `[ -x ./setup-gui.sh ]`,
so both the direct instruction and the console installer's own pointer to it
failed on every fresh clone.

The mode is read from GIT, not from the filesystem: a Windows checkout has no
POSIX permission bits, so os.access would answer a different question here
than the one a Linux user's clone asks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The .sh files a user or an installer invokes directly.
ENTRY_POINTS = [
    "setup.sh",
    "setup-gui.sh",
    "install.sh",
    "localm.sh",
    "localm-launcher.sh",
    "report-issue.sh",
    "rollback.sh",
]


def _index_modes() -> dict:
    out = subprocess.run(
        ["git", "ls-files", "-s", "--", *ENTRY_POINTS],
        cwd=str(ROOT), capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip(f"git could not read the index: {out.stderr.strip()}")
    modes = {}
    for line in out.stdout.splitlines():
        if not line.strip():
            continue
        meta, _, path = line.partition("\t")
        modes[path.strip()] = meta.split()[0]
    return modes


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_shell_entry_point_is_executable_in_git(name):
    modes = _index_modes()
    if name not in modes:
        pytest.skip(f"{name} is not tracked")
    assert modes[name] == "100755", (
        f"{name} is stored {modes[name]}; a fresh clone cannot run ./{name}")
