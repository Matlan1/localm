# SPDX-License-Identifier: AGPL-3.0-or-later
"""The installer guard must actually FIRE, must BLOCK, and must not cry wolf.

conftest's installer guard exists because a test reaching a real package
installer against the suite's own interpreter is not a tidiness problem: while a
wheel unpacks, its directory exists with no ``__init__.py``, which is a PEP 420
namespace package, so the module imports SUCCESSFULLY with no attributes. That
took a day to attribute because it fails an unrelated test in another worker,
hundreds of lines from the cause, and then erases itself when the install
finishes.

A guard nobody has proven can fire is worth nothing, and a guard that fires on
everything is worse because it gets exempted. So both directions are covered.

The live guard is reached through ``subprocess.Popen.__init__.__globals__``
rather than by importing conftest again: re-importing it would RE-EXECUTE the
module, creating a second temp home and re-arming every guard in it (the sibling
syspath-guard test documents the same hazard). This way the assertions are
against the object that is actually installed in this session, which is the only
one that matters.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _live():
    """The conftest globals of the guard installed in THIS session."""
    return subprocess.Popen.__init__.__globals__


def _disposable_python() -> str:
    """A plausible python inside a throwaway venv, i.e. NOT this interpreter."""
    return os.path.join("C:" + os.sep if os.name == "nt" else os.sep,
                        "tmp", "user_comfy0", "venv", "Scripts", "python.exe")


def test_the_guard_is_armed_in_this_session():
    """Without this, every assertion below could pass while the shipped guard sat
    inert, and green would mean 'never ran'."""
    assert subprocess.Popen.__name__ == "_GuardedPopen", (
        "conftest's installer guard is not armed; the rest of this file proves "
        "nothing about the shipped code")
    assert "_installs_into_this_interpreter" in _live()


def test_it_blocks_an_install_into_this_interpreter():
    """FIRES-CONTROL, end to end through the real patched Popen.

    Nothing is spawned: the guard raises before delegating to the real Popen, so
    this proves the block without installing anything anywhere."""
    hits = _live()["_INSTALLER_HITS"]
    before = dict(hits)
    with pytest.raises(RuntimeError) as excinfo:
        subprocess.Popen([sys.executable, "-m", "pip", "install", "definitely-not-real"])
    assert "BLOCKED" in str(excinfo.value)
    assert "interpreter running the suite" in str(excinfo.value)

    # This test TRIPS the guard deliberately, and the autouse fixture fails any
    # test that leaves a new hit behind. Drop only the entries this test added,
    # so the fixture sees a clean diff and a real hit from anywhere else in the
    # session is still reported.
    for key in [k for k in hits if k not in before]:
        del hits[key]


@pytest.mark.parametrize("argv", [
    # the exact command captured from the CI process table when this was found
    ["uv", "pip", "install", "--python", sys.executable, "faster-whisper>=1.2.1"],
    [sys.executable, "-m", "pip", "install", "x>=1"],
    # No explicit target: uv without --python, and the bare pip shims found on
    # PATH. Under the suite, PATH leads to the venv running it. The bare forms are
    # the ones the FIRST version of this guard missed - it matched the NAME but
    # then resolved the target as a path, and a bare `pip` has no directory
    # component, so it landed on the CWD and matched nothing. That made the
    # original 4/4 fires-control a 4/4 on fully qualified commands only.
    ["uv", "pip", "install", "somepkg"],
    ["pip", "install", "somepkg"],
    ["pip3", "install", "somepkg"],
    ["pip3.12", "install", "somepkg"],
])
def test_the_oracle_catches_installs_aimed_at_this_interpreter(argv):
    assert _live()["_installs_into_this_interpreter"](argv), argv


@pytest.mark.parametrize("argv", [
    # LEGITIMATE: installing into a disposable venv under tmp_path. The
    # managed-ComfyUI tests do exactly this and must keep working - the guard
    # draws its line at the TARGET, not at the act of installing.
    [_disposable_python(), "-m", "pip", "install", "--no-index", "localms2pkg"],
    ["uv", "pip", "install", "--python", _disposable_python(), "somepkg"],
    # a path-qualified pip that is genuinely NOT this interpreter
    ["/usr/bin/pip.exe", "install", "somepkg"],
    # real pip children that install NOTHING (the cache-containment tests)
    [_disposable_python(), "-m", "pip", "cache", "dir", "--disable-pip-version-check"],
    [sys.executable, "-m", "pip", "freeze"],
    [sys.executable, "-m", "pip", "list"],
    ["pip", "list"],
    ["pip", "--version"],
    # pipx is a different tool and must not be swept up by the pip[0-9.]* shim match
    ["pipx", "run", "somepkg"],
    # not an installer at all, and a stray 'install' word must not be enough
    ["git", "status"],
    ["git", "commit", "-m", "install docs"],
    ["grep", "-rn", "pip install", "docs/"],
    [sys.executable, "-c", "print('hi')"],
])
def test_the_oracle_stays_quiet_on_everything_else(argv):
    assert _live()["_installs_into_this_interpreter"](argv) is None, argv
