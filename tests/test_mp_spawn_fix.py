# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression pin for #617: a GGUF model load (or the voice/STT worker) must not
fail with "[WinError 2] The system cannot find the file specified" when running
under the branded LocaLM.exe launcher.

Root cause: CPython's multiprocessing spawns a Windows child via
``sys._base_executable`` whenever it differs from ``sys.executable`` (its
definition of "running in a venv") - and computes that path as
``<base_prefix>/<basename of the running exe>``, which does not exist once the
running exe has been renamed (LocaLM.exe, see applaunch.py's
``make_windows_launcher``). ``ensure_spawn_uses_venv_python`` repoints
multiprocessing at the venv's own (never renamed) python.exe first, so this
substitution has a real file to resolve. See localm/_mp_spawn.py.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.spawn
import sys

import pytest

from localm import _mp_spawn


@pytest.fixture(autouse=True)
def _restore_mp_executable():
    # multiprocessing.set_executable mutates module-global state; never leak a
    # test-set value into a later test or the real interpreter's own default.
    original = multiprocessing.spawn.get_executable()
    yield
    multiprocessing.set_executable(original)


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    before = multiprocessing.spawn.get_executable()
    _mp_spawn.ensure_spawn_uses_venv_python()
    assert multiprocessing.spawn.get_executable() == before


def test_repoints_to_venv_python_on_windows(monkeypatch, tmp_path):
    # A fake venv layout: sys.prefix is the venv root, and a never-renamed
    # python.exe sits at <prefix>/Scripts/python.exe - exactly the file the
    # branded LocaLM.exe copy (a DIFFERENT, renamed file) must be bypassed in
    # favour of, per the module docstring.
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    venv_python = scripts_dir / "python.exe"
    venv_python.write_bytes(b"")   # existence is all ensure_spawn_uses_venv_python checks

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))

    _mp_spawn.ensure_spawn_uses_venv_python()

    assert multiprocessing.spawn.get_executable() == str(venv_python)


def test_missing_venv_python_leaves_default_untouched(monkeypatch, tmp_path):
    # No Scripts/python.exe under this prefix (an unexpected layout) - must not
    # point multiprocessing at a file that does not exist.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    before = multiprocessing.spawn.get_executable()

    _mp_spawn.ensure_spawn_uses_venv_python()

    assert multiprocessing.spawn.get_executable() == before
