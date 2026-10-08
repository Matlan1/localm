# SPDX-License-Identifier: AGPL-3.0-or-later
"""Temp files stay inside the data dir.

``localm.config`` points tempfile and TMP/TEMP/TMPDIR at ``<data dir>/tmp`` when it
is imported, so nothing localm (or a library or child process it starts) writes to
a temp location lands in the system temp folder. ``LOCALM_TMPDIR`` is the explicit
way to choose another place. Each case runs in a fresh interpreter, because the
containment happens at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

_PROBE = r"""
import json, os, subprocess, sys, tempfile
import localm.config
fd, made = tempfile.mkstemp(prefix="probe_")
os.close(fd)
child = subprocess.run(
    [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
    capture_output=True, text=True).stdout.strip()
out = {
    "gettempdir": tempfile.gettempdir(),
    "env": [os.environ.get(v) for v in ("TMP", "TEMP", "TMPDIR")],
    "made": made,
    "child": child,
}
if sys.platform == "win32":
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.GetTempPathW(1024, buf)
    out["win32_temp_path"] = buf.value
print(json.dumps(out))
"""


def _run(tmp_path, *, home, system_temp, extra_env=None):
    system_temp.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop("LOCALM_TMPDIR", None)
    env["LOCALM_HOME"] = str(home)
    for var in ("TMP", "TEMP", "TMPDIR"):
        env[var] = str(system_temp)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, "-c", _PROBE], env=env, cwd=str(_REPO_ROOT),
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1]), r.stderr


def _same(a, b) -> bool:
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(
        os.path.normpath(str(b)))


class TestTempStaysInsideTheDataDir:
    def test_tempfile_and_env_point_into_the_data_dir(self, tmp_path):
        home = tmp_path / "data"
        out, _ = _run(tmp_path, home=home, system_temp=tmp_path / "systemtemp")
        want = home / "tmp"
        assert _same(out["gettempdir"], want)
        assert all(_same(v, want) for v in out["env"]), out["env"]
        assert _same(Path(out["made"]).parent, want)
        assert want.is_dir()

    def test_nothing_lands_in_the_system_temp_folder(self, tmp_path):
        system_temp = tmp_path / "systemtemp"
        _run(tmp_path, home=tmp_path / "data", system_temp=system_temp)
        assert list(system_temp.iterdir()) == []

    def test_a_child_process_inherits_the_data_dir_temp(self, tmp_path):
        home = tmp_path / "data"
        out, _ = _run(tmp_path, home=home, system_temp=tmp_path / "systemtemp")
        assert _same(out["child"], home / "tmp")

    @pytest.mark.skipif(sys.platform != "win32", reason="Win32 temp path")
    def test_native_code_asking_windows_for_the_temp_path_gets_the_data_dir(
            self, tmp_path):
        home = tmp_path / "data"
        out, _ = _run(tmp_path, home=home, system_temp=tmp_path / "systemtemp")
        assert _same(out["win32_temp_path"], home / "tmp")


class TestExplicitChoice:
    def test_localm_tmpdir_is_honoured(self, tmp_path):
        chosen = tmp_path / "chosen"
        out, _ = _run(tmp_path, home=tmp_path / "data",
                      system_temp=tmp_path / "systemtemp",
                      extra_env={"LOCALM_TMPDIR": str(chosen)})
        assert _same(out["gettempdir"], chosen)
        assert chosen.is_dir()


class TestFailureIsSaidNotHidden:
    def test_an_uncreatable_temp_dir_warns_and_leaves_temp_alone(self, tmp_path):
        home = tmp_path / "data"
        home.mkdir()
        (home / "tmp").write_text("a file where the folder belongs", encoding="utf-8")
        system_temp = tmp_path / "systemtemp"
        out, err = _run(tmp_path, home=home, system_temp=system_temp)
        assert "cannot create the temp folder" in err
        assert _same(out["gettempdir"], system_temp)
