# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the GPU registry lives: inside the install's data dir unless the user opts out.

Nothing is written outside the install by default. Two processes sharing a data
dir resolve the same registry whatever their TMP/TEMP/TMPDIR; installs with
different data dirs share a registry only through the explicit
``LOCALM_GPU_REGISTRY_DIR`` override.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from localm import gpu_registry

# Captured at import: test_gpu_registry.py patches the module attribute per test.
_real_registry_dir = gpu_registry.registry_dir
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def default_env(tmp_path, monkeypatch):
    """No override, and a LOCALM_HOME under tmp_path."""
    home = tmp_path / "data"
    monkeypatch.delenv(gpu_registry.REGISTRY_DIR_ENV, raising=False)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    return home


def _subprocess_registry_dir(tmp_path, *, home, temp_name):
    t = tmp_path / temp_name
    t.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.pop(gpu_registry.REGISTRY_DIR_ENV, None)
    env["LOCALM_HOME"] = str(home)
    for var in ("TMP", "TEMP", "TMPDIR"):
        env[var] = str(t)
    env["PYTHONPATH"] = str(_REPO_ROOT)
    r = subprocess.run(
        [sys.executable, "-c",
         "from localm import gpu_registry; print(gpu_registry.registry_dir())"],
        env=env, cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return Path(r.stdout.strip().splitlines()[-1])


class TestDefaultIsInsideTheDataDir:
    def test_default_is_under_the_data_dir(self, default_env):
        assert _real_registry_dir() == default_env / "run" / "gpu"

    def test_default_never_depends_on_the_temp_environment(
            self, default_env, tmp_path, monkeypatch):
        seen = []
        for name in ("tempA", "tempB"):
            t = tmp_path / name
            t.mkdir()
            for var in ("TMP", "TEMP", "TMPDIR"):
                monkeypatch.setenv(var, str(t))
            monkeypatch.setattr(tempfile, "tempdir", None)
            seen.append(_real_registry_dir())
        assert seen[0] == seen[1] == default_env / "run" / "gpu"

    def test_two_real_interpreters_sharing_a_data_dir_agree_despite_temp(
            self, tmp_path):
        home = tmp_path / "data"
        a = _subprocess_registry_dir(tmp_path, home=home, temp_name="tempA")
        b = _subprocess_registry_dir(tmp_path, home=home, temp_name="tempB")
        assert a == b == home / "run" / "gpu"

    def test_different_data_dirs_do_not_share_a_registry_by_default(self, tmp_path):
        a = _subprocess_registry_dir(tmp_path, home=tmp_path / "one", temp_name="t")
        b = _subprocess_registry_dir(tmp_path, home=tmp_path / "two", temp_name="t")
        assert a != b
        assert a == tmp_path / "one" / "run" / "gpu"
        assert b == tmp_path / "two" / "run" / "gpu"


class TestExplicitOverride:
    def test_override_env_wins(self, default_env, tmp_path, monkeypatch):
        target = tmp_path / "shared"
        monkeypatch.setenv(gpu_registry.REGISTRY_DIR_ENV, str(target))
        assert _real_registry_dir() == target

    def test_two_data_dirs_with_the_same_override_share_a_registry(self, tmp_path):
        shared = tmp_path / "shared"
        outs = []
        for name in ("one", "two"):
            env = dict(os.environ)
            env["LOCALM_HOME"] = str(tmp_path / name)
            env[gpu_registry.REGISTRY_DIR_ENV] = str(shared)
            env["PYTHONPATH"] = str(_REPO_ROOT)
            r = subprocess.run(
                [sys.executable, "-c",
                 "from localm import gpu_registry; print(gpu_registry.registry_dir())"],
                env=env, cwd=str(_REPO_ROOT), capture_output=True, text=True,
                timeout=120)
            assert r.returncode == 0, r.stderr
            outs.append(Path(r.stdout.strip().splitlines()[-1]))
        assert outs[0] == outs[1] == shared

    def test_empty_override_is_ignored(self, default_env, monkeypatch):
        monkeypatch.setenv(gpu_registry.REGISTRY_DIR_ENV, "")
        assert _real_registry_dir() == default_env / "run" / "gpu"


class TestSuiteIsolation:
    def test_conftest_pins_the_registry_inside_the_throwaway_home(self):
        override = os.environ.get(gpu_registry.REGISTRY_DIR_ENV)
        assert override, "conftest must pin LOCALM_GPU_REGISTRY_DIR"
        assert _real_registry_dir() == Path(override)
