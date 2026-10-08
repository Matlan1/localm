# SPDX-License-Identifier: AGPL-3.0-or-later
"""Where the cross-install GPU registry lives: independent of TMP/TEMP/TMPDIR.

Two localm processes of one OS user must resolve the SAME registry directory
even when they were started with different temp environments (a launcher, a
service, an installer, a test harness), otherwise VRAM cooperation is blind.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest

from localm import gpu_registry

# Captured at import: test_gpu_registry.py patches the module attribute per test.
_real_registry_dir = gpu_registry.registry_dir
_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """No override, and a fake user home / cache root under tmp_path."""
    home = tmp_path / "home"
    local = tmp_path / "localappdata"
    xdg = tmp_path / "xdgcache"
    monkeypatch.delenv(gpu_registry.REGISTRY_DIR_ENV, raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))
    return types.SimpleNamespace(home=home, local=local, xdg=xdg)


def _with_platform(monkeypatch, platform):
    monkeypatch.setattr(gpu_registry, "sys", types.SimpleNamespace(platform=platform))


class TestRegistryDirIgnoresTempEnvironment:
    def test_same_directory_under_different_temp_environments(
            self, clean_env, tmp_path, monkeypatch):
        seen = []
        for name in ("tempA", "tempB"):
            t = tmp_path / name
            t.mkdir()
            for var in ("TMP", "TEMP", "TMPDIR"):
                monkeypatch.setenv(var, str(t))
            monkeypatch.setattr(tempfile, "tempdir", None)
            seen.append(_real_registry_dir())
        assert seen[0] == seen[1]
        assert tmp_path / "tempA" not in seen[0].parents
        assert tmp_path / "tempB" not in seen[1].parents

    def test_two_real_interpreters_with_different_temp_agree(self, clean_env, tmp_path):
        code = ("from localm import gpu_registry; "
                "print(gpu_registry.registry_dir())")
        outs = []
        for name in ("tempA", "tempB"):
            t = tmp_path / name
            t.mkdir()
            env = dict(os.environ)
            env.pop(gpu_registry.REGISTRY_DIR_ENV, None)
            for var in ("TMP", "TEMP", "TMPDIR"):
                env[var] = str(t)
            env["PYTHONPATH"] = str(_REPO_ROOT)
            r = subprocess.run([sys.executable, "-c", code], env=env,
                               cwd=str(_REPO_ROOT), capture_output=True,
                               text=True, timeout=120)
            assert r.returncode == 0, r.stderr
            outs.append(r.stdout.strip().splitlines()[-1])
        assert outs[0] == outs[1]
        assert str(tmp_path / "tempA") not in outs[0]

    def test_override_env_wins(self, clean_env, tmp_path, monkeypatch):
        target = tmp_path / "explicit"
        monkeypatch.setenv(gpu_registry.REGISTRY_DIR_ENV, str(target))
        assert _real_registry_dir() == target

    def test_empty_override_is_ignored(self, clean_env, monkeypatch):
        monkeypatch.setenv(gpu_registry.REGISTRY_DIR_ENV, "")
        assert _real_registry_dir().name == "gpu"
        assert clean_env.xdg in _real_registry_dir().parents or \
            clean_env.local in _real_registry_dir().parents


class TestPerPlatformCacheRoot:
    def test_windows_uses_localappdata(self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "win32")
        assert _real_registry_dir() == clean_env.local / "localm" / "gpu"

    def test_windows_without_localappdata_falls_back_under_home(
            self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "win32")
        monkeypatch.delenv("LOCALAPPDATA")
        assert _real_registry_dir() == (
            clean_env.home / "AppData" / "Local" / "localm" / "gpu")

    def test_macos_uses_library_caches(self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "darwin")
        assert _real_registry_dir() == (
            clean_env.home / "Library" / "Caches" / "localm" / "gpu")

    def test_linux_honours_absolute_xdg_cache_home(self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "linux")
        assert _real_registry_dir() == clean_env.xdg / "localm" / "gpu"

    def test_linux_without_xdg_uses_dot_cache(self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "linux")
        monkeypatch.delenv("XDG_CACHE_HOME")
        assert _real_registry_dir() == clean_env.home / ".cache" / "localm" / "gpu"

    def test_linux_ignores_a_relative_xdg_cache_home(self, clean_env, monkeypatch):
        _with_platform(monkeypatch, "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "relative/cache")
        assert _real_registry_dir() == clean_env.home / ".cache" / "localm" / "gpu"


class TestSuiteIsolation:
    def test_conftest_pins_the_registry_away_from_the_user_cache(self):
        override = os.environ.get(gpu_registry.REGISTRY_DIR_ENV)
        assert override, "conftest must pin LOCALM_GPU_REGISTRY_DIR"
        assert _real_registry_dir() == Path(override)
        assert gpu_registry._user_cache_root() not in Path(override).parents
