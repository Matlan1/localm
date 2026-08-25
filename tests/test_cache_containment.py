# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm's OWN pip/uv caches stay INSIDE the data dir (AGENTS.md rule 4)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import localm.config as config


# --------------------------------------------------------------------------- #
#  config helpers: pip/uv cache dirs + the contained-env builder              #
# --------------------------------------------------------------------------- #

def test_pip_and_uv_cache_dirs_live_inside_the_data_dir(tmp_path):
    """Both caches are siblings under <LOCALM_HOME>/cache, never the user profile."""
    home = tmp_path / ".localm"                      # the autouse LOCALM_HOME
    assert config.pip_cache_dir() == home / "cache" / "pip"
    assert config.uv_cache_dir() == home / "cache" / "uv"
    for p in (config.pip_cache_dir(), config.uv_cache_dir()):
        assert home in p.parents                     # inside the data dir
        if Path.home() not in home.parents:
            assert Path.home() not in p.parents      # NOT the home profile


def test_cache_dirs_follow_localm_home(tmp_path, monkeypatch):
    """Derived from home_dir(), so pointing the data dir moves the caches with it (rule 1: never a hardcoded absolute path)."""
    elsewhere = tmp_path / "moved-home"
    monkeypatch.setenv("LOCALM_HOME", str(elsewhere))
    assert config.pip_cache_dir() == elsewhere / "cache" / "pip"
    assert config.uv_cache_dir() == elsewhere / "cache" / "uv"


def test_contained_pip_env_pins_both_caches(tmp_path):
    """The env handed to an installer subprocess sets PIP_CACHE_DIR AND UV_CACHE_DIR to the contained dirs."""
    env = config.contained_pip_env()
    assert env["PIP_CACHE_DIR"] == str(config.pip_cache_dir())
    assert env["UV_CACHE_DIR"] == str(config.uv_cache_dir())
    # The rest of the environment is carried through, not replaced.
    assert env.get("PATH") == os.environ.get("PATH")


def test_contained_pip_env_overrides_ambient_values(tmp_path, monkeypatch):
    """Containment a stray environment variable can silently switch off is not a guarantee (see config.cache_dir()): an ambient PIP_CACHE_DIR / UV_CACHE_DIR must NOT win over localm's contained location."""
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "ambient-pip"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "ambient-uv"))
    env = config.contained_pip_env()
    assert env["PIP_CACHE_DIR"] == str(config.pip_cache_dir())
    assert env["UV_CACHE_DIR"] == str(config.uv_cache_dir())
    assert "ambient-pip" not in env["PIP_CACHE_DIR"]
    assert "ambient-uv" not in env["UV_CACHE_DIR"]


def test_contained_pip_env_accepts_a_base_env(tmp_path):
    """A caller-supplied base is copied (not mutated) and augmented with the two cache vars."""
    base = {"SOME_VAR": "kept"}
    env = config.contained_pip_env(base=base)
    assert env["SOME_VAR"] == "kept"
    assert env["PIP_CACHE_DIR"] == str(config.pip_cache_dir())
    assert env["UV_CACHE_DIR"] == str(config.uv_cache_dir())
    assert base == {"SOME_VAR": "kept"}              # base not mutated in place


def test_managed_comfy_pip_cache_delegates_to_config(tmp_path):
    """One source of truth for the pip-cache location: the managed-ComfyUI helper resolves to exactly config.pip_cache_dir()."""
    from localm.media import managed_comfy_provision as prov
    assert prov.pip_cache_dir() == config.pip_cache_dir()


# --------------------------------------------------------------------------- #
#  Wiring: the installers pass the contained env to their subprocess          #
#  (these FAIL on the pre-fix code, which passed no env= at all)              #
# --------------------------------------------------------------------------- #

class _FakeProc:
    def __init__(self, lines, rc):
        self.stdout = iter(lines)
        self.returncode = rc

    def wait(self):
        pass


def test_deps_run_pip_hands_the_contained_env_to_the_child(monkeypatch):
    """plugins/deps.py::_run_pip must launch pip/uv with the contained cache env, or a plugin-extra install leaks wheels to the user profile."""
    from localm.plugins import deps
    captured = {}

    def fake_popen(cmd, **kw):
        captured["env"] = kw.get("env")
        return _FakeProc(["installed x\n"], 0)      # succeed on the first (uv) try

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    ok, _out = deps._run_pip(["x>=1"])
    assert ok is True
    env = captured["env"]
    assert env is not None, "no env= passed to the installer subprocess (cache leaks)"
    assert env["PIP_CACHE_DIR"] == str(config.pip_cache_dir())
    assert env["UV_CACHE_DIR"] == str(config.uv_cache_dir())


def test_setup_llama_runtime_wheel_hands_the_contained_env_to_the_child(monkeypatch):
    """setup_llama.py::_install_runtime_wheel must launch pip/uv with the contained cache env, same as the plugin-extra installer."""
    from localm import setup_llama
    captured = {}

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kw):
        captured["env"] = kw.get("env")
        return _R()

    monkeypatch.setattr(setup_llama.subprocess, "run", fake_run)
    ok = setup_llama._install_runtime_wheel(Path("some/pkg/dir"))
    assert ok is True
    env = captured["env"]
    assert env is not None, "no env= passed to the runtime-wheel install (cache leaks)"
    assert env["PIP_CACHE_DIR"] == str(config.pip_cache_dir())
    assert env["UV_CACHE_DIR"] == str(config.uv_cache_dir())


# --------------------------------------------------------------------------- #
#  The EFFECT, not the setting: a REAL pip child honours the contained cache  #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pip_venv(tmp_path_factory):
    """A throwaway venv WITH pip (localm's own .venv ships without one)."""
    venv_dir = tmp_path_factory.mktemp("pipvenv") / "v"
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                       check=True, capture_output=True, text=True, timeout=180)
    except Exception as e:                            # no venv/ensurepip available
        pytest.skip(f"could not build a pip venv: {e}")
    py = (venv_dir / ("Scripts" if os.name == "nt" else "bin") /
          ("python.exe" if os.name == "nt" else "python"))
    if not py.is_file():
        pytest.skip("venv python not found")
    probe = subprocess.run([str(py), "-c", "import pip"], capture_output=True)
    if probe.returncode != 0:
        pytest.skip("venv has no pip (ensurepip unavailable)")
    return py


def test_real_pip_child_resolves_its_cache_inside_the_data_dir(pip_venv, tmp_path):
    """A real pip, launched with config.contained_pip_env() (the exact env the installers hand their subprocess), reports a cache dir inside the data dir."""
    home = tmp_path / ".localm"
    env = config.contained_pip_env(
        base=dict(os.environ, PIP_CACHE_DIR=str(tmp_path / "ambient")))
    out = subprocess.run(
        [str(pip_venv), "-m", "pip", "cache", "dir", "--disable-pip-version-check"],
        capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    reported = Path(out.stdout.strip().splitlines()[-1].strip())
    assert reported == config.pip_cache_dir(), out.stdout   # Windows: case-insensitive eq
    assert home in reported.parents, out.stdout
    if Path.home() not in home.parents:
        assert Path.home() not in reported.parents, out.stdout  # not the user profile
