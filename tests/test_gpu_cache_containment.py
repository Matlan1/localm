# SPDX-License-Identifier: AGPL-3.0-or-later
"""The AMD GPU compiler caches stay inside the data folder.

The HIP runtime's comgr and MIOpen write kernel caches under the user profile unless
AMD_COMGR_CACHE_DIR, MIOPEN_USER_DB_PATH and MIOPEN_CUSTOM_CACHE_DIR are set.
config.contain_gpu_caches() pins them inside the data dir when localm.config is imported.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm import config

_PINNED = {
    "AMD_COMGR_CACHE_DIR": ("cache", "comgr"),
    "MIOPEN_USER_DB_PATH": ("cache", "miopen", "db"),
    "MIOPEN_CUSTOM_CACHE_DIR": ("cache", "miopen", "cache"),
}
_VARS = tuple(_PINNED) + tuple(config._gpu_pin_marker(v) for v in _PINNED)


def _clean_env(monkeypatch):
    for var in _VARS:
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def test_every_gpu_cache_is_pinned_inside_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_gpu_cache_env({})
    for var, parts in _PINNED.items():
        assert Path(env[var]) == tmp_path.joinpath("data", *parts)
        assert env[config._gpu_pin_marker(var)] == env[var]
    assert {v: Path(p) for v, p in config.gpu_cache_dirs().items()} == {
        v: Path(env[v]) for v in _PINNED}


@pytest.mark.parametrize("var", list(_PINNED))
def test_the_pin_marker_name_is_the_variable_under_the_localm_pinned_prefix(var):
    assert config._gpu_pin_marker(var) == "LOCALM_PINNED_" + var


@pytest.mark.parametrize("var", list(_PINNED))
def test_a_user_set_variable_is_left_exactly_as_set(monkeypatch, tmp_path, var):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_gpu_cache_env({var: str(tmp_path / "mine")})
    assert env[var] == str(tmp_path / "mine")
    assert config._gpu_pin_marker(var) not in env
    for other in _PINNED:
        if other != var:
            assert Path(env[other]) == tmp_path.joinpath("data", *_PINNED[other])


def test_the_comgr_on_off_switch_is_never_touched(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    assert "AMD_COMGR_CACHE" not in config.contained_gpu_cache_env({})
    assert config.contained_gpu_cache_env({"AMD_COMGR_CACHE": "0"})["AMD_COMGR_CACHE"] == "0"


def test_an_inherited_pin_is_repinned_for_another_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "first"))
    inherited = config.contained_gpu_cache_env({})
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "second"))
    env = config.contained_gpu_cache_env(inherited)
    for var, parts in _PINNED.items():
        assert Path(env[var]) == tmp_path.joinpath("second", *parts)


def test_a_relative_data_dir_is_pinned_as_absolute_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCALM_HOME", "reldata")
    env = config.contained_gpu_cache_env({})
    for var, parts in _PINNED.items():
        assert os.path.isabs(env[var])
        assert _norm(env[var]) == _norm(tmp_path.joinpath("reldata", *parts))


def test_contain_gpu_caches_applies_to_the_process_environment(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    config.contain_gpu_caches()
    for var, parts in _PINNED.items():
        assert Path(os.environ[var]) == tmp_path.joinpath("data", *parts)


def test_contain_gpu_caches_leaves_a_user_value_alone(monkeypatch, tmp_path):
    _clean_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("AMD_COMGR_CACHE_DIR", str(tmp_path / "mine"))
    config.contain_gpu_caches()
    assert os.environ["AMD_COMGR_CACHE_DIR"] == str(tmp_path / "mine")
    assert config._gpu_pin_marker("AMD_COMGR_CACHE_DIR") not in os.environ
    assert Path(os.environ["MIOPEN_USER_DB_PATH"]) == tmp_path / "data" / "cache" / "miopen" / "db"


def _fresh_interpreter(tmp_path: Path, code: str, extra_env=None) -> str:
    env = {k: v for k, v in os.environ.items() if k not in _VARS}
    env.update({
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "LOCALM_HOME": str(tmp_path / "data"),
        "PYTHONPATH": str(Path(config.__file__).resolve().parents[1]),
    })
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_a_child_process_inherits_the_pins(tmp_path):
    out = _fresh_interpreter(tmp_path, (
        "import localm.config, subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-c', 'import os; "
        "print(os.environ[\"AMD_COMGR_CACHE_DIR\"]); "
        "print(os.environ[\"MIOPEN_USER_DB_PATH\"]); "
        "print(os.environ[\"MIOPEN_CUSTOM_CACHE_DIR\"])'], capture_output=True, text=True)\n"
        "print(r.stdout.strip())\n"))
    got = [_norm(line) for line in out.splitlines()]
    assert got == [_norm(tmp_path.joinpath("data", *parts)) for parts in _PINNED.values()]


def test_a_user_value_survives_import_in_a_fresh_interpreter(tmp_path):
    mine = tmp_path / "mine"
    out = _fresh_interpreter(tmp_path, (
        "import localm.config, os\n"
        "print(os.environ['MIOPEN_USER_DB_PATH'])\n"),
        extra_env={"MIOPEN_USER_DB_PATH": str(mine)})
    assert _norm(out) == _norm(mine)
