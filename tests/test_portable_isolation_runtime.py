# SPDX-License-Identifier: AGPL-3.0-or-later
"""A Portable or Shared install uses only what localm brought.

Covers the runtime side: the install record decides Portable, an account-wide
LOCALM_HOME or LOCALM_TMPDIR does not move a Portable install's folders, a
Portable install keeps its temp files inside the install, child processes get no
inherited uv / pip / Python search settings, every cache a library writes is
pinned into the data folder (whatever the user set, in Portable), the updater's
dependency step uses the install's own store, find_uv never takes a uv from PATH
or the user profile, and icacls runs by full path without a window.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from localm import config, install_manifest, updater, uv_tool


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def _write_record(root: Path, **fields) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / config.INSTALL_RECORD_NAME).write_text(json.dumps(fields), encoding="utf-8")


# --------------------------------------------------------------------------- #
#  The install record                                                          #
# --------------------------------------------------------------------------- #

def test_the_record_names_match_the_install_manifest():
    assert config.INSTALL_RECORD_NAME == install_manifest.MANIFEST_NAME
    assert config.PORTABLE_TEMP_NAME == install_manifest.PORTABLE_TEMP_NAME


def test_portable_is_read_from_the_install_record(tmp_path):
    assert not config.setup_install(tmp_path)
    assert not config.portable_install(tmp_path)
    _write_record(tmp_path, runtime_contained=False)
    assert config.setup_install(tmp_path)
    assert not config.portable_install(tmp_path)
    _write_record(tmp_path, runtime_contained=True, extra="x" * 10)
    assert config.portable_install(tmp_path)


def test_an_unreadable_record_is_not_portable(tmp_path):
    (tmp_path / config.INSTALL_RECORD_NAME).write_text("{not json", encoding="utf-8")
    assert config.setup_install(tmp_path)
    assert not config.portable_install(tmp_path)


# --------------------------------------------------------------------------- #
#  R3: LOCALM_HOME / LOCALM_TMPDIR in a Portable install                       #
# --------------------------------------------------------------------------- #

@pytest.fixture
def portable(monkeypatch):
    monkeypatch.setattr(config, "portable_install", lambda root=None: True)


@pytest.fixture
def account_wide(monkeypatch):
    stored: dict = {}
    monkeypatch.setattr(config, "_account_wide_env_value", lambda name: stored.get(name))
    return stored


def test_an_account_wide_localm_home_is_ignored_in_a_portable_install(
        portable, account_wide, monkeypatch, tmp_path, capsys):
    other = str(tmp_path / "someone-elses-data")
    account_wide["LOCALM_HOME"] = other
    monkeypatch.setenv("LOCALM_HOME", other)
    assert config.home_override() is None
    assert _norm(config._detect_home()) != _norm(other)
    assert "ignores it" in capsys.readouterr().err


def test_a_localm_home_set_for_this_process_is_used_in_a_portable_install(
        portable, account_wide, monkeypatch, tmp_path):
    account_wide["LOCALM_HOME"] = str(tmp_path / "account")
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "this-run"))
    assert config.home_override() == str(tmp_path / "this-run")
    assert config._detect_home() == tmp_path / "this-run"


def test_an_account_wide_localm_home_is_used_outside_a_portable_install(
        account_wide, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "portable_install", lambda root=None: False)
    account_wide["LOCALM_HOME"] = str(tmp_path / "account")
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "account"))
    assert config._detect_home() == tmp_path / "account"


def test_an_account_wide_localm_tmpdir_is_ignored_in_a_portable_install(
        portable, account_wide, monkeypatch, tmp_path):
    account_wide["LOCALM_TMPDIR"] = str(tmp_path / "elsewhere")
    monkeypatch.setenv("LOCALM_TMPDIR", str(tmp_path / "elsewhere"))
    assert config.temp_dir(tmp_path / "data", warn=False) != tmp_path / "elsewhere"


@pytest.mark.skipif(sys.platform != "win32", reason="the account environment lives in the registry")
def test_the_account_wide_value_is_read_from_the_registry():
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            expected, _ = winreg.QueryValueEx(key, "TEMP")
        except OSError:
            pytest.skip("no TEMP in HKCU\\Environment")
    assert config._account_wide_env_value("TEMP") == os.path.expandvars(expected)
    assert config._account_wide_env_value("LOCALM_NO_SUCH_VARIABLE_X") is None


# --------------------------------------------------------------------------- #
#  R2: temp follows the Portable tooling choice                                #
# --------------------------------------------------------------------------- #

def test_portable_temp_stays_in_the_install_with_a_data_folder_outside_it(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "portable_containment", lambda: True)
    monkeypatch.delenv("LOCALM_TMPDIR", raising=False)
    got = config.temp_dir(tmp_path / "custom-data", warn=False)
    assert got == config.install_root() / config.PORTABLE_TEMP_NAME


def test_portable_temp_uses_the_data_folder_when_it_is_inside_the_install(monkeypatch):
    monkeypatch.setattr(config, "portable_containment", lambda: True)
    monkeypatch.delenv("LOCALM_TMPDIR", raising=False)
    home = config.install_root() / "home-for-this-test-never-created"
    assert config.temp_dir(home, warn=False) == home / "tmp"


def test_without_portable_a_data_folder_outside_the_checkout_keeps_the_system_temp(
        monkeypatch, tmp_path):
    monkeypatch.setattr(config, "portable_containment", lambda: False)
    monkeypatch.delenv("LOCALM_TMPDIR", raising=False)
    assert config.temp_dir(tmp_path / "custom-data", warn=False) is None


def test_portable_containment_needs_the_record_and_no_process_home(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "portable_install", lambda root=None: True)
    monkeypatch.setattr(config, "_account_wide_env_value", lambda name: None)
    monkeypatch.delenv("LOCALM_HOME", raising=False)
    assert config.portable_containment()
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    assert not config.portable_containment()
    monkeypatch.setattr(config, "portable_install", lambda root=None: False)
    monkeypatch.delenv("LOCALM_HOME")
    assert not config.portable_containment()


def test_uninstall_removes_the_portable_temp_folder(tmp_path):
    (tmp_path / config.PORTABLE_TEMP_NAME / "x").mkdir(parents=True)
    items, _ = install_manifest._plan(tmp_path, None, purge_data=False, defer_runtime=False)
    assert any(_norm(i.path) == _norm(tmp_path / config.PORTABLE_TEMP_NAME)
               and i.status == "remove" for i in items)


# --------------------------------------------------------------------------- #
#  R4: no inherited uv / pip / Python settings for children                    #
# --------------------------------------------------------------------------- #

_POISON = {"UV_INDEX_URL": "http://evil.invalid/simple", "UV_PYTHON": "C:/Python39/python.exe",
           "PIP_INDEX_URL": "http://evil.invalid/simple", "PIP_CONFIG_FILE": "C:/pip.ini",
           "PYTHONPATH": "C:/elsewhere", "PYTHONHOME": "C:/elsewhere",
           "PYTHONSTARTUP": "C:/elsewhere/s.py", "PYTHONUSERBASE": "C:/elsewhere",
           "VIRTUAL_ENV": "C:/elsewhere/venv", "CONDA_PREFIX": "C:/conda",
           "uv_lower_case": "x"}


def test_isolated_tool_env_drops_inherited_settings_and_sets_its_own(tmp_path):
    base = dict(_POISON, KEEP_ME="yes")
    env = config.isolated_tool_env(base)
    for key in _POISON:
        if key not in config.tool_env_settings():
            assert key not in env, key
    assert env["KEEP_ME"] == "yes"
    assert env["UV_NO_CONFIG"] == "1"
    assert env["PIP_CONFIG_FILE"] == os.devnull
    assert env["PYTHONNOUSERSITE"] == "1"
    assert _norm(env["UV_CREDENTIALS_DIR"]) == _norm(config.cache_dir() / "uv-credentials")
    assert _norm(env["UV_PYTHON_BIN_DIR"]) == _norm(config.cache_dir() / "uv-python-bin")
    assert base == dict(_POISON, KEEP_ME="yes")


def _isolate(monkeypatch, *, made_by_setup: bool):
    monkeypatch.setattr(config, "setup_install", lambda root=None: made_by_setup)
    monkeypatch.setattr(config, "_dropped_tool_env", {})
    monkeypatch.setattr(config, "_tool_env_isolated", False)
    for key, value in _POISON.items():
        monkeypatch.setenv(key, value)
    for key in config.tool_env_settings():
        if key not in _POISON:
            monkeypatch.delenv(key, raising=False)
    config.isolate_tool_env()


def test_a_setup_install_clears_inherited_settings_for_its_children(monkeypatch):
    _isolate(monkeypatch, made_by_setup=True)
    for key in _POISON:
        if key not in config.tool_env_settings():
            assert key not in os.environ, key
    for key, value in config.tool_env_settings().items():
        assert os.environ[key] == value
    out = subprocess.run([sys.executable, "-c",
                          "import os; print(os.environ.get('PYTHONPATH', '-'), "
                          "os.environ.get('UV_INDEX_URL', '-'))"],
                         capture_output=True, text=True, timeout=60).stdout.split()
    assert out == ["-", "-"]


def test_a_source_checkout_without_a_record_is_left_alone(monkeypatch):
    _isolate(monkeypatch, made_by_setup=False)
    for key, value in _POISON.items():
        assert os.environ[key.upper() if os.name == "nt" else key] == value


def test_a_users_own_command_gets_the_cleared_settings_back(monkeypatch):
    _isolate(monkeypatch, made_by_setup=True)
    env = {k.upper(): v for k, v in config.user_command_env().items()}
    for key, value in _POISON.items():
        assert env.get(key.upper()) == value, key
    for key in ("UV_NO_CONFIG", "PYTHONNOUSERSITE", "UV_CREDENTIALS_DIR"):
        assert key not in env


def test_localms_own_installers_never_see_inherited_settings(monkeypatch):
    for key, value in _POISON.items():
        monkeypatch.setenv(key, value)
    env = config.contained_pip_env()
    assert "UV_INDEX_URL" not in env and "PIP_INDEX_URL" not in env
    assert "PYTHONPATH" not in env
    assert env["PIP_CONFIG_FILE"] == os.devnull and env["UV_NO_CONFIG"] == "1"


def test_uv_keeps_its_credentials_store_in_the_data_folder():
    """uv 0.13.6 writes credentials.toml.lock under %APPDATA%\\uv\\credentials
    unless UV_CREDENTIALS_DIR moves it (``uv auth dir`` reports the folder)."""
    import shutil
    uv = uv_tool.find_uv(config.install_root()) or shutil.which("uv")
    if uv is None:
        pytest.skip("no uv to ask")
    env = config.contained_pip_env()
    out = subprocess.run([str(uv), "auth", "dir"], env=env, capture_output=True, text=True,
                         timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert out.returncode == 0, out.stderr
    assert _norm(out.stdout.strip()) == _norm(config.cache_dir() / "uv-credentials")


# --------------------------------------------------------------------------- #
#  R4: caches pinned into the data folder                                      #
# --------------------------------------------------------------------------- #

_TOOL = {"TORCH_HOME": ("torch",), "TORCHINDUCTOR_CACHE_DIR": ("torchinductor",),
         "PYTORCH_KERNEL_CACHE_PATH": ("torch", "kernels"), "TRITON_CACHE_DIR": ("triton",),
         "MPLCONFIGDIR": ("matplotlib",), "NUMBA_CACHE_DIR": ("numba",),
         "XDG_CACHE_HOME": ("xdg",), "PLAYWRIGHT_BROWSERS_PATH": ("ms-playwright",)}


def test_library_and_tool_caches_are_pinned_inside_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_gpu_cache_env({}, portable=False)
    for var, parts in _TOOL.items():
        assert Path(env[var]) == tmp_path.joinpath("data", "cache", *parts), var
    assert set(config.tool_cache_dirs()) == set(_TOOL)


@pytest.mark.parametrize("var", list(_TOOL) + ["MIOPEN_USER_DB_PATH"])
def test_a_user_value_wins_outside_portable_and_not_inside(monkeypatch, tmp_path, var):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    mine = str(tmp_path / "mine")
    assert config.contained_gpu_cache_env({var: mine}, portable=False)[var] == mine
    pinned = config.contained_gpu_cache_env({var: mine}, portable=True)[var]
    assert Path(pinned).is_relative_to(tmp_path / "data" / "cache")


def test_portable_pins_hugging_face_whatever_the_user_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    user = {"HF_HOME": str(tmp_path / "u"), "HF_HUB_CACHE": str(tmp_path / "u" / "hub"),
            "HF_XET_CACHE": str(tmp_path / "u" / "xet"), "HUGGINGFACE_HUB_CACHE": "x",
            "HF_TOKEN_PATH": str(tmp_path / "u" / "token")}
    env = config.contained_hf_env(user, portable=True)
    home = tmp_path / "data" / "cache" / "huggingface"
    assert Path(env["HF_HOME"]) == home
    assert Path(env["HF_TOKEN_PATH"]) == home / "token"
    for var in ("HF_HUB_CACHE", "HF_XET_CACHE", "HUGGINGFACE_HUB_CACHE"):
        assert var not in env
    assert config.hf_cache_user_placement(env) == {}
    assert config.contained_hf_env(user, portable=False) == user


def test_a_users_command_does_not_inherit_the_portable_login_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_gpu_cache_env(config.contained_hf_env({}, portable=True),
                                         portable=True)
    cleaned = config.user_command_env(env)
    for var in ("HF_HOME", "HF_TOKEN_PATH", "XDG_CACHE_HOME", "TORCH_HOME"):
        assert var not in cleaned, var
    assert not [k for k in cleaned if k.startswith("LOCALM_PINNED_")]


def test_the_non_portable_login_default_survives_the_xdg_pin(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_hf_env({}, portable=False)
    env = config.contained_gpu_cache_env(env, portable=False)
    assert "HF_TOKEN_PATH" not in config.user_command_env(env)


# --------------------------------------------------------------------------- #
#  R5: the updater's dependency step uses the install's own store              #
# --------------------------------------------------------------------------- #

def test_a_portable_update_uses_its_own_store_over_inherited_values(monkeypatch, tmp_path):
    target = tmp_path / "install"
    _write_record(target, runtime_contained=True, cache_dir=str(target / ".cache"),
                  python_dir=str(target / ".python"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "user-cache"))
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(tmp_path / "user-python"))
    monkeypatch.setenv("UV_INDEX_URL", "http://evil.invalid/simple")
    env = updater._deps_env(target)
    assert env["UV_CACHE_DIR"] == str(target / ".cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(target / ".python")
    assert env["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert "UV_INDEX_URL" not in env and env["UV_NO_CONFIG"] == "1"


def test_a_moved_portable_install_uses_the_store_in_its_new_folder(tmp_path):
    target = tmp_path / "moved"
    _write_record(target, runtime_contained=True, cache_dir=str(tmp_path / "old" / ".cache"),
                  python_dir=str(tmp_path / "old" / ".python"))
    env = updater._deps_env(target)
    assert env["UV_CACHE_DIR"] == str(target / ".cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(target / ".python")


def test_a_shared_update_uses_the_recorded_store(monkeypatch, tmp_path):
    store = tmp_path / "store"
    target = tmp_path / "install"
    _write_record(target, runtime_contained=False, cache_dir=str(store / "cache"),
                  python_dir=str(store / "python"))
    monkeypatch.setenv("UV_CACHE_DIR", str(tmp_path / "user-cache"))
    env = updater._deps_env(target)
    assert env["UV_CACHE_DIR"] == str(store / "cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(store / "python")


# --------------------------------------------------------------------------- #
#  R1: find_uv takes only the uv localm brought                                #
# --------------------------------------------------------------------------- #

def _fake_uv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#!/bin/sh\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def poisoned(monkeypatch, tmp_path):
    """A uv on PATH and in every user-profile folder Astral's installer uses."""
    on_path = _fake_uv(tmp_path / "pathdir" / uv_tool.UV_NAME)
    fake_home = tmp_path / "profile"
    for rel in ((".local", "bin"), (".cargo", "bin")):
        _fake_uv(fake_home.joinpath(*rel) / uv_tool.UV_NAME)
    monkeypatch.setenv("PATH", str(on_path.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "lad"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "lad"))
    return on_path


def test_a_uv_on_path_or_in_the_profile_is_never_used(poisoned, tmp_path):
    root = tmp_path / "install"
    root.mkdir()
    assert uv_tool.find_uv(root) is None
    assert str(poisoned.parent) not in uv_tool.describe_search(root)


def test_the_contained_uv_is_found(poisoned, tmp_path):
    root = tmp_path / "install"
    own = _fake_uv(root / ".uv" / uv_tool.UV_NAME)
    assert uv_tool.find_uv(root) == str(own)


def test_the_shared_store_uv_the_record_names_is_found(poisoned, tmp_path):
    root = tmp_path / "install"
    store_uv = _fake_uv(uv_tool.shared_store_dir() / "uv" / uv_tool.UV_NAME)
    _write_record(root, uv_exe=str(store_uv))
    assert uv_tool.find_uv(root) == str(store_uv)


def test_a_recorded_uv_outside_the_store_is_not_used(poisoned, tmp_path):
    root = tmp_path / "install"
    _write_record(root, uv_exe=str(poisoned))
    assert uv_tool.find_uv(root) is None


# --------------------------------------------------------------------------- #
#  icacls by full path, without a window                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(sys.platform != "win32", reason="icacls is a Windows program")
def test_icacls_runs_by_full_path_without_a_window(monkeypatch, tmp_path):
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    target = tmp_path / "f.txt"
    target.write_text("x", encoding="utf-8")
    assert config.restrict_file_perms(target)
    (argv, kw), = calls
    assert os.path.isabs(argv[0]) and Path(argv[0]).name.lower() == "icacls.exe"
    assert Path(argv[0]).parent.name.lower() == "system32"
    assert kw.get("creationflags", 0) & subprocess.CREATE_NO_WINDOW


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows system folder")
def test_a_bare_name_on_path_never_stands_in_for_a_system_program(monkeypatch, tmp_path):
    (tmp_path / "icacls.exe").write_bytes(b"MZ")
    monkeypatch.setenv("PATH", str(tmp_path))
    from localm import sysprog
    found = sysprog.find_system("icacls")
    assert found and _norm(Path(found).parent) != _norm(tmp_path)
    assert sysprog.find_system("localm-no-such-program") is None


# --------------------------------------------------------------------------- #
#  R7: the app window profile on macOS, the launcher settings fallback         #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("contained, private", [(True, True), (False, False)])
def test_the_macos_window_keeps_no_profile_outside_a_portable_install(
        monkeypatch, contained, private):
    from localm import appface
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(config, "portable_containment", lambda: contained)
    monkeypatch.setattr(appface, "icon_path", lambda: None)
    kwargs = appface._window_start_kwargs()
    assert kwargs["private_mode"] is private
    assert "storage_path" not in kwargs


def test_the_launcher_settings_fallback_stays_in_the_install():
    import ast
    src = (config.install_root() / "launcher.pyw").read_text(encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_settings_file")
    names = {ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    assert "Path.home" not in names
    assert "REPO_DIR" in ast.unparse(fn)


@pytest.mark.parametrize("contained", [True, False])
def test_a_setup_install_without_its_uv_installs_the_pinned_one_where_it_belongs(
        poisoned, tmp_path, monkeypatch, contained):
    root = tmp_path / "install"
    _write_record(root, runtime_contained=contained)
    got = []

    def fake_bootstrap(r, *, emit=None, dest=None):
        got.append(dest)
        return _fake_uv(Path(dest) / uv_tool.UV_NAME)

    monkeypatch.setattr(uv_tool, "bootstrap_uv", fake_bootstrap)
    exe = uv_tool.uv_for_installs(root)
    want = (root / ".uv") if contained else (uv_tool.shared_store_dir() / "uv")
    assert got == [want] and exe == str(want / uv_tool.UV_NAME)
    assert uv_tool.find_uv(root) == exe
    assert str(poisoned) != exe


def test_a_source_checkout_never_installs_uv_by_itself(poisoned, tmp_path, monkeypatch):
    monkeypatch.setattr(uv_tool, "bootstrap_uv",
                        lambda *a, **k: pytest.fail("bootstrap in a source checkout"))
    assert uv_tool.uv_for_installs(tmp_path / "checkout") is None
