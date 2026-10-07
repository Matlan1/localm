# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hugging Face caches stay inside the data folder.

huggingface_hub and its xet layer write logs, staging, hub refs and stamps under
HF_HOME, which defaults to the user profile. config.contain_hf_cache() pins it
inside the data dir when localm.config is imported.
"""
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import config
from localm.model_manager import pull

_HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_TOKEN_PATH",
            "HF_TOKEN", "XDG_CACHE_HOME", config._HF_PIN_MARKER)


def _clean_hf_env(monkeypatch):
    for var in _HF_VARS:
        monkeypatch.setenv(var, "x")
        monkeypatch.delenv(var)


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(str(p)))


def test_hf_home_is_pinned_inside_the_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_hf_env({"XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert Path(env["HF_HOME"]) == tmp_path / "data" / "cache" / "huggingface"
    assert Path(env["HF_HOME"]) == config.hf_home_dir()
    assert env[config._HF_PIN_MARKER] == env["HF_HOME"]


def test_the_saved_login_stays_where_huggingface_hub_keeps_it(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_hf_env({"XDG_CACHE_HOME": str(tmp_path / "xdg")})
    assert _norm(env["HF_TOKEN_PATH"]) == _norm(tmp_path / "xdg" / "huggingface" / "token")


def test_the_login_default_follows_the_home_directory_without_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_hf_env({})
    expected = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "token"
    assert _norm(env["HF_TOKEN_PATH"]) == _norm(expected)


def test_a_user_set_hf_home_is_left_exactly_as_set(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    base = {"HF_HOME": str(tmp_path / "mine")}
    env = config.contained_hf_env(base)
    assert env == base
    assert config.hf_cache_user_placement(base) == {"HF_HOME": str(tmp_path / "mine")}


def test_explicit_hub_and_xet_caches_and_token_path_are_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    base = {"HF_HUB_CACHE": str(tmp_path / "hub"), "HF_XET_CACHE": str(tmp_path / "xet"),
            "HF_TOKEN_PATH": str(tmp_path / "tok")}
    env = config.contained_hf_env(base)
    assert env["HF_HUB_CACHE"] == base["HF_HUB_CACHE"]
    assert env["HF_XET_CACHE"] == base["HF_XET_CACHE"]
    assert env["HF_TOKEN_PATH"] == base["HF_TOKEN_PATH"]
    assert Path(env["HF_HOME"]) == config.hf_home_dir()
    assert config.hf_cache_user_placement(env) == {
        "HF_HUB_CACHE": base["HF_HUB_CACHE"], "HF_XET_CACHE": base["HF_XET_CACHE"]}


def test_localms_own_pin_is_not_reported_as_a_user_choice(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    env = config.contained_hf_env({})
    assert config.hf_cache_user_placement(env) == {}


def test_an_inherited_pin_is_repinned_for_another_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "first"))
    inherited = config.contained_hf_env({})
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "second"))
    env = config.contained_hf_env(inherited)
    assert Path(env["HF_HOME"]) == tmp_path / "second" / "cache" / "huggingface"
    assert env["HF_TOKEN_PATH"] == inherited["HF_TOKEN_PATH"]


def test_contain_hf_cache_applies_to_the_process_environment(monkeypatch, tmp_path):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    config.contain_hf_cache()
    assert Path(os.environ["HF_HOME"]) == tmp_path / "data" / "cache" / "huggingface"
    assert "HF_TOKEN_PATH" in os.environ


def test_contain_hf_cache_leaves_a_user_hf_home_alone(monkeypatch, tmp_path):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
    config.contain_hf_cache()
    assert os.environ["HF_HOME"] == str(tmp_path / "mine")
    assert "HF_TOKEN_PATH" not in os.environ
    assert config._HF_PIN_MARKER not in os.environ


def _fresh_interpreter(tmp_path: Path, code: str, extra_env=None) -> str:
    """Run *code* in a new interpreter whose home, XDG cache and data dir are
    all under tmp_path and which has no Hugging Face variables set."""
    env = {k: v for k, v in os.environ.items() if k not in _HF_VARS}
    env.update({
        "HOME": str(tmp_path / "home"),
        "USERPROFILE": str(tmp_path / "home"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg"),
        "LOCALM_HOME": str(tmp_path / "data"),
        "PYTHONPATH": str(Path(config.__file__).resolve().parents[1]),
    })
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_huggingface_hub_resolves_every_cache_inside_the_data_dir(tmp_path):
    out = _fresh_interpreter(tmp_path, (
        "import localm.config\n"
        "from huggingface_hub import constants as c\n"
        "print(c.HF_HOME); print(c.HF_HUB_CACHE); print(c.HF_XET_CACHE)\n"))
    hf_home, hub, xet = (Path(line) for line in out.splitlines())
    data_cache = tmp_path / "data" / "cache" / "huggingface"
    assert _norm(hf_home) == _norm(data_cache)
    assert _norm(hub).startswith(_norm(data_cache))
    assert _norm(xet).startswith(_norm(data_cache))


def test_a_login_saved_outside_localm_still_authenticates(tmp_path):
    token_file = tmp_path / "xdg" / "huggingface" / "token"
    token_file.parent.mkdir(parents=True)
    token_file.write_text("hf_dummy_login", encoding="utf-8")
    out = _fresh_interpreter(tmp_path, (
        "import localm.config\n"
        "import huggingface_hub\n"
        "print(huggingface_hub.get_token())\n"))
    assert out == "hf_dummy_login"


def test_a_child_process_inherits_the_pin(tmp_path):
    out = _fresh_interpreter(tmp_path, (
        "import localm.config, subprocess, sys\n"
        "r = subprocess.run([sys.executable, '-c', "
        "'import os; print(os.environ[\"HF_HOME\"])'], capture_output=True, text=True)\n"
        "print(r.stdout.strip())\n"))
    assert _norm(out) == _norm(tmp_path / "data" / "cache" / "huggingface")


def test_a_user_hf_home_survives_import_in_a_fresh_interpreter(tmp_path):
    mine = tmp_path / "mine"
    out = _fresh_interpreter(tmp_path, (
        "import localm.config\n"
        "from huggingface_hub import constants as c\n"
        "print(c.HF_HOME)\n"), extra_env={"HF_HOME": str(mine)})
    assert _norm(out) == _norm(mine)


def test_pull_notes_a_user_placed_cache_once(monkeypatch, tmp_path):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    log = MagicMock()
    monkeypatch.setattr(pull, "logger", log)
    monkeypatch.setattr(pull, "_HF_PLACEMENT_NOTED", False)
    pull._note_user_hf_placement()
    pull._note_user_hf_placement()
    assert log.info.call_count == 1
    assert "HF_HUB_CACHE" in log.info.call_args.args[-1]


def test_the_pull_entry_wiring_notes_a_user_placed_cache(monkeypatch, tmp_path):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "xet"))
    log = MagicMock()
    monkeypatch.setattr(pull, "logger", log)
    monkeypatch.setattr(pull, "_HF_PLACEMENT_NOTED", False)
    pull._ensure_hf_resumable_download()
    assert log.info.call_count == 1
    assert "HF_XET_CACHE" in log.info.call_args.args[-1]


def test_pull_stays_quiet_when_the_contained_home_is_in_effect(monkeypatch, tmp_path):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    config.contain_hf_cache()
    log = MagicMock()
    monkeypatch.setattr(pull, "logger", log)
    monkeypatch.setattr(pull, "_HF_PLACEMENT_NOTED", False)
    pull._note_user_hf_placement()
    log.info.assert_not_called()


@pytest.mark.parametrize("name", ["cache"])
def test_the_data_folder_entry_that_holds_the_hf_home_is_listed(name):
    from localm import install_manifest as im
    assert im.is_data_entry(name)
    assert config.hf_home_dir().parent.name == name


def test_a_relative_data_dir_is_pinned_as_an_absolute_hf_home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LOCALM_HOME", "reldata")
    env = config.contained_hf_env({})
    assert os.path.isabs(env["HF_HOME"])
    assert _norm(env["HF_HOME"]) == _norm(tmp_path / "reldata" / "cache" / "huggingface")


def test_legacy_and_asset_cache_variables_count_as_user_placement(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    base = {"HUGGINGFACE_HUB_CACHE": str(tmp_path / "a"), "HF_ASSETS_CACHE": str(tmp_path / "b")}
    assert config.hf_cache_user_placement(base) == base


def test_containment_that_cannot_take_effect_is_reported(tmp_path):
    code = "import huggingface_hub.constants; import localm.config"
    env = {k: v for k, v in os.environ.items() if k not in _HF_VARS}
    env.update({"HOME": str(tmp_path / "home"), "USERPROFILE": str(tmp_path / "home"),
                "XDG_CACHE_HOME": str(tmp_path / "xdg"), "LOCALM_HOME": str(tmp_path / "data"),
                "PYTHONPATH": str(Path(config.__file__).resolve().parents[1])})
    proc = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "huggingface_hub was imported before localm" in proc.stderr
    quiet = _fresh_interpreter(tmp_path, "import localm.config; print('ok')")
    assert quiet == "ok"


def _loaded_hub(monkeypatch, hub_home):
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants",
                        SimpleNamespace(HF_HOME=hub_home))


def test_a_hub_loaded_with_another_home_is_reported_in_process(monkeypatch, tmp_path, capsys):
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    elsewhere = str(tmp_path / "profile" / "huggingface")
    _loaded_hub(monkeypatch, elsewhere)
    config.contain_hf_cache()
    pinned = os.environ["HF_HOME"]
    assert capsys.readouterr().err == (
        f"[localm] WARNING: huggingface_hub was imported before localm, so its caches "
        f"stay at {elsewhere} instead of the data folder ({pinned}).\n")


@pytest.mark.parametrize("case", ["same-home", "not-loaded", "user-home", "no-home-attr"])
def test_containment_in_effect_or_left_to_the_user_is_not_reported(
        monkeypatch, tmp_path, capsys, case):
    from types import SimpleNamespace
    _clean_hf_env(monkeypatch)
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "data"))
    pinned = str(config.contained_hf_env({})["HF_HOME"])
    if case == "same-home":
        _loaded_hub(monkeypatch, pinned + os.sep)
    elif case == "not-loaded":
        monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    elif case == "user-home":
        monkeypatch.setenv("HF_HOME", str(tmp_path / "mine"))
        _loaded_hub(monkeypatch, str(tmp_path / "elsewhere"))
    else:
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", SimpleNamespace())
    config.contain_hf_cache()
    assert capsys.readouterr().err == ""
