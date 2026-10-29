# SPDX-License-Identifier: AGPL-3.0-or-later
"""The ComfyUI process localm spawns gets MIOpen's FAST find mode unless the user
set a find mode, driven through ensure_comfy's real launch path with a faked Popen."""

import os

from localm.media import comfy_client
from tests.test_comfy_func_shim import _capture_spawn_env


def test_child_env_sets_fast_miopen_find_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    env = _capture_spawn_env(monkeypatch, tmp_path)["env"]
    assert env is not None
    assert env["MIOPEN_FIND_MODE"] == comfy_client.COMFY_MIOPEN_FIND_MODE == "FAST"


def test_child_env_keeps_a_user_set_miopen_find_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("MIOPEN_FIND_MODE", "NORMAL")
    env = _capture_spawn_env(monkeypatch, tmp_path)["env"]
    assert env["MIOPEN_FIND_MODE"] == "NORMAL"


def test_child_env_does_not_change_localm_own_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    env = _capture_spawn_env(monkeypatch, tmp_path)["env"]
    assert env["MIOPEN_FIND_MODE"] == "FAST"
    assert "MIOPEN_FIND_MODE" not in os.environ


def test_child_env_keeps_the_rest_of_the_inherited_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    monkeypatch.setenv("LOCALM_TEST_MARKER_VAR", "kept")
    env = _capture_spawn_env(monkeypatch, tmp_path)["env"]
    assert env["LOCALM_TEST_MARKER_VAR"] == "kept"
