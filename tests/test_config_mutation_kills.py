# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pins the exact behaviour of two localm.config helpers whose outcomes the other
config tests only reach indirectly: the environment handed to a command the
user asked to run (which pins to drop, which user values to keep) and the
stale-lock reclaim's age test."""

import os
import types

import pytest

import localm.config as cfg

_HF_PIN = "LOCALM_PINNED_HF_HOME"


def _token_path(env):
    return cfg._default_hf_token_path(env)


class TestUserCommandEnvHuggingFace:
    def test_a_pinned_home_and_the_default_login_location_are_dropped_with_the_marker(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "KEEP": "1"}
        env["HF_TOKEN_PATH"] = _token_path(env)
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_a_home_the_user_chose_is_kept_and_the_marker_is_dropped(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/mine", "HF_TOKEN_PATH": "/t"}
        assert cfg.user_command_env(env) == {"HF_HOME": "/mine", "HF_TOKEN_PATH": "/t"}

    def test_a_login_location_the_user_chose_is_kept_when_the_home_is_dropped(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "HF_TOKEN_PATH": "/my/token"}
        assert cfg.user_command_env(env) == {"HF_TOKEN_PATH": "/my/token"}

    def test_a_home_without_a_marker_is_the_users_own_and_is_kept(self):
        env = {"HF_HOME": "/mine", "HF_TOKEN_PATH": _token_path({})}
        assert cfg.user_command_env(env) == env

    def test_a_marker_without_a_home_leaves_nothing_behind(self):
        assert cfg.user_command_env({_HF_PIN: "/pin", "KEEP": "1"}) == {"KEEP": "1"}

    def test_an_environment_with_none_of_the_variables_is_returned_unchanged(self):
        env = {"KEEP": "1"}
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_the_login_location_follows_the_xdg_cache_of_the_environment(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin", "XDG_CACHE_HOME": "/xdg"}
        env["HF_TOKEN_PATH"] = _token_path(env)
        assert env["HF_TOKEN_PATH"] == os.path.join("/xdg", "huggingface", "token")
        assert cfg.user_command_env(env) == {"XDG_CACHE_HOME": "/xdg"}

    def test_the_callers_dict_is_not_modified(self):
        env = {_HF_PIN: "/pin", "HF_HOME": "/pin"}
        cfg.user_command_env(env)
        assert env == {_HF_PIN: "/pin", "HF_HOME": "/pin"}


@pytest.mark.parametrize("var", [v for v, _ in cfg._GPU_CACHE_PINS])
class TestUserCommandEnvGpuCaches:
    def test_a_pinned_cache_variable_is_dropped_with_its_marker(self, var):
        marker = cfg._gpu_pin_marker(var)
        env = {var: "/pin", marker: "/pin", "KEEP": "1"}
        assert cfg.user_command_env(env) == {"KEEP": "1"}

    def test_a_value_the_user_chose_is_kept_and_the_marker_is_dropped(self, var):
        marker = cfg._gpu_pin_marker(var)
        env = {var: "/mine", marker: "/pin"}
        assert cfg.user_command_env(env) == {var: "/mine"}

    def test_a_variable_without_a_marker_is_the_users_own_and_is_kept(self, var):
        assert cfg.user_command_env({var: "/mine"}) == {var: "/mine"}

    def test_a_marker_without_the_variable_is_dropped(self, var):
        assert cfg.user_command_env({cfg._gpu_pin_marker(var): "/pin"}) == {}

    def test_an_absent_variable_without_a_marker_is_left_absent(self, var):
        assert cfg.user_command_env({"KEEP": "1"}) == {"KEEP": "1"}


class TestReclaimStaleLockAge:
    def _lock(self, tmp_path, age):
        lockpath = tmp_path / "x.lock"
        lockpath.write_bytes(b"1:tok")
        stamp = lockpath.stat().st_mtime - age
        os.utime(lockpath, (stamp, stamp))
        return lockpath, b"1:tok", lockpath.stat().st_mtime

    def test_a_lock_younger_than_the_stale_age_is_left_in_place(self, tmp_path):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
        assert lockpath.exists()

    def test_a_lock_older_than_the_stale_age_is_removed(self, tmp_path):
        lockpath, token, mtime = self._lock(tmp_path, age=cfg._CROSS_LOCK_STALE_AGE + 60)
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
        assert not lockpath.exists()

    def test_a_lock_exactly_as_old_as_the_stale_age_is_not_yet_stale(
            self, tmp_path, monkeypatch):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
            time=lambda: mtime + cfg._CROSS_LOCK_STALE_AGE))
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is False
        assert lockpath.exists()

    def test_a_lock_just_past_the_stale_age_is_removed(self, tmp_path, monkeypatch):
        lockpath, token, mtime = self._lock(tmp_path, age=0)
        monkeypatch.setattr(cfg, "time", types.SimpleNamespace(
            time=lambda: mtime + cfg._CROSS_LOCK_STALE_AGE + 0.001))
        assert cfg._reclaim_stale_lock(lockpath, token, mtime) is True
        assert not lockpath.exists()
