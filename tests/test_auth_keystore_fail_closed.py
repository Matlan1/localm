# SPDX-License-Identifier: AGPL-3.0-or-later
"""_load_keystore() used to fail OPEN on any read error (return []), and
create_key()/revoke_key() both read-modify-write on that result - so a single
reported-unreadable read during either call silently replaced the whole
keystore with just that one change, while the caller reported success.

These tests pin the fix: a read that comes back unreadable must REFUSE the
write (config.ConfigUnreadable), never silently persist a near-empty list
over the existing keys. See test_auth_keystore_cross_process.py for the
cross-process contention half of the same class of bug, and
test_config_cross_process_lock.py for the sibling coverage this mirrors for
config.json/registry.json."""

from pathlib import Path

import pytest


@pytest.fixture
def auth(tmp_path, monkeypatch):
    """localm.auth with a throwaway data dir and a clean auth environment.
    Mirrors test_auth.py's own `auth` fixture."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    import localm.auth as a
    return a


def _fail_next_read_once(monkeypatch, target_name: str) -> None:
    """Make the NEXT read-mode open() of a file named *target_name* raise a
    plain (non-Permission) OSError exactly once, then behave normally for
    every call after - including a retry of the very same read.

    Patches BOTH builtins.open and io.open: config._read_json_checked's own
    bare `open(candidate, ...)` call resolves via builtins, while
    pathlib.Path.read_text() (what the pre-fix _load_keystore used directly)
    calls io.open(...) as a module attribute, which a builtins-only patch
    does not reach. A plain OSError (not PermissionError) is deliberate: it
    lands in config._read_json_checked's non-transient except branch on
    EITHER platform, so the simulated failure is not silently absorbed by
    that function's own retry-on-transient-PermissionError loop."""
    import builtins
    import io as io_mod
    real_open = builtins.open
    state = {"armed": True}

    def flaky(file, mode="r", *args, **kwargs):
        name = getattr(file, "name", None)
        if name is None:
            name = str(file).replace("\\", "/").rsplit("/", 1)[-1]
        if (state["armed"] and name == target_name
                and "r" in mode and "w" not in mode and "a" not in mode):
            state["armed"] = False
            raise OSError(f"simulated persistent read failure for {target_name}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", flaky)
    monkeypatch.setattr(io_mod, "open", flaky)


def test_create_key_refuses_rather_than_wipes_on_unreadable_keystore(auth, monkeypatch):
    alpha = auth.create_key("alpha", [])
    beta = auth.create_key("beta", [])
    assert {r["name"] for r in auth.list_keys()} == {"alpha", "beta"}

    import localm.config as cfg
    _fail_next_read_once(monkeypatch, Path(auth.keystore_file()).name)

    exc = None
    try:
        auth.create_key("gamma", [])
    except Exception as e:  # noqa: BLE001 - captured for an isinstance check below
        exc = e

    # DATA FIRST (diff-review-discipline item 24): alpha and beta must survive
    # a reported-unreadable read, whether create_key raised or - on the
    # unfixed code - silently "succeeded" by replacing the store.
    names_on_disk = {r["name"] for r in auth.list_keys()}
    assert names_on_disk >= {"alpha", "beta"}, (
        f"a reported-unreadable read destroyed the existing keystore instead "
        f"of refusing the write; keystore now holds {names_on_disk}")

    # And the failure must be LOUD, never a false "New key created".
    assert isinstance(exc, cfg.ConfigUnreadable), (
        f"create_key() must raise ConfigUnreadable on an unreadable keystore "
        f"read rather than silently succeed; got {exc!r}")

    # One-shot: a normal call right after succeeds, and all three persist.
    gamma = auth.create_key("gamma", [])
    assert {r["name"] for r in auth.list_keys()} == {"alpha", "beta", "gamma"}
    assert len({alpha["id"], beta["id"], gamma["id"]}) == 3


def test_revoke_key_refuses_rather_than_wipes_on_unreadable_keystore(auth, monkeypatch):
    alpha = auth.create_key("alpha", [])
    auth.create_key("beta", [])

    import localm.config as cfg
    _fail_next_read_once(monkeypatch, Path(auth.keystore_file()).name)

    exc = None
    try:
        auth.revoke_key(alpha["id"])
    except Exception as e:  # noqa: BLE001
        exc = e

    names_on_disk = {r["name"] for r in auth.list_keys()}
    assert names_on_disk >= {"alpha", "beta"}, (
        f"a reported-unreadable read destroyed the existing keystore during "
        f"revoke_key() instead of refusing; keystore now holds {names_on_disk}")
    assert isinstance(exc, cfg.ConfigUnreadable), (
        f"revoke_key() must raise ConfigUnreadable on an unreadable keystore "
        f"read rather than silently report success; got {exc!r}")

    assert auth.revoke_key(alpha["id"]) is True
    assert {r["name"] for r in auth.list_keys()} == {"beta"}
