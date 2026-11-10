# SPDX-License-Identifier: AGPL-3.0-or-later
"""A config.json served from its .bak is not current, and the shared store lock
decides staleness by whether its holder is still running.

Two real interpreters hold the lock: one is killed (a dead holder, reclaimed at
once however fresh the file is), one is left running (a live holder that stalls
past the stale age keeps the lock)."""

import json
import logging
import os
import subprocess
import sys
import time

import pytest

import localm.config as cfg
from localm import instances


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / ".localm"
    h.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", h)
    monkeypatch.setattr(cfg, "MODELS_DIR", h / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", h / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", h / "registry.json")
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    return h


def _config_with_bak(net_mode_now="off", net_mode_before="allow"):
    cfg.CONFIG_FILE.with_name("config.json.bak").write_text(
        json.dumps({"net_mode": net_mode_before}), encoding="utf-8")
    cfg.CONFIG_FILE.write_text(json.dumps({"net_mode": net_mode_now}),
                               encoding="utf-8")


def _corrupt_config():
    cfg.CONFIG_FILE.write_text("{ not json,", encoding="utf-8")


# --------------------------------------------------------------------------- #
#  .bak is not current                                                         #
# --------------------------------------------------------------------------- #

def test_corrupt_config_with_a_looser_bak_resolves_net_mode_off(home):
    from localm import netpolicy
    _config_with_bak(net_mode_now="off", net_mode_before="allow")
    assert netpolicy.network_mode() == "off"
    _corrupt_config()
    assert netpolicy.network_mode() == "off"


def test_corrupt_config_without_a_bak_resolves_net_mode_off(home):
    from localm import netpolicy
    cfg.CONFIG_FILE.write_text(json.dumps({"net_mode": "off"}), encoding="utf-8")
    _corrupt_config()
    assert netpolicy.network_mode() == "off"


def test_check_url_refuses_when_config_is_served_from_bak(home):
    from localm import netpolicy
    _config_with_bak(net_mode_now="off", net_mode_before="allow")
    _corrupt_config()
    with pytest.raises(netpolicy.NetworkPolicyError):
        netpolicy.check_url("https://example.com/")


def test_read_state_names_the_bak_case(home):
    _config_with_bak()
    _corrupt_config()
    value, state = cfg._read_json_state(cfg.CONFIG_FILE, {})
    assert state == cfg.READ_FROM_BACKUP
    assert value == {"net_mode": "allow"}
    assert cfg._read_json_checked(cfg.CONFIG_FILE, {})[1] is False


def test_read_state_names_current_absent_and_unreadable(home):
    assert cfg._read_json_state(cfg.CONFIG_FILE, {})[1] == cfg.READ_ABSENT
    cfg.CONFIG_FILE.write_text("{}", encoding="utf-8")
    assert cfg._read_json_state(cfg.CONFIG_FILE, {})[1] == cfg.READ_CURRENT
    _corrupt_config()
    assert cfg._read_json_state(cfg.CONFIG_FILE, {})[1] == cfg.READ_UNREADABLE


def test_a_missing_primary_with_a_bak_is_still_read_ok(home):
    cfg.CONFIG_FILE.with_name("config.json.bak").write_text(
        json.dumps({"n_ctx": 1}), encoding="utf-8")
    value, state = cfg._read_json_state(cfg.CONFIG_FILE, {})
    assert state == cfg.READ_CURRENT and value == {"n_ctx": 1}


def test_require_auth_fails_closed_when_config_is_served_from_bak(home):
    from localm import auth
    cfg.CONFIG_FILE.write_text(json.dumps({"require_auth": False}), encoding="utf-8")
    cfg.CONFIG_FILE.with_name("config.json.bak").write_text(
        json.dumps({"require_auth": False}), encoding="utf-8")
    assert auth.require_auth_enabled() is False
    _corrupt_config()
    assert auth.require_auth_enabled() is True


def test_effective_mode_is_privacy_when_config_is_served_from_bak(home):
    from localm import audit
    cfg.CONFIG_FILE.write_text(json.dumps({"mode": "full"}), encoding="utf-8")
    cfg.CONFIG_FILE.with_name("config.json.bak").write_text(
        json.dumps({"mode": "full"}), encoding="utf-8")
    assert audit.effective_mode("chat") == audit.SessionMode.FULL
    _corrupt_config()
    assert audit.effective_mode("chat") == audit.SessionMode.PRIVACY


def test_update_config_refuses_when_only_the_bak_reads_and_leaves_both_files(home):
    _config_with_bak()
    _corrupt_config()
    before = cfg.CONFIG_FILE.read_bytes()
    bak_before = cfg.CONFIG_FILE.with_name("config.json.bak").read_bytes()
    with pytest.raises(cfg.ConfigUnreadable):
        cfg.update_config(lambda c: c.__setitem__("temperature", 0.5))
    assert cfg.CONFIG_FILE.read_bytes() == before
    assert cfg.CONFIG_FILE.with_name("config.json.bak").read_bytes() == bak_before


def test_an_unreadable_primary_is_reported_once_through_the_logger(
        home, caplog, capsys):
    _config_with_bak()
    _corrupt_config()
    cfg._warned_unreadable.clear()
    with caplog.at_level(logging.WARNING, logger="localm"):
        for _ in range(5):
            cfg._read_json_checked(cfg.CONFIG_FILE, {})
    assert capsys.readouterr().err == ""
    reports = [r for r in caplog.records if "config.json" in r.getMessage()]
    assert len(reports) == 1


# --------------------------------------------------------------------------- #
#  Lock staleness is liveness                                                  #
# --------------------------------------------------------------------------- #

pytestmark_identity = pytest.mark.skipif(
    instances.process_start_identity(os.getpid()) is None,
    reason="this platform cannot read a process start identity")

_HOLDER = (
    "import sys, time\n"
    "from pathlib import Path\n"
    "import localm.config as c\n"
    "target = Path(sys.argv[1])\n"
    "with c._cross_process_lock(target):\n"
    "    print('held', flush=True)\n"
    "    time.sleep(120)\n"
)


def _spawn_holder(target):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["LOCALM_HOME"] = str(target.parent)
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(target)],
                            env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held", proc.stderr.read()
    return proc


def _lockpath(target):
    return target.with_name(target.name + ".lock")


@pytestmark_identity
def test_a_killed_holder_is_reclaimed_at_once_whatever_the_stale_age(
        home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    target = home / "store.json"
    proc = _spawn_holder(target)
    try:
        proc.kill()
        proc.wait(timeout=30)
        assert _lockpath(target).exists()
        started = time.monotonic()
        with cfg._cross_process_lock(target, timeout=5.0) as lock:
            lock.assert_held()
        assert time.monotonic() - started < 3.0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)


@pytestmark_identity
def test_a_live_holder_past_the_stale_age_keeps_the_lock(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    target = home / "store.json"
    proc = _spawn_holder(target)
    try:
        lockpath = _lockpath(target)
        held_bytes = lockpath.read_bytes()
        old = time.time() - 600
        os.utime(lockpath, (old, old))
        with pytest.raises(cfg.ConfigLockTimeout):
            with cfg._cross_process_lock(target, timeout=0.5):
                pytest.fail("acquired a lock a live process holds")
        assert lockpath.read_bytes() == held_bytes
    finally:
        proc.kill()
        proc.wait(timeout=30)


def test_a_token_without_a_start_identity_still_goes_stale_by_age(home, monkeypatch):
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 0.05)
    target = home / "store.json"
    lockpath = _lockpath(target)
    lockpath.write_bytes(b"4242:deadbeef")
    old = time.time() - 10
    os.utime(lockpath, (old, old))
    with cfg._cross_process_lock(target, timeout=2.0):
        pass


@pytestmark_identity
def test_this_processs_leaked_token_under_a_reused_pid_is_dead(home):
    target = home / "store.json"
    leaked = cfg._lock_token(os.getpid(), "abandoned")
    assert cfg._lock_holder_state(leaked) == "dead"
    _lockpath(target).write_bytes(leaked)
    with cfg._cross_process_lock(target, timeout=2.0):
        pass


@pytestmark_identity
def test_a_token_held_by_this_process_is_alive(home):
    target = home / "store.json"
    with cfg._cross_process_lock(target):
        assert cfg._lock_holder_state(_lockpath(target).read_bytes()) == "alive"


def test_assert_held_raises_when_the_lock_was_taken_over(home):
    target = home / "store.json"
    with cfg._cross_process_lock(target) as lock:
        lock.assert_held()
        _lockpath(target).write_bytes(b"777:feedface")
        with pytest.raises(cfg.ConfigLockLost):
            lock.assert_held()
        with pytest.raises(cfg.ConfigLockLost):
            cfg.assert_store_lock_held(target)


def test_assert_held_raises_when_the_lock_file_was_removed(home):
    target = home / "store.json"
    with cfg._cross_process_lock(target) as lock:
        _lockpath(target).unlink()
        with pytest.raises(cfg.ConfigLockLost):
            lock.assert_held()


def test_update_config_writes_nothing_when_its_lock_was_taken_over(home):
    cfg.CONFIG_FILE.write_text(json.dumps({"n_ctx": 111}), encoding="utf-8")
    lockpath = _lockpath(cfg.CONFIG_FILE)

    def mutator(c):
        lockpath.write_bytes(b"777:feedface")
        c["temperature"] = 0.9

    with pytest.raises(cfg.ConfigLockLost):
        cfg.update_config(mutator)
    assert json.loads(cfg.CONFIG_FILE.read_text(encoding="utf-8")) == {"n_ctx": 111}


def test_assert_store_lock_held_is_a_no_op_without_a_lock(home):
    cfg.assert_store_lock_held(home / "store.json")


# --------------------------------------------------------------------------- #
#  The stores share one contract (storekit)                                    #
# --------------------------------------------------------------------------- #

def test_set_api_key_waits_on_the_store_lock(home, monkeypatch):
    from localm import auth
    monkeypatch.setattr(cfg, "_CROSS_LOCK_TIMEOUT", 0.3)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(auth, "key_file", lambda: home / "auth.key")
    lockpath = _lockpath(home / "auth.key")
    lockpath.write_bytes(b"12345:foreign")
    monkeypatch.setattr(cfg, "_CROSS_LOCK_STALE_AGE", 3600.0)
    try:
        with pytest.raises(cfg.ConfigLockTimeout):
            auth.set_api_key("abc-123")
    finally:
        lockpath.unlink()
    assert not (home / "auth.key").exists()


def test_set_api_key_writes_under_the_lock_and_releases_it(home, monkeypatch):
    from localm import auth
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.setattr(auth, "key_file", lambda: home / "auth.key")
    monkeypatch.setattr(auth, "owner_kdf_file", lambda: home / "auth.kdf.json")
    auth.set_api_key("abc-123")
    assert (home / "auth.key").read_text(encoding="utf-8").strip() == "abc-123"
    assert not _lockpath(home / "auth.key").exists()


def test_storekit_write_private_uses_a_temp_name_unique_to_the_writer(home, monkeypatch):
    from localm import storekit
    seen = []
    real = cfg.os.replace

    def spy(src, dst):
        seen.append(os.path.basename(str(src)))
        return real(src, dst)

    monkeypatch.setattr(cfg.os, "replace", spy)
    storekit.write_private(home / "x.json", "{}", retrying=False)
    assert seen and str(os.getpid()) in seen[0] and seen[0].endswith(".tmp")
    assert seen[0] != "x.json.tmp"


def test_storekit_write_private_refuses_after_the_lock_was_lost(home):
    from localm import storekit
    target = home / "x.json"
    target.write_text("old", encoding="utf-8")
    with storekit.store_lock(target):
        _lockpath(target).write_bytes(b"777:feedface")
        with pytest.raises(cfg.ConfigLockLost):
            storekit.write_private(target, "new")
    assert target.read_text(encoding="utf-8") == "old"


def test_storekit_read_for_write_refuses_a_bak_served_store(home):
    from localm import storekit
    target = home / "x.json"
    target.write_text("[1, 2]", encoding="utf-8")
    target.with_name("x.json.bak").write_text("[1]", encoding="utf-8")
    assert storekit.read_for_write(target, [], list, "refused") == [1, 2]
    target.write_text("[", encoding="utf-8")
    with pytest.raises(cfg.ConfigUnreadable, match="refused"):
        storekit.read_for_write(target, [], list, "refused")


def test_keystore_credentials_and_sessions_take_the_lock_through_storekit():
    import inspect

    from localm import auth, model_source_credentials, sessions
    for mod in (auth, model_source_credentials, sessions):
        src = inspect.getsource(mod)
        assert "storekit.store_lock" in src, mod.__name__
