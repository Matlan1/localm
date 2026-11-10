"""The management-event log: closed kinds, content-free lines, every session mode,
rotation, write failures, and the key and login emitters."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

import localm.config as cfg
from localm import mgmt_events

SECRET = "mgmt-events-secret-key-0123456789"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


def _events(home):
    path = home / "logs" / "management.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def test_record_writes_one_line_with_only_the_given_fields(home):
    assert mgmt_events.record("plugin_installed", home=home, plugin="demo", version_to="1.2") is True
    (ev,) = _events(home)
    assert set(ev) == {"ts", "kind", "plugin", "version_to"}
    assert ev["kind"] == "plugin_installed" and ev["plugin"] == "demo"


def test_config_saved_keeps_key_names_not_values(home):
    mgmt_events.record("config_saved", home=home, keys=["b_key", "a_key"])
    (ev,) = _events(home)
    assert ev["keys"] == ["a_key", "b_key"]


def test_unknown_kind_raises_and_writes_nothing(home):
    with pytest.raises(ValueError):
        mgmt_events.record("key_exploded", home=home)
    assert _events(home) == []


def test_every_kind_is_accepted(home):
    for kind in sorted(mgmt_events.KINDS):
        assert mgmt_events.record(kind, home=home) is True
    assert [e["kind"] for e in _events(home)] == sorted(mgmt_events.KINDS)


def test_text_fields_are_length_capped(home):
    mgmt_events.record("plugin_installed", home=home, plugin="x" * 5000)
    (ev,) = _events(home)
    assert len(ev["plugin"]) == 200


def test_rotation_keeps_one_previous_file(home, monkeypatch):
    monkeypatch.setattr(mgmt_events, "MAX_BYTES", 300)
    for _ in range(40):
        mgmt_events.record("config_saved", home=home, keys=["k"])
    logs = home / "logs"
    assert sorted(p.name for p in logs.iterdir()) == [
        "management.jsonl", "management.jsonl.1"]
    assert (logs / "management.jsonl").stat().st_size < 600
    assert _events(home)


def test_write_failure_returns_false_and_warns(home, monkeypatch, caplog):
    (home / "logs").write_text("a file where the directory should be")
    with caplog.at_level("WARNING", logger="localm"):
        assert mgmt_events.record("key_minted", home=home, principal="x") is False
    assert any("management event log" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("mode", ["privacy", "full"])
def test_written_in_every_session_mode(home, monkeypatch, mode):
    (home / "config.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
    import localm.auth as auth
    rec = auth.create_key("dev", ["chat"])
    (ev,) = _events(home)
    assert ev["kind"] == "key_minted" and ev["principal"] == rec["id"]


def test_create_key_writes_one_event_and_never_the_key(home):
    import localm.auth as auth
    rec = auth.create_key("dev", ["chat"])
    raw = (home / "logs" / "management.jsonl").read_text(encoding="utf-8")
    assert rec["key"] not in raw
    (ev,) = _events(home)
    assert ev == {"ts": ev["ts"], "kind": "key_minted", "principal": rec["id"]}


def test_revoke_key_writes_event_only_when_a_key_was_removed(home):
    import localm.auth as auth
    rec = auth.create_key("dev", ["chat"])
    assert auth.revoke_key("nonexistent") is False
    assert [e["kind"] for e in _events(home)] == ["key_minted"]
    assert auth.revoke_key(rec["id"]) is True
    evs = _events(home)
    assert [e["kind"] for e in evs] == ["key_minted", "key_revoked"]
    assert evs[1]["principal"] == rec["id"]


def test_owner_key_set_and_clear_are_recorded_without_the_key(home):
    import localm.auth as auth
    auth.set_api_key(SECRET)
    assert auth.clear_api_key() == []
    raw = (home / "logs" / "management.jsonl").read_text(encoding="utf-8")
    assert SECRET not in raw
    evs = _events(home)
    assert [(e["kind"], e["principal"]) for e in evs] == [
        ("key_minted", "owner"), ("key_revoked", "all")]


def test_clear_with_nothing_to_clear_writes_nothing(home):
    import localm.auth as auth
    assert auth.clear_api_key() == []
    assert _events(home) == []


def test_key_operation_succeeds_when_the_log_cannot_be_written(home):
    import localm.auth as auth
    (home / "logs").write_text("blocker")
    rec = auth.create_key("dev", ["chat"])
    assert rec["key"] and auth.revoke_key(rec["id"]) is True


def _client():
    from fastapi.testclient import TestClient
    from localm.inference.http_server import create_app
    engine = MagicMock()
    engine.display_name = "m"
    return TestClient(create_app(engine))


def test_failed_login_records_loopback_flag_and_not_the_presented_key(home):
    with patch.dict(os.environ, {"LOCALM_API_KEY": SECRET}):
        with _client() as c:
            bad = "wrong-key-presented-9876543210"
            assert c.post("/api/session", json={"key": bad}).status_code == 401
            assert c.post("/api/session", json={}).status_code == 401
            ok = c.post("/api/session", json={"key": SECRET})
            assert ok.status_code == 200
    raw = (home / "logs" / "management.jsonl").read_text(encoding="utf-8")
    assert bad not in raw and SECRET not in raw
    evs = [e for e in _events(home) if e["kind"] != "config_saved"]
    assert [e["kind"] for e in evs] == ["login_failed", "login_failed"]
    assert all(isinstance(e["loopback"], bool) for e in evs)


SECRET_VALUE = "super-secret-value-should-never-be-logged"


def test_update_config_records_the_changed_key_names_never_the_values(home):
    cfg.update_config(lambda c: c.update(hf_token=SECRET_VALUE, max_tokens=918273645))
    raw = (home / "logs" / "management.jsonl").read_text(encoding="utf-8")
    assert SECRET_VALUE not in raw and "918273645" not in raw
    (ev,) = _events(home)
    assert ev["kind"] == "config_saved"
    assert ev["keys"] == ["hf_token", "max_tokens"]


def test_save_config_records_only_what_changed(home):
    cfg.save_config({**cfg.load_config(), "max_tokens": 555})
    assert [e["keys"] for e in _events(home)] == [["max_tokens"]]
    cfg.save_config({**cfg.load_config(), "max_tokens": 555})
    assert len(_events(home)) == 1
    cfg.save_config({**cfg.load_config(), "max_tokens": 556, "hf_token": SECRET_VALUE})
    assert _events(home)[-1]["keys"] == ["hf_token", "max_tokens"]


def test_config_save_that_changes_nothing_writes_no_event(home):
    cfg.update_config(lambda c: None)
    assert _events(home) == []


@pytest.mark.parametrize("mode", ["privacy", "full"])
def test_config_saved_is_written_in_every_session_mode(home, mode):
    (home / "config.json").write_text(json.dumps({"mode": mode}), encoding="utf-8")
    cfg.update_config(lambda c: c.update(max_tokens=1234))
    assert [e["kind"] for e in _events(home)] == ["config_saved"]


def _plugin_manager(home):
    import textwrap

    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    store = home / "store"
    pdir = store / "p1"
    pdir.mkdir(parents=True)
    (pdir / "plugin.toml").write_text(
        '[plugin]\nname = "p1"\nscope = "p1"\nversion = "2.5.0"\nregister = "plug"\n',
        encoding="utf-8")
    (pdir / "plug.py").write_text(textwrap.dedent('''
        def register(host):
            pass

        def unregister():
            pass
    '''), encoding="utf-8")
    return PluginManager(FastAPI(), store_root=store, installed_root=home / "installed")


def _plugin_events(home):
    return [e for e in _events(home) if e["kind"].startswith("plugin_")]


def test_install_and_uninstall_each_write_one_plugin_event(home):
    mgr = _plugin_manager(home)
    mgr.install("p1")
    (ev,) = _plugin_events(home)
    assert (ev["kind"], ev["plugin"], ev["version_to"]) == ("plugin_installed", "p1", "2.5.0")
    assert mgr.uninstall("p1") is True
    evs = _plugin_events(home)
    assert [e["kind"] for e in evs] == ["plugin_installed", "plugin_uninstalled"]
    assert (evs[1]["plugin"], evs[1]["version_from"]) == ("p1", "2.5.0")


def test_uninstall_of_an_unknown_plugin_writes_nothing(home):
    mgr = _plugin_manager(home)
    with pytest.raises(KeyError):
        mgr.uninstall("nope")
    assert _plugin_events(home) == []


def test_plugin_install_succeeds_when_the_log_cannot_be_written(home):
    mgr = _plugin_manager(home)
    (home / "logs").write_text("blocker")
    mgr.install("p1")
    assert mgr.is_installed("p1")
