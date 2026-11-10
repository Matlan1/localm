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
    assert mgmt_events.record("plugin_installed", plugin="demo", version_to="1.2") is True
    (ev,) = _events(home)
    assert set(ev) == {"ts", "kind", "plugin", "version_to"}
    assert ev["kind"] == "plugin_installed" and ev["plugin"] == "demo"


def test_config_saved_keeps_key_names_not_values(home):
    mgmt_events.record("config_saved", keys=["b_key", "a_key"])
    (ev,) = _events(home)
    assert ev["keys"] == ["a_key", "b_key"]


def test_unknown_kind_raises_and_writes_nothing(home):
    with pytest.raises(ValueError):
        mgmt_events.record("key_exploded")
    assert _events(home) == []


def test_every_kind_is_accepted(home):
    for kind in sorted(mgmt_events.KINDS):
        assert mgmt_events.record(kind) is True
    assert [e["kind"] for e in _events(home)] == sorted(mgmt_events.KINDS)


def test_text_fields_are_length_capped(home):
    mgmt_events.record("plugin_installed", plugin="x" * 5000)
    (ev,) = _events(home)
    assert len(ev["plugin"]) == 200


def test_rotation_keeps_one_previous_file(home, monkeypatch):
    monkeypatch.setattr(mgmt_events, "MAX_BYTES", 300)
    for _ in range(40):
        mgmt_events.record("config_saved", keys=["k"])
    logs = home / "logs"
    assert sorted(p.name for p in logs.iterdir()) == [
        "management.jsonl", "management.jsonl.1"]
    assert (logs / "management.jsonl").stat().st_size < 600
    assert _events(home)


def test_write_failure_returns_false_and_warns(home, monkeypatch, caplog):
    (home / "logs").write_text("a file where the directory should be")
    with caplog.at_level("WARNING", logger="localm"):
        assert mgmt_events.record("key_minted", principal="x") is False
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
    evs = _events(home)
    assert [e["kind"] for e in evs] == ["login_failed", "login_failed"]
    assert all(isinstance(e["loopback"], bool) for e in evs)
