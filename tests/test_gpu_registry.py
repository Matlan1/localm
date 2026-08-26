# SPDX-License-Identifier: AGPL-3.0-or-later
"""Multi-instance GPU/VRAM coordination: localm/gpu_registry.py + the
switch_engine() cooperative-unload fallback + POST /v1/instances/cooperate-
unload.

Covers: atomic registry write/read/reap (mirrors instances.py's own tested
pattern), list_gpu_peers' liveness+identity double-check (mocked pid_alive
and the /whoami handshake), request_cooperative_unload's advisory HTTP call,
switch_engine()'s new eviction-exhausted branch (falls back to today's exact
503 when no coordination/no peers/cooperation fails; succeeds and loads when
a peer cooperates), and the new endpoint's token-only auth (never reachable
via a real API key/shell token alone).

Every test here redirects gpu_registry.registry_dir() to a per-test tmp_path
(autouse fixture below) so nothing ever touches the real machine-wide
``%TEMP%/localm/gpu`` directory - that directory could hold a REAL running
localm instance's entry, and this suite must never probe or ask a real
process to unload its model.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from localm import gpu_registry
from localm.inference import http_server as hs
from localm.inference.http_server import create_app
from tests.conftest import probe_double


@pytest.fixture(autouse=True)
def _isolated_registry_dir(tmp_path, monkeypatch):
    """Redirect the module-wide registry location to a throwaway directory for
    every test in this file, and guarantee hs._gpu_coord starts and ends each
    test as None (no cross-test leakage of coordination state)."""
    d = tmp_path / "gpu"
    monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
    hs._gpu_coord = None
    yield d
    hs._gpu_coord = None


# ------------------------------------------------------------------ #
#  gpu_registry: write / read / reap                                 #
# ------------------------------------------------------------------ #

class TestRegistryReadWrite:
    def test_write_entry_writes_full_schema(self, tmp_path):
        d = tmp_path / "reg"
        path = gpu_registry.write_entry(
            d, instance_id="iid1", pid=os.getpid(), port=8642, host="127.0.0.1",
            scheme="http", model="my-model", vram_estimate_bytes=123456,
            gpu_index=0, coordination_token="coord-secret")
        assert path is not None and path.exists()
        entry = json.loads(path.read_text(encoding="utf-8"))
        for key in ("instance_id", "pid", "port", "host", "scheme", "model",
                    "vram_estimate_bytes", "gpu_index", "updated_at",
                    "coordination_token"):
            assert key in entry, f"missing {key}"
        assert entry["instance_id"] == "iid1"
        assert entry["model"] == "my-model"
        assert entry["coordination_token"] == "coord-secret"

    @pytest.mark.skipif(__import__("sys").platform == "win32", reason="POSIX modes only")
    def test_entry_file_is_owner_only_on_posix(self, tmp_path):
        import stat
        d = tmp_path / "reg"
        path = gpu_registry.write_entry(
            d, instance_id="iid1", pid=1, port=1, host="h", scheme="http",
            model=None, vram_estimate_bytes=None, gpu_index=0, coordination_token="t")
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600

    def test_list_entries_and_remove(self, tmp_path):
        d = tmp_path / "reg"
        path = gpu_registry.write_entry(
            d, instance_id="iid2", pid=1, port=1, host="h", scheme="http",
            model=None, vram_estimate_bytes=None, gpu_index=0, coordination_token="t")
        entries = gpu_registry.list_entries(d)
        assert len(entries) == 1 and entries[0]["instance_id"] == "iid2"
        gpu_registry.remove_entry(path)
        assert gpu_registry.list_entries(d) == []

    def test_list_entries_empty_when_dir_missing(self, tmp_path):
        assert gpu_registry.list_entries(tmp_path / "does-not-exist") == []

    def test_list_entries_skips_corrupt(self, tmp_path):
        d = tmp_path / "reg"
        d.mkdir(parents=True)
        (d / "bad.json").write_text("{not json", encoding="utf-8")
        assert gpu_registry.list_entries(d) == []

    def test_write_entry_failure_is_best_effort(self, tmp_path, monkeypatch):
        """A write failure must never raise into the caller - it returns None
        instead, logged rather than crashing the caller's request path.

        The fault is injected at ``config.atomic_write_private``, the writer this
        function delegates to, not at ``pathlib.Path.write_text``, which the
        shared writer no longer calls: a fault injector that silently fails to
        fire is indistinguishable from a guard that correctly found nothing to
        refuse. The ``is None`` assertion is itself the proof the injection took,
        since without it the function returns the written path.
        """
        d = tmp_path / "reg"

        def boom(*a, **k):
            raise OSError("disk full")

        import localm.config as cfg
        monkeypatch.setattr(cfg, "atomic_write_private", boom)
        result = gpu_registry.write_entry(
            d, instance_id="iid3", pid=1, port=1, host="h", scheme="http",
            model=None, vram_estimate_bytes=None, gpu_index=0, coordination_token="t")
        assert result is None


class TestReapStale:
    def _raw(self, d, iid, pid):
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{iid}.json"
        p.write_text(json.dumps({
            "instance_id": iid, "pid": pid, "port": 1, "host": "h",
            "scheme": "http", "model": None, "vram_estimate_bytes": None,
            "gpu_index": 0, "updated_at": "now", "coordination_token": "t",
        }), encoding="utf-8")
        return p

    def test_reap_removes_dead_keeps_live_and_self(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        dead = self._raw(d, "dead1", 111)
        live = self._raw(d, "live1", 222)
        mine = self._raw(d, "self1", 333)
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: pid in (222, 333))
        removed = gpu_registry.reap_stale(d, self_id="self1")
        assert "dead1" in removed
        assert not dead.exists()
        assert live.exists()
        assert mine.exists()

    def test_reap_removes_corrupt(self, tmp_path):
        d = tmp_path / "reg"
        d.mkdir(parents=True)
        bad = d / "corrupt.json"
        bad.write_text("xxx", encoding="utf-8")
        gpu_registry.reap_stale(d)
        assert not bad.exists()

    def test_reap_missing_dir_is_noop(self, tmp_path):
        assert gpu_registry.reap_stale(tmp_path / "nope") == []


class TestAgeSeconds:
    def test_age_seconds_parses_iso(self):
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc) - timedelta(seconds=42)).isoformat()
        age = gpu_registry.age_seconds(ts)
        assert age is not None and 40 <= age <= 50

    def test_age_seconds_none_on_bad_input(self):
        assert gpu_registry.age_seconds(None) is None
        assert gpu_registry.age_seconds("not-a-date") is None


# ------------------------------------------------------------------ #
#  list_gpu_peers: liveness + identity double-check                  #
# ------------------------------------------------------------------ #

class TestListGpuPeers:
    def _write(self, d, iid, port, model=None, pid=None):
        # The default pid is NOT os.getpid(): these entries stand in for a
        # genuinely different process. A pid equal to this test process's own is
        # excluded as self by list_gpu_peers() regardless of exclude_self_id.
        if pid is None:
            pid = os.getpid() + 1
        return gpu_registry.write_entry(
            d, instance_id=iid, pid=pid, port=port, host="127.0.0.1",
            scheme="http", model=model, vram_estimate_bytes=None, gpu_index=0,
            coordination_token=f"tok-{iid}")

    def test_verified_live_peer_is_returned(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        self._write(d, "peer1", 9001, model="peer-model")
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
        monkeypatch.setattr(gpu_registry, "_try_whoami",
                            lambda scheme, port, iid, timeout: True)
        peers = gpu_registry.list_gpu_peers(d)
        assert len(peers) == 1 and peers[0]["instance_id"] == "peer1"

    def test_dead_pid_is_excluded(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        self._write(d, "peer2", 9002, model="peer-model")
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: False)
        monkeypatch.setattr(gpu_registry, "_try_whoami",
                            lambda scheme, port, iid, timeout: True)
        assert gpu_registry.list_gpu_peers(d) == []

    def test_failed_whoami_handshake_is_excluded(self, tmp_path, monkeypatch):
        """A live PID whose /whoami identity check fails (impostor on a reused
        port, or just unreachable) must NOT be trusted as a peer - file
        contents alone are never enough."""
        d = tmp_path / "reg"
        self._write(d, "peer3", 9003, model="peer-model")
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
        monkeypatch.setattr(gpu_registry, "_try_whoami",
                            lambda scheme, port, iid, timeout: False)
        assert gpu_registry.list_gpu_peers(d) == []

    def test_self_is_excluded(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        self._write(d, "self-id", 9004, model="my-model")
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
        monkeypatch.setattr(gpu_registry, "_try_whoami",
                            lambda scheme, port, iid, timeout: True)
        peers = gpu_registry.list_gpu_peers(d, exclude_self_id="self-id")
        assert peers == []

    def test_self_pid_excluded_even_without_exclude_self_id(self, tmp_path, monkeypatch):
        """The caller may have no instance_id to pass at all -
        llamacpp/_sizing.py's _vram_holder_hint has none - and must still never
        see itself as a peer. A registry entry whose pid is THIS test process's
        own pid must be excluded even with no exclude_self_id given, and even
        though pid_alive/_try_whoami would happily vouch for it; otherwise a
        low-VRAM warning blames "another localm instance" that is actually
        itself, port and all."""
        d = tmp_path / "reg"
        self._write(d, "self-by-pid", 9010, model="my-model", pid=os.getpid())
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: True)
        monkeypatch.setattr(gpu_registry, "_try_whoami",
                            lambda scheme, port, iid, timeout: True)
        peers = gpu_registry.list_gpu_peers(d)   # no exclude_self_id passed
        assert peers == []

    def test_missing_directory_returns_empty(self, tmp_path):
        assert gpu_registry.list_gpu_peers(tmp_path / "nope") == []


# ------------------------------------------------------------------ #
#  own_entry: find THIS process's own registry entry by pid           #
# ------------------------------------------------------------------ #

class TestOwnEntry:
    def test_returns_entry_matching_own_pid(self, tmp_path):
        d = tmp_path / "reg"
        gpu_registry.write_entry(
            d, instance_id="me", pid=os.getpid(), port=8642, host="127.0.0.1",
            scheme="http", model="gemma", vram_estimate_bytes=None,
            gpu_index=0, coordination_token="t")
        entry = gpu_registry.own_entry(d)
        assert entry is not None and entry["instance_id"] == "me"

    def test_none_when_no_entry_matches(self, tmp_path):
        d = tmp_path / "reg"
        gpu_registry.write_entry(
            d, instance_id="someone-else", pid=os.getpid() + 1, port=1,
            host="h", scheme="http", model=None, vram_estimate_bytes=None,
            gpu_index=0, coordination_token="t")
        assert gpu_registry.own_entry(d) is None

    def test_none_when_directory_missing(self, tmp_path):
        assert gpu_registry.own_entry(tmp_path / "nope") is None

    def test_default_directory_uses_registry_dir(self, tmp_path, monkeypatch):
        d = tmp_path / "reg"
        monkeypatch.setattr(gpu_registry, "registry_dir", lambda: d)
        gpu_registry.write_entry(
            d, instance_id="me2", pid=os.getpid(), port=1, host="h",
            scheme="http", model=None, vram_estimate_bytes=None,
            gpu_index=0, coordination_token="t")
        entry = gpu_registry.own_entry()   # no directory arg -> registry_dir()
        assert entry is not None and entry["instance_id"] == "me2"


# ------------------------------------------------------------------ #
#  request_cooperative_unload: advisory HTTP call                    #
# ------------------------------------------------------------------ #

class TestRequestCooperativeUnload:
    def test_missing_port_or_token_returns_false(self):
        assert gpu_registry.request_cooperative_unload({}) is False
        assert gpu_registry.request_cooperative_unload({"port": 1}) is False

    def test_success_response_returns_true(self, monkeypatch):
        class _Resp:
            status_code = 200
            def json(self):
                return {"status": "unloaded"}

        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None, verify=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

        monkeypatch.setattr("requests.post", fake_post)
        peer = {"port": 9005, "scheme": "http", "coordination_token": "peer-tok"}
        assert gpu_registry.request_cooperative_unload(peer) is True
        assert "127.0.0.1:9005" in captured["url"]
        assert captured["json"]["coordination_token"] == "peer-tok"
        assert captured["headers"]["X-LocalM-Coordination-Token"] == "peer-tok"

    def test_non_200_returns_false(self, monkeypatch):
        class _Resp:
            status_code = 403
            def json(self):
                return {}

        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp())
        peer = {"port": 9006, "scheme": "http", "coordination_token": "t"}
        assert gpu_registry.request_cooperative_unload(peer) is False

    def test_network_error_returns_false(self, monkeypatch):
        import requests

        def boom(*a, **k):
            raise requests.RequestException("connection refused")

        monkeypatch.setattr("requests.post", boom)
        peer = {"port": 9007, "scheme": "http", "coordination_token": "t"}
        assert gpu_registry.request_cooperative_unload(peer) is False

    def test_malformed_json_returns_false(self, monkeypatch):
        class _Resp:
            status_code = 200
            def json(self):
                raise ValueError("bad json")

        monkeypatch.setattr("requests.post", lambda *a, **k: _Resp())
        peer = {"port": 9008, "scheme": "http", "coordination_token": "t"}
        assert gpu_registry.request_cooperative_unload(peer) is False


# ------------------------------------------------------------------ #
#  switch_engine(): cooperative-unload eviction fallback              #
# ------------------------------------------------------------------ #

class _FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, event):
        pass


def _make_engine(name):
    return _FakeEngine(name)


class _UnfittableEngine(_FakeEngine):
    """Simulates the backend's OWN final sizing decision genuinely refusing
    (GgufBackend._check_vram raising because the model cannot fit even at 0 GPU
    layers - llamacpp/_sizing.py). switch_engine does not hard-refuse on its own
    crude whole-model estimate once local and cooperative eviction are exhausted:
    it falls through to a real load attempt and lets the backend decide,
    converting a genuine backend RuntimeError into a clean 503 (see
    switch_engine's `except RuntimeError` around new_engine.load). These
    cooperative-unload tests are about the COOPERATION SEQUENCING (was the
    registry queried, did a peer get asked, does failure never escalate past
    503), not about whole-model sizing, so the model-b factory here must simulate
    a load that genuinely cannot fit or the fall-through would just succeed
    (200) instead of reaching a 503 to assert on."""

    def load(self):
        raise RuntimeError("VRAM exhausted: cannot fit even at 0 GPU layers")


def _make_unfittable_engine(name):
    return _UnfittableEngine(name)


@pytest.fixture
def multi_model_registry(monkeypatch):
    fake_registry = {
        "model-a": {"path": "Z:/models/model-a.gguf", "source": "local"},
        "model-b": {"path": "Z:/models/model-b.gguf", "source": "local"},
    }
    monkeypatch.setattr("localm.config.load_registry", lambda: fake_registry)
    monkeypatch.setattr("localm.model_manager.get_model_info",
                        lambda name: (f"Z:/models/{name}.gguf", "hint"))
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None
    yield fake_registry


def _dynamic_vram(free_gate=None):
    """10 GB total; ~8 GB consumed per loaded model, UNLESS free_gate()
    reports the coordination freed things up (used to simulate a peer's
    unload actually releasing driver-level VRAM)."""
    def _read():
        if free_gate is not None and free_gate():
            return {"free": 10 * 1024 ** 3, "total": 10 * 1024 ** 3}
        loaded = sum(1 for e in hs._engines.values() if e.loaded)
        free = (10 * 1024 ** 3) - int(loaded * 8 * 1024 ** 3)
        return {"free": free, "total": 10 * 1024 ** 3}
    return _read


class TestSwitchEngineCooperativeUnload:
    def test_falls_back_to_503_without_coordination(self, multi_model_registry, monkeypatch):
        """hs._gpu_coord unset (the default for every existing test and every
        --isolated run) -> the coordination branch is a pure no-op, cooperation
        is never attempted, and the load still ends in a clean 503 when the
        backend's own sizing (simulated here - see _UnfittableEngine) genuinely
        cannot fit it."""
        assert hs._gpu_coord is None
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1  # not locally evictable
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503
        assert "VRAM exhausted" in exc.detail

    def test_cooperation_attempted_but_no_holder_falls_back_to_503(
            self, multi_model_registry, monkeypatch):
        """Coordination IS configured, but no live peer holds a model - the
        attempt is genuinely made (proving the wiring runs), and the load
        still ends in a clean 503 once the backend's own sizing (simulated -
        see _UnfittableEngine) genuinely cannot fit it, never a harder
        failure."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "selftok"}
        calls = {"n": 0}

        def fake_list_peers(exclude_self_id=None):
            calls["n"] += 1
            assert exclude_self_id == "self1"
            return []

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", fake_list_peers)
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503
        assert calls["n"] >= 1, "must actually query the registry before giving up"

    def test_cooperation_failure_falls_back_to_503_not_harder(
            self, multi_model_registry, monkeypatch):
        """A peer exists but declines/fails cooperation - the load still ends
        in a clean 503 once the backend's own sizing (simulated -
        see _UnfittableEngine) genuinely cannot fit it, never escalated."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "selftok"}
        peer_entry = {"instance_id": "peer1", "port": 9100, "scheme": "http",
                      "model": "peer-model", "coordination_token": "peertok"}
        monkeypatch.setattr(gpu_registry, "list_gpu_peers", lambda exclude_self_id=None: [peer_entry])
        monkeypatch.setattr(gpu_registry, "request_cooperative_unload", lambda peer, **k: False)
        monkeypatch.setattr("localm.discover.vram_info", probe_double(_dynamic_vram()))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            with pytest.raises(HTTPException) as exc:
                await hs.switch_engine("model-b", _make_unfittable_engine)
            return exc.value

        exc = asyncio.run(scenario())
        assert exc.status_code == 503

    def test_successful_cooperation_frees_vram_and_load_succeeds(
            self, multi_model_registry, monkeypatch):
        """A live peer holding a model cooperates - the request IS made (with
        the peer's own coordination_token) and, once it reports success, the
        load proceeds without any LOCAL eviction (model-a stays resident)."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "selftok"}
        peer_entry = {"instance_id": "peer1", "port": 9200, "scheme": "http",
                      "model": "peer-model", "coordination_token": "peertok"}
        state = {"cooperated": False}

        def fake_list_peers(exclude_self_id=None):
            return [] if state["cooperated"] else [peer_entry]

        def fake_request(peer, **k):
            assert peer["coordination_token"] == "peertok"
            state["cooperated"] = True
            return True

        monkeypatch.setattr(gpu_registry, "list_gpu_peers", fake_list_peers)
        monkeypatch.setattr(gpu_registry, "request_cooperative_unload", fake_request)
        monkeypatch.setattr("localm.discover.vram_info",
                            probe_double(_dynamic_vram(free_gate=lambda: state["cooperated"])))

        async def scenario():
            await hs.switch_engine("model-a", _make_engine)
            hs._engines["model-a"].active_requests = 1
            return await hs.switch_engine("model-b", _make_engine)

        result = asyncio.run(scenario())
        assert result["status"] == "loaded"
        assert result["model"] == "model-b"
        assert state["cooperated"] is True
        assert "model-b" in hs._engines
        assert "model-a" in hs._engines, "freed via cooperation, never locally evicted"


# ------------------------------------------------------------------ #
#  POST /v1/instances/cooperate-unload: token-only auth              #
# ------------------------------------------------------------------ #

class TestCooperateUnloadEndpointAuth:
    def _client(self):
        return TestClient(create_app(None))

    def test_accepts_correct_token_via_header(self):
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "the-real-token"}
        client = self._client()
        r = client.post("/v1/instances/cooperate-unload",
                        headers={"X-LocalM-Coordination-Token": "the-real-token"})
        assert r.status_code == 200
        assert r.json()["status"] in ("unloaded", "already_unloaded")

    def test_accepts_correct_token_via_json_body(self):
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "the-real-token"}
        client = self._client()
        r = client.post("/v1/instances/cooperate-unload",
                        json={"coordination_token": "the-real-token"})
        assert r.status_code == 200

    def test_rejects_missing_token(self):
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "the-real-token"}
        client = self._client()
        r = client.post("/v1/instances/cooperate-unload")
        assert r.status_code == 403

    def test_rejects_wrong_token(self):
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "the-real-token"}
        client = self._client()
        r = client.post("/v1/instances/cooperate-unload",
                        headers={"X-LocalM-Coordination-Token": "wrong-token"})
        assert r.status_code == 403

    def test_not_reachable_via_a_real_bearer_credential_alone(self):
        """A caller presenting a real per-process secret (standing in for a
        real API key / shell token) as Authorization, WITHOUT the
        coordination_token, must still be refused - proves this is a
        genuinely separate auth path from require_scope/MODELS_WRITE, not
        just 'also accepts a real key'."""
        hs._gpu_coord = {"instance_id": "self1", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "the-real-token"}
        app = create_app(None)
        real_secret = app.state.shell_token
        client = TestClient(app)
        r = client.post("/v1/instances/cooperate-unload",
                        headers={"Authorization": f"Bearer {real_secret}"})
        assert r.status_code == 403

    def test_disabled_when_coordination_not_registered(self):
        """No _gpu_coord at all (isolated run / plain app) - the endpoint
        refuses every token rather than silently accepting one."""
        hs._gpu_coord = None
        client = self._client()
        r = client.post("/v1/instances/cooperate-unload",
                        headers={"X-LocalM-Coordination-Token": "anything"})
        assert r.status_code == 403


# ------------------------------------------------------------------ #
#  _gpu_registry_sync: reflects live state into the registry          #
# ------------------------------------------------------------------ #

class TestGpuRegistrySync:
    def test_sync_noop_without_coordination(self):
        assert hs._gpu_coord is None
        hs._gpu_registry_sync()   # must not raise, must not create the dir

    def test_sync_writes_current_model_state(self, tmp_path, monkeypatch):
        monkeypatch.setattr("localm.model_manager.get_model_info",
                            lambda name: (None, "hint"))  # unresolvable -> size None
        hs._gpu_coord = {"instance_id": "sync1", "port": 4242, "host": "127.0.0.1",
                         "scheme": "http", "token": "tok"}
        hs._active_model_name = "some-model"
        try:
            hs._gpu_registry_sync()
            entries = gpu_registry.list_entries(gpu_registry.registry_dir())
            assert len(entries) == 1
            assert entries[0]["instance_id"] == "sync1"
            assert entries[0]["model"] == "some-model"
            assert entries[0]["port"] == 4242
            assert entries[0]["coordination_token"] == "tok"
        finally:
            hs._active_model_name = None

    def test_sync_survives_write_failure(self, monkeypatch):
        hs._gpu_coord = {"instance_id": "sync2", "port": 1, "host": "127.0.0.1",
                         "scheme": "http", "token": "tok"}

        def boom(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(gpu_registry, "write_entry", boom)
        hs._gpu_registry_sync()   # must not raise


# ------------------------------------------------------------------ #
#  Lifespan wiring: register on startup, clean up on shutdown         #
# ------------------------------------------------------------------ #

class TestLifespanRegistersGpuCoordination:
    """Mirrors test_surfaces.py's `_api_app` pattern: an app wired the way
    instances.advertise() wires it (instance_id/port/scheme/bind_host set on
    app.state), then `with TestClient(app) as client:` to actually run the
    ASGI lifespan startup/shutdown events this feature hooks into."""

    def _advertised_app(self, *, isolated=False):
        app = create_app(None)
        app.state.instance_id = "iid-gpu-lifespan"
        app.state.instance_port = 18642
        app.state.instance_scheme = "http"
        app.state.bind_host = "127.0.0.1"
        app.state.instance_isolated = isolated
        return app

    def test_registers_on_startup_and_removes_on_shutdown(self, tmp_path):
        app = self._advertised_app()
        with TestClient(app):
            entries = gpu_registry.list_entries(gpu_registry.registry_dir())
            assert len(entries) == 1
            assert entries[0]["instance_id"] == "iid-gpu-lifespan"
            assert entries[0]["port"] == 18642
            assert "coordination_token" in entries[0]
            assert hs._gpu_coord is not None
        # Shutdown ran: entry removed, module state reset.
        assert gpu_registry.list_entries(gpu_registry.registry_dir()) == []
        assert hs._gpu_coord is None

    def test_isolated_instance_never_registers(self, tmp_path):
        app = self._advertised_app(isolated=True)
        with TestClient(app):
            assert gpu_registry.list_entries(gpu_registry.registry_dir()) == []
            assert hs._gpu_coord is None

    def test_plain_app_without_instance_id_never_registers(self, tmp_path):
        """A bare create_app() (no instances.advertise() wiring at all - every
        pre-existing test in this codebase) must never touch the registry."""
        app = create_app(None)
        with TestClient(app):
            assert gpu_registry.list_entries(gpu_registry.registry_dir()) == []
            assert hs._gpu_coord is None

    def test_startup_reaps_a_dead_peers_leftover_entry(self, tmp_path, monkeypatch):
        """A prior instance that crashed (SIGKILL, no shutdown cleanup) leaves
        its entry on disk forever unless something sweeps it. Confirms the
        NEXT instance to start does that sweep (gpu_registry.reap_stale wired
        into the same startup path instances.advertise() uses), not merely
        that a dead entry is filtered out of list_gpu_peers - the entry must
        actually be gone from disk afterward."""
        d = gpu_registry.registry_dir()
        d.mkdir(parents=True, exist_ok=True)
        dead = d / "dead-peer.json"
        dead.write_text(json.dumps({
            "instance_id": "dead-peer", "pid": 999999, "port": 1, "host": "h",
            "scheme": "http", "model": None, "vram_estimate_bytes": None,
            "gpu_index": 0, "updated_at": "now", "coordination_token": "t",
        }), encoding="utf-8")
        monkeypatch.setattr(gpu_registry, "pid_alive", lambda pid: pid != 999999)

        app = self._advertised_app()
        with TestClient(app):
            entries = {e["instance_id"]
                      for e in gpu_registry.list_entries(gpu_registry.registry_dir())}
            assert entries == {"iid-gpu-lifespan"}
        assert not dead.exists()
