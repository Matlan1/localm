# SPDX-License-Identifier: AGPL-3.0-or-later
"""Permission-gated, NON-PERSISTENT network-policy override for the two
prerequisite model downloads (the embedding model and the Whisper STT model).

The properties pinned here, most important first:

1. net_mode=off is the DEFAULT floor: even the explicit allow_download=True
   authorization refuses under off, on every surface (voice prefetch, the voice
   download route, the embedding download route), UNLESS the owner has set
   net_allow_model_downloads (admin_only, default False - see
   test_off_but_downloads_allowed_bypasses and
   test_embedder_off_but_downloads_allowed_bypasses below). The bypass rule is
   bypass-ASK-respect-OFF, never bypass-both - net_allow_model_downloads is the
   one deliberate, explicit exception to "never bypass-both".
2. The one-time authorization cannot persist BY CONSTRUCTION: the whole
   download flow leaves config.json byte-identical and never calls
   update_config. Asserted on the DATA (the file, the spy) before any status
   code.
3. bypass-ask: allow_download=True downloads under net_mode=ask, while the
   IMPLICIT paths do not - the transcribe worker is dispatched with
   local_files_only=True whenever the policy did not authorize a download, so
   the child process is structurally unable to fetch.
4. The block reason is always surfaced (stt_available / voice_status reason,
   the can_download flags), never a silent degrade.
5. The bypass is scope-gated on config:write - the same scope that could
   change net_mode itself; a key without it gets a 403 and no download.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import huggingface_hub

from localm import voice

OWNER_KEY = "owner-admin-key-f1-net-override"

_REAL_FIND_SPEC = importlib.util.find_spec


def _fake_faster_whisper_present(monkeypatch):
    """Pretend the faster-whisper package is importable WITHOUT importing it
    (CI installs [dev,rag], not [voice], so the real find_spec answers None
    there and these policy tests would silently test the wrong branch)."""
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a, **kw: (object() if name == "faster_whisper"
                                else _REAL_FIND_SPEC(name, *a, **kw)))


def _voice_cfg(monkeypatch, tmp_path, net_mode: str, model: str = "base"):
    """Isolated home + a controlled config for the pure voice-policy tests."""
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    import localm.config as _cfg
    monkeypatch.setattr(
        _cfg, "load_config",
        lambda: {"voice_stt_model": model, "net_mode": net_mode})


def _lay_whisper_snapshot(name: str = "base") -> None:
    """Materialise the cached-model layout the probe (and the worker) read.
    Reads stt_cache_dir() live, so it follows the test's LOCALM_HOME."""
    snap = (voice.stt_cache_dir()
            / f"models--Systran--faster-whisper-{name}" / "snapshots" / "rev0")
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "model.bin").write_bytes(b"\x00")


def _snapshot_spy(monkeypatch, *, create: bool = False):
    """Replace huggingface_hub.snapshot_download with a recorder. With
    ``create`` it also lays down the snapshot, so the post-download honesty
    check in prefetch_stt_model sees a real model.bin."""
    calls = []

    def _fake(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        if create:
            name = repo_id.split("/", 1)[1].replace("faster-whisper-", "")
            _lay_whisper_snapshot(name)
        return "unused"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake)
    return calls


# --------------------------------------------------------------------------- #
#  1. prefetch_stt_model: bypass-ask-respect-off, one call = one authorization #
# --------------------------------------------------------------------------- #

class TestPrefetchPolicy:
    def test_off_beats_explicit_consent(self, monkeypatch, tmp_path):
        """THE property: net_mode=off refuses even allow_download=True, by
        DEFAULT. If this ever goes green while the spy recorded a call, either
        the kill switch has an unintended bypass, or net_allow_model_downloads
        leaked into this config - _voice_cfg's dict never sets it, so .get(...,
        False) must resolve False here."""
        _voice_cfg(monkeypatch, tmp_path, "off")
        calls = _snapshot_spy(monkeypatch)
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert calls == [], "off must mean NO network call, even authorized"
        assert ok is False
        assert "net_mode=off" in reason

    def test_off_but_downloads_allowed_bypasses(self, monkeypatch, tmp_path):
        """net_allow_model_downloads exempts an explicit prefetch from the off
        floor. Asserted on the spy (a real call happened), the same discipline
        as its sibling above."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(
            _cfg, "load_config",
            lambda: {"voice_stt_model": "base", "net_mode": "off",
                     "net_allow_model_downloads": True})
        calls = _snapshot_spy(monkeypatch, create=True)
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert len(calls) == 1, "the override must let the real download through"
        assert (ok, reason) == (True, "")

    def test_ask_without_consent_refuses_with_reason(self, monkeypatch, tmp_path):
        _voice_cfg(monkeypatch, tmp_path, "ask")
        calls = _snapshot_spy(monkeypatch)
        ok, reason = voice.prefetch_stt_model()          # policy default
        assert calls == []
        assert ok is False
        assert "net_mode=ask" in reason and "download" in reason.lower()

    def test_ask_with_consent_downloads_once(self, monkeypatch, tmp_path):
        _voice_cfg(monkeypatch, tmp_path, "ask")
        calls = _snapshot_spy(monkeypatch, create=True)
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert len(calls) == 1
        repo, kwargs = calls[0]
        assert repo == "Systran/faster-whisper-base"
        # containment: the download lands in localm's own cache dir.
        assert kwargs.get("cache_dir") == str(voice.stt_cache_dir())
        assert (ok, reason) == (True, "")
        assert voice.stt_model_cached() == (True, "base")

    def test_allow_auto_downloads_without_consent(self, monkeypatch, tmp_path):
        _voice_cfg(monkeypatch, tmp_path, "allow")
        calls = _snapshot_spy(monkeypatch, create=True)
        ok, _ = voice.prefetch_stt_model()               # policy default
        assert len(calls) == 1
        assert ok is True

    def test_already_cached_never_touches_the_network(self, monkeypatch, tmp_path):
        _voice_cfg(monkeypatch, tmp_path, "off")
        _lay_whisper_snapshot()
        calls = _snapshot_spy(monkeypatch)
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert calls == []
        assert (ok, reason) == (True, "")

    def test_fetch_without_a_model_bin_reports_failure(self, monkeypatch, tmp_path):
        """Rule 5: a download that 'succeeded' without producing a loadable
        snapshot must not report success."""
        _voice_cfg(monkeypatch, tmp_path, "allow")
        calls = _snapshot_spy(monkeypatch, create=False)   # fetches nothing
        ok, reason = voice.prefetch_stt_model(allow_download=True)
        assert len(calls) == 1
        assert ok is False
        assert "model.bin" in reason

    def test_authorization_never_reaches_config(self, monkeypatch, tmp_path):
        """Non-persistable BY CONSTRUCTION: the whole prefetch path never calls
        update_config, and the config file's bytes are untouched."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        import localm.config as _cfg
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(
            {"voice_stt_model": "base", "net_mode": "ask"}), encoding="utf-8")
        monkeypatch.setattr(_cfg, "CONFIG_FILE", cfg_file)
        writes = []
        monkeypatch.setattr(_cfg, "update_config",
                            lambda *a, **kw: writes.append(a))
        before = cfg_file.read_bytes()
        _snapshot_spy(monkeypatch, create=True)
        ok, _ = voice.prefetch_stt_model(allow_download=True)
        assert cfg_file.read_bytes() == before, \
            "the one-time authorization leaked into config.json"
        assert writes == [], "the prefetch path must never call update_config"
        assert ok is True


# --------------------------------------------------------------------------- #
#  2. stt_available surfaces the real reason                                   #
# --------------------------------------------------------------------------- #

class TestSttAvailableReasons:
    def test_cached_is_available(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "off")
        _lay_whisper_snapshot()
        assert voice.stt_available() == (True, "")

    def test_uncached_allow_is_available(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "allow")
        assert voice.stt_available() == (True, "")

    def test_uncached_ask_reports_the_policy(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "ask")
        ok, reason = voice.stt_available()
        assert ok is False
        assert "net_mode=ask" in reason
        assert "base" in reason                     # names the model

    def test_uncached_off_reports_the_kill_switch(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "off")
        ok, reason = voice.stt_available()
        assert ok is False
        assert "net_mode=off" in reason

    def test_missing_package_still_reported_first(self, monkeypatch, tmp_path):
        _voice_cfg(monkeypatch, tmp_path, "allow")
        monkeypatch.setattr(
            importlib.util, "find_spec",
            lambda name, *a, **kw: (None if name == "faster_whisper"
                                    else _REAL_FIND_SPEC(name, *a, **kw)))
        ok, reason = voice.stt_available()
        assert ok is False
        assert "faster-whisper" in reason


# --------------------------------------------------------------------------- #
#  3. transcribe dispatch: the worker executes the parent's policy decision    #
# --------------------------------------------------------------------------- #

class TestTranscribeDispatchPolicy:
    def _capture_dispatch(self, monkeypatch, response=("ok", "hello")):
        sent = []

        class _FakeQ:
            def put(self, msg):
                sent.append(msg)

            def get(self, timeout=None):
                return response

        monkeypatch.setattr(voice, "_ensure_worker", lambda: None)
        monkeypatch.setattr(voice, "_proc",
                            types.SimpleNamespace(is_alive=lambda: True))
        monkeypatch.setattr(voice, "_req_q", _FakeQ())
        monkeypatch.setattr(voice, "_resp_q", _FakeQ())
        return sent

    def test_uncached_ask_dispatches_offline_only(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "ask")
        sent = self._capture_dispatch(monkeypatch)
        assert voice.transcribe_bytes(b"blob") == "hello"
        assert sent[0][4] is True, \
            "ask must dispatch local_files_only=True (no download in the child)"

    def test_uncached_allow_dispatches_with_download(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "allow")
        sent = self._capture_dispatch(monkeypatch)
        voice.transcribe_bytes(b"blob")
        assert sent[0][4] is False, \
            "allow is the one mode where first use may download"

    def test_cached_loads_offline_even_under_allow(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "allow")
        _lay_whisper_snapshot()
        sent = self._capture_dispatch(monkeypatch)
        voice.transcribe_bytes(b"blob")
        assert sent[0][4] is True, \
            "a cached model must load with no network access at all"

    def test_blocked_load_failure_reports_the_policy(self, monkeypatch, tmp_path):
        """When the policy refused the download and the offline load then
        fails, the error is the POLICY reason (code download-blocked), not a
        mysterious loader message."""
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "ask")
        self._capture_dispatch(monkeypatch,
                               response=("error", "load", "no local snapshot"))
        with pytest.raises(voice.VoiceError) as ei:
            voice.transcribe_bytes(b"blob")
        assert ei.value.code == "download-blocked"
        assert "net_mode=ask" in str(ei.value)
        assert "no local snapshot" in str(ei.value)   # loader detail kept

    def test_allow_load_failure_stays_a_load_error(self, monkeypatch, tmp_path):
        _fake_faster_whisper_present(monkeypatch)
        _voice_cfg(monkeypatch, tmp_path, "allow")
        self._capture_dispatch(monkeypatch,
                               response=("error", "load", "corrupt file"))
        with pytest.raises(voice.VoiceError) as ei:
            voice.transcribe_bytes(b"blob")
        assert ei.value.code == "load"


# --------------------------------------------------------------------------- #
#  App fixtures for the route-level tests (real auth, real keys)               #
# --------------------------------------------------------------------------- #

def _isolated_home(tmp_path, monkeypatch):
    import localm.config as cfg
    home = tmp_path / ".localm"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
    monkeypatch.setattr(cfg, "HOME_DIR", home)
    monkeypatch.setattr(cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", home / "registry.json")
    (home / "config.json").write_text(
        json.dumps({"net_mode": "ask"}), encoding="utf-8")
    monkeypatch.setenv("LOCALM_API_KEY", OWNER_KEY)
    return home


def _join_prefetch_threads():
    for t in threading.enumerate():
        if t.name == voice.PREFETCH_THREAD_NAME:
            t.join(timeout=10)


@pytest.fixture
def voice_client(tmp_path, monkeypatch):
    """The voice plugin on a real auth-enforcing app: an owner key plus a
    voice-only key and a voice+config:write key. install() fires on_install,
    whose prefetch is stubbed and whose thread is JOINED before yielding, so it
    cannot outlive the monkeypatch scope (see the voice_app fixture note in
    test_gui.py)."""
    home = _isolated_home(tmp_path, monkeypatch)
    import localm.voice as _voice
    hook_calls = []
    monkeypatch.setattr(
        _voice, "prefetch_stt_model",
        lambda allow_download=None: (hook_calls.append(allow_download)
                                     or (False, "stubbed")))
    from localm import auth
    voice_only = auth.create_key("mic", ["voice"], allow_privileged=True)["key"]
    voice_writer = auth.create_key(
        "micadmin", ["voice", "config:write"], allow_privileged=True)["key"]
    from localm.inference.http_server import create_app
    app = create_app(None)
    from localm.plugins.engine import PluginManager
    PluginManager(app, external_root=tmp_path / "noplugins").install("voice")
    _join_prefetch_threads()
    with TestClient(app) as c:
        yield types.SimpleNamespace(c=c, home=home, voice_only=voice_only,
                                    voice_writer=voice_writer,
                                    hook_calls=hook_calls, app=app)


def _hdr(key):
    return {"Authorization": f"Bearer {key}"}


def _wait_job(app, job_id, timeout=30.0):
    job = app.state.jobs.get(job_id)
    assert job is not None
    deadline = time.monotonic() + timeout
    while job.status == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
    return job


# --------------------------------------------------------------------------- #
#  4. voice routes                                                             #
# --------------------------------------------------------------------------- #

class TestVoiceRoutes:
    def test_on_install_prefetches_with_one_time_consent(self, voice_client):
        """Installing the plugin is the user's explicit action: the hook runs
        prefetch with allow_download=True (bypass-ask), exactly once, on the
        named background thread the fixture joined."""
        assert voice_client.hook_calls == [True]

    def test_status_reports_reason_and_can_download(self, voice_client, monkeypatch):
        _fake_faster_whisper_present(monkeypatch)
        body = voice_client.c.get("/api/voice/status",
                                  headers=_hdr(OWNER_KEY)).json()
        assert body["available"] is False
        assert "net_mode=ask" in body["reason"]
        assert body["model_cached"] is False
        assert body["can_download"] is True          # owner may authorize

    def test_status_withholds_download_offer_without_scope(self, voice_client,
                                                           monkeypatch):
        _fake_faster_whisper_present(monkeypatch)
        body = voice_client.c.get(
            "/api/voice/status", headers=_hdr(voice_client.voice_only)).json()
        assert body["available"] is False
        assert body["can_download"] is False

    def test_status_offers_download_to_config_write_key(self, voice_client,
                                                        monkeypatch):
        _fake_faster_whisper_present(monkeypatch)
        body = voice_client.c.get(
            "/api/voice/status", headers=_hdr(voice_client.voice_writer)).json()
        assert body["can_download"] is True

    def test_status_never_offers_download_under_off(self, voice_client,
                                                    monkeypatch):
        _fake_faster_whisper_present(monkeypatch)
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        body = voice_client.c.get("/api/voice/status",
                                  headers=_hdr(OWNER_KEY)).json()
        assert body["can_download"] is False
        assert "net_mode=off" in body["reason"]

    def test_status_offers_download_under_off_when_downloads_allowed(
            self, voice_client, monkeypatch):
        _fake_faster_whisper_present(monkeypatch)
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        update_config(lambda c: c.__setitem__("net_allow_model_downloads", True))
        body = voice_client.c.get("/api/voice/status",
                                  headers=_hdr(OWNER_KEY)).json()
        assert body["can_download"] is True

    def test_download_route_403_without_config_write(self, voice_client):
        r = voice_client.c.post("/api/voice/model/download",
                                headers=_hdr(voice_client.voice_only))
        assert r.status_code == 403
        assert "config:write" in r.text

    def test_download_route_409_under_off_even_for_owner(self, voice_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        r = voice_client.c.post("/api/voice/model/download",
                                headers=_hdr(OWNER_KEY))
        assert r.status_code == 409
        assert "net_mode=off" in r.text

    def test_download_route_200_under_off_when_downloads_allowed(self, voice_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        update_config(lambda c: c.__setitem__("net_allow_model_downloads", True))
        r = voice_client.c.post("/api/voice/model/download",
                                headers=_hdr(OWNER_KEY))
        assert r.status_code == 200, r.text

    def test_download_route_runs_one_authorized_job_persists_nothing(
            self, voice_client):
        cfg_file = voice_client.home / "config.json"
        before = cfg_file.read_bytes()
        r = voice_client.c.post("/api/voice/model/download",
                                headers=_hdr(voice_client.voice_writer))
        assert r.status_code == 200, r.text
        job = _wait_job(voice_client.app, r.json()["job_id"])
        # hook_calls[0] is the on_install prefetch; the route adds exactly one
        # more explicit (allow_download=True) authorization.
        assert voice_client.hook_calls == [True, True]
        assert cfg_file.read_bytes() == before, \
            "the download flow must leave config.json byte-identical"
        # The EFFECTIVE mode, not the raw file: config persistence stores only
        # the delta from defaults (_user_delta), so "ask" - the default - never
        # appears as a literal key.
        from localm.config import load_config
        assert load_config()["net_mode"] == "ask"
        # The stubbed prefetch reports failure, so the job status is "failed".
        assert job.status == "failed"

    def test_download_route_short_circuits_when_cached(self, voice_client,
                                                       monkeypatch):
        import localm.voice as _voice
        monkeypatch.setattr(_voice, "stt_model_cached", lambda: (True, "base"))
        r = voice_client.c.post("/api/voice/model/download",
                                headers=_hdr(OWNER_KEY))
        assert r.status_code == 200
        assert r.json()["status"] == "already_cached"


# --------------------------------------------------------------------------- #
#  5. embedding routes                                                         #
# --------------------------------------------------------------------------- #

@pytest.fixture
def rag_client(tmp_path, monkeypatch):
    """The rag plugin on a real auth-enforcing app (mirrors rag_app_env in
    test_config_admin_gating.py) plus a rag-only and a rag+config:write key."""
    home = _isolated_home(tmp_path, monkeypatch)
    from localm import auth
    rag_only = auth.create_key("ragbot", ["rag"], allow_privileged=True)["key"]
    rag_writer = auth.create_key(
        "ragadmin", ["rag", "config:write"], allow_privileged=True)["key"]
    from localm.inference.embedder import reset_embedder
    reset_embedder()
    from localm.inference.http_server import create_app
    app = create_app(None)
    from localm.plugins.engine import PluginManager
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    with TestClient(app) as c:
        yield types.SimpleNamespace(c=c, home=home, rag_only=rag_only,
                                    rag_writer=rag_writer, app=app)
    reset_embedder()


class TestEmbeddingRoutes:
    def test_status_reports_can_download(self, rag_client):
        body = rag_client.c.get("/api/rag/embedding",
                                headers=_hdr(OWNER_KEY)).json()
        assert body["installed"] is False
        assert body["can_download"] is True

    def test_status_withholds_download_offer_without_scope(self, rag_client):
        body = rag_client.c.get("/api/rag/embedding",
                                headers=_hdr(rag_client.rag_only)).json()
        assert body["installed"] is False
        assert body["can_download"] is False

    def test_status_never_offers_download_under_off(self, rag_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        body = rag_client.c.get("/api/rag/embedding",
                                headers=_hdr(OWNER_KEY)).json()
        assert body["can_download"] is False

    def test_status_offers_download_under_off_when_downloads_allowed(self, rag_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        update_config(lambda c: c.__setitem__("net_allow_model_downloads", True))
        body = rag_client.c.get("/api/rag/embedding",
                                headers=_hdr(OWNER_KEY)).json()
        assert body["can_download"] is True

    def test_download_route_403_without_config_write(self, rag_client):
        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(rag_client.rag_only))
        assert r.status_code == 403
        assert "config:write" in r.text

    def test_download_route_409_under_off_even_for_owner(self, rag_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(OWNER_KEY))
        assert r.status_code == 409
        assert "net_mode=off" in r.text

    def test_download_route_200_under_off_when_downloads_allowed(
            self, rag_client, monkeypatch):
        """The route's own redundant pre-check (ahead of resolve_embedding_
        model_path's inner one) honours the override too."""
        from localm.inference.embedder import (
            DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, _embeddings_dir)
        _repo, filename = KNOWN_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
        dest = _embeddings_dir() / filename
        fetched = []

        def _fake_hf_download(repo, fname, local_dir=None, **kw):
            fetched.append((repo, fname))
            target = (dest.parent if local_dir is None else Path(local_dir)) / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"GGUF")
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_download)
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("net_mode", "off"))
        update_config(lambda c: c.__setitem__("net_allow_model_downloads", True))
        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(rag_client.rag_writer))
        assert r.status_code == 200, r.text
        job = _wait_job(rag_client.app, r.json()["job_id"])
        assert fetched == [(_repo, filename)]
        assert job.status == "done"

    def test_download_route_409_for_non_internal_model(self, rag_client):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__(
            "embedding_model", str(rag_client.home / "x.gguf")))
        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(OWNER_KEY))
        assert r.status_code == 409
        assert "internal" in r.text

    def test_download_route_fetches_under_ask_persists_nothing(
            self, rag_client, monkeypatch):
        """The whole point end to end: net_mode=ask blocks the lazy fetch, the
        explicit route fetches anyway (config:write key), and NOTHING lands in
        config - the file is byte-identical and net_mode is still ask."""
        from localm.inference.embedder import (
            DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, _embeddings_dir)
        _repo, filename = KNOWN_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
        dest = _embeddings_dir() / filename
        fetched = []

        def _fake_hf_download(repo, fname, local_dir=None, **kw):
            fetched.append((repo, fname))
            target = (dest.parent if local_dir is None else Path(local_dir)) / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"GGUF")
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hf_download)
        cfg_file = rag_client.home / "config.json"
        before = cfg_file.read_bytes()

        # The lazy path really is blocked under ask:
        from localm.inference.embedder import resolve_embedding_model_path
        assert resolve_embedding_model_path() is None
        assert fetched == []

        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(rag_client.rag_writer))
        assert r.status_code == 200, r.text
        job = _wait_job(rag_client.app, r.json()["job_id"])
        assert dest.is_file(), "the model must actually land on disk"
        assert fetched == [(_repo, filename)]
        assert cfg_file.read_bytes() == before, \
            "the download flow must leave config.json byte-identical"
        # Effective value, not the raw file key - "ask" is the default, and the
        # config store persists only the delta from defaults.
        from localm.config import load_config
        assert load_config()["net_mode"] == "ask"
        assert job.status == "done"
        # And the status flips without any config change:
        body = rag_client.c.get("/api/rag/embedding",
                                headers=_hdr(OWNER_KEY)).json()
        assert body["installed"] is True
        assert body["can_download"] is False

    def test_embedder_off_beats_explicit_consent_at_the_inner_layer(
            self, tmp_path, monkeypatch):
        """The embedder-side twin of the voice off-floor test, pinned at
        _download_known itself: allow_download=True (the explicit consent the
        download route AND the existing change-model route pass) still refuses
        under net_mode=off, by DEFAULT. This inner layer is what keeps the
        floor in place even if a future route forgets its own off pre-check;
        net_allow_model_downloads (see the sibling test below) is the one
        deliberate way past it."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "load_config", lambda: {"net_mode": "off"})
        monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
        from localm.inference import embedder as emb
        emb.reset_embedder()
        fetched = []
        monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                            lambda *a, **kw: fetched.append(a))
        try:
            path = emb.resolve_embedding_model_path(allow_download=True)
            assert fetched == [], "off must mean NO network call, even authorized"
            assert path is None
            assert "network is off" in (emb.last_error() or "")
        finally:
            emb.reset_embedder()

    def test_embedder_off_but_downloads_allowed_bypasses(self, tmp_path, monkeypatch):
        """net_allow_model_downloads exempts an explicit embedding-model fetch
        from the off floor, at the same inner layer the test above pins."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(
            _cfg, "load_config",
            lambda: {"net_mode": "off", "net_allow_model_downloads": True})
        monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
        from localm.inference import embedder as emb
        emb.reset_embedder()
        fetched = []

        def _fake_download(repo, filename, local_dir=None, **kw):
            fetched.append((repo, filename))
            target = Path(local_dir) / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"GGUF")
            return str(target)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
        try:
            path = emb.resolve_embedding_model_path(allow_download=True)
            assert len(fetched) == 1, "the override must let the real fetch through"
            assert path is not None and Path(path).is_file()
        finally:
            emb.reset_embedder()

    def test_ask_mode_message_distinct_from_off(self, tmp_path, monkeypatch):
        """netpolicy.py's documented 'ask' contract: allowed, but surfaces
        confirmation first - NOT the same refusal as net_mode=off. The lazy/
        auto resolve path (allow_download=None, what get_embedder() and every
        memory/RAG caller actually uses) must say the one-time download
        action will work RIGHT NOW under ask, and must not collapse into
        off's wording."""
        monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
        monkeypatch.delenv("LOCALM_NET_MODE", raising=False)
        import localm.config as _cfg
        monkeypatch.setattr(_cfg, "MODELS_DIR", tmp_path / "models")
        from localm.inference import embedder as emb
        try:
            monkeypatch.setattr(_cfg, "load_config", lambda: {"net_mode": "ask"})
            emb.reset_embedder()
            assert emb.resolve_embedding_model_path() is None
            ask_reason = emb.last_error()
            assert ask_reason is not None and "net_mode=ask" in ask_reason
            assert "right now" in ask_reason, \
                "ask must say the one-time action works NOW, not just name the mode"

            monkeypatch.setattr(_cfg, "load_config", lambda: {"net_mode": "off"})
            emb.reset_embedder()
            assert emb.resolve_embedding_model_path() is None
            off_reason = emb.last_error()
            assert off_reason is not None and "net_mode=off" in off_reason

            assert ask_reason != off_reason, \
                "ask and off must not collapse into the identical refusal text"
            assert "right now" not in off_reason, \
                "off's own wording must stay untouched by the ask-mode split"
        finally:
            emb.reset_embedder()

    def test_download_route_short_circuits_when_installed(self, rag_client,
                                                          monkeypatch):
        from localm.inference.embedder import (
            DEFAULT_EMBEDDING_MODEL, KNOWN_EMBEDDING_MODELS, _embeddings_dir)
        _repo, filename = KNOWN_EMBEDDING_MODELS[DEFAULT_EMBEDDING_MODEL]
        dest = _embeddings_dir() / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"GGUF")
        r = rag_client.c.post("/api/rag/embedding/download",
                              headers=_hdr(OWNER_KEY))
        assert r.status_code == 200
        assert r.json()["status"] == "already_installed"


# --------------------------------------------------------------------------- #
#  6. GET /api/memory's can_download_embedder must never advertise an action  #
#     the rag PLUGIN's own mount-level gate would refuse                     #
# --------------------------------------------------------------------------- #

class TestMemoryEmbedderDownloadHint:
    def test_withheld_when_rag_not_installed(self, tmp_path, monkeypatch):
        """rag is a SEPARATE, optional plugin from memory (no `requires` link
        between them). With only memory installed, POST
        /api/rag/embedding/download does not exist at all - the hint must not
        offer an action that would 404."""
        _isolated_home(tmp_path, monkeypatch)
        from localm.inference.embedder import reset_embedder
        reset_embedder()
        from localm.inference.http_server import create_app
        app = create_app(None)
        from localm.plugins.engine import PluginManager
        PluginManager(app, external_root=tmp_path / "noplugins").install("memory")
        try:
            with TestClient(app) as c:
                body = c.get("/api/memory", headers=_hdr(OWNER_KEY)).json()
                assert body["can_download_embedder"] is False
                assert body["embedder_model"] is None
                # Confirm the premise the hint must respect: the route is
                # genuinely absent, not merely unauthorized.
                r = c.post("/api/rag/embedding/download", headers=_hdr(OWNER_KEY))
                assert r.status_code == 404
        finally:
            reset_embedder()

    def test_withheld_without_the_rag_scope(self, tmp_path, monkeypatch):
        """rag installed, but a key scoped to memory+config:write (no "rag")
        hits the download route's MOUNT-level scope gate (require_scope("rag"),
        checked before the route's own config:write check ever runs) and 403s
        regardless of holding config:write. The hint must not offer this key
        an action it cannot take; an owner/rag-scoped caller is unaffected."""
        _isolated_home(tmp_path, monkeypatch)
        from localm.inference.embedder import reset_embedder
        reset_embedder()
        from localm import auth
        mem_writer = auth.create_key(
            "memadmin", ["memory", "config:write"], allow_privileged=True)["key"]
        from localm.inference.http_server import create_app
        app = create_app(None)
        from localm.plugins.engine import PluginManager
        pm = PluginManager(app, external_root=tmp_path / "noplugins")
        pm.install("memory")
        pm.install("rag")
        try:
            with TestClient(app) as c:
                body = c.get("/api/memory", headers=_hdr(mem_writer)).json()
                assert body["can_download_embedder"] is False
                assert body["embedder_model"] is None
                # Confirm the premise: this exact key really is refused there.
                r = c.post("/api/rag/embedding/download", headers=_hdr(mem_writer))
                assert r.status_code == 403
                # The owner key, which passes every scope gate, still gets it.
                owner_body = c.get("/api/memory", headers=_hdr(OWNER_KEY)).json()
                assert owner_body["can_download_embedder"] is True
                assert owner_body["embedder_model"]
        finally:
            reset_embedder()
