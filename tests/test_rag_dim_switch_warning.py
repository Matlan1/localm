# SPDX-License-Identifier: AGPL-3.0-or-later
"""Switching the embedding model (``POST /api/rag/embedding``) must enumerate the
collections it is about to invalidate, comparing each stored dimension, rather
than ending with a generic "click reindex on a collection below".

A collection built under the OLD model still shows "hybrid" in
``/api/rag/collections`` right after the switch (``has_vectors`` is a purely
offline fact, never compared against the live model) - only an actual query
discovers the mismatch and silently drops to BM25.

Covers, bottom-up: ``Collection.vector_dim()`` (the accessor), then
``_collection_dim_report()`` (the enumeration/comparison), then the real
``POST /api/rag/embedding`` job end to end with only the embedder mocked, so the
wiring between them - not just each piece in isolation - is exercised.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from localm.rag.store import Collection


# --------------------------------------------------------------------------- #
#  Fixture helpers                                                            #
# --------------------------------------------------------------------------- #

def _embedder(dim):
    def embed(texts):
        return [[0.1] * dim for _ in texts]
    return embed


def _collection(base: Path, name: str, texts: list[str], dim: int | None = None
                 ) -> Collection:
    """A saved collection with *texts* as chunks and, when *dim* is given,
    real vectors at that dimension - mirrors test_rag_reembed.py's helper of
    the same shape, parametrised on *base* so it can point at a fixture's
    rag_dir() instead of always tmp_path directly."""
    c = Collection(name, base=base).create()
    c._chunks = [{"source": f"doc{i}.txt", "pos": i, "text": t}
                 for i, t in enumerate(texts)]
    if dim is not None:
        c._vectors = _embedder(dim)(texts)
    c._save()
    return c


@pytest.fixture
def rag_home(tmp_path, monkeypatch) -> Path:
    """Points rag_dir() (via home_dir()) at tmp_path, mirroring the pattern in
    test_disclosure.py's rag_app / test_rag_api_mode.py's api_mode_app - so
    collection_names()/Collection(name) (both default base=rag_dir()) resolve
    under this test's own tmp_path with no explicit base= needed."""
    home = tmp_path / "userhome"
    home.mkdir()
    data = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(data))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", data)
    monkeypatch.setattr(cfg, "MODELS_DIR", data / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", data / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", data / "registry.json")
    from localm.rag.store import rag_dir
    return rag_dir()


# --------------------------------------------------------------------------- #
#  Collection.vector_dim()                                                    #
# --------------------------------------------------------------------------- #

class TestVectorDim:
    def test_known_dim_from_a_freshly_saved_collection(self, tmp_path):
        c = _collection(tmp_path, "kb", ["alpha", "beta"], dim=384)
        assert c.vector_dim() == 384
        assert Collection("kb", base=tmp_path).vector_dim() == 384

    def test_none_when_never_embedded(self, tmp_path):
        c = _collection(tmp_path, "kb", ["alpha", "beta"], dim=None)
        assert c.vector_dim() is None
        assert Collection("kb", base=tmp_path).stats()["has_vectors"] is False

    def test_falls_back_to_the_first_vector_for_a_legacy_file_with_no_dim_key(
            self, tmp_path):
        """A vectors.json written before the top-level "dim" field existed -
        _load()'s own fallback (data.get("dim") or _first_dim(vectors)) must
        still resolve a real number, matching what the C3 add-time guard
        already trusts as this collection's dimension."""
        _collection(tmp_path, "kb", ["alpha", "beta"], dim=None)
        (tmp_path / "kb" / "vectors.json").write_text(
            json.dumps({"vectors": [[0.1] * 12, [0.2] * 12]}), encoding="utf-8")
        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.vector_dim() == 12

    def test_none_when_the_vectors_file_is_corrupt(self, tmp_path):
        _collection(tmp_path, "kb", ["alpha", "beta"], dim=8)
        (tmp_path / "kb" / "vectors.json").write_text("{not json", encoding="utf-8")
        reloaded = Collection("kb", base=tmp_path)
        assert reloaded.vector_dim() is None
        assert reloaded.stats()["has_vectors"] is False
        assert reloaded.stats()["vector_degrade_reason"]


# --------------------------------------------------------------------------- #
#  _collection_dim_report()                                                   #
# --------------------------------------------------------------------------- #

def _report(target_dim):
    from localm.plugins.builtin.rag.plug import _collection_dim_report
    return _collection_dim_report(target_dim)


class TestCollectionDimReport:
    def test_a_collection_at_a_different_dim_is_flagged_to_degrade(self, rag_home):
        _collection(rag_home, "docs", ["a", "b", "c"], dim=768)
        report = _report(384)
        assert report["degrades"] == [{"name": "docs", "dim": 768, "n_chunks": 3}]
        assert report["unknown"] == []
        assert report["unaffected"] == 0

    def test_a_collection_already_at_the_target_dim_is_only_counted(self, rag_home):
        _collection(rag_home, "docs", ["a", "b"], dim=384)
        report = _report(384)
        assert report["degrades"] == []
        assert report["unaffected"] == 1

    def test_a_never_embedded_collection_is_not_mentioned_at_all(self, rag_home):
        """It was already BM25-only before the switch and stays BM25-only
        after it - nothing about THIS action changed anything for it (the
        existing 're-embed needed' GUI badge already covers it, independent
        of any model switch)."""
        _collection(rag_home, "notes", ["a"], dim=None)
        report = _report(384)
        assert report == {"degrades": [], "unknown": [], "unaffected": 0}

    def test_multiple_collections_are_sorted_into_the_right_buckets(self, rag_home):
        _collection(rag_home, "old", ["a", "b"], dim=768)          # degrades
        _collection(rag_home, "current", ["a"], dim=384)           # unaffected
        _collection(rag_home, "empty", ["a"], dim=None)            # not mentioned
        report = _report(384)
        assert [c["name"] for c in report["degrades"]] == ["old"]
        assert report["unaffected"] == 1

    def test_a_corrupt_vectors_file_is_not_reported_as_will_degrade(self, rag_home):
        """A collection whose vectors were already unusable BEFORE this
        switch is a pre-existing state (has_vectors already False, surfaced
        by the GUI's own 're-embed needed' badge) - it must not be
        misreported as something THIS switch newly broke."""
        _collection(rag_home, "broken", ["a", "b"], dim=8)
        (rag_home / "broken" / "vectors.json").write_text("{not json", encoding="utf-8")
        report = _report(384)
        assert report["degrades"] == []
        assert not any(c["name"] == "broken" for c in report["unknown"])

    def test_dim_none_while_has_vectors_true_renders_unknown_not_fine(
            self, rag_home, monkeypatch):
        """Not reachable under the current has_vectors/vector_dim coupling
        (has_vectors implies a resolvable vector_dim - see vector_dim()'s
        docstring), but the report must still treat that combination as
        UNKNOWN rather than silently skipping it as fine or reporting a
        false dim mismatch, in case that coupling ever changes."""
        _collection(rag_home, "docs", ["a", "b"], dim=768)
        from localm.rag.store import Collection as StoreCollection
        monkeypatch.setattr(StoreCollection, "vector_dim", lambda self: None)
        report = _report(384)
        assert report["degrades"] == []
        assert [c["name"] for c in report["unknown"]] == ["docs"]

    def test_an_unreadable_collection_directory_is_reported_unknown_not_dropped(
            self, rag_home):
        """A hand-placed or half-deleted directory that fails to even
        construct as a Collection must not silently vanish from the report:
        best-effort here means naming the failure, not folding it into a
        false "nothing to see"."""
        rogue = rag_home / "not.a.valid.name"
        rogue.mkdir(parents=True)
        (rogue / "meta.json").write_text("{}", encoding="utf-8")
        report = _report(384)
        assert len(report["unknown"]) == 1
        assert "not.a.valid.name" in report["unknown"][0]["name"]
        assert report["unknown"][0]["reason"]

    def test_no_collections_at_all_reports_cleanly(self, rag_home):
        assert _report(384) == {"degrades": [], "unknown": [], "unaffected": 0}


# --------------------------------------------------------------------------- #
#  POST /api/rag/embedding end to end: the job's own output lines             #
# --------------------------------------------------------------------------- #

@pytest.fixture
def embedding_route_app(rag_home, monkeypatch):
    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui

    app = FastAPI()
    PluginManager(app, external_root=rag_home.parent / "noplugins").install("rag")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


def _run_job_and_get_lines(client, app, model: str) -> str:
    # confirm=True: these tests exercise the ACTUAL switch (the job that
    # writes config, loads the model, and reports the dim-mismatch warning) -
    # see TestEmbeddingSetConfirmGate for the unconfirmed dry-run itself.
    r = client.post("/api/rag/embedding", json={"model": model, "confirm": True})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    job = app.state.jobs.get(job_id)
    import time
    deadline = time.time() + 10
    while job.status == "running" and time.time() < deadline:
        time.sleep(0.02)
    assert job.status == "done", f"job did not finish cleanly: {job.status}"
    return "\n".join(e.get("text", "") for e in job._history if e.get("type") == "line")


class TestEmbeddingSwitchRouteEndToEnd:
    def test_switch_names_the_collection_that_will_degrade(
            self, embedding_route_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["alpha", "beta"], dim=768)
        import localm.inference.embedder as emb

        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder",
                            lambda **kw: type("E", (), {"embed": staticmethod(
                                lambda texts: [[0.1] * 384 for _ in texts])})())

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            text = _run_job_and_get_lines(c, embedding_route_app, "new-model")

        assert "Ready: new-model (384-dim)" in text
        assert "docs" in text
        assert "768-dim" in text
        assert "2 chunks" in text
        assert "re-embed" in text
        assert "1 existing collection(s) will fall back to BM25" in text

    def test_switch_says_nothing_extra_when_nothing_is_affected(
            self, embedding_route_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["alpha"], dim=384)
        import localm.inference.embedder as emb

        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder",
                            lambda **kw: type("E", (), {"embed": staticmethod(
                                lambda texts: [[0.1] * 384 for _ in texts])})())

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            text = _run_job_and_get_lines(c, embedding_route_app, "new-model")

        assert "Ready: new-model (384-dim)" in text
        assert "will fall back to BM25" not in text
        assert "could not be checked" not in text

    def test_switch_with_no_collections_at_all_is_unaffected_by_the_new_code(
            self, embedding_route_app, monkeypatch):
        import localm.inference.embedder as emb

        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder",
                            lambda **kw: type("E", (), {"embed": staticmethod(
                                lambda texts: [[0.1] * 384 for _ in texts])})())

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            text = _run_job_and_get_lines(c, embedding_route_app, "new-model")

        assert "Ready: new-model (384-dim)" in text
        assert "will fall back to BM25" not in text


# --------------------------------------------------------------------------- #
#  POST /api/rag/embedding without confirm=True is a DRY RUN: the warning      #
#  lands BEFORE the switch takes effect. No embedder mocking here: an           #
#  unconfirmed request must never touch                                         #
#  resolve_embedding_model_path/get_embedder at all - it answers from           #
#  meta.json alone (embedding_model()), same as _collection_dim_report.         #
# --------------------------------------------------------------------------- #

class TestEmbeddingSetConfirmGate:
    def test_unconfirmed_does_not_write_config_or_start_a_job(
            self, embedding_route_app, rag_home):
        from localm.config import load_config
        before = load_config().get("embedding_model")

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            r = c.post("/api/rag/embedding", json={"model": "new-model"})

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["needs_confirm"] is True
        assert "job_id" not in data
        assert load_config().get("embedding_model") == before, \
            "an unconfirmed request must not write embedding_model"

    def test_unconfirmed_names_collections_with_embeddings_but_asserts_no_dim(
            self, embedding_route_app, rag_home):
        c = _collection(rag_home, "docs", ["alpha", "beta"], dim=768)
        c._meta["embedding_model"] = "old-model"
        c._save()

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as post_client:
            r = post_client.post("/api/rag/embedding", json={"model": "new-model"})

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["needs_confirm"] is True
        assert data["model"] == "new-model"
        assert data["collections"] == [
            {"name": "docs", "built_with": "old-model", "n_chunks": 2}]
        assert "may invalidate" in data["note"]
        assert "1 existing collection" in data["note"]
        # No dimension is asserted anywhere in the report - that would require
        # loading the candidate model, which confirm=False must never do.
        assert "768" not in data["note"]
        assert "dim" not in str(data["collections"])

    def test_unconfirmed_with_no_embedded_collections_says_nothing_to_invalidate(
            self, embedding_route_app, rag_home):
        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            r = c.post("/api/rag/embedding", json={"model": "new-model"})

        data = r.json()
        assert data["collections"] == []
        assert "nothing to invalidate" in data["note"]

    def test_unconfirmed_omits_a_collection_with_no_vectors(
            self, embedding_route_app, rag_home):
        # Lexical-only (never embedded) collections have nothing a model
        # switch could invalidate, so they must not pad the count.
        _collection(rag_home, "lexical-only", ["alpha"], dim=None)

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            r = c.post("/api/rag/embedding", json={"model": "new-model"})

        assert r.json()["collections"] == []

    def test_unconfirmed_names_an_unreadable_collection_without_leaking_the_exception(
            self, embedding_route_app, rag_home, caplog):
        """A construction failure must still be NAMED in the response, not
        silently dropped (mirroring _collection_dim_report's own 'unknown'
        bucket), but the raw exception text must never reach the HTTP response
        body - only the server-side log. _collection_dim_report's identical
        'reason' field has no such exposure because its only reader is _setup(),
        which logs just the collection NAME and never re-serializes 'reason';
        this route serializes its whole report straight into JSON."""
        rogue = rag_home / "not.a.valid.name"
        rogue.mkdir(parents=True)
        (rogue / "meta.json").write_text("{}", encoding="utf-8")

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c, \
                caplog.at_level("WARNING", logger="localm"):
            r = c.post("/api/rag/embedding", json={"model": "new-model"})

        data = r.json()
        assert len(data["collections"]) == 1
        entry = data["collections"][0]
        assert entry["name"] == "not.a.valid.name"
        assert entry["reason"] == "could not be read"
        # The property under test: nothing exception-shaped in the RESPONSE.
        body_text = r.text
        assert "ValueError" not in body_text
        assert "Traceback" not in body_text
        # The failure is still surfaced, just server-side: noted, not muted.
        assert "not.a.valid.name" in caplog.text
        assert "ValueError" in caplog.text

    def test_confirming_after_a_dry_run_actually_switches(
            self, embedding_route_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["alpha", "beta"], dim=768)
        import localm.inference.embedder as emb

        monkeypatch.setattr(emb, "resolve_embedding_model_path",
                            lambda **kw: "/fake/new-model.gguf")
        monkeypatch.setattr(emb, "get_embedder",
                            lambda **kw: type("E", (), {"embed": staticmethod(
                                lambda texts: [[0.1] * 384 for _ in texts])})())

        from fastapi.testclient import TestClient
        with TestClient(embedding_route_app) as c:
            dry = c.post("/api/rag/embedding", json={"model": "new-model"})
            assert dry.json()["needs_confirm"] is True

            text = _run_job_and_get_lines(c, embedding_route_app, "new-model")

        assert "Ready: new-model (384-dim)" in text
        from localm.config import load_config
        assert load_config().get("embedding_model") == "new-model"


# --------------------------------------------------------------------------- #
#  PATCH /v1/config is the OTHER GUI writer of embedding_model (the Settings   #
#  page's editable field). It shares collection_provenance_report()/_note()    #
#  (in localm/rag/store.py) with the RAG picker above. Unlike that             #
#  always-two-step route, PATCH /v1/config is a generic multi-key settings     #
#  save, so it only gates on a REAL value change with something to lose - a    #
#  no-op or nothing-at-risk write completes in one round trip.                 #
# --------------------------------------------------------------------------- #

CONFIG_OWNER_KEY = "owner-admin-key-rag-dim-no-reembed"


def _owner_headers():
    return {"Authorization": f"Bearer {CONFIG_OWNER_KEY}"}


@pytest.fixture
def config_app(rag_home, monkeypatch):
    """A real core app (PATCH /v1/config included), sharing rag_home's
    monkeypatched HOME_DIR/Path.home so collection_names()/Collection(name)
    resolve into the same directory the TestClient's requests operate
    against. An owner API key (mirrors test_config_admin_gating.py's
    app_env): a bare TestClient supplies no GUI shell token, and an actual
    config WRITE (unlike the dry-run/needs_confirm response, which never
    reaches update_config()) requires either that or an owner key."""
    monkeypatch.setenv("LOCALM_API_KEY", CONFIG_OWNER_KEY)
    from localm.inference.http_server import create_app
    return create_app(None)


class TestPatchConfigEmbeddingConfirmGate:
    def test_change_with_affected_collections_needs_confirm_and_does_not_write(
            self, config_app, rag_home):
        c = _collection(rag_home, "docs", ["alpha", "beta"], dim=768)
        c._meta["embedding_model"] = "old-model"
        c._save()
        from localm.config import load_config
        before = load_config().get("embedding_model")

        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            r = client.patch("/v1/config", headers=_owner_headers(),
                             json={"embedding_model": "new-model"})

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["needs_confirm"] is True
        assert data["model"] == "new-model"
        assert data["collections"] == [
            {"name": "docs", "built_with": "old-model", "n_chunks": 2}]
        assert "may invalidate" in data["note"]
        assert load_config().get("embedding_model") == before, \
            "an unconfirmed PATCH must not write embedding_model"

    def test_confirm_true_actually_writes(self, config_app, rag_home):
        _collection(rag_home, "docs", ["alpha"], dim=768)

        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            r = client.patch(
                "/v1/config", headers=_owner_headers(),
                json={"embedding_model": "new-model", "confirm": True})

        assert r.status_code == 200, r.text
        assert "needs_confirm" not in r.json()
        from localm.config import load_config
        assert load_config().get("embedding_model") == "new-model"

    def test_no_affected_collections_writes_directly_no_gate(
            self, config_app, rag_home):
        # Nothing has embeddings yet - nothing at risk, so a plain PATCH (no
        # confirm) still writes in one round trip. The gate warns about real
        # risk, not every embedding_model write.
        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            r = client.patch("/v1/config", headers=_owner_headers(),
                             json={"embedding_model": "new-model"})

        assert r.status_code == 200, r.text
        assert "needs_confirm" not in r.json()
        from localm.config import load_config
        assert load_config().get("embedding_model") == "new-model"

    def test_setting_the_same_value_is_a_noop_not_gated(self, config_app, rag_home):
        from localm.config import update_config
        update_config(lambda c: c.__setitem__("embedding_model", "same-model"))
        c = _collection(rag_home, "docs", ["alpha"], dim=768)
        c._meta["embedding_model"] = "same-model"
        c._save()

        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            r = client.patch("/v1/config", headers=_owner_headers(),
                             json={"embedding_model": "same-model"})

        assert r.status_code == 200, r.text
        assert "needs_confirm" not in r.json()

    def test_other_fields_in_the_same_batch_land_together_once_confirmed(
            self, config_app, rag_home):
        # A Settings-page save can bundle unrelated fields with the
        # embedding_model change (one Save button per section). The confirm
        # gate must not silently drop them once the user proceeds.
        c = _collection(rag_home, "docs", ["alpha"], dim=768)
        c._meta["embedding_model"] = "old-model"
        c._save()

        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            dry = client.patch(
                "/v1/config", headers=_owner_headers(),
                json={"embedding_model": "new-model", "net_allow": ["x.com"]})
            assert dry.json()["needs_confirm"] is True

            r = client.patch(
                "/v1/config", headers=_owner_headers(),
                json={"embedding_model": "new-model", "net_allow": ["x.com"],
                      "confirm": True})

        assert r.status_code == 200, r.text
        from localm.config import load_config
        cfg = load_config()
        assert cfg.get("embedding_model") == "new-model"
        assert cfg.get("net_allow") == ["x.com"]

    def test_confirm_key_itself_is_never_persisted_as_a_config_key(
            self, config_app, rag_home):
        from fastapi.testclient import TestClient
        with TestClient(config_app) as client:
            r = client.patch(
                "/v1/config", headers=_owner_headers(),
                json={"embedding_model": "new-model", "confirm": True})

        assert r.status_code == 200, r.text
        from localm.config import load_config
        assert "confirm" not in load_config()


# --------------------------------------------------------------------------- #
#  The one-shot job-log warning above loads and test-embeds the NEW model, so  #
#  it only runs at switch time and never reaches the PERSISTENT list/detail    #
#  badge. dim_mismatch is the best-effort badge (never a load, never a false   #
#  "matches"): it compares each collection's own vector_dim (already cached,   #
#  free) against embedder.loaded_dim() (whatever is already resident).         #
# --------------------------------------------------------------------------- #

def _dim_mismatch(stats, active_dim):
    from localm.plugins.builtin.rag.plug import _dim_mismatch as fn
    return fn(stats, active_dim)


class TestDimMismatchHelper:
    def test_none_when_no_embedder_is_resident(self):
        assert _dim_mismatch({"has_vectors": True, "vector_dim": 768}, None) is None

    def test_none_when_the_collection_has_no_vectors(self):
        assert _dim_mismatch({"has_vectors": False, "vector_dim": None}, 384) is None

    def test_none_when_the_collections_own_dim_is_unknown(self):
        assert _dim_mismatch({"has_vectors": True, "vector_dim": None}, 384) is None

    def test_true_when_dims_disagree(self):
        assert _dim_mismatch({"has_vectors": True, "vector_dim": 768}, 384) is True

    def test_false_when_dims_agree(self):
        assert _dim_mismatch({"has_vectors": True, "vector_dim": 384}, 384) is False


@pytest.fixture
def dim_badge_app(rag_home):
    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui

    app = FastAPI()
    PluginManager(app, external_root=rag_home.parent / "noplugins").install("rag")

    async def switch_model(name):
        pass

    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "model-a")
    return app


class TestListDetailRoutesSurfaceDimMismatch:
    def test_list_route_flags_a_collection_whose_vectors_predate_the_resident_model(
            self, dim_badge_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["a", "b"], dim=768)
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: 384)

        from fastapi.testclient import TestClient
        with TestClient(dim_badge_app) as c:
            r = c.get("/api/rag/collections")
        assert r.status_code == 200
        row = next(x for x in r.json()["collections"] if x["name"] == "docs")
        assert row["has_vectors"] is True, "still reports hybrid - the vectors are real, just stale"
        assert row["dim_mismatch"] is True

    def test_list_route_does_not_flag_a_collection_matching_the_resident_model(
            self, dim_badge_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["a", "b"], dim=384)
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: 384)

        from fastapi.testclient import TestClient
        with TestClient(dim_badge_app) as c:
            r = c.get("/api/rag/collections")
        row = next(x for x in r.json()["collections"] if x["name"] == "docs")
        assert row["dim_mismatch"] is False

    def test_list_route_reports_unknown_rather_than_a_false_match_when_no_embedder_is_loaded(
            self, dim_badge_app, rag_home, monkeypatch):
        """The common cold-start case: nothing loaded yet. Must read as 'cannot
        tell' (None), never silently as 'matches' (False) - a false False here
        would be indistinguishable from a genuine match and hide exactly the
        collections a user most needs to be warned about."""
        _collection(rag_home, "docs", ["a", "b"], dim=768)
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: None)

        from fastapi.testclient import TestClient
        with TestClient(dim_badge_app) as c:
            r = c.get("/api/rag/collections")
        row = next(x for x in r.json()["collections"] if x["name"] == "docs")
        assert row["dim_mismatch"] is None

    def test_list_route_never_flags_a_bm25_only_collection(
            self, dim_badge_app, rag_home, monkeypatch):
        _collection(rag_home, "notes", ["a"], dim=None)
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: 384)

        from fastapi.testclient import TestClient
        with TestClient(dim_badge_app) as c:
            r = c.get("/api/rag/collections")
        row = next(x for x in r.json()["collections"] if x["name"] == "notes")
        assert row["dim_mismatch"] is None, "nothing to compare - it was BM25-only before this too"

    def test_detail_route_flags_the_same_way_as_the_list_route(
            self, dim_badge_app, rag_home, monkeypatch):
        _collection(rag_home, "docs", ["a", "b"], dim=768)
        monkeypatch.setattr("localm.inference.embedder.loaded_dim", lambda: 384)

        from fastapi.testclient import TestClient
        with TestClient(dim_badge_app) as c:
            r = c.get("/api/rag/collections/docs")
        assert r.status_code == 200
        assert r.json()["dim_mismatch"] is True
