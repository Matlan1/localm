# SPDX-License-Identifier: AGPL-3.0-or-later
"""The Knowledge routes, the scheduled re-sync and the store lose, hide or
misreport nothing the person did not choose.

Every case drives the real store, the real click commands, the real routes and
the real scheduled-job runner under a throwaway LOCALM_HOME. Only the embedder
(a server call) is replaced, by a deterministic function that can be made to
fail.

  * a re-index that cannot re-embed a document refuses, before anything is
    saved, instead of dropping the document's vectors;
  * the scheduled re-sync keeps listing a document that is still missing;
  * the repair route names the sources that vanished from disk;
  * the add and upload routes refuse (409) a re-index that would drop vectors
    until the caller confirms;
  * summary lines say "1 chunk".
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag import Collection


def _embed_ok(texts):
    return [[float(len(t) % 7), 1.0, 0.5] for t in texts]


class _Flaky:
    """An embedder that works until its *n*-th call, then raises."""

    def __init__(self, fail_on):
        self.fail_on = fail_on
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        if self.calls >= self.fail_on:
            raise RuntimeError("embedder went away")
        return _embed_ok(texts)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path / "lh"))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    import localm.config as cfg
    lh = tmp_path / "lh"
    lh.mkdir()
    monkeypatch.setattr(cfg, "HOME_DIR", lh)
    monkeypatch.setattr(cfg, "MODELS_DIR", lh / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", lh / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", lh / "registry.json")
    (lh / "config.json").write_text(
        '{"rag_indexing_mode": "blacklist", "rag_denied_roots": []}',
        encoding="utf-8")
    return lh


@pytest.fixture
def docs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    for n in ("a", "b", "c"):
        (d / f"{n}.txt").write_text(f"{n} text about {n * 5} things", encoding="utf-8")
    return d


def _hybrid(name, docs):
    c = Collection(name)
    c.create()
    c.add_paths([docs], embed_fn=_embed_ok, model_name="fake")
    s = Collection(name, cache=False).stats()
    assert s["has_vectors"] and s["n_chunks"] == 3, "fixture precondition"
    return c


def _disk(name):
    """What a fresh reader sees: (chunks, vectors present, has_vectors)."""
    c = Collection(name, cache=False)
    s = c.stats()
    present = sum(1 for v in (c._vectors or []) if v)
    return s["n_chunks"], present, s["has_vectors"]


# --------------------------------------------------------------------------- #
#  The store: a re-index never drops vectors it cannot replace                 #
# --------------------------------------------------------------------------- #

class TestStoreKeepsVectors:
    def test_force_without_an_embedder_refuses_and_changes_nothing(self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        before = _disk("kb")
        with pytest.raises(EmbeddingLossRefused) as ei:
            Collection("kb").add_paths([docs], force=True, embed_fn=None)
        assert "no embedding model" in str(ei.value)
        assert _disk("kb") == before == (3, 3, True)

    def test_force_with_an_embedder_that_dies_half_way_keeps_every_vector(
            self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        with pytest.raises(EmbeddingLossRefused) as ei:
            Collection("kb").add_paths([docs], force=True, embed_fn=_Flaky(2),
                                       model_name="fake")
        assert "embedder went away" in str(ei.value)
        assert _disk("kb") == (3, 3, True)

    def test_a_short_answer_from_the_embedder_is_refused_too(self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        with pytest.raises(EmbeddingLossRefused):
            Collection("kb").add_paths([docs], force=True,
                                       embed_fn=lambda texts: [], model_name="fake")
        assert _disk("kb") == (3, 3, True)

    def test_non_finite_vectors_are_refused_too(self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        with pytest.raises(EmbeddingLossRefused):
            Collection("kb").add_paths(
                [docs], force=True, model_name="fake",
                embed_fn=lambda texts: [[float("nan"), 1.0, 0.5] for _ in texts])
        assert _disk("kb") == (3, 3, True)

    def test_a_chosen_loss_goes_through(self, home, docs):
        _hybrid("kb", docs)
        r = Collection("kb").add_paths([docs], force=True, embed_fn=None,
                                       allow_vector_loss=True)
        assert r["updated"] == 3
        assert _disk("kb") == (3, 0, False)

    def test_a_working_embedder_re_embeds_under_force(self, home, docs):
        _hybrid("kb", docs)
        r = Collection("kb").add_paths([docs], force=True, embed_fn=_embed_ok,
                                       model_name="fake")
        assert r["updated"] == 3
        assert _disk("kb") == (3, 3, True)

    def test_a_collection_without_vectors_still_indexes_lexical_only(
            self, home, docs):
        c = Collection("plain")
        c.create()
        assert c.add_paths([docs], force=True, embed_fn=None)["added"] == 3
        assert c.add_paths([docs], force=True, embed_fn=None)["updated"] == 3
        assert _disk("plain") == (3, 0, False)

    def test_a_changed_document_without_force_still_degrades_not_refuses(
            self, home, docs):
        _hybrid("kb", docs)
        (docs / "a.txt").write_text("a brand new text", encoding="utf-8")
        r = Collection("kb").add_paths([docs], embed_fn=_Flaky(1), model_name="fake")
        assert r["updated"] == 1 and r["skipped"] == 2
        assert _disk("kb")[0] == 3

    def test_force_on_uploads_refuses_too(self, home):
        from localm.rag import EmbeddingLossRefused
        c = Collection("kb")
        c.create()
        c.add_uploads([{"filename": "u.txt", "data": b"upload text here"}],
                      embed_fn=_embed_ok, model_name="fake")
        assert _disk("kb") == (1, 1, True)
        with pytest.raises(EmbeddingLossRefused):
            Collection("kb").add_uploads(
                [{"filename": "u.txt", "data": b"upload text here"}], force=True,
                embed_fn=None)
        assert _disk("kb") == (1, 1, True)

    def test_a_forced_resync_refuses_too(self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        with pytest.raises(EmbeddingLossRefused):
            Collection("kb").resync(force=True, embed_fn=None)
        assert _disk("kb") == (3, 3, True)

    def test_the_collection_object_is_usable_after_a_refusal(self, home, docs):
        from localm.rag import EmbeddingLossRefused
        _hybrid("kb", docs)
        (docs / "a.txt").write_text("a brand new text", encoding="utf-8")
        c = Collection("kb")
        with pytest.raises(EmbeddingLossRefused):
            c.add_paths([docs], force=True, embed_fn=_Flaky(2), model_name="fake")
        on_disk = Collection("kb", cache=False)
        assert c._chunks == on_disk._chunks and c._vectors == on_disk._vectors
        assert all("brand new" not in ch["text"] for ch in c._chunks)


# --------------------------------------------------------------------------- #
#  The CLI                                                                     #
# --------------------------------------------------------------------------- #

@pytest.fixture
def ragcli():
    from localm.cli import rag as ragcli
    return ragcli


@pytest.fixture
def embedder(ragcli, monkeypatch):
    """The embedder `--embed` and the no-terminal path reach; the test sets
    ``state["fn"]``."""
    state = {"fn": _embed_ok}
    monkeypatch.setattr(ragcli, "_cli_rag_embed_fn",
                        lambda url: (lambda texts: state["fn"](texts)))
    return state


class TestCliRefusesToLoseVectors:
    def test_add_force_with_no_terminal_and_no_embedder_refuses(
            self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(ragcli.rag_group, ["add", "kb", str(docs), "--force"])
        assert r.exit_code == 1, r.output
        assert "nothing was changed" in r.output
        assert "setup-embeddings" in r.output
        assert _disk("kb") == (3, 3, True)

    def test_add_force_embed_with_no_embedder_refuses(
            self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(
            ragcli.rag_group, ["add", "kb", str(docs), "--force", "--embed"])
        assert r.exit_code == 1, r.output
        assert _disk("kb") == (3, 3, True)

    def test_repair_with_no_embedder_refuses(self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(ragcli.rag_group, ["repair", "kb"])
        assert r.exit_code == 1, r.output
        assert "nothing was changed" in r.output
        assert _disk("kb") == (3, 3, True)

    def test_repair_embed_that_dies_half_way_leaves_the_old_vectors(
            self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(2)
        r = CliRunner().invoke(ragcli.rag_group, ["repair", "kb", "--embed"])
        assert r.exit_code == 1, r.output
        assert _disk("kb") == (3, 3, True)

    def test_yes_without_embed_is_the_chosen_loss(self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(
            ragcli.rag_group, ["add", "kb", str(docs), "--force", "--yes"])
        assert r.exit_code == 0, r.output
        assert _disk("kb") == (3, 0, False)

    def test_repair_yes_is_the_chosen_loss(self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(ragcli.rag_group, ["repair", "kb", "--yes"])
        assert r.exit_code == 0, r.output
        assert _disk("kb") == (3, 0, False)

    def test_an_answered_yes_is_the_chosen_loss(self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(
            ragcli.rag_group, ["add", "kb", str(docs), "--force"], input="y\n")
        assert r.exit_code == 0, r.output
        assert _disk("kb") == (3, 0, False)

    def test_yes_with_embed_does_not_choose_the_loss(
            self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        embedder["fn"] = _Flaky(1)
        r = CliRunner().invoke(
            ragcli.rag_group, ["repair", "kb", "--embed", "--yes"])
        assert r.exit_code == 1, r.output
        assert _disk("kb") == (3, 3, True)

    def test_a_working_embedder_still_repairs(self, home, docs, ragcli, embedder):
        _hybrid("kb", docs)
        r = CliRunner().invoke(ragcli.rag_group, ["repair", "kb"])
        assert r.exit_code == 0, r.output
        assert _disk("kb") == (3, 3, True)


# --------------------------------------------------------------------------- #
#  The scheduled re-sync                                                       #
# --------------------------------------------------------------------------- #

def _rag_job(collection="kb"):
    from localm.plugins.builtin.jobs.store import Job
    return Job(name="sync-kb", task_kind="rag", collection=collection,
               schedule_kind="interval", schedule=3600)


class TestScheduledResyncOutput:
    def test_a_document_still_missing_on_the_next_run_is_still_listed(
            self, home, docs, monkeypatch):
        from localm.plugins.builtin.jobs import runner
        monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)
        c = Collection("kb")
        c.create()
        c.add_paths([docs])
        (docs / "b.txt").unlink()
        first = runner.run_job(_rag_job(), engine=None)
        second = runner.run_job(_rag_job(), engine=None)
        for out in (first["output"], second["output"]):
            assert "no longer on disk" in out
            assert "b.txt" in out
        assert "still" in second["output"].lower()

    def test_the_listing_is_capped_and_says_how_many_more(
            self, home, docs, monkeypatch):
        from localm.plugins.builtin.jobs import runner
        monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)
        for i in range(12):
            (docs / f"m{i:02}.txt").write_text(f"gone soon {i}", encoding="utf-8")
        c = Collection("kb")
        c.create()
        c.add_paths([docs])
        for i in range(12):
            (docs / f"m{i:02}.txt").unlink()
        runner.run_job(_rag_job(), engine=None)
        out = runner.run_job(_rag_job(), engine=None)["output"]
        assert "12 document(s) are no longer on disk" in out
        assert "and 2 more" in out

    def test_the_summary_says_one_chunk(self, home, tmp_path, monkeypatch):
        from localm.plugins.builtin.jobs import runner
        monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)
        d = tmp_path / "one"
        d.mkdir()
        (d / "x.txt").write_text("single chunk", encoding="utf-8")
        c = Collection("kb")
        c.create()
        c.add_paths([d])
        out = runner.run_job(_rag_job(), engine=None)["output"]
        assert "1 chunk over 1 folder(s)" in out


# --------------------------------------------------------------------------- #
#  The Knowledge routes                                                        #
# --------------------------------------------------------------------------- #

@pytest.fixture
def app(home, tmp_path, monkeypatch):
    from localm.jobmanager import JobManager
    from localm.plugins.engine import PluginManager
    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    app.state.jobs = JobManager()
    return app


@pytest.fixture
def route_embedder(monkeypatch):
    """What the routes' ``_self_services`` hands out as the embedder."""
    from localm.plugins.builtin.rag import plug
    state = {"fn": None}
    real = plug._self_services

    def _services(request):
        _e, classify, describe = real(request)
        fn = state["fn"]
        return ((lambda texts: fn(texts)) if fn else None), classify, describe

    monkeypatch.setattr(plug, "_self_services", _services)
    return state


def _await(app, job_id, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = app.state.jobs.get(job_id)
        assert job is not None
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def _lines(job):
    return "\n".join(e.get("text", "") for e in job._history if e.get("type") == "line")


class TestAddAndUploadRoutesAskBeforeDroppingVectors:
    def test_reindex_without_embed_is_a_409_until_confirmed(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": False, "reindex": True})
            assert r.status_code == 409, r.text
            body = r.json()
            assert body["needs_confirm"] is True
            assert body["reason"] == "drops_embeddings"
            assert "job_id" not in body
            assert _disk("kb") == (3, 3, True)
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": False, "reindex": True,
                             "confirm": True})
            assert r.status_code == 200, r.text
            job = _await(app, r.json()["job_id"])
            assert job.status == "done"
            assert _disk("kb") == (3, 0, False)

    def test_reindex_with_no_embedder_available_is_a_409_too(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": True, "reindex": True})
            assert r.status_code == 409, r.text
            assert _disk("kb") == (3, 3, True)

    def test_a_plain_add_without_embed_is_unaffected(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": False})
            assert r.status_code == 200, r.text
            assert _await(app, r.json()["job_id"]).status == "done"
            assert _disk("kb") == (3, 3, True)

    def test_reindex_with_a_working_embedder_needs_no_confirmation(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        route_embedder["fn"] = _embed_ok
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": True, "reindex": True})
            assert r.status_code == 200, r.text
            assert _await(app, r.json()["job_id"]).status == "done"
            assert _disk("kb") == (3, 3, True)

    def test_an_embedder_that_dies_half_way_ends_the_job_with_the_vectors_kept(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        route_embedder["fn"] = _Flaky(2)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(docs)], "embed": True, "reindex": True})
            assert r.status_code == 200, r.text
            job = _await(app, r.json()["job_id"])
            assert job.status != "done"
            assert "nothing was changed" in _lines(job)
            assert "confirm: true" in _lines(job)
            assert _disk("kb") == (3, 3, True)

    def test_upload_reindex_without_embed_is_a_409_until_confirmed(
            self, app, home, route_embedder):
        c0 = Collection("kb")
        c0.create()
        c0.add_uploads([{"filename": "u.txt", "data": b"upload text here"}],
                       embed_fn=_embed_ok, model_name="fake")
        body = {"files": [{"filename": "u.txt", "content_b64": "dXBsb2FkIHRleHQgaGVyZQ=="}],
                "embed": False, "reindex": True}
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/upload", json=body)
            assert r.status_code == 409, r.text
            assert r.json()["needs_confirm"] is True
            assert _disk("kb") == (1, 1, True)
            r = c.post("/api/rag/collections/kb/upload", json={**body, "confirm": True})
            assert r.status_code == 200, r.text
            assert _await(app, r.json()["job_id"]).status == "done"
            assert _disk("kb") == (1, 0, False)


class TestRepairRoute:
    def test_vanished_sources_are_named(self, app, home, docs, route_embedder):
        c0 = Collection("kb")
        c0.create()
        c0.add_paths([docs])
        (docs / "b.txt").unlink()
        (docs / "c.txt").unlink()
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/repair", json={"embed": False})
            assert r.status_code == 200, r.text
            job = _await(app, r.json()["job_id"])
            text = _lines(job)
            assert job.status == "done"
            assert "2 source(s) no longer exist on disk" in text
            assert "b.txt" in text and "c.txt" in text
            assert Collection("kb").stats()["n_chunks"] == 3

    def test_an_embedder_that_dies_half_way_keeps_the_vectors(
            self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        route_embedder["fn"] = _Flaky(2)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            job = _await(app, r.json()["job_id"])
            assert job.status != "done"
            assert "nothing was changed" in _lines(job)
            assert _disk("kb") == (3, 3, True)

    def test_confirm_true_is_the_chosen_loss(self, app, home, docs, route_embedder):
        _hybrid("kb", docs)
        route_embedder["fn"] = _Flaky(2)
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/repair", json={"confirm": True})
            assert _await(app, r.json()["job_id"]).status == "done"
            assert _disk("kb")[0] == 3

    def test_the_summary_says_one_chunk(self, app, home, tmp_path, route_embedder):
        d = tmp_path / "one"
        d.mkdir()
        (d / "x.txt").write_text("single chunk", encoding="utf-8")
        c0 = Collection("kb")
        c0.create()
        c0.add_paths([d])
        with TestClient(app) as c:
            r = c.post("/api/rag/collections/kb/repair", json={"embed": False})
            assert "- 1 chunk total" in _lines(_await(app, r.json()["job_id"]))
            r = c.post("/api/rag/collections/kb/add",
                       json={"paths": [str(d)], "embed": False})
            assert "- 1 chunk total" in _lines(_await(app, r.json()["job_id"]))
            Collection("kb2").create()
            r = c.post("/api/rag/collections/kb2/upload",
                       json={"files": [{"filename": "u.txt",
                                        "content_b64": "dXBsb2Fk"}],
                             "embed": False})
            assert "- 1 chunk total" in _lines(_await(app, r.json()["job_id"]))


# --------------------------------------------------------------------------- #
#  The wording                                                                 #
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).resolve().parent.parent
_PLURAL_SOURCES = [
    "localm/plugins/builtin/rag/plug.py",
    "localm/plugins/builtin/jobs/runner.py",
    "localm/cli/rag.py",
]


@pytest.mark.parametrize("rel", _PLURAL_SOURCES)
def test_no_count_is_followed_by_a_bare_plural_noun(rel):
    """``{n} chunks`` reads "1 chunks": every count goes through count_noun."""
    text = (_ROOT / rel).read_text(encoding="utf-8")
    bad = [m.group(0) for m in re.finditer(
        r"\}\s+(?:chunks|docs)\b", text)]
    assert not bad, f"{rel}: use count_noun() for {bad}"
