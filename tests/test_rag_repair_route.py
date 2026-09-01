# SPDX-License-Identifier: AGPL-3.0-or-later
"""HTTP route POST /api/rag/collections/{name}/repair.

Mirrors the CLI command (add with force=True from coll.documents()),
job-backed and collection-locked like add/upload/reembed (Collection.add_paths
takes both locks itself), and keeps two behaviours the CLI has:

  - the embeddings-loss guard (cli/rag.py's --embed/--yes prompt), here as a
    needs_confirm dry-run response instead of a job, mirroring
    rag_embedding_set's own two-step confirm shape;
  - refusing honestly instead of running a "repaired: 0 re-indexed" no-op
    when a collection has nothing rebuildable because every document was
    added via /upload (the uploaded bytes are never retained - see
    Collection.add_uploads' own docstring).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localm.rag import Collection


@pytest.fixture
def repair_app(tmp_path, monkeypatch):
    """A headless rag app (no attach_gui, so no self_url/active_model
    published). self_embed is therefore None by default, which is what the
    needs_confirm / no-embedder-available tests need; tests that need an
    embedder available build vectors directly through the Collection
    primitive."""
    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.jobs import JobManager
    home = tmp_path / "userhome"
    home.mkdir()
    localm = home / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")
    app.state.jobs = JobManager()
    return app, home


def _await_job(app, job_id, timeout=30.0):
    jobs = app.state.jobs
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        assert job is not None, f"job {job_id} vanished from the registry"
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


class TestRepairRefusesHonestly:
    def test_no_indexed_documents_400(self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 400, r.text
            assert "no indexed documents" in r.text.lower()

    def test_corrupt_with_no_documents_names_that_specifically(self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            (home / ".localm" / "rag" / "kb" / "meta.json").write_text(
                "{not valid json", encoding="utf-8")
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 400, r.text
            assert "corrupt" in r.text.lower()

    def test_all_upload_only_refuses_instead_of_a_noop_job(self, repair_app):
        """A collection built entirely from uploads has no server-side source
        add_paths(force=True) could rebuild from: it must refuse with an honest
        reason, never start a job that would silently touch nothing and report
        success."""
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            Collection("kb").add_uploads(
                [{"filename": "notes.txt", "data": b"upload only content"}])
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 400, r.text
            assert "upload" in r.text.lower()
            assert "job_id" not in r.json()


class TestRepairRebuildsFiles:
    def test_rebuilds_a_damaged_file_backed_collection(self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = home / "doc.txt"
            target.write_text("alpha beta gamma about turbines", encoding="utf-8")
            Collection("kb").add_paths([target])

            rag_dir = home / ".localm" / "rag" / "kb"
            with (rag_dir / "chunks.jsonl").open("a", encoding="utf-8") as f:
                f.write("\nnot json at all")
            assert Collection("kb").stats()["chunks_bad_lines"] == 1, (
                "fixture precondition")

            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            job = _await_job(app, r.json()["job_id"])
            assert job.status == "done", f"job ended {job.status}"

            healed = Collection("kb")
            assert healed.stats()["chunks_bad_lines"] == 0
            assert healed.stats()["corrupt"] is False

    def test_mixed_collection_rebuilds_files_and_names_the_upload_only_ones(
            self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = home / "doc.txt"
            target.write_text("alpha beta gamma about turbines", encoding="utf-8")
            coll = Collection("kb")
            coll.add_paths([target])
            coll.add_uploads(
                [{"filename": "notes.txt", "data": b"upload only content"}])
            assert Collection("kb").stats()["n_docs"] == 2, "fixture precondition"

            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            job = _await_job(app, r.json()["job_id"])
            assert job.status == "done", f"job ended {job.status}"
            log_text = "\n".join(
                e.get("text", "") for e in job._history if e.get("type") == "line")
            assert "1" in log_text and "upload" in log_text.lower(), (
                f"the job log must name the upload-only document it could not "
                f"rebuild, not just silently skip it: {log_text!r}")
            # Both documents are still present - the upload-only one was left
            # as-is, not dropped.
            assert Collection("kb").stats()["n_docs"] == 2


class TestRepairEmbeddingsLossGuard:
    """Mirrors cli/rag.py's --embed/--yes prompt: repairing without an
    embedder available would silently drop an existing hybrid collection back
    to BM25-only. The route answers with needs_confirm instead of a prompt."""

    def _hybrid_collection(self, home):
        target = home / "doc.txt"
        target.write_text("alpha beta gamma about turbines", encoding="utf-8")
        coll = Collection("kb")
        coll.add_paths([target], embed_fn=lambda ts: [[1.0, 0.0, 0.0] for _ in ts])
        assert coll.stats()["has_vectors"] is True, "fixture precondition"
        return coll

    def test_needs_confirm_when_no_embedder_available_and_has_vectors(
            self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            self._hybrid_collection(home)
            # This app never published self_url/active_model, so the route's
            # own self_embed is None regardless of req.embed's default True.
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["needs_confirm"] is True
            assert "job_id" not in data
            assert "embeddings" in data["detail"].lower()
            # Nothing was touched - still hybrid, still 1 doc.
            assert Collection("kb").stats()["has_vectors"] is True

    def test_confirm_true_proceeds_and_drops_embeddings(self, repair_app):
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            self._hybrid_collection(home)
            r = c.post("/api/rag/collections/kb/repair", json={"confirm": True})
            assert r.status_code == 200, r.text
            job = _await_job(app, r.json()["job_id"])
            assert job.status == "done", f"job ended {job.status}"
            assert Collection("kb").stats()["has_vectors"] is False, (
                "an explicit confirm means the user accepted the drop")

    def test_needs_confirm_when_gui_shell_attached_but_no_embedder_installed(
            self, repair_app):
        """The GUI-reachable case: self_url/active_model published exactly as
        attach_gui does (a GUI shell IS attached), but this fresh LOCALM_HOME
        has no embedding model on disk. _self_services must withhold
        self_embed here too, not only when no shell is attached at all -
        otherwise this guard is unreachable from the GUI's own repair click."""
        app, home = repair_app
        app.state.self_url = "http://127.0.0.1:0/v1"
        app.state.active_model = lambda: "dummy-chat-model"
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            self._hybrid_collection(home)
            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["needs_confirm"] is True
            assert "job_id" not in data
            assert Collection("kb").stats()["has_vectors"] is True

    def test_no_confirm_needed_when_collection_has_no_vectors(self, repair_app):
        """Nothing at risk (BM25-only already): no confirm is requested. The
        CLI's own guard is gated on coll.stats().get('has_vectors') the same
        way."""
        app, home = repair_app
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            target = home / "doc.txt"
            target.write_text("alpha beta gamma about turbines", encoding="utf-8")
            Collection("kb").add_paths([target])   # no embed_fn: BM25-only

            r = c.post("/api/rag/collections/kb/repair", json={})
            assert r.status_code == 200, r.text
            assert "job_id" in r.json(), (
                "a BM25-only collection has nothing to lose and must run "
                "the job directly, not ask for confirmation")
