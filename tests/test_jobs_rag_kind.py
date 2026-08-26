# SPDX-License-Identifier: AGPL-3.0-or-later
"""The `rag` scheduled-job kind: a folder re-sync you can put on a schedule.

Covers the job DEFINITION side (it is expressible, validated, persisted, and
survives a scheduler round-trip) and the RUNNER side (it drives the real
Collection.resync against a real folder, with the confinement policy applied and
no chat model loaded).

Everything that touches disk points LOCALM_HOME at a tmp dir and patches the
config module, so nothing touches the user's real data.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALM_HOME", str(tmp_path))
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", tmp_path)
    monkeypatch.setattr(cfg, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", tmp_path / "registry.json")
    return tmp_path


@pytest.fixture
def docs(tmp_path, home):
    """An indexable folder, plus a config that allows indexing it.

    The default policy is whitelist (home folder + cwd + configured roots), and
    a pytest tmp dir is under neither on most boxes - so without this every
    runner test would be asserting against a BLOCKED root by accident. The two
    policy tests below set their own config explicitly.
    """
    d = tmp_path / "papers"
    d.mkdir()
    (d / "one.txt").write_text("the first paper is about turbines", encoding="utf-8")
    _write_config(home, {"rag_indexing_mode": "blacklist", "rag_denied_roots": []})
    return d


def _write_config(home, data: dict) -> None:
    import json
    (home / "config.json").write_text(json.dumps(data), encoding="utf-8")


def _rag_job(**kw):
    from localm.plugins.builtin.jobs.store import Job
    base = dict(name="sync-kb", task_kind="rag", collection="kb",
                schedule_kind="interval", schedule=3600)
    base.update(kw)
    return Job(**base)


# --------------------------------------------------------------------------- #
#  The job definition                                                          #
# --------------------------------------------------------------------------- #

def test_rag_is_an_accepted_task_kind(home):
    from localm.plugins.builtin.jobs.store import TASK_KINDS
    assert "rag" in TASK_KINDS
    job = _rag_job()
    assert job.task_kind == "rag" and job.collection == "kb"


def test_a_rag_job_needs_no_prompt(home):
    """It re-syncs a named collection against folders it already knows, so it is
    fully specified without one (same as the memory kind)."""
    job = _rag_job(prompt="")
    assert job.prompt == ""


def test_a_rag_job_without_a_collection_is_refused(home):
    with pytest.raises(ValueError, match="collection"):
        _rag_job(collection=None)
    with pytest.raises(ValueError, match="collection"):
        _rag_job(collection="   ")


def test_a_bad_collection_name_is_refused_at_definition_time(home):
    """A typo must fail when the job is CREATED, not silently on every
    unattended tick."""
    with pytest.raises(ValueError):
        _rag_job(collection="../escape")
    with pytest.raises(ValueError):
        _rag_job(collection="has spaces")


def test_other_kinds_still_require_a_prompt(home):
    """Negative control: relaxing the prompt rule for rag must not relax it for
    chat/coder."""
    from localm.plugins.builtin.jobs.store import Job
    with pytest.raises(ValueError, match="prompt"):
        Job(name="x", task_kind="chat", prompt="")
    with pytest.raises(ValueError, match="prompt"):
        Job(name="x", task_kind="coder", prompt="", cwd=".")


def test_a_rag_job_round_trips_through_the_store(home):
    from localm.plugins.builtin.jobs.store import JobStore
    store = JobStore()
    job = store.add(_rag_job(name="nightly-kb", collection="manuals"))

    # A fresh store instance reads it back off disk.
    got = JobStore().get(job.id)
    assert got is not None
    assert got.task_kind == "rag"
    assert got.collection == "manuals"
    assert got.prompt == ""


def test_a_rag_job_survives_a_scheduler_round_trip(home):
    """Persisted -> listed -> found due -> run -> result recorded."""
    from localm.plugins.builtin.jobs.scheduler import JobScheduler
    from localm.plugins.builtin.jobs.store import JobStore

    store = JobStore()
    job = store.add(_rag_job(schedule_kind="interval", schedule=60))
    seen: list = []

    def fake_run(j, *, engine=None):
        seen.append((j.task_kind, j.collection))
        return {"status": "ok", "output": "re-synced", "error": None,
                "started": 0.0, "finished": 1.0}

    sched = JobScheduler(store, run_job=fake_run)
    assert sched.due(job, now=1000.0) is True          # never run -> due
    assert sched.tick(now=1000.0) == [job.id]
    assert seen == [("rag", "kb")]

    after = JobStore().get(job.id)
    assert after.last_status == "ok"
    assert sched.due(after, now=after.last_run + 5) is False    # not due again
    assert sched.due(after, now=after.last_run + 61) is True


def test_a_rag_job_is_creatable_over_the_api(home, monkeypatch):
    """Through the real plugin engine (open mode, no key), mirroring the jobs
    plugin's own route test."""
    from pathlib import Path

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from localm.plugins.engine import PluginManager

    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.delenv("LOCALM_REQUIRE_AUTH", raising=False)

    store_root = Path(__file__).resolve().parents[1] / "localm" / "plugins" / "builtin"
    app = FastAPI()
    PluginManager(app, store_root=store_root,
                  installed_root=home / "plugins").install("jobs")

    with TestClient(app) as client:
        r = client.post("/api/jobs", json={
            "name": "kb-sync", "task_kind": "rag", "collection": "kb",
            "schedule_kind": "cron", "schedule": "0 3 * * *"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["task_kind"] == "rag" and body["collection"] == "kb"
        # No prompt was sent at all: a rag job must not need one.
        assert body["prompt"] == ""

        # It is really persisted, not just echoed.
        from localm.plugins.builtin.jobs.store import JobStore
        stored = JobStore().get(body["id"])
        assert stored is not None and stored.collection == "kb"

        # And the collection is required, with a 400.
        bad = client.post("/api/jobs", json={
            "name": "kb-sync-2", "task_kind": "rag",
            "schedule_kind": "interval", "schedule": 3600})
        assert bad.status_code == 400
        assert "collection" in bad.json()["detail"]


def test_the_cli_can_add_a_rag_job(home):
    from click.testing import CliRunner
    from localm.plugins.builtin.jobs.cli import main
    from localm.plugins.builtin.jobs.store import JobStore

    res = CliRunner().invoke(
        main, ["add", "kb-sync", "--rag", "--collection", "kb",
               "--cron", "0 3 * * *"])
    assert res.exit_code == 0, res.output
    jobs = JobStore().list()
    assert len(jobs) == 1
    assert (jobs[0].task_kind, jobs[0].collection) == ("rag", "kb")

    # Mutually exclusive with the other kind flags.
    clash = CliRunner().invoke(
        main, ["add", "x", "--rag", "--coder", "--collection", "kb",
               "--cwd", "."])
    assert clash.exit_code == 1
    assert "only one of" in clash.output.lower()


# --------------------------------------------------------------------------- #
#  The runner: it really re-syncs                                              #
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def no_shared_embedder(monkeypatch):
    """Keep the runner tests off the process-wide embedder singleton.

    ``localm.inference.embedder`` caches its embedder (and its load-failure and
    download-attempt latches) in module globals for the whole pytest process, so
    whether a resync here indexes with vectors would depend on what an unrelated
    test did earlier. Default every test in this module to lexical-only; the
    embedding-specific tests below set their own. The real resolution path is
    covered by ``test_rag_embed_fn_is_none_without_an_embedding_model``.

    Yields the ORIGINAL function so that one test can still drive it."""
    from localm.plugins.builtin.jobs import runner
    original = runner._rag_embed_fn
    monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)
    return original


def test_rag_embed_fn_is_none_without_an_embedding_model(home, monkeypatch,
                                                         no_shared_embedder):
    """No embedding model available -> lexical-only, not a crash. It must not
    hand back ``embed_texts``, whose None return ``add_paths`` cannot consume."""
    from localm.inference import embedder
    monkeypatch.setattr(embedder, "get_embedder", lambda: None)
    assert no_shared_embedder() is None


def _collection_with(home, folder, name="kb"):
    from localm.rag import Collection
    coll = Collection(name)
    coll.create()
    coll.add_paths([folder])
    return coll


def test_run_job_picks_up_a_file_added_since_the_index(home, docs):
    """The end-to-end point of the feature, through run_job itself."""
    from localm.plugins.builtin.jobs.runner import run_job
    from localm.rag import Collection

    _collection_with(home, docs)
    (docs / "two.txt").write_text("the second paper is about gearboxes",
                                  encoding="utf-8")

    result = run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "1 added" in result["output"]
    hits = Collection("kb").query("gearboxes", k=3)
    assert hits and "gearboxes" in hits[0]["text"]


def test_run_job_reports_a_deleted_file_without_removing_it(home, docs):
    from localm.plugins.builtin.jobs.runner import run_job
    from localm.rag import Collection

    _collection_with(home, docs)
    (docs / "one.txt").unlink()

    result = run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "no longer on disk" in result["output"]
    assert "FLAGGED, not removed" in result["output"]
    coll = Collection("kb")
    assert coll.stats()["n_missing"] == 1
    assert coll.query("turbines", k=3)          # still searchable


def test_run_job_reports_an_unreachable_folder_instead_of_deleting(home, docs):
    import shutil
    from localm.plugins.builtin.jobs.runner import run_job
    from localm.rag import Collection

    _collection_with(home, docs)
    shutil.rmtree(docs)

    result = run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "skipped folder" in result["output"]
    assert "nothing under it was indexed, flagged, or removed" in result["output"]
    assert Collection("kb").stats()["n_missing"] == 0


def test_run_job_applies_the_indexing_policy(home, docs):
    """A scheduled run must never index what an interactive API add would
    refuse. The folder was legal when it was indexed; the owner has since put it
    on the deny list, so the re-sync must skip it and SAY so - not index it
    because it used to be allowed."""
    from localm.plugins.builtin.jobs.runner import run_job
    from localm.rag import Collection

    _collection_with(home, docs)
    _write_config(home, {"rag_indexing_mode": "blacklist",
                         "rag_denied_roots": [str(docs)]})
    (docs / "two.txt").write_text("a second paper about gearboxes",
                                  encoding="utf-8")

    result = run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "skipped folder" in result["output"]
    assert "denied list" in result["output"]
    coll = Collection("kb")
    assert not any(s.endswith("two.txt") for s in coll.documents())
    assert coll.stats()["n_missing"] == 0      # blocked is not a delete verdict


def test_the_same_run_indexes_once_the_folder_is_allowed(home, docs):
    """Negative control for the policy test: with the deny lifted, the identical
    run picks the new file up - so the skip really was the policy."""
    from localm.plugins.builtin.jobs.runner import run_job
    from localm.rag import Collection

    _collection_with(home, docs)
    _write_config(home, {"rag_indexing_mode": "blacklist", "rag_denied_roots": []})
    (docs / "two.txt").write_text("a second paper about gearboxes",
                                  encoding="utf-8")

    result = run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "skipped folder" not in result["output"]
    assert "1 added" in result["output"]
    assert any(s.endswith("two.txt") for s in Collection("kb").documents())


def test_run_job_errors_cleanly_on_an_unknown_collection(home):
    from localm.plugins.builtin.jobs.runner import run_job
    result = run_job(_rag_job(collection="nope"), engine=None)
    assert result["status"] == "error"
    assert "no such collection" in result["error"].lower()


def test_a_rag_job_never_loads_a_chat_engine(home, docs, monkeypatch):
    """A folder re-sync needs no chat model; loading one would evict the user's
    live model for nothing."""
    from localm.plugins.builtin.jobs import runner

    _collection_with(home, docs)
    called: list = []

    def _boom(model):
        called.append(model)
        raise AssertionError("a rag job must not load a chat engine")

    monkeypatch.setattr(runner, "_load_engine", _boom)
    result = runner.run_job(_rag_job(collection="kb"), engine=None)
    assert result["status"] == "ok", result.get("error")
    assert called == []


def test_run_job_flags_lost_embedding_coverage(home, docs, monkeypatch):
    """A re-sync that indexes new documents lexical-only into a collection that
    HAS vectors degrades semantic search. That must be said, not implied."""
    from localm.plugins.builtin.jobs import runner
    from localm.rag import Collection

    coll = Collection("kb")
    coll.create()
    coll.add_paths([docs], embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    assert coll.stats()["has_vectors"]

    (docs / "two.txt").write_text("a second paper about gearboxes",
                                  encoding="utf-8")
    monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)

    result = runner.run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "no embedding model was available" in result["output"]
    assert "localm setup-embeddings" in result["output"]


def test_no_embedding_note_when_the_collection_never_had_vectors(home, docs,
                                                                 monkeypatch):
    """Negative control: a lexical-only collection loses nothing, so the warning
    must not fire (a warning that always fires teaches people to ignore it)."""
    from localm.plugins.builtin.jobs import runner

    _collection_with(home, docs)
    (docs / "two.txt").write_text("a second paper", encoding="utf-8")
    monkeypatch.setattr(runner, "_rag_embed_fn", lambda: None)

    result = runner.run_job(_rag_job(collection="kb"), engine=None)

    assert result["status"] == "ok", result.get("error")
    assert "no embedding model was available" not in result["output"]
