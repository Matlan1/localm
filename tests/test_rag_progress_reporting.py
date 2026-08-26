# SPDX-License-Identifier: AGPL-3.0-or-later
"""rag-reembed and rag-upload report STRUCTURED progress (``Job.progress``)
instead of prose (reembed) or nothing (upload).

``Collection.reembed``'s batch loop computes the exact ``n/total`` for every
batch. It passes ``phase``/``done``/``total``/``unit`` as keywords on the SAME
``on_progress`` call that carries the line text, so ``plug._job_progress``
forwards the identical numbers the line was built from rather than deriving
them a second time.

``POST /api/rag/collections/{name}/upload`` knows the file count and the total
DECODED byte size before any indexing work starts - the whole request is
already base64-decoded by the time the job function runs - and reports both as
one t=0 progress event, so a client sees a real denominator immediately instead
of silence until the first file finishes.

Neither call site computes a percentage itself; ``Job.progress`` is the only
place that divides, and an unknown total reports ``pct: null`` rather than a
fabricated 0.
"""
from __future__ import annotations

import base64
import time

from localm.plugins.gui.jobs import Job
from localm.rag.store import Collection


def _embedder(dim):
    def embed(texts):
        return [[0.1] * dim for _ in texts]
    return embed


def _collection(tmp_path, texts, dim=8):
    c = Collection("kb", base=tmp_path).create()
    c._chunks = [{"source": f"doc{i}.txt", "pos": i, "text": t}
                 for i, t in enumerate(texts)]
    c._vectors = _embedder(dim)(texts)
    c._save()
    return c


# --------------------------------------------------------------------------- #
#  reembed's loop hands over numbers, not a percentage                        #
# --------------------------------------------------------------------------- #

class TestReembedStructuredProgress:
    def test_batches_report_the_same_numbers_the_line_was_built_from(self, tmp_path):
        c = _collection(tmp_path, [f"chunk {i}" for i in range(5)], dim=8)
        calls = []
        c.reembed(embed_fn=_embedder(16), batch=2,
                  on_progress=lambda text, **kw: calls.append((text, kw)))

        assert len(calls) == 3, "one call per batch: 2, 2, then the last 1 of 5"
        texts, kwargs = zip(*calls)
        assert texts == ("re-embedding 2/5", "re-embedding 4/5", "re-embedding 5/5")
        assert kwargs[0] == {"phase": "re-embedding", "done": 2, "total": 5,
                              "unit": "chunks"}
        assert kwargs[-1] == {"phase": "re-embedding", "done": 5, "total": 5,
                               "unit": "chunks"}

    def test_never_hands_on_progress_a_precomputed_percentage(self, tmp_path):
        """pct is derived in exactly one place (Job.progress). reembed must not
        do that division itself and pass a ``pct`` kwarg along."""
        c = _collection(tmp_path, ["a", "b", "c"], dim=8)
        calls = []
        c.reembed(embed_fn=_embedder(8), on_progress=lambda t, **kw: calls.append(kw))
        assert calls and all("pct" not in kw for kw in calls)

    def test_a_kwargs_tolerant_string_only_sink_still_works(self, tmp_path):
        """Any sink reused for reembed must accept and ignore the new keywords:
        the contract is additive, not a breaking change to on_progress's
        single-string-argument callers elsewhere in store.py."""
        c = _collection(tmp_path, ["a", "b"], dim=8)
        messages = []

        def sink(text, **_ignored):
            messages.append(text)

        c.reembed(embed_fn=_embedder(8), on_progress=sink)
        assert messages == ["re-embedding 2/2"]


# --------------------------------------------------------------------------- #
#  plug._job_progress forwards the structured call to Job.progress            #
# --------------------------------------------------------------------------- #

def _job() -> Job:
    return Job(id="j1", kind="rag-reembed", argv=[])


def _progress_events(job: Job) -> list:
    return [e for e in job._history if e.get("type") == "progress"]


class TestJobProgressWrapperForwardsReembedNumbers:
    def test_structured_call_reaches_job_progress_with_correct_pct(self):
        from localm.plugins.builtin.rag.plug import _job_progress

        job = _job()
        cb = _job_progress(job)
        cb("re-embedding 2/5", phase="re-embedding", done=2, total=5, unit="chunks")

        lines = [e for e in job._history if e.get("type") == "line"]
        assert lines[-1]["text"] == "re-embedding 2/5"
        ev = _progress_events(job)[-1]
        assert (ev["pct"], ev["done"], ev["total"], ev["unit"], ev["phase"]) == (
            40.0, 2, 5, "chunks", "re-embedding")

    def test_plain_text_only_call_does_not_fabricate_a_progress_event(self):
        """add_paths/add_uploads call this SAME wrapper with only a message
        (e.g. "indexed foo.txt (3 chunks)", "embeddings unavailable ..."). That
        must keep producing a line and nothing else - no invented progress
        event with no numbers behind it."""
        from localm.plugins.builtin.rag.plug import _job_progress

        job = _job()
        cb = _job_progress(job)
        cb("indexed foo.txt (3 chunks)")

        assert not _progress_events(job), (
            "a plain-text call must not synthesize a progress event")
        lines = [e for e in job._history if e.get("type") == "line"]
        assert lines[-1]["text"] == "indexed foo.txt (3 chunks)"


# --------------------------------------------------------------------------- #
#  Composed end to end: reembed through a REAL Job, no HTTP needed            #
# --------------------------------------------------------------------------- #

class TestReembedProgressReachesARealJobEndToEnd:
    def test_progress_climbs_to_100_percent_and_lands_in_the_listing(self, tmp_path):
        from localm.plugins.builtin.rag.plug import _job_progress

        c = _collection(tmp_path, [f"chunk {i}" for i in range(4)], dim=8)
        job = _job()

        c.reembed(embed_fn=_embedder(16), batch=1, on_progress=_job_progress(job))

        pcts = [e["pct"] for e in _progress_events(job)]
        assert pcts == [25.0, 50.0, 75.0, 100.0], (
            f"expected a monotonic climb to 100%, got {pcts}")
        summary = job.summary()
        assert summary["pct"] == 100.0
        assert summary["phase"] == "re-embedding"


# --------------------------------------------------------------------------- #
#  /upload reports the known file count + byte total at t=0                   #
# --------------------------------------------------------------------------- #

def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _await_job(app, job_id, timeout=10.0):
    jobs = app.state.jobs
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        assert job is not None, f"job {job_id} vanished from the registry"
        if job.status != "running":
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still running after {timeout}s")


def _upload_app(tmp_path, monkeypatch):
    from fastapi import FastAPI

    from localm.plugins.engine import PluginManager
    from localm.plugins.gui.web import attach_gui

    localm = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(localm))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    import localm.config as cfg
    monkeypatch.setattr(cfg, "HOME_DIR", localm)
    monkeypatch.setattr(cfg, "MODELS_DIR", localm / "models")
    monkeypatch.setattr(cfg, "CONFIG_FILE", localm / "config.json")
    monkeypatch.setattr(cfg, "REGISTRY_FILE", localm / "registry.json")

    app = FastAPI()
    PluginManager(app, external_root=tmp_path / "noplugins").install("rag")

    async def switch_model(name):
        pass
    attach_gui(app, self_url="http://127.0.0.1:9/v1",
               switch_model=switch_model, active_model=lambda: "m")
    return app


class TestUploadReportsKnownTotalsAtJobStart:
    def test_reports_file_count_and_byte_total_before_indexing_starts(
            self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        app = _upload_app(tmp_path, monkeypatch)
        file_a, file_b = b"gfx1030 rocm runtime notes", b"a shorter one"
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [
                    {"filename": "a.md", "content_b64": _b64(file_a)},
                    {"filename": "b.md", "content_b64": _b64(file_b)},
                ],
                "embed": False})
            assert r.status_code == 200, r.text
            job = _await_job(app, r.json()["job_id"])

        assert job.status == "done", f"job ended {job.status}"
        progress = [e for e in job._history if e.get("type") == "progress"]
        assert progress, "no structured progress event was ever pushed"
        first = progress[0]
        assert first["phase"] == "uploading"
        assert first["done"] == 0
        assert first["total"] == 2, "must know the file count up front"
        assert first["total_bytes"] == len(file_a) + len(file_b), (
            "must report the DECODED byte total, not the base64 string length")
        assert first["pct"] == 0.0, (
            "0 of a KNOWN total is a measured zero, not the fabricated-0 case "
            "P5/R1 forbids")
        # add_uploads ticks per item, so the listing ends at 100 and the t=0 event
        # is the first of several. The known file count and the decoded byte total
        # are reported before any indexing starts.
        assert job.summary()["pct"] == 100.0, (
            "the listing did not advance past the t=0 report")

    def test_byte_total_reflects_decoded_size_not_the_base64_length(
            self, tmp_path, monkeypatch):
        """base64 inflates by ~4/3, so reporting the encoded string length
        instead of len(data) would overstate every upload's byte total by
        ~33%."""
        from fastapi.testclient import TestClient

        app = _upload_app(tmp_path, monkeypatch)
        payload = b"x" * 300         # length not a multiple of 3: exercises padding
        with TestClient(app) as c:
            c.post("/api/rag/collections", json={"name": "kb"})
            r = c.post("/api/rag/collections/kb/upload", json={
                "files": [{"filename": "a.bin", "content_b64": _b64(payload)}],
                "embed": False})
            assert r.status_code == 200, r.text
            job = _await_job(app, r.json()["job_id"])

        progress = [e for e in job._history if e.get("type") == "progress"]
        assert progress[0]["total_bytes"] == 300
