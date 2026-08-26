# SPDX-License-Identifier: AGPL-3.0-or-later
"""The upload bar has to MOVE, not just have a denominator: `_add_uploads_locked`
owns the loop and ticks per file.

WHAT THE FIXTURES HAVE TO EXPRESS:

* The loop body has three exits, two of which `continue`. A fixture of only
  indexable files can never reach either, so the mix below is deliberate.
* Asserting the FINAL event cannot distinguish "advanced through the skips" from
  "jumped at the end", so every count assertion here is on the SEQUENCE.
* Most callers pass no `on_progress` at all, so the no-op path is the live one
  and gets its own test - the structured keywords would raise TypeError against
  a one-positional lambda.
"""

import pytest

from localm.rag import store as store_mod


class _Recorder:
    """Captures both channels the way `_job_progress` receives them."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, text, **kw):
        self.calls.append((text, kw))

    @property
    def dones(self):
        return [kw["done"] for _t, kw in self.calls if "done" in kw]

    @property
    def lines(self):
        return [t for t, _kw in self.calls]


@pytest.fixture()
def coll(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "RAG_DIR", tmp_path, raising=False)
    c = store_mod.Collection("p7b")
    c._load()
    return c


def _embed(texts):
    return [[float(len(t)), 1.0] for t in texts]


def _files(n):
    return [{"filename": f"u{i}.txt", "data": f"content number {i}".encode()}
            for i in range(n)]


class TestTheBarMoves:
    def test_every_file_reports_against_a_known_total(self, coll):
        rec = _Recorder()
        coll.add_uploads(_files(4), embed_fn=_embed, on_progress=rec)

        assert rec.dones == [1, 2, 3, 4], (
            f"the upload bar did not advance per file: {rec.calls}")
        totals = {kw["total"] for _t, kw in rec.calls if "total" in kw}
        assert totals == {4}, f"lost the denominator it had at t=0: {totals}"
        assert all(kw["unit"] == "files"
                   for _t, kw in rec.calls if "unit" in kw)

    def test_it_reaches_the_end(self, coll):
        rec = _Recorder()
        coll.add_uploads(_files(3), embed_fn=_embed, on_progress=rec)
        assert rec.dones[-1] == 3, (
            f"never reported the last file finishing: {rec.dones}")


class TestEveryExitFromTheLoopTicks:
    """The three exits, each with its own fixture. A forgotten one is the whole
    risk of the chosen shape, so each is pinned separately AND by the contiguous
    sequence, which is what catches a FOURTH exit added later."""

    def test_a_skipped_duplicate_advances_and_is_no_longer_silent(self, coll):
        payload = [{"filename": "same.txt", "data": b"unchanging"}]
        coll.add_uploads(payload, embed_fn=_embed)          # seed the hash

        rec = _Recorder()
        coll.add_uploads(payload + [{"filename": "new.txt", "data": b"fresh"}],
                         embed_fn=_embed, on_progress=rec)

        assert rec.dones == [1, 2], (
            f"a duplicate skip did not advance the count: {rec.calls}")
        assert any("same.txt" in ln for ln in rec.lines), (
            f"a skipped file was still reported as nothing at all: {rec.lines}")

    def test_an_unextractable_item_advances(self, coll, monkeypatch):
        real = store_mod.extract_bytes

        def _boom(data, filename, **kw):
            if filename == "bad.bin":
                raise store_mod.ExtractError("cannot extract this")
            return real(data, filename, **kw)

        monkeypatch.setattr(store_mod, "extract_bytes", _boom)

        rec = _Recorder()
        result = coll.add_uploads(
            [{"filename": "good.txt", "data": b"fine"},
             {"filename": "bad.bin", "data": b"\x00\x01"},
             {"filename": "ok.txt", "data": b"also fine"}],
            embed_fn=_embed, on_progress=rec)

        assert [f["path"] for f in result["failed"]] == ["upload:bad.bin"], (
            f"the fixture never reached the failure branch: {result}")
        assert rec.dones == [1, 2, 3], (
            f"a failed item did not advance the count: {rec.calls}")

    def test_the_sequence_is_contiguous_with_every_outcome_mixed(self, coll,
                                                                 monkeypatch):
        """All three exits in ONE run. This is the assertion that catches a
        fourth exit being added later without a tick, which no single-outcome
        fixture can."""
        dup = {"filename": "dup.txt", "data": b"seen before"}
        coll.add_uploads([dup], embed_fn=_embed)

        real = store_mod.extract_bytes

        def _boom(data, filename, **kw):
            if filename == "bad.bin":
                raise store_mod.ExtractError("nope")
            return real(data, filename, **kw)

        monkeypatch.setattr(store_mod, "extract_bytes", _boom)

        rec = _Recorder()
        coll.add_uploads(
            [dup,
             {"filename": "bad.bin", "data": b"\x00"},
             {"filename": "fresh.txt", "data": b"brand new"}],
            embed_fn=_embed, on_progress=rec)

        assert rec.dones == [1, 2, 3], (
            f"the sequence skipped an outcome: {rec.calls}")


class TestBothChannelsTravelTogether:
    def test_the_line_and_the_numbers_arrive_on_one_call(self, coll):
        """The prose and the structured numbers ride the SAME call, so they
        cannot drift; this asserts neither is dropped."""
        rec = _Recorder()
        coll.add_uploads(_files(2), embed_fn=_embed, on_progress=rec)

        for text, kw in rec.calls:
            assert text, f"a structured tick carried no line: {kw}"
            assert kw.get("done") is not None, f"a line carried no numbers: {text!r}"

    def test_no_callback_at_all_does_not_raise(self, coll):
        """The live path for almost every caller. The structured keywords would
        hit a one-positional no-op lambda and raise TypeError, turning a progress
        change into a broken upload."""
        out = coll.add_uploads(_files(2), embed_fn=_embed)
        assert out["added"] == 2


class TestItReachesAJobListing:
    def test_a_percentage_lands_where_a_watching_client_reads_it(self, coll):
        """End to end through the real adapter, because the value is the whole
        path: `_job_progress` forwards the same numbers to `Job.progress`, which
        is the only place that divides."""
        from localm.plugins.builtin.rag.plug import _job_progress
        from localm.plugins.gui.jobs import Job

        job = Job(id="j1", kind="rag-upload", argv=[])
        coll.add_uploads(_files(4), embed_fn=_embed,
                         on_progress=_job_progress(job))

        summary = job.summary()
        assert summary["pct"] == 100.0, (
            f"the upload's progress never reached the listing: {summary}")
        pcts = [e["pct"] for e in job._history if e.get("type") == "progress"]
        assert pcts == [25.0, 50.0, 75.0, 100.0], (
            f"the listing saw a frozen or jumping bar: {pcts}")
