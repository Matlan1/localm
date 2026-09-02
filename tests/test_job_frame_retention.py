# SPDX-License-Identifier: AGPL-3.0-or-later
"""A live-view frame must not accumulate in a job's replay history.

Every other event a job pushes is small and worth replaying, so ``push`` keeps
them all (bounded at 10,000) and hands the whole backlog to each new subscriber.
A screencast frame is neither: it is a base64 image arriving several times a
second, so retaining them would hold hundreds of megabytes for the life of the
job, and replaying them would hand a late viewer thousands of stale pictures
before the current one.

Only the most recent frame is kept, and a new subscriber gets exactly that -
the same shape ``_last_progress`` already uses.
"""

from __future__ import annotations

import asyncio

from localm.plugins.gui.jobs import FRAME_EVENT, Job


def _job() -> Job:
    return Job(id="j-frames", kind="browser", argv=[])


def _frame(n: int) -> dict:
    return {"type": FRAME_EVENT, "data": "x" * 64, "seq": n}


class TestFramesDoNotAccumulate:
    def test_many_frames_leave_the_history_empty(self):
        job = _job()
        for i in range(500):
            job.push(_frame(i))
        assert list(job._history) == [], (
            "frames entered the replay history: %d entries" % len(job._history))

    def test_only_the_latest_frame_is_retained(self):
        job = _job()
        for i in range(5):
            job.push(_frame(i))
        assert job._last_frame is not None
        assert job._last_frame["seq"] == 4

    def test_ordinary_events_still_accumulate(self):
        """The control: without it, an implementation that dropped EVERY event
        would pass the two tests above."""
        job = _job()
        for i in range(5):
            job.push({"type": "log", "line": str(i)})
        assert len(job._history) == 5

    def test_progress_is_still_latched(self):
        job = _job()
        job.push({"type": "progress", "pct": 12.5})
        assert job._last_progress is not None
        assert job._last_progress["pct"] == 12.5
        assert len(job._history) == 1


class TestANewSubscriberGetsTheCurrentFrame:
    def _drain(self, q) -> list:
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    def test_the_backlog_carries_the_latest_frame_and_no_older_one(self):
        async def run():
            job = _job()
            job.push({"type": "log", "line": "start"})
            for i in range(200):
                job.push(_frame(i))
            q = job.subscribe()
            return self._drain(q)
        events = asyncio.run(run())
        frames = [e for e in events if e.get("type") == FRAME_EVENT]
        assert len(frames) == 1, "a late subscriber was replayed %d frames" % len(frames)
        assert frames[0]["seq"] == 199, frames[0]
        assert any(e.get("type") == "log" for e in events), \
            "the ordinary backlog must still be replayed"

    def test_no_frame_yet_means_no_frame_in_the_backlog(self):
        async def run():
            job = _job()
            job.push({"type": "log", "line": "only"})
            return self._drain(job.subscribe())
        events = asyncio.run(run())
        assert [e.get("type") for e in events] == ["log"]

    def test_a_frame_pushed_after_subscribing_still_arrives_live(self):
        """Retention must not cost delivery: a subscriber present at push time
        receives every frame, not just the last."""
        async def run():
            job = _job()
            q = job.subscribe()
            for i in range(3):
                job.push(_frame(i))
            await asyncio.sleep(0)
            return self._drain(q)
        events = asyncio.run(run())
        seqs = [e["seq"] for e in events if e.get("type") == FRAME_EVENT]
        assert seqs == [0, 1, 2], seqs
