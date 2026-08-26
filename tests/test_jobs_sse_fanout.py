# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI job-progress SSE must fan out to every concurrent viewer independently.

A job's event queue must not be a single ``queue.Queue`` shared by every reader of
``/api/jobs/{id}/events``. ``queue.Queue`` is single-consumer: two concurrent SSE
connections to the same job id split its events between them - whichever reader
happens to be waiting when an item is pushed gets it, the other does not, so one
reader can receive every event including the terminating 'end' frame while the
other receives none and hangs with no 'end' frame to break its read loop on. That
is ordinary use: reload the page during a model pull/comfy-setup/image-gen job, or
open the same job in two tabs/devices.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from localm.plugins.gui.jobs import Job
from localm.plugins.gui.web import attach_gui


@pytest.fixture
def app(tmp_path, monkeypatch):
    home = tmp_path / ".localm"
    monkeypatch.setenv("LOCALM_HOME", str(home))
    monkeypatch.delenv("LOCALM_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    import localm.config as _cfg
    monkeypatch.setattr(_cfg, "HOME_DIR", home)
    monkeypatch.setattr(_cfg, "MODELS_DIR", home / "models")
    monkeypatch.setattr(_cfg, "CONFIG_FILE", home / "config.json")
    monkeypatch.setattr(_cfg, "REGISTRY_FILE", home / "registry.json")
    from localm.plugins.engine import attach_engine
    a = FastAPI()
    attach_engine(a)
    attach_gui(a, self_url="http://127.0.0.1:9/v1",
               switch_model=lambda n: None, active_model=lambda: "model-a")
    return a


def _line_events(n):
    return [{"type": "line", "text": f"line {i}"} for i in range(n)]


async def _read_events(client, job_id):
    events = []
    async with client.stream("GET", f"/api/jobs/{job_id}/events") as r:
        assert r.status_code == 200
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            events.append(json.loads(line[len("data: "):]))
            if events[-1].get("type") == "end":
                break
    return events


@pytest.mark.anyio
async def test_two_concurrent_readers_each_get_the_full_preloaded_stream(app):
    """Events are already queued up (as if two clients open the same in-flight
    job) before either connects."""
    job = Job(id=uuid.uuid4().hex[:12], kind="test", argv=[])
    app.state.jobs._jobs[job.id] = job
    for ev in _line_events(20):
        job.push(ev)
    job.push({"type": "end", "status": "done"})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events1, events2 = await asyncio.wait_for(
            asyncio.gather(_read_events(client, job.id), _read_events(client, job.id)),
            timeout=10,
        )

    for events in (events1, events2):
        assert [e for e in events if e["type"] == "line"] == _line_events(20)
        assert events[-1] == {"type": "end", "status": "done"}


@pytest.mark.anyio
async def test_two_concurrent_readers_each_get_the_full_live_stream(app):
    """The realistic case: both viewers are already connected (e.g. two open
    tabs) when the job's live events arrive, not just replayed backlog."""
    job = Job(id=uuid.uuid4().hex[:12], kind="test", argv=[])
    app.state.jobs._jobs[job.id] = job

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        t1 = asyncio.create_task(_read_events(client, job.id))
        t2 = asyncio.create_task(_read_events(client, job.id))

        deadline = asyncio.get_event_loop().time() + 5
        while job.subscriber_count < 2:
            if asyncio.get_event_loop().time() > deadline:
                raise AssertionError("both readers never subscribed to the job")
            await asyncio.sleep(0.01)

        for ev in _line_events(20):
            job.push(ev)
        job.push({"type": "end", "status": "done"})

        events1, events2 = await asyncio.wait_for(asyncio.gather(t1, t2), timeout=10)

    for events in (events1, events2):
        assert [e for e in events if e["type"] == "line"] == _line_events(20)
        assert events[-1] == {"type": "end", "status": "done"}

    assert job.subscriber_count == 0   # both connections unsubscribed on disconnect
