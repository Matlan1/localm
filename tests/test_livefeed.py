# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.livefeed: the one sampler behind the GUI's pushed hardware stats.

The meter used to lag a freed model by up to ~5 s because the VRAM reading is
refreshed at most every 10 s and nothing asked for a fresh one when a model went
away. These tests pin the replacement: a kick forces a reading at once (a routine
tick cannot, here, because the fake sampler only reports the truth when asked for
a fresh reading), frames are sent only for changes a tab would display, and the
sampler exists once however many tabs are open and not at all when none is.
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

import pytest

from localm import livefeed
from localm.livefeed import LiveFeed, display_key

GIB = 1024 ** 3


class Sampler:
    """A fake ``sample(fresh)``. ``truth`` is what the GPU holds now; a routine
    reading reports ``shown``, the last value a fresh reading saw, as the real
    throttled one does."""

    def __init__(self, used_gib=10.0, cpu=12.0):
        self.truth = used_gib
        self.shown = used_gib
        self.cpu = cpu
        self.calls: list[bool] = []

    async def __call__(self, fresh):
        self.calls.append(fresh)
        if fresh:
            self.shown = self.truth
        return {"cpu": {"percent": self.cpu},
                "vram": {"used": int(self.shown * GIB), "total": 16 * GIB}}


def run(coro):
    return asyncio.run(coro)


async def until(cond, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        await asyncio.sleep(0.005)
    return bool(cond())


def used_gib(feed):
    return feed.latest["vram"]["used"] / GIB


# ------------------------------------------------------------ the kick

def test_a_kick_delivers_a_reading_the_routine_tick_would_not():
    async def go():
        s = Sampler(used_gib=10.0)
        feed = LiveFeed(s, interval_s=60, follow_ups_s=())
        sub = feed.subscribe()
        try:
            assert await until(lambda: feed.latest is not None)
            assert used_gib(feed) == 10.0
            s.truth = 2.0
            t0 = time.monotonic()
            loop = asyncio.get_running_loop()
            assert await loop.run_in_executor(None, feed.kick, "unload"), \
                "a kick from another thread reaches a connected feed"
            assert await until(lambda: used_gib(feed) == 2.0, timeout=2.0), \
                "the freed VRAM never reached the feed"
            assert time.monotonic() - t0 < 1.0
        finally:
            sub.close()

    run(go())


def test_a_kick_is_followed_by_forced_readings_for_a_late_driver_release():
    async def go():
        s = Sampler(used_gib=10.0)
        inner = s.__call__

        async def sample(fresh):
            reading = await inner(fresh)
            if len(s.calls) == 2:
                s.truth = 2.0
            return reading

        feed = LiveFeed(sample, interval_s=60, follow_ups_s=(0.05, 0.1))
        sub = feed.subscribe()
        try:
            assert await until(lambda: feed.latest is not None)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, feed.kick, "unload")
            assert await until(lambda: len(s.calls) >= 4), s.calls
            assert used_gib(feed) == 2.0, \
                "the kick's own reading saw the old value, so only a follow-up can have"
            assert all(s.calls), "every reading around a kick is forced"
        finally:
            sub.close()

    run(go())


def test_the_follow_ups_are_not_retaken_when_the_kick_reading_ran_past_them():
    async def go():
        s = Sampler()
        slow = {"on": False}
        inner = s.__call__

        async def sample(fresh):
            if slow["on"]:
                await asyncio.sleep(0.3)
            return await inner(fresh)

        feed = LiveFeed(sample, interval_s=60, follow_ups_s=(0.05, 0.1, 5.0))
        sub = feed.subscribe()
        try:
            assert await until(lambda: len(s.calls) == 1)
            slow["on"] = True
            await asyncio.get_running_loop().run_in_executor(None, feed.kick, "x")
            assert await until(lambda: len(s.calls) == 2)
            await asyncio.sleep(0.5)
            assert len(s.calls) == 2, \
                "follow-ups already past when the reading finished were taken again"
        finally:
            sub.close()

    run(go())


def test_kick_does_nothing_when_no_tab_is_connected_and_the_sampler_stops_with_the_last():
    async def go():
        s = Sampler()
        feed = LiveFeed(s, interval_s=0.01, follow_ups_s=())
        assert feed.kick("x") is False, "nobody is waiting for it"
        assert s.calls == [], "no sampler runs without a tab"
        sub = feed.subscribe()
        assert await until(lambda: len(s.calls) >= 3)
        sub.close()
        await asyncio.sleep(0)
        settled = len(s.calls)
        await asyncio.sleep(0.1)
        assert len(s.calls) == settled, "the sampler kept running with no tab"
        assert feed.kick("x") is False
        sub2 = feed.subscribe()
        try:
            assert await until(lambda: len(s.calls) > settled)
            assert s.calls[settled] is True, "a returning tab starts with a forced reading"
        finally:
            sub2.close()

    run(go())


# ------------------------------------------------------ what gets pushed

def test_unchanged_displayed_values_are_not_pushed_again():
    async def go():
        s = Sampler(cpu=12.0)
        n = {"i": 0}
        inner = s.__call__

        async def sample(fresh):
            n["i"] += 1
            s.cpu = 12.0 + n["i"] * 0.001
            return await inner(fresh)

        feed = LiveFeed(sample, interval_s=0.005, follow_ups_s=())
        sub = feed.subscribe()
        try:
            assert await until(lambda: feed.samples >= 15)
            assert feed.seq == 1, "sub-percent CPU jitter is not a displayed change"
            s.cpu = 40.0

            async def sample2(fresh):
                return {"cpu": {"percent": 40.0},
                        "vram": {"used": 10 * GIB, "total": 16 * GIB}}

            feed._sample = sample2
            assert await until(lambda: feed.seq == 2)
            assert feed.latest["cpu"]["percent"] == 40.0
        finally:
            sub.close()

    run(go())


def test_two_tabs_share_one_sampler():
    async def go():
        s = Sampler()
        feed = LiveFeed(s, interval_s=60, follow_ups_s=())
        a = feed.subscribe()
        b = feed.subscribe()
        try:
            assert await a.wait(2.0)
            assert await b.wait(2.0)
            tasks = [t for t in asyncio.all_tasks() if t.get_name() == "localm-live-feed"]
            assert len(tasks) == 1
            assert s.calls == [True], "the second tab started a reading of its own"
            assert a.seen_seq == b.seen_seq == 0
            assert feed.seq == 1 and feed.latest is not None
        finally:
            a.close()
            b.close()

    run(go())


def test_a_failing_sample_is_reported_once_and_the_sampler_recovers(monkeypatch):
    import localm.debuglog as dl
    warnings, debugs = [], []
    monkeypatch.setattr(dl.logger, "warning", lambda *a, **k: warnings.append(a))
    monkeypatch.setattr(dl.logger, "debug", lambda *a, **k: debugs.append(a))

    async def go():
        n = {"i": 0}

        async def sample(fresh):
            n["i"] += 1
            if n["i"] <= 3:
                raise RuntimeError("probe died")
            return {"cpu": {"percent": 5.0}}

        feed = LiveFeed(sample, interval_s=0.005, follow_ups_s=())
        sub = feed.subscribe()
        try:
            assert await until(lambda: feed.latest is not None), \
                "the sampler died on a failed sample"
        finally:
            sub.close()

    run(go())
    assert len(warnings) == 1, warnings
    assert "probe died" in str(warnings[0])
    assert any("still failing" in str(d) for d in debugs)


# ------------------------------------------------------- display_key

@pytest.mark.parametrize("a, b, same", [
    ({"cpu": {"percent": 12.4}}, {"cpu": {"percent": 12.45}}, True),
    ({"cpu": {"percent": 12.4}}, {"cpu": {"percent": 12.5}}, False),
    ({"ram": {"used": 4 * GIB, "total": 8 * GIB}},
     {"ram": {"used": 4 * GIB + 1000, "total": 8 * GIB}}, True),
    ({"ram": {"used": 4 * GIB, "total": 8 * GIB}},
     {"ram": {"used": 4 * GIB + GIB // 5, "total": 8 * GIB}}, False),
    ({"vram": {"used": int(0.6996 * 16 * GIB), "total": 16 * GIB}},
     {"vram": {"used": int(0.7004 * 16 * GIB), "total": 16 * GIB}}, False),
    ({"vram": {"total": 16 * GIB}}, {"vram": {"used": 0, "total": 16 * GIB}}, False),
    ({"gpu": {"percent": 30.2}}, {"gpu": {"percent": 29.6}}, True),
    ({"gpu": {"percent": 30.2}}, {"gpu": {"percent": 30.6}}, False),
    ({}, {}, True),
])
def test_display_key_changes_exactly_when_the_status_bar_would(a, b, same):
    assert (display_key(a) == display_key(b)) is same


def test_display_key_sees_one_card_of_several_change():
    def board(second):
        return {"vram": {"used": 6 * GIB, "total": 32 * GIB, "devices": [
            {"index": 0, "used": 4 * GIB, "total": 16 * GIB},
            {"index": 1, "used": second * GIB, "total": 16 * GIB}]}}
    assert display_key(board(2)) != display_key(board(8))
    assert display_key(board(2)) == display_key(board(2))


# ------------------------------------------------------ the hook sites

class _Recorder:
    def __init__(self):
        self.reasons: list[str] = []

    def kick(self, reason):
        self.reasons.append(reason)
        return True


@pytest.fixture
def recorder():
    rec = _Recorder()
    livefeed.install(rec)
    yield rec
    livefeed.install(None)


def test_unloading_an_engine_kicks_the_feed_even_when_the_backend_raises(recorder):
    from localm.inference.engine import Engine

    class Backend:
        def __init__(self, fail):
            self.fail = fail

        def unload(self):
            if self.fail:
                raise RuntimeError("native free failed")

    Engine.unload(types.SimpleNamespace(_backend=Backend(False)))
    assert recorder.reasons == ["model unloaded"]
    with pytest.raises(RuntimeError):
        Engine.unload(types.SimpleNamespace(_backend=Backend(True)))
    assert recorder.reasons == ["model unloaded"] * 2, \
        "a failed unload may still have freed VRAM; the meter must be re-read"


def test_loading_an_engine_kicks_the_feed_after_the_load_ends_either_way(recorder):
    from localm.inference.engine import Engine

    def fake(fail):
        e = types.SimpleNamespace(
            _load_lock=threading.RLock(),
            _backend=types.SimpleNamespace(loaded=False), loading=False)

        def _load_locked():
            assert e.loading is True
            if fail:
                raise RuntimeError("load failed")
        e._load_locked = _load_locked
        return e

    Engine.load(fake(False))
    assert recorder.reasons == ["model loaded"]
    with pytest.raises(RuntimeError):
        Engine.load(fake(True))
    assert recorder.reasons == ["model loaded"] * 2
    already = fake(False)
    already._backend.loaded = True
    Engine.load(already)
    assert recorder.reasons == ["model loaded"] * 2, "nothing loaded, nothing changed"


def test_a_job_starting_and_finishing_kicks_the_feed(recorder):
    from localm.jobmanager import JobManager

    class Store:
        def write(self, rows):
            return True

        def clear(self):
            return None

        def reap(self, *, skip=None):
            return []

    mgr = JobManager(store=Store(), reconcile=False)
    done = threading.Event()
    job = mgr.start_fn("pull", lambda j: done.wait(5) or True, label="x")
    assert "job registered" in recorder.reasons, "the start must reach the feed"
    before = len(recorder.reasons)
    done.set()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and job.finished_at is None:
        time.sleep(0.01)
    assert job.finished_at is not None
    assert len(recorder.reasons) > before and recorder.reasons[-1] == "job changed", \
        "the end of a job must reach the feed"


def test_module_kick_is_false_without_a_feed():
    livefeed.install(None)
    assert livefeed.kick("x") is False


class _Resident:
    """Stands in for a resident embedder, reranker or speech engine."""

    def __init__(self):
        self.closed = False
        self._stopped = False

    def close(self):
        self.closed = True

    active_requests = 0
    alive = True


def test_releasing_a_resident_embedder_reranker_or_speech_model_kicks_the_feed(
        recorder, monkeypatch):
    from localm.inference import embedder, reranker, speech

    monkeypatch.setattr(embedder, "_EMBEDDER", _Resident())
    assert embedder.reset_embedder() is True
    monkeypatch.setattr(reranker, "_RERANKER", _Resident())
    assert reranker.reset_reranker() is True
    monkeypatch.setattr(speech, "_ENGINE", _Resident())
    assert speech.reset_speech() is True
    assert recorder.reasons == ["embedding model unloaded", "reranker unloaded",
                                "speech model unloaded"]


def test_releasing_nothing_does_not_kick_the_feed(recorder, monkeypatch):
    from localm.inference import embedder, reranker, speech

    monkeypatch.setattr(embedder, "_EMBEDDER", None)
    monkeypatch.setattr(reranker, "_RERANKER", None)
    monkeypatch.setattr(speech, "_ENGINE", None)
    assert (embedder.reset_embedder(), reranker.reset_reranker(),
            speech.reset_speech()) == (False, False, False)
    assert recorder.reasons == []


def test_a_restarted_sampler_does_not_serve_the_reading_from_before_it_stopped():
    async def go():
        s = Sampler(used_gib=10.0)
        feed = LiveFeed(s, interval_s=60, follow_ups_s=())
        first = feed.subscribe()
        assert await until(lambda: feed.latest is not None)
        first.close()
        await asyncio.sleep(0)
        s.truth = 2.0
        gate = asyncio.Event()
        inner = s.__call__

        async def slow(fresh):
            await gate.wait()
            return await inner(fresh)

        feed._sample = slow
        second = feed.subscribe()
        try:
            await asyncio.sleep(0.05)
            assert feed.latest is None, "a returning tab was shown the old reading"
            gate.set()
            assert await until(lambda: feed.latest is not None and used_gib(feed) == 2.0)
        finally:
            second.close()

    run(go())


def test_every_direct_backend_load_in_engine_is_followed_by_a_kick():
    """Engine.load() kicks in its finally block. The two other places that load
    the backend (the embed path and the chat reload) call it directly and so
    need a kick of their own."""
    import ast
    import inspect

    from localm.inference import engine

    tree = ast.parse(inspect.getsource(engine))
    bad, seen = [], 0

    def is_backend_load(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "self._backend.load")

    def is_kick(node):
        return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and ast.unparse(node.value.func) == "_livefeed.kick")

    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name == "_load_locked":
            continue
        for node in ast.walk(fn):
            for field in ("body", "orelse", "finalbody"):
                stmts = getattr(node, field, None)
                if not isinstance(stmts, list):
                    continue
                for i, st in enumerate(stmts):
                    if is_backend_load(st):
                        seen += 1
                        if i + 1 >= len(stmts) or not is_kick(stmts[i + 1]):
                            bad.append((fn.name, st.lineno))
    assert seen >= 2, "the sentinel found no direct backend loads; it is looking in the wrong place"
    assert bad == [], f"a direct backend load with no kick after it: {bad}"


def test_loading_an_embedder_kicks_the_feed(recorder, monkeypatch, tmp_path):
    from localm.inference import embedder as emb

    p = tmp_path / "e.gguf"
    p.write_bytes(b"0" * 1024)
    monkeypatch.setattr("localm.config.load_config",
                        lambda: {"embedding_model": str(p), "n_gpu_layers": 0,
                                 "net_mode": "off"})
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)

    class Fake:
        dim = 3
        effective_pooling = None

        def __init__(self, path, **kw):
            self.model_path = path

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", Fake)
    emb.reset_embedder()
    recorder.reasons.clear()
    try:
        assert emb.get_embedder() is not None
        assert recorder.reasons == ["embedding model loaded"]
        assert emb.get_embedder() is not None
        assert recorder.reasons == ["embedding model loaded"], "a reuse is not a load"
    finally:
        emb.reset_embedder()


def test_loading_a_speech_model_kicks_the_feed(recorder, monkeypatch, tmp_path):
    from localm.inference import speech

    class FakeEngine:
        alive = True
        active_requests = 0
        sample_rate = 24000
        projector_on_gpu = False

        def __init__(self, model, n_gpu_layers=0):
            self.model = model

        def claim(self):
            pass

        def close(self, grace=5.0):
            pass

    monkeypatch.setattr(speech, "SpeechEngine", FakeEngine)
    monkeypatch.setattr(speech, "_choose_gpu_layers", lambda model, n_ctx: (0, None))
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"GGUF" + b"m")
    proj = tmp_path / "mmproj-m.gguf"
    proj.write_bytes(b"GGUF")
    model = speech.SpeechModel("m", str(gguf), str(proj))
    speech.reset_speech()
    recorder.reasons.clear()
    try:
        speech.get_engine(model)
        assert recorder.reasons == ["speech model loaded"]
        speech.get_engine(model)
        assert recorder.reasons == ["speech model loaded"], "a reuse is not a load"
    finally:
        speech.reset_speech()


def test_loading_a_reranker_kicks_the_feed(recorder, monkeypatch, tmp_path):
    from localm.inference import embedder as emb
    from localm.inference import reranker as rr

    path = tmp_path / "r.gguf"
    path.write_bytes(b"GGUF" + b"0" * 1024)
    monkeypatch.setattr(rr, "_check_head", lambda p: None)
    monkeypatch.setattr(emb, "_maybe_swap_for_embedder", lambda *a, **k: None)
    monkeypatch.setattr(emb, "_choose_embedder_gpu_layers", lambda p, cfg: (0, None))

    class Fake:
        effective_pooling = emb._POOLING_RANK
        cls_labels = None

        def __init__(self, p, **kw):
            self.model_path = p

        def close(self):
            pass

    monkeypatch.setattr(emb, "IsolatedEmbedder", Fake)
    rr.reset_reranker()
    recorder.reasons.clear()
    try:
        assert rr.get_reranker(str(path)) is not None
        assert recorder.reasons == ["reranker loaded"]
        assert rr.get_reranker(str(path)) is not None
        assert recorder.reasons == ["reranker loaded"], "a reuse is not a load"
    finally:
        rr.reset_reranker()
