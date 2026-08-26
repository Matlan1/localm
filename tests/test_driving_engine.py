# SPDX-License-Identifier: AGPL-3.0-or-later
"""localm.inference.http_server.driving_engine: the plugin-facing primitive that
pins active_requests and touches the per-model activity clock for the DURATION
of a plugin-driven generation call (memory auto-consolidate, a scheduled job,
...), so idle-unload cannot evict a model a plugin is actively using with no
concurrent HTTP traffic.
"""

from __future__ import annotations

import asyncio
import time

from localm.inference import http_server as hs


class _FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self.loaded = True
        self.active_requests = 0
        self.unloads = 0

    def unload(self):
        self.loaded = False
        self.unloads += 1


def _reset(*names):
    hs._engines.clear()
    hs._engines_lru.clear()
    hs._inference_sems.clear()
    hs._inference_sem = asyncio.Semaphore(1)
    for n in names:
        hs._last_activity_per_model.pop(n, None)


def test_driving_engine_pins_active_requests_for_the_duration():
    eng = _FakeEngine("m")
    assert eng.active_requests == 0
    with hs.driving_engine(eng):
        assert eng.active_requests == 1
    assert eng.active_requests == 0


def test_driving_engine_unpins_even_on_exception():
    """A plugin's generation call can raise (a backend error, a cancelled
    request); the pin is released anyway rather than left stuck at 1."""
    eng = _FakeEngine("m")
    try:
        with hs.driving_engine(eng):
            assert eng.active_requests == 1
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert eng.active_requests == 0


def test_driving_engine_touches_activity_on_enter_and_exit():
    _reset("m")
    eng = _FakeEngine("m")
    before = time.monotonic()
    with hs.driving_engine(eng):
        during = hs._last_activity_per_model["m"]
        assert during >= before
        time.sleep(0.02)
    after_exit = hs._last_activity_per_model["m"]
    assert after_exit > during, (
        "exit must re-touch the clock - otherwise the idle countdown restarts "
        "from task START, not from when the task actually finished")


def test_idle_unload_does_not_evict_while_pinned_even_when_stale():
    """A long plugin task pauses between rounds long enough for the per-model
    timestamp to look stale; the active_requests pin from driving_engine vetoes
    eviction regardless of that timestamp."""
    async def scenario():
        _reset("busy")
        eng = _FakeEngine("busy")
        hs._engines["busy"] = eng

        with hs.driving_engine(eng):
            # Simulate a long pause mid-task WITHOUT releasing the pin.
            hs._last_activity_per_model["busy"] = time.monotonic() - 1000
            did = await hs._idle_unload_once(60)
            return did, eng.loaded

    did, loaded = asyncio.run(scenario())
    assert loaded is True, "a pinned engine must not be evicted regardless of timestamp staleness"
    assert did is False


def test_idle_unload_can_evict_once_the_pin_releases_and_time_passes():
    """Once the task finishes (pin released) and enough real idle time passes,
    eviction proceeds normally: the pin is not a permanent exemption."""
    async def scenario():
        _reset("done")
        eng = _FakeEngine("done")
        hs._engines["done"] = eng

        with hs.driving_engine(eng):
            pass   # task finishes; exit re-touches the clock to "now"
        # Simulate enough idle time having passed SINCE the task ended.
        hs._last_activity_per_model["done"] = time.monotonic() - 1000
        did = await hs._idle_unload_once(60)
        return did, eng.loaded

    did, loaded = asyncio.run(scenario())
    assert did is True
    assert loaded is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
