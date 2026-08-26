# SPDX-License-Identifier: AGPL-3.0-or-later
"""Production code must not detect tests or mocks and fabricate behavior.

Two facades in http_server accommodated test mocks in shipped code:

  - switch_engine fabricated a fake model path + 4 GB size whenever
    `"pytest" in sys.modules`, instead of raising the real 404 "Model files not
    found" - making that contract untestable and running all in-test eviction
    math on fiction;

  - _complete had hasattr-guarded fallbacks (100 prompt tokens, 4096 capacity,
    a literal "ok" completion, 10 completion tokens) so a method-less engine
    passed through the real endpoint with a fabricated 200 instead of surfacing
    the failure.
"""

import asyncio

import pytest
from fastapi import HTTPException

from localm.inference import http_server as hs


class FakeEngine:
    def __init__(self, name):
        self.display_name = name
        self._loaded = False
        self.active_requests = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    def set_load_cancel(self, ev):
        pass


def _reset():
    for d in (hs._engines, hs._engines_lru, hs._inference_sems, hs._last_activity_per_model):
        d.clear()
    hs._active_model_name = None
    hs._default_model_name = None
    hs._engine = None
    hs._inference_sem = None
    hs._switch_desired = None
    hs._switch_loading = None
    hs._switch_cancel = None


def test_switch_engine_404s_a_missing_model_even_under_pytest(monkeypatch):
    """A non-empty registry where the model's info cannot be resolved must raise
    the real 404 - not fabricate a path/size just because pytest is importable."""
    _reset()
    monkeypatch.setattr("localm.config.load_registry",
                        lambda: {"other-model": {"path": "x", "source": "local"}})
    monkeypatch.setattr("localm.model_manager.get_model_info", lambda n: None)
    monkeypatch.setattr("localm.discover.vram_info",
                        lambda: {"free": 10 * 1024 ** 3, "total": 16 * 1024 ** 3})

    with pytest.raises(HTTPException) as ei:
        asyncio.run(hs.switch_engine("ghost", lambda n: FakeEngine(n)))
    assert ei.value.status_code == 404
    _reset()


def test_complete_surfaces_a_method_less_engine_instead_of_faking_ok(monkeypatch):
    """An engine missing chat_stream must raise (a real failure), not return the
    literal fabricated "ok" completion."""

    class PartialEngine:
        def count_messages_tokens(self, messages):
            return 5

        def context_capacity(self):
            return 4096

        def count_tokens(self, text):
            return 3
        # no chat_stream

    sem = asyncio.Semaphore(1)
    with pytest.raises(AttributeError):
        asyncio.run(hs._complete(PartialEngine(),
                                 [{"role": "user", "content": "hi"}], "m", sem))
