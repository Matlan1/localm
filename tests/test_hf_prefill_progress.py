# SPDX-License-Identifier: AGPL-3.0-or-later
"""A HuggingFace-format model reading a long prompt is not killed as a stall: the
worker reports each decoder layer the prompt passes, the status reaches the parent
intact, and every status restarts the first-token deadline. A worker that reports
nothing is still caught."""

from __future__ import annotations

import multiprocessing
import threading
import time

import pytest

from localm.inference import protocol
from localm.inference.backends import _hf_runner as runner_mod
from localm.inference.backends import _hf_worker as worker_mod
from tests.test_hf_status_callback import _make_hf_runner


class _Handle:
    def __init__(self, hooks, fn):
        self._hooks, self._fn = hooks, fn

    def remove(self):
        self._hooks.remove(self._fn)


class _Layer:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return _Handle(self.hooks, fn)

    def run(self):
        for fn in list(self.hooks):
            fn(self, (), None)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _pass(layers, clock, step):
    for layer in layers:
        clock.t += step
        layer.run()


def test_the_prompt_pass_reports_layers_once_per_interval_and_decode_reports_nothing():
    layers = [_Layer() for _ in range(6)]
    clock = _Clock()
    seen = []
    progress = worker_mod._PrefillLayerProgress(seen.append, layers, 5.0, clock=clock)
    _pass(layers, clock, 3.0)
    assert [str(s) for s in seen] == ["Processing prompt... (layer 2 of 6)",
                                      "Processing prompt... (layer 4 of 6)"]
    assert all(s.code == "processing_layers" for s in seen)
    assert seen[-1].args == {"done": "4", "total": "6"}
    _pass(layers, clock, 10.0)
    assert len(seen) == 2, "a decode step after the prompt is not prompt progress"
    progress.close()
    assert all(layer.hooks == [] for layer in layers)
    progress.close()


def test_close_before_the_prompt_ends_stops_the_reports():
    layers = [_Layer() for _ in range(4)]
    clock = _Clock()
    seen = []
    progress = worker_mod._PrefillLayerProgress(seen.append, layers, 0.0, clock=clock)
    clock.t = 1.0
    layers[0].run()
    progress.close()
    _pass(layers, clock, 1.0)
    assert [s.args["done"] for s in seen] == ["1"]


def test_a_raising_status_callback_does_not_break_the_forward_pass():
    layers = [_Layer() for _ in range(3)]
    clock = _Clock()

    def boom(_s):
        raise RuntimeError("queue gone")

    worker_mod._PrefillLayerProgress(boom, layers, 0.0, clock=clock)
    _pass(layers, clock, 1.0)


def test_the_decoder_stack_is_found_in_a_real_torch_model_and_its_hooks_fire():
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision = nn.Module()
            self.vision.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(5)])

        def forward(self, x):
            for layer in self.model.layers:
                x = layer(x)
            return x

    m = Tiny()
    layers = worker_mod._decoder_layers(m)
    assert layers is m.model.layers
    seen = []
    progress = worker_mod._PrefillLayerProgress(seen.append, layers, 0.0)
    try:
        m(torch.zeros(1, 4))
    finally:
        progress.close()
    assert [s.args["done"] for s in seen] == ["1", "2", "3", "4"]
    assert worker_mod._decoder_layers(nn.Linear(2, 2)) is None


def test_the_layer_status_survives_the_process_boundary():
    s = protocol.prefill_layers_status(12, 48)
    q = multiprocessing.get_context("spawn").Queue()
    q.put(s)
    back = q.get(timeout=10)
    assert str(back) == "Processing prompt... (layer 12 of 48)"
    assert back.code == "processing_layers"
    assert back.args == {"done": "12", "total": "48"}
    assert protocol.status_code_of(back) == "processing_layers"


def test_a_prefill_longer_than_the_budget_that_reports_layers_is_not_killed(monkeypatch):
    monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
    r = _make_hf_runner()
    stop = threading.Event()

    def _child():
        r._req_q.get(timeout=5)
        for done in range(1, 9):              # 8 x 0.3 s = 2.4 s, budget 1 s
            if stop.wait(0.3):
                return
            r._resp_q.put(("status", protocol.prefill_layers_status(done, 8)))
        r._resp_q.put(("chunk", "ok"))
        r._resp_q.put(("done", {"finish_reason": "stop"}))

    child = threading.Thread(target=_child, daemon=True)
    child.start()
    statuses = []
    try:
        t0 = time.monotonic()
        out = list(r.chat_stream(first_chunk_timeout=1.0, on_status=statuses.append,
                                 messages=[]))
        assert out == ["ok"]
        assert time.monotonic() - t0 > 2.0
        assert not r._proc.terminated
        assert getattr(statuses[-1], "code", None) == "processing_layers"
    finally:
        stop.set()
        child.join(2)


def test_a_prefill_that_stops_reporting_is_still_caught(monkeypatch):
    monkeypatch.setattr(runner_mod, "_STREAM_CHUNK_TIMEOUT", 0.5)
    r = _make_hf_runner()

    def _child():
        r._req_q.get(timeout=5)
        r._resp_q.put(("status", protocol.prefill_layers_status(1, 8)))

    child = threading.Thread(target=_child, daemon=True)
    child.start()
    with pytest.raises(RuntimeError) as exc:
        list(r.chat_stream(first_chunk_timeout=1.0, messages=[]))
    child.join(2)
    msg = str(exc.value)
    assert "made no progress for 1s" in msg
    assert "'HuggingFace first-token timeout (s)'" in msg
    assert "hf_first_token_timeout_s" not in msg
    assert r._proc.terminated
