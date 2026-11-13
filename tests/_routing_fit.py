# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fit stand-ins for tests whose subject is not how a candidate fits.

Capability routing needs a fit for every candidate (``plan_route``'s required
``fit``). A test about the latch, the routing header or a surface uses these
so every candidate reads as fitting the GPU; tests about fit itself use real
readings or their own ``Fit`` values."""

from __future__ import annotations

import pytest

from localm.inference import capability_routing as cr


def fits_on_gpu(names):
    """A ``fit`` assessor under which every candidate fits the GPU."""
    return {n: cr.Fit(cr.FIT_GPU) for n in names}


def plan_route(*args, **kwargs):
    """``capability_routing.plan_route`` with ``fits_on_gpu`` unless the test
    passes its own ``fit``."""
    kwargs.setdefault("fit", fits_on_gpu)
    return cr.plan_route(*args, **kwargs)


def reset_routing_state() -> None:
    """Drop the capability-routing state the HTTP server keeps for the life of
    its process: the cached planning GPU reading, the models routing loaded,
    the loads in flight and the failed-load latch. Every routing fixture calls
    it, so no test sees state an earlier test in the same worker left behind.
    See test_no_test_sets_the_routing_reading_but_through_pin_vram_reading."""
    from localm.inference import http_server as hs
    hs._forget_routing_reading()
    hs._routing_loaded.clear()
    hs._loads_in_flight.clear()
    hs._routing_latch.clear()


def pin_vram_reading(monkeypatch, reading) -> None:
    """Make capability routing read *reading* (a ``routing_fit.VramReading``,
    or a callable returning one) as the GPU for this test, with the routing
    state reset (``reset_routing_state``) so no reading cached by an earlier
    test is used instead."""
    from localm.inference import http_server as hs
    reset_routing_state()
    monkeypatch.setattr(hs, "routing_vram_reading",
                        reading if callable(reading) else (lambda: reading))


def make_every_candidate_fit(monkeypatch) -> None:
    """Make every routing fit measurement in the HTTP and MCP servers read
    "fits the GPU": a large usable VRAM and RAM reading and a classification
    of ``FIT_GPU``. Eviction and ask rules still run on top of it."""
    from localm.inference import routing_fit
    from localm.inference.backends.llamacpp import _sizing
    big = 1 << 50
    pin_vram_reading(monkeypatch, routing_fit.VramReading(free=big, total=big, usable=True))
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (big, big))
    monkeypatch.setattr(routing_fit, "classify", lambda *a, **k: (cr.FIT_GPU, 0))


@pytest.fixture
def every_candidate_fits(monkeypatch):
    make_every_candidate_fit(monkeypatch)
