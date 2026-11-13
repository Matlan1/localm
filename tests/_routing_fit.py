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


def make_every_candidate_fit(monkeypatch) -> None:
    """Make every routing fit measurement in the HTTP and MCP servers read
    "fits the GPU": a large usable VRAM and RAM reading and a classification
    of ``FIT_GPU``. Eviction and ask rules still run on top of it."""
    from localm.inference import http_server as hs
    from localm.inference import routing_fit
    from localm.inference.backends.llamacpp import _sizing
    big = 1 << 50
    monkeypatch.setattr(hs, "routing_vram_reading", lambda: routing_fit.VramReading(
        free=big, total=big, usable=True))
    monkeypatch.setattr(_sizing, "ram_reading", lambda: (big, big))
    monkeypatch.setattr(routing_fit, "classify", lambda *a, **k: (cr.FIT_GPU, 0))


@pytest.fixture
def every_candidate_fits(monkeypatch):
    make_every_candidate_fit(monkeypatch)
