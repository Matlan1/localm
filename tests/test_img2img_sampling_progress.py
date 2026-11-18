"""Image-to-image jobs run a trimmed schedule (steps * denoise); its progress is
shown as sampling, and the step count that ran is available to the sidecar."""

from __future__ import annotations

import queue

import pytest

from localm.media.sdcpp import _child
from localm.plugins.builtin.image.backends import native as image_native


@pytest.mark.parametrize("steps,strength,expected", [
    (10, 0.6, (6, 7)),
    (20, 0.25, (5, 6)),
    (10, 1.0, ()),
    (10, None, ()),
    (10, 0.0, ()),
    (2, 0.2, ()),
    (2, 0.75, (1, 2)),
    (10, 0.95, (9, 10)),
])
def test_trimmed_sample_steps(steps, strength, expected):
    assert _child.trimmed_sample_steps(steps, strength) == expected


@pytest.fixture
def progress_cb():
    q: queue.Queue = queue.Queue()
    _, progress, _ = _child._make_callbacks(q)
    saved = (_child._State.sample_steps, _child._State.trimmed_steps,
             _child._State.hires_steps, _child._State.decoding,
             _child._State.upscaling, _child._State.last_progress)
    _child._State.decoding = False
    _child._State.upscaling = False
    _child._State.last_progress = 0.0
    yield progress, q
    (_child._State.sample_steps, _child._State.trimmed_steps,
     _child._State.hires_steps, _child._State.decoding,
     _child._State.upscaling, _child._State.last_progress) = saved


def test_a_trimmed_schedule_reports_sampling_progress(progress_cb):
    progress, q = progress_cb
    _child._State.sample_steps = 10
    _child._State.trimmed_steps = (6, 7)
    _child._State.hires_steps = 0
    progress(1, 7, 0.5, None)
    kind, info = q.get_nowait()
    assert kind == "progress" and info["phase"] == "sampling" and info["steps"] == 7


def test_an_untrimmed_job_still_reports_sampling_at_the_requested_steps(progress_cb):
    progress, q = progress_cb
    _child._State.sample_steps = 10
    _child._State.trimmed_steps = ()
    progress(1, 10, 0.5, None)
    assert q.get_nowait()[1]["phase"] == "sampling"


def test_the_relay_hands_the_sidecar_the_steps_that_ran():
    said: list[str] = []
    relay = image_native._event_relay(said.append)
    relay(("progress", {"phase": "sampling", "step": 1, "steps": 6, "secs": 0.5}))
    relay(("progress", {"phase": "sampling", "step": 2, "steps": 6, "secs": 0.5}))
    assert said == ["Step 1/6 (0.50 s/step)", "Step 2/6 (0.50 s/step)"]
    assert relay.take_sampled_steps() == 6
    assert relay.take_sampled_steps() is None
