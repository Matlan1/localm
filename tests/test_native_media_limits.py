# SPDX-License-Identifier: AGPL-3.0-or-later
"""The native steps, CFG and flow-shift limits are the same in Settings, in a
saved native workflow, in the native backends and in the generate routes: a
value the page accepts can be saved as a preset and set in Settings."""

from __future__ import annotations

import pytest

from localm import settings_schema as ss
from localm.media import native_workflows as nw
from localm.media.sdcpp import sampling


def test_settings_accept_the_largest_steps_and_refuse_one_more():
    for media in ("image", "video"):
        assert ss.validate_media_block(media, {"native_steps": str(ss.NATIVE_STEPS_MAX)})
        with pytest.raises(ValueError):
            ss.validate_media_block(media, {"native_steps": str(ss.NATIVE_STEPS_MAX + 1)})


@pytest.mark.parametrize("media", ["image", "video", "music"])
def test_a_saved_workflow_takes_the_same_steps_and_cfg_range(media):
    assert nw._param(media, "steps", ss.NATIVE_STEPS_MAX) == ss.NATIVE_STEPS_MAX
    assert nw._param(media, "cfg", ss.NATIVE_CFG_MAX) == ss.NATIVE_CFG_MAX
    with pytest.raises(ValueError):
        nw._param(media, "steps", ss.NATIVE_STEPS_MAX + 1)


def test_the_flow_shift_range_is_the_same_in_settings_the_workflow_and_the_route():
    from localm.plugins.builtin.video.plug import VideoRequest
    assert nw._param("video", "flow_shift", ss.NATIVE_FLOW_SHIFT_MAX) == ss.NATIVE_FLOW_SHIFT_MAX
    assert ss.validate_media_block("video", {"native_flow_shift": str(ss.NATIVE_FLOW_SHIFT_MAX)})
    le = [m.le for m in VideoRequest.model_fields["flow_shift"].metadata if hasattr(m, "le")]
    assert le == [ss.NATIVE_FLOW_SHIFT_MAX]


def test_the_image_route_and_the_backends_take_the_same_steps_maximum():
    from localm.plugins.builtin.image import plug
    assert plug.MAX_STEPS == ss.NATIVE_STEPS_MAX
    assert sampling.check_steps(ss.NATIVE_STEPS_MAX) == ss.NATIVE_STEPS_MAX
    with pytest.raises(ValueError):
        sampling.check_steps(ss.NATIVE_STEPS_MAX + 1)


@pytest.mark.parametrize("value", [0, -3, True, 2.5, "20"])
def test_steps_that_are_not_a_whole_number_in_range_are_refused(value):
    with pytest.raises(ValueError):
        sampling.check_steps(value)
