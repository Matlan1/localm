# SPDX-License-Identifier: AGPL-3.0-or-later
"""video()/music() reject out-of-range width, height and steps before doing
any work, the same way they already reject an out-of-range seconds/fps/
duration_seconds. Each oversized value is asserted to fail with the specific
validation HTTPException, not merely "some exception" - a broad
pytest.raises(Exception) would also match the 503 raised further down the
same function for a missing job registry, and could pass even if the bound
check were deleted entirely. Each in-range value is asserted to pass the
bound check by reaching that same later 503 (no job registry configured on
the fake request), proving the request cleared validation rather than never
being exercised.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from unittest.mock import MagicMock

from localm.plugins.builtin.video import plug as video_plug
from localm.plugins.builtin.music import plug as music_plug


def _fake_request():
    request = MagicMock()
    request.app.state.jobs = None  # forces a 503 past any bound check that passes
    request.headers = {}
    request.cookies = {}
    return request


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("field,value", [
    ("width", 100000),
    ("width", 15),
    ("height", 100000),
    ("height", 15),
    ("steps", 100000),
    ("steps", 0),
])
def test_video_rejects_out_of_range_dimension(field, value):
    req = video_plug.VideoRequest(prompt="a cat", **{field: value})
    with pytest.raises(HTTPException) as exc_info:
        _run(video_plug.video(req, _fake_request()))
    assert exc_info.value.status_code == 400
    assert field in exc_info.value.detail.lower()


@pytest.mark.parametrize("kwargs", [
    {"width": 1024},
    {"height": 1024},
    {"steps": 30},
    {"width": 16, "height": 16, "steps": 1},   # lower boundary, inclusive
    {"width": 4096, "height": 4096, "steps": 200},  # upper boundary, inclusive
    {},  # all three left None - must not be rejected by the new checks
])
def test_video_accepts_in_range_dimension(kwargs):
    req = video_plug.VideoRequest(prompt="a cat", **kwargs)
    with pytest.raises(HTTPException) as exc_info:
        _run(video_plug.video(req, _fake_request()))
    # cleared the new bound checks and hit the pre-existing "no job registry"
    # 503 further down the same function, not a 400 from this change
    assert exc_info.value.status_code == 503


@pytest.mark.parametrize("value", [100000, 0, -1])
def test_music_rejects_out_of_range_steps(value):
    req = music_plug.MusicRequest(tags="lofi", steps=value)
    with pytest.raises(HTTPException) as exc_info:
        _run(music_plug.music(req, _fake_request()))
    assert exc_info.value.status_code == 400
    assert "steps" in exc_info.value.detail.lower()


@pytest.mark.parametrize("kwargs", [
    {"steps": 30},
    {"steps": 1},
    {"steps": 200},
    {},  # steps left None - must not be rejected by the new check
])
def test_music_accepts_in_range_steps(kwargs):
    req = music_plug.MusicRequest(tags="lofi", **kwargs)
    with pytest.raises(HTTPException) as exc_info:
        _run(music_plug.music(req, _fake_request()))
    assert exc_info.value.status_code == 503
