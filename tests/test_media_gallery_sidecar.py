# SPDX-License-Identifier: AGPL-3.0-or-later
"""The settings sidecar of a generated file is read through one function for
the image, music and video histories: a missing one is no settings, and one
that cannot be read or is not a JSON object is no settings plus a WARNING that
names the file."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from localm.media import gallery


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == "localm" and r.levelno == logging.WARNING]


def test_a_sidecar_object_is_returned(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="localm")
    (tmp_path / "a.wav.json").write_text('{"tags": "lofi"}', encoding="utf-8")
    assert gallery.read_sidecar("music", tmp_path / "a.wav") == {"tags": "lofi"}
    assert _warnings(caplog) == []


def test_no_sidecar_is_no_settings_and_no_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="localm")
    assert gallery.read_sidecar("image", tmp_path / "a.png") == {}
    assert _warnings(caplog) == []


@pytest.mark.parametrize("text,why", [("{not json", "Expecting property name"),
                                      ("[1, 2]", "not a JSON object")])
def test_an_unreadable_sidecar_is_no_settings_and_a_warning(tmp_path, caplog, text, why):
    caplog.set_level(logging.WARNING, logger="localm")
    (tmp_path / "clip.mp4.json").write_text(text, encoding="utf-8")
    assert gallery.read_sidecar("video", tmp_path / "clip.mp4") == {}
    (warning,) = _warnings(caplog)
    assert warning.startswith("video: the settings file of clip.mp4 could not be read")
    assert why in warning


def test_no_media_history_reads_a_sidecar_itself():
    import localm
    root = Path(localm.__file__).parent / "plugins" / "builtin"
    for media in ("image", "music", "video"):
        src = (root / media / "plug.py").read_text(encoding="utf-8")
        assert re.search(r'gallery\.read_sidecar\("' + media + '", ', src), media
        assert not re.search(r"json\.loads\(sidecar", src), media
