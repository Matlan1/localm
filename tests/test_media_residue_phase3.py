"""Media-containment residue regression tests (M2 Phase 3).

Two leaks are covered here, both about a generation leaving a trace that the
per-plugin containment knobs were supposed to prevent:

FAC-3  - the per-plugin ``comfy.output_dir`` is honoured for image generation
         but was silently dropped by the music and video backends, so their
         containment knob did nothing. ComfyUI's output dir is what
         ``contain_comfy_artifacts`` needs to delete its on-disk copy AND to
         locate the input/ dir for the uploaded img2img/i2v source. The real
         ``generate_music`` / ``generate_video`` take no ``comfy_output_dir``
         argument (resolution is via the ``COMFY_OUTPUT_DIR`` env var, the same
         fallback the shipped containment test relies on), so the backend must
         publish the per-plugin value on that env var for the duration of the
         call. We mock ``generate_*`` and assert the value is visible to it.

GAP-MG-1 - the img2img / image-to-video ``input_image`` accepted ANY readable
         local path (only ``is_file()`` was checked) and uploaded it to
         ComfyUI's input/ dir. That is an arbitrary-file-read-into-ComfyUI
         primitive (e.g. /etc/passwd, a private key, another user's docs). The
         path must be confined to the localm home (or the project checkout);
         anything outside is rejected with a 400.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from fastapi import HTTPException

from localm.plugins.builtin.image import plug as image_plug
from localm.plugins.builtin.music import backend as music_backend
from localm.plugins.builtin.video import backend as video_backend
from localm.plugins.builtin.video import plug as video_plug


# --------------------------------------------------------------------------- #
#  FAC-3: music / video backends forward the per-plugin output dir
# --------------------------------------------------------------------------- #

def _cfg_with_output_dir(name: str, output_dir: str) -> dict:
    return {
        "plugins": {
            name: {
                "backend": "comfy",
                "comfy": {"output_dir": output_dir},
            }
        }
    }


def test_music_settings_exposes_output_dir():
    s = music_backend.settings(_cfg_with_output_dir("music", "/tmp/comfy-music-out"))
    assert s.get("output_dir") == "/tmp/comfy-music-out"


def test_video_settings_exposes_output_dir():
    s = video_backend.settings(_cfg_with_output_dir("video", "/tmp/comfy-video-out"))
    assert s.get("output_dir") == "/tmp/comfy-video-out"


def test_music_backend_publishes_output_dir_to_generate(monkeypatch):
    """The per-plugin output_dir must reach the (env-resolved) containment path.

    The real generate_music has NO comfy_output_dir parameter; resolution is via
    the COMFY_OUTPUT_DIR env var. So the backend must set that env var to the
    per-plugin value while generate_music runs. We capture what the callee sees.
    """
    seen = {}

    def fake_generate_music(tags, out_path, **kwargs):
        seen["env"] = os.environ.get("COMFY_OUTPUT_DIR")
        # the real callee never receives this kwarg - assert the backend did not
        # try to pass it (that would TypeError against the real function)
        seen["kwargs_has_comfy_output_dir"] = "comfy_output_dir" in kwargs
        return True, "ok"

    monkeypatch.setattr(music_backend, "_music_gen",
                        type("M", (), {"generate_music": staticmethod(fake_generate_music)}))
    monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)

    s = music_backend.settings(_cfg_with_output_dir("music", "/tmp/music-out-dir"))
    ok, msg = music_backend.generate(
        s, "lofi", Path("track.flac"), self_url="", write_sidecar=False)

    assert ok, msg
    assert seen["env"] == "/tmp/music-out-dir"
    assert seen["kwargs_has_comfy_output_dir"] is False
    # env must be restored afterwards (no global leak)
    assert os.environ.get("COMFY_OUTPUT_DIR") is None


def test_video_backend_publishes_output_dir_to_generate(monkeypatch):
    seen = {}

    def fake_generate_video(prompt, out_path, **kwargs):
        seen["env"] = os.environ.get("COMFY_OUTPUT_DIR")
        seen["kwargs_has_comfy_output_dir"] = "comfy_output_dir" in kwargs
        return True, "ok"

    monkeypatch.setattr(video_backend, "_video_gen",
                        type("V", (), {"generate_video": staticmethod(fake_generate_video)}))
    monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)

    s = video_backend.settings(_cfg_with_output_dir("video", "/tmp/video-out-dir"))
    ok, msg = video_backend.generate(
        s, "a cat walks", Path("clip.mp4"), self_url="", write_sidecar=False)

    assert ok, msg
    assert seen["env"] == "/tmp/video-out-dir"
    assert seen["kwargs_has_comfy_output_dir"] is False
    assert os.environ.get("COMFY_OUTPUT_DIR") is None


def test_video_backend_restores_prior_output_dir_env(monkeypatch):
    """A pre-existing COMFY_OUTPUT_DIR must be restored, not clobbered, after the
    per-plugin override is applied for the call."""
    seen = {}

    def fake_generate_video(prompt, out_path, **kwargs):
        seen["env"] = os.environ.get("COMFY_OUTPUT_DIR")
        return True, "ok"

    monkeypatch.setattr(video_backend, "_video_gen",
                        type("V", (), {"generate_video": staticmethod(fake_generate_video)}))
    monkeypatch.setenv("COMFY_OUTPUT_DIR", "/pre/existing")

    s = video_backend.settings(_cfg_with_output_dir("video", "/tmp/override-dir"))
    ok, _ = video_backend.generate(
        s, "p", Path("clip.mp4"), self_url="", write_sidecar=False)

    assert ok
    assert seen["env"] == "/tmp/override-dir"          # override applied
    assert os.environ.get("COMFY_OUTPUT_DIR") == "/pre/existing"   # restored


def test_music_backend_no_output_dir_leaves_env_untouched(monkeypatch):
    """With no per-plugin output_dir configured, the backend must not invent one
    (it must leave whatever COMFY_OUTPUT_DIR the environment already had)."""
    seen = {}

    def fake_generate_music(tags, out_path, **kwargs):
        seen["env"] = os.environ.get("COMFY_OUTPUT_DIR")
        return True, "ok"

    monkeypatch.setattr(music_backend, "_music_gen",
                        type("M", (), {"generate_music": staticmethod(fake_generate_music)}))
    monkeypatch.delenv("COMFY_OUTPUT_DIR", raising=False)

    s = music_backend.settings({"plugins": {"music": {"backend": "comfy"}}})
    ok, _ = music_backend.generate(
        s, "t", Path("track.flac"), self_url="", write_sidecar=False)

    assert ok
    assert seen["env"] is None


# --------------------------------------------------------------------------- #
#  GAP-MG-1: input_image must be confined to an allowed root
# --------------------------------------------------------------------------- #

def _outside_path(tmp_path) -> Path:
    """A real, readable file guaranteed to be OUTSIDE the localm home + project
    roots (an arbitrary-file-read target)."""
    p = tmp_path / "secret.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nNOTYOURS")
    return p


def _run(coro):
    return asyncio.run(coro)


def _fake_request():
    """Minimal stand-in: the route reads request.app.state.{jobs,self_url}. We
    never reach the job manager for the rejection path; for the accepted path it
    is intentionally absent so the route 503s right after confinement passes."""
    ns = type("NS", (), {})
    req, app, state = ns(), ns(), ns()
    state.jobs = None
    state.self_url = ""
    app.state = state
    req.app = app
    return req


def test_image_input_image_outside_root_rejected(tmp_path, monkeypatch):
    # point the localm home somewhere isolated so tmp_path is genuinely outside
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    evil = _outside_path(tmp_path)
    assert evil.is_file()                         # the old is_file() check passes

    req = image_plug.ImagineRequest(prompt="x", input_image=str(evil))
    with pytest.raises(HTTPException) as ei:
        _run(image_plug.imagine(req, _fake_request()))
    assert ei.value.status_code == 400
    assert "input" in str(ei.value.detail).lower()


def test_video_input_image_outside_root_rejected(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    evil = _outside_path(tmp_path)
    assert evil.is_file()

    req = video_plug.VideoRequest(prompt="x", input_image=str(evil))
    with pytest.raises(HTTPException) as ei:
        _run(video_plug.video(req, _fake_request()))
    assert ei.value.status_code == 400
    assert "input" in str(ei.value.detail).lower()


def test_image_input_image_inside_home_accepted(tmp_path, monkeypatch):
    """A file INSIDE the localm home passes confinement (it still 503s later for
    lack of a job manager, which proves we got past the input check)."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    inside = home / "ref.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\nMINE")

    req = image_plug.ImagineRequest(prompt="x", input_image=str(inside))
    with pytest.raises(HTTPException) as ei:
        _run(image_plug.imagine(req, _fake_request()))
    # passed confinement; failed later on the missing GUI job manager
    assert ei.value.status_code == 503


def test_video_input_image_inside_home_accepted(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    inside = home / "ref.png"
    inside.write_bytes(b"\x89PNG\r\n\x1a\nMINE")

    req = video_plug.VideoRequest(prompt="x", input_image=str(inside))
    with pytest.raises(HTTPException) as ei:
        _run(video_plug.video(req, _fake_request()))
    assert ei.value.status_code == 503


def test_image_nonexistent_input_image_still_400(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    missing = home / "nope.png"     # inside root but does not exist
    req = image_plug.ImagineRequest(prompt="x", input_image=str(missing))
    with pytest.raises(HTTPException) as ei:
        _run(image_plug.imagine(req, _fake_request()))
    assert ei.value.status_code == 400
