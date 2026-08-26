# SPDX-License-Identifier: AGPL-3.0-or-later
"""Media-containment residue regression tests.

Two leaks, both about a generation leaving a trace the per-plugin containment
knobs are supposed to prevent:

  * the per-plugin ``comfy.output_dir`` must reach the music and video backends.
    ComfyUI's output dir is what ``contain_comfy_artifacts`` needs to delete its
    on-disk copy AND to locate the input/ dir for the uploaded img2img/i2v
    source. The real ``generate_music`` / ``generate_video`` take no
    ``comfy_output_dir`` argument (resolution is via the ``COMFY_OUTPUT_DIR``
    env var), so the backend publishes the per-plugin value on that env var for
    the duration of the call. These mock ``generate_*`` and assert the value is
    visible to it.

  * the img2img / image-to-video ``input_image`` must be confined. Accepting any
    readable local path on an ``is_file()`` check alone is an
    arbitrary-file-read-into-ComfyUI primitive (e.g. /etc/passwd, a private key,
    another user's docs), so anything outside an allowed root is a 400.

    The allowed set is the upload inbox plus the generated-media galleries - see
    ``localm.media.paths.allowed_input_roots`` - never the whole localm home,
    which is the credential store (auth.key, auth.json, sessions.json, rag/,
    coder/, bug-reports/). The data-dir ROOT is therefore rejected, which is
    asserted below.
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
#  music / video backends forward the per-plugin output dir
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


@pytest.mark.parametrize(
    "backend_mod,plugin_name,output_dir",
    [
        (music_backend, "music", "/tmp/comfy-music-out"),
        (video_backend, "video", "/tmp/comfy-video-out"),
    ],
    ids=["music", "video"],
)
def test_backend_settings_exposes_output_dir(backend_mod, plugin_name, output_dir):
    s = backend_mod.settings(_cfg_with_output_dir(plugin_name, output_dir))
    assert s.get("output_dir") == output_dir


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
#  input_image must be confined to an allowed root
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


@pytest.mark.parametrize(
    "request_cls,handler_coro_fn",
    [
        (image_plug.ImagineRequest, image_plug.imagine),
        (video_plug.VideoRequest, video_plug.video),
    ],
    ids=["image", "video"],
)
@pytest.mark.parametrize("subdir", ["uploads", "gui_images", "gui_video"],
                         ids=["uploads", "image-gallery", "video-gallery"])
def test_input_image_in_an_allowed_root_accepted(tmp_path, monkeypatch, subdir,
                                                 request_cls, handler_coro_fn):
    """A file in the upload inbox or a gallery passes confinement (it still 503s
    later for lack of a job manager, which proves we got past the input check).

    Both plugins accept both galleries: the GUI's "use as input" button on the
    image page fills the field with a gui_images path, and image-to-video takes
    an image, so neither plugin may be restricted to its own directory."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    inside = home / subdir / "ref.png"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"\x89PNG\r\n\x1a\nMINE")

    req = request_cls(prompt="x", input_image=str(inside))
    with pytest.raises(HTTPException) as ei:
        _run(handler_coro_fn(req, _fake_request()))
    # passed confinement; failed later on the missing GUI job manager
    assert ei.value.status_code == 503


@pytest.mark.parametrize(
    "request_cls,handler_coro_fn",
    [
        (image_plug.ImagineRequest, image_plug.imagine),
        (video_plug.VideoRequest, video_plug.video),
    ],
    ids=["image", "video"],
)
@pytest.mark.parametrize("relname", ["ref.png", "auth.key", "rag/index.json"],
                         ids=["data-root-file", "owner-key", "rag-store"])
def test_input_image_in_the_data_dir_root_rejected(tmp_path, monkeypatch, relname,
                                                   request_cls, handler_coro_fn):
    """The data dir itself is NOT an allowed root.

    It holds auth.key - the plaintext owner key - plus auth.json, sessions.json
    and the rag/coder stores, and these routes are mounted under the
    non-privileged image/video scopes, so "confined to the data dir" would leave
    a scoped key able to have localm's own secrets uploaded to a ComfyUI that may
    legitimately be a LAN or public host over plaintext http."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    secret = home / relname
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"\x89PNG\r\n\x1a\nSECRET")   # readable, even image-shaped

    req = request_cls(prompt="x", input_image=str(secret))
    with pytest.raises(HTTPException) as ei:
        _run(handler_coro_fn(req, _fake_request()))
    assert ei.value.status_code == 400
    assert "input" in str(ei.value.detail).lower()
    # The rejection must not disclose the data dir (it carries the OS username)
    # to a non-owner caller.
    assert str(home) not in str(ei.value.detail)


def test_image_nonexistent_input_image_still_400(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("LOCALM_HOME", str(home))

    missing = home / "uploads" / "nope.png"    # inside a root but does not exist
    missing.parent.mkdir(parents=True)
    req = image_plug.ImagineRequest(prompt="x", input_image=str(missing))
    with pytest.raises(HTTPException) as ei:
        _run(image_plug.imagine(req, _fake_request()))
    assert ei.value.status_code == 400
