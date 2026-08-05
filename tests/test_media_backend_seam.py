# SPDX-License-Identifier: AGPL-3.0-or-later
"""The media backend seam (I1).

Each media plugin (image/music/video) dispatches to the backend named by the
``backend`` config key: ``"comfy"`` (default) is the inline ComfyUI reference,
any other name loads ``backends/<name>.py`` via ``media_config.load_backend``.
These tests prove the seam actually SWITCHES implementations - it was previously a
config value that was read and then ignored (every call hard-wired to ComfyUI)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from localm import music_gen as _music_gen_pkg
from localm import video_gen as _video_gen_pkg
from localm.image_gen import comfy as _image_comfy
from localm.plugins import media_config
from localm.plugins.builtin.image import backend as image_backend
from localm.plugins.builtin.music import backend as music_backend
from localm.plugins.builtin.video import backend as video_backend


class _StubBackend:
    """A non-comfy backend module stand-in recording what it was asked to do."""

    def __init__(self):
        self.calls = []

    def ensure_available(self, s, on_progress=None):
        self.calls.append("ensure_available")
        return True, "stub up"

    def free_vram(self, s):
        self.calls.append("free_vram")
        return True

    def generate(self, s, *a, **k):
        self.calls.append(("generate", a, k))
        return True, "stub generated"


@pytest.mark.parametrize("backend_mod, gen_args", [
    (image_backend, ("a cat", Path("o.png"))),
    (music_backend, ("lofi", Path("o.flac"))),
    (video_backend, ("a cat walks", Path("o.mp4"))),
])
def test_non_comfy_backend_is_dispatched(backend_mod, gen_args, monkeypatch):
    stub = _StubBackend()
    loaded = {}

    def fake_load(package, name):
        loaded["args"] = (package, name)
        return stub

    monkeypatch.setattr(media_config, "load_backend", fake_load)
    s = {"backend": "myremote"}
    ok, msg = backend_mod.generate(s, *gen_args, self_url="", write_sidecar=False)
    assert ok and msg == "stub generated"
    assert any(isinstance(c, tuple) and c[0] == "generate" for c in stub.calls)
    assert loaded["args"][1] == "myremote"          # loaded by the configured name
    assert loaded["args"][0].startswith("localm.plugins.builtin.")
    # ensure_available / free_vram dispatch through the same seam.
    assert backend_mod.ensure_available(s) == (True, "stub up")
    assert backend_mod.free_vram(s) is True


@pytest.mark.parametrize("backend_mod", [image_backend, music_backend, video_backend])
def test_comfy_default_uses_the_inline_reference(backend_mod, monkeypatch):
    # The default 'comfy' must use the inline reference and NOT consult
    # load_backend (so it keeps working with no backends/ dir present).
    called = {"n": 0}

    def spy(*a, **k):
        called["n"] += 1
        raise AssertionError("load_backend should not be called for comfy")

    monkeypatch.setattr(media_config, "load_backend", spy)
    assert backend_mod._impl({"backend": "comfy"}) is backend_mod._COMFY_REF
    assert backend_mod._impl({}) is backend_mod._COMFY_REF          # empty -> comfy
    assert called["n"] == 0


def test_unknown_backend_falls_back_to_comfy(monkeypatch):
    # A configured-but-missing backend module must not hard-crash a generate; it
    # falls back to the comfy reference (the settings 'warning' carries notes).
    def boom(package, name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(media_config, "load_backend", boom)
    assert image_backend._impl({"backend": "ghost"}) is image_backend._COMFY_REF


_S = {"api_url": "http://127.0.0.1:8188", "launch_cmd": "", "workdir": "",
      "output_dir": "", "swap_policy": "never", "fast_dequant": True,
      "delete_outputs": False, "float_type": None}


@pytest.mark.parametrize("backend_mod, gen_args, underlying_mod, underlying_fn", [
    (image_backend, ("a cat", Path("o.png")), _image_comfy, "generate_image"),
    (music_backend, ("lofi", Path("o.flac")), _music_gen_pkg, "generate_music"),
    (video_backend, ("a cat walks", Path("o.mp4")), _video_gen_pkg, "generate_video"),
])
def test_instance_token_survives_the_backend_seam(
        backend_mod, gen_args, underlying_mod, underlying_fn, monkeypatch):
    """The plug.py route's instance_token must reach the per-plugin comfy.py
    generate_* call THROUGH the backend seam (plug.py -> backend.generate ->
    backend.py's _comfy_generate -> comfy.py), not just when comfy.py is
    called directly.

    Distinct from tests/test_comfy_containment.py, tests/test_music_gen.py
    and tests/test_video_gen.py's instance_token tests, which call
    comfy.generate_image/music/video directly and would stay green even if
    backend.py's adapter silently dropped instance_token on the floor between
    the seam and comfy.py - the exact one-layer-removed blind spot
    diff-review-discipline.md item 23 describes. This test goes through the
    actual public seam (backend_mod.generate) plug.py calls."""
    spy = MagicMock(return_value=(True, "ok"))
    monkeypatch.setattr(underlying_mod, underlying_fn, spy)

    backend_mod.generate(dict(_S), *gen_args, self_url="http://127.0.0.1:8642/v1",
                         write_sidecar=False, instance_token="tok-xyz")

    spy.assert_called_once()
    assert spy.call_args.kwargs.get("instance_token") == "tok-xyz"
